"""物化视图：sqlite 连接、schema、日志重放、全量重建。

这一层是**可重建的派生层**（删掉 `index.sqlite` 能从 `log.jsonl` 完整恢复），
它的全部难点在「重放幂等」，与上层「哪些事实当前有效」的查询是两回事。

依赖方向：模块级拿 `ops.apply_op`。`ops` 正文里反过来用本模块的
`_write_fact` / `_invalidate` —— 那处反向引用在**函数内**取属性，所以不会成环。
`render_facts_md` 用函数内惰性 import：`render` 依赖 `facts`、`facts` 反过来
惰性依赖本模块，模块级取它会成环。

**与原实现的一处结构性差别：schema 只有一版。**
原实现带着一条迁移链（`_ADDED_COLUMNS` / `_VECTOR_ADDED_COLUMNS` / `_migrate`），
因为线上库是分四个阶段长出来的。这里是一个全新的项目，没有任何存量库要照顾 ——
把演进史搬过来只会让「当前 schema 长什么样」要多读一百行才看得出来。取而代之的是
一个 `schema_version`：版本对不上就重跑 schema（物化层是可重建的，这是安全的），
之后加列时把版本号 +1 并补一段迁移即可。
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import render
from .facts import _fact_text_index
from .ops import apply_op, read_ops
from .paths import _lock, index_path

# 改 schema 时把它 +1，并在下面 `_migrate()` 里补一段从旧版升上来的语句。
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  id               TEXT PRIMARY KEY,
  subject          TEXT NOT NULL,
  predicate        TEXT NOT NULL,
  object           TEXT NOT NULL,
  valid_from       TEXT,
  valid_to         TEXT,
  confidence       REAL DEFAULT 0.0,
  importance       REAL DEFAULT 0.0,
  persona_attention REAL DEFAULT 0.0,
  pinned           INTEGER DEFAULT 0,
  status           TEXT DEFAULT 'active',
  source           TEXT DEFAULT 'extract',
  kind             TEXT DEFAULT 'fact',
  due              TEXT,
  turn_ref         TEXT,
  quote            TEXT,
  last_used_at     TEXT,
  use_count        INTEGER DEFAULT 0,
  text_index       TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_slot ON facts(status, subject, predicate);
CREATE INDEX IF NOT EXISTS idx_facts_live ON facts(status, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_kind ON facts(status, kind);
-- 热路径的取数（先截断再打分）：WHERE 的两列就是这条索引的前两列。
-- 排序里的 importance 项 SQLite 仍在内存里做 —— 它省的是「读全表」，
-- 不是「排全表」，但读全表正是大库那几十毫秒里的大头。
CREATE INDEX IF NOT EXISTS idx_facts_hot ON facts(status, valid_to, pinned, importance);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
-- vectors 的 text_hash 分不清「这条向量是哪个版本的事实编码出来的」时，
-- 事实改了内容、向量还留着旧的，检索就会按旧内容召回（用户看到的是
-- 「她记着我三年前说过的话」）。model 是编码器身份（repo + 维度）：只有 dim 的话，
-- 换一个**同维度**的模型会把两个向量空间混在一张表里，相似度全是噪声。
CREATE TABLE IF NOT EXISTS vectors (fact_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB,
                                    text_hash TEXT, model TEXT);
CREATE TABLE IF NOT EXISTS summaries (
  id         TEXT PRIMARY KEY,
  day        TEXT,
  created_at TEXT,
  text       TEXT,
  turn_ref   TEXT
);
-- 主动话题。它**不是事实**：没有槽位、没有回引，不能被引用；
-- 但它和纪要一样不可再生（模型生成），所以有一条 TOPIC 操作把它放进日志。
-- `used` 不进日志（它是排序状态，不是内容）。
CREATE TABLE IF NOT EXISTS topics (
  id         TEXT PRIMARY KEY,
  day        TEXT,
  created_at TEXT,
  kind       TEXT,
  text       TEXT,
  due_day    TEXT,
  ref        TEXT,
  used       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_topics_live ON topics(used, due_day);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
  text_index, content='facts', content_rowid='rowid');

-- external content 表必须自己同步，否则 FTS 永远是空的
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, text_index) VALUES (new.rowid, new.text_index);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, text_index) VALUES ('delete', old.rowid, old.text_index);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, text_index) VALUES ('delete', old.rowid, old.text_index);
  INSERT INTO facts_fts(rowid, text_index) VALUES (new.rowid, new.text_index);
END;
"""

# 会被写进 facts 表的字段（op 里的同名字段直通）。
# `persona_attention` 在前身里叫 `her_attention` —— 那个名字把「谁的视角」
# 写死进了 schema，而画像是宿主注入的（见 `PersonaProfile`），所以改名。
_FACT_FIELDS = ("subject", "predicate", "object", "valid_from", "valid_to",
                "confidence", "importance", "persona_attention",
                "pinned", "status", "source", "kind", "due", "turn_ref",
                "quote", "last_used_at", "use_count")

# 缺省值必须在这里补，**不能指望建表时的 DEFAULT**：我们是把全部字段显式写进
# INSERT 的，op 里没有的字段会写成 NULL，而 DEFAULT 只在「该列没出现在 INSERT 里」
# 时才生效 —— status 变成 NULL 之后每一条事实都查不出来（`status='active'`
# 匹配不上），而且不报任何错。
_FACT_DEFAULTS: dict[str, Any] = {
    "status": "active", "source": "extract", "pinned": 0, "use_count": 0,
    "confidence": 0.0, "importance": 0.0, "persona_attention": 0.0,
    "kind": "fact",
}

# 本进程里已经确认 schema 就绪的库。**这不是优化，是正确性问题**：
# ensure_schema 挂在检索的冷路径上（预算 50ms），而 executescript 整套 schema
# （含 CREATE VIRTUAL TABLE fts5）第一次要几十毫秒 —— 实测会把首次检索顶出预算，
# 冷路径静默返回空，看起来就像「记忆功能没生效」。
_SCHEMA_READY: set[str] = set()
_SCHEMA_LOCK = threading.Lock()


def connect(uid: str) -> sqlite3.Connection:
    """打开（必要时建）物化库。WAL：写入串行化，读不挡写。"""
    p = index_path(uid)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _schema_ok(con: sqlite3.Connection) -> bool:
    """这个库真的建好表、而且**已经是当前版本**吗。

    只查 facts 表在不在是不够的：老库有 facts 表但没有后加的列，那样看起来
    「有表」就返回 True，于是 schema 与迁移都被跳过 —— 新代码读新列拿到
    KeyError 或 None，新能力**永远不生效**，而且不报错。

    所以判据是「版本号 + 关键表的关键列都在」。物化层是可重建的，
    这里判错的代价最多是重跑一次 `CREATE TABLE IF NOT EXISTS`，方向安全。
    """
    try:
        if _meta_get(con, "schema_version", "") != str(SCHEMA_VERSION):
            return False
        cols = {r["name"] for r in con.execute("PRAGMA table_info(facts)").fetchall()}
        if not cols or not set(_FACT_FIELDS).issubset(cols):
            return False
        vcols = {r["name"] for r in con.execute("PRAGMA table_info(vectors)").fetchall()}
        if not vcols or not {"text_hash", "model"}.issubset(vcols):
            return False
        tcols = {r["name"] for r in con.execute("PRAGMA table_info(topics)").fetchall()}
        if not tcols or "used" not in tcols:
            return False
        con.execute("SELECT 1 FROM meta LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def ensure_schema(con: sqlite3.Connection) -> None:
    """建表并把版本号写上。幂等。"""
    con.executescript(_SCHEMA)
    _meta_set(con, "schema_version", str(SCHEMA_VERSION))
    con.commit()


def _migrate(con: sqlite3.Connection) -> None:
    """从旧版本升到 `SCHEMA_VERSION`。

    现在只有 v1，所以它是空的 —— 但**留着这个函数**，因为下次加列时
    「该在哪儿写」不该需要重新想一遍。加列的正确姿势是：
    `ALTER TABLE ... ADD COLUMN`（老库靠它），同时把新列写进 `_SCHEMA`
    （新库靠它），最后 `SCHEMA_VERSION += 1`。
    """
    return None


def open_index(uid: str) -> sqlite3.Connection:
    """连接 + 保证 schema 就绪。schema 每个库在每个进程里只建一次。

    别把它挪出计时区间就算了 —— 建好之后单次检索不到 1ms，真正的成本是
    「第一次」那一下；把它摊掉，冷路径的延迟预算才对得上。
    """
    con = connect(uid)
    key = str(index_path(uid))
    if key not in _SCHEMA_READY or not _schema_ok(con):
        with _SCHEMA_LOCK:
            if key not in _SCHEMA_READY or not _schema_ok(con):
                con.executescript(_SCHEMA)
                _migrate(con)
                _meta_set(con, "schema_version", str(SCHEMA_VERSION))
                con.commit()
                _SCHEMA_READY.add(key)
    return con


def _meta_get(con: sqlite3.Connection, key: str, default: str = "") -> str:
    try:
        row = con.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    except sqlite3.Error:
        return default
    return row["v"] if row else default


def _meta_set(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT INTO meta(k, v) VALUES(?, ?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (key, value))


def _write_fact(con: sqlite3.Connection, fid: str, fact: dict[str, Any]) -> None:
    row = {k: fact.get(k) for k in _FACT_FIELDS}
    for k, v in _FACT_DEFAULTS.items():
        if row.get(k) is None:
            row[k] = v
    if not row.get("valid_from"):
        row["valid_from"] = time.strftime("%Y-%m-%d")
    row["id"] = fid
    row["pinned"] = int(bool(row.get("pinned") or 0))
    row["use_count"] = int(row.get("use_count") or 0)
    row["text_index"] = _fact_text_index(fact)

    # 用 UPSERT 而不是 INSERT OR REPLACE。这是一个**功能性**选择，不是风格：
    # REPLACE 冲突时的隐式 DELETE 只在 recursive_triggers 打开时才触发删除触发器，
    # 而它默认是关的 —— 于是 facts_fts 里会留下这一行的旧版本，
    # 改了事实之后「旧词照样搜得到」。
    # UPSERT 是真正的 UPDATE，AFTER UPDATE 触发器会把旧 text_index 删掉、写进新的。
    cols = list(row.keys())
    marks = ", ".join("?" for _ in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id")
    con.execute(
        f"INSERT INTO facts ({', '.join(cols)}) VALUES ({marks}) "
        f"ON CONFLICT(id) DO UPDATE SET {updates}",
        list(row.values()))


def _invalidate(con: sqlite3.Connection, fid: str, day: str, reason: str) -> None:
    """失效而不是删除（双时间轴）：新事实让旧事实**不再被注入**，
    但「他以前说过什么」还查得到 —— 陪伴产品里，忘记比记错更容易被原谅，
    真正伤人的是「她矢口否认说过」。"""
    con.execute("UPDATE facts SET status='superseded', valid_to=? WHERE id=?",
                (day, fid))


def materialize(uid: str, *, force: bool = False) -> dict[str, Any]:
    """把日志里还没应用的操作投影到物化视图。

    返回 `{"applied": n, "bad_ops": [...]}`。

    **一条坏操作不许卡死整个用户。** 以前是一条 op 抛异常就整批回滚，
    于是同一个坏 op 每次重试都卡在同一位置，后面的永远不应用 —— 症状是
    「她记性变差」，而没有任何报错。

    **隔离不许静默**：丢掉的那条要连 id 与原因一起报出来。没有调用方消费它
    也不打紧 —— 静默丢数据是这套系统最不能接受的一种失败。
    """
    bad_ops: list[dict[str, str]] = []
    with _lock(uid):
        con = open_index(uid)
        try:
            ops = read_ops(uid)
            applied = 0 if force else int(_meta_get(con, "applied_ops", "0") or 0)
            if applied > len(ops):       # 日志被换过 / 截断过：重新来
                applied = 0
            n = 0
            for op in ops[applied:]:
                # 每条一个 SAVEPOINT：回滚只会退掉这一条，前面已经应用的好 op
                # 留在事务里（直接 `con.rollback()` 会把它们一起带走 ——
                # 那又变回「一条坏行卡死全库」，只是卡的位置换了一下）。
                con.execute("SAVEPOINT mat_op")
                try:
                    apply_op(con, op)
                except Exception as e:   # noqa: BLE001  一条坏 op 不该卡死整个用户
                    con.execute("ROLLBACK TO mat_op")
                    bad_ops.append({
                        "id": str(op.get("id") or ""),
                        "op": str(op.get("op") or ""),
                        "error": f"{type(e).__name__}: {e}"})
                else:
                    con.execute("RELEASE mat_op")
                n += 1                   # 读过就算推进，下次不再撞同一条
            if n:
                _meta_set(con, "applied_ops", str(len(ops)))
                con.commit()
            if n and render.enabled():
                # 写路径的最后一步是重渲染人类可读视图。只在真的应用了操作时做，
                # 别让一次空调用去重写整份 md。
                render.render_facts_md(uid)
            return {"applied": n, "bad_ops": bad_ops}
        finally:
            con.close()


def wipe(uid: str) -> None:
    """把这个用户的物化层**清空**：表留着、行全没。

    它是整库重置的第一步，也是「purge 真的让她不再记得」的落点：Windows 上
    有打开的句柄时 `index.sqlite` 这个**文件**删不掉，于是先把库里的内容清空 ——
    检索、面板、巩固读的都是这张库，空表就等于她一条都不记得，与文件能不能
    删掉无关。旧实现只删文件、失败被吞掉，结果是日志与原话没了、
    事实**原样留在库里**，接口还回成功。

    `DELETE FROM facts` 会经 `facts_ad` 触发器把 facts_fts 里的词条一起带走，
    所以不用手工清 FTS。
    """
    with _lock(uid):
        # **先看文件在不在**：open_index() 会顺手建库建 schema，而 wipe 是删除
        # 路径上的一步 —— 对着一个从没有过记忆的 uid 调它，不该凭空造出一个目录。
        if not index_path(uid).exists():
            return
        try:
            con = open_index(uid)
        except Exception:                    # noqa: BLE001  库都打不开 = 本来就是空的
            return
        try:
            with con:
                con.execute("DELETE FROM vectors")
                con.execute("DELETE FROM summaries")
                con.execute("DELETE FROM topics")
                con.execute("DELETE FROM facts")
                con.execute("DELETE FROM meta")
        finally:
            con.close()


def rebuild(uid: str) -> int:
    """从 `log.jsonl` 全量重建物化视图。返回重放的操作条数。

    这是「只有日志不可再生」这句话的可执行版本：删掉 `index.sqlite` 也应该完整
    恢复，包括 FTS 索引（触发器会跟着重放自动同步，末尾再显式 rebuild 一次兜底 ——
    索引坏了而表还在时它才救得回来）。
    """
    with _lock(uid):
        p = index_path(uid)
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(p) + suffix).unlink()
            except OSError:
                pass
        with _SCHEMA_LOCK:
            _SCHEMA_READY.discard(str(p))     # 文件没了，缓存也跟着失效
        con = open_index(uid)
        try:
            ops = read_ops(uid)
            for op in ops:
                apply_op(con, op)
            con.execute("INSERT INTO facts_fts(facts_fts) VALUES('rebuild')")
            _meta_set(con, "applied_ops", str(len(ops)))
            con.commit()
            return len(ops)
        finally:
            con.close()
