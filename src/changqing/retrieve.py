"""检索：热 / 冷两条路径 + 融合 + 画像重排。

**热路径**（每轮都跑，常数预算）：钉住的、承诺类的、以及重要性高的事实，按
`hot_tokens` 封顶。它把「记忆块随轮次线性膨胀」那张表变成一条**常数**线 ——
聊了 30 轮还是 3000 轮，注入的量一样。

**冷路径**（按需触发）：这句话指向过去、或者它和已知事实有词面交叠时才检索，
超过 `recall_ms` 就放弃。

两条路径都做两件事：

  1. **去重**：刚说过的事已经在上下文里，再当记忆喂一遍既费 token 又会让她过度强调
     （「你上次说」连说三遍就很假）；
  2. **画像重排**：乘以 `persona_attention`。通用的「重要」和「她会在意的」不是
     一回事 —— 她该记住的是「你不吃香菜」，而不是一段履历字段。

**绝不抛异常**：超时、索引坏、分词器缺失、向量模型没加载，都只是「这一轮没有可注入的」。
但**「不吞」也不等于「不说」** —— 被吞掉的异常记在 `last_error()` 里，排查时看得见。
这条是被坑出来的：冷路径因为一个内部错误一直返回空，表现是「记忆功能好像没生效」，
而日志里一个字都没有。
"""

from __future__ import annotations

import array
import math
import time
import traceback
from collections.abc import Sequence
from operator import mul
from typing import Any

from .config import MemoryConfig
from .runtime import runtime
from .store import list_summaries, list_topics, mark_used, open_index
from .tokenize import to_query, tokenize
from .tokens import clip_to_tokens, estimate_tokens

# 进程级的一次性开销先在这里付掉：jieba 建词典、numpy 首次 import 都在百毫秒级，
# 而冷路径的预算是几十毫秒。不预热的话**第一次**冷路径检索会为了 import 顶爆预算、
# 静默返回空，看起来就是「记忆好像没生效」。所以这类开销属于「谁用谁先付」，
# 不属于每轮检索。

# 「绝不抛异常」不等于「静默失败」。检索吞掉的异常记在这里，自检和排查才看得见。
_ERRORS: list[str] = []
_ERROR_KEEP = 5

# **超时**和**没命中**都会返回空列表，但原因和处置完全相反：
#   超时 = 这次算慢了，该看预算、看是不是事实太多；
#   没命中 = 库里确实没有相关的，属于正常。
# 两者在指标里长得一模一样的话，「记忆老是空的」就没法定位。
STATS: dict[str, int] = {
    "search": 0,
    "timeout": 0,
    "no_hit": 0,
    "ok": 0,
    "vec": 0,
    "error": 0,
}
LAST: dict[str, Any] = {"reason": "", "ms": 0, "vec": 0}


def _cfg() -> MemoryConfig:
    """当前配置。**每次调用现取** —— 模块级快照会让「改了配置不生效」且不报错。"""
    return runtime().config


def _mark(reason: str, **kw: Any) -> None:
    LAST.update({"reason": reason, "ms": 0, "vec": 0})
    LAST.update(kw)


def stats() -> dict[str, int]:
    """累计计数。给运维视图与自检用。"""
    return dict(STATS)


def last_error() -> str:
    """最近一次被吞掉的异常（没有就是空串）。「不抛」不等于「不留痕」。"""
    return _ERRORS[-1] if _ERRORS else ""


def _note_error() -> None:
    STATS["error"] += 1
    _ERRORS.append(traceback.format_exc())
    del _ERRORS[:-_ERROR_KEEP]  # 只留最近几条，别把内存吃光


def card_text(f: dict[str, Any]) -> str:
    """一条事实渲染成一行卡片文本。注入预算按这个长度算。

    承诺 / 约定带 `due` 时把日期写进去 —— 「答应过 2026-09-16 之前带你去看展」比
    「答应过带你看展」有用得多：前者她自己就能判断是不是已经过期了。
    """
    text = f"{f.get('subject', '')}{f.get('predicate', '')}{f.get('object', '')}"
    if str(f.get("kind") or "fact") in ("commitment", "promise") and f.get("due"):
        text += f"（{str(f['due'])[:10]} 之前）"
    return text


def _live(f: dict[str, Any], exclude_refs: Sequence[str], min_conf: float) -> bool:
    """这条事实这一轮能不能进注入集合。

    `pinned` 的放行**判在去重之前**：用户亲手钉住的是**显式意图**，压得过
    「刚说过的话不要再喂一遍」这条默认规则。顺序反了的话，只要它的 `turn_ref`
    落在最近那批消息里，被钉住的那条承诺就会静默消失。取舍理由：漏掉一条他亲手
    钉住的承诺，比多注入一句刚说过的话更伤（前者他看得见自己钉过）。
    """
    if str(f.get("status") or "active") != "active":
        return False
    if f.get("valid_to"):
        return False  # 已失效：不再注入，但还查得到
    if f.get("pinned"):
        return True  # 用户钉住的无条件进（去重也压不过）
    if exclude_refs and f.get("turn_ref") in exclude_refs:
        return False  # 刚说过，上下文里有
    return float(f.get("confidence") or 0.0) >= min_conf


def _age_days(f: dict[str, Any]) -> float:
    """这条事实「多久以前说起的」—— 只认 `valid_from`。

    不在这后面挂一个 `or last_used_at` 的兜底：物化层保证 `valid_from` 非空，
    那个兜底永远不会轮到，却会让人以为打分读了 `last_used_at`。它只由 `mark_used`
    记账，现行口径里不参与打分。
    """
    day = str(f.get("valid_from") or "")[:10]
    if not day:
        return 0.0
    try:
        t = time.mktime(time.strptime(day, "%Y-%m-%d"))
    except (ValueError, OverflowError):
        return 0.0
    return max(0.0, (time.time() - t) / 86400.0)


def _recency(f: dict[str, Any]) -> float:
    """时间衰减，**带下限 0.6**。

    它是排序信号，不是准入门槛：「他不吃香菜」放一年也还是真的，不该被时间直接
    判死、然后永远注入不进来。真正把关的是 `min_score`（相关度）与
    `min_confidence`（抽取置信度）。
    """
    return 0.6 + 0.4 / (1.0 + _age_days(f) / 180.0)


# 热路径候选的 SQL 排序 —— 必须与 `hot_facts` 的三档**同序**：
#   1. pinned 优先；
#   2. 承诺 / 约定排在一般事实前，组内按 due（最早的先）；
#      **这一档不做条数截断**：Python 那边承诺整档都在里面，截断点只有 token
#      预算一个 —— 两边截断点不同的话，`LIMIT` 砍掉的就不只是本来就排在预算
#      之外的那些了；
#   3. 其余按 importance × persona_attention，id 兜底。
# 同序是「先截断」成立的前提：只有两边顺序一致，`LIMIT` 砍掉的才确定是本来就
# 排在预算之外的那些。
_HOT_ORDER_BY = (
    " ORDER BY COALESCE(pinned,0) DESC, "
    "CASE WHEN kind IN ('commitment','promise') THEN 0 ELSE 1 END ASC, "
    "CASE WHEN kind IN ('commitment','promise') "
    "     THEN COALESCE(due,'9999') END ASC, "
    "(COALESCE(importance,0) * (0.5 + COALESCE(persona_attention,0))) DESC, "
    "id ASC"
)


def _slot_key(f: dict[str, Any]) -> tuple[str, str]:
    """槽位：同一件事的两种说法应该落在这个键上（subject + predicate）。"""
    return (str(f.get("subject") or "").strip(), str(f.get("predicate") or "").strip())


def _slot_rank(f: dict[str, Any]) -> tuple[int, int, float, str]:
    """同槽位里留谁：**pinned 优先，其次 `valid_from` 新的，再其次 importance**。

    返回的元组按**升序**排，所以偏好全部折成正负号写在这里，调用方只比大小。
    `valid_from` 用 `int(...)` 而不是字符串比较：`YYYY-MM-DD` 里带 `-`，而 ASCII 里
    `'-'`（0x2D）小于数字，所以 `"2026-09-05" < "2026-09-01"` —— 直接比字符串会把
    「新的」判成「小的」。日期形状不对的（手写进库的）记 0，排在有日期的之后。
    """
    day = str(f.get("valid_from") or "")[:10].replace("-", "")
    try:
        stamp = int(day)
    except ValueError:
        stamp = 0
    return (
        0 if f.get("pinned") else 1,
        -stamp,  # 新的在前
        -float(f.get("importance") or 0.0),
        str(f.get("id") or ""),
    )


def _dedup_slots(facts: Sequence[dict]) -> list[dict]:
    """同槽位（subject+predicate）只留一条 —— 读取侧的兜底。

    写入侧的冲突消解（`extract.resolve_ops`）只管正常路径；用户点一下「让她想起来」
    （把 status 改回 active、清空 valid_to），或者历史遗留的两条同槽位 active，
    都会让**互斥的两个 object 同时进注入集合** —— 她会同时说猫叫团子和猫叫咪咪。
    这里按槽位兜一层，**不改**写入侧的语义。

    顺序敏感：必须在 `_live`（status / valid_to / 置信度 / 去重）**之后**、
    取预算**之前**跑。理由：`_live` 挡掉的（失效的、刚说过的）本来就不该占坑，
    先兜槽位等于让一条死事实把活着的那条挤掉。
    """
    best: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    for f in facts:
        key = _slot_key(f)
        cur = best.get(key)
        if cur is None:
            best[key] = f
            order.append(key)
        elif _slot_rank(f) < _slot_rank(cur):
            best[key] = f
    return [best[k] for k in order]


def _fetch_live(
    uid: str, exclude_refs: Sequence[str], min_conf: float, limit: int = 0
) -> list[dict]:
    """这一轮的活跃事实候选。`limit > 0` 时在 **SQL 侧**先按热路径顺序截断。

    为什么要它：把活跃事实全取回来、在 Python 侧逐条建对象再打分，是纯粹的浪费
    —— 预算是几百 token、只装几十条，而库里有几千条。取数上界在规模上来之后
    是第一件要修的事。

    `limit <= 0` 时不带 LIMIT（旧行为，规模脚本与对拍用）。

    边界要说清：`LIMIT` 在 `_live` 过滤**之前**，被去重 / 低置信挡掉的行同样占名额。
    默认值相对注入预算是几十倍余量，这种挤占不影响最终选中；把
    `hot_fetch_max` 调到贴近预算才会。

    最后一步是**同槽位兜底去重**：顺序放在 `_live` 之后 —— 死的那条不该把活着的挤掉。
    """
    sql = "SELECT * FROM facts WHERE status='active' AND (valid_to IS NULL OR valid_to='')"
    params: tuple[Any, ...] = ()
    if int(limit or 0) > 0:
        sql += _HOT_ORDER_BY + " LIMIT ?"
        params = (int(limit),)
    con = open_index(uid)
    try:
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    live = [dict(r) for r in rows if _live(dict(r), exclude_refs, min_conf)]
    return _dedup_slots(live)


def _take_budget(facts: list[dict], budget_tokens: int) -> tuple[list[dict], int]:
    """按 token 预算裁剪，返回 (选中, 用了多少 token)。

    **第一条豁免**：预算装不下也留着。事实是原子 —— 切一半会让模型拿着半个事实
    说话，那比多花几十 token 贵得多。散文（纪要 / 话题）不适用这条，走
    `clip_to_tokens`。
    """
    out: list[dict] = []
    used = 0
    for f in facts:
        cost = estimate_tokens(card_text(f))
        if out and used + cost > budget_tokens:
            break
        out.append(f)
        used += cost
    return out, used


def hot_facts(
    uid: str, *, exclude_refs: Sequence[str] = (), budget_tokens: int | None = None
) -> list[dict]:
    """热路径：每轮都跑，常数预算。

    排序分**三档**：

      1. 用户钉住的 —— 无条件进；
      2. **未兑现的承诺 / 约定** —— 「他答应周末带我去的」这类如果被一般事实挤掉，
         她就会显得不守约。**整档**排在一般事实之前，不靠 importance 去和别的记忆
         抢；档内按 due 最早的先；
      3. 其余按 importance × persona_attention 排。

    第 2 档刻意**不做条数截断**。只放前 N 条的话，第 N+1 条起既不在钉住那一档、
    也不在承诺那一档，于是永远进不了注入集合 —— 而承诺又只被同槽位的 SUPERSEDE
    清掉、巩固还特意把它排除在降权之外，那几个名额会被几条过期的旧承诺永久占着。
    结果是卡片里显示五条承诺、她说话时只看得到最旧三条，而「她答应过的事」攒得
    越多她越看不见。现在名额之外的承诺继续留在这一档里，只是可能被 token 预算裁掉。

    取数有上界（`hot_fetch_max`）：SQL 侧按**与下面完全同序**的 `ORDER BY ... LIMIT`
    只交回可能进预算的那一段。同序意味着这里**不能**对承诺做 SQL 那边没有的截断，
    否则 `LIMIT` 砍掉的就不只是排在预算外的那些。
    """
    cfg = _cfg()
    budget = int(budget_tokens if budget_tokens is not None else cfg.hot_tokens)
    cands = _fetch_live(uid, exclude_refs, cfg.min_confidence, limit=int(cfg.hot_fetch_max or 0))
    pinned = [f for f in cands if f.get("pinned")]
    commits = [
        f
        for f in cands
        if not f.get("pinned") and str(f.get("kind") or "fact") in ("commitment", "promise")
    ]
    rest = [f for f in cands if f not in pinned and f not in commits]

    commits.sort(key=lambda f: (str(f.get("due") or "9999"), str(f.get("id"))))
    rest.sort(
        key=lambda f: (
            -(float(f.get("importance") or 0.0) * (0.5 + float(f.get("persona_attention") or 0.0))),
            str(f.get("id")),
        )
    )
    ordered = pinned + commits + rest
    picked, _ = _take_budget(ordered, budget)
    return picked


def hot_summaries(
    uid: str, *, k: int | None = None, budget_tokens: int | None = None
) -> list[dict]:
    """热路径的故事线：最近几场对话的小结，按 token 预算裁剪。

    为什么只取**最近**的：故事线的价值在连续性（「上次你说……」），久远的具体事实
    由事实层与冷路径负责。在这里再搭一套按语义检索纪要的索引，等于把「记得准不准、
    能不能查」复制一遍 —— 而那套索引本来就该只服务事实层。

    取数失败只是「这轮没有故事」，**绝不抛异常**：说话永远优先于记忆。
    """
    cfg = _cfg()
    k = int(k if k is not None else cfg.story_k)
    if k <= 0:
        return []
    budget = int(budget_tokens if budget_tokens is not None else cfg.story_tokens)
    out: list[dict] = []
    used = 0
    try:
        # 多取几条再裁：预算先到、条数先到，两种都要能停
        for s in list_summaries(uid, limit=max(8, k * 4)):
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            cost = estimate_tokens(text)
            if out and used + cost > budget:
                break
            if not out and cost > budget:
                # 单条就超预算：**截断**而不是豁免。纪要是散文，少说一句可以；
                # 不截的话 story_tokens 就不是上界。
                text = clip_to_tokens(text, budget)
                if not text:
                    break
                cost = estimate_tokens(text)
            out.append(
                {
                    "id": str(s.get("id") or ""),
                    "day": str(s.get("day") or "")[:10],
                    "text": text,
                }
            )
            used += cost
            if len(out) >= k:
                break
    except Exception:  # noqa: BLE001  取不到只是少一段故事
        _note_error()
        return []
    return out


def hot_topics(uid: str, *, k: int | None = None, budget_tokens: int | None = None) -> list[dict]:
    """她下次主动开口时可以挑的话题（**只在主动开口那一轮读**）。

    为什么不在普通回话时读：手里握一张「你可以问他这个」的清单，模型就会去执行它
    —— 他刚说了一件事，她反问另一件。普通那一轮她该说的只有他刚说的那件。

    口径与故事线一致：散文预算（超了**裁剪**而非豁免）。取数失败只是「这次没得挑」，
    继续说话 —— 绝不抛异常。
    """
    cfg = _cfg()
    k = int(k if k is not None else cfg.topic_k)
    budget = int(budget_tokens if budget_tokens is not None else cfg.topic_tokens)
    if k <= 0 or budget <= 0:
        return []
    out: list[dict] = []
    used = 0
    try:
        for t in list_topics(uid, limit=max(4, k * 3)):
            text = str(t.get("text") or "").strip()
            if not text:
                continue
            cost = estimate_tokens(text)
            if out and used + cost > budget:
                break
            if not out and cost > budget:
                text = clip_to_tokens(text, budget)
                if not text:
                    break
                cost = estimate_tokens(text)
            out.append(
                {
                    "id": str(t.get("id") or ""),
                    "day": str(t.get("day") or "")[:10],
                    "due_day": str(t.get("due_day") or "")[:10],
                    "kind": str(t.get("kind") or "share"),
                    "text": text,
                }
            )
            used += cost
            if len(out) >= k:
                break
    except Exception:  # noqa: BLE001  没话题 = 少一个开场理由
        _note_error()
        return []
    return out


# ---------------------------------------------------------------- 冷路径
# 取数形状：**先三路召回拿 id，再按 id 取行**。每一步都有界：
#   三路召回各自带 LIMIT → 候选 id 是个几百个的量级 → `WHERE id IN (...)` 只取这几行。
# 旧形状是把全部活跃事实取回来、在 Python 侧逐行建对象、全量打分，最后才按预算截断
# —— 那是「截断」不是「省活」。
# 钉住的那批单独并一条 `pinned=1`（有界）：它是**用户显式**表达的意图，三路召回
# 一条都没撞上时也必须进。
PIN_FETCH_MAX = 64
# `IN (...)` 的分批大小：SQLite 的变量上限（编译期 SQLITE_MAX_VARIABLE_NUMBER）
# 默认是 999，取 400 留足余量。
_ID_BATCH = 400

_LIVE_PROBE_SQL = (
    "SELECT 1 FROM facts WHERE status='active' AND (valid_to IS NULL OR valid_to='') LIMIT 1"
)


def _any_live_fact(con) -> bool:
    """这个库里到底有没有**活跃事实**（在同一把连接上问）。

    它必须与「本轮有没有候选」分开：`no_fact` 的含义是「她库里一条活跃事实都没有」，
    那是**真的没料**；而「有事实、这一问一条都没召回」是另一件事（`no_hit`）。
    混成一个的话，运营者会拿「没命中」去查「她是不是失忆了」—— 两件事的处置完全相反。
    """
    return con.execute(_LIVE_PROBE_SQL).fetchone() is not None


def _pinned_ids(con) -> list[str]:
    """库里的钉住事实（有界）。"""
    rows = con.execute(
        "SELECT id FROM facts WHERE status='active' "
        "AND (valid_to IS NULL OR valid_to='') AND pinned=1 "
        "ORDER BY id ASC LIMIT ?",
        (PIN_FETCH_MAX,),
    ).fetchall()
    return [str(r["id"]) for r in rows]


def _fetch_rows(con, ids: Sequence[str]) -> dict[str, dict]:
    """按 id 取事实行（去重、分批）。取不到（id 过期 / 写坏了）就不返回那一条。"""
    uniq = [str(i) for i in dict.fromkeys(ids) if i]
    out: dict[str, dict] = {}
    for i in range(0, len(uniq), _ID_BATCH):
        part = uniq[i : i + _ID_BATCH]
        sql = "SELECT * FROM facts WHERE id IN ({})".format(",".join("?" * len(part)))
        for r in con.execute(sql, part).fetchall():
            d = dict(r)
            out[str(d.get("id"))] = d
    return out


def _slot_hits(con, tokens: Sequence[str], allowed: set | None = None) -> list[str]:
    """槽位精确命中：查询词正好等于某条事实的 predicate / object。

    排在融合第一位 —— 它比全文检索精确得多：「不吃香菜」对上了就是对的，
    而不是碰巧共享一个「吃」字。

    `allowed` 为空 = 不再额外交一遍活跃过滤：WHERE 里已经有 status / valid_to，
    而「这一轮的注入集合」是在取行**之后**才算得出来的。测试与规模脚本仍可以传一个
    集合做白名单。
    """
    ids: list[str] = []
    for t in tokens:
        if len(t) < 2:
            continue
        for row in con.execute(
            "SELECT id FROM facts WHERE status='active' "
            "AND (valid_to IS NULL OR valid_to='') "
            "AND (predicate=? OR object=?) LIMIT 8",
            (t, t),
        ).fetchall():
            if allowed is not None and row["id"] not in allowed:
                continue
            if row["id"] not in ids:
                ids.append(row["id"])
    return ids


def _fts_hits(con, query: str, allowed: set | None = None, limit: int = 16) -> list[str]:
    """全文召回。空 MATCH 表达式在 SQLite 里是**语法错误**，所以先判空。"""
    expr = to_query(query)
    if not expr:
        return []
    try:
        rows = con.execute(
            "SELECT f.id FROM facts_fts JOIN facts f ON f.rowid = facts_fts.rowid "
            "WHERE facts_fts MATCH ? AND f.status='active' "
            "AND (f.valid_to IS NULL OR f.valid_to='') "
            "ORDER BY bm25(facts_fts) LIMIT ?",
            (expr, limit),
        ).fetchall()
    except Exception:  # noqa: BLE001  FTS 坏了不该让检索整条挂掉
        return []
    return [r["id"] for r in rows if allowed is None or r["id"] in allowed]


def vectors_available(con, dim: int) -> bool:
    """库里有没有**当前维度**的向量。不加载模型、不编码。

    与写入侧的判断必须是**同一句**：两边各写一份的话，会出现「检索以为有、
    重建以为没有」这种各说各的的空转。所以这个谓词只定义在这里，写入侧 import 它。
    """
    try:
        row = con.execute("SELECT 1 FROM vectors WHERE dim=? LIMIT 1", (int(dim),)).fetchone()
    except Exception:  # noqa: BLE001  还没建过 vectors 表（空库）
        return False
    return row is not None


def _unpack(blob: bytes) -> array.array:
    """把库里存的小端 float32 还原成一维数组。用 `array` 而不是 numpy：
    核心零第三方依赖，而这里只需要一个扁平数组。"""
    a = array.array("f")
    a.frombytes(bytes(blob))
    return a


def _vec_hits(con, query: str, dim: int, allowed: set | None = None, limit: int = 16) -> list[str]:
    """向量召回。没有注入 `Embedder`（或它没加载好）时**整支不进**。

    这一支是「口语化转述」那一档的补丁：字面一个词都不重叠时，槽位与全文检索
    都召不回，向量还能捞一把。它是可选增强，坏了不影响主路径 —— 所以整段
    吞异常，只在 `last_error()` 里留一笔。

    **余弦下限（`embed_min_cos`）不是可有可无的调参**：融合用的是 RRF，任何一张表里
    排第一都算「相关度 1.0」，所以向量的一条噪声命中会和全文检索的精确命中同分。
    而中文短句的余弦本来就挤在一起，不设门槛就是用一堆似是而非的记忆把正确的挤出去。

    **相似度是纯标准库算的**（`sum(map(mul, ...))`，见 `docs/retrieval.md`）：
    写入侧已经归一化，所以这里只需一次点积。512 维 2000 条本机实测约 44ms，
    与冷路径预算同一量级；扫描条数由 `embed_scan_max` 封顶。这不是「差不多就行」
    的估算 —— 超时的代价是**整条冷路径返回空**，所以闸门必须自己算得清楚。
    """
    cfg = _cfg()
    emb = runtime().embedder
    if not _embedding_on(emb):
        return []
    try:
        if not vectors_available(con, dim):
            return []
        vecs = emb.encode([query])
        if not vecs:
            return []
        q = _unpack_encoded(vecs[0])
        qn = math.sqrt(sum(x * x for x in q))
        if qn == 0.0:
            return []
        # 只读当前维度的行：换过模型的话库里会混着两种维度，混着读出来的相似度
        # 全是垃圾（而且不报错）。
        rows = con.execute(
            "SELECT v.fact_id, v.vec FROM vectors v JOIN facts f ON f.id = v.fact_id "
            "WHERE v.dim=? AND f.status='active' "
            "AND (f.valid_to IS NULL OR f.valid_to='') LIMIT ?",
            (int(dim), int(cfg.embed_scan_max)),
        ).fetchall()
        if not rows:
            return []
        floor = float(cfg.embed_min_cos)
        scored: list[tuple[float, str]] = []
        for r in rows:
            fid = str(r["fact_id"])
            if allowed is not None and fid not in allowed:
                continue
            sim = sum(map(mul, _unpack(r["vec"]), q)) / qn
            if sim >= floor:
                scored.append((sim, fid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [fid for _sim, fid in scored[:limit]]
    except Exception:  # noqa: BLE001  向量是可选增强，坏了不影响主路径
        _note_error()
        return []


def _embedding_on(emb) -> bool:
    """有没有一条可用的向量路。

    判据是「注入的实现自己说它开着」而不是「配置里写了什么」：向量能力现在是
    注入进来的，配置里没有 provider 这一项 —— 也就没有「配置说开、实现是空」的
    那种不一致。
    """
    name = str(getattr(emb, "name", "") or "").strip().lower()
    if not name or name in ("none", "off", "0"):
        return False
    return bool(getattr(emb, "loaded", lambda: False)())


def _unpack_encoded(vec: Sequence[float]) -> array.array:
    """把注入的 Embedder 返回的一维向量装进 `array`，让下面的点积走 C 层迭代。"""
    return array.array("f", (float(x) for x in vec))


def _rrf(rank_lists: Sequence[Sequence[str]], k: int = 0) -> dict[str, float]:
    """Reciprocal Rank Fusion：只用名次不用分数。

    这样省掉了「BM25 和余弦怎么归一化」这个永远调不对的问题 —— 两张表的分数量纲
    根本不可比，硬凑一个权重函数只会得到一组只在某次评测里好看的系数。
    """
    if not k:
        k = int(_cfg().rrf_k)
    out: dict[str, float] = {}
    for lst in rank_lists:
        for rank, fid in enumerate(lst):
            out[fid] = out.get(fid, 0.0) + 1.0 / (k + rank + 1)
    return out


def search(
    uid: str,
    query: str,
    *,
    k: int | None = None,
    timeout_ms: int | None = None,
    track: bool = True,
    exclude_refs: Sequence[str] = (),
    budget_tokens: int | None = None,
) -> list[dict]:
    """冷路径。**绝不抛异常**：超时、索引坏、分词器缺失都只是「这轮没有可注入的」。

    `budget_tokens` 是调用方给的注入预算（对外契约见 `Memory.recall`）；不给就取
    `hot_tokens`。这个参数曾经在门面上被收下就扔了 —— 契约与实现不一致，而且是
    「静默忽略」那种：调用方以为限额生效了。

    计时口径（重要）：`recall_ms` 覆盖**整条检索** —— 打开连接、三路召回、按 id 取行、
    逐行过滤、融合与打分。起点必须在取数**之前**；放在取数之后的话，最贵的那一段
    压根不在预算里，超时保护形同虚设。

    仍然不进预算的是**进程级一次性开销**（jieba 建词典、向量实现首次 import）——
    它们属于启动预热，不属于每一轮。
    """
    cfg = _cfg()
    k = int(k if k is not None else cfg.recall_k)
    budget = int(budget_tokens if budget_tokens is not None else cfg.hot_tokens)
    budget_ms = float(timeout_ms if timeout_ms is not None else cfg.recall_ms)
    STATS["search"] += 1
    t_start = time.perf_counter()
    deadline = t_start + budget_ms / 1000.0
    try:
        con = open_index(uid)
        try:
            # ① 三路召回**先**拿 id（各自有 LIMIT），不把全表取回来建对象
            slots = _slot_hits(con, tokenize(query))
            fts = _fts_hits(con, query, limit=max(8, k * 4))
            vec = _vec_hits(con, query, int(cfg.embed_dim), limit=max(8, k * 4))
            # ② 用户钉住的那批**显式**并进来（有界）。`recall()` 是对外契约，单独
            #    调它时也得给全 pinned —— 三路召回一条都没撞上就不给的话，
            #    「用户亲手钉住的承诺无条件进」在冷路径上是假的。
            pins = _pinned_ids(con)
            # ③ 只按 id 取这几行，再逐行判「能不能进这一轮的注入集合」
            rows = _fetch_rows(con, [*pins, *slots, *fts, *vec])
            live_rows = _dedup_slots(
                [r for r in rows.values() if _live(r, exclude_refs, float(cfg.min_confidence))]
            )
            if not live_rows:
                STATS["no_hit"] += 1
                # 「库里没有活跃事实」与「这一问什么都没召回」是两件事，
                # 靠 reason 分开。探针只在空结果这一条路上跑，`LIMIT 1`、走索引。
                _mark("no_fact" if not _any_live_fact(con) else "no_hit")
                return []
            live = {f["id"]: f for f in live_rows}
            if time.perf_counter() > deadline:
                # 算都算完了才超时，这笔时间已经花出去了 —— 这里只是决定
                # 「这轮算不算数」。所以指标必须把它和「没命中」分开。
                STATS["timeout"] += 1
                _mark("timeout", ms=int((time.perf_counter() - t_start) * 1000))
                return []  # 超时宁可不注入，也不拖首字
        finally:
            con.close()

        # 三条召回表的分值归一：**任一**表里排第一就算相关度 1.0，多表同时命中会被
        # `min(1.0)` 截断（那是「更可信」的加成，不是分数爆表）。不能按「三张表都
        # 命中」当满分 —— 现实中绝大多数查询只会命中一张表，那样归一出来的相关度
        # 永远只有 0.33，再乘上画像权重与时间衰减，就没有任何一条能过 `min_score`，
        # 检索等于形同虚设。
        rrf = _rrf([slots, fts, vec])
        top = 1.0 / (int(cfg.rrf_k) + 1)
        scored = []
        for fid, raw in rrf.items():
            f = live.get(fid)
            if not f:
                continue
            if f.get("pinned"):
                continue  # 钉住的**单独一档**，不参与打分门槛
            rel = min(1.0, raw / top) if top else 0.0
            # 画像重排 + 时间衰减
            qual = 0.5 * float(f.get("importance") or 0.0) + 0.5 * float(
                f.get("persona_attention") or 0.0
            )
            score = rel * (0.75 + 0.25 * qual) * _recency(f)
            if score < float(cfg.min_score):
                # 门槛的真实力气只在**召回窗口的尾巴**上：窗口内的分数下限本来就
                # 高于它，能拦下的是「只命中一张表、又排在尾部、qual 与 recency
                # 都垫底」那一小段。真正把关的是**召回本身**：槽位要精确相等、
                # 全文要有词面交叠、向量要过 `embed_min_cos`。所以调门槛之前先想清楚
                # 它到底在挡什么 —— 把它调高不会让检索更准，只会让注入更少。
                continue
            f = dict(f)
            f["_score"] = round(score, 4)
            f["_match"] = "slot" if fid in slots else "fts" if fid in fts else "vec"
            scored.append(f)
        scored.sort(key=lambda x: (-x["_score"], str(x["id"])))
        # 钉住的排在**所有**打分结果之前，且不过分数门槛 —— 与 `hot_facts` 的分档
        # 同序（pinned → 承诺 → 其余），也才配得上「无条件进」这句话。
        pinned = []
        for fid in pins:
            f = live.get(fid)
            if not f or not f.get("pinned"):
                continue
            f = dict(f)
            f["_score"] = 1.0
            f["_match"] = "pin"
            pinned.append(f)
        pinned.sort(key=lambda x: str(x["id"]))
        picked, _ = _take_budget(pinned + scored, budget)
        picked = picked[:k]
        STATS["vec"] += len(vec)
        if picked:
            STATS["ok"] += 1
            _mark("ok", vec=sum(1 for f in picked if f.get("_match") == "vec"))
        else:
            STATS["no_hit"] += 1
            _mark("no_hit")
        if picked and track:
            # track=False 是给 `retrieve_for_turn` 用的：它会把热 + 冷合起来写一次，
            # 多写一遍就多一次 WAL 提交（Windows 上是十毫秒级），而这段正好压在
            # 首字延迟上。
            try:
                mark_used(uid, [f["id"] for f in picked])
            except Exception:  # noqa: BLE001
                _note_error()
        return picked
    except Exception:  # noqa: BLE001  检索失败 = 这轮没有记忆可注入
        _note_error()
        _mark("error")
        return []
