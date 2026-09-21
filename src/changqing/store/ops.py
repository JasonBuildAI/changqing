"""操作日志：追加、读回、事实 id 分配、重放到物化视图。

`log.jsonl` 是**唯一不可再生**的那一层，「改不是改，是再追加一条操作」这条语义
全部落在这个文件里；而 `apply_op` 是它的另一半 —— 把日志投影成当前状态。
两者必须一起读才看得懂，所以同住一个文件。

依赖方向：`index` 只能在函数体内 import。`index.py` 在模块级就要
`from .ops import apply_op`（`materialize` / `rebuild` 要用），若这里也在模块级
`from . import index`，先加载哪一个都会拿到半初始化的另一半。把这一侧的 import
放进函数体，环就断在「调用时」而不是「导入时」。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from .paths import _load_state, _lock, _save_state, log_path


def append_op(uid: str, op: dict[str, Any]) -> None:
    """往日志追加一条操作。单行完整 + flush + fsync，语义与 L0 完全一致。

    调用方负责决定 op 的内容；这里只保证「要么完整落盘，要么是能被丢掉的残行」。
    """
    rec = dict(op)
    rec.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _lock(uid):
        p = log_path(uid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def read_ops(uid: str) -> list[dict[str, Any]]:
    """读回全部操作。坏行（含被截断的尾行）直接跳过，不抛异常。

    日志是「丢了就没了」的那个文件，读它的时候更不能因为一行坏了就整份打不开。
    """
    p = log_path(uid)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue  # 残行 / 坏行：当它不存在
        if isinstance(rec, dict) and rec.get("op"):
            out.append(rec)
    return out


def next_fact_id(uid: str) -> str:
    """分配一个事实 id：`F-` + 递增，**永不复用**。

    先占号再写日志（同 L0 的轮次 id）：崩溃最多浪费一个号，绝不会重号。
    id 是回引与「这条凭什么是真的」的锚点，重号比断号严重得多。
    """
    with _lock(uid):
        st = _load_state(uid)
        n = int(st.get("next_fact_seq") or 1)
        st["next_fact_seq"] = n + 1
        _save_state(uid, st)
    return f"F-{n:04d}"


def apply_op(con: sqlite3.Connection, op: dict[str, Any]) -> None:
    """把一条操作应用到物化视图。

    必须**幂等**：同一段日志重放两次结果要一样 —— 唯一事实源是日志，
    重放是常态而不是异常。
    """
    from . import index  # 函数内 import：模块级会成环，见文件头说明

    kind = str(op.get("op") or "").upper()
    fid = op.get("id")
    day = str(op.get("ts") or "")[:10] or time.strftime("%Y-%m-%d")

    if kind in ("ADD", "SUPERSEDE"):
        if not fid:
            return
        if kind == "SUPERSEDE" and op.get("replaces"):
            index._invalidate(con, str(op["replaces"]), day, f"被 {fid} 取代")
        index._write_fact(con, str(fid), op)
    elif kind == "EDIT":
        if not fid:
            return
        sets = op.get("set")
        if not isinstance(sets, dict):
            return
        cur = con.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone()
        if cur is None:
            return
        # **`cur.keys()` 不能写成 `for k in cur`**：`sqlite3.Row` 迭代出来的是
        # **值**不是列名，`cur[k]` 会拿一个值当索引去查，抛 `IndexError`。
        # 而 `materialize` 把一条坏 op 隔离成 `bad_ops` 里的一行就继续走 ——
        # 于是「编辑一条记忆」变成了「编辑被丢掉、界面照旧报成功」。
        merged = dict(zip(cur.keys(), cur, strict=True))
        merged.update({k: v for k, v in sets.items() if k in index._FACT_FIELDS})
        index._write_fact(con, str(fid), merged)
    elif kind == "INVALIDATE":
        index._invalidate(
            con, str(fid or ""), str(op.get("valid_to") or day), str(op.get("reason") or "")
        )
    elif kind == "PIN":
        con.execute("UPDATE facts SET pinned=? WHERE id=?", (int(bool(op.get("pinned"))), fid))
    elif kind == "CONFIRM":
        status = str(op.get("status") or "active")
        if status == "active":
            # 「确认这条是真的」也包括把**有效期放回来**：界面上那条「恢复」
            # 走的就是这里。只改 status 的话 valid_to 还挂着，检索层照样把它
            # 挡在外面 —— 用户点了恢复，界面上却什么都没发生。
            con.execute("UPDATE facts SET status='active', valid_to=NULL WHERE id=?", (fid,))
        else:
            con.execute("UPDATE facts SET status=? WHERE id=?", (status, fid))
    elif kind == "FORGET":
        mode = str(op.get("mode") or "archive").lower()
        if mode == "purge":
            con.execute("DELETE FROM facts WHERE id=?", (fid,))
            con.execute("DELETE FROM vectors WHERE fact_id=?", (fid,))
        else:
            con.execute("UPDATE facts SET status='forgotten', valid_to=? WHERE id=?", (day, fid))
    elif kind == "MERGE":
        # 巩固期把重复的合成一条：被并的那条标 merged 并指向胜者，**不删** ——
        # 「她合并了两条记忆」和「她丢了一条记忆」是两回事，后者要能解释
        # （误伤要可恢复）。
        #
        # 「指向胜者」的指针**在日志里**（op 的 merged_into），facts 表没有
        # merged_into 这一列。曾经图省事把它写进 object，结果是内容字段里躺着
        # 一个 id（渲染成「他养的猫叫F-0003」），而且 rebuild 重放日志时会把
        # 这份脏数据再放一遍。要查两条的合并关系就读日志。
        con.execute(
            "UPDATE facts SET status='merged', valid_to=? WHERE id=? AND status='active'",
            (day, fid),
        )
    elif kind == "DEROGATE":
        # 降权：改的是 importance，不改内容。它**会**进日志 —— 因为 importance
        # 是排序依据，不可审计的降权等于悄悄改变她的记忆权重。
        con.execute(
            "UPDATE facts SET importance=? WHERE id=?", (float(op.get("importance") or 0.0), fid)
        )
    elif kind == "SUMMARY":
        # L2 会话纪要：它**不是事实**（没有回引、没有槽位），所以进 summaries 表，
        # 绝不混进 facts —— 混进去的话「每条事实都能追回原话」就保不住了。
        # 但它和事实一样不可再生（模型生成，重算复现不出来），所以要进日志；
        # 只写表的话 rebuild() 会把库整个删掉重建，纪要就没了。
        if not fid:
            return
        con.execute(
            "INSERT OR REPLACE INTO summaries(id, day, created_at, text, turn_ref) "
            "VALUES (?,?,?,?,?)",
            (
                str(fid),
                op.get("day") or day,
                op.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                op.get("text") or "",
                op.get("turn_ref") or "",
            ),
        )
    elif kind == "TOPIC":
        # 主动话题：与纪要同类 —— 不是事实，但不可再生，所以要进日志；
        # 只写表的话 rebuild() 删库重建时它就没了，于是「她下次想主动提的事」
        # 整丢，而不报错。
        # `used` 不写在这条里：它是派生层的排序状态，不是内容。
        if not fid:
            return
        con.execute(
            "INSERT OR REPLACE INTO topics(id, day, created_at, kind, text, "
            "due_day, ref) VALUES (?,?,?,?,?,?,?)",
            (
                str(fid),
                op.get("day") or day,
                op.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
                op.get("kind") or "share",
                op.get("text") or "",
                op.get("due_day") or "",
                op.get("ref") or "",
            ),
        )
