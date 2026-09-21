"""主动话题：她下次想主动找他聊什么。

它与 facts / summaries 的区别要说清楚，因为**混层会毁掉一条不变式**：

    facts      有槽位、有回引（turn_ref + quote），可被引用
    summaries  一段对话的纪要，没有回引，但记「发生过什么」
    topics     整理时**顺带**生成的「值得由她主动提起」的条目 ——
               它是**计划**，不是记录。所以它既不该被当成事实回引，
               也不该混进 summaries（那会让「她记得你们的故事」与
               「她想找你聊」这两件事在同一个表里说不清）。

为什么需要它：主动开口如果只有一句「你可以自己起个头」，模型手里有记忆却没有
**挑哪一件来说**的判断，于是产出「在吗 / 最近怎么样」这种空转开场 ——
用户评的是「跟她说话不像跟真人」。挑话题需要一次模型调用，而它**绝不能出现在
首字路径上**，所以放在后台整理里顺手做掉。

落盘形态与纪要一致：内容不可再生（模型生成），走操作日志（`TOPIC`），
由物化视图读出来。唯一不进日志的是 `used` 标记（它是排序状态，不是内容）。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from . import index
from .ops import append_op
from .paths import _lock

# 主动开口的三种意图。它只用来让面板看得出「她当时打算干什么」，
# 不参与任何检索与打分 —— 加新值不会有任何代码认得它。
TOPIC_KINDS = ("followup", "share", "ask")

_TOPICS_FILE = "topics.md"


def topics_path(uid: str) -> Path:
    return index.index_path(uid).with_name(_TOPICS_FILE)


def _topic_id(text: str, day: str) -> str:
    """id 由**内容**决定，跨进程也稳定（同 `_summary_id` 的理由）。"""
    h = hashlib.sha1(f"{day}\n{text}".encode("utf-8")).hexdigest()[:10]
    return f"TP-{(day or '').replace('-', '')}-{h}"


def append_topic(uid: str, text: str, day: str, *, kind: str = "share",
                 due_day: str = "", ref: str = "") -> str:
    """追加一条「她下次想主动提的事」。返回 id（空文本返回 ""）。

    去重靠 id：同一段对话整理两次，内容一样 → 同一条（`INSERT OR REPLACE`），
    这正是幂等要的样子。
    """
    text = str(text or "").strip()
    if not text:
        return ""
    kind = str(kind or "share").strip().lower()
    if kind not in TOPIC_KINDS:
        kind = "share"                 # 没认出的意图不丢条目，只退回中性那一档
    tid = _topic_id(text, day)
    append_op(uid, {"op": "TOPIC", "id": tid, "day": day,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "kind": kind, "text": text,
                    "due_day": str(due_day or "")[:10], "ref": str(ref or "")})
    index.materialize(uid)
    render_topics_md(uid)
    return tid


def list_topics(uid: str, *, limit: int = 8, include_used: bool = False) -> list[dict[str, Any]]:
    """她攒着想找他说的事，按「有日期的先、再按生成的日子」排。

    有 `due_day` 的排前面：那条通常是**他这几天要面对的事**（面试、体检、出差），
    过了那几天再说就没意义了；`due_day` 空的按生成日期倒序 —— 新的更像
    「刚想起的」。
    """
    con = index.open_index(uid)
    try:
        sql = "SELECT * FROM topics"
        args: list[Any] = []
        if not include_used:
            sql += " WHERE COALESCE(used,0)=0"
        sql += (" ORDER BY CASE WHEN COALESCE(due_day,'')='' THEN 1 ELSE 0 END ASC, "
                "COALESCE(due_day,'9999') ASC, day DESC, created_at DESC, id ASC "
                "LIMIT ?")
        args.append(max(1, int(limit)))
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def mark_topics_used(uid: str, topic_ids: list[str]) -> None:
    """说过了就标上。它是**排序状态**，不是内容，所以不进日志
    （丢了只会让她再提一次，不会让记忆失真）。"""
    if not topic_ids:
        return
    con = index.open_index(uid)
    try:
        con.executemany("UPDATE topics SET used=1 WHERE id=?",
                        [(t,) for t in topic_ids])
        con.commit()
    finally:
        con.close()


def reset_topics(uid: str, *, wipe: bool = False) -> int:
    """清掉她攒着的话题。

    `wipe=False`（默认）只把未用的标成用过（「这一轮别再提了」）；
    `wipe=True` 连表一起清空（用户主动重置时用）。返回影响的行数。
    """
    con = index.open_index(uid)
    try:
        with con:
            if wipe:
                cur = con.execute("DELETE FROM topics")
            else:
                cur = con.execute("UPDATE topics SET used=1 WHERE COALESCE(used,0)=0")
        return int(cur.rowcount or 0)
    finally:
        con.close()


def topic_stats(uid: str) -> dict[str, Any]:
    con = index.open_index(uid)
    try:
        row = con.execute(
            "SELECT count(*) AS total, "
            "sum(CASE WHEN COALESCE(used,0)=0 THEN 1 ELSE 0 END) AS open "
            "FROM topics").fetchone()
        return {"total": int(row["total"] or 0), "open": int(row["open"] or 0)}
    finally:
        con.close()


def render_topics_md(uid: str, limit: int = 200) -> str | None:
    """把话题渲染成人类可读的 md。**只读产物**。

    它对人有用是因为「她这次主动开口是编的还是真有依据」需要能一眼对出来 ——
    渲染视图在这里不是装饰，是「她主动说的事有没有出处」的检查面板。
    """
    rows = list_topics(uid, limit=limit, include_used=True)
    lines = ["# 她想主动提的事（渲染视图，只读）", "",
             "> 这是从 log.jsonl 物化出来的视图，**不要手工编辑** —— 下次渲染会覆盖它。",
             "> 它由后台整理顺手生成：不是她说过的话，也不是事实，是**她打算找你聊什么**。",
             ""]
    if not rows:
        lines += ["（还没有话题。它随一次后台整理生成，也可能这次就没有值得提的事。）", ""]
    for r in rows:
        marks = [r["id"], str(r.get("kind") or "share")]
        if int(r.get("used") or 0):
            marks.append("已提过")
        if r.get("due_day"):
            marks.append(f"{r['due_day']} 之前")
        lines.append(f"- {str(r.get('day') or '')[:10]} {r['text']}  ({' · '.join(marks)})")
    lines.append("")
    p = topics_path(uid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        return str(p)
    except OSError:
        return None

