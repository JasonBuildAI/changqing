"""L2 会话纪要：id 生成、追加、读取、渲染。

纪要**不是事实**（没有回引、没有槽位），它走单独的 `summaries` 表与单独的 md
文件；把它和 facts 混在一起，「每条事实都能追回原话」这条边界读起来就不清楚了。

但它和事实一样**不可再生**（模型生成，重算复现不出来），所以走操作日志、
由物化视图读出来，而不是直接写表：表是派生层，`rebuild()` 会连库一起删。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from . import index
from .ops import append_op
from .paths import summary_path


def _summary_id(text: str, day: str) -> str:
    """纪要 id 由**内容**决定，跨进程也稳定。

    不用内置 `hash()`：它按 PYTHONHASHSEED 随机化 —— 同一个进程里稳定，
    换个进程就变一个值。那样同一段纪要会以两个 id 落库（`INSERT OR REPLACE`
    去不掉重），日志重放还会在库里堆出一串副本。
    """
    h = hashlib.sha1(f"{day}\n{text}".encode()).hexdigest()[:12]
    return f"S-{(day or '').replace('-', '')}-{h}"


def append_summary(uid: str, text: str, day: str, turn_ref: str = "") -> str:
    """追加一段 L2 会话纪要，返回它的 id（空文本返回 ""）。"""
    text = str(text or "").strip()
    if not text:
        return ""
    sid = _summary_id(text, day)
    append_op(
        uid,
        {
            "op": "SUMMARY",
            "id": sid,
            "day": day,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "text": text,
            "turn_ref": str(turn_ref or ""),
        },
    )
    index.materialize(uid)
    render_summaries_md(uid)
    return sid


def list_summaries(uid: str, limit: int = 20) -> list[dict[str, Any]]:
    con = index.open_index(uid)
    try:
        # 按「对话发生在哪天」倒序，同一天再按生成时间 —— id 是内容哈希，
        # 只按 id 排的话同一天里的先后就乱了。
        rows = con.execute(
            "SELECT * FROM summaries ORDER BY day DESC, created_at DESC, id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def render_summaries_md(uid: str, limit: int = 500) -> Path | None:
    """把纪要渲染成人类可读的 md。**只读产物**（和 `facts.md` 一样）。

    渲染视图而不是「追加写」：整理重试（前一段失败、整段重跑）会在文件里堆出
    重复段落，而库里因为 id 稳定只有一条 —— 两边对不上。渲染没有这个问题：
    文件永远是当前库内容的一个投影。
    """
    rows = list_summaries(uid, limit=limit)
    lines = [
        "# 会话纪要（渲染视图，只读）",
        "",
        "> 这是从 log.jsonl 物化出来的视图，**不要手工编辑** —— 下次渲染会覆盖它。",
        "",
    ]
    if not rows:
        lines += ["（还没有纪要。它由后台整理从一段对话里总结出来。）", ""]
    for r in reversed(rows):  # 时间正序：读起来就是对话先后
        lines.append(f"## {str(r.get('day') or '').strip()} {r['id']}".rstrip())
        lines.append("")
        lines.append(str(r.get("text") or ""))
        if r.get("turn_ref"):
            lines.append("")
            lines.append(f"- 从 {r['turn_ref']} 起")
        lines.append("")
    p = summary_path(uid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        return p
    except OSError:
        return None
