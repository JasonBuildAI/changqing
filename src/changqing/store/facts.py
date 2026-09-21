"""事实的增删查改：列出、待确认、单条、标记使用、计数。

这一层回答的是「**现在**哪些事实有效」，是检索层与面板真正消费的那一面。
它与「日志怎么重放成表」是两种问题，混在一起时「为什么这条查不出来」
要同时怀疑查询与物化两处。

依赖方向：函数内惰性 import `index`。这里不能写模块级
`from .index import open_index` —— `index.py` 模块级要用 `ops.apply_op`，
而 `ops` 又要 `index._write_fact`，成环。
"""

from __future__ import annotations

import time
from typing import Any


def _fact_text_index(fact: dict[str, Any]) -> str:
    from ..tokenize import to_index

    return to_index(" ".join(str(fact.get(k) or "") for k in ("subject", "predicate", "object")))


def list_facts(
    uid: str, *, status: str | None = None, include_dead: bool = False
) -> list[dict[str, Any]]:
    """列出事实。默认只给**当前有效**的（active 且 `valid_to` 为空）。"""
    from . import index

    con = index.open_index(uid)
    try:
        sql = "SELECT * FROM facts"
        args: list[Any] = []
        where = []
        if status:
            where.append("status=?")
            args.append(status)
        if not include_dead and not status:
            where.append("status='active'")
            where.append("(valid_to IS NULL OR valid_to='')")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY pinned DESC, importance DESC, persona_attention DESC, id ASC"
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def pending_facts(uid: str) -> list[dict[str, Any]]:
    """待确认的事实（回引近似命中的那批）。界面要给用户一个「是不是这样」的入口。"""
    return list_facts(uid, status="pending")


def get_fact(uid: str, fact_id: str) -> dict[str, Any] | None:
    from . import index

    con = index.open_index(uid)
    try:
        row = con.execute("SELECT * FROM facts WHERE id=?", (fact_id,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def mark_used(uid: str, fact_ids: list[str]) -> None:
    """记一次「这条被用上了」。

    `used` 统计只影响派生层的排序，**不进日志** —— 它不是「记忆内容」，
    丢了也不影响可审计性。
    """
    if not fact_ids:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    from . import index

    con = index.open_index(uid)
    try:
        con.executemany(
            "UPDATE facts SET last_used_at=?, use_count=use_count+1 WHERE id=?",
            [(now, f) for f in fact_ids],
        )
        con.commit()
    finally:
        con.close()


def fact_stats(uid: str) -> dict[str, Any]:
    """事实层的计数，给面板与自检用。"""
    from . import index

    con = index.open_index(uid)
    try:
        row = con.execute(
            "SELECT count(*) AS total, "
            "sum(CASE WHEN status='active' AND (valid_to IS NULL OR valid_to='') "
            "THEN 1 ELSE 0 END) AS live, "
            "sum(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending "
            "FROM facts"
        ).fetchone()
        return {
            "total": int(row["total"] or 0),
            "live": int(row["live"] or 0),
            "pending": int(row["pending"] or 0),
            "applied_ops": int(index._meta_get(con, "applied_ops", "0") or 0),
        }
    finally:
        con.close()
