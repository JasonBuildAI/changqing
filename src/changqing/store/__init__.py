"""存储层：L0 原话 + 操作日志 + 可重建的事实物化视图 + 派生内容。

四层的关系是这个包的全部重点：

    L0 原话   sessions/YYYY-MM-DD.md      只追加    **不可再生**
              （超期只归档到 sessions.archive/YYYY-MM-DD.md.gz，
                一个字节都不删；见 turnlog.archive_old_turns）
    log.jsonl                             只追加    **不可再生**
    index.sqlite                          物化      可重建
    facts.md / summaries.md / topics.md   渲染      可重建

落盘形态（`root` 见 `MemoryConfig.root`，默认 `~/.changqing`）：

    {root}/{uid[:2]}/{uid}/sessions/YYYY-MM-DD.md
    {root}/{uid[:2]}/{uid}/sessions.archive/YYYY-MM-DD.md.gz
    {root}/{uid[:2]}/{uid}/log.jsonl        # 操作日志（唯一事实源）
    {root}/{uid[:2]}/{uid}/index.sqlite     # 物化视图（删了能重建）
    {root}/{uid[:2]}/{uid}/facts.md         # 人类可读渲染（只读产物）
    {root}/{uid[:2]}/{uid}/state.json       # 轮次/事实序号 + 整理游标

**包内 import 一律走「调用时」**：子模块之间互相依赖（index ↔ ops ↔ facts），
模块级互相 import 会成环，所以反向的那一侧都放在函数体内。详见各子模块的
docstring —— 那里的说明不是考古，是「别把它挪回模块级」。
"""

from .facts import fact_stats, get_fact, list_facts, mark_used, pending_facts
from .index import SCHEMA_VERSION, connect, ensure_schema, materialize, open_index, rebuild, wipe
from .ops import append_op, apply_op, next_fact_id, read_ops
from .paths import (
    day_path,
    facts_md_path,
    index_path,
    load_state,
    log_path,
    mutate_state,
    root_dir,
    safe_name,
    save_state,
    sessions_dir,
    summary_path,
    update_state,
    user_dir,
    watermark,
)
from .render import enabled as render_enabled
from .render import render_facts_md
from .summaries import append_summary, list_summaries, render_summaries_md
from .topics import (
    TOPIC_KINDS,
    append_topic,
    list_topics,
    mark_topics_used,
    render_topics_md,
    reset_topics,
    topic_stats,
    topics_path,
)
from .turnlog import (
    append_turn,
    archive_old_turns,
    archive_stats,
    escape_text,
    read_turns,
    remember,
    reset_memory,
    stats,
    unescape_text,
)

__all__ = [
    # paths / state
    "root_dir",
    "user_dir",
    "sessions_dir",
    "day_path",
    "safe_name",
    "log_path",
    "index_path",
    "facts_md_path",
    "summary_path",
    "topics_path",
    "load_state",
    "save_state",
    "update_state",
    "mutate_state",
    "watermark",
    # L0
    "append_turn",
    "remember",
    "read_turns",
    "reset_memory",
    "stats",
    "escape_text",
    "unescape_text",
    "archive_old_turns",
    "archive_stats",
    # L1 日志与物化
    "append_op",
    "read_ops",
    "apply_op",
    "next_fact_id",
    "connect",
    "open_index",
    "ensure_schema",
    "materialize",
    "rebuild",
    "wipe",
    "SCHEMA_VERSION",
    # 事实
    "list_facts",
    "pending_facts",
    "get_fact",
    "mark_used",
    "fact_stats",
    # L2 纪要
    "append_summary",
    "list_summaries",
    "render_summaries_md",
    # 话题
    "TOPIC_KINDS",
    "append_topic",
    "list_topics",
    "mark_topics_used",
    "reset_topics",
    "topic_stats",
    "render_topics_md",
    # 渲染
    "render_facts_md",
    "render_enabled",
]
