"""一轮对话的**工作面**：这一轮到底有什么可说出口的东西。

调用方把 `retrieve_for_turn()` 的结果挂在自己的会话对象上（键是 `CTX_KEY`），
这一层负责从那里读回来，并回答两个问题：

  1. 卡片该往哪几栏渲染（`MEMORY_SECTIONS`）；
  2. 这一轮到底**有没有素材**（`has_recall_material`）——它是门控与渲染的**唯一判据**。

**为什么这两件事必须住在一起。** 曾经它们各写一份字段清单，门控那份漏了一栏：
只有「她答应过他的事」时，门控说没料，而卡片里明明写着那条 —— 于是她拿着一句刚
答应的承诺，却说「（我还不知道你什么呢）」。同一句话只能有一处权威定义。
"""

from __future__ import annotations

from typing import Any

# 卡片分栏的**唯一清单**：(会话字段, 分组标题, 有日期时的归属说法)。
# 渲染与「算不算素材」都从这里取 —— 抄第二份的那次教训见文件头。
MEMORY_SECTIONS = (
    ("user_profile", "你对他的了解", "他说的"),
    ("shared_moments", "你们的共同经历", "你们那天聊过"),
    ("promises", "她答应过他的事", "你们那天聊过"),
    ("commitments", "约定 / 待办", "你们那天聊过"),
)
MEMORY_MATERIAL_KEYS = tuple(k for k, _, _ in MEMORY_SECTIONS)

# 本轮检索结果的落点。它不是记忆内容，是这一轮的工作面 —— 所以调用方应该把它
# 放进「瞬态字段」，别跟着会话落盘：下一轮就被整个覆盖，写进去只会让它随轮次线性变大。
CTX_KEY = "_mem"


def context_facts(sess: dict[str, Any]) -> list[dict]:
    """本轮检索出来的事实。门控、卡片渲染、指标三处都读它 —— 这是「同源」的落点。"""
    ctx = sess.get(CTX_KEY)
    if isinstance(ctx, dict):
        facts = ctx.get("facts")
        if isinstance(facts, list):
            return facts
    return []


def context_summaries(sess: dict[str, Any]) -> list[dict]:
    """本轮检索出来的故事线。与 `context_facts` 同一时刻落盘、同一份口径。"""
    ctx = sess.get(CTX_KEY)
    if isinstance(ctx, dict):
        rows = ctx.get("summaries")
        if isinstance(rows, list):
            return rows
    return []


def context_topics(sess: dict[str, Any]) -> list[dict]:
    """本轮可用的主动话题（只在**她先开口那一轮**被填进来）。

    它不是「她已经知道的事」，而是「她打算说什么」，所以不带编号进卡片、
    也不进「引用是否有效」那套对比。判「本轮有没有素材」时必须算它一份：
    漏判的结果是「她手里有开场理由，而门控说她没料」，两处各说各的。
    """
    ctx = sess.get(CTX_KEY)
    if isinstance(ctx, dict):
        rows = ctx.get("topics")
        if isinstance(rows, list):
            return rows
    return []


def memory_facts(sess: dict[str, Any]) -> dict:
    """会话里的结构化记忆（**降级通道**）。

    事实层上线之后它是兜底：抽取还没跑过这一段原话、或者整个记忆系统被关掉时，
    卡片还得有东西可渲染。
    """
    m = sess.get("memory")
    return m if isinstance(m, dict) else {}


def has_recall_material(sess: dict[str, Any]) -> bool:
    """这一轮她手里有没有「能说出口的东西」—— 门控与渲染的**唯一判据**。

    四类都算，按可信度排：本轮事实 → **故事线** → 主动话题 → 结构化记忆（降级通道）。

    后两类必须**单独判**：它们不住在会话里那一份结构化记忆里，所以
    `MEMORY_MATERIAL_KEYS` 那份字段清单**结构上**覆盖不到 —— 漏判的后果是
    「她看得见你们之间发生过什么 / 知道该找他聊什么，而门控说她没料」。
    """
    if context_facts(sess) or context_summaries(sess) or context_topics(sess):
        return True
    m = memory_facts(sess)
    return any(bool(m.get(k)) for k in MEMORY_MATERIAL_KEYS)


def has_any_material(uid: str) -> bool:
    """这个人的库里到底有没有「能说出口的东西」（事实 / 纪要 / 话题任一即算）。

    为什么不用 `has_recall_material(sess)`：主动开口的闸门跑在检索**之前**，
    那时工作面上还是上一轮的东西（而且它不落盘，进程重启就是空的）—— 拿它当判据的
    结果是「她永远不开口」，而且看起来像功能坏了。所以这里直查库。

    三条都是 `LIMIT 1` 的索引查询；它跑在一轮对话的开始，而那时本来就要开库。
    库都不存在（新用户）时直接 False —— 不为了问一句话而顺手把目录与 schema 建出来。

    查库报错时**放行**（返回 True）：一次探针异常就永久不让她开口，比偶尔一次
    没话找话难查得多。

    本库从中解耦出来的那份宿主应用，2026-09-21 起把**主动开口的内容闸**选成了
    这一条：话题清单太稀（整理一天一两条），以「手里有没用过的话题」为条件等于
    把功能关掉 —— 实测那个号话题已空而事实层有料，主动开口一次都没发生过。
    「话题优先」改由 prompt 承担：有可用话题时那张清单照旧注入，用尽了才退到
    事实与纪要。**挑哪一条是宿主的决定**，库只保证两条都能用（窄的那条见
    `has_openable_topic`）。
    """
    from .runtime import runtime
    from .store import index_path, open_index

    if not runtime().config.enabled or not uid:
        return False
    try:
        if not index_path(uid).exists():
            return False
        con = open_index(uid)
        try:
            for sql in (
                "SELECT 1 FROM facts WHERE status='active' "
                "AND (valid_to IS NULL OR valid_to='') LIMIT 1",
                "SELECT 1 FROM summaries LIMIT 1",
                "SELECT 1 FROM topics LIMIT 1",
            ):
                if con.execute(sql).fetchone() is not None:
                    return True
            return False
        finally:
            con.close()
    except Exception:  # noqa: BLE001  探针坏了按宽松那边走
        return True


def has_openable_topic(uid: str) -> bool:
    """她手里有没有**能主动开一次口**的东西：话题清单里还有没用过的那条。

    这道闸比 `has_any_material` 更窄：事实与纪要说的是「她记得他」，但它们不是
    「能起个头的话题」；只有话题清单里那条没用过的（`list_topics` 默认就滤掉用过的）
    才撑得起一句具体的、跟他有关的话。手里没话题就只能空转打招呼 —— 那串
    「你好」「你好」就是这么来的。

    它是**给宿主的一个可选项，不是唯一正确的闸**：话题清单很稀，拿它当默认条件
    容易变成「她永远不开口」（原始宿主应用实测如此，后来换用了 `has_any_material`）。
    想让开场一定贴着具体话题的宿主仍然可以用它 —— 用哪一条由宿主自己定。

    与 `has_any_material` 同样的道理直查库而不是读工作面（那道闸跑在检索之前）。
    查库报错仍然放行。
    """
    from .runtime import runtime
    from .store import index_path, list_topics

    if not runtime().config.enabled or not uid:
        return False
    try:
        if not index_path(uid).exists():
            return False
        return bool(list_topics(uid, limit=1))
    except Exception:  # noqa: BLE001  探针坏了按宽松那边走
        return True


__all__ = [
    "CTX_KEY",
    "MEMORY_MATERIAL_KEYS",
    "MEMORY_SECTIONS",
    "context_facts",
    "context_summaries",
    "context_topics",
    "has_any_material",
    "has_openable_topic",
    "has_recall_material",
    "memory_facts",
]
