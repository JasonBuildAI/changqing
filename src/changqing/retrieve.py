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

import time
import traceback
from collections.abc import Sequence
from typing import Any

from .config import MemoryConfig
from .runtime import runtime
from .store import open_index
from .tokens import estimate_tokens

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
