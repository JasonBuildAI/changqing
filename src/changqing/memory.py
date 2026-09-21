"""门面：`Memory` —— 一个聊天对象（一个 uid）的记忆。

    from changqing import Memory, MemoryConfig
    from changqing.adapters.mock import MockEmbedder, MockLLM

    mem = Memory("user-1", config=MemoryConfig(root="./data"),
                 embedder=MockEmbedder(), llm=MockLLM())
    mem.remember({"user": "我家猫叫团子", "assistant": "记住了，团子。"})
    mem.recall("我家猫叫什么？")

**它是句柄，不是全局单例。** 一个进程里可以有很多个 `Memory`，每个绑一个 uid；
它们共用同一份配置与同一组注入的能力（见 `runtime`）。要「同一个进程里两个库
各用各的配置」，就各自传 `config` —— 这正是构造器上那些参数存在的理由。

**能力在**调用时**解析，不在构造时**。构造完之后宿主再 `configure(llm=...)`，
已经建好的句柄一样看得到。构造时把运行期拷成一份的话，`Memory` 与
`runtime()` 就成了两个真源，而它们不一致时**不报错** —— 表现是「我明明配好了，
它就是不用」。所以这里只存**显式的覆盖**，每次调用现拼。

**接口分两套名字，指向同一件事。**

    remember / recall / forget        这套是这套库自己的说法
    add / search / delete             mem0 风格的那套，纯别名
    get / get_all / update / history  同一族的读改查

两套并存是**故意**的：写惯了 mem0 的人不必先学一套新词。但它们必须是
**同一份实现**（别名就是别名，不是第二份实现），否则「同一个动作两个名字」
会慢慢长成两种行为 —— 而它们不一致时不报错，只在对账的时候才发现。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .config import MemoryConfig
from .edit import EDITABLE_FIELDS, EditError, validate
from .persona import PersonaProfile
from .ports import LLM, Embedder, UsageSink
from .runtime import Runtime, runtime, using
from .store import (
    append_op,
    append_summary,
    append_topic,
    append_turn,
    get_fact,
    list_facts,
    list_summaries,
    list_topics,
    materialize,
    pending_facts,
    read_ops,
    render_facts_md,
    reset_memory,
    topic_stats,
)


class Memory:
    """一个 uid 的记忆句柄。"""

    __slots__ = ("_overrides", "_overrides_ready", "uid")

    def __init__(
        self,
        uid: str,
        *,
        config: MemoryConfig | None = None,
        persona: PersonaProfile | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        usage: UsageSink | None = None,
    ) -> None:
        self.uid = str(uid or "")
        # 只存**写下来的**那些；None 一个都不存（它的意思是「跟着 runtime 走」）。
        # 存下 None 的话，`configure()` 之后再补的能力就会被这里的空值盖住。
        self._overrides: dict[str, Any] = {}
        for name, value in (
            ("config", config),
            ("persona", persona),
            ("embedder", embedder),
            ("llm", llm),
            ("usage", usage),
        ):
            if value is not None:
                self._overrides[name] = value
        self._overrides_ready = bool(self._overrides)

    # ---------------------------------------------------------------- 装配
    def effective(self) -> Runtime:
        """这个句柄当下实际生效的运行期。**每次现拼**，见模块 docstring。"""
        base = runtime()
        if not self._overrides_ready:
            return base
        return replace(base, **self._overrides)

    @contextmanager
    def activated(self):
        """进入「这个句柄的配置生效」的作用域。

        内部方法都套着它。它同时保证**并发安全**：`using()` 换的是进程级的那一份，
        而作用域结束一定会还原 —— 不套的话，这个句柄的配置会漏给别的线程。
        """
        with using(self.effective()):
            yield self

    def __repr__(self) -> str:
        return f"<Memory uid={self.uid!r}>"

    # ---------------------------------------------------------------- L0 写入
    def remember(self, turn: dict[str, Any]) -> list[str]:
        """追加一轮原话（L0），返回写进去的轮次 id。

        原话是**唯一不可再生**的那一层：事实能从它抽出来重建，它自己不能。
        所以这里只追加，不修改、不合并。

        写不动也不抛：调用方通常在「刚说完一句话」的那条路上，那里的一次
        磁盘异常不该让整个对话崩掉。返回空列表就是「这轮没记下来」。
        """
        if not self.uid:
            return []
        with self.activated() as mem:
            if not mem.effective().config.enabled:
                return []
            try:
                return append_turn(mem.uid, turn)
            except Exception:  # noqa: BLE001  记忆写不动也得能说话
                return []

    def add(self, messages: Any, *, ts: float | None = None) -> list[str]:
        """`remember` 的别名，输入更宽：一轮 dict / 一段话 / mem0 那种消息列表。

        只有**在用户与助手之间**切分：一段用户独白里的换行不该被当成两轮，
        而她连着发三条消息也不该被并成一轮（原话的边界在这里并掉就再也查不回来）。
        """
        out: list[str] = []
        for turn in _turns(messages, ts):
            out.extend(self.remember(turn))
        return out

    # ---------------------------------------------------------------- 检索
    def recall(
        self,
        query: str,
        *,
        budget_tokens: int = 0,
        timeout_ms: int = 0,
        k: int = 0,
    ) -> list[dict[str, Any]]:
        """按 query 检索事实。**绝不抛异常**，超时或分数不够就是空列表。

        `budget_tokens` / `timeout_ms` 传 0 表示「用配置里的默认值」——
        这个「0 等于默认」的约定要留着（`test` 里有钉住它的用例）：显式传
        `budget_tokens=0` 的人想要的是「按默认预算裁」，不是「一条都不给」。
        """
        if not self.uid:
            return []
        from .retrieve import search

        cfg = self.effective().config
        with self.activated():
            return search(
                self.uid,
                query,
                k=int(k) or None,
                budget_tokens=int(budget_tokens) or None,
                timeout_ms=int(timeout_ms or cfg.recall_ms),
            )

    # mem0 叫这个
    def search(
        self,
        query: str,
        *,
        budget_tokens: int = 0,
        timeout_ms: int = 0,
        k: int = 0,
    ) -> list[dict[str, Any]]:
        """`recall` 的别名。"""
        return self.recall(query, budget_tokens=budget_tokens, timeout_ms=timeout_ms, k=k)

    def context(
        self,
        query: str,
        *,
        recent_refs: list[str] | None = None,
        proactive: bool = False,
    ) -> dict[str, Any]:
        """一轮对话要注入的**全部**素材：热路径 + 冷路径 + 故事线 + 话题。

        与 `recall` 的分工：`recall` 回答「跟这句话有关的事有哪些」，
        `context` 回答「这一轮该把什么摆到她面前」。前者是后者的一个输入。

        `recent_refs` 是「刚刚这两轮说过的话」对应的轮次 id —— 刚说过的事还在
        上下文里，冷路径不该把它再捞一遍当成一条「记忆」注入。
        """
        from .retrieve import retrieve_for_turn

        sess: dict[str, Any] = {"recent": list(recent_refs or [])}
        with self.activated():
            return retrieve_for_turn(self.uid, query, sess, proactive=proactive)

    # ---------------------------------------------------------------- 读
    def get(self, fact_id: str) -> dict[str, Any] | None:
        """取一条事实（含已经失效的）—— 「这条到底还在不在」要能问。"""
        with self.activated():
            return get_fact(self.uid, str(fact_id))

    def get_all(self, *, include_dead: bool = True) -> list[dict[str, Any]]:
        """列出全部事实。

        默认**含已失效的**（与 `recall` 相反）：这个方法是给「导出 / 审计 /
        面板」用的，那里最怕的是「明明发生过，列表里没有」。
        """
        with self.activated():
            return list_facts(self.uid, include_dead=bool(include_dead))

    def pending(self) -> list[dict[str, Any]]:
        """回引只近似命中的那批，等人点一下「是这样吗」。"""
        with self.activated():
            return pending_facts(self.uid)

    def summaries(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.activated():
            return list_summaries(self.uid, limit=int(limit))

    def topics(self, *, limit: int = 8, include_used: bool = False) -> list[dict[str, Any]]:
        """她攒着想主动提起的事。`include_used=True` 才是「连提过的也给我」。"""
        with self.activated():
            return list_topics(self.uid, limit=int(limit), include_used=bool(include_used))

    def history(self, fact_id: str = "") -> list[dict[str, Any]]:
        """读**操作日志**：这条（或全部）事实被怎么写、改、忘过。

        走日志而不是物化视图：日志是唯一事实源、只追加，物化视图可以重建。
        「她为什么变成这样记得 / 不记得」只有日志答得出来。
        不传 id 就是全部操作，按发生顺序。
        """
        with self.activated():
            ops = read_ops(self.uid)
        if not fact_id:
            return ops
        want = str(fact_id)
        return [op for op in ops if str(op.get("id") or "") == want]

    # ---------------------------------------------------------------- 写 / 改
    def update(self, fact_id: str, **fields: Any) -> dict[str, Any] | None:
        """改一条事实。**追加一条 EDIT 操作**，不直接改派生物。

        白名单与校验见 `edit` —— 这里只负责「先校验、再追加、再重建派生层」。
        顺序不能换：坏值必须在**写下任何东西之前**被挡住，否则那条坏操作会
        永久卡在这个用户的重放上（见 `edit` 的 docstring）。

        改完立刻重建向量：不重建的话，要等到下次后台整理才生效，而中间这段时间
        检索按**旧内容**召回这条事实 —— 用户看到的是「她记着我上次说的那句」。
        """
        if fact_id is None:
            raise EditError("缺少事实 id")
        sets = validate(fields)
        if not sets:
            raise EditError(f"没有可改的字段（只能改 {sorted(EDITABLE_FIELDS)}）")
        if self.get(fact_id) is None:
            return None
        with self.activated():
            append_op(self.uid, {"op": "EDIT", "id": str(fact_id), "set": sets})
            materialize(self.uid)
            render_facts_md(self.uid)
            self.reindex()
            return get_fact(self.uid, str(fact_id))

    def forget(self, fact_id: str, mode: str = "") -> str:
        """删一条事实，返回真正用的那种模式（`archive` / `purge`）。

        「忘记」在人心里就是删除，所以 `purge` 删掉的是**事实行**（facts 表 +
        全文索引 + 向量）。但引文与 L0 原话仍在 —— 那是「这条凭什么是真的」的
        证据，真要连它一起清掉，走整库 `reset`。两种模式都只是往日志追加一条
        操作，由它决定物化视图怎么变：派生层从不手工改。

        **向量必须显式重建**：`FORGET` 只改 facts 那一行的状态，向量行不属于
        物化层。不重建的话，被归档的那条向量还留在库里继续占着召回名额 ——
        而冷路径的向量召回是「先按相似度取满、再按存活过滤」，它会把还活着的
        那条挤出去。
        """
        with self.activated() as mem:
            use = (mode or mem.effective().config.forget_mode or "archive").lower()
            append_op(mem.uid, {"op": "FORGET", "id": str(fact_id), "mode": use})
            materialize(mem.uid)
            render_facts_md(mem.uid)
            mem.reindex()
        return use

    # mem0 叫这个
    def delete(self, fact_id: str, mode: str = "") -> str:
        """`forget` 的别名。"""
        return self.forget(fact_id, mode)

    def confirm(self, fact_id: str, accept: bool = True) -> str:
        """确认 / 否决一条待确认的事实，返回它落到的状态。

        否决走**归档**而不是删除：用户改主意了还能找回来。
        确认时会同时清掉 `valid_to` —— 只把状态改回 active 是不够的，那条事实
        还挂着失效日期，检索层照样不会再用它。
        """
        if self.get(fact_id) is None:
            return ""
        if not accept:
            self.forget(fact_id, "archive")
            return "forgotten"
        with self.activated():
            append_op(self.uid, {"op": "CONFIRM", "id": str(fact_id), "status": "active"})
            materialize(self.uid)
            self.reindex()
        return "active"

    def pin(self, fact_id: str, pinned: bool = True) -> bool:
        """钉住 / 取消钉住。钉住的事实无条件进热路径。"""
        if self.get(fact_id) is None:
            return False
        with self.activated():
            append_op(self.uid, {"op": "PIN", "id": str(fact_id), "pinned": bool(pinned)})
            materialize(self.uid)
            render_facts_md(self.uid)
        return bool(pinned)

    def note_summary(self, text: str, day: str = "", turn_ref: str = "") -> str:
        """补一段 L2 纪要。后台整理会自己写，手工补是给「这场聊得很值」用的。"""
        with self.activated():
            return append_summary(self.uid, text, day or time.strftime("%Y-%m-%d"), turn_ref)

    def note_topic(
        self, text: str, day: str = "", *, kind: str = "share", due_day: str = ""
    ) -> str:
        """补一条「她下次想提的事」。"""
        with self.activated():
            return append_topic(
                self.uid, text, day or time.strftime("%Y-%m-%d"), kind=kind, due_day=due_day
            )

    # ---------------------------------------------------------------- 维护
    def reindex(self, *, batch: int = 0, limit: int = 0) -> dict[str, Any]:
        """把向量索引补到最新。没注入 `Embedder` 时是空操作。

        惰性 import：向量是可选层，关掉的时候连它的模块都不该被拉起来。
        """
        from .vectors import reindex

        with self.activated():
            return reindex(self.uid, batch=int(batch), limit=int(limit))

    def extract_now(self, *, call: Any = None) -> dict[str, Any]:
        """立刻整理一次（不等后台的三个触发条件）。同步、会调模型。"""
        from .worker import extract_now

        with self.activated():
            return extract_now(self.uid, call=call)

    def stats(self) -> dict[str, Any]:
        """这一份记忆的规模（轮数、事实数、向量数、话题数）。面板与自检用。"""
        from .store import fact_stats
        from .vectors import stats as vector_stats

        out: dict[str, Any] = {}
        with self.activated():
            out.update(dict(fact_stats(self.uid)))
            out["vectors"] = vector_stats(self.uid)
            out["topics"] = topic_stats(self.uid)
        return out

    def export(self) -> dict[str, Any]:
        """导出全部记忆。**从日志与事实渲染**，不是快照某个内部结构。

        操作日志一起给：它是唯一事实源，有了它就能把这份记忆完整重建一遍。
        """
        with self.activated():
            return {
                "uid": self.uid,
                "facts": list_facts(self.uid, include_dead=True),
                "pending": pending_facts(self.uid),
                "summaries": list_summaries(self.uid, limit=100),
                "topics": list_topics(self.uid, limit=100, include_used=True),
                "ops": read_ops(self.uid),
            }

    def reset(self, mode: str | None = None) -> dict[str, Any]:
        """整库重置：**这一份记忆**（不是别人的、也不是配置）。

        返回的是**实际做了什么**，包括删不掉的东西 —— Windows 上只要还有人
        开着文件句柄就会拒绝删除。把 leftover 吞掉的后果是接口回一个 `ok`，
        而事实原样留着：她照样记得。
        """
        with self.activated():
            return reset_memory(self.uid, mode)


# ---------------------------------------------------------------- 输入归一
def _turns(messages: Any, ts: float | None = None) -> list[dict[str, Any]]:
    """把 `add` 收得下的三种形状都变成轮次列表。

    · `{"user": ..., "assistant": ...}` —— 本来就是一轮，原样通过；
    · 一个字符串 —— 当成一轮用户独白（她还没回）；
    · `[{"role": ..., "content": ...}, ...]` —— 按「用户 → 助手」的边界切。

    切分只在角色切换处发生：用户连说两句是一轮（那是同一口气），
    助手连发三条也是一轮（她一轮里可以发好几条）。
    """
    if isinstance(messages, dict):
        turn = dict(messages)
        if ts is not None:
            turn["ts"] = ts
        return [turn] if (turn.get("user") or turn.get("assistant")) else []
    if isinstance(messages, str):
        return [{"ts": ts if ts is not None else time.time(), "user": messages}]

    out: list[dict[str, Any]] = []
    user: list[str] = []
    assistant: list[str] = []
    for item in messages or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            continue
        if role == "user":
            if assistant:  # 上一轮的「他说 + 她回」已经齐了，落盘
                out.append(_pair(user, assistant, ts))
                user, assistant = [], []
            user.append(text)
        else:
            assistant.append(text)
    if user or assistant:
        out.append(_pair(user, assistant, ts))
    return out


def _pair(user: list[str], assistant: list[str], ts: float | None) -> dict[str, Any]:
    turn: dict[str, Any] = {"ts": ts if ts is not None else time.time()}
    if user:
        turn["user"] = "\n".join(user)
    if assistant:
        # 保持**多条**而不是拼成一段：L0 里「她当时是分两次说的」这件事
        # 拼掉之后就永远查不回来了。
        turn["assistant"] = list(assistant) if len(assistant) > 1 else assistant[0]
    return turn


__all__ = ["Memory"]
