"""核心数据形状。

这里只放**形状**，不放行为。它们都是 `TypedDict` 而不是 dataclass，
因为存储层从头到尾按 `dict` 在传（SQLite 的 `row_factory` 直接给 dict、
日志是 JSONL），转成 dataclass 只会增加一层无谓的往返。
形状写在这里的价值是：唯一权威定义，别处引用它。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Role = Literal["user", "assistant"]

# 事实的生命周期。`superseded` 是**失效而非删除**：新事实让旧事实带上
# `valid_to`，旧事实仍然留在库里（可审计），只是不再注入。
FactStatus = Literal["active", "pending", "superseded", "rejected"]


class Turn(TypedDict, total=False):
    """一轮对话的输入形状（`remember()` 收的就是它）。

    `assistant` 可以是**多条**：她一轮里可能发好几条消息，而 L0 是唯一不可
    再生的那一层，消息边界不能在这里并掉。
    """

    ts: float
    user: str
    assistant: str | list[str]


class TurnRow(TypedDict):
    """从 L0 读回来的一行（读的是原话，不是结构化事实）。"""

    id: str
    day: str
    hour: str
    time: str
    role: Role
    tags: list[str]
    text: str


class Fact(TypedDict, total=False):
    """一条事实。

    `turn_ref` 与 `quote` 是低幻觉的落点：每条事实都要能指回它出自哪一轮、
    原话是怎么说的。`valid_from` / `valid_to` 是双时间轴 —— 新事实让旧事实
    **失效**而不是删除（照 Zep / Graphiti 的做法）。
    """

    id: str
    subject: str
    predicate: str
    object: str
    status: FactStatus
    importance: float
    # 画像重排用的权重：这条事实对「她」来说有多值得在意。
    # 名字故意中性化 —— 画像由宿主注入（见 `PersonaProfile`），不是写死的。
    persona_attention: float
    confidence: float
    kind: str
    pinned: int
    valid_from: str
    valid_to: str
    quote: str
    turn_ref: str
    created_at: str
    last_used_at: str
    use_count: int


class Summary(TypedDict, total=False):
    """L2 纪要：一场对话之后写下的小结。它**不带编号**，因此不是可回引的事实。"""

    id: str
    day: str
    text: str
    created_at: str


class Topic(TypedDict, total=False):
    """主动话题：她手里「可以找他聊一句」的由头。"""

    id: str
    kind: str
    text: str
    used: int
    created_at: str


class MemoryContext(TypedDict, total=False):
    """一轮检索的**工作面**：这一轮要注入什么。

    它是瞬态的（不落盘）—— 里面是完整的事实行与用户原话，下一轮就被整个
    覆盖；写进会话文件只会让它随轮次线性变大。
    """

    facts: list[Fact]
    summaries: list[Summary]
    topics: list[Topic]
    enabled: bool
    ms: int
    hot: int
    cold: int
    used_tokens: int


class ResetResult(TypedDict):
    """`reset()` 的返回：**如实报出实际做了什么**，而不是只说一句成功。

    `leftover` 是删不掉的东西（Windows 上只要有打开的句柄就会拒绝删除）。
    吞掉它会让接口回一个 `ok`，而库里的事实原样留着 —— 她照样记得。
    """

    mode: str
    leftover: list[str]


class ArchiveStats(TypedDict, total=False):
    """归档层的现状，给「少了的去哪儿了」一个同一份返回值里的答案。"""

    archived_days: list[str]
    archived_bytes: int


class Stats(TypedDict, total=False):
    uid: str
    dir: str
    days: list[str]
    turns: int
    bytes: int
    watermark: dict[str, Any]
