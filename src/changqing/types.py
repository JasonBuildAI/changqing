"""核心数据形状：**只放形状，不放行为**。

它们都是 `TypedDict` 而不是 dataclass，因为存储层从头到尾按 `dict` 在传
（SQLite 的 `row_factory` 直接给 dict、日志是 JSONL），转成 dataclass 只会增加
一层无谓的往返。形状写在这里的价值是「唯一权威定义，别处引用它」。

**为什么这份文件必须被机械校验**（`tests/test_types.py`）。
它曾经漂过，而且漂得毫无征兆：状态那个 `Literal` 里躺着一个代码从来不写的
`rejected`，而代码真的会写的 `forgotten` / `merged` 一个都没列进去；
`Fact` 里还写着一个表里根本不存在的列 `created_at`。

形状漂移**不报错**：没人 import 的定义不会在运行时被检查，它只在有人照着这份
定义写代码的时候现形 —— 而那时候错误已经在调用方那边了。所以这里的每一条
判据都由测试拿实现去对：状态与操作种类从源码里扫，形状与被测代码的真实返回
逐键比对。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Role = Literal["user", "assistant"]

# 事实的生命周期 = 代码真的会写进去的那五个（见 `store/index.py` 的 `_invalidate`、
# `store/ops.py` 的 FORGET / CONFIRM / MERGE 与 `store/facts.py` 的 pending 分支）。
# `superseded` 是**失效而非删除**：新事实让旧事实带上 `valid_to`，旧事实仍然
# 留在库里（可审计），只是不再注入。
FactStatus = Literal["active", "pending", "superseded", "forgotten", "merged"]

# 操作日志里的操作种类，就是 `store/ops.py` 的 `apply_op` 认得的那一组。
# 它是这个库的**唯一事实源的结构**：日志只追加，物化视图是它的投影。
OpKind = Literal[
    "ADD",
    "SUPERSEDE",
    "EDIT",
    "INVALIDATE",
    "PIN",
    "CONFIRM",
    "FORGET",
    "MERGE",
    "DEROGATE",
    "SUMMARY",
    "TOPIC",
]

# 事实的种类。它与 `edit.KINDS` 是同一份东西 —— 之所以在这里再写一遍类型，
# 是因为那边是一个给运行时用的元组，这边是给类型检查用的 `Literal`；
# `tests/test_types.py` 钉住两者永远相等。多出来的值不是「以后可能用」，
# 而是**当下没有任何一处处理**：巩固只认 commitment / promise。
FactKind = Literal["fact", "commitment", "promise"]

# 冷路径「为什么是空的」：**超时与没命中都会返回空，但处置完全相反**
# （超时要调预算，没命中是正常）。所以它必须是一个能被上层分辨的值，
# 而不是一句日志。
RetrievalReason = Literal["hot_only", "no_fact", "no_hit", "timeout", "error"]


class Turn(TypedDict, total=False):
    """一轮对话的输入形状（`remember()` 收的就是它）。

    `assistant` 可以是**多条**：她一轮里可能发好几条消息，而 L0 是唯一不可
    再生的那一层，消息边界不能在这里并掉。
    """

    ts: float
    user: str
    assistant: str | list[str]


class TurnRow(TypedDict):
    """从 L0 读回来的一行（读的是原话，不是结构化事实）。

    `tags` 是行上挂的标记（现在是空的，留着给宿主自己用）；
    `hour` 是它所在的小节标题，用于「同一小时内的原话」这类判断。
    """

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
    **失效**而不是删除。

    `_score` 与 `_match` 是检索算完才加上去的（不是库里的列）：前者是融合后的
    分数，后者说明这一条是被哪几路召回的。`text_index` 是全文检索用的分词结果。
    """

    id: str
    subject: str
    predicate: str
    object: str
    status: FactStatus
    kind: FactKind
    importance: float
    # 画像重排用的权重：这条事实对「她」来说有多值得在意。
    # 名字故意中性化 —— 画像由宿主注入（见 `PersonaProfile`），不是写死的。
    persona_attention: float
    confidence: float
    pinned: int
    valid_from: str
    valid_to: str
    quote: str
    turn_ref: str
    due: str
    source: str
    last_used_at: str
    use_count: int
    text_index: str
    _score: float
    _match: str


class Summary(TypedDict, total=False):
    """L2 纪要：一场对话之后写下的小结。

    它**不是事实**（没有槽位、没有回引，不能被引用成「你说过」），
    但和事实一样不可再生 —— 模型生成的东西重算复现不出来，所以要进日志。
    """

    id: str
    day: str
    text: str
    created_at: str
    turn_ref: str


class Topic(TypedDict, total=False):
    """主动话题：她手里「可以找他聊一句」的由头。

    `due_day` 是「别早于这一天提」；`used` 是派生层的排序状态（不进日志）——
    挑过一次就划掉，同一件事反复主动提，是「她只会这一句」的样子。
    """

    id: str
    day: str
    created_at: str
    kind: str
    text: str
    due_day: str
    ref: str
    used: int


class MemoryContext(TypedDict, total=False):
    """一轮检索的**工作面**：这一轮要注入什么。

    它是瞬态的（不落盘）—— 里面是完整的事实行，下一轮就被整个覆盖；
    写进会话文件只会让它随轮次线性变大。

    `used_tokens` 是**三份加起来的**实际注入量（事实 + 故事线 + 话题），
    只算事实的话，下一个人会拿它当成「这一轮的注入量」，而另外两份凭空消失。
    """

    facts: list[Fact]
    summaries: list[Summary]
    topics: list[Topic]
    enabled: bool
    ms: int
    hot: int
    cold: int
    used_tokens: int
    backend: str
    reason: RetrievalReason | str


class ResetResult(TypedDict):
    """`reset()` 的返回：**如实报出实际做了什么**，而不是只说一句成功。

    `leftover` 是删不掉的东西（Windows 上只要有打开的句柄就会拒绝删除）。
    吞掉它会让接口回一个 `ok`，而库里的事实原样留着 —— 她照样记得。
    """

    mode: str
    leftover: list[str]


class ArchiveStats(TypedDict):
    """归档目录的现状：`sessions.archive/` 下有几个包、共多少字节。"""

    files: int
    bytes: int


class VectorStats(TypedDict):
    """向量索引的现状。

    `stale` 是「事实改了内容、向量还是旧的」那种条数；`model_mismatch` 是
    「当前编码器与索引里那批不是同一个模型」——后者为真时相似度全是噪声，
    而它**不会报错**，只能靠这个标志发现。
    """

    facts: int
    vectors: int
    stale: int
    dim: int
    model: str
    models: list[str]
    model_mismatch: bool


class TopicStats(TypedDict):
    total: int
    open: int


class MemoryStats(TypedDict, total=False):
    """`Memory.stats()` 的返回：规模与派生层的计数。"""

    total: int
    live: int
    pending: int
    applied_ops: int
    vectors: VectorStats
    topics: TopicStats


class StoreStats(TypedDict, total=False):
    """`store.stats()` 的返回：L0 与归档层的现状（轮数、天数、字节）。"""

    uid: str
    dir: str
    days: list[str]
    turns: int
    bytes: int
    archived_days: list[str]
    archived_bytes: int
    watermark: dict[str, Any]
