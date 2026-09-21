"""外部能力的协议：嵌入、对话模型、用量记账。

这个包**不自己**去调任何模型。它需要三种能力，全部由宿主注入：

    Embedder   把文本变成向量（可选 —— 不注入就等于关掉向量召回）
    LLM        走一次非流式补全（抽取与巩固要用）
    UsageSink  记一笔用量（可选 —— 不注入就是不记账）

为什么用 Protocol 而不是基类：宿主多半已经有自己的客户端了，让它为了接入
这个库去继承一个类，是把我们的形状强加给别人。结构化子类型（鸭子类型 +
静态检查）只要方法签名对得上就能用。

`@runtime_checkable` 是为了能在装配处做一次 `isinstance` 自检（「你传进来的
这个东西真的实现了这四个方法吗」），而不是等到第一次检索时才炸。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


class LLMNotConfigured(RuntimeError):
    """没有注入 `LLM` 就调用了抽取 / 巩固。"""


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。

    **`encode` 失败返回 `None`，不抛异常。** 这是刻意的：向量召回是「锦上添花」
    的一路，模型没加载好、网络断了都不该让这一轮对话取不到任何记忆 ——
    降级到只走全文检索，而不是整条链路失败。
    """

    name: str
    dim: int
    # 模型的标识。向量索引靠它认「这批向量是谁编的」—— 只按 dim 判是不够的：
    # 换一个**同维度**的模型时旧向量会被原样留着，查询向量来自新模型，
    # 相似度全是噪声，而统计还报一切正常。
    repo: str

    def ready(self, download: bool = True) -> bool:
        """能不能用。`download=False` 表示「只问在不在，别去下」。"""

    def encode(self, texts: Sequence[str]) -> list[list[float]] | None:
        """批量编码。失败返回 None。"""

    def warm(self) -> bool | None:
        """把一次性的加载开销提前付掉。返回 None = 这个实现不需要预热。"""

    def loaded(self) -> bool:
        """加载好了没。**不能顺带触发加载** —— 状态查询不该去下模型。"""


@runtime_checkable
class LLM(Protocol):
    """一次非流式补全。

    形状与 `openai` 的习惯一致：`messages` 是 `[{"role": ..., "content": ...}]`。

    `on_finish(finish_reason)` 是个**观测**钩子：「这次输出被上限砍断了」
    是抽取那条路上必须能分辨的一件事（否则截断与「没什么可抽的」长得一模一样，
    症状是「她忘了」而不报错，而且只在长对话上出现）。
    """

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        max_tokens: int = 0,
        on_usage: Any = None,
        on_finish: Any = None,
    ) -> str: ...


@runtime_checkable
class UsageSink(Protocol):
    """记一笔用量。宿主拿它接自己的账单，这个库不解释数字。"""

    def note_llm(self, *, tin: int = 0, tout: int = 0, **extra: Any) -> None: ...


# ---------------------------------------------------------------- 默认实现
# 表示「这条路关着」的名字。`NullEmbedder` 用它，各处的判断也认它 ——
# 这条判断曾经在检索层与写入层各准备写一份，两处一旦漂移就会出现
# 「检索以为没有向量、写入以为有」这种**静默**的空转。
_OFF_NAMES = ("none", "off", "0")


def embedder_enabled(emb: Any) -> bool:
    """注入的向量实现自己说它开着吗。

    判据是**实现自己声明的名字**，不是配置里的开关：向量能力是注入进来的，
    配置里根本没有 provider 这一项 —— 也就没有「配置说开、实现是空」的不一致。

    **它不等于「现在能用」**：能用还要看 `ready()` / `loaded()`
    （见 `retrieve._embedding_on`）。这里只回答「这条路装没装」。
    """
    name = str(getattr(emb, "name", "") or "").strip().lower()
    return bool(name) and name not in _OFF_NAMES


class NullEmbedder:
    """关掉向量召回。它存在是为了让调用方不必到处写 `if`。"""

    name = _OFF_NAMES[0]
    dim = 0
    repo = ""

    def ready(self, download: bool = True) -> bool:
        return False

    def encode(self, texts: Sequence[str]) -> list[list[float]] | None:
        return None

    def warm(self) -> bool | None:
        return None  # None = 不需要预热（而不是预热失败）

    def loaded(self) -> bool:
        return False


class NoUsage:
    """不记账。"""

    def note_llm(self, *, tin: int = 0, tout: int = 0, **extra: Any) -> None:
        return None


class NullLLM:
    """没有注入模型时的替身：**大声报错**，不静默返回空串。

    静默返回空串的症状是「她什么都记不住」，而日志里一行异常都没有 ——
    这种缺陷要花几个小时才能定位。宁可当场抛。
    """

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        max_tokens: int = 0,
        on_usage: Any = None,
        on_finish: Any = None,
    ) -> str:
        raise LLMNotConfigured(
            "changqing 需要注入一个 LLM 才能做抽取与巩固："
            "Memory(config=..., llm=OpenAILLM(...))，"
            "或把 config.enabled 设为 False 关掉整个记忆系统。"
        )
