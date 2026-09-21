"""changqing · 长情：为**长期相处**的对话关系设计的记忆系统。

它解决一个问题：一个陪你聊了很久的模型，怎么才能真的记得你。做法是把记忆分成
四层，让模型自己负责整理，并且每一条事实都能回引到产生它的那句原话：

    L0 原话   sessions/YYYY-MM-DD.md   只追加、超期只归档，一个字节都不删
    L1 事实   log.jsonl → index.sqlite 操作日志不可再生，索引可重建
    L2 纪要   一场对话之后写下的小结
    L3 画像   事实上的权重（`persona_attention`），由注入的 `PersonaProfile` 决定

**这个包不自己调任何模型。** 它需要三种能力，全部由宿主注入（见 `ports`）：
`Embedder`（可选，不注入就是关掉向量召回）、`LLM`（抽取与巩固要用）、
`UsageSink`（可选，不注入就是不记账）。默认替身是离线、确定、零网络的
（见 `adapters.mock`），所以 `examples/quickstart.py` 不需要任何 API Key。

公开面就是下面 `__all__` 里那些：门面（`Memory` / `MemoryConfig` / `PersonaProfile`）
加装配（`configure` / `runtime` / `using`）。内部模块（`store` / `extract` /
`retrieve` / `worker` …）可以有，但不承诺稳定 —— 要直接调它们就得接受接口会动。
"""

from __future__ import annotations

from .config import MemoryConfig
from .edit import EDITABLE_FIELDS, EditError
from .memory import Memory
from .persona import NEUTRAL, PersonaProfile
from .ports import LLM, Embedder, NoUsage, NullEmbedder, NullLLM, UsageSink
from .runtime import Runtime, configure, reset, runtime, using
from .tokens import clip_to_tokens, estimate_tokens

__version__ = "0.1.0"

__all__ = [
    # 门面
    "Memory",
    "MemoryConfig",
    "PersonaProfile",
    # 手动改一条事实的字段白名单与校验
    "EDITABLE_FIELDS",
    "EditError",
    # 装配
    "Runtime",
    "configure",
    "reset",
    "runtime",
    "using",
    # 能力协议与默认替身
    "Embedder",
    "LLM",
    "UsageSink",
    "NullEmbedder",
    "NullLLM",
    "NoUsage",
    "NEUTRAL",
    # token 折算（预算与裁剪共用同一把尺）
    "estimate_tokens",
    "clip_to_tokens",
    "__version__",
]
