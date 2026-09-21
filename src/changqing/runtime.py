"""进程级的运行期装配：配置、画像与三个注入的能力。

**为什么是进程级的一份，而不是每个函数一个参数。**

存储层与检索层有上百处要读配置（根目录、预算、阈值）。一条路是把配置顺着
参数一层层传下去，代价是改几百个调用点与全部测试；另一条路是模块级常量，
代价就是「导入时拷贝，之后改不动、也不报错」那个坑。

这里取第三条：一个**显式的、可替换的**进程级对象，子模块每次用的时候现取
（`runtime()`），而不是在导入时把值拷走。测试用 `using()` 换一份，作用域结束
自动还原。

**这是个有意的取舍，它的边界写在 `docs/architecture.md`**：一个进程里只有
一份生效的配置。多租户服务需要「同一个进程里两个库各用各的配置」时，得把它
改成显式传参 —— 那是接口变更，不是改个默认值。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .config import MemoryConfig
from .persona import NEUTRAL, PersonaProfile
from .ports import LLM, Embedder, NoUsage, NullEmbedder, NullLLM, UsageSink


@dataclass
class Runtime:
    """当前生效的配置、画像与外部能力。"""

    config: MemoryConfig = field(default_factory=MemoryConfig)
    persona: PersonaProfile = field(default_factory=lambda: NEUTRAL)
    embedder: Embedder = field(default_factory=NullEmbedder)
    llm: LLM = field(default_factory=NullLLM)
    usage: UsageSink = field(default_factory=NoUsage)


_LOCK = threading.Lock()
_ACTIVE: Runtime = Runtime()


def runtime() -> Runtime:
    """当前生效的运行期。子模块**每次用的时候**现取，别在导入时拷走。"""
    return _ACTIVE


def configure(
    config: MemoryConfig | None = None,
    *,
    persona: PersonaProfile | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
    usage: UsageSink | None = None,
) -> Runtime:
    """替换进程级的运行期装配，返回新的那一份。没传的项保持原样。"""
    global _ACTIVE
    with _LOCK:
        _ACTIVE = Runtime(
            config=config if config is not None else _ACTIVE.config,
            persona=persona if persona is not None else _ACTIVE.persona,
            embedder=embedder if embedder is not None else _ACTIVE.embedder,
            llm=llm if llm is not None else _ACTIVE.llm,
            usage=usage if usage is not None else _ACTIVE.usage,
        )
        return _ACTIVE


@contextmanager
def using(rt: Runtime) -> Iterator[Runtime]:
    """在这个作用域里换成 `rt`，出来自动还原。测试与「临时换一个库」用它。"""
    global _ACTIVE
    with _LOCK:
        previous = _ACTIVE
        _ACTIVE = rt
    try:
        yield rt
    finally:
        with _LOCK:
            _ACTIVE = previous


def reset() -> Runtime:
    """还原成出厂状态。测试之间清场用。"""
    return configure(
        MemoryConfig(),
        persona=NEUTRAL,
        embedder=NullEmbedder(),
        llm=NullLLM(),
        usage=NoUsage(),
    )
