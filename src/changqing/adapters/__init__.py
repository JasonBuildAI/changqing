"""可直接用的外部能力实现。

`mock` 里的两个是**离线**的：不联网、不下模型、结果确定，默认档测试与
`examples/quickstart.py` 都用它们。`openai` 里的两个走 OpenAI 兼容接口，
需要装 `changqing[openai]`。

**`openai` 是惰性导入的**（PEP 562 的模块级 `__getattr__`）：`import changqing`
与 `import changqing.adapters` 都**不许**拉进 httpx —— 核心包零第三方依赖，
只在真的要用远程模型时才需要它。写成模块级 `from .openai import ...` 的话，
「装没装 httpx」会变成「能不能用这个库」，而这两件事本来无关。
"""

from __future__ import annotations

from typing import Any

from .mock import MockEmbedder, MockLLM

# 需要 `[openai]` 才能拉起来的那两个。名字在这里列一次，`__getattr__` 与
# `__all__` 共用同一份 —— 两处各抄一份的话，漏掉的那个会表现成
# `from changqing.adapters import *` 少一个名字，而不报错。
_LAZY = ("OpenAIEmbedder", "OpenAILLM")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from . import openai as _openai

        return getattr(_openai, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MockEmbedder", "MockLLM", *_LAZY]
