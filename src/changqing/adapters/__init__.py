"""可直接用的外部能力实现。

`mock` 里的两个是**离线**的：不联网、不下模型、结果确定，默认档测试与
`examples/quickstart.py` 都用它们。`openai` 里的两个走 OpenAI 兼容接口，
需要装 `changqing[openai]`。
"""

from .mock import MockEmbedder, MockLLM

__all__ = ["MockEmbedder", "MockLLM"]
