"""离线替身：不联网、不下模型、结果确定。

它们不是「测试专用的假货」，而是这个库**默认就能跑起来**的那条路：
没有 API Key、没有模型文件的人，也应该能把 `examples/quickstart.py` 跑通，
看到记忆写进去、又检索出来。
"""

from __future__ import annotations

import contextlib
import zlib
from collections.abc import Sequence
from typing import Any


class MockEmbedder:
    """字符 n-gram 哈希向量。

    为什么不用 `hash()`：CPython 的字符串哈希**每个进程都加了随机盐**
    （PYTHONHASHSEED），拿它当桶号会让「同一个字符串在两个进程里得到不同的
    向量」—— 离线索引写完，下次启动就查不到了，而且看起来一切正常。
    `zlib.crc32` 是稳定的。

    中文用二元组（bigram）：单字做特征是噪声，整句做特征是零召回。
    """

    name = "mock"
    repo = "changqing/mock-ngram-v1"

    def __init__(self, dim: int = 64) -> None:
        self.dim = max(1, int(dim))

    @staticmethod
    def _grams(text: str) -> list[str]:
        s = "".join(str(text or "").split())
        if len(s) < 2:
            return [s] if s else []
        return [s[i : i + 2] for i in range(len(s) - 1)]

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            grams = self._grams(text)
            for g in grams:
                vec[zlib.crc32(g.encode("utf-8")) % self.dim] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            out.append([v / norm for v in vec] if norm else vec)
        return out

    def ready(self, download: bool = True) -> bool:
        return True

    def warm(self) -> bool | None:
        return None  # 没有一次性开销

    def loaded(self) -> bool:
        return True


class MockLLM:
    """按脚本回话的模型替身。

    `responses` 用完之后一直返回 `default`。每次调用都记进 `self.calls`，
    方便测试断言「抽取向模型要了什么」——那一份 prompt 才是抽取质量的关键，
    而它平时是不可见的。
    """

    def __init__(self, responses: Sequence[str] | None = None, default: str = "") -> None:
        self.responses: list[str] = list(responses or [])
        self.default = default
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        max_tokens: int = 0,
        on_usage: Any = None,
        on_finish: Any = None,
    ) -> str:
        self.calls.append({"messages": messages, "model": model, "max_tokens": max_tokens})
        text = self.responses.pop(0) if self.responses else self.default
        if on_usage is not None:
            # 粗估一笔用量，让宿主的记账路径也有东西可记（不是真实 token 数）。
            chars = sum(len(str(m.get("content") or "")) for m in messages)
            # 记账是观测，自己炸了别带走这一轮。
            with contextlib.suppress(Exception):
                on_usage({"tin": int(chars / 1.5) + 1, "tout": int(len(text) / 1.5) + 1})
        if on_finish is not None:
            with contextlib.suppress(Exception):
                on_finish("stop")  # 档不会截断，别把档当截断
        return text
