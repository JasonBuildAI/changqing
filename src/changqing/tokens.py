"""token 折算：中文按 1.5 字符/token 的粗估。

**这个公式在本包里只有这一处。** 注入预算（`retrieve`）、散文裁剪
（`clip_to_tokens`）、向量重建的批量大小都读它 —— 两处各写一份的话，同一个
「300 token 预算」会在两条路上算出不同的数，而两边都不会报错。

**一处有意为之的分叉**：做过类似系统的宿主往往已经在自己的用量账本里有一份
`estimate_tokens`。那两份的**公式可以一样，但要各留一份** —— 本库不能反过来
import 宿主的用量模块（那就没解耦），也不能指望宿主把账本换成本库的。
代价是一旦哪边调了系数，两边会差一点点；这一点写在 `docs/design.md` 里，
不假装它不存在。

这个估法只用来**分配预算与排序**，不用来算钱：真实账单必须以端点上返回的
实测用量为准，估出来的数只配当兜底。
"""

from __future__ import annotations

# 一个中文字大约 1.5 个字符折 1 个 token（比英文密，比 GBK 计法松）。
_CHARS_PER_TOKEN = 1.5


def estimate_tokens(text: str) -> int:
    """粗估一段文本占多少 token。空串也要给 1：一条「占了位置」的直觉。"""
    return int(len(str(text or "")) / _CHARS_PER_TOKEN) + 1


def clip_to_tokens(text: str, budget_tokens: int) -> str:
    """把一段**散文**截到 token 预算内（与 `estimate_tokens` 同一把尺）。

    只给散文用：一条事实是原子，切一半会让模型拿着半个事实说话（那比多花几十
    token 贵得多），所以事实那边走的是「整条进或整条不进」。纪要、话题这种
    成句的文本才适用裁剪。

    推导：`estimate_tokens(t) = int(len/1.5) + 1 ≤ len/1.5 + 1`，所以只要
    `len ≤ 1.5 × (budget - 1)` 就一定有 `estimate ≤ budget`。留一个字符给省略号。
    预算 ≤ 0 时返回空串 —— 「没有任何预算」就该什么都不注入。
    """
    if budget_tokens <= 0:
        return ""
    if estimate_tokens(text) <= budget_tokens:
        return text
    keep = max(0, int((budget_tokens - 1) * _CHARS_PER_TOKEN) - 1)
    return text[:keep].rstrip() + "…"


__all__ = ["clip_to_tokens", "estimate_tokens"]
