"""手动改一条事实时，允许改哪些字段、值长什么样。

**为什么白名单与校验必须住在一起、而且是公开的。** 面板的路由与库的门面
（`Memory.update`）是两条入口，但它们是同一句话：「用户可以把这条事实改成
什么样」。各写一份的后果不是报错，而是有一条入口悄悄比另一条松 —— 于是
「从面板改不坏的输入，从代码里能改坏」，而且两条路各自看起来都自洽。

**校验为什么必须在写日志之前挡掉，而不是「先进库，坏了再修」**
（这不是假想，是本层真踩过的两类坏法）：

1. `object` 传 `None` 或一个 dict 会一路走到 SQL 的 NOT NULL / 类型检查上抛
   `sqlite3.ProgrammingError`。而操作日志是**只追加**的、物化是按游标重放的 ——
   这一条坏操作会**永久**卡在那个用户的重放上：此后所有新事实都不再落库，
   界面上只看得出「她记性变差了」。
2. `importance` 传 `"high"` 会以 TEXT 落进 REAL 列。之后 `float()` 会在渲染、
   打分、巩固三处抛异常，而那三处都被宽 `except` 吞掉 —— 表现是两条注入路径
   每轮静默返回空，日志里一行都没有。

物化层另有一层「单条隔离」兜住**历史上已经存在**的坏行。两层都要：这里管
「不再产生坏行」，那里管「已经产生的坏行不扩散」。

**不 import 任何 Web 框架。** 校验失败抛 `EditError`，由调用方决定它是 400
还是一个异常 —— 核心库不该知道 HTTP 的存在。
"""

from __future__ import annotations

import re
from typing import Any

# 允许用户改的字段。
#
# **`turn_ref` 与 `quote` 故意不在里面**：那是「这条凭什么是真的」的证据。
# 允许改就等于允许伪造出处，整套回引校验就白做了。要改内容就改 `object`，
# 要否定整条走 confirm / forget。
EDITABLE_FIELDS = (
    "subject",
    "predicate",
    "object",
    "importance",
    "persona_attention",
    "due",
    "kind",
)

# 事实内容字段的长度上限。全局的消息长度上限挡不住这里 —— 面板一次 POST
# 就能塞进一个十万字的 `object`，而注入预算是按它折 token 的。
TEXT_MAX = 200

# `kind` 的取值。多出来的值不是「以后可能用」，而是**当下没有任何一处处理**：
# 巩固只认 commitment / promise，一个未知的 kind 会静默地既不被保护、也不被扫。
KINDS = ("fact", "commitment", "promise")

# 日期字段只认这一个形状。检索与渲染都按 `[:10]` 切，别的形状会被**悄悄截断**
# （「下周三」切完还是「下周三」，`strptime` 抛异常 → 时间衰减退化成一个常数）。
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class EditError(ValueError):
    """用户给的字段值不合法。调用方自己决定它对应 400 还是别的什么。"""


def _number(name: str, value: Any) -> float:
    """0–1 之间的数值。`bool` 要显式挡掉：`True` 是 `int` 的子类，会悄悄变成 1.0。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EditError(f"{name} 必须是 0–1 之间的数值")
    num = float(value)
    if not (0.0 <= num <= 1.0):
        raise EditError(f"{name} 必须在 0–1 之间（收到 {num}）")
    return num


def validate(fields: dict[str, Any]) -> dict[str, Any]:
    """挑出白名单内的字段并校验，返回可以直接进日志的那一份。

    只挑不报「有字段被忽略」：调用方通常是把用户提交的整块 dict 递进来，
    对它来说「多给了几个不认识的键」是正常的，报错反而挡住正常操作。
    **但白名单里的字段值不合法一律抛** —— 那才是会写坏数据的那一类。
    """
    out: dict[str, Any] = {}
    for key, value in (fields or {}).items():
        if key not in EDITABLE_FIELDS:
            continue
        if key in ("subject", "predicate", "object"):
            text = value.strip() if isinstance(value, str) else ""
            if not text:
                raise EditError(f"{key} 必须是非空字符串")
            if len(text) > TEXT_MAX:
                raise EditError(f"{key} 太长了（上限 {TEXT_MAX} 字，收到 {len(text)}）")
            out[key] = text
        elif key in ("importance", "persona_attention"):
            out[key] = _number(key, value)
        elif key == "due":
            text = "" if value is None else str(value).strip()
            if text and not DATE_RE.match(text):
                raise EditError("due 必须是 YYYY-MM-DD 或留空")
            out[key] = text
        elif key == "kind":
            text = value.strip() if isinstance(value, str) else ""
            if text not in KINDS:
                raise EditError(f"kind 只能是 {list(KINDS)}")
            out[key] = text
    return out


__all__ = ["DATE_RE", "EDITABLE_FIELDS", "KINDS", "TEXT_MAX", "EditError", "validate"]
