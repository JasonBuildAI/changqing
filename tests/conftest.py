"""测试的公共底座：每个用例一份干净的运行期 + 一个临时库根。

**为什么必须有这个 fixture，而不是各用例自己传 `MemoryConfig(root=...)`。**
`runtime()` 是**进程级**的一份（见 `changqing/runtime.py` 的取舍），跨用例残留会
让「上一条用例换掉了配置或脚本化的模型桩」泄漏到下一条 —— 那种失败看起来像
被测代码错了，实际是用例之间互相污染，查起来最贵。

**为什么根目录一律指到 tmp_path。** `~/.changqing`（或任何真实目录）里装的是
真实对话；测试往里写一次，之后就再也分不清哪条是测试造的了。这里的 fixture
把根目录按用例换成临时目录，效果是 `tests/` 里**没有**任何一条用例碰得到真实库。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from changqing import MemoryConfig
from changqing.runtime import Runtime, using


@pytest.fixture()
def rt(tmp_path: Path) -> Iterator[Runtime]:
    """换上一份隔离的运行期：库根在临时目录，能力全是出厂替身。作用域结束还原。"""
    fresh = Runtime(config=MemoryConfig(root=tmp_path / "changqing"))
    with using(fresh) as active:
        yield active


@pytest.fixture()
def mem_root(rt: Runtime) -> Path:
    """隔离后的库根（临时目录）。用例要断言「文件真的落在哪儿」时用它。"""
    return Path(rt.config.root)
