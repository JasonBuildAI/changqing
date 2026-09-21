"""护栏一：仓库里不许出现任何指向宿主的痕迹。

为什么这条要机械地扫，而不是靠人记住：解耦最贵的失败形态是「**看起来解耦了
其实没有**」。一行 `from app.config import MEMORY_DIR` 在宿主里跑得好好的；
藏在函数体里的那种更糟 —— 它连 import 时都不报错，只在某条分支上炸。

这条护栏只管**耦合**（import 那个包），隐私词汇是下一条
`tests/test_no_private_terms.py` 的事。
"""

from __future__ import annotations

from _guards import host_import_lines, iter_text_files


# ---------------------------------------------------------------- 先证明它会红
def test_the_detector_catches_a_host_import():
    """护栏自己先过一遍：这三种写法必须被认出来。"""
    assert host_import_lines("from app.config import MEMORY_DIR\n") == [1]
    assert host_import_lines("import app.memory\n") == [1]
    assert host_import_lines("x = 1\nfrom app import config\n") == [2]
    assert host_import_lines("import app\n") == [1]


def test_the_detector_does_not_fire_on_a_local_name_called_app():
    """**假红也是缺陷**：`app` 在这里是 FastAPI 的变量，不是宿主模块。

    按 `app\\.` 一网打尽的话，`examples/server_app.py` 与路由自己就会被判违规
    —— 而真正的耦合会淹没在假红里，人就学会忽略这条护栏了。
    """
    assert host_import_lines("app.include_router(create_router(), prefix='/api')\n") == []
    assert host_import_lines("from fastapi import FastAPI\napp = FastAPI()\n") == []
    assert host_import_lines("applications = []\n") == []


def test_the_scan_actually_covers_the_repository():
    """扫不到文件时这条护栏会「全部通过」—— 那是它最危险的状态。"""
    names = [name for name, _ in iter_text_files()]
    assert len(names) > 50, f"只扫到 {len(names)} 个文件，扫描器多半坏了"
    assert "src/changqing/config.py" in names
    assert "README.md" in names


# ---------------------------------------------------------------- 再扫仓库
def test_no_file_imports_a_host_package():
    offenders: list[str] = []
    for name, text in iter_text_files():
        for line in host_import_lines(text):
            offenders.append(f"{name}:{line}")
    assert offenders == [], f"这些地方还在 import 宿主包：{offenders}"
