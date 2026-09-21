"""四条隐私 / 解耦 / 行尾护栏共用的扫描器。

为什么放在一个模块里，而不是三份各写一遍：它们查的是**同一批文件**。
「哪些目录算仓库、哪些文件算文本」如果各写一份，迟早有一份漏掉某个目录 ——
于是那条护栏在悄悄放行，而它看起来完全正常（这正是它要防的那类缺陷）。

**这套扫描器的值全在「它真的会失败吗」上。** 所以每条护栏都先证明自己
认得违规样本（`test_the_detector_...`），再去扫仓库 —— 一条从不失败的护栏
比没有护栏更糟：它让人以为这件事有人在管。
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 不扫的目录：版本库内务、缓存、草稿。**`.tmp` 是草稿目录**（注入脚本、
# 还没提交的片段都放在那儿），它不该被当成仓库内容。
SKIP_DIRS = {
    ".git",
    ".tmp",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    ".idea",
    ".vscode",
}

# 二进制/打包产物。它们不是「文本内容」，扫了只会得到一堆解码噪声。
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".gz",
    ".zip",
    ".whl",
    ".sqlite",
    ".db",
}

# 工具产物：名字本身就在说「这不是仓库内容」。
#
# 这份名单必须**按名字**匹配，不能并进后缀表：`.coverage` 的 `suffix` 是空串
# （它整个就是一个没有扩展名的文件名），后缀表拦不住它。踩过一次：本地跑过
# pytest-cov 之后根目录多出一个 `.coverage`，那是 SQLite 产物、里面天然带
# `\r`，于是行尾护栏对着一个 git 根本不跟踪的文件变红。
SKIP_NAMES = {".coverage", ".DS_Store", "Thumbs.db"}
SKIP_NAME_PREFIXES = (".coverage.",)

# 单文件上限：一个几兆的文件不是仓库内容，但它能让整套测试慢下来。
MAX_BYTES = 2_000_000

# 护栏自己往仓库里放的**一次性探针**，见 `test_a_carriage_return_in_the_tree_
# _is_actually_reported`：光证明「扫描器认得违规样本」说明不了它还覆盖着这棵树，
# 所以真的放一份进去，看它会不会被报出来。
#
# 探针**照常被扫描**（不然那个证明就无从谈起），只是全仓库的「一处都没有」
# 断言会把它们滤掉。不滤的话两条护栏会互相打架：行尾那条正拿着一份带 `\r` 的
# 探针，隐私那条扫到它 —— 于是随机某一条变红，而红的不是代码。
PROBE_PREFIX = ".changqing-probe-"


def is_probe(name: str) -> bool:
    """这个仓库相对路径是一次性探针吗（`name` 是 `iter_*` 产出的那种字符串）。"""
    return Path(name).name.startswith(PROBE_PREFIX)


def _skipped(path: Path) -> bool:
    if path.name.startswith(SKIP_NAME_PREFIXES):
        return True
    if path.name in SKIP_NAMES:
        return True
    return any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts)


def iter_raw_files(*, only_suffix: str = "") -> Iterator[tuple[str, bytes]]:
    """仓库里的文件，产出 `(相对路径, 原始字节)`。

    要字节不要文本，是因为**行尾检查必须看原始字节**：`read_text` 会把
    `\\r\\n` 归一成 `\\n`，于是「这份文件是 CRLF 的」这件事在读的那一刻就没了。
    """
    for path in sorted(REPO.rglob("*")):
        if not path.is_file() or _skipped(path):
            continue
        if only_suffix and not path.name.endswith(only_suffix):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            yield path.relative_to(REPO).as_posix(), path.read_bytes()
        except OSError:
            continue


def iter_text_files(*, only_suffix: str = "") -> Iterator[tuple[str, str]]:
    """仓库里的文本文件，产出 `(相对路径, 内容)`。

    读不出文本的（二进制、编码不对）静默跳过 —— 这里不是「解析器」，
    是「扫一遍仓库」，为一条解码失败而整条护栏失败没有意义。
    """
    for name, raw in iter_raw_files(only_suffix=only_suffix):
        try:
            yield name, raw.decode("utf-8")
        except UnicodeDecodeError:
            continue


# ---------------------------------------------------------------- 宿主耦合
# 指向宿主顶层包 `app` 的 **import 语句**。
#
# 只认 import 语句这一件事很关键：`app.include_router(...)` 里的 `app` 是
# FastAPI 的变量，不是宿主模块 —— 按 `app\.` 一网打尽的话，`examples/` 与
# 服务端路由自己就会被判违规，而真正的耦合（`from app.memory import x`）
# 反倒淹没在假红里。
_HOST_IMPORT_RE = re.compile(r"^[ \t]*(?:from|import)[ \t]+app(?:\.[A-Za-z_]|\s|$)", re.MULTILINE)


def host_import_lines(text: str) -> list[int]:
    """文本里指向宿主包的 import 在第几行（1 起）。"""
    return [text.count("\n", 0, m.start()) + 1 for m in _HOST_IMPORT_RE.finditer(text)]


# ---------------------------------------------------------------- 环境变量
def env_read_lines(source: str) -> list[int]:
    """`source` 里读环境变量的行号（1 起）。

    **走 AST 而不是正则。** `adapters/openai.py` 的 docstring 里就写着
    `OpenAILLM(api_key=os.environ["OPENAI_API_KEY"])` —— 那是在教宿主**自己**
    去读，正是这条规矩想要的写法。正则会把那句当成违规，于是护栏逼着人删掉
    正确的文档。
    """
    tree = ast.parse(source)
    hits: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv", "putenv"):
            hits.append(node.lineno)
        elif isinstance(node, ast.Name) and node.id == "getenv":
            hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module == "os" and node.names:
            if any(alias.name in ("environ", "getenv") for alias in node.names):
                hits.append(node.lineno)
    return sorted(set(hits))
