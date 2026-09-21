"""行尾：仓库里**不许出现任何 `\\r`**。

`.gitattributes` 里写着 `* text=auto eol=lf`，理由也写在那一行旁边：
落单的 `\\r`（`\\r` 后面不是 `\\n`）会让 git 把整份文件判成**二进制**、
跳过 CRLF→LF 归一化，结果是下一次 diff 长成「整份文件重写」—— 而它一个字
都不报错。规矩写了却没人执行，就是这条护栏存在的理由。

**这条护栏是有来历的**：一次用脚本改源码，脚本用默认的文本模式写回，
整份文件悄悄变成了 CRLF（94 个 `\\r`），`git add` 只给了一句
「CRLF will be replaced by LF」的警告。真正把它挖出来的是下一节的字节扫描 ——
所以这里扫的是**原始字节**：`read_text` 会把 `\\r\\n` 归一掉，那正好把要查的
东西擦干净了。
"""

from __future__ import annotations

import os

from _guards import PROBE_PREFIX, REPO, is_probe, iter_raw_files


def _carriage_returns() -> list[tuple[str, int]]:
    """带 `\\r` 的文件与它有几个。**探针也算在内** —— 下面有一条要证明它会被看到。"""
    return [(name, raw.count(13)) for name, raw in iter_raw_files() if b"\r" in raw]


def test_the_detector_sees_a_carriage_return():
    """先证明自己认得违规样本：`\\r\\n` 与落单的 `\\r` 都要算。"""
    assert b"\r" in b"a\r\nb"
    assert b"\r" in b"a\rb"
    assert b"\r" not in b"a\nb"


def test_the_scan_actually_covers_the_repository():
    """扫描面本身也要有下限，否则「全绿」可能只是「什么都没扫」。"""
    names = [name for name, _ in iter_raw_files()]
    assert len(names) > 50, f"只扫到 {len(names)} 个文件，扫描器多半坏了"
    assert "src/changqing/config.py" in names


def test_a_carriage_return_in_the_tree_is_actually_reported():
    """把违规样本**真的放进仓库里**，再确认扫描器会点名它。

    前面那条只证明了「`\\r` 是 `\\r`」—— 它说明不了 `iter_raw_files` 会走到
    这个文件。而这条护栏最可能坏掉的方式恰恰是「扫描器不再覆盖整棵树」
    （往 `SKIP_DIRS` 里加错一个名字、`rglob` 被换成只扫 `src/`），
    那时它照样全绿。所以这里端到端地走一遍：放进去 → 被报出来 → 删掉。
    """
    probe = REPO / f"{PROBE_PREFIX}crlf-{os.getpid()}.txt"
    probe.write_bytes(b"a\r\nb")
    try:
        hits = _carriage_returns()
    finally:
        probe.unlink(missing_ok=True)
    assert any(probe.name in name for name, _ in hits), f"扫描器没报出放进仓库的样本：{hits}"


def test_no_file_carries_a_carriage_return():
    offenders = [
        f"{name}（{count} 个）" for name, count in _carriage_returns() if not is_probe(name)
    ]
    assert offenders == [], f"这些文件里有 \\r（`.gitattributes` 要求 eol=lf）：{offenders}"
