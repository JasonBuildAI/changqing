"""护栏二：仓库里不许出现任何属于宿主、作者本机或那个具体角色的词。

这一条守的是**隐私**，不是耦合（那是 `tests/test_no_host_coupling.py`）。
为什么它必须机械地扫：这类词是「混进来」的，不是「写上去」的 —— 一句
从宿主那边顺手带过来的真实对话样例、一个本机绝对路径、一个角色名，
在 review 里看起来都无害，而它们会跟着这个库一起被公开。

**名单本身也不写成明文。** 一个公开仓库里为了「挡住某个名字」而把那个名字
抄一遍，是把要挡的东西自己送出去（这个文件最初就是这么写的）。所以名单存的是
UTF-8 字节的十六进制，理由按行写在旁边。这挡的是「随手搜一下」，不是有心人 ——
但随手搜一下正是最常发生的那种。

**没有任何豁免文件。** 这一条一开始把自己从扫描里排除掉了（名单是明文，扫自己
必然命中）。改成十六进制之后文件里不再有那些词，豁免就撤了 —— 少一条豁免，
就少一个「看起来守住了其实没守」的角落。
"""

from __future__ import annotations

import os

from _guards import PROBE_PREFIX, REPO, is_probe, iter_text_files

# 十六进制写出来的名单，一行一个理由。逐条都要有理由，不然下一个人会往里加
# 一些「看着像」的词，把护栏变成噪声。
_TERMS_HEX = (
    "6e756d62657268756d616e",  # 宿主项目的名字
    "e69e97e4b880e6ad86",  # 宿主那个角色的名字
    "e9bb84e6b19fe58d97",  # 作者本名（本仓库的署名统一用 JasonBuildAI）
    "4167656e7450726f6a65637473",  # 本机的工程目录
    "443a5c",  # 本机绝对路径的开头
)

PRIVATE_TERMS = tuple(bytes.fromhex(hexed).decode("utf-8") for hexed in _TERMS_HEX)


def _hits(name: str, text: str) -> list[str]:
    """这个文件里出现的违规词，带上行号（行号是给人去改的）。"""
    found: list[str] = []
    for term in PRIVATE_TERMS:
        pos = text.find(term)
        while pos != -1:
            found.append(f"{name}:{text.count(chr(10), 0, pos) + 1}: {term}")
            pos = text.find(term, pos + len(term))
    return found


def _scan() -> list[str]:
    """整棵树扫一遍，**探针也算在内**（下面那条要证明它会被看到）。"""
    offenders: list[str] = []
    for name, text in iter_text_files():
        offenders.extend(_hits(name, text))
    return offenders


def _scan_besides_probes() -> list[str]:
    """全仓库那条断言用的版本：护栏自己放的探针不算违规。"""
    return [hit for hit in _scan() if not is_probe(hit.split(":", 1)[0])]


# ---------------------------------------------------------------- 先证明它会红
def test_every_listed_term_is_actually_detectable():
    """名单里每一条都要真的能被认出来。

    十六进制写错一个字符就会得到一条**永远不触发**的名单项 —— 而它看起来与
    别的项一模一样。所以每一条都先塞进一段文本里走一遍。
    """
    for term in PRIVATE_TERMS:
        assert term, "解码出空串：这一条永远不会触发"
        assert _hits("sample", f"前面 {term} 后面") != [], f"这条认不出来：{term!r}"


def test_the_detector_reports_the_line_number():
    """报行号是为了能直接去改；只报文件名的话，一次命中就是一次全文件通读。"""
    term = PRIVATE_TERMS[0]
    assert _hits("a.py", f"第一行\n第二行 {term}\n") == [f"a.py:2: {term}"]


def test_every_term_is_found_wherever_it_appears_in_a_file():
    """同一个词出现两次要报两次 —— 只报第一处会让人改一半就以为改完了。"""
    term = PRIVATE_TERMS[0]
    assert _hits("a.py", f"{term} 和再一次 {term}") == [f"a.py:1: {term}", f"a.py:1: {term}"]


def test_an_offending_file_in_the_tree_is_actually_reported():
    """真的往仓库里放一份违规内容，确认**扫描整棵树**会点名它。

    `_hits` 认得违规词，说明不了 `iter_text_files` 会走到那个文件 ——
    这条护栏最可能坏掉的方式恰恰是后者（`rglob` 被换掉、`SKIP_DIRS` 加错名字）。
    """
    probe = REPO / f"{PROBE_PREFIX}term-{os.getpid()}.txt"
    probe.write_text(f"中间那句 {PRIVATE_TERMS[0]} 在这里\n", encoding="utf-8", newline="\n")
    try:
        hits = _scan()
    finally:
        probe.unlink(missing_ok=True)
    assert any(probe.name in hit for hit in hits), f"扫描器没报出放进仓库的样本：{hits}"


# ---------------------------------------------------------------- 再扫仓库
def test_no_private_terms_anywhere_in_the_repository():
    offenders = _scan_besides_probes()
    scanned = sum(1 for _ in iter_text_files())
    assert scanned > 50, f"只扫到 {scanned} 个文件，扫描器多半坏了"
    assert offenders == [], f"这些地方出现了不该出现的词：{offenders}"
