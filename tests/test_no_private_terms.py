"""护栏二：仓库里不许出现任何属于宿主、作者本机或那个具体角色的词。

这一条守的是**隐私**，不是耦合（那是 `tests/test_no_host_coupling.py`）。
为什么它必须机械地扫：这类词是「混进来」的，不是「写上去」的 —— 一句
从宿主那边顺手带过来的真实对话样例、一个 `\u0044\u003a\u005c...` 的绝对路径、一个角色名，
在 review 里看起来都无害，而它们会跟着这个库一起被公开。

**名单只此一处。** 它是这个仓库里唯一写下这些词的地方，所以扫描时把本文件
排除掉；除此之外没有任何豁免（`_guards.py` 自己也在扫描范围内）。
"""

from __future__ import annotations

from _guards import iter_text_files

# 逐条都要有理由，不然下一个人会往里加一些「看着像」的词，把护栏变成噪声：
PRIVATE_TERMS = (
    "\u006e\u0075\u006d\u0062\u0065\u0072\u0068\u0075\u006d\u0061\u006e",  # 宿主项目的名字
    "\u6797\u4e00\u6b46",  # 宿主那个角色的名字
    "\u9ec4\u6c5f\u5357",  # 作者本名（本仓库的署名统一用 JasonBuildAI）
    "\u0041\u0067\u0065\u006e\u0074\u0050\u0072\u006f\u006a\u0065\u0063\u0074\u0073",  # 本机的工程目录
    "\u0044\u003a\u005c",  # 本机绝对路径
)

# 名单自己住在这个文件里，扫它必然命中自己。**这是唯一一处豁免。**
SELF = "tests/test_no_private_terms.py"


def _hits(name: str, text: str) -> list[str]:
    """这个文件里出现的违规词，带上行号（行号是给人去改的）。"""
    found: list[str] = []
    for term in PRIVATE_TERMS:
        pos = text.find(term)
        while pos != -1:
            found.append(f"{name}:{text.count(chr(10), 0, pos) + 1}: {term}")
            pos = text.find(term, pos + len(term))
    return found


# ---------------------------------------------------------------- 先证明它会红
def test_every_listed_term_is_actually_detectable():
    """名单里每一条都要真的能被认出来。

    写错一个转义（`\u0044\u003a\u005c` 与 `\u0044\u003a\u005c\\`）就会出现一条**永远不触发**的名单项 ——
    而它看起来与别的项一模一样。
    """
    for term in PRIVATE_TERMS:
        assert _hits("sample", f"前面 {term} 后面") != [], f"这条认不出来：{term!r}"


def test_the_detector_reports_the_line_number():
    """报行号是为了能直接去改；只报文件名的话，一次命中就是一次全文件通读。"""
    assert _hits("a.py", "第一行\n第二行 \u006e\u0075\u006d\u0062\u0065\u0072\u0068\u0075\u006d\u0061\u006e\n") == ["a.py:2: \u006e\u0075\u006d\u0062\u0065\u0072\u0068\u0075\u006d\u0061\u006e"]


# ---------------------------------------------------------------- 再扫仓库
def test_no_private_terms_anywhere_in_the_repository():
    offenders: list[str] = []
    scanned = 0
    for name, text in iter_text_files():
        scanned += 1
        if name == SELF:
            continue
        offenders.extend(_hits(name, text))
    assert scanned > 50, f"只扫到 {scanned} 个文件，扫描器多半坏了"
    assert offenders == [], f"这些地方出现了不该出现的词：{offenders}"
