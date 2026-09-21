"""文档里写的每一个 extra，都必须在 pyproject 里真的存在。

**为什么值得一条护栏**：extra 打错一个字母时，pip **退出码是 0** ——
它只在那几百行输出里留一句 `WARNING: changqing 0.1.0 does not provide the
extra 'al'`，然后「装成功」。于是「照文档装一下」变成「静默地什么都没装」，
而用户以为装上了。本机实测确认过（pip 26，`--dry-run` 也一样）。

这就是本仓库最贵的那类缺陷：**看起来守住了，其实没有** —— 只不过这次
静默失效的是**安装说明**，不是代码。所以这里真的一行一行去读文档。

真相取自**已安装发行版的元数据**（`Provides-Extra`），不另起一份解析器：
能跑这套用例的环境一定装过这个包（`src` 布局下不装连 `import changqing`
都失败），所以这里不需要「装了才检查、没装就跳过」那种会让护栏变哑的分支。
"""

from __future__ import annotations

import re
from importlib.metadata import metadata

from _guards import iter_text_files

DIST = "changqing"

# 只认 `.[name]` 这种写法（`pip install ".[server]"`）。带上那个点很关键：
# 改版本的链接写的是 `[Unreleased]`、`[0.1.0]`，不带点 —— 一网打尽会把
# CHANGELOG 的章节名当成 extra。
_EXTRA_RE = re.compile(r"\.\[([A-Za-z0-9_-]+)\]")

# 讲安装的文档。新增一份讲安装的 `.md` 时要往这里加。
DOCS = (
    "README.md",
    "README.en.md",
    "CONTRIBUTING.md",
    "docs/api.md",
    "docs/configuration.md",
)


def declared_extras() -> set[str]:
    """pyproject 里声明的 extra 名字（经由已安装发行版的元数据）。"""
    return set(metadata(DIST).get_all("Provides-Extra") or [])


def extras_in(text: str) -> list[str]:
    """一段 Markdown 里出现的 extra 名字（按出现顺序，重复的保留）。"""
    found: list[str] = []
    for line in text.splitlines():
        if "pip install" not in line:
            continue
        found.extend(_EXTRA_RE.findall(line))
    return found


# ---------------------------------------------------------------- 先证明它会红
def test_the_scan_finds_extras_in_an_install_line():
    """先证明正则认得 `pip install ".[x]"` —— 认不出来时它是**静默**全绿的。"""
    assert extras_in('pip install ".[server]" ".[dev]"') == ["server", "dev"]
    assert extras_in('pip install -e ".[all]"     # 上面全部') == ["all"]
    # 不带点的不算：那是 CHANGELOG 的章节名，不是 extra。
    assert extras_in("见 [Unreleased] 与 [0.1.0] 两节") == []
    # 不是安装行也不算。
    assert extras_in("extra 叫 `all`，见 [all]") == []


def test_the_metadata_reports_the_extras_we_expect():
    """真相本身也要有个下限：读不到 extra 时，下面两条会一起静默变绿。"""
    extras = declared_extras()
    assert {"all", "dev"} <= extras, f"元数据里的 extra 不对劲：{sorted(extras)}"


# ---------------------------------------------------------------- 再对文档
def test_every_extra_named_in_the_docs_exists():
    extras = declared_extras()
    offenders: list[str] = []
    seen: list[str] = []
    for name, text in iter_text_files(only_suffix=".md"):
        if name not in DOCS:
            continue
        for extra in extras_in(text):
            seen.append(extra)
            if extra not in extras:
                offenders.append(f"{name}: .[{extra}]")
    assert len(seen) >= 8, f"只从文档里扫到 {len(seen)} 处 extra，扫描器多半坏了"
    assert offenders == [], f"这些 extra 在 pyproject 里不存在（pip 只会警告一声）：{offenders}"


def test_every_declared_extra_is_documented_in_the_readme():
    """反方向：声明了却没人写的 extra，用户不会知道它存在。"""
    readme = dict(iter_text_files(only_suffix=".md")).get("README.md", "")
    missing = sorted(e for e in declared_extras() if f".[{e}]" not in readme)
    assert missing == [], f"这些 extra 没有出现在 README.md 里：{missing}"
