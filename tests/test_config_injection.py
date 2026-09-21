"""护栏三：`config.py` 之外，库里没有任何一处读环境变量。

这条规矩的理由在 `docs/configuration.md` 里，一句话是：**值只在
`MemoryConfig.from_env()` 里读一次**。在别处读一次环境变量，那份值就会在
导入时被固化成一个模块级名字 —— 之后测试里换掉的配置**改不到它，而且不报错**。
这个包的前身就踩在这里：测试把根目录指向临时目录，某个模块里那份拷贝还指着
真实数据目录，于是测试写进了生产数据。

所以护栏查的是「读」这个动作，而不是「有没有出现 `os.environ` 这个词」——
`adapters/openai.py` 的 docstring 里有一句
`OpenAILLM(api_key=os.environ["OPENAI_API_KEY"])`，那是在教宿主**自己**去读
一次再传进来，正是这条规矩想要的写法。走 AST 才分得开这两种。
"""

from __future__ import annotations

from _guards import env_read_lines, iter_text_files

LIBRARY = "src/changqing/"
CONFIG = "src/changqing/config.py"


# ---------------------------------------------------------------- 先证明它会红
def test_the_detector_sees_every_way_of_reading_the_environment():
    assert env_read_lines("import os\nX = os.environ['A']\n") == [2]
    assert env_read_lines("import os\nX = os.getenv('A')\n") == [2]
    assert env_read_lines("from os import environ\nX = environ['A']\n") == [1]
    assert env_read_lines("from os import getenv\nX = getenv('A')\n") == [1, 2]


def test_the_detector_ignores_a_mention_inside_a_docstring():
    """文档字符串里教宿主怎么读，不算违规 —— 认错这一条就会逼人删掉正确的文档。"""
    assert env_read_lines('"""OpenAILLM(api_key=os.environ["OPENAI_API_KEY"])"""\n') == []
    assert env_read_lines("# os.getenv('A')\nX = 1\n") == []


# ---------------------------------------------------------------- 再扫仓库
def test_only_config_reads_the_environment():
    offenders: list[str] = []
    scanned = 0
    for name, text in iter_text_files(only_suffix=".py"):
        if not name.startswith(LIBRARY) or name == CONFIG:
            continue
        scanned += 1
        offenders.extend(f"{name}:{line}" for line in env_read_lines(text))
    assert scanned > 20, f"只扫到 {scanned} 个库内文件，扫描器多半坏了"
    assert offenders == [], f"这些地方在自己读环境变量：{offenders}"
