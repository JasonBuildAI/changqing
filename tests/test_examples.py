"""例子必须**真的跑得起来** —— 没 Key、没网络、没模型文件。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_example(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """在**另一个进程**里跑例子，库根指到临时目录。

    为什么要另起进程而不是 import 进来：`from_env()` 与进程级的运行期都是
    导入时/进程级的东西，同进程跑会读到测试自己的状态 —— 那样验的就不是
    「用户照 README 敲那两行会怎样」，而是「我们的测试环境恰好配成了什么」。
    库根必须换掉：例子默认写 `~/.changqing`，测试往那儿写一次就再也洗不干净。
    """
    env = dict(os.environ, CHANGQING_DIR=str(tmp_path / "memory"), PYTHONIOENCODING="utf-8")
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=ROOT,
        timeout=300,
    )


def test_quickstart_runs_without_any_api_key(tmp_path: Path) -> None:
    """快速上手：写了原话、抽出了事实、检索到了、没相关的事时不硬凑。"""
    done = run_example("examples/quickstart.py", tmp_path)
    assert done.returncode == 0, done.stderr[-2000:]
    out = done.stdout
    assert "L0 写下 2 轮原话" in out, out
    assert "他养的猫叫团子" in out, out
    # 第二问故意问一件没说过的事：空列表是正确答案，不是失败 —— 这条钉住
    # 「分数不够就返回空」在**默认配置**下真的成立。
    assert "（没有够格的事）" in out, out
    assert (tmp_path / "memory").is_dir(), "例子没写在 CHANGQING_DIR 指的地方"
