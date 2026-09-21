"""基准与评测脚本本身也要被测 —— 它们是最容易烂掉的一类代码。

一份没人跑的基准，过两个月就会安静地对不上现实：它照样打印一屏数字，只是那些
数字不再来自真实的代码路径。所以这里用**子进程**把它们真的跑一遍。

默认档里唯一不跑的是需要真模型的那条（`online`）：它要 Key 要网络，
一条会在别人机器上假红的用例比没有用例更糟。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_script(
    script: str, *args: str, env: dict | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONIOENCODING="utf-8", **(env or {})),
        cwd=ROOT,
        timeout=900,
    )


@pytest.mark.slow
def test_scale_benchmark_runs_offline() -> None:
    """规模基准：不联网也能跑完，而且四段报告都在。"""
    done = run_script("benchmarks/scale.py", "--users", "1", "--turns", "20", "--queries", "4")
    assert done.returncode == 0, done.stderr[-2000:]
    for marker in ("[L0]", "[整理]", "[检索]", "[占用]", "[投影]"):
        assert marker in done.stdout, f"少了 {marker} 这一段：\n{done.stdout}"
    assert "ms/轮" in done.stdout


class _FakeModel(BaseHTTPRequestHandler):
    """一个假端点：永远回「第一条原话的前八个字」。

    它刻意**不是很准**：有些标注的关键内容落在这八个字里（算命中），有些落在
    后面（算漏抽）—— 于是召回不可能满，阈值那一行才验得成。
    """

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"] or 0)))
        prompt = str((body.get("messages") or [{}])[-1].get("content") or "")
        lines = re.findall(r"^\[(T-\d+)\][^\n]*?他说：(.*)$", prompt, re.M)
        facts = []
        if lines:
            tid, text = lines[0][0], lines[0][1].strip()
            facts.append(
                {
                    "subject": "他",
                    "predicate": "提到",
                    "object": text[:8],
                    "quote": text[:8],
                    "turn_ref": tid,
                    "kind": "fact",
                    "confidence": 0.9,
                }
            )
        payload = {
            "choices": [
                {
                    "message": {"content": json.dumps({"facts": facts}, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 22},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        return None  # 别把请求日志混进被测输出


@pytest.fixture()
def fake_model() -> str:
    """起一个真的 HTTP 服务，返回它的 base_url。

    **不用 httpx 的 MockTransport**：那条路绕过了真实 HTTP 这一层（状态码、
    Content-Length、编码），而评测脚本与适配器之间的接缝恰好全在那儿。
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_extraction_eval_scores_and_fails_its_threshold(fake_model: str) -> None:
    """端到端跑一遍评测：指标真的算出来了，而且阈值**真的会拦**。"""
    env = {"OPENAI_API_KEY": "test-key-not-real"}
    done = run_script("benchmarks/extraction_eval.py", "--base-url", fake_model, env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    assert "召回" in done.stdout and "精确" in done.stdout
    # 假模型只答对第一例，所以召回不可能满 —— 阈值该把它拦下来。
    strict = run_script(
        "benchmarks/extraction_eval.py",
        "--base-url",
        fake_model,
        "--min-recall",
        "1.0",
        env=env,
    )
    assert strict.returncode == 1, strict.stdout[-800:]
    # 反方向也要成立：阈值调低就该过 —— 否则上一行的红可能只是脚本崩了。
    loose = run_script(
        "benchmarks/extraction_eval.py",
        "--base-url",
        fake_model,
        "--min-recall",
        "0.1",
        env=env,
    )
    assert loose.returncode == 0, loose.stderr[-800:]


def test_extraction_eval_refuses_to_run_without_a_key() -> None:
    """没有 Key 退出码 2，不静默通过：静默跳过的评测跑一百遍也还是「全绿」。"""
    done = run_script("benchmarks/extraction_eval.py", env={"OPENAI_API_KEY": ""})
    assert done.returncode == 2
    assert "OPENAI_API_KEY" in (done.stderr + done.stdout)


@pytest.mark.online
def test_extraction_eval_against_a_real_model() -> None:
    """真模型那一趟。默认档不选它（`-m 'not online'`）。"""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("没有 OPENAI_API_KEY")
    done = run_script("benchmarks/extraction_eval.py", "--min-recall", "0.6")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
