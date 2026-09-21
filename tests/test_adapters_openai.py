"""OpenAI 兼容适配器：**不联网**就能测到的部分。

用 `httpx.MockTransport` 把请求截下来 —— 于是「发出去的 payload 长什么样」与
「坏响应怎么处理」这两件事都能断言。它们正是这一层仅有的两件事：真正的模型质量
不归适配器管（那是 `benchmarks/extraction_eval.py` 的活）。

这里刻意**不**测网络本身：一条要外网的用例就是一条会在别人的机器上假红的用例。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

httpx = pytest.importorskip("httpx", reason="适配器需要 changqing[openai]")

from changqing.adapters.openai import EMBED_DIMS, OpenAIEmbedder, OpenAILLM  # noqa: E402
from changqing.ports import LLM, Embedder, embedder_enabled  # noqa: E402


def fake_client(handler) -> httpx.Client:
    """把请求交给一段函数，而不是交给网络。"""
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://x/v1")


def test_embedder_sends_the_whole_batch_in_one_request() -> None:
    """一批一次请求：逐条发的话，一次整理要几百个来回。"""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        seen[-1]["auth"] = request.headers.get("authorization", "")
        seen[-1]["path"] = request.url.path
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    emb = OpenAIEmbedder("k", model="m", dim=2, client=fake_client(handler))
    assert emb.encode(["甲", "乙"]) == [[1.0, 0.0], [0.0, 1.0]], "必须按 index 归位"
    assert seen[0]["model"] == "m"
    assert seen[0]["input"] == ["甲", "乙"]
    assert seen[0]["path"] == "/v1/embeddings"
    assert seen[0]["auth"] == "Bearer k"


def test_embedder_returns_none_instead_of_raising() -> None:
    """向量挂掉只是这一轮降级到全文检索，不是整条链路失败。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    emb = OpenAIEmbedder("k", model="m", dim=2, client=fake_client(handler))
    assert emb.encode(["甲"]) is None
    assert "HTTPStatusError" in emb.last_error, emb.last_error


def test_embedder_refuses_a_dimension_mismatch() -> None:
    """维度不符当**这次编码失败**，而不是静默收下。

    收下的后果不是报错，是相似度变成噪声 —— 而且统计还报一切正常。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]})

    emb = OpenAIEmbedder("k", model="m", dim=2, client=fake_client(handler))
    assert emb.encode(["甲"]) is None
    assert "维度不符" in emb.last_error, emb.last_error


def test_unknown_embedding_model_requires_an_explicit_dim() -> None:
    """表里没有的模型宁可当场报错：默认猜一个维度等于把索引废掉还不说。"""
    with pytest.raises(ValueError, match="dim"):
        OpenAIEmbedder("k", model="某家自研的-embedding")
    assert EMBED_DIMS["text-embedding-3-small"] == 1536


def test_embedder_names_itself_for_the_index_and_for_the_kill_switch() -> None:
    """`name` 管「这条路装没装」，`repo` 管「这批向量是谁编的」，两者不能混。"""
    emb = OpenAIEmbedder("k", model="text-embedding-3-small")
    assert embedder_enabled(emb), "非空且不是 none/off/0 才算装上了"
    assert emb.repo == "openai/text-embedding-3-small"
    assert emb.dim == 1536


def test_llm_reports_finish_reason_and_usage() -> None:
    """`finish_reason` 与用量都要回传：前者是「输出被砍断」的唯一信号。"""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["max_tokens"] == 99
        assert body["model"] == "m"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    why: list[str] = []
    used: list[dict] = []
    llm = OpenAILLM("k", model="m", client=fake_client(handler))
    text = llm(
        [{"role": "user", "content": "嗨"}],
        max_tokens=99,
        on_finish=why.append,
        on_usage=used.append,
    )
    assert text == "{}"
    assert why == ["length"], "截断与「没什么可抽的」必须分得开"
    assert used == [{"tin": 7, "tout": 3}]


def test_llm_raises_so_the_extraction_cursor_does_not_advance() -> None:
    """调用失败照实抛：上游把它读成「这次失败，游标不许推进，下次重试」。

    吞成空串的话，那几轮原话会被当成**已经整理过** —— 静默丢记忆。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    llm = OpenAILLM("k", model="m", client=fake_client(handler))
    with pytest.raises(httpx.HTTPStatusError):
        llm([{"role": "user", "content": "嗨"}])


def test_api_key_is_required_and_must_be_explicit() -> None:
    """不读环境变量（理由见模块 docstring）：缺 Key 当场报错，不留一个会联网的默认。"""
    with pytest.raises(ValueError, match="api_key"):
        OpenAILLM("")
    with pytest.raises(ValueError, match="api_key"):
        OpenAIEmbedder("   ", model="m", dim=2)


def test_both_adapters_satisfy_the_protocols_they_are_injected_as() -> None:
    """结构化子类型要在**装配处**就能问出来，而不是等第一次检索时才炸。

    注入点收的是 `Embedder` / `LLM` 两个 Protocol（见 `changqing.ports`）。
    少一个方法（比如 `encode`）在这里就该是 False —— 这条断言是会失败的那种：
    把 `OpenAIEmbedder.encode` 改个名字，它立刻红。
    """
    assert isinstance(OpenAIEmbedder("k", model="m", dim=2), Embedder)
    assert isinstance(OpenAILLM("k", model="m"), LLM)


def test_core_never_pulls_in_httpx() -> None:
    """核心包**零第三方依赖**这句话得能验：把 httpx 变成一个导入即炸的模块，

    `import changqing` 与 `import changqing.adapters` 都还要能活。
    这条会失败 —— 谁在模块级写上一句 `from .openai import ...`，它立刻红。
    另起进程是因为「这个进程已经导入过 httpx 了」，在本进程里验不出什么。
    """
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'httpx' or name.startswith('httpx.'):\n"
        "            raise ImportError('httpx 不在这台机器上')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import changqing, changqing.adapters\n"
        "print(changqing.__version__, 'httpx' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr[-1500:]
    assert done.stdout.strip().endswith("False"), done.stdout


def test_close_is_safe_to_call_twice_and_never_closes_an_injected_client() -> None:
    """注入进来的 client 由注入方负责关 —— 替他关掉是「我借了你的车还把它报废」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    client = fake_client(handler)
    emb = OpenAIEmbedder("k", model="m", dim=2, client=client)
    emb.close()
    emb.close()
    assert client.is_closed is False
