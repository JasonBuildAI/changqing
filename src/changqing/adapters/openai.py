"""OpenAI 兼容的两种能力：嵌入与对话模型（装 `changqing[openai]`）。

    from changqing.adapters.openai import OpenAIEmbedder, OpenAILLM

**为什么 API Key 必须显式传，这个库不读 `OPENAI_API_KEY`。**
读环境变量是**宿主**的事：它可能从 `.env` 读、从密钥服务读、从当前请求的上下文
里取（多租户就是一家一个 Key）。一个库背着调用方去读环境，症状是「我以为它没配，
它却在联网」—— 而这一条正是配置边界要守的地方：`config.py` 是**唯一**读环境变量
的模块（护栏见 `tests/test_no_env_leaks.py`）。想读环境的写法只有一行：

    OpenAILLM(api_key=os.environ["OPENAI_API_KEY"])

**为什么用 httpx 而不是官方 SDK。** 这里只需要两个 POST。少一个重依赖，也少一次
版本同步；代价是错误形状要自己按 HTTP 的状态码处理，下面每处都写清了取舍。

**`base_url` 是给兼容端点留的**：自建 vLLM / Ollama / 各家兼容网关都只用这一个参数
就能接上，不必为每家写一个适配器。
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import Any

import httpx

# 已知嵌入模型的维度。**它必须是对的**：向量索引按维度判「这批向量还能不能用」，
# 维度报错的话旧向量会被原样留着、查询向量来自另一个空间，相似度全是噪声，
# 而统计还报一切正常。表里没有的模型要求显式传 `dim`（宁可当场报错）。
EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class _Endpoint:
    """两种能力共用的那一点点：连接、鉴权头、超时、一次 POST。"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float,
        client: Any = None,
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError("必须显式传 api_key（这个库不读环境变量，理由见模块 docstring）")
        self.api_key = str(api_key)
        self.base_url = str(base_url or "").rstrip("/")
        self.timeout = float(timeout)
        self.last_error = ""
        # 允许注入现成的 client：测试用 `httpx.MockTransport` 挂上来，
        # 于是这些适配器可以在**不联网**的情况下被测到。
        self._client = client
        # **谁建的谁关**。注入进来的 client 属于注入方（宿主可能拿它跑别的请求），
        # 替他关掉是「我借了你的车还把它报废」——而且他那边下一次请求会炸在一个
        # 与本库毫无关系的地方。
        self._owns_client = client is None

    def client(self) -> Any:
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # 鉴权头**挂在这次请求上**，不挂在 client 上：注入进来的 client 是我们
        # 借来的，往它身上写 header 等于改别人的东西（而且他会拿去发别的请求）。
        resp = self.client().post(
            path, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError(f"{path} 返回的不是一个对象：{type(data).__name__}")
        return data

    def close(self) -> None:
        """放掉连接池。**谁建的谁关**（见 `__init__`）。重复调用是安全的。"""
        client = self._client
        if not self._owns_client or client is None:
            return
        self._client = None
        # 关连接失败不该再抛：这时候调用方已经在收尾了。
        with contextlib.suppress(Exception):
            client.close()


class OpenAIEmbedder(_Endpoint):
    """OpenAI 兼容的嵌入端点。

    `name` 是 `"openai"` 而不是模型名：`ports.embedder_enabled` 判的是
    「这条路装没装」（它认 `none` / `off` / `0` 为关），而「用的是哪个模型」由
    `repo` 回答 —— 向量索引靠 `repo` 认「这批向量是谁编的」，两者混在一起会让
    换模型的检测失效。
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 0,
        timeout: float = 30.0,
        client: Any = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout=timeout, client=client)
        self.model = str(model or "")
        self.dim = int(dim or EMBED_DIMS.get(self.model, 0))
        if self.dim < 1:
            raise ValueError(f"不认识的嵌入模型 {self.model!r}：请显式传 dim=<该模型的向量维度>")
        self.name = "openai"
        self.repo = f"openai/{self.model}"

    def encode(self, texts: Sequence[str]) -> list[list[float]] | None:
        """批量编码。**失败返回 None，不抛** —— 向量是锦上添花的一路，

        它挂掉只该让这一轮降级到全文检索，而不是让整条对话取不到任何记忆。
        """
        batch = [str(t or "") for t in texts]
        if not batch:
            return []
        try:
            data = self.post("/embeddings", {"model": self.model, "input": batch})
            rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
            out = [[float(v) for v in row.get("embedding") or []] for row in rows]
        except Exception as exc:  # noqa: BLE001  见 docstring：降级而不是抛
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None
        if len(out) != len(batch):
            self.last_error = f"返回 {len(out)} 条，请求了 {len(batch)} 条"
            return None
        wrong = [len(v) for v in out if len(v) != self.dim]
        if wrong:
            # 静默收下的后果是「向量召回一直不命中」或更糟：拿不同维度的向量
            # 算相似度。宁可当这次编码失败 —— 下一条路径（全文检索）是好的。
            self.last_error = (
                f"维度不符：声明 dim={self.dim}（模型 {self.model}），"
                f"实际拿到 {wrong[0]}。请把 dim 改成正确的值"
            )
            return None
        return out

    def ready(self, download: bool = True) -> bool:
        """远程端点没有「本地有没有」这一问：Key 在就可用。"""
        return bool(self.api_key)

    def warm(self) -> bool | None:
        return None  # 没有一次性加载开销（None = 不需要预热）

    def loaded(self) -> bool:
        return bool(self.api_key)


class OpenAILLM(_Endpoint):
    """OpenAI 兼容的对话补全（非流式）。

    出错**照实抛**（与 `OpenAIEmbedder.encode` 相反）：这条路上游的抽取把它读成
    「这次调用失败了，游标不许推进」—— 吞成一个空串的话，那几轮原话会被当成
    「已经整理过」，表现就是**静默丢记忆**。
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float | None = 0.0,
        timeout: float = 60.0,
        client: Any = None,
    ) -> None:
        super().__init__(api_key, base_url, timeout=timeout, client=client)
        self.model = str(model or "")
        self.temperature = temperature

    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        max_tokens: int = 0,
        on_usage: Any = None,
        on_finish: Any = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": str(model or self.model),
            "messages": list(messages),
        }
        if int(max_tokens) > 0:
            # 兼容端点（vLLM / 各家网关）认这个名字；新一点的 OpenAI 模型改用
            # max_completion_tokens，宿主可以自己传 `model=` 去换端点，但**上限
            # 不能省**：抽取的输出被砍断与「没什么可抽的」长得一模一样。
            payload["max_tokens"] = int(max_tokens)
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        data = self.post("/chat/completions", payload)
        choice: dict[str, Any] = (data.get("choices") or [{}])[0]
        if on_finish is not None:
            on_finish(str(choice.get("finish_reason") or ""))
        usage = data.get("usage") or {}
        if on_usage is not None:
            on_usage(
                {
                    "tin": int(usage.get("prompt_tokens") or 0),
                    "tout": int(usage.get("completion_tokens") or 0),
                }
            )
        return str((choice.get("message") or {}).get("content") or "")


__all__ = ["EMBED_DIMS", "OpenAIEmbedder", "OpenAILLM"]
