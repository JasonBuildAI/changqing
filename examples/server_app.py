"""把记忆路由挂进一个真实的 FastAPI 应用。**不需要任何 API Key。**

    pip install "changqing[server]"
    python examples/server_app.py
    # 然后：
    curl -s localhost:8000/api/memory | python -m json.tool

这里演示三件在真实宿主里必须做的事，而它们在库内部是**做不到**的：

1. **身份从哪来。** 记忆只认 `request.state.uid`（服务端签发），这里用一段
   中间件从 `X-Demo-User` 头里取。真实宿主该换成签名 Cookie / 网关 / 会话 ——
   换的是这段中间件，路由一个字都不用改。
2. **配置与模型从哪来。** `memory_of` 是宿主注入的地方：一个 `MemoryConfig`
   加一组能力。这里用离线 mock（确定、零网络），换成 `OpenAILLM` 就上线。
3. **前端不在库里。** 库给的是接口，长什么样由宿主决定。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, Request

from changqing import Memory, MemoryConfig, PersonaProfile
from changqing.adapters.mock import MockEmbedder, MockLLM
from changqing.server import create_router

# 演示用临时目录。真实宿主会写一个固定的路径（配置里那一项就是干这个的）。
ROOT = Path(tempfile.mkdtemp(prefix="changqing-demo-"))
CONFIG = MemoryConfig(root=ROOT)

# 一份**中性**画像：抽取时给模型一份「她是哪种人」的说明，事实的
# `persona_attention` 就是按它打的分。换成你自己的角色设定即可。
NEUTRAL_DEMO = PersonaProfile(
    name="assistant",
    description="一个长期陪在身边的助手：话不多，记得住细节。",
    attention_hint="跟他的日常、约定、和他在意的人有关的事。",
)

# 抽取的输出格式。真实模型会自己按 prompt 生成这一段，这里写死一份，
# 好让例子在没有 Key 的机器上也能真的把事实记下来。
SCRIPTED = json.dumps(
    {
        "facts": [
            {
                "subject": "他",
                "predicate": "养的猫叫",
                "object": "团子",
                "quote": "我家猫叫团子",
                "turn_ref": "T-000001",
                "kind": "fact",
                "confidence": 0.9,
                "importance": 0.6,
                "persona_attention": 0.5,
            }
        ],
        "summary": "第一次聊到他的猫。",
        "topics": [],
    },
    ensure_ascii=False,
)

app = FastAPI(title="changqing demo")


@app.middleware("http")
async def attach_identity(request: Request, call_next):
    """把「这是谁」写成路由读得到的那个位置。带 `X-Demo-User` 就换人。"""
    request.state.uid = str(request.headers.get("X-Demo-User") or "demo-user")
    return await call_next(request)


def memory_of(uid: str) -> Memory:
    """宿主注入点：这个 uid 用哪份配置、哪组能力。"""
    return Memory(
        uid,
        config=CONFIG,
        persona=NEUTRAL_DEMO,
        embedder=MockEmbedder(),
        llm=MockLLM(default=SCRIPTED),
    )


app.include_router(create_router(memory_of=memory_of), prefix="/api")


@app.post("/api/demo/say")
def say(text: str, request: Request) -> dict:
    """走一遍完整链路：记下原话 → 整理成事实 → 检索回来。

    真实宿主不会这么调：原话由对话链路写，整理由后台线程按三个触发条件跑。
    这里合成一步，是为了让例子里能立刻看到结果。
    """
    mem = memory_of(str(request.state.uid))
    turns = mem.remember({"user": text, "assistant": "记住了。"})
    extracted = mem.extract_now()
    return {
        "turns": turns,
        "extracted": extracted,
        "facts": mem.get_all(),
    }


if __name__ == "__main__":
    import uvicorn

    print(f"memory root: {ROOT}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
