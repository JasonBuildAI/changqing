"""可挂载的记忆路由。**路径是相对的，挂载前缀由宿主决定。**

    from fastapi import FastAPI
    from changqing.server import create_router

    app = FastAPI()
    app.include_router(create_router(), prefix="/api")

于是这里的 `/memory` 在宿主上就是 `/api/memory`。不写死前缀是必须的：写死了就
得让每个宿主都去挪自己的路由表，而「挪一下」这件事在别人的代码里通常等于不改，
于是要么冲突、要么把记忆挂到一个谁也没想到的地址上。

## 这一域共享的那条铁律

    uid **一律从服务端签发的身份取**，绝不接受客户端传参。

`/memory?uid=xxx` 是典型的越权读取（IDOR）：把参数改一个字符就能读别人的记忆。
所以这里**没有任何**一个接口接受 uid —— 它只来自 `uid_of(request)`，默认实现读
`request.state.uid`（那正是「服务端签发」在 FastAPI 里的落点）。宿主用签名 Cookie
或网关注入都行，换的是 `uid_of`，不是这里的路由。

`create_router` 接收它而不是「自己去读某个全局」：记忆库不该猜宿主的会话机制。

## 依赖

这个模块要 FastAPI 与 pydantic（`pip install "changqing[server]"`）。**它不在
核心包里被 import** —— `changqing/__init__.py` 一个字都不提它，装不装 FastAPI
都不影响 `import changqing`。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..edit import EDITABLE_FIELDS, EditError
from ..memory import Memory

__all__ = ["MemoryConfirmReq", "MemoryEditReq", "MemoryForgetReq", "MemoryPinReq", "create_router"]


class MemoryEditReq(BaseModel):
    """改一条事实。`set` 里只认 `EDITABLE_FIELDS`，其余键静默忽略。"""

    id: str
    set: dict[str, Any] = Field(default_factory=dict)


class MemoryForgetReq(BaseModel):
    """删一条。`mode` 留空走配置里的默认（默认是归档，可恢复）。"""

    id: str
    mode: str | None = None


class MemoryConfirmReq(BaseModel):
    id: str
    accept: bool = True


class MemoryPinReq(BaseModel):
    id: str
    pinned: bool = True


def _state_uid(request: Request) -> str:
    """出厂实现：uid 从 `request.state.uid` 取（由宿主的鉴权中间件写上）。

    取不到就 401，**绝不退回空串或默认用户** —— 退回默认用户的症状是
    「所有人都看同一个人的记忆」，而它不会报任何错。
    """
    uid = str(getattr(request.state, "uid", "") or "")
    if not uid:
        raise HTTPException(401, "没有身份：请在鉴权中间件里写 request.state.uid")
    return uid


def create_router(
    *,
    uid_of: Callable[[Request], str] = _state_uid,
    memory_of: Callable[[str], Memory] = Memory,
) -> APIRouter:
    """建一个只含记忆接口的 `APIRouter`。

    `uid_of` 决定「这是谁」，`memory_of` 决定「用哪个句柄」——后者是宿主把
    自己的配置与模型注入进来的地方（`lambda uid: Memory(uid, llm=..., embedder=...)`）。
    两个都做成参数而不是读全局：这样同一个进程里挂两份路由、各自一套配置，
    是一个显式的选择，而不是一件要改库才能做到的事。
    """
    router = APIRouter(tags=["memory"])

    def _mem(request: Request) -> Memory:
        return memory_of(uid_of(request))

    def _dig(mem: Memory, fact_id: str) -> dict[str, Any]:
        """取一条，取不到就是 404。

        **在动任何东西之前判**：不判的话，一条打错 id 的请求会往日志里追加一条
        指向死 id 的操作，而它永远不会有对应的物化结果 —— 之后每次重放都要为它
        走一趟空转。
        """
        fact = mem.get(fact_id)
        if fact is None:
            raise HTTPException(404, f"没有这条事实：{fact_id}")
        return fact

    @router.get("/memory")
    def overview(request: Request) -> dict[str, Any]:
        """这一份记忆的全貌：事实、待确认、纪要、话题、计数。

        `include_dead=True` 与 `include_used=True` 是刻意的：这里看的是
        **她到底记过什么**，不是「现在还能不能用」。已经失效 / 已经提过的那些
        必须看得见 —— 否则「她提过那件事吗」在界面上永远无法回答。
        """
        mem = _mem(request)
        return {
            "facts": mem.get_all(),
            "pending": mem.pending(),
            "summaries": mem.summaries(limit=10),
            "topics": mem.topics(limit=8, include_used=True),
            "stats": mem.stats(),
        }

    @router.post("/memory/edit")
    def edit(req: MemoryEditReq, request: Request) -> dict[str, Any]:
        """改一条事实。追加一条 EDIT 操作，不直接改派生物。"""
        mem = _mem(request)
        wanted = {k: v for k, v in (req.set or {}).items() if k in EDITABLE_FIELDS}
        if not wanted:
            raise HTTPException(400, f"没有可改的字段（只能改 {sorted(EDITABLE_FIELDS)}）")
        # 值校验在**查库之前**：坏值不该因为「这条不存在」被报成 404，
        # 那样用户以为换个 id 就行，实际是值的问题。
        try:
            fact = mem.update(req.id, **wanted)
        except EditError as e:
            raise HTTPException(400, str(e)) from e
        if fact is None:
            raise HTTPException(404, f"没有这条事实：{req.id}")
        return {"ok": True, "fact": fact}

    @router.post("/memory/forget")
    def forget(req: MemoryForgetReq, request: Request) -> dict[str, Any]:
        """删一条。默认归档（可恢复），`mode=purge` 才是真删。"""
        mem = _mem(request)
        _dig(mem, req.id)
        mode = mem.forget(req.id, req.mode or "")
        return {
            "ok": True,
            "mode": mode,
            # **别写成「已彻底删除」**：purge 删掉的是事实行，那一轮的 ADD 操作里
            # 还存着引文、L0 里还存着整轮原话。如实说，用户才知道还有什么留着。
            "note": "已归档，可恢复"
            if mode == "archive"
            else "已从事实库删除（引文与 L0 原话仍在，要走整库重置才会清）",
        }

    @router.post("/memory/confirm")
    def confirm(req: MemoryConfirmReq, request: Request) -> dict[str, Any]:
        """确认 / 否决一条待确认的事实（回引近似命中的那批）。

        否决走归档而不是删除：用户改主意了还能找回来。
        """
        mem = _mem(request)
        _dig(mem, req.id)
        status = mem.confirm(req.id, req.accept)
        return {"ok": True, "status": status}

    @router.post("/memory/pin")
    def pin(req: MemoryPinReq, request: Request) -> dict[str, Any]:
        """钉住 / 取消钉住。钉住的事实无条件进热路径。"""
        mem = _mem(request)
        _dig(mem, req.id)
        return {"ok": True, "pinned": mem.pin(req.id, bool(req.pinned))}

    @router.get("/memory/export")
    def export(request: Request) -> dict[str, Any]:
        """导出全部记忆。**只从日志与事实渲染**，不是快照某个内部结构。"""
        mem = _mem(request)
        out = mem.export()
        out["note"] = "ops 是唯一事实源，可以从它完整重建这一份记忆"
        return out

    return router
