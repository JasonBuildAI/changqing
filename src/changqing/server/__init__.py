"""可选的服务端接入层：一个可挂载的 FastAPI 路由。

**它在核心的 import 图上不存在。** `import changqing` 不碰这里，所以没装
FastAPI 的机器照样能用整套记忆能力 —— 这个包只在你 `from changqing.server
import create_router` 的那一刻才需要 `[server]` 那一组依赖。

它提供的是**接口**，不是界面：查、改、删、确认、钉、导出六条。任何前端
都不在这个仓库里 —— 各家要呈现的东西不一样，硬塞一份进去只会限制别人。
"""

from __future__ import annotations

from .router import (
    MemoryConfirmReq,
    MemoryEditReq,
    MemoryForgetReq,
    MemoryPinReq,
    create_router,
)

__all__ = [
    "MemoryConfirmReq",
    "MemoryEditReq",
    "MemoryForgetReq",
    "MemoryPinReq",
    "create_router",
]
