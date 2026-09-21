"""目录布局与共享底座：路径、每个 uid 的写锁、`state.json`。

这一层落在依赖图的最底下，零包内依赖 —— `turnlog` 要写锁与 state、
`ops` 要日志路径与事实序号、`index` 要索引路径，放在任何一个业务子模块里，
别人都得反过来依赖它（「日志」依赖「对话流水」是假依赖）。

**与宿主版本最大的一处差别**：这里没有「活值回退」。
宿主把库根目录在导入时拷进本模块，于是测试里改根目录会静默失效，只能再写一段
「去父包上读那一份活值、如果它变了就用它」的补偿逻辑。那个补偿本身就是缺陷的
证据。这里根目录来自 `runtime().config.root`，**每次调用现取**，改配置就是改配置。
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..runtime import runtime

_STATE_FILE = "state.json"
_SESSIONS = "sessions"

# 每个 uid 一把写锁：后台整理与「编辑一条事实」这两条路必须串行化，
# 否则并发改写会基于旧快照写出错误的 SUPERSEDE。
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(uid: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lk = _LOCKS.get(uid)
        if lk is None:
            lk = _LOCKS[uid] = threading.Lock()
        return lk


# ---------------------------------------------------------------- 路径
def safe_name(name: str, fallback: str = "default") -> str:
    """只允许字母数字和 `_` `-`，防路径注入。

    uid 一般是调用方签发的，正常情况下一定合法；但目录名最终会拼进文件系统
    路径，多一道白名单的成本是零。
    """
    s = "".join(c for c in str(name or "") if c.isalnum() or c in ("_", "-"))
    return s or fallback


def root_dir() -> Path:
    """当前生效的库根。每次现取，不在导入时拷走。"""
    return Path(runtime().config.root)


def user_dir(uid: str) -> Path:
    """用户根目录。按 uid 前两位分桶，避免一个目录塞进十万个用户。"""
    u = safe_name(uid, "local")
    return root_dir() / u[:2] / u


def sessions_dir(uid: str) -> Path:
    return user_dir(uid) / _SESSIONS


def day_path(uid: str, day: str) -> Path:
    return sessions_dir(uid) / f"{safe_name(day, 'unknown')}.md"


def _state_path(uid: str) -> Path:
    return user_dir(uid) / _STATE_FILE


# ---------------------------------------------------------------- state
def _load_state(uid: str) -> dict[str, Any]:
    p = _state_path(uid)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        # state 坏了不该让记忆写不下去：下一轮从 1 重新编号只是浪费 id，
        # 「永不复用」靠的是单调递增，不靠它绝对连续。
        return {}


def _save_state(uid: str, st: dict[str, Any]) -> None:
    """`state.json` 不是追加日志，是一个**小**可变文件，所以这里可以用
    「临时文件 + `os.replace`」拿原子性。

    **临时文件名必须带 pid 与线程 id**：固定叫 `state.tmp` 时，两个写者同时写
    会让 `os.replace` 把对方正在写的那个文件搬走 —— 于是其中一次写入**静默丢掉**，
    而丢的正好可能是刚推上去的 `next_seq` / `next_fact_seq`。
    """
    p = _state_path(uid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{os.getpid()}-{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass


def update_state(uid: str, **changes: Any) -> dict[str, Any]:
    """**读-改-写**，整段在写锁里。

    为什么不能「先 load 再 save」：整理线程手上那份快照中间隔着几百毫秒的活
    （模型调用），期间前台可能已经推了几轮原话上去。直接 `save_state` 会把
    前台这期间的写入盖掉 —— 症状是「她偶尔丢掉刚说过的一轮」，不报错。

    改的是**顶层**键。要合并嵌套字段（`watermark` 那种）走 `mutate_state`。
    """
    return mutate_state(uid, lambda st: st.update(changes))


def mutate_state(uid: str, fn: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """把**整个改动**放进同一把锁里：锁内读、`fn` 改、锁内写。

    为什么需要它，而不只是 `update_state(**changes)`：有些字段是嵌套的
    （`watermark` 是一个小字典，上面挂着 `last_ts` 与 `extracted_upto`）。
    写它必须基于**锁内读到的那一份**做合并 —— 在锁外先把 watermark 读出来、
    算好整个新字典再写回，中间夹着的那些前台写入（每轮都在更新的 `last_ts`）
    会被静静盖掉，而症状只是「静默整理的时机偶尔不对」，不报错。
    """
    with _lock(uid):
        st = _load_state(uid)
        fn(st)
        _save_state(uid, st)
        return st


def load_state(uid: str) -> dict[str, Any]:
    return _load_state(uid)


def save_state(uid: str, st: dict[str, Any]) -> None:
    """**整个写回**。只在「手上这份就是最新」时用（例如测试预置状态）。

    整理那条路要走 `update_state`。
    """
    _save_state(uid, st)


def watermark(uid: str) -> dict[str, Any]:
    """整理游标：上一次整理读到哪一轮。它决定「下次从哪儿接着抽」。"""
    w = _load_state(uid).get("watermark")
    return w if isinstance(w, dict) else {}


_LOG_FILE = "log.jsonl"
_INDEX_FILE = "index.sqlite"


def log_path(uid: str) -> Path:
    return user_dir(uid) / _LOG_FILE


def index_path(uid: str) -> Path:
    return user_dir(uid) / _INDEX_FILE


def facts_md_path(uid: str) -> Path:
    return user_dir(uid) / "facts.md"


def summary_path(uid: str) -> Path:
    return user_dir(uid) / "summaries.md"
