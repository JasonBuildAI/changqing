"""配置：**这个包里唯一读环境变量的地方**。

为什么要单独立一条规矩，而不是各模块各读各的：这个包的前身是一份宿主应用，
它把几十个配置常量在**导入时**拷进每个子模块（`from ... import MEMORY_DIR`）。
那样做的症状是「改不生效、而且不报错」—— 测试里把根目录换成临时目录，
另一个模块拷的那份还指着真实数据目录，于是测试写进了生产数据。
所以这里定死：值只在 `load_config()` 里读一次，读出来的东西是一个
**冻结的数据类**，谁要就往构造器里传（见 `runtime.py` 的 `configure()`）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

# 环境变量前缀。所有配置项都挂在这个前缀下，避免和宿主的变量名打架。
ENV_PREFIX = "CHANGQING_"

# 每个字段读哪个环境变量。这张表同时是 `docs/configuration.md` 里那张表的真源 ——
# 文档里再抄一份的话，两边迟早各自漂，而漂的方向永远是「文档说 A、代码读 B」。
ENV_MAP: dict[str, str] = {
    "enabled": "ENABLED",
    "root": "DIR",
    "reset_mode": "RESET_MODE",
    "forget_mode": "FORGET_MODE",
    "retain_months": "RETAIN_MONTHS",
    "render_views": "RENDER_VIEWS",
    "hot_tokens": "HOT_TOKENS",
    "hot_fetch_max": "HOT_FETCH_MAX",
    "story_k": "STORY_K",
    "story_tokens": "STORY_TOKENS",
    "topic_k": "TOPIC_K",
    "topic_tokens": "TOPIC_TOKENS",
    "recall_k": "RECALL_K",
    "recall_ms": "RECALL_MS",
    "min_score": "MIN_SCORE",
    "min_confidence": "MIN_CONFIDENCE",
    "pending_tolerance": "PENDING_TOLERANCE",
    "embed_dim": "EMBED_DIM",
    "embed_min_cos": "EMBED_MIN_COS",
    "embed_batch": "EMBED_BATCH",
    "idle_min": "IDLE_MIN",
    "max_turns": "MAX_TURNS",
    "extract_max_calls": "EXTRACT_MAX_CALLS",
    "extract_model": "EXTRACT_MODEL",
    "extract_tokens_per_turn": "EXTRACT_TOKENS_PER_TURN",
    "extract_max_tokens": "EXTRACT_MAX_TOKENS",
    "consolidate_days": "CONSOLIDATE_DAYS",
    "decay_floor": "DECAY_FLOOR",
    "merge_similarity": "MERGE_SIMILARITY",
    "commitment_slots": "COMMITMENT_SLOTS",
    "idle_split_sec": "IDLE_SPLIT_SEC",
}


def _raw(suffix: str) -> str:
    """取环境变量原始值。**空串当没设置**。

    较真这一条的理由：`.env` 里写 `CHANGQING_DIR=`（留空表示"用默认"）时，
    `os.environ` 给的是空串而不是默认值，于是根目录变成当前目录 ——
    在用户眼里就是「我什么都没改，记忆库突然搬家了」。
    """
    return (os.environ.get(ENV_PREFIX + suffix) or "").strip()


def _env_str(suffix: str, default: str) -> str:
    return _raw(suffix) or default


def _env_int(suffix: str, default: int) -> int:
    try:
        return int(_raw(suffix))
    except ValueError:
        return default          # 写错一行不该让整个库起不来


def _env_float(suffix: str, default: float) -> float:
    try:
        return float(_raw(suffix))
    except ValueError:
        return default


def _env_bool(suffix: str, default: bool) -> bool:
    v = _raw(suffix).lower()
    if not v:
        return default
    return v not in ("0", "off", "no", "false")


def default_root() -> Path:
    """默认库根：`~/.changqing`。

    放在用户主目录而不是当前目录，是因为这个库的默认用法是一个长期存在的
    服务进程；把记忆写进「启动时的工作目录」意味着换个地方启动就读不到
    昨天的会话了。
    """
    return Path.home() / ".changqing"


@dataclass(frozen=True)
class MemoryConfig:
    """这份记忆系统的全部可调项。冻结的（`frozen=True`）。

    冻结不是洁癖：配置对象会在多个线程之间共享（后台整理线程 + 前台请求），
    可变的话「这一轮读到一半被改了」是一种无法复现的偶发。要改就
    `dataclasses.replace(cfg, hot_tokens=500)` 造一个新的。
    """

    # ---------------------------------------------------------------- 开关与位置
    enabled: bool = True
    root: Path = None  # type: ignore[assignment]  # __post_init__ 里给默认值
    reset_mode: str = "purge"
    forget_mode: str = "archive"
    retain_months: int = 24
    render_views: bool = True

    # ---------------------------------------------------------------- 注入预算
    # 这几个数直接决定「每轮花多少 token」，是成本的主要闸门。
    hot_tokens: int = 300
    hot_fetch_max: int = 400
    story_k: int = 3
    story_tokens: int = 160
    topic_k: int = 2
    topic_tokens: int = 80
    recall_k: int = 4
    recall_ms: int = 50
    min_score: float = 0.35
    min_confidence: float = 0.5
    pending_tolerance: float = 0.6

    # ---------------------------------------------------------------- 向量
    # 注意这里**没有** provider / model / mirror：向量能力由宿主注入
    # `Embedder`（见 `ports.py`），库本身不再自带任何模型下载逻辑。
    embed_dim: int = 512
    embed_min_cos: float = 0.48
    embed_batch: int = 32

    # ---------------------------------------------------------------- 后台整理
    idle_min: int = 30
    max_turns: int = 200
    extract_max_calls: int = 8
    extract_model: str = ""
    extract_tokens_per_turn: int = 160
    extract_max_tokens: int = 8000
    consolidate_days: int = 90
    decay_floor: float = 0.05
    merge_similarity: float = 0.8
    commitment_slots: int = 3

    # ---------------------------------------------------------------- 唯一的空闲判据
    # **不许引入第二个空闲概念**。它既决定喂给模型的上下文何时重开，
    # 也决定「这一场」的边界（整理按场做）。
    idle_split_sec: int = 600

    def __post_init__(self) -> None:
        if self.root is None:
            object.__setattr__(self, "root", default_root())
        else:
            object.__setattr__(self, "root", Path(self.root).expanduser())
        if self.retain_months < 1:
            object.__setattr__(self, "retain_months", 1)
        if self.embed_dim < 1:
            object.__setattr__(self, "embed_dim", 1)
        if self.idle_split_sec < 0:
            object.__setattr__(self, "idle_split_sec", 0)
        if not 0.0 <= self.decay_floor <= 1.0:
            object.__setattr__(self, "decay_floor", 0.05)

    # ---------------------------------------------------------------- 构造
    @classmethod
    def from_env(cls, *, prefix: str | None = None) -> MemoryConfig:
        """从环境变量读一份配置。缺省值就是上面那些类属性。"""
        global ENV_PREFIX
        old = ENV_PREFIX
        if prefix is not None:
            ENV_PREFIX = prefix
        try:
            return cls(
                enabled=_env_bool("ENABLED", True),
                root=Path(_env_str("DIR", str(default_root()))),
                reset_mode=_env_str("RESET_MODE", "purge").lower(),
                forget_mode=_env_str("FORGET_MODE", "archive").lower(),
                retain_months=_env_int("RETAIN_MONTHS", 24),
                render_views=_env_bool("RENDER_VIEWS", True),
                hot_tokens=_env_int("HOT_TOKENS", 300),
                hot_fetch_max=_env_int("HOT_FETCH_MAX", 400),
                story_k=_env_int("STORY_K", 3),
                story_tokens=_env_int("STORY_TOKENS", 160),
                topic_k=_env_int("TOPIC_K", 2),
                topic_tokens=_env_int("TOPIC_TOKENS", 80),
                recall_k=_env_int("RECALL_K", 4),
                recall_ms=_env_int("RECALL_MS", 50),
                min_score=_env_float("MIN_SCORE", 0.35),
                min_confidence=_env_float("MIN_CONFIDENCE", 0.5),
                pending_tolerance=_env_float("PENDING_TOLERANCE", 0.6),
                embed_dim=_env_int("EMBED_DIM", 512),
                embed_min_cos=_env_float("EMBED_MIN_COS", 0.48),
                embed_batch=_env_int("EMBED_BATCH", 32),
                idle_min=_env_int("IDLE_MIN", 30),
                max_turns=_env_int("MAX_TURNS", 200),
                extract_max_calls=_env_int("EXTRACT_MAX_CALLS", 8),
                extract_model=_env_str("EXTRACT_MODEL", ""),
                extract_tokens_per_turn=_env_int("EXTRACT_TOKENS_PER_TURN", 160),
                extract_max_tokens=_env_int("EXTRACT_MAX_TOKENS", 8000),
                consolidate_days=_env_int("CONSOLIDATE_DAYS", 90),
                decay_floor=_env_float("DECAY_FLOOR", 0.05),
                merge_similarity=_env_float("MERGE_SIMILARITY", 0.8),
                commitment_slots=_env_int("COMMITMENT_SLOTS", 3),
                idle_split_sec=_env_int("IDLE_SPLIT_SEC", 600),
            )
        finally:
            ENV_PREFIX = old

    def evolved(self, **changes: object) -> MemoryConfig:
        """改几个字段，返回**新的**一份（原配置不动）。"""
        return replace(self, **changes)

