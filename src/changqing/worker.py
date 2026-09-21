"""后台整理：把 L0 原话变成 L1 事实。

**三个触发条件（任一）**：
  a. 静默超过 `idle_min` 分钟 —— 「用户不说了」是整理最自然的时机
  b. 距上次整理累计超过 `max_turns` 轮 —— 长对话中途也要落一次
  c. 用户下次活跃时扫 watermark —— 补做没做完的（用户直接关页面是常态）

**为什么 c 是「下次活跃时」而不是「启动时」**：服务器上用户成千上万，开机
全量扫描会变成启动风暴。按用户懒扫描即可。

静默扫描（a）随之要**按批轮转**（`SCAN_BATCH`）：每次空转只看一批，游标往后走。
固定看前一批的话，排在后面的人永远轮不到 —— 而「用户直接关页面」恰恰是最
常见的结束方式，漏掉就等于那段对话永远停在 L0。

**调用频率纪律（写死在代码里）**：每轮对话的记忆类模型调用 = **0**；
单场对话整理调用 ≤ `config.extract_max_calls`（默认 8）。
绝不允许滑向「每轮一次」——那是 300 次/场，成本差两个数量级。

**幂等**：整理两次的结果必须一样。落点在 `extract.resolve_ops` 的 NOOP，
以及游标只在整段成功之后才推进（失败就下次重试，不会漏也不会重）。

**没有注入模型时不静默降级成「空事实」**：`NullLLM` 会抛
`LLMNotConfigured`，这里把它识别成「这条路本来就不通」（`_call_extract`
返回 `""`，游标照常推进），而不是「这次调用失败了」（返回 `None`，
游标不动、下次重试）。两者混起来会造成**静默丢数据或无限重试**。
"""

from __future__ import annotations

import contextlib
import queue
import re
import threading
import time
from typing import Any

from .config import MemoryConfig
from .extract import (
    build_messages,
    clean_fact,
    parse_facts,
    parse_summary,
    parse_topics,
    resolve_ops,
    verify,
)
from .ports import LLMNotConfigured, embedder_enabled
from .runtime import runtime
from .store import (
    append_op,
    append_summary,
    append_topic,
    archive_old_turns,
    load_state,
    materialize,
    mutate_state,
    read_turns,
    root_dir,
    update_state,
)
from .tokenize import warm as warm_tokenizer
from .vectors import reindex

# 一次送给模型多少轮。太大 => prompt 长、抽取质量下降；太小 => 调用次数上升。
CHUNK_TURNS = 40

# ---------------------------------------------------------------- 承诺行预筛
# 「她答应过他的事」的**预筛**：从 L0 的 assistant 行里挑出可能带承诺 / 约定
# 的那一小批夹进抽取请求。为什么预筛而不是全送：她的回复量是他说的话的好几倍，
# 全送会把抽取 prompt 撑大几倍（每一分钱都在 8 次/场的硬上限里）；而判据不在这里、
# 在抽取 prompt 的铁律与 `verify` 的 kind=promise 分支 —— 预筛宽一点
# 只会多花一点上下文，抠紧却会**静默漏掉**真正的承诺（漏一条的表现只是
# 「她忘了自己说过」，不报错、不进任何指标）。
_HER_COMMIT_RE = re.compile(
    r"答应|说好|一定|保证|"
    r"[我带等你来][你去]|给你|帮你|陪你|送你|留给|带回|买给|"
    r"周末|明天|后天|下次|改天|一起去"
)
_HER_COMMIT_MAX = 6  # 每个 chunk 最多夹入几条她的行（成本上界）
_HER_COMMIT_CHARS = 120  # 每条最多截多长：超长的行基本是长篇自述，不是承诺


def _cfg() -> MemoryConfig:
    """现取配置。**不要在模块级拷走任何一项** —— 那正是这套库要解耦掉的东西。"""
    return runtime().config


def _lines_for_model(all_turns: list[dict], chunk: list[dict]) -> list[dict]:
    """这一段送给模型的行走列：本段 user 轮全送 + 夹在中间「她说过的承诺行」。

    归属规则：**她的行跟随它前面最近的那个 user 轮** —— 那个 user 轮在本段，
    这一行就进本段；在上一段（或用户本轮之后、下一段之前）就进对应那一段。
    这样最后一个 chunk 会带上 L0 尾部的她的行（用户说完最后一句、她回复、
    然后就没声了 —— 那正是承诺最容易出现的位置）。

    **游标单位不受影响**：返回的只是「送给模型的行」，游标推进用的是 chunk
    （user 轮）。两件事混在一起会把游标推快 —— 表现是「她再也想不起他刚说的话」。
    """
    if not chunk:
        return []
    ids = {str(t.get("id")) for t in chunk}
    lines: list[dict] = []
    n_her = 0
    seg_open = False  # 当前行是否属于本段（最近一个 user 轮在不在本段）
    for t in all_turns:
        role = str(t.get("role"))
        if role == "user":
            seg_open = str(t.get("id")) in ids
            if seg_open:
                lines.append(t)
            continue
        if not seg_open or role != "assistant" or n_her >= _HER_COMMIT_MAX:
            continue
        text = str(t.get("text") or "")
        if not _HER_COMMIT_RE.search(text):
            continue
        # 截断而不是丢弃：承诺句通常在前半段（角色是先答应、再解释），
        # 截掉的长尾基本是长篇自述。
        lines.append(dict(t, text=text[:_HER_COMMIT_CHARS]) if len(text) > _HER_COMMIT_CHARS else t)
        n_her += 1
    return lines


def user_rounds_of(st: dict) -> int:
    """这个用户**有发言**的轮数 —— 抽取游标的进度基准。

    为什么不能直接用 `rounds`：`rounds` 数的是「轮」（含主动开口那些他一个字
    都没说的轮），而待抽的列表是从**只有用户发言**的那批里切的。两个单位一旦
    错位，游标就会被推过用户真正说过的话 —— 表现是「她什么都记不住」，
    而且**不报错**（实测：聊了三轮，一条事实都没有）。

    读不到 `user_rounds`（更老的 `state.json`）时退回 `rounds`：老库照旧能跑，
    第一次写盘之后就用新计数了。
    """
    v = st.get("user_rounds")
    return int(v if v is not None else (st.get("rounds") or 0))


class MemoryWorker:
    """单后台线程 + 队列。

    **为什么不做成「每轮同步抽一次」**：那是首字延迟，而整理慢一点完全没关系。
    一个线程、一个队列、三个触发条件，成本上界与延迟上界都是可算的。
    """

    # 队列空转时一次看多少个「活跃 uid」。**不能只扫固定的一批**：
    # `list(set)[:64]` 每次都取同一批，排在后面的人永远不会被静默整理
    # （只有满 max_turns 轮才整理一次）。也不能一次全扫 ——
    # 那等于把「启动时全量扫描」搬到每次 notify 上。
    SCAN_BATCH = 32

    # `_known` / `_last` 的条数上限。两条纪律：
    #   1. **只丢闲下来的**（超过 `idle_split_sec` 没再 notify 过）——
    #      丢一个还在说话的人，等于下一次静默触发永远轮不到他；
    #   2. **压不下去就不压**：全是活跃 uid 时宁可暂时超出上限。
    # 上限是内存护栏，不是不变量。
    KNOWN_MAX = 2000

    # 磁盘欠账扫描的目录清单缓存多久重建一次（秒）。清单本身不便宜
    # （1 万用户 = 256 个前缀目录 + 1 万个 uid 目录），而**新用户不需要靠它进来**：
    # 活跃的人由 `notify()` 直接覆盖，这个扫描只负责「早就该整理、却没人再来的」。
    DISK_LIST_TTL = 600.0

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._known: set = set()  # 活跃过的 uid，静默触发要遍历它
        self._seen_at: dict[str, float] = {}  # uid -> 最后一次 notify 的时刻（淘汰判据）
        self._scan_at = 0  # 轮转游标（见 _sweep_idle）
        self._last: dict[str, Any] = {}  # uid -> 最近一次结果（自检与面板用）
        self._last_at: dict[str, float] = {}  # uid -> 那份结果的时刻（淘汰判据）
        self._disk_at = 0  # 磁盘欠账的轮转游标（见 _sweep_disk）
        self._disk_uids: list[str] = []  # 磁盘上的 uid 清单（带 TTL 缓存）
        self._disk_listed_at = 0.0
        self._vec_at = 0  # 补向量的轮转游标（见 _ensure_vectors）
        # 整理时模型还没就绪、向量欠着的那批 uid（见 _ensure_vectors）
        self._vector_backlog: set = set()

    # ---------------------------------------------------------- 生命周期
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="changqing-worker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)

    def notify(self, uid: str) -> None:
        """一轮结束时的登记。**O(1)、不阻塞** —— 它在回复的收尾路径上。"""
        if not _cfg().enabled or not uid:
            return
        self.start()
        self._known.add(uid)
        self._seen_at[uid] = time.time()
        self._evict_stale()
        self._q.put(uid)

    # ---------------------------------------------------------- 主循环
    def _loop(self) -> None:
        # 分词器要在后台线程里也能用：它建词典要 0.7-1.6 秒，
        # 不能在第一次整理时现付。建不起来只是「检索粗一点」，
        # 不该让整条后台线程起不来。
        with contextlib.suppress(Exception):
            warm_tokenizer()
        while not self._stop.is_set():
            try:
                uid = self._q.get(timeout=30)
            except queue.Empty:
                uid = ""
            if uid is None:
                break
            if uid:
                try:
                    self.maybe_extract(uid)
                except Exception as e:  # noqa: BLE001  后台任务不许把进程带走
                    self._last[uid] = {"ok": False, "error": str(e)}
                    self._last_at[uid] = time.time()
            else:
                # 队列空转（30 秒没人要整理）＝ 没人等着，正好把两件欠账补上：
                # 磁盘上「早就该整理却没人再来」的，以及模型晚到导致的向量欠账。
                # 两件都属于「没人等着时才做」的活，所以同居这一支。
                self._ensure_vectors()
                self._sweep_disk()
            # 顺手看一眼「静默触发」：用户不说了，正是把这段对话
            # 整理成事实的时机（条件 a）。队列空转时做，不占额外线程。
            self._sweep_idle()

    def _evict_stale(self, now: float | None = None) -> None:
        """`_known` / `_last` 超过 `KNOWN_MAX` 时，丢掉闲下来的那些。

        两条纪律：
          1. **只丢闲下来的**（超过 `idle_split_sec` 没再 notify 过）。
             丢一个还在说话的人，等于把他的静默整理从轮转里摘掉 ——
             而症状只是「有的人记忆好像没整理」，不报错。
          2. **压不下去就不压**。全是活跃 uid 时宁可暂时超出上限，也不为了守住
             一个数字去动正在用的状态。上限是内存护栏，不是不变量。

        `_last`（每个 uid 最近一次整理结果）跟着一起丢：它比 `_known` 大得多
        （一份抽取统计几百字节），留着只为了排查时看一眼，闲下来的那份没有留的价值。

        `now` 只为自检留的注入点（正常路径不要传）。
        """
        cap = int(self.KNOWN_MAX or 0)
        if cap <= 0 or len(self._known) <= cap:  # 0 = 不限
            return
        now = time.time() if now is None else float(now)
        idle = sorted(
            (
                u
                for u in list(self._known)
                if now - float(self._seen_at.get(u) or 0.0) > _cfg().idle_split_sec
            ),
            key=lambda u: float(self._seen_at.get(u) or 0.0),
        )
        for uid in idle:
            if len(self._known) <= cap:
                break
            self._known.discard(uid)
            self._seen_at.pop(uid, None)
            self._last.pop(uid, None)
            self._last_at.pop(uid, None)

    def _sweep_idle(self) -> None:
        """挨个看一眼静默的那批 uid，每次看 SCAN_BATCH 个，下一次接着上一批。

        轮转的意义：用户数上万时，固定取前 N 个的话后面的人永远轮不到，
        而「用户直接关页面」恰恰是最常见的结束方式 —— 漏在那一批上，
        记忆就永远停在 L0，一次也不会被抽成事实。
        """
        uids = list(self._known)
        if not uids:
            return
        start = self._scan_at % len(uids)
        batch = uids[start : start + self.SCAN_BATCH]
        self._scan_at = start + len(batch)
        for uid in batch:
            if self._stop.is_set():
                return
            # 一个 uid 坏了不拖累别人：这一批是轮转的，下次还会轮到它
            with contextlib.suppress(Exception):
                self.maybe_extract(uid)

    # ---------------------------------------------------------- 欠账
    def _ensure_vectors(self) -> None:
        """把欠着的向量索引补上（模型晚到、或整理时模型还没就绪）。

        **为什么需要它**：模型只由宿主自己的预热去准备。启动时没网、或运维事后
        才把模型文件放好，那几轮整理就会把向量欠着 —— 而「向量少了」是**静默**的：
        召回率变差，没有任何报错。在这里补，成本可控：一次最多补 SCAN_BATCH 个
        uid，不跟正常整理抢时间。

        **要轮转**：取 `list(self._vector_backlog)[:SCAN_BATCH]` 的话，集合没变时
        每次都取同一批，于是**持续失败**的那些 uid（模型在、但这几条编不出来）
        会永远占着那前 32 个名额，排在后面的欠账一次都补不上，而表现只是
        「召回差一点」，不报错。与 `_sweep_idle` 同一条教训。
        """
        if not _cfg().enabled:
            return
        try:
            emb = runtime().embedder
            if not embedder_enabled(emb):
                return
            if not self._vector_backlog:
                return  # 没有欠账就什么都不做（含不碰任何模型）
            # **只读本地地问**，绝不去加载模型：这是唯一那条整理线程，
            # 一次加载最坏几十秒，卡住的是整条整理队列（所有用户）。
            # 加载只归宿主的预热；预热还没好就继续欠着。
            if not emb.ready(False):
                return
            uids = list(self._vector_backlog)
            start = self._vec_at % len(uids)
            batch = uids[start : start + self.SCAN_BATCH]
            self._vec_at = start + len(batch)
            for uid in batch:
                if self._stop.is_set():
                    return
                out = reindex(uid)
                if out.get("ok"):
                    self._vector_backlog.discard(uid)
        except Exception:  # noqa: BLE001  派生层坏了不该带走后台线程
            pass

    def _disk_uid_list(self) -> list[str]:
        """磁盘上所有 uid 目录的清单（带 TTL 缓存，见 `DISK_LIST_TTL`）。

        **只认带 `state.json` 的目录** —— 那是「这个 uid 真的聊过」的最小证据。
        空目录（半途创建的桶、清理的残留）扫它只是白读一次盘。

        坐标一律来自 `user_dir()` 的同一个权威定义（按 uid 前两位分桶的两级布局），
        不在这里另写一份路径拼接：写第二份的后果是扫描**静默跑到别的库上**。
        """
        now = time.time()
        if self._disk_uids and now - self._disk_listed_at < self.DISK_LIST_TTL:
            return self._disk_uids
        uids: list[str] = []
        try:
            root = root_dir()
            if root.exists():
                for bucket in root.iterdir():
                    if not bucket.is_dir():
                        continue
                    for ud in bucket.iterdir():
                        if ud.is_dir() and (ud / "state.json").exists():
                            uids.append(ud.name)
        except OSError:  # 盘上一时读不到：这一轮不扫，下次再来
            return self._disk_uids
        self._disk_uids = sorted(uids)
        self._disk_listed_at = now
        return self._disk_uids

    def _sweep_disk(self) -> None:
        """按批轮转扫磁盘上的 uid，把「早就该整理却没人再来」的欠账补上。

        **为什么必须有它**：整理的三条触发条件里，b（满 max_turns 轮）与
        c（下次活跃时扫 watermark）都要**用户再来**才成立，而「用户直接关页面」
        恰恰是最常见的结束方式 —— 那一段对话就永远停在 L0，一次也不会被想起。
        `_known` 帮不上：它只装**本进程见过**的 uid，重启就空了。

        每批 `SCAN_BATCH` 个，游标往后走（固定看前一批的话排在后面的人永远轮不到，
        与 `_sweep_idle` 同一条教训）。判据仍然是 `maybe_extract` → `should_run`：
        不无脑抽，没欠账的 uid 一次模型调用都不会花。
        """
        if not _cfg().enabled:
            return
        uids = self._disk_uid_list()
        if not uids:
            return
        start = self._disk_at % len(uids)
        batch = uids[start : start + self.SCAN_BATCH]
        self._disk_at = start + len(batch)
        for uid in batch:
            if self._stop.is_set():
                return
            with contextlib.suppress(Exception):
                self.maybe_extract(uid)

    # ---------------------------------------------------------- 触发判定
    def should_run(self, uid: str, *, now: float | None = None, force: bool = False) -> str:
        """返回 "" 表示还不该跑，否则返回触发原因。"""
        if force:
            return "forced"
        cfg = _cfg()
        st = load_state(uid)
        wm = st.get("watermark") or {}
        rounds = user_rounds_of(st)  # **有用户发言的轮**，不是「轮」
        done = int(st.get("extracted_rounds") or 0)
        if rounds <= done:
            return ""  # 没有新东西
        now = now if now is not None else time.time()
        idle_min = (now - float(wm.get("last_ts") or 0)) / 60.0
        if idle_min >= cfg.idle_min:
            return "idle"
        if rounds - done >= cfg.max_turns:
            return "turns"
        return ""

    # ---------------------------------------------------------- L0 归档
    def _archive_if_due(self, uid: str) -> dict[str, Any]:
        """超期的 L0 原话搬进 gzip 包（`archive_old_turns`）—— **每天最多一次**。

        为什么挂在整理 worker 上：它已经是一个「按 uid 定期转一圈」的后台线程，
        归档再起一条线程/定时器只是多一个要运维的东西，而这件事**慢一点完全
        没关系**（晚一天搬走不影响任何读取）。

        **为什么在 `maybe_extract` 的最开头**：归档与「有没有新原话要抽」是两件事
        —— 用户今天没说话、或者早就抽完了（`should_run` 返回空），超期的月份照样
        得搬走。放在判据后面的话，一个不再说话的用户永远等不到归档。

        **开销纪律**：`maybe_extract` 会被 `_sweep_idle` / `_sweep_disk` **按批批量**
        调用，所以日常开销必须只是「一次 state 读 + 一次日期比较」；`state.json`
        的 `last_archive_scan` 挡第二次，只在**今天第一次**时才真去扫目录、写 state。

        `retain_months <= 0` 直接返回（一行关掉 = 永不清理）。归档失败绝不影响
        抽取：异常在这里吞掉，但**如实放进返回值**（`maybe_extract` 再把它挂到
        自己的返回体上），不静默。
        """
        cfg = _cfg()
        if int(cfg.retain_months or 0) <= 0:
            return {}  # 一行关掉 = 永不清理
        today = time.strftime("%Y-%m-%d")
        if str(load_state(uid).get("last_archive_scan") or "") == today:
            return {}  # 今天已经扫过：就这一次 state 读
        try:
            out = archive_old_turns(uid, months=cfg.retain_months)
        except Exception as e:  # noqa: BLE001  归档坏了也得能整理
            out = {"files": 0, "bytes": 0, "months": [], "error": str(e)}
        # 扫过就记下来，**哪怕这次没搬、哪怕报了错**：这是一条「节流」记录，不是
        # 「成功」记录。不记的话，一个坏文件会让这个 uid 每次被扫到都重撞一遍
        # （每 30 秒一次），而重试的收益是零。
        # state 写不动不影响归档结果（下一次顶多再扫一遍）
        with contextlib.suppress(Exception):
            update_state(uid, last_archive_scan=today)
        return out

    # ---------------------------------------------------------- 跑一次
    def maybe_extract(self, uid: str, *, force: bool = False, call: Any = None) -> dict[str, Any]:
        arch = self._archive_if_due(uid)
        reason = self.should_run(uid, force=force)
        if not reason:
            out: dict[str, Any] = {"ok": True, "skipped": True}
            if arch:
                out["archive"] = arch
            return out
        out = self.extract_uid(uid, call=call)
        out["reason"] = reason
        if arch:
            out["archive"] = arch
        self._last[uid] = out
        self._last_at[uid] = time.time()
        return out

    def extract_uid(self, uid: str, *, call: Any = None) -> dict[str, Any]:
        """把还没整理过的 L0 抽成事实。返回统计。**绝不抛异常**。

        **游标自愈**：`user_rounds` / `extracted_rounds` 是「L0 里第 i 条 user 行
        = 第 i 轮」这套下标的一部分。L0 一旦比 state 少（整库清理时 `state.json`
        被句柄占着删不掉、`sessions/` 却删掉了 —— 在 Windows 上这是常态），
        这套下标就断了。

        把「pending 为空」当成唯一解释（⇒「游标已经追上用户发言了」）会把漂移
        **固化**成「已经抽完」：此后每来一句新话两个计数一起 +1，pending 永远为空
        —— 抽取永久变成空操作，而返回值照旧 `ok: True` / `calls: 0`，不报错、
        不进任何指标。

        现在把两个计数夹回 L0 的真实长度，并**保留 state 声称的「还没抽」条数**
        （`user_rounds - extracted_rounds`）锚到 L0 的尾巴上。为什么不是直接夹成
        `len(turns)`（那样也能解冻）：夹完的下标正好落在切片边界上，这一次刚到的
        发言会被当成「已经抽完」而**永久漏掉**。代价是 L0 在中段丢过东西时可能
        重抽一小段已经抽过的 —— 抽取是幂等的（`resolve_ops` 的 NOOP、纪要 id 是
        内容哈希），多花一次调用换「一句话都不漏」，这个取舍是刻意的。
        漂移本身记进 `out["l0_drift"]`，不再静默。
        """
        cfg = _cfg()
        out: dict[str, Any] = {
            "ok": False,
            "calls": 0,
            "active": 0,
            "pending": 0,
            "dropped": 0,
            "truncated": 0,
            "ADD": 0,
            "SUPERSEDE": 0,
            "NOOP": 0,
        }
        try:
            st = load_state(uid)
            done = int(st.get("extracted_rounds") or 0)
            # 与下面那个列表**同一个单位**：只数用户发言的轮（见 user_rounds_of）
            rounds_total = user_rounds_of(st)

            # 只拿用户说的话当**游标与切片**的依据：抽取的第一条铁律就是
            # 「她说的事实不算」，在这里先把它们滤掉，模型连看都看不到，省得它犯规。
            # 但她的行不丢：可能带承诺的那些由 `_lines_for_model` 夹回请求里
            # （铁律 2 的唯一来源，缺了它「她答应过他的事」永远进不了库）。
            # **读在早返回之前**：漂移只有拿 L0 的真实长度才看得出来，
            # 放在后面的话 `rounds_total <= done` 那条早返回会把漂移藏起来。
            all_turns = read_turns(uid)
            turns = [t for t in all_turns if t.get("role") == "user"]

            if rounds_total > len(turns) or done > len(turns):
                # 下标断了（见 docstring）：自愈而不是固化。
                claimed = max(0, rounds_total - done)  # state 声称还没抽的条数
                clamped_done = max(0, len(turns) - claimed)
                out["l0_drift"] = {
                    "state_user_rounds": rounds_total,
                    "state_extracted_rounds": done,
                    "l0_user_rounds": len(turns),
                    "user_rounds": len(turns),
                    "extracted_rounds": clamped_done,
                    # 时间戳不是装饰：`state.json` 的 `updated_at` 每轮都在变，
                    # 分不出「这次自愈是什么时候发生的」。面板要能回答
                    # 「他刚反馈记不住，是不是因为这个」。
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                }
                rounds_total, done = len(turns), clamped_done
                # 走 `update_state`：读改写必须在 `_lock(uid)` 里 —— 中间那段
                # `read_turns()` 是几百毫秒的活，拿旧快照整个写回会把前台这期间的
                # 写入（next_seq / user_rounds）盖掉。
                update_state(
                    uid, user_rounds=rounds_total, extracted_rounds=done, l0_drift=out["l0_drift"]
                )

            if rounds_total <= done:
                out["ok"] = True
                return out

            pending = turns[done:rounds_total]

            calls = 0
            written = 0
            # `calls` 用 enumerate 数，不另开一个自增变量：它就是「这是第几次
            # 模型调用」，而它与上限的比较正是成本闸门，两处数字必须同源。
            for calls, offset in enumerate(range(0, len(pending), CHUNK_TURNS), 1):
                if calls > cfg.extract_max_calls:
                    break  # 硬上限，写死
                chunk = pending[offset : offset + CHUNK_TURNS]  # user 轮（游标单位）
                # 送给模型的是「本段 user 轮 + 夹在中间她的承诺行」；
                # `calls` / `written` 和切片都只数 user 轮 —— 她的行只进 prompt、
                # 不进下标（混进去会把游标推快 = 静默漏抽他的话）。
                lines = _lines_for_model(all_turns, chunk)
                raw = self._call_extract(lines, call=call, out=out)
                if raw is None:
                    # 调用失败：**不推进游标**，这段下次重试。
                    # 推进了的话这几轮原话就永远不会再被整理 —— 静默丢数据。
                    # 「被上限砍断」也走这一条，但说清是哪一种 —— 两者的修法不同：
                    # 前者等下次重试，后者要把抽取上限调大。
                    out["calls"] = calls
                    out["error"] = (
                        "抽取输出撞上 token 上限，游标不推进，下次重试"
                        if out.get("truncated")
                        else "抽取调用失败，游标不推进，下次重试"
                    )
                    break
                result, stats = self._ingest(uid, lines, chunk[-1], raw)
                out["calls"] = calls
                for k in ("active", "pending", "dropped"):
                    out[k] = out.get(k, 0) + result.get(k, 0)
                for k in ("ADD", "SUPERSEDE", "NOOP"):
                    out[k] = out.get(k, 0) + stats.get(k, 0)
                written += len(chunk)

            # watermark 只在整段成功之后推进：失败就下次重试，不会漏也不会重
            if written:

                def _advance(s: dict, _base: int = done + written) -> None:
                    # 读改写整体在 `_lock(uid)` 里（上面那几百毫秒的抽取调用在锁外，
                    # 所以这里必须重新读一遍再写 —— 否则前台这期间的 append_turn
                    # 会被旧快照盖掉）
                    s["extracted_rounds"] = _base
                    wm = s.get("watermark") or {}
                    wm["extracted_upto"] = wm.get("last_turn_id")
                    s["watermark"] = wm

                # `watermark` 是**嵌套**的，所以走 `mutate_state` 而不是
                # `update_state`：整个合并必须在锁内基于锁里读到的那一份做，
                # 在锁外拼好新字典再写回会把前台每轮更新的 `last_ts` 盖掉。
                mutate_state(uid, _advance)
                # 整理完顺手巩固一轮：新增的事实里可能有和旧的重复，
                # 也可能是「这次被用上了」需要更新权重。放在同一个后台线程里，
                # 不额外占线程、也不额外花 token（巩固是纯本地的）
                try:
                    from .consolidate import consolidate

                    out["consolidate"] = consolidate(uid)
                except Exception as e:  # noqa: BLE001
                    out["consolidate"] = {"ok": False, "error": str(e)}
                # 向量索引跟着事实走。**放在整理这一侧**是设计决定：事实刚被冲突
                # 消解 / 巩固过，这里编码一次就是最新的；换成在对话里编码，
                # 每轮要多花几十毫秒，而那是首字延迟。
                try:
                    out["vectors"] = reindex(uid)
                except Exception as e:  # noqa: BLE001
                    out["vectors"] = {"ok": False, "error": str(e)}
                if out["vectors"].get("ok"):
                    self._vector_backlog.discard(uid)
                elif out["vectors"].get("reason") == "embed_unavailable":
                    # 模型还没就绪：先欠着，等模型好了由 `_ensure_vectors` 补
                    self._vector_backlog.add(uid)
            out["ok"] = True
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)
        return out

    # ---------------------------------------------------------- 内部
    def _call_extract(
        self, lines: list[dict], call: Any = None, out: dict | None = None
    ) -> str | None:
        """一次抽取调用。`lines` 是给模型的那批行（含夹回来的她的承诺行）。

        返回值分三种，调用方必须区分 —— 混在一起会造成**静默丢数据**：
          None  这次调用**失败**了 → 不许推进游标，下次重试
          ""    没有可抽的 / 没配模型（不是失败，重试也不会变好）→ 正常推进
          其他  模型输出的原文，交给 `parse_facts`

        **被输出上限砍断也算失败**：上限原先跟聊天共用一个常数，而它一次要吐
        一整批。撞上限时 JSON 从中间断掉，`parse_facts` 解不出来就返回 []，
        而下游把 [] 读成「没什么可抽的」**照常推进游标** —— 事实没了，
        一个字都不报。所以这里按批大小自己算一个上限，并把
        `finish_reason == "length"` 当失败：游标不推进、下次重试，
        次数记在 `out["truncated"]` 里。
        """
        cfg = _cfg()
        today = time.strftime("%Y-%m-%d")
        msgs = build_messages(lines, today)
        # 按**这一批的行数**算：她的行也在里面，它们一样占输出。
        cap = min(cfg.extract_max_tokens, cfg.extract_tokens_per_turn * max(1, len(lines)))
        try:
            if call is not None:
                return call(msgs)
            finish: dict = {}
            raw = runtime().llm(
                msgs,
                model=cfg.extract_model,
                max_tokens=cap,
                on_finish=lambda why: finish.setdefault("why", why),
                on_usage=lambda u: runtime().usage.note_llm(extract=True, **u),
            )
            if finish.get("why") == "length":
                if out is not None:
                    out["truncated"] = int(out.get("truncated") or 0) + 1
                return None
            return raw
        except LLMNotConfigured:
            # 没注入模型：只写 L0，抽取跳过。这是降级路径而不是失败，
            # 所以游标照常推进 —— 否则每次触发都会重跑一遍注定失败的整理。
            return ""
        except Exception:  # noqa: BLE001  抽取失败不影响说话
            return None

    def _ingest(self, uid: str, lines: list[dict], anchor: dict, raw: str) -> tuple:
        """解析 → 回引校验 → 冲突消解 → 追加日志 → 物化。

        `lines` 是**送给模型的那批行**（本段 user 轮 + 夹进来的她的行）：回引校验
        要对着它查 —— 她的承诺引用的正是她自己那一行。
        `anchor` 是本段最后一个 user 轮：纪要 / 话题的日期与 turn_ref 以它为准
        （与「只数 user 轮」的游标同一套下标，不受中间夹进来的她的行影响）。
        """
        by_id = {t["id"]: t for t in lines}
        result = {"active": 0, "pending": 0, "dropped": 0}
        cleaned: list[dict] = []
        for raw_fact in parse_facts(raw):
            tid = str(raw_fact.get("turn_ref") or "").strip()
            turn = by_id.get(tid)
            if turn is None:
                result["dropped"] += 1
                continue
            status, _score = verify(raw_fact, by_id)
            if status == "drop":
                result["dropped"] += 1
                continue
            f = clean_fact(raw_fact, turn)
            if f is None:
                result["dropped"] += 1
                continue
            f["status"] = "active" if status == "active" else "pending"
            cleaned.append(f)
            result[status] += 1

        ops, stats = resolve_ops(uid, cleaned)
        for op in ops:
            append_op(uid, op)
        if ops:
            materialize(uid)

        # L2 会话纪要：它**不是事实** —— 没有回引、没有槽位，所以进 summaries
        # 而不是 facts。混进 facts 的话「每条事实都能追回原话」这条就保不住了。
        summary = parse_summary(raw)
        if summary:
            day = str(anchor.get("day") or "")[:10] or time.strftime("%Y-%m-%d")
            try:
                append_summary(uid, summary, day, turn_ref=str(anchor.get("id") or ""))
                result["summary"] = 1
            except Exception:  # noqa: BLE001
                pass

        # 主动话题：**同一次调用顺手要出来**，这是它存在的理由 ——
        # 挑话题需要一次模型调用，而它绝不能出现在首字路径上，所以只能住在
        # 后台整理里。写失败只是「下次开口没得挑」，绝不影响说话。
        topics = parse_topics(raw, str(anchor.get("day") or "")[:10])
        wrote = 0
        for t in topics:
            try:
                if append_topic(
                    uid,
                    t["text"],
                    str(anchor.get("day") or "")[:10] or time.strftime("%Y-%m-%d"),
                    kind=t["kind"],
                    due_day=t["due_day"],
                    ref=t["ref"],
                ):
                    wrote += 1
            except Exception:  # noqa: BLE001  派生层坏了不卡整理
                break
        if wrote:
            result["topics"] = wrote
        return result, stats


WORKER = MemoryWorker()


def notify(uid: str) -> None:
    """一轮结束时的登记口（宿主在收尾时调它）。**O(1)、不阻塞**。"""
    WORKER.notify(uid)


def extract_now(uid: str, *, call: Any = None) -> dict[str, Any]:
    """同步跑一次整理。自检与手动触发用。"""
    return WORKER.extract_uid(uid, call=call)


__all__ = ["CHUNK_TURNS", "MemoryWorker", "WORKER", "extract_now", "notify", "user_rounds_of"]
