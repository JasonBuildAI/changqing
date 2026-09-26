"""后台整理：三个触发条件、调用上限、幂等，以及游标那套下标。

这组用例的形状都来自同一个教训：整理这条路错了**从不报错**。丢一批话只是
「她记性不好」，游标推快一点只是「她再也想不起他刚说的话」，多调一次模型只是
账单变厚。所以断言写得比通常啰嗦，每一条都对应一种静默失败。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from changqing.adapters.mock import MockEmbedder
from changqing.runtime import Runtime, using
from changqing.store import (
    append_turn,
    list_facts,
    list_summaries,
    list_topics,
    load_state,
    open_index,
    read_turns,
    update_state,
)
from changqing.worker import (
    _HER_LINE_CHARS,
    _HER_LINES_MAX,
    CHUNK_TURNS,
    MemoryWorker,
    user_rounds_of,
)
from changqing.worker import _lines_for_model as lines_for_model

UID = "u" + "b" * 16


def say(uid: str, text: str, reply: str = "嗯嗯", ts: float | None = None) -> None:
    """追加一轮：他说一句、她回一句。`ts` 默认是「现在」。"""
    append_turn(uid, {"user": text, "assistant": reply, "ts": ts or time.time()})


def facts_json(*facts: dict, summary: str = "", topics: list | None = None) -> str:
    """拼一份抽取输出。字段名与抽取 prompt 要的完全一致。"""
    body = {
        "facts": [
            {
                "subject": "他",
                "kind": "fact",
                "confidence": 0.9,
                "importance": 0.5,
                "persona_attention": 0.5,
                **f,
            }
            for f in facts
        ],
        "summary": summary,
        "topics": topics or [],
    }
    return json.dumps(body, ensure_ascii=False)


@contextmanager
def with_llm(rt: Runtime, llm) -> Iterator[Runtime]:
    """在这段作用域里换上一个模型替身（配置与其余能力沿用 fixture 那份）。"""
    with using(Runtime(config=rt.config, embedder=rt.embedder, llm=llm, usage=rt.usage)) as active:
        yield active


# ---------------------------------------------------------------- 送给模型的行
def test_her_lines_follow_the_user_turn_before_them(rt: Runtime):
    """她的行跟随它前面最近的那个 user 轮 —— 归属错了，承诺就挂到别人头上。"""
    turns = [
        {"id": "T-000001", "role": "user", "text": "周末有空吗"},
        {"id": "T-000002", "role": "assistant", "text": "周末带你去看展"},
        {"id": "T-000003", "role": "user", "text": "好啊"},
    ]
    chunk = [turns[0]]
    got = [t["id"] for t in lines_for_model(turns, chunk)]
    assert got == ["T-000001", "T-000002"], "本段的 user 轮 + 它后面她的行"


def test_her_lines_ride_along_whatever_she_said(rt: Runtime):
    """她的**整行**都进请求，预筛只剩长度与条数两个上界。

    关键词预筛看不见「我平时都喝美式」这类句子（一个承诺词都没有），而她的偏好
    只存在于她的行里 —— 漏掉的表现正是她今天说爱喝美式、明天说从来不喝咖啡。
    """
    turns = [
        {"id": "T-000001", "role": "user", "text": "在吗"},
        {"id": "T-000002", "role": "assistant", "text": "在的，今天画了一下午"},
    ]
    assert [t["id"] for t in lines_for_model(turns, [turns[0]])] == ["T-000001", "T-000002"], (
        "没有承诺词的她的行也要进 prompt"
    )


def test_her_lines_are_capped_truncated_and_drop_the_oldest(rt: Runtime):
    """上限是成本上界：一场对话里她的回复量是他说的话的好几倍。

    超过条数上界时丢**最旧**的那几条：她刚说的那段才最可能被下一场提起。
    """
    turns = [{"id": "T-000000", "role": "user", "text": "在吗"}]
    for i in range(1, _HER_LINES_MAX + 4):
        turns.append({"id": f"T-{i:06d}", "role": "assistant", "text": f"我第{i}条。" + "话" * 300})
    got = lines_for_model(turns, [turns[0]])
    her = [t for t in got if t["role"] == "assistant"]
    assert len(her) == _HER_LINES_MAX, f"每个 chunk 最多夹 {_HER_LINES_MAX} 条她的行"
    assert all(len(t["text"]) == _HER_LINE_CHARS for t in her), "超长的截断，不是整条丢掉"
    assert her[0]["id"] == f"T-{4:06d}", "丢的是最旧的那几条"
    assert her[-1]["id"] == f"T-{_HER_LINES_MAX + 3:06d}", "最新的一条必须留住"


def test_rounds_are_never_counted_from_her_side(rt: Runtime):
    """游标的单位是 **user 轮**：混进她的行会把游标推快，表现是永久漏抽。"""
    turns = [
        {"id": "T-000001", "role": "user", "text": "在吗"},
        {"id": "T-000002", "role": "assistant", "text": "我明天给你带书"},
        {"id": "T-000003", "role": "user", "text": "谢谢"},
    ]
    got = lines_for_model(turns, [turns[0], turns[2]])
    assert len(got) == 3, "送出去 3 行"
    assert len([t for t in got if t["role"] == "user"]) == 2, "但游标只该走 2 格"


# ---------------------------------------------------------------- 游标基准
def test_user_rounds_prefers_the_new_counter(rt: Runtime):
    assert user_rounds_of({"rounds": 9, "user_rounds": 4}) == 4


def test_user_rounds_falls_back_for_older_state(rt: Runtime):
    """更老的 state.json 里没有 `user_rounds` —— 退回 `rounds`，不报错。"""
    assert user_rounds_of({"rounds": 9}) == 9
    assert user_rounds_of({}) == 0


def test_proactive_rounds_do_not_move_the_cursor(rt: Runtime):
    """主动开口那一轮他一个字都没说：`rounds` 走一格，用户轮数不走。"""
    append_turn(UID, {"user": "", "assistant": "今天光线很好"})
    append_turn(UID, {"user": "在吗", "assistant": "在"})
    st = load_state(UID)
    assert st["rounds"] == 2
    assert user_rounds_of(st) == 1, "只有一轮有用户发言"


# ---------------------------------------------------------------- 触发判定
def test_no_new_turns_means_no_trigger(rt: Runtime):
    w = MemoryWorker()
    assert w.should_run(UID) == "", "什么都没说过，一次都不该跑"


def test_idle_trigger(rt: Runtime):
    """用户不说了，正是整理最自然的时机。"""
    say(UID, "我家猫叫团子")
    old = time.time() - 3600
    update_state(UID, watermark={"last_turn_id": "T-000002", "last_ts": old})
    assert MemoryWorker().should_run(UID) == "idle"


def test_turn_threshold_trigger(rt: Runtime):
    """长对话中途也要落一次 —— 不能只等静默。"""
    for i in range(5):
        say(UID, f"第{i}句")
    update_state(UID, watermark={"last_ts": time.time()})  # 刚刚才说过话
    with using(Runtime(config=rt.config.evolved(max_turns=3))):
        assert MemoryWorker().should_run(UID) == "turns"


def test_force_runs_even_with_nothing_new(rt: Runtime):
    assert MemoryWorker().should_run(UID, force=True) == "forced"


# ---------------------------------------------------------------- 跑一轮
def test_extract_writes_facts_summary_and_topics(rt: Runtime):
    """一次调用三样都要得回来：事实、纪要、主动话题共用同一次模型调用。"""
    say(UID, "我家猫叫团子")
    raw = facts_json(
        {
            "predicate": "养的猫叫",
            "object": "团子",
            "quote": "我家猫叫团子",
            "turn_ref": "T-000001",
        },
        summary="他聊了他的猫",
        topics=[{"text": "问他团子最近怎么样", "kind": "followup"}],
    )
    out = MemoryWorker().extract_uid(UID, call=lambda _msgs: raw)
    assert out["ok"] is True
    assert out["calls"] == 1, "一轮对话的记忆类调用就这一次"
    assert out["active"] == 1, "抽到一条事实"
    assert [f["object"] for f in list_facts(UID)] == ["团子"]
    assert [s["text"] for s in list_summaries(UID)] == ["他聊了他的猫"], "纪要在库里"
    assert [t["text"] for t in list_topics(UID)] == ["问他团子最近怎么样"], (
        "主动话题也在库里（下次开口才有得挑）"
    )
    assert load_state(UID)["extracted_rounds"] == 1, "游标推进了"


def test_a_second_run_has_nothing_left_to_do(rt: Runtime):
    """幂等：整理两次的结果一样 —— 第二次一条事实都不该新增。"""
    say(UID, "我家猫叫团子")
    raw = facts_json(
        {"predicate": "养的猫叫", "object": "团子", "quote": "我家猫叫团子", "turn_ref": "T-000001"}
    )
    w = MemoryWorker()
    w.extract_uid(UID, call=lambda _m: raw)
    again = w.extract_uid(UID, call=lambda _m: raw)
    assert again["calls"] == 0, "没有新话要抽"
    assert len(list_facts(UID)) == 1, "事实没有翻倍"


def test_her_own_fact_lands_with_her_as_subject(rt: Runtime):
    """她自己的偏好也进事实层，`subject` 是「她」（铁律 1）。

    这是她「前后不一致」的正解：库里没有立足点时，她每一场对同一件事的说法
    都只靠这一轮上下文发挥 —— 今天说爱喝美式、明天说从来不喝咖啡。
    """
    say(UID, "你平时喝什么", "我平时都喝美式。")
    her = next(t for t in read_turns(UID) if t["role"] == "assistant")
    raw = facts_json(
        {
            "subject": "她",
            "predicate": "平时都喝",
            "object": "美式",
            "quote": "我平时都喝美式",
            "turn_ref": her["id"],
        }
    )
    out = MemoryWorker().extract_uid(UID, call=lambda _m: raw)
    assert out["active"] == 1, f"她自己的事实要落进事实层（实际 {out}）"
    fact = list_facts(UID)[0]
    assert fact["subject"] == "她" and fact["kind"] == "fact", f"subject=她、kind 不变：{fact}"
    assert fact["turn_ref"] == her["id"], "回引的是她自己那一行"


def test_her_restating_him_is_not_a_fact(rt: Runtime):
    """她复述他的话**不许**变成事实 —— 那是把他的信息反向当成她提供的。

    「你上次说你不吃香菜」与她真说了「我不吃香菜」在文本上分不开，所以判据只在
    prompt 的铁律 2 里；模型犯规时，回引到她那一轮仍会被 `verify` 丢掉
    （他的信息只能从他的行里抽）。
    """
    say(UID, "我不吃香菜", "你上次说你不吃香菜。")
    her = next(t for t in read_turns(UID) if t["role"] == "assistant")
    raw = facts_json(
        {
            "subject": "他",
            "predicate": "不吃",
            "object": "香菜",
            "quote": "你上次说你不吃香菜",
            "turn_ref": her["id"],
        }
    )
    out = MemoryWorker().extract_uid(UID, call=lambda _m: raw)
    assert out["active"] == 0, f"她复述他的话不许落库（实际 {out}）"
    assert list_facts(UID) == [], "事实层里一条都不该有"


def test_a_failed_call_does_not_advance_the_cursor(rt: Runtime):
    """失败就下次重试。推了游标 = 这几轮原话永远不会再被整理 —— 静默丢数据。"""
    say(UID, "我家猫叫团子")

    def boom(_msgs):
        raise RuntimeError("网络断了")

    out = MemoryWorker().extract_uid(UID, call=boom)
    assert out["ok"] is True, "后台整理不该把异常抛给调用方"
    assert "error" in out
    assert load_state(UID).get("extracted_rounds", 0) == 0, "游标一动不动"
    assert list_facts(UID) == []


def test_truncated_output_counts_as_failure_not_as_empty(rt: Runtime):
    """撞上输出上限时 JSON 从中间断掉 —— 把它读成「没什么可抽的」会静默丢事实。"""
    say(UID, "我家猫叫团子")

    class TruncatingLLM:
        """吐到一半撞上上限的模型：`finish_reason` 是这件事唯一的可判依据。"""

        def __call__(self, messages, *, model="", max_tokens=0, on_usage=None, on_finish=None):
            if on_finish:
                on_finish("length")
            return '{"facts": [{"subject": "他"'

    with with_llm(rt, TruncatingLLM()):
        out = MemoryWorker().extract_uid(UID)
    assert out["truncated"] == 1
    assert load_state(UID).get("extracted_rounds", 0) == 0, "游标不推进，下次重试"
    assert "上限" in out["error"], "说清是哪一种失败：两者的修法不同"


def test_empty_output_counts_as_failure_not_as_empty(rt: Runtime):
    """空正文不是「没有可抽的」：契约要的是一份 JSON，空白永远不合格，按失败走。

    2026-09-23 真机抓到一次「抽取 0 条、out 里没有 error、游标照常推进」——
    根因是空响应与「没配模型」共用 `""` 这一个返回值，而上游把它读成合格的空答案。
    重跑就绿，等于把丢事实推给运气。
    """
    say(UID, "我家猫叫团子")

    class SilentLLM:
        """一句话都不说的模型：不报错，也没有任何 JSON。"""

        def __init__(self, body: str) -> None:
            self.body = body

        def __call__(self, messages, *, model="", max_tokens=0, on_usage=None, on_finish=None):
            if on_finish:
                on_finish("stop")
            return self.body

    for body in ("", "   \n"):
        with with_llm(rt, SilentLLM(body)):
            out = MemoryWorker().extract_uid(UID)
        assert "游标不推进" in out.get("error", ""), f"空正文（{body!r}）要按失败走"
        assert load_state(UID).get("extracted_rounds", 0) == 0, "游标不推进，下次重试"
        assert list_facts(UID) == []


def test_missing_llm_degrades_and_still_advances(rt: Runtime):
    """没注入模型是**降级**而不是失败：重试也不会变好，所以游标照常推进。

    不推进的话每次触发都会重跑一遍注定失败的整理，而用户什么都没得到。
    """
    say(UID, "我家猫叫团子")
    out = MemoryWorker().extract_uid(UID)  # fixture 里是 NullLLM
    assert out["ok"] is True
    assert out["calls"] == 1
    assert list_facts(UID) == []
    assert load_state(UID)["extracted_rounds"] == 1, "降级路径要推进游标"


def test_the_call_cap_bounds_one_run(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    """一场对话的调用次数有硬上限 —— 滑向「每轮一次」是成本差两个数量级的事。"""
    monkeypatch.setattr("changqing.worker.CHUNK_TURNS", 1)
    for i in range(5):
        say(UID, f"第{i}句")

    seen: list = []

    def spy(msgs):
        seen.append(msgs)
        return facts_json()

    with using(Runtime(config=rt.config.evolved(extract_max_calls=2))):
        out = MemoryWorker().extract_uid(UID, call=spy)
    assert out["calls"] == 2, "5 段只肯花 2 次调用"
    assert len(seen) == 2, "真的只调了 2 次"
    assert load_state(UID)["extracted_rounds"] == 2, "只推进了两段"


def test_the_call_cap_is_configurable(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("changqing.worker.CHUNK_TURNS", 1)
    say(UID, "第一句")
    say(UID, "第二句")
    with using(Runtime(config=rt.config.evolved(extract_max_calls=1))):
        out = MemoryWorker().extract_uid(UID, call=lambda _m: facts_json())
    assert out["calls"] == 1


# ---------------------------------------------------------------- 游标自愈
def test_cursor_self_heals_when_l0_is_shorter_than_state(rt: Runtime):
    """L0 比 state 少（清理时 state.json 删不掉、sessions/ 删掉了）时，
    下标就断了。把它当成「已经抽完」会把漂移固化：此后**抽取永久变成空操作**，
    而返回值照旧 `ok: True` / `calls: 0`，不报错、不进任何指标。
    """
    say(UID, "我家猫叫团子")
    update_state(UID, user_rounds=9, extracted_rounds=9)  # state 声称抽完了
    assert len([t for t in read_turns(UID) if t["role"] == "user"]) == 1, "L0 里只有一轮"

    seen: list = []
    w = MemoryWorker()
    out = w.extract_uid(UID, call=lambda m: (seen.append(m), facts_json())[1])
    assert "l0_drift" in out, "漂移要如实报出来，不再静默"
    assert out["l0_drift"]["l0_user_rounds"] == 1
    assert len(seen) == 0, "state 声称没有欠账，这一趟确实无事可做"

    st = load_state(UID)
    assert st["user_rounds"] == 1, "计数夹回 L0 的真实长度"
    assert st["l0_drift"]["at"], "时间戳不是装饰：面板要能回答「他刚反馈记不住是不是因为这个」"

    # **这才是自愈的意义**：下标已经对齐，下一句新话不再被当成「早就抽过」。
    # 修之前 state 停在 9/9：每来一句两个计数一起 +1，pending 永远为空 ——
    # 抽取永久变成空操作，而返回值照旧 ok / calls: 0。
    say(UID, "它三岁了")
    out2 = w.extract_uid(UID, call=lambda m: (seen.append(m), facts_json())[1])
    assert len(seen) == 1, "新到的那一句终于被抽了"
    assert out2["calls"] == 1


def test_the_latest_turn_is_never_lost_to_a_clamp(rt: Runtime):
    """夹下标时保留 state 声称的欠账条数。

    直接夹成 `len(turns)` 也能解冻，但夹完的下标正好落在切片边界上 ——
    这一次刚到的发言会被当成「已经抽完」而**永久漏掉**。
    """
    for i in range(3):
        say(UID, f"第{i}句")
    update_state(UID, user_rounds=5, extracted_rounds=2)  # 声称「还欠 3 条」

    out = MemoryWorker().extract_uid(UID, call=lambda _m: facts_json())
    assert out["l0_drift"]["extracted_rounds"] == 0, "L0 只有 3 条，欠账 3 条 → 一条都还没抽"
    assert out["l0_drift"]["user_rounds"] == 3
    assert load_state(UID)["extracted_rounds"] == 3, "3 条全抽了"


# ---------------------------------------------------------------- 登记与淘汰
def test_notify_only_registers_when_enabled(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    w = MemoryWorker()
    started: list = []
    monkeypatch.setattr(w, "start", lambda: started.append(1))
    with using(Runtime(config=rt.config.evolved(enabled=False))):
        w.notify(UID)
    assert started == [], "关掉整个记忆系统时连线程都不该起"
    assert w._known == set()

    w.notify(UID)
    assert started == [1], "开着的时候才起线程"
    assert UID in w._known, "登记进活跃集合（静默触发要遍历它）"
    assert w._q.qsize() == 1, "并且把一个待整理的 uid 放进队列"


def test_evict_stale_only_drops_idle_uids(rt: Runtime):
    """丢一个还在说话的人，等于把他的静默整理从轮转里摘掉 —— 而症状只是
    「有的人记忆好像没整理」，不报错。"""
    w = MemoryWorker()
    w.KNOWN_MAX = 2
    now = time.time()
    w._known = {"busy-a", "busy-b", "idle-old"}
    w._seen_at = {"busy-a": now, "busy-b": now - 5, "idle-old": now - rt.config.idle_split_sec - 10}
    w._last = {"idle-old": {"ok": True}}
    w._last_at = {"idle-old": now}
    w._evict_stale(now=now)
    assert w._known == {"busy-a", "busy-b"}, "只丢了闲下来的那个"
    assert w._last == {}, "`_last` 跟着一起丢"


def test_evict_stale_gives_up_instead_of_dropping_active_uids(rt: Runtime):
    """全是活跃 uid 时宁可暂时超出上限 —— 上限是内存护栏，不是不变量。"""
    w = MemoryWorker()
    w.KNOWN_MAX = 1
    now = time.time()
    w._known = {"a", "b", "c"}
    w._seen_at = dict.fromkeys(("a", "b", "c"), now)
    w._evict_stale(now=now)
    assert w._known == {"a", "b", "c"}, "压不下去就不压"


def test_chunk_turns_is_the_documented_default():
    assert CHUNK_TURNS == 40


# ---------------------------------------------------------------- 磁盘欠账
def test_disk_sweep_finds_conversations_nobody_came_back_to(rt: Runtime):
    """三条触发条件里有一条半都要「用户再来」，而关页面才是最常见的结束方式。

    没有这一条路，那段对话就永远停在 L0 —— 一次也不会被想起。
    """
    w = MemoryWorker()
    assert w._disk_uid_list() == [], "还没人聊过"

    for uid in ("u" + "1" * 16, "u" + "2" * 16):
        say(uid, "我家猫叫团子")
    got = w._disk_uid_list()
    assert got == sorted(["u" + "1" * 16, "u" + "2" * 16]), "只认真的聊过的目录"


def test_disk_sweep_skips_directories_without_state(rt: Runtime, mem_root):
    """`state.json` 是「这个 uid 真的聊过」的最小证据。

    空目录（半途创建的桶、清理的残留）扫它只是白读一次盘 —— 而上万用户的
    目录清单正是这条路上最贵的一次 IO。
    """
    w = MemoryWorker()
    say(UID, "在吗")
    stray = mem_root / "z9" / ("u" + "z" * 16)
    stray.mkdir(parents=True)
    assert w._disk_uid_list() == [UID], "空目录不算聊过"


def test_disk_sweep_rotates_so_everyone_gets_a_turn(rt: Runtime):
    """固定看前一批的话排在后面的人永远轮不到 —— 与静默扫描同一条教训。"""
    w = MemoryWorker()
    w.SCAN_BATCH = 2
    for i in range(5):
        say(f"u{i:016d}", f"第{i}句")
    assert len(w._disk_uid_list()) == 5
    seen: list = []
    w.maybe_extract = lambda uid, **kw: seen.append(uid)  # 只关心「轮到谁」
    for _ in range(3):
        w._sweep_disk()
    assert len(set(seen)) == 5, "三轮下来五个 uid 都轮到了"


def test_disk_sweep_is_a_noop_when_the_whole_system_is_off(rt: Runtime):
    """关掉记忆系统时连目录都不该列 —— 空转的 IO 也是成本。"""
    say(UID, "我家猫叫团子")
    w = MemoryWorker()
    w._disk_uids = []  # 清掉缓存，逼它真去列目录
    w._disk_listed_at = 0.0
    with using(Runtime(config=rt.config.evolved(enabled=False))):
        w._sweep_disk()
    assert w._disk_uids == [], "一次目录都没列"


def test_disk_uid_list_is_cached(rt: Runtime, monkeypatch: pytest.MonkeyPatch):
    """清单要 1 万条目录 + 1 万个 uid 目录，不能每轮空转都重列一遍。"""
    say(UID, "在吗")
    w = MemoryWorker()
    assert w._disk_uid_list() == [UID]
    say("u" + "c" * 16, "新的人")
    assert w._disk_uid_list() == [UID], "TTL 内用缓存"
    w._disk_listed_at = 0.0  # 假装 TTL 过期
    assert len(w._disk_uid_list()) == 2, "过期后重新列，新人进来了"


# ---------------------------------------------------------------- L0 归档
def test_the_loop_actually_drives_both_debt_sweeps(rt: Runtime, monkeypatch):
    """空转那一支必须**真的接上** `_ensure_vectors` 与 `_sweep_disk`。

    这里守的是一个具体踩过的坑：两个方法都写好了、单测也都直接调它们、全绿 ——
    而 `_loop` 里那一支忘了调。于是「磁盘欠账」这条设计上存在的路**根本没跑过**，
    而所有测试与统计都显示一切正常。
    """
    w = MemoryWorker()
    calls: list[str] = []

    def fake_get(timeout: int = 30) -> str:
        # 空转一次，然后让下一轮开头就退出循环
        w._stop.set()
        return ""

    monkeypatch.setattr(w._q, "get", fake_get)
    monkeypatch.setattr(w._q, "put", lambda item: None)
    monkeypatch.setattr(w, "_ensure_vectors", lambda: calls.append("vectors"))
    monkeypatch.setattr(w, "_sweep_disk", lambda: calls.append("disk"))
    monkeypatch.setattr(w, "_sweep_idle", lambda: calls.append("idle"))
    w._loop()
    assert calls == ["vectors", "disk", "idle"], f"空转那一轮该做的三件事：{calls}"


# ---------------------------------------------------------------- 向量欠账
class NotReadyEmbedder(MockEmbedder):
    """装上了但还没加载好 —— 模型在下载中、或宿主还没预热。"""

    def ready(self, download: bool = True) -> bool:
        return False


def vector_count(uid: str) -> int:
    con = open_index(uid)
    try:
        return int(con.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"])
    finally:
        con.close()


def test_a_missing_model_puts_the_uid_on_the_backlog(rt: Runtime):
    """模型没就绪时**先欠着**：那轮整理照常成功，只是向量留给下一次补。

    不记这笔账的话，欠下的那些向量永远补不回来 —— 而「向量少了」是静默的：
    召回差一点，没有任何报错、没有进任何指标。
    """
    w = MemoryWorker()
    say(UID, "我家猫叫团子")
    with using(Runtime(config=rt.config, embedder=NotReadyEmbedder())):
        out = w.extract_uid(
            UID,
            call=lambda _m: facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            ),
        )
        assert out["ok"] is True, "向量欠着不影响整理本身"
        assert out["vectors"]["reason"] == "embed_unavailable"
        assert UID in w._vector_backlog
        w._ensure_vectors()
    assert UID in w._vector_backlog, "模型还是没好：接着欠"


def test_the_backlog_is_drained_once_the_model_shows_up(rt: Runtime):
    """模型到位之后由维护循环补上，整理不用重跑。"""
    w = MemoryWorker()
    say(UID, "我家猫叫团子")
    with using(Runtime(config=rt.config, embedder=NotReadyEmbedder())):
        w.extract_uid(
            UID,
            call=lambda _m: facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            ),
        )
    assert UID in w._vector_backlog

    with using(Runtime(config=rt.config, embedder=MockEmbedder())):
        w._ensure_vectors()
    assert w._vector_backlog == set(), "补上了就从欠账里划掉"
    assert vector_count(UID) == 1, "向量真的写进去了"


def test_a_healthy_model_indexes_during_the_extraction_pass(rt: Runtime):
    """正常情况下向量跟着整理走 —— 放在对话里编码就是每轮几十毫秒的首字延迟。"""
    w = MemoryWorker()
    say(UID, "我家猫叫团子")
    with using(Runtime(config=rt.config, embedder=MockEmbedder())):
        out = w.extract_uid(
            UID,
            call=lambda _m: facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            ),
        )
    assert out["vectors"]["encoded"] == 1
    assert w._vector_backlog == set()
    assert vector_count(UID) == 1


def test_the_backlog_rotates_so_stuck_uids_do_not_block_the_queue(rt: Runtime, monkeypatch):
    """持续失败的那些 uid 会永远占着前 SCAN_BATCH 个名额 —— 排在后面的
    欠账一次都补不上，而表现只是「召回差一点」，不报错。
    """
    w = MemoryWorker()
    w.SCAN_BATCH = 1
    w._vector_backlog = {"stuck-a", "stuck-b", "stuck-c"}
    seen: list[str] = []

    class Stuck(MockEmbedder):
        def encode(self, texts):
            return None  # 永远编不出来（比如模型在、但输入格式不对）

    monkeypatch.setattr(
        "changqing.worker.reindex", lambda uid, **kw: seen.append(uid) or {"ok": False}
    )
    with using(Runtime(config=rt.config, embedder=Stuck())):
        for _ in range(3):
            w._ensure_vectors()
    assert sorted(seen) == ["stuck-a", "stuck-b", "stuck-c"], (
        "轮转之后三个都轮到过，谁也不占着名额不放"
    )


# ---------------------------------------------------------------- L0 归档
def _old_say(uid: str, text: str, days_ago: int) -> None:
    """往「很多天以前」那天写一轮原话（归档的判据是那一天属于哪个月）。"""
    append_turn(uid, {"user": text, "assistant": "嗯", "ts": time.time() - days_ago * 86400})


def test_overdue_turns_are_archived(rt: Runtime, mem_root):
    """超期的原话搬进 gzip —— **不是删除**，`read_turns` 透明读回两处。"""
    _old_say(UID, "很久以前说过的话", 900)
    say(UID, "今天说的话")
    before = read_turns(UID)

    out = MemoryWorker()._archive_if_due(UID)
    assert out["files"] == 1, "搬走了一个月"
    assert read_turns(UID) == before, "读回来一字不差 —— 归档不该改变任何读取结果"
    assert list(mem_root.rglob("sessions.archive/*.gz")), "确实落在归档目录里"


def test_archiving_happens_at_most_once_a_day(rt: Runtime):
    """它挂在 `maybe_extract` 最开头，而那是被**按批批量**调用的 ——
    日常开销必须只是「一次 state 读 + 一次日期比较」。
    """
    _old_say(UID, "很久以前说过的话", 900)
    w = MemoryWorker()
    # 这一天是**代码在它自己那一刻**写下的，所以期望值要把跨越午夜的那一次也认下来
    # —— 否则这条断言会在 00:00 前后偶发变红，而红的是环境不是代码。
    before = time.strftime("%Y-%m-%d")
    assert w._archive_if_due(UID)["files"] == 1
    assert w._archive_if_due(UID) == {}, "今天第二次是空的（没再扫目录）"
    assert load_state(UID)["last_archive_scan"] in (before, time.strftime("%Y-%m-%d"))


def test_archiving_is_skipped_when_retention_is_off(rt: Runtime, mem_root):
    """一行关掉 = 永不清理：`retain_months <= 0` 连目录都不该扫。"""
    _old_say(UID, "很久以前说过的话", 900)
    w = MemoryWorker()
    with using(Runtime(config=rt.config.evolved(retain_months=0))):
        assert w._archive_if_due(UID) == {}
    assert not list(mem_root.rglob("sessions.archive/*.gz"))


def test_archiving_runs_even_when_there_is_nothing_to_extract(rt: Runtime):
    """判据在归档**之后**：一个不再说话的用户也得轮得到归档。

    顺序反了的话，`should_run` 一返回空就整个早退 —— 那段永远停在保留期外的
    原话不但搬不走，还每次空转都被重扫一遍。
    """
    _old_say(UID, "很久以前说过的话", 900)
    update_state(UID, user_rounds=1, extracted_rounds=1)  # 早就抽完了，没有新东西
    out = MemoryWorker().maybe_extract(UID)
    assert out["skipped"] is True, "确实没有新原话要抽"
    assert out["archive"]["files"] == 1, "但归档照样跑了"
