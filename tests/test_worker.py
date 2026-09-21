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

from changqing.runtime import Runtime, using
from changqing.store import (
    append_turn,
    list_facts,
    list_summaries,
    list_topics,
    load_state,
    read_turns,
    update_state,
)
from changqing.worker import CHUNK_TURNS, MemoryWorker, user_rounds_of
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
    assert got == ["T-000001", "T-000002"], "本段的 user 轮 + 它后面她的承诺行"


def test_her_lines_are_only_added_when_they_look_like_a_promise(rt: Runtime):
    """预筛宽一点只多花一点上下文，抠紧会**静默漏掉**真正的承诺。"""
    turns = [
        {"id": "T-000001", "role": "user", "text": "在吗"},
        {"id": "T-000002", "role": "assistant", "text": "在的，今天画了一下午"},
    ]
    assert [t["id"] for t in lines_for_model(turns, [turns[0]])] == ["T-000001"], (
        "不带承诺味道的她的行不进 prompt"
    )


def test_her_lines_are_capped_and_truncated(rt: Runtime):
    """上限是成本上界：一场对话里她的回复量是他说的话的好几倍。"""
    turns = [{"id": "T-000000", "role": "user", "text": "在吗"}]
    for i in range(1, 12):
        turns.append({"id": f"T-{i:06d}", "role": "assistant", "text": "我明天给你带" + "话" * 300})
    got = lines_for_model(turns, [turns[0]])
    her = [t for t in got if t["role"] == "assistant"]
    assert len(her) == 6, "每个 chunk 最多夹 6 条她的行"
    assert all(len(t["text"]) == 120 for t in her), "超长的截断，不是整条丢掉"


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
