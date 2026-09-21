"""形状必须与实现对齐 —— 这一条只能靠机械校验，读代码看不出来。

`types.py` 里的定义没人 import，所以它们**不会**在运行时被检查（这是这类缺陷
最贵的地方）。它漂过一次：状态那个 `Literal` 里躺着一个代码从来不写的
`rejected`，而真的会写的 `forgotten` / `merged` 一个都没有；`Fact` 里还写着
一个表里根本不存在的列 `created_at`。

所以这里全部拿实现去对：状态与操作种类从源码里扫出来，形状与被测代码的真实
返回逐键比对，而且**两个方向都要** —— 少声明一个键是文档漏了，多声明一个键
是文档里有幻觉，两者的症状都是「有人照着它写代码，然后写错」。
"""

from __future__ import annotations

import json
import re
import typing
from pathlib import Path

import pytest

from changqing import Memory
from changqing.adapters.mock import MockEmbedder, MockLLM
from changqing.edit import KINDS
from changqing.runtime import Runtime
from changqing.store import archive_stats, read_turns
from changqing.store import stats as store_stats
from changqing.types import (
    ArchiveStats,
    Fact,
    FactKind,
    FactStatus,
    MemoryContext,
    MemoryStats,
    OpKind,
    ResetResult,
    RetrievalReason,
    StoreStats,
    Summary,
    Topic,
    TurnRow,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "changqing"
UID = "u" + "t" * 16


# ---------------------------------------------------------------- 取源码与形状
def _all_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SRC.rglob("*.py")))


def _source(*parts: str) -> str:
    return SRC.joinpath(*parts).read_text(encoding="utf-8")


def _declared(shape: type) -> set[str]:
    """一个 `TypedDict` 声明了哪些键。"""
    return set(shape.__annotations__)


def _literal(alias: object) -> set[str]:
    return set(typing.get_args(alias))


def _assert_same_keys(shape: type, actual: dict, name: str) -> None:
    declared, real = _declared(shape), set(actual)
    missing = sorted(declared - real)
    phantom = sorted(real - declared)
    assert declared == real, (
        f"{name} 与实现不一致：只声明没出现的 {missing}，出现了没声明的 {phantom}"
    )


# ---------------------------------------------------------------- 字面量
def test_the_status_literal_is_exactly_what_the_code_writes():
    """状态白名单 vs 源码里真的会写进去的那些值。

    少一个：有人按这个 `Literal` 做分支时会漏掉一个真实状态。
    多一个：定义里躺着一个从不存在的状态（`rejected` 就是这么躺了很久的）。
    """
    written = set(re.findall(r"status='([a-z]+)'", _all_source()))
    assert written, "一个状态都没扫到 —— 正则或代码写法变了，这条护栏已经失效"
    assert _literal(FactStatus) == written, (
        f"FactStatus 与代码不一致：只声明没写的 {sorted(_literal(FactStatus) - written)}，"
        f"写了没声明的 {sorted(written - _literal(FactStatus))}"
    )


def test_the_op_kind_literal_is_exactly_what_apply_op_handles():
    """操作种类 vs `apply_op` 真正认的那一组。

    多一个 = 日志里可能出现一条谁都不处理的操作（它会静默地什么也不做）；
    少一个 = 定义漏了一种真实操作。
    """
    body = _source("store", "ops.py")
    body = body[body.index("def apply_op") :]
    handled = set(re.findall(r'"([A-Z][A-Z_]{2,})"', body))
    assert handled, "一个操作都没扫到 —— 这条护栏已经失效"
    assert _literal(OpKind) == handled, (
        f"OpKind 与 apply_op 不一致：只声明没处理的 "
        f"{sorted(_literal(OpKind) - handled)}，处理了没声明的 {sorted(handled - _literal(OpKind))}"
    )


def test_the_fact_kind_literal_matches_the_edit_whitelist():
    """事实种类与 `edit.KINDS` 是同一份东西，两处写法不同（元组 vs `Literal`）。"""
    assert _literal(FactKind) == set(KINDS)


def test_the_retrieval_reasons_are_exactly_what_retrieve_reports():
    """冷路径的「为什么是空的」：超时与没命中都返回空，但处置完全相反。"""
    reported = set(re.findall(r'"(hot_only|no_fact|no_hit|timeout|error)"', _source("retrieve.py")))
    assert reported, "一个原因都没扫到 —— 这条护栏已经失效"
    assert _literal(RetrievalReason) == reported, (
        f"RetrievalReason 与 retrieve 不一致：只声明没报的 "
        f"{sorted(_literal(RetrievalReason) - reported)}，报了没声明的 "
        f"{sorted(reported - _literal(RetrievalReason))}"
    )


# ---------------------------------------------------------------- 形状
@pytest.fixture()
def full(rt: Runtime) -> Memory:
    """一个走完整链路之后、每种形状都至少有一条的句柄。"""
    payload = json.dumps(
        {
            "facts": [
                {
                    "subject": "他",
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                    "confidence": 0.9,
                    "importance": 0.6,
                }
            ],
            "summary": "第一次聊到他的猫。",
            "topics": [{"kind": "share", "text": "问问他团子多大了"}],
        },
        ensure_ascii=False,
    )
    mem = Memory(UID, config=rt.config, embedder=MockEmbedder(), llm=MockLLM(default=payload))
    mem.remember({"user": "我家猫叫团子", "assistant": "记住了，团子。"})
    mem.extract_now()
    mem.note_summary("这场聊到了他的猫。")
    mem.note_topic("问问他团子多大了。")
    return mem


def test_the_declared_shapes_match_what_the_code_returns(full: Memory):
    """逐键比对：声明的键与真实返回的键必须**完全相等**。"""
    cards = full.recall("我家猫叫什么")
    assert cards, "一条都没召回 —— 夹具没跑通，这条护栏也就没意义"
    _assert_same_keys(Fact, cards[0], "Fact")

    turns = read_turns(UID)
    assert turns, "L0 是空的 —— 夹具没跑通"
    _assert_same_keys(TurnRow, turns[0], "TurnRow")

    summaries = full.summaries()
    assert summaries, "没有纪要 —— 夹具没跑通"
    _assert_same_keys(Summary, summaries[0], "Summary")

    topics = full.topics(include_used=True)
    assert topics, "没有话题 —— 夹具没跑通"
    _assert_same_keys(Topic, topics[0], "Topic")

    _assert_same_keys(MemoryContext, full.context("你好呀"), "MemoryContext")
    _assert_same_keys(MemoryStats, full.stats(), "MemoryStats")
    _assert_same_keys(StoreStats, store_stats(UID), "StoreStats")
    _assert_same_keys(ArchiveStats, archive_stats(UID), "ArchiveStats")


def test_the_reset_shape_matches(full: Memory):
    """`reset()` 的返回：**如实报出实际做了什么**，而不是只说一句成功。

    单独一条用例：它会把这个夹具的库清掉，不该影响上面那条。
    """
    _assert_same_keys(ResetResult, full.reset(), "ResetResult")
