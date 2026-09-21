"""一条链路走到底：原话 → 整理 → 事实 → 检索 → 忘记 → 导出重建。

前面每一份用例都在验一个局部，这一份验的是**它们接起来还成不成立**。
接缝处的缺陷不是任何一段的错，所以只有整条跑一遍才看得见：

  · 写进去的原话，整理得看得到（游标对齐）；
  · 抽出来的事实，检索真的拿得到（索引、全文、槽位三路都通）；
  · **刚说过的**与 **100 天前说过的**一样能想起（这是这个库存在的全部理由）；
  · 忘了之后检索拿不到，而原话与操作日志仍在；
  · 导出的操作日志能把这份记忆**完整重建**出来。

全程离线：`MockEmbedder` 加脚本化的 `MockLLM`，没有网络也没有模型文件。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from changqing import Memory
from changqing.adapters.mock import MockEmbedder, MockLLM
from changqing.runtime import Runtime
from changqing.store import list_facts, read_ops, read_turns, rebuild

UID = "u" + "e" * 16
OTHER = "u" + "x" * 16


def reply(*facts: dict, summary: str = "", topics: list | None = None) -> str:
    body = {
        "facts": [{"subject": "他", "kind": "fact", "confidence": 0.9, **f} for f in facts],
        "summary": summary,
        "topics": topics or [],
    }
    return json.dumps(body, ensure_ascii=False)


def facts_for(quote: str, predicate: str, obj: str, ref: str) -> str:
    return reply({"predicate": predicate, "object": obj, "quote": quote, "turn_ref": ref})


def mem_of(rt: Runtime, llm: MockLLM, uid: str = UID) -> Memory:
    return Memory(uid, config=rt.config, embedder=MockEmbedder(), llm=llm)


# ---------------------------------------------------------------- 核心承诺
def test_she_remembers_both_what_was_just_said_and_what_was_said_long_ago(rt: Runtime):
    """**这个库存在的全部理由。** 两条都要成立，缺一条这套分层就没意义：

      · 100 天前说过的那件事 —— 光靠上下文早就不在了（一味加长一定会爆）；
      · 刚刚才说的那一件 —— 光靠检索不行（它还没进任何索引）。

    所以两层必须都有：L0 原话负责「刚说过的」，抽出来的事实负责「很久以前的」。
    """
    long_ago = time.time() - 100 * 86400.0
    assert read_turns(UID) == []

    llm = MockLLM(
        [
            facts_for("我家猫叫团子，三岁了", "养的猫叫", "团子", "T-000001"),
            facts_for("我下周要搬到杭州去", "要搬到", "杭州", "T-000003"),
        ],
        default=reply(),
    )
    mem = mem_of(rt, llm)
    mem.remember({"user": "我家猫叫团子，三岁了", "assistant": "记住了。", "ts": long_ago})
    mem.extract_now()
    mem.remember({"user": "我下周要搬到杭州去", "assistant": "嗯。", "ts": time.time()})
    mem.extract_now()

    assert [f["object"] for f in mem.recall("我家猫叫什么")] == ["团子"]
    assert [f["object"] for f in mem.recall("我要搬到哪里去")] == ["杭州"]

    rows = read_turns(UID)
    assert [r["text"] for r in rows if r["role"] == "user"] == [
        "我家猫叫团子，三岁了",
        "我下周要搬到杭州去",
    ]
    assert rows[0]["day"] == time.strftime("%Y-%m-%d", time.localtime(long_ago))


def test_a_fact_carries_the_evidence_that_makes_it_checkable(rt: Runtime):
    """每条事实都要能指回它出自哪一轮、原话是怎么说的。

    这是「她是不是编的」这个问题唯一可查的答案 —— 没有它，防幻觉那几道闸就只是
    抽取时的一次性检查，事后谁也复核不了。
    """
    llm = MockLLM([facts_for("我家猫叫团子", "养的猫叫", "团子", "T-000001")], default=reply())
    mem = mem_of(rt, llm)
    mem.add("我家猫叫团子")
    mem.extract_now()
    fact = mem.get_all()[0]
    assert fact["quote"] == "我家猫叫团子"
    assert any(
        r["id"] == fact["turn_ref"] and fact["quote"] in r["text"] for r in read_turns(UID)
    ), "引文真的在原话里 —— 可核对"


# ---------------------------------------------------------------- 忘掉
def test_forgetting_hides_it_from_recall_but_keeps_the_record(rt: Runtime):
    """「忘记」在人心里是删除，但原话与操作日志必须留着 —— 前者是「这条凭什么是
    真的」的证据，后者是唯一事实源。"""
    llm = MockLLM([facts_for("我家猫叫团子", "养的猫叫", "团子", "T-000001")], default=reply())
    mem = mem_of(rt, llm)
    mem.add("我家猫叫团子")
    mem.extract_now()
    fid = mem.get_all()[0]["id"]
    assert mem.recall("我家猫叫什么"), "先证明忘之前真的搜得到，否则下面那条是空转"

    mem.forget(fid)

    assert mem.recall("我家猫叫什么") == [], "忘了就搜不到了"
    assert read_turns(UID), "原话还在（真删只走整库重置）"
    assert [op["op"] for op in read_ops(UID)] == ["ADD", "FORGET"], "日志记着发生过什么"
    assert mem.get(fid)["status"] != "active", "行没被删掉，只是不再算数"


# ---------------------------------------------------------------- 可重建
def test_the_whole_memory_can_be_rebuilt_from_the_export(rt: Runtime):
    """导出的操作日志是**唯一事实源**：拿它重建出来的事实必须一模一样。

    这是「日志不可再生、索引可重建」这条分工的端到端版本。
    """
    llm = MockLLM(
        [
            facts_for("我家猫叫团子", "养的猫叫", "团子", "T-000001"),
            facts_for("我不吃香菜", "不吃", "香菜", "T-000002"),
        ],
        default=reply(),
    )
    mem = mem_of(rt, llm)
    mem.add("我家猫叫团子")
    mem.extract_now()
    mem.add("我不吃香菜")
    mem.extract_now()
    mem.update(mem.get_all()[0]["id"], object="团子（三岁）")

    exported = mem.export()
    snapshot = sorted((f["id"], f["object"], f["status"]) for f in exported["facts"])
    assert len(snapshot) == 2

    assert rebuild(UID) == len(exported["ops"]), "重建重放了全部操作"
    assert (
        sorted((f["id"], f["object"], f["status"]) for f in list_facts(UID, include_dead=True))
        == snapshot
    ), "重建之后事实一条不差"


# ---------------------------------------------------------------- 隔离
def test_two_peoples_memories_never_mix_even_in_one_process(rt: Runtime):
    """同一个进程里两个句柄、两个库。串库里没有任何一处会报错 ——
    它只是把 A 的猫摆到 B 的对话里。"""
    llm = MockLLM([facts_for("我家猫叫团子", "养的猫叫", "团子", "T-000001")], default=reply())
    mine = mem_of(rt, llm)
    mine.add("我家猫叫团子")
    mine.extract_now()

    theirs = mem_of(rt, MockLLM(default=reply()), uid=OTHER)
    # 先看盘：**问一句不该在盘上留下东西**。等下面调过 get_all 再断言就没意义了
    # —— 那一步会（正常地）把索引建出来。
    assert not (Path(rt.config.root) / "ux" / OTHER).exists(), "别人的库连目录都不该有"
    assert theirs.get_all() == []
    assert theirs.recall("我家猫叫什么") == []
    assert read_turns(OTHER) == []
