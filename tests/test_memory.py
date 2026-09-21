"""门面：公开面、两套名字、以及「能力在调用时解析」这条装配规则。

门面是最容易被悄悄改坏的一层 —— 它没有复杂的算法，只有**名字与顺序**。
而名字改了不报错：宿主那边是 `AttributeError`（还算好），
顺序改了更糟：`update` 先追加再校验的话，一条坏值会永久卡住那个用户的重放。
所以这里逐条钉住。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changqing import EDITABLE_FIELDS, EditError, Memory, MemoryConfig
from changqing.adapters.mock import MockEmbedder, MockLLM
from changqing.runtime import Runtime, runtime

UID = "u" + "m" * 16


def facts_json(*facts: dict) -> str:
    return json.dumps(
        {
            "facts": [{"subject": "他", "kind": "fact", "confidence": 0.9, **f} for f in facts],
            "summary": "",
            "topics": [],
        },
        ensure_ascii=False,
    )


def scripted(*payloads: str) -> MockLLM:
    return MockLLM(list(payloads), default=json.dumps({"facts": []}))


def handle(rt: Runtime, **kw) -> Memory:
    """一个绑在隔离库根上的句柄。默认带上离线编码器，向量相关才好断言。"""
    kw.setdefault("embedder", MockEmbedder())
    return Memory(UID, config=rt.config, **kw)


# ---------------------------------------------------------------- 公开面
def test_the_public_surface_is_pinned(rt: Runtime):
    """`__all__` 快照。接口静默漂移比接口报错贵得多 —— 报错当场就发现，
    漂移要等到有人按文档写代码的时候。
    """
    import changqing

    assert changqing.__all__ == [
        "Memory",
        "MemoryConfig",
        "PersonaProfile",
        "EDITABLE_FIELDS",
        "EditError",
        "Runtime",
        "configure",
        "reset",
        "runtime",
        "using",
        "Embedder",
        "LLM",
        "UsageSink",
        "NullEmbedder",
        "NullLLM",
        "NoUsage",
        "NEUTRAL",
        "estimate_tokens",
        "clip_to_tokens",
        "__version__",
    ]


def test_the_aliases_point_at_the_same_implementation():
    """别名必须是**同一个方法对象**，不能是「也叫这个名字的第二份实现」。

    第二份实现不会立刻出错：它只在有人改了一边之后才与另一边分家，
    而那已经是几个月以后的事了。
    """
    assert Memory.search is not Memory.recall, "签名不同，是包装（下一条用例钉住语义）"
    assert Memory.delete is not Memory.forget


def test_the_mem0_aliases_do_the_same_thing(rt: Runtime):
    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    via_recall = mem.recall("我家猫叫什么")
    via_search = mem.search("我家猫叫什么")
    assert [f["id"] for f in via_recall] == [f["id"] for f in via_search]
    assert mem.get_all() == mem.get_all()
    assert mem.get(via_recall[0]["id"])["object"] == "团子"
    assert mem.delete(via_recall[0]["id"]) == "archive"
    assert mem.get(via_recall[0]["id"])["status"] != "active"


# ---------------------------------------------------------------- 输入归一
def test_a_message_list_splits_on_role_changes(rt: Runtime):
    """只在**角色切换**处切轮：用户的连续两句是同一口气，
    她连发三条也是同一轮（一轮里发好几条是常态，并成一条就再也查不回来）。
    """
    mem = handle(rt)
    ids = mem.add(
        [
            {"role": "user", "content": "在吗"},
            {"role": "user", "content": "我今天面试了"},
            {"role": "assistant", "content": "怎么样"},
            {"role": "assistant", "content": "紧张吗"},
            {"role": "user", "content": "还行"},
        ]
    )
    # 返回的是 **L0 每一条消息**的 id，不是「轮次」的 id —— 所以这里是 4：
    # 一句用户 + 两条她的消息 + 一句用户。
    assert len(ids) == 4
    from changqing.store import read_turns

    turns = read_turns(UID)
    assert [t["role"] for t in turns] == ["user", "assistant", "assistant", "user"]
    assert "我今天面试了" in turns[0]["text"] and "在吗" in turns[0]["text"], (
        "同一口气里的两句并成一条，但**没丢**"
    )


def test_a_bare_string_is_one_turn_he_has_not_been_answered_on_yet(rt: Runtime):
    """她还没回的时候也要能记：等他下一句来的时候再补助手那一半。"""
    mem = handle(rt)
    ids = mem.add("我先睡了啊")
    assert len(ids) == 1
    from changqing.store import read_turns

    assert [t["role"] for t in read_turns(UID)] == ["user"]


def test_a_turn_dict_passes_through_untouched(rt: Runtime):
    mem = handle(rt)
    ids = mem.add({"user": "在", "assistant": ["嗯", "怎么了"]})
    assert len(ids) == 3, "她一轮里的两条消息各自留一条 L0 记录"


# ---------------------------------------------------------------- 装配
def test_capabilities_are_resolved_at_call_time_not_construction(rt: Runtime):
    """构造完之后再 `configure(llm=...)`，已经建好的句柄必须**看得到**。

    构造时把运行期拷一份的话，句柄与 `runtime()` 就成了两个真源 ——
    它们不一致时不报错，表现是「我明明配好了，它就是不用」。
    """
    mem = Memory(UID, config=rt.config)
    llm = MockLLM([json.dumps({"facts": []})])
    runtime().llm = llm  # type: ignore[misc]
    try:
        assert mem.effective().llm is llm
    finally:
        runtime().llm = rt.llm  # type: ignore[misc]


def test_an_explicit_override_wins_over_the_process_runtime(rt: Runtime):
    """显式传进来的那一项要压过进程级的那一份 —— 这是「一个进程两个库」的入口。"""
    mine = MockEmbedder()
    mem = Memory(UID, config=rt.config, embedder=mine)
    assert mem.effective().embedder is mine
    assert mem.effective().config is rt.config


def test_one_handle_does_not_leak_its_config_to_the_process(rt: Runtime, tmp_path: Path):
    """句柄作用域结束后，进程级那一份必须原样还原。

    不还原的后果是**跨用户串库**：下一个请求拿到的根目录是上一个人的。
    """
    other = MemoryConfig(root=tmp_path / "elsewhere")
    Memory(UID, config=other).add("在吗")
    assert runtime().config is rt.config, "进程级那一份没被动过"


def test_an_empty_uid_is_refused_rather_than_written_into_the_root(rt: Runtime):
    """没有 uid 就没有「谁的记忆」—— 写进库根会在根目录里造出一堆孤儿文件，
    而它看起来只是「多了几个不认识的目录」。
    """
    mem = Memory("")
    assert mem.remember({"user": "在吗"}) == []
    assert mem.recall("在吗") == []
    assert not any(Path(rt.config.root).glob("*.md"))


# ---------------------------------------------------------------- 改一条
def test_update_appends_an_op_instead_of_editing_the_view(rt: Runtime):
    """改内容追加一条 EDIT 操作，不动物化视图 —— 视图是能被重建的派生物。"""
    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    fid = mem.get_all()[0]["id"]

    out = mem.update(fid, object="团子（三岁）")
    assert out["object"] == "团子（三岁）"
    ops = mem.history(fid)
    assert [op["op"] for op in ops] == ["ADD", "EDIT"], "日志里两条：原来那条 + 这次改"
    assert ops[-1]["set"] == {"object": "团子（三岁）"}


def test_a_bad_value_is_refused_before_anything_is_written(rt: Runtime):
    """**校验必须在追加之前。** 坏值一旦进了只追加的日志，就会永久卡住那个
    用户的重放：此后所有新事实都不再落库，界面上只看得出「她记性变差」。
    """
    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    fid = mem.get_all()[0]["id"]
    before = mem.history()

    with pytest.raises(EditError):
        mem.update(fid, importance="high")
    with pytest.raises(EditError):
        mem.update(fid, object=None)
    with pytest.raises(EditError):
        mem.update(fid, due="下周三")
    with pytest.raises(EditError):
        mem.update(fid, kind="guess")
    with pytest.raises(EditError):
        mem.update(fid, body="不在白名单里")
    assert mem.history() == before, "一条坏 op 都没进日志"


def test_update_says_no_to_a_field_that_is_not_editable(rt: Runtime):
    """`turn_ref` / `quote` 是「这条凭什么是真的」的证据 —— 改得了就等于能伪造出处。"""
    assert "turn_ref" not in EDITABLE_FIELDS
    assert "quote" not in EDITABLE_FIELDS
    assert "persona_attention" in EDITABLE_FIELDS, "画像权重是可调的"


def test_update_returns_none_for_a_fact_that_does_not_exist(rt: Runtime):
    """不存在的 id 不该被静默地写进日志（那条 EDIT 会永远挂在一个死 id 上）。"""
    mem = handle(rt)
    assert mem.update("F-9999", object="随便") is None
    assert mem.history() == []


def test_editing_a_fact_refreshes_its_vector(rt: Runtime):
    """内容变了向量要跟着变。不重建的话，要等到下次后台整理才生效，
    而中间这段时间检索按**旧内容**召回 —— 用户看到「她记着我上次说的那句」。
    """
    from changqing.store import open_index

    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    fid = mem.get_all()[0]["id"]

    def hashes() -> list[str]:
        con = open_index(UID)
        try:
            return [r["text_hash"] for r in con.execute("SELECT text_hash FROM vectors")]
        finally:
            con.close()

    before = hashes()
    mem.update(fid, object="团子（三岁）")
    after = hashes()
    assert len(after) == 1, "一条事实一条向量"
    assert after != before, "向量按新内容重编了"


# ---------------------------------------------------------------- 忘一条
def test_forget_leaves_the_evidence_and_says_so(rt: Runtime):
    """`purge` 删的是**事实行**，引文与 L0 原话都还在。返回值要说清楚是哪一种，
    调用方（界面）才能如实告诉用户「可恢复」还是「已经删了」。
    """
    from changqing.store import read_ops, read_turns

    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    fid = mem.get_all()[0]["id"]

    assert mem.forget(fid) == "archive", "默认归档，可恢复"
    assert read_turns(UID), "L0 原话一个字节都没删"
    assert [op["op"] for op in read_ops(UID)][-1] == "FORGET"
    assert mem.forget(fid, "purge") == "purge"


def test_forget_refreshes_the_vector_index(rt: Runtime):
    """被归档的那条向量必须离开索引。冷路径的向量召回是「先按相似度取满、
    再按存活过滤」，留着它会继续占名额，把还活着的那条挤出去。
    """
    from changqing.store import open_index

    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    mem.forget(mem.get_all()[0]["id"])
    con = open_index(UID)
    try:
        n = con.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"]
    finally:
        con.close()
    assert n == 0


# ---------------------------------------------------------------- 读日志
def test_history_can_be_narrowed_to_one_fact(rt: Runtime):
    """「她为什么变成这样记得 / 不记得」只有日志答得出来。"""
    mem = handle(rt)
    mem.add({"user": "甲"})
    mem.add({"user": "乙"})
    mem.note_topic("想问他面试怎么样")
    everything = mem.history()
    assert len(everything) >= 1
    assert mem.history("T-000001") == [], "轮次不是操作，日志里没有它"
    assert mem.history("nope") == []


def test_export_hands_over_the_source_of_truth(rt: Runtime):
    """导出要连操作日志一起给：它是唯一事实源，有它就能把这份记忆完整重建。"""
    mem = handle(
        rt,
        llm=scripted(
            facts_json(
                {
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                }
            )
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    out = mem.export()
    assert out["uid"] == UID
    assert [f["object"] for f in out["facts"]] == ["团子"]
    assert out["ops"], "操作日志一起给出去"


# ---------------------------------------------------------------- 手动改写
def test_validation_accepts_what_the_panel_would_send(rt: Runtime):
    """面板发的就是一块 dict：多几个不认识的键是正常的，不该报错；
    但白名单里的字段值不合法必须报。
    """
    from changqing.edit import validate

    assert validate(
        {"object": "团子", "importance": 0.7, "due": "", "kind": "fact", "whatever": 1}
    ) == {"object": "团子", "importance": 0.7, "due": "", "kind": "fact"}
    assert validate({}) == {}


def test_a_true_boolean_is_not_a_valid_number():
    """`True` 是 `int` 的子类，顺手放进去会悄悄变成 1.0 ——
    一个「0 到 1 之间」的校验没挡住布尔值，看起来像通过。
    """
    from changqing.edit import validate

    with pytest.raises(EditError):
        validate({"importance": True})
