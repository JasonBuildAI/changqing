"""抽取的四道闸与它们的失败模式。

这些用例的形状有条共同点：**每一条都对应一种静默失败**（丢一条事实不报错、
编一个日期不报错、把她的承诺当成他说过的话不报错）。断言写得比通常啰嗦，
是因为这里错了之后症状只是「她记性不好」。
"""

from __future__ import annotations

from changqing import MemoryConfig, PersonaProfile
from changqing.extract import (
    EXTRACT_SYSTEM,
    absolutize,
    build_messages,
    build_system,
    clean_fact,
    parse_facts,
    parse_summary,
    parse_topics,
    resolve_ops,
    verify,
)
from changqing.runtime import Runtime, using


def _turn(tid: str = "T-000001", role: str = "user", text: str = "我家猫叫团子") -> dict:
    return {"id": tid, "day": "2026-09-21", "time": "10:00", "role": role, "text": text}


# ---------------------------------------------------------------- prompt
def test_prompt_never_hardcodes_a_persona():
    """出厂 prompt 里不许出现任何具体角色的设定 —— 那是注入进来的内容。"""
    assert "{persona}" not in build_system()
    assert "{attention}" not in build_system()
    assert "以她的视角" in build_system()


def test_prompt_carries_the_injected_persona(rt: Runtime):
    mine = PersonaProfile(name="小满", description="爱做饭，话不多")
    with using(Runtime(config=rt.config, persona=mine)):
        text = build_system()
    assert "以小满的视角（爱做饭，话不多）" in text
    assert "小满这个人会在意这条吗" in text
    # 字段名跟着换过：老名字留在 prompt 里等于让模型填一个没人读的字段
    assert "her_attention" not in text
    assert '"persona_attention":0.8' in text


def test_prompt_marks_who_said_what():
    """行首的说话人是铁律 1 / 2 唯一的可判依据，缺了它整条抽取没有依据。"""
    msgs = build_messages(
        [_turn(), _turn("T-000002", "assistant", "我周末带你去看展")], "2026-09-21"
    )
    body = msgs[1]["content"]
    assert "他说：我家猫叫团子" in body
    assert "她说：我周末带你去看展" in body
    assert msgs[0]["role"] == "system"


def test_prompt_still_mentions_the_invariants():
    """改 prompt 的人最容易顺手删掉这几样，而它们各自挡一类错。"""
    assert "宁少勿错" in EXTRACT_SYSTEM
    assert "她答应的事" in EXTRACT_SYSTEM, "承诺类事实不能从口径里消失"
    assert "一字不改" in EXTRACT_SYSTEM, "quote 是回引的证据"


def test_prompt_keeps_her_restating_rule():
    """「她复述他的话不是新信息」只在 prompt 里 —— 一句话是她真说的还是她在
    复述，文本上分不开，所以这条口径没有第二处落点可守。"""
    assert "复述" in EXTRACT_SYSTEM
    assert "她自己的" in EXTRACT_SYSTEM


# ---------------------------------------------------------------- 解析
def test_parse_facts_accepts_both_shapes():
    fenced = '```json\n{"facts": [{"subject": "他"}], "summary": "s"}\n```'
    assert parse_facts(fenced) == [{"subject": "他"}]
    bare = '[{"subject": "他"}, "垃圾"]'
    assert parse_facts(bare) == [{"subject": "他"}]


def test_parse_facts_survives_a_trailing_comma():
    """模型爱在最后一项后面留个逗号，而 `json.loads` 对此零容忍。"""
    assert parse_facts('{"facts": [{"a": 1},]}') == [{"a": 1}]


def test_parse_facts_returns_empty_on_garbage():
    assert parse_facts("今天天气不错") == []
    assert parse_facts("") == []


def test_parse_summary_reads_the_summary_field():
    assert parse_summary('{"summary": " 他养了只猫 "}') == "他养了只猫"
    assert parse_summary("没有 JSON") == ""


def test_parse_topics_caps_and_validates(rt: Runtime):
    raw = (
        '{"topics": ['
        '{"text": "他那个方案改了没有", "kind": "followup", "due_day": "2026-09-23"},'
        '{"text": "第二个", "kind": "乱写"},'
        '{"text": "第三个"},'
        '{"text": "第四个会被砍掉"}]}'
    )
    out = parse_topics(raw, day="2026-09-21")
    assert len(out) == 3
    assert out[0] == {
        "text": "他那个方案改了没有",
        "kind": "followup",
        "due_day": "2026-09-23",
        "ref": "",
    }
    assert out[1]["kind"] == "share"  # 非法 kind 归一到 share
    assert out[1]["due_day"] == ""  # 没给日期就是空串，不是 None


def test_parse_topics_drops_an_unparsable_due_day(rt: Runtime):
    """算不出来的日期当没给 —— 一个错的截止日期比没有更糟。"""
    out = parse_topics('{"topics": [{"text": "体检", "due_day": "下辈子"}]}', day="2026-09-21")
    assert out[0]["due_day"] == ""


# ---------------------------------------------------------------- 闸 4
def test_absolutize_uses_the_turn_date_not_today():
    """2026-09-21 是周一。相对词一律按**那一轮**的日期算，不是按今天。"""
    assert absolutize("上周五去看了展", "2026-09-21") == "2026-09-18去看了展"
    assert absolutize("这周三", "2026-09-21") == "2026-09-23"
    assert absolutize("下周一", "2026-09-21") == "2026-09-28"
    assert absolutize("昨天", "2026-09-21") == "2026-09-20"


def test_absolutize_drops_bare_anchors_but_keeps_aspect_words():
    """算不准的锚点直接去掉；「以前 / 之前 / 最近」是内容，动了就把意思改反。"""
    assert absolutize("上个月搬了家", "2026-09-21") == "搬了家"
    assert absolutize("我以前住北京", "2026-09-21") == "我以前住北京"
    assert absolutize("周末", "2026-09-21") == "周末"


def test_absolutize_leaves_text_alone_when_the_date_is_bad():
    """日期形状不对时**原样返回**：拿一个瞎猜的基准去算，比不换算更糟。"""
    assert absolutize("上周五", "不是日期") == "上周五"
    assert absolutize("", "2026-09-21") == ""


# ---------------------------------------------------------------- 闸 1 / 2
def test_verify_drops_a_fact_without_a_back_reference():
    turns = {"T-000001": _turn()}
    assert verify({"quote": "我家猫叫团子"}, turns) == ("drop", 0.0)
    assert verify({"turn_ref": "T-000001"}, turns) == ("drop", 0.0)


def test_verify_drops_a_reference_to_a_turn_that_is_not_there():
    """回引到一个不存在的轮次 = 编的。这一条挡的是「模型自己发明了编号」。"""
    state, _ = verify({"turn_ref": "T-999999", "quote": "我家猫叫团子"}, {"T-000001": _turn()})
    assert state == "drop"


def test_verify_accepts_a_verbatim_quote():
    turns = {"T-000001": _turn(text="我家猫叫团子，三岁了")}
    assert verify({"turn_ref": "T-000001", "quote": "我家猫叫团子"}, turns) == ("active", 1.0)


def test_verify_marks_a_paraphrase_pending_and_keeps_it():
    """近似不是编造，标 pending 留着让人确认 —— 丢掉它等于丢一条真的记忆。"""
    turns = {"T-000001": _turn(text="我最近在接私活，画插画那种")}
    fact = {"turn_ref": "T-000001", "quote": "私活是插画"}
    state, score = verify(fact, turns)
    assert state == "pending"
    assert 0.0 < score < 1.0


def test_verify_takes_her_own_things_but_not_her_restating_him():
    """铁律 1 / 2：她的行只收**她自己的事** —— `kind=promise`（她答应的事）或
    `subject=她`（她的偏好 / 习惯 / 正在做的事），其余一律不算他说过的事实。

    放宽少了：她的承诺与偏好都进不了库，表现是「她忘了自己答应过什么」，
    以及今天说爱喝美式、明天说从来不喝咖啡。放宽多了：她复述他的话会被当成
    他提供的信息，再反过来变成「你上次说……」。
    """
    turns = {"T-000002": _turn("T-000002", "assistant", "我周末带你去看展")}
    assert verify({"turn_ref": "T-000002", "quote": "我周末带你去看展"}, turns)[0] == "drop"
    assert verify(
        {"turn_ref": "T-000002", "quote": "我周末带你去看展", "kind": "promise"}, turns
    ) == ("active", 1.0), "她答应过的事要能落进事实层"
    assert verify(
        {"turn_ref": "T-000002", "quote": "我周末带你去看展", "subject": "她", "kind": "fact"},
        turns,
    ) == ("active", 1.0), "她自己的事（subject=她）不标 promise 一样要收"
    assert (
        verify({"turn_ref": "T-000002", "quote": "我周末带你去看展", "subject": "他"}, turns)[0]
        == "drop"
    ), "她复述他的话标成 subject=他，不许收"
    assert (
        verify({"turn_ref": "T-000002", "quote": "我周末带你去看展", "kind": "commitment"}, turns)[
            0
        ]
        == "drop"
    ), "「他答应的事」不能从她的行里抽"


def test_verify_reads_the_tolerance_from_the_runtime(rt: Runtime):
    """阈值从运行期现取 —— 模块级快照会让「改了配置不生效」且不报错。

    这条用例的力气全在「同一个 fact、同一份 turns，只换配置」上：把阈值挪回模块级，
    两次结果会一模一样，而它**不会报错**。
    """
    turns = {"T-000001": _turn(text="我最近在接私活，画插画那种")}
    fact = {"turn_ref": "T-000001", "quote": "私活是插画"}
    strict = MemoryConfig(root=rt.config.root, pending_tolerance=0.9)
    loose = MemoryConfig(root=rt.config.root, pending_tolerance=0.5)
    with using(Runtime(config=strict)):
        assert verify(fact, turns)[0] == "drop"
    with using(Runtime(config=loose)):
        assert verify(fact, turns)[0] == "pending"


def test_verify_treats_a_zero_tolerance_as_the_configured_default(rt: Runtime):
    """`0.0` 的含义是「用配置里那一份」，不是「阈值 0」。

    这是沿用下来的口径，藏着一个陷阱：想表达「什么都收」得传一个正数。
    钉住它是为了让下一个人改口径时看见自己动了什么。
    """
    turns = {"T-000001": _turn(text="我最近在接私活，画插画那种")}
    fact = {"turn_ref": "T-000001", "quote": "私活是插画"}
    with using(Runtime(config=MemoryConfig(root=rt.config.root, pending_tolerance=0.9))):
        assert verify(fact, turns, tolerance=0.0)[0] == "drop"


# ---------------------------------------------------------------- 清洗
def test_clean_fact_keeps_facts_without_an_object():
    """中文里「怕黑」根本没有宾语。要求 object 非空会把它们**静默丢光**。"""
    out = clean_fact({"predicate": "怕黑", "object": ""}, _turn(), PersonaProfile())
    assert out is not None
    assert out["object"] == ""
    assert out["predicate"] == "怕黑"


def test_clean_fact_still_drops_an_empty_pair():
    assert clean_fact({"predicate": " ", "object": ""}, _turn()) is None


def test_clean_fact_absolutizes_the_object_but_never_the_quote():
    """quote 是回引校验的证据，改过就不叫证据了。"""
    out = clean_fact(
        {"predicate": "去看了展", "object": "上周五", "quote": "我上周五去看了展"},
        _turn(),
        PersonaProfile(),
    )
    assert out is not None
    assert out["object"] == "2026-09-18"
    assert out["quote"] == "我上周五去看了展"


def test_clean_fact_takes_her_subject_from_the_persona():
    """她那一侧（承诺与她自己的事）的归属文案是**形状**，不是自由内容。

    模型把她的承诺写成「他」，她会拿自己的承诺说成「你上次说……」；把她的偏好
    写成「我们」，卡片会读成「我们养的猫叫团子」—— 所以两种都要收口。
    """
    out = clean_fact(
        {"subject": "他", "predicate": "带你去看展", "kind": "promise"},
        _turn(role="assistant"),
        PersonaProfile(name="小满"),
    )
    assert out is not None
    assert out["subject"] == "小满"

    own = clean_fact(
        {"subject": "她", "predicate": "平时都喝", "object": "美式"},
        _turn(role="assistant"),
        PersonaProfile(name="小满"),
    )
    assert own is not None
    assert own["subject"] == "小满", "她自己的事跟承诺同一套归属"

    third = clean_fact(
        {"subject": "我们", "predicate": "养的猫叫", "object": "团子"},
        _turn(),
        PersonaProfile(name="小满"),
    )
    assert third is not None
    assert third["subject"] == "他", "模型给的第三种写法一律收口到「他」"


def test_clean_fact_normalizes_numbers_and_renames_the_weight_field():
    out = clean_fact(
        {"predicate": "重要", "confidence": "abc", "persona_attention": 3},
        _turn(),
        PersonaProfile(),
    )
    assert out is not None
    assert out["confidence"] == 0.7  # 不是数 → 默认值
    assert out["persona_attention"] == 1.0  # 越界 → 夹到上界
    assert "her_attention" not in out


# ---------------------------------------------------------------- 冲突消解
def test_resolve_ops_adds_then_supersedes_then_noops(rt: Runtime):
    """三种结局各跑一次：这正是**幂等**的落点（同一段原话整理两次必须一样）。"""
    from changqing import store as ST

    uid = "alice"
    base = {"subject": "他", "predicate": "养的猫叫", "quote": "我家猫叫团子"}

    ops, stats = resolve_ops(uid, [dict(base, object="团子")])
    assert stats == {"ADD": 1, "SUPERSEDE": 0, "NOOP": 0}
    for op in ops:
        ST.append_op(uid, op)
    ST.materialize(uid)

    ops, stats = resolve_ops(uid, [dict(base, object="团子")])
    assert stats == {"ADD": 0, "SUPERSEDE": 0, "NOOP": 1}
    assert ops == []

    ops, stats = resolve_ops(uid, [dict(base, object="咪咪")])
    assert stats == {"ADD": 0, "SUPERSEDE": 1, "NOOP": 0}
    assert ops[0]["op"] == "SUPERSEDE"
    assert ops[0]["replaces"]


def test_resolve_ops_collapses_repeats_inside_one_batch(rt: Runtime):
    """同一次整理里出现两条同槽位事实时，第二条取代的是刚写下的那条。"""
    base = {"subject": "他", "predicate": "养的猫叫", "quote": "我家猫叫团子"}
    ops, stats = resolve_ops("bob", [dict(base, object="团子"), dict(base, object="咪咪")])
    assert stats == {"ADD": 1, "SUPERSEDE": 1, "NOOP": 0}
    assert ops[1]["replaces"] == ops[0]["id"]
