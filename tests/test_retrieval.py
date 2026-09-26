"""检索：门控与素材同源、预算是真的、故事线是常数、坏掉不抛异常。

这一层最贵的一类缺陷是**两处各说各的**：门控说她「手里没料」，而卡片里明明
写着一条承诺 —— 于是她拿着一句刚答应过的事，却说「我还不知道你什么呢」。
所以这里第一条就是把「什么算素材」那份清单钉死。

第二类是**静默忽略参数**：`recall(..., budget_tokens=...)` 被收下就扔了，
调用方以为限额生效了，账单才知道没有。契约与实现不一致，而它不报错。
"""

from __future__ import annotations

from changqing import retrieve
from changqing.context import (
    MEMORY_SECTIONS,
    has_any_material,
    has_openable_topic,
    has_recall_material,
)
from changqing.runtime import Runtime, using
from changqing.store import append_op, append_summary, append_topic, materialize
from changqing.tokenize import warm as warm_tokenizer
from changqing.tokens import estimate_tokens

UID = "u" + "r" * 16


def add(fid: str, predicate: str, obj: str, **fields) -> None:
    op = {
        "op": "ADD",
        "id": fid,
        "subject": "他",
        "predicate": predicate,
        "object": obj,
        "quote": f"{predicate}{obj}",
        "turn_ref": "T-000001",
        "kind": "fact",
        "valid_from": "2026-09-01",
        "confidence": 0.9,
        "importance": 0.9,
        "persona_attention": 0.9,
    }
    op.update(fields)
    append_op(UID, op)


# ---------------------------------------------------------------- 门控
def test_the_gate_and_the_card_sections_read_one_list(rt: Runtime):
    """**同一句话只有一处权威定义。** 门控与卡片渲染必须读同一份字段清单。

    这里真踩过：两边各写一份，门控那份漏了 `promises` —— 只有「她答应过他的事」
    时，门控说没料，而卡片里明明写着那条。这一条逐个字段验一遍，抄第二份就会红。
    """
    assert has_recall_material({}) is False, "空记忆：没有素材"
    for key, _title, _when in MEMORY_SECTIONS:
        sess = {"memory": {key: ["有东西"]}}
        assert has_recall_material(sess) is True, f"{key} 也算素材（与卡片同一份清单）"


def test_a_story_alone_counts_as_material(rt: Runtime):
    """纪要**不住在会话的字段清单里**，所以那份清单结构上覆盖不到它。

    漏判的后果是「她看得见你们之间发生过什么，而门控说她没料」。
    """
    story_only = {"_mem": {"facts": [], "summaries": [{"id": "S-1", "text": "他说他有点怕黑。"}]}}
    assert has_recall_material(story_only) is True

    topics_only = {
        "_mem": {"facts": [], "summaries": [], "topics": [{"id": "P-1", "text": "问问他面试"}]}
    }
    assert has_recall_material(topics_only) is True, "话题也算 —— 它是她开口的理由"

    nothing = {"_mem": {"facts": [], "summaries": [], "topics": []}}
    assert has_recall_material(nothing) is False, "三样都空才是真没料（证明上面不是恒真）"


def test_the_opening_gate_asks_the_library_not_last_turns_worksheet(rt: Runtime):
    """两条开场探针都跑在检索**之前**，那时工作面上还是上一轮的东西。

    拿工作面当判据的结果是「她永远不开口」，而且看起来像功能坏了 ——
    工作面不落盘，进程重启就是空的。窄的那条（有没有可用话题）是**宿主可选项**：
    事实多、话题稀的号拿它当唯一的闸，等于把主动开口关掉。
    """
    assert has_any_material(UID) is False, "新用户：库里什么都没有"
    assert has_openable_topic(UID) is False
    add("F-0001", "养的猫叫", "团子")
    materialize(UID)
    assert has_any_material(UID) is True, "有事实就算有料"
    assert has_openable_topic(UID) is False, "但事实不是「能起个头的话题」"
    append_topic(UID, "问问他面试怎么样", "2026-09-01")
    assert has_openable_topic(UID) is True


def test_a_brand_new_person_gets_no_directories_created_for_asking(rt: Runtime, mem_root):
    """只是「问一句」不该顺手把目录与 schema 建出来。"""
    assert has_any_material("u" + "z" * 16) is False
    assert not (mem_root / "uz").exists(), "探针不该在盘上留下东西"


# ---------------------------------------------------------------- 预算
def test_recall_honours_the_budget_it_was_given(rt: Runtime):
    """`recall(..., budget_tokens=...)` 必须**真的**限制返回量。

    这个参数曾经被收下就扔了：裁剪用的永远是配置里那个默认值。契约与实现不一致，
    而且是「静默忽略」那种 —— 调用方以为限额生效了，账单才知道没有。
    所以这里三个方向都验：给小预算真的裁、不给用默认、给大预算照旧不裁。
    """
    for i, word in enumerate(("团子的细节甲", "团子的细节乙", "团子的细节丙", "团子的细节丁"), 1):
        # 四个**不同槽位**的事实：同槽位在读取侧只留一条，那样就测不出预算了
        add(f"F-97{i:02d}", word, word)
    materialize(UID)
    warm_tokenizer()
    query = "团子的细节甲"
    assert len(retrieve.search(UID, query)) == 4, "不带预算时四条都在（先确认召回本身没问题）"
    assert len(retrieve.search(UID, query, budget_tokens=1)) == 1, (
        "1 token 只装得下第一条（第一条豁免，见 _take_budget）"
    )
    assert len(retrieve.search(UID, query, budget_tokens=10**6)) == 4, "给大预算照旧不裁"


def test_a_fact_card_is_never_cut_in_half(rt: Runtime):
    """事实卡片是**原子**：预算再小也是整条进或整条不进。

    切半条的症状是「她记得你一半」——而被切掉的那半里往往正是关键的那个词。
    """
    long_fact = "他讲了很多关于他自己的事" * 6
    add("F-0001", long_fact, long_fact)
    materialize(UID)
    warm_tokenizer()
    out = retrieve.search(UID, long_fact, budget_tokens=1)
    assert len(out) == 1, "预算比一条还小时，第一条仍然完整地进（那条豁免）"
    assert out[0]["object"] == long_fact, "而且没有被截断"


# ---------------------------------------------------------------- 故事线
def test_the_story_line_is_a_constant_budget(rt: Runtime):
    """纪要从 3 条涨到 30 条，注入量不跟着涨 —— 否则它就是个越用越贵的洞。"""
    counts: list[int] = []
    used: list[int] = []
    for n in (3, 30):
        for i in range(n):
            append_summary(UID, f"第{i}场对话的小结，讲了他那天的一件小事。", "2026-09-11")
        cards = retrieve.hot_summaries(UID)
        counts.append(len(cards))
        used.append(sum(estimate_tokens(c["text"]) for c in cards))
    assert counts == [3, 3], f"两次都只注入 story_k 条（实际 {counts}）"
    assert used[1] <= used[0] * 2, f"注入量不随纪要条数增长（{used}）"


def test_a_single_long_summary_is_clipped_not_dropped(rt: Runtime):
    """`story_tokens` 是**硬上界**。原先的裁剪对第一条豁免，于是一条五百字的纪要
    能塞进三百多 token 而预算写着 160 —— 文档里那两行成本估算因此是假的。

    纪要是散文，少说一句可以；事实卡片是原子，不能切半条。所以两种裁剪不一样，
    这条钉住散文那一种。
    """
    long_text = "他讲了很多关于他自己的事。" * 40
    budget = rt.config.story_tokens
    assert estimate_tokens(long_text) > budget, "先证明这条真的超预算，否则下面全是空转"
    append_summary(UID, long_text, "2026-09-11")
    cards = retrieve.hot_summaries(UID)
    assert len(cards) == 1, "截断而不是丢：故事线还在"
    assert cards[0]["text"].endswith("…"), "截断处有标记"
    assert estimate_tokens(cards[0]["text"]) <= budget, "超了硬上界"


def test_the_story_line_can_be_switched_off(rt: Runtime):
    """`story_k=0` 必须等于「没有这段故事」—— 一行关掉。"""
    append_summary(UID, "他提到妈妈给他寄了腊肠。", "2026-09-02")
    assert retrieve.hot_summaries(UID), "先证明开着的时候确实取得到，否则这条在测「什么都没有」"
    off = rt.config.evolved(story_k=0)
    with using(Runtime(config=off)):
        assert retrieve.hot_summaries(UID) == []


def test_the_story_index_is_pinned_by_the_config(rt: Runtime):
    """纪要不带编号进卡片 —— 它不是可回引的事实，`S-` 那种 id 不该出现在里面。"""
    append_summary(UID, "他刚搬了家，说新房子朝南。", "2026-09-05")
    cards = retrieve.hot_summaries(UID)
    assert cards and cards[0]["day"] == "2026-09-05", "故事卡片带绝对日期"
    assert cards[0]["id"].startswith("S-"), "内部 id 仍留着（可审计），但不是给人看的编号"


# ---------------------------------------------------------------- 编排
def test_topics_are_only_read_on_the_turn_where_she_speaks_first(rt: Runtime):
    """普通那一轮手里不该握一张「你可以问他这个」的清单 —— 那会让对话变成查表。"""
    append_topic(UID, "问问他面试怎么样", "2026-09-01")
    normal = retrieve.retrieve_for_turn(UID, "在吗", {})
    assert normal["topics"] == [], "她答话那一轮不读话题"
    active = retrieve.retrieve_for_turn(UID, "", {}, proactive=True)
    assert active["topics"], "她先开口那一轮才读"


def test_the_used_token_count_covers_all_three_kinds_of_material(rt: Runtime):
    """`used_tokens` 是这一轮**实际注入**的量：事实 + 故事线 + 话题。

    只算事实的话，下一个人会拿它当「这一轮的注入量」，而另外两份凭空消失 ——
    成本估算与真实账单就此分家。
    """
    add("F-0001", "养的猫叫", "团子")
    materialize(UID)
    warm_tokenizer()
    append_summary(UID, "第一次聊到他的猫。", "2026-09-01")
    append_topic(UID, "问问他团子最近怎么样", "2026-09-01")
    ctx = retrieve.retrieve_for_turn(UID, "", {}, proactive=True)
    expected = (
        sum(estimate_tokens(retrieve.card_text(f)) for f in ctx["facts"])
        + sum(estimate_tokens(s["text"]) for s in ctx["summaries"])
        + sum(estimate_tokens(t["text"]) for t in ctx["topics"])
    )
    assert ctx["used_tokens"] == expected
    assert ctx["facts"] and ctx["summaries"] and ctx["topics"], "三样都要真的在场"


def test_a_broken_hot_path_is_counted_instead_of_swallowed(rt: Runtime, monkeypatch):
    """**吞掉不等于静默。** 热路径每次报错都要留一笔。

    少了这一句，「热路径因为内部错误一直是空的」在日志与统计里一个字都没有，
    表现只是「她记性不太好」—— 而那是全套设计里最贵的一种症状。
    """
    before = retrieve.stats().get("error", 0)

    def boom(*_a, **_kw):
        raise RuntimeError("内部坏了")

    monkeypatch.setattr(retrieve, "hot_facts", boom)
    ctx = retrieve.retrieve_for_turn(UID, "在吗", {})
    assert ctx["facts"] == [], "坏了就是空的"
    assert retrieve.stats().get("error", 0) == before + 1, "但要留一笔"
    assert "内部坏了" in retrieve.last_error(), "而且要看得出是什么坏了"


def test_a_new_person_never_runs_the_cold_path(rt: Runtime):
    """库里一条事实都没有时不该去查 —— 冷路径的价值是「挑出相关的那几条」，
    没有候选可挑的时候它只是白花一次查询的时间。"""
    retrieve.STATS["search"] = 0
    ctx = retrieve.retrieve_for_turn(UID, "我家猫叫什么", {})
    assert retrieve.STATS["search"] == 0, "一次都没查"
    assert {k: ctx[k] for k in ("facts", "hot", "cold", "used_tokens", "summaries", "topics")} == {
        "facts": [],
        "hot": 0,
        "cold": 0,
        "used_tokens": 0,
        "summaries": [],
        "topics": [],
    }
    assert ctx["enabled"] is True, "系统开着，只是没料可给"
