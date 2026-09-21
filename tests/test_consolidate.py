"""巩固：重复合并成一条，而不是越攒越多。

这些用例守的是「她合并了两条记忆」与「她丢了一条记忆」的区别 —— 前者可以
从日志重放，后者不可逆。断言写得比通常啰嗦，是因为这里错了之后症状只是
「她记性有点乱」，不报错、不进任何指标。
"""

from __future__ import annotations

import time

from changqing.consolidate import consolidate, find_duplicates
from changqing.runtime import Runtime
from changqing.store import (
    append_op,
    get_fact,
    list_facts,
    materialize,
    open_index,
    read_ops,
)

UID = "u" + "4" * 16
OLD = time.time() - 400 * 86400  # 一年多以前


def add(uid: str, fid: str, pred: str, obj: str, **kw) -> None:
    """往某个 uid 追加一条 ADD（valid_from 是「一年多以前」）。"""
    op = {
        "op": "ADD",
        "id": fid,
        "subject": "他",
        "predicate": pred,
        "object": obj,
        "valid_from": time.strftime("%Y-%m-%d", time.localtime(OLD)),
        "confidence": 0.9,
        "importance": 0.5,
        "persona_attention": 0.5,
        "kind": "fact",
        "turn_ref": "T-000001",
        "quote": obj,
    }
    op.update(kw)
    append_op(uid, op)


def test_duplicates_merge_into_the_richest_one(rt: Runtime):
    """合并同槽位重复 —— 留信息量最大的，被并的行剪掉。

    「可审计」不是靠留一行派生数据，而是靠那两样不可再生的：
    MERGE 操作在日志里，原话在 L0 里。
    """
    add(UID, "F-0001", "养的猫叫", "团子")
    add(UID, "F-0002", "养的猫叫", "团子。")  # 只有标点不同
    add(UID, "F-0003", "养的猫叫", "团子（三岁）")  # 更详细，应该它胜出
    add(UID, "F-0004", "不吃", "香菜")
    materialize(UID)
    assert len(list_facts(UID)) == 4, "合并前 4 条"

    groups = find_duplicates(UID)
    assert len(groups) == 1, "找到 1 组重复"
    assert len(groups[0]) == 3, "这一组 3 条"

    stats = consolidate(UID)
    assert stats["ok"] is True, "巩固成功"
    assert stats["merged"] == 2, "合并了 2 条"

    live = list_facts(UID)
    assert len(live) == 2, "当前有效剩 2 条（猫 + 香菜）"
    cat = next(f for f in live if f["predicate"] == "养的猫叫")
    assert cat["object"] == "团子（三岁）", "留下的是信息量最大的那条"
    assert stats["pruned"] == 2, "被并的行剪掉了（派生行，留久了会线性膨胀）"

    merges = [o for o in read_ops(UID) if o["op"] == "MERGE"]
    assert len(merges) == 2, "MERGE 操作留在日志里"
    assert all(o.get("merged_into") == "F-0003" for o in merges), "日志里记了并到哪一条"


def test_winner_ignores_punctuation(rt: Runtime):
    """选胜者时标点不该赢（内容量比字符数重要）。"""
    w = "u" + "5" * 16
    for fid, obj in (("F-1", "香菜"), ("F-2", "香菜。")):
        add(w, fid, "不吃", obj)
    materialize(w)
    consolidate(w)
    assert [f["object"] for f in list_facts(w)] == ["香菜"], "留下的不带多余标点"

    # 但内容确实更多时，带括号的那条该赢 —— 不能因为要治标点就矫枉过正
    w2 = "u" + "6" * 16
    for fid, obj in (("F-1", "团子"), ("F-2", "团子（三岁）")):
        add(w2, fid, "养的猫叫", obj)
    materialize(w2)
    consolidate(w2)
    assert [f["object"] for f in list_facts(w2)] == ["团子（三岁）"], "内容更多的仍然赢"


def test_similar_values_in_different_slots_never_merge(rt: Runtime):
    """槽位不同就不算重复 —— 谓词是「同一件事」的判据，不是值本身。"""
    w = "u" + "7" * 16
    add(w, "F-1", "养的猫叫", "团子")
    add(w, "F-2", "养的狗叫", "团子")
    materialize(w)
    assert find_duplicates(w) == [], "值一样但槽位不同"
    consolidate(w)
    assert len(list_facts(w)) == 2, "两条都还在"


def test_different_values_are_not_swallowed(rt: Runtime):
    """「不吃香菜」与「不吃芹菜」只差一个字，但恰好是决定意思的那个 —— 不能并。

    这是覆盖率判据最危险的一面：短的那条被长的那条完全覆盖也判 1.0 时，
    这类「包含但不同」会被悄悄吞掉一条。
    """
    w = "u" + "8" * 16
    add(w, "F-1", "不吃", "香菜")
    add(w, "F-2", "不吃", "芹菜")
    materialize(w)
    consolidate(w)
    assert sorted(f["object"] for f in list_facts(w)) == ["芹菜", "香菜"], "两条都得留下"


def test_dry_run_changes_nothing(rt: Runtime):
    """dry_run 回答的是「现在跑一轮会动什么」，它自己一个字都不许写。"""
    w = "u" + "9" * 16
    add(w, "F-1", "养的猫叫", "团子")
    add(w, "F-2", "养的猫叫", "团子（三岁）")
    materialize(w)
    stats = consolidate(w, dry_run=True)
    assert stats["merged"] == 1, "算出来了"
    assert stats["ops"] >= 1, "也报了会追加几条操作"
    assert stats["swept"] == 0, "但索引一条都没动"
    assert stats["pruned"] == 0, "一行都没剪"
    assert len(list_facts(w)) == 2, "库里还是两条"
    assert all(o["op"] != "MERGE" for o in read_ops(w)), "一条 MERGE 都没写"


# ---------------------------------------------------------------- 衰减
TODAY = time.strftime("%Y-%m-%d")


def test_stale_facts_lose_weight_but_are_not_deleted(rt: Runtime):
    """一年多没用过、又不重要的：权重压一档，但**不删**。

    降权是「她该多快想起这件事」的旋钮，不是「这件事还在不在」的开关 ——
    压下去的仍然查得到，只是不再抢热路径的位置。
    """
    w = "u" + "a" * 16
    add(w, "F-1", "喜欢的乐队", "慢慢说", importance=0.8)
    materialize(w)
    stats = consolidate(w)
    assert stats["derogated"] == 1, "压了一条"
    assert stats["kept"] == 0, "没有别的可保留"
    assert get_fact(w, "F-1")["importance"] == 0.56, "0.8 * 0.7"
    assert len(list_facts(w)) == 1, "事实还在"
    derogations = [o for o in read_ops(w) if o["op"] == "DEROGATE"]
    assert len(derogations) == 1, "降权也是追加的操作，不是改库"
    assert derogations[0]["id"] == "F-1"


def test_decay_stops_at_the_floor(rt: Runtime):
    """压到地板就停 —— 否则同一条事实每巩固一次就再写一条 DEROGATE。"""
    w = "u" + "b" * 16
    add(w, "F-1", "喜欢的乐队", "慢慢说", importance=0.05)
    materialize(w)
    stats = consolidate(w)
    assert stats["derogated"] == 0, "已经在地板上"
    assert stats["kept"] == 1, "算它一条保住了"
    assert get_fact(w, "F-1")["importance"] == 0.05
    assert all(o["op"] != "DEROGATE" for o in read_ops(w)), "一条 DEROGATE 都没写"


def test_recent_facts_are_left_alone(rt: Runtime):
    """刚说的事不降权 —— 判据是「多久没用过」，不是「重不重要」。"""
    w = "u" + "c" * 16
    add(w, "F-1", "喜欢的乐队", "慢慢说", importance=0.8, valid_from=TODAY)
    materialize(w)
    stats = consolidate(w)
    assert stats["derogated"] == 0
    assert stats["kept"] == 1
    assert get_fact(w, "F-1")["importance"] == 0.8, "权重一个字没动"


def test_pinned_and_promises_never_decay(rt: Runtime):
    """钉住的、以及承诺类的永不降权。

    承诺是「她欠他的一件事」，与「她记不记得」无关；拿排序权重去压一句还没
    兑现的承诺，等于让它悄悄沉底 —— 那正是用户能察觉到的「她忘了自己说过」。
    """
    w = "u" + "d" * 16
    add(w, "F-1", "喜欢的乐队", "慢慢说", importance=0.8, pinned=1)
    add(w, "F-2", "答应", "周末带他去看展", importance=0.8, kind="promise")
    add(w, "F-3", "约好", "国庆一起去爬山", importance=0.8, kind="commitment")
    materialize(w)
    stats = consolidate(w)
    assert stats["derogated"] == 0, "一条都没压"
    assert stats["kept"] == 3
    assert all(get_fact(w, fid)["importance"] == 0.8 for fid in ("F-1", "F-2", "F-3"))


def test_invalidated_facts_leave_the_full_text_index(rt: Runtime):
    """失效的行从全文索引里摘掉 —— 数据还在，但检索不该再命中一个作废的值。

    「她想起一件早就作废的事」不报任何错，也不进任何指标，所以这里断言的是
    索引那一列真的被清空了，而不只是返回值里那个数字。
    """
    w = "u" + "e" * 16
    add(w, "F-1", "养的猫叫", "团子")
    materialize(w)
    append_op(w, {"op": "INVALIDATE", "id": "F-1", "reason": "猫送人了"})
    materialize(w)
    con = open_index(w)
    try:
        before = con.execute("SELECT text_index FROM facts WHERE id='F-1'").fetchone()
    finally:
        con.close()
    assert before["text_index"], "失效之前索引里是有词的"

    stats = consolidate(w)
    assert stats["swept"] == 1, "摘了一条"
    con = open_index(w)
    try:
        after = con.execute("SELECT text_index FROM facts WHERE id='F-1'").fetchone()
    finally:
        con.close()
    assert after is not None, "行还在（摘索引不等于删数据）"
    assert not after["text_index"], "索引那一列空了"
