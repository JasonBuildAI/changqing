"""操作日志 → 物化视图。每条 op 都要真的改到东西，坏 op 要被点名。"""

from __future__ import annotations

from changqing.runtime import Runtime
from changqing.store import append_op, get_fact, list_facts, materialize, read_ops

UID = "u" + "o" * 16


def add(uid: str, fid: str, **fields) -> None:
    op = {
        "op": "ADD",
        "id": fid,
        "subject": "他",
        "predicate": "养的猫叫",
        "object": "团子",
        "quote": "我家猫叫团子",
        "turn_ref": "T-000001",
        "kind": "fact",
        "valid_from": "2026-09-01",
    }
    op.update(fields)
    append_op(uid, op)


def test_edit_actually_changes_the_fact(rt: Runtime):
    """「编辑一条记忆」必须真的改到。

    这一条守的是一个具体踩过的坑：`sqlite3.Row` 迭代出来的是**值**不是列名，
    写成 `{k: cur[k] for k in cur}` 会拿一个值当索引去查、抛 `IndexError`，
    而 `materialize` 把坏 op 隔离进 `bad_ops` 就继续走 —— 于是编辑被丢掉，
    调用方那边看起来一切正常。
    """
    add(UID, "F-1")
    assert materialize(UID)["bad_ops"] == []
    append_op(UID, {"op": "EDIT", "id": "F-1", "set": {"object": "团子（三岁）"}})
    out = materialize(UID)
    assert out["bad_ops"] == [], "EDIT 不该被当成坏 op"
    assert get_fact(UID, "F-1")["object"] == "团子（三岁）"
    assert get_fact(UID, "F-1")["predicate"] == "养的猫叫", "没点名的字段原样留着"


def test_edit_only_touches_known_fields(rt: Runtime):
    """`set` 里混进来的野字段不该进 SQL（那条 SQL 是拼的列名）。"""
    add(UID, "F-1")
    materialize(UID)
    append_op(
        UID, {"op": "EDIT", "id": "F-1", "set": {"object": "团子", "; DROP TABLE facts --": "x"}}
    )
    assert materialize(UID)["bad_ops"] == []
    assert get_fact(UID, "F-1")["object"] == "团子"


def test_edit_of_a_missing_fact_is_a_noop(rt: Runtime):
    append_op(UID, {"op": "EDIT", "id": "F-404", "set": {"object": "x"}})
    assert materialize(UID)["bad_ops"] == []
    assert list_facts(UID) == []


def test_a_bad_op_is_named_and_does_not_block_the_rest(rt: Runtime):
    """**一条坏操作不许卡死整个用户**：以前是一条 op 抛异常就整批回滚，
    于是同一个坏 op 每次重试都卡在同一位置，后面的永远不应用 ——
    症状是「她记性变差」，而没有任何报错。
    """
    add(UID, "F-1")
    materialize(UID)
    # 绑一个 sqlite 绑不了的值：这才是「真的会失败」的那种坏 op
    append_op(UID, {"op": "EDIT", "id": "F-1", "set": {"object": {"不是": "字符串"}}})
    add(UID, "F-2", object="香菜")
    out = materialize(UID)
    assert len(out["bad_ops"]) == 1, "坏的那条被点名"
    assert out["bad_ops"][0]["id"] == "F-1"
    assert get_fact(UID, "F-2")["object"] == "香菜", "它后面的那条照样应用了"


def test_materialize_does_not_reapply_the_whole_log(rt: Runtime):
    """物化是增量的：重跑只应用新来的那几条（`applied_ops` 水位）。"""
    add(UID, "F-1")
    assert materialize(UID)["applied"] == 1
    assert materialize(UID)["applied"] == 0, "已经应用过的不再重放"
    add(UID, "F-2")
    assert materialize(UID)["applied"] == 1


def test_edit_is_append_only(rt: Runtime):
    """改不是改，是**再追加一条操作** —— 日志是唯一事实源，可重放。"""
    add(UID, "F-1")
    materialize(UID)
    append_op(UID, {"op": "EDIT", "id": "F-1", "set": {"object": "团子（三岁）"}})
    materialize(UID)
    assert [o["op"] for o in read_ops(UID)] == ["ADD", "EDIT"]
