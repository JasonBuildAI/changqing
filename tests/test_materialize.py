"""L1：操作日志 → 物化视图。**只有日志不可再生**，视图是它的投影。

这套设计的可执行版本是这一句话：把 `index.sqlite` 删掉，也应该从 `log.jsonl`
完整恢复，一条不差、包括全文索引。所以这里逐条钉住：

  · 重放幂等（同一段日志应用两次结果一样）；
  · 删掉索引文件能重建回来；
  · 换事实是**失效旧的那条**而不是删它（双时间，可审计）；
  · 全文索引跟着 UPDATE 走 —— external content 表**不会自己同步**；
  · 日志尾部那半行不许让整份日志打不开（那等于这个人的记忆全部消失）。

最后一条最容易漏：`read_ops` 一旦抛异常，`materialize` 就一条都应用不了，
而它的表现只是「她记性突然变差」，没有任何一处报错。
"""

from __future__ import annotations

import json

from changqing.runtime import Runtime
from changqing.store import (
    append_op,
    ensure_schema,
    fact_stats,
    facts_md_path,
    get_fact,
    index_path,
    list_facts,
    log_path,
    materialize,
    open_index,
    read_ops,
    rebuild,
    render_facts_md,
)

UID = "u" + "f" * 16


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
    }
    op.update(fields)
    append_op(UID, op)


def search_index(word: str) -> int:
    con = open_index(UID)
    try:
        ensure_schema(con)
        return int(
            con.execute(
                "SELECT count(*) FROM facts_fts WHERE facts_fts MATCH ?", (f'"{word}"',)
            ).fetchone()[0]
        )
    finally:
        con.close()


# ---------------------------------------------------------------- 日志 → 视图
def test_the_log_materializes_into_facts(rt: Runtime):
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    assert len(read_ops(UID)) == 2
    assert materialize(UID)["applied"] == 2, "应用条数"
    assert materialize(UID)["applied"] == 0, "再物化一次不重复应用"

    live = list_facts(UID)
    assert len(live) == 2
    assert (live[0]["status"], bool(live[0]["valid_from"])) == ("active", True), (
        "默认字段有值，不能是 NULL"
    )
    for line in log_path(UID).read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["op"], "日志是 JSON Lines，一行一条"


def test_the_full_text_index_does_not_sync_itself(rt: Runtime):
    """external content 表**不会自己同步**：没有触发器的话索引是空的，
    于是「按词召回」这条路静默返回空 —— 她答不出「我家猫叫什么」。"""
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)
    assert search_index("团子") == 1, "中文能搜到（分词真的生效了）"


def test_the_full_text_index_follows_an_edit(rt: Runtime):
    """UPDATE 触发器那一条：改过之后旧词必须查不到。

    只测「新词查得到」是不够的 —— 索引里同时留着新旧两份的话，新词照样查得到，
    旧词却会把这条事实重新捞回来，于是她按一句早就改掉的话记得你。
    """
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)
    append_op(UID, {"op": "EDIT", "id": "F-0002", "set": {"object": "芹菜"}})
    materialize(UID)
    assert search_index("香菜") == 0, "改过之后旧词查不到"
    assert search_index("芹菜") == 1, "新词查得到"


# ---------------------------------------------------------------- 可重建
def test_the_index_can_be_rebuilt_from_the_log(rt: Runtime):
    """「只有日志不可再生」的可执行版本。"""
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)
    before = [(f["id"], f["object"]) for f in list_facts(UID)]
    ops = len(read_ops(UID))

    assert rebuild(UID) == ops, "重建重放了全部操作"
    assert [(f["id"], f["object"]) for f in list_facts(UID)] == before

    # 真的把文件删掉，走一遍「索引损坏」那条路
    index_path(UID).unlink()
    assert materialize(UID, force=True)["applied"] == ops, "从日志重建回来"
    assert [(f["id"], f["object"]) for f in list_facts(UID)] == before


def test_replaying_the_same_log_twice_changes_nothing(rt: Runtime):
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    append_op(UID, {"op": "EDIT", "id": "F-0002", "set": {"object": "芹菜"}})
    materialize(UID)
    snap = [(f["id"], f["object"], f["status"]) for f in list_facts(UID, include_dead=True)]
    materialize(UID, force=True)
    assert [(f["id"], f["object"], f["status"]) for f in list_facts(UID, include_dead=True)] == snap


# ---------------------------------------------------------------- 双时间
def test_a_superseded_fact_is_invalidated_not_deleted(rt: Runtime):
    """换工作那种：旧事实**失效而不是删除**。她不该再说「你是老师」，
    但「他以前是老师」这句话必须仍然查得到 —— 那是可审计的过去。"""
    add("F-0010", "的职业是", "老师")
    materialize(UID)
    append_op(
        UID,
        {
            "op": "SUPERSEDE",
            "id": "F-0011",
            "replaces": "F-0010",
            "subject": "他",
            "predicate": "的职业是",
            "object": "插画师",
            "valid_from": "2026-10-01",
            "confidence": 0.9,
            "importance": 0.6,
            "persona_attention": 0.4,
            "turn_ref": "T-000042",
        },
    )
    materialize(UID)

    old = get_fact(UID, "F-0010")
    assert old["status"] == "superseded"
    assert bool(old["valid_to"]), "旧事实填了 valid_to"
    assert old["object"] == "老师", "行还在、内容还在"

    live = [f["object"] for f in list_facts(UID)]
    assert "老师" not in live, "当前有效里只剩新的"
    assert "插画师" in live


def test_pin_edit_and_both_kinds_of_forget(rt: Runtime):
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)

    append_op(UID, {"op": "PIN", "id": "F-0001", "pinned": 1})
    materialize(UID)
    assert get_fact(UID, "F-0001")["pinned"] == 1

    append_op(
        UID, {"op": "EDIT", "id": "F-0001", "set": {"importance": 0.95, "object": "团子（三岁）"}}
    )
    materialize(UID)
    f = get_fact(UID, "F-0001")
    assert (f["object"], f["importance"]) == ("团子（三岁）", 0.95)

    append_op(UID, {"op": "FORGET", "id": "F-0001", "mode": "archive"})
    materialize(UID)
    assert get_fact(UID, "F-0001")["status"] == "forgotten", "归档：状态变了但行还在"
    assert not any(x["id"] == "F-0001" for x in list_facts(UID)), "不再出现在当前有效里"

    append_op(UID, {"op": "FORGET", "id": "F-0002", "mode": "purge"})
    materialize(UID)
    assert get_fact(UID, "F-0002") is None, "彻底删除：行没了"


# ---------------------------------------------------------------- 残行
def test_a_torn_log_tail_does_not_take_the_whole_log_down(rt: Runtime):
    """写日志的进程被杀在 write 中间，只丢那半行。

    `read_ops` 一旦抛异常，`materialize` 就**一条都应用不了** ——
    一个用户的整份记忆从这里开始全部消失，而表现只是「她记性突然变差」。
    """
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)
    n_before = len(read_ops(UID))

    with open(log_path(UID), "a", encoding="utf-8", newline="\n") as f:
        f.write('{"op": "ADD", "id": "F-0003", "subject": "他", "pred')

    assert len(read_ops(UID)) == n_before, "残行被跳过"
    assert materialize(UID)["applied"] == 0, "它没有冒充成一条操作"
    assert len(list_facts(UID)) == 2, "已经有的事实一条没少"

    add("F-0004", "喜欢", "画画")
    assert materialize(UID)["applied"] == 1, "坏行之后照样能继续应用"


# ---------------------------------------------------------------- 渲染视图
def test_the_rendered_view_is_read_only_but_readable(rt: Runtime):
    """渲染出来的 md 是给人看的**只读产物**。手工改它会被下次渲染覆盖 ——
    要改就得走编辑接口，由它往日志追加一条操作。"""
    add("F-0001", "养的猫叫", "团子")
    materialize(UID)
    path = render_facts_md(UID)
    assert path is not None and path.exists()
    body = path.read_text(encoding="utf-8")
    assert "只读" in body and "不要手工编辑" in body, "产物自己写明它只读"
    assert "他养的猫叫团子" in body, "事实有它自己的小标题"
    assert "#F-0001" in body and "T-000001" in body, "编号与出处都带上了"
    assert "团子" in body


def test_the_rendered_view_keeps_the_dead_in_their_own_section(rt: Runtime):
    """失效的事实不能从人读的那份视图里消失 —— 否则「他以前是老师」这件事
    在盘上就只剩数据库里那一行，人翻文件永远看不到。"""
    add("F-0010", "的职业是", "老师")
    materialize(UID)
    append_op(
        UID,
        {
            "op": "SUPERSEDE",
            "id": "F-0011",
            "replaces": "F-0010",
            "subject": "他",
            "predicate": "的职业是",
            "object": "插画师",
            "valid_from": "2026-10-01",
        },
    )
    materialize(UID)
    body = render_facts_md(UID).read_text(encoding="utf-8")
    assert "已经失效的" in body, "失效的那一段要有自己的小标题"
    assert "老师" in body, "失效的事实还在视图里"


def test_the_rendered_view_says_so_when_there_is_nothing_yet(rt: Runtime):
    """空库也要写一份 —— 一份空文件与「还没有事实」是两件事。"""
    materialize(UID)
    body = render_facts_md(UID).read_text(encoding="utf-8")
    assert "还没有事实" in body
    assert facts_md_path(UID).exists()


# ---------------------------------------------------------------- 计数
def test_the_counts_tell_live_pending_and_dead_apart(rt: Runtime):
    add("F-0001", "养的猫叫", "团子")
    add("F-0002", "不吃", "香菜")
    materialize(UID)
    append_op(UID, {"op": "FORGET", "id": "F-0002", "mode": "archive"})
    materialize(UID)
    st = fact_stats(UID)
    assert st["live"] == 1, "归档的那条不算活着"
    assert st["total"] >= 2, "但它仍在总数里 —— 「她记过什么」不该被抹掉"
