"""向量索引的写入侧：幂等、可换模型、坏了不影响说话。

这几条守的都是**静默**的坏法：索引没跟上时检索只是差一点，而统计那几行
看起来一切正常。
"""

from __future__ import annotations

from changqing.adapters.mock import MockEmbedder
from changqing.runtime import Runtime, using
from changqing.store import append_op, materialize, open_index
from changqing.vectors import LIVE_STATUS, reindex

UID = "u" + "v" * 16


def add(uid: str, fid: str, pred: str, obj: str) -> None:
    append_op(
        uid,
        {
            "op": "ADD",
            "id": fid,
            "subject": "他",
            "predicate": pred,
            "object": obj,
            "quote": f"{pred}{obj}",
            "turn_ref": "T-000001",
            "confidence": 0.9,
            "importance": 0.5,
            "persona_attention": 0.5,
            "kind": "fact",
            "valid_from": "2026-09-01",
        },
    )


def vec_count(uid: str) -> int:
    con = open_index(uid)
    try:
        return int(con.execute("SELECT COUNT(*) AS n FROM vectors").fetchone()["n"])
    finally:
        con.close()


def vector_models(uid: str) -> list[str]:
    con = open_index(uid)
    try:
        return sorted(
            str(r["model"] or "")
            for r in con.execute("SELECT DISTINCT model FROM vectors").fetchall()
        )
    finally:
        con.close()


def with_embedder(rt: Runtime, emb):
    return using(Runtime(config=rt.config, embedder=emb, llm=rt.llm, usage=rt.usage))


# ---------------------------------------------------------------- 关掉这条路
def test_no_embedder_means_the_whole_layer_is_off(rt: Runtime):
    """向量是可选增强：不注入就等于关掉，调用方不必到处写 `if`。"""
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    out = reindex(UID)  # fixture 里是 NullEmbedder
    assert out == {"ok": True, "skipped": "disabled", "encoded": 0, "removed": 0}
    assert vec_count(UID) == 0


def test_a_model_that_is_not_ready_yet_is_a_reason_not_a_crash(rt: Runtime):
    """**不许在这里加载模型**：这是后台整理线程，一次加载最坏几十秒，
    卡住的不是一个用户的首字、而是整条整理队列。
    """
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    emb = MockEmbedder()
    emb.ready = lambda download=True: False  # type: ignore[method-assign]

    class NotReady(MockEmbedder):
        def ready(self, download: bool = True) -> bool:
            return False

    out = with_embedder(rt, NotReady())
    with out:
        res = reindex(UID)
    assert res["ok"] is False
    assert res["reason"] == "embed_unavailable", "调用方按 reason 决定「先欠着」"
    assert vec_count(UID) == 0


# ---------------------------------------------------------------- 幂等
def test_reindex_is_idempotent(rt: Runtime):
    """派生层重跑一次结果必须一样 —— 判据是 (fact_id, dim, text_hash)。"""
    add(UID, "F-1", "养的猫叫", "团子")
    add(UID, "F-2", "不吃", "香菜")
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        assert reindex(UID)["encoded"] == 2
        again = reindex(UID)
    assert again["encoded"] == 0, "内容没变就跳过"
    assert again["pending"] == 0
    assert vec_count(UID) == 2


def test_a_changed_fact_is_re_encoded(rt: Runtime):
    """事实改了内容、向量还留着旧的，检索就会按**旧内容**召回 ——
    用户看到的是「她记着我三年前说过的话」。
    """
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        reindex(UID)
        append_op(UID, {"op": "EDIT", "id": "F-1", "set": {"object": "团子（三岁）"}})
        materialize(UID)
        out = reindex(UID)
    assert out["encoded"] == 1, "内容变了就得重编"
    assert vec_count(UID) == 1, "而且是原地更新，不是插第二条"


def test_facts_that_died_leave_the_index(rt: Runtime):
    """维度对不上或事实不在世上的先清掉：留着不只是占地方，混维度的行
    会被 `WHERE dim=?` 读不到，检索静默少一半。
    """
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        reindex(UID)
        append_op(UID, {"op": "INVALIDATE", "id": "F-1", "reason": "猫送人了"})
        materialize(UID)
        out = reindex(UID)
    assert out["removed"] == 1
    assert vec_count(UID) == 0


def test_pending_facts_are_encoded_too(rt: Runtime):
    """pending 也要编：用户点了「确认」之后它就是 active，那时才补编码的话，
    面板上刚确认的记忆当轮还搜不到。
    """
    assert "pending" in LIVE_STATUS
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    append_op(UID, {"op": "CONFIRM", "id": "F-1", "status": "pending"})
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        assert reindex(UID)["encoded"] == 1


# ---------------------------------------------------------------- 换模型
class OtherModel(MockEmbedder):
    """同一个维度、另一个编码器 —— 这就是「同维度换模型」那个坑的形状。"""

    name = "mock-other"
    repo = "changqing/mock-ngram-v2"


def test_switching_model_wipes_and_rebuilds(rt: Runtime):
    """换一个**同维度**的模型时，旧向量在新模型的查询向量下相似度全是噪声，
    而库里混着两个向量空间这件事**没有任何报错**。
    """
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        reindex(UID)
    assert vector_models(UID) == ["changqing/mock-ngram-v1"]

    with with_embedder(rt, OtherModel()):
        out = reindex(UID)
    assert out["encoded"] == 1, "整批重编"
    assert out["removed"] == 1, "旧空间的那条被删了"
    assert vector_models(UID) == ["changqing/mock-ngram-v2"], "表里只剩一个向量空间"


def test_the_dimension_is_recorded_per_row(rt: Runtime):
    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    with with_embedder(rt, MockEmbedder(dim=32)):
        out = reindex(UID)
    assert out["dim"] == 32
    con = open_index(UID)
    try:
        row = con.execute("SELECT dim, length(vec) AS n FROM vectors").fetchone()
    finally:
        con.close()
    assert row["dim"] == 32, "维度按**实际返回的向量**记，不是按配置"
    assert row["n"] == 32 * 4, "float32 小端：4 字节一个分量"


# ---------------------------------------------------------------- 坏了不抛
def test_a_failing_encoder_is_reported_not_raised(rt: Runtime):
    """模型没加载好、推理报错都只是「这轮没有向量可召回」，全文检索还在。"""

    class Broken(MockEmbedder):
        def encode(self, texts):
            return None

    add(UID, "F-1", "养的猫叫", "团子")
    materialize(UID)
    with with_embedder(rt, Broken()):
        out = reindex(UID)
    assert out["ok"] is False
    assert out["reason"] == "encode_failed"
    assert vec_count(UID) == 0


def test_limit_is_only_for_self_checks(rt: Runtime):
    """`limit` 一次只处理 N 条：自检要能抽一小批看行为。"""
    for i in range(3):
        add(UID, f"F-{i}", "喜欢", f"第{i}件")
    materialize(UID)
    with with_embedder(rt, MockEmbedder()):
        first = reindex(UID, limit=2)
        assert first["encoded"] == 2
        assert first["pending"] == 0, "pending 数的是「这一趟没编完的」"
        second = reindex(UID)
    assert second["encoded"] == 1, "下一趟把剩下的补上"
