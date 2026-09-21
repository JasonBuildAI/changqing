"""可挂载路由：相对路径、身份只从服务端来、坏值在动日志之前挡住。

这一层最贵的缺陷是**越权**：`/memory?uid=xxx` 那种写法，把参数改一个字符就能
读别人的记忆。它不会报错、不会变红，只是安静地把每个人的对话摆到所有人面前。
所以这里用「整张路由表逐行钉成常量」加「没有一条接口收 uid」两道来守。

需要 `changqing[server]`（FastAPI + httpx）。没装就整份跳过 —— 跳过是**说得出
理由**的那种：这一层的能力本来就不在核心包里。
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from changqing import Memory
from changqing.adapters.mock import MockEmbedder, MockLLM
from changqing.runtime import Runtime
from changqing.server import create_router

UID = "u" + "s" * 16
OTHER = "u" + "o" * 16


def facts_json(*facts: dict) -> str:
    return json.dumps(
        {
            "facts": [{"subject": "他", "kind": "fact", "confidence": 0.9, **f} for f in facts],
            "summary": "",
            "topics": [],
        },
        ensure_ascii=False,
    )


def seeded(rt: Runtime) -> Memory:
    """一个已经有一条事实的句柄（顺手把向量也建起来，好断言派生层跟着动）。"""
    mem = Memory(
        UID,
        config=rt.config,
        embedder=MockEmbedder(),
        llm=MockLLM(
            [
                facts_json(
                    {
                        "predicate": "养的猫叫",
                        "object": "团子",
                        "quote": "我家猫叫团子",
                        "turn_ref": "T-000001",
                    }
                )
            ]
        ),
    )
    mem.add("我家猫叫团子")
    mem.extract_now()
    return mem


def client(rt: Runtime, *, prefix: str = "", uid: str = UID) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_router(
            uid_of=lambda _request: uid,
            memory_of=lambda who: Memory(who, config=rt.config, embedder=MockEmbedder()),
        ),
        prefix=prefix,
    )
    return TestClient(app)


# ---------------------------------------------------------------- 路由表
def test_the_router_paths_are_pinned():
    """整张表逐行钉成常量。多一条接口就变红 —— 一个顺手加上的调试接口
    （「导出全部用户的记忆」之类）能一路悄悄发到线上，就是缺了这条。
    """
    from changqing.server import create_router as make

    paths = sorted(
        (r.path, sorted(r.methods))  # type: ignore[attr-defined]
        for r in make().routes
    )
    assert paths == [
        ("/memory", ["GET"]),
        ("/memory/confirm", ["POST"]),
        ("/memory/edit", ["POST"]),
        ("/memory/export", ["GET"]),
        ("/memory/forget", ["POST"]),
        ("/memory/pin", ["POST"]),
    ]


def test_every_path_is_relative_so_the_host_owns_the_prefix(rt: Runtime):
    """路径不带前缀：写死了就得让每个宿主去挪自己的路由表，而「挪一下」
    在别人的代码里通常等于不改 —— 于是要么冲突、要么挂到一个谁也没想到的地址上。
    """
    with client(rt, prefix="/api") as c:
        assert c.get("/api/memory").status_code == 200
        assert c.get("/memory").status_code == 404, "没有宿主前缀时不该在这里"


def test_no_endpoint_accepts_a_uid_from_the_client():
    """**越权那道闸。** 逐条看签名里有没有 uid 参数，再看请求体里有没有 uid 字段。

    这条测试的价值在于它会**真的失败**：谁哪天为了「方便调试」加一个
    `uid: str = Query("")`，这里立刻红，而不是等到有人发现能读别人的记忆。
    """
    from changqing.server import create_router as make
    from changqing.server import router as router_mod

    for route in make().routes:
        params = getattr(route, "dependant", None)
        names = {p.name for p in (getattr(params, "query_params", None) or [])}
        names |= {p.name for p in (getattr(params, "body_params", None) or [])}
        assert "uid" not in names, f"{route.path} 收了一个 uid 参数"

    for model in (
        router_mod.MemoryEditReq,
        router_mod.MemoryForgetReq,
        router_mod.MemoryConfirmReq,
        router_mod.MemoryPinReq,
    ):
        assert "uid" not in model.model_fields, f"{model.__name__} 收了一个 uid 字段"


def test_a_request_without_an_identity_is_refused_not_defaulted(rt: Runtime):
    """取不到身份就 401，**绝不退回空串或默认用户** —— 退回默认用户的症状是
    「所有人都看同一个人的记忆」，而它不报任何错。
    """
    app = FastAPI()
    app.include_router(create_router(memory_of=lambda who: Memory(who, config=rt.config)))
    with TestClient(app) as c:
        assert c.get("/memory").status_code == 401
        assert c.get("/memory/export").status_code == 401


def test_the_default_rule_reads_the_identity_the_middleware_wrote(rt: Runtime):
    """出厂实现读 `request.state.uid` —— 那是「服务端签发」在 FastAPI 里的落点。"""
    app = FastAPI()

    @app.middleware("http")
    async def who(request: Request, call_next):
        request.state.uid = UID
        return await call_next(request)

    app.include_router(create_router(memory_of=lambda who: Memory(who, config=rt.config)))
    with TestClient(app) as c:
        assert c.get("/memory").status_code == 200


# ---------------------------------------------------------------- 六条接口
def test_the_overview_shows_what_she_has_recorded(rt: Runtime):
    seeded(rt)
    with client(rt) as c:
        body = c.get("/memory").json()
    assert [f["object"] for f in body["facts"]] == ["团子"]
    assert body["stats"]["live"] == 1


def test_the_overview_keeps_the_dead_and_the_already_used(rt: Runtime):
    """这里看的是**她到底记过什么**，不是「现在还能不能用」。已经失效 / 已经
    提过的那些必须看得见，否则「她提过那件事吗」在界面上永远无法回答。
    """
    mem = seeded(rt)
    mem.note_topic("想问问他面试怎么样")
    mem.forget(mem.get_all()[0]["id"])
    with client(rt) as c:
        body = c.get("/memory").json()
    assert len(body["facts"]) == 1, "失效的那条还在列表里，只是状态变了"
    assert body["facts"][0]["status"] != "active"
    assert body["topics"], "提过的话题也要在"


def test_edit_appends_an_op_and_returns_the_new_value(rt: Runtime):
    seeded(rt)
    fid = Memory(UID, config=rt.config).get_all()[0]["id"]
    with client(rt) as c:
        body = c.post("/memory/edit", json={"id": fid, "set": {"object": "团子（三岁）"}}).json()
    assert body["ok"] is True
    assert body["fact"]["object"] == "团子（三岁）"


def test_a_bad_value_is_a_400_and_not_a_500(rt: Runtime):
    """校验失败是**用户的问题**，必须回 400。回 500 的话调用方会去查服务端日志，
    而真相在请求体里。
    """
    seeded(rt)
    fid = Memory(UID, config=rt.config).get_all()[0]["id"]
    with client(rt) as c:
        bad = c.post("/memory/edit", json={"id": fid, "set": {"importance": "high"}})
        empty = c.post("/memory/edit", json={"id": fid, "set": {"nope": 1}})
    assert bad.status_code == 400
    assert empty.status_code == 400
    assert "importance" in bad.json()["detail"]


def test_a_bad_value_is_refused_before_it_reaches_the_log(rt: Runtime):
    """坏值一旦进了只追加的日志，就会**永久**卡住那个用户的重放。"""
    seeded(rt)
    mem = Memory(UID, config=rt.config)
    fid = mem.get_all()[0]["id"]
    before = len(mem.history())
    with client(rt) as c:
        c.post("/memory/edit", json={"id": fid, "set": {"object": None}})
        c.post("/memory/edit", json={"id": fid, "set": {"due": "下周三"}})
    assert len(Memory(UID, config=rt.config).history()) == before


def test_editing_something_that_does_not_exist_is_a_404(rt: Runtime):
    """而且**不往日志里写**那条指向死 id 的 EDIT —— 写了之后每次重放都要为它
    空转一趟，而谁也不会有对应的物化结果。
    """
    mem = seeded(rt)
    before = len(mem.history())
    with client(rt) as c:
        r = c.post("/memory/edit", json={"id": "F-9999", "set": {"object": "随便"}})
    assert r.status_code == 404
    assert len(Memory(UID, config=rt.config).history()) == before


def test_forget_says_which_mode_it_used(rt: Runtime):
    seeded(rt)
    mem = Memory(UID, config=rt.config)
    fid = mem.get_all()[0]["id"]
    with client(rt) as c:
        archived = c.post("/memory/forget", json={"id": fid}).json()
        purged = c.post("/memory/forget", json={"id": fid, "mode": "purge"}).json()
    assert archived["mode"] == "archive"
    assert "可恢复" in archived["note"]
    assert purged["mode"] == "purge"
    assert "原话" in purged["note"], "如实说清还有什么留着"


def test_forget_of_a_missing_fact_is_a_404(rt: Runtime):
    with client(rt) as c:
        assert c.post("/memory/forget", json={"id": "F-9999"}).status_code == 404
        assert c.post("/memory/confirm", json={"id": "F-9999"}).status_code == 404
        assert c.post("/memory/pin", json={"id": "F-9999"}).status_code == 404


def test_confirm_accepts_and_also_works_as_reject(rt: Runtime):
    """否决走归档而不是删除：用户改主意了还能找回来。"""
    seeded(rt)
    mem = Memory(UID, config=rt.config)
    fid = mem.get_all()[0]["id"]
    with client(rt) as c:
        assert c.post("/memory/confirm", json={"id": fid}).json()["status"] == "active"
        rejected = c.post("/memory/confirm", json={"id": fid, "accept": False}).json()
    assert rejected["status"] == "forgotten"
    assert mem.get(fid)["status"] != "active"


def test_pin_and_unpin(rt: Runtime):
    seeded(rt)
    mem = Memory(UID, config=rt.config)
    fid = mem.get_all()[0]["id"]
    with client(rt) as c:
        assert c.post("/memory/pin", json={"id": fid}).json()["pinned"] is True
        assert mem.get(fid)["pinned"] == 1
        assert c.post("/memory/pin", json={"id": fid, "pinned": False}).json()["pinned"] is False
        assert mem.get(fid)["pinned"] == 0


def test_export_hands_over_the_source_of_truth(rt: Runtime):
    seeded(rt)
    with client(rt) as c:
        body = c.get("/memory/export").json()
    assert body["uid"] == UID
    assert next(op["op"] for op in body["ops"]) == "ADD"
    assert "唯一事实源" in body["note"]


def test_one_identity_cannot_see_anothers_memory(rt: Runtime):
    """**越权的行为面那一条。** 同一条路由、同一份配置，换一个身份就该看到
    另一个库 —— 而这个库是空的。
    """
    seeded(rt)
    with client(rt, uid=OTHER) as c:
        assert c.get("/memory").json()["facts"] == []
        assert c.get("/memory/export").json()["ops"] == []
