"""`state.json` 的两条写路：改顶层键、以及合并嵌套字段。

为什么值得单独测：这两条路都长成「读-改-写」，而**错的那一版不报错** ——
它只是偶尔把前台刚写上去的一轮盖掉，症状是「她偶尔丢掉刚说过的话」。
"""

from __future__ import annotations

from changqing.runtime import Runtime
from changqing.store import load_state, mutate_state, update_state


def test_update_state_touches_only_the_keys_it_was_given(rt: Runtime):
    """只改我点名的那几个键 —— 没点名的（前台刚写的 `user_rounds`）必须还在。"""
    update_state("u1", user_rounds=3, extracted_rounds=1)
    update_state("u1", extracted_rounds=2)
    st = load_state("u1")
    assert st["user_rounds"] == 3, "没被点名的键原样留着"
    assert st["extracted_rounds"] == 2


def test_mutate_state_merges_inside_the_lock(rt: Runtime):
    """嵌套字段走 `mutate_state`：合并基于锁内读到的那一份，不是锁外的快照。"""
    update_state("u1", watermark={"last_ts": 111})

    def advance(s: dict) -> None:
        wm = s.get("watermark") or {}
        wm["extracted_upto"] = "T-000009"
        s["watermark"] = wm

    mutate_state("u1", advance)
    st = load_state("u1")
    assert st["watermark"] == {"last_ts": 111, "extracted_upto": "T-000009"}, (
        "两个键都在 —— 覆盖写会把 last_ts 弄丢"
    )


def test_mutate_state_returns_the_state_it_wrote(rt: Runtime):
    """返回值就是落盘的那一份，调用方不必再读一次盘。"""
    out = mutate_state("u2", lambda s: s.update({"rounds": 7}))
    assert out["rounds"] == 7
    assert load_state("u2")["rounds"] == 7


def test_missing_state_reads_as_empty(rt: Runtime):
    """没聊过的人没有 state.json —— 读它要返回空字典而不是抛。"""
    assert load_state("nobody") == {}
