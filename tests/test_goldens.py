"""行为金标准：把「解耦有没有改行为」钉成一次可执行的比对。

**为什么非要有这一份。** 这套代码是从别处原样搬过来的，而搬运最容易出的错
不是「跑不起来」—— 那会立刻报错。是**跑起来但结果悄悄偏了一点**：排序换了
方向、少了一道过滤、预算算错一格。这种偏差不报错、不变红，只在你某天发现
「她怎么把上周说过的猫忘了」时才浮出来，而那时已经查不动了。

冻结的做法：用同一组固定输入跑**搬运前的那份实现**（它的根目录指到临时目录，
不碰真实记忆），把输出写进 `goldens/behavior.json`，再由这里逐条比对。

两处**有意**的不一致写在 json 的 `_note` 里，不在代码里做特例：

  · 字段 `her_attention` 改名成 `persona_attention`（人设解耦）；
  · 渲染视图那句「要改走 HTTP 接口」改成「要改走编辑接口」—— 存储层不该提 HTTP。

这份金标准只覆盖**纯函数**与**同一组操作下的检索结果**。时间、随机、网络
一概不在里面：这个库本来就不依赖它们（见 `docs/architecture.md`），所以冻结
得住；哪天它开始依赖了，这里会先红。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from changqing import extract, retrieve
from changqing.consolidate import _core_len, _similar
from changqing.runtime import Runtime, runtime
from changqing.store import append_op, list_facts, materialize, render_facts_md
from changqing.store.turnlog import escape_text, unescape_text
from changqing.tokens import estimate_tokens
from changqing.worker import user_rounds_of

GOLDEN = Path(__file__).resolve().parent / "goldens" / "behavior.json"
DAY = "2026-09-21"


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    """冻结的那份金标准。模块级读一次 —— 它是一份只读常量，不是随用例变的东西。"""
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_pure_functions_reproduce_the_golden(golden: dict, rt: Runtime) -> None:
    """纯函数逐条比对：同一份输入，一字不差。"""
    fx, ex = golden["fixtures"], golden["expected"]
    assert retrieve.card_text(fx["fact"]) == ex["card_text"], "卡片文本变了"
    assert [estimate_tokens(t) for t in fx["corpus"]] == ex["estimate_tokens"], "token 估算变了"
    assert [extract.absolutize(t, DAY) for t in fx["corpus"]] == ex["absolutize"], "时间绝对化变了"
    assert [[escape_text(t), unescape_text(escape_text(t))] for t in fx["corpus"]] == ex[
        "escape"
    ], "转义往返变了"
    assert [extract.parse_facts(r) for r in fx["raws"]] == ex["parse_facts"], "抽取结果解析变了"
    assert [extract.parse_summary(r) for r in fx["raws"]] == ex["parse_summary"], "纪要解析变了"
    assert [extract.parse_topics(r, DAY) for r in fx["raws"]] == ex["parse_topics"], "话题解析变了"
    assert [_core_len(t) for t in fx["corpus"]] == ex["core_len"], "内容量口径变了"
    assert [user_rounds_of(s) for s in fx["states"]] == ex["user_rounds_of"], "用户轮数变了"
    assert retrieve._rrf([["a", "b", "c"], ["b", "c", "d"], ["c"]]) == ex["rrf"], "RRF 变了"
    floor = runtime().config.merge_similarity
    assert [[_similar(a, b, floor=floor) for a, b in fx["pairs"]]] == ex["similar"], "相似度变了"


def test_verification_and_slot_resolution_reproduce_the_golden(golden: dict, rt: Runtime) -> None:
    """低幻觉闸门与同槽位冲突消解：这两处决定「她记错了什么」，最不能漂。"""
    fx, ex = golden["fixtures"], golden["expected"]
    got = [
        list(extract.verify({**c, "kind": c.get("kind", "fact")}, fx["turns"], 0.6))
        for c in fx["cands"]
    ]
    assert got == ex["verify"], "回引校验的判决变了"
    resolved = [list(x) for x in extract.resolve_ops(fx["uid"], [dict(f) for f in fx["batch"]])]
    assert resolved == ex["resolve_ops"], "冲突消解产出的操作变了"


def test_recall_and_views_reproduce_the_golden(golden: dict, rt: Runtime) -> None:
    """同一段操作日志之下：检索结果、热素材顺序、渲染视图都要一模一样。"""
    fx, ex = golden["fixtures"], golden["expected"]
    uid = fx["uid"]
    for op in fx["ops"]:
        append_op(uid, dict(op))
    materialize(uid)
    assert {q: [f["id"] for f in retrieve.search(uid, q)] for q in fx["queries"]} == ex["recall"], (
        "检索顺序变了"
    )
    assert [f["id"] for f in retrieve.hot_facts(uid)] == ex["hot"], "热素材顺序变了"
    assert [[f["id"], f["status"], f["pinned"]] for f in list_facts(uid, include_dead=True)] == ex[
        "facts"
    ], "事实集合或状态变了"
    path = render_facts_md(uid)
    assert path is not None, "渲染视图没落盘"
    assert path.read_text(encoding="utf-8") == ex["render"], "渲染视图的文本变了"
