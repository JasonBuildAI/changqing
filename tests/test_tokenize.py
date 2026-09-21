"""分词：装了 jieba 用它，没装退化成相邻两字的 bigram —— **两条路都要有测试**。

**为什么退化路径也必须被测。** 开发与 CI 环境里 jieba 是装着的（`[dev]` 带它），
所以「没装 jieba」那条路只有**用户**才会走到。它一旦坏掉，症状是「中文检索什么都
搜不到」，而那不会报错：FTS5 默认的 unicode61 把连续汉字当成一个 token，
插入「团子三岁了」之后查「团子」命中 0 —— 一切正常，只是什么都搜不到。

所以这里把分词器状态直接按到 bigram 上，再断言它真的还能搜到东西。
判据不是「和 jieba 一样」（本来就不一样），而是**不会一条都搜不到**。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from changqing import Memory
from changqing.retrieve import search
from changqing.runtime import Runtime
from changqing.store import append_op, materialize
from changqing.tokenize import backend, to_index, to_query, tokenize

UID = "u" + "k" * 16


@pytest.fixture()
def bigram(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """把分词器按到「没装 jieba」那条路上（`_load()` 认这两个模块级状态）。"""
    import changqing.tokenize as tk

    monkeypatch.setattr(tk, "_JIEBA", None)
    monkeypatch.setattr(tk, "_BACKEND", "bigram")
    yield


def _fact() -> None:
    """往库里放一条能靠词面命中的事实（不调模型）。"""
    append_op(
        UID,
        {
            "op": "ADD",
            "id": "F-0001",
            "subject": "他",
            "predicate": "养的猫叫",
            "object": "团子",
            "quote": "我家猫叫团子",
            "turn_ref": "T-000001",
            "confidence": 0.9,
            "importance": 0.8,
            "persona_attention": 0.5,
            "status": "active",
        },
    )
    materialize(UID)


# ---------------------------------------------------------------- 退化路径
def test_the_fallback_cuts_adjacent_characters(bigram: None):
    """bigram 的切法：相邻两字一个单位，标点与空白丢掉。"""
    assert tokenize("今天画室很安静") == ["今天", "天画", "画室", "室很", "很安", "安静"]
    assert tokenize("猫，三岁") == ["猫三", "三岁"], "标点不该出现在 token 里"
    assert tokenize("好") == ["好"], "一个字也要能搜"
    assert tokenize("") == []


def test_the_fallback_still_finds_a_hit(rt: Runtime, bigram: None):
    """退化路径的判据：**不会一条都搜不到**。

    这一条才是这条护栏的意义所在 —— 切法不同可以接受，搜不到不行。
    """
    _fact()
    assert [f["id"] for f in search(UID, "我家猫叫什么")] == ["F-0001"]


def test_the_index_and_the_query_use_the_same_tokenizer(rt: Runtime, bigram: None):
    """写索引与拼查询必须走同一把尺子。

    两边各用一套切法时，词面对不上 —— 而它不报错，只是永远搜不到，
    看起来像「这个库的中文检索很弱」。
    """
    assert to_index("我家猫叫团子").split() == tokenize("我家猫叫团子")


# ---------------------------------------------------------------- 两条路共有的判据
def test_every_token_is_quoted_whatever_the_backend():
    """FTS5 的 MATCH 里 AND / OR / NOT / NEAR 是**真的操作符**。

    每个 token 都加引号之后，用户随口说一句「not bad」不会把检索打挂；
    不包的话那不是搜不到，是**语法错误**。
    """
    for text in ("not bad", "AND OR NOT NEAR", "团子三岁了"):
        expr = to_query(text)
        if not expr:
            continue
        for piece in expr.split(" OR "):
            assert piece.startswith('"') and piece.endswith('"'), f"没加引号：{piece!r}"


def test_the_backend_is_one_of_the_two_known_ones():
    """状态查询本身要能回答「现在用的是哪个」—— 面板与自检靠它。"""
    assert backend() in ("jieba", "bigram")


def test_the_backend_is_visible_in_the_turn_context(rt: Runtime, bigram: None):
    """「现在用的是哪个分词器」必须**看得出来**。

    `context()` 的返回里带着 `backend`。少了它，「她的中文检索好像变弱了」
    在日志与面板里一个字都没有 —— 只能靠猜，而这是本仓库最贵的缺陷类型。
    """
    mem = Memory(UID, config=rt.config)
    mem.remember({"user": "在吗", "assistant": "嗯"})
    assert mem.context("你好")["backend"] == "bigram"
