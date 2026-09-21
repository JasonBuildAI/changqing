"""画像：默认中性、可以换、换完之后抽取用的那句说明真的跟着变。"""

from __future__ import annotations

from changqing.persona import NEUTRAL, PersonaProfile
from changqing.runtime import Runtime, configure, using


def test_default_profile_is_neutral():
    """出厂画像不指向任何具体角色 —— 这是「换个角色不用改库」的前提。"""
    assert NEUTRAL.name == "她"
    assert NEUTRAL.description == ""
    assert NEUTRAL.attention_hint == ""
    assert NEUTRAL.angle() == "以她的视角"
    assert NEUTRAL.attention_question() == "她这个人会在意这条吗"


def test_description_shows_up_in_the_angle():
    p = PersonaProfile(name="小满", description="一个爱做饭、话不多的人")
    assert p.angle() == "以小满的视角（一个爱做饭、话不多的人）"


def test_attention_hint_replaces_the_derived_question():
    """给了整句就用整句：判据是内容，不该被我们拼的模板改一个字。"""
    p = PersonaProfile(name="小满", attention_hint="她会不会想接着听这件事")
    assert p.attention_question() == "她会不会想接着听这件事"


def test_promise_subject_falls_back_to_a_neutral_noun():
    """`name` 留空也要能落库：承诺类事实的 subject 不能是空串。"""
    assert PersonaProfile(name="").promise_subject == "她"
    assert PersonaProfile(name="小满").promise_subject == "小满"


def test_configure_carries_the_persona(rt: Runtime):
    """注入路径：`configure(persona=...)` 之后 `runtime().persona` 就是它。

    这条断言看着平淡，它挡的是「装配里加了字段但 `configure` 忘了转发」——
    那种情况下画像会被**静默忽略**，抽取照旧用中性画像，谁也不会报错。
    """
    mine = PersonaProfile(name="小满", description="爱做饭")
    with using(Runtime(config=rt.config, persona=mine)):
        from changqing import runtime

        assert runtime().persona is mine
    assert configure().persona is NEUTRAL
