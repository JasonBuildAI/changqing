"""画像：**由宿主注入**的一份「她是谁」的说明。

为什么画像必须可注入，而不是写死在抽取 prompt 里：抽取要回答两个不同的问题 ——

  importance         这条事实**有多重要**（通用，与角色无关）
  persona_attention  **这个角色会不会在意**这条（取决于她是谁）

把某个具体角色的性格写进 prompt，等于把这个库绑死在那一个角色上：换一个角色，
权重重排就全错了，而且**不报错** —— 它只是让「她会记得什么」这件事悄悄跑偏。
所以形状固定在这里，内容归宿主：

    from changqing import Memory, MemoryConfig, PersonaProfile

    Memory(
        config=MemoryConfig(...),
        persona=PersonaProfile(
            name="她",
            description="一个在意光线和颜色的人，话不多。",
        ),
    )

**默认是中性画像**：机制照跑（每条事实仍然带 `persona_attention`、抽取时仍然收到
一份角色说明），只是不偏向任何一种性格。要让它在某个角色上更好用，传一份自己的
—— 那是内容，不是机制。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersonaProfile:
    """一份角色画像。冻结的：它会在多个线程之间共享，读它的人不该看见半截改写。

    `name` 同时是「她答应过他的事」那条事实的 `subject`（见 `promise_subject`）：
    它是**形状**而不是自由内容，所以不交给模型发挥。模型把 subject 写成「他」的话，
    她会拿自己的承诺说成「你上次说……」，编出根本不存在的对话。
    """

    name: str = "她"
    description: str = ""
    attention_hint: str = ""

    @property
    def promise_subject(self) -> str:
        """承诺类事实（`kind=promise`）的 subject。留空时退回一个中性的称呼。"""
        return self.name or "她"

    def angle(self) -> str:
        """抽取 prompt 里那个「以谁的视角」的短语。没给 description 时只留主体。"""
        who = self.name or "她"
        if self.description:
            return f"以{who}的视角（{self.description}）"
        return f"以{who}的视角"

    def attention_question(self) -> str:
        """「这条她会在意吗」那句判据。给了 `attention_hint` 就整句用它。"""
        if self.attention_hint:
            return self.attention_hint
        who = self.name or "她"
        return f"{who}这个人会在意这条吗"


# 中性画像：不指向任何具体角色，但机制完整（抽取仍然收到一份角色说明）。
NEUTRAL = PersonaProfile()

__all__ = ["NEUTRAL", "PersonaProfile"]
