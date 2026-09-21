"""配置的夹取规则：坏值兜住，但**合法的关掉**不能被悄悄改成别的意思。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from changqing import MemoryConfig
from changqing.config import ENV_MAP


def test_retain_months_zero_means_never_clean():
    """0 是一行关掉超期归档的那个值，不是「保留一个月」。

    夹到 1 的话用户以为关掉了，实际每个月都在搬走原话 —— 而搬走这件事
    在读取上是透明的，所以谁都不会发现，直到去找一个早就不在该月的文件。
    """
    assert MemoryConfig(retain_months=0).retain_months == 0


def test_negative_retention_clamps_to_off_not_to_one():
    """负数是写错了的配置：按最保守的那边解释（永不清理），而不是替用户做清理。"""
    assert MemoryConfig(retain_months=-5).retain_months == 0


def test_reasonable_values_pass_through():
    assert MemoryConfig(retain_months=3).retain_months == 3
    assert MemoryConfig().retain_months == 24


def test_root_is_expanded_and_defaulted():
    """`~` 要展开 —— 不展开的话每个用户目录都会叫一个字面量 `~`。"""
    assert MemoryConfig(root="~/x").root == Path.home() / "x"
    assert MemoryConfig().root.is_absolute()


def test_idle_split_seconds_never_goes_negative():
    """**唯一的空闲判据**，负数会让「超过它就算闲」永远成立。"""
    assert MemoryConfig(idle_split_sec=-1).idle_split_sec == 0
    assert MemoryConfig().idle_split_sec == 600


def test_evolved_returns_a_new_object():
    base = MemoryConfig()
    other = base.evolved(hot_tokens=500)
    assert other is not base
    assert other.hot_tokens == 500
    assert base.hot_tokens == 300, "原配置不动"


# ---------------------------------------------------------------- 环境变量
def test_env_map_covers_every_field_and_nothing_else():
    """`ENV_MAP` 与字段必须一一对应。

    它是 `docs/configuration.md` 那张表的真源。漏一个字段的症状是「这个开关读不到，
    而它看起来完全正常」—— 部署脚本里写着 `CHANGQING_XXX`，库里没有一处认领它；
    多一个名字则是文档里多了一个不存在的开关。**两个方向都不报错**，所以钉住。
    """
    fields = {f.name for f in dataclasses.fields(MemoryConfig)}
    assert set(ENV_MAP) == fields, f"ENV_MAP 与字段不一致：{sorted(set(ENV_MAP) ^ fields)}"


def test_env_map_names_are_unique():
    """两个字段不能共用一个环境变量名：后一个会永远读不到自己的值。"""
    names = list(ENV_MAP.values())
    assert len(names) == len(set(names)), "有重复的环境变量名"


def test_from_env_reads_the_prefixed_variables(monkeypatch):
    monkeypatch.setenv("CHANGQING_HOT_TOKENS", "777")
    assert MemoryConfig.from_env().hot_tokens == 777


def test_blank_environment_value_means_unset(monkeypatch):
    """空串当没设置：`CHANGQING_DIR=` 写下来表示「用默认」。

    不当没设置的话，`os.environ` 给的是空串而不是默认值，于是根目录变成当前目录
    —— 在用户眼里就是「我什么都没改，记忆库突然搬家了」。
    """
    monkeypatch.setenv("CHANGQING_DIR", "")
    assert MemoryConfig.from_env().root == MemoryConfig().root


def test_a_broken_number_falls_back_to_the_default(monkeypatch):
    """写错一行不该让整个库起不来。"""
    monkeypatch.setenv("CHANGQING_HOT_TOKENS", "不是数字")
    assert MemoryConfig.from_env().hot_tokens == 300


def test_from_env_can_take_another_prefix(monkeypatch):
    """宿主的变量名带着自己的前缀时，不必为了接这个库去改部署脚本。"""
    monkeypatch.setenv("MYAPP_RECALL_K", "9")
    assert MemoryConfig.from_env(prefix="MYAPP_").recall_k == 9
