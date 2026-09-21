"""配置的夹取规则：坏值兜住，但**合法的关掉**不能被悄悄改成别的意思。"""

from __future__ import annotations

from pathlib import Path

from changqing import MemoryConfig


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
