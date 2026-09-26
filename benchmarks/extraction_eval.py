"""抽取评测：**需要真实 API Key 与网络**，所以它不在默认档里。

    set OPENAI_API_KEY=sk-...
    python benchmarks/extraction_eval.py
    python benchmarks/extraction_eval.py --model gpt-4o-mini --min-recall 0.6

它回答一个只有真模型才能回答的问题：**那份 prompt 抽出来的东西对不对。**
规模基准（`benchmarks/scale.py`）量的是形状，这个量的是质量 —— 而质量没有
「本机实测」以外的办法，只能拿一组带标注的对话跑一遍看。

**判据是「关键内容串在不在这条事实里」，不逐字比对。**
比逐字宽：模型把「养的猫叫团子」说成「有一只叫团子的猫」也算命中 —— 换个说法
不算错。比不判严：一个关键串都对不上的事实，算**多抽**（编东西的代理指标）。
另外单独报「被闸门挡掉」的条数：那些是回引对不上原话的（编造 quote、或者引用
一个不存在的轮次）—— 七道闸拦下来的数量，比多抽更能说明她会不会瞎编。

**没有 Key 就退出码 2**，不静默跳过：一条静默跳过的评测，跑一百遍也还是
「全都通过」。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from changqing import Memory, MemoryConfig
from changqing.adapters.openai import OpenAILLM
from changqing.runtime import Runtime, using

# 标注语料。每一例都是「一段对话 + 这段对话里必须被记住的关键内容」。
# 覆盖的正是那七道闸各自最容易出错的地方：
#
#   · 他说的日常事实（最基本的一条路）
#   · **她答应过的事** —— 它只存在于 assistant 那一行里（铁律 2 的唯一来源），
#     漏了它的症状是「她忘了自己答应过什么」
#   · **她自己的事** —— 偏好 / 习惯 / 去过的地方（铁律 1）。它同样只存在于她的
#     行里，漏了它会让她今天说爱喝美式、明天说从来不喝咖啡
#   · 相对时间（「下周三」）必须落成绝对日期，否则存进去的是一个会过期的词
#   · 闲聊：「抽不出东西」才是正确答案，能抽出东西反而不对
#   · 同槽位换值：「我换工作了」—— 旧值失效而不是被删
CASES: list[dict[str, Any]] = [
    {
        "name": "养了一只猫",
        "turns": [
            {"user": "我家猫叫团子，三岁了", "assistant": "团子这个名字真好听。"},
        ],
        "keys": ["团子"],
    },
    {
        "name": "她答应过的事",
        "turns": [
            {"user": "周末带我去看展好不好", "assistant": "好呀，周末带你去。"},
        ],
        "keys": ["看展"],
    },
    {
        "name": "她自己的事",
        "turns": [
            {"user": "你老家哪儿的", "assistant": "……我绍兴的。"},
            {"user": "你周末一般都干嘛", "assistant": "……在家画画。有时候去看展。"},
            {"user": "你去过哪儿玩吗", "assistant": "……青岛。去年去的，海边风大。"},
            {"user": "你爱吃什么", "assistant": "……甜的。桂花糕。"},
        ],
        "keys": ["绍兴", "青岛", "桂花糕"],
    },
    {
        "name": "相对时间落成绝对日期",
        "turns": [
            {"user": "我下周三要出差，去两天", "assistant": "那我到时候找你。"},
        ],
        "keys": ["出差"],
        "want_date": True,
    },
    {
        "name": "闲聊不该抽出任何事实",
        "turns": [{"user": "今天有点累", "assistant": "那就早点睡。"}],
        "keys": [],
    },
    {
        "name": "换了工作（同槽位换值）",
        "turns": [
            {"user": "我换工作了，现在在一家做地图的公司", "assistant": "恭喜你！"},
        ],
        "keys": ["地图"],
    },
]


def key_covered(fact: dict[str, Any], keys: list[str]) -> bool:
    """这条事实里有任何一个关键串吗。判据见模块 docstring。"""
    text = f"{fact.get('subject', '')}{fact.get('predicate', '')}{fact.get('object', '')}"
    text += str(fact.get("quote") or "")
    return any(key in text for key in keys)


def run_case(case: dict[str, Any], cfg: MemoryConfig, llm: Any, index: int) -> dict[str, Any]:
    """一个 uid 一例：互相隔离，免得前一段的记忆帮后一段蒙对。"""
    mem = Memory(f"eval-{index}", config=cfg, llm=llm)
    for turn in case["turns"]:
        mem.remember(turn)
    report = mem.extract_now()
    facts = [f for f in mem.get_all() if f.get("status") != "superseded"]
    keys = list(case.get("keys") or [])
    hits = sum(1 for f in facts if key_covered(f, keys)) if keys else 0
    extra = sum(1 for f in facts if not key_covered(f, keys)) if keys else len(facts)
    missing = [k for k in keys if not any(k in str(f.get("object") or "") for f in facts)]
    dated = any(
        len(str(f.get("object") or "")) >= 10 and str(f["object"])[:4].isdigit() for f in facts
    )
    return {
        "case": case,
        "extra": extra,
        "missing": missing,
        "hits": hits,
        "dated": dated,
        "dropped": int(report.get("dropped", 0)),
        "facts": facts,
        "calls": int(report.get("calls", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("EXTRACT_EVAL_MODEL") or "gpt-4o-mini")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or "")
    parser.add_argument("--min-recall", type=float, default=0.0, help="低于它就退出码 1")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    # 这里**是宿主**，所以读环境变量是它的活。库本身不读（见 adapters/openai.py）。
    api_key = os.environ.get("OPENAI_API_KEY") or ""
    if not api_key:
        print("没有 OPENAI_API_KEY：这个评测要真模型，跳过不是「通过」。", file=sys.stderr)
        return 2

    root = Path(tempfile.mkdtemp(prefix="changqing-eval-"))
    cfg = MemoryConfig(root=root)
    kwargs: dict[str, Any] = {"model": args.model}
    if args.base_url:
        kwargs["base_url"] = args.base_url
    llm = OpenAILLM(api_key, **kwargs)
    print(f"模型 {args.model} · 库根 {root}\n")

    hit = miss = extra = dropped = 0
    started = time.perf_counter()
    with using(Runtime(config=cfg)):
        for index, case in enumerate(CASES):
            got = run_case(case, cfg, llm, index)
            hit += len(case["keys"]) - len(got["missing"])
            miss += len(got["missing"])
            extra += got["extra"]
            dropped += got["dropped"]
            mark = "✓" if not got["missing"] and not got["extra"] else "✗"
            print(f"{mark} {case['name']}  （{got['calls']} 次调用）")
            for fact in got["facts"]:
                print(f"    · {fact.get('predicate', '')}{fact.get('object', '')}")
            if got["missing"]:
                print(f"    漏: {', '.join(got['missing'])}")
            if case.get("want_date"):
                print(f"    相对时间落成绝对日期: {'是' if got['dated'] else '否'}")
    elapsed = time.perf_counter() - started

    recall = hit / max(1, hit + miss)
    precision = hit / max(1, hit + extra)
    print(f"\n命中 {hit} · 漏抽 {miss} · 多抽 {extra} · 被闸门挡掉 {dropped}   用时 {elapsed:.1f}s")
    print(f"召回 {recall:.0%} · 精确 {precision:.0%}")
    print("（被闸门挡掉的那些是回引对不上原话的 —— 拦下来比多抽更能说明会不会瞎编）")
    if args.keep:
        print(f"库根留着：{root}")
    if recall < args.min_recall:
        print(f"召回 {recall:.0%} 低于阈值 {args.min_recall:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
