"""30 秒跑通：**不需要任何 API Key，也不联网。**

    python examples/quickstart.py

它会走完整条链路 —— 写下原话、整理成事实、按一句话检索回来、拼出这一轮该
注入的素材 —— 全程只用离线替身（`MockEmbedder` / `MockLLM`），所以每次输出
都一样。这两个替身不是「测试用的假货」，而是这个库**默认就该能跑起来**的
那条路：先看见它工作，再决定接哪个模型。

上线只换一行：把替身换成 `changqing.adapters.openai` 里的
`OpenAIEmbedder` / `OpenAILLM`，其余代码一个字都不用动 —— 抽取的 prompt、
检索的算式、存储的格式都不归模型管。

库根走 `CHANGQING_DIR`（默认 `~/.changqing`）。想让它写在别处：

    CHANGQING_DIR=./my-memory python examples/quickstart.py
"""

from __future__ import annotations

import json

from changqing import Memory, MemoryConfig, PersonaProfile
from changqing.adapters.mock import MockEmbedder, MockLLM


def scripted_extraction() -> str:
    """模型本该吐出来的那段 JSON，这里写死一份。

    真实模型会按 `changqing/extract.py` 里那份 prompt 自己生成它。写死是为了让
    例子的输出确定：验证「检索能把它捞回来」这件事，不该依赖网络。

    `quote` 必须是那一轮原话的**子串**，`turn_ref` 指向产生它的轮次 ——
    抽取层用这条回引挡幻觉，编出来的 quote 会被判成 `drop`（见 `docs/extraction.md`）。
    """
    return json.dumps(
        {
            "facts": [
                {
                    "subject": "他",
                    "predicate": "养的猫叫",
                    "object": "团子",
                    "quote": "我家猫叫团子",
                    "turn_ref": "T-000001",
                    "kind": "fact",
                    "confidence": 0.9,
                    "importance": 0.6,
                    "persona_attention": 0.5,
                }
            ],
            "summary": "第一次聊到他的猫。",
            "topics": [{"kind": "share", "text": "问问他团子多大了"}],
        },
        ensure_ascii=False,
    )


def main() -> int:
    cfg = MemoryConfig.from_env()
    mem = Memory(
        "demo-user",
        config=cfg,
        # 中性画像：抽取向模型交代「她是哪种人」，事实的 persona_attention
        # 就按它打分。换成你自己的角色设定即可（见 docs/configuration.md）。
        persona=PersonaProfile(
            name="assistant",
            description="一个长期陪在身边的助手：话不多，记得住细节。",
            attention_hint="跟他的日常、约定、和他在意的人有关的事。",
        ),
        embedder=MockEmbedder(),
        llm=MockLLM(default=scripted_extraction()),
    )
    print(f"库根: {cfg.root}")

    # L0：原话落盘。这是**唯一不可再生**的一层，所以只追加、不修改。
    ids = mem.remember({"user": "我家猫叫团子", "assistant": "记住了，团子。"})
    print(f"L0 写下 {len(ids)} 轮原话: {ids}")

    # 整理：真实部署里由后台线程按三个触发条件跑，这里手动催一次。
    print(f"整理: {json.dumps(mem.extract_now(), ensure_ascii=False)}")

    # 冷路径：按这句话去找相关的事。分数不够就返回空 —— 宁可不说，也不编。
    # 第二个问题故意问一件**从没说过**的事：空列表就是正确答案，而不是故障。
    for query in ("我家猫叫什么？", "我不吃什么？"):
        cards = mem.recall(query)
        print(f"\n问到「{query}」:")
        if not cards:
            print("  （没有够格的事）")
        for fact in cards:
            head = f"{fact['subject']}{fact['predicate']}{fact['object']}"
            print(f"  · {head}  (来自 {fact['turn_ref']})")

    # 这一轮该把什么摆到她面前：热路径 + 冷路径 + 故事线 + 待提话题，
    # 每份各有预算，`used_tokens` 是三份加起来。
    ctx = mem.context("你好呀")
    print(
        f"\n这一轮注入 {ctx['used_tokens']} token："
        f"热 {ctx['hot']} 条 / 冷 {ctx['cold']} 条 / 故事线 {len(ctx['summaries'])} 条"
    )
    # L1 的账：哪些事还在、哪些已经失效 —— 失效的是**不删**，见 docs/storage.md。
    print(f"事实 {len(mem.get_all())} 条 · 规模 {json.dumps(mem.stats(), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
