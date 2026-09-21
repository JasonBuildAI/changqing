"""规模基准：这东西在「一个人天天用上很久」的量级上是什么形状。

    python benchmarks/scale.py                       # 3 个人 × 300 轮
    python benchmarks/scale.py --users 10 --turns 500
    python benchmarks/scale.py --keep                # 留下库根，自己进去翻

**它是离线的**：合成语料 + 离线替身 + 一个「照着 prompt 里那些行推导事实」的
假模型（`DerivedLLM`）。不联网、不下模型，所以同一台机器上两次跑出来的形状一致，
可以拿来比「改了这个参数之后是变好还是变坏」。

**数字只用来读形状，不是产品承诺。** 机器不同、语料不同、真模型的事实密度也不同。
能带走的只有两样：**每一轮占多少字节**、**检索随索引变大的斜率**。任何写进
文档的容量结论都必须是「假设 + 算式」，不能写成「本机实测」（见 `docs/design.md`）。

量四件事：

  1. **L0 写入**：一轮原话的落盘耗时 —— 它压在「他刚说完一句话」那条路上；
  2. **整理**：模型调用次数与过闸门的事实数 —— 成本的主体在这里；
  3. **检索**：索引变大之后冷路径还稳不稳（p50 / p95），以及一轮注入多少 token；
  4. **占用与投影**：字节花在哪一层，以及乘上规模假设之后是多少。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from changqing import Memory, MemoryConfig
from changqing.adapters.mock import MockEmbedder
from changqing.runtime import Runtime, using

# 合成语料的模板。**刻意设计成会互相打架**：同一个人在不一样的轮次里说不一样的
# 猫名、不一样的住处 —— 于是 `resolve_ops` 会真的产出 SUPERSEDE（旧事实失效而
# 不是删除），索引里不只有 ADD。全是新事实的语料量出来的东西不真实。
TEMPLATES = (
    ("我养的猫叫{v}", "养的猫叫", ("团子", "豆包", "雪球", "芝麻", "团子（三岁）")),
    ("我住在{v}", "住在", ("杭州", "成都", "厦门", "南京")),
    ("我喜欢的颜色是{v}", "喜欢的颜色是", ("深蓝", "墨绿", "灰", "米白")),
    ("我不吃{v}", "不吃", ("香菜", "芹菜", "苦瓜", "内脏")),
    ("我下个月要搬到{v}", "要搬到", ("上海", "苏州", "广州", "北京")),
)

# 闲聊句。每 5 轮里挑 1 轮当它 —— 真实对话里大部分轮次什么都不产出，
# 全是模板句的语料会让「事实密度」虚高好几倍。
SMALL_TALK = ("今天有点累", "嗯，我在听", "外面下雨了", "刚吃完饭", "有点困了")

# 探测句：与模板一一对应。它们**不含**答案里的那个词，所以能命中的只可能是
# 槽位或全文检索，而不是把答案又抄了一遍。
PROBES = ("我养的猫叫什么", "我住在哪儿", "我喜欢的颜色是什么", "我不吃什么")


class DerivedLLM:
    """一个「照着 prompt 里那些行推导事实」的离线模型。

    **为什么不用 `MockLLM` 直接吐一段写死的 JSON**：那样每条事实的 `turn_ref`
    都对不上真实轮次，回引校验会把它们全部判成 `drop` —— 索引永远是空的，
    而这个基准要量的偏偏是「索引里有东西之后的样子」。

    这里做的正是真模型该做的事：从带编号的行里产出事实，`quote` 取原话、
    `turn_ref` 取那一行的编号。于是它过得了那七道闸，量出来的数字才有意义。
    （它的「智商」远不如真模型 —— 但基准量的是**形状**，抽取质量归
    `benchmarks/extraction_eval.py`。）
    """

    _LINE = re.compile(r"^\[(?P<tid>T-\d+)\][^\n]*?他说：(?P<text>.*)$", re.M)

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.calls += 1
        facts: list[dict[str, Any]] = []
        prompt = str(messages[-1].get("content") or "")
        for hit in self._LINE.finditer(prompt):
            text = hit.group("text").strip()
            for pattern, predicate, _values in TEMPLATES:
                head = pattern.partition("{v}")[0]
                if text.startswith(head) and len(text) > len(head):
                    facts.append(
                        {
                            "subject": "他",
                            "predicate": predicate,
                            "object": text[len(head) :],
                            "quote": text,
                            "turn_ref": hit.group("tid"),
                            "kind": "fact",
                            "confidence": 0.9,
                            "importance": 0.5,
                            "persona_attention": 0.5,
                        }
                    )
                    break
        return json.dumps({"facts": facts, "summary": "", "topics": []}, ensure_ascii=False)


def synth_turn(rng: random.Random, index: int) -> str:
    """一句合成原话。第 `5n+4` 轮是闲聊，其余按模板轮流取（五个模板都会轮到）。"""
    if index % 5 == 4:
        return rng.choice(SMALL_TALK)
    pattern, _predicate, values = TEMPLATES[(index // 5) % len(TEMPLATES)]
    return pattern.format(v=rng.choice(values))


def size_breakdown(root: Path) -> dict[str, int]:
    """字节花在哪一层。**分层报出来**：只说总量看不出该优化哪一层。

    三层的性质完全不同：`sessions/` 是不可再生的原话（省不了），
    `log.jsonl` 是不可再生的操作日志（省不了），`index.sqlite` 是**可重建**的
    物化视图（要省就省它）。
    """
    out = {"sessions": 0, "archive": 0, "log": 0, "index": 0, "views": 0, "other": 0}
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        size = item.stat().st_size
        name = item.name
        if item.parent.name == "sessions.archive" or name.endswith(".gz"):
            out["archive"] += size
        elif name == "log.jsonl":
            out["log"] += size
        elif name.startswith("index.sqlite"):
            out["index"] += size
        elif item.parent.name == "sessions":
            out["sessions"] += size
        elif name.endswith(".md"):
            out["views"] += size
        else:
            out["other"] += size
    return out


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def run(args: argparse.Namespace, cfg: MemoryConfig, root: Path) -> int:
    rng = random.Random(args.seed)
    llm = DerivedLLM()
    users = max(1, args.users)
    turns = max(1, args.turns)
    print(f"规模 {users} 人 × {turns} 轮（合成语料，离线替身）\n")

    # ---------------------------------------------------------------- L0 写入
    members: list[Memory] = []
    started = time.perf_counter()
    for user in range(users):
        mem = Memory(f"bench-{user}", config=cfg, embedder=MockEmbedder(), llm=llm)
        for index in range(turns):
            mem.remember({"user": synth_turn(rng, index), "assistant": "嗯，我记着了。"})
        members.append(mem)
    write_s = time.perf_counter() - started
    total_turns = users * turns
    print(f"[L0] 写入 {total_turns} 轮  {write_s * 1000 / total_turns:.2f} ms/轮")

    # ---------------------------------------------------------------- 整理
    started = time.perf_counter()
    reports = [mem.extract_now() for mem in members]
    extract_s = time.perf_counter() - started
    passed = sum(int(r.get("active", 0)) for r in reports)
    added = sum(int(r.get("ADD", 0)) for r in reports)
    replaced = sum(int(r.get("SUPERSEDE", 0)) for r in reports)
    dropped = sum(int(r.get("dropped", 0)) for r in reports)
    live = sum(int(mem.stats().get("live", 0)) for mem in members)
    print(
        f"[整理] {llm.calls} 次模型调用  {extract_s:.2f}s  "
        f"{extract_s / max(1, llm.calls) * 1000:.0f} ms/次"
    )
    # 这个「ms/次」是**平均**，第一次那笔里含建索引/开库的一次性开销
    # （实测把它单独摊出来能差出七八倍）。要看稳态就得跑大一点再看趋势 ——
    # 这个基准量的是形状，不是某一次的绝对值。
    print("       （首次数含建索引的一次性开销，别当稳态）")
    print(
        f"       过闸门 {passed} 条（新增 {added} / 换旧 {replaced} / 被闸门丢掉 {dropped}）"
        f"，其中还活着 {live} 条"
    )
    print(f"       平均每次调用产出 {passed / max(1, llm.calls):.1f} 条")

    # ---------------------------------------------------------------- 检索
    latencies: list[float] = []
    injected: list[int] = []
    hits = 0
    for index in range(int(args.queries)):
        mem = members[index % len(members)]
        started = time.perf_counter()
        found = mem.recall(PROBES[index % len(PROBES)])
        latencies.append((time.perf_counter() - started) * 1000)
        hits += 1 if found else 0
    for mem in members:
        injected.append(int(mem.context("你好呀").get("used_tokens", 0)))
    print(
        f"[检索] {args.queries} 次  命中 {hits}  "
        f"p50 {pct(latencies, 0.5):.1f} ms / p95 {pct(latencies, 0.95):.1f} ms"
        f"   （冷路径预算 {cfg.recall_ms} ms）"
    )
    print(
        f"       一轮注入中位数 {int(statistics.median(injected or [0]))} token"
        f"（事实+故事线+话题，预算 {cfg.hot_tokens}）"
    )

    # ---------------------------------------------------------------- 占用
    parts: dict[str, int] = {}
    files = 0
    for user in range(users):
        uid = f"bench-{user}"
        one = size_breakdown(root / uid[:2] / uid)
        for key, value in one.items():
            parts[key] = parts.get(key, 0) + value
        files += sum(1 for item in (root / uid[:2] / uid).rglob("*") if item.is_file())
    total_bytes = sum(parts.values())
    per_turn = total_bytes / max(1, total_turns)
    print(f"\n[占用] {users} 个库 {files} 个文件 {total_bytes / 1024:.0f} KiB")
    for key in ("sessions", "log", "index", "views", "archive", "other"):
        if parts.get(key):
            print(f"       {key:<9} {parts[key] / 1024:8.1f} KiB  {parts[key] / total_bytes:5.1%}")
    print(f"       每一轮 ≈ {per_turn:.0f} 字节（上面几层加起来分摊）")
    print(
        f"       其中原话那一层 ≈ {parts.get('sessions', 0) / max(1, total_turns):.0f} 字节/轮"
        "（这一层随轮数线性长）"
    )

    # ---------------------------------------------------------------- 投影
    # **这是算式，不是实测**：把「每一轮多少字节」乘到规模假设上。它成立的理由
    # 只有一条 —— 每轮的成本与前面聊过多少轮无关。撑着这一条的是上面那两个
    # p50 / p95：索引长大了，延迟没跟着涨。
    per_user_turns = 90_000  # 每天 3 场 × 每场 300 轮 × 100 天
    print(
        "\n       索引那一层里有 sqlite 的页对齐开销，样本越小占比越大："
        "所以下面的投影是**偏保守**的一侧。"
    )
    print(f"\n[投影] 每人 {per_user_turns:,} 轮（每天 3 场 × 每场 300 轮 × 100 天）")
    print(f"       每人 ≈ {per_turn * per_user_turns / 1024 / 1024:.0f} MiB")
    print(f"       一万人 ≈ {per_turn * per_user_turns * 10_000 / 1024**3:.0f} GiB")
    print("       （未计归档压缩；那一层见 docs/storage.md 的保留策略）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--users", type=int, default=3, help="合成几个人")
    parser.add_argument("--turns", type=int, default=300, help="每人合成多少轮")
    parser.add_argument("--queries", type=int, default=40, help="量多少次检索延迟")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--keep", action="store_true", help="留下库根，方便进去翻")
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="changqing-bench-"))
    cfg = MemoryConfig(root=root)
    print(f"库根 {root}")
    # 进程级那一份也指到临时根：万一哪条路径绕过了 Memory 的显式配置，
    # 它落在临时目录里，而不是用户的 ~/.changqing。
    with using(Runtime(config=cfg)):
        code = run(args, cfg, root)
    if args.keep:
        print(f"库根留着：{root}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
