# 长情 · changqing

> 为**长期相处**的对话关系设计的记忆系统。

[English](README.en.md) · [架构](docs/architecture.md) · [设计](docs/design.md) · [变更记录](CHANGELOG.md)

[![CI](https://github.com/JasonBuildAI/changqing/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonBuildAI/changqing/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 它解决什么问题

一个陪你聊了一百天、说过九万句话的模型，怎么才能**真的记得你** —— 包括你刚才
提过的那件小事，和三个月前随口说过的那个名字。

两条常见做法都不够，而且失败的方式不一样：

- **只靠检索（RAG）**：你刚说的话还没进检索库，所以她「想不起来」你上一句说了什么；
- **只靠上下文**：一味加长一定会爆，而且爆之前先变贵 —— 每轮都在为一百天前的话付钱。

`changqing` 的分工是：**上下文管这一轮，记忆系统管越久越要整理**。整理由模型做
（抽取 + 巩固），而不是把原话无限堆进去。它把记忆分成四层，并且**每一条事实都能
回引到产生它的那句原话** —— 所以「她为什么记得这个」永远答得出来。

## 30 秒跑起来

**不需要任何 API Key，也不联网。**

```bash
pip install -e .
python examples/quickstart.py
```

它会写下两轮原话、整理成事实、按一句话检索回来，全程用离线替身。
接真模型只换一处（见「接自己的模型」）。

```python
from changqing import Memory, MemoryConfig
from changqing.adapters.mock import MockEmbedder, MockLLM

mem = Memory(
    "user-1",
    config=MemoryConfig(root="./data"),
    embedder=MockEmbedder(),
    llm=MockLLM(default=抽取输出),
)
mem.remember({"user": "我家猫叫团子", "assistant": "记住了，团子。"})
mem.extract_now()  # 真实部署里由后台线程按触发条件跑
for fact in mem.recall("我家猫叫什么？"):
    print(fact["subject"] + fact["predicate"] + fact["object"])  # 他养的猫叫团子
```

## 四层

| 层 | 存什么 | 落在哪 | 性质 |
|---|---|---|---|
| **L0 原话** | 一句一句的原文 | `sessions/YYYY-MM-DD.md` | 只追加；超期只**归档**（gzip），一个字节都不删 |
| **L1 事实** | 结构化的事实（他的、她自己的）与承诺 | `log.jsonl` → `index.sqlite` | 日志不可再生，索引可重建 |
| **L2 纪要** | 一段对话之后的小结 | `summaries` | 有预算的散文，可裁剪 |
| **L3 画像** | 事实上的权重 | `persona_attention` | 由注入的 `PersonaProfile` 决定 |

```
   他说 ──► L0 原话（只追加）
              │
              │  后台整理（三个触发条件；每轮对话的记忆类调用 = 0）
              ▼
           抽取 ── 七道闸 ──► L1 操作日志（ADD / SUPERSEDE / …）
                                  │  物化
                                  ▼
                             index.sqlite（槽位 + 全文 + 向量）
                                  │
      这一轮的话 ──► 检索：热路径（恒定）+ 冷路径（按需）──► 预算裁剪 ──► 卡片
                        ＋ L2 故事线 ＋ 主动话题
```

## 三条不会回头的结论

1. **光靠 RAG 不行**（刚说过的话还没进检索库），**光靠上下文也不行**（一定会爆）。
2. **整理必须由模型做**，不能靠规则堆原话 —— 但整理是**独立的一次调用**，
   不复用陪聊那条 prompt：一个要「像她」，一个要「有一说一」，目标互相打架。
3. **没有依据的事实不进库**。每条事实都要带 `turn_ref` 与 `quote`，`quote` 必须
   能在被回引的那一轮里定位；对不上就丢，近似就进「待确认」。宁可少记一条，
   也不记错一条 —— 详见 [抽取](docs/extraction.md)。

## 安装

核心是**零第三方依赖**的纯标准库实现；其余按需装：

```bash
pip install .                    # 核心：存储 + 抽取 + 检索 + 整理（离线替身可用）
pip install ".[openai]"          # OpenAI 兼容的嵌入与对话模型
pip install ".[server]"          # 可挂载的 FastAPI 路由
pip install ".[chinese]"         # jieba 分词（缺失时退化成相邻两字的 bigram：召回面窄一些）
pip install ".[all]" ".[dev]"    # 全都要 / 开发用
```

`requires-python >= 3.10`。**不发布 PyPI**，用 `pip install -e .` 或直接从仓库装。

## 接自己的模型

这个库**不自己调任何模型**：三种能力都由宿主注入（Protocol 见
[`src/changqing/ports.py`](src/changqing/ports.py)）。

| 能力 | 做什么 | 不注入会怎样 |
|---|---|---|
| `Embedder` | 把文本变成向量 | 向量召回整支关闭（槽位 + 全文召回照常） |
| `LLM` | 走一次补全（抽取与巩固） | 只写 L0，不整理；调用时**大声报错**，不静默返回空串 |
| `UsageSink` | 记一笔用量 | 不记账 |

```python
import os
from changqing.adapters.openai import OpenAIEmbedder, OpenAILLM

mem = Memory(
    "user-1",
    config=MemoryConfig.from_env(),
    embedder=OpenAIEmbedder(os.environ["OPENAI_API_KEY"]),
    llm=OpenAILLM(os.environ["OPENAI_API_KEY"], model="gpt-4o-mini"),
)
```

**API Key 必须显式传**：这个库不读 `OPENAI_API_KEY`。读环境变量是宿主的事
（多租户就是一家一个 Key），一个库背着调用方去读，症状是「我以为它没配，它却在联网」。
`base_url` 可以指向任何 OpenAI 兼容端点（自建 vLLM / Ollama / 各家网关）。

## 挂到 Web 服务

```python
from fastapi import FastAPI
from changqing.server import create_router

app = FastAPI()
app.include_router(create_router(), prefix="/api")  # 于是 /api/memory 就是记忆接口
```

路径是**相对的**，前缀由宿主决定。**没有任何接口接受 uid 参数** —— 身份只从服务端
签发处取（默认读 `request.state.uid`），否则改一个字符就能读别人的记忆。
接口清单见 [API](docs/api.md)。

## 成本

写死的纪律：**每轮对话的记忆类模型调用 = 0**；单场对话的整理调用
≤ `extract_max_calls`（默认 8）。这一条不是优化，而是这个库能不能用在「一万个人
天天用」上的分水岭 —— 滑向「每轮一次」就是 300 次/场，差两个数量级。

规模基准不联网也能跑（合成语料 + 离线替身）：

```bash
python benchmarks/scale.py                # 量形状：字节花在哪一层、检索的斜率
python benchmarks/extraction_eval.py      # 量质量：需要真实 Key
```

**里面任何一个数字都不是产品承诺**：能带走的只有量级与斜率。容量结论一律写成
「假设 + 算式」，见 [设计](docs/design.md)。

## 文档

| 文档 | 讲什么 |
|---|---|
| [架构](docs/architecture.md) | L0–L3 的职责边界、数据流、进程级运行期的取舍 |
| [设计](docs/design.md) | 为什么光靠 RAG 不行、为什么必须模型整理、容量怎么算 |
| [抽取](docs/extraction.md) | 低幻觉的七道闸、prompt 的铁律、时间绝对化 |
| [检索](docs/retrieval.md) | 热 / 冷两条路径、RRF 融合、画像重排、预算裁剪 |
| [存储](docs/storage.md) | 落盘格式、归档、可重建的边界、删与不删 |
| [配置](docs/configuration.md) | 全部环境变量与默认值、从旧名字迁移 |
| [API](docs/api.md) | `Memory` 门面、HTTP 路由、适配器矩阵 |
| [对比](docs/comparison.md) | 与 mem0 / MemOS / Zep / TencentDB-Agent-Memory 的机制差异 |

## 参与

见 [CONTRIBUTING.md](CONTRIBUTING.md)。提交前必须全绿：
`ruff check .` · `ruff format --check .` · `mypy` · `python -m pytest`。

## 许可

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
