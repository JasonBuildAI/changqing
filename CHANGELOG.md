# Changelog

> English summary: every notable change to this project is recorded here. The format
> follows Keep a Changelog and the version numbers follow Semantic Versioning.

本项目的每一处值得写下来的改动都记在这一份里。
格式沿用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号沿用[语义化版本](https://semver.org/lang/zh-CN/)。

**读者该知道的只有一条**：`0.y.z` 期间，`__all__` 里那批名字保证向后兼容
（有快照测试钉住，见 [API](docs/api.md)）；名字**之外**的内部模块
（`store` / `extract` / `retrieve` / `worker` …）可以有，但不承诺稳定。

## [Unreleased]

### Added

- **`her_self`：她自己说过的当轮就进候选卡片**（上游 2026-09-23 的口径）。
  `context.MEMORY_SECTIONS` 新增 `("her_self", "你说过但还没核实", "她说的")`：
  模型这轮说出的、关于她自己的生活细节由宿主当轮写进会话，下一轮起卡片上就有 ——
  后台整理要等触发，这段时差里同一件事只能靠上下文发挥，于是她会今天说去过苏州、
  明天说没出过远门。归属留在她这一侧（并进他的候选分组，人称整个反过来）；
  它在这份清单里，门控（`has_recall_material`）与渲染一次到位。

### Changed

- **抽取口径扩到「她自己的事」**：她的行不再只收 `kind=promise`（她答应的事），
  `subject=她` 的偏好 / 习惯 / 正在做的事 / 去过的地方一并进事实层（闸 3）。
  她复述他的话仍然一律丢弃 —— 那是把他的信息反向当成他提供的。
  `subject` 在落库前收口到两个值（她的那一侧用画像的称呼，其余是「他」）。
- `extract_tokens_per_turn` 默认 `160 → 200`：一批里也夹着她的行，输出比只抽他时更长。
- 后台整理送给模型的是**她的整行**（不再按承诺关键词预筛），上界从 6 条 × 120 字
  放宽到 16 条 × 160 字；超过条数上界时丢**最旧**的那几条（她刚说的最可能被再提起）。

### Fixed

- **空正文不再被当成「没有可抽的」**：抽取返回空响应时按失败走（游标不推进、下次重试）。
  原先空响应与「没配模型」共用 `""` 这一个返回值，上游把它读成合格的空答案 ——
  形状是「抽取 0 条、没有 error、游标照常推进」，同一条链路重跑就绿了。
  契约要的是一份 JSON，空白永远不是合格答案。

## [0.1.0] - 2026-09-21

第一个版本。四层记忆、依赖全部注入、零第三方运行时依赖。

### Added

- **四层记忆**：L0 原话（只追加、超期只归档）、L1 事实（只追加的操作日志 +
  可重建的 SQLite 物化视图）、L2 纪要、L3 画像权重。
  见 [架构](docs/architecture.md) 与 [存储](docs/storage.md)。
- **抽取的低幻觉七道闸**：回引轮次、回引校验、角色归属、时间绝对化、清洗、
  冲突消解、幂等。见 [抽取](docs/extraction.md)。
- **检索的热 / 冷两条路径**：热路径常数预算，冷路径按需触发，槽位 / 全文 /
  向量三路 RRF 融合，再按画像权重与时间衰减重排，最后按预算裁剪。
  见 [检索](docs/retrieval.md)。
- **后台整理**：三个触发条件（静默 / 累计轮数 / 手动催）、游标、幂等与单场调用上限；
  巩固期的衰减与合并。
- **依赖注入**：`Embedder` / `LLM` / `UsageSink` 三个 Protocol 加默认替身，
  库本身不调任何模型；不注入 `Embedder` 就等于关掉向量召回。
- **公开门面** `Memory` + `MemoryConfig`：`remember` / `recall` / `forget`
  三方法契约，外加 mem0 风格的 `add` / `search` / `delete` / `get` / `get_all` /
  `update` / `history` 别名。见 [API](docs/api.md)。
- **可注入的画像** `PersonaProfile`：机制保留（每条事实带 `persona_attention`），
  内容归宿主，默认是一份中性画像。
- **可挂载的服务端**（`changqing[server]`）：相对路径的 FastAPI 路由，
  挂载前缀由宿主决定；uid 一律从服务端签发的身份取，不接受客户端传参。
- **适配器**：离线确定性的 `MockEmbedder` / `MockLLM`（默认能跑，不需要 Key），
  以及 OpenAI 兼容的 `OpenAIEmbedder` / `OpenAILLM`。
- **示例与基准**：`examples/quickstart.py`（无 Key 跑通全链路）、
  `benchmarks/scale.py`（离线合成，看长期相处的规模与成本）、
  `benchmarks/extraction_eval.py`（抽取质量评测，需要真实 Key）。

### Notes

- 许可证 Apache-2.0。版权只出现在 `LICENSE` 与 `NOTICE`，**源码文件不带逐文件版权头**。
- 不发布到 PyPI。安装方式是从仓库 `pip install -e .`。
- 支持 Python 3.10 及以上。
- 本仓库的文档为中文，每份开头附三行英文摘要；README 中英双语。

[Unreleased]: https://github.com/JasonBuildAI/changqing/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JasonBuildAI/changqing/releases/tag/v0.1.0
