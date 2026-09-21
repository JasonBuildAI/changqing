# 配置

> English summary: every knob is a field on one frozen dataclass, `MemoryConfig`, and
> `config.py` is the only module in the package that reads the environment. Variables use
> the `CHANGQING_` prefix; the older `MEMORY_*` / `EMBED_*` families map onto the new names
> by prefix substitution.

这一页是**全部可调项**的清单，也是从旧名字搬过来时的对照表。
名字与默认值的真源是 `src/changqing/config.py` 的 `ENV_MAP` 与字段本身 ——
下面那张表由它渲染而来，两者由 `tests/test_config.py` 钉住一一对应
（漏一个字段的症状是「这个开关读不到，而它看起来完全正常」）。

## 一条规矩：只有 config.py 读环境变量

```python
from changqing import MemoryConfig

cfg = MemoryConfig.from_env()  # 读 CHANGQING_*
cfg = MemoryConfig.from_env(prefix="APP_")  # 换一个前缀
cfg = MemoryConfig(root="./data", hot_tokens=200)  # 或者直接写死
```

为什么把这一条立成规矩：这个包的前身是一份宿主应用，它把几十个配置常量在
**导入时**拷进每个子模块（`from ... import MEMORY_DIR`）。那样做的症状是
**改不生效、而且不报错** —— 测试里把根目录换成临时目录，另一个模块拷的那份
还指着真实数据目录，于是测试写进了生产数据。所以这里定死：值只在
`MemoryConfig.from_env()` 里读一次，读出来的东西是一个冻结的数据类，
谁要就往构造器里传。

## 读取语义

四条，都写在 `_env_*` 那四个小函数里：

| 规则 | 为什么 |
|---|---|
| **空串当没设置** | `CHANGQING_DIR=` 是常见的「我没改，用默认」写法。不当没设置的话，`os.environ` 给的是空串，根目录会变成当前目录 —— 用户眼里就是「我什么都没改，记忆库突然搬家了」 |
| **数字写坏 → 默认值** | 写错一行不该让整个库起不来 |
| **布尔**：`0` / `off` / `no` / `false`（不分大小写）为假，其余非空为真 | 让人按自己的习惯写，而不是逼着大家记 `0`/`1` |
| **路径里的 `~` 展开** | 不展开的话每个用户目录都会叫一个字面量的 `~` |

## 全部变量

### 开关与位置

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | `CHANGQING_ENABLED` | `1` | 总开关。关掉之后整条链路不写不读 |
| `root` | `CHANGQING_DIR` | `~/.changqing` | 库根目录。放主目录而不是当前目录：默认用法是个长期存在的服务进程，写「启动时的工作目录」意味着换个地方启动就读不到昨天的会话了 |
| `reset_mode` | `CHANGQING_RESET_MODE` | `purge` | `reset()` 的默认行为：`purge` 真删 / `archive` 改名留档 |
| `forget_mode` | `CHANGQING_FORGET_MODE` | `archive` | `forget()` 的默认行为。`archive` 只标失效，可恢复 |
| `retain_months` | `CHANGQING_RETAIN_MONTHS` | `24` | 超过这个月数的 L0 原话搬进 gzip。**`0` = 永不清理**；负数也夹回 0 |
| `render_views` | `CHANGQING_RENDER_VIEWS` | `1` | 要不要顺手渲染 `facts.md` 之类的给人看的视图。自检里成百上千次写操作时关掉能省一堆文件 I/O |

### 注入预算

这几个数直接决定**每轮花多少 token**，是成本的主要闸门。
它们的取数逻辑见 [检索](retrieval.md)。

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `hot_tokens` | `CHANGQING_HOT_TOKENS` | `300` | 热路径注入的 token 上限。它把「记忆块随轮次膨胀」变成一条常数线 |
| `hot_fetch_max` | `CHANGQING_HOT_FETCH_MAX` | `400` | 热路径从库里取多少条候选再说。取数是 SQL 里的 `LIMIT`，与打分同序 |
| `story_k` | `CHANGQING_STORY_K` | `3` | 一次注入几条 L2 纪要 |
| `story_tokens` | `CHANGQING_STORY_TOKENS` | `160` | 纪要那一份的 token 预算 |
| `topic_k` | `CHANGQING_TOPIC_K` | `2` | 一次注入几条主动话题 |
| `topic_tokens` | `CHANGQING_TOPIC_TOKENS` | `80` | 话题那一份的 token 预算 |
| `recall_k` | `CHANGQING_RECALL_K` | `4` | 冷路径最多返回几条 |
| `recall_ms` | `CHANGQING_RECALL_MS` | `50` | 冷路径的墙钟预算。**超时的结果是整条冷路径返回空**（原因记在 `retrieve.last_error()`，不抛异常） |
| `min_score` | `CHANGQING_MIN_SCORE` | `0.35` | 融合后的分数门槛。真正把关的是它，不是向量余弦 |
| `min_confidence` | `CHANGQING_MIN_CONFIDENCE` | `0.5` | 抽取置信度门槛。低于它的候选连进入打分的资格都没有 |
| `pending_tolerance` | `CHANGQING_PENDING_TOLERANCE` | `0.6` | 两条事实相似到这个程度就算「同一个槽位在换值」，走冲突消解 |
| `rrf_k` | `CHANGQING_RRF_K` | `60` | RRF 融合的平滑常数。调大 = 更平均，调小 = 更看头名 |

### 向量召回

注意这里**没有** provider / model / mirror：向量能力由宿主注入 `Embedder`
（见 [API](api.md) 的适配器矩阵），库本身不带任何模型下载逻辑。
不注入就整支关掉。

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `embed_dim` | `CHANGQING_EMBED_DIM` | `512` | 向量维度。至少要 `1` |
| `embed_min_cos` | `CHANGQING_EMBED_MIN_COS` | `0.48` | 余弦相似度的召回门槛 |
| `embed_batch` | `CHANGQING_EMBED_BATCH` | `32` | 写索引时一次编几条 |
| `embed_scan_max` | `CHANGQING_EMBED_SCAN_MAX` | `2000` | 一次向量召回最多扫多少条索引行。**这是纯 Python 相似度的成本闸门**：512 维 2000 条在本机实测约 44ms，与 `recall_ms` 的 50ms 是同一量级 —— 调大它就得同时想清楚 `recall_ms` 给多少 |

### 后台整理

三个触发条件与巩固的细节见 [架构](architecture.md) 与 [抽取](extraction.md)。

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `idle_min` | `CHANGQING_IDLE_MIN` | `30` | 静默超过这么多分钟就整理一次 ——「用户不说了」是最自然的时机 |
| `max_turns` | `CHANGQING_MAX_TURNS` | `200` | 距上次整理累计这么多轮也整理一次：长对话中途也要落一次 |
| `extract_max_calls` | `CHANGQING_EXTRACT_MAX_CALLS` | `8` | **单场对话**的整理调用上限。没有它，一次整理会按轮数把账单乘上去 |
| `extract_model` | `CHANGQING_EXTRACT_MODEL` | 空 | 抽取用哪个模型。留空 = 交给注入的 `LLM` 自己决定 |
| `extract_tokens_per_turn` | `CHANGQING_EXTRACT_TOKENS_PER_TURN` | `160` | 折算一次整理要读多少 token 的输入 |
| `extract_max_tokens` | `CHANGQING_EXTRACT_MAX_TOKENS` | `8000` | 单次整理的输出上限 |
| `consolidate_days` | `CHANGQING_CONSOLIDATE_DAYS` | `90` | 多久没被用过的事实开始衰减 |
| `decay_floor` | `CHANGQING_DECAY_FLOOR` | `0.05` | 衰减的下限。**不取 0**：降到 0 等于判死，之后永远注入不进来 |
| `merge_similarity` | `CHANGQING_MERGE_SIMILARITY` | `0.8` | 两条事实相似到这个程度才算重复、可以合并 |
| `commitment_slots` | `CHANGQING_COMMITMENT_SLOTS` | `3` | 热路径给「她答应过他的事」留几个位置 |

### 上下文窗口

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `history_window` | `CHANGQING_HISTORY_WINDOW` | `24` | 宿主喂给模型的最近消息**条数**（这是调用方的选择，本库猜不出来）。记忆去重的窗口必须与它相等：小了 = 模型刚看过的话又被当记忆喂一遍；大了 = 掉出窗口的话因为还被排除，所以**永远**不再注入 |

### 唯一的空闲判据

| 字段 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `idle_split_sec` | `CHANGQING_IDLE_SPLIT_SEC` | `600` | **不许引入第二个空闲概念**。它既决定喂给模型的上下文何时重开，也决定「这一场」的边界（整理按场做） |

## 配置对象是冻结的

`MemoryConfig` 是 `frozen=True` 的。冻结不是洁癖：这个对象会在多个线程之间共享
（后台整理线程 + 前台请求），可变的话「这一轮读到一半被改了」是一种无法复现的
偶发。要改就造一份新的：

```python
from changqing import Memory, MemoryConfig, configure

configure(MemoryConfig.from_env())  # 进程级：一次装配，之后每处现取
mem = Memory("user-1", config=MemoryConfig(root="./data"))
mem.effective().config.hot_tokens  # 这个句柄当下实际生效的那一份
```

三种给法，优先级从高到低：

1. `Memory(..., config=...)` —— 只影响这个句柄；
2. `configure(config=...)` —— 影响整个进程；
3. 谁都不给 —— 用 `MemoryConfig()` 的出厂默认值。

`Memory` 只记**写下来的**覆盖项，能力在**调用时**解析（不在构造时）。构造完之后
再 `configure(llm=...)`，已经建好的句柄一样看得到 —— 构造时把运行期拷成一份的话，
`Memory` 与 `runtime()` 就成了两个真源，而它们不一致时**不报错**，
表现是「我明明配好了，它就是不用」。

## 从旧名字搬过来

这个包是从一份宿主应用里解耦出来的，那套变量名沿用 `MEMORY_` 与 `EMBED_`
两个前缀。现在统一收到 `CHANGQING_` 一个前缀下：

| 旧 | 新 | 备注 |
|---|---|---|
| `MEMORY_<X>` | `CHANGQING_<X>` | 后缀不变，整批做前缀替换即可 |
| `MEMORY_TURNS` | `CHANGQING_MAX_TURNS` | **唯一改了后缀的一个**（顺便换了名字：它管的是轮数上限） |
| `CHAT_IDLE_SPLIT_SEC` | `CHANGQING_IDLE_SPLIT_SEC` | 换前缀 |
| `EMBED_DIM` · `EMBED_MIN_COS` · `EMBED_BATCH` | `CHANGQING_EMBED_*` | 换前缀 |
| `EMBED_PROVIDER` · `EMBED_MODEL` · `EMBED_DIR` · `EMBED_MIRROR` | **没有对应项** | 向量能力改成注入 `Embedder`，库不再自带模型下载与镜像 |

新增（旧的那套里没有）：`CHANGQING_RENDER_VIEWS`、`CHANGQING_EMBED_SCAN_MAX`、
`CHANGQING_HISTORY_WINDOW`。

**两个前缀想一次替换掉的话**，`MemoryConfig.from_env(prefix="MEMORY_")` 能整批换，
但只覆盖 `MEMORY_*` 那一族：`CHAT_IDLE_SPLIT_SEC` 与 `EMBED_*` 得各自处理。
而且要留意这条路的失败姿态 —— **读不到的变量落回默认值，不报错**。所以
「换了前缀但漏了几项」的症状是「某个开关悄悄回到默认」，而不是启动失败；
`MEMORY_TURNS` 就是那个会静默失效的例子（它会去找 `MEMORY_MAX_TURNS`，
找不到就用默认的 200）。搬完最好对一遍上面那张表的每一行。

## 相关文档

- 一轮注入哪些、按什么排 → [检索](retrieval.md)
- 落盘格式与「归档 / 删除」的区别 → [存储](storage.md)
- 进程级装配的取舍与它的边界 → [架构](architecture.md)
