# API

> English summary: the public surface is one facade (`Memory`) over one frozen config
> (`MemoryConfig`), three methods carry the contract (`remember` / `recall` / `forget`)
> with mem0-style aliases beside them, and every external capability arrives through an
> injected protocol. The optional server extra exposes the same thing as relative paths.

公开面就是下面这些。它被 `tests/test_memory.py` 逐条钉住（含一份 `__all__` 快照）
—— 接口**静默漂移**比接口报错贵得多：报错当场就发现，漂移要等到有人按文档写代码
的时候。

```python
import changqing

changqing.__all__  # 就是这一页说的东西，没有别的
```

## 装什么

```bash
pip install -e .            # 核心：零第三方运行时依赖
pip install -e ".[openai]"  # 加 OpenAI 兼容的适配器
pip install -e ".[server]"  # 加可挂载的 FastAPI 路由
pip install -e ".[chinese]" # 加中文分词（不装也能跑，只是分词退化成按字切）
pip install -e ".[all]"     # 上面全部
```

核心包**不 import 任何第三方库**。`import changqing` 不需要装 FastAPI、
不需要分词器、不需要 HTTP 客户端。

## 门面：`Memory`

```python
Memory(
    uid,
    *,
    config=None,      # MemoryConfig；不给就跟着 runtime() 走
    persona=None,     # PersonaProfile；不给就是中性画像
    embedder=None,    # ports.Embedder；不给就是 NullEmbedder（向量整支关掉）
    llm=None,         # ports.LLM；不给就是 NullLLM（抽取会大声报错）
    usage=None,       # ports.UsageSink；不给就是不记账
)
```

**它是句柄，不是单例。** 一个进程里可以有很多个 `Memory`，每个绑一个 uid。
构造器上只记**写下来的**那几项覆盖，能力在**调用时**解析 —— 所以构造完之后
再 `configure(llm=...)`，已经建好的句柄一样看得到。反过来（构造时拷一份）会让
`Memory` 与 `runtime()` 变成两个真源，而它们不一致时**不报错**，
表现是「我明明配好了，它就是不用」。

`uid` 为空字符串时，`remember` / `add` 与 `recall` / `search` **直接返回空结果**，
不碰盘 —— 这是给「还没登录」的那一轮留的路：没有身份时既不该写下原话，
也不该读到任何东西。（其余方法没有这道门：它们查的是一个空的库，
返回的本来就是空。）

### 三方法契约

| 方法 | 签名 | 说明 |
|---|---|---|
| `remember` | `(turn: dict) -> list[str]` | 追加一轮原话（L0），返回轮次 id。`turn` 是 `{"user": ..., "assistant": ...}`，可以带 `ts`。**只追加、不修改**：这一层不可再生 |
| `recall` | `(query, *, budget_tokens=0, timeout_ms=0, k=0) -> list[dict]` | 按一句话检索事实。**绝不抛异常**：超时或分数不够就是空列表。三个 `0` 都表示「用配置里的默认值」（显式写 0 的人想要的是「按默认预算裁」，不是「一条都不给」） |
| `forget` | `(fact_id, mode="") -> str` | 忘记一条。`mode` 留空走 `CHANGQING_FORGET_MODE`（默认 `archive`，可恢复）。返回**实际用的那个 mode** |

### mem0 风格的那一套名字

写惯了 mem0 的人不必先学一套新词。**别名就是别名**，是同一个方法对象，
不是第二份实现 —— 第二份实现只在有人改了一边之后才与另一边分家，
而那已经是几个月以后的事了（`test_the_aliases_point_at_the_same_implementation`
钉住了这一条）。

| 别名 | 就是 | 备注 |
|---|---|---|
| `add(messages, *, ts=None)` | `remember` 的家族 | 收三种形状：一轮 dict、一个字符串（一轮用户独白）、`[{"role", "content"}, …]`（按「用户 → 助手」的边界切）。切分**只在角色切换处**发生：用户连说两句算一轮，助手连发三条也算一轮 |
| `search(query, …)` | `recall` | 参数一模一样 |
| `delete(fact_id, mode="")` | `forget` | |
| `get(fact_id)` | —— | 取一条，没有就是 `None` |
| `get_all(*, include_dead=True)` | —— | 全部事实。默认**含已失效的**：这里看的是「她记过什么」，不是「现在还能不能用」 |
| `update(fact_id, **fields)` | —— | 手改一条。只认 `EDITABLE_FIELDS`，其余键静默忽略；坏值抛 `EditError` |
| `history(fact_id="")` | —— | 从 `log.jsonl` 里挑出跟这条有关（或全部）的操作。**它不是第二套机制**，只是把操作日志按 id 过滤出来 |

### 其余读与维护

| 方法 | 返回 | 说明 |
|---|---|---|
| `context(query, *, recent_refs=None, proactive=False)` | dict | 这一轮要注入的**全部**素材。与 `recall` 的分工：`recall` 回答「跟这句话有关的事有哪些」，`context` 回答「这一轮该把什么摆到她面前」 |
| `pending()` | list | 待确认的事实（回引近似命中的那批） |
| `summaries(limit=20)` | list | L2 纪要 |
| `topics(*, limit=8, include_used=False)` | list | 主动话题 |
| `confirm(fact_id, accept=True)` | str | 确认 / 否决。**它就是「恢复」**：确认时会把有效期一起放回来 |
| `pin(fact_id, pinned=True)` | bool | 钉住。钉住的事实无条件进热路径 |
| `note_summary(text, day="", turn_ref="")` | str | 直接补一条纪要 |
| `note_topic(text, day="", *, kind="share", due_day="")` | str | 补一条「她下次想提的事」 |
| `reindex(*, batch=0, limit=0)` | dict | 把向量索引补到最新。没注入 `Embedder` 时是空操作 |
| `extract_now(*, call=None)` | dict | 立刻整理一次（不等后台的三个触发条件）。同步、会调模型 |
| `stats()` | dict | 轮数、事实数、向量数、话题数 |
| `export()` | dict | 全量导出，**含操作日志** —— 有了它就能把这份记忆完整重建 |
| `reset(mode=None)` | dict | 整库重置，返回 `{"mode", "leftover"}`，见 [存储](storage.md) |
| `effective()` | `Runtime` | 这个句柄当下实际生效的配置与能力 |

### 事实卡片长什么样

`recall` / `search` / `context` 里的事实都是同一形状（`dict`），
外加一个 `_score`（只在排过序的结果里）：

| 键 | 说明 |
|---|---|
| `id` | `F-000001` 风格，全局递增（不复用） |
| `subject` · `predicate` · `object` | 三段式的正文。渲染出来就是「他养的猫叫团子」 |
| `kind` | `fact` / `commitment` / `promise` |
| `confidence` · `importance` · `persona_attention` | 抽取给的三个权重（0–1） |
| `valid_from` · `valid_to` | 有效期。`valid_to` 非空 = 已经失效（**失效不是删除**） |
| `status` | `active` / `forgotten` / `merged` / … |
| `pinned` · `due` · `source` | 钉住、到期日、这条从哪来 |
| `turn_ref` · `quote` | **回引**：产生这条事实的轮次 id 与那段原话。每条事实都能追回它，这是这套系统敢说自己不编的依据 |
| `last_used_at` · `use_count` | 派生层记账，只影响排序 |
| `_score` | 融合后的分数（`rel × (0.75 + 0.25 × qual) × recency`） |

`context()` 的返回多几样：`facts` / `summaries` / `topics` 三份素材，
加上 `hot` · `cold`（各几条）、`used_tokens`（**三份加起来的实际注入量**）、
`ms`、`backend`、`reason`。`reason` 是「冷路径为什么是空的」：
`hot_only` / `no_fact` / `no_hit` / `timeout` / `error` ——
**超时与没命中都返回空，但处置完全相反**，所以必须能分开。

## 手改一条事实：`EDITABLE_FIELDS`

```python
from changqing import EDITABLE_FIELDS

EDITABLE_FIELDS  # ('subject', 'predicate', 'object', 'importance',
#  'persona_attention', 'due', 'kind')
```

只有这几个字段可以手改。`id` / `turn_ref` / `quote` / `status` 不在里面：
前三个是**证据**（改了它这条事实就不可追了），`status` 有自己的操作
（`confirm` / `forget`），绕过去会让日志与物化视图对不上。

**别把这张名单在别处再抄一份。** 它曾经在多处各有一份，其中一处漏掉了承诺类
—— 结果只有「她答应过他的事」时 recall 会被静默挡掉，而卡片里明明写着这条。
现在只有一处定义，路由直接引它。

## 三个 Protocol

外部能力全部注入，形状在 `changqing.ports`：

| 能力 | 必填成员 | 不注入会怎样 |
|---|---|---|
| `Embedder` | `name` · `dim` · `repo` · `ready(download=True)` · `encode(texts)` · `warm()` · `loaded()` | `NullEmbedder`：向量召回整支关掉，降级到槽位 + 全文检索 |
| `LLM` | `__call__(messages, *, model="", max_tokens=0, on_usage=None, on_finish=None) -> str` | `NullLLM`：抽取抛 `LLMNotConfigured`，**大声报错而不是静默返回空串** |
| `UsageSink` | `note_llm(*, tin=0, tout=0, **extra)` | `NoUsage`：不记账 |

三条容易写错的约定：

- **`encode` 失败返回 `None`，不抛异常。** 向量是锦上添花的一路，模型没加载好、
  网络断了都不该让这一轮取不到任何记忆。
- **`NullLLM` 抛异常是有意的。** 静默返回空串的症状是「她什么都记不住」，
  而日志里一行异常都没有 —— 这种缺陷要花几个小时才能定位。
- **`repo` 是编码器身份**（仓库 + 维度），不只是给人看的名字：换一个**同维度**
  的模型时，旧向量必须被认出来，不能与新向量混在一张表里。

### 现成的适配器

| | 离线替身（核心包，零网络） | OpenAI 兼容（`changqing[openai]`） |
|---|---|---|
| `Embedder` | `MockEmbedder(dim=64)` —— 按字符 bi-gram 做的确定性哈希向量 | `OpenAIEmbedder(api_key, *, model="text-embedding-3-small", base_url=..., dim=0)` |
| `LLM` | `MockLLM(responses=None, default="")` —— 按顺序吐回写死的字符串 | `OpenAILLM(api_key, *, model="gpt-4o-mini", base_url=..., temperature=0.0)` |

```python
from changqing import Memory, MemoryConfig
from changqing.adapters.mock import MockEmbedder, MockLLM

mem = Memory(
    "user-1",
    config=MemoryConfig(root="./data"),
    embedder=MockEmbedder(),
    llm=MockLLM(default='{"facts": []}'),
)
```

两个替身不是「测试用的假货」，而是这个库**默认就该能跑起来**的那条路：
先看见它工作，再决定接哪个模型。`examples/quickstart.py` 就是用它们跑的，
不需要任何 API Key。

OpenAI 兼容的那两个：

- **`api_key` 必须显式传**，适配器**不读环境变量** —— 与 [配置](configuration.md)
  那条「只有 `config.py` 读环境变量」是同一条规矩。要环境变量就在构造处自己读一次。
- `base_url` 可以指任何 OpenAI 兼容端点（自建网关、其他厂商都行）。
- 可以注入现成的 `client`，于是它们能在**不联网**的情况下被测到。
  **谁建的谁关**：注入进来的 client 不会被 `close()` 关掉。
- `EMBED_DIMS` 是已知嵌入模型的维度表；用了表里没有的模型就显式传 `dim=`，
  否则构造时直接报错（而不是等索引写完才发现维度不对）。

## 服务端（`changqing[server]`）

```python
from fastapi import FastAPI
from changqing.server import create_router

app = FastAPI()
app.include_router(create_router(), prefix="/api")  # 挂上去就是 /api/memory
```

**路径是相对的，前缀归宿主决定。** 写死前缀就得让每个宿主去挪自己的路由表，
而「挪一下」这件事在别人的代码里通常等于不改，于是要么冲突、要么把记忆挂到一个
谁也没想到的地址上。

```python
create_router(
    *,
    uid_of=...,     # (Request) -> str，决定「这是谁」
    memory_of=...,  # (uid) -> Memory，宿主注入配置与模型的地方
)
```

出厂 `uid_of` 读 `request.state.uid`（宿主的鉴权中间件写的那个），取不到就 **401**，
**绝不退回空串或默认用户** —— 退回默认用户的症状是「所有人都看同一个人的记忆」，
而它不会报任何错。

### 路由表

| 方法 | 路径 | 请求体 | 成功返回 |
|---|---|---|---|
| `GET` | `/memory` | —— | `facts` · `pending` · `summaries` · `topics` · `stats`。默认含已失效 / 已提过的：这里看的是**她到底记过什么** |
| `POST` | `/memory/edit` | `{"id", "set": {…}}` | `{"ok", "fact"}`。没有可改的字段 → 400；值不合法 → 400（**在查库之前判**，否则用户会以为是 id 的问题）；没有这条 → 404 |
| `POST` | `/memory/forget` | `{"id", "mode"?}` | `{"ok", "mode", "note"}`。`note` 如实说明还剩什么没删 |
| `POST` | `/memory/confirm` | `{"id", "accept"}` | `{"ok", "status"}` |
| `POST` | `/memory/pin` | `{"id", "pinned"}` | `{"ok", "pinned"}` |
| `GET` | `/memory/export` | —— | 全量导出，`note` 说明操作日志是唯一事实源 |

### 这一域共享的那条铁律

> **uid 一律从服务端签发的身份取，绝不接受客户端传参。**

`/memory?uid=xxx` 是典型的越权读取（IDOR）：改一个字符就能读别人的记忆。
所以这里**没有任何**一个接口接受 uid。宿主用签名 Cookie 或网关注入都行，
换的是 `uid_of`，不是这里的路由。

这个模块要 FastAPI 与 pydantic，**它不在核心包里被 import** ——
`changqing/__init__.py` 一个字都不提它，装不装 FastAPI 都不影响 `import changqing`。
它也不含任何前端：界面是宿主的形状，不是这个库的。

## 相关文档

- 一轮注入哪些、按什么排 → [检索](retrieval.md)
- 每个配置项与环境变量 → [配置](configuration.md)
- 与 mem0 / MemOS / Zep / TencentDB-Agent-Memory 的机制差异 → [对比](comparison.md)
