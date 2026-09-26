# changqing

> A memory system built for **long-lived conversational relationships**.

[中文](README.md) · [Architecture](docs/architecture.md) · [Design](docs/design.md) · [Changelog](CHANGELOG.md)

[![CI](https://github.com/JasonBuildAI/changqing/actions/workflows/ci.yml/badge.svg)](https://github.com/JasonBuildAI/changqing/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## The problem

An assistant that has talked with you for a hundred days and ninety thousand turns
should **actually remember you** — both the small thing you mentioned a minute ago
and the name you dropped three months ago.

Two common approaches fall short, and they fail differently:

- **Retrieval alone (RAG)**: what you just said is not in the index yet, so it
  cannot surface the sentence you finished a second ago;
- **Context alone**: growing the window eventually blows up — and gets expensive
  long before that, because every turn pays for a hundred days of history.

`changqing` splits the job: **the context window owns this turn, the memory system
owns "the longer it goes, the more it needs consolidating"**. Consolidation is done
by a model (extraction plus consolidation), not by piling raw utterances into the
prompt. Memory is organized into four layers, and **every fact can be traced back to
the words that produced it** — so "why does she remember this" always has an answer.

## Thirty seconds

**No API key, no network.**

```bash
pip install -e .
python examples/quickstart.py
```

It writes two turns, consolidates them into facts, and recalls them by question,
using offline doubles for the model and the embedder. Swapping in a real model is a
one-line change (see "Bring your own model").

```python
from changqing import Memory, MemoryConfig
from changqing.adapters.mock import MockEmbedder, MockLLM

mem = Memory(
    "user-1",
    config=MemoryConfig(root="./data"),
    embedder=MockEmbedder(),
    llm=MockLLM(default=extraction_json),
)
mem.remember({"user": "my cat is called Tuanzi", "assistant": "noted."})
mem.extract_now()  # in production a background thread drives this
for fact in mem.recall("what is my cat called?"):
    print(fact["predicate"] + fact["object"])
```

## The four layers

| Layer | Holds | Lives in | Nature |
|---|---|---|---|
| **L0 utterances** | the raw dialogue, turn by turn | `sessions/YYYY-MM-DD.md` | append-only; expired months are **archived** (gzip), never deleted |
| **L1 facts** | structured facts (his and her own) and promises | `log.jsonl` → `index.sqlite` | the log is irreplaceable, the index is rebuildable |
| **L2 summaries** | a short recap per conversation | `summaries` | prose with a budget, clip-able |
| **L3 persona** | weights on top of facts | `persona_attention` | decided by the injected `PersonaProfile` |

```
   what he says ──► L0 utterances (append-only)
                        │
                        │  background consolidation (3 triggers; 0 model calls per turn)
                        ▼
                     extraction ── 7 gates ──► L1 operation log (ADD / SUPERSEDE / …)
                                                   │  materialize
                                                   ▼
                                            index.sqlite (slots + full-text + vectors)
                                                   │
   this turn's words ──► retrieval: hot path (constant) + cold path (on demand)
                              ──► budget clipping ──► cards
                          plus L2 story line and proactive topics
```

## Three conclusions we do not revisit

1. **Retrieval alone is not enough** (what you just said is not indexed yet), and
   **context alone is not enough** (it will blow up).
2. **Consolidation must be done by a model**, not by heaping up raw text — but it is
   a **separate call**, not a reuse of the companion prompt: one wants to sound like
   her, the other wants to be strictly factual, and those goals fight each other.
3. **Facts without evidence do not enter the store.** Every fact carries a `turn_ref`
   and a `quote`, and the `quote` must be locatable inside the cited turn; a mismatch
   is dropped, a near match goes to "pending confirmation". Better to miss one than
   to remember one wrong — see [extraction](docs/extraction.md).

## Install

The core is **pure standard library, zero third-party runtime dependencies**:

```bash
pip install .                    # core: storage + extraction + retrieval + consolidation
pip install ".[openai]"          # OpenAI-compatible embedder and chat model
pip install ".[server]"          # mountable FastAPI router
pip install ".[chinese]"         # jieba tokenizer (falls back to character bigrams)
pip install ".[all]" ".[dev]"    # everything / development
```

`requires-python >= 3.10`. **Not published to PyPI**; install from the repository.

## Bring your own model

The library never calls a model itself: three capabilities are injected by the host
(Protocols in [`src/changqing/ports.py`](src/changqing/ports.py)).

| Capability | Does | If omitted |
|---|---|---|
| `Embedder` | turns text into vectors | vector recall is off (slot + full-text recall still work) |
| `LLM` | one completion (extraction, consolidation) | L0 is still written, nothing is consolidated; calling it raises loudly rather than returning empty |
| `UsageSink` | records usage | nothing is metered |

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

**The API key is explicit.** This library does not read `OPENAI_API_KEY`: reading the
environment is the host's job (per-tenant keys, secret managers), and a library that
quietly does it surprises you the day it starts making network calls. `base_url`
points at any OpenAI-compatible endpoint (self-hosted vLLM, Ollama, gateways).

## Mount it in a web service

```python
from fastapi import FastAPI
from changqing.server import create_router

app = FastAPI()
app.include_router(create_router(), prefix="/api")  # /api/memory is now the memory API
```

Paths are **relative**: the prefix belongs to the host. **No endpoint accepts a uid
parameter** — identity comes only from what the server itself issued (by default
`request.state.uid`), otherwise changing one character reads somebody else's memory.
See [API](docs/api.md).

## Cost

Hard-coded discipline: **zero memory-related model calls per conversation turn**, and
at most `extract_max_calls` (8 by default) consolidation calls per conversation. This
is not a micro-optimization; it is the difference between "usable by ten thousand
people every day" and "300 calls per conversation", two orders of magnitude apart.

The scale benchmark runs offline (synthetic dialogue plus offline doubles):

```bash
python benchmarks/scale.py                # shape: which layer holds the bytes, retrieval slope
python benchmarks/extraction_eval.py      # quality: needs a real key
```

**No number in there is a product promise.** Only magnitudes and slopes travel
between machines; capacity statements are written as assumptions plus arithmetic,
see [design](docs/design.md).

## Documentation

| Document | Covers |
|---|---|
| [Architecture](docs/architecture.md) | layer boundaries, data flow, process-level runtime trade-off |
| [Design](docs/design.md) | why RAG alone fails, why a model must consolidate, how capacity is computed |
| [Extraction](docs/extraction.md) | the seven low-hallucination gates, prompt rules, date normalization |
| [Retrieval](docs/retrieval.md) | hot and cold paths, RRF fusion, persona reranking, budget clipping |
| [Storage](docs/storage.md) | on-disk formats, archiving, what is rebuildable, what deletion means |
| [Configuration](docs/configuration.md) | every environment variable and default, migration from older names |
| [API](docs/api.md) | the `Memory` facade, HTTP routes, adapter matrix |
| [Comparison](docs/comparison.md) | mechanisms compared with mem0 / MemOS / Zep / TencentDB-Agent-Memory |

All documentation is written in Chinese with a three-line English summary at the top,
except this file and its Chinese sibling.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Everything must be green before you push:
`ruff check .` · `ruff format --check .` · `mypy` · `python -m pytest`.

## License

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).
