# changqing

> A Chinese-first memory system for **long-lived conversational relationships**.

[中文](README.md)

`changqing` solves exactly one problem: an AI that has talked with you for a
hundred days and ninety thousand turns should **actually remember you** — both
the small thing you mentioned a minute ago and the name you dropped three
months ago.

It does not stuff every raw utterance into the context window (that will blow
up), and it does not rely on vector retrieval alone (what you just said is not
in the index yet). Memory is organized into four layers, a model does the
consolidation work, and **every fact can be traced back to the words that
produced it**.

## Status

Early development (`0.1.0`). The public API is not frozen yet.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

