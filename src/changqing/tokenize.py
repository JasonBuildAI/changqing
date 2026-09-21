"""中文分词：jieba 优先，没装就退化成字符 bigram。

**为什么必须有这一层。** FTS5 默认的 unicode61 把连续汉字当成**一个** token。
实测：插入「团子三岁了」，查「团子」命中 0、查「三岁」命中 0 —— 中文检索基本
失效。jieba 与 bigram 两种切法都能救回来（同一组查询 4/4 命中），所以装了用
jieba、没装用 bigram，功能不降级成「搜不到」。

**为什么必须预热。** 实测 `import jieba` 862ms、首次 `lcut` 732ms（要建词典并写
缓存），稳态只要 0.087ms。惰性加载的话第一次冷路径检索要 0.8–1.6 秒，而检索的
预算是几十毫秒 —— 更糟的是它发生在「用户说了一句话之后」那条路上，
正好压在首字延迟那条命门上。所以启动时先热一次（`warm()`）。
"""

from __future__ import annotations

import re
import threading

# FTS5 的 MATCH 表达式里，双引号包起来才是「词组」；不包的话 AND/OR/NOT/NEAR
# 这些词会被当成操作符，用户随口说一句「not bad」就能把检索打挂。
_SAFE = re.compile(r"[^\w\u4e00-\u9fff]+")

_JIEBA = None
_LOCK = threading.Lock()
_BACKEND = "unknown"


def _load():
    """惰性 import jieba，只成功一次。失败就永久退化成 bigram。"""
    global _JIEBA, _BACKEND
    if _BACKEND != "unknown":
        return _JIEBA
    with _LOCK:
        if _BACKEND != "unknown":
            return _JIEBA
        try:
            import jieba
            jieba.setLogLevel(jieba.logging.ERROR)   # 别让它往 stdout 打建词典的日志
            _JIEBA = jieba
            _BACKEND = "jieba"
        except Exception:                            # noqa: BLE001  可选依赖
            _JIEBA = None
            _BACKEND = "bigram"
    return _JIEBA


def backend() -> str:
    """当前用的是哪个分词器（自检与面板用）。"""
    _load()
    return _BACKEND


def tokenize(text: str) -> list[str]:
    """切成检索用的 token。标点与空白丢掉 —— 它们对召回没贡献，只会撑大索引。"""
    s = str(text or "")
    if not s.strip():
        return []
    jb = _load()
    if jb is not None:
        return [w for w in jb.lcut(s) if _SAFE.sub("", w)]
    # bigram 降级：没有词典，就把相邻两字当一个单位（「猫叫团」→「猫叫 叫团」）
    chars = [c for c in s if _SAFE.sub("", c)]
    if len(chars) < 2:
        return chars
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def to_index(text: str) -> str:
    """给 FTS5 的 `text_index` 字段用：空格分隔的 token 串。"""
    return " ".join(tokenize(text))


def to_query(text: str) -> str:
    """给 FTS5 的 MATCH 表达式用：每个 token 加引号再 OR 起来。

    引号是必须的（见 `_SAFE` 的注释）；一个 token 都没有时返回空串，
    调用方要跳过 FTS —— 空 MATCH 表达式在 SQLite 里是语法错误。
    """
    toks = []
    for t in tokenize(text):
        t = t.replace('"', "")
        if t:
            toks.append(f'"{t}"')
    return " OR ".join(toks)


def warm() -> str:
    """启动预热：把词典建好、缓存落盘。返回分词器名字。"""
    _load()
    try:
        tokenize("今天画室很安静，光线很好。")
    except Exception:                                # noqa: BLE001
        pass
    return _BACKEND

