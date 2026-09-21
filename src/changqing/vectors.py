"""向量索引的**写入侧**。

三条纪律：

1. **只在后台整理里跑**。对话热路径（检索）永远只读，绝不在这里编码事实 ——
   那是每轮几十毫秒，而首字延迟只有 1-2 秒。
2. **幂等**。`vectors` 是派生层，重跑一次结果必须一样：判据是
   `(fact_id, dim, text_hash)`，内容没变就跳过。所以把库删了之后再跑一次就能
   全部补回来，不必重放日志。
3. **失败不影响说话**。模型没加载好、推理报错、写库失败 —— 都只是「这轮没有
   向量可召回」，全文检索与槽位召回还在。返回值里带 reason 供排查，但绝不往外抛。

**这个文件在没注入 `Embedder` 时整支不进**：`reindex` 返回
`{"ok": True, "skipped": "disabled"}`，`stats` 报零。向量是可选增强，
关掉它不该让任何调用点写 `if`。
"""

from __future__ import annotations

import array
import hashlib
from typing import Any

from .ports import Embedder, embedder_enabled
from .retrieve import card_text
from .runtime import runtime
from .store import open_index

# 活着、值得编码的事实状态。pending 也要编 —— 用户点了「确认」之后它就是
# active，那时候才补编码的话，刚确认的记忆当轮还搜不到。
LIVE_STATUS = ("active", "pending")

# `IN (?, ?, ...)` 的占位符。写成常量而不是在两处各 join 一遍：
# 两处的个数必须与 LIVE_STATUS 严格对齐，差一个就会静默少编码一类事实。
_MARKS = ", ".join("?" for _ in LIVE_STATUS)


def _text_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def _model_of(client: Embedder) -> str:
    """这批向量是**哪个编码器**编的（写进 `vectors.model` 的那个值）。

    取仓库名 / 模型名，不拼维度 —— 维度已经在单独的 `dim` 列里了，拼进来会让
    「同维度换模型」这件事看起来像换了两次。拿不到 repo 的实现（测试的桩、
    云端 provider）退回 `name`，**绝不留空**：空值会和老库的 NULL 混成一种情况，
    `model_mismatch` 就永远判不出来。

    为什么不是只看 `dim`：换一个同维度的模型时，旧向量在新模型的查询向量下
    相似度全是噪声，而库里混着两个向量空间这件事**没有任何报错** ——
    `stats()` 还会报 `stale=0`（看起来一切正常）。把身份写进索引，
    `reindex` 才能发现「这不是我编的」并整批重来。

    写成一个函数而不是各处现算：`reindex` 与 `stats` 必须算出**同一个**串，
    两处各写一份的话，改了一边就会变成「永远不一致」或「永远一致」。
    """
    repo = str(getattr(client, "repo", "") or "").strip()
    return repo or str(getattr(client, "name", "") or "unknown")


def _pack(vec: Any) -> bytes:
    """float32 小端打包。用 `array` 而不是 numpy：写入侧在后台线程，
    能少一个依赖就少一个。"""
    return array.array("f", [float(x) for x in vec]).tobytes()


def _local_model() -> str:
    """当前注入的编码器的身份。读不到、或压根没装，返回空串（宁可不判，不误判）。

    空串在这里是**有意义**的：`stats` 用它来回答「库里那批向量是不是它编的」，
    拿不到身份时唯一诚实的答案是「不知道」而不是「不匹配」——
    后者会让一个没装向量的人天天看到一条红告警。
    """
    try:
        emb = runtime().embedder
        if not embedder_enabled(emb):
            return ""
        return _model_of(emb)
    except Exception:  # noqa: BLE001  派生层的事不该往外抛
        return ""


def reindex(uid: str, *, batch: int = 0, limit: int = 0) -> dict[str, Any]:
    """把缺失 / 变旧的向量补齐，并清掉不该留的。返回统计。

    `limit` 只给自检用（一次只处理 N 条），线上传 0 = 不限。
    """
    client = runtime().embedder
    cfg = runtime().config
    if not embedder_enabled(client):
        return {"ok": True, "skipped": "disabled", "encoded": 0, "removed": 0}
    # **不许在这里加载模型**：这是后台整理线程，一次加载最坏要几十秒到两分钟，
    # 卡住的不是一个用户的首字、而是整条整理队列 —— 后面排队的人跟着一起等。
    # 模型只由宿主自己的预热去准备；还没好就先欠着，由维护循环在它就绪之后补
    # （worker 的 `_vector_backlog`）。
    if not client.ready(False):
        return {
            "ok": False,
            "reason": "embed_unavailable",
            "encoded": 0,
            "removed": 0,
            "error": getattr(client, "last_error", ""),
        }
    dim = int(client.dim or 0)
    model = _model_of(client)
    size = max(1, int(batch or cfg.embed_batch))

    con = open_index(uid)
    try:
        # 换过模型（哪怕维度一样）就**整批重编**：两个向量空间混在一张表里，
        # 相似度是噪声而没有任何报错，是这套索引最难查的一种坏法。
        # 更早的库里 model 是 NULL（这一列是后加的），同样按「不是当前模型」处理。
        prev = con.execute(
            "SELECT COUNT(*) AS n FROM vectors WHERE COALESCE(model,'')<>?", (model,)
        ).fetchone()
        stale_model = int(prev["n"] or 0) if prev else 0
        if stale_model:
            con.execute("DELETE FROM vectors")

        # 维度对不上、或事实已经不在世上的，先清掉。留着它们不只是占地方：
        # `SELECT ... WHERE dim=?` 读不到混维度的行，检索会静默少一半。
        cur = con.execute(
            "DELETE FROM vectors WHERE dim<>? OR fact_id NOT IN "
            f"(SELECT id FROM facts WHERE status IN ({_MARKS}))",
            (dim, *LIVE_STATUS),
        )
        removed = int(cur.rowcount or 0) + stale_model
        # **删完立刻提交**：下面「没有要补的」时整个函数直接返回，谁都不会再走到
        # 循环里那个 commit —— 而 sqlite3 默认是隐式事务，`close()` 会把未提交的
        # DELETE 一起丢掉。于是失效事实的向量永远清不掉：`removed` 每次都报 1
        # （DELETE 确实执行过），表里那行却一直在。
        con.commit()

        have = {
            r["fact_id"]: r["text_hash"]
            for r in con.execute("SELECT fact_id, text_hash FROM vectors").fetchall()
        }

        todo: list[tuple[str, str, str]] = []
        for row in con.execute(
            f"SELECT * FROM facts WHERE status IN ({_MARKS})",
            LIVE_STATUS,
        ).fetchall():
            f = dict(row)
            text = card_text(f)
            h = _text_hash(text)
            if have.get(str(f["id"])) != h:
                todo.append((str(f["id"]), text, h))
        if limit:
            todo = todo[: int(limit)]

        encoded = 0
        for i in range(0, len(todo), size):
            part = todo[i : i + size]
            vecs = client.encode([t for _, t, _ in part])
            if not vecs or len(vecs) != len(part):
                # 编码失败就停在这里，已经写进去的那几批留着（下次接着补）。
                # 半途中断留下的不是坏数据：每行都是完整的一条向量。
                con.commit()
                return {
                    "ok": False,
                    "reason": "encode_failed",
                    "encoded": encoded,
                    "removed": removed,
                    "error": getattr(client, "last_error", ""),
                }
            for (fid, _text, h), v in zip(part, vecs, strict=True):
                con.execute(
                    "INSERT INTO vectors(fact_id, dim, vec, text_hash, model) "
                    "VALUES(?,?,?,?,?) "
                    "ON CONFLICT(fact_id) DO UPDATE SET dim=excluded.dim, "
                    "vec=excluded.vec, text_hash=excluded.text_hash, "
                    "model=excluded.model",
                    (fid, len(v), _pack(v), h, model),
                )
            con.commit()
            encoded += len(part)
        return {
            "ok": True,
            "encoded": encoded,
            "removed": removed,
            "pending": len(todo) - encoded,
            "dim": dim,
            "model": model,
        }
    except Exception as e:  # noqa: BLE001  派生层坏了不该带走主流程
        return {
            "ok": False,
            "reason": "error",
            "encoded": 0,
            "removed": 0,
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        con.close()


def stats(uid: str) -> dict[str, Any]:
    """这个用户的向量索引状态。面板与自检用。**只读，不碰模型权重。**

    `model` 是**当前编码器**的身份，`model_mismatch` 回答「库里那批不是它编的」
    —— 这两个字段是「换过模型、索引还没跟上」那个窗口里**唯一**的可观测点。

    为什么单看 `stale`（事实数 - 向量数）不够：换一个**同维度**的模型时，
    `facts == vectors == stale=0`，面板一片绿，而检索已经在拿两个不同的向量空间
    算余弦了 —— 结果是「她突然什么都想不起来了」，没有任何报错。所以这里坚持
    报出库里实际存在的每一个 `model` 取值，让「混了两个空间」有地方看得见。
    """
    out: dict[str, Any] = {
        "facts": 0,
        "vectors": 0,
        "stale": 0,
        "dim": 0,
        "model": "",
        "models": [],
        "model_mismatch": False,
    }
    cur_model = _local_model()
    out["model"] = cur_model
    try:
        con = open_index(uid)
    except Exception:  # noqa: BLE001  派生层坏了不该把面板也带走
        return out
    try:
        row = con.execute(
            f"SELECT COUNT(*) AS n FROM facts WHERE status IN ({_MARKS})",
            LIVE_STATUS,
        ).fetchone()
        out["facts"] = int(row["n"] if row else 0)
        row = con.execute(
            "SELECT COUNT(*) AS n, MAX(dim) AS d, "
            "COUNT(DISTINCT COALESCE(model,'')) AS m FROM vectors"
        ).fetchone()
        out["vectors"] = int(row["n"] if row else 0)
        out["dim"] = int((row["d"] if row else 0) or 0)
        out["stale"] = max(0, out["facts"] - out["vectors"])
        out["models"] = sorted(
            str(r["model"] or "")
            for r in con.execute(
                "SELECT DISTINCT COALESCE(model,'') AS model FROM vectors"
            ).fetchall()
        )
        # 空库、或没有当前编码器身份时不判 —— 那两种情况下「不匹配」都是误报。
        out["model_mismatch"] = bool(out["vectors"] and cur_model and out["models"] != [cur_model])
        return out
    except Exception:  # noqa: BLE001
        return out
    finally:
        con.close()


__all__ = ["LIVE_STATUS", "reindex", "stats"]
