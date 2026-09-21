"""巩固与遗忘：让事实数不无序膨胀，让该记得的浮上来。

两件事，都写成**追加的操作**（MERGE / DEROGATE）而不是改库：

  1. **合并同槽位重复**：同一个 (subject, predicate) 下值相同或高度相似的多条
     合成一条，保留信息量最大的（引文最长、置信度最高）。被并的标 `merged`
     —— **不删**，「并到哪一条」记在 MERGE 操作的 `merged_into` 里（事实行
     内容一字不改）。「合并了两条记忆」和「丢了一条记忆」是两回事，而且
     MERGE 是日志里的一行，整库可以从 L0 + log.jsonl 重放出来。

  2. **降权**：长期没被用过、又没什么重要性的，把 importance 压下去。
     它不会消失（还能查到），只是不再抢热路径的位置。
     钉住的、承诺类的、画像里很在意的**永不降权** —— 权重是「她该多快想起
     这件事」的旋钮，不能用来惩罚一句还没兑现的承诺。

  3. **清出索引**：不再是 active 的行从全文索引里摘掉。它们仍在 facts 表里，
     仍然可审计 —— 但检索不该再命中一个已经失效的值。

为什么降权也要进日志：importance 是**排序依据**。悄悄改权重等于悄悄改变
她记得什么 —— 那正是这套系统存在的理由的反面。

**合并之后要剪派生行**：被并的那几条已经没有任何读取方会碰（status=merged），
而它们的内容在胜者那一条里，原始依据在 L0 与 log.jsonl 里。留着只会让
facts 表随对话场次线性膨胀 —— 而上万用户 × 上百场对话正是这个库的设计规模。
`superseded` 的行**不剪**：那是「他换工作了」这种真实变化，少一条就少一段
可审计的历史。
"""

from __future__ import annotations

import re
import time
from typing import Any

from .runtime import runtime
from .store import append_op, list_facts, materialize, open_index
from .tokenize import tokenize

# 永不降权的 kind：承诺与约定是「她欠他的一件事」，与「她记不记得」无关，
# 而 importance 是排序依据 —— 拿它去压承诺等于让一句答应过的话悄悄沉底。
_PROTECTED_KINDS = ("commitment", "promise")


def _similar(a: str, b: str, *, floor: float) -> float:
    """两条 fact 的值有多像 —— 用**覆盖率**而不是 Jaccard。

    为什么不是 Jaccard：真实重复几乎总是「同一件事，一次说得粗、一次说得细」
    （「团子」vs「团子（三岁）」）。Jaccard 把「细的那条多出来的词」算成不相似
    （1/2 = 0.5），于是永远合不了；覆盖率问的是「短的那条有没有被长的完全覆盖」，
    这才是「同一件事」的正确判据。

    反向的风险是过度合并，所以门槛不低（默认 0.8），而且
    「不吃香菜」vs「不吃芹菜」这种仍然是 0.5，不会被并掉。
    """
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 1.0 if a.strip() == b.strip() else 0.0
    overlap_n = len(ta & tb)
    contain = overlap_n / min(len(ta), len(tb))
    jaccard = overlap_n / len(ta | tb)
    # 覆盖率之外再加一道 Jaccard 地板：只看覆盖率的话，
    # 「一部老电影」和「另一部老电影」也是 1.0（短的那条被完全覆盖），
    # 可它们的差别恰好是决定意思的那个词。地板挡住这类「包含但不同」。
    return contain if (contain >= floor and jaccard >= floor * 0.6) else 0.0


def _core_len(s: Any) -> int:
    """去掉标点空白之后的长度。选胜者要比的是**内容量**，不是字符数。"""
    return len(
        re.sub(r"[\s，。！？、；：,.!?;:…—\-~～·|/\\\"'“”‘’（）()\[\]【】〔〕]+", "", str(s or ""))
    )


def _best_of(group: list[dict]) -> dict:
    """一组重复里留哪条：**内容量最大**的优先。

    比的是去掉标点之后的长度，不是原始长度 —— 否则「香菜。」会赢过「香菜」，
    纯粹因为多一个句号。这是同一个坑的第二层：更早一版按 quote 长度选，
    挑中的是带句号的那条噪声；改成 object 长度之后，句号**照样**能赢。
    内容一样时取更短的那条（干净的那个），再其次看置信度和 id。
    """
    return sorted(
        group,
        key=lambda f: (
            -_core_len(f.get("object")),
            len(str(f.get("object") or "")),
            -float(f.get("confidence") or 0.0),
            str(f["id"]),
        ),
    )[0]


def find_duplicates(uid: str) -> list[list[dict]]:
    """找出同槽位下值相同或高度相似的重复组（每组 >= 2 条）。"""
    floor = float(runtime().config.merge_similarity)
    groups: dict[tuple, list[dict]] = {}
    for f in list_facts(uid):
        groups.setdefault((f["subject"], f["predicate"]), []).append(f)
    out: list[list[dict]] = []
    for _slot, items in groups.items():
        if len(items) < 2:
            continue
        used: set = set()
        for i, a in enumerate(items):
            if a["id"] in used:
                continue
            same = [a]
            for b in items[i + 1 :]:
                if b["id"] in used:
                    continue
                if _similar(str(a["object"]), str(b["object"]), floor=floor) >= floor:
                    same.append(b)
                    used.add(b["id"])
            if len(same) >= 2:
                used.add(a["id"])
                out.append(same)
    return out


def consolidate(uid: str, *, now: float = 0.0, dry_run: bool = False) -> dict[str, Any]:
    """跑一轮巩固。返回统计。**绝不抛异常**（它跑在后台线程里）。

    `dry_run` 只算不写：自检与面板要能回答「现在跑一轮会动什么」，而不真的动。
    `now` 是给自检用的时间注入点（正常路径不要传）。
    """
    cfg = runtime().config
    now = now or time.time()
    # `pruned` / `swept` 也要有初值：dry_run 时不进那两步，缺了它读取方拿到的是
    # KeyError —— 而「算一算会动什么」正是自检与面板最常用的那一次调用。
    stats: dict[str, Any] = {
        "merged": 0,
        "derogated": 0,
        "swept": 0,
        "pruned": 0,
        "kept": 0,
        "ops": 0,
    }
    try:
        ops: list[dict] = []

        # ---- 1. 合并重复
        for group in find_duplicates(uid):
            keep = _best_of(group)
            for f in group:
                if f["id"] == keep["id"]:
                    continue
                ops.append({"op": "MERGE", "id": f["id"], "merged_into": keep["id"]})
                stats["merged"] += 1

        # ---- 2. 降权（长期没用 + 不重要）
        cutoff = now - cfg.consolidate_days * 86400
        for f in list_facts(uid):
            if f["pinned"] or str(f.get("kind")) in _PROTECTED_KINDS:
                stats["kept"] += 1
                continue
            last = str(f.get("last_used_at") or f.get("valid_from") or "")
            try:
                ts = time.mktime(time.strptime(last[:10], "%Y-%m-%d"))
            except (ValueError, TypeError):
                ts = 0.0  # 没有日期：按「从没用过」处理
            if ts >= cutoff:
                stats["kept"] += 1
                continue
            imp = float(f.get("importance") or 0.0)
            new_imp = max(cfg.decay_floor, round(imp * 0.7, 3))
            if new_imp < imp:
                ops.append({"op": "DEROGATE", "id": f["id"], "importance": new_imp})
                stats["derogated"] += 1
            else:
                # 已经压到地板了：再追加一条同样权重的操作只是让日志变长，
                # 而下一次巩固还会再算一遍同样的结果。
                stats["kept"] += 1

        # ---- 3. 先落库，再收拾派生层
        # 顺序不能反：后面两步动的是**已经不是 active** 的行，而 MERGE / DEROGATE
        # 刚写下的状态还在日志里没进库 —— 先收拾的话那批刚被改掉的事实还是
        # active，一条都碰不到，索引里继续留着它们的词条。
        if not dry_run:
            for op in ops:
                append_op(uid, op)
            if ops:
                materialize(uid)
            stats["swept"] = _sweep_index(uid)
            stats["pruned"] = _prune_merged(uid)
        stats["ops"] = len(ops)
        stats["ok"] = True
    except Exception as e:  # noqa: BLE001  巩固失败不能影响说话
        stats["ok"] = False
        stats["error"] = str(e)
    return stats


def _sweep_index(uid: str) -> int:
    """把不再是 active 的行从全文索引里摘掉，返回摘了几条。

    索引是 external content 表：索引里的词条跟 `facts.text_index` 那一列连着，
    把那一列置空，触发器就把词条摘了。**这不是删数据** —— 事实行还在 facts
    表里（`facts.md` 的「已失效」一节照样列得出来），只是检索不该再命中一个
    已经作废的值。

    摘之前不能省：失效的事实留在索引里只会是噪声（bm25 里它们的行还在，
    词条还占着倒排表的位），而「她想起一件早就作废的事」不会报任何错。

    失败返回 0 而不是抛：清理做不动最多是索引脏一点，绝不能让派生层的家务
    把整轮巩固判成失败。
    """
    con = open_index(uid)
    try:
        rows = con.execute(
            "SELECT rowid FROM facts WHERE status != 'active' AND coalesce(text_index,'') != ''"
        ).fetchall()
        swept = 0
        for r in rows:
            con.execute("UPDATE facts SET text_index='' WHERE rowid=?", (r["rowid"],))
            swept += 1
        con.commit()
        return swept
    except Exception:  # noqa: BLE001
        return 0
    finally:
        con.close()


def _prune_merged(uid: str) -> int:
    """剪掉 `status='merged'` 的行，返回剪了几行。

    失败返回 0 而不是抛：这是清理，清理不了最多是表大一点，
    绝不能让「派生层的家务」把这一轮巩固整体判成失败。
    """
    con = open_index(uid)
    try:
        cur = con.execute("DELETE FROM facts WHERE status='merged'")
        pruned = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        con.commit()
        return pruned
    except Exception:  # noqa: BLE001
        return 0
    finally:
        con.close()


__all__ = ["consolidate", "find_duplicates"]
