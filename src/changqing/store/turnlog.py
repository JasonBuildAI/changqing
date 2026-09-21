"""对话流水（L0 原话）：单行化、追加、读回、超期归档。

`sessions/YYYY-MM-DD.md` 是**唯一不可再生**的那一层：用户真的说过的话。
它的全部关注点是「单行完整 + 崩溃只丢尾行」，所以这套追加语义值得单独一个文件。

轮次 id（`T-000001`）全局递增、**永不复用** —— 它是事实回引（`turn_ref`）的锚点。
先占号再写盘：崩溃最多浪费一个号，绝不会重号。

「归档」也在这个文件里，因为它改的是同一份文件（`sessions/*.md`）。
但方向和 `reset` 相反：`archive_old_turns` **一个字节都不删**，超期只是搬进
gzip 包，`read_turns` 透明读回两处。
"""

from __future__ import annotations

import contextlib
import gzip
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from ..runtime import runtime
from .paths import (
    _load_state,
    _lock,
    _save_state,
    append_text,
    day_path,
    sessions_dir,
    user_dir,
    watermark,
)

_TURN_RE = re.compile(r"^- (T-\d{6,}) (\d{2}:\d{2}:\d{2}) (user|assistant) (.*)$")

# 只有长得像 `YYYY-MM-DD` 的才当 L0 的一天：备份、手写笔记混进 `sessions/` 时
# 不该被当成「某个月的原话」处理。
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# 归档目录的名字与后缀。后缀是常量而不是字面量：`Path.stem` 只剥一层，
# `2026-01-05.md.gz` 的 stem 是 `2026-01-05.md` —— 那个坑在 `_day_sources`
# 与 `archive_old_turns` 两处都要躲开。
_ARCHIVE_NAME = "sessions.archive"
_GZ_SUFFIX = ".md.gz"


# ---------------------------------------------------------------- 单行化
def escape_text(text: str) -> str:
    """把原话压成**一行**。

    记录必须单行完整，否则「崩溃时丢掉尾部残行」这条容错就不成立 ——
    一条跨行记录被截断后，前半截看起来是完全合法的一行。
    换行转义而不是替换成空格：原话要**无损**，所以 `\\r` 与 `\\n` 分开转义，
    `unescape_text` 之后必须能一字不差地还原（含 CRLF）。
    """
    s = str(text or "").replace("\\", "\\\\")
    return s.replace("\r", "\\r").replace("\n", "\\n")


def unescape_text(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "r":
                out.append("\r")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _parse_tags(text: str) -> tuple[list[str], str]:
    """剥掉行首的 `[标签]`，返回（标签列表，剩下的正文）。"""
    tags: list[str] = []
    rest = text
    while rest.startswith("["):
        end = rest.find("]")
        if end < 0:
            break
        tags.append(rest[1:end])
        rest = rest[end + 1 :].lstrip()
    return tags, rest


def _fmt_line(tid: str, ts: float, role: str, text: str, tags: list[str] | None = None) -> str:
    when = time.strftime("%H:%M:%S", time.localtime(ts))
    prefix = ("[" + "] [".join(tags) + "] ") if tags else ""
    return f"- {tid} {when} {role} {prefix}{escape_text(text)}"


# ---------------------------------------------------------------- 追加
def append_turn(uid: str, turn: dict[str, Any]) -> list[str]:
    """把一轮（他说 + 她回）追加到当天的 md，返回写入的轮次 id。

    自带日期 / 小时标题：一天一个文件、文件内按小时分小节，这样人翻起来是一份
    能读的对话记录，而不是一堆裸行。标题只在需要时写，重复调用不会反复插同一行。
    """
    ts = float(turn.get("ts") or time.time())
    day = time.strftime("%Y-%m-%d", time.localtime(ts))
    hour = time.strftime("%H:%M", time.localtime(ts))

    msgs: list[tuple[str, str]] = []
    if str(turn.get("user") or "").strip():
        msgs.append(("user", str(turn["user"])))
    # 她这一轮可能发了好几条消息。原话的**边界**不能在这里并掉 ——
    # 一条并成一段之后，「她当时是分两次说的」这件事就永远查不回来了。
    assistant = turn.get("assistant")
    parts = assistant if isinstance(assistant, (list, tuple)) else [assistant]
    for part in parts:
        if str(part or "").strip():
            msgs.append(("assistant", str(part)))

    with _lock(uid):
        path = day_path(uid, day)
        st = _load_state(uid)
        out: list[str] = []

        fresh = (not path.exists()) or path.stat().st_size == 0
        if fresh:
            out.append(f"# {day} 对话\n\n")
            last_hour = None
        else:
            # 只有同一天才认 last_hour；跨天文件是新的，标题必须重写
            last_hour = st.get("last_hour") if st.get("day") == day else None
        if last_hour != hour:
            out.append(f"## {hour}\n\n")

        seq = int(st.get("next_seq") or 1)
        ids: list[str] = []
        for role, body in msgs:
            tid = f"T-{seq:06d}"
            seq += 1
            out.append(_fmt_line(tid, ts, role, body) + "\n")
            ids.append(tid)

        # 崩溃时最多丢最后一条，且是残行；而**下一条不能被它吞掉** ——
        # 尾部是残行时先断开换行（原因见 `paths.append_text`）。
        append_text(path, "".join(out))

        rounds = int(st.get("rounds") or 0) + 1
        # **单独记「有用户发言的轮数」**：抽取的游标必须按它走，不能按 rounds。
        # 主动开口的轮（他一个字都没说）只写 assistant 行，而抽取只认 user 行 ——
        # 两个计数一旦错位，游标会被推过用户真正说过的话，症状是「她什么都记不住」
        # 而且**不报错**。
        user_rounds = int(st.get("user_rounds") or 0) + (
            1 if any(r == "user" for r, _ in msgs) else 0
        )
        st.update(
            {
                "next_seq": seq,
                "day": day,
                "last_hour": hour,
                "rounds": rounds,
                "user_rounds": user_rounds,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)),
                "watermark": {
                    "last_turn_id": ids[-1] if ids else "",
                    "last_ts": ts,
                    "day": day,
                    "rounds": rounds,
                    # 整理游标：**保留**上一轮的值，不在这里重置。写它的只有
                    # 后台整理一处 —— 每一轮把它抹成 None 的话这个字段永远是空的，
                    # 而事实明明抽出来了（面板会一直显示 null）。
                    "extracted_upto": (st.get("watermark") or {}).get("extracted_upto"),
                },
            }
        )
        _save_state(uid, st)
    return ids


def remember(uid: str, turn: dict[str, Any]) -> None:
    """追加一轮到 L0。**绝不抛异常、绝不阻塞说话**。

    写入不进热路径是刻意的：首字延迟的大头是模型的 TTFT，记忆不能挡在前面。
    """
    if not runtime().config.enabled:
        return
    # 记忆写不动也得能说话：吞掉写入异常，但**不吞掉返回值**（没有返回值）。
    with contextlib.suppress(Exception):
        append_turn(uid, turn)


# ---------------------------------------------------------------- 读取
def _turn_rows(day: str, text: str) -> list[dict[str, Any]]:
    """把**一天**的文本解析成轮次行。"""
    if not text.endswith("\n"):
        # 末尾没有换行 = 最后一行是进程被杀在 write 中途的半条记录。
        # 判据**不能**是「正则认不认」：`- T-000100 12:05:33 user 他说了一半`
        # 这种残行的前缀完全合法，会被当成一整轮读进去，让后台整理从半句话里
        # 抽出一条「事实」。每写一条都带换行，所以缺换行就是残行。
        text = text.rsplit("\n", 1)[0] if "\n" in text else ""
    rows: list[dict[str, Any]] = []
    hour = ""
    for line in text.splitlines():
        if line.startswith("## "):
            hour = line[3:].strip()
            continue
        m = _TURN_RE.match(line)
        if not m:
            continue  # 标题、空行、以及被截断的残行
        tid, when, role, payload = m.groups()
        tags: list[str] = []
        if role == "assistant":
            tags, payload = _parse_tags(payload)
        rows.append(
            {
                "id": tid,
                "day": day,
                "hour": hour,
                "time": when,
                "role": role,
                "tags": tags,
                "text": unescape_text(payload),
            }
        )
    return rows


def _day_sources(uid: str) -> list[tuple[str, Path]]:
    """这个 uid 的每一天各自住在哪：`sessions/YYYY-MM-DD.md` 还是归档包。

    返回**按天排好序**的 `(day, path)`。排序必须在合并两处**之后**做：
    各排各的再拼，归档过的那几天会整段跑到别的天前面，读回来的顺序就与归档前
    不一样了 —— 而顺序正是这条改动的核心护栏。

    同一天两处都有时用 `sessions/` 里那份（它是只追加的那份权威，归档只是副本；
    归档器只有在逐字节确认同源之后才会删掉 live 那份）。
    """
    live = sessions_dir(uid)
    arch = _archive_dir(uid)
    found: dict[str, Path] = {}
    for p in sorted(live.glob("*.md")) if live.is_dir() else []:
        if _DAY_FILE_RE.match(p.stem):
            found[p.stem] = p
    for p in sorted(arch.glob("*" + _GZ_SUFFIX)) if arch.is_dir() else []:
        # 不能写 `p.stem`：`2026-01-05.md.gz` 的 stem 是 `2026-01-05.md`
        # （只剥一层后缀），拿去当「天」用会让 day 字段凭空多一个 `.md`。
        found.setdefault(p.name[: -len(_GZ_SUFFIX)], p)
    return [(d, found[d]) for d in sorted(found)]


def read_turns(uid: str, day: str | None = None) -> list[dict[str, Any]]:
    """按顺序读回 L0，**归档过的月份透明读回**。`day` 为空则读全部。

    解析时**丢掉尾部不完整的那一行**：进程被杀在 write 中间时文件末尾会留下
    半条记录，直接抛异常会让整个记忆层打不开。「追加 + 单行完整」换来的容错，
    不需要 replace 那种重量级方案。

    调用方看不到归档这件事：两处的天在合并之后统一排序、统一解析，
    归档前后逐条一致。
    """
    out: list[dict[str, Any]] = []
    for dd, p in _day_sources(uid):
        if day and dd != day:
            continue
        try:
            if p.name.endswith(_GZ_SUFFIX):
                with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
        except (OSError, EOFError):
            # 读不动就当这一天读不到：一天读不出来不该让整份记忆打不开。
            continue
        out.extend(_turn_rows(dd, text))
    return out


# ---------------------------------------------------------------- 归档
def _archive_dir(uid: str) -> Path:
    """归档目录：`sessions/` 的**同级兄弟**，不是它的子目录。

    同级是为了让 `sessions/*.md` 这一族 glob（统计、运维台的按天清单）不会被
    归档包混进来 —— 塞进子目录也能躲开 glob，但那样「这个人在盘上占多少」
    要在两棵子树里数。
    """
    return sessions_dir(uid).with_name(_ARCHIVE_NAME)


def _cutoff_month(months: int, *, now: float | None = None) -> str:
    """保留期里**最早的那个月**（`YYYY-MM`）：比它更早的月份全是超期。

    为什么截止点落在「某个月的 1 号」很要紧：判据因此简化成一次字符串比较，
    **不用看那天是几号** —— 月内切割（只搬月初那几天）得重写文件里的行，
    而 L0 的语义是「只追加、不可再生」：重写一次，崩溃时丢的就不再是尾行
    而是中段。

    `months` 数法：从当月往前数 `months` 个月、**当月算第 1 个**
    （`retain_months=24` = 当月 + 前 23 个月，正好 24 个月）。
    """
    t = time.localtime(time.time() if now is None else now)
    total = t.tm_year * 12 + (t.tm_mon - 1) - (int(months) - 1)
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _same_bytes(src: Path, gz: Path) -> bool:
    """`sessions/` 里那份与归档包里那份是不是**同一批字节**。

    只有确认同源才敢删 live 那份。坏掉 / 被截断的 `.gz` 一律判「不同源」：
    判错的方向必须是「留着原话」，不是「删了它」。
    """
    try:
        return gzip.decompress(gz.read_bytes()) == src.read_bytes()
    except (OSError, EOFError):
        return False


def _unlink_quiet(p: Path) -> None:
    """删不掉就算了（临时文件残留不致命，原话才致命）。"""
    with contextlib.suppress(OSError):
        p.unlink()


def archive_old_turns(uid: str, *, months: int | None = None) -> dict[str, Any]:
    """把**整月**都早于保留期的 L0 原话搬进 `sessions.archive/*.md.gz`。

    **这是归档不是删除**：原话一个字节都没少，只是换个 `.gz` 待着，
    `read_turns` 透明读回两处。真正的删除只有用户主动发起的那两条路。

    **按整月判、不在月内切**：只要这个月整体早于截止月，就搬走这个月的每一天。
    `months <= 0` = 永不清理、直接返回。

    **无损 + 幂等**靠三件事：

      1. 先把内容写进临时文件、再 `os.replace` 到 `.gz`（同目录内 = 原子），
         **最后**才 `unlink` 原 `.md`：崩在任何一步都只会「留下没搬完的」，
         绝不会先删后写；
      2. `.gz` 已经在的那一天不重复搬运 —— 只有逐字节确认两边同源
         （= 崩在 replace 与 unlink 之间）才补完最后那一步；
      3. gzip 的 `mtime=0`：同一份内容两次归档得到同一批字节。

    返回可观测的统计而不是 None —— 「今天到底搬没搬」要看得到。
    """
    out: dict[str, Any] = {
        "files": 0,
        "bytes": 0,
        "months": [],
        "cutoff": "",
        "skipped": 0,
        "finished": 0,
        "errors": [],
    }
    if months is None:
        months = runtime().config.retain_months
    try:
        months = int(months)
    except (TypeError, ValueError):
        return out  # 配置写坏了按「关掉」处理，不动盘
    if months <= 0:
        return out  # 永不清理
    out["cutoff"] = cutoff = _cutoff_month(months)

    moved: list[str] = []
    with _lock(uid):
        d = sessions_dir(uid)
        if not d.is_dir():
            return out
        try:
            files = sorted(p for p in d.glob("*.md") if _DAY_FILE_RE.match(p.stem))
        except OSError:
            return out
        for p in files:
            month = p.stem[:7]
            if month >= cutoff:
                continue  # 整月还在保留期里：这一个月一天都不动
            target = _archive_dir(uid) / f"{p.name}.gz"
            raw = b""
            try:
                if target.exists():
                    if _same_bytes(p, target):
                        p.unlink()  # 同源：崩在半路，补完最后那一步
                        out["finished"] += 1
                    else:
                        # 不同源 = 归档之后又有人往这一天写过（时间戳回拨）。
                        # 判断不了该信哪一份时**两边都留着** —— 原话不许丢。
                        out["skipped"] += 1
                    continue
                raw = p.read_bytes()
                target.parent.mkdir(parents=True, exist_ok=True)
                # 临时文件名带 pid 与线程 id：固定名会让两个写者互相搬走对方
                # 正在写的文件。
                tmp = target.with_name(f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp")
                try:
                    tmp.write_bytes(gzip.compress(raw, mtime=0))
                    os.replace(tmp, target)
                except OSError:
                    _unlink_quiet(tmp)
                    raise
                p.unlink()  # **只有 .gz 落好之后**才删原话
            except OSError as e:
                # 一个文件搬不动不拖累其余月份（Windows 上句柄占着是常态），
                # 但要如实报出来：静默 = 看不出「这几天一直没归档」。
                out["errors"].append({"day": p.stem, "error": str(e)})
                continue
            out["files"] += 1
            out["bytes"] += len(raw)
            moved.append(month)
    out["months"] = sorted(set(moved))
    return out


def archive_stats(uid: str) -> dict[str, Any]:
    """归档包的小结：`{"files": n, "bytes": n}`。

    `bytes` 是 `.gz` 在盘上占的字节（**压缩后**，回答「占多少盘」），而
    `archive_old_turns` 返回的 `bytes` 是搬走前原话的字节数（压缩前，回答
    「搬走了多少原话」）—— 两个数各有各的用处，别混着比。

    **只 stat，不读内容**：它跑在面板的自动刷新路径上。没归档过的用户返回
    两个 0，不建目录。
    """
    files = 0
    size = 0
    d = _archive_dir(uid)
    if d.is_dir():
        for p in d.glob("*" + _GZ_SUFFIX):
            try:
                size += p.stat().st_size
                files += 1
            except OSError:
                pass
    return {"files": files, "bytes": size}


def stats(uid: str) -> dict[str, Any]:
    """给自检与面板用的小结。

    **`days` / `bytes` 只算 `sessions/`（还没归档的那部分），`turns` 是两处之和**
    （它走 `read_turns`）—— 这对看起来会打架：全归档之后 `days == []` 而
    `turns > 0`。所以这里把归档那一侧**显式报出来**（`archived_days` /
    `archived_bytes`），让「少了的去哪儿了」在同一份返回值里看得见。
    """
    d = sessions_dir(uid)
    days = sorted(p.stem for p in d.glob("*.md")) if d.exists() else []
    turns = read_turns(uid)
    size = 0
    for p in d.glob("*.md"):
        with contextlib.suppress(OSError):
            size += p.stat().st_size
    arch = _archive_dir(uid)
    arch_days: list[str] = []
    arch_size = 0
    if arch.exists():
        for p in arch.glob("*" + _GZ_SUFFIX):
            arch_days.append(p.name[: -len(_GZ_SUFFIX)])
            with contextlib.suppress(OSError):
                arch_size += p.stat().st_size
    return {
        "uid": uid,
        "dir": str(user_dir(uid)),
        "days": days,
        "turns": len(turns),
        "bytes": size,
        "archived_days": sorted(arch_days),
        "archived_bytes": arch_size,
        "watermark": watermark(uid),
    }


# ---------------------------------------------------------------- 重置
def reset_memory(uid: str, mode: str | None = None) -> dict[str, Any]:
    """整库重置，返回**实际做了什么**：`{"mode", "leftover"}`。

    `purge`（默认）= 真删：用户在界面上的预期就是「删了」，而这是用户主动发起的
    整库操作（区别于系统内部的「事实失效不删」）。
    `archive` = 改名留档，可恢复。两种都可以，但**界面文案必须与之一致**。

    **为什么返回的不是一个模式字符串**：只要返回字符串，调用方就只能说「成功」。
    而这条路径上「删不掉」是常态而不是异常 —— Windows 上只要有一个打开的句柄
    （前台检索、后台整理随时可能正在读 `index.sqlite`），`shutil.rmtree` 与
    `Path.rename` **都会**失败。旧实现把失败吞进 `ignore_errors=True`：
    日志、原话、状态都删了，而库里的事实**原样留着** —— 她照样记得、
    接口却回成功；而且不可再生的那份已经没了，`rebuild()` 也救不回来。所以：

      1. **先把物化层清空**（`index.wipe`）：检索、面板、巩固读的都是那张库，
         空表 = 她一条都不记得，这一步与「文件能不能删掉」无关；
      2. 再删文件（日志 / 原话 / 状态这些不可再生的）；
      3. **如实报出没删掉的**（`leftover`），让调用方决定怎么告诉用户。
    """
    mode = (mode or runtime().config.reset_mode or "purge").lower()
    d = user_dir(uid)
    if not d.exists():
        return {"mode": mode, "leftover": []}

    if mode != "archive":
        # 1) 先让她真的不记得。
        # **这一步必须在下面那把 `_lock(uid)` 之外**：`index.wipe` 自己也要拿同一把
        # 锁，而 `threading.Lock` 不可重入 —— 套在里面会当场死锁。它自己拿锁，
        # 所以单独调用也是安全的。
        from .index import wipe  # 惰性：store 内部按包分层，别在导入期成环

        wipe(uid)

    with _lock(uid):
        if mode == "archive":
            stamp = time.strftime("%Y%m%d%H%M%S")
            target = d.with_name(f"{d.name}.archived-{stamp}")
            n = 1
            while target.exists():  # 同一秒里连点两次也不能互相覆盖
                target = d.with_name(f"{d.name}.archived-{stamp}-{n}")
                n += 1
            try:
                d.rename(target)
            except OSError:
                # archive **没有**「删掉」这层语义：改名失败就是没归档，必须抛。
                # 悄悄返回成功会让调用方以为留档成功了。
                raise
            return {"mode": mode, "leftover": []}

        with contextlib.suppress(OSError):
            shutil.rmtree(d, ignore_errors=True)  # 2) 再删文件，尽力而为
        leftover = sorted(p.name for p in d.iterdir()) if d.exists() else []
        return {"mode": mode, "leftover": leftover}
