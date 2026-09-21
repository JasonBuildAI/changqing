"""对话流水（L0 原话）：单行化、追加、读回。

`sessions/YYYY-MM-DD.md` 是**唯一不可再生**的那一层：用户真的说过的话。
它的全部关注点是「单行完整 + 崩溃只丢尾行」，所以这套追加语义值得单独一个文件。

轮次 id（`T-000001`）全局递增、**永不复用** —— 它是事实回引（`turn_ref`）的锚点。
先占号再写盘：崩溃最多浪费一个号，绝不会重号。
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from ..runtime import runtime
from .paths import _load_state, _lock, _save_state, day_path, sessions_dir

_TURN_RE = re.compile(r"^- (T-\d{6,}) (\d{2}:\d{2}:\d{2}) (user|assistant) (.*)$")

# 只有长得像 `YYYY-MM-DD` 的才当 L0 的一天：备份、手写笔记混进 `sessions/` 时
# 不该被当成「某个月的原话」处理。
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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
        rest = rest[end + 1:].lstrip()
    return tags, rest


def _fmt_line(tid: str, ts: float, role: str, text: str,
              tags: list[str] | None = None) -> str:
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

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(out))
            f.flush()
            os.fsync(f.fileno())      # 崩溃时最多丢最后一条，且是残行

        rounds = int(st.get("rounds") or 0) + 1
        # **单独记「有用户发言的轮数」**：抽取的游标必须按它走，不能按 rounds。
        # 主动开口的轮（他一个字都没说）只写 assistant 行，而抽取只认 user 行 ——
        # 两个计数一旦错位，游标会被推过用户真正说过的话，症状是「她什么都记不住」
        # 而且**不报错**。
        user_rounds = int(st.get("user_rounds") or 0) + \
            (1 if any(r == "user" for r, _ in msgs) else 0)
        st.update({
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
        })
        _save_state(uid, st)
    return ids


def remember(uid: str, turn: dict[str, Any]) -> None:
    """追加一轮到 L0。**绝不抛异常、绝不阻塞说话**。

    写入不进热路径是刻意的：首字延迟的大头是模型的 TTFT，记忆不能挡在前面。
    """
    if not runtime().config.enabled:
        return
    try:
        append_turn(uid, turn)
    except Exception:                 # noqa: BLE001  记忆写不动也得能说话
        pass


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
            continue                  # 标题、空行、以及被截断的残行
        tid, when, role, payload = m.groups()
        tags: list[str] = []
        if role == "assistant":
            tags, payload = _parse_tags(payload)
        rows.append({"id": tid, "day": day, "hour": hour, "time": when,
                     "role": role, "tags": tags, "text": unescape_text(payload)})
    return rows


def _day_sources(uid: str) -> list[tuple[str, Path]]:
    """这个 uid 的每一天各自住在哪个文件。返回**按天排好序**的 `(day, path)`。"""
    live = sessions_dir(uid)
    found: dict[str, Path] = {}
    for p in (sorted(live.glob("*.md")) if live.is_dir() else []):
        if _DAY_FILE_RE.match(p.stem):
            found[p.stem] = p
    return [(d, found[d]) for d in sorted(found)]


def read_turns(uid: str, day: str | None = None) -> list[dict[str, Any]]:
    """按顺序读回 L0。`day` 为空则读全部。

    解析时**丢掉尾部不完整的那一行**：进程被杀在 write 中间时文件末尾会留下
    半条记录，直接抛异常会让整个记忆层打不开。「追加 + 单行完整」换来的容错，
    不需要 replace 那种重量级方案。
    """
    out: list[dict[str, Any]] = []
    for dd, p in _day_sources(uid):
        if day and dd != day:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # 读不动就当这一天读不到：一天读不出来不该让整份记忆打不开。
            continue
        out.extend(_turn_rows(dd, text))
    return out
