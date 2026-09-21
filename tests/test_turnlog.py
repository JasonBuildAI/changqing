"""L0：只追加的原始对话。这一层**一个字都不能丢**。

事实、纪要、话题全都能从原话重新抽出来，原话自己不能 —— 它是唯一不可再生的
那一层。所以这里每一条断言都在守同一件事：**写进去什么，读回来就是什么**。

原话的存储形状是人能直接读的 markdown（一天一个文件、文件里按小时分节），
代价是「按行解析」。由此长出两条必须钉住的判据：

  · 一条消息必须**单行完整**地写进去（正文里的换行要转义），否则下面那条判据不成立；
  · 文件末尾那半行（进程被杀在写中间）必须**丢掉但不抛异常**。

第二条有一个非常容易漏的分支：前缀**完全合法**（有时间、有角色）只是正文被
截断的半行。正则认它、只看正则就会把它当成一整条消息，于是后台整理从半句话里
抽出一条「事实」。判据只能是「文件末尾有没有换行」。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from changqing.runtime import Runtime
from changqing.store import (
    append_turn,
    day_path,
    load_state,
    read_turns,
    remember,
    safe_name,
    save_state,
    stats,
    user_dir,
    watermark,
)

UID_A = "u" + "a" * 16
UID_B = "u" + "b" * 16
# 「现在」在这一份里**只取一次**。`append_turn` 是按 ts 自己算日期与小时的，
# 所以期望值必须来自**同一个** ts：期望值在导入时取、写入时又取一次的话，
# 跨过整点的那一次运行会掉在两条记录中间 —— 那是环境抖，不是代码错
# （真踩过：`## {HOUR}` 那条断言在整点后 0.1 秒变红，单跑又是绿的）。
TS = time.time()
DAY = time.strftime("%Y-%m-%d", time.localtime(TS))
HOUR = time.strftime("%H:%M", time.localtime(TS))


def turn(user: str, assistant: str, ts: float | None = None) -> dict:
    return {"user": user, "assistant": assistant, "ts": ts or TS}


# ---------------------------------------------------------------- 追加写
def test_a_turn_becomes_one_file_per_day_with_stable_ids(rt: Runtime):
    """一天一个文件；日期与小节标题**只插一次**；轮次 id 单调递增。"""
    ids1 = append_turn(UID_A, turn("我家猫叫团子，三岁了", "……嗯，团子。我记住了。"))
    assert len(ids1) == 2, "一轮两个 id"
    assert tuple(ids1) == ("T-000001", "T-000002"), "id 格式"

    ids2 = append_turn(UID_A, turn("今天画了一下午", "画室很安静吧。"))
    assert ids2[0] == "T-000003", "id 连续递增"

    body = day_path(UID_A, DAY).read_text(encoding="utf-8")
    assert body.count(f"# {DAY} 对话") == 1, "日期标题只出现一次"
    assert body.count(f"## {HOUR}") == 1, "小时标题只出现一次"


def test_what_he_said_comes_back_word_for_word(rt: Runtime):
    append_turn(UID_A, turn("我家猫叫团子，三岁了", "……嗯，团子。我记住了。"))
    rows = read_turns(UID_A, DAY)
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[0]["text"] == "我家猫叫团子，三岁了"
    assert rows[1]["text"] == "……嗯，团子。我记住了。"


def test_her_messages_keep_their_boundaries(rt: Runtime):
    """她一轮里发好几条，**每一条各自留一行**。

    并成一段之后，「她当时是分两次说的」这件事就永远查不回来了 —— 而整理是从
    这些行里抽事实的，边界没了，抽出的事实也跟着糊。
    """
    ids = append_turn(UID_A, {"user": "在吗", "assistant": ["嗯", "怎么了"], "ts": TS})
    assert len(ids) == 3
    assert [r["role"] for r in read_turns(UID_A)] == ["user", "assistant", "assistant"]


def test_she_can_be_remembered_before_she_answers(rt: Runtime):
    """他先说完、她还没回的时候也要能记：只写他那一半。"""
    append_turn(UID_A, {"user": "我先睡了啊", "ts": TS})
    assert [r["role"] for r in read_turns(UID_A)] == ["user"]


# ---------------------------------------------------------------- 单行完整
def test_original_text_stays_on_one_line(rt: Runtime):
    """正文里的换行要转义。不转义的话，下面「丢掉尾部残行」那条判据不成立 ——
    一条带换行的原话会被读成两条，后半截还没有角色。
    """
    raw = "第一行\n第二行\r\n第三行"
    append_turn(UID_B, turn(raw, "嗯。"))
    assert len(read_turns(UID_B, DAY)) == 2, "两条消息"

    lines = day_path(UID_B, DAY).read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if ln.startswith("- T-")]
    assert len(hits) == 2, "两条消息两行"
    assert all("\n" not in ln for ln in hits), "正文里没有裸换行"
    assert read_turns(UID_B, DAY)[0]["text"] == raw, "转义之后仍能原样还原"


# ---------------------------------------------------------------- 半截行
def _append_raw(uid: str, text: str) -> None:
    with open(day_path(uid, DAY), "a", encoding="utf-8", newline="\n") as f:
        f.write(text)


def test_a_torn_tail_is_dropped_without_raising(rt: Runtime):
    """进程被杀在 write 中间：那半条既没有角色也没有时间，正则不认它。"""
    append_turn(UID_B, turn("第一行\n第二行\r\n第三行", "嗯。"))
    before = len(read_turns(UID_B, DAY))
    _append_raw(UID_B, "- T-000099 12:0")

    rows = read_turns(UID_B, DAY)
    assert len(rows) == before, "残行被丢掉，而且不抛异常"
    assert not any(r["id"] == "T-000099" for r in rows), "它没有冒充成一条消息"


def test_a_torn_tail_with_a_legal_prefix_is_also_dropped(rt: Runtime):
    """**更容易漏的那一种**：前缀完全合法（有时间、有角色），只是正文被截断。

    正则认它 —— 只看正则就会把它当成一整条，让后台整理从半句话里抽出一条
    「事实」。判据只能是「文件末尾有没有换行」。
    """
    append_turn(UID_B, turn("第一行\n第二行\r\n第三行", "嗯。"))
    before = len(read_turns(UID_B, DAY))
    _append_raw(UID_B, "\n- T-000100 12:05:33 user 他说了一半")

    rows = read_turns(UID_B, DAY)
    assert len(rows) == before, "前缀合法的半截行也要丢"
    assert not any(r["id"] == "T-000100" for r in rows)


def test_a_torn_tail_does_not_swallow_the_next_turn(rt: Runtime):
    """**残行不能吃掉下一条记录。** 这是本层真踩过的一个数据丢失路径。

    上一个进程被杀在 write 中间，留下的半行没有换行符。此时朴素地
    `open(path, "a")` 会把下一条轮次直接接在残行后面：

        - T-000099 12:0- T-000100 12:05:33 user 他刚说的话

    读侧只丢掉坏那一行，而**被接上去的那一整条也在同一行里** —— 于是它跟着一起
    被丢掉。如果是原话，那就是**永久丢失**（L0 不可再生）；如果是操作日志，
    那条操作再也不会被物化。两种都不报错，读侧只是照常跳过。

    所以「丢掉尾部残行」这条判据之外，还必须有一条「断开换行再写」。
    """
    append_turn(UID_B, turn("第一行\n第二行", "嗯。"))
    before = len(read_turns(UID_B, DAY))
    _append_raw(UID_B, "- T-000099 12:0")  # 半行，没有换行符

    ids = append_turn(UID_B, turn("他刚说的话", "嗯。"))
    rows = read_turns(UID_B, DAY)
    assert len(rows) == before + 2, f"新的一轮被残行吃掉了：{rows}"
    assert ids[0] == "T-000003"
    assert rows[-2]["text"] == "他刚说的话", "他刚说的那句话读得回来"
    assert not any(r["id"] == "T-000099" for r in rows), "残行自己仍然被丢掉"


def test_a_legacy_tagged_line_still_parses(rt: Runtime):
    """写侧早就不产标签了，但**读侧仍认**：磁盘上已有的老文件还在。

    这条钉的是单向兼容。读侧哪天顺手把标签解析删掉，老文件里的正文就会多出
    一截 `[shy_warm@0.4] [low_pressure_opening]`，而它看起来只是「她说话怪怪的」。
    """
    append_turn(UID_A, turn("在吗", "嗯。"))
    _append_raw(
        UID_A,
        "- T-000003 12:00:01 assistant [shy_warm@0.4] [low_pressure_opening] 老格式行\n",
    )
    last = read_turns(UID_A, DAY)[-1]
    assert last["tags"] == ["shy_warm@0.4", "low_pressure_opening"], "老标签被解析出来"
    assert last["text"] == "老格式行", "正文里没有标签残留"


# ---------------------------------------------------------------- 水位
def test_the_watermark_advances_with_every_turn(rt: Runtime):
    """水位是「整理到哪儿了」的落点，乱推一格就是「她再也想不起他刚说的话」。"""
    append_turn(UID_A, turn("我家猫叫团子，三岁了", "……嗯，团子。我记住了。"))
    append_turn(UID_A, turn("今天画了一下午", "画室很安静吧。"))
    wm = watermark(UID_A)
    assert wm.get("last_turn_id") == "T-000004", "指到最后一个 id"
    assert wm.get("rounds") == 2, "轮数累加"
    assert wm.get("day") == DAY
    assert stats(UID_A)["turns"] == 4


def test_a_new_turn_never_erases_the_extraction_cursor(rt: Runtime):
    """写它的是后台整理，`append_turn` 只负责**保留**。

    这里真踩过：早期写法把 `extracted_upto` 写成常量 `None`，于是每一轮都把它
    抹掉一次 —— 那个字段永远是空的，界面一直显示 null，而事实明明抽出来了。
    """
    append_turn(UID_A, turn("我家猫叫团子", "嗯。"))
    st = load_state(UID_A)
    st["watermark"] = dict(st.get("watermark") or {}, extracted_upto="T-000001")
    save_state(UID_A, st)
    append_turn(UID_A, turn("又聊了一句", "……嗯。"))
    assert watermark(UID_A).get("extracted_upto") == "T-000001"


# ---------------------------------------------------------------- 写不动
def _block_the_user_dir(root: Path, uid: str) -> None:
    """把 uid 那一级**占成一个文件**，于是建目录必然失败。"""
    path = user_dir(uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x", encoding="utf-8")


def test_append_turn_reports_an_unwritable_dir_honestly(rt: Runtime, mem_root: Path):
    """底层如实抛。往上那一层（`remember`）才是负责吞的地方。"""
    uid = "u" + "c" * 16
    _block_the_user_dir(mem_root, uid)
    with pytest.raises(OSError):
        append_turn(uid, turn("在吗", "嗯。"))


def test_remember_is_silent_so_she_can_still_speak(rt: Runtime, mem_root: Path):
    """对调用方必须是静默的：记忆写不动也得出得了声。

    它通常跑在「刚说完一句话」的那条路上，那里的一次磁盘异常不该让对话崩掉。
    """
    uid = "u" + "d" * 16
    _block_the_user_dir(mem_root, uid)
    assert remember(uid, turn("在吗", "嗯。")) is None


# ---------------------------------------------------------------- 隔离
def test_two_people_never_see_each_other(rt: Runtime):
    append_turn(UID_A, turn("我家猫叫团子", "嗯。"))
    append_turn(UID_B, turn("我养的是一条蛇", "哦。"))
    assert user_dir(UID_A) != user_dir(UID_B), "两个 uid 目录不同"
    assert not any("蛇" in r["text"] for r in read_turns(UID_A))
    assert not any("团子" in r["text"] for r in read_turns(UID_B))


def test_a_uid_cannot_climb_out_of_its_own_bucket(rt: Runtime):
    """路径注入：uid 里的分隔符必须被剥掉，`../` 不能真的往上走。

    不剥的后果不是报错，而是**所有 uid 都写进同一个目录** —— 于是所有人都看得见
    所有人的原话，而日志里一行异常都没有。
    """
    assert safe_name("../../etc/passwd") == "etcpasswd"
    assert "/" not in safe_name("a/b")
    assert "\\" not in safe_name("a\\b")
    assert safe_name("") == "default", "空名字不许拼出一个空目录名"
