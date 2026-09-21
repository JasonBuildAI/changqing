"""L0 归档：把过了保留期的原话换成 `.gz` 待着，**一个字节都不删**。

这一份用例只有一个核心判据，而它是这套设计唯一能证明「她什么都还记得」的东西：

    **归档前后 `read_turns` 逐条完全一致。**

id / day / hour / time / role / tags / text 一个字段都不许变，天的顺序也不许变。
必须拿归档前后的两份输出**直接比**：少一个字段、错一天顺序，表现都是「她记得
的过去变了」，而它不报错、不变红，只是安静地少一点。

日期一律**从「现在」往回推**，不写死年月。保留期是按当月算的，写死日期的话这些
用例过几个月就自己失效了 —— 而且失效的方式是**悄悄恒真**（永远归档不到东西，
断言却全过），不是变红。
"""

from __future__ import annotations

import gzip
import time

from changqing.runtime import Runtime
from changqing.store import (
    append_turn,
    archive_old_turns,
    archive_stats,
    read_turns,
    sessions_dir,
    stats,
)

BACK_DAYS = (0, 1, 3, 35, 38, 40, 70, 75, 80, 110, 120, 130)


def _seed_across_months(uid: str) -> list[str]:
    """造跨月原话，返回它们落在哪些「天」上（同月的几天落在不同文件里）。"""
    now = time.time()
    days: list[str] = []
    for i, back in enumerate(BACK_DAYS):
        ts = now - back * 86400.0
        append_turn(uid, {"user": f"第{i}条原话", "assistant": f"第{i}条回复", "ts": ts})
        days.append(time.strftime("%Y-%m-%d", time.localtime(ts)))
    return days


def _split(days: list[str], months: int = 1) -> tuple[list[str], list[str]]:
    """按「保留 months 个月」分成 (会被归档的, 留着的)。

    这里**独立写一遍**月份比较，不调实现里那个算截止月的函数 —— 直接抄实现的话，
    实现把截止月算错了这些用例也跟着错，两边一起绿。
    """
    cur = time.strftime("%Y-%m")
    old = sorted({d for d in days if d[:7] < cur})
    live = sorted({d for d in days if d[:7] >= cur})
    return old, live


def _archive_dir(uid: str):
    return sessions_dir(uid).with_name("sessions.archive")


# ---------------------------------------------------------------- 核心护栏
def test_archiving_does_not_change_what_she_remembers(rt: Runtime):
    uid = "u" + "4" * 16
    days = _seed_across_months(uid)
    old, live = _split(days)
    before = read_turns(uid)
    assert before, "对照组：先得有原话可读"
    assert len({d[:7] for d in old}) >= 3, f"要真的跨 3 个月（实际 {old}）"
    assert live, "当月要留至少一天，否则「留下的没动」这条没被测到"

    out = archive_old_turns(uid, months=1)

    assert out["files"] == len(old), f"搬走的天数不对：{out}"
    assert out["bytes"] > 0, "搬走的字节数要报出来（不然看不出到底动了没有）"
    assert out["months"] == sorted({d[:7] for d in old}), "报出被归档的月份"
    assert out["cutoff"] == time.strftime("%Y-%m"), "截止月 = 当月"
    assert out["skipped"] == 0 and out["errors"] == [], f"不该有搬不动的：{out}"

    sess = sessions_dir(uid)
    arch = _archive_dir(uid)
    for d in old:
        assert not (sess / f"{d}.md").exists(), f"旧月 {d} 的 .md 还在 sessions/"
        assert (arch / f"{d}.md.gz").exists(), f"旧月 {d} 没被搬进 sessions.archive/"
    for d in live:
        assert (sess / f"{d}.md").exists(), f"当月的 {d} 不该被搬走"
        assert not (arch / f"{d}.md.gz").exists(), f"当月的 {d} 不该归档"

    # gzip 头 4-8 字节是 MTIME。写 0 才对同一份内容给出同一批字节 ——
    # 不然「两次归档的结果一样」这件事就没法按字节比。
    head = (arch / f"{old[0]}.md.gz").read_bytes()[:8]
    assert head[4:8] == b"\x00\x00\x00\x00", f"gzip mtime 没写 0：{head!r}"

    after = read_turns(uid)
    assert after == before, f"归档之后读回来的东西变了\n前 {before}\n后 {after}"
    assert read_turns(uid, old[0]) == [t for t in before if t["day"] == old[0]], (
        "指定归档里的那一天也读得回来"
    )
    assert read_turns(uid, live[-1]) == [t for t in before if t["day"] == live[-1]], (
        "指定没归档的那一天照旧"
    )
    seq = [t["day"] for t in after]
    assert seq == sorted(seq), f"跨两处的读回顺序不是按天递增：{seq}"

    ast = archive_stats(uid)
    assert ast["files"] == len(old), f"归档小结的文件数不对：{ast}"
    assert ast["bytes"] > 0, "归档小结要报出占用（压缩后的字节）"


def test_archiving_twice_moves_nothing_the_second_time(rt: Runtime):
    uid = "u" + "5" * 16
    days = _seed_across_months(uid)
    old, _live = _split(days)
    before = read_turns(uid)
    arch = _archive_dir(uid)

    first = archive_old_turns(uid, months=1)
    assert first["files"] == len(old), f"第一次要搬走旧月那几天：{first}"
    assert len(list(arch.glob("*.md.gz"))) == len(old), "归档包数量不对"

    second = archive_old_turns(uid, months=1)
    assert second["files"] == 0 and second["bytes"] == 0, f"第二次不该再搬：{second}"
    assert second["months"] == [], f"第二次不该再报月份：{second}"
    assert second["skipped"] == 0, f"不该有「两边都有」的可疑日子：{second}"
    assert len(list(arch.glob("*.md.gz"))) == len(old), "归档包数量变了"
    assert read_turns(uid) == before, "第二次跑完之后读回来不一致"


def test_retention_zero_never_touches_anything(rt: Runtime):
    """`retain_months=0` = 永不清理（文档里那一行「关掉它」）。

    这条必须**真的跑一遍归档再断言盘上没变**：只看返回值的话，「先搬走再报 0」
    也过得去。
    """
    uid = "u" + "6" * 16
    _seed_across_months(uid)
    sess = sessions_dir(uid)
    files_before = sorted(p.name for p in sess.glob("*.md"))
    before = read_turns(uid)

    out = archive_old_turns(uid, months=0)

    assert isinstance(out, dict), f"关掉时也要回一个可观测的统计（拿到 {out!r}）"
    assert out["files"] == 0 and out["bytes"] == 0 and out["months"] == [], out
    assert sorted(p.name for p in sess.glob("*.md")) == files_before, "有文件被动了"
    assert not _archive_dir(uid).exists(), "连归档目录都不该建出来"
    assert read_turns(uid) == before, "关掉时读回来也必须一模一样"


def test_two_copies_that_differ_are_both_kept(rt: Runtime):
    """两份内容不同源时**两边都留着**：判断不了就偏「留原话」那一侧。

    现实的来源是「时间戳回拨，写进了已经归档的那一天」：包是旧的、live 是新的，
    谁也说不清该信哪一份。判错的方向只能是「多留一份」，绝不能是「删掉原话」。
    """
    uid = "u" + "7" * 16
    days = _seed_across_months(uid)
    old, _live = _split(days)
    day = old[0]
    sess = sessions_dir(uid)
    arch = _archive_dir(uid)
    arch.mkdir(parents=True, exist_ok=True)
    (arch / f"{day}.md.gz").write_bytes(gzip.compress("别的什么".encode(), mtime=0))

    out = archive_old_turns(uid, months=1)

    assert out["skipped"] == 1, f"这一天该被记成「没搬」：{out}"
    assert (sess / f"{day}.md").exists(), "内容不同源时 live 那份被删了 —— 原话丢了"
    assert read_turns(uid, day), "那一天还得读得出来（以 live 那份为准）"


def test_a_move_that_crashed_after_the_gzip_is_finished_later(rt: Runtime):
    """崩在 `os.replace` 与 `unlink` 之间：包已经落好、live 的 .md 还在。

    下一次归档要**补完最后那一步**。不补的话同一天会被读两遍（她记得的过去凭空
    多一份）；反过来「先删后写」会丢原话 —— 所以只有**逐字节同源**才敢删 live 那份。
    """
    uid = "u" + "8" * 16
    days = _seed_across_months(uid)
    old, _live = _split(days)
    before = read_turns(uid)
    sess = sessions_dir(uid)
    arch = _archive_dir(uid)
    arch.mkdir(parents=True, exist_ok=True)
    for d in old:  # 手工造出那个中间态：包在、原文件也在
        (arch / f"{d}.md.gz").write_bytes(gzip.compress((sess / f"{d}.md").read_bytes(), mtime=0))

    out = archive_old_turns(uid, months=1)

    assert out["files"] == 0, f"包已经在，不该重复搬运（幂等）：{out}"
    assert out["finished"] == len(old), f"该补完的那几天要补完：{out}"
    assert out["skipped"] == 0, f"同源的不该被当成「两份都留」：{out}"
    for d in old:
        assert not (sess / f"{d}.md").exists(), f"崩在半路的那天没补完：{d}"
    assert read_turns(uid) == before, "补完之后读回来必须一致（而且没有读两遍）"


def test_stats_shows_where_the_missing_turns_went(rt: Runtime):
    """`days` / `bytes` 只算 live，而 `turns` 是两处之和 —— 那对看起来会打架：
    全归档之后 `days == []` 而 `turns > 0`。所以归档那一侧要显式报出来，
    让「少了的去哪儿了」在**同一份返回值**里看得见。
    """
    uid = "u" + "9" * 16
    days = _seed_across_months(uid)
    old, live = _split(days)
    turns_before = stats(uid)["turns"]
    assert stats(uid)["days"] == sorted(days), "归档之前都在 live 那一侧"
    assert stats(uid)["archived_days"] == [], "还没归档过"

    archive_old_turns(uid, months=1)
    st = stats(uid)
    assert st["days"] == sorted(live), f"live 那一侧只剩当月：{st['days']}"
    assert st["archived_days"] == sorted(old), f"归档那几天要露出来：{st}"
    assert st["archived_bytes"] > 0, "归档占用也要报（压缩后）"
    assert st["turns"] == turns_before, "turns 走 read_turns，归档前后一条不差"
