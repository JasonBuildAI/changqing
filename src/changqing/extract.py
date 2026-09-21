"""抽取：把 L0 原话变成带**回引**的事实。

**为什么不复用陪聊那条 prompt。** 陪聊模型的 KPI 是「像她」；让它同时干严谨抽取，
两个目标互相打架：一边要自然、可以含糊，一边要有一说一、宁少勿错。所以抽取是一次
**独立**调用：自己的 prompt、自己的输出契约、**允许返回空数组**（没有值得记的
东西是完全合格的答案）。

这个文件里有四道闸：

  闸 1 回引轮次   每条事实必须带 `turn_ref` + `quote`，缺一个就丢
  闸 2 回引校验   `quote` 必须能在 **`turn_ref` 指的那一轮**里定位。
                  校验范围限定在被回引的那一轮，而不是全库字符串匹配 ——
                  既放过「我最近在接私活，画插画那种」→「职业：插画师」这种合理
                  转述，也挡住「上下文里碰巧出现过这个词」这种巧合命中。
                  命中 → `active`；近似 → `pending`（待确认，**不丢**）；对不上 → 丢弃
  闸 4 时间绝对化  相对时间词按那一轮的**时间戳**换算成绝对日期再落盘 ——
                  模型不知道今天是几号，它写的「上周五」是编的

「她是谁」不在这里写死：prompt 里那句角色说明由 `PersonaProfile` 注入
（见 `persona.py`），本模块只负责把它插进模板。画像决定的是 `persona_attention`
那一栏的口径 —— 通用的「重要」和「她会在意的」不是一回事。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from .persona import PersonaProfile
from .runtime import runtime
from .tokenize import tokenize

# ---------------------------------------------------------------- 抽取 prompt
# 画像在模板里留一个记号，最后用 `replace` 插进去（不用 `str.format`：
# prompt 正文里到处是 JSON 的花括号，转义它们只会让下一次改 prompt 变成踩雷）。
_PERSONA = "{persona}"
_ATTENTION = "{attention}"

EXTRACT_SYSTEM = """你是一个严谨的信息抽取器。你的唯一任务是从对话里抽出**用户亲口说过的、关于他自己的**事实。

铁律：
1. **事实**只从他说的（行首是「他说：」）的话里抽。她说的一律不算事实 —— 哪怕她是在复述他、
   哪怕她说的是「你上次说你不吃香菜」。
2. **恰好一条例外：她答应他的事。** 只从行首是「她说：」的行里抽，`kind=promise`、
   `subject` 填「她」、`quote` 用她那句话的原文片段（一字不改）。这是「她答应过他的事」
   进库的唯一来源 —— 她说了「周末带你去看展」，之后她必须记得自己说过。
   他答应的事是 `kind=commitment`（只从他的行抽），两者绝不要混。
3. 每条事实必须附 `turn_ref`（那一轮的编号，如 T-000123）和 `quote`（**原文片段，一字不改**）。
   找不到原文片段就不要输出这一条。
4. **宁少勿错。** 没有值得记住的事实就返回空数组 []，这是完全合格的答案。
   推测、脑补、把两个轮次的信息拼在一起，都是严重错误。
5. 不要抽：**一时的情绪**（「今天好累」「烦死了」这种说过就过去的）、她说过的话（除了铁律 2 的承诺）、
   你已经抽过的同一条事实。
   **但要抽**：长期的偏好、习惯、恐惧、身体状况、家里人、承诺与约定。
   「我有点怕黑」「我不吃香菜」「我妈身体不好」「我周末得加班」都是要记很久的，
   不要因为「听起来像在说情绪」就漏掉 —— 漏掉这类比多抽一条一次性的情绪糟得多。
6. subject 固定用「他」（kind=promise 时用「她」）；predicate 要写成能和 subject、object 连成一句通顺中文的动词短语
   （例：subject=他, predicate=养的猫叫, object=团子 → 「他养的猫叫团子」）。
   **没有宾语的事实把内容整段写进 predicate、object 留空字符串**（例：subject=他,
   predicate=怕黑, object="" → 「他怕黑」）。反过来把内容塞进 object、让 predicate
   空着是不合格的 —— 那样的提取结果会被丢弃。
7. `kind` 三选一：`fact`（关于他的信息）、`commitment`（**他答应的事、约定、
   待办**，例如「这周末带你去看展」）、`promise`（**她答应的事**，只从「她说：」的行抽，
   见铁律 2）。承诺类一定要标对 —— 它们会被单独顶到注入集合的前面，不能被普通信息挤掉。
8. `due` 只在承诺类**明确说出了时间**时给：把「这周末」「下周三」按下面给的
   今天换算成 YYYY-MM-DD；说不清的就给 null，不要猜。
9. 只输出一个 JSON 对象，字段是 `facts`、`summary` 与 `topics`，不要在围栏之外写任何文字。
10. `topics` 是「**她下次主动找他时值得提的事**」（不是事实，不是纪要）：
   他只说了一半的事、他这几天要面对的事（面试 / 体检 / 出差 / 项目截止）、
   她答应过他要做的事。每条一句短话，写成**她自己能开口说的那句**（不是给你的指令）。
   最多 3 条；**没有就返空数组 []，这是完全合格的答案**。
   绝对不要写「最近怎么样」「在吗」「你还好吗」这类空转开场 —— 写了就是错。
   `kind` 三选一：`followup`（接着说一件他没说完的事）、`share`（说一件她自己的事）、
   `ask`（问他一件她记得该问的事）；`due_day` 只在事情有具体日子时给（按今天换算）。

输出结构：
{
  "facts": [
    {"turn_ref":"T-000123","quote":"我家猫叫团子，三岁了","subject":"他",
     "predicate":"养的猫叫","object":"团子","kind":"fact","due":null,
     "confidence":0.95,"importance":0.7,"persona_attention":0.8}
  ],
  "summary": "他养了只三岁的猫叫团子。",
  "topics": [
    {"text":"他那个方案后来改了没有","kind":"followup","due_day":null,"ref":"T-000123"}
  ]
}

`summary` 是这**一段对话的纪要**：1-2 句、只写已经确认的事、不要复述每一条事实。
没有值得写的就给空字符串。

每条事实的字段：
  turn_ref     必需，那一轮的编号
  quote        必需，用户原话片段
  subject      通常就是「他」
  predicate    动词短语
  object       具体内容
  kind         `fact` 或 `commitment`
  due          承诺类的日期 YYYY-MM-DD，说不清给 null
  confidence   0-1，你有多确定这是用户说的、且不是玩笑。**要分出层次**：
               原话字面直述 = 0.9 以上；需要你归纳或转述（「我最近在接私活，
               画插画那种」→「职业是插画师」）= 0.6~0.8；有歧义或像玩笑 = 0.5 以下。
               全填 1.0 等于这个字段没填 —— 下游要靠它筛掉不可靠的抽取。
  importance   0-1，这条对长期了解他有多重要
  persona_attention 0-1，**{persona}**：{attention}。
               通用的「重要」和「她会在意的」不是一回事
"""

EXTRACT_USER = """今天是 {today}。下面是这一段对话（编号 + 时间 + 说话人 + 原话）：

{lines}

请抽取出值得长期记住的、关于他本人的事实（外加她答应过他的事，见铁律 2），
并给这一段对话写一句纪要。
没有值得记的就返回 {{"facts": [], "summary": ""}}。"""


def build_system(persona: PersonaProfile | None = None) -> str:
    """把画像插进抽取 prompt。**画像只影响「她会在意吗」这一栏的口径**，
    其余铁律一个字都不改 —— 换角色不该顺手改掉抽取的正确性要求。"""
    p = persona if persona is not None else runtime().persona
    return EXTRACT_SYSTEM.replace(_PERSONA, p.angle()).replace(_ATTENTION, p.attention_question())


def build_messages(
    turns: Sequence[dict], today: str, persona: PersonaProfile | None = None
) -> list[dict]:
    """拼出这一段的抽取请求。

    行首标出说话人：这是铁律 1 / 2 **唯一**的可判依据。不标的话模型分不清哪句是
    他说的、哪句是她说的 —— 她的承诺整段抽不出来，而错标的一句会让她把自己复述
    的话当成「他说过的事实」。
    """
    lines = []
    for t in turns:
        who = "她说" if str(t.get("role")) == "assistant" else "他说"
        lines.append(
            f"[{t['id']}] {t.get('day', '')} {t.get('time', '')} {who}：{t.get('text', '')}"
        )
    return [
        {"role": "system", "content": build_system(persona)},
        {"role": "user", "content": EXTRACT_USER.format(today=today, lines="\n".join(lines))},
    ]


# 主动话题的上限与长度。它是「她想说什么」的一句话，不是一篇文档：
# 超长一条就能把整个话题预算占光，把其他几条挤掉。
TOPIC_MAX = 3
TOPIC_TEXT_CHARS = 60
# 日期的唯一形状。检索与渲染都按 `[:10]` 切，别的形状会被**悄悄截断**，
# 所以这里只认这一种，其余一律当没给。
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_topics(raw: str, day: str = "") -> list[dict]:
    """取出「她下次想主动提的事」。坏 JSON / 缺字段 / 空数组都是「没话题」，绝不抛。

    这一层只做形状校验（字段存在、长度合适、kind 合法），**不做回引校验**：
    话题不是事实，进不了「有没有依据」那套对比（渲染时也不带编号）。所以它可以为空、
    也可以不贴 `ref` —— 但一条都不许多于 `TOPIC_MAX` 条：它是预算有限的 prompt
    片段，不是待办清单。

    `day` 是闸 4 的同一个基准（那一轮的时间）：模型爱把 `due_day` 写成「下周三」，
    不换算就等于存了一个会过期的词进去。换算不出具体日期的一律当没给（宁缺勿错）。
    """
    obj = _first_json_obj(raw)
    if not isinstance(obj, dict):
        return []
    rows = obj.get("topics")
    if not isinstance(rows, list):
        return []
    out: list[dict] = []
    for x in rows:
        if not isinstance(x, dict):
            continue
        text = str(x.get("text") or "").strip()
        if not text:
            continue
        kind = str(x.get("kind") or "share").strip().lower()
        if kind not in ("followup", "share", "ask"):
            kind = "share"
        due = str(x.get("due_day") or "").strip()
        if day and due:
            due = absolutize(due, day)
        out.append(
            {
                "text": text[:TOPIC_TEXT_CHARS],
                "kind": kind,
                # 与 facts 的 due 同一口径：算不出具体日期就当没给。
                # 一个错的日期比没有更糟 —— 她会照着一个不存在的日子问你。
                "due_day": due if _DATE_RE.match(due) else "",
                "ref": str(x.get("ref") or "").strip()[:12],
            }
        )
        if len(out) >= TOPIC_MAX:
            break
    return out


# ---------------------------------------------------------------- 闸 4：时间绝对化
_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
_REL_DAY = {"今天": 0, "昨天": -1, "前天": -2, "明天": 1, "后天": 2}
_REL_RE = re.compile(r"(上周|这周|本周|下周)([一二三四五六日天])|(今天|昨天|前天|明天|后天)")
# 光秃秃的周/月锚点：「上个月搬了家」没法唯一还原成某一天，
# 但留在**长期事实**里也会随时间失效（三个月后「上周」就不再是那一周了）。
_BARE_ANCHOR_RE = re.compile(r"(上周|这周|本周|下周|上个月|这个月|下个月)")


def absolutize(text: str, day: str) -> str:
    """把相对时间词换成绝对日期。换算基准是**那一轮的时间戳**，不是今天。

    模型不知道今天是几号，它写「上周五」纯属编的；而我们手里有那一轮的真实日期，
    能算准。两层处理：

      1. 能算准的（上周五 / 昨天 / 下周三）换成具体日期；
      2. 算不准的纯时间锚点（光秃秃的「上周」「这个月」）**直接去掉** ——
         把一句会随时间长歪的话留在一条要记很久的事实里，比不写它更糟。
         准确时间由事实自己的 `valid_from` 承担。

    刻意**不动**「以前 / 之前 / 最近」：它们带的是体貌意义，「我以前住北京」
    去掉「以前」就把意思改反了 —— 那是内容，不是时间锚点。
    """
    if not text:
        return text
    try:
        base = datetime.strptime(day, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return text

    def sub(m: re.Match) -> str:
        if m.group(1):  # 上周五 / 这周三
            which, wd = m.group(1), _WEEKDAYS.get(m.group(2), 0)
            monday = base - timedelta(days=base.weekday())
            offset = {"上周": -7, "这周": 0, "本周": 0, "下周": 7}[which]
            return (monday + timedelta(days=offset + wd)).isoformat()
        return (base + timedelta(days=_REL_DAY[m.group(3)])).isoformat()

    return _BARE_ANCHOR_RE.sub("", _REL_RE.sub(sub, text))


def _num(v, lo: float, hi: float, default: float) -> float:
    """夹到 [lo, hi]。不是数（None / "abc" / NaN）就给默认值。

    NaN 必须单独判：`max(lo, min(hi, nan))` 在 Python 里一路是 NaN，
    它不会报错，只会让这一条事实的分数永远比不出大小。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:
        return default
    return max(lo, min(hi, f))


def _norm_due(v, day: str) -> str | None:
    """承诺的截止日期：绝对化之后必须是 YYYY-MM-DD，否则宁可没有。

    模型给的 due 是自由文本（「下周三」「周末」），先过 absolutize。算不出具体
    日期的就丢掉 —— 一个错的截止日期比没有更糟：她会照着它说「你上周就该带我去了」。
    """
    s = str(v or "").strip()
    if not s or s.lower() in ("null", "none", "无"):
        return None
    s = absolutize(s, day)
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else None


# ---------------------------------------------------------------- 解析
def _first_json_obj(raw: str) -> dict | None:
    """从模型输出里抠出第一个能解析的 JSON 对象。坏 JSON 一律返回 None。"""
    if not raw:
        return None
    cands = []
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        cands.append(m.group(1).strip())
    a, b = raw.find("{"), raw.rfind("}")
    if 0 <= a < b:
        cands.append(raw[a : b + 1])
    for c in cands:
        # 第二遍去掉尾随逗号：模型很爱在数组最后一项后面留一个逗号，
        # 而 `json.loads` 对此零容忍 —— 那一条会连累整批抽取结果被判成「没抽到」。
        for attempt in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                obj = json.loads(attempt)
            except Exception:  # noqa: BLE001  容错函数的本分：坏 JSON 当没抽到
                continue
            if isinstance(obj, dict):
                return obj
    return None


def parse_facts(raw: str) -> list[dict]:
    """从模型输出里抠出事实数组。坏 JSON 一律当「没抽到」，**绝不抛异常**。

    兼容两种输出：裸数组（老约定），以及 `{"facts": [...], "summary": "..."}`。
    两种都认是有意的：模型换了版本、或者哪天回了老格式，都不该让整段整理变成 0 条。
    """
    if not raw:
        return []
    obj = _first_json_obj(raw)
    if obj is not None and isinstance(obj.get("facts"), list):
        return [x for x in obj["facts"] if isinstance(x, dict)]
    # 裸数组
    a, b = raw.find("["), raw.rfind("]")
    if 0 <= a < b:
        chunk = raw[a : b + 1]
        for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                arr = json.loads(attempt)
            except Exception:  # noqa: BLE001  同上：解析失败不该让整段整理炸掉
                continue
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
    return []


def parse_summary(raw: str) -> str:
    """取出这一段对话的纪要。没有、或者长得不像纪要就不要（宁可没有）。"""
    obj = _first_json_obj(raw)
    if not obj:
        return ""
    return str(obj.get("summary") or "").strip()[:500]


# ---------------------------------------------------------------- 闸 1 / 2：回引校验
def _norm(s: str) -> str:
    """去掉空白与标点，只留内容。回引比对要在「内容」上做，不在排版上做。"""
    return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】…—-]+", "", str(s or ""))


def overlap(quote: str, text: str) -> float:
    """`quote` 有多少比例能在 `text` 里找到（按 token 覆盖率）。

    为什么不用整串相似度：quote 是短句、text 是整轮，字符级相似度天然很低，
    会把合理的转述全判成编造 —— 而「合理转述」恰恰是我们想留下的那一类。
    """
    qt = [t for t in tokenize(quote) if _norm(t)]
    if not qt:
        return 0.0
    body = _norm(text)
    hit = sum(1 for t in qt if _norm(t) and _norm(t) in body)
    return hit / len(qt)


def verify(fact: dict, turns_by_id: dict[str, dict], tolerance: float = 0.0) -> tuple[str, float]:
    """回引校验。返回 `(active | pending | drop, 分数)`。

    **只看 `turn_ref` 指的那一轮。** 全库匹配的话，「上下文里碰巧出现过那个词」
    也会被当成命中，而那不是证据。
    """
    tol = tolerance or runtime().config.pending_tolerance
    tid = str(fact.get("turn_ref") or "").strip()
    if not tid:
        return "drop", 0.0  # 闸 1：没有回引，直接丢
    turn = turns_by_id.get(tid)
    if turn is None:
        return "drop", 0.0  # 回引到一个不存在的轮次 = 编的
    text = str(turn.get("text") or "")
    role = str(turn.get("role"))
    kind = str(fact.get("kind") or "").strip().lower()
    # 他说的话默认收；她的话**只收 kind=promise**（她答应他的事，铁律 2）。
    # 不放宽这一条的话，「她答应过他的事」永远进不了事实层 —— 她的承诺只存在于
    # assistant 行里，一律 drop 的表现就是「她忘了自己答应过什么」。
    if role != "user" and not (role == "assistant" and kind == "promise"):
        return "drop", 0.0
    quote = str(fact.get("quote") or "").strip()
    if not quote:
        return "drop", 0.0
    if _norm(quote) in _norm(text):
        return "active", 1.0
    score = overlap(quote, text)
    if score >= tol:
        return "pending", round(score, 3)  # 近似：待确认，**不丢**
    return "drop", round(score, 3)


# ---------------------------------------------------------------- 清洗
_RANGES = {"confidence": (0.0, 1.0), "importance": (0.0, 1.0), "persona_attention": (0.0, 1.0)}


def clean_fact(fact: dict, turn: dict, persona: PersonaProfile | None = None) -> dict | None:
    """把一条原始抽取结果变成可以落盘的事实。缺关键字段就返回 None。

    `persona` 决定承诺类事实的 `subject`（默认取运行期里那份画像）。
    """
    kind = str(fact.get("kind") or "fact").strip().lower()
    if kind not in ("fact", "commitment", "promise"):
        kind = "fact"
    # kind=promise（她答应的事）的 subject 统一成画像的称呼：它决定卡片的归属文案，
    # 是**形状**而不是自由内容，不该交给模型发挥 —— 模型把 subject 写成「他」的话，
    # 她会拿自己的承诺说成「你上次说……」，编出根本不存在的对话。
    who = (persona if persona is not None else runtime().persona).promise_subject
    subject = str(fact.get("subject") or (who if kind == "promise" else "他")).strip()[:40]
    if kind == "promise":
        subject = who
    predicate = str(fact.get("predicate") or "").strip()[:80]
    obj = str(fact.get("object") or "").strip()[:200]
    # **object 允许为空。** 中文里「怕黑」「我妈身体不好」这类事实根本没有宾语，
    # 模型会把内容整段写进 predicate。这里要求 object 非空的话，它们会被判成无效
    # **静默丢弃** —— 而丢一条的表现只是「她忘了」，不报错、不进任何指标。
    # 判据改成「合起来得有点内容」，两边都空的仍然是垃圾。
    if len(predicate) + len(obj) < 2:
        return None
    day = str(turn.get("day") or "")[:10] or date.today().isoformat()
    return {
        "subject": subject,
        "predicate": predicate,
        # 闸 4：事实内容里的相对时间词换成绝对日期（基准是那一轮的时间戳）
        "object": absolutize(obj, day),
        "kind": kind,
        "due": _norm_due(fact.get("due"), day),
        "turn_ref": str(turn.get("id") or ""),
        # quote 保持**一字不改** —— 它是回引校验的证据，改过就不叫证据了。
        # 所以 absolutize 只作用在 object 上，不动 quote。
        "quote": str(fact.get("quote") or "")[:300],
        "confidence": _num(fact.get("confidence"), 0, 1, 0.7),
        "importance": _num(fact.get("importance"), 0, 1, 0.5),
        "persona_attention": _num(fact.get("persona_attention"), 0, 1, 0.5),
        "valid_from": day,
        "source": "extract",
    }


# ---------------------------------------------------------------- 冲突消解
def resolve_ops(uid: str, facts: Sequence[dict]) -> tuple[list[dict], dict[str, int]]:
    """同槽位（subject + predicate）的冲突消解 → 要追加的操作。

    三种结局（外加一种「不改」）：

      ADD        槽位是空的 → 新增
      SUPERSEDE  同槽位已有别的值（「我换工作了」）→ 新事实入，**旧事实失效而不是删除**
                 （双时间轴）。长期相处里忘记比记错更容易被原谅，但真正伤人的是
                 「她矢口否认说过」
      NOOP       同槽位、同值 → 什么都不做。**这就是幂等的落点**：同一段原话整理两次，
                 第二次全是 NOOP，状态一模一样

    NOOP 只出现在返回的统计里、**不写进日志** —— 日志是状态，不是决策流水；
    把「这次什么都没发生」也记下来的话，重复整理会让日志无限膨胀。
    """
    from . import store as ST

    stats = {"ADD": 0, "SUPERSEDE": 0, "NOOP": 0}
    ops: list[dict] = []
    existing: dict[tuple[str, str], list[dict]] = {}
    for f in ST.list_facts(uid):
        existing.setdefault((f["subject"], f["predicate"]), []).append(f)

    for f in facts:
        key = (f["subject"], f["predicate"])
        cands = existing.get(key) or []
        if any(c["object"] == f["object"] for c in cands):
            stats["NOOP"] += 1
            continue
        fid = ST.next_fact_id(uid)
        payload = dict(f)
        if cands:
            op = dict(payload, op="SUPERSEDE", id=fid, replaces=cands[0]["id"])
            stats["SUPERSEDE"] += 1
        else:
            op = dict(payload, op="ADD", id=fid)
            stats["ADD"] += 1
        ops.append(op)
        # 同一次整理里出现第二条同槽位事实时，它要取代的是刚写下的这条
        existing[key] = [dict(payload, id=fid)]
    return ops, stats
