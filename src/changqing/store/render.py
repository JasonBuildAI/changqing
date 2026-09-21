"""派生视图：`facts.md` 的渲染。

这份 md 是**只读产物**（手工改会被下次渲染覆盖），渲染逻辑与「事实存了什么」
是两回事。

依赖方向：`facts.list_facts` 只能在函数内 import。`index.py` 模块级要
`from . import render`，若这里也在模块级 `from .facts import ...`，
链条会变成 index → render → facts → index 而成环。放在函数体里，
环就断在「调用时」；本模块对外的模块级依赖只剩 `paths`。
"""

from __future__ import annotations

from pathlib import Path

from ..runtime import runtime
from .paths import facts_md_path


def enabled() -> bool:
    """物化之后要不要顺手重渲染 `facts.md`。

    它只是给人看的派生视图；自检里成百上千次写操作时关掉能省一堆无意义的文件 I/O。
    开关住在配置里（每次现取），而不是这个模块的一个全局变量 ——
    后者会让「改配置」对已经导入过本模块的进程静默失效。
    """
    return bool(runtime().config.render_views)


def render_facts_md(uid: str) -> Path | None:
    """把物化视图渲染成人类可读的 md。**只读产物**：手工改会被下次渲染覆盖，
    要改走编辑接口（由它追加一条操作）。"""
    from .facts import list_facts     # 函数内 import：模块级会成环，见文件头说明
    facts = list_facts(uid)
    dead = list_facts(uid, include_dead=True, status="superseded")
    lines = ["# 关于他的事实（渲染视图，只读）", "",
             "> 这是从 log.jsonl 物化出来的视图，**不要手工编辑** —— 下次渲染会覆盖它。",
             "> 要改走编辑接口，由接口往日志追加一条操作。", ""]
    if not facts:
        lines += ["（还没有事实。事实由后台整理从原话里抽出来。）", ""]
    for f in facts:
        head = f"{f['subject']}{f['predicate']}{f['object']}"
        marks = [f"#{f['id']}", f["status"],
                 f"重要 {float(f['importance'] or 0):.1f}",
                 f"她会在意 {float(f['persona_attention'] or 0):.1f}"]
        if f["pinned"]:
            marks.append("已钉住")
        lines.append(f"## {head}  ({' · '.join(marks)})")
        lines.append("")
        when = (f["valid_from"] or "")[:10]
        quote = f["quote"] or ""
        ref = f["turn_ref"] or ""
        if quote:
            lines.append(f"- {when} 他说：「{quote}」  ({ref})")
        else:
            lines.append(f"- {when}  ({ref})")
        lines.append("")
    if dead:
        lines += ["## 已经失效的（不再是当前事实，但没删）", ""]
        for f in dead:
            lines.append(f"- {f['id']} {f['subject']}{f['predicate']}{f['object']}"
                         f" —— {f['valid_to'] or '?'} 起失效")
        lines.append("")
    p = facts_md_path(uid)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines), encoding="utf-8")
        return p
    except OSError:
        return None

