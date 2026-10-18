# -*- coding: utf-8 -*-
"""对话框底色/正文色只有一处定义。

这条规则原先在十一个对话框里各写了一遍，而且写法还不统一（有的用 `theme`、
有的用 `t`）—— 换配色时很容易漏掉一个。现在集中在
`ThemeManager.get_dialog_style()`。
"""
from pathlib import Path

from labelforge.ui.theme import ThemeManager, ThemeMode

ROOT = Path(__file__).resolve().parents[1]


def test_dialog_style_matches_the_rule_the_dialogs_used_to_inline():
    """helper 的输出必须与原先内联的那条一模一样，否则外观会变。"""
    for mode in (ThemeMode.LIGHT, ThemeMode.DARK):
        ThemeManager.set_mode(mode)
        t = ThemeManager.get_theme()
        assert ThemeManager.get_dialog_style() == (
            f"QDialog {{ background-color: {t['widget_bg']}; "
            f"color: {t['text_primary']}; }}")


def test_no_dialog_re_inlines_the_rule():
    """拦住「又抄一遍」：除了定义处，别处不该再出现这条内联规则。"""
    pattern = "QDialog {{ background-color:"
    offenders = []
    for path in (ROOT / "labelforge").rglob("*.py"):
        if path.name == "theme.py":
            continue
        if pattern in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        "这些文件又内联了对话框底色规则，请改用 "
        "ThemeManager.get_dialog_style():\n  " + "\n  ".join(offenders))