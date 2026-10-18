# -*- coding: utf-8 -*-
"""对话框共用的两条样式只有一处定义。

这两条原先在十来个文件里各写一遍，而且写法还不统一（有的用 `theme`、有的
用 `t`）—— 换配色时很容易漏掉一个。现在集中在 ThemeManager 上：
`get_dialog_style()` 管对话框本身的底色/文字色，`get_label_style()` 管正文色。
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


def test_label_style_matches_the_rule_the_panels_used_to_inline():
    for mode in (ThemeMode.LIGHT, ThemeMode.DARK):
        ThemeManager.set_mode(mode)
        t = ThemeManager.get_theme()
        assert ThemeManager.get_label_style() == (
            f"QLabel {{ color: {t['text_primary']}; }}")


def _files_containing(fragment):
    return sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "labelforge").rglob("*.py")
        if p.name != "theme.py" and fragment in p.read_text(encoding="utf-8"))


def test_no_dialog_re_inlines_the_dialog_rule():
    """拦住「又抄一遍」：除了定义处，别处不该再出现这条内联规则。"""
    offenders = _files_containing("QDialog {{ background-color:")
    assert not offenders, (
        "这些文件又内联了对话框底色规则，请改用 "
        "ThemeManager.get_dialog_style():\n  " + "\n  ".join(offenders))


def test_no_panel_re_inlines_the_label_rule():
    offenders = _files_containing("QLabel {{ color: {t['text_primary']}; }}")
    offenders += _files_containing("QLabel {{ color: {theme['text_primary']}; }}")
    assert not sorted(set(offenders)), (
        "这些文件又内联了标签正文色，请改用 "
        "ThemeManager.get_label_style():\n  " + "\n  ".join(sorted(set(offenders))))