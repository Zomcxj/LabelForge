"""Annotation-mode shortcuts and hint bar (Seg/Pose/OBB)."""
from labelforge.core import config
from labelforge.engine.event_handler import EventHandlerMixin
from labelforge.ui import i18n
from labelforge.ui.settings_dialog import SettingsDialog


def test_shortcut_config_has_polygon_point_obb_without_conflicting_r():
    sc = config.SHORTCUT_CONFIG
    assert sc["draw_polygon"] == "P"
    assert sc["draw_point"] == "K"
    assert sc["draw_obb"] == "O"
    assert sc["toggle_labels"] == "R"
    singles = [v for v in sc.values() if "+" not in v]
    assert len(singles) == len(set(singles)), singles


def test_shortcut_hint_includes_three_draw_modes():
    class Editor(EventHandlerMixin):
        def __init__(self):
            self.shortcut_config = dict(config.SHORTCUT_CONFIG)
            self.shortcut_status_label = type("L", (), {"text": "", "setText": lambda s, t: setattr(s, "text", t)})()

    editor = Editor()
    editor._update_shortcut_status_label()
    text = editor.shortcut_status_label.text
    assert "P" in text
    assert "K" in text
    assert "O" in text


def test_settings_dialog_lists_new_draw_shortcuts():
    src = open("labelforge/ui/settings_dialog.py", encoding="utf-8").read()
    assert "'draw_polygon'" in src
    assert "'draw_point'" in src
    assert "'draw_obb'" in src
    assert SettingsDialog is not None


def test_i18n_has_polygon_point_obb_keys():
    for key in ("绘制多边形", "标注关键点", "绘制旋转框", "分组"):
        assert key in i18n._strings["zh"]
        assert key in i18n._strings["en"]


def test_settings_dialog_has_max_polygon_points():
    src = open("labelforge/ui/settings_dialog.py", encoding="utf-8").read()
    assert "max_polygon_points" in src
    assert '"多边形最大点数"' in src or "多边形最大点数" in src


def test_i18n_has_polygon_vertex_keys():
    for key in ("多边形最大点数", "删除顶点", "已达最大点数"):
        assert key in i18n._strings["zh"]
        assert key in i18n._strings["en"]


def test_group_input_allows_empty_group():
    src = open("labelforge/ui/dialogs.py", encoding="utf-8").read()
    assert "setSpecialValueText" in src
    assert "setMinimum(0)" in src
    for key in ("留空",):
        assert key in i18n._strings["zh"]
        assert key in i18n._strings["en"]


def test_group_input_wired_into_all_label_dialogs():
    draw_src = open("labelforge/canvas/canvas_drawing.py", encoding="utf-8").read()
    assert draw_src.count("show_group=True") == 3  # 矩形 + 多边形 + 关键点
    menu_src = open("labelforge/canvas/canvas_menu.py", encoding="utf-8").read()
    assert "show_group=True" in menu_src  # 右键修改标签
    assert "current_group_id=" in menu_src
    lm_src = open("labelforge/engine/label_manager.py", encoding="utf-8").read()
    assert "show_group=True" in lm_src  # 标签管理逐框改名
    assert "current_group_id=" in lm_src
