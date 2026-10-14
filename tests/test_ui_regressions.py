"""Regression tests for recently reported UI failures."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canvas_imports_qpixmap_for_brightness_contrast_updates():
    source = (ROOT / "pastelabel" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")

    assert "from PyQt5.QtGui import QPainter, QPixmap, QColor, QPen, QFontMetrics" in source


def test_brightness_contrast_uses_vectorized_pixel_processing():
    source = (ROOT / "pastelabel" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_adjusted_background"
    )

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(method))
    assert "np.frombuffer" in source
    assert "np.clip" in source


def test_message_box_relocalizes_detail_button_after_click():
    source = (ROOT / "pastelabel" / "ui" / "dialog_helpers.py").read_text(encoding="utf-8")

    assert "btn.clicked" in source
    assert "QTimer.singleShot(0, self._fix_detail_buttons)" in source


def test_memory_action_dispatches_memory_records_dialog(monkeypatch):
    from pastelabel.ui import memory_dialog
    from pastelabel.ui.mixins.options_popup import OptionsPopupMixin

    calls = {}

    class FakeMemoryRecordsDialog:
        def __init__(self, parent):
            calls["parent"] = parent

        def exec_(self):
            calls["executed"] = True

    monkeypatch.setattr(memory_dialog, "MemoryRecordsDialog", FakeMemoryRecordsDialog)

    owner = object()
    OptionsPopupMixin._show_memory_records(owner)

    assert calls == {"parent": owner, "executed": True}


def test_canvas_adjustment_panel_localizes_via_project_i18n():
    source = (ROOT / "pastelabel" / "widgets" / "canvas_adjustment.py").read_text(encoding="utf-8")

    # 不允许绕过项目 i18n 的 Qt 自带 tr()（无翻译文件时恒为英文）
    assert "self.tr(" not in source
    assert "from ..ui.i18n import t as tr" in source
    for key in ("画布显示", "标签透明度", "画面亮度", "对比度", "恢复默认"):
        assert f'tr("{key}")' in source
    assert "def retranslate_ui" in source


def test_language_switch_refreshes_canvas_adjustment_texts():
    source = (ROOT / "pastelabel" / "ui" / "mixins" / "translation.py").read_text(encoding="utf-8")

    assert "self.canvas_adjustment.retranslate_ui()" in source


def test_canvas_adjustment_i18n_entries_exist_in_both_languages():
    from pastelabel.ui import i18n

    keys = ("画布显示", "标签透明度", "画面亮度", "对比度", "恢复默认", "收起调节", "展开调节")
    for lang in ("zh", "en"):
        table = i18n._strings[lang]
        for key in keys:
            assert key in table, (lang, key)
            assert table[key] != ""


def test_options_popup_uses_correct_core_import_depth():
    source = (ROOT / "pastelabel" / "ui" / "mixins" / "options_popup.py").read_text(encoding="utf-8")

    # 这个文件在 ui/mixins/ 下，core 在 pastelabel/ 下 → 必须是三级 `...`。
    # 不断言具体条数：写死数字会在每次新增一个用到配置的开关时假红，
    # 而它真正要防的是「用了少一层点的错误 import」。
    assert "from ..core import config_manager" not in source
    assert "from ...core import config_manager" in source
    # 所有 config_manager 的导入都必须走三级，不允许混用其它层级
    import re
    imports = re.findall(r"from (\.+)core import config_manager", source)
    assert imports, "没找到 config_manager 导入"
    assert set(imports) == {"..."}, imports
