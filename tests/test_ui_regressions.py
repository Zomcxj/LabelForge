"""Regression tests for recently reported UI failures."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canvas_imports_qpixmap_for_brightness_contrast_updates():
    source = (ROOT / "labelforge" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # 断言「这些名字都被导入」而不是钉死整行：之前钉字符串，加一个 QFont
    # 就假红（那是给滑窗格子标号用的，与亮度对比度无关）。
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "PySide6.QtGui":
            imported.update(a.name for a in node.names)
    for name in ("QPainter", "QPixmap", "QColor", "QPen", "QFontMetrics"):
        assert name in imported, f"canvas_renderer 没导入 {name}"


def test_brightness_contrast_uses_vectorized_pixel_processing():
    source = (ROOT / "labelforge" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")
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
    source = (ROOT / "labelforge" / "ui" / "dialog_helpers.py").read_text(encoding="utf-8")

    assert "btn.clicked" in source
    assert "QTimer.singleShot(0, self._fix_detail_buttons)" in source


def test_memory_action_dispatches_memory_records_dialog(monkeypatch):
    from labelforge.ui import memory_dialog
    from labelforge.ui.mixins.options_popup import OptionsPopupMixin

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
    source = (ROOT / "labelforge" / "widgets" / "canvas_adjustment.py").read_text(encoding="utf-8")

    # 不允许绕过项目 i18n 的 Qt 自带 tr()（无翻译文件时恒为英文）
    assert "self.tr(" not in source
    assert "from ..ui.i18n import t as tr" in source
    for key in ("画布显示", "标签透明度", "画面亮度", "对比度", "恢复默认"):
        assert f'tr("{key}")' in source
    assert "def retranslate_ui" in source


def test_language_switch_refreshes_canvas_adjustment_texts():
    source = (ROOT / "labelforge" / "ui" / "mixins" / "translation.py").read_text(encoding="utf-8")

    assert "self.canvas_adjustment.retranslate_ui()" in source


def test_canvas_adjustment_i18n_entries_exist_in_both_languages():
    from labelforge.ui import i18n

    keys = ("画布显示", "标签透明度", "画面亮度", "对比度", "恢复默认", "收起调节", "展开调节")
    for lang in ("zh", "en"):
        table = i18n._strings[lang]
        for key in keys:
            assert key in table, (lang, key)
            assert table[key] != ""


def test_options_popup_uses_correct_core_import_depth():
    source = (ROOT / "labelforge" / "ui" / "mixins" / "options_popup.py").read_text(encoding="utf-8")

    # 这个文件在 ui/mixins/ 下，core 在 labelforge/ 下 → 必须是三级 `...`。
    # 不断言具体条数：写死数字会在每次新增一个用到配置的开关时假红，
    # 而它真正要防的是「用了少一层点的错误 import」。
    assert "from ..core import config_manager" not in source
    assert "from ...core import config_manager" in source
    # 所有 config_manager 的导入都必须走三级，不允许混用其它层级
    import re
    imports = re.findall(r"from (\.+)core import config_manager", source)
    assert imports, "没找到 config_manager 导入"
    assert set(imports) == {"..."}, imports
