import os
import subprocess
import sys
from pathlib import Path

import pytest

from pastelabel.ui.theme import ThemeManager, ThemeMode


ROOT = Path(__file__).resolve().parents[1]


# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死。断言通过后立即
# flush + os._exit(0) 绕开析构；os._exit 不冲刷缓冲，必须先 flush。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _method_source(path, method_name, next_method_name):
    source = path.read_text(encoding="utf-8")
    return source.split(f"def {method_name}", 1)[1].split(
        f"def {next_method_name}", 1
    )[0]


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_theme_tokens_drive_typography_and_active_interaction(mode):
    ThemeManager.set_mode(mode)
    theme = ThemeManager.get_theme()
    stylesheet = ThemeManager.get_stylesheet()

    assert theme["font_family"] in stylesheet
    assert f"font-size: {theme['font_size']}px;" in stylesheet
    assert f"background-color: {theme['interaction_active']};" in stylesheet
    assert f"color: {theme['interaction_active_text']};" in stylesheet


def test_global_stylesheet_owns_normal_mode_and_cache_widget_rules():
    stylesheet = ThemeManager.get_stylesheet()

    for selector in (
        "QFrame#modeSeg",
        "QPushButton#modeSegBtn",
        "QLineEdit#cacheSlotName",
        "QWidget#optionsPopup",
    ):
        assert selector in stylesheet


def test_mode_and_cache_widgets_use_global_qss_contract_without_local_overrides():
    root = __import__("pathlib").Path(__file__).parents[1]
    mode_setup = _method_source(root / "pastelabel" / "ui" / "mixins" / "options_popup.py", "_create_options_menu", "_rebuild_options_popup")
    cache_setup = _method_source(root / "pastelabel" / "ui" / "mixins" / "cache_menu.py", "_rebuild_label_cache_menu", "_handle_cache_slot_row_click")

    assert 'self.mode_seg.setObjectName("modeSeg")' in mode_setup
    assert mode_setup.count('setObjectName("modeSegBtn")') == 2
    assert 'slot_name_input.setObjectName("cacheSlotName")' in cache_setup
    assert "setTextMargins" not in cache_setup
    assert 'slot_name_input.setProperty("active", index == getattr(self, \'active_label_cache_slot\', 0))' in cache_setup
    assert "self.mode_seg.setStyleSheet" not in mode_setup
    assert "mode_button_style" not in mode_setup
    assert "slot_name_input.setStyleSheet" not in cache_setup


def test_mode_refresh_only_syncs_state_and_indicator_without_local_qss():
    from pastelabel.ui.main_window import ImageEditor
    import inspect

    refresh = inspect.getsource(ImageEditor._update_mode_seg_style)

    assert "setChecked(is_paste)" in refresh
    assert "setChecked(not is_paste)" in refresh
    assert 'self.mode_seg_ctrl.set_accent(ThemeManager.get_theme()["interaction_active"])' in refresh
    assert "self.mode_seg_ctrl.update_position(animated=animated)" in refresh
    assert "setStyleSheet" not in refresh


def test_apply_theme_refreshes_the_mode_indicator_from_current_tokens():
    from pastelabel.ui.main_window import ImageEditor
    import inspect

    refresh = inspect.getsource(ImageEditor._apply_theme)

    assert "self._update_mode_seg_style()" in refresh


def test_settings_and_label_dialog_normal_controls_inherit_global_qss():
    root = __import__("pathlib").Path(__file__).parents[1]
    settings = (root / "pastelabel" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")
    dialogs = (root / "pastelabel" / "ui" / "dialogs.py").read_text(encoding="utf-8")
    settings_setup = _method_source(root / "pastelabel" / "ui" / "settings_dialog.py", "_init_ui", "_switch_page")
    label_setup = _method_source(root / "pastelabel" / "ui" / "dialogs.py", "__init__", "showEvent")

    assert "setStyleSheet" not in settings_setup
    assert "setStyleSheet" not in label_setup
    assert "dialog.setStyleSheet" not in settings


def test_progress_dialog_uses_global_button_states_and_color_swatch_keeps_its_color():
    root = __import__("pathlib").Path(__file__).parents[1]
    settings = (root / "pastelabel" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")
    dialogs = (root / "pastelabel" / "ui" / "dialogs.py").read_text(encoding="utf-8")
    progress_setup = _method_source(root / "pastelabel" / "ui" / "dialogs.py", "create_progress_dialog", "_center_dialog")
    crosshair_style = _method_source(root / "pastelabel" / "ui" / "settings_dialog.py", "_update_crosshair_color_button", "_choose_crosshair_color")

    assert "QPushButton" not in progress_setup
    assert "#2950ff" not in progress_setup
    assert "background-color: {self._crosshair_color}" in crosshair_style
    assert "#2950ff" not in crosshair_style


@pytest.mark.parametrize("mode, color", [("LIGHT", "#1f1f1f"), ("DARK", "#f5f5f5")])
def test_real_qt_theme_switch_styles_settings_and_label_dialog_inputs(mode, color):
    script = f'''
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.dialogs import LabelSelectionDialog
from pastelabel.ui.main_window import ImageEditor
from pastelabel.ui.settings_dialog import SettingsDialog
from pastelabel.ui.theme import ThemeManager, ThemeMode

app = QApplication.instance() or QApplication([])
window = ImageEditor()
ThemeManager.set_mode(ThemeMode.{mode})
window._apply_theme()
settings = SettingsDialog(window)
label_dialog = LabelSelectionDialog(labels=["cat"])
for widget, role in (
    (window.label_list, QPalette.Text),
    (window.prev_img_btn, QPalette.ButtonText),
    (settings.grid_width_spin, QPalette.Text),
    (label_dialog.new_label_input, QPalette.Text),
):
    widget.ensurePolished()
    assert widget.styleSheet() == "", (type(widget).__name__, widget.styleSheet())
    assert widget.palette().color(role).name() == "{color}", type(widget).__name__
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_real_qt_canvas_keeps_business_overlay_colors_across_a_light_to_dark_switch():
    script = '''
from PyQt5.QtCore import QPoint, QRectF, Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget
from pastelabel.canvas.canvas_renderer import CanvasRendererMixin
from pastelabel.ui.main_window import ImageEditor
from pastelabel.ui.theme import ThemeManager, ThemeMode

class Check:
    def __init__(self, checked=True): self.checked = checked
    def isChecked(self): return self.checked

class Editor:
    def __init__(self):
        self.current_background = QPixmap(100, 100)
        self.current_background.fill(Qt.white)
        self.show_grid_checkbox = Check(False)
        self.show_labels_checkbox = Check(True)
        self.show_label_names_checkbox = Check(False)
        self.show_paste_names_checkbox = Check(False)
        self.detection_boxes = [{"x": 10, "y": 10, "width": 30, "height": 30, "label": "cat"}]
        self.canvas_items = []
        self.selected_item = None
        self.edit_mode = "annotate"
        self.pressed_label = None
    def get_label_color(self, label): return "#E53935"

class Canvas(CanvasRendererMixin, QWidget):
    def __init__(self):
        super().__init__()
        self._editor = Editor()
        self.background_scale = 1
        self.selected_box = 0
        self.selected_boxes = [0]
        self.shape_opacity = 1.0
        self.is_drawing_box = True
        self.temp_draw_box = QRectF(50, 50, 20, 20)
        self.draw_start_pos = QPoint(50, 50)
        self.mouse_inside = True
        self.mouse_pos = QPoint(80, 80)
        self.hover_resize_target = None
        self.hover_resize_handle = None
    def get_background_rect(self): return QRectF(0, 0, 100, 100)

app = QApplication.instance() or QApplication([])
window = ImageEditor()
canvas = Canvas()
canvas.resize(100, 100)
samples = []
for mode in (ThemeMode.LIGHT, ThemeMode.DARK):
    ThemeManager.set_mode(mode)
    window._apply_theme()
    image = QImage(100, 100, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    canvas.render(image)
    samples.append((
        image.pixelColor(20, 10).name().lower(), # detection border
        image.pixelColor(10, 10).name().lower(), # detection handle
        image.pixelColor(55, 50).name().lower(), # temporary box
        image.pixelColor(80, 20).name().lower(), # crosshair
    ))

assert samples[0] == samples[1], samples
assert QColor(samples[0][1]).red() > 200, samples
assert QColor(samples[0][0]).red() > QColor(samples[0][0]).green(), samples
assert QColor(samples[0][2]).green() > 200, samples
assert samples[0][3] != "#ffffff", samples
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_options_popup_uses_global_qss_instead_of_a_stale_local_override():
    popup_setup = _method_source(
        ROOT / "pastelabel" / "ui" / "mixins" / "options_popup.py",
        "_rebuild_options_popup",
        "_refresh_options_popup_texts",
    )
    assert 'popup.setObjectName("optionsPopup")' in popup_setup
    assert "popup.setStyleSheet" not in popup_setup


def test_log_area_font_is_one_step_up_from_default():
    """日志框文字比全局字号大一号，小字长时间看容易吃力。"""
    stylesheet = ThemeManager.get_stylesheet()
    block = stylesheet.split("QTextEdit#logArea", 1)[1].split("}\n", 1)[0]
    assert "Consolas" in block, "日志框应保持等宽字体"
    assert "font-size: 12px;" in block
    assert "font-size: 11px;" not in block
    assert "border-radius: 6px;" in block, "圆角样式不应被误改"
