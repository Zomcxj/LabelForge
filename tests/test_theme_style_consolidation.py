import os
import subprocess
import sys
from pathlib import Path

import pytest

from labelforge.ui.theme import ThemeManager, ThemeMode


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
    mode_setup = _method_source(root / "labelforge" / "ui" / "mixins" / "options_popup.py", "_create_options_menu", "_rebuild_options_popup")
    cache_setup = _method_source(root / "labelforge" / "ui" / "mixins" / "cache_menu.py", "_rebuild_label_cache_menu", "_handle_cache_slot_row_click")

    assert 'self.mode_seg.setObjectName("modeSeg")' in mode_setup
    assert mode_setup.count('setObjectName("modeSegBtn")') == 2
    assert 'slot_name_input.setObjectName("cacheSlotName")' in cache_setup
    assert "setTextMargins" not in cache_setup
    assert 'slot_name_input.setProperty("active", index == getattr(self, \'active_label_cache_slot\', 0))' in cache_setup
    assert "self.mode_seg.setStyleSheet" not in mode_setup
    assert "mode_button_style" not in mode_setup
    assert "slot_name_input.setStyleSheet" not in cache_setup


def test_mode_refresh_only_syncs_state_and_indicator_without_local_qss():
    from labelforge.ui.main_window import ImageEditor
    import inspect

    refresh = inspect.getsource(ImageEditor._update_mode_seg_style)

    assert "setChecked(is_paste)" in refresh
    assert "setChecked(not is_paste)" in refresh
    assert 'self.mode_seg_ctrl.set_accent(ThemeManager.get_theme()["interaction_active"])' in refresh
    assert "self.mode_seg_ctrl.update_position(animated=animated)" in refresh
    assert "setStyleSheet" not in refresh


def test_apply_theme_refreshes_the_mode_indicator_from_current_tokens():
    from labelforge.ui.main_window import ImageEditor
    import inspect

    refresh = inspect.getsource(ImageEditor._apply_theme)

    assert "self._update_mode_seg_style()" in refresh


def test_settings_and_label_dialog_normal_controls_inherit_global_qss():
    root = __import__("pathlib").Path(__file__).parents[1]
    settings = (root / "labelforge" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")
    dialogs = (root / "labelforge" / "ui" / "dialogs.py").read_text(encoding="utf-8")
    settings_setup = _method_source(root / "labelforge" / "ui" / "settings_dialog.py", "_init_ui", "_switch_page")
    label_setup = _method_source(root / "labelforge" / "ui" / "dialogs.py", "__init__", "showEvent")

    assert "setStyleSheet" not in settings_setup
    assert "setStyleSheet" not in label_setup
    assert "dialog.setStyleSheet" not in settings


def test_progress_dialog_uses_global_button_states_and_color_swatch_keeps_its_color():
    root = __import__("pathlib").Path(__file__).parents[1]
    settings = (root / "labelforge" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")
    dialogs = (root / "labelforge" / "ui" / "dialogs.py").read_text(encoding="utf-8")
    progress_setup = _method_source(root / "labelforge" / "ui" / "dialogs.py", "create_progress_dialog", "_center_dialog")
    crosshair_style = _method_source(root / "labelforge" / "ui" / "settings_dialog.py", "_update_crosshair_color_button", "_choose_crosshair_color")

    assert "QPushButton" not in progress_setup
    assert "#2950ff" not in progress_setup
    assert "background-color: {self._crosshair_color}" in crosshair_style
    assert "#2950ff" not in crosshair_style


@pytest.mark.parametrize("mode, color", [("LIGHT", "#1f1f1f"), ("DARK", "#f5f5f5")])
def test_real_qt_theme_switch_styles_settings_and_label_dialog_inputs(mode, color):
    script = f'''
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication
from labelforge.ui.dialogs import LabelSelectionDialog
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.settings_dialog import SettingsDialog
from labelforge.ui.theme import ThemeManager, ThemeMode

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
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QWidget
from labelforge.canvas.canvas_renderer import CanvasRendererMixin
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.theme import ThemeManager, ThemeMode

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
        ROOT / "labelforge" / "ui" / "mixins" / "options_popup.py",
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
    # 圆角跟全局一致（第 11 轮统一为 4px）；这条只盯「别又被单独改掉」
    assert "border-radius: 4px;" in block, "圆角样式不应被误改"


def test_real_qt_options_popup_paints_rounded_corners():
    """选项悬浮窗四角必须真的被切掉（用户报「底部还是直角」）。

    只写 QSS 的 `border-radius` 不够：Qt 会给顶层窗口铺一块不透明的方角
    背景，把圆角盖住。必须同时开 `WA_TranslucentBackground`（不铺那块方底）
    与 `WA_StyledBackground`（裸 QWidget 默认不画 QSS 的背景色，会变成全透明）。

    通过 `QStyle.PE_Widget` 让样式引擎真画一遍背景再数四角 alpha：直接
    `widget.grab()`/`render()` 都拿不到 QSS 背景（量出来全 0，会假绿）。
    """
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication, QStyle, QStyleOption
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.theme import ThemeManager

app = QApplication.instance() or QApplication([])
app.setStyleSheet(ThemeManager.get_stylesheet())
window = ImageEditor()
window.resize(900, 600)
window.show()
app.processEvents()
window._rebuild_options_popup()
popup = window.options_menu

assert popup.objectName() == "optionsPopup", popup.objectName()
assert popup.testAttribute(Qt.WA_TranslucentBackground), "没开透明背景，圆角会被方底盖住"
assert popup.testAttribute(Qt.WA_StyledBackground), "没开样式背景，QSS 背景根本不会画"

popup.resize(220, 300)
popup.show()
app.processEvents()

image = QImage(popup.size(), QImage.Format_ARGB32)
image.fill(QColor(0, 0, 0, 0))
painter = QPainter(image)
option = QStyleOption()
option.initFrom(popup)
popup.style().drawPrimitive(QStyle.PE_Widget, option, painter, popup)
painter.end()

width, height = image.width(), image.height()

def alpha(x, y):
    return image.pixelColor(x, y).alpha()

# 四角（含底部两角，用户报的就是底部）都要被切掉
for name, (x, y) in (
    ("左上", (0, 0)),
    ("右上", (width - 1, 0)),
    ("左下", (0, height - 1)),
    ("右下", (width - 1, height - 1)),
):
    assert alpha(x, y) < 120, (name, alpha(x, y))

# 但边中点与中心必须仍然画出来——否则是「整个背景没画」而非「圆角」
assert alpha(width // 2, height // 2) > 200, alpha(width // 2, height // 2)
assert alpha(width // 2, height - 1) > 200, alpha(width // 2, height - 1)
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
