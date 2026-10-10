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


def _rule(style, selector):
    return style.split(f"{selector} {{", 1)[1].split("}", 1)[0]


@pytest.mark.parametrize("mode, color", [("LIGHT", "#1f1f1f"), ("DARK", "#f5f5f5")])
def test_real_qt_uses_theme_text_for_representative_normal_controls(mode, color):
    script = f'''
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.main_window import ImageEditor
from pastelabel.ui.settings_dialog import SettingsDialog
from pastelabel.ui.theme import ThemeManager, ThemeMode

app = QApplication.instance() or QApplication([])
window = ImageEditor()
ThemeManager.set_mode(ThemeMode.{mode})
window._apply_theme()
settings = SettingsDialog(window)
widgets = [
    (window.background_list, QPalette.Text),
    (window.label_list, QPalette.Text),
    (window.prev_img_btn, QPalette.ButtonText),
    (settings.findChild(type(window.paste_count_lbl)), QPalette.WindowText),
    (settings.grid_width_spin, QPalette.Text),
]
for widget, role in widgets:
    widget.ensurePolished()
    actual = widget.palette().color(role).name()
    assert actual == "{color}", (type(widget).__name__, actual, widget.styleSheet())
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode, color", [(ThemeMode.LIGHT, "#1F1F1F"), (ThemeMode.DARK, "#F5F5F5")])
def test_hover_button_rules_do_not_override_the_unified_text_color(mode, color):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()

    for selector in ("QPushButton:hover", "QPushButton#navBtn:hover", "QPushButton#langBtn:hover"):
        block = _rule(style, selector)
        if "color:" in block:
            assert f"color: {color};" in block
