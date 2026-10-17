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


def _rule(style, selector):
    return style.split(f"{selector} {{", 1)[1].split("}", 1)[0]


@pytest.mark.parametrize("mode, color", [("LIGHT", "#1f1f1f"), ("DARK", "#f5f5f5")])
def test_real_qt_uses_theme_text_for_representative_normal_controls(mode, color):
    script = f'''
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.settings_dialog import SettingsDialog
from labelforge.ui.theme import ThemeManager, ThemeMode

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


# ---------- 右键悬浮菜单圆角 ----------

def _menu_corner_probe(script, timeout=60):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=ROOT, env=env,
        text=True, capture_output=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]


@pytest.mark.parametrize("mode", ["LIGHT", "DARK"])
def test_real_qt_context_menu_has_rounded_corners(mode):
    """右键悬浮菜单的**四角必须被渲染成透明**（真圆角，不是只写了 QSS）。

    只断言样式表里有 `border-radius` 是空转的：菜单没设无边框窗口标志时，
    Qt 会照旧画方角，QSS 的圆角被窗口背景盖住，肉眼看还是方的。所以这里
    真的把菜单渲染一遍，然后检查四个角上的像素是不是透明的。
    """
    out = _menu_corner_probe(f'''
from PySide6.QtWidgets import QApplication, QMenu
from PySide6.QtGui import QAction, QImage, QPainter
from PySide6.QtCore import QPoint
from labelforge.ui.theme import ThemeManager, ThemeMode

app = QApplication.instance() or QApplication([])
ThemeManager.set_mode(ThemeMode.{mode})
app.setStyleSheet(ThemeManager.get_stylesheet())

# 必须走真的构造路径：QSS 的圆角只有配上无边框+透明窗口才生效，
# 自己 new 一个裸 QMenu 测不到这一点。
from labelforge.canvas.canvas_menu import CanvasMenuMixin
menu = CanvasMenuMixin._rounded_menu(None)
menu.addAction(QAction("删除", menu))
menu.addAction(QAction("复制", menu))
menu.resize(160, 80)
menu.ensurePolished()

img = QImage(160, 80, QImage.Format_ARGB32)
img.fill(0)
painter = QPainter(img)
menu.render(painter, QPoint())
painter.end()

# 四角与中心：角应透明、中心应不透明
corners = [(0, 0), (159, 0), (0, 79), (159, 79)]
for x, y in corners:
    print("corner", x, y, img.pixelColor(x, y).alpha())
print("center", img.pixelColor(80, 40).alpha())
''')
    alphas = [int(ln.split()[-1]) for ln in out if ln.startswith("corner")]
    center = int([ln for ln in out if ln.startswith("center")][0].split()[-1])
    assert len(alphas) == 4, out
    assert center > 200, ("菜单中心应不透明", center)
    assert all(a < 200 for a in alphas), ("四角应被切成透明（圆角）", alphas)


@pytest.mark.parametrize("mode", ["LIGHT", "DARK"])
def test_options_menu_keeps_its_own_rule(mode):
    """选项菜单（optionsMenu）的专用样式仍在：全局 QMenu 规则不能把它顶掉。

    这条在进程内断言（不需要渲染）：全局加圆角时最容易踩的坑是顺手把
    对象名规则的圆角/内边距删掉，或写错层级让专用规则失效。
    """
    ThemeManager.set_mode(getattr(ThemeMode, mode))
    css = ThemeManager.get_stylesheet()
    # 全局 QMenu 规则存在（右键悬浮菜单靠它拿圆角）
    assert "QMenu {" in css
    # 选项菜单的专用规则仍在，且同样有圆角
    assert "QMenu#optionsMenu {" in css
    block = css[css.index("QMenu#optionsMenu {"):]
    block = block[:block.index("}")]
    assert "border-radius" in block, block
