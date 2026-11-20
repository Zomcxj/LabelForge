"""图像处理工具的入口与文案测试（真实 Qt 子进程）。"""
import os
import subprocess
import sys
from pathlib import Path

from labelforge.engine import image_tools as it
from labelforge.ui import i18n

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen",
                        "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c",
         script + "\nimport sys\nsys.stdout.flush()\nsys.exit(0)\n"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=240,
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_export_menu_lists_image_tools():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication, QMenu
from labelforge.ui.main_window import ImageEditor

captured = {}
def fake_exec(self, *_args, **_kwargs):
    captured["texts"] = [a.text() for a in self.actions()]
    return None

QMenu.exec_ = fake_exec
app = QApplication([])
window = ImageEditor()
window._show_export_menu()
assert any("图像处理" in text for text in captured["texts"]), captured["texts"]
print("OK")
''')
    assert "OK" in out


def test_open_image_tools_creates_the_dialog():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
assert hasattr(window, "_open_image_tools")
window._open_image_tools()
assert window._image_tools_panel is not None
assert window._image_tools_panel.windowTitle() == "图像处理"
window._open_image_tools()
assert window._image_tools_panel is not None
print("OK")
''')
    assert "OK" in out


def test_every_operation_label_and_group_has_english():
    """操作名与分组名都要有英文，否则英文界面会显示中文。"""
    missing = []
    for key in it.OPERATION_ORDER:
        group, label, _params = it.OPERATION_META[key]
        for text in (group, label):
            if i18n._strings["en"].get(text, text) == text and not text.isascii():
                missing.append(text)
    assert not missing, sorted(set(missing))


def test_dialog_strings_have_english():
    keys = [
        "图像处理", "未打开图片", "选区:", "无选区", "矩形", "椭圆", "多边形",
        "清除选区", "仅选区内生效", "打开图片", "保存结果", "无法读取该图片",
        "选择图片", "请先打开一张图片", "没有勾选任何处理项", "已保存", "保存失败",
        "几何类操作会改变画布尺寸, 标注不再适用",
        "图片文件 (*.png *.jpg *.jpeg *.bmp)",
    ]
    for key in keys:
        assert key in i18n._strings["zh"], key
        assert i18n._strings["en"].get(key, key) != key, key


def test_preview_swaps_the_canvas_and_restores_on_close():
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "a.png")
img = QImage(64, 48, QImage.Format_RGB32)
img.fill(QColor(10, 20, 30))
img.save(path)
window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
before = window.current_background.toImage().pixelColor(32, 24).name()
assert window._image_tools_active is False
window._open_image_tools()
assert window._image_tools_active is True
assert window._image_tools_original is not None
window._image_tools_panel._checks["invert"].setChecked(True)
window._image_tools_apply()
after = window.current_background.toImage().pixelColor(32, 24).name()
assert after != before, (before, after)
window._image_tools_close()
restored = window.current_background.toImage().pixelColor(32, 24).name()
assert restored == before, (before, restored)
assert window._image_tools_active is False
print("OK")
''')
    assert "OK" in out


def test_open_panel_without_a_dataset_does_not_crash():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
window._open_image_tools()
assert window._image_tools_original is None
window._image_tools_apply()
window._image_tools_close()
assert window._image_tools_active is False
print("OK")
''')
    assert "OK" in out


def test_switch_background_does_not_overrun_a_single_image_dataset():
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "only.png")
img = QImage(16, 16, QImage.Format_RGB32)
img.fill(QColor(9, 9, 9))
img.save(path)
window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
window.switch_background(1)
assert window.current_background_index == 0
window.switch_background(-1)
assert window.current_background_index == 0
print("OK")
''')
    assert "OK" in out
