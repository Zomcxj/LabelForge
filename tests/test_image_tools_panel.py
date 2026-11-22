"""图像编辑面板的控件结构测试（真实 Qt 子进程）。"""
import os
import subprocess
import sys
from pathlib import Path

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


def test_panel_lists_every_operation_and_has_no_list():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication, QListWidget, QPushButton
from labelforge.engine import image_tools as it
from labelforge.ui.image_tools_panel import ImageToolsPanel
app = QApplication([])
panel = ImageToolsPanel(None)
assert sorted(panel._checks) == sorted(it.OPERATION_ORDER)
assert not panel.findChildren(QListWidget)
assert panel.region_mode() == "rect"
print("OK")
''')
    assert "OK" in out


def test_flatten_alpha_composites_transparent_pixels_onto_white():
    out = _run_qt('''
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.image_tools_panel import _flatten_alpha
app = QApplication([])
src = QImage(4, 4, QImage.Format_ARGB32)
src.fill(QColor(0, 0, 0, 0))
flat = _flatten_alpha(src)
assert not flat.hasAlphaChannel()
assert flat.pixelColor(1, 1).lightness() > 200, flat.pixelColor(1, 1).name()
print("OK")
''')
    assert "OK" in out


def test_annotations_stay_hidden_while_editing():
    """图像编辑开着就提示标注已隐藏，与勾没勾处理项无关。"""
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "a.png")
img = QImage(40, 30, QImage.Format_RGB32)
img.fill(QColor(10, 20, 30))
img.save(path)
window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
# 状态提示常显：勾不勾处理项都一样
assert "标注已隐藏" in window._image_tools_panel._status.text()
window._image_tools_panel._checks["fliph"].setChecked(True)
window._image_tools_apply()
assert "标注已隐藏" in window._image_tools_panel._status.text()
window._image_tools_panel._checks["fliph"].setChecked(False)
window._image_tools_apply()
assert "标注已隐藏" in window._image_tools_panel._status.text(), "取消勾选后提示仍在"
print("OK")
''')
    assert "OK" in out


def test_saving_asks_the_host_and_uses_the_engine_writer():
    """面板只发信号，写盘一律走 engine 那一份实现。"""
    out = _run_qt('''
from PySide6.QtWidgets import QApplication, QPushButton
from labelforge.engine import image_tools as it
from labelforge.ui.image_tools_panel import ImageToolsPanel
app = QApplication([])
panel = ImageToolsPanel(None)
texts = [b.text() for b in panel.findChildren(QPushButton)]
assert "保存图片" in texts, texts
assert "另存为…" in texts, texts
# 保存与另存为是两个独立按钮，不再藏在菜单里
assert not hasattr(panel, "_save_menu"), "保存结果不该再挂菜单"
calls = []
panel.save_new_requested.connect(lambda: calls.append("new"))
panel.save_copy_requested.connect(lambda: calls.append("copy"))
panel._save_btn.click()
panel._save_copy_btn.click()
assert calls == ["new", "copy"], calls
assert callable(it.save_result)
source = open("labelforge/ui/image_tools_panel.py", encoding="utf-8").read()
assert "_processed" not in source
print("OK")
''')
    assert "OK" in out
