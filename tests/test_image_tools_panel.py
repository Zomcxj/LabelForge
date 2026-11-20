"""图像处理面板的控件结构测试（真实 Qt 子进程）。"""
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


def test_panel_lists_every_operation_and_has_no_save_or_list():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication, QListWidget, QPushButton
from labelforge.engine import image_tools as it
from labelforge.ui.image_tools_panel import ImageToolsPanel
app = QApplication([])
panel = ImageToolsPanel(None)
assert sorted(panel._checks) == sorted(it.OPERATION_ORDER)
texts = [b.text() for b in panel.findChildren(QPushButton)]
assert "保存结果" not in texts, texts
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


def test_geometric_preview_reports_that_annotations_are_hidden():
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
window._image_tools_panel._checks["fliph"].setChecked(True)
window._image_tools_apply()
assert "标注已隐藏" in window._image_tools_panel._status.text()
window._image_tools_panel._checks["fliph"].setChecked(False)
window._image_tools_apply()
assert "标注已隐藏" not in window._image_tools_panel._status.text()
print("OK")
''')
    assert "OK" in out


def test_no_save_paths_remain():
    source = (ROOT / "labelforge/ui/image_tools_panel.py").read_text(encoding="utf-8")
    assert "_processed" not in source
    assert "_save_btn" not in source
