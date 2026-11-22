"""图像编辑预览期间的画布锁：标注编辑与显示（真实 Qt 子进程）。"""
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


def test_annotations_are_locked_but_regions_stay_editable():
    out = _run_qt('''
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
src = QImage(64, 48, QImage.Format_RGB32)
src.fill(QColor(1, 2, 3))
window.current_background = src
assert window.canvas._can_edit_canvas() is True
window._image_tools_active = True
assert window.canvas._can_edit_canvas() is True, "区域拖动需要总闸放行"
assert window.canvas._region_editing_allowed() is True
before = len(window.detection_boxes)
for name, handler in (("box", window.canvas._handle_drawing_press),
                      ("polygon", window.canvas._handle_polygon_press),
                      ("point", window.canvas._handle_point_press)):
    assert handler(QPointF(5, 5)) is True, name
assert len(window.detection_boxes) == before, window.detection_boxes
assert window.canvas._hide_detection_boxes() is True, "图像编辑开着就该藏标注框"
window._image_tools_active = False
assert window.canvas._hide_detection_boxes() is False
print("OK")
''')
    assert "OK" in out
