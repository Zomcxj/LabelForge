"""图像编辑面板的选区落在主画布的区域层上（真实 Qt 子进程）。"""
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


def test_image_tools_regions_take_over_the_canvas_region_layer():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
window.region_boxes = [{"shape_type": "rectangle", "x": 1, "y": 2, "width": 3, "height": 4}]
window.image_tools_regions = []
window._image_tools_active = True
assert window.canvas._active_regions() is window.image_tools_regions
assert window.canvas._show_regions() is True
assert window.canvas._region_editing_allowed() is True
window._image_tools_active = False
assert window.canvas._active_regions() is window.region_boxes
print("OK")
''')
    assert "OK" in out


def test_panel_region_mode_drives_the_canvas_drawing_state():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
window.image_tools_regions = []
window._image_tools_active = True
panel = window._image_tools_panel if hasattr(window, "_image_tools_panel") else None
from labelforge.ui.image_tools_panel import ImageToolsPanel
panel = ImageToolsPanel(window, window)
window._image_tools_panel = panel
panel._mode_combo.setCurrentIndex(1)
assert window.image_tools_region_mode == "rect"
panel._add_btn.click()
assert window.canvas.is_drawing_region is True
window.canvas._reset_region_drawing_state()
panel._mode_combo.setCurrentIndex(2)
assert window.image_tools_region_mode == "polygon"
panel._add_btn.click()
assert window.canvas.is_drawing_region_polygon is True
window.canvas._reset_region_drawing_state()
panel._mode_combo.setCurrentIndex(0)
assert window.image_tools_region_mode == "none"
panel._add_btn.click()
assert window.canvas.is_drawing_region is False
assert window.canvas.is_drawing_region_polygon is False
print("OK")
''')
    assert "OK" in out
