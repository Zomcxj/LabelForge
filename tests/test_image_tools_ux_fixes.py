"""面板交互问题：选区不生效、来源按钮、输入框高度。"""
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


_SETUP = '''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

def build(size=(640, 480)):
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "a.png")
    img = QImage(size[0], size[1], QImage.Format_RGB32)
    img.fill(QColor(0, 0, 0))
    img.save(path)
    window = ImageEditor()
    window.resize(1400, 900)
    window.show()
    window.background_images = [path]
    window.background_list.addItem("a.png")
    window.current_background_index = 0
    window.load_image_by_index(0)
    QTest.qWait(80)
    window._open_image_tools()
    QTest.qWait(80)
    return window

app = QApplication([])
'''


def test_selection_applies_without_touching_the_mode_combo():
    """默认就该能拉选区：下拉不再是生效的必要条件。"""
    out = _run_qt(_SETUP + '''
from PySide6.QtCore import QPoint, Qt
window = build()
panel = window._image_tools_panel
canvas = window.canvas
assert canvas.is_drawing_region is True, "打开面板后画布应当已进入选区绘制态"
rect = canvas.get_background_rect()
p0 = QPoint(int(rect.left()) + 10, int(rect.top()) + 10)
p1 = QPoint(int(rect.left()) + 60, int(rect.top()) + 60)
QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, p0)
QTest.mouseMove(canvas, p1)
QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, p1)
QTest.qWait(50)
assert window.image_tools_regions, "拖拽后应当产生选区"
panel._checks["invert"].setChecked(True)
QTest.qWait(400)
out = window.current_background.toImage()
inside = out.pixelColor(20, 20).name()
outside = out.pixelColor(300, 300).name()
assert inside == "#ffffff", inside
assert outside == "#000000", outside
print("OK")
''')
    assert "OK" in out


def test_selection_limits_processing_without_an_extra_checkbox():
    """有选区就只在选区内生效：不再提供「仅选区内生效」开关。"""
    out = _run_qt(_SETUP + '''
from PySide6.QtWidgets import QCheckBox
window = build()
panel = window._image_tools_panel
assert not any("仅选区内" in cb.text()
               for cb in panel.findChildren(QCheckBox)), "开关应当去掉"
assert panel.mask_enabled() is False, "没有选区时整图生效"
window.image_tools_regions.append(
    {"shape_type": "rectangle", "x": 10, "y": 10, "width": 20, "height": 20})
assert panel.mask_enabled() is True, "画了选区就只在选区内生效"
window.image_tools_regions.clear()
assert panel.mask_enabled() is False
print("OK")
''')
    assert "OK" in out


def test_geometric_operations_keep_the_selection_but_skip_the_mask():
    """几何类改尺寸不能做蒙版合成，但选区不该被删掉。"""
    out = _run_qt(_SETUP + '''
window = build()
panel = window._image_tools_panel
window.image_tools_regions.append(
    {"shape_type": "rectangle", "x": 10, "y": 10, "width": 20, "height": 20})
panel._checks["rotate"].setChecked(True)
assert panel.geometric_active() is True
assert len(window.image_tools_regions) == 1, "选区不该被清掉"
panel._checks["rotate"].setChecked(False)
assert panel.geometric_active() is False
assert panel.mask_enabled() is True, "取消几何项后选区照样生效"
print("OK")
''')
    assert "OK" in out


def test_panel_has_no_source_buttons():
    """图片只能从主界面选：面板里不留记忆目录/选文件夹。"""
    out = _run_qt(_SETUP + '''
from PySide6.QtWidgets import QPushButton
window = build()
texts = [b.text() for b in window._image_tools_panel.findChildren(QPushButton)]
assert not any("记忆" in t for t in texts), texts
assert not any("文件夹" in t for t in texts), texts
print("OK")
''')
    assert "OK" in out


def test_parameter_widgets_are_not_tall():
    """数字输入框比旁边的勾选框高出一大截，行距很难看。"""
    out = _run_qt(_SETUP + '''
window = build()
panel = window._image_tools_panel
heights = []
for widgets in panel._param_widgets.values():
    for _name, widget, _getter in widgets:
        heights.append(widget.maximumHeight())
assert heights, heights
assert max(heights) <= 26, sorted(set(heights))
check = panel._checks["bright"]
assert max(heights) <= check.sizeHint().height() + 6, (max(heights), check.sizeHint().height())
print("OK")
''')
    assert "OK" in out
