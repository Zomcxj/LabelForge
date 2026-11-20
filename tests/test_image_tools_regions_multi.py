"""多选区与区域拖动（参考贴图区域的交互）。"""
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
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

def build():
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "a.png")
    img = QImage(640, 480, QImage.Format_RGB32)
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

def drag(canvas, x0, y0, x1, y1):
    rect = canvas.get_background_rect()
    p0 = QPoint(int(rect.left()) + x0, int(rect.top()) + y0)
    p1 = QPoint(int(rect.left()) + x1, int(rect.top()) + y1)
    QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, p0)
    QTest.mouseMove(canvas, p1)
    QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, p1)
    QTest.qWait(30)

app = QApplication([])
'''


def test_two_drags_keep_two_regions_and_the_mask_covers_both():
    out = _run_qt(_SETUP + '''
window = build()
panel = window._image_tools_panel
canvas = window.canvas
drag(canvas, 20, 20, 90, 90)
window._image_tools_panel._add_btn.click()
drag(canvas, 300, 250, 380, 330)
assert len(window.image_tools_regions) == 2, window.image_tools_regions
panel._checks["invert"].setChecked(True)
QTest.qWait(400)
out = window.current_background.toImage()
r1, r2 = window.image_tools_regions
c1 = (int(r1["x"] + r1["width"] / 2), int(r1["y"] + r1["height"] / 2))
c2 = (int(r2["x"] + r2["width"] / 2), int(r2["y"] + r2["height"] / 2))
first = out.pixelColor(*c1).name()
second = out.pixelColor(*c2).name()
gap = out.pixelColor(int(r1["x"] + r1["width"] + 40),
                     int(r1["y"] + r1["height"] + 40)).name()
assert first == "#ffffff", first
assert second == "#ffffff", second
assert gap == "#000000", gap
print("OK")
''')
    assert "OK" in out


def test_an_existing_region_can_be_dragged():
    """面板开着时，点住已有区域拖动应当移动它，而不是再画一个。"""
    out = _run_qt(_SETUP + '''
window = build()
canvas = window.canvas
drag(canvas, 20, 20, 120, 120)
assert len(window.image_tools_regions) == 1, window.image_tools_regions
before = window.image_tools_regions[0]["x"]
drag(canvas, 70, 70, 170, 170)
assert len(window.image_tools_regions) == 1, window.image_tools_regions
after = window.image_tools_regions[0]["x"]
assert after > before, (before, after)
print("OK")
''')
    assert "OK" in out


def test_panel_shows_no_path_and_no_navigation():
    """面板顶部不再显示路径，也没有上一张/下一张。"""
    out = _run_qt(_SETUP + '''
from PySide6.QtWidgets import QPushButton
window = build()
panel = window._image_tools_panel
texts = [b.text() for b in panel.findChildren(QPushButton)]
assert not any("上一张" in t or "下一张" in t for t in texts), texts
assert "a.png" not in panel._info.text() if hasattr(panel, "_info") else True
assert not hasattr(panel, "_info"), "顶部信息行应当整行去掉"
print("OK")
''')
    assert "OK" in out
