"""整支评审提出的 2 条 Critical 与 6 条 Important 的回归测试。"""
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


_TWO_IMAGES = '''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

def make_dataset():
    folder = tempfile.mkdtemp()
    paths = []
    for name, color in (("a.png", (10, 0, 0)), ("b.png", (0, 20, 0))):
        path = os.path.join(folder, name)
        img = QImage(32, 32, QImage.Format_RGB32)
        img.fill(QColor(*color))
        img.save(path)
        paths.append(path)
    return paths

app = QApplication([])
'''

_STRIPES = '''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

def make_stripes():
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "stripes.png")
    img = QImage(2400, 1600, QImage.Format_RGB32)
    for x in range(img.width()):
        img.setPixelColor(x, 0, QColor(0, 0, 0) if x % 2 == 0 else QColor(255, 255, 255))
        for y in range(1, img.height()):
            img.setPixelColor(x, y, img.pixelColor(x, 0))
    img.save(path)
    return path

app = QApplication([])
'''


def test_switching_images_from_the_sidebar_reloads_the_panel():
    """Critical 1：快捷键/侧栏切图不会发 itemClicked，面板不得把旧图的结果画上去。"""
    out = _run_qt(_TWO_IMAGES + '''
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
for path in paths:
    window.background_list.addItem(os.path.basename(path))
window.background_list.setCurrentRow(0)
window.load_image_by_index(0)
window._open_image_tools()
window.switch_background_to_index(1)
assert window.current_background_index == 1
reloaded = window._image_tools_original.pixelColor(16, 16).name()
assert reloaded == "#001400", reloaded
window._image_tools_panel._checks["invert"].setChecked(True)
window._image_tools_apply()
inverted = window.current_background.toImage().pixelColor(16, 16).name()
assert inverted == "#ffebff", inverted
print("OK")
''')
    assert "OK" in out


def test_preview_pixels_are_never_saved_to_disk():
    """Critical 2：预览态下任何落盘路径都必须返回空。"""
    out = _run_qt(_TWO_IMAGES + '''
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window.canvas_items = [{"x": 0, "y": 0, "width": 8, "height": 8, "path": ""}]
assert window.save_manager._get_save_info_or_none() is not None
window._open_image_tools()
window._image_tools_panel._checks["invert"].setChecked(True)
window._image_tools_apply()
assert window.save_manager._get_save_info_or_none() is None
window.save_manager.auto_save_background()
puts = [p for p in os.listdir(os.path.dirname(paths[0])) if "output" in p]
assert not puts, puts
print("OK")
''')
    assert "OK" in out


def test_preview_refresh_is_debounced():
    """Important 3：参数连续变动时不应每一下都同步跑完处理链。"""
    out = _run_qt(_TWO_IMAGES + '''
from PySide6.QtTest import QTest
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
before = window.current_background.toImage().pixelColor(16, 16).name()
window._image_tools_panel._checks["invert"].setChecked(True)
now = window.current_background.toImage().pixelColor(16, 16).name()
assert now == before, "参数变化后应当还没刷新（防抖）"
QTest.qWait(400)
after = window.current_background.toImage().pixelColor(16, 16).name()
assert after == "#f5ffff", after
print("OK")
''')
    assert "OK" in out


def test_reopening_the_panel_does_not_leave_an_extra_panel():
    """Important 4：重开面板要销毁旧面板，不能留下第二份子树。"""
    out = _run_qt(_TWO_IMAGES + '''
from PySide6.QtCore import QCoreApplication, QEvent
from labelforge.ui.image_tools_panel import ImageToolsPanel
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
window._open_image_tools()
QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
panels = window.findChildren(ImageToolsPanel)
assert len(panels) == 1, len(panels)
print("OK")
''')
    assert "OK" in out


def test_selections_accumulate_on_the_canvas():
    """Important 5 的后续：多选区是需求本身，落笔后不再互相顶掉。"""
    out = _run_qt(_TWO_IMAGES + '''
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
canvas = window.canvas
canvas._start_add_region()
window.image_tools_regions.append(
    {"shape_type": "rectangle", "x": 1, "y": 2, "width": 4, "height": 5})
canvas._after_region_committed()
window.image_tools_regions.append(
    {"shape_type": "rectangle", "x": 3, "y": 3, "width": 6, "height": 6})
canvas._after_region_committed()
assert len(window.image_tools_regions) == 2, window.image_tools_regions
assert window.image_tools_regions[1]["x"] == 3, window.image_tools_regions
print("OK")
''')
    assert "OK" in out


def test_right_click_is_ignored_while_the_panel_is_open():
    """Important 6：预览态右键菜单能删标签文件/移除图片，必须拦住。"""
    out = _run_qt(_TWO_IMAGES + '''
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
calls = []
window.canvas._handle_right_click = lambda pos: calls.append(pos)
event = QMouseEvent(QEvent.MouseButtonPress, QPointF(10, 10), QPointF(10, 10),
                    Qt.RightButton, Qt.RightButton, Qt.NoModifier)
window.canvas.mousePressEvent(event)
assert calls, "面板没开时右键照常"
window._open_image_tools()
calls.clear()
window.canvas.mousePressEvent(event)
assert not calls, calls
print("OK")
''')
    assert "OK" in out


def test_opening_the_panel_with_no_operations_keeps_the_original_pixels():
    """Important 7：零勾选时不得把降采样再放大的糊图顶上画布。"""
    out = _run_qt(_STRIPES + '''
window = ImageEditor()
path = make_stripes()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
image = window.current_background.toImage()
assert image.pixelColor(0, 0).name() == "#000000", image.pixelColor(0, 0).name()
assert image.pixelColor(1, 0).name() == "#ffffff", image.pixelColor(1, 0).name()
print("OK")
''')
    assert "OK" in out


def test_failed_reload_clears_the_geometric_hide():
    """Important 8：读不到图时不能让上一轮的几何状态继续藏标注。"""
    out = _run_qt(_TWO_IMAGES + '''
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
window._image_tools_panel._checks["fliph"].setChecked(True)
window._image_tools_apply()
QTest_wait = None
assert window._image_tools_geometric is True
window.background_images = [os.path.join(os.path.dirname(paths[0]), "missing.png")]
window._image_tools_reload()
assert window._image_tools_original is None
assert window._image_tools_geometric is False
assert window.canvas._hide_detection_boxes() is False
assert window.image_tools_regions == []
print("OK")
''')
    assert "OK" in out


def test_panel_does_not_open_in_the_delete_view():
    """评审的 Declined to judge 项：删除视图下画布内容与数据集索引不一致。"""
    out = _run_qt(_TWO_IMAGES + '''
window = ImageEditor()
paths = make_dataset()
window.background_images = list(paths)
window.current_background_index = 0
window.load_image_by_index(0)
window._is_delete_view = True
window._open_image_tools()
assert getattr(window, "_image_tools_panel", None) is None
assert window._image_tools_active is False
print("OK")
''')
    assert "OK" in out
