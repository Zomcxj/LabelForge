"""画布渲染回归测试。"""
import os
from pathlib import Path
import subprocess
import sys

from PySide6.QtCore import QRectF

from labelforge.engine.paste_engine import PasteEngineMixin


ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死。断言通过后立即
# flush + os._exit(0) 绕开析构；os._exit 不冲刷缓冲，必须先 flush。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def test_magnifier_shows_in_side_mode_regardless_of_selection():
    source = (ROOT / "labelforge" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")

    # v1.1.4: side magnifier is always visible; the old gating on selection must not exist.
    assert "if mag_pos == 'side' and self.selected_box is None and self._editor.selected_item is None and not self.is_drawing_box:" not in source
    assert "MAGNIFIER_CONFIG.get('position', 'side')" in source


def test_crosshair_is_drawn_only_in_annotate_mode_while_mouse_is_inside_canvas():
    source = (ROOT / "labelforge" / "canvas" / "canvas_renderer.py").read_text(encoding="utf-8")

    assert "if (getattr(self._editor, 'edit_mode', 'paste') == 'annotate' and" in source
    assert "self.mouse_inside and self._editor.current_background is not None and" in source
    assert "background_rect is not None):" in source
    assert "def _draw_crosshair(self, painter):" in source
    assert "CROSSHAIR_CONFIG" in source
    assert "CROSSHAIR_CONFIG.get('alpha', 160)" in source


def test_paste_engine_offsets_fully_overlapping_group_by_three_pixels():
    source = (ROOT / "labelforge" / "engine" / "paste_engine.py").read_text(encoding="utf-8")

    assert "OVERLAP_OFFSET_STEP = 3" in source
    assert "def _offset_overlapping_paste_group(" in source
    # 组里每项是 (rect, label[, box])，多出来的尾随元素要原样带着走，
    # 否则粘贴会丢掉置信度/形状类型。
    assert "for item in pasted_group:" in source
    assert "tuple(item[1:])" in source


def test_offset_overlapping_paste_group_keeps_boxes_inside_background_bounds():
    class FakeBackground:
        def width(self):
            return 10

        def height(self):
            return 10

    class Editor(PasteEngineMixin):
        def __init__(self):
            self.current_background = FakeBackground()
            self.detection_boxes = [{"x": 7, "y": 7, "width": 3, "height": 3, "label": "cat"}]

    editor = Editor()

    adjusted = editor._offset_overlapping_paste_group([(QRectF(7, 7, 3, 3), "cat")])

    rect, label = adjusted[0]
    assert rect.x() >= 0
    assert rect.y() >= 0
    assert rect.x() + rect.width() <= editor.current_background.width()
    assert rect.y() + rect.height() <= editor.current_background.height()


def test_point_brush_does_not_leak_into_later_rectangles():
    """画完关键点后残留的实心画刷不能污染后续矩形，否则矩形被填成实心色块。"""
    script = '''
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_label_names_checkbox = Check()
    def get_label_color(self, label):
        return {"nose": "#43A047", "car": "#1E88E5"}[label]

class Rect:
    def left(self): return 0
    def top(self): return 0

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.background_scale = 1.0
        self.selected_box = None
        self.selected_boxes = []

app = QApplication.instance() or QApplication([])

# 关键点（绿色）在前，矩形（蓝色）在后
renderer = Renderer()
renderer._editor.detection_boxes = [
    {"label": "nose", "shape_type": "point", "points": [[10.0, 10.0]],
     "x": 10.0, "y": 10.0, "width": 0, "height": 0},
    {"label": "car", "shape_type": "rectangle",
     "x": 40.0, "y": 40.0, "width": 60.0, "height": 60.0},
]

image = QImage(120, 120, QImage.Format_ARGB32)
image.fill(Qt.white)
painter = QPainter(image)
renderer._draw_detection_boxes(painter, Rect())
painter.end()

# 矩形中心应是白底 + 蓝色 alpha60 淡填充，而不是不透明绿色
px = image.pixelColor(70, 70)
expected = (round((0x1E * 60 + 255 * 195) / 255),
            round((0x88 * 60 + 255 * 195) / 255),
            round((0xE5 * 60 + 255 * 195) / 255))
assert px.getRgb()[:3] == expected, (px.getRgb()[:3], expected)
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_point_with_zero_bbox_still_renders():
    """载入的关键点 width/height 为 0，不能被矩形的退化判断跳过而不显示。"""
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_label_names_checkbox = Check()
    def get_label_color(self, label): return "#E53935"

class Rect:
    def left(self): return 0
    def top(self): return 0

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.background_scale = 1.0
        self.selected_box = None
        self.selected_boxes = []

app = QApplication.instance() or QApplication([])

image = QImage(60, 60, QImage.Format_ARGB32)
image.fill(Qt.white)
painter = QPainter(image)
editor = Renderer()
editor._editor.detection_boxes = [{
    "label": "nose", "shape_type": "point", "points": [[30.0, 30.0]],
    "x": 30.0, "y": 30.0, "width": 0, "height": 0,
}]
editor._draw_detection_boxes(painter, Rect())
painter.end()

# 圆心必须被点形状涂成标签色，而不是保持白底
px = image.pixelColor(30, 30)
assert (px.red(), px.green(), px.blue()) != (255, 255, 255), px.getRgb()
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_keypoint_selected_state_is_visually_distinct_from_idle():
    script = '''
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_label_names_checkbox = Check()
    def get_label_color(self, label): return "#E53935"

class Rect:
    def left(self): return 0
    def top(self): return 0

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.background_scale = 1.0

# 正常关键点：同组框存在且点在其内，走圆点绘制
box = {"label": "nose", "shape_type": "point", "points": [[30.0, 30.0]],
       "x": 30.0, "y": 30.0, "width": 1, "height": 1, "group_id": 1}
group_box = {"label": "person", "shape_type": "rectangle",
             "x": 10.0, "y": 10.0, "width": 40.0, "height": 40.0, "group_id": 1}

def render(is_selected):
    image = QImage(60, 60, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    r = Renderer()
    r._editor.detection_boxes = [group_box, box]
    r._draw_point_shape(painter, box, Rect(), is_selected, False)
    painter.end()
    # 采样圆心外沿：编辑态有白色外圈/更大实心点
    return [image.pixelColor(30 + r, 30).getRgb() for r in (2, 4, 5, 6)]

app = QApplication.instance() or QApplication([])
idle = render(False)
selected = render(True)
assert idle != selected, (idle, selected)
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_warned_keypoint_renders_as_square_with_white_outline():
    """异常关键点（无同组框/不在框内）画成方形 + 白色描边，与正常圆点不同。"""
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_label_names_checkbox = Check()
    detection_boxes = []
    def get_label_color(self, label): return "#E53935"

class Rect:
    def left(self): return 0
    def top(self): return 0

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.background_scale = 1.0

app = QApplication.instance() or QApplication([])

def render(box, boxes):
    image = QImage(60, 60, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    r = Renderer()
    r._editor.detection_boxes = boxes
    r._draw_point_shape(painter, box, Rect(), False, False)
    painter.end()
    return image

# 无同组框 -> 方形：角上 (cx-4, cy-4) 有颜色，圆点在该处应为白
warned = {"label": "nose", "shape_type": "point", "points": [[30.0, 30.0]],
          "x": 30.0, "y": 30.0, "width": 1, "height": 1, "group_id": 1}
ok = {"label": "nose", "shape_type": "point", "points": [[30.0, 30.0]],
      "x": 30.0, "y": 30.0, "width": 1, "height": 1, "group_id": 1}
group_box = {"label": "person", "shape_type": "rectangle",
             "x": 10.0, "y": 10.0, "width": 40.0, "height": 40.0, "group_id": 1}

warned_img = render(warned, [])
ok_img = render(ok, [group_box])
# 采样方形左上角区域：方形覆盖，圆点不覆盖
wp = warned_img.pixelColor(26, 26).getRgb()[:3]
op = ok_img.pixelColor(26, 26).getRgb()[:3]
assert wp != (255, 255, 255), wp
assert op == (255, 255, 255), op
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_keypoint_drawn_on_top_of_its_box():
    """关键点即使排在框之前，也要画在框上方（不被框填充遮盖）。"""
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_label_names_checkbox = Check()
    def get_label_color(self, label):
        return {"person": "#1E88E5", "nose": "#43A047"}[label]

class Rect:
    def left(self): return 0
    def top(self): return 0

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.background_scale = 1.0
        self.selected_box = None
        self.selected_boxes = []

app = QApplication.instance() or QApplication([])
renderer = Renderer()
# 关键点在前，同组框在后；框会覆盖该位置
renderer._editor.detection_boxes = [
    {"label": "nose", "shape_type": "point", "points": [[30.0, 30.0]],
     "x": 30.0, "y": 30.0, "width": 0, "height": 0, "group_id": 1},
    {"label": "person", "shape_type": "rectangle",
     "x": 10.0, "y": 10.0, "width": 40.0, "height": 40.0, "group_id": 1},
]

image = QImage(60, 60, QImage.Format_ARGB32)
image.fill(Qt.white)
painter = QPainter(image)
renderer._draw_detection_boxes(painter, Rect())
painter.end()

# 圆心应为关键点绿色，而不是被框的蓝色填充盖住
px = image.pixelColor(30, 30)
assert px.green() > px.blue(), px.getRgb()
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_boxes_use_label_color_fills_with_expected_alpha_states():
    script = '''
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication
from labelforge.canvas.canvas_renderer import CanvasRendererMixin

class Check:
    def isChecked(self): return False

class Editor:
    show_paste_names_checkbox = Check()
    show_label_names_checkbox = Check()
    def get_label_color(self, label): return "#e53935"

class Renderer(CanvasRendererMixin):
    def __init__(self):
        self._editor = Editor()
        self.hover_resize_target = None
        self.hover_resize_handle = None

def pixel_after(draw):
    image = QImage(60, 60, QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    draw(painter)
    painter.end()
    return image.pixelColor(30, 30)

def expected(alpha):
    color = QColor("#e53935")
    return QColor(
        round((color.red() * alpha + 255 * (255 - alpha)) / 255),
        round((color.green() * alpha + 255 * (255 - alpha)) / 255),
        round((color.blue() * alpha + 255 * (255 - alpha)) / 255),
    )

renderer = Renderer()
app = QApplication.instance() or QApplication([])
pixmap = QPixmap(40, 40)
pixmap.fill(Qt.white)
for draw in (
    lambda painter: renderer._draw_single_detection_box(painter, 10, 10, 40, 40, "cat", False, False),
    lambda painter: renderer._draw_single_paste_item(painter, pixmap, QRectF(10, 10, 40, 40), "cat", False, False),
):
    actual = pixel_after(draw)
    assert actual == expected(60), (actual.getRgb(), expected(60).getRgb())

for draw in (
    lambda painter: renderer._draw_single_detection_box(painter, 10, 10, 40, 40, "cat", True, False),
    lambda painter: renderer._draw_single_detection_box(painter, 10, 10, 40, 40, "cat", False, True),
    lambda painter: renderer._draw_single_paste_item(painter, pixmap, QRectF(10, 10, 40, 40), "cat", True, False),
    lambda painter: renderer._draw_single_paste_item(painter, pixmap, QRectF(10, 10, 40, 40), "cat", False, True),
):
    actual = pixel_after(draw)
    assert actual == expected(155), (actual.getRgb(), expected(155).getRgb())
'''
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT,
        env=os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
