# -*- coding: utf-8 -*-
"""贴图融合接入画布渲染：旋转、投影外扩、缓存与老路径分流。

conftest 在导入期把 PySide6 换成桩（`QRectF` 恒返回同一个矩形、`QPainter`
全是空操作），几何、落笔位置和融合位图在桩下都无从断言，所以这里按项目
既有做法跑真 Qt 子进程 —— 与 `tests/test_canvas_hit_test.py` 同一套写法：
子进程里用假 editor / 假 painter 记录绘制调用，父进程按 `KEY value` 行断言。

脚本只跑一次（五段检查写在同一份脚本里），避免每个用例都重新起一次 Qt 和
cv2。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程退出用 sys.exit（走正常 finalize），不用 os._exit：后者仍会经
# ExitProcess 卸载 DLL 触发 Qt 静态析构。
_QT_EXIT_GUARD = """
import sys
sys.stdout.flush()
sys.exit(0)
"""

_SCRIPT = r"""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication

from labelforge.canvas.canvas_renderer import CanvasRendererMixin
from labelforge.engine.paste_item import PasteItem

app = QApplication.instance() or QApplication([])

BACKGROUND_RECT = QRectF(0, 0, 400, 400)


class Check:
    '''复选框替身：类别名关掉，绘制只看本体。'''

    def isChecked(self):
        return False


class Editor:
    show_paste_names_checkbox = Check()
    show_label_names_checkbox = Check()
    pressed_label = None
    selected_item = None
    current_background = None

    def __init__(self):
        self.canvas_items = []

    def get_label_color(self, label):
        return "#E53935"


class RecordingPainter:
    '''记录落笔位置的 painter 替身，只覆盖贴图绘制用到的方法。'''

    def __init__(self):
        self.ops = []
        self.origin = [0.0, 0.0]
        self.drawn = []     # 每次 drawPixmap 的中心/尺寸/当时原点
        self.last_angle = None

    def save(self):
        self.ops.append("save")

    def restore(self):
        self.ops.append("restore")

    def translate(self, point):
        self.origin[0] += point.x()
        self.origin[1] += point.y()
        self.ops.append("translate")

    def rotate(self, angle):
        self.ops.append("rotate")
        self.last_angle = angle

    def setPen(self, pen):
        self.ops.append("setPen")

    def drawRect(self, *args):
        self.ops.append("drawRect")

    def drawPolygon(self, polygon):
        self.ops.append("drawPolygon")

    def fillRect(self, *args):
        self.ops.append("fillRect")

    def drawPixmap(self, *args):
        self.ops.append("drawPixmap")
        if len(args) == 5:                     # 老路径：x, y, w, h, pixmap
            x, y, w, h = (float(v) for v in args[:4])
            center = (x + w / 2.0, y + h / 2.0)
            size = (w, h)
        else:                                  # 旋转路径：(QRect, pixmap)
            rect = args[0]
            center = (rect.center().x(), rect.center().y())
            size = (float(rect.width()), float(rect.height()))
        self.drawn.append({
            "center": center,
            "size": size,
            "origin": tuple(self.origin),
            "pixmap_size": (args[-1].width(), args[-1].height()),
        })


class Renderer(CanvasRendererMixin):

    def __init__(self, scale=1.0):
        self._editor = Editor()
        self.background_scale = scale
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.shape_opacity = 1.0


def sprite(size=40):
    image = QImage(size, size, QImage.Format_ARGB32)
    image.fill(QColor(200, 30, 30))
    return QPixmap.fromImage(image)


def item(rect, **kw):
    return PasteItem(sprite(), rect, "cat", **kw)


def draw(items, scale=1.0, background=None):
    renderer = Renderer(scale)
    renderer._editor.canvas_items = items
    renderer._editor.current_background = background
    painter = RecordingPainter()
    renderer._draw_paste_items(painter, BACKGROUND_RECT)
    return renderer, painter


# ---------- 1. 角度为 0 的贴图仍走老路径 ----------

default_item = item(QRectF(10, 10, 40, 40))

legacy_calls = []
legacy_renderer = Renderer(1.0)
legacy_renderer._editor.canvas_items = [default_item]
legacy_renderer._draw_single_paste_item = (
    lambda *a, **kw: legacy_calls.append(a))
legacy_painter = RecordingPainter()
legacy_renderer._draw_paste_items(legacy_painter, BACKGROUND_RECT)
print("LEGACY_CALLS", len(legacy_calls))
if legacy_calls:
    # (painter, pixmap, item_rect, label, is_selected, is_pressed, index)
    print("LEGACY_ARGS", legacy_calls[0][3], legacy_calls[0][4],
          legacy_calls[0][5], legacy_calls[0][6])
print("LEGACY_OPS", ",".join(legacy_painter.ops))

# 老路径的绘制顺序（边框 → 本体）靠 `_draw_single_paste_item` 保持
order_painter = RecordingPainter()
order_renderer = Renderer(1.0)
order_renderer._draw_single_paste_item(
    order_painter, sprite(), QRectF(10, 10, 40, 40), "cat", False, False)
print("ORDER_OPS", *order_painter.ops)

# ---------- 2. 有角度时绕贴图中心旋转，不再走老路径 ----------

rotated_item = item(QRectF(20, 30, 60, 40), angle=90.0)
rotated, rotated_painter = draw([rotated_item])
rotated_draw = rotated_painter.drawn[0]
print("ROTATE", rotated_painter.last_angle)
print("TRANSLATE", float(rotated_painter.origin[0]),
      float(rotated_painter.origin[1]))
print("PIXMAP_CENTER",
      float(rotated_draw["center"][0] + rotated_draw["origin"][0]),
      float(rotated_draw["center"][1] + rotated_draw["origin"][1]))
print("PIXMAP_SIZE", int(rotated_draw["size"][0]), int(rotated_draw["size"][1]))
print("SAVE_RESTORE", rotated_painter.ops.count("save"),
      rotated_painter.ops.count("restore"))

rotated_single_calls = []
rotated_renderer = Renderer(1.0)
rotated_renderer._editor.canvas_items = [rotated_item]
rotated_renderer._draw_single_paste_item = (
    lambda *a, **kw: rotated_single_calls.append(a))
rotated_renderer._draw_paste_items(RecordingPainter(), BACKGROUND_RECT)
print("ROTATED_SINGLE_CALLS", len(rotated_single_calls))

print("OK")
"""


def _run():
    """跑一次真 Qt 子进程脚本，按 `KEY value...` 行解析成字典。"""
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
        # 子进程按 UTF-8 写、父进程按 UTF-8 读，不看本机 locale
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, (result.stdout or "") + (result.stderr or "")
    out = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split()
        if parts:
            out[parts[0]] = parts[1:]
    return out


_OUTPUT = {}


def _output():
    """记录式脚本只跑一次：所有断言共用同一份真 Qt 输出。"""
    if not _OUTPUT:
        _OUTPUT.update(_run())
    return _OUTPUT


def test_default_item_keeps_the_legacy_draw_path():
    out = _output()
    assert out["LEGACY_CALLS"] == ["1"], out["LEGACY_CALLS"]
    assert out["LEGACY_ARGS"] == ["cat", "False", "False", "0"], out["LEGACY_ARGS"]
    # 有角度的贴图不走老路径，改由旋转分支画
    assert out["ROTATED_SINGLE_CALLS"] == ["0"], out["ROTATED_SINGLE_CALLS"]


def test_legacy_draw_order_is_frame_body_extras():
    """拆成 frame/body/extras 三段之后，顺序与像素都不能变。"""
    out = _output()
    assert out["ORDER_OPS"] == [
        "setPen", "drawRect", "drawPixmap", "fillRect"], out["ORDER_OPS"]


def test_rotated_item_rotates_the_painter():
    out = _output()
    assert [float(v) for v in out["ROTATE"]] == [90.0], out["ROTATE"]
    # 绕贴图中心旋转：transl(translate) 落在贴图矩形中心 (20+60/2, 30+40/2)
    assert [float(v) for v in out["TRANSLATE"]] == [50.0, 50.0], out["TRANSLATE"]
    cx, cy = (float(v) for v in out["PIXMAP_CENTER"])
    assert abs(cx - 50.0) <= 1.0 and abs(cy - 50.0) <= 1.0, out["PIXMAP_CENTER"]
    # 没有投影时落笔位图就是贴图本身的大小
    assert [int(v) for v in out["PIXMAP_SIZE"]] == [60, 40], out["PIXMAP_SIZE"]
    # 旋转包在 save/restore 之间，不能污染后续绘制
    assert out["SAVE_RESTORE"] == ["1", "1"], out["SAVE_RESTORE"]


# ---------- 真像素：离屏 Qt 画一遍，回读画布 ----------

# 记录式替身只能证明"方法被调用过"，证明不了旋转后落笔范围真的变了、也证明
# 不了标签色没涂到投影上。这一段在子进程里真画到 QImage 上，再按像素核对，
# 写法与 tests/test_video_render.py 的 `_run_qt` 一致。
_PAINT_SCRIPT = '''
import sys
sys.path.insert(0, r"%s")

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

app = QApplication([])

from labelforge.canvas.canvas_renderer import CanvasRendererMixin
from labelforge.engine.paste_item import PasteItem

SIZE = 200
SPRITE_W, SPRITE_H = 40, 20
ITEM_RECT = QRectF(80, 90, SPRITE_W, SPRITE_H)      # 居中在 200x200 里


class Check:
    def isChecked(self):
        return False


class Editor:
    show_paste_names_checkbox = Check()
    show_label_names_checkbox = Check()
    pressed_label = None
    current_background = None

    def __init__(self, selected_item):
        self.canvas_items = []
        self.selected_item = selected_item

    def get_label_color(self, label):
        # 纯绿：和纯红贴图不同通道，涂错位置一眼就能数出来
        return "#00FF00"


class Renderer(CanvasRendererMixin):
    def __init__(self, selected_item=None):
        self._editor = Editor(selected_item)
        self.background_scale = 1.0
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.shape_opacity = 1.0


def sprite():
    image = QImage(SPRITE_W, SPRITE_H, QImage.Format_RGBA8888)
    image.fill(QColor(255, 0, 0, 255))
    return QPixmap.fromImage(image)


def paint(selected=False, **params):
    renderer = Renderer(0 if selected else None)
    renderer._editor.canvas_items = [
        PasteItem(sprite(), ITEM_RECT, "cat", **params)]
    canvas = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    try:
        renderer._draw_paste_items(painter, QRectF(0, 0, SIZE, SIZE))
    finally:
        painter.end()
    return canvas


def extent(canvas):
    """不透明像素的外接框 (x0, y0, x1, y1)。"""
    xs, ys = [], []
    for y in range(canvas.height()):
        for x in range(canvas.width()):
            if canvas.pixelColor(x, y).alpha() > 200:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


box0 = extent(paint(angle=0.0))
box90 = extent(paint(angle=90.0))

selected = paint(selected=True)
# 贴图自身覆盖 [80, 120) x [90, 110) 的整数像素；留 1px 余量避开填充边界。
# 选中态的缩放手柄是标签色实心圆、画在右下角，它本身不是「色层溢出」，
# 连它所在的一角一起排除（右下角那块之外的贴图外像素只有边框）。
green_outside = 0
outside_painted = 0
green_inside = 0
for y in range(SIZE):
    for x in range(SIZE):
        color = selected.pixelColor(x, y)
        inside = 79 <= x <= 120 and 89 <= y <= 110
        if inside:
            if color.green() > color.red() + 40:
                green_inside += 1
            continue
        if x >= 108 and y >= 100:
            continue
        if color.alpha() <= 20:
            continue
        outside_painted += 1
        if color.green() > color.red() + 40:
            green_outside += 1

print("RESULT", repr({
    "box0": box0,
    "box90": box90,
    "green_outside": green_outside,
    "outside_painted": outside_painted,
    "green_inside": green_inside,
}))
'''


def _run_qt(script):
    """子进程里跑一遍离屏 Qt 脚本，返回子进程打印的 RESULT 行（已 eval）。"""
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-c", script % str(ROOT)],
        cwd=ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


def test_painted_pixels_rotate_and_tint_stays_on_the_paste():
    """真画一遍：0° 是横的、90° 是竖的，标签色不落到贴图本体之外。"""
    result = _run_qt(_PAINT_SCRIPT)

    x0, y0, x1, y1 = result["box0"]
    assert (x1 - x0 + 1) > (y1 - y0 + 1), result["box0"]
    assert abs((x1 - x0 + 1) - 40) <= 2, result["box0"]
    assert abs((y1 - y0 + 1) - 20) <= 2, result["box0"]

    x0, y0, x1, y1 = result["box90"]
    assert (y1 - y0 + 1) > (x1 - x0 + 1), result["box90"]
    assert abs((x1 - x0 + 1) - 20) <= 2, result["box90"]
    assert abs((y1 - y0 + 1) - 40) <= 2, result["box90"]
    # 绕贴图中心 (99.5, 99.5) 旋转：两个方向都必须居中
    assert abs((x0 + x1) / 2 - 99.5) <= 2, result["box90"]
    assert abs((y0 + y1) / 2 - 99.5) <= 2, result["box90"]

    # 标签色只盖在贴图本体上：本体内有，本体外一个绿色占优的都没有
    assert result["green_inside"] > 0, result
    assert result["green_outside"] == 0, result


# ---------- 旋转：形状派生 / 菜单接线 / 保存写盘 / 状态栏任务 ----------


def test_rotating_a_paste_item_switches_it_to_rotation():
    from labelforge.engine.paste_item import PasteItem

    item = PasteItem(pixmap=object(), rect=(0, 0, 10, 10), label="car")
    assert item.shape_type == "rectangle"
    item.angle = 90.0
    assert item.shape_type == "rotation"
    item.angle = 0.0
    assert item.shape_type == "rectangle"


def test_rotated_paste_is_written_as_a_rotation_shape():
    from labelforge.engine.save_manager import SaveManagerMixin
    from labelforge.engine.paste_item import PasteItem
    from labelforge.engine.shape_io import labelme_shape_from_box

    item = PasteItem(pixmap=object(), rect=(10, 10, 20, 10), label="car", angle=45.0)
    shape = SaveManagerMixin._paste_item_to_shape(item)
    assert shape["shape_type"] == "rotation"
    assert shape["flags"] == {"paste": True}
    assert len(shape["points"]) == 4


def test_unrotated_paste_stays_a_rectangle_shape():
    from labelforge.engine.save_manager import SaveManagerMixin
    from labelforge.engine.paste_item import PasteItem

    item = PasteItem(pixmap=object(), rect=(10, 10, 20, 10), label="car")
    shape = SaveManagerMixin._paste_item_to_shape(item)
    assert shape["shape_type"] == "rectangle"
    assert shape["points"] == [[10, 10], [30, 10], [30, 20], [10, 20]]


def test_export_task_follows_paste_shapes():
    from labelforge.engine.yolo_exporter import detect_task
    from labelforge.engine.paste_item import PasteItem

    def boxes_for(items):
        return [{"shape_type": i.shape_type} for i in items]

    plain = [PasteItem(object(), (0, 0, 5, 5), "a")]
    assert detect_task(boxes_for(plain)) == "det"

    rotated = [PasteItem(object(), (0, 0, 5, 5), "a", angle=30.0)]
    assert detect_task(boxes_for(rotated)) == "obb"


def test_rotated_paste_corners_go_clockwise_from_the_top_left():
    """旋转框四点按 左上、右上、右下、左下 的顺序写盘（顺时针 90° 的例子）。"""
    import pytest

    from labelforge.engine.paste_item import PasteItem
    from labelforge.engine.save_manager import SaveManagerMixin

    item = PasteItem(pixmap=object(), rect=(10, 10, 20, 10), label="car",
                     angle=90.0)
    points = SaveManagerMixin._paste_item_to_shape(item)["points"]
    # 外接框 20x10 绕中心 (20, 15) 顺时针转 90°：变成 10x20 竖着的框
    assert [c for p in points for c in p] == pytest.approx(
        [25.0, 5.0, 25.0, 25.0, 15.0, 25.0, 15.0, 5.0], abs=1e-9)
    # 相邻两点依次相隔 20、10、20、10（绕行一圈，不是交叉顺序）
    lengths = []
    for i in range(4):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % 4]
        lengths.append(round(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, 6))
    assert lengths == [20.0, 10.0, 20.0, 10.0]


def _rotate_fixture(items):
    """一套最小的假 canvas + editor，够跑 `_rotate_paste_item`。"""
    from labelforge.canvas.canvas_menu import CanvasMenuMixin
    from labelforge.engine.undo_manager import UndoManager

    class Status:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    class Editor:
        _is_delete_view = False

        def __init__(self):
            self.canvas_items = list(items)
            self.detection_boxes = []
            self.undo_manager = UndoManager()
            self.status_label = Status()
            self.undo_calls = 0

        def save_undo_state(self):
            self.undo_calls += 1
            self.undo_manager.save_state(self.canvas_items, self.detection_boxes)

    class Canvas(CanvasMenuMixin):
        def __init__(self, editor):
            self._editor = editor
            self.painted = 0

        def update(self):
            self.painted += 1

    editor = Editor()
    return editor, Canvas(editor)


def test_rotate_paste_item_turns_the_item_into_an_oriented_box():
    from labelforge.engine.paste_item import PasteItem

    editor, canvas = _rotate_fixture([PasteItem(object(), (0, 0, 10, 10), "car")])
    canvas._rotate_paste_item(0, 90)

    item = editor.canvas_items[0]
    assert isinstance(item, PasteItem)
    assert item.angle == 90.0
    assert item.shape_type == "rotation"
    # 撤销栈推了一次，画布要重绘
    assert editor.undo_calls == 1
    assert canvas.painted == 1


def test_rotate_paste_item_normalizes_the_angle_and_can_be_undone():
    from labelforge.engine.paste_item import PasteItem
    from labelforge.ui.i18n import t as tr

    editor, canvas = _rotate_fixture([PasteItem(object(), (0, 0, 10, 10), "car")])

    canvas._rotate_paste_item(0, -90)
    # 逆时针 90° 归一化到 [0, 360)
    assert editor.canvas_items[0].angle == 270.0
    assert editor.status_label.text == tr("已切换为旋转框")

    # 撤销要能拿回 0°：快照里存的是对象引用，原地改角度会让撤销失效
    editor.canvas_items, editor.detection_boxes = editor.undo_manager.undo(
        editor.canvas_items, editor.detection_boxes)
    assert editor.canvas_items[0].angle == 0.0
    assert editor.canvas_items[0].shape_type == "rectangle"

    canvas._rotate_paste_item(0, 90)
    canvas._rotate_paste_item(0, None)
    assert editor.canvas_items[0].angle == 0.0
    assert editor.canvas_items[0].shape_type == "rectangle"
    assert editor.status_label.text == tr("已切回检测框")


def test_rotate_paste_item_ignores_a_missing_item():
    editor, canvas = _rotate_fixture([])

    canvas._rotate_paste_item(0, 90)

    assert editor.canvas_items == []
    # 没有可旋转的项：不推撤销栈、不提示
    assert editor.undo_calls == 0
    assert editor.status_label.text == ""


def test_export_task_label_shows_the_resolved_task():
    from labelforge.engine.paste_item import PasteItem
    from labelforge.ui.i18n import t as tr
    from labelforge.ui.main_window import ImageEditor

    prefix = f"{tr('导出任务')}: "

    class Label:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    class Editor:
        export_task_label = Label()
        canvas_items = []
        detection_boxes = []

    editor = Editor()
    ImageEditor._refresh_export_task_label(editor)
    assert editor.export_task_label.text == prefix + "det"

    editor.canvas_items = [PasteItem(object(), (0, 0, 5, 5), "a", angle=30.0)]
    ImageEditor._refresh_export_task_label(editor)
    assert editor.export_task_label.text == prefix + "obb"

    # 贴图归零后回到 det；检测框里的多边形/关键点同样参与判定
    editor.canvas_items = [PasteItem(object(), (0, 0, 5, 5), "a")]
    editor.detection_boxes = [{"shape_type": "polygon", "points": [[0, 0]]}]
    ImageEditor._refresh_export_task_label(editor)
    assert editor.export_task_label.text == prefix + "seg"

    editor.detection_boxes = [{"shape_type": "point", "group_id": 1}]
    ImageEditor._refresh_export_task_label(editor)
    assert editor.export_task_label.text == prefix + "pose"


def test_export_task_label_refresh_without_the_widget_is_a_no_op():
    from labelforge.ui.main_window import ImageEditor

    class Editor:
        canvas_items = []
        detection_boxes = []

    ImageEditor._refresh_export_task_label(Editor())


# ---------- 真像素：保存合成图按角度落笔 ----------

_SAVE_PAINT_SCRIPT = '''
import sys
sys.path.insert(0, r"%s")

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

app = QApplication([])

from labelforge.engine.paste_item import PasteItem
from labelforge.engine.save_manager import SaveManagerMixin

SIZE = 200
SPRITE_W, SPRITE_H = 40, 20
ITEM_RECT = QRectF(80, 90, SPRITE_W, SPRITE_H)      # 居中在 200x200 里


def sprite():
    image = QImage(SPRITE_W, SPRITE_H, QImage.Format_RGBA8888)
    image.fill(QColor(255, 0, 0, 255))
    return QPixmap.fromImage(image)


def composite(angle):
    canvas = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    try:
        SaveManagerMixin._draw_paste_items_on(
            painter, [PasteItem(sprite(), ITEM_RECT, "cat", angle=angle)])
    finally:
        painter.end()
    return canvas


def extent(canvas):
    xs, ys = [], []
    for y in range(canvas.height()):
        for x in range(canvas.width()):
            if canvas.pixelColor(x, y).alpha() > 200:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


print("RESULT", repr({"box0": extent(composite(0.0)),
                      "box90": extent(composite(90.0))}))
'''


def test_saved_composite_paints_the_paste_rotated():
    """保存用的合成图必须与画布一致：0° 横向、90° 竖向、绕中心转。"""
    result = _run_qt(_SAVE_PAINT_SCRIPT)

    x0, y0, x1, y1 = result["box0"]
    assert (x1 - x0 + 1) > (y1 - y0 + 1), result["box0"]
    assert abs((x1 - x0 + 1) - 40) <= 2, result["box0"]
    assert abs((y1 - y0 + 1) - 20) <= 2, result["box0"]

    x0, y0, x1, y1 = result["box90"]
    assert (y1 - y0 + 1) > (x1 - x0 + 1), result["box90"]
    assert abs((x1 - x0 + 1) - 20) <= 2, result["box90"]
    assert abs((y1 - y0 + 1) - 40) <= 2, result["box90"]
    assert abs((x0 + x1) / 2 - 99.5) <= 2, result["box90"]
    assert abs((y0 + y1) / 2 - 99.5) <= 2, result["box90"]


# ---------- 真像素：保存合成图带上羽化 / 调色 / 投影 / 泊松 ----------
#
# 画布预览与保存成品必须同一套像素效果（spec A1/A2/A4/A5）。这里在真 Qt 子
# 进程里按各参数画一遍再回读像素，父进程按 RESULT 里的数字断言。

_SAVE_FUSION_SCRIPT = '''
import sys
sys.path.insert(0, r"%s")

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication

app = QApplication([])

from labelforge.core.config import PASTE_FUSION_CONFIG
from labelforge.engine.paste_item import PasteItem
from labelforge.engine.save_manager import SaveManagerMixin

SIZE = 200
SPRITE_W, SPRITE_H = 40, 20
X0, Y0 = 80, 90
ITEM_RECT = QRectF(X0, Y0, SPRITE_W, SPRITE_H)
BG_COLOR = QColor(20, 60, 220)


def sprite(w=SPRITE_W, h=SPRITE_H, color=QColor(255, 0, 0, 255)):
    image = QImage(w, h, QImage.Format_RGBA8888)
    image.fill(color)
    return QPixmap.fromImage(image)


def base_image(fill_bg=False):
    canvas = QImage(SIZE, SIZE, QImage.Format_ARGB32)
    canvas.fill(Qt.transparent)
    if fill_bg:
        painter = QPainter(canvas)
        painter.fillRect(0, 0, SIZE, SIZE, BG_COLOR)
        painter.end()
    return canvas


def draw(items, base=None):
    canvas = base if base is not None else base_image()
    painter = QPainter(canvas)
    try:
        SaveManagerMixin._draw_paste_items_on(painter, items)
    finally:
        painter.end()
    return canvas


def compose(items, base=None):
    canvas = base if base is not None else base_image()
    SaveManagerMixin._compose_pastes(canvas, items)
    return canvas


def rgba(canvas):
    image = canvas.convertToFormat(QImage.Format_RGBA8888)
    return np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )[:, :image.width() * 4].reshape(
        image.height(), image.width(), 4).copy()


def extent(arr):
    ys, xs = np.where(arr[:, :, 3] > 200)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


result = {}

# 1. 默认：与「直接把 pixmap 画到同一位置」逐像素一致
plain_item = PasteItem(sprite(), ITEM_RECT, "cat")
plain = base_image()
p = QPainter(plain)
p.drawPixmap(ITEM_RECT.toRect(), plain_item.pixmap)
p.end()
default = rgba(draw([plain_item]))
result["default_identical"] = bool(np.array_equal(rgba(plain), default))
result["default_extent"] = extent(default)

# 2. 旋转：绕贴图中心画，外接框跟着转
rotated = rgba(draw([PasteItem(sprite(), ITEM_RECT, "cat", angle=90.0)]))
result["rotated_extent"] = extent(rotated)

# 3. 泊松：开关打开时不抛异常且真的改像素；关掉时与普通绘制一致
bg = QPixmap.fromImage(base_image(True))
plain_second = rgba(draw([plain_item], base_image(True)))

PASTE_FUSION_CONFIG["seamless"] = False
off = rgba(compose([plain_item], base_image(True)))

PASTE_FUSION_CONFIG["seamless"] = True
try:
    on = rgba(compose([plain_item], base_image(True)))
    raised = False
except Exception:
    on = None
    raised = True
finally:
    PASTE_FUSION_CONFIG["seamless"] = False

result["seamless_raised"] = raised
result["off_equals_plain"] = bool(np.array_equal(off, plain_second))
result["seamless_nonempty"] = (None if on is None
                               else int(np.count_nonzero(on[:, :, 3] > 0)))
result["seamless_changed"] = (None if on is None else int(np.count_nonzero(
    (on[:, :, :3] != off[:, :, :3]).any(axis=2))))

print("RESULT", repr(result))
'''


def test_saved_composite_matches_the_canvas_draw_path():
    result = _run_qt(_SAVE_FUSION_SCRIPT)

    X0, Y0 = 80, 90

    # 默认仍与旧落笔逐像素一致
    assert result["default_identical"] is True, result["default_identical"]
    assert list(result["default_extent"]) == [X0, Y0, X0 + 39, Y0 + 19],         result["default_extent"]

    # 90° 旋转：不透明区变成竖的，且仍以贴图中心为中心
    x0, y0, x1, y1 = result["rotated_extent"]
    assert (y1 - y0 + 1) > (x1 - x0 + 1), result["rotated_extent"]
    # 贴图中心 (100, 100)：旋转后外接框仍以它为中心
    assert abs((x0 + x1) / 2 - 100) <= 2, result["rotated_extent"]
    assert abs((y0 + y1) / 2 - 100) <= 2, result["rotated_extent"]

    # 泊松：开关打开时不抛异常且真的改像素；关掉时与普通路径一致
    assert result["seamless_raised"] is False, result
    assert result["off_equals_plain"] is True, result
    assert result["seamless_nonempty"] and result["seamless_nonempty"] > 0, result
    assert result["seamless_changed"] > 0, result


def test_rotated_paste_label_shows_the_angle():
    """旋转后的贴图标签带「NN°」，重置回 0° 后只剩名字。"""
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin

    captured = []

    class Check:
        def isChecked(self):
            return True

    class Editor:
        show_paste_names_checkbox = Check()

        def get_label_color(self, label):
            return "#E53935"

    class Renderer(CanvasRendererMixin):
        def __init__(self):
            self._editor = Editor()

        def _draw_label_above_rect(self, painter, x, y, text, color):
            captured.append(text)

    renderer = Renderer()
    renderer._draw_paste_label(None, 0, 0, "cat", False, 0, 45.0)
    renderer._draw_paste_label(None, 0, 0, "cat", False, 0, 0.0)

    assert captured == ["cat 45°", "cat"], captured
