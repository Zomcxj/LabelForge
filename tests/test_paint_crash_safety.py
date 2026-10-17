"""paintEvent 崩溃兜底回归测试。

背景：`paintEvent` 内未捕获的异常会让 Qt 直接终止进程（Windows 0xC0000409 /
Linux SIGSEGV），没有 traceback，日志里什么都看不到。更隐蔽的是，光包
try/except 并不够——异常时若 QPainter 没 end()，traceback 会持有该帧，
painter 活到 except 块结束，Qt 在 render() 上下文里析构它就段错误。

这些用例必须在真实 Qt 里跑（conftest 的 mock 无法复现崩溃），
所以走 offscreen 子进程；返回码非 0 即视为崩溃。
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RENDERER = ROOT / "labelforge" / "canvas" / "canvas_renderer.py"

_HARNESS = '''
from PySide6.QtCore import Qt, QRectF, QPoint
from PySide6.QtGui import QPixmap, QColor, QImage
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication([])
from labelforge.canvas.canvas_renderer import CanvasRendererMixin


class Check:
    def __init__(self, c=True): self.c = c
    def isChecked(self): return self.c


class Editor:
    def __init__(self):
        self.current_background = QPixmap(100, 100)
        self.current_background.fill(Qt.white)
        self.show_grid_checkbox = Check(False)
        self.show_labels_checkbox = Check(True)
        self.show_label_names_checkbox = Check(False)
        self.show_paste_names_checkbox = Check(False)
        self.detection_boxes = [{"x": 10, "y": 10, "width": 30, "height": 30, "label": "cat"}]
        self.canvas_items = []
        self.selected_item = None
        self.edit_mode = "annotate"
        self.pressed_label = None
    def get_label_color(self, label): return "#E53935"


class Canvas(CanvasRendererMixin, QWidget):
    def __init__(self):
        super().__init__()
        self._editor = Editor()
        self.background_scale = 1
        self.shape_opacity = 1.0
        self.selected_box = 0
        self.selected_boxes = [0]
        self.selected_item_size = None
        self.is_drawing_box = False
        self.temp_draw_box = None
        self.draw_start_pos = None
        self.mouse_inside = True
        self.mouse_pos = QPoint(30, 30)
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self._brightness = 50
        self._contrast = 50
        self._adjusted_source = None
        self._adjusted_background = None
        self._adjusted_brightness = None
        self._adjusted_contrast = None
    def get_background_rect(self): return QRectF(0, 0, 100, 100)


def boom(*a, **k): raise RuntimeError("injected")


def render_once(canvas):
    img = QImage(100, 100, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    canvas.render(img)
    return img
'''


def _run(script):
    # Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死。断言通过、标记打印后
    # 立即 flush + os._exit(0) 绕开析构；os._exit 不冲刷缓冲，必须先 flush。
    script = script + '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, (
        f"子进程退出码 {result.returncode}（负值/大值表示 Qt 硬崩）\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return result.stdout


def test_paint_failure_in_every_draw_step_does_not_kill_the_process():
    """7 个绘制阶段各自抛异常，进程都必须存活。"""
    out = _run(_HARNESS + '''
cases = {
    "apply_display_adjustments": lambda c: setattr(c, "apply_display_adjustments", boom),
    "get_background_rect": lambda c: setattr(c, "get_background_rect", boom),
    "draw_background": lambda c: setattr(c, "_draw_background", boom),
    "draw_grid": lambda c: setattr(c, "_draw_grid", boom),
    "draw_detection_boxes": lambda c: setattr(c, "_draw_detection_boxes", boom),
    "draw_magnifier": lambda c: setattr(c, "_draw_magnifier", boom),
    "draw_crosshair": lambda c: setattr(c, "_draw_crosshair", boom),
}
for name, inject in cases.items():
    canvas = Canvas()
    canvas.resize(100, 100)
    inject(canvas)
    render_once(canvas)
    print("survived", name)
print("ALL OK")
''')
    for step in ("apply_display_adjustments", "get_background_rect", "draw_background",
                 "draw_grid", "draw_detection_boxes", "draw_magnifier", "draw_crosshair"):
        assert f"survived {step}" in out, step
    assert "ALL OK" in out


def test_paint_failure_is_logged_instead_of_silently_swallowed():
    """失败必须落到 labelforge.log，否则崩溃现场无从追查。"""
    out = _run(_HARNESS + '''
import labelforge.core.exception_hook as hook
logged = []
hook._write_log = lambda msg: logged.append(msg)

canvas = Canvas()
canvas.resize(100, 100)
canvas._draw_background = boom
render_once(canvas)

assert logged, "未记录任何日志"
assert "画布绘制失败" in logged[0], logged
assert "RuntimeError" in logged[0], logged
print("ALL OK")
''')
    assert "ALL OK" in out


def test_healthy_paint_still_produces_pixels():
    """兜底不能把正常绘制也吞掉。"""
    out = _run(_HARNESS + '''
canvas = Canvas()
canvas.resize(100, 100)
img = render_once(canvas)
assert img.pixelColor(50, 50).alpha() > 0, "正常渲染应有内容"
print("ALL OK")
''')
    assert "ALL OK" in out


def test_painters_are_ended_even_when_drawing_raises():
    """异常路径下 QPainter 必须被 end()，否则 Qt 会在 render() 里段错误。"""
    source = RENDERER.read_text(encoding="utf-8")
    scene_body = source.split("def _paint_scene", 1)[1].split("def _report_paint_failure", 1)[0]

    assert scene_body.count("finally:") == 2, "两个 QPainter 都必须有 finally 收尾"
    assert "if sp.isActive():" in scene_body
    assert "sp.end()" in scene_body
    assert "if painter.isActive():" in scene_body
    assert "painter.end()" in scene_body


def test_failure_handler_does_not_open_a_new_painter():
    """兜底里不能再开 painter：可能处在 render() 上下文中，会段错误。"""
    source = RENDERER.read_text(encoding="utf-8")
    handler = source.split("def _report_paint_failure", 1)[1].split("\n    def ", 1)[0]

    assert "QPainter(" not in handler, "兜底不得新建 QPainter"
    assert "_write_log" in handler


def test_paint_event_guards_the_whole_scene():
    source = RENDERER.read_text(encoding="utf-8")
    paint = source.split("def paintEvent", 1)[1].split("def _paint_scene", 1)[0]

    assert "try:" in paint
    assert "self._paint_scene(event)" in paint
    assert "except Exception as exc:" in paint
    assert "self._report_paint_failure(exc)" in paint
