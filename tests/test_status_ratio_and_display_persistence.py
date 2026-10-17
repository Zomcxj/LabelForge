"""状态栏宽高占比 + 画布显示（亮度/对比度）常驻的回归测试。

conftest 把 PySide6 全部 mock 掉了，因此这里不实例化真实 Qt：
- 纯函数走真实调用；
- 画布行为用 stub 编辑器 + mock QPixmap 驱动真实方法体。
"""
from pathlib import Path

import numpy as np
import pytest

from labelforge.canvas import canvas_renderer as renderer_module
from labelforge.canvas.canvas import Canvas
from labelforge.core.utils import format_size_ratio


ROOT = Path(__file__).resolve().parents[1]
CANVAS_SRC = ROOT / "labelforge" / "canvas" / "canvas.py"
RENDERER_SRC = ROOT / "labelforge" / "canvas" / "canvas_renderer.py"


# --------------------------------------------------------------------------
# 1. 宽高占比：纯函数
# --------------------------------------------------------------------------
def test_format_size_ratio_reports_percentage_of_image():
    assert format_size_ratio(120, 80, 500, 800) == "W:120 H:80 (24.0% × 10.0%)"


def test_format_size_ratio_rounds_to_one_decimal():
    assert format_size_ratio(1, 1, 3, 3) == "W:1 H:1 (33.3% × 33.3%)"


@pytest.mark.parametrize("img_w,img_h", [(0, 0), (None, None), (0, 800), (500, 0)])
def test_format_size_ratio_degrades_when_image_size_unknown(img_w, img_h):
    """图片尺寸缺失或非正时只显示像素值，不能除零。"""
    assert format_size_ratio(120, 80, img_w, img_h) == "W:120 H:80"


def test_format_size_ratio_truncates_fractional_pixels():
    assert format_size_ratio(120.7, 80.2, 500, 800) == "W:120 H:80 (24.1% × 10.0%)"


# --------------------------------------------------------------------------
# 2. 宽高占比：状态栏接入
# --------------------------------------------------------------------------
class _Label:
    def __init__(self):
        self.text = None

    def setText(self, value):
        self.text = value


class _Point:
    def x(self):
        return 10

    def y(self):
        return 20


class _Rect:
    def left(self):
        return 0

    def top(self):
        return 0


class _StatusEditor:
    _max_labels = 3
    _is_delete_view = False

    def __init__(self):
        self.current_background = object()
        self.status_label = _Label()
        self.detection_boxes = [
            {"label": "cat", "x": 0, "y": 0, "width": 120, "height": 80},
        ]

    def get_image_info(self):
        return {"width": 500, "height": 800,
                "box_count": 1, "paste_count": 0}

    def get_label_stats(self):
        return {"cat": 1}


class _StatusStub:
    """只提供 update_status_label 需要读取的属性。"""

    def __init__(self, **overrides):
        self._editor = _StatusEditor()
        self.mouse_inside = True
        self.mouse_pos = _Point()
        self.background_scale = 1.0
        self.selected_item_size = None
        self.selected_box = 0
        self.is_drawing_box = False
        self.temp_draw_box = None
        for key, value in overrides.items():
            setattr(self, key, value)

    def get_background_rect(self):
        return _Rect()


def test_status_label_shows_selected_box_ratio_to_image():
    stub = _StatusStub()
    Canvas.update_status_label(stub)
    assert stub._editor.status_label.text == (
        "Box:1 Paste:0 | X:10 Y:20 | W:120 H:80 (24.0% × 10.0%) | cat:1")


def test_status_label_ratio_uses_image_size_not_canvas_size():
    """占比基准是图片原始宽高，跟缩放比例无关。"""
    stub = _StatusStub(background_scale=0.25)
    Canvas.update_status_label(stub)
    assert "W:120 H:80 (24.0% × 10.0%)" in stub._editor.status_label.text


def test_status_label_falls_back_without_image_size():
    stub = _StatusStub()
    stub._editor.get_image_info = lambda: None
    Canvas.update_status_label(stub)
    assert "W:120 H:80" in stub._editor.status_label.text
    assert "%" not in stub._editor.status_label.text


def test_status_label_uses_ratio_for_paste_item_selection():
    stub = _StatusStub(selected_box=None, selected_item_size=(50, 100))
    Canvas.update_status_label(stub)
    assert "W:50 H:100 (10.0% × 12.5%)" in stub._editor.status_label.text


# --------------------------------------------------------------------------
# 3. 画布显示常驻：源图跟踪
# --------------------------------------------------------------------------
class _ByteBuffer(bytearray):
    """np.frombuffer 需要缓冲协议；Qt 的 bits() 还会被调 setsize()。"""

    def setsize(self, size):
        pass


class _FakeImage:
    def __init__(self, width=4, height=4):
        self._w, self._h = width, height
        self._buf = _ByteBuffer(width * height * 4)

    def convertToFormat(self, *_args):
        return self

    def width(self):
        return self._w

    def height(self):
        return self._h

    def bits(self):
        return self._buf


class _FakePixmap:
    def __init__(self, name, width=4, height=4):
        self.name = name
        self._w, self._h = width, height

    def width(self):
        return self._w

    def height(self):
        return self._h

    def toImage(self):
        return _FakeImage(self._w, self._h)

    def __repr__(self):
        return f"<pixmap {self.name}>"


class _DisplayEditor:
    def __init__(self, pixmap):
        self.current_background = pixmap


def _make_canvas(pixmap):
    return Canvas(_DisplayEditor(pixmap))


@pytest.fixture
def fake_from_image(monkeypatch):
    """让 QPixmap.fromImage 返回可辨识的假图，避免依赖真实 Qt。"""
    made = []

    def _from_image(_img):
        pixmap = _FakePixmap(f"adjusted-{len(made)}")
        made.append(pixmap)
        return pixmap

    monkeypatch.setattr(renderer_module.QPixmap, "fromImage",
                        staticmethod(_from_image), raising=False)
    return made


def test_adjust_source_follows_image_switch():
    """切图后源图必须换成新图，否则滑块会把画布拉回上一张。"""
    first, second = _FakePixmap("first"), _FakePixmap("second")
    canvas = _make_canvas(first)

    assert canvas._resolve_adjust_source() is first

    canvas._adjusted_background = first          # 假装 first 已产出调整结果
    canvas._editor.current_background = second   # 模拟 image_loader 直接换图

    assert canvas._resolve_adjust_source() is second


def test_apply_display_adjustments_is_idempotent_across_repaints():
    first = _FakePixmap("first")
    canvas = _make_canvas(first)
    canvas.apply_display_adjustments()

    assert canvas._adjusted_source is first
    assert canvas._adjusted_background is first
    assert canvas._editor.current_background is first

    canvas.apply_display_adjustments()
    canvas.apply_display_adjustments()
    assert canvas._editor.current_background is first


def test_switching_image_reapplies_current_brightness_contrast(fake_from_image):
    """切图后新图要重新套用当前亮度/对比度，而不是保持未调整的原图。"""
    first, second = _FakePixmap("first"), _FakePixmap("second")
    canvas = _make_canvas(first)

    canvas.apply_brightness_contrast(70, 60)
    assert canvas._brightness == 70 and canvas._contrast == 60
    assert canvas._editor.current_background is fake_from_image[0]

    canvas._editor.current_background = second
    canvas.apply_display_adjustments()

    assert canvas._adjusted_source is second
    assert canvas._editor.current_background is fake_from_image[1]
    assert canvas._adjusted_background is fake_from_image[1]


def test_switching_image_keeps_slider_values(fake_from_image):
    canvas = _make_canvas(_FakePixmap("first"))
    canvas.apply_brightness_contrast(70, 60)

    canvas._editor.current_background = _FakePixmap("second")
    canvas.apply_display_adjustments()

    assert (canvas._brightness, canvas._contrast) == (70, 60)


def test_neutral_brightness_contrast_passes_source_through():
    first = _FakePixmap("first")
    canvas = _make_canvas(first)
    canvas.apply_brightness_contrast(50, 50)

    assert canvas._adjusted_background is first
    assert canvas._editor.current_background is first


def test_clearing_background_drops_tracked_source():
    first = _FakePixmap("first")
    canvas = _make_canvas(first)
    canvas.apply_display_adjustments()

    canvas._editor.current_background = None
    canvas.apply_display_adjustments()

    assert canvas._adjusted_source is None
    assert canvas._adjusted_background is None


def test_apply_brightness_contrast_still_vectorized_without_loops():
    """原有约束：像素处理必须向量化，不能退化成逐像素循环。"""
    import ast

    tree = ast.parse(RENDERER_SRC.read_text(encoding="utf-8"))
    source = RENDERER_SRC.read_text(encoding="utf-8")
    for name in ("apply_brightness_contrast", "_render_adjusted_background"):
        method = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        assert not any(isinstance(node, (ast.For, ast.While))
                       for node in ast.walk(method)), name
    assert "np.frombuffer" in source
    assert "np.clip" in source


# --------------------------------------------------------------------------
# 4. 画布显示常驻：接入点
# --------------------------------------------------------------------------
def test_paint_event_applies_display_adjustments():
    """所有切图路径最后都会重绘，收敛在 paintEvent 里单点生效。"""
    source = RENDERER_SRC.read_text(encoding="utf-8")
    paint = source.split("def paintEvent", 1)[1].split("def _draw_background", 1)[0]
    assert "self.apply_display_adjustments()" in paint


def test_display_adjustments_are_available_to_the_renderer_mixin():
    """paintEvent 属于 renderer mixin，调整方法必须同处一个 mixin。

    放在 Canvas 上会让只继承 renderer 的调用方在 paintEvent 里抛
    AttributeError —— Qt 会把绘制期异常变成进程崩溃（0xC0000409）。
    """
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin

    assert callable(getattr(CanvasRendererMixin, "apply_display_adjustments", None))


def test_apply_display_adjustments_swallows_render_errors(monkeypatch):
    """调整失败只能退化为不调整，绝不能让绘制路径抛异常。"""
    logged = []
    import labelforge.core.exception_hook as hook
    monkeypatch.setattr(hook, "_write_log", lambda msg: logged.append(msg))

    canvas = _make_canvas(_FakePixmap("first"))

    def _boom(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(canvas, "_render_adjusted_background", _boom)
    canvas._brightness, canvas._contrast = 70, 60

    canvas.apply_display_adjustments()  # 不得抛出

    assert canvas._adjusted_background is canvas._editor.current_background


def test_apply_display_adjustments_skips_work_when_nothing_changed():
    """鼠标移动会触发重绘，参数没变时不能反复跑像素处理。"""
    canvas = _make_canvas(_FakePixmap("first"))
    calls = []

    original = canvas._render_adjusted_background

    def _counting(*args):
        calls.append(args)
        return original(*args)

    canvas._render_adjusted_background = _counting
    canvas.apply_display_adjustments()
    assert len(calls) == 1

    canvas.apply_display_adjustments()
    canvas.apply_display_adjustments()
    assert len(calls) == 1

    canvas._brightness = 70
    canvas.apply_display_adjustments()
    assert len(calls) == 2


def test_canvas_drops_stale_original_background_field():
    """旧的 _original_background 切图后不失效，是本次修复的根因。"""
    for path in (CANVAS_SRC, RENDERER_SRC):
        assert "_original_background" not in path.read_text(encoding="utf-8")
