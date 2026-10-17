"""语义区域（semantic region）功能测试。

覆盖区域绘制、拖动/缩放、固定锁定、删除与命中，以及贴图引擎的
区域中心采样（_sample_center_in_regions / random_paste_images）。

conftest 已 mock 整个 PySide6，故此处通过假对象（duck typing）直接
调用真实 mixin 逻辑，不实例化真实 Qt 控件。
"""
import random

import pytest

from labelforge.canvas.canvas_interaction import CanvasInteractionMixin
from labelforge.canvas.canvas_renderer import CanvasRendererMixin
from labelforge.canvas.canvas_menu import CanvasMenuMixin
from labelforge.engine.shape_io import point_in_polygon
from labelforge.engine.paste_engine import PasteEngineMixin
from labelforge.engine import paste_engine as paste_engine_module
from labelforge.core.config import REGION_CONFIG


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def x(self):
        return self._x

    def y(self):
        return self._y

    def setX(self, x):
        self._x = x

    def setY(self, y):
        self._y = y

    def __sub__(self, other):
        return Point(self._x - other.x(), self._y - other.y())


class Rect:
    def __init__(self, x=0, y=0, w=100, h=100):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def left(self):
        return self._x

    def top(self):
        return self._y

    def right(self):
        return self._x + self._w

    def bottom(self):
        return self._y + self._h

    def width(self):
        return self._w

    def height(self):
        return self._h

    def contains(self, point):
        return (self._x <= point.x() <= self._x + self._w and
                self._y <= point.y() <= self._y + self._h)


class Background:
    def __init__(self, w=400, h=300):
        self._w = w
        self._h = h

    def width(self):
        return self._w

    def height(self):
        return self._h


class Editor:
    def __init__(self):
        self.edit_mode = 'paste'
        self._is_delete_view = False
        self.background_images = ["a.png"]
        self.current_background = Background(400, 300)
        self.current_background_index = 0
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        # 第 10 轮把区域拆成两份：标注模式看 `region_boxes`（推理范围），
        # 贴图模式看 `paste_region_boxes`（贴图落点）。假 Editor 用一个
        # `region_boxes` 属性把两者统一起来（见下），测试里 `e.region_boxes`
        # 始终表示「当前模式下那份区域」，不必每处都判模式。
        self._inference_regions = []
        self._paste_regions = []
        self._inference_fixed = False
        self._paste_fixed = False
        self._scope_panel_open = False
        self.selected_item = None

    @property
    def region_fixed(self):
        """当前模式那份区域的固定开关（测试统一入口，与 `region_boxes` 配套）。"""
        return (self._paste_fixed if self.edit_mode == 'paste'
                else self._inference_fixed)

    @region_fixed.setter
    def region_fixed(self, value):
        if self.edit_mode == 'paste':
            self._paste_fixed = bool(value)
        else:
            self._inference_fixed = bool(value)

    @property
    def paste_region_fixed(self):
        return self._paste_fixed

    @paste_region_fixed.setter
    def paste_region_fixed(self, value):
        self._paste_fixed = bool(value)

    @property
    def region_boxes(self):
        """当前模式那份区域（测试统一入口）。"""
        return (self._paste_regions if self.edit_mode == 'paste'
                else self._inference_regions)

    @region_boxes.setter
    def region_boxes(self, value):
        if self.edit_mode == 'paste':
            self._paste_regions = list(value or [])
        else:
            self._inference_regions = list(value or [])

    @property
    def paste_region_boxes(self):
        return self._paste_regions

    @paste_region_boxes.setter
    def paste_region_boxes(self, value):
        self._paste_regions = list(value or [])

    def save_undo_state(self):
        pass


class Canvas(CanvasInteractionMixin):
    def __init__(self, regions=None, fixed=False, mode='paste'):
        self._editor = Editor()
        self._editor.edit_mode = mode
        # `regions=` 落到**当前模式**那份上：贴图模式给 `paste_region_boxes`，
        # 标注模式给 `region_boxes`。测试因此不必关心"两份"这件事，它表达的
        # 始终是「我这个模式下有哪些区域」。
        if mode == 'paste':
            self._editor.paste_region_boxes = list(regions) if regions else []
            self._editor.paste_region_fixed = fixed
        else:
            self._editor.region_boxes = list(regions) if regions else []
            self._editor.region_fixed = fixed
        # 面板开着才允许编辑区域。第 11 轮去掉贴图区域面板后，贴图模式
        # 恒允许编辑（右键就是它的入口），标注模式仍要求范围面板开着。
        self._editor._scope_panel_open = True
        self.background_scale = 1
        self.mouse_pos = None
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.is_drawing_region = False
        self.is_drawing_region_polygon = False
        self.temp_region_points = []
        self.selected_region = None
        self.selected_region_vertex = None
        self.is_dragging_region = False
        self.is_resizing_region = False
        self.is_dragging_region_vertex = False
        self.region_vertex_drag_index = None
        self.region_resize_handle = None
        self.region_drag_start = None
        self.region_resize_start = None
        self.selected_box = None
        self.selected_boxes = []
        self.updated = 0
        self.set_cursor_count = 0

    def update(self):
        self.updated += 1

    def setCursor(self, _cursor):
        self.set_cursor_count += 1

    def update_status_label(self):
        pass

    def get_background_rect(self):
        return Rect(0, 0, 400, 300)

    def find_item_at_position(self, _pos):
        return None

    def _can_edit_canvas(self):
        return True

    def _sync_detection_box_to_dict(self, _box_index):
        pass


class WheelEvent:
    def __init__(self, delta, modifiers=0):
        self._delta = delta
        self._modifiers = modifiers

    def angleDelta(self):
        delta = self._delta

        class _D:
            def y(self_inner):
                return delta
        return _D()

    def modifiers(self):
        return self._modifiers


class Spin:
    def __init__(self, value=30):
        self._value = value

    def value(self):
        return self._value


class Pixmap:
    def __init__(self, w=50, h=50):
        self._w = w
        self._h = h

    def width(self):
        return self._w

    def height(self):
        return self._h


class FakeQRectF:
    def __init__(self, x=0, y=0, w=0, h=0):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h


class CanvasUpdate:
    def __init__(self):
        self.calls = 0

    def update(self):
        self.calls += 1


class Engine(PasteEngineMixin):
    def __init__(self, regions=None, seed=None):
        self._is_delete_view = False
        self.small_images = [("a.png", Pixmap(50, 50)), ("b.png", Pixmap(40, 60))]
        self.current_background = Background(640, 480)
        self.current_background_index = 0
        self.canvas_items = []
        self.canvas_items_dict = {}
        self.detection_boxes = []
        self.paste_count_spin = Spin(5)
        self.min_size_spin = Spin(30)
        self.max_size_spin = Spin(60)
        # 第 10 轮起贴图落点看的是**贴图那份**区域（`paste_region_boxes`），
        # 不再是推理那份。测试里的 `regions=` 表达的就是「贴图区域」。
        self.paste_region_boxes = list(regions) if regions else []
        self.region_boxes = []
        self.canvas = CanvasUpdate()
        self._paste_labels = []

    def save_undo_state(self):
        pass

    def _validate_size_range(self):
        pass

    def _get_paste_label(self, index):
        return "paste"


def _sample(regions, new_w=40, new_h=40, bg_w=400, bg_h=300, rng=None):
    rng = rng or random.Random(0)
    return PasteEngineMixin._sample_center_in_regions(None, regions, new_w, new_h, bg_w, bg_h, rng)


def test_config_constants():
    from labelforge.core.config import DETECTION_BOX_CONFIG

    assert REGION_CONFIG['min_size'] == 10
    # 第 12 轮：区域句柄 = 检测框 + 1（用户要求），所以跟着检测框走，
    # 不再写死数字——否则改检测框时这里会骗人。
    assert REGION_CONFIG['handle_size'] == DETECTION_BOX_CONFIG['resize_handle_size'] + 1


def test_sample_center_single_region_500_all_inside():
    regions = [{'x': 50.0, 'y': 60.0, 'width': 120.0, 'height': 100.0}]
    rng = random.Random(42)
    for _ in range(500):
        sample = _sample(regions, rng=rng)
        assert sample is not None
        cx, cy = sample
        # 贴图 40x40 必须完整落在区域内（含四边）
        assert 50.0 + 20 <= cx <= 170.0 - 20
        assert 60.0 + 20 <= cy <= 160.0 - 20


def test_sample_center_multiple_regions_both_hit():
    regions = [
        {'x': 0.0, 'y': 0.0, 'width': 80.0, 'height': 80.0},
        {'x': 300.0, 'y': 200.0, 'width': 80.0, 'height': 80.0},
    ]
    rng = random.Random(7)
    in_a = in_b = 0
    for _ in range(500):
        sample = _sample(regions, new_w=20, new_h=20, rng=rng)
        assert sample is not None
        cx, cy = sample
        if 0.0 <= cx <= 80.0 and 0.0 <= cy <= 80.0:
            in_a += 1
        elif 300.0 <= cx <= 380.0 and 200.0 <= cy <= 280.0:
            in_b += 1
    assert in_a > 0
    assert in_b > 0


def test_sample_center_region_too_small_returns_none():
    regions = [{'x': 0.0, 'y': 0.0, 'width': 20.0, 'height': 20.0}]
    for seed in range(10):
        rng = random.Random(seed)
        assert _sample(regions, new_w=60, new_h=60, rng=rng) is None


def test_sample_center_edge_region_clamped_and_inside():
    regions = [{'x': 0.0, 'y': 0.0, 'width': 300.0, 'height': 300.0}]
    rng = random.Random(1)
    seen = set()
    for _ in range(500):
        sample = _sample(regions, new_w=60, new_h=60, bg_w=640, bg_h=480, rng=rng)
        assert sample is not None
        cx, cy = sample
        # 贴图完整在区域内：中心 ∈ [30, 270]
        assert 30.0 <= cx <= 270.0
        assert 30.0 <= cy <= 270.0
        seen.add((round(cx, 1), round(cy, 1)))
    assert len(seen) > 50


def test_random_paste_images_fully_inside_regions(monkeypatch):
    """贴图四边都不能超出区域。"""
    monkeypatch.setattr(paste_engine_module, "QRectF", FakeQRectF)
    region = {'x': 50.0, 'y': 50.0, 'width': 500.0, 'height': 380.0}
    engine = Engine(regions=[region], seed=99)
    engine.random_paste_images(seed=99)

    assert len(engine.canvas_items) > 0
    for _pixmap, rect, label in engine.canvas_items:
        assert region['x'] <= rect.x()
        assert region['y'] <= rect.y()
        assert rect.x() + rect.width() <= region['x'] + region['width']
        assert rect.y() + rect.height() <= region['y'] + region['height']
        assert 0 <= rect.x() <= 640 - rect.width()
        assert 0 <= rect.y() <= 480 - rect.height()


def test_random_paste_images_no_regions_deterministic(monkeypatch):
    monkeypatch.setattr(paste_engine_module, "QRectF", FakeQRectF)
    engine_a = Engine(seed=None)
    engine_a.random_paste_images(seed=1234)
    engine_b = Engine(seed=None)
    engine_b.random_paste_images(seed=1234)

    assert len(engine_a.canvas_items) > 0
    assert len(engine_a.canvas_items) == len(engine_b.canvas_items)
    for (ra, rb) in zip(engine_a.canvas_items, engine_b.canvas_items):
        rect_a, rect_b = ra[1], rb[1]
        assert rect_a.x() == rect_b.x()
        assert rect_a.y() == rect_b.y()
        assert rect_a.width() == rect_b.width()
        assert rect_a.height() == rect_b.height()


def test_region_drawing_drag_creates_region():
    """按下-拖-松开：press 定起点，release 完成。"""
    canvas = Canvas()
    canvas.is_drawing_region = True
    canvas._handle_region_press(Point(50, 50))
    assert canvas.draw_start_pos is not None
    assert canvas._editor.region_boxes == []
    canvas._complete_region_drawing(Point(150, 150))

    assert len(canvas._editor.region_boxes) == 1
    region = canvas._editor.region_boxes[0]
    assert region['x'] == 50.0
    assert region['y'] == 50.0
    assert region['width'] == 100.0
    assert region['height'] == 100.0
    assert canvas._editor.detection_boxes == []
    assert canvas.is_drawing_region is False
    assert canvas.draw_start_pos is None


def test_region_drawing_press_does_not_complete():
    """第二次 press 不应完成绘制（改为拖动语义）。"""
    canvas = Canvas()
    canvas.is_drawing_region = True
    canvas._handle_region_press(Point(50, 50))
    canvas._handle_region_press(Point(150, 150))
    assert canvas._editor.region_boxes == []
    assert canvas.draw_start_pos is not None


@pytest.mark.parametrize("p1,p2", [
    ((10, 10), (20, 20)),
    ((10, 10), (200, 15)),
    ((200, 10), (10, 10)),
])
def test_region_drawing_rejects_too_small(p1, p2):
    canvas = Canvas()
    canvas.is_drawing_region = True
    canvas._handle_region_press(Point(*p1))
    canvas._complete_region_drawing(Point(*p2))
    assert canvas._editor.region_boxes == []
    assert canvas.is_drawing_region is False


def test_region_drawing_rejected_when_fixed():
    canvas = Canvas(fixed=True)
    canvas.is_drawing_region = True
    assert canvas._handle_region_press(Point(50, 50)) is True
    assert canvas.draw_start_pos is None
    assert canvas._editor.region_boxes == []


def test_drag_region_moves_and_clamps():
    canvas = Canvas(regions=[{'x': 10.0, 'y': 10.0, 'width': 50.0, 'height': 50.0}])
    canvas.selected_region = 0

    canvas.region_drag_start = Point(0, 0)
    canvas.mouse_pos = Point(30, 40)
    canvas._drag_region()
    region = canvas._editor.region_boxes[0]
    assert region['x'] == 40.0
    assert region['y'] == 50.0

    canvas.region_drag_start = Point(0, 0)
    canvas.mouse_pos = Point(100000, -100000)
    canvas._drag_region()
    region = canvas._editor.region_boxes[0]
    assert region['x'] == 350.0
    assert region['y'] == 0.0
    assert region['width'] == 50.0
    assert region['height'] == 50.0


def test_resize_region_br_handle_and_min_size():
    canvas = Canvas(regions=[{'x': 10.0, 'y': 10.0, 'width': 50.0, 'height': 50.0}])
    canvas.selected_region = 0
    canvas.region_resize_handle = "br"

    canvas.region_resize_start = Point(0, 0)
    canvas.mouse_pos = Point(20, 30)
    canvas._resize_region()
    region = canvas._editor.region_boxes[0]
    assert region['x'] == 10.0
    assert region['y'] == 10.0
    assert region['width'] == 70.0
    assert region['height'] == 80.0

    canvas.region_resize_start = Point(0, 0)
    canvas.mouse_pos = Point(-100000, -100000)
    canvas._resize_region()
    region = canvas._editor.region_boxes[0]
    assert region['width'] == REGION_CONFIG['min_size']
    assert region['height'] == REGION_CONFIG['min_size']


def test_fixed_blocks_drag_resize_and_click():
    canvas = Canvas(regions=[{'x': 10.0, 'y': 10.0, 'width': 100.0, 'height': 100.0}], fixed=True)
    canvas.selected_region = 0
    canvas.region_resize_handle = "br"
    canvas.region_drag_start = Point(0, 0)
    canvas.region_resize_start = Point(0, 0)
    canvas.mouse_pos = Point(100, 100)

    canvas._drag_region()
    canvas._resize_region()
    assert canvas._editor.region_boxes[0] == {'x': 10.0, 'y': 10.0, 'width': 100.0, 'height': 100.0}

    canvas.selected_region = None
    assert canvas._handle_region_click(Point(50, 50)) is False
    assert canvas.selected_region is None
    assert canvas.is_dragging_region is False
    assert canvas.is_resizing_region is False


def test_delete_region_removes_single():
    canvas = Canvas(regions=[
        {'x': 0.0, 'y': 0.0, 'width': 50.0, 'height': 50.0},
        {'x': 100.0, 'y': 100.0, 'width': 60.0, 'height': 60.0},
    ])
    canvas._delete_region(0)
    assert len(canvas._editor.region_boxes) == 1
    assert canvas._editor.region_boxes[0]['x'] == 100.0
    assert canvas.selected_region is None


def test_delete_region_rejected_when_fixed():
    canvas = Canvas(regions=[{'x': 0.0, 'y': 0.0, 'width': 50.0, 'height': 50.0}], fixed=True)
    canvas._delete_region(0)
    assert len(canvas._editor.region_boxes) == 1


def test_region_at_reverse_order_hit():
    canvas = Canvas(regions=[
        {'x': 0.0, 'y': 0.0, 'width': 200.0, 'height': 200.0},
        {'x': 100.0, 'y': 100.0, 'width': 200.0, 'height': 200.0},
    ])
    assert canvas._region_at(Point(150, 150)) == 1
    assert canvas._region_at(Point(50, 50)) == 0
    assert canvas._region_at(Point(250, 250)) == 1
    assert canvas._region_at(Point(350, 350)) is None


def test_region_handle_at_returns_corner():
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}])
    size = REGION_CONFIG['handle_size']
    assert canvas._region_handle_at(Point(100 + size / 2, 100 - size / 2), 0) == 'tl'
    assert canvas._region_handle_at(Point(200 + size / 2, 100 - size / 2), 0) == 'tr'
    assert canvas._region_handle_at(Point(100 + size / 2, 200 + size / 2), 0) == 'bl'
    assert canvas._region_handle_at(Point(200 + size / 2, 200 + size / 2), 0) == 'br'


def test_wheel_scales_hovered_region():
    """悬停在区域上滚轮缩放（绕中心），夹紧最小尺寸。"""
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}])
    canvas.mouse_pos = Point(150, 150)

    canvas.wheelEvent(WheelEvent(120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] > 100.0
    assert region['height'] > 100.0
    assert canvas.selected_region == 0

    before = dict(region)
    canvas.wheelEvent(WheelEvent(-120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] < before['width']
    assert region['height'] < before['height']


def test_wheel_region_not_scaled_when_outside():
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}])
    canvas.mouse_pos = Point(10, 10)
    canvas.wheelEvent(WheelEvent(120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] == 100.0
    assert region['height'] == 100.0


def test_wheel_region_blocked_when_fixed():
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}], fixed=True)
    canvas.mouse_pos = Point(150, 150)
    canvas.wheelEvent(WheelEvent(120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] == 100.0
    assert region['height'] == 100.0


def test_wheel_region_can_fill_background():
    """区域可以放大到铺满背景，不能被 90% 的假上限夹住。"""
    canvas = Canvas(regions=[{'x': 190.0, 'y': 140.0, 'width': 20.0, 'height': 20.0}])
    canvas.mouse_pos = Point(200, 150)

    for _ in range(90):
        canvas.wheelEvent(WheelEvent(120))

    region = canvas._editor.region_boxes[0]
    assert region['width'] == 400.0
    assert region['height'] == 300.0


def test_wheel_scales_selected_box_not_region_in_annotate_mode():
    """回归：标注模式下悬停区域不得吞掉滚轮，选中的检测框仍要缩放。"""
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}],
                    mode='annotate')
    canvas._editor.detection_boxes = [
        {'label': 'cat', 'x': 120.0, 'y': 120.0, 'width': 60.0, 'height': 60.0},
    ]
    canvas.selected_box = 0
    canvas.mouse_pos = Point(150, 150)

    canvas.wheelEvent(WheelEvent(120))

    region = canvas._editor.region_boxes[0]
    assert region['width'] == 100.0
    assert region['height'] == 100.0
    box = canvas._editor.detection_boxes[0]
    assert box['width'] > 60.0
    assert box['height'] > 60.0


def test_wheel_still_scales_region_in_paste_mode():
    """对照：贴图模式下滚轮仍缩放悬停区域。"""
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 100.0, 'height': 100.0}])
    canvas._editor.detection_boxes = [
        {'label': 'cat', 'x': 120.0, 'y': 120.0, 'width': 60.0, 'height': 60.0},
    ]
    canvas.selected_box = 0
    canvas.mouse_pos = Point(150, 150)

    canvas.wheelEvent(WheelEvent(120))

    region = canvas._editor.region_boxes[0]
    assert region['width'] > 100.0
    assert canvas._editor.detection_boxes[0]['width'] == 60.0


def test_wheel_region_min_size_clamped():
    canvas = Canvas(regions=[{'x': 100.0, 'y': 100.0, 'width': 30.0, 'height': 30.0}])
    canvas.mouse_pos = Point(115, 115)
    for _ in range(50):
        canvas.wheelEvent(WheelEvent(-120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] >= REGION_CONFIG['min_size']
    assert region['height'] >= REGION_CONFIG['min_size']


def test_is_paste_mode():
    canvas = Canvas()
    assert canvas._is_paste_mode() is True
    canvas._editor.edit_mode = 'annotate'
    assert canvas._is_paste_mode() is False
    canvas._editor.edit_mode = 'paste'
    canvas._editor._is_delete_view = True
    assert canvas._is_paste_mode() is False


class RegionCanvas(CanvasInteractionMixin, CanvasRendererMixin):
    """同时带交互与渲染 mixin：`_show_regions` 定义在渲染 mixin 上。"""

    def __init__(self, regions=None, fixed=False):
        Canvas.__init__(self, regions=regions, fixed=fixed)
        self._editor.edit_mode = 'paste'
        self._editor._is_delete_view = False


def test_show_regions_follows_paste_mode_by_default():
    """默认只在贴图模式画区域轮廓（不设范围时行为不变）。"""
    canvas = RegionCanvas()
    assert canvas._show_regions() is True
    canvas._editor.edit_mode = 'annotate'
    assert canvas._show_regions() is False


def test_show_regions_in_annotate_mode_when_scope_uses_roi():
    """推理范围选了区域时，标注模式也画区域轮廓。

    否则用户看不到范围，不知道框为什么只出现在某块地方。编辑仍只在
    贴图模式：`_is_paste_mode()` 保持 False，贴图交互不受影响。
    """
    canvas = RegionCanvas()
    canvas._editor.edit_mode = 'annotate'
    canvas._editor._show_scope_regions = True

    assert canvas._show_regions() is True
    assert canvas._is_paste_mode() is False, "编辑权限不能被放宽"


def test_show_regions_stays_off_without_scope_flag():
    """没设范围标志时不显示（默认属性缺失也不能崩）。"""
    canvas = RegionCanvas()
    canvas._editor.edit_mode = 'annotate'

    assert canvas._show_regions() is False
    assert getattr(canvas._editor, '_show_scope_regions', False) is False


def test_show_regions_ignores_delete_view():
    """删除视图下 `_is_paste_mode()` 为假，但范围显示按自己的判断走。"""
    canvas = RegionCanvas()
    canvas._editor.edit_mode = 'annotate'
    canvas._editor._is_delete_view = True
    canvas._editor._show_scope_regions = True

    assert canvas._show_regions() is True
    assert canvas._is_paste_mode() is False

# ==================== 多边形区域 ====================

DIAMOND = {
    'shape_type': 'polygon',
    'points': [[200.0, 50.0], [350.0, 150.0], [200.0, 250.0], [50.0, 150.0]],
    'x': 50.0, 'y': 50.0, 'width': 300.0, 'height': 200.0,
}


def _poly_region(points, **kw):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    region = {
        'shape_type': 'polygon',
        'points': [list(p) for p in points],
        'x': float(min(xs)), 'y': float(min(ys)),
        'width': float(max(xs) - min(xs)), 'height': float(max(ys) - min(ys)),
    }
    region.update(kw)
    return region


def test_polygon_region_drawing_creates_region():
    canvas = Canvas()
    canvas.is_drawing_region_polygon = True
    for pt in [(100, 100), (200, 100), (200, 200), (100, 200)]:
        canvas._handle_region_polygon_press(Point(*pt))
    assert len(canvas.temp_region_points) == 4
    assert canvas._editor.region_boxes == []
    canvas._prompt_finish_region_polygon()
    assert len(canvas._editor.region_boxes) == 1
    region = canvas._editor.region_boxes[0]
    assert region['shape_type'] == 'polygon'
    assert len(region['points']) == 4
    assert region['x'] == 100.0 and region['y'] == 100.0
    assert region['width'] == 100.0 and region['height'] == 100.0
    assert canvas.is_drawing_region_polygon is False
    assert canvas.temp_region_points == []


def test_polygon_region_requires_three_points():
    canvas = Canvas()
    canvas.is_drawing_region_polygon = True
    canvas._handle_region_polygon_press(Point(100, 100))
    canvas._handle_region_polygon_press(Point(200, 100))
    canvas._prompt_finish_region_polygon()
    assert canvas._editor.region_boxes == []
    assert canvas.is_drawing_region_polygon is False


def test_polygon_region_pop_last_point():
    canvas = Canvas()
    canvas.is_drawing_region_polygon = True
    for pt in [(100, 100), (200, 100), (200, 200)]:
        canvas._handle_region_polygon_press(Point(*pt))
    canvas._region_polygon_pop_last_point()
    assert len(canvas.temp_region_points) == 2
    assert canvas.is_drawing_region_polygon is True
    canvas._region_polygon_pop_last_point()
    canvas._region_polygon_pop_last_point()
    assert canvas.temp_region_points == []
    assert canvas.is_drawing_region_polygon is False


def test_polygon_region_rejects_too_small_bbox():
    canvas = Canvas()
    canvas.is_drawing_region_polygon = True
    for pt in [(100, 100), (105, 100), (105, 105), (100, 105)]:
        canvas._handle_region_polygon_press(Point(*pt))
    canvas._prompt_finish_region_polygon()
    assert canvas._editor.region_boxes == []


def test_polygon_region_rejected_when_fixed():
    canvas = Canvas(fixed=True)
    canvas.is_drawing_region_polygon = True
    assert canvas._handle_region_polygon_press(Point(100, 100)) is True
    assert canvas.temp_region_points == []


def test_region_at_polygon_inside_outside():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    assert canvas._region_at(Point(200, 150)) == 0
    assert canvas._region_at(Point(50, 50)) is None


def test_region_vertex_at_hit_and_miss():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    assert canvas._region_vertex_at(Point(200, 50)) == (0, 0)
    assert canvas._region_vertex_at(Point(350, 150)) == (0, 1)
    assert canvas._region_vertex_at(Point(200, 150)) is None


def test_delete_region_vertex():
    points = [[100, 100], [200, 100], [200, 200], [100, 200]]
    canvas = Canvas(regions=[_poly_region(points)])
    assert canvas._delete_region_vertex(0, 0) is True
    region = canvas._editor.region_boxes[0]
    assert len(region['points']) == 3
    assert region['x'] == 100.0 and region['y'] == 100.0
    assert canvas._delete_region_vertex(0, 0) is False


def test_delete_region_vertex_rejected_when_fixed():
    points = [[100, 100], [200, 100], [200, 200], [100, 200]]
    canvas = Canvas(regions=[_poly_region(points)], fixed=True)
    assert canvas._delete_region_vertex(0, 0) is False
    assert len(canvas._editor.region_boxes[0]['points']) == 4


def test_sync_region_bbox():
    region = _poly_region([[100, 100], [200, 100], [200, 200], [100, 200]])
    region['points'] = [[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]]
    CanvasMenuMixin._sync_region_bbox(region)
    assert region['x'] == 10.0
    assert region['y'] == 20.0
    assert region['width'] == 100.0
    assert region['height'] == 60.0


def test_drag_polygon_region_moves_all_vertices():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.selected_region = 0
    before = [list(p) for p in canvas._editor.region_boxes[0]['points']]

    canvas.region_drag_start = Point(200, 150)
    canvas.mouse_pos = Point(220, 170)
    canvas._drag_region()

    region = canvas._editor.region_boxes[0]
    for i, p in enumerate(region['points']):
        assert abs(p[0] - (before[i][0] + 20)) < 1e-6
        assert abs(p[1] - (before[i][1] + 20)) < 1e-6
    assert region['x'] == before[0][0] + 20 or abs(region['x'] - (50 + 20)) < 1e-6


def test_drag_region_vertex_moves_single_point():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.selected_region = 0
    canvas.region_vertex_drag_index = 0
    canvas.is_dragging_region_vertex = True

    canvas.region_drag_start = Point(200, 50)
    canvas.mouse_pos = Point(230, 80)
    canvas._drag_region_vertex()

    region = canvas._editor.region_boxes[0]
    assert region['points'][0] == [230.0, 80.0]
    assert region['x'] == 50.0  # bbox 左边界由其它点决定
    assert region['y'] == 80.0


def test_drag_region_vertex_clamped_to_background():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.selected_region = 0
    canvas.region_vertex_drag_index = 0
    canvas.is_dragging_region_vertex = True

    canvas.region_drag_start = Point(200, 50)
    canvas.mouse_pos = Point(-10000, -10000)
    canvas._drag_region_vertex()
    assert canvas._editor.region_boxes[0]['points'][0] == [0.0, 0.0]


def test_region_vertex_click_then_drag_moves_by_delta_not_absolute():
    """回归：点击顶点进入拖动后，第一次拖动必须按鼠标位移移动，
    不能瞬移到光标绝对坐标（region_drag_start 未初始化）。"""
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])

    assert canvas._handle_region_click(Point(200, 50)) is True
    assert canvas.is_dragging_region_vertex is True
    assert canvas.region_vertex_drag_index == 0
    assert canvas.region_drag_start is not None

    canvas.mouse_pos = Point(205, 55)
    canvas._drag_region_vertex()
    region = canvas._editor.region_boxes[0]
    assert region['points'][0] == [205.0, 55.0]

    canvas.mouse_pos = Point(210, 60)
    canvas._drag_region_vertex()
    assert region['points'][0] == [210.0, 60.0]


def test_drag_region_vertex_entry_path_scales_delta_by_background_scale():
    """scale != 1 时按位移/scale 移动，而不是跳到绝对坐标。"""
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.background_scale = 2.0
    canvas._handle_region_click(Point(400, 100))
    assert canvas.is_dragging_region_vertex is True

    canvas.mouse_pos = Point(410, 110)
    canvas._drag_region_vertex()
    region = canvas._editor.region_boxes[0]
    assert region['points'][0] == [205.0, 55.0]


def test_insert_region_vertex():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    assert canvas._insert_region_vertex(0, 1, [275.0, 100.0]) is True
    region = canvas._editor.region_boxes[0]
    assert len(region['points']) == 5
    assert region['points'][1] == [275.0, 100.0]


def test_insert_region_vertex_max_points():
    points = [[100.0 + i, 100.0 + (i % 2) * 50] for i in range(REGION_CONFIG['polygon_max_points'])]
    canvas = Canvas(regions=[_poly_region(points)])
    assert canvas._insert_region_vertex(0, 1, [150.0, 150.0]) is False


def test_region_polygon_edge_at():
    canvas = Canvas(regions=[_poly_region([[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]])])
    edge = canvas._region_polygon_edge_at(Point(200, 100))
    assert edge is not None
    r_idx, v_idx, img_pt = edge
    assert r_idx == 0
    assert v_idx == 1
    assert img_pt == [200.0, 100.0]
    assert canvas._region_polygon_edge_at(Point(200, 200)) is None


def test_resize_polygon_region_scales_vertices():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.selected_region = 0
    canvas.region_resize_handle = 'br'
    canvas.is_resizing_region = True

    canvas.region_resize_start = Point(350, 250)
    canvas.mouse_pos = Point(450, 350)
    canvas._resize_region()

    region = canvas._editor.region_boxes[0]
    assert region['x'] == 50.0 and region['y'] == 50.0
    assert region['width'] == 350.0
    assert region['height'] == 250.0
    assert region['points'][1] == [400.0, 175.0]


def test_wheel_scales_polygon_region_vertices():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])])
    canvas.mouse_pos = Point(200, 150)
    canvas.wheelEvent(WheelEvent(120))
    region = canvas._editor.region_boxes[0]
    assert region['width'] > 300.0
    assert len(region['points']) == 4
    assert region['points'][0][1] < 50.0  # 顶点随缩放上移


def test_fixed_blocks_polygon_editing():
    canvas = Canvas(regions=[_poly_region(DIAMOND['points'])], fixed=True)
    before = [list(p) for p in canvas._editor.region_boxes[0]['points']]

    assert canvas._handle_region_click(Point(200, 150)) is False
    canvas.selected_region = 0
    canvas.region_vertex_drag_index = 0
    canvas.is_dragging_region_vertex = True
    canvas.region_drag_start = Point(200, 50)
    canvas.mouse_pos = Point(300, 100)
    canvas._drag_region_vertex()
    assert canvas._editor.region_boxes[0]['points'] == before

    assert canvas._insert_region_vertex(0, 1, [275.0, 100.0]) is False
    assert canvas._delete_region_vertex(0, 0) is False


def test_sample_in_polygon_region_all_inside():
    region = _poly_region(DIAMOND['points'])
    rng = random.Random(42)
    hits = 0
    for _ in range(500):
        sample = Engine()._sample_in_polygon_region(
            region, 40, 40, 400, 300, rng, 200
        )
        if sample is None:
            continue
        hits += 1
        cx, cy = sample
        assert point_in_polygon(cx, cy, region['points'])
        for x, y in ((cx - 20, cy - 20), (cx + 20, cy - 20),
                     (cx - 20, cy + 20), (cx + 20, cy + 20)):
            assert point_in_polygon(x, y, region['points'])
    assert hits > 0


def test_sample_in_polygon_region_too_small_returns_none():
    region = _poly_region([[100.0, 100.0], [110.0, 100.0], [110.0, 110.0], [100.0, 110.0]])
    rng = random.Random(1)
    assert Engine()._sample_in_polygon_region(
        region, 60, 60, 400, 300, rng, 50
    ) is None


def test_sample_center_mixed_rect_and_polygon():
    rect = {'shape_type': 'rectangle', 'x': 20.0, 'y': 20.0, 'width': 100.0, 'height': 100.0}
    poly = _poly_region([[200.0, 200.0], [380.0, 200.0], [380.0, 290.0], [200.0, 290.0]])
    regions = [rect, poly]
    rng = random.Random(7)
    hit_rect = hit_poly = 0
    for _ in range(500):
        sample = Engine()._sample_center_in_regions(
            regions, 30, 30, 400, 300, rng
        )
        assert sample is not None
        cx, cy = sample
        if 20 <= cx <= 120 and 20 <= cy <= 120:
            hit_rect += 1
        elif point_in_polygon(cx, cy, poly['points']):
            hit_poly += 1
        else:
            raise AssertionError(f"sample outside regions: {sample}")
    assert hit_rect > 0
    assert hit_poly > 0


def test_sample_center_all_regions_too_small_returns_none():
    rect = {'shape_type': 'rectangle', 'x': 0.0, 'y': 0.0, 'width': 10.0, 'height': 10.0}
    poly = _poly_region([[200.0, 200.0], [210.0, 200.0], [210.0, 210.0], [200.0, 210.0]])
    rng = random.Random(3)
    assert Engine()._sample_center_in_regions(
        [rect, poly], 60, 60, 400, 300, rng
    ) is None

def test_temp_region_box_rendered_during_drag():
    """矩形区域拖动时必须实时渲染预览（回归：曾漏 _draw_temp_region_box 分支）。"""
    import inspect
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin

    source = inspect.getsource(CanvasRendererMixin._draw_temp_region_box)
    assert "temp_draw_box" in source
    assert "draw_start_pos" in source
    assert "REGION_CONFIG" in source


def test_paint_scene_dispatches_region_rect_preview():
    """paintEvent 必须在 is_drawing_region 时调用 _draw_temp_region_box。"""
    import inspect
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin
    source = inspect.getsource(CanvasRendererMixin._paint_scene)
    assert "is_drawing_region" in source
    assert "_draw_temp_region_box" in source


class MouseEvent:
    def __init__(self, pos, button=1):
        self._pos = pos
        self._button = button
        self._accepted = False

    def pos(self):
        return self._pos

    def button(self):
        return self._button

    def accept(self):
        self._accepted = True


def test_region_polygon_press_ignored_outside_paste_mode():
    """回归：切到 annotate 后，残留的多边形区域绘制不得继续加点/完成。"""
    canvas = Canvas()
    canvas.setFocus = lambda: None
    canvas.is_drawing_region_polygon = True
    canvas._handle_region_polygon_press(Point(100, 100))
    assert len(canvas.temp_region_points) == 1

    canvas._editor.edit_mode = 'annotate'
    canvas.mousePressEvent(MouseEvent(Point(200, 200)))
    assert canvas.temp_region_points == []
    assert canvas.is_drawing_region_polygon is False
    assert canvas._editor.region_boxes == []


def test_region_rect_press_ignored_outside_paste_mode():
    """回归：切到 annotate 后，残留的矩形区域绘制不得继续。"""
    canvas = Canvas()
    canvas.setFocus = lambda: None
    canvas.is_drawing_region = True
    canvas.draw_start_pos = Point(10, 10)
    canvas._editor.edit_mode = 'annotate'
    canvas.mousePressEvent(MouseEvent(Point(200, 200)))
    assert canvas.is_drawing_region is False
    assert canvas._editor.region_boxes == []
