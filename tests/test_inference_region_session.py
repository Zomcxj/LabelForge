"""区域绘制：面板「画一个区域」按钮驱动，点一下画一个。

取代原先的「区域绘制会话」（选中「区域」就自动进入可编辑态）。会话会**独占
左键**——每个左键都被区域吃掉，于是标签框再也点不到。用户这一轮要求「推理
范围打开时恢复标签绘制」，而恢复的前提就是别再有东西劫持鼠标，所以会话整个
拆掉，改成按需的一次性按钮。

钉死两侧：
- 按钮触发后：标注模式下拖动 → 生成一个区域，且**不会**拖走已有检测框；
- 画完一个就退出绘制态（「点一下画一个」），其余时间交互与平时完全一致。

**宿主用真 `Canvas`**，而不是「混入 mixin + 手设状态」的桩。试过桩方案：
`Canvas.__init__` 把 parent 直接交给 `QWidget.__init__`，所以 editor 必须是
真 QWidget；而绕过 `QWidget.__init__` 后，`mousePressEvent` 沿途会进真 Qt
实现并报 "super-class __init__() was never called"。既然横竖要真控件，
不如整个用真的——这样测的是生产代码路径，脚手架不会与产品行为漂移。

坐标注意：`get_background_rect()` 会按 1024×768 的最小尺寸算缩放（实测
2.56），所以这里**不手设** `is_manual_scale`/`background_scale`，让画布自己
算，再把鼠标点按同一套映射换算，避免脚手架与产品两套坐标系打架。
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_HOST = """
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent, QPixmap
from labelforge.canvas import Canvas

class Editor(QWidget):
    def __init__(self):
        super().__init__()
        self.edit_mode = 'annotate'
        self._is_delete_view = False
        self.region_boxes = []
        self.region_fixed = False
        self.background_images = ['a.png']
        self.current_background_index = 0
        self.detection_boxes = []
        self.current_background = None
        self.selected_item = None
        self.status_label = QLabel()
        self.canvas_items = []
        self._max_labels = 3
        self.show_labels_checkbox = type('C', (), {'isChecked': lambda self: True})()
        # 探针：画区域绝不该调 _set_edit_mode
        self.edit_mode_calls = []
        # 探针：区域增删/落笔后要通知宿主刷新面板列表
        self.region_changed = 0
    def save_undo_state(self):
        self.undo_calls = getattr(self, 'undo_calls', 0) + 1
    def get_image_info(self):
        return {'box_count': len(self.detection_boxes), 'paste_count': 0,
                'width': 400, 'height': 300}
    def get_label_stats(self):
        return {}
    def _set_edit_mode(self, mode):
        self.edit_mode_calls.append(mode)
        self.edit_mode = mode
    def _on_region_changed(self):
        self.region_changed += 1

app = QApplication([])
editor = Editor()
canvas = Canvas(editor)
canvas.resize(400, 300)
editor.current_background = QPixmap(400, 300)

def img_to_canvas(ix, iy):
    '''图像坐标 → 画布坐标（与 get_background_rect 同一套映射）。'''
    r = canvas.get_background_rect()
    return QPoint(int(r.left() + ix * canvas.background_scale),
                  int(r.top() + iy * canvas.background_scale))

def press(x, y, button=Qt.LeftButton):
    r = canvas.get_background_rect()
    pos = QPoint(int(r.left() + x * canvas.background_scale),
                 int(r.top() + y * canvas.background_scale))
    ev = QMouseEvent(QMouseEvent.MouseButtonPress, pos,
                     button, button, Qt.NoModifier)
    canvas.mousePressEvent(ev)

def make_box():
    editor.detection_boxes = [{'x': 20, 'y': 20, 'width': 100, 'height': 80,
                               'label': 'a', 'shape_type': 'rectangle'}]
"""


def test_the_session_api_is_gone():
    """会话这套 API 必须彻底拆掉，不能留个「永远 False」的空壳。

    留着空壳最危险的是它会被误当成「还开着会话」，于是又有人往上面挂逻辑；
    真正的判据是画布上根本没这几个方法。
    """
    out = _run_real_qt(_HOST + """
for name in ('region_session_active', 'begin_region_session',
             'end_region_session', '_notify_region_session_changed'):
    print(name, hasattr(canvas, name))
""")
    lines = _lines(out)
    for line in lines:
        assert line.endswith("False"), f"会话 API 没拆干净：{line}"


def test_add_region_arms_a_one_shot_draw():
    """「画一个区域」按钮：点一下进入绘制态，拖动画出一个，随后自动退出。"""
    out = _run_real_qt(_HOST + """
canvas._start_add_region()
print(canvas.is_drawing_region)
canvas._handle_region_press(img_to_canvas(30, 30))
print(canvas.draw_start_pos is not None)
canvas._complete_region_drawing(img_to_canvas(120, 110))
print(len(editor.region_boxes))
print(editor.region_boxes[0]['shape_type'])
# 画完一个就退出绘制态（「点一下画一个」）
print(canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "True", f"按钮没进入绘制态：{lines[0]}"
    assert lines[1] == "True", lines[1]
    assert lines[2] == "1", f"没画出区域：{lines[2]}"
    assert lines[3] == "rectangle", lines[3]
    assert lines[4] == "False", (
        f"画完一个还赖在绘制态，就变成「一直画」了：{lines[4]}")


def test_drawing_a_second_region_needs_another_click():
    """再点一次按钮才能画第二个——不能「画完一个就再也画不了」。"""
    out = _run_real_qt(_HOST + """
canvas._start_add_region()
canvas._handle_region_press(img_to_canvas(10, 10))
canvas._complete_region_drawing(img_to_canvas(100, 100))
print(len(editor.region_boxes), canvas.is_drawing_region)

# 没再点按钮：这次按下走的是生产里的分支（`mousePressEvent`），应当什么也不画
press(150, 150)
canvas._complete_region_drawing(img_to_canvas(250, 250))
print(len(editor.region_boxes))

# 再点一次按钮，就又能画
canvas._start_add_region()
canvas._handle_region_press(img_to_canvas(150, 150))
canvas._complete_region_drawing(img_to_canvas(250, 250))
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "1 False", lines[0]
    assert lines[1] == "1", (
        f"没点按钮却又画了一个（绘制态没收干净）：{lines[1]}")
    assert lines[2] == "2", f"再点按钮后画不出第二个：{lines[2]}"


def test_add_region_does_not_change_edit_mode():
    """画区域不切模式：edit_mode 全程不变，`_set_edit_mode` 探针始终为空。

    探针是必要的：只断言 edit_mode 不变的话，「切过去又切回来」也能骗过它。
    """
    out = _run_real_qt(_HOST + """
canvas._start_add_region()
print(editor.edit_mode)
canvas._handle_region_press(img_to_canvas(10, 10))
canvas._complete_region_drawing(img_to_canvas(100, 100))
print(editor.edit_mode)
print(editor.edit_mode_calls)
""")
    lines = _lines(out)
    assert lines[0] == "annotate", lines[0]
    assert lines[1] == "annotate", lines[1]
    assert lines[2] == "[]", f"画区域时动了编辑模式：{lines[2]}"


def test_cancel_region_drawing_resets_state():
    """右键取消：只收掉这一次绘制，不影响别的。"""
    out = _run_real_qt(_HOST + """
canvas._start_add_region()
canvas._handle_region_press(img_to_canvas(50, 50))
print(canvas.is_drawing_region, canvas.draw_start_pos is not None)
press(50, 50, Qt.RightButton)
print(canvas.is_drawing_region, canvas.draw_start_pos is None)
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]
    assert lines[1] == "False True", f"右键没收掉绘制态：{lines[1]}"
    assert lines[2] == "0", f"取消时不该落成区域：{lines[2]}"


def test_armed_draw_takes_priority_over_touching_an_existing_box():
    """已经点了「画区域」后，按在已有检测框上也是去画区域，不拖动那个框。"""
    out = _run_real_qt(_HOST + """
make_box()
canvas._start_add_region()
press(60, 60)
print(canvas.is_dragging_box)
print(canvas.draw_start_pos is not None)
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "False", f"点了画区域却去拖了检测框：{lines[0]}"
    assert lines[1] == "True", lines[1]


def test_without_arming_annotate_press_on_box_still_drags_it():
    """没点按钮时的基线：按在检测框上照常进入拖动（恢复标签编辑的核心）。

    这正是拆掉会话的意义：以前会话一开，每个左键都被区域吃掉，标签框再也
    点不到。
    """
    out = _run_real_qt(_HOST + """
make_box()
press(60, 60)
print(canvas.is_dragging_box)
print(canvas.selected_box)
print(canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "True", f"没点按钮时检测框拖不动了：{lines[0]}"
    assert lines[1] == "0", lines[1]
    assert lines[2] == "False", lines[2]


def test_without_arming_annotate_press_on_empty_starts_box_draw():
    """没点按钮时：标注模式在空白处按下仍按原有语义（开始画标注框）。"""
    out = _run_real_qt(_HOST + """
canvas.is_drawing_box = True
press(300, 250)
print(canvas.draw_start_pos is not None)
print(canvas.is_drawing_region)
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "False", lines[1]
    assert lines[2] == "0", lines[2]


def test_committing_a_region_notifies_the_editor():
    """落笔后通知宿主：面板的区域列表、状态栏要跟着刷出新的一条。"""
    out = _run_real_qt(_HOST + """
editor.region_changed = 0
canvas._start_add_region()
canvas._handle_region_press(img_to_canvas(50, 50))
canvas._complete_region_drawing(img_to_canvas(150, 150))
print(editor.region_changed > 0)
""")
    lines = _lines(out)
    assert lines[0] == "True", "落笔后没通知宿主，面板列表不会出现新区域"


def test_drawing_a_region_does_not_touch_undo():
    """画区域不碰撤销栈（区域只限本次运行、不持久化）。"""
    out = _run_real_qt(_HOST + """
editor.undo_calls = 0
canvas._start_add_region()
canvas._handle_region_press(img_to_canvas(50, 50))
canvas._complete_region_drawing(img_to_canvas(150, 150))
print(editor.undo_calls)
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "0", lines[0]
    assert lines[1] == "1", lines[1]


def test_temporary_region_box_visible_in_paste_mode_and_while_drawing():
    """临时区域框可见性：贴图模式可见；绘制态下标注模式也可见；其余不可见。

    绘制态下必须可见，否则用户点了「画一个区域」在图上拖时看不到画的是什么。
    """
    out = _run_real_qt(_HOST + """
editor.edit_mode = 'paste'
print(canvas._region_draw_visible())
editor.edit_mode = 'annotate'
print(canvas._region_draw_visible())
canvas._start_add_region()
print(canvas._region_draw_visible())
canvas._cancel_region_drawing()
print(canvas._region_draw_visible())
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "False", lines[1]
    assert lines[2] == "True", (
        f"绘制态下看不见临时区域框：{lines[2]}")
    assert lines[3] == "False", lines[3]


def test_stale_drawing_state_is_dropped_when_nothing_armed_it():
    """残留的绘制态必须就地清掉，不能让它继续吃掉左键。

    绘制态只在两种情况下算数：贴图模式（右键菜单那个老入口）、或范围面板
    开着（新的「添加区域」按钮）。两个都不成立时还挂着 `is_drawing_region`，
    就是残留——比如从贴图模式切到标注模式、或面板关掉后忘了复位。此时若照
    旧放行，用户会看到「切到标注模式后画不了标注框，反而冒出一个区域」。

    这条与「点了按钮就能画」是一对：既要能用，也不能被残留状态劫持。
    """
    out = _run_real_qt(_HOST + """
# 标注模式、面板没开：没人 arm 过，这是残留
editor.edit_mode = 'annotate'
canvas.is_drawing_region = True
canvas.draw_start_pos = None
press(200, 180)
print(canvas.is_drawing_region)
print(len(editor.region_boxes))

# 贴图模式下同一状态是合法的（右键菜单入口）
editor.edit_mode = 'paste'
canvas.is_drawing_region = True
canvas._region_draw_armed_flag = True
press(200, 180)
print(canvas.is_drawing_region)

# 面板开着 + 真的点过「添加区域」：算数
editor.edit_mode = 'annotate'
editor._scope_panel_open = True
canvas._start_add_region()
press(200, 180)
print(canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "False", f"残留绘制态没被清掉：{lines[0]}"
    assert lines[1] == "0", (
        f"残留绘制态竟然真画出了一个区域：{lines[1]}")
    assert lines[2] == "True", (
        f"贴图模式下的绘制态被误清了：{lines[2]}")
    assert lines[3] == "True", (
        f"点了「添加区域」之后的绘制态被误清了：{lines[3]}")
