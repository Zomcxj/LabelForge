"""区域绘制会话：不切贴图模式也能画区域，且会话外行为零变化。

这是本轮唯一碰交互层的改动，所以两侧都要钉死：
- 会话内：标注模式下拖拽 → 生成区域，且不会拖走已有检测框；
- 会话外：标注模式拖拽行为与会话功能加入前逐项一致。

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
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from pastelabel.canvas import Canvas

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
        # 探针：会话绝不该调 _set_edit_mode
        self.edit_mode_calls = []
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


def test_session_starts_and_ends():
    """会话起止：begin 返回 True，重复 begin 返回 False，end 后回到未激活。"""
    out = _run_real_qt(_HOST + """
print(canvas.region_session_active())
print(canvas.begin_region_session())
print(canvas.region_session_active())
print(canvas.begin_region_session())
canvas.end_region_session()
print(canvas.region_session_active())
canvas.end_region_session()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "False", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "False", lines[3]
    assert lines[4] == "False", lines[4]
    assert lines[5] == "ok", lines[5]


def test_session_does_not_change_edit_mode():
    """会话不切模式：edit_mode 全程不变，且 `_set_edit_mode` 探针始终为空。

    这是 ③「点编辑区域不切贴图模式」的核心断言。探针是必要的：只断言
    edit_mode 不变的话，「切过去又切回来」也能骗过它。
    """
    out = _run_real_qt(_HOST + """
canvas.begin_region_session()
print(editor.edit_mode)
canvas.end_region_session()
print(editor.edit_mode)
print(editor.edit_mode_calls)
print(canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "annotate", lines[0]
    assert lines[1] == "annotate", lines[1]
    # 会话全程绝不碰 _set_edit_mode
    assert lines[2] == "[]", lines[2]
    # 会话结束后绘制态要复位
    assert lines[3] == "False", lines[3]


def test_session_drag_creates_region_in_annotate_mode():
    """会话内、标注模式下拖动 → 生成一个矩形区域。"""
    out = _run_real_qt(_HOST + """
canvas.begin_region_session()
canvas._handle_region_press(img_to_canvas(30, 30))
print(canvas.draw_start_pos is not None)
canvas._complete_region_drawing(img_to_canvas(120, 110))
print(len(editor.region_boxes))
print(editor.region_boxes[0]['shape_type'])
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "1", lines[1]
    assert lines[2] == "rectangle", lines[2]


def test_session_allows_drawing_several_regions_in_a_row():
    """会话内可连续画多个区域：落笔后不退出绘制态。

    钉 `_after_region_committed()`：若它恒走 `_reset_region_drawing_state()`，
    `is_drawing_region` 变 False、`draw_start_pos` 也不再被重置，第二次
    `_handle_region_press` 就画不出东西——用户「画完一个想再画一个」会静默
    失败。区域绘制由面板「确定」收尾，不是画完一个就退出。
    """
    out = _run_real_qt(_HOST + """
canvas.begin_region_session()
canvas._handle_region_press(img_to_canvas(10, 10))
canvas._complete_region_drawing(img_to_canvas(100, 100))
print(len(editor.region_boxes), canvas.is_drawing_region)
canvas._handle_region_press(img_to_canvas(150, 150))
canvas._complete_region_drawing(img_to_canvas(250, 250))
print(len(editor.region_boxes), canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "1 True", lines[0]
    assert lines[1] == "2 True", lines[1]


def test_session_press_on_existing_box_starts_region_not_box_drag():
    """会话内按下时，落在已有检测框上也不能进入拖动（先去画区域）。

    基线（同一坐标、无会话）见下一条：那里 `is_dragging_box` 为 True。
    两条一起才说明会话**改变了**优先级，而不是坐标恰好没命中。
    """
    out = _run_real_qt(_HOST + """
make_box()
canvas.begin_region_session()
press(60, 60)
print(canvas.is_dragging_box)
print(canvas.draw_start_pos is not None)
print(canvas.is_drawing_region)
print(editor.region_boxes)
""")
    lines = _lines(out)
    assert lines[0] == "False", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]


def test_without_session_annotate_press_on_box_still_drag_it():
    """会话外基线：同一坐标按在已有检测框上仍然进入拖动（行为零变化）。

    构造与上一条完全相同，只有「有没有开会话」不同——差异必须只有这一点。
    """
    out = _run_real_qt(_HOST + """
make_box()
press(60, 60)
print(canvas.is_dragging_box)
print(canvas.selected_box)
print(canvas.is_drawing_region)
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "0", lines[1]
    assert lines[2] == "False", lines[2]


def test_without_session_annotate_press_on_empty_starts_box_draw():
    """会话外：标注模式在空白处按下仍按原有语义（开始画标注框）。"""
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


def test_session_right_click_cancels_current_region_and_keeps_session():
    """会话内右键：只取消**当前这一个**区域，会话继续（还能接着画）。

    这是 ② 的核心：用户要求「删除编辑区域按钮，直接选中后就编辑」——选中
    「区域」即进入可编辑态、可连续画多个。原来右键会把整个会话一起收掉
    （`end_region_session()`），于是「画完一个就再也画不了第二个」，正是用户
    报的「只能画一个」。
    """
    out = _run_real_qt(_HOST + """
canvas.begin_region_session()
canvas._handle_region_press(img_to_canvas(50, 50))
print(canvas.draw_start_pos is not None)
press(50, 50, Qt.RightButton)
print(canvas.region_session_active())
print(canvas.draw_start_pos is None)
print(len(editor.region_boxes))
# 会话还在：接着按下仍能起一个新区域
canvas._handle_region_press(img_to_canvas(120, 120))
print(canvas.draw_start_pos is not None)
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", (
        f"右键把整个会话收掉了——那样就「只能画一个」：{lines[1]}")
    assert lines[2] == "True", lines[2]
    assert lines[3] == "0", lines[3]
    assert lines[4] == "True", (
        f"右键之后画不了下一个区域了：{lines[4]}")


def test_session_end_resets_drawing_state():
    """end 会复位绘制态（会话结束后标注模式恢复正常交互）。"""
    out = _run_real_qt(_HOST + """
canvas.begin_region_session()
canvas._handle_region_press(img_to_canvas(50, 50))
print(canvas.is_drawing_region, canvas.draw_start_pos is not None)
canvas.end_region_session()
print(canvas.is_drawing_region, canvas.draw_start_pos is None)
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]
    assert lines[1] == "False True", lines[1]


def test_session_notifies_editor_on_start_and_end():
    """会话起止都通知编辑器：状态栏与面板上的区域计数要跟着刷。"""
    out = _run_real_qt(_HOST + """
notifications = []
editor._on_region_session_changed = lambda: notifications.append(1)
canvas.begin_region_session()
after_begin = len(notifications)
canvas.end_region_session()
after_end = len(notifications)
print(after_begin, after_end)
""")
    lines = _lines(out)
    assert lines[0] == "1 2", lines[0]


def test_session_does_not_touch_undo():
    """会话不碰撤销栈：begin/落笔/end 都不该调 save_undo_state。

    与「区域不进撤销」一致（区域只限本次运行、不持久化）。
    注意 `_handle_region_press` 自身也不该存撤销——下一条同时钉持久化。
    """
    out = _run_real_qt(_HOST + """
editor.undo_calls = 0
canvas.begin_region_session()
canvas._handle_region_press(img_to_canvas(50, 50))
canvas._complete_region_drawing(img_to_canvas(150, 150))
canvas.end_region_session()
print(editor.undo_calls)
print(len(editor.region_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "0", lines[0]
    assert lines[1] == "1", lines[1]


def test_region_draw_visible_in_session_and_paste_mode():
    """临时区域框可见性：贴图模式可见；会话内标注模式也可见；其余不可见。

    会话内必须可见，否则用户「在标注模式画区域」时看不到自己画的东西。
    """
    out = _run_real_qt(_HOST + """
editor.edit_mode = 'paste'
print(canvas._region_draw_visible())
editor.edit_mode = 'annotate'
print(canvas._region_draw_visible())
canvas.begin_region_session()
print(canvas._region_draw_visible())
canvas.end_region_session()
print(canvas._region_draw_visible())
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "False", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "False", lines[3]
