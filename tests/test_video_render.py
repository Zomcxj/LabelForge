"""视频合成时的标签叠加绘制测试。

绘制必须真实跑 Qt：mock 一个 QPainter 只能证明方法被调用过，证明不了
中文能画出来、颜色通道没搞反、行步长对得上。所以这里用子进程 +
`QImage` 离屏渲染，再读像素核对。

约定（同 test_video_ui）：纯逻辑断言不实例化 Qt，真实绘制走子进程。
"""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(script):
    """在子进程里跑真实 Qt 脚本，返回打印出来的 RESULT 行内容。"""
    proc = subprocess.run([sys.executable, "-c", script % str(ROOT)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


# ---------- 模块契约 ----------

def test_render_module_does_not_import_qt_at_module_level():
    """渲染模块不在导入期拉 PyQt5，纯逻辑测试与无 Qt 环境才能导入。"""
    from pastelabel.engine import video_render

    src = Path(video_render.__file__).read_text(encoding="utf-8")
    top_level = [
        line for line in src.splitlines()
        if (line.startswith("import ") or line.startswith("from "))
        and "PyQt5" in line
    ]
    assert not top_level, top_level


def test_render_spec_fields():
    """RenderSpec 收集绘制开关与颜色，颜色由主线程预先取好传进来。"""
    from pastelabel.engine.video_render import RenderSpec

    spec = RenderSpec(label_colors={"person": "#FF0000"})
    assert spec.show_labels is False
    assert spec.show_scores is False
    assert spec.labels is None
    assert spec.label_colors == {"person": "#FF0000"}


def test_filter_boxes_by_label_selection():
    """类别筛选：只留勾选的类别；labels=None 表示全画。"""
    from pastelabel.engine.video_render import filter_boxes

    boxes = [{"label": "person"}, {"label": "car"}, {"label": "person"}]
    assert filter_boxes(boxes, None) == boxes
    assert filter_boxes(boxes, {"person"}) == [boxes[0], boxes[2]]
    assert filter_boxes(boxes, set()) == []


def test_color_for_label_falls_back_to_default():
    """没配颜色的类别用兜底色，不能抛异常。"""
    from pastelabel.engine.video_render import color_for_label

    assert color_for_label({"person": "#FF0000"}, "person") == (255, 0, 0)
    r, g, b = color_for_label({"person": "#FF0000"}, "car")
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    # 空标签也要能拿到颜色
    assert len(color_for_label({}, "")) == 3


def test_bad_color_hex_falls_back():
    """颜色串损坏时兜底，不能把 #GGGGGG 解析炸掉。"""
    from pastelabel.engine.video_render import color_for_label

    for bad in ("#GGGGGG", "red", "#FFF", ""):
        assert len(color_for_label({"x": bad}, "x")) == 3


# ---------- 真实 Qt 绘制 ----------

def test_draw_boxes_paints_border_and_returns_same_shape():
    """画一个框：输出尺寸不变、框边出现类别色、框内不被填充。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((80, 120, 3), np.uint8)          # RGB 黑底
spec = RenderSpec(label_colors={"person": "#FF0000"})
out = draw_boxes(img, [{"label": "person", "shape_type": "rectangle",
                        "x": 20, "y": 20, "width": 60, "height": 40}], spec)

border = tuple(int(v) for v in out[20, 40])     # 上边中点
inside = tuple(int(v) for v in out[40, 50])     # 框内
outside = tuple(int(v) for v in out[5, 5])      # 框外
print("RESULT", repr((out.shape == img.shape, out.dtype == img.dtype,
                      border, inside, outside)))
''')
    same_shape, same_dtype, border, inside, outside = result
    assert same_shape and same_dtype, "绘制不能改变数组尺寸与类型"
    assert border == (255, 0, 0), f"框边应为类别色，实际 {border}"
    assert inside == (0, 0, 0), f"框内不该被填充，实际 {inside}"
    assert outside == (0, 0, 0), f"框外不该被画到，实际 {outside}"


def test_draw_boxes_does_not_mutate_input():
    """输入数组不能被就地改掉（调用方还要用它编码）。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((60, 60, 3), np.uint8)
spec = RenderSpec(label_colors={"person": "#00FF00"})
out = draw_boxes(img, [{"label": "person", "shape_type": "rectangle",
                        "x": 10, "y": 10, "width": 30, "height": 30}], spec)
print("RESULT", repr((int(img.sum()), out is not img)))
''')
    img_sum, is_new = result
    assert img_sum == 0, "原数组被就地修改了"
    assert is_new is True, "应返回新数组"


def test_draw_boxes_handles_polygon_and_rotation():
    """多边形与旋转框按各自几何绘制，不是当成外接矩形。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

spec = RenderSpec(label_colors={"a": "#FF0000"})

poly = np.zeros((100, 100, 3), np.uint8)
out = draw_boxes(poly, [{"label": "a", "shape_type": "polygon",
                         "points": [[10, 10], [90, 10], [50, 90]]}], spec)
# 三角形顶点附近有边，外接矩形的右上角空白区不应被画到
near_edge = int(out[10, 50].sum())          # 上边中点
corner = int(out[12, 12].sum())             # 外接矩形左上角，三角之外
inside = int(out[30, 50].sum())             # 三角形内部

rot = np.zeros((100, 100, 3), np.uint8)
out2 = draw_boxes(rot, [{"label": "a", "shape_type": "rotation",
                         "points": [[50, 10], [90, 50], [50, 90], [10, 50]]}], spec)
# 旋转框的四条边中点：菱形边上有色，菱形的四个角外侧无色
edge_r = int(out2[50, 90].sum())
corner2 = int(out2[12, 12].sum())

print("RESULT", repr((near_edge, corner, inside, edge_r, corner2)))
''')
    near_edge, corner, inside, edge_r, corner2 = result
    assert near_edge > 0, "多边形上边没画出来"
    assert corner == 0, "多边形被画成了外接矩形"
    assert inside == 0, "多边形内部不该被填充"
    assert edge_r > 0, "旋转框边没画出来"
    assert corner2 == 0, "旋转框被画成了外接矩形"


def test_draw_boxes_handles_point_shape():
    """关键点按点绘制（画在坐标处的小方块），不是零宽矩形。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((60, 60, 3), np.uint8)
spec = RenderSpec(label_colors={"k": "#00FF00"})
out = draw_boxes(img, [{"label": "k", "shape_type": "point",
                        "x": 30, "y": 30, "width": 0, "height": 0,
                        "points": [[30, 30]]}], spec)
at_point = int(out[30, 30].sum())
far = int(out[5, 5].sum())
print("RESULT", repr((at_point, far)))
''')
    at_point, far = result
    assert at_point > 0, "关键点没有画出来"
    assert far == 0, "关键点画到了别处"


def test_chinese_label_text_is_drawn():
    """中文类别名要真的画出字形：标签条上必须出现文字色的笔画像素。

    白底作图好判读：标签条=类别色、字形=黑色、未画到=白色。
    只统计「有条、有色」是不够的——实心色条也能满足，必须证明
    条上有字形笔画（黑像素），否则中文没渲染出来时测试照样绿。

    平台默认字体可能没有汉字字形（offscreen 平台默认 Helvetica 就没有），
    那样 drawText 一个像素都画不出来，看着像代码 bug。这里显式挑一个
    能画汉字的字体设成应用字体，把「有没有画字」和「系统装了什么字体」分开。
    """
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from PyQt5.QtGui import QFont, QFontMetrics
from pastelabel.engine.video_render import RenderSpec, draw_boxes

CANDIDATES = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC")
chosen = None
for fam in CANDIDATES:
    probe = QFont(fam)
    if QFontMetrics(probe).inFont("缺"):
        chosen = fam
        break
if chosen is None:
    print("RESULT", repr(("NOFONT",)))
else:
    app.setFont(QFont(chosen, 10))
    img = np.full((100, 200, 3), 255, np.uint8)          # 白底
    spec = RenderSpec(label_colors={"缺陷": "#FF0000"}, show_labels=True)
    out = draw_boxes(img, [{"label": "缺陷", "shape_type": "rectangle",
                            "x": 20, "y": 50, "width": 100, "height": 30}], spec)
    above = out[10:48]
    flat = above.reshape(-1, 3)
    red = int((flat == (255, 0, 0)).all(axis=1).sum())    # 标签条
    black = int((flat == (0, 0, 0)).all(axis=1).sum())    # 字形笔画
    white = int((flat == (255, 255, 255)).all(axis=1).sum())
    print("RESULT", repr((red, black, white)))
''')
    if result == "NOFONT":
        pytest.skip("系统没有可画汉字的字体，无法验证中文渲染")
    red, black, white = result
    assert red > 20, f"标签条没画出来：{red}"
    assert black > 5, f"标签条上没有字形笔画，中文可能没渲染：{black}"
    assert white > 0, "框上方除标签条外应保持背景色"


def test_label_bar_uses_the_category_color():
    """不同类别的标签条各用各自的类别色，不是一律同色。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.full((120, 260, 3), 255, np.uint8)
spec = RenderSpec(label_colors={"a": "#FF0000", "b": "#0000FF"}, show_labels=True)
out = draw_boxes(img, [
    {"label": "a", "shape_type": "rectangle", "x": 10, "y": 50, "width": 80, "height": 20},
    {"label": "b", "shape_type": "rectangle", "x": 150, "y": 50, "width": 80, "height": 20},
], spec)
flat = out[10:48].reshape(-1, 3)
red = int((flat == (255, 0, 0)).all(axis=1).sum())
blue = int((flat == (0, 0, 255)).all(axis=1).sum())
print("RESULT", repr((red, blue)))
''')
    red, blue = result
    assert red > 20, f"类别 a 的红条没画出来：{red}"
    assert blue > 20, f"类别 b 的蓝条没画出来：{blue}"


def test_draw_boxes_result_owns_its_memory():
    """返回数组最终必须落在一块 numpy 自持的内存上。

    回归：曾用 `ascontiguousarray(切片)` 读回，本就连续时它原样返回视图，
    数组指着 QImage 缓冲；QImage 在函数返回时析构，数组随即悬空——
    表现为随机花屏、随机丢框，而且完全不报错。

    这里查的是缓冲血缘的**末端**：`copy()` 出来的是自持 ndarray，
    而视图链末端会是 `memoryview` / `sip.voidptr`（谁也没保证它活着）。
    比「反复分配看看内容变不变」可靠：那取决于分配器什么时候复用，
    不稳定，可能漏掉。
    """
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((40, 80, 3), np.uint8)
out = draw_boxes(img, [{"label": "x", "shape_type": "rectangle",
                        "x": 5, "y": 5, "width": 30, "height": 20}],
                 RenderSpec(label_colors={"x": "#FF0000"}))

node = out
chain = [type(node).__name__]
while isinstance(node, np.ndarray) and node.base is not None:
    node = node.base
    chain.append(type(node).__name__)
terminal_owns = isinstance(node, np.ndarray) and bool(node.flags["OWNDATA"])
print("RESULT", repr((terminal_owns, chain)))
''')
    terminal_owns, chain = result
    assert terminal_owns is True, f"返回数组是外部缓冲的视图，会悬空：{chain}"


def test_score_shown_only_when_enabled():
    """置信度开关：关掉时标签只有类别名，打开才带两位小数分数。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
from pastelabel.engine.video_render import RenderSpec, label_text

box = {"label": "person", "score": 0.8765}
off = label_text(box, RenderSpec(show_labels=True, show_scores=False))
on = label_text(box, RenderSpec(show_labels=True, show_scores=True))
manual = label_text({"label": "person"}, RenderSpec(show_labels=True, show_scores=True))
print("RESULT", repr((off, on, manual)))
''')
    off, on, manual = result
    assert off == "person", off
    assert on == "person 0.88", on
    assert manual == "person", f"人工框没有分数，不该冒出数字：{manual}"


def test_labels_off_means_no_text_drawn():
    """关掉「类别名」时只画框，不画文字。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((100, 200, 3), np.uint8)
spec = RenderSpec(label_colors={"person": "#FF0000"}, show_labels=False)
out = draw_boxes(img, [{"label": "person", "shape_type": "rectangle",
                        "x": 20, "y": 50, "width": 100, "height": 30}], spec)
above = int(out[10:48].sum())
print("RESULT", repr((above,)))
''')
    (above,) = result
    assert above == 0, f"关掉类别名后框上方不该有任何绘制：{above}"


def test_bgr_and_rgb_are_not_swapped():
    """颜色通道顺序：RGB 数组里 (255,0,0) 必须画成红色而非蓝色。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((80, 80, 3), np.uint8)
spec = RenderSpec(label_colors={"x": "#0000FF"})     # 纯蓝
out = draw_boxes(img, [{"label": "x", "shape_type": "rectangle",
                        "x": 20, "y": 20, "width": 40, "height": 40}], spec)
print("RESULT", repr(tuple(int(v) for v in out[20, 40])))
''')
    pixel = tuple(result)
    assert pixel == (0, 0, 255), f"通道被交换了：{pixel}"


def test_out_of_bounds_boxes_do_not_crash():
    """框超出画面（推理框可能越界）时钳制绘制，不能崩。

    注意区分两种情况：**部分重叠**的框要画出落在画面内的边；四条边
    都在画面外的框（比如把整个画面包住的超大框）本来就无可见边，
    什么都不画才是对的。
    """
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.zeros((60, 60, 3), np.uint8)
spec = RenderSpec(label_colors={"x": "#FF0000"})
# 部分重叠：左边与上边在画面外，右边/下边落在画面内
partial = draw_boxes(img, [{"label": "x", "shape_type": "rectangle",
                            "x": -20, "y": 10, "width": 50, "height": 30}], spec)
# 完全包住画面：四条边都在画面外，无可见边
enclosing = draw_boxes(img, [{"label": "x", "shape_type": "rectangle",
                              "x": -20, "y": -20, "width": 200, "height": 200}], spec)
# 退化尺寸与完全在画面外
others = draw_boxes(img, [
    {"label": "x", "shape_type": "rectangle", "x": 59, "y": 59, "width": 0, "height": 0},
    {"label": "x", "shape_type": "rectangle", "x": 100, "y": 100, "width": 20, "height": 20},
], spec)
print("RESULT", repr((partial.shape, int((partial > 0).sum()),
                     int((enclosing > 0).sum()), int((others > 0).sum()))))
''')
    shape, partial_px, enclosing_px, others_px = result
    assert shape == (60, 60, 3)
    assert partial_px > 0, "部分重叠的框应画出画面内的边"
    assert enclosing_px == 0, "四条边都在画面外的框本来就没有可见边"
    assert others_px == 0, "退化/完全在外的框不该画出东西"


def test_empty_boxes_returns_equivalent_image():
    """没有框时返回等价图像，不额外改动像素。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

img = np.full((40, 40, 3), 7, np.uint8)
out = draw_boxes(img, [], RenderSpec())
print("RESULT", repr((bool((out == img).all()), out is not img)))
''')
    identical, is_new = result
    assert identical is True
    assert is_new is True, "也应返回副本，调用方语义统一"


def test_non_contiguous_input_is_handled():
    """切片来的非连续数组（如 RGB 通道逆序后的视图）也能画。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, draw_boxes

base = np.zeros((40, 80, 3), np.uint8)
view = base[:, ::-1, :]                      # 非连续
spec = RenderSpec(label_colors={"x": "#FF0000"})
out = draw_boxes(view, [{"label": "x", "shape_type": "rectangle",
                         "x": 10, "y": 10, "width": 20, "height": 20}], spec)
print("RESULT", repr((out.shape, int(out.sum()) > 0)))
''')
    shape, drew = result
    assert shape == (40, 80, 3)
    assert drew is True
