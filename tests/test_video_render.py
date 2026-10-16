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


def test_label_text_matches_the_label_bar_font():
    """标签条上的文字必须用标签条自己那套字号画。

    回归：`_draw_label` 拿 `QFontMetrics(font)` 量好了条子尺寸（1080p 下
    27px 高），却**没有 `painter.setFont(font)`**，于是 `drawText` 用
    painter 默认的（应用）字号渲染——条子是按 27px 画的，字只有十多像素，
    切片推理成片里的标注因此看着「字比条小一大截」。画布那条路
    （`_draw_label_above_rect`）设了字体，所以只有视频这条路不一致。

    判据不看具体像素数，看**墨迹与标签字号的比例**：墨迹宽度要接近该字号
    的文字总宽 `horizontalAdvance`，墨迹高度要占 `metrics.height()` 的
    合理比例。这样换字体、换平台都不影响判定；而「没 setFont」时字宽只有
    一半左右、字高也矮一截，两条都过不去。

    显式把应用字体设成 10pt：真实场景里 painter 默认字号就是应用字体，
    必须让「应用字号 ≠ 标签字号」这个前提确定成立，否则本用例会随
    环境默认字号漂移。条高不按实测像素算——标签条与框的上边线同色相接，
    行掩膜会把边线并进来，按 `metrics.height()` 判更稳。
    """
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from PyQt5.QtGui import QFont, QFontMetrics
from pastelabel.engine.video_render import (RenderSpec, draw_boxes,
                                            _auto_font_pixels)

app.setFont(QFont(QFont().family(), 10))        # 应用字体：明显小于标签字号

img = np.full((1080, 1920, 3), 255, np.uint8)
spec = RenderSpec(label_colors={"car": "#FF0000"},
                  show_labels=True, show_scores=True)
BOX_TOP = 1000
out = draw_boxes(img, [{"label": "car", "score": 0.72,
                        "shape_type": "rectangle", "x": 400, "y": BOX_TOP,
                        "width": 300, "height": 60}], spec)

band = out[BOX_TOP - 60:BOX_TOP - 3]            # 只留标签条那一条带
red = (band[:, :, 0] == 255) & (band[:, :, 1] == 0) & (band[:, :, 2] == 0)
rows = np.where(red.any(axis=1))[0]
cols = np.where(red.any(axis=0))[0]
bar = band[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
ink = (bar[:, :, 0] < 100) & (bar[:, :, 1] < 100) & (bar[:, :, 2] < 100)
ink_rows = np.where(ink.any(axis=1))[0]
ink_cols = np.where(ink.any(axis=0))[0]

font = QFont()
font.setPixelSize(_auto_font_pixels(1080))
metrics = QFontMetrics(font)
text = "car 0.72"
print("RESULT", repr((
    int(bar.shape[0]),
    int(bar.shape[1]),
    int(ink_rows.max() - ink_rows.min() + 1) if len(ink_rows) else 0,
    int(ink_cols.max() - ink_cols.min() + 1) if len(ink_cols) else 0,
    int(metrics.height()),
    int(metrics.horizontalAdvance(text)),
)))
''')
    bar_h, bar_w, ink_h, ink_w, metrics_h, advance = result
    assert ink_h > 0 and ink_w > 0, "标签条上一个字形像素都没有，文字根本没画出来"
    assert ink_w > 0.7 * advance, (
        f"文字宽度远小于标签字号应有的宽度（{ink_w} vs {advance}）："
        "drawText 用的不是标签字号的字体")
    assert ink_h > 0.45 * metrics_h, (
        f"文字墨迹高 {ink_h} 不到标签字号高度 {metrics_h} 的一半："
        "字比条小，painter 没设标签字体")
    # 条宽 = 文字宽 + 左右内边距；墨迹不能溢出条子
    assert ink_w <= bar_w, f"文字画出了标签条：{ink_w} > {bar_w}"
    assert bar_w < advance + 4 * 6, (
        f"标签条宽 {bar_w} 相对文字宽 {advance} 过大，条子与文字不匹配")


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


# ---------- 第 12 轮：类别名字号可调 + 预览 ----------

def test_font_slider_zero_means_auto():
    """滑块 0 = 按画面高度自动，返回 0；1..72 是手动像素值。

    这个 0 极易在下游被 `or` 吃掉，所以换算单独成函数并在这里钉住。
    """
    from pastelabel.engine.video_render import (
        FONT_SLIDER_MAX, font_slider_to_pixels)

    assert font_slider_to_pixels(0) == 0
    assert font_slider_to_pixels(-5) == 0
    assert font_slider_to_pixels(None) == 0
    assert font_slider_to_pixels("nonsense") == 0
    assert font_slider_to_pixels(1) == 1
    assert font_slider_to_pixels(24) == 24
    assert font_slider_to_pixels(FONT_SLIDER_MAX) == FONT_SLIDER_MAX
    # 越界夹到上限，不会把字号撑爆
    assert font_slider_to_pixels(9999) == FONT_SLIDER_MAX


def test_font_pixels_to_slider_round_trips():
    """界面初始化滑块用反向换算，两个方向必须一致。"""
    from pastelabel.engine.video_render import (
        font_pixels_to_slider, font_slider_to_pixels)

    for value in (0, 1, 12, 40, 72):
        assert font_pixels_to_slider(font_slider_to_pixels(value)) == value


def test_resolve_font_pixels_falls_back_to_auto():
    """`font_pixels=0` 时按画面高度算；非 0 时原样用。"""
    from pastelabel.engine.video_render import RenderSpec, resolve_font_pixels

    assert resolve_font_pixels(RenderSpec(font_pixels=0), 1080) == 27
    assert resolve_font_pixels(RenderSpec(font_pixels=40), 1080) == 40


def test_preview_size_only_shrinks():
    """预览按最长边等比缩到上限；本来就更小的图不放大。"""
    from pastelabel.engine.video_render import preview_size

    assert preview_size(1920, 1080, 360) == (360, 202)   # 202.5 取偶
    assert preview_size(100, 80, 360) == (100, 80)      # 不放大
    assert preview_size(360, 360, 360) == (360, 360)    # 正好等于上限


def test_preview_font_size_changes_rendered_text():
    """字号滑块真的改变预览里文字的像素量（用户要的就是这个）。

    只断言「函数被调用过」证明不了字号生效，所以数类别色像素：
    字号越大，标签条越大，红像素越多。
    """
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, render_preview

img = np.zeros((480, 640, 3), np.uint8)
img[:, :] = (34, 51, 68)
boxes = [{"label": "person", "shape_type": "rectangle",
          "x": 150, "y": 200, "width": 300, "height": 200}]

counts = []
pixels = []
for font_pixels in (0, 14, 24, 40, 72):
    spec = RenderSpec(labels={"person"}, show_labels=True, show_scores=True,
                      label_colors={"person": "#FF0000"},
                      font_pixels=font_pixels)
    out, used = render_preview(img, boxes, spec, max_side=360)
    red = (out[:, :, 0] > 200) & (out[:, :, 1] < 60) & (out[:, :, 2] < 60)
    counts.append(int(red.sum()))
    pixels.append(used)

print("RESULT", repr((counts, pixels)))
''')
    counts, pixels = result
    # 0 解析成「按 270 高的画面自动」= 10px
    assert pixels == [10, 14, 24, 40, 72], pixels
    assert counts == sorted(counts), f"字号变大像素却没变多：{counts}"
    assert counts[-1] > counts[0] * 5, f"字号 0 与 72 差别太小：{counts}"


def test_render_preview_does_not_mutate_the_input():
    """预览返回新数组，不动调用方的原图（和 draw_boxes 同一契约）。"""
    result = _run_qt('''
import sys
sys.path.insert(0, r"%s")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from pastelabel.engine.video_render import RenderSpec, render_preview

img = np.zeros((60, 60, 3), np.uint8)
img[:, :] = 9
before = img.copy()
spec = RenderSpec(labels={"x"}, show_labels=True,
                  label_colors={"x": "#FF0000"}, font_pixels=20)
out, _ = render_preview(img, [{"label": "x", "shape_type": "rectangle",
                               "x": 5, "y": 5, "width": 30, "height": 30}],
                        spec, max_side=0)
print("RESULT", repr((bool((img == before).all()), bool((out != before).any()),
                     out is not img)))
''')
    unchanged, drew, is_new = result
    assert unchanged is True, "原图被改了"
    assert drew is True, "什么都没画"
    assert is_new is True


def test_render_preview_rejects_bad_shapes():
    """非 HxWx3 输入照旧报错，和 draw_boxes 同一口径。"""
    from pastelabel.engine.video_render import render_preview

    import numpy as np
    with pytest.raises(ValueError):
        render_preview(np.zeros((10, 10), np.uint8), [])
