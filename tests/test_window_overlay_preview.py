"""原图叠加窗口网格预览：参数 → 窗口数、网格落位与非法参数处理。

窗口数学必须是「绘制」与「引擎」两处一致，否则用户看到的网格与实际跑的
窗口对不上——这是本控件存在的唯一理由，所以逐点核对窗口起点。

「非法参数」的真实口径（已实测，勿凭直觉改写）：
  - 会抛 `ScopeError`：窗口尺寸 <=0 或 >MAX_WINDOW、重叠为负、窗口数超
    MAX_WINDOWS。
  - **不会**抛：重叠 >= 窗口尺寸——`normalized_window()` 把它静默钳到
    `min(cw,ch)-1`。所以「重叠太大」不报错，只会让窗口数暴涨并可能撞上
    MAX_WINDOWS。预览必须对这两种情况都给得出一致的结论。
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


def _spec_size(line):
    """把 `print(w.spec())` 的输出解析回「三个尺寸键」的元组。

    窗口 spec 会随功能增加多出键（比如 `min_visible`），所以断言尺寸时不该
    钉死整份字典的字符串，否则每加一个合法参数都要来改一批测试。
    """
    import ast
    spec = ast.literal_eval(line)
    return (spec["w"], spec["h"], spec["overlap"])


def test_real_preview_window_count_matches_engine():
    """预览算出的窗口数必须等于 iter_windows() 的长度。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
from labelforge.engine.inference_scope import iter_windows

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((2000, 1000))
p.set_spec({'w': 640, 'h': 640, 'overlap': 200})
print(p.window_count(), len(iter_windows(2000, 1000,
                                         {'w': 640, 'h': 640, 'overlap': 200})))
""")
    count, engine = _lines(out)[0].split()
    assert count == engine, out


def test_real_preview_clamps_overlap_like_engine():
    """重叠 >= 窗口尺寸时**不报错**：与引擎一样按 min(cw,ch)-1 钳制。

    引擎的 `normalized_window()` 就是这么做的，预览若在这里报「非法」，
    用户会看到「预览说不行、但推理跑得动」的矛盾。

    实测值：500×500 图配 400×400 窗、重叠 400 → 钳成 (400,400,399)，
    每轴 400/1 → 101 个起点，101×101 = 10201 窗。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
from labelforge.engine.inference_scope import iter_windows

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((500, 500))
p.set_spec({'w': 400, 'h': 400, 'overlap': 400})
print(p.window_count(), p.is_valid())
print(p.window_count() == len(iter_windows(500, 500,
                                           {'w': 400, 'h': 400, 'overlap': 400})))
""")
    lines = _lines(out)
    assert lines[0] == "10201 True", lines[0]
    assert lines[1] == "True", lines[1]


def test_real_preview_rejects_window_size_out_of_range():
    """窗口尺寸超上限：引擎抛 ScopeError，预览必须显示为不可用而不是崩溃。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((500, 500))
p.set_spec({'w': 20000, 'h': 20000, 'overlap': 0})
print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_preview_rejects_zero_and_negative_window():
    """窗口尺寸为 0 / 负数：同样显示为不可用，不抛异常。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((500, 500))
for bad in ({'w': 0, 'h': 100, 'overlap': 0}, {'w': -5, 'h': 100, 'overlap': 0}):
    p.set_spec(bad)
    print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "None False", lines[1]
    assert lines[2] == "ok", lines[2]


def test_real_preview_rejects_too_many_windows():
    """窗口数超 MAX_WINDOWS：引擎抛错，预览必须给「参数非法」而不是画几万个框。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((20000, 20000))
p.set_spec({'w': 8, 'h': 8, 'overlap': 7})
print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_preview_without_image_is_not_valid():
    """没加载图片时预览不可用，但仍能重绘不崩。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_spec({'w': 640, 'h': 640, 'overlap': 200})
print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_preview_without_spec_is_not_valid():
    """参数还没设过时不可用（弹窗刚建、还没填参数）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((2000, 1000))
print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_preview_draws_grid_over_image():
    """重绘后画布上确实出现**网格线**像素，颜色是主题的 accent。

    不能只断言「有非白像素」：`paintEvent` 开头就 `fillRect(canvas_bg)`，
    底色本身就非白，把 `_draw_grid` 整个改成 no-op 那条断言照样绿（实测过）。
    这里改为数**accent 色**的像素，且要求它成片出现（格子是线，不是一两点
    抗锯齿噪点）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QImage
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
from labelforge.ui.theme import ThemeManager

app = QApplication([])
host = QWidget()
p = WindowOverlayPreview(host)
p.resize(400, 300)
p.set_image((2000, 1000))
p.set_spec({'w': 640, 'h': 640, 'overlap': 200})
grid = QColor(ThemeManager.get_theme()['accent'])
img = QImage(400, 300, QImage.Format_ARGB32)
img.fill(QColor('white'))
painter = QPainter(img)
p.render(painter, QPoint())
painter.end()

def is_grid(c):
    # 允许一点抗锯齿容差，但必须明显偏蓝（accent #2196F3）
    return (abs(c.red() - grid.red()) < 40 and abs(c.green() - grid.green()) < 40
            and abs(c.blue() - grid.blue()) < 40)

n = sum(1 for y in range(300) for x in range(400) if is_grid(img.pixelColor(x, y)))
print('grid pixels', n)
print('enough', n > 200)
# 底色不是 accent：否则上面的计数会把底色也算进去，断言退化
print('bg is not grid', not is_grid(QColor(ThemeManager.get_theme()['canvas_bg'])))
""")
    lines = _lines(out)
    n = int(lines[0].split()[2])
    assert n > 200, lines[0]
    assert lines[1] == "enough True", lines[1]
    assert lines[2] == "bg is not grid True", lines[2]


def test_real_preview_grid_density_tracks_window_count():
    """窗口数变了，画出来的网格跟着变：逐窗口画，不是只画一个外框。

    断言 accent 色像素数**随窗口数显著增长**，而不是只断言「画了东西」——
    后者在 `_draw_grid` 成 no-op 时也为真（底色 fillRect 就够），是空转。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QImage, QColor
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
from labelforge.ui.theme import ThemeManager

app = QApplication([])
host = QWidget()
grid = QColor(ThemeManager.get_theme()['accent'])

def count_grid(spec):
    p = WindowOverlayPreview(host)
    p.resize(400, 300)
    p.set_image((2000, 1000))
    p.set_spec(spec)
    img = QImage(400, 300, QImage.Format_ARGB32)
    img.fill(QColor('white'))
    painter = QPainter(img)
    p.render(painter, QPoint())
    painter.end()
    return p.window_count(), sum(
        1 for y in range(300) for x in range(400)
        if abs(img.pixelColor(x, y).red() - grid.red()) < 40
        and abs(img.pixelColor(x, y).green() - grid.green()) < 40
        and abs(img.pixelColor(x, y).blue() - grid.blue()) < 40)

few_n, few_px = count_grid({'w': 1000, 'h': 1000, 'overlap': 0})
many_n, many_px = count_grid({'w': 200, 'h': 200, 'overlap': 0})
print('windows', few_n, many_n)
print('grid pixels', few_px, many_px)
# 两者都真的画了网格线
print('few drew', few_px > 50)
print('many drew', many_px > 50)
# 窗口多的必须画得多：只画一个外框时两者像素数会一样
print('grows', many_px > few_px * 1.5)
""")
    lines = _lines(out)
    assert lines[0] == "windows 2 50", lines[0]
    few_px, many_px = (int(v) for v in lines[1].split()[-2:])
    assert few_px > 50, lines[1]
    assert many_px > 50, lines[1]
    assert lines[2] == "few drew True", lines[2]
    assert lines[3] == "many drew True", lines[3]
    assert lines[4] == "grows True", lines[4]


def test_real_preview_skips_offscreen_windows_when_clipping():
    """裁剪是「逻辑」保证，不是靠耗时：越界窗口不参与绘制。

    为什么不测耗时：实测在 4489 窗、400×300 控件下，裁剪版 0.0155s、
    未裁剪版 0.0156s——两者无法区分，用「耗时 < N 秒」写不出能杀死
    裁剪缺失的用例（我试过，变异体存活）。所以改为直接验行为：
    窗口全在可视区外时，一帧画下来不该出现任何窗口边线。

    做法：把图放大到远大于控件，只保留左上一个窗口在视口内，其余全在外。
    若裁剪生效，越界窗口不画；若失效，它们会被 drawRect 到控件外——
    对渲染结果无可见影响，所以这里改为直接检查 `_draw_grid` 的调用集合。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QRectF
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
host = QWidget()
p = WindowOverlayPreview(host)
p.resize(100, 100)
p.set_image((1000, 1000))
p.set_spec({'w': 100, 'h': 100, 'overlap': 0})     # 100 窗

# 记录实际被 drawRect 的矩形：包一层替身 painter。
# scale 取 1.0（原图 1:1）→ 1000px 的图铺在 100px 的控件上，
# 只有左上角那一窗与视口相交，其余 99 个都在视口外。
class Spy:
    def __init__(self):
        self.rects = []
    def setPen(self, *a): pass
    def drawRect(self, rect):
        self.rects.append(rect)

spy = Spy()
p._draw_grid(spy, p._window_rects(), 1.0, 0.0, 0.0, {'accent': '#FF0000'})
visible = QRectF(p.rect())
inside = [r for r in spy.rects if r.intersects(visible)]
outside = [r for r in spy.rects if not r.intersects(visible)]
print('total windows :', len(p._window_rects()))
print('drawn         :', len(spy.rects))
print('all drawn visible:', len(outside) == 0)
print('drew something :', len(inside) > 0)
""")
    lines = _lines(out)
    assert lines[0] == "total windows : 100", lines[0]
    assert lines[2] == "all drawn visible: True", lines[2]
    assert lines[3] == "drew something : True", lines[3]
    drawn = int(lines[1].split(":")[1])
    assert drawn < 100, f"裁剪后不该画满 100 个，实测 {drawn}"


def test_real_preview_survives_large_window_count():
    """数千窗口时仍能取到窗口数并完成一次重绘，不抛异常。

    这条**不测耗时**：实测在 4489 窗、400×300 控件下，裁剪版 0.0155s、
    未裁剪版 0.0156s——两者无法区分，任何「耗时 < N 秒」的门槛都杀不死
    「裁剪缺失」这个变异体（我试过，变异体存活）。裁剪由
    `test_real_preview_skips_offscreen_windows_when_clipping` 从行为上钉死；
    这里只保证大规模输入下不崩、数目正确。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QImage
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
host = QWidget()
p = WindowOverlayPreview(host)
p.resize(400, 300)
p.set_image((20000, 20000))
p.set_spec({'w': 300, 'h': 300, 'overlap': 0})
print('windows:', p.window_count())
img = QImage(400, 300, QImage.Format_ARGB32)
img.fill(0)
painter = QPainter(img)
p.render(painter, QPoint())
painter.end()
print('rendered ok')
""")
    lines = _lines(out)
    windows = int(lines[0].split(":")[1])
    # 20000/300 → 每轴 67 个起点 → 4489 窗（实测值，且 < MAX_WINDOWS）
    assert windows == 4489, windows
    assert lines[1] == "rendered ok", lines[1]


def test_real_preview_paints_the_background_image_not_just_a_fill():
    """预览画的是**真实原图**，不是纯色占位（复审 B-4）。

    做法：给一张左半绿、右半蓝的图，缩略后两侧都应出现对应色相。若只填
    `panel_bg` 纯色（改动前的行为），两侧都没有绿色/蓝色像素，这条会红。

    用色相而不是精确 RGB：绘制路径上有缩放与混合，精确值不稳定；色相关系
    （左偏绿、右偏蓝）才是「图真的画了」的判别量。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.resize(200, 200)
img = QImage(400, 400, QImage.Format_RGB888)
img.fill(QColor(0, 220, 0))                       # 左半 绿
for y in range(400):
    for x in range(200, 400):
        img.setPixelColor(x, y, QColor(0, 0, 220))  # 右半 蓝
p.set_pixmap(QPixmap.fromImage(img))
p.set_spec({'w': 400, 'h': 400, 'overlap': 0})

out = QImage(200, 200, QImage.Format_ARGB32)
out.fill(QColor('white'))
painter = QPainter(out)
p.render(painter, QPoint())
painter.end()

def green_ish(c): return c.green() > 120 and c.red() < 100 and c.blue() < 100
def blue_ish(c): return c.blue() > 120 and c.red() < 100 and c.green() < 100

print('image size synced', p._image_size)
left = [out.pixelColor(x, 100) for x in range(2, 30)]
right = [out.pixelColor(x, 100) for x in range(170, 198)]
print('left is green', any(green_ish(c) for c in left))
print('right is blue', any(blue_ish(c) for c in right))
""")
    lines = _lines(out)
    # 图比尺寸更权威：设了 pixmap 就用它的尺寸
    assert lines[0] == "image size synced (400, 400)", lines[0]
    assert lines[1] == "left is green True", lines[1]
    assert lines[2] == "right is blue True", lines[2]


def test_real_preview_without_pixmap_falls_back_to_placeholder():
    """没有原图时退回纯色占位并仍画网格：预览是调参参照，缺图不该报错。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QImage, QColor
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
host = QWidget()
p = WindowOverlayPreview(host)
p.resize(200, 200)
p.set_image((400, 400))
p.set_spec({'w': 200, 'h': 200, 'overlap': 0})
print('no pixmap', p._pixmap is None)
print('count', p.window_count())
img = QImage(200, 200, QImage.Format_ARGB32)
img.fill(QColor('white'))
painter = QPainter(img)
p.render(painter, QPoint())
painter.end()
n = sum(1 for y in range(0, 200, 2) for x in range(0, 200, 2)
        if img.pixelColor(x, y) != QColor('white'))
print('drew something', n > 0)
# 传空 pixmap 等于没有：退回占位而不是画一片空白
from PySide6.QtGui import QPixmap
p.set_pixmap(QPixmap())
print('null pixmap ignored', p._pixmap is None)
""")
    lines = _lines(out)
    assert lines[0] == "no pixmap True", lines[0]
    assert lines[1] == "count 4", lines[1]
    assert lines[2] == "drew something True", lines[2]
    assert lines[3] == "null pixmap ignored True", lines[3]


def test_real_params_widget_estimates_and_validates_like_dialog():
    """参数控件是滑窗弹窗与范围面板的共同内核，口径必须一致。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_params import WindowParamsWidget

app = QApplication([])
w = WindowParamsWidget(reference_size=(3840, 2160),
                       spec={'w': 640, 'h': 640, 'overlap': 200})
print(w.spec())
print(w.window_count(), w.window_grid(), w.is_valid())
# 非法参数：超上限时无效
w._w.setValue(64); w._h.setValue(64); w._ov.setValue(64)
print(w.window_count(), w.is_valid())
print('非法' in w.estimate_text())
# 坏 spec 逐项兜底，不整个崩
w2 = WindowParamsWidget(reference_size=(100, 100),
                        spec={'w': 'x', 'h': None, 'overlap': [1]})
print(w2.spec())
""")
    lines = _lines(out)
    # 三个尺寸键照旧，另外必须带上 `min_visible`（跨窗去重要按它过滤低可见框）。
    # 不整份比较：spec 里的合法键以后还会增加。
    assert _spec_size(lines[0]) == (640, 640, 200), lines[0]
    assert "min_visible" in lines[0], lines[0]
    assert lines[1] == "45 (5, 9) True", lines[1]
    assert lines[2] == "None False", lines[2]
    assert lines[3] == "True", lines[3]
    # 坏值回退到默认
    assert _spec_size(lines[4]) == (640, 640, 200), lines[4]
    assert "min_visible" in lines[4], lines[4]

    """参数里混进 None / 字符串 / 列表时：显示不可用，绝不抛异常。

    `iter_windows()` 现在把这些一律归成 ScopeError；这条用例钉的是
    「预览层面对坏输入不崩」这个承诺本身，而不依赖 iter_windows 具体抛哪种
    异常——它以后若改成抛 TypeError，这里就该红，逼我们回来收敛。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((500, 500))
for bad in ({'w': None, 'h': 100, 'overlap': 0},
            {'w': 'x', 'h': 100, 'overlap': 0},
            {'w': [1], 'h': 100, 'overlap': 0},
            {'w': 100, 'h': 100, 'overlap': [1]},
            {}, {'w': 100}):
    p.set_spec(bad)
    print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert all(ln == "None False" for ln in lines[:-1]), lines
    assert lines[-1] == "ok", lines[-1]


def test_real_preview_survives_malformed_spec_types():
    """参数里混进 None / 字符串 / 列表时：显示不可用，绝不抛异常。

    `iter_windows()` 现在把这些一律归成 ScopeError；这条用例钉的是
    「预览层面对坏输入不崩」这个承诺本身，而不依赖 iter_windows 具体抛哪种
    异常——它以后若改成抛 TypeError，这里就该红，逼我们回来收敛。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview

app = QApplication([])
p = WindowOverlayPreview()
p.set_image((500, 500))
for bad in ({'w': None, 'h': 100, 'overlap': 0},
            {'w': 'x', 'h': 100, 'overlap': 0},
            {'w': [1], 'h': 100, 'overlap': 0},
            {'w': 100, 'h': 100, 'overlap': [1]},
            {}, {'w': 100}):
    p.set_spec(bad)
    print(p.window_count(), p.is_valid())
p.repaint()
print('ok')
""")
    lines = _lines(out)
    assert all(ln == "None False" for ln in lines[:-1]), lines
    assert lines[-1] == "ok", lines[-1]
