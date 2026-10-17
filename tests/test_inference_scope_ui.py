"""推理范围 UI 的接线测试：面板、窗口弹窗、参数行入口、快照与配置持久化。

纯逻辑部分用假控件；真实控件行为（显隐、真实数值、预估文本）走
`_run_real_qt` 子进程，避免 conftest 的 mock PySide6 把结论带偏。
"""
import ast
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from labelforge.engine.inference_scope import (
    MODE_FULL, MODE_ROI, MODE_WINDOW, ScopeError,
)

_QT_EXIT_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    """在真实 Qt（offscreen）子进程里跑断言，返回 stdout。"""
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def _spec_dict(line):
    """把 `print(wp.spec())` 的输出解析回字典。

    窗口 spec 随着功能增加会多出键（比如 `min_visible`），所以断言「三个
    尺寸键对不对」时不该钉死整份字典的字符串，那样每加一个合法参数都要来
    改一批测试。
    """
    import ast
    value = ast.literal_eval(line)
    assert isinstance(value, dict), line
    return value


def _spec_size(line):
    """从 spec 的打印里取出 (w, h, overlap)，供只关心尺寸的断言使用。"""
    spec = _spec_dict(line)
    return (spec["w"], spec["h"], spec["overlap"])


# ---------- 参数行入口 ----------

def test_real_row_has_scope_button_disabled_until_model_loads():
    """真实控件：没加载模型时范围按钮禁用，加载后启用。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
print(row.scope_btn.isEnabled(), row.scope_btn.text())
row.set_model('m.onnx', 0.25, 0.45, (640, 640), ['CPUExecutionProvider'])
print(row.scope_btn.isEnabled(), row.scope_btn.text())
row.clear_model()
print(row.scope_btn.isEnabled(), row.scope_btn.text())
""")
    lines = _lines(out)
    assert lines[0] == "False 推理范围", lines[0]
    assert lines[1] == "True 推理范围", lines[1]
    assert lines[2] == "False 推理范围", lines[2]


def test_real_row_scope_summary_shows_mode_next_to_button():
    """真实控件：摘要写在按钮右侧的常驻 label 上，按钮文字不自带摘要。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_model('m.onnx', 0.25, 0.45, (640, 640), ['CPUExecutionProvider'])
for name in ('全图', '滑窗', '区域'):
    row.set_scope_summary(name)
    print(row.scope_btn.text(), '|', row.scope_summary_label.text())
""")
    lines = _lines(out)
    assert lines[0] == "推理范围 | 全图", lines[0]
    assert lines[1] == "推理范围 | 滑窗", lines[1]
    assert lines[2] == "推理范围 | 区域", lines[2]


def test_real_row_scope_summary_sits_right_of_button_in_same_row():
    """摘要 label 必须在**同一行**且紧贴按钮右侧（不能被 stretch 推到行尾）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.resize(1100, 40)
row.show()
app.processEvents()
btn_tl = row.scope_btn.mapTo(row, row.scope_btn.rect().topLeft())
lbl_tl = row.scope_summary_label.mapTo(row, row.scope_summary_label.rect().topLeft())
print(lbl_tl.x() >= btn_tl.x() + row.scope_btn.width())
# 同一行：纵向中心差不超过按钮高度的一半
btn_cy = btn_tl.y() + row.scope_btn.height() / 2
lbl_cy = lbl_tl.y() + row.scope_summary_label.height() / 2
print(abs(btn_cy - lbl_cy) <= row.scope_btn.height() / 2 + 1)
# 紧贴按钮：中间只隔布局间距，不能夹着一个 stretch。
# （只比「在关闭按钮左边」不够——stretch 在按钮与 label 之间时，
#   两者仍都在关闭按钮左边，得直接量按钮到 label 的间隙。）
gap = lbl_tl.x() - (btn_tl.x() + row.scope_btn.width())
print(gap >= 0 and gap <= 20)
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]


def test_real_row_scope_summary_label_is_visible_and_nonempty():
    """摘要标签可见且有文字（不能是个隐藏或空的占位）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.show()
app.processEvents()
print(row.scope_summary_label.isVisible())
print(bool(row.scope_summary_label.text()))
print(row.scope_summary_label.objectName())
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "modelRowScope", lines[2]


def test_real_row_scope_summary_resets_with_model():
    """真实控件：清掉模型后摘要回到全图。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_model('m.onnx', 0.25, 0.45, (640, 640), ['CPUExecutionProvider'])
row.set_scope_summary('区域')
row.clear_model()
print(row.scope_summary_label.text())
""")
    assert _lines(out)[0] == "全图", out


def test_real_row_scope_summary_ignores_detail_argument():
    """`detail` 保留但本轮不拼进摘要文字（模式名足够）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_scope_summary('滑窗', ' 640×640/200')
print(row.scope_summary_label.text())
# 参数应出现在 tooltip 里，仍可见但不占行
print('640×640' in row.scope_summary_label.toolTip())
""")
    lines = _lines(out)
    assert lines[0] == "滑窗", lines[0]
    assert lines[1] == "True", lines[1]


def test_real_row_scope_summary_treats_empty_name_as_full_image():
    """空名字（没模型/未初始化）也要落成「全图」，不能把摘要留空。

    摘要 label 是常驻控件，空字符串会让行上出现一块无意义的空白；
    `scope_summary_text()` 的既有调用方也期望拿到一个真模式名。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
# 刚构造出来、还没设过任何范围时也不该是空的
print(repr(row.scope_summary_text()))
row.set_scope_summary('区域')
row.set_scope_summary('')
print(repr(row.scope_summary_text()))
row.set_scope_summary(None)
print(repr(row.scope_summary_text()))
""")
    lines = _lines(out)
    assert lines[0] == "'全图'", lines[0]
    assert lines[1] == "'全图'", lines[1]
    assert lines[2] == "'全图'", lines[2]


def test_real_row_scope_summary_text_matches_label():
    """`scope_summary_text()` 读的必须是 label（它是既有调用方的取值口）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_scope_summary('区域')
print(row.scope_summary_text())
print(row.scope_summary_text() == row.scope_summary_label.text())
""")
    lines = _lines(out)
    assert lines[0] == "区域", lines[0]
    assert lines[1] == "True", lines[1]


def test_real_row_scope_button_emits_request():
    """真实控件：点范围按钮会发出请求信号。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
got = []
row.scope_requested.connect(lambda: got.append(1))
# 没加载模型时按钮是禁用的，点击不发信号
row.scope_btn.click()
print(len(got))
row.set_model('m.onnx', 0.25, 0.45, (640, 640), ['CPUExecutionProvider'])
row.scope_btn.click()
print(len(got))
""")
    lines = _lines(out)
    assert lines[0] == "0", lines[0]
    assert lines[1] == "1", lines[1]


# ---------- 窗口弹窗 ----------

def test_real_window_dialog_estimate_matches_plan_baseline():
    """真实控件：3840×2160、640×640、重叠 200 → 9×5 = 45 窗/图（计划 §3.4）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(3840, 2160))
print(d.spec())
print(d.window_count(), d.window_grid())
print(d.ok_btn.isEnabled())
print(d._estimate.text())
""")
    lines = _lines(out)
    assert _spec_size(lines[0]) == (640, 640, 200), lines[0]
    assert lines[1] == "45 (5, 9)", lines[1]
    assert lines[2] == "True", lines[2]
    assert "5 × 9" in lines[3] and "45" in lines[3], lines[3]


def test_real_window_dialog_disables_ok_when_count_explodes():
    """真实控件：窗口数超上限时禁用确定并说明，不能排出天文数字。

    20000² 图配 64 窗口、重叠 64 → 步长 1 → 3.9 亿窗，列表没建完就 OOM。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(20000, 20000))
d._w.setValue(64); d._h.setValue(64); d._ov.setValue(64)
print(d.window_count())
print(d.ok_btn.isEnabled())
print(d._estimate.text())
""")
    lines = _lines(out)
    assert lines[0] == "None", lines[0]
    assert lines[1] == "False", lines[1]
    assert "非法" in lines[2], lines[2]


def test_real_window_dialog_small_image_collapses_to_one_window():
    """真实控件：图比窗口小 → 自动缩为整图，只有 1 个窗口。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(200, 100))
print(d.window_count(), d.window_grid())
""")
    assert _lines(out)[0] == "1 (1, 1)", _lines(out)[0]


def test_real_window_dialog_survives_broken_saved_spec():
    """真实控件：配置里存了坏窗口参数也不能崩，回退到默认值。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
for bad in ({'w': 'abc', 'h': None, 'overlap': 'x'},
            {'w': 0, 'h': -5, 'overlap': -1},
            {'w': 99999, 'h': 1, 'overlap': 10**9}):
    d = SlidingWindowDialog(reference_size=(1000, 1000), spec=bad)
    print(d.spec())
""")
    lines = _lines(out)
    assert _spec_size(lines[0]) == (640, 640, 200), lines[0]
    assert _spec_size(lines[1]) == (8, 8, 0), lines[1]
    assert _spec_size(lines[2]) == (8192, 8, 8192), lines[2]


def test_real_window_dialog_roundtrips_spec_through_engine():
    """真实控件：弹窗给的 spec 能被引擎直接接受并排出同样数量的窗口。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog
from labelforge.engine.inference_scope import iter_windows, normalize_scope

app = QApplication([])
d = SlidingWindowDialog(reference_size=(3840, 2160))
spec = d.spec()
print(len(iter_windows(3840, 2160, spec)))
ns = normalize_scope({'mode': 'window', 'windows': spec}, (3840, 2160))
print(len(iter_windows(3840, 2160, ns['windows'])))
""")
    lines = _lines(out)
    assert lines[0] == "45" and lines[1] == "45", lines


# ---------- 范围面板 ----------

def test_real_scope_panel_warns_when_region_mode_has_no_regions():
    """真实控件：选区域但没画区域时给出警告（不再有按钮可禁用）。

    否则会「成功但 0 框」，默认覆盖模式会因此删掉原有自动标注。面板现在是
    实时的、没有确定按钮，所以这条约束的落点变成 `warn_label`：宿主启动前
    仍会拦，但用户必须先在面板上看到为什么。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=0,
               reference_size=(3840, 2160))
# 全图模式不看区域，没有警告
print(p.mode() == MODE_FULL, repr(p.warn_label.text()))
p.set_mode(MODE_ROI)
print('没有可用区域，请先画好区域' in p.warn_label.text())
p.set_region_count(2)
print('没有可用区域，请先画好区域' in p.warn_label.text())
print(p.mode() == MODE_ROI)
""")
    lines = _lines(out)
    assert lines[0] == "True ''", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "False", lines[2]
    assert lines[3] == "True", lines[3]


def test_real_scope_panel_shows_controls_per_mode():
    """真实控件：三段的控件**始终显示**，能不能改由选中哪段决定。

    用户要求「滑窗参数打开窗口就要存在」+「还在一个窗口，补全内容，分割一下」
    ——参数区与区域明细都不再随模式显隐，一打开面板就能看到三段各自的内容
    （当前窗口尺寸/模板、区域条数与明细）。可编辑性仍由模式决定。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
p.show()
app.processEvents()
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    p.set_mode(mode)
    app.processEvents()
    # 用 isVisible()（会连带看父控件）而不是 isHidden()：子控件没有单独
    # 隐藏过时 isHidden() 恒为 False，断言会变成空转。
    print(mode, p.window_params.isVisible(), p.region_label.isVisible(),
          p.region_list.isVisible())
# 面板本身必须是非模态的：否则弹窗开着时画布收不到鼠标事件，「就地画区域」失效
print(p.isModal())
""")
    lines = _lines(out)
    # 三段的控件三种模式下都可见（「全部显示各自内容」）
    assert lines[0] == "full True True True", lines[0]
    assert lines[1] == "window True True True", lines[1]
    assert lines[2] == "roi True True True", lines[2]
    assert lines[3] == "False", lines[3]


def test_real_scope_params_are_disabled_unless_window_mode_selected():
    """只有选中「滑窗」时宽/高/重叠与可见比例才**可点**，否则禁用。

    用户要求「未选中滑窗时不能修改点击」。只隐藏是不够的：隐藏的控件
    在某些布局路径下仍会被键盘焦点/程序化调用改到，而且用户看不到
    当前值就没法核对「这次要跑多大窗口」。所以分别断言 disabled（可见
    或不可见都好，但一定不能可用）。

    模板那一排已经删掉了（用户要求「推理滑窗不要模板了」），所以这里只
    钉输入框。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
# 数据增强侧即使有模板，推理面板也不该再显示出模板按钮
from labelforge.core import config_manager as cm
cfg = cm.load_config()
cfg['crop_templates'] = [{'w': 512, 'h': 512, 'overlap': 64,
                          'min_visible': 0.7, 'square': False,
                          'fill': False, 'fill_rgb': [0, 0, 0]}]
cm.save_config(cfg)

p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
p.show()
app.processEvents()
print(p.window_params.spec()['w'])
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    p.set_mode(mode)
    app.processEvents()
    wp = p.window_params
    # 区域那一段没有「编辑区域」按钮了，可编辑性看它的区域提示 label
    print(mode, wp.spins_enabled(), p.region_label.isEnabled())
# 禁用状态下用户**点不动**：这是「不能修改点击」的真实保证。
# 不断言 setValue() 会失败——那是程序化写入，Qt 有意让它绕过 enabled，
# 拿它当断言是量的东西与结论不同构。
p.set_mode(MODE_ROI)
app.processEvents()
print(wp.is_editable(), wp.spins_enabled(), wp._w.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "640", lines[0]
    # 全图：滑窗参数禁用、区域那段也置灰
    assert lines[1] == "full False False", lines[1]
    # 滑窗：参数可用、区域那段置灰
    assert lines[2] == "window True False", lines[2]
    # 区域：参数禁用、区域那段可用
    assert lines[3] == "roi False True", lines[3]
    assert lines[4] == "False False False", lines[4]


def test_real_scope_panel_never_shows_a_thumbnail_preview():
    """面板里**不再有**缩略预览小图：网格/区域直接画在主画布上。

    用户要求「直接在画布展示，不要单独小窗显示全图」。所以面板里不该再有
    `WindowOverlayPreview` 这类控件——留着就会又出现一个「单独小窗显示全图」。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160))
p.set_mode(MODE_WINDOW)
p.show()
app.processEvents()
leftovers = p.findChildren(WindowOverlayPreview)
print(len(leftovers))
# 参数控件里也不该有
print(len(p.window_params.findChildren(WindowOverlayPreview)))
""")
    lines = _lines(out)
    assert lines[0] == "0", lines[0]
    assert lines[1] == "0", lines[1]


def test_real_scope_panel_top_aligns_to_the_label_group():
    """面板顶边对齐「标签管理」区顶边；放不下时被钳制上移，但绝不越过锚点。

    契约来自 `_scope_anchor_rect`（用户明确要求「范围推理窗口左上角始终对齐
    标签管理区域的左上角」）。此处断言它，并钉住下面这条不变量：屏幕上装不下
    整个面板时，面板整体上移，**不会被裁掉**。

    这条测试原来断言「面板竖直中心落在两个列表之间」——那是 35f9c3c 改用左上角
    对齐**之前**的契约，之后没跟着更新，只因 PyQt5 离屏屏幕只有 600 高、把面板
    从 567 压到 433 才碰巧成立。离屏屏幕尺寸在两代 Qt 下不同（Qt5 800x600、
    Qt6 800x800），这个巧合随之消失，于是暴露出来。此处改为直接断言真实契约，
    两个分支都不依赖屏幕尺寸。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1400, 900)
w.show()
app.processEvents()

# 让侧栏列表有真实几何
w._detector = type("D", (), {"model_path": "m.onnx", "conf_threshold": .25,
                             "iou_threshold": .45, "class_names": []})()
w._open_scope_panel()
app.processEvents()
panel = w._scope_panel
bg = w.bg_list_group
lb = w.label_group
bg_top = bg.mapTo(w, bg.rect().topLeft()).y()
bg_bottom = bg_top + bg.height()
lb_top = lb.mapTo(w, lb.rect().topLeft()).y()
p_top = panel.mapTo(w, panel.rect().topLeft()).y()
p_bottom = p_top + panel.height()
screen_bottom = app.primaryScreen().availableGeometry().bottom()
print('bg', bg_top, bg_bottom)
print('label_top', lb_top)
print('panel_top', p_top)
print('panel_bottom', p_bottom)
print('screen_bottom', screen_bottom)
panel.close()
app.processEvents()
""")
    lines = _lines(out)
    bg_top, bg_bottom = (int(v) for v in lines[0].split()[1:])
    lb_top = int(lines[1].split()[1])
    p_top = int(lines[2].split()[1])
    p_bottom = int(lines[3].split()[1])
    screen_bottom = int(lines[4].split()[1])

    # 量的是「确实有列表」而不是退化成 0 高度
    assert bg_bottom > bg_top, lines[0]
    assert lb_top > bg_top, (bg_top, lb_top)

    # 顶边不会落到锚点下方（对齐，或被钳制到更上面）
    assert p_top <= lb_top, f"面板顶边低于标签区顶边：{p_top} > {lb_top}"
    # 放得下就必须精确对齐
    panel_h = p_bottom - p_top
    if lb_top + panel_h <= screen_bottom:
        assert p_top == lb_top, (
            f"屏幕放得下时面板顶边应与标签区顶边对齐：{p_top} != {lb_top}")
    # 装不下时上移，但不能被裁掉
    assert p_bottom <= screen_bottom, (
        f"面板底边超出屏幕（被裁掉）：{p_bottom} > {screen_bottom}")


def test_place_within_top_left_aligns_vertically_and_clamps_at_the_bottom():
    """`place_within(align="top-left")` 的两个分支，都不依赖离屏屏幕尺寸。

    上面那条面板级测试在离屏屏幕（800x800）上**永远走不到**「精确对齐」分支：
    面板 511 高、锚点在 ~399，必然被钳到屏幕上移。这里直接喂一个装得下的
    锚点，把「顶边精确对齐锚点顶边」和「装不下时上移但不被裁掉」都钉住。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QWidget, QDialog
from PySide6.QtCore import QRect
from labelforge.ui.dialog_helpers import place_within

app = QApplication([])
screen = app.primaryScreen().availableGeometry()

parent = QWidget()
parent.resize(600, 600)
parent.move(20, 20)
parent.show()
app.processEvents()
origin = parent.mapToGlobal(parent.rect().topLeft())

# ① 装得下：顶边精确对齐锚点顶边
fits_rect = QRect(50, 20, 200, 100)
d1 = QDialog(parent)
d1.resize(120, 80)
place_within(d1, fits_rect, align='top-left')
app.processEvents()
print('FIT_Y', d1.y(), origin.y() + fits_rect.y())
print('FIT_IN_SCREEN', d1.frameGeometry().bottom() <= screen.bottom())

# ② 装不下：整体上移，但底边仍在屏幕内
tall = QDialog(parent)
tall.resize(120, screen.height() - 40)
tall.show()
app.processEvents()
low_rect = QRect(50, screen.height() - 60, 200, 100)
place_within(tall, low_rect, align='top-left')
app.processEvents()
print('CLAMP_BOTTOM_IN_SCREEN',
      tall.frameGeometry().bottom() <= screen.bottom())
print('CLAMP_TOP_ABOVE_ANCHOR', tall.y() <= origin.y() + low_rect.y())
d1.close()
tall.close()
""")
    lines = _lines(out)
    _, got_y, want_y = lines[0].split()
    assert got_y == want_y, (
        f"装得下时顶边没有精确对齐锚点顶边：{got_y} != {want_y}")
    assert lines[1] == "FIT_IN_SCREEN True", lines[1]
    assert lines[2] == "CLAMP_BOTTOM_IN_SCREEN True", lines[2]
    assert lines[3] == "CLAMP_TOP_ABOVE_ANCHOR True", lines[3]


def test_real_scope_panel_embeds_window_params_without_second_dialog():
    """真实控件：滑窗参数就在面板里，点它**不会**再开第二个窗口。

    这是用户报的问题（「点击滑窗怎么多了一个弹窗」）：面板里必须有真实可用的
    宽/高/重叠输入框，而不是一个再弹窗的按钮。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QSpinBox, QPushButton
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160))
p.set_mode(MODE_WINDOW)
p.show()
app.processEvents()
wp = p.window_params
# 三个输入框可见、可编辑，且值就是传入的 spec
spins = [s for s in wp.findChildren(QSpinBox) if s.isVisible()]
print(len(spins))
print(wp.spec())
print(all(s.isEnabled() for s in spins))
# 面板里不该再有一个「滑窗参数」按钮（那是旧设计：点它再开窗）
old_btn = [b for b in p.findChildren(QPushButton) if "滑窗参数" in b.text()]
print(len(old_btn))
""")
    lines = _lines(out)
    assert lines[0] == "4", f"窗口参数现在有四个输入框（含最低可见比例）：{lines[0]}"
    assert _spec_size(lines[1]) == (640, 640, 200), lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "0", lines[3]


def test_real_scope_panel_inline_params_track_edits():
    """真实控件：**面板内**改参数，窗口数立刻跟着变（不需要开第二个窗）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160))
p.set_mode(MODE_WINDOW)
p.show()
app.processEvents()
print(p.window_params.window_count())
p.window_params._w.setValue(1280); p.window_params._h.setValue(1280)
p.window_params._ov.setValue(0)
app.processEvents()
# 3840×2160 配 1280×1280、重叠 0 → 3 × 2 = 6 窗
print(p.window_params.window_count())
print(p.selection())
""")
    lines = _lines(out)
    assert lines[0] == "45", lines[0]
    assert lines[1] == "6", lines[1]
    mode, spec = ast.literal_eval(lines[2])
    assert mode == "window", lines[2]
    assert (spec["w"], spec["h"], spec["overlap"]) == (1280, 1280, 0), lines[2]


def test_real_scope_panel_asks_the_canvas_to_redraw_on_change():
    """改参数要发 `overlay_changed`：主画布上的网格靠这条信号重画。

    网格已经不在面板里画了，所以「参数变了但信号没发」的后果是**画面不更新**
    （用户改完宽高，画布上的格子还是旧的），而不是某个控件没刷新。这条钉住
    信号真的会发，且切模式也会发。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160))
signals = []
p.overlay_changed.connect(lambda: signals.append(len(signals)))
p.set_mode(MODE_WINDOW)
app.processEvents()
after_mode = len(signals)
p.window_params._w.setValue(1280)
app.processEvents()
after_param = len(signals)
p.set_mode(MODE_ROI)
app.processEvents()
after_switch = len(signals)
print(after_mode > 0, after_param > after_mode, after_switch > after_param)
# 面板里不该有任何缩略预览控件（已在画布上画）
from labelforge.ui.widgets.window_overlay_preview import WindowOverlayPreview
print(len(p.findChildren(WindowOverlayPreview)))
""")
    lines = _lines(out)
    assert lines[0] == "True True True", lines[0]
    assert lines[1] == "0", lines[1]


def test_real_canvas_draws_scope_grid_when_window_mode_selected():
    """主画布真的画出了滑窗网格，且只有选中滑窗时才画。

    这是用户要求的落点：「滑窗…直接在画布展示，不要单独小窗显示全图」。
    画布把网格画到一张图上，数「青色调」像素（网格色是 #00D4FF）。

    不能按精确 RGB 比：网格是半透明叠在背景图上的，实际像素是混合色
    （实测背景 40,70,100 + 网格 → #0fa0c6 之类），拿纯色比会假红。
    按「蓝 > 绿 > 红」这个青色签名数，再要求「区域模式下这些像素消失」
    ——后者才是「滑窗与区域各自的内容分开显示」的证据。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW, MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()

img = QImage(1600, 1200, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
app.processEvents()

w._detector = type("D", (), {"model_path": "m.onnx", "conf_threshold": .25,
                             "iou_threshold": .45, "class_names": []})()
# 1600×1200 配 800×600、重叠 0 → 2×2 = 4 窗，缩放后格子仍够大
w._scope_window_spec = {'w': 800, 'h': 600, 'overlap': 0}
w._scope_mode = MODE_WINDOW
w._scope_panel_open = True

canvas = w.canvas
canvas.resize(1000, 700)
app.processEvents()

def count_grid():
    im = QImage(canvas.size(), QImage.Format_ARGB32)
    im.fill(QColor('white'))
    pnt = QPainter(im)
    canvas.render(pnt, QPoint())
    pnt.end()
    n = 0
    for y in range(im.height()):
        for x in range(im.width()):
            c = im.pixelColor(x, y)
            # 青色签名：蓝最大、红最小（网格 #00D4FF 叠在深蓝底上）
            if c.blue() > 150 and c.blue() > c.green() > c.red():
                n += 1
    return n

win_px = count_grid()
w._scope_mode = MODE_ROI
roi_px = count_grid()
w._scope_mode = MODE_WINDOW
w._scope_panel_open = False
off_px = count_grid()

print(win_px)
print(roi_px)
print(off_px)
print(win_px > roi_px and win_px > off_px)
""")
    lines = _lines(out)
    win_px = int(lines[0])
    roi_px = int(lines[1])
    off_px = int(lines[2])
    assert win_px > 100, f"滑窗模式没画出网格（青色像素 {win_px}）"
    assert roi_px < 20, f"区域模式仍画了网格（{roi_px}）"
    assert off_px < 20, f"面板关掉仍画了网格（{off_px}）"
    assert lines[3] == "True", lines[3]


def test_real_canvas_scope_grid_follows_the_parameters():
    """改窗口参数，画布上的网格跟着变（不是画一次就定格）。

    这条防的是「信号发了但画布用的是快照」：宿主要在收到 `overlay_changed`
    时把新参数取过去，否则用户改完宽高，画面还是旧格子。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()
img = QImage(1600, 1200, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w._detector = type("D", (), {"model_path": "m.onnx", "conf_threshold": .25,
                             "iou_threshold": .45, "class_names": []})()
app.processEvents()

canvas = w.canvas
canvas.resize(1000, 700)
app.processEvents()
from PySide6.QtWidgets import QSpinBox

w._open_scope_panel()
app.processEvents()
panel = w._scope_panel
panel.set_mode(MODE_WINDOW)
app.processEvents()

def count_grid():
    im = QImage(canvas.size(), QImage.Format_ARGB32)
    im.fill(QColor('white'))
    pnt = QPainter(im)
    canvas.render(pnt, QPoint())
    pnt.end()
    return sum(1 for y in range(im.height()) for x in range(im.width())
               if im.pixelColor(x, y).blue() > 150
               and im.pixelColor(x, y).blue() > im.pixelColor(x, y).green()
               > im.pixelColor(x, y).red())

few = count_grid()
# 窗口变小 → 格子变多 → 网格像素显著增加
panel.window_params._w.setValue(400)
panel.window_params._h.setValue(300)
app.processEvents()
many = count_grid()
print(few > 0)
print(many > few)
print(w._scope_window_spec)
panel.close()
app.processEvents()
""")
    lines = _lines(out)
    assert lines[0] == "True", lines
    assert lines[1] == "True", f"改参数后网格没变: {lines}"
    assert _spec_size(lines[2]) == (400, 300, 200), lines[2]


def test_real_scope_panel_selection_reports_windows_only_when_needed():
    """真实控件：全图/区域不带窗口参数，滑窗类带。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2)
for mode in (MODE_FULL, MODE_ROI, MODE_WINDOW):
    p.set_mode(mode)
    print(p.selection())
""")
    lines = _lines(out)
    assert lines[0] == "('full', None)", lines[0]
    assert lines[1] == "('roi', None)", lines[1]
    mode, spec = ast.literal_eval(lines[2])
    assert mode == "window", lines[2]
    assert (spec["w"], spec["h"], spec["overlap"]) == (640, 640, 200), lines[2]


def test_real_scope_panel_window_summary_text():
    """真实控件：窗口摘要带上尺寸与重叠；没传参数时用控件默认值。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 512, 'overlap': 100})
print(repr(p.window_summary()))
# 参数内嵌后控件总有值（默认 640×640 / 重叠 200），所以摘要不会是空串——
# 这正是「滑窗模式直接就有参数可调」的体现。
print(repr(ScopePanel().window_summary()))
""")
    lines = _lines(out)
    assert lines[0] == "'640×512 重叠 100'", lines[0]
    assert lines[1] == "'640×640 重叠 200'", lines[1]


def test_real_scope_panel_roi_mode_offers_the_add_region_button():
    """真实控件：「添加区域」按钮取代了自动进入编辑态。

    用户要求「画区域也加一个按钮啊 点一下画一个」——所以面板这侧的契约变成：
    区域段有个按钮，点它发 `add_region_requested`；画布上就不再自动进入绘制
    态（会独占左键，标签框点不到）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(region_count=1)
fired = []
p.add_region_requested.connect(lambda: fired.append(1))

# 旧的「编辑区域」按钮与它的信号必须都不在了
print(hasattr(p, 'edit_regions_requested'), hasattr(p, 'region_btn'))

p.set_mode(MODE_FULL)
print(p.add_region_btn.isEnabled())
p.set_mode(MODE_WINDOW)
print(p.add_region_btn.isEnabled())
p.set_mode(MODE_ROI)
print(p.add_region_btn.isEnabled())
p.add_region_btn.click()
print(len(fired))
print(p.region_label.text())
""")
    lines = _lines(out)
    assert lines[0] == "False False", (
        f"旧的「编辑区域」按钮/信号没拆干净：{lines[0]}")
    # 只有选中「区域」时那段内容体才可编辑（其余置灰）
    assert lines[1] == "False", f"全图下按钮不该可点：{lines[1]}"
    assert lines[2] == "False", f"滑窗下按钮不该可点：{lines[2]}"
    assert lines[3] == "True", f"区域下按钮应当可点：{lines[3]}"
    assert lines[4] == "1", f"点按钮没发出 add_region_requested：{lines[4]}"
    assert lines[5] == "1 个区域", (
        f"区域计数没显示对（且不该再挂「在图上拖动」提示）：{lines[5]}")


def test_real_scope_panel_retranslates():
    """真实控件：切语言后面板上的文字跟着变。

    确定/取消/编辑区域三个按钮都没了，所以钉的是**当前还在屏上的那几处**
    文案：模式单选按钮、段标题、区域计数与「添加区域」按钮。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.ui import i18n
from labelforge.engine.inference_scope import MODE_FULL, MODE_ROI

app = QApplication([])
p = ScopePanel(region_count=1)
p.set_mode(MODE_ROI)
print(p._buttons[MODE_FULL].text(), p.sections[MODE_ROI].title(),
      p.region_label.text())
print(p.add_region_btn.text())
i18n.set_lang('en')
p.retranslate()
print(p._buttons[MODE_FULL].text(), p.sections[MODE_ROI].title(),
      p.region_label.text())
print(p.add_region_btn.text())
i18n.set_lang('zh')
""")
    lines = _lines(out)
    assert lines[0] == "全图 区域 1 个区域", lines[0]
    assert lines[1] == "添加区域", lines[1]
    assert lines[2] == "Full image Regions 1 regions", lines[2]
    assert lines[3] == "Add Region", lines[3]


def test_real_scope_panel_region_row_has_lock_button_in_both_languages():
    """区域列表每行都有锁定按钮，文案跟随语言与当前锁状态。

    用户要求「区域列表增加锁定按钮」。按钮显示的是**下一步动作**：未锁时写
    「锁定」，已锁时写「解锁」，切语言两边都要跟着变。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.ui import i18n
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
p = ScopePanel()
p.set_mode(MODE_ROI)
p.set_regions([
    {'shape_type': 'rectangle', 'x': 10, 'y': 10, 'width': 40, 'height': 40},
    {'shape_type': 'rectangle', 'x': 50, 'y': 50, 'width': 40, 'height': 40,
     'locked': True},
])
print([b.text() for b in p.findChildren(type(p.add_region_btn))
       if b.objectName() == 'scopeRegionLock'])
i18n.set_lang('en')
p.retranslate()
print([b.text() for b in p.findChildren(type(p.add_region_btn))
       if b.objectName() == 'scopeRegionLock'])
i18n.set_lang('zh')
""")
    lines = _lines(out)
    assert lines[0] == "['锁定', '解锁']", (
        f"锁定按钮文案没反映当前状态：{lines[0]}")
    assert lines[1] == "['Lock', 'Unlock']", (
        f"锁定按钮没跟着切语言：{lines[1]}")


def test_real_scope_panel_lock_button_emits_index_and_target_state():
    """点某行的锁按钮 → 发出 (该行下标, 目标状态)，不是别的行、也不是空信号。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
p = ScopePanel()
p.set_mode(MODE_ROI)
p.set_regions([
    {'shape_type': 'rectangle', 'x': 10, 'y': 10, 'width': 40, 'height': 40},
    {'shape_type': 'rectangle', 'x': 50, 'y': 50, 'width': 40, 'height': 40},
])
fired = []
p.region_lock_toggled.connect(lambda i, v: fired.append((i, v)))
btns = [b for b in p.findChildren(type(p.add_region_btn))
        if b.objectName() == 'scopeRegionLock']
btns[1].click()
print(fired)
""")
    lines = _lines(out)
    assert lines[0] == "[(1, True)]", (
        f"锁按钮没把「第几条、锁成什么」报对：{lines[0]}")


# ---------- mixin 快照 ----------

def _scope_host(mode=MODE_FULL, regions=None, size=(200, 100), windows=None,
                saved_windows=None):
    """最小宿主：只替换 `_scope_for_run` 依赖的画布/配置读取。

    `_reference_image_size`、`_region_boxes`、`_saved_window_spec` 都由
    `InferenceMixin` 自己定义，子类里覆盖它们即可（不能靠 MRO 让另一个基类
    的方法胜出——mixin 在左边，永远先命中）。
    """
    from labelforge.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        def __init__(self):
            self._scope_mode = mode
            self.region_boxes = list(regions or [])
            self._window_spec = windows

        def _reference_image_size(self):
            return size

        def _saved_window_spec(self):
            return dict(saved_windows or {"w": 640, "h": 640, "overlap": 200})

    return _Host()


def test_scope_for_run_full_mode_returns_none():
    """全图模式回 None，让引擎走原来的单次 detect 路径。"""
    host = _scope_host(mode=MODE_FULL)
    assert host._scope_for_run() is None


def test_scope_for_run_window_mode_carries_windows():
    host = _scope_host(mode=MODE_WINDOW, windows={"w": 512, "h": 512,
                                                  "overlap": 64})
    scope = host._scope_for_run()
    assert scope["mode"] == MODE_WINDOW
    assert scope["windows"] == {"w": 512, "h": 512, "overlap": 64}


def test_scope_for_run_window_mode_falls_back_to_saved_spec():
    """没显式设过窗口参数时用配置里存的。"""
    host = _scope_host(mode=MODE_WINDOW, windows=None,
                       saved_windows={"w": 320, "h": 320, "overlap": 16})
    assert host._scope_for_run()["windows"] == {"w": 320, "h": 320,
                                                "overlap": 16}


def test_effective_window_spec_falls_back_to_defaults():
    """首次使用（没设过也没存过）要回默认窗口参数，不能回空表。

    回空表时面板会带着「没有窗口参数」进滑窗模式，确定后启动任务才报错。
    """
    host = _scope_host(mode=MODE_WINDOW, windows=None, saved_windows={})
    spec = host._effective_window_spec()
    assert spec == {"w": 640, "h": 640, "overlap": 200}, spec
    assert host._scope_for_run()["windows"] == spec


def test_effective_window_spec_prefers_current_session_value():
    """本次会话设过的优先于配置里存的。"""
    host = _scope_host(mode=MODE_WINDOW, windows={"w": 256, "h": 256,
                                                  "overlap": 32},
                       saved_windows={"w": 640, "h": 640, "overlap": 200})
    assert host._effective_window_spec() == {"w": 256, "h": 256, "overlap": 32}


def test_effective_window_spec_uses_saved_when_session_unset():
    """没设过就用配置里存的。"""
    host = _scope_host(mode=MODE_WINDOW, windows=None,
                       saved_windows={"w": 512, "h": 512, "overlap": 64})
    assert host._effective_window_spec() == {"w": 512, "h": 512, "overlap": 64}


def test_scope_for_run_roi_mode_snapshots_regions_and_reference():
    """区域模式带上区域快照与参考图尺寸。"""
    regions = [{"x": 10, "y": 10, "width": 50, "height": 50}]
    host = _scope_host(mode=MODE_ROI, regions=regions, size=(200, 100))

    scope = host._scope_for_run()

    assert scope["mode"] == MODE_ROI
    assert scope["regions"] == regions
    assert scope["reference_size"] == (200, 100)
    assert scope["regions"] is not regions, "必须是快照，不能是同一个列表"


def test_scope_for_run_roi_snapshot_ignores_later_edits():
    """任务启动后再改画布区域，不能影响已定稿的 scope。"""
    regions = [{"x": 10, "y": 10, "width": 50, "height": 50}]
    host = _scope_host(mode=MODE_ROI, regions=regions, size=(200, 100))

    scope = host._scope_for_run()
    regions[0]["x"] = 9999
    regions.append({"x": 0, "y": 0, "width": 5, "height": 5})

    assert scope["regions"][0]["x"] == 10
    assert len(scope["regions"]) == 1


def test_scope_for_run_roi_without_regions_raises():
    """选了区域却没有区域 → 启动前就报错，不留给后台变成 0 框。"""
    host = _scope_host(mode=MODE_ROI, regions=[])
    with pytest.raises(ScopeError):
        host._scope_for_run()


def test_scope_for_run_rejects_bad_window_params_at_start():
    """坏窗口参数在启动时报错，不能进后台。"""
    for bad in ({"w": 0, "h": 640, "overlap": 0},
                {"w": -5, "h": 640, "overlap": 0},
                {"w": 640, "h": 640, "overlap": -1},
                {"w": "x", "h": 640, "overlap": 0}):
        host = _scope_host(mode=MODE_WINDOW, windows=bad)
        with pytest.raises(ScopeError):
            host._scope_for_run()


def test_scope_for_run_without_reference_size_skips_the_check():
    """拿不到参考图尺寸时不带 reference_size（不做尺寸校验）。"""
    host = _scope_host(mode=MODE_ROI,
                       regions=[{"x": 0, "y": 0, "width": 50, "height": 50}],
                       size=None)
    scope = host._scope_for_run()
    assert "reference_size" not in scope


def test_region_boxes_tolerates_missing_attribute():
    """宿主没有 region_boxes 时回空表，不抛 AttributeError。"""
    from labelforge.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._region_boxes() == []


def test_reference_image_size_tolerates_missing_image():
    """没有当前图时返回 None，不崩。"""
    from labelforge.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._reference_image_size() is None


def test_scope_mode_defaults_to_full_when_unset():
    """从未设置过范围时视为全图（保持既有行为）。"""
    from labelforge.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._scope_for_run() is None

# ---------- 主窗口接线（真实 Qt） ----------

_MIXIN_HOST = """
import os
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QImage
app = QApplication([])
from labelforge.ui.mixins.inference import InferenceMixin
# 圆点刷新（`_refresh_background_item_status`）在列表 mixin 里；范围刷新用例要
# 走真实实现，别在替身里重写一遍「刷圆点」的语义。
from labelforge.ui.mixins.background_list import BackgroundListMixin as _BackgroundListMixin
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI


class _Lbl:
    def __init__(self):
        self.text_ = ""

    def setText(self, text):
        self.text_ = text


class _Detector:
    model_path = "m.onnx"
    conf_threshold = 0.25
    iou_threshold = 0.45
    active_providers = ["CPUExecutionProvider"]
    class_names = ["person", "car"]

    def resolve_format(self):
        return "v8"

    def is_dynamic_input(self):
        return False


class _FakeCanvas:
    # 范围接线用的画布替身：只暴露「画一个区域」与「逐条锁定」这两个入口。
    # 真实交互（一次性绘制、层级命中）由 tests/test_inference_region_session.py
    # 和 tests/test_inference_scope_ui.py 里的真 Canvas 用例钉；
    # 这里只验 mixin 有没有把信号正确落到画布上。

    def __init__(self):
        self.is_drawing_region = False
        self.add_count = 0
        self.set_locked_calls = []

    def _start_add_region(self):
        self.is_drawing_region = True
        self.add_count += 1

    def _cancel_region_drawing(self):
        self.is_drawing_region = False

    def set_region_locked(self, index, locked):
        regions = getattr(self._editor_ref, 'region_boxes', None) or []
        if not (0 <= index < len(regions)):
            return False
        regions[index]['locked'] = bool(locked)
        self.set_locked_calls.append((index, bool(locked)))
        return True

    def update(self):
        pass


class _Host(InferenceMixin, _BackgroundListMixin, QWidget):
    def __init__(self):
        super().__init__()
        self.status_label = _Lbl()
        self.region_boxes = []
        self.background_images = ["a.jpg"]
        self.current_background_index = 0
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        self.canvas_items = []
        self._inference_dataset = ("a.jpg",)
        self._busy = False
        self._detector = None
        self._model_worker = None
        self._run_worker = None
        self._is_delete_view = False
        self._canvas_layout = None
        self.current_background = None
        self.switched_to = []
        self.canvas = _FakeCanvas()
        # 替身要能拿到宿主真身去改 `region_boxes`（与真裁布一个口径）
        self.canvas._editor_ref = self
        # 圆点刷新要向 `background_list` 写 BG_ROLE_STATUS，`_sync_current_boxes_from_cache`
        # 还要 `update_label_list`。给真控件（QListWidget）而不是替身：状态图标是
        # 通过 `setData/setIcon` 落到 item 上的，替身会把断言变成「测替身」。
        from PySide6.QtWidgets import QListWidget
        self.background_list = QListWidget()
        self.update_label_list = lambda: None
        self.save_undo_state = lambda: None
        # 范围面板定位要用：侧栏几何块 + 两个「列表」。锚点按两个列表的
        # mapTo/width/height 算，所以这里放的是带几何信息的**替身**（不用真
        # QWidget：在子进程退出阶段销毁宿主子控件会踩访问违例，实测崩）。
        class _Rect:
            def __init__(self, x, y, w, h):
                self._r = (x, y, w, h)

            def rect(self):
                from PySide6.QtCore import QRect
                return QRect(0, 0, self._r[2], self._r[3])

            def mapTo(self, parent, point):
                from PySide6.QtCore import QPoint
                return QPoint(self._r[0] + point.x(), self._r[1] + point.y())

            def width(self):
                return self._r[2]

            def height(self):
                return self._r[3]

        # 侧栏放在屏幕左侧一块，并且让面板宽度塞得进 offscreen 的 800×600 屏幕。
        # 面板最小宽度 440；若侧栏放 x=700，`place_within` 会把面板拉回屏幕内，
        # 位置断言就变成「测屏幕钳制」而不是「测锚点」了（实测 frame.x()=263）。
        #
        # 侧栏故意给得**够宽**（760）：锚点窄的时候面板居中位置会被
        # `if gx < left: gx = left` 兜到左边界，此时「把面板左移」这类变异
        # 看不出差别（实测变异体存活）。
        self.bg_list_group = _Rect(20, 40, 760, 300)
        self.label_group = _Rect(20, 350, 760, 300)

    def _side_panel_geometry(self):
        from PySide6.QtCore import QRect
        return QRect(20, 40, 760, 610)

    def _set_edit_mode(self, mode, animated=False):
        self.switched_to.append((mode, animated))
"""


def test_real_mixin_wiring_end_to_end():
    """真实控件：加载模型 → 范围按钮可用 → 切模式 → 摘要与快照跟着变。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._open_scope_panel()
print(h.status_label.text_)
h._on_model_loaded(_Detector(), (640, 640), "model")
print(h.model_row.scope_btn.isEnabled(), h.model_row.scope_btn.text())

img = QImage(400, 300, QImage.Format_RGB888)
img.fill(0)
h.current_background = img
print(h._reference_image_size())

h._scope_mode = MODE_WINDOW
h._window_spec = {'w': 100, 'h': 100, 'overlap': 50}
h._refresh_scope_button()
# 摘要读按钮右侧的常驻 label；按钮文字恒为「推理范围」
print(h.model_row.scope_summary_label.text())
print(h._scope_for_run())

h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
h._scope_mode = MODE_ROI
h._refresh_scope_button()
print(h.model_row.scope_summary_label.text())
print(h._scope_for_run())

h._scope_mode = MODE_FULL
h._refresh_scope_button()
print(h.model_row.scope_summary_label.text(), h._scope_for_run())
""")
    lines = _lines(out)
    assert lines[0] == "请先加载模型", lines[0]
    assert lines[1] == "True 推理范围", lines[1]
    assert lines[2] == "(400, 300)", lines[2]
    assert lines[3] == "滑窗", lines[3]
    assert lines[4] == "{'mode': 'window', 'windows': {'w': 100, 'h': 100, 'overlap': 50}}", lines[4]
    assert lines[5] == "区域", lines[5]
    assert "'reference_size': (400, 300)" in lines[6], lines[6]
    assert lines[7] == "全图 None", lines[7]


def test_real_mixin_open_scope_panel_constructs_non_modal_panel():
    """真实控件：加载模型后点「推理范围」要真的把面板建起来，且是**非模态**。

    踩过的坑：`ScopePanel(self)` 把 editor 当成第一个位置参数 `windows`，
    运行时报 `TypeError: 'ImageEditor' object is not iterable`。原用例在
    **加载模型前**调 `_open_scope_panel()`，撞上「请先加载模型」的提前返回，
    根本没走到构造那行，所以一直是绿的。

    现在**真的** `show()` 面板（不再是拦下 `exec_()`）：非模态后 `show()` 会
    立刻返回，测试不会挂住，而且能顺带验证「面板确实是非模态的」——应用级
    模态会吃掉发往画布的鼠标事件，让就地画区域失效。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtWidgets import QApplication

h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h._open_scope_panel()

panel = h._scope_panel
print("面板已建:", panel is not None)
print("parent 是宿主:", panel.parent() is h)
print("windows 是字典:", isinstance(panel.windows(), dict), panel.windows())
print("非模态:", not panel.isModal())
print("可见:", panel.isVisible())
# 关键：没有 activeModalWidget 拦截输入，画布才能收到鼠标事件
print("无模态窗:", QApplication.activeModalWidget() is None)
panel.close()
print("关闭后待画态已收:", h.canvas.is_drawing_region is False)
""")
    lines = _lines(out)
    assert lines[0] == "面板已建: True", lines
    assert lines[1] == "parent 是宿主: True", lines
    assert lines[2].startswith("windows 是字典: True"), lines
    assert lines[3] == "非模态: True", lines
    assert lines[4] == "可见: True", lines
    assert lines[5] == "无模态窗: True", lines
    assert lines[6] == "关闭后待画态已收: True", lines


def test_real_mixin_add_region_arms_drawing_without_switching_mode():
    """`_on_add_region_requested()` 点了「添加区域」后：画布进入一次性绘制态，且**不切贴图模式**。

    取代原先「选中『区域』就自动开会话」的联动。原先那条入口要真的产生可观察
    效果（否则就是「点了没反应」），现在这个效果是 `is_drawing_region`。
    `switched_to` 探针专门抓「偷偷切过去又切回来」——只断言 edit_mode 值不变
    是不够的。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_add_region_requested()
print(h.switched_to)
print(h.canvas.is_drawing_region)
print(h.status_label.text_)
""")
    lines = _lines(out)
    assert lines[0] == "[]", f"画区域时切了模式：{lines[0]}"
    assert lines[1] == "True", (
        f"点了「添加区域」没进入绘制态：{lines[1]}")
    assert "拖动" in lines[2] or "区域" in lines[2], lines[2]


def test_real_mixin_add_region_is_noop_without_canvas():
    """宿主没有 canvas（或画布没有该方法）时不炸：防御式取属性，静默返回。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h.canvas = None
h._on_add_region_requested()
print(h.switched_to)
h2 = _Host()
h2.canvas = type('C', (), {})()
h2._on_add_region_requested()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "[]", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_mixin_add_region_respects_region_fixed():
    """`region_fixed`（贴图模式那个全局固定）开着时不进入绘制态，只提示。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h.region_fixed = True
h._on_add_region_requested()
print(h.canvas.is_drawing_region)
print(h.status_label.text_)
""")
    lines = _lines(out)
    assert lines[0] == "False", (
        f"区域被固定时不该进入绘制态：{lines[0]}")
    assert "固定" in lines[1] or "解锁" in lines[1], (
        f"没提示为什么不能加：{lines[1]}")


def test_real_mixin_region_lock_toggles_and_refreshes_the_panel():
    """宿主收到锁定信号后：写到画布真身上，并让面板按最新状态重建列表。

    面板拿到的只是快照，所以宿主必须回推一次 `set_regions()`，否则按钮文案
    会停在旧状态（锁完还是写着「锁定」）。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h.region_boxes = [{'shape_type': 'rectangle', 'x': 10, 'y': 10,
                   'width': 40, 'height': 40}]
h._on_region_lock_toggled(0, True)
print(h.region_boxes[0].get('locked'))
print(h.canvas.set_locked_calls)
# 越界下标是空操作，不该炸
h._on_region_lock_toggled(9, True)
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "True", f"锁定没写到画布真身上：{lines[0]}"
    assert lines[1] == "[(0, True)]", (
        f"没落到画布那块区域上：{lines[1]}")
    assert lines[2] == "ok", lines[2]


def test_real_mixin_region_display_follows_the_roi_mode_only():
    """区域轮廓要**同时**满足「面板开着」与「选了区域」。

    两条要求叠起来：
    - 切到滑窗/全图就不能再挂着区域框（否则和滑窗网格混在一起分不清）。
    - 关掉面板（点确定/取消/直接关）也不能留着——用户报的正是「点确认后
      画布上的区域框还在」。

    但编辑能力不受影响：贴图模式下区域本来就看得见、画得动。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

# 面板开着 + 区域：显示
h._scope_mode = MODE_ROI
h._scope_panel_open = True
h._sync_scope_region_display()
print('open-roi', getattr(h, '_show_scope_regions', None))

# 面板开着 + 滑窗：不显示——用户报的「点滑窗还展示区域画框」
h._scope_mode = MODE_WINDOW
h._sync_scope_region_display()
print('open-window', getattr(h, '_show_scope_regions', None))

# 面板开着 + 全图：不显示
h._scope_mode = MODE_FULL
h._sync_scope_region_display()
print('open-full', getattr(h, '_show_scope_regions', None))

# 面板关掉 + 区域：也不显示——用户报的「点确认后区域框还在」
h._scope_mode = MODE_ROI
h._scope_panel_open = False
h._sync_scope_region_display()
print('closed-roi', getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "open-roi True", lines[0]
    assert lines[1] == "open-window False", (
        f"切到滑窗后区域轮廓还在显示（{lines}）：用户报的就是这个")
    assert lines[2] == "open-full False", lines[2]
    assert lines[3] == "closed-roi False", (
        f"关掉面板后区域轮廓还在显示（{lines}）："
        "用户报的「点确认后画布上的区域框还在」")


def test_scope_modules_have_all_translations():
    """范围面板/滑窗弹窗里的 tr() 键都要有两种语言。

    两个模块的键有一部分是**变量**（`tr(name)` / `tr(names[mode])`），正则
    扫不到，所以这里连 `_MODE_KEYS` 一起查，别只依赖源码扫描。
    """
    import re

    from labelforge.ui import i18n
    from labelforge.ui.scope_panel import _MODE_KEYS

    keys = set()
    for rel in ("labelforge/ui/scope_panel.py",
                "labelforge/ui/sliding_window_dialog.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        keys |= set(re.findall(r"tr\(['\"]([^'\"]+)['\"]\)", source))
    for _mode, name, hint in _MODE_KEYS:
        keys |= {name, hint}
    keys |= {"全图", "滑窗", "区域"}

    missing = sorted((key, lang) for key in keys for lang in ("zh", "en")
                     if key not in i18n._strings[lang])
    assert not missing, (
        f"{missing}\n"
        "产品代码在 scope_panel.py 里用了这些 tr() 键，但 i18n 里一条都没有；"
        "要补的是 labelforge/ui/i18n.py（产品代码），不是本测试。")


def test_real_mixin_toggles_scope_region_display():
    """真实控件：区域轮廓的开关是「面板开着 + 选了区域」两个条件的与。

    标注模式下也要看得见范围，否则用户不知道框为什么只出现在某块地方；
    但面板一关就得收起，否则「点确认后画布上的区域框还在」。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

h._scope_panel_open = True
for mode in (MODE_ROI, MODE_WINDOW, MODE_FULL):
    h._scope_mode = mode
    h._refresh_scope_button()
    print('open', mode, getattr(h, '_show_scope_regions', None))

h._scope_panel_open = False
for mode in (MODE_ROI, MODE_WINDOW):
    h._scope_mode = mode
    h._refresh_scope_button()
    print('closed', mode, getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "open roi True", lines[0]
    assert lines[1] == "open window False", lines[1]
    assert lines[2] == "open full False", lines[2]
    assert lines[3] == "closed roi False", lines[3]
    assert lines[4] == "closed window False", lines[4]


def test_real_window_dialog_warns_when_many_windows():
    """真实控件：窗口数很多时给耗时提醒，但仍允许确定。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(20000, 20000))
d._w.setValue(640); d._h.setValue(640); d._ov.setValue(0)
print(d.window_count(), d.ok_btn.isEnabled())
print(d._estimate.text().replace(chr(10), ' / '))
""")
    lines = _lines(out)
    assert lines[0] == "1024 True", lines[0]
    assert "1024" in lines[1] and "倍" in lines[1], lines[1]


def test_real_mixin_roi_without_regions_raises_before_start():
    """真实控件：区域模式但没有区域时，启动前就抛错（不是跑到后台变 0 框）。"""
    out = _run_real_qt(_MIXIN_HOST + """
from labelforge.engine.inference_scope import ScopeError
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.current_background = QImage(400, 300, QImage.Format_RGB888)
h.region_boxes = []
h._scope_mode = MODE_ROI
try:
    h._scope_for_run()
    print("NO RAISE")
except ScopeError as exc:
    print("ScopeError", '没有可用区域' in str(exc))
""")
    assert _lines(out)[0] == "ScopeError True", _lines(out)[0]


def test_real_mixin_load_resets_scope_to_full():
    """真实控件：换模型后范围回到全图，不沿用上一个模型的区域/窗口模式。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
h._scope_mode = MODE_ROI
h._refresh_scope_button()
print(h.model_row.scope_summary_label.text())
h._on_model_loaded(_Detector(), (640, 640), "model")
print(h.model_row.scope_summary_label.text(), h._scope_for_run())
""")
    lines = _lines(out)
    assert lines[0] == "区域", lines[0]
    assert lines[1] == "全图 None", lines[1]


def test_real_mixin_saves_and_restores_window_spec():
    """真实控件：窗口参数存配置后能读回来（范围模式不存）。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h._window_spec = {'w': 512, 'h': 512, 'overlap': 64}
h._scope_mode = MODE_WINDOW
h._save_inference_settings()
print(h._saved_window_spec())
from labelforge.core import config_manager
saved = config_manager.load_all()['inference_settings']
print('mode' in saved, 'window_spec' in saved)
""")
    lines = _lines(out)
    # 窗口参数现在还会带上 `min_visible`（跨窗去重的最低可见比例），所以
    # 这里按「三个尺寸键要原样存取回来」断言，而不是钉死整份字典——否则
    # 每加一个合法的窗口参数都要来改这条测试。
    saved_spec = lines[0]
    assert "'w': 512" in saved_spec, saved_spec
    assert "'h': 512" in saved_spec, saved_spec
    assert "'overlap': 64" in saved_spec, saved_spec
    assert "'min_visible'" in saved_spec, saved_spec
    assert lines[1] == "False True", lines[1]


def test_real_scope_panel_positions_on_its_anchor():
    """真实控件：弹窗**左上角对齐**锚点的左上角（全局坐标），而不是屏幕正中。

    用户要求「范围推理窗口左上角始终对齐标签管理区域的左上角」——竖直方向
    也要跟，所以不再是「竖直居中在锚点那块」，而是顶边跟顶边、左边跟左边。
    锚点由 `_scope_anchor_rect()` 给出，就是「标签管理」区那块。

    锚点取「屏幕上偏左」是为了避开钳制：贴到中右部会被钳回右边（那是
    `test_real_scope_panel_clamps_anchor_to_screen` 的事）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QRect
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
avail = app.primaryScreen().availableGeometry()
# 左侧锚点：给 440px 弹窗留得下，钳制不会介入。
anchor = QRect(avail.left() + 10, avail.top() + 30, 350, 200)
host = QWidget()
host.resize(avail.width(), avail.height())
host.show()
panel = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, parent=host)
panel.set_anchor_rect(anchor)
panel.show()
app.processEvents()
# 用全局坐标比较：panel.geometry() 是相对父窗口的，边界值易混
panel_tl = panel.mapToGlobal(panel.rect().topLeft())
anchor_tl = host.mapToGlobal(anchor.topLeft())
# 水平：左边不越过锚点左边界（面板比锚点宽，向右溢出，不盖画布）
print(anchor_tl.x() <= panel_tl.x() <= anchor_tl.x() + 40)
# 竖直：顶边与锚点顶边对齐
print(abs(panel_tl.y() - anchor_tl.y()) <= 2)
panel.close()
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", f"没顶对齐到锚点左上角: {lines}"


def test_real_scope_panel_clamps_anchor_to_screen():
    """锚点超出屏幕可用区时钳回屏内：不能把弹窗推到看不见的地方。

    量的是 `frameGeometry()`（**全局**坐标）：弹窗是顶层窗口，`geometry()`
    给的是父窗坐标，拿去和屏幕可用区比是两套坐标系，会假红。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import QRect
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
avail = app.primaryScreen().availableGeometry()
host = QWidget()
host.resize(avail.width(), avail.height())
host.show()
panel = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, parent=host)
# 故意把锚点放到屏幕右边缘之外
panel.set_anchor_rect(QRect(avail.right() + 50, avail.top(), 400, 400))
panel.show()
app.processEvents()
g = panel.frameGeometry()
print(avail.left() <= g.left() and g.right() <= avail.right())
print(avail.top() <= g.top() and g.bottom() <= avail.bottom())
panel.close()
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", lines[1]


def test_real_scope_panel_without_anchor_centers_on_parent():
    """没给锚点（取不到右侧栏）时退回居中，行为与改动前一致。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QWidget
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
host = QWidget()
host.resize(800, 600)
host.show()
panel = ScopePanel(parent=host)
panel.show()
g = panel.geometry()
# 居中：左右留白大致相等。800-440=360 是奇数，整数除法的 1px 舍入
# 加上宿主自身的窗口装饰偏移（实测 4px），容差给到 6。
left_gap = g.left() - host.geometry().left()
right_gap = host.geometry().right() - g.right()
print(abs(left_gap - right_gap) <= 6)
panel.close()
""")
    assert _lines(out)[0] == "True", out


def test_real_sliding_dialog_count_follows_params():
    """真实控件：改参数 → 窗口数跟着变，且与引擎一致。

    弹窗自身不再画预览（网格改画在主画布上），所以这里量 `window_count()`；
    它仍是「画面上的格子数」的来源，与引擎同源这条不变。

    方向要注意：**减小重叠会让窗口数变少**（重叠越大越密）。实测
    2000×1000 图、默认 640×640/重叠 200 得 10 窗；把宽度降到 500、
    重叠清零后 500×640/重叠 0 得 8 窗。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog
from labelforge.engine.inference_scope import iter_windows

app = QApplication([])
d = SlidingWindowDialog(reference_size=(2000, 1000))
before = d.window_count()
d._w.setValue(500)
d._ov.setValue(0)
after = d.window_count()
print(before, after)
print(before != after)
print(d.params.is_valid())
print(after == len(iter_windows(2000, 1000, d.spec())))
""")
    lines = _lines(out)
    assert lines[0] == "10 8", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", lines[3]


def test_real_sliding_dialog_count_matches_engine_count():
    """窗口数必须等于 iter_windows()：画面上的网格就是实际会跑的窗口。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog
from labelforge.engine.inference_scope import iter_windows

app = QApplication([])
d = SlidingWindowDialog(reference_size=(2000, 1000))
spec = d.spec()
print(d.window_count() == len(iter_windows(2000, 1000, spec)))
""")
    assert _lines(out)[0] == "True", out


def test_real_sliding_dialog_count_unusable_without_reference():
    """没有参考图时算不出窗口数（返回 None），但弹窗仍能开、不崩。

    注意 `is_valid()` 此时是 **True**：没有参考图意味着「还不知道要跑多少窗」，
    按不可用禁掉确定反而是错的（用户可能先配参数、后加载图）。这条只钉
    「算不出数目但不崩」，不把 is_valid 的口径写死在这。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=None)
print(d.window_count(), d.params.is_valid())
d.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None True", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_sliding_dialog_has_no_template_buttons():
    """滑窗弹窗里**没有**模板按钮：用户要求「推理滑窗不要模板了」。

    早先这里有一排「模板」按钮，与数据增强共用 `crop_templates`。现在两边
    脱钩——推理侧只认输入框里的值，数据增强侧的模板列表不该出现在这里，也不
    该被这里的操作写坏。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog
from labelforge.core import config_manager

app = QApplication([])
# 数据增强侧**有**模板：推理面板仍不该显示
config_manager.add_crop_template({'w': 320, 'h': 320, 'overlap': 100,
                                  'min_visible': 0.5})
d = SlidingWindowDialog(reference_size=(2000, 1000),
                        spec={'w': 640, 'h': 640, 'overlap': 200})
print(hasattr(d, 'template_buttons'))
print(hasattr(d.params, 'template_buttons'))
print(len(d.params.template_buttons()) if hasattr(d.params, 'template_buttons') else 0)
print(d.spec())
""")
    lines = _lines(out)
    assert lines[0] == "False", (
        f"滑窗弹窗还有 template_buttons API，容易再被接回模板：{lines[0]}")
    assert lines[1] == "False", (
        f"参数控件还有 template_buttons API：{lines[1]}")
    assert lines[2] == "0", f"还显示出模板了：{lines[2]}"
    # 参数是调用方传进来的那份，不受数据增强模板影响
    assert _spec_size(lines[3]) == (640, 640, 200), lines[3]


def test_real_sliding_dialog_does_not_write_the_shared_template_list():
    """弹窗的操作**绝不**改数据增强的模板列表（两边已脱钩）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.sliding_window_dialog import SlidingWindowDialog
from labelforge.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 384, 'h': 384, 'overlap': 64,
                                  'min_visible': 0.5})
before = [tuple(sorted(t.items(), key=lambda kv: kv[0]))
          for t in config_manager.load_crop_templates()]
d = SlidingWindowDialog(reference_size=(2000, 1000))
d._w.setValue(777)
d.window_count()
after = [tuple(sorted(t.items(), key=lambda kv: kv[0]))
         for t in config_manager.load_crop_templates()]
print(before == after, len(before) > 0)
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]


def test_real_window_params_defaults_are_640_visible_80_no_fill():
    """默认值：640×640、最低可见比例 80%；**填充不开**。

    用户明确「不开 fill」，也明确可见比例默认 80%。这三个数一起钉：只钉一个
    的话，另两个改回旧值照样绿。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import (
    WindowParamsWidget, DEFAULT_WINDOW, DEFAULT_OVERLAP, DEFAULT_MIN_VISIBLE)

app = QApplication([])
w = WindowParamsWidget()
s = w.spec()
print(DEFAULT_WINDOW, DEFAULT_OVERLAP, DEFAULT_MIN_VISIBLE)
print(s['w'], s['h'], s['overlap'], s['min_visible'])
print(w.min_visible_spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "640 200 0.8", f"常量不对：{lines[0]}"
    assert lines[1] == "640 640 200 0.8", f"默认值不对：{lines[1]}"
    assert lines[2] == "80", f"输入框没按 80% 显示：{lines[2]}"


def test_real_window_params_reads_a_stale_min_visible_from_old_config():
    """旧配置里存着 0.7 时按 0.7 显示：只改**默认值**，不篡改用户已存的值。

    用户的旧配置是他自己的选择，换默认值不该把已有配置一起改掉；而坏值
    （字符串/越界）仍要退回新默认 0.8。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget

app = QApplication([])
w = WindowParamsWidget(spec={'min_visible': 0.7})
print(w.min_visible_spin.value())
print(w.spec()['min_visible'])

bad = WindowParamsWidget(spec={'min_visible': 'oops'})
print(bad.min_visible_spin.value())

out_of_range = WindowParamsWidget(spec={'min_visible': 0.2})
print(out_of_range.min_visible_spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "70", f"旧配置的 0.7 被改了：{lines[0]}"
    assert lines[1] == "0.7", lines[1]
    assert lines[2] == "80", f"坏值没退回新默认 0.8：{lines[2]}"
    assert lines[3] == "50", f"越界值应被输入框夹到下限 50：{lines[3]}"


def test_real_window_params_never_reads_the_shared_templates():
    """推理侧参数不再从 `crop_templates` 读任何东西。

    写一条数据增强模板（w/h 与推理默认都不同），新建的参数控件必须还是
    默认值——否则「模板同步」就又从别处溜回来了。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget
from labelforge.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 320, 'h': 320, 'overlap': 100,
                                  'min_visible': 0.5})
w = WindowParamsWidget()
s = w.spec()
print(s['w'], s['h'], s['overlap'], s['min_visible'])
""")
    lines = _lines(out)
    assert lines[0] == "640 640 200 0.8", (
        f"推理默认被数据增强模板带跑了：{lines[0]}")


def test_real_mixin_panel_close_drops_pending_region_draw_on_every_path():
    """关面板一定收掉待画的区域并关掉区域显示：确定 / 取消 / 直接关闭都一样。

    绘制态是面板按钮开的（「点一下画一个」），面板都没了就不该还挂着「下一次
    在图上拖动会画出一个区域」——那正是旧会话劫持鼠标的老毛病换个壳。
    用**真的** ScopePanel + 真的 `close()`/`accept()`/`reject()`：非模态后
    这些调用会立即返回，不再需要替身拦 `exec_()`；替身恰好会把新机制
    （`finished_ok` 信号回传）整条绕过去，测不到。

    场景取「区域」范围：这里要验的正是「面板开着 → 轮廓在显示 → 关掉 →
    待画态收了、轮廓也收起」。轮廓的收起对应用户报的「点确认后画布上的区域框
    还在」——面板只是调范围时的参照，关掉就不该继续叠加在画布上。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtWidgets import QApplication
from labelforge.engine.inference_scope import MODE_ROI

for call in ('accept', 'reject', 'close'):
    h = _Host()
    h._on_model_loaded(_Detector(), (640, 640), "model")
    h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
    h._scope_mode = MODE_ROI
    h._open_scope_panel()
    # 面板开着时点一下「添加区域」，模拟用户正要画一个
    h._on_add_region_requested()
    during_draw = h.canvas.is_drawing_region
    during_show = h._show_scope_regions
    getattr(h._scope_panel, call)()
    QApplication.processEvents()
    print(call, during_draw, during_show, h.canvas.is_drawing_region,
          h._scope_panel_open, h._show_scope_regions)
""")
    lines = _lines(out)
    for i, call in enumerate(("accept", "reject", "close")):
        assert lines[i] == f"{call} True True False False False", lines[i]


def test_real_mixin_scope_panel_closing_never_leaves_a_pending_draw():
    """无论以哪条路径关闭，都不能留下「下次拖动画区域」的待命态。"""
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtWidgets import QApplication

h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h._open_scope_panel()
h._on_add_region_requested()
# 模拟异常路径：宿主在收尾途中抛异常，也必须已经清干净
panel = h._scope_panel
try:
    panel.reject()
    QApplication.processEvents()
    raise RuntimeError('boom')
except RuntimeError:
    pass
print("drawing", h.canvas.is_drawing_region)
print("open flag", getattr(h, '_scope_panel_open', None))
print("show", getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "drawing False", lines[0]
    assert lines[1] == "open flag False", lines[1]
    assert lines[2] == "show False", lines[2]




def test_real_mixin_switching_to_window_hides_the_region_outlines():
    """① 切到「滑窗」后画布上不能再出现区域画框。

    用户报的原始现象：推理范围里画好的区域框，点了「滑窗」还在画面上，和滑窗
    网格叠在一起，分不清哪个才是这次要跑的范围。

    用真 Canvas 的 `render()` 落到图上再数「区域描边色」像素（不是只看
    `_show_scope_regions` 这个中间标记——标记对了但绘制路径没接到也是白搭）。
    区域色是 `REGION_CONFIG['border_color']`，按「是那个色相」判定，不按精确
    RGB：描边叠在背景上同样是混合色。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.core.config import REGION_CONFIG
from labelforge.engine.inference_scope import MODE_WINDOW, MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()

img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
app.processEvents()
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()

# 画一个覆盖画面大半的区域，保证描边一定进画面
w.region_boxes = [{'x': 100, 'y': 100, 'width': 700, 'height': 500}]
w.region_fixed = False

edge = QColor(REGION_CONFIG['border_color'])

def count_region_edge():
    im = QImage(canvas.size(), QImage.Format_ARGB32)
    im.fill(QColor('white'))
    pnt = QPainter(im)
    canvas.render(pnt, QPoint())
    pnt.end()
    n = 0
    for y in range(im.height()):
        for x in range(im.width()):
            c = im.pixelColor(x, y)
            # 区域描边是青色系：蓝绿明显高于红
            if c.blue() > 120 and c.green() > 110 and c.red() < 90:
                n += 1
    return n

w._scope_panel_open = True
w._scope_mode = MODE_ROI
w._sync_scope_region_display()
roi_px = count_region_edge()

w._scope_mode = MODE_WINDOW
w._sync_scope_region_display()
win_px = count_region_edge()

print(roi_px)
print(win_px)
print(roi_px > win_px)
""")
    lines = _lines(out)
    roi_px = int(lines[0])
    win_px = int(lines[1])
    assert roi_px > 0, f"区域模式没画出区域框（像素 {roi_px}）"
    assert lines[2] == "True", (
        f"切到滑窗后区域框还在画（区域 {roi_px} vs 滑窗 {win_px}）："
        "用户报的就是这个")


def test_real_scope_params_stay_visible_but_locked_outside_window_mode():
    """② 参数区打开面板就要在；未选「滑窗」时置灰不可改。

    两个要求各自断言：可见性（「要存在」）与可编辑性（「无法编辑」）分开量，
    免得一个变了另一个跟着"看起来对"。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
p.show()
app.processEvents()
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    p.set_mode(mode)
    app.processEvents()
    wp = p.window_params
    print(mode, wp.isVisible(), wp.is_editable())
""")
    lines = _lines(out)
    # 三种模式下参数区都在（② 的「打开窗口就要存在」）
    assert lines[0] == "full True False", lines[0]
    assert lines[1] == "window True True", lines[1]
    assert lines[2] == "roi True False", lines[2]


def test_real_scope_panel_mode_selection_is_single_choice():
    """② 的隐藏前提：三个模式按钮必须真正单选。

    QRadioButton 的自动互斥只在同一个父控件内生效，而这里每个按钮各占一行、
    父控件不同，若不显式维护就会同时选中多个，`mode()` 只返回字典序里第一个
    （永远是全图）——表现为「怎么点都切不过去」。

    两条路径都要量，因为它们各自都可能单独失效：
      * 用户**点击**按钮 → 走 `toggled` → `_on_mode_toggled` 清理其他项；
      * 程序**set_mode()** → 由 `set_mode` 自己清掉旧的选中项。
    只测一条的话，另一条的实现被删掉也照样绿。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel()
p.show()
app.processEvents()

# 路径一：程序调用 set_mode
for want in (MODE_WINDOW, MODE_ROI, MODE_FULL, MODE_WINDOW):
    p.set_mode(want)
    app.processEvents()
    checked = [m for m, b in p._buttons.items() if b.isChecked()]
    print('set', want, p.mode(), len(checked))

# 路径二：模拟用户点击（走 toggled -> _on_mode_toggled）
for want in (MODE_ROI, MODE_FULL, MODE_WINDOW):
    p._buttons[want].click()
    app.processEvents()
    checked = [m for m, b in p._buttons.items() if b.isChecked()]
    print('click', want, p.mode(), len(checked))
""")
    lines = _lines(out)
    expected_set = ("window", "roi", "full", "window")
    for i, want in enumerate(expected_set):
        assert lines[i] == f"set {want} {want} 1", (
            f"set_mode 没有单选（第 {i+1} 次 wanted={want}）：{lines}")
    expected_click = ("roi", "full", "window")
    for j, want in enumerate(expected_click):
        row = lines[len(expected_set) + j]
        assert row == f"click {want} {want} 1", (
            f"点击按钮没有单选（第 {j+1} 次 clicked={want}）：{lines}")


def test_real_mixin_reopening_the_panel_keeps_the_chosen_mode():
    """② 的关联缺陷：重开面板不能把上次选的范围重置成「全图」。

    开面板时的一串 set_* 都会经 `overlay_changed` 回调把**面板**当时的 mode
    写回宿主，而面板那一刻还停在构造默认值（全图），于是「选了区域 → 关掉 →
    再打开」会静默变回全图。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
for name, mode in (("区域", MODE_ROI), ("滑窗", MODE_WINDOW)):
    h._scope_mode = mode
    h._open_scope_panel()
    print(name, h._scope_panel.mode(), h._scope_mode)
""")
    lines = _lines(out)
    assert lines[0] == "区域 roi roi", (
        f"重开面板把「区域」重置了（{lines}）")
    assert lines[1] == "滑窗 window window", (
        f"重开面板把「滑窗」重置了（{lines}）")


def test_real_scope_panel_lists_region_coordinates_with_delete_buttons():
    """① 区域画完后，面板里逐条列出坐标，每条右侧有删除按钮。

    用户要求「区域画完后在推理范围窗口显示该框的坐标 加一个删除按钮」。
    坐标取外接矩形（多边形只有点集，显示第一个点会误导人），退化区域不能
    把面板构造炸掉。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
p = ScopePanel(region_count=0)
p.show()
app.processEvents()

p.set_regions([
    {'shape_type': 'rectangle', 'x': 10, 'y': 20, 'width': 100, 'height': 50},
    {'shape_type': 'polygon', 'points': [[5, 5], [105, 15], [55, 95]]},
    {'x': 0, 'y': 0, 'width': 0, 'height': 0},
])
app.processEvents()

def row_text(i):
    w = p.region_list.itemWidget(p.region_list.item(i))
    return w.findChild(QLabel, 'scopeRegionText').text()

def has_delete(i):
    w = p.region_list.itemWidget(p.region_list.item(i))
    return w.findChild(QPushButton, 'scopeRegionDelete') is not None

print(p.region_list.count())
print(row_text(0))
print(row_text(1))
print(row_text(2))
print(has_delete(0), has_delete(1), has_delete(2))
# 明细只在「区域」模式显示（与「编辑区域」按钮同步）
p.set_mode(MODE_ROI)
app.processEvents()
print('roi', p.region_list.isVisible())
""")
    lines = _lines(out)
    assert lines[0] == "3", lines[0]
    assert lines[1] == "#1  (10, 20) 100×50", lines[1]
    # 多边形取外接矩形，不是第一个顶点 (5, 5)
    assert lines[2] == "#2  (5, 5) 100×90", lines[2]
    # 退化区域不能让整行崩掉
    assert lines[3] == "#3  (未知)", lines[3]
    assert lines[4] == "True True True", lines[4]
    assert lines[5] == "roi True", lines[5]


def test_real_scope_panel_delete_removes_the_right_region():
    """① 点第 N 行的删除，删的就是第 N 条（不是最后一条或第一条）。

    下标必须逐行绑定：闭包直接引用循环变量的话，所有按钮都会删最后一条，
    而界面上看起来「按钮都在」，很容易漏掉。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtWidgets import QApplication, QPushButton
from labelforge.engine.inference_scope import MODE_ROI

h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [
    {'shape_type': 'rectangle', 'x': 10, 'y': 10, 'width': 100, 'height': 80},
    {'shape_type': 'rectangle', 'x': 200, 'y': 50, 'width': 60, 'height': 60},
    {'shape_type': 'rectangle', 'x': 400, 'y': 77, 'width': 30, 'height': 30},
]
h._open_scope_panel()
h._scope_panel.set_mode(MODE_ROI)
app.processEvents()

def click_delete(i):
    w = h._scope_panel.region_list.itemWidget(h._scope_panel.region_list.item(i))
    w.findChild(QPushButton, 'scopeRegionDelete').click()
    app.processEvents()

click_delete(1)   # 删中间那条
print(len(h.region_boxes))
print(h.region_boxes[0]['x'], h.region_boxes[1]['x'])
print(h._scope_panel.region_list.count())
""")
    lines = _lines(out)
    assert lines[0] == "2", lines[0]
    # 剩下的应是第 1 条与第 3 条（x=10 与 x=400），中间的 x=200 被删
    assert lines[1] == "10.0 400.0" or lines[1] == "10 400", lines[1]
    assert lines[2] == "2", lines[2]


def test_real_mixin_confirming_the_panel_clears_region_outlines():
    """② 点「确定」后画布上不能再留着区域框。

    用户报的原始现象：「点击确认后关闭推理范围窗口还是显示在界面」——
    指的是画布上的区域框还在。用真 Canvas 数「区域描边色」像素，
    断言确定后归零（不是只看 `_show_scope_regions` 这个中间标记）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()

img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.show_grid_checkbox.setChecked(False)
app.processEvents()
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()

w.region_boxes = [{'x': 100, 'y': 100, 'width': 700, 'height': 500}]
w.region_fixed = False
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()

def count_edges():
    im = QImage(canvas.size(), QImage.Format_ARGB32)
    im.fill(QColor('white'))
    pnt = QPainter(im)
    canvas.render(pnt, QPoint())
    pnt.end()
    n = 0
    for y in range(im.height()):
        for x in range(im.width()):
            c = im.pixelColor(x, y)
            if c.blue() > 120 and c.green() > 110 and c.red() < 90:
                n += 1
    return n

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
before = count_edges()
w._scope_panel.close()
app.processEvents()
after_ok = count_edges()

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w._scope_panel.close()
app.processEvents()
after_cancel = count_edges()

print(before)
print(after_ok)
print(after_cancel)
""")
    lines = _lines(out)
    before = int(lines[0])
    after_ok = int(lines[1])
    after_cancel = int(lines[2])
    assert before > 0, f"区域模式下面板开着却没画出区域框（像素 {before}）"
    assert after_ok == 0, (
        f"点确定后画布上还留着区域框（{before} → {after_ok}）：用户报的就是这个")
    assert after_cancel == 0, (
        f"点取消后画布上还留着区域框（{before} → {after_cancel}）")


def test_real_scope_panel_floats_over_the_side_panel_not_the_canvas():
    """① 范围面板要浮在**右侧侧栏**上层，不能压到画布上。

    用户报的现象：「推理范围窗口怎么在屏幕左边了？如果我软件移过去不就遮挡
    画布了」。根因有两个，这里一并钉住：
      * 锚点曾经是 `QRect(0, ...)`——x 被写死成 0，而画布在主窗左侧，
        x=0 就是画布左端；
      * `move()` 对顶层窗口是**全局**坐标，早先在 `place_within` 里又减了一次
        父窗原点，等于把面板整体左移一个主窗 x 的量。

    量的是 `frameGeometry()`（全局）与侧栏的实际全局位置，两者同构。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication

h = _Host()
h.resize(1200, 800)
h.show()
app.processEvents()
h.background_images = ["a.jpg"]
h.current_background_index = 0
h.current_background = QPixmap.fromImage(QImage(400, 300, QImage.Format_RGB888))
h.detection_boxes = []
h.canvas_items = []
h.detection_boxes_dict = {0: []}
h._detector = _Detector()
h._open_scope_panel()
app.processEvents()
panel = h._scope_panel

side = h._side_panel_geometry()
side_left = h.mapToGlobal(side.topLeft()).x()
frame = panel.frameGeometry()
screen_bottom = app.primaryScreen().availableGeometry().bottom()
# 竖直契约：顶边对齐锚点（`_scope_anchor_rect` =「标签管理」区左上角），
# 放不下时上移。锚点由宿主提供，这里直接问它，比猜 label_group 更稳。
anchor = h._scope_anchor_rect()
label_top = h.mapToGlobal(anchor.topLeft()).y()

# 面板要落在侧栏这块的左边界起；「不压画布」由锚点 x 不是 0 保证
# （画布在主窗左侧，x=0 就是画布左端）。
#
# 水平用「中心线对齐侧栏中心」而不是只判左右边界：锚点够宽时，把 gx 整体
# 平移某个量后左边界仍可能落在 side_left 右侧（被 `gx = left` 兜住或屏幕钳住），
# 只判边界的话『面板左移』这类变异会存活（实测）。
#
# 竖直**不再**断言「居中」：那是 35f9c3c 改用左上角对齐之前的契约；当时这条
# 之所以绿，是因为 PyQt5 离屏屏幕只有 600 高把面板压扁、中心恰好落在侧栏中心
# 附近。离屏屏幕尺寸在 Qt6 变成 800x800，巧合没了。改成钉真实不变量：
# 顶边不越过锚点、底边不被屏幕裁掉。
side_center_x = h.mapToGlobal(side.center()).x()
print(frame.x() >= side_left)
print(abs(side_left - frame.x()) < side.width())
print(frame.top() <= label_top)
print(frame.bottom() <= screen_bottom)
print(abs(frame.center().x() - side_center_x) < 8)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        f"面板左边缘跑到侧栏左边之外了（{lines}）：用户报的「跑到屏幕左边」")
    assert lines[1] == "True", (
        f"面板整个跑出侧栏那一块了（{lines}）")
    assert lines[2] == "True", (
        f"面板顶边越过了「标签管理」区顶边（{lines}）：应左上角对齐或上移")
    assert lines[3] == "True", (
        f"面板底边被屏幕裁掉了（{lines}）")
    assert lines[4] == "True", (
        f"面板没有对齐侧栏的水平中心（{lines}）：整体平移能被这条抓住")


def test_real_scope_panel_anchor_uses_the_real_side_panel_x():
    """① 的根因回归：锚点必须带侧栏的真实 x，不能再是 0。

    只钉「面板最终位置」的话，换个屏幕尺寸/窗口位置就可能测不出来；这里直接
    断言锚点矩形与侧栏左边界对齐。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication

h = _Host()
h.resize(1200, 800)
h.show()
app.processEvents()
h.background_images = ["a.jpg"]
h.current_background_index = 0
h.current_background = QPixmap.fromImage(QImage(400, 300, QImage.Format_RGB888))
h.detection_boxes = []
h.canvas_items = []
h.detection_boxes_dict = {0: []}
h._detector = _Detector()

anchor = h._scope_anchor_rect()
label_tl = h.label_group.mapTo(h, h.label_group.rect().topLeft())
bg_left = h.bg_list_group.mapTo(h, h.bg_list_group.rect().topLeft()).x()

print(anchor.x() > 0)
print(abs(anchor.x() - bg_left) <= 12)
print(anchor.width() > 0)
# ④：锚点就是「标签管理」区那块，**左上角**对齐它（含竖直方向）
print(abs(anchor.y() - label_tl.y()) <= 12)
print(anchor.height() <= h.label_group.height() + 20)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        f"锚点 x 还是 0（{lines}）：这正是「面板跑到屏幕左边」的根因")
    assert lines[1] == "True", f"锚点左边界没和侧栏对齐（{lines}）"
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", (
        f"锚点顶边没和「标签管理」区对齐（{lines}）：面板左上角对不上")
    assert lines[4] == "True", (
        f"锚点高度不是「标签管理」区的高度（{lines}）")


def test_real_batch_target_runs_the_span_between_current_and_target():
    """② 「张数范围」的语义是**结束张号**（1 基，含），方向自动决定。

    用户明确过：「假设当前图是第 200 张，我输入 300 就是 200~300；我输入 100
    就是 100~200」。填 0 = 不限制、到末尾。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h.background_images = [f"img{i}.jpg" for i in range(400)]
h.current_background_index = 199          # 第 200 张
h._detector = _Detector()
h._on_model_loaded(_Detector(), (640, 640), "model")
row = h._ensure_model_row()
row.set_batch_to_range(len(h.background_images))

captured = {}
def spy(indices, current_only):
    captured['i'] = list(indices)
h._start_inference = spy

for value in (300, 100, 0, 200):
    row.set_batch_to(500 if value == 0 else value)
    if value == 0:
        row.set_batch_to(0)
    h._run_inference_from_here()
    idx = captured['i']
    print(value, idx[0] + 1, idx[-1] + 1, len(idx))
""")
    lines = _lines(out)
    assert lines[0] == "300 200 300 101", lines[0]
    assert lines[1] == "100 100 200 101", lines[1]
    # 0 = 到末尾：第 200 张 ~ 第 400 张
    assert lines[2] == "0 200 400 201", lines[2]
    assert lines[3] == "200 200 200 1", lines[3]


def test_real_batch_target_spin_is_spinnable_and_typable():
    """② 输入框要「可滚动可输入」，并且上限跟着数据集张数收窄。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_batch_to_range(350)
spin = row.batch_to_spin
print(spin.maximum())
print(row.batch_to_target())
# 可滚动：stepUp/stepDown 能改值
row.set_batch_to(300); spin.stepUp()
print(spin.value())
spin.stepDown()
print(spin.value())
# 可输入：setValue 走同一条路（键盘输入最终也是 setValue）
row.set_batch_to(123)
print(row.batch_to_target())
# 超出数据集张数的值会被上限挡住
row.set_batch_to(99999)
print(spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "350", f"上限没跟着数据集张数走：{lines[0]}"
    # 默认值 = 最后一张（① 的要求）。这条断言原先钉的是旧默认值 0，
    # 现在的需求恰好把「默认末张」定为新行为，所以跟着改；其余断言不动。
    assert lines[1] == "350", f"默认值不是最后一张：{lines[1]}"
    assert lines[2] == "301", f"不能滚动加一：{lines[2]}"
    assert lines[3] == "300", f"不能滚动减一：{lines[3]}"
    assert lines[4] == "123", f"不能直接输入：{lines[4]}"
    assert lines[5] == "350", f"超出张数的值没被上限挡住：{lines[5]}"


def test_real_size_fields_hidden_for_static_shown_for_dynamic():
    """③ 输入尺寸：静态模型整组不显示，动态模型才显示。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.show()
app.processEvents()
# 未加载模型：先收起
print(row.size_fields_visible())

row.set_model("static.onnx", 0.25, 0.45, (640, 640), dynamic=False)
app.processEvents()
print(row.size_fields_visible())
print(row.size_lbl.isVisible(), row.size_spin_w.isVisible(),
      row.size_spin_h.isVisible(), row.size_sep_label.isVisible())

row.set_model("dynamic.onnx", 0.25, 0.45, (640, 640), dynamic=True)
app.processEvents()
print(row.size_fields_visible())
print(row.size_spin_w.isEnabled(), row.size_spin_h.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "False", f"未加载模型就该收起输入尺寸：{lines[0]}"
    assert lines[1] == "False", f"静态模型仍显示输入尺寸：{lines[1]}"
    assert lines[2] == "False False False False", (
        f"静态模型只藏了一部分控件：{lines[2]}")
    assert lines[3] == "True", f"动态模型没显示输入尺寸：{lines[3]}"
    assert lines[4] == "True True", f"动态模型的输入框不可编辑：{lines[4]}"


def test_real_threshold_spins_step_by_one_hundredth():
    """③ 置信度 / NMS 的步长是 0.01，且宽度收窄到 58px。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
print(round(row.conf_spin.singleStep(), 4), round(row.iou_spin.singleStep(), 4))
row.conf_spin.setValue(0.25)
row.conf_spin.stepUp()
print(round(row.conf_spin.value(), 4))
row.iou_spin.setValue(0.45)
row.iou_spin.stepDown()
print(round(row.iou_spin.value(), 4))
print(row.conf_spin.width(), row.iou_spin.width())
""")
    lines = _lines(out)
    assert lines[0] == "0.01 0.01", f"步长不是 0.01：{lines[0]}"
    assert lines[1] == "0.26", f"置信度没有按 0.01 步进：{lines[1]}"
    assert lines[2] == "0.44", f"NMS 没有按 0.01 步进：{lines[2]}"
    assert lines[3] == "58 58", f"阈值输入框宽度没收到 58：{lines[3]}"


def test_real_batch_cancel_refreshes_rows_without_switching_images():
    """取消后**当场**刷新已跑完的圆点，不必切到那张图才更新。

    用户报的原始现象：「取消后没有刷新新的标签，还有切到那张图才更新圆点」。
    原因：逐张路径虽然各自写过一次盘，但取消时收尾没有再按最终磁盘状态统一
    刷一遍，于是这些行的圆点停在跑批前。
    """
    out = _run_real_qt(_MIXIN_HOST + """
import os
import tempfile

from PySide6.QtGui import QImage
from labelforge.engine.image_loader.mixin import BG_ROLE_STATUS
from labelforge.engine.image_loader.status import (
    STATUS_ANNOTATED, STATUS_UNANNOTATED)

# `_MIXIN_HOST` 里已经建好 QApplication；这里**不能再建一个**——同进程里
# 第二个 QApplication 会在构造时把进程 abort 掉（实测 rc=0xC0000409），
# 表现成「测试没输出、直接崩」。
tmp = tempfile.mkdtemp()
paths = []
for i in range(4):
    p = os.path.join(tmp, f"img{i}.jpg")
    QImage(120, 120, QImage.Format_RGB888).save(p)
    paths.append(p)

h = _Host()
h.background_images = list(paths)
h.current_background_index = 0
h.detection_boxes_dict = {i: [] for i in range(len(paths))}
h.canvas_items_dict = {i: [] for i in range(len(paths))}
# 建出列表行（真控件），否则圆点无处可刷
from labelforge.engine.image_loader.status import decorate_background_list_item
from PySide6.QtWidgets import QListWidgetItem
for i, p in enumerate(paths):
    item = QListWidgetItem(os.path.basename(p))
    decorate_background_list_item(item, p, i)
    h.background_list.addItem(item)
app.processEvents()

# 手写两个 sidecar，模拟「本轮已经跑完两张」
import json
for i in (0, 1):
    j = os.path.splitext(paths[i])[0] + ".json"
    with open(j, "w", encoding="utf-8") as f:
        json.dump({"version": "1", "shapes": [
            {"label": "cat", "shape_type": "rectangle", "x": 1, "y": 1,
             "width": 9, "height": 9, "flags": {"inference": True}}]}, f)

# 刷新前：四行都还是「未标注」
before = [h.background_list.item(r).data(BG_ROLE_STATUS)
          for r in range(h.background_list.count())]
print(before.count(STATUS_ANNOTATED))

# 走**收尾入口**（`_on_batch_finished`）而不是直接调刷新 helper：
# 用户遇到的是「取消后收尾没刷新」，要钉的是收尾有没有接上这一步，
# 绕过收尾去调 helper 的话，把收尾里的刷新删掉测试也照样绿。
from labelforge.engine.inference_task import ImageResult
results = {0: ImageResult(0, paths[0], [], None, 120, 120),
           1: ImageResult(1, paths[1], [], None, 120, 120)}
h._close_progress_dialog = lambda: None
h._apply_inference_results = lambda *a, **k: None
h.update_label_list = lambda: None
h._batch_handled_keys = {0, 1}
h._batch_written_keys = set()
h._batch_write_failed = set()
h._on_batch_finished(results)
app.processEvents()

lst = h.background_list
statuses = []
for r in range(lst.count()):
    statuses.append(lst.item(r).data(BG_ROLE_STATUS))
print(statuses.count(STATUS_ANNOTATED))
print(statuses[0] == STATUS_ANNOTATED and statuses[1] == STATUS_ANNOTATED)
print(statuses[2] == STATUS_UNANNOTATED and statuses[3] == STATUS_UNANNOTATED)
""")
    lines = _lines(out)
    assert lines[0] == "0", f"刷新前就已是已标注，用例失去意义：{lines}"
    assert lines[1] == "2", f"已写盘的两行没有当场刷成已标注：{lines}"
    assert lines[2] == "True", f"刷新的不是前两行：{lines}"
    assert lines[3] == "True", (
        f"没跑的图被误刷成已标注了：{lines}")


def test_real_batch_finish_syncs_current_canvas_from_cache():
    """取消后当前图的画布要从缓存回灌，不能停在旧框上。

    用户报的另一半现象：「要么显示旧的框」。批量跑的是当前图之后的若干张，
    取消后用户往往还停在当前图，它的内存框与缓存可能已经不一致。
    """
    out = _run_real_qt(_MIXIN_HOST + """
import os
import tempfile
import json

from PySide6.QtGui import QImage
from PySide6.QtWidgets import QListWidgetItem
from labelforge.engine.image_loader.status import decorate_background_list_item
from labelforge.engine.inference_task import ImageResult

# 同上：不能重复建 QApplication（`_MIXIN_HOST` 已建）。
tmp = tempfile.mkdtemp()
paths = []
for i in range(2):
    p = os.path.join(tmp, f"img{i}.jpg")
    QImage(120, 120, QImage.Format_RGB888).save(p)
    paths.append(p)

h = _Host()
h.background_images = list(paths)
h.current_background_index = 0
h.detection_boxes = [{"x": 1, "y": 1, "width": 5, "height": 5}]   # 旧框
h.detection_boxes_dict = {0: [
    {"x": 10, "y": 10, "width": 20, "height": 20},
    {"x": 40, "y": 40, "width": 20, "height": 20},
], 1: []}
h.canvas_items = []
h._inference_dataset = tuple(paths)
for i, p in enumerate(paths):
    item = QListWidgetItem(os.path.basename(p))
    decorate_background_list_item(item, p, i)
    h.background_list.addItem(item)

# 直接调 helper：钉「不一致才回灌、一致时不动」这条自身语义
print(len(h.detection_boxes))
h._sync_current_boxes_from_cache()
print(len(h.detection_boxes))
h._sync_current_boxes_from_cache()
print(len(h.detection_boxes))

# 再钉**收尾接线**：把内存改回旧值，走 `_on_batch_finished` 也应该被回灌。
# 只测 helper 的话，把收尾里的调用删掉测试照样绿（变异体存活过）。
h.detection_boxes = [{"x": 1, "y": 1, "width": 5, "height": 5}]
h._close_progress_dialog = lambda: None
h._apply_inference_results = lambda *a, **k: None
h.update_label_list = lambda: None
h._batch_handled_keys = {0}
h._batch_written_keys = set()
h._batch_write_failed = set()
results = {0: ImageResult(0, paths[0], [], None, 120, 120)}
h._on_batch_finished(results)
print(len(h.detection_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "1", lines[0]
    assert lines[1] == "2", f"画布没从缓存回灌（还是旧框）：{lines}"
    assert lines[2] == "2", f"一致时不该重复回灌：{lines}"
    assert lines[3] == "2", (
        f"收尾没接上画布回灌：走 `_on_batch_finished` 后画布还是旧框：{lines}")


def test_real_batch_target_has_a_prefix_label():
    """① 张数输入框要有前缀名，不能光秃秃一个数字。

    用户要求「范围推理右侧的序号加一个前缀名」——指的是「推理范围」按钮右侧
    那个序号输入框，前缀加在它自己左边。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
app = QApplication([])
row = ModelRow()
print(row.batch_to_label.text())
# 前缀要真的在行里显示，而不是造了个看不见的控件
print(row.batch_to_label.isVisibleTo(row))
print(row.batch_to_label.parent() is row)
""")
    lines = _lines(out)
    assert lines[0] == "目标张号", f"前缀名不对：{lines[0]}"
    assert lines[1] == "True", f"前缀控件没加入行布局：{lines}"
    assert lines[2] == "True", lines[2]


def test_real_batch_target_label_retranslates():
    """切语言后前缀名要跟着变，不能留在旧语言。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
from labelforge.ui import i18n
app = QApplication([])
row = ModelRow()
print(row.batch_to_label.text())
i18n.set_lang('en')
row.retranslate()
print(row.batch_to_label.text())
i18n.set_lang('zh')
""")
    lines = _lines(out)
    assert lines[0] == "目标张号", lines[0]
    assert lines[1] == "Target image", f"切语言后前缀名没跟着变：{lines[1]}"


def test_real_batch_target_defaults_to_the_last_image():
    """① 默认值 = 数据集最后一张的张号（用户明确要求）。

    换数据集后上限与默认值都要跟着走，否则会留一个指向旧数据集的张号。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
app = QApplication([])
row = ModelRow()
row.set_batch_to_range(400)
print(row.batch_to_spin.value())
row.set_batch_to_range(50)
print(row.batch_to_spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "400", f"默认值不是最后一张：{lines[0]}"
    assert lines[1] == "50", f"换数据集后没跟着走：{lines[1]}"


def test_real_batch_target_keeps_a_user_edited_value():
    """用户手动改过就不该被默认值覆盖——否则改完一换数据集就被抹掉。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
app = QApplication([])
row = ModelRow()
row.set_batch_to_range(400)
row.batch_to_spin.setValue(123)   # 模拟用户真手输
row.set_batch_to_range(400)
print(row.batch_to_spin.value())
""")
    assert _lines(out)[0] == "123", "用户改过的值被默认值覆盖了"


def test_real_batch_target_ceiling_and_default_track_a_new_dataset():
    """换数据集：上限与默认值同时跟着新数据集走。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
app = QApplication([])
row = ModelRow()
row.set_batch_to_range(400)
print(row.batch_to_spin.maximum(), row.batch_to_spin.value())
row.set_batch_to_range(30)
print(row.batch_to_spin.maximum(), row.batch_to_spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "400 400", lines[0]
    assert lines[1] == "30 30", f"上限与默认值没跟着新数据集：{lines[1]}"


# ---------- min_visible：窗口参数里的最低可见比例 ----------

def test_real_window_params_carry_min_visible():
    """窗口参数要带上「最低可见比例」，默认 0.8、按百分比显示。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget
app = QApplication([])
w = WindowParamsWidget(reference_size=(1000, 1000),
                       spec={"w": 640, "h": 640, "overlap": 200})
print(w.spec()["min_visible"])
w.set_spec({"w": 640, "h": 640, "overlap": 200, "min_visible": 0.9})
print(w.spec()["min_visible"])
print(w.min_visible_spin.value())
""")
    lines = _lines(out)
    assert lines[0] == "0.8", f"缺省不是 0.8：{lines[0]}"
    assert lines[1] == "0.9", f"没读回 spec：{lines[1]}"
    assert lines[2] == "90", f"百分比没同步：{lines[2]}"


def test_real_min_visible_is_locked_when_window_mode_is_not_selected():
    """未选中滑窗时「最低可见比例」也要禁用（与其他窗口参数一致）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget
app = QApplication([])
w = WindowParamsWidget(reference_size=(1000, 1000))
w.set_editable(False)
print(w.min_visible_spin.isEnabled())
w.set_editable(True)
print(w.min_visible_spin.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "False", f"未选中滑窗时没禁用：{lines}"
    assert lines[1] == "True", f"选中滑窗后没启用：{lines}"


def test_real_min_visible_is_never_overwritten_by_the_augment_templates():
    """数据增强的模板**不再**影响推理侧的最低可见比例（两边已脱钩）。

    早先这里叫 `test_real_min_visible_comes_from_the_template`，断言点模板会
    把 `min_visible` 一起带过来。用户要求「不与数据增强同步模板」，所以这条
    反转：共用模板里有一条 0.85 的，推理侧的默认仍必须是 0.8。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget
from labelforge.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 512, 'h': 512, 'overlap': 64,
                                  'min_visible': 0.85})
w = WindowParamsWidget(reference_size=(1000, 1000),
                       spec={"w": 640, "h": 640, "overlap": 200})
print(w.min_visible_spin.value())
print(w.spec()["min_visible"])
print(hasattr(w, '_apply_template'))
""")
    lines = _lines(out)
    assert lines[0] == "80", f"可见比例被数据增强模板带跑了：{lines}"
    assert lines[1] == "0.8", lines[1]
    assert lines[2] == "False", (
        f"`_apply_template` 还在，模板同步容易再溜回来：{lines[2]}")


def test_real_min_visible_accepts_a_value_written_by_an_older_version():
    """旧配置没这个键时退回默认，不能让整个控件崩掉。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.window_params import WindowParamsWidget
app = QApplication([])
w = WindowParamsWidget(reference_size=(1000, 1000),
                       spec={"w": 640, "h": 640, "overlap": 200})
print(w.min_visible_spin.value())
w.set_spec({"w": 640, "h": 640, "overlap": 200, "min_visible": "坏的"})
print(w.spec()["min_visible"])
""")
    lines = _lines(out)
    assert lines[0] == "80", f"旧配置应退回 80%%：{lines[0]}"
    assert lines[1] == "0.8", f"坏值应退回默认而不是崩：{lines[1]}"


# ---------- ② 三段分区：各自显示自己的内容 ----------

def test_real_panel_shows_all_three_sections_with_their_contents():
    """三段（全图/滑窗/区域）在同一个窗口里，且各自的内容都显示。

    用户要求「还在一个窗口，补全内容，分割一下」：不拆成多个弹窗，但要分成
    三块、每块显示自己的东西。未选中的段**置灰而不是隐藏**，这样用户一眼
    能看到另外两种模式当前的参数/区域，不用来回切。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_FULL

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160), region_count=1)
p.set_regions([{'x': 0, 'y': 0, 'width': 100, 'height': 100}])
p.set_mode(MODE_FULL)
p.show()
app.processEvents()
print(len(p.sections))
# 三段各自的内容都要在（不是整块显隐）
print(p.window_params.isVisibleTo(p))
print(p.region_label.isVisibleTo(p))
print(p.region_list.isVisibleTo(p))
# 未选中的段置灰
print(p.window_params.isEnabled())
print(p.region_label.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "3", f"不是三段：{lines[0]}"
    assert lines[1:4] == ["True", "True", "True"], (
        f"未选中段的控件被藏起来了，用户要求「全部显示各自内容」：{lines}")
    assert lines[4:6] == ["False", "False"], (
        f"未选中段的控件该置灰：{lines}")


def test_real_panel_sections_track_the_selected_mode():
    """选中哪段，那段就可编辑，其余置灰。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.engine.inference_scope import MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160), region_count=1)
p.set_regions([{'x': 0, 'y': 0, 'width': 100, 'height': 100}])
p.set_mode(MODE_WINDOW)
p.show()
app.processEvents()
print(p.window_params.isEnabled(), p.region_label.isEnabled())
p.set_mode(MODE_ROI)
app.processEvents()
print(p.window_params.isEnabled(), p.region_label.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "True False", f"选中滑窗时该只有滑窗可编辑：{lines[0]}"
    assert lines[1] == "False True", f"选中区域时该只有区域可编辑：{lines[1]}"


def test_real_panel_sections_are_grouped_in_three_blocks():
    """`sections` 给的是三个真实控件，且分别装着各自的内容。

    只判数量会漏掉「三段其实指着同一个控件」这种错，所以顺带核对每段
    确实包含自己那份内容。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200},
               reference_size=(3840, 2160), region_count=1)
p.set_regions([{'x': 0, 'y': 0, 'width': 100, 'height': 100}])
p.show()
app.processEvents()
secs = p.sections
ids = sorted(id(w) for w in secs.values())
print(len(secs), len(set(ids)))

def inside(widget, target):
    # target 是否在 widget 这棵子树里（用 isAncestorOf，不看名字）
    return widget is target or widget.isAncestorOf(target)

# 滑窗参数必须在**滑窗**段里；区域提示 label 必须在**区域**段里。
# 不能写成 any(...)：那样两样东西挤在同一段也会通过。
print(inside(secs['window'], p.window_params))
print(inside(secs['roi'], p.region_label))
print(inside(secs['roi'], p.region_list))
# 并且**不该**跑到别人家里去
print(inside(secs['window'], p.region_label))
print(inside(secs['roi'], p.window_params))
# 单选项分布在各自段里，但每段的内容体才是可禁用对象
print(inside(secs['full'], p._buttons['full']))
print(inside(secs['window'], p._buttons['window']))
print(inside(secs['roi'], p._buttons['roi']))
""")
    lines = _lines(out)
    assert lines[0] == "3 3", f"三段不是三个不同控件：{lines[0]}"
    assert lines[1] == "True", "滑窗参数不在「滑窗」段里"
    assert lines[2] == "True", "区域提示不在「区域」段里"
    assert lines[3] == "True", "区域明细不在「区域」段里"
    assert lines[4] == "False", "区域提示跑进了「滑窗」段"
    assert lines[5] == "False", "滑窗参数跑进了「区域」段"
    assert lines[6:9] == ["True", "True", "True"], (
        f"三个模式按钮没有分列在各自段里：{lines[6:9]}")


def test_real_panel_section_titles_follow_the_language():
    """切语言后三段的**标题**也要跟着变，不能只换按钮文字。

    段标题是 `QGroupBox.setTitle()` 设的，与 `QRadioButton.setText()` 是两套
    东西；只改按钮不改标题的话，中文界面里段标题会留着上一门语言。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel
from labelforge.ui import i18n

app = QApplication([])
p = ScopePanel()
p.show()
app.processEvents()

i18n.set_lang("zh")
p.retranslate()
app.processEvents()
print([p.sections[m].title() for m in ("full", "window", "roi")])

i18n.set_lang("en")
p.retranslate()
app.processEvents()
print([p.sections[m].title() for m in ("full", "window", "roi")])
# 标题与按钮文字必须一致（同一门语言）
print(all(p.sections[m].title() == p._buttons[m].text()
          for m in ("full", "window", "roi")))
""")
    lines = _lines(out)
    assert lines[0] == "['全图', '滑窗', '区域']", lines[0]
    assert lines[1] == "['Full image', 'Sliding window', 'Regions']", (
        f"切到英文后段标题没跟着变：{lines[1]}")
    assert lines[2] == "True", "段标题与按钮文字不是同一门语言"


def test_real_panel_open_flag_tracks_the_panel_lifetime():
    """④ 的前提：`_scope_panel_open` 必须准确跟着面板起落（真 `ImageEditor`）。

    这个标记是「面板开着时锁标签框」（Task 8）和「只在面板开着时画区域描边」
    的唯一依据：漏置/漏清会让标签框该锁的时候能编辑、或关掉面板后画布还留着
    区域框。三条关闭路径（确定 / 取消 / 直接 close）都要清干净。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(900, 700)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()

# 开面板前：必须是关的（否则「锁标签框」会一开始就误锁）
print(getattr(w, '_scope_panel_open', False))

w._open_scope_panel()
app.processEvents()
print(w._scope_panel_open)

w._scope_panel.close()
app.processEvents()
print(w._scope_panel_open)

w._open_scope_panel()
app.processEvents()
w._scope_panel.close()
app.processEvents()
print(w._scope_panel_open)

# 第三条：直接关闭（点右上角 X / close()），不经确定/取消
w._open_scope_panel()
app.processEvents()
w._scope_panel.close()
app.processEvents()
print(w._scope_panel_open)
""")
    lines = _lines(out)
    assert lines[0] == "False", f"面板还没开就已经是「开着」：{lines[0]}"
    assert lines[1] == "True", f"开面板后标记没置起来：{lines[1]}"
    assert lines[2] == "False", f"点确定后标记没清：{lines[2]}"
    assert lines[3] == "False", f"点取消后标记没清：{lines[3]}"
    assert lines[4] == "False", f"直接关闭后标记没清：{lines[4]}"


def test_real_closing_the_panel_drops_a_pending_region_draw():
    """关掉面板必须收掉「待画区域」，别一直劫持鼠标。

    待命态若漏收，关掉面板后标注模式下的拖框会被当成画区域抢走，表现为
    「关掉窗口后画不了框」。

    ⑥ 之后进入待命态的方式是**点面板上的「添加区域」**（不再是选中「区域」
    就自动进入）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(900, 700)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()

w._open_scope_panel()
app.processEvents()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
# 选中「区域」本身不该进入待画态（那会独占左键，标签框就点不到了）
print(w.canvas.is_drawing_region)

w._scope_panel.add_region_btn.click()
app.processEvents()
print(w.canvas.is_drawing_region)

w._scope_panel.close()
app.processEvents()
print(w.canvas.is_drawing_region)
print(w._scope_panel_open)
""")
    lines = _lines(out)
    assert lines[0] == "False", (
        f"选中「区域」就自动进入待画态了（会独占左键）：{lines[0]}")
    assert lines[1] == "True", f"点了「添加区域」却没进入待画态：{lines[1]}"
    assert lines[2] == "False", f"关掉面板后待画态还在：{lines[2]}"
    assert lines[3] == "False", lines[3]


def test_real_panel_flag_is_independent_of_region_drawing_state():
    """「面板开着」与「正在等画区域」是两个状态，不能互相误清。

    这两件事常被混成一个标记。方向是单向的：取消一次绘制不该动「面板开着」
    （否则区域描边会跟着失效、面板也像被关了），而面板一关则必须连带把待画
    态收掉。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(900, 700)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
w._scope_panel.add_region_btn.click()
app.processEvents()
print(w.canvas.is_drawing_region, w._scope_panel_open)

# 取消这一次绘制（等价于用户按 Esc / 右键放弃这一笔）
w.canvas._cancel_region_drawing()
app.processEvents()
# 面板标记不受影响；待画态收掉了
print(w.canvas.is_drawing_region, w._scope_panel_open)

# 关掉面板：标记也必须清
w._scope_panel.close()
app.processEvents()
print(w._scope_panel_open)
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]
    assert lines[1] == "False True", (
        f"取消绘制时误清了「面板开着」：{lines[1]}")
    assert lines[2] == "False", lines[2]


# ---------- ⑥ 面板开着时**恢复**标签框编辑（推翻第四轮的锁） ----------

def test_real_scope_panel_open_allows_label_box_editing():
    """⑥ 面板开着时，画布上的标签框照常能选中/拖动（真 `ImageEditor`）。

    用户要求「推理范围打开时恢复标签绘制」——第四轮那句「面板开着时不能编辑
    标签框」已被推翻。保护改由**每条区域自己的锁**承担（锁定的区域挡在下面
    且不可改），面板本身不再拦标签框。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
# 要有真背景图，`get_background_rect()` 才有意义（否则命中点算不准）
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
# 状态栏/信息面板会读它，缺了会 IndexError
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()

w.detection_boxes = [{'label': 'cat', 'x': 200, 'y': 200,
                      'width': 200, 'height': 200}]
w.selected_box = None
canvas.selected_box = None
canvas.selected_boxes = []
w.update_label_list = lambda: None
# 这个用例只关心「框有没有被改」，不该真去写盘（后台图是内存里造的，
# 没有真实路径）。存盘是另一条链，已有专门用例覆盖。
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None

# 框在画布上的中心点（背景图坐标 -> 画布坐标）
from PySide6.QtCore import QRectF
rect = canvas.get_background_rect()
scale = canvas.background_scale
px = int(rect.x() + 300 * scale)
py = int(rect.y() + 300 * scale)

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

# --- 面板开着：按框照样选中、照样能拖 ---
w._open_scope_panel()
app.processEvents()
before = dict(w.detection_boxes[0])
press(px, py)
print(canvas.selected_box)
print(canvas.is_dragging_box)
move(px + 80, py + 80)
release(px + 80, py + 80)
print(w.detection_boxes[0] != before)

# --- 关掉面板：同一操作结果一致（开关面板不该改变标签框的编辑能力）---
w._scope_panel.close()
app.processEvents()
after_open = dict(w.detection_boxes[0])
press(px, py)
print(canvas.selected_box)
print(canvas.is_dragging_box)
move(px + 80, py + 80)
release(px + 80, py + 80)
print(w.detection_boxes[0] != after_open)
""")
    lines = _lines(out)
    assert lines[0] == "0", (
        f"面板开着时按标签框没选中它：{lines[0]}")
    assert lines[1] == "True", (
        f"面板开着时没能进入标签框拖动：{lines[1]}")
    assert lines[2] == "True", (
        "面板开着时标签框拖不动，没做到「和面板没开时一样」")
    assert lines[3] == "0", (
        f"面板关着时按标签框没选中它：{lines[3]}")
    assert lines[4] == "True", f"面板关着时没能进入拖动：{lines[4]}"
    assert lines[5] == "True", "面板关着时拖动没生效"


def test_real_scope_panel_open_allows_drawing_a_new_label_box():
    """⑥ 面板开着时也**能新画**标签框（恢复「标签绘制」）。

    判据是「拖完有没有多出一个框」，不是看 `is_drawing_box` 标记：标注模式下
    本来就不是靠那个标记画矩形的，盯它等于空转（变异体会活下来）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w.detection_boxes = []
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
# 状态栏/信息面板会读它，缺了会 IndexError
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
# 画完矩形会弹「选标签」模态框；这里不需要人机交互，直接喂一个标签。
from labelforge.canvas import canvas_drawing as _cd
import labelforge.ui.dialogs as _dlg
_dlg.LabelSelectionDialog.select_label = staticmethod(lambda *a, **k: 'cat')

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

def drag_to_draw():
    # 先进入画框模式（真实入口：`_begin_draw_mode` 会置 `is_drawing_box`），
    # 否则 `_handle_drawing_press` 根本不会被调用，测的就不是画框那条路。
    w._begin_draw_mode('rectangle')
    app.processEvents()
    # 画矩形是**两次点击**（第二下按下的地方收尾），不是按住拖。
    press(300, 250)
    move(500, 400)
    press(500, 400)
    app.processEvents()

# --- 面板开着：在空白处拖一下就该画出框 ---
w._open_scope_panel()
app.processEvents()
drag_to_draw()
print(len(w.detection_boxes) == 1)

# --- 关掉面板：同样的拖动结果一致 ---
w._scope_panel.close()
app.processEvents()
drag_to_draw()
print(len(w.detection_boxes) == 2)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        "面板开着时画不出新标签框（用户要求恢复标签绘制）")
    assert lines[1] == "True", "面板关着时画不出框"


def test_real_panel_open_still_allows_panning():
    """例外：面板开着时**平移画布**照旧可用。

    恢复的是标签框编辑，不是把整个画布交出去——面板常浮在侧栏上，用户仍要
    拖动底图看位置；点在空白处应当进入平移。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w.detection_boxes = []
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
# 状态栏/信息面板会读它，缺了会 IndexError
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

# 面板开着 + 点图内空白：应当进入平移（不是画框、也不是被锁挡下）
w._open_scope_panel()
app.processEvents()
press(400, 350)
print(canvas.is_dragging_background)
print(not canvas.is_drawing_box)
""")
    lines = _lines(out)
    assert lines[0] == "True", "面板开着时连平移画布都被锁死了"
    assert lines[1] == "True", lines[1]


# ---------- ③ 面板开着时允许调整区域框 ----------

def test_real_panel_open_allows_dragging_a_region_box():
    """③ 面板开着时，区域框必须能拖动改位置。

    用户要求「区域框可以调整大小和位置，只在范围面板开着时允许」——面板正是
    调区域的时候，此时框不能拖就完全没法用。注意这与「锁标签框」（Task 8）
    方向相反：同一时刻标签框锁住、区域框可动。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.detection_boxes = []
w.region_fixed = False
w.region_boxes = [{'x': 300, 'y': 300, 'width': 200, 'height': 200}]

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

# 区域中心在画布上的位置（图像坐标 -> 画布坐标）
rect = canvas.get_background_rect()
sc = canvas.background_scale
cx = int(rect.x() + 400 * sc)
cy = int(rect.y() + 400 * sc)

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w.edit_mode = 'annotate'          # 面板开着时通常是标注模式
app.processEvents()

before = dict(w.region_boxes[0])
press(cx, cy)
print(canvas.is_dragging_region)
move(cx + 60, cy + 40)
release(cx + 60, cy + 40)
print(w.region_boxes[0] != before)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        "面板开着时按在区域框上没进入拖动，用户没法调整区域位置")
    assert lines[1] == "True", "面板开着时拖动区域框没有生效"


def test_real_panel_open_allows_resizing_a_region_box():
    """③ 区域框还要能**改大小**（拖右下角手柄）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.detection_boxes = []
w.region_fixed = False
w.region_boxes = [{'x': 300, 'y': 300, 'width': 200, 'height': 200}]

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

rect = canvas.get_background_rect()
sc = canvas.background_scale
# 区域右下角 (500,500) 的手柄
hx = int(rect.x() + 500 * sc)
hy = int(rect.y() + 500 * sc)

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w.edit_mode = 'annotate'
app.processEvents()

before = dict(w.region_boxes[0])
press(hx, hy)
print(canvas.is_resizing_region)
move(hx + 50, hy + 50)
release(hx + 50, hy + 50)
after = w.region_boxes[0]
print(after['width'] > before['width'] and after['height'] > before['height'])
""")
    lines = _lines(out)
    assert lines[0] == "True", "面板开着时按区域右下角没进入缩放"
    assert lines[1] == "True", "面板开着时区域框尺寸没被改大"


def test_real_region_is_not_editable_when_the_panel_is_closed():
    """对照：面板关着、且是标注模式时，区域框不该被动（别锁反了）。

    用户明说「只在范围面板开着时允许」——关掉面板后标注模式下点区域框，
    不能把它拖走（那会误改用户画好的区域）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.detection_boxes = []
w.region_fixed = False
w.region_boxes = [{'x': 300, 'y': 300, 'width': 200, 'height': 200}]

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

rect = canvas.get_background_rect()
sc = canvas.background_scale
cx = int(rect.x() + 400 * sc)
cy = int(rect.y() + 400 * sc)

w.edit_mode = 'annotate'          # 标注模式，面板**没开**
app.processEvents()
print(w._scope_panel_open)

before = dict(w.region_boxes[0])
press(cx, cy)
move(cx + 60, cy + 40)
release(cx + 60, cy + 40)
print(w.region_boxes[0] == before)
""")
    lines = _lines(out)
    assert lines[0] == "False", f"前置不成立，面板不该是开着的：{lines[0]}"
    assert lines[1] == "True", (
        "面板关着时标注模式下竟然把区域框拖走了（应当不允许）")


def test_real_closing_the_panel_mid_drag_does_not_corrupt_the_region():
    """拖动区域框中途关掉面板：不能崩，也不能把区域搞成半截状态。

    面板是可以用快捷键/X 随时关掉的，拖动过程中关掉属于正常操作。这里只要求
    「不崩 + 区域还是个合法框」，不要求把拖动结果回滚——用户可能正是想边看
    边调。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.detection_boxes = []
w.region_fixed = False
w.region_boxes = [{'x': 300, 'y': 300, 'width': 200, 'height': 200}]

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

rect = canvas.get_background_rect()
sc = canvas.background_scale
cx = int(rect.x() + 400 * sc)
cy = int(rect.y() + 400 * sc)

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w.edit_mode = 'annotate'
app.processEvents()

press(cx, cy)
move(cx + 60, cy + 40)
# 拖动中途关面板（模拟用户按 Esc / 点 X）
w._scope_panel.close()
app.processEvents()
print(w._scope_panel_open)
# 再继续移动并松手：不应该抛异常
move(cx + 90, cy + 70)
release(cx + 90, cy + 70)
app.processEvents()
print(w.canvas.is_dragging_region)

region = w.region_boxes[0]
print(region['width'] > 0 and region['height'] > 0)
print(set(region) >= {'x', 'y', 'width', 'height'})
""")
    lines = _lines(out)
    assert lines[0] == "False", f"关面板没生效：{lines[0]}"
    assert lines[1] == "False", f"关面板后拖动状态没收干净：{lines[1]}"
    assert lines[2] == "True", f"区域尺寸被搞坏了：{lines}"
    assert lines[3] == "True", f"区域字段缺失：{lines}"


def test_real_panel_window_edits_reach_the_run_scope():
    """面板里改的窗口参数必须真的进到本次任务的范围里。

    用户报过「面板里改了参数，跑起来还是老样子」这类问题的根因就在这：`_scope_for_run`
    读的是「本次会话设过的 / 配置里存的 / 默认值」（`_effective_window_spec`），
    而面板里**当下**的输入是记在 `_scope_window_spec`——两者没关系，于是用户在
    面板里改了宽/高/重叠/最低可见比例，任务仍然用旧值，界面上却显示新值。

    这条把面板 → 范围快照这条链接上（含 `min_visible`，它是新加的一项）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
w = ImageEditor()
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w._open_scope_panel()
w._scope_panel.set_mode(MODE_WINDOW)
app.processEvents()
wp = w._scope_panel.window_params
wp._w.setValue(1234)
wp._h.setValue(567)
wp._ov.setValue(88)
wp.min_visible_spin.setValue(85)
app.processEvents()

spec = w._scope_for_run()['windows']
print(spec['w'], spec['h'], spec['overlap'])
print(spec.get('min_visible'))
""")
    lines = _lines(out)
    assert lines[0] == "1234 567 88", (
        f"面板里改的窗口尺寸没进到任务范围里（跑起来还是旧值）：{lines[0]}")
    assert lines[1] == "0.85", (
        f"面板里改的最低可见比例没进到任务范围里：{lines[1]}")


def test_real_default_window_spec_carries_min_visible():
    """没有任何已存参数时，兜底的默认窗口表也必须带 `min_visible`。

    默认表少了这一项，`_normalize_window_spec` 只好补默认值——今天恰好也是
    0.7 所以看不出问题，但一旦两处默认值分家，用户会莫名其妙拿到另一个值。
    这里钉住「默认表与界面默认一致」。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.widgets.window_params import DEFAULT_MIN_VISIBLE

app = QApplication([])
w = ImageEditor()
w.show()
app.processEvents()
# 清掉「本次会话设过」与配置里存的，逼它走默认分支
w._window_spec = None
w._saved_window_spec = lambda: None
app.processEvents()

spec = w._effective_window_spec()
print('min_visible' in spec)
print(spec.get('min_visible'))
print(DEFAULT_MIN_VISIBLE)
""")
    lines = _lines(out)
    assert lines[0] == "True", "默认窗口表没带 min_visible"
    assert lines[1] == lines[2], (
        f"默认表里的 min_visible（{lines[1]}）与界面默认（{lines[2]}）不一致")


# ---------- ③ 面板改动即时生效（无确定/取消）----------

def test_real_scope_panel_has_no_ok_or_cancel_button():
    """③ 面板不再有「确定/取消」：改了就是改了，不需要点按钮确认。

    用户要求「选中某推理方式立马刷新不需要点击确定，可以把确定取消按钮删除」。
    按钮删掉后，唯一剩下的关闭方式是右上角 X（`close()`）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.scope_panel import ScopePanel

app = QApplication([])
p = ScopePanel()
p.show()
app.processEvents()
print(hasattr(p, 'ok_btn'), hasattr(p, 'cancel_btn'))
# 按钮文字也不该还留在任何子控件上
texts = [w.text() for w in p.findChildren(type(p.findChild(
    __import__('PySide6.QtWidgets', fromlist=['QPushButton']).QPushButton)))]
print('确定' in texts, '取消' in texts)
""")
    lines = _lines(out)
    assert lines[0] == "False False", (
        f"面板上还留着确定/取消按钮：{lines[0]}")
    assert lines[1] == "False False", (
        f"按钮控件删了但文字还在：{lines[1]}")


def test_real_switching_mode_applies_without_confirming():
    """③ 换推理方式立刻写入宿主状态，不必点确定。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW, MODE_ROI

app = QApplication([])
w = ImageEditor()
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w._on_model_loaded(w._detector, (640, 640), 'model')
w.resize(900, 700)
w.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
w.region_fixed = False
# 默认（还没开过面板）是「全图」；`_scope_mode` 要等模型加载才建，用 getattr
print(getattr(w, '_scope_mode', 'full'))

w._open_scope_panel()
app.processEvents()
print(w._scope_mode)

w._scope_panel.set_mode(MODE_WINDOW)
app.processEvents()
print(w._scope_mode)

w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
print(w._scope_mode)

# 不关面板，直接问「跑的时候用哪个范围」
print(w._scope_for_run()['mode'])
""")
    lines = _lines(out)
    assert lines[0] == "full", lines
    assert lines[1] == "full", f"开面板不该改动状态：{lines}"
    assert lines[2] == "window", (
        f"切到滑窗没立刻生效（还要点确定）：{lines[2]}")
    assert lines[3] == "roi", f"切到区域没立刻生效：{lines[3]}"
    assert lines[4] == "roi", f"跑起来没用刚选的模式：{lines[4]}"


def test_real_window_edits_apply_without_confirming():
    """③ 滑窗参数改了立刻生效：不点确定，`_scope_for_run()` 就用新值。

    这条把上一轮「面板参数要进到任务范围」（`_window_spec_for_run`）从
    「面板开着」升级为「面板开着且即时」——不再依赖任何确认动作。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
w = ImageEditor()
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w._open_scope_panel()
w._scope_panel.set_mode(MODE_WINDOW)
app.processEvents()

wp = w._scope_panel.window_params
wp._w.setValue(1234)
wp._h.setValue(567)
wp._ov.setValue(88)
wp.min_visible_spin.setValue(85)
app.processEvents()

spec = w._scope_for_run()['windows']
print(spec['w'], spec['h'], spec['overlap'])
print(spec['min_visible'])
# 宿主侧的窗口参数也要跟着更新（关掉面板后仍要用这份）
print(w._window_spec['w'], w._window_spec['overlap'])
""")
    lines = _lines(out)
    assert lines[0] == "1234 567 88", f"改的参数没即时生效：{lines[0]}"
    assert lines[1] == "0.85", f"min_visible 没即时生效：{lines[1]}"
    assert lines[2] == "1234 88", (
        f"宿主的窗口参数没随面板更新（关面板后会退回旧值）：{lines[2]}")


# ---------- 第五轮：参数行宽度 / 分隔符 / conf 标签 ----------

def test_real_batch_target_spin_is_a_quarter_narrower():
    """① 「目标张号」输入框宽度减到原来的 3/4。

    用户要求「目标张号输入框宽度减少 1/4」：原宽 84 → 63。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
print(row.batch_to_spin.width())
print(row.batch_to_spin.minimumWidth(), row.batch_to_spin.maximumWidth())
""")
    lines = _lines(out)
    assert lines[0] == "63", (
        f"目标张号输入框宽度不是 63（84 的 3/4）：{lines[0]}")
    assert lines[1] == "63 63", (
        f"宽度没真正钉住，布局还能把它撑开：{lines[1]}")


def test_real_input_size_spins_match_threshold_width():
    """输入尺寸的两个输入框与 conf / NMS **等宽**。

    用户要求「输入尺寸的按钮宽度和 conf、NMS 保持一致」。原来是写死的 72px，
    同一行里并排就参差不齐；现在四者共用 `_THRESHOLD_SPIN_WIDTH`。

    同时钉住**最长的值还看得全**：8192 是范围上限（4 位数），收窄到 58px
    之后不能把它截断成省略号——那样就成「改了宽度、坏了读数」。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.resize(1600, 40)
row.show()
app.processEvents()
row.set_size_fields_visible(True)
app.processEvents()

print(row.conf_spin.width(), row.iou_spin.width())
print(row.size_spin_w.width(), row.size_spin_h.width())
row.size_spin_w.setValue(1280)
row.size_spin_h.setValue(8192)
app.processEvents()
print(row.size_spin_w.text())
print(row.size_spin_h.text())
""")
    lines = _lines(out)
    conf, iou = (int(v) for v in lines[0].split())
    size_w, size_h = (int(v) for v in lines[1].split())
    assert conf == iou, f"conf 与 NMS 本身就不等宽：{lines[0]}"
    assert (size_w, size_h) == (conf, iou), (
        f"输入尺寸没和 conf/NMS 对齐：{lines[1]} vs {lines[0]}")
    # 收窄不能把值截掉（Qt 在放不下时会显示成省略/截断的文本）
    assert lines[2] == "1280", f"输入尺寸显示被截断了：{lines[2]}"
    assert lines[3] == "8192", f"上限值显示被截断了：{lines[3]}"


def test_real_param_row_groups_are_separated_by_dashes():
    """⑤ 参数行各功能之间用 `-` 隔开（共 7 组，置信度与 NMS 同组）。

    组：加载模型+模型名 | 置信度+NMS | 输入尺寸 | 覆盖/补充 | 类别筛选
        | 推理范围+摘要 | 目标张号
    所以中间应有 **6** 个分隔符，且「置信度」和「NMS」之间**没有**分隔符。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QLabel
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()

# 按布局里的实际顺序收集所有 QLabel 的文字
order = []
lay = row.layout()
for i in range(lay.count()):
    w = lay.itemAt(i).widget()
    if isinstance(w, QLabel):
        order.append(w.text())
print(order)
print(sum(1 for t in order if t.strip() == '-'))
# conf 与 NMS 之间只隔一个数字输入框（QDoubleSpinBox），没有分隔符
ci = order.index('conf')
ni = order.index('NMS')
print(ci, ni)
""")
    lines = _lines(out)
    order = lines[0]
    assert "-" in order, f"参数行里没有 '-' 分隔符：{order}"
    assert lines[1] == "6", (
        f"分隔符不是 6 个（7 组之间有 6 个间隔）：{lines[1]}")
    # conf 与 NMS 之间只有一个控件（conf 的输入框），说明它们同组
    ci, ni = (int(v) for v in lines[2].split())
    assert ni - ci == 1, (
        f"conf 与 NMS 被拆开了，应当同组：{lines[2]}")


def test_real_static_model_hides_input_size_without_doubling_dashes():
    """静态轴模型收起「输入尺寸」时，不许留下相邻的两条 `-`。

    用户看到 `NMS [0.45] - - 覆盖/补充` 报「NMS 后面有两个 -，删除一个」。
    病根不是多插了分隔符，而是这组控件收起后，它**左边**那条分隔符还露着，
    和**右边**那条挨在了一起。所以断言「任何相邻两条分隔符之间都有实际
    内容」，而不是数分隔符个数——数个数反而会在动态模型下误判。
    """
    script = """
from PySide6.QtWidgets import QApplication, QLabel
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.resize(1600, 40)
row.show()
app.processEvents()


def visible_labels():
    out = []
    lay = row.layout()
    for i in range(lay.count()):
        w = lay.itemAt(i).widget()
        if w is None or not w.isVisible():
            continue
        if isinstance(w, QLabel):
            out.append(w.text())
        else:
            out.append('<widget>')
    return out


# 静态轴模型 / 未加载模型：整组「输入尺寸」是收起的
row.set_size_fields_visible(False)
app.processEvents()
static = visible_labels()
print(static)

# 动态轴模型：整组「输入尺寸」显示，两条分隔符中间隔着这组控件
row.set_size_fields_visible(True)
app.processEvents()
print(visible_labels())
"""
    out = _run_real_qt(script)
    lines = _lines(out)

    static = ast.literal_eval(lines[0])
    dynamic = ast.literal_eval(lines[1])

    # 静态模型下看不到「输入尺寸」，而且**没有**任何一对相邻的 `-`
    assert "输入尺寸" not in static, f"静态模型不该显示输入尺寸：{static}"
    adjacent = [(a, b) for a, b in zip(static, static[1:])
                if a.strip() == "-" and b.strip() == "-"]
    assert not adjacent, f"静态模型下出现了相邻的两条 '-'：{static}"

    # 动态模型下这组要回来，仍不许有相邻的两条 `-`
    assert "输入尺寸" in dynamic, f"动态模型应当显示输入尺寸：{dynamic}"
    adjacent = [(a, b) for a, b in zip(dynamic, dynamic[1:])
                if a.strip() == "-" and b.strip() == "-"]
    assert not adjacent, f"动态模型下出现了相邻的两条 '-'：{dynamic}"


def test_real_conf_label_reads_conf_in_both_languages():
    """⑤ 「置信度」的默认标签中英都显示 `conf`。

    用户要求「置信度默认中英都是 conf」。视频弹窗那个同名复选框不受影响
    （它用的是另一套键），所以这里只钉参数行。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui import i18n
from labelforge.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
i18n.set_lang("zh")
row.retranslate()
print(row.conf_lbl.text())
i18n.set_lang("en")
row.retranslate()
print(row.conf_lbl.text())
""")
    lines = _lines(out)
    assert lines[0] == "conf", f"中文下置信度标签不是 conf：{lines[0]}"
    assert lines[1] == "conf", f"英文下置信度标签不是 conf：{lines[1]}"


def test_real_video_overlay_confidence_label_is_untouched():
    """⑤ 的护栏：视频弹窗的「置信度」复选框**不**跟着变成 conf。

    它和参数行原先共用 `tr("置信度")`。给参数行换成新键之后，这一处必须
    保持原样，否则就是「改一处坏一处」。只查中文即可——`置信度` 这个键的
    英文值本来就是 `Confidence`（`i18n` 里另有一条），改键名才不会连累它。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QCheckBox
from labelforge.ui import i18n
from labelforge.ui.video_dialog import VideoDialog

app = QApplication([])
i18n.set_lang("zh")
d = VideoDialog()
print([cb.text() for cb in d.findChildren(QCheckBox)])
""")
    lines = _lines(out)
    assert "置信度" in lines[0], (
        f"视频弹窗的「置信度」被连带改掉了：{lines[0]}")
    assert "conf" not in lines[0], (
        f"视频弹窗也变成 conf 了：{lines[0]}")


def test_real_canvas_grid_labels_each_window_with_row_col():
    """② 滑窗模式下每个窗口左下角标 `r1c1`（同数据增强的口径）。

    这条直接**数出画的字**而不是数像素：给画布装一个假的 painter，把
    `drawText` 的调用全记下来，检查标签文字与所在格子对得上。数像素会因为
    字体/抗锯齿而不稳，数调用才是「真的标了号」。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()

img = QImage(1600, 1200, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
app.processEvents()

# 1600x1200 配 800x600、重叠 0 → 2 行 x 2 列 = 4 窗
w._scope_window_spec = {'w': 800, 'h': 600, 'overlap': 0}
w._scope_mode = MODE_WINDOW
w._scope_panel_open = True

canvas = w.canvas
canvas.resize(1000, 700)
app.processEvents()

texts = []
class Spy(QPainter):
    def drawText(self, *args):
        texts.append(str(args[-1]))

# `_paint_scene` 内部自己 new 了一个 QPainter，外面塞的 painter 拦不到它，
# 所以这里直接驱动 `_draw_scope_grid`（它就是真正画格子+标号的那一层）。
im = QImage(canvas.size(), QImage.Format_ARGB32)
p = Spy(im)
canvas._draw_scope_grid(p, canvas.get_background_rect())
p.end()
grid_texts = sorted(t for t in texts if t.startswith('r') and 'c' in t)
print(grid_texts)
""")
    lines = _lines(out)
    labels = lines[0]
    assert "'r1c1'" in labels, (
        f"画布上没标出 r1c1：{labels}")
    assert "'r2c2'" in labels, (
        f"2x2 的网格没标到 r2c2：{labels}")
    assert lines[0].count("'r") == 4, (
        f"4 个窗口应该标 4 个号：{lines[0]}")


def test_real_canvas_grid_labels_shift_with_window_params():
    """② 改窗口参数后，标号跟着重排（不是只画一次就定住）。

    1600x1200 配 400x300、重叠 0 → 4 行 x 4 列 = 16 窗，最大号 `r4c4`。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPainter, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_WINDOW

app = QApplication([])
w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()
img = QImage(1600, 1200, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
app.processEvents()

w._scope_mode = MODE_WINDOW
w._scope_panel_open = True
# 1600x1200 配 400x300、重叠 0 → 4 行 x 4 列 = 16 窗
w._scope_window_spec = {'w': 400, 'h': 300, 'overlap': 0}
canvas = w.canvas
canvas.resize(1000, 700)
app.processEvents()

texts = []
class Spy(QPainter):
    def drawText(self, *args):
        texts.append(str(args[-1]))

im = QImage(canvas.size(), QImage.Format_ARGB32)
p = Spy(im)
canvas._draw_scope_grid(p, canvas.get_background_rect())
p.end()
all_lbl = sorted(t for t in texts if t.startswith('r') and 'c' in t)
print(all_lbl[0], all_lbl[-1], len(all_lbl))
""")
    lines = _lines(out)
    # 1600x1200 配 400x300、重叠 0 → 4 行 x 4 列 = 16 窗
    assert lines[0] == "r1c1 r4c4 16", (
        f"标号没跟着窗口参数重排：{lines[0]}")


# ---------- ③ 区域框缩放：四边手柄 + 更大的命中区 ----------

_REGION_RESIZE_HOST = """
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QMouseEvent, QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_scope import MODE_ROI

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
canvas = w.canvas
canvas.resize(900, 700)
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.detection_boxes = []
w.region_fixed = False
w.region_boxes = [{'shape_type': 'rectangle', 'x': 300, 'y': 300,
                   'width': 200, 'height': 200}]

def press(x, y):
    canvas.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

def move(x, y):
    canvas.mouseMoveEvent(QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y), Qt.NoButton,
        Qt.LeftButton, Qt.NoModifier))

def release(x, y):
    canvas.mouseReleaseEvent(QMouseEvent(
        QMouseEvent.MouseButtonRelease, QPointF(x, y), Qt.LeftButton,
        Qt.NoButton, Qt.NoModifier))

rect = canvas.get_background_rect()
sc = canvas.background_scale

def C(ix, iy):
    # 图像坐标 -> 画布坐标
    return int(rect.x() + ix * sc), int(rect.y() + iy * sc)

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w.edit_mode = 'annotate'
app.processEvents()
"""


def test_real_region_has_four_edge_handles():
    """③ 区域框有**四条边**的缩放手柄，不只是四角。

    用户要求「复用检测框的缩放，单边调整和角点拖动」：只拖一条边时，只有
    那一个方向应该变，另一个方向保持不变。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
# 右边中点：只把宽度拖大 60 像素，高度必须纹丝不动
rx, ry = C(500, 400)
press(rx, ry)
print(canvas.is_resizing_region, canvas.is_dragging_region)
move(rx + 60, ry + 40)
release(rx + 60, ry + 40)
r = w.region_boxes[0]
print(round(r['width']), round(r['height']))
print(round(r['x']), round(r['y']))
""")
    lines = _lines(out)
    assert lines[0] == "True False", (
        f"按在区域右边中点没进入缩放（当成拖动了）：{lines[0]}")
    # 宽度变大了，高度与 x/y 都该保持（单边调整只改一个方向）
    w0, h0 = (int(v) for v in lines[1].split())
    x0, y0 = (int(v) for v in lines[2].split())
    assert w0 > 200, f"拖右边没把宽度拖大：{lines[1]}"
    assert h0 == 200, f"拖右边不该改高度：{lines[1]}"
    assert (x0, y0) == (300, 300), f"拖右边不该改 x/y：{lines[2]}"


def test_real_region_bottom_edge_resizes_height_only():
    """③ 下边中点：只改高度，宽度与 x/y 不动。"""
    out = _run_real_qt(_REGION_RESIZE_HOST + """
bx, by = C(400, 500)
press(bx, by)
print(canvas.is_resizing_region)
move(bx + 50, by + 70)
release(bx + 50, by + 70)
r = w.region_boxes[0]
print(round(r['width']), round(r['height']))
print(round(r['x']), round(r['y']))
""")
    lines = _lines(out)
    assert lines[0] == "True", f"按在下边中点没进入缩放：{lines[0]}"
    w0, h0 = (int(v) for v in lines[1].split())
    x0, y0 = (int(v) for v in lines[2].split())
    assert h0 > 200, f"拖下边没把高度拖大：{lines[1]}"
    assert w0 == 200, f"拖下边不该改宽度：{lines[1]}"
    assert (x0, y0) == (300, 300), f"拖下边不该改 x/y：{lines[2]}"


def test_real_region_left_edge_resizes_width_and_x_only():
    """③ 左边中点：改宽度与 x，高度与 y 不动。"""
    out = _run_real_qt(_REGION_RESIZE_HOST + """
lx, ly = C(300, 400)
press(lx, ly)
print(canvas.is_resizing_region)
move(lx - 40, ly + 30)
release(lx - 40, ly + 30)
r = w.region_boxes[0]
print(round(r['width']), round(r['height']))
print(round(r['x']), round(r['y']))
""")
    lines = _lines(out)
    assert lines[0] == "True", f"按在左边中点没进入缩放：{lines[0]}"
    w0, h0 = (int(v) for v in lines[1].split())
    x0, y0 = (int(v) for v in lines[2].split())
    assert w0 > 200, f"拖左边没把宽度拖大：{lines[1]}"
    assert h0 == 200, f"拖左边不该改高度：{lines[1]}"
    assert x0 < 300, f"拖左边没跟着改 x：{lines[2]}"
    assert y0 == 300, f"拖左边不该改 y：{lines[2]}"


def test_real_region_handle_size_is_constant_box_plus_three():
    """区域手柄 = 检测框 + 3，且**不随缩放变化**（用户第 12 轮口径）。

    这条替换了早先的「命中区随缩放放大」：那个做法图一缩小就把手柄放大到
    24px，于是区域句柄看着比检测框大两三倍，用户原话「我参数拉到 15 都没你的
    区域框句柄大」。现在两边同一套尺度——恒定像素、随图缩放而缩放，只差 3。

    绘制尺寸与命中尺寸必须同源，否则会出现「看着能抓却抓不到」。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from labelforge.core.config import DETECTION_BOX_CONFIG, REGION_HANDLE_EXTRA

box = DETECTION_BOX_CONFIG['resize_handle_size']
sizes = []
for sc in (1.0, 0.5, 0.25, 0.1):
    canvas.background_scale = sc
    sizes.append(round(canvas._region_handle_hit_size(), 2))
canvas.background_scale = 0.25
print('sizes', sizes, box + REGION_HANDLE_EXTRA)
# 绘制与命中必须一致
print('draw_hit', round(canvas._region_handle_draw_size()),
      round(canvas._region_handle_hit_size()))
""")
    lines = _lines(out)
    sizes_line = next(x for x in lines if x.startswith("sizes"))
    draw_line = next(x for x in lines if x.startswith("draw_hit"))
    # sizes_line 形如 `sizes [11, 11, 11, 11] 11`，列表里有空格，
    # 不能按空白切分——用正则把列表和末尾的数字分开取。
    import re as _re
    match = _re.search(r"\[([^\]]*)\]\s+(\S+)\s*$", sizes_line)
    assert match, f"解析不了这行：{sizes_line}"
    sizes = [float(v) for v in match.group(1).split(",")]
    expected = float(match.group(2))
    assert sizes == [expected] * 4, (
        f"区域手柄应恒为 {expected}px、不随缩放变，实际 {sizes}")
    assert draw_line.split()[1:] == [str(int(expected))] * 2, (
        f"手柄「看到的」和「抓得到的」必须一致：{draw_line}")


def test_real_region_resize_never_shrinks_below_min_size():
    """③ 边手柄也要守住最小尺寸，不能把区域拖成 0。"""
    out = _run_real_qt(_REGION_RESIZE_HOST + """
rx, ry = C(500, 400)
press(rx, ry)
move(rx - 500, ry)          # 往左猛拖，想拖成负宽
release(rx - 500, ry)
r = w.region_boxes[0]
print(r['width'] >= 10)
""")
    lines = _lines(out)
    assert lines[0] == "True", f"区域被拖到小于最小尺寸了：{lines[0]}"


# ---------- 第六轮：区域逐条锁定 + 面板开着时恢复标签绘制 ----------

_REGION_LOCK_HOST = _REGION_RESIZE_HOST + """
def region_center(index=0):
    r = w.region_boxes[index]
    return C(r['x'] + r['width'] / 2, r['y'] + r['height'] / 2)
"""


def test_real_region_lock_round_trip_and_default():
    """区域默认不锁；锁上/解开能往返，且只动被点名的那一条。"""
    out = _run_real_qt(_REGION_LOCK_HOST + """
w.region_boxes.append({'shape_type': 'rectangle', 'x': 100, 'y': 100,
                       'width': 50, 'height': 50})
print(canvas.region_locked(w.region_boxes[0]), canvas.region_locked(w.region_boxes[1]))
print(canvas.set_region_locked(0, True))
print(canvas.region_locked_at(0), canvas.region_locked_at(1))
print(w.region_boxes[0].get('locked'), w.region_boxes[1].get('locked'))
print(canvas.set_region_locked(0, True))   # 已经是锁的，不算改
print(canvas.set_region_locked(0, False))
print(canvas.region_locked_at(0))
print(canvas.set_region_locked(99, True))  # 越界是空操作
""")
    lines = _lines(out)
    assert lines[0] == "False False", f"区域默认应当是不锁的：{lines[0]}"
    assert lines[1] == "True", f"锁上没报告「改了」：{lines[1]}"
    assert lines[2] == "True False", (
        f"锁定串到了别的区域上：{lines[2]}")
    assert lines[3] == "True None", (
        f"锁只该写在被点名那条身上：{lines[3]}")
    assert lines[4] == "False", f"重复锁定不该再报「改了」：{lines[4]}"
    assert lines[5] == "True", f"没能解锁：{lines[5]}"
    assert lines[6] == "False", f"解锁后 region_locked_at 没跟上：{lines[6]}"
    assert lines[7] == "False", f"越界下标应当返回 False：{lines[7]}"


def test_real_locked_region_yields_clicks_to_the_box_below():
    """锁定的区域「最底层、不可修改」：点上去要穿透，落到下面的检测框。

    用户要求「不锁定默认最上层与检测框一样可修改，锁定后最底层不可修改」。
    所以锁定后 `_region_at` 要够不着它，而它下面那张检测框要被选中。
    """
    out = _run_real_qt(_REGION_LOCK_HOST + """
# 区域在 (300,300)-(500,500)；叠一张检测框压住同一块地方
w.detection_boxes = [{'label': 'cat', 'x': 320, 'y': 320,
                      'width': 160, 'height': 160}]
w.update_label_list = lambda: None
canvas._after_detection_boxes_changed() if hasattr(
    canvas, '_after_detection_boxes_changed') else None
app.processEvents()

cx, cy = region_center()
print(canvas._region_at(QPointF(cx, cy)))

canvas.set_region_locked(0, True)
app.processEvents()
print(canvas._region_at(QPointF(cx, cy)))
# 纯几何查询（拖出文件的守卫）仍然要看得见它，与能否编辑无关
print(canvas._region_at(QPointF(cx, cy), include_locked=True))

# 点一下：应当选中下面那张检测框，而不是区域
press(cx, cy)
app.processEvents()
print(canvas.selected_box)
print(canvas.selected_region)
release(cx, cy)
""")
    lines = _lines(out)
    assert lines[0] == "0", f"未锁定的区域应当能被命中：{lines[0]}"
    assert lines[1] == "None", (
        f"锁定的区域还能被点中（应当穿透到下面的框）：{lines[1]}")
    assert lines[2] == "0", (
        f"纯几何查询被锁挡住了，往外拖会被误判成空白：{lines[2]}")
    assert lines[3] == "0", (
        f"锁定的区域没把点击让给下面的检测框：{lines[3]}")
    assert lines[4] == "None", (
        f"锁定的区域仍被选中了：{lines[4]}")


def test_real_unlocked_region_wins_over_the_box_it_covers():
    """未锁区域在检测框**之上**，重叠处点击归区域（与上一条互为对照）。"""
    out = _run_real_qt(_REGION_LOCK_HOST + """
w.detection_boxes = [{'label': 'cat', 'x': 320, 'y': 320,
                      'width': 160, 'height': 160}]
w.update_label_list = lambda: None
app.processEvents()

cx, cy = region_center()
press(cx, cy)
app.processEvents()
print(canvas.selected_region)
print(canvas.selected_box)
release(cx, cy)
""")
    lines = _lines(out)
    assert lines[0] == "0", (
        f"未锁定的区域没能优先于检测框拿到点击：{lines[0]}")
    assert lines[1] == "None", (
        f"重叠时检测框也被选中了，两个不该同时动：{lines[1]}")


_LAYER_ORDER_HOST = """
from PySide6.QtWidgets import QApplication, QWidget, QLabel
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from labelforge.canvas import Canvas
from labelforge.canvas import canvas_renderer as _cr

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
        self.show_grid_checkbox = type('C', (), {'isChecked': lambda self: False})()
        self.show_paste_names_checkbox = type('C', (), {'isChecked': lambda self: False})()
        self.shape_opacity = 1.0
        self.mouse_inside = False
        self.is_drawing_box = False
        self.is_drawing_region = False
        self.is_drawing_region_polygon = False
        self.is_drawing_polygon = False
        self.selected_region = None
        self.selected_box = None
        self.selected_boxes = []
        # 区域轮廓要画出来：标注模式下靠这个开关（面板开着时的提示层）
        self._show_scope_regions = True
    def save_undo_state(self):
        pass
    def get_image_info(self):
        return {'box_count': 0, 'paste_count': 0, 'width': 400, 'height': 300}
    def get_label_stats(self):
        return {}
    def get_label_color(self, label):
        return '#ff0000'

app = QApplication([])
editor = Editor()
canvas = Canvas(editor)
canvas.resize(400, 300)
editor.current_background = QPixmap(400, 300)

# 记下三个绘制步骤的调用顺序：`_paint_scene` 内部自己 new 一个 QPainter，
# 外层塞进去的假 painter 拦不到任何东西，所以只能换成给**方法**打桩。
order = []
for name in ('_draw_regions', '_draw_paste_items', '_draw_detection_boxes'):
    def make(name):
        orig = getattr(canvas, name)
        def spy(self, painter, rect, *a, **k):
            order.append(name)
            return orig(painter, rect, *a, **k)
        return spy
    setattr(canvas, name, make(name).__get__(canvas, type(canvas)))

def draw():
    order.clear()
    canvas._paint_scene(None)
"""


def test_real_locked_regions_draw_below_boxes_and_unlocked_above():
    """未锁区域画在检测框**之上**，锁定的画在**之下**。

    用户要求「不锁定默认最上层与检测框一样可修改，锁定后最底层不可修改」。
    层级必须与点击优先级一致（未锁优先拿到点击），否则会出现「看着在上面、
    点下去却选中了别的框」。

    这里数的是三个绘制步骤的**调用顺序**，不是像素：叠在同一个位置的两条
    区域/框，靠像素颜色分不出谁压谁，方法顺序才是「先画谁」的直接证据。
    """
    out = _run_real_qt(_LAYER_ORDER_HOST + """
editor.region_boxes = [
    {'shape_type': 'rectangle', 'x': 10, 'y': 10, 'width': 50, 'height': 50},
    {'shape_type': 'rectangle', 'x': 20, 'y': 20, 'width': 50, 'height': 50,
     'locked': True},
]
editor.detection_boxes = [{'label': 'a', 'x': 15, 'y': 15,
                           'width': 40, 'height': 40,
                           'shape_type': 'rectangle'}]
draw()
# 简化成「区域标记」列表：两次 _draw_regions 中间夹着框和贴图
print([n.replace('_draw_', '') for n in order])
print(order.index('_draw_detection_boxes'))
""")
    lines = _lines(out)
    names = [n.strip().strip("'") for n in lines[0].strip("[]").split(",")]
    assert names.count("regions") == 2, (
        f"区域没有分两趟画：{lines[0]}")
    first_regions = names.index("regions")
    second_regions = names.index("regions", first_regions + 1)
    box_at = names.index("detection_boxes")
    assert first_regions < box_at < second_regions, (
        f"层级不对，应当「锁定区域 → 检测框 → 未锁区域」：{lines[0]}")


def test_real_locked_region_is_the_one_drawn_first():
    """两次 `_draw_regions` 的先后不能颠倒：先画的是**锁定**的那趟。

    上一条只证明「夹着框」，这条钉住夹的顺序（否则第一趟画未锁的也能满足
    `first < box < second`，层级就反了）。
    """
    out = _run_real_qt(_LAYER_ORDER_HOST + """
locks = []
orig = canvas._draw_regions
def spy(painter, rect, locked_only=None):
    locks.append(locked_only)
    return orig(painter, rect, locked_only)
canvas._draw_regions = spy

editor.region_boxes = [{'shape_type': 'rectangle', 'x': 10, 'y': 10,
                        'width': 50, 'height': 50}]
editor.detection_boxes = [{'label': 'a', 'x': 15, 'y': 15, 'width': 40,
                           'height': 40, 'shape_type': 'rectangle'}]
draw()
print(locks)
""")
    lines = _lines(out)
    assert lines[0] == "[True, False]", (
        f"第一趟应当是锁定的那趟（最底层）：{lines[0]}")


# ---------- 第七轮：区域编辑与标签一致 + 覆盖清手画框 ----------

def test_real_region_wheel_scale_works_while_scope_panel_is_open():
    """② 面板开着（画布停在标注模式）时，悬停区域滚轮照样能缩放。

    平移/缩放/手柄三件事本来就都实现了，但滚轮那条分支当时只认
    `_is_paste_mode()`，面板一开（标注模式）就静默失效——用户看到的正是
    「区域不能像标签那样缩放」。
    """
    out = _run_real_qt(_REGION_LOCK_HOST + """
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPointF, QPoint

def wheel(x, y, up=True):
    canvas.mouse_pos = QPointF(x, y)
    delta = 120 if up else -120
    canvas.wheelEvent(QWheelEvent(QPointF(x, y), QPointF(x, y),
                                  QPoint(0, 0), QPoint(0, delta),
                                  Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase,
                                  False))

# 面板开着 + 标注模式：滚轮应当能缩放区域
w._open_scope_panel()
canvas.is_manual_scale = True          # 免得缩放被「自适应」重置
w.edit_mode = 'annotate'
app.processEvents()
before = (w.region_boxes[0]['x'], w.region_boxes[0]['y'],
          w.region_boxes[0]['width'], w.region_boxes[0]['height'])
cx, cy = region_center()
wheel(cx, cy, up=True)
after = (w.region_boxes[0]['x'], w.region_boxes[0]['y'],
         w.region_boxes[0]['width'], w.region_boxes[0]['height'])
print('面开标注模式 区域变了吗 ->', before != after)
print('编辑模式 ->', w.edit_mode)
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), (
        f"面板开着时滚轮缩放区域无效：{lines[0]}")
    assert lines[1].endswith("annotate"), lines[1]


def test_real_region_lock_keeps_wheel_scale_inert():
    """② 的边界：锁定的区域滚轮也不该缩放（锁 = 不可修改）。"""
    out = _run_real_qt(_REGION_LOCK_HOST + """
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPointF, QPoint

def wheel(x, y):
    canvas.mouse_pos = QPointF(x, y)
    canvas.wheelEvent(QWheelEvent(QPointF(x, y), QPointF(x, y),
                                  QPoint(0, 0), QPoint(0, 120),
                                  Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase,
                                  False))

w._open_scope_panel()
canvas.is_manual_scale = True
app.processEvents()
canvas.set_region_locked(0, True)
before = (w.region_boxes[0]['width'], w.region_boxes[0]['height'])
cx, cy = region_center()
wheel(cx, cy)
after = (w.region_boxes[0]['width'], w.region_boxes[0]['height'])
print('锁定后滚轮缩放了吗 ->', before != after)
""")
    lines = _lines(out)
    assert lines[0].endswith("False"), (
        f"锁定的区域被滚轮缩放了：{lines[0]}")


def test_real_region_corner_and_edge_handles_work_while_panel_is_open():
    """② 四个角 + 四条边在面板开着时都能拖（单边调整也要真的有效）。"""
    out = _run_real_qt(_REGION_LOCK_HOST + """
w._open_scope_panel()
canvas.is_manual_scale = True
app.processEvents()

def bbox():
    r = w.region_boxes[0]
    return (round(r['x']), round(r['y']), round(r['width']), round(r['height']))

# 拖右下角：宽高一起变，左上角不动
brx, bry = C(w.region_boxes[0]['x'] + w.region_boxes[0]['width'],
             w.region_boxes[0]['y'] + w.region_boxes[0]['height'])
before = bbox()
press(brx, bry)
move(brx + 60, bry + 60)
release(brx + 60, bry + 60)
corner = bbox()
print('角点拖后 (x,y 不变, w/h 变大) ->',
      corner[0] == before[0] and corner[1] == before[1],
      corner[2] > before[2] and corner[3] > before[3])

# 拖右边中点：只宽变，高与左上角都不动
before2 = bbox()
rmx, rmy = C(w.region_boxes[0]['x'] + w.region_boxes[0]['width'],
             w.region_boxes[0]['y'] + w.region_boxes[0]['height'] / 2)
press(rmx, rmy)
move(rmx + 50, rmy)
release(rmx + 50, rmy)
edge = bbox()
print('单边拖后 (只有 w 变大) ->',
      edge[2] > before2[2], edge[3] == before2[3],
      edge[0] == before2[0], edge[1] == before2[1])
""")
    lines = _lines(out)
    assert lines[0].endswith("True True"), (
        f"角点拖拽不对：{lines[0]}")
    assert lines[1].endswith("True True True True"), (
        f"单边调整不对（应只有宽度变）：{lines[1]}")


def test_real_replace_mode_clears_hand_drawn_boxes_too():
    """③ 覆盖：手画的检测框也被清掉，只剩本轮推理结果（用户明确要求）。

    用真 `ImageEditor` 走一遍 `_apply_inference_to_current`，钉住端到端行为
    ——只测 `merge_boxes()` 的话，中间任何一层把 `replace_auto` 丢了都测不到。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_task import ImageResult

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(1000, 800, QImage.Format_RGB888)
img.fill(QColor(40, 70, 100))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
app.processEvents()
w.update_label_list = lambda: None
w._save_current_detection_boxes = lambda *a, **k: None
w.save_json = lambda *a, **k: None
w.update_inference_menu = lambda *a, **k: None

# 手画的框
w.detection_boxes = [
    {'label': 'hand', 'x': 10, 'y': 10, 'width': 40, 'height': 40},
]
# `_replace_mode()` 读的是参数行上的按钮，所以必须把它真的建出来：
# 只传 replace_auto=True 给函数的话，测不到「界面默认就是覆盖」这件事。
row = w._ensure_model_row()
row.show()
app.processEvents()
print('replace_mode ->', w._replace_mode())

result = ImageResult(0, 'work.png',
                     [{'label': 'new', 'x': 200, 'y': 200,
                       'width': 30, 'height': 30}], None, 1000, 800)
w._apply_inference_to_current(result, True)
app.processEvents()
print('labels ->', sorted(b['label'] for b in w.detection_boxes))
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), f"默认应是覆盖模式：{lines[0]}"
    assert lines[1] == "labels -> ['new']", (
        f"覆盖后手画框没被清掉：{lines[1]}")


# ---------- 第九轮：区域滚轮单边调整 + 边框只画四角 ----------

def test_real_region_wheel_on_edge_resizes_only_that_edge():
    """② 滚轮压在**边上**只改那一条边（用户：单边调整包括滚轮，复用检测框的）。

    检测框早有这套（`_get_selected_box_edge` / `_adjust_selected_box_edge`），
    区域侧原先只有「绕中心等比缩放」。这里钉住四条边各自的效果。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPoint

def wheel(x, y, up=True):
    canvas.mouse_pos = QPointF(x, y)
    d = 120 if up else -120
    canvas.wheelEvent(QWheelEvent(QPointF(x, y), QPointF(x, y),
                                  QPoint(0, 0), QPoint(0, d),
                                  Qt.NoButton, Qt.NoModifier,
                                  Qt.NoScrollPhase, False))

def bbox():
    r = w.region_boxes[0]
    return (round(r['x']), round(r['y']), round(r['width']), round(r['height']))

r = w.region_boxes[0]
cases = []

# 右边：只有宽变
ex, ey = C(r['x'] + r['width'], r['y'] + r['height'] / 2)
b = bbox(); wheel(ex, ey); a = bbox()
cases.append(('right', a[2] > b[2], a[0] == b[0], a[1] == b[1], a[3] == b[3]))

# 左边：x 减小、宽变
ex, ey = C(r['x'], r['y'] + r['height'] / 2)
b = bbox(); wheel(ex, ey); a = bbox()
cases.append(('left', a[0] < b[0], a[2] > b[2], a[1] == b[1], a[3] == b[3]))

# 下边：只有高变
ex, ey = C(r['x'] + r['width'] / 2, r['y'] + r['height'])
b = bbox(); wheel(ex, ey); a = bbox()
cases.append(('bottom', a[3] > b[3], a[0] == b[0], a[1] == b[1], a[2] == b[2]))

# 上边：y 减小、高变
ex, ey = C(r['x'] + r['width'] / 2, r['y'])
b = bbox(); wheel(ex, ey); a = bbox()
cases.append(('top', a[1] < b[1], a[3] > b[3], a[0] == b[0], a[2] == b[2]))

for name, *flags in cases:
    print(name, all(flags))
""")
    lines = _lines(out)
    for name in ("right", "left", "bottom", "top"):
        assert f"{name} True" in lines, (
            f"区域贴{name}边滚轮没做到单边调整：{lines}")


def test_real_region_wheel_accepts_the_mouse_outside_the_edge():
    """区域滚轮贴边命中**含外侧**，与检测框同一手感（用户第 11 轮）。

    用户报「鼠标放在一侧滚动滚轮会调整一边，区域都没有」——检测框那边
    `_get_selected_box_edge()` 在框**外**多远都认正对着的那条边（实测偏移
    60px 仍返回 'right'），区域却只认「正好压在边线上」的 8px 带。于是同样是
    「贴着右边滚」，框的宽在变、区域纹丝不动。

    这条钉住：边线上、外侧 20、外侧 60 三处都能单边调整。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPoint

def wheel(x, y):
    canvas.mouse_pos = QPointF(x, y)
    canvas.wheelEvent(QWheelEvent(QPointF(x, y), QPointF(x, y),
                                  QPoint(0, 0), QPoint(0, 120),
                                  Qt.NoButton, Qt.NoModifier,
                                  Qt.NoScrollPhase, False))

def bbox():
    r = w.region_boxes[0]
    return (round(r['x']), round(r['y']), round(r['width']), round(r['height']))

def fresh():
    w.region_boxes = [{'shape_type': 'rectangle', 'x': 300, 'y': 300,
                       'width': 200, 'height': 200}]
    canvas.selected_region = None

r = w.region_boxes[0]
right = r['x'] + r['width']

for off in (0, 20, 60):
    fresh()
    ex, ey = C(right + off, r['y'] + r['height'] / 2)
    b = bbox(); wheel(ex, ey); a = bbox()
    # 只有宽度变、其余三项不动
    print('外侧', off, a[2] > b[2], a[0] == b[0], a[1] == b[1], a[3] == b[3])
""")
    lines = _lines(out)
    for off in (0, 20, 60):
        line = next((x for x in lines if x.startswith(f"外侧 {off} ")), None)
        assert line is not None, f"没有外侧 {off} 的输出：{lines}"
        assert line.endswith("True True True True"), (
            f"右边外侧 {off}px 滚轮没做到单边调整：{line}")


def test_real_region_handles_show_a_white_square_on_hover():
    """区域手柄悬停时必须变成**白色正方形**，与检测框完全一致（用户第 11 轮）。

    用户：「检测框选中句柄会变成正方形……区域都没有」。检测框
    `_draw_box_handles()` 里悬停那一支画的是 `drawRect`（实心白方 + 边框），
    另一个画白底圆 + 彩色实心圆。区域过去**只有后者、没有悬停态**。

    这条从两条正交的角度采样区分两种形状：
    - 正方形：水平与对角线方向都白到半径外
    - 圆    ：对角线方向比水平方向更早退出（勾股）
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from PySide6.QtGui import QWheelEvent, QPainter
from PySide6.QtCore import QPoint, QSize

r = w.region_boxes[0]
canvas.selected_region = 0
br = canvas.get_background_rect()
s = canvas.background_scale
cx = br.left() + (r['x'] + r['width']) * s
cy = br.top() + (r['y'] + r['height']) * s


def render(handle):
    canvas.hover_region_index = 0
    canvas.hover_region_handle = handle
    img = QImage(canvas.size(), QImage.Format_RGB888)
    img.fill(QColor(0, 0, 0))
    p = QPainter(img)
    canvas._draw_region_handles(
        p, br.left() + r['x'] * s, br.top() + r['y'] * s,
        r['width'] * s, r['height'] * s, QColor('#FF3B30'))
    p.end()
    return img


def white(img, x, y):
    c = img.pixelColor(int(x), int(y))
    return c.red() > 200 and c.green() > 200 and c.blue() > 200


norm = render(None)
hov = render('br')
# 采样半径 2：在**填充区**内。半径 3 会落到边长 2px 的描边圈上（那是区域色
# 而不是白），所以两种态都以 2 为准。
# 正方形：水平与对角线方向都白到同样远；圆：对角线方向先退出（勾股）。
print('常态 水平', white(norm, cx + 2, cy))
print('常态 对角', white(norm, cx + 2, cy + 2))
print('悬停 水平', white(hov, cx + 2, cy))
print('悬停 对角', white(hov, cx + 2, cy + 2))
print('悬停 更远对角', white(hov, cx + 3, cy + 3))

# 渲染之外还要确认**悬停态真的被 hover 逻辑记下来**：只靠手动塞
# `hover_region_handle` 的话，`_check_region_hover()` 整段删掉测试也照样过。
canvas.selected_region = None
canvas.hover_region_index = None
canvas.hover_region_handle = None
canvas.mouse_pos = QPointF(cx, cy)          # 正压在右下角手柄上
canvas._check_region_hover()
print('hover 记录', canvas.hover_region_handle, canvas.hover_region_index)
""")
    lines = _lines(out)
    # 悬停：正方形是实心白，水平与对角线一样远
    assert lines[2].endswith("True"), (
        f"悬停手柄水平方向应为白色: {lines[2]}")
    assert lines[3].endswith("True"), (
        f"悬停手柄在对角线方向应为白色（正方形四角）: {lines[3]}")
    # 常态：白色只是圆外那半圈月牙，正方形填充位置上是区域色
    assert lines[0].endswith("False") and lines[1].endswith("False"), (
        f"常态手柄不该在填充区全白（那是悬停正方形）: {lines[:2]}")
    # 悬停在右下角 → 必须记成 ('br', 0)，渲染端才有东西可画
    hover_line = next((x for x in lines if x.startswith("hover 记录")), None)
    assert hover_line is not None, f"没有 hover 记录的输出：{lines}"
    assert hover_line.endswith("br 0"), (
        f"悬停在右下角应记为 br/0，实际：{hover_line}")


def test_real_region_wheel_in_the_middle_scales_both_axes():
    """② 的边界：鼠标在区域**中间**时仍是整体等比缩放（不误判成单边）。"""
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from PySide6.QtGui import QWheelEvent
from PySide6.QtCore import QPoint

def wheel(x, y):
    canvas.mouse_pos = QPointF(x, y)
    canvas.wheelEvent(QWheelEvent(QPointF(x, y), QPointF(x, y),
                                  QPoint(0, 0), QPoint(0, 120),
                                  Qt.NoButton, Qt.NoModifier,
                                  Qt.NoScrollPhase, False))

r = w.region_boxes[0]
cx, cy = C(r['x'] + r['width'] / 2, r['y'] + r['height'] / 2)
b = (r['width'], r['height'])
wheel(cx, cy)
a = (r['width'], r['height'])
print('中间滚轮宽高都变大', a[0] > b[0] and a[1] > b[1])
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), (
        f"区域中间滚轮没有整体缩放：{lines[0]}")


def test_real_region_border_draws_corner_handles_only():
    """② 区域选中时**只画四角**手柄，与检测框逐像素一致。

    检测框 `_draw_box_handles()` 的 `corners` 就是四个角，边中点一个都不画
    （第 11 轮实测核对过）。区域过去只画四角是对的，第 11 轮中途一度被改成
    八个边中点也画，用户当场纠正「检测框也只有四个点」，所以这里钉死：
    **边中点的位置必须是空白**。

    边中点仍可抓（`_region_handle_at()` 认八处），见下一条——不画只影响视觉。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
from PySide6.QtGui import QImage, QPainter

# `_draw_regions` 只在**选中**该区域时才画手柄（`is_selected`）
w.region_fixed = False
canvas.selected_region = 0
print('selected_region ->', canvas.selected_region)

drawn = []
orig = canvas._draw_region_handles

def spy(painter, x, y, wd, ht, color):
    drawn.append((x, y, wd, ht))
    return orig(painter, x, y, wd, ht, color)

canvas._draw_region_handles = spy
canvas.update()
app.processEvents()
img = QImage(canvas.width() or 100, canvas.height() or 100,
             QImage.Format_ARGB32)
p = QPainter(img)
canvas._paint_scene(p)
p.end()
print('走到手柄绘制', len(drawn) > 0)

# 直接问源码：四边中点那四个坐标不该再出现在绘制函数里
import inspect
fn = None
for klass in type(canvas).__mro__:
    if '_draw_region_handles' in klass.__dict__:
        fn = klass.__dict__['_draw_region_handles']
        break
src = inspect.getsource(fn)
print('源码里还有四边中点', ('w / 2' in src) and ('h / 2' in src))
print('四角坐标仍在', '(x, y + h)' in src or '(x + w, y + h)' in src)

# 逐像素：边中点上必须是空白（与检测框同一套「只有四角」）
r = w.region_boxes[0]
canvas.hover_region_index = None
canvas.hover_region_handle = None
raw = QImage(canvas.width() or 100, canvas.height() or 100,
             QImage.Format_RGB888)
raw.fill(QColor(0, 0, 0))
p = QPainter(raw)
br = canvas.get_background_rect()
sc = canvas.background_scale
canvas._draw_region_handles(
    p, br.left() + r['x'] * sc, br.top() + r['y'] * sc,
    r['width'] * sc, r['height'] * sc, QColor('#FF3B30'))
p.end()


def has_pixels(cx, cy, half=6):
    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            c = raw.pixelColor(int(cx) + dx, int(cy) + dy)
            if c.red() or c.green() or c.blue():
                return True
    return False


def px(ix, iy):
    return br.left() + ix * sc, br.top() + iy * sc


print('右边中点空白',
      not has_pixels(*px(r['x'] + r['width'], r['y'] + r['height'] / 2)))
print('上边中点空白',
      not has_pixels(*px(r['x'] + r['width'] / 2, r['y'])))
print('右下角有像素',
      has_pixels(*px(r['x'] + r['width'], r['y'] + r['height'])))
""")
    lines = _lines(out)
    assert lines[1].endswith("True"), f"没走到手柄绘制：{lines}"
    assert lines[2].endswith("False"), (
        f"区域手柄仍在画四边中点，与检测框不一致：{lines[2]}")
    assert lines[3].endswith("True"), (
        f"四角手柄被误删：{lines[3]}")
    assert lines[4].endswith("True"), (
        f"右边中点位置不该有手柄：{lines[4]}")
    assert lines[5].endswith("True"), (
        f"上边中点位置不该有手柄：{lines[5]}")
    assert lines[6].endswith("True"), (
        f"右下角应该有手柄：{lines[6]}")


def test_real_region_edge_drag_still_resizes_with_a_drawn_handle():
    """② 的边界：边中点画了句柄之后，**鼠标在边上拖动**仍能单边调整。

    与上一条配套：手柄画出来是给眼睛看的，拖动靠的还是 `_region_handle_at()`
    的命中判定，两处必须一致（「看到的点」=「抓得到的点」）。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
r = w.region_boxes[0]
before = (round(r['x']), round(r['y']), round(r['width']), round(r['height']))

# 鼠标移到右边中点并拖动
ex, ey = C(r['x'] + r['width'], r['y'] + r['height'] / 2)
press(ex, ey)
move(ex + 40, ey)
release(ex + 40, ey)

after = (round(r['x']), round(r['y']), round(r['width']), round(r['height']))
print('边上拖动只改宽', after[2] > before[2], after[3] == before[3],
      after[0] == before[0], after[1] == before[1])
""")
    lines = _lines(out)
    assert lines[0].endswith("True True True True"), (
        f"边上不画点之后拖动也不生效了：{lines[0]}")


# ---------- 第九轮：贴图模式收起推理 ----------

def test_real_model_row_is_hidden_and_inference_disabled_in_paste_mode():
    """④ 切到贴图模式：参数行自动收起，两个推理按钮随之不可用。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
img = QImage(600, 400, QImage.Format_RGB888)
img.fill(QColor(20, 30, 40))
w.current_background = QPixmap.fromImage(img)
w.current_background_index = 0
w.background_images = ['work.png']
app.processEvents()

row = w._ensure_model_row()
# 走真实的模型加载收尾（它会 show 参数行并刷新推理按钮可用性）
w._on_model_loaded(w._detector, (640, 640), 'test')
app.processEvents()
print('标注模式参数行可见', row.isVisible())
print('标注模式能推理', w.model_single_btn.isEnabled())

w._set_edit_mode('paste')
app.processEvents()
print('贴图模式参数行可见', row.isVisible())
print('贴图模式能推理', w.model_single_btn.isEnabled())
print('贴图模式全图推理', w.model_all_btn.isEnabled())

w._set_edit_mode('annotate')
app.processEvents()
print('切回标注参数行可见', row.isVisible())
print('切回标注能推理', w.model_single_btn.isEnabled())
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), lines[0]
    assert lines[1].endswith("True"), lines[1]
    assert lines[2].endswith("False"), f"贴图模式没收起参数行：{lines[2]}"
    assert lines[3].endswith("False"), f"贴图模式推理按钮没禁用：{lines[3]}"
    assert lines[4].endswith("False"), lines[4]
    assert lines[5].endswith("True"), f"切回标注没恢复参数行：{lines[5]}"
    assert lines[6].endswith("True"), f"切回标注没恢复推理：{lines[6]}"


def test_real_model_row_not_restored_if_user_had_closed_it():
    """④ 的边界：用户自己在标注模式下收了参数行，切一圈回来仍是收起的。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
app.processEvents()

row = w._ensure_model_row()
w._on_model_loaded(w._detector, (640, 640), 'test')
app.processEvents()
row.hide()                    # 用户自己收起
app.processEvents()

w._set_edit_mode('paste')
app.processEvents()
w._set_edit_mode('annotate')
app.processEvents()
print('没有被强行展开', row.isVisible())
""")
    lines = _lines(out)
    assert lines[0].endswith("False"), (
        f"用户原本收起的参数行被模式切换强行展开了：{lines[0]}")


def test_real_start_inference_is_refused_in_paste_mode():
    """④ 绕过按钮直接调 `_start_inference` 也要被挡（快捷键/代码路径）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
w._on_model_loaded(w._detector, (640, 640), 'test')
app.processEvents()
w._set_edit_mode('paste')
app.processEvents()

w._start_inference([0], current_only=True)
print('起了 worker 吗', w._run_worker is not None)
print('状态栏', w.status_label.text())
""")
    lines = _lines(out)
    assert lines[0].endswith("False"), (
        f"贴图模式下仍能起推理 worker：{lines[0]}")
    assert "贴图模式" in lines[1], f"没有给出原因提示：{lines[1]}"


# ---------- 第九轮：区域右键入口在两种模式下都可达 ----------

def test_real_region_menu_is_reachable_only_in_paste_mode():
    """⑤ 区域菜单**只在贴图模式**给（用户第 11 轮改了口径）。

    第 9/10 轮的做法是两种模式都挂区域入口；第 11 轮用户明确：

        标注模式不要，只走推理范围的区域。

    理由是「推理是推理的操作，贴图是贴图的操作」——推理区域的增删改全部走
    「推理范围」面板，标注模式右键再留一份入口，同一份数据就有两个地方能改。

    贴图模式的入口必须保留：贴图区域没有面板，右键是它唯一的编辑入口。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QMenu
from PySide6.QtGui import QAction, QImage, QColor, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
w.current_background_index = 0
w.canvas.resize(900, 700)
app.processEvents()
w._detector = type('D', (), {'model_path': 'm.onnx', 'conf_threshold': .25,
                            'iou_threshold': .45, 'class_names': [],
                            'resolve_format': lambda s: 'v8',
                            'is_dynamic_input': lambda s: False,
                            'active_providers': ['CPU']})()
w._on_model_loaded(w._detector, (640, 640), 'test')
w.region_boxes = []
w.paste_region_boxes = []
w.region_fixed = False
w.paste_region_fixed = False
w._open_scope_panel()
app.processEvents()

captured = {}
orig_exec_ = QMenu.exec_
def fake_exec(self, *a, **k):
    captured['texts'] = [act.text() for act in self.actions()]
    return None
QMenu.exec_ = fake_exec
try:
    for mode in ('annotate', 'paste'):
        w._set_edit_mode(mode)
        app.processEvents()
        captured.pop('texts', None)
        w.canvas._show_background_context_menu(QPoint(10, 10))
        app.processEvents()
        texts = captured.get('texts') or []
        print(mode, any('添加区域' in t for t in texts))
finally:
    QMenu.exec_ = orig_exec_
""")
    lines = _lines(out)
    assert lines[0].endswith("False"), (
        f"标注模式不该再有区域入口（用户：只走推理范围的区域）：{lines[0]}")
    assert lines[1].endswith("True"), (
        f"贴图模式下右键没有「添加区域」（贴图区域唯一的入口被弄坏了）：{lines[1]}")


def test_real_region_menu_uses_the_same_region_list_in_both_modes():
    """⑤ 两种模式共用同一份 `region_boxes`（不是两套区域）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QColor, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
w.current_background_index = 0
w.region_boxes = [{'shape_type': 'rectangle', 'x': 10, 'y': 10,
                   'width': 50, 'height': 50}]
app.processEvents()

w._set_edit_mode('annotate')
app.processEvents()
a = w._region_boxes()
w._set_edit_mode('paste')
app.processEvents()
b = w._region_boxes()
print('模式切换不丢区域', len(a) == 1 and len(b) == 1)
print('两种模式读的是同一个列表',
      len(a) == 1 and a[0] is w.region_boxes[0])
rect = w.canvas.get_background_rect()
sc = w.canvas.background_scale
from PySide6.QtCore import QPointF
ctr = QPointF(rect.x() + 35 * sc, rect.y() + 35 * sc)   # 区域中心 (10+25, 10+25)
print('贴图模式下画布也认这条区域',
      w.canvas._region_at(ctr) is not None)
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), lines[0]
    assert lines[1].endswith("True"), (
        f"贴图模式下画布读不到同一份区域：{lines[1]}")


# ---------- 第九轮：固定区域菜单项左对齐 ----------

def test_real_region_menu_items_are_left_aligned():
    """⑥ 各项文字缩进必须一致（不再有「空了一个字」）。

    用 Qt 画菜单项的口径复算每项的文字缩进：可勾选项会被额外加一个
    `PM_MenuButtonIndicator`（本主题 12px），于是「固定区域」比同菜单其它项
    多缩进一个字——用户报的「固定区域没有左对齐空了一个字」。

    ⚠️ 必须按**每项**算，不能看 `actionGeometry().width()`：菜单宽度会被最长项
    撑满，几种写法都是同一个值，看宽度永远发现不了这个 bug。

    第 11 轮去掉多边形区域后菜单只剩三项（添加矩形/固定/清空）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QMenu, QStyle, QStyleOptionMenuItem
from PySide6.QtGui import QImage, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
st = app.style()
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
w.current_background_index = 0
w.region_boxes = []
w.region_fixed = False
app.processEvents()


def indents(fixed):
    m = QMenu()
    w.canvas._add_region_actions(m, fixed)
    out = []
    for a in m.actions():
        opt = QStyleOptionMenuItem()
        opt.initFrom(m)
        opt.font = m.font()
        opt.text = a.text()
        opt.menuItemType = QStyleOptionMenuItem.Normal
        opt.checkType = (QStyleOptionMenuItem.Exclusive if a.isCheckable()
                         else QStyleOptionMenuItem.NotCheckable)
        opt.checked = a.isChecked()
        opt.rect = m.actionGeometry(a)
        # Qt 只给 checkable 项加这一列
        out.append(st.pixelMetric(QStyle.PM_MenuButtonIndicator, opt, m)
                   if a.isCheckable() else 0)
    return out


i_off = indents(False)
i_on = indents(True)
print('未固定缩进:', i_off)
print('已固定缩进:', i_on)
print('没有可勾选项:', not any(
    a.isCheckable() for a in (lambda m: (w.canvas._add_region_actions(m, False),
                                         m.actions())[1])(QMenu())))
""")
    lines = _lines(out)
    assert lines[0] == "未固定缩进: [0, 0, 0]", (
        f"未固定时各项没有齐平：{lines[0]}")
    assert lines[1] == "已固定缩进: [0, 0, 0]", (
        f"已固定时各项没有齐平：{lines[1]}")
    assert lines[2].endswith("True"), lines[2]


def test_real_region_menu_keeps_a_visible_check_state():
    """⑥ 勾**仍然看得见**，且在两种状态间切换。

    用户明确「我不是固定区域左边有打钩吗，不勾选不就解除了」——修对齐不能把勾
    去掉。勾现在画进文案（`✔ 固定区域`），点一下照样来回切。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QMenu
from PySide6.QtGui import QImage, QPixmap
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(600, 400, QImage.Format_RGB888))
w.background_images = ['work.png']
w.current_background_index = 0
w.region_boxes = []
w.region_fixed = False
app.processEvents()


def fix_text(fixed):
    menu = QMenu()
    w.canvas._add_region_actions(menu, fixed)
    for a in menu.actions():
        if '固定区域' in a.text():
            return a.text()
    return None


print('未固定文案:', fix_text(False))
print('已固定文案:', fix_text(True))
print('已固定时带勾:', fix_text(True).startswith('✔'))
print('未固定时不带勾:', not fix_text(False).startswith('✔'))
""")
    lines = _lines(out)
    assert lines[0] == "未固定文案: 固定区域", lines[0]
    assert lines[1] == "已固定文案: ✔ 固定区域", (
        f"已固定时没有显示勾：{lines[1]}")
    assert lines[2].endswith("True"), lines[2]
    assert lines[3].endswith("True"), lines[3]


# ---------- 第 12 轮：区域几何参数 = 检测框 + 1（逐项对齐） ----------

def test_region_geometry_is_exactly_one_more_than_the_box():
    """区域句柄与边框 = 检测框参数 **+1**，逐项钉住（用户要求）。

    这条把「复用检测框」写成可执行的对照，而不是散落在各处的一句注释：
    句柄基准、边框未选中、边框选中三项都必须正好差 1（或 1.5×），
    任何一项被单独改动都会在这里失败。

    第 12 轮之前这里漏过一次：句柄改了 8→9，边框却还是自己那套
    `× (2 if selected else 1)`，与检测框的 `× 1.5` 不是同一把尺子。
    """
    from labelforge.core.config import (BOX_BORDER_CONFIG, DETECTION_BOX_CONFIG,
                                        REGION_CONFIG)

    box_handle = DETECTION_BOX_CONFIG['resize_handle_size']
    assert REGION_CONFIG['handle_size'] == box_handle + 1, (
        f"区域句柄应为检测框 {box_handle} + 1，实际 {REGION_CONFIG['handle_size']}")

    assert BOX_BORDER_CONFIG['width'] == 1.0, (
        "检测框基准宽度变了，区域那边用的是同一个常量，跟着自动变")

    # 边框粗细要比 .width()，进程内的 QPen 是假对象，所以走真实 Qt 子进程
    out = _run_real_qt(_REGION_RESIZE_HOST + """
probe = canvas
box_plain = probe._get_box_border_pen(QColor('#FF0000'), False).widthF()
region_plain = probe._get_region_border_pen(QColor('#FF0000'), False).widthF()
box_sel = probe._get_box_border_pen(QColor('#FF0000'), True).widthF()
region_sel = probe._get_region_border_pen(QColor('#FF0000'), True).widthF()
print('pens', box_plain, region_plain, box_sel, region_sel)
""")
    lines = _lines(out)
    pen_line = next(x for x in lines if x.startswith("pens"))
    _, box_plain, region_plain, box_sel, region_sel = pen_line.split()
    box_plain, region_plain = float(box_plain), float(region_plain)
    box_sel, region_sel = float(box_sel), float(region_sel)
    assert region_plain == box_plain + 1, (
        f"区域边框未选中应为检测框 {box_plain} + 1，实际 {region_plain}")
    # 选中态：**先 +1 再 ×1.5**（基准参数 +1），所以是 (1+1)*1.5 = 3.0，
    # 不是 1.5+1。用户说的「参数 +1」指的是基准参数。
    assert region_sel == (box_plain + 1) * 1.5, (
        f"区域边框选中应为 (检测框 {box_plain} + 1) × 1.5，实际 {region_sel}")


def test_region_handle_cursor_matches_the_box():
    """悬停区域手柄的光标必须和检测框**一样**，是**小手**而不是十字。

    用户报「区域的句柄选中会变出十字光标」。检测框那边
    （`_select_hovered_detection_box`）给的是 `PointingHandCursor`；区域这里
    曾错写成 `CrossCursor`。这条直接拿两边比，谁改都能发现。
    """
    out = _run_real_qt(_REGION_RESIZE_HOST + """
r = w.region_boxes[0]
canvas.selected_region = None
canvas.hover_region_handle = None
canvas.hover_region_index = None
br = canvas.get_background_rect()
s = canvas.background_scale
# 鼠标正压在右下角手柄上
canvas.mouse_pos = QPointF(br.left() + (r['x'] + r['width']) * s,
                           br.top() + (r['y'] + r['height']) * s)
canvas._check_region_hover()
region_cursor = canvas.cursor().shape()

# 检测框的对照值：同一套 hover 逻辑给出的光标
w.detection_boxes = [{'label': 'a', 'shape_type': 'rectangle', 'x': 120,
                      'y': 120, 'width': 80, 'height': 60, 'points': []}]
canvas.selected_box = 0
canvas.mouse_pos = QPointF(br.left() + (120 + 80) * s,
                           br.top() + (120 + 60) * s)
canvas._select_hovered_detection_box(br)
box_cursor = canvas.cursor().shape()

print('handles', canvas.hover_region_handle, canvas.hover_resize_handle)
print('cursors', region_cursor.value, box_cursor.value,
      Qt.PointingHandCursor.value, Qt.CrossCursor.value)
""")
    lines = _lines(out)
    handle_line = next(x for x in lines if x.startswith("handles"))
    curves_line = next(x for x in lines if x.startswith("cursors"))
    assert handle_line.split()[1] == "br", (
        f"悬停右下角应命中 br 手柄：{handle_line}")
    parts = curves_line.split()
    region_cursor, box_cursor, pointing, cross = (int(parts[1]), int(parts[2]),
                                                  int(parts[3]), int(parts[4]))
    assert region_cursor != cross, "区域手柄不该是十字光标（用户明确报过）"
    assert region_cursor == pointing, (
        f"区域手柄光标应为小手 PointingHandCursor({pointing})，"
        f"实际 {region_cursor}")
    assert box_cursor == region_cursor, (
        f"区域({region_cursor}) 与检测框({box_cursor}) 的光标必须一致")
