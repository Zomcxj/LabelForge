"""推理范围 UI 的接线测试：面板、窗口弹窗、参数行入口、快照与配置持久化。

纯逻辑部分用假控件；真实控件行为（显隐、真实数值、预估文本）走
`_run_real_qt` 子进程，避免 conftest 的 mock PyQt5 把结论带偏。
"""
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from pastelabel.engine.inference_scope import (
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


# ---------- 参数行入口 ----------

def test_real_row_has_scope_button_disabled_until_model_loads():
    """真实控件：没加载模型时范围按钮禁用，加载后启用。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(3840, 2160))
print(d.spec())
print(d.window_count(), d.window_grid())
print(d.ok_btn.isEnabled())
print(d._estimate.text())
""")
    lines = _lines(out)
    assert lines[0] == "{'w': 640, 'h': 640, 'overlap': 200}", lines[0]
    assert lines[1] == "45 (5, 9)", lines[1]
    assert lines[2] == "True", lines[2]
    assert "5 × 9" in lines[3] and "45" in lines[3], lines[3]


def test_real_window_dialog_disables_ok_when_count_explodes():
    """真实控件：窗口数超上限时禁用确定并说明，不能排出天文数字。

    20000² 图配 64 窗口、重叠 64 → 步长 1 → 3.9 亿窗，列表没建完就 OOM。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=(200, 100))
print(d.window_count(), d.window_grid())
""")
    assert _lines(out)[0] == "1 (1, 1)", _lines(out)[0]


def test_real_window_dialog_survives_broken_saved_spec():
    """真实控件：配置里存了坏窗口参数也不能崩，回退到默认值。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
for bad in ({'w': 'abc', 'h': None, 'overlap': 'x'},
            {'w': 0, 'h': -5, 'overlap': -1},
            {'w': 99999, 'h': 1, 'overlap': 10**9}):
    d = SlidingWindowDialog(reference_size=(1000, 1000), spec=bad)
    print(d.spec())
""")
    lines = _lines(out)
    assert lines[0] == "{'w': 640, 'h': 640, 'overlap': 200}", lines[0]
    assert lines[1] == "{'w': 8, 'h': 8, 'overlap': 0}", lines[1]
    assert lines[2] == "{'w': 8192, 'h': 8, 'overlap': 8192}", lines[2]


def test_real_window_dialog_roundtrips_spec_through_engine():
    """真实控件：弹窗给的 spec 能被引擎直接接受并排出同样数量的窗口。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.engine.inference_scope import iter_windows, normalize_scope

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

def test_real_scope_panel_blocks_region_mode_without_regions():
    """真实控件：选区域但没画区域时禁用确定。

    否则会「成功但 0 框」，默认覆盖模式会因此删掉原有自动标注。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=0,
               reference_size=(3840, 2160))
print(p.ok_btn.isEnabled())
p.set_mode(MODE_ROI)
print(p.ok_btn.isEnabled())
print('没有可用区域' in p.warn_label.text())
p.set_region_count(2)
print(p.ok_btn.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "False", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", lines[3]


def test_real_scope_panel_shows_controls_per_mode():
    """真实控件：滑窗参数控件只在滑窗显示，区域按钮只在区域显示。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

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
    print(mode, p.window_params.isVisible(), p.region_btn.isVisible())
# 面板本身必须是非模态的：否则弹窗开着时画布收不到鼠标事件，「就地画区域」失效
print(p.isModal())
""")
    lines = _lines(out)
    assert lines[0] == "full False False", lines[0]
    assert lines[1] == "window True False", lines[1]
    assert lines[2] == "roi False True", lines[2]
    assert lines[3] == "False", lines[3]


def test_real_scope_params_are_disabled_unless_window_mode_selected():
    """只有选中「滑窗」时宽/高/重叠与模板按钮才**可点**，否则禁用。

    用户要求「未选中滑窗时不能修改点击」。只隐藏是不够的：隐藏的控件
    在某些布局路径下仍会被键盘焦点/程序化调用改到，而且用户看不到
    当前值就没法核对「这次要跑多大窗口」。所以分别断言 disabled（可见
    或不可见都好，但一定不能可用）。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
# 造出模板按钮：先写一条共用模板
from pastelabel.core import config_manager as cm
cfg = cm.load_config()
cfg['crop_templates'] = [{'w': 512, 'h': 512, 'overlap': 64,
                          'min_visible': 0.7, 'square': False,
                          'fill': False, 'fill_rgb': [0, 0, 0]}]
cm.save_config(cfg)

p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
p.show()
app.processEvents()
btns = p.window_params.template_buttons()
print(len(btns))
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    p.set_mode(mode)
    app.processEvents()
    wp = p.window_params
    print(mode, wp.spins_enabled(), wp.templates_enabled(),
          p.region_btn.isEnabled())
# 禁用状态下用户**点不动**：这是「不能修改点击」的真实保证。
# 不断言 setValue() 会失败——那是程序化写入，Qt 有意让它绕过 enabled，
# 拿它当断言是量的东西与结论不同构。
p.set_mode(MODE_ROI)
app.processEvents()
print(wp.is_editable(), wp.spins_enabled(), wp._w.isEnabled())
""")
    lines = _lines(out)
    assert lines[0] == "1", lines[0]
    # 全图：滑窗参数与模板都禁用、区域按钮也不可用
    assert lines[1] == "full False False False", lines[1]
    # 滑窗：参数可用、模板可用、区域按钮禁用
    assert lines[2] == "window True True False", lines[2]
    # 区域：参数与模板禁用、区域按钮可用
    assert lines[3] == "roi False False True", lines[3]
    assert lines[4] == "False False False", lines[4]


def test_real_scope_panel_never_shows_a_thumbnail_preview():
    """面板里**不再有**缩略预览小图：网格/区域直接画在主画布上。

    用户要求「直接在画布展示，不要单独小窗显示全图」。所以面板里不该再有
    `WindowOverlayPreview` 这类控件——留着就会又出现一个「单独小窗显示全图」。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.ui.widgets.window_overlay_preview import WindowOverlayPreview
from pastelabel.engine.inference_scope import MODE_WINDOW

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


def test_real_scope_panel_anchors_between_the_two_lists():
    """面板竖直位置落在「背景图列表」与「标签列表」之间。

    用户澄清「上面是层级不是位置」——它指的是插入层级/顺序上处在两者中间，
    而不是像素上贴着列表顶端。这里用两个真实列表控件的几何来钉：弹窗的
    竖直中心必须落在背景图列表底边与标签列表顶边之间。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.main_window import ImageEditor

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
p_center = p_top + panel.height() // 2
print('bg', bg_top, bg_bottom)
print('label_top', lb_top)
print('panel_top', p_top)
print('panel_center', p_center)
print('panel_top >= bg_top', p_top >= bg_top)
print('panel_center <= lb_top', p_center <= lb_top)
panel.close()
app.processEvents()
""")
    lines = _lines(out)
    bg_top, bg_bottom = (int(v) for v in lines[0].split()[1:])
    lb_top = int(lines[1].split()[1])
    p_top = int(lines[2].split()[1])
    p_center = int(lines[3].split()[1])
    assert lines[4] == "panel_top >= bg_top True", lines[4]
    assert lines[5] == "panel_center <= lb_top True", lines[5]
    # 量的是「确实有列表」而不是退化成 0 高度
    assert bg_bottom > bg_top, lines[0]
    assert lb_top > bg_top, (bg_top, lb_top)


def test_real_scope_panel_embeds_window_params_without_second_dialog():
    """真实控件：滑窗参数就在面板里，点它**不会**再开第二个窗口。

    这是用户报的问题（「点击滑窗怎么多了一个弹窗」）：面板里必须有真实可用的
    宽/高/重叠输入框，而不是一个再弹窗的按钮。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QSpinBox
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_WINDOW

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
old_btn = [b for b in p.findChildren(type(p.ok_btn))
           if "滑窗参数" in b.text()]
print(len(old_btn))
""")
    lines = _lines(out)
    assert lines[0] == "3", lines[0]
    assert lines[1] == "{'w': 640, 'h': 640, 'overlap': 200}", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "0", lines[3]


def test_real_scope_panel_inline_params_track_edits():
    """真实控件：**面板内**改参数，窗口数立刻跟着变（不需要开第二个窗）。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_WINDOW

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
    assert lines[2] == ("('window', {'w': 1280, 'h': 1280, 'overlap': 0})"), lines[2]


def test_real_scope_panel_asks_the_canvas_to_redraw_on_change():
    """改参数要发 `overlay_changed`：主画布上的网格靠这条信号重画。

    网格已经不在面板里画了，所以「参数变了但信号没发」的后果是**画面不更新**
    （用户改完宽高，画布上的格子还是旧的），而不是某个控件没刷新。这条钉住
    信号真的会发，且切模式也会发。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

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
from pastelabel.ui.widgets.window_overlay_preview import WindowOverlayPreview
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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap
from pastelabel.ui.main_window import ImageEditor
from pastelabel.engine.inference_scope import MODE_WINDOW, MODE_ROI

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
    canvas.render(pnt)
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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap
from pastelabel.ui.main_window import ImageEditor
from pastelabel.engine.inference_scope import MODE_WINDOW

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
from PyQt5.QtWidgets import QSpinBox

w._open_scope_panel()
app.processEvents()
panel = w._scope_panel
panel.set_mode(MODE_WINDOW)
app.processEvents()

def count_grid():
    im = QImage(canvas.size(), QImage.Format_ARGB32)
    im.fill(QColor('white'))
    pnt = QPainter(im)
    canvas.render(pnt)
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
    assert lines[2] == "{'w': 400, 'h': 300, 'overlap': 200}", lines[2]


def test_real_scope_panel_selection_reports_windows_only_when_needed():
    """真实控件：全图/区域不带窗口参数，滑窗类带。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2)
for mode in (MODE_FULL, MODE_ROI, MODE_WINDOW):
    p.set_mode(mode)
    print(p.selection())
""")
    lines = _lines(out)
    assert lines[0] == "('full', None)", lines[0]
    assert lines[1] == "('roi', None)", lines[1]
    assert lines[2] == "('window', {'w': 640, 'h': 640, 'overlap': 200})", lines[2]


def test_real_scope_panel_window_summary_text():
    """真实控件：窗口摘要带上尺寸与重叠；没传参数时用控件默认值。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel

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


def test_real_scope_panel_edit_regions_signal():
    """真实控件：点「编辑区域」发出跳转请求。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_ROI

app = QApplication([])
p = ScopePanel(region_count=1)
p.set_mode(MODE_ROI)
got = []
p.edit_regions_requested.connect(lambda: got.append(1))
p.region_btn.click()
print(len(got))
""")
    assert _lines(out)[0] == "1"


def test_real_scope_panel_retranslates():
    """真实控件：切语言后按钮文字跟着变。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.ui import i18n
from pastelabel.engine.inference_scope import MODE_FULL

app = QApplication([])
p = ScopePanel(region_count=1)
print(p._buttons[MODE_FULL].text(), p.region_btn.text(), p.ok_btn.text())
i18n.set_lang('en')
p.retranslate()
print(p._buttons[MODE_FULL].text(), p.region_btn.text(), p.ok_btn.text())
i18n.set_lang('zh')
""")
    lines = _lines(out)
    assert lines[0] == "全图 编辑区域 确定", lines[0]
    assert lines[1] == "Full image Edit regions OK", lines[1]


# ---------- mixin 快照 ----------

def _scope_host(mode=MODE_FULL, regions=None, size=(200, 100), windows=None,
                saved_windows=None):
    """最小宿主：只替换 `_scope_for_run` 依赖的画布/配置读取。

    `_reference_image_size`、`_region_boxes`、`_saved_window_spec` 都由
    `InferenceMixin` 自己定义，子类里覆盖它们即可（不能靠 MRO 让另一个基类
    的方法胜出——mixin 在左边，永远先命中）。
    """
    from pastelabel.ui.mixins.inference import InferenceMixin

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
    from pastelabel.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._region_boxes() == []


def test_reference_image_size_tolerates_missing_image():
    """没有当前图时返回 None，不崩。"""
    from pastelabel.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._reference_image_size() is None


def test_scope_mode_defaults_to_full_when_unset():
    """从未设置过范围时视为全图（保持既有行为）。"""
    from pastelabel.ui.mixins.inference import InferenceMixin

    class _Host(InferenceMixin):
        pass

    assert _Host()._scope_for_run() is None

# ---------- 主窗口接线（真实 Qt） ----------

_MIXIN_HOST = """
import os
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QImage
app = QApplication([])
from pastelabel.ui.mixins.inference import InferenceMixin
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI


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
    # 范围接线用的画布替身：只暴露会话接口。
    # 「会话内交互到底怎么变」由 tests/test_inference_region_session.py 用真
    # Canvas 钉；这里只验 mixin 有没有正确开/收会话。

    def __init__(self):
        self._active = False
        self.begin_count = 0
        self.end_count = 0

    def region_session_active(self):
        return self._active

    def begin_region_session(self):
        if self._active:
            return False
        self._active = True
        self.begin_count += 1
        return True

    def end_region_session(self):
        if not self._active:
            return
        self._active = False
        self.end_count += 1

    def update(self):
        pass


class _Host(InferenceMixin, QWidget):
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
from PyQt5.QtWidgets import QApplication

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
print("关闭后会话已收:", h.canvas.region_session_active() is False)
""")
    lines = _lines(out)
    assert lines[0] == "面板已建: True", lines
    assert lines[1] == "parent 是宿主: True", lines
    assert lines[2].startswith("windows 是字典: True"), lines
    assert lines[3] == "非模态: True", lines
    assert lines[4] == "可见: True", lines
    assert lines[5] == "无模态窗: True", lines
    assert lines[6] == "关闭后会话已收: True", lines


def test_real_mixin_edit_regions_starts_session_without_switching_mode():
    """「编辑区域」就地开绘制会话，**不切贴图模式**（③）。

    这条是 spec 第 9 条的反向断言：以前这里切 `_set_edit_mode('paste')`，
    现在必须一次都不切。`switched_to` 是宿主上的探针，专门用来抓
    「偷偷切过去又切回来」——只断言 edit_mode 值不变是不够的。

    踩过的坑仍保留在本用例的意图里：入口必须真的产生可观察效果
    （会话激活），否则就是「按钮点了没反应」的旧毛病。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._edit_regions_for_scope()
print(h.switched_to)
print(h.canvas.region_session_active())
print(h.status_label.text_)
""")
    lines = _lines(out)
    assert lines[0] == "[]", lines[0]
    assert lines[1] == "True", lines[1]
    assert "拖动" in lines[2] or "区域" in lines[2], lines[2]


def test_real_mixin_edit_regions_is_noop_without_canvas():
    """宿主没有 canvas 时不炸：防御式取属性，静默返回。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h.canvas = None
h._edit_regions_for_scope()
print(h.switched_to)
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "[]", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_mixin_edit_regions_twice_keeps_one_session():
    """重复点「编辑区域」不重复开会话：第二次 begin 返回 False。"""
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._edit_regions_for_scope()
h._edit_regions_for_scope()
print(h.canvas.region_session_active())
print(h.canvas.begin_count)
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]


def test_real_mixin_panel_open_drives_region_display():
    """区域轮廓显示绑「范围弹窗是否打开」：开则显示、关则不显示（B）。

    弹窗关着时仍保留原有行为：选了区域范围就显示（否则用户不知道框为什么
    只出现在某块地方）。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]

# 弹窗打开（全图模式）：也要显示区域轮廓，用户才能边看边画
h._scope_mode = MODE_FULL
h._scope_panel_open = True
h._sync_scope_region_display()
print('open-full', getattr(h, '_show_scope_regions', None))

# 弹窗关闭、全图模式：不显示
h._scope_panel_open = False
h._sync_scope_region_display()
print('closed-full', getattr(h, '_show_scope_regions', None))

# 弹窗关闭、区域模式：仍显示（原有行为）
h._scope_mode = MODE_ROI
h._sync_scope_region_display()
print('closed-roi', getattr(h, '_show_scope_regions', None))

# 弹窗打开、滑窗模式：显示
h._scope_mode = MODE_WINDOW
h._scope_panel_open = True
h._sync_scope_region_display()
print('open-window', getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "open-full True", lines[0]
    assert lines[1] == "closed-full False", lines[1]
    assert lines[2] == "closed-roi True", lines[2]
    assert lines[3] == "open-window True", lines[3]


def test_scope_modules_have_all_translations():
    """范围面板/滑窗弹窗里的 tr() 键都要有两种语言。

    两个模块的键有一部分是**变量**（`tr(name)` / `tr(names[mode])`），正则
    扫不到，所以这里连 `_MODE_KEYS` 一起查，别只依赖源码扫描。
    """
    import re

    from pastelabel.ui import i18n
    from pastelabel.ui.scope_panel import _MODE_KEYS

    keys = set()
    for rel in ("pastelabel/ui/scope_panel.py",
                "pastelabel/ui/sliding_window_dialog.py"):
        source = (ROOT / rel).read_text(encoding="utf-8")
        keys |= set(re.findall(r"tr\(['\"]([^'\"]+)['\"]\)", source))
    for _mode, name, hint in _MODE_KEYS:
        keys |= {name, hint}
    keys |= {"全图", "滑窗", "区域"}

    missing = sorted((key, lang) for key in keys for lang in ("zh", "en")
                     if key not in i18n._strings[lang])
    assert not missing, missing


def test_real_mixin_toggles_scope_region_display():
    """真实控件：范围含区域时打开画布区域轮廓，切回全图/滑窗时关掉。

    标注模式下也要看得见范围，否则用户不知道框为什么只出现在某块地方。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
for mode in (MODE_ROI, MODE_WINDOW, MODE_FULL):
    h._scope_mode = mode
    h._refresh_scope_button()
    print(mode, getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "roi True", lines[0]
    assert lines[1] == "window False", lines[1]
    assert lines[2] == "full False", lines[2]


def test_real_window_dialog_warns_when_many_windows():
    """真实控件：窗口数很多时给耗时提醒，但仍允许确定。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

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
from pastelabel.engine.inference_scope import ScopeError
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
from pastelabel.core import config_manager
saved = config_manager.load_all()['inference_settings']
print('mode' in saved, 'window_spec' in saved)
""")
    lines = _lines(out)
    assert lines[0] == "{'w': 512, 'h': 512, 'overlap': 64}", lines[0]
    assert lines[1] == "False True", lines[1]


def test_real_scope_panel_positions_on_its_anchor():
    """真实控件：弹窗**竖直居中**在锚点那块（全局坐标），而不是屏幕正中。

    锚点现在是「背景图列表底边 ~ 标签列表顶边」那条带，所以竖直方向要按
    带的中心对齐——只把顶边对齐会让弹窗整块压到下面的标签列表上，那就不是
    用户要的「两个列表中间」了。

    水平仍取锚点左侧 +8：不要求装进侧栏（弹窗最小宽 440px、侧栏 350px，
    装不下），要的是「落在列表中间的层级上、别盖画布」。
    锚点取「屏幕上偏左」是为了避开钳制：贴到中右部会被钳回右边（那是
    `test_real_scope_panel_clamps_anchor_to_screen` 的事）。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QRect
from pastelabel.ui.scope_panel import ScopePanel

app = QApplication([])
avail = app.primaryScreen().availableGeometry()
# 左侧锚点：给 440px 弹窗留得下，钳制不会介入
anchor = QRect(avail.left() + 10, avail.top() + 30, 350,
               min(400, avail.height() - 40))
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
anchor_center_global_y = host.mapToGlobal(anchor.center()).y()
panel_center_global_y = panel_tl.y() + panel.height() // 2
# 水平：贴着锚点左边（+8 左右）
print(anchor_tl.x() <= panel_tl.x() <= anchor_tl.x() + 40)
# 竖直：弹窗中心与锚点中心对齐（允许几像素的取整误差）
print(abs(panel_center_global_y - anchor_center_global_y) <= 4)
# 且确实不再贴顶（贴顶就说明居中没生效）
print(panel_tl.y() != anchor_tl.y())
panel.close()
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", f"没竖直居中于锚点: {lines}"
    assert lines[2] == "True", f"仍在贴锚点顶边: {lines}"


def test_real_scope_panel_clamps_anchor_to_screen():
    """锚点超出屏幕可用区时钳回屏内：不能把弹窗推到看不见的地方。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QRect
from pastelabel.ui.scope_panel import ScopePanel

app = QApplication([])
avail = app.primaryScreen().availableGeometry()
host = QWidget()
host.resize(avail.width(), avail.height())
host.show()
panel = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, parent=host)
# 故意把锚点放到屏幕右边缘之外
panel.set_anchor_rect(QRect(avail.right() + 50, avail.top(), 400, 400))
panel.show()
g = panel.geometry()
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
from PyQt5.QtWidgets import QApplication, QWidget
from pastelabel.ui.scope_panel import ScopePanel

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.engine.inference_scope import iter_windows

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.engine.inference_scope import iter_windows

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=None)
print(d.window_count(), d.params.is_valid())
d.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None True", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_sliding_dialog_template_fills_and_updates_count():
    """真实控件：点模板 → 回填输入框 + 窗口数**确实变了**。

    断言「变了」而不是「不为 None」：后者在完全不重算的情况下也会过
    （旧 spec 还在，仍算得出窗口数）——变异验证时发现该断言杀不死
    「点击不回填/不重算」这个变异体。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 320, 'h': 320, 'overlap': 100,
                                  'min_visible': 0.5})
d = SlidingWindowDialog(reference_size=(2000, 1000),
                        spec={'w': 640, 'h': 640, 'overlap': 200})
before = d.window_count()
print(len(d.template_buttons()) > 0)
d.template_buttons()[0].click()
print(d.spec())
after = d.window_count()
print(before, after, before != after)
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "{'w': 320, 'h': 320, 'overlap': 100}", lines[1]
    # 2000×1000 图上：640/重叠 200 → 10 窗；320/重叠 100 → 45 窗
    assert lines[2] == "10 45 True", lines[2]


def test_real_sliding_dialog_template_still_editable_after_click():
    """点模板后仍可继续改宽高：不被模板值锁死，也不禁用输入框。

    断言 disabled 状态是必要的：只断言 setValue 生效的话，
    「点模板后 disable 掉输入框」这个变异体杀不死——程序化 setValue
    在 disabled 控件上一样能改值（变异验证时证实过）。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 384, 'h': 384, 'overlap': 64,
                                  'min_visible': 0.5})
d = SlidingWindowDialog(reference_size=(2000, 1000))
d.template_buttons()[0].click()
print(d._w.isEnabled(), d._h.isEnabled(), d._ov.isEnabled())
d._w.setValue(900)
print(d.spec())
""")
    lines = _lines(out)
    assert lines[0] == "True True True", lines[0]
    assert lines[1] == "{'w': 900, 'h': 384, 'overlap': 64}", lines[1]


def test_real_sliding_dialog_has_no_templates_when_config_empty():
    """配置里没有模板时不显示模板按钮（也不崩）。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.core import config_manager

app = QApplication([])
cfg = config_manager.load_config()
cfg['crop_templates'] = []
config_manager.save_config(cfg)
d = SlidingWindowDialog(reference_size=(2000, 1000))
print(len(d.template_buttons()))
""")
    assert _lines(out)[0] == "0", out


def test_real_sliding_dialog_does_not_write_template_by_itself():
    """弹窗自身只读不写：操作后配置里的模板数不变（写模板由调用方在确定后做）。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.core import config_manager

app = QApplication([])
config_manager.add_crop_template({'w': 384, 'h': 384, 'overlap': 64,
                                  'min_visible': 0.5})
before = len(config_manager.load_crop_templates())
d = SlidingWindowDialog(reference_size=(2000, 1000))
d._w.setValue(777)
d.window_count()
d.template_buttons()[0].click() if d.template_buttons() else None
after = len(config_manager.load_crop_templates())
print(before == after, before > 0)
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]


def test_real_mixin_records_template_only_on_accept():
    """点「确定」才记模板；「取消」与直接关闭都不记（两者等价）。

    这是 spec 第 7/8 条的落地断言。参数内嵌进面板后不再有第二个窗口，所以
    直接驱动 `finished_ok` 走真实路径，而不是拿替身拦 `exec_()`——那样恰好
    把「非模态 + 信号回传」这条新机制绕过去了。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from pastelabel.core import config_manager as cm
from pastelabel.engine.inference_scope import MODE_WINDOW

h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")

# 1) 非滑窗模式下点确定：不该记模板（窗口参数对全图模式没有意义）
h._scope_mode = MODE_FULL
before = len(cm.load_crop_templates())
h._on_scope_panel_closed(True)
print("full", len(cm.load_crop_templates()) - before)

# 2) 取消：不该记模板，也不该改范围
h._scope_mode = MODE_FULL
h._window_spec = {'w': 640, 'h': 640, 'overlap': 200}
before = len(cm.load_crop_templates())
h._on_scope_panel_closed(False)
print("cancel", len(cm.load_crop_templates()) - before)
print("cancel mode", h._scope_mode)

# 3) 滑窗模式点确定：记一条
import pastelabel.ui.scope_panel as sp

class _Panel:
    Accepted = 1
    def __init__(self):
        pass
    def selection(self):
        return (MODE_WINDOW, {'w': 512, 'h': 512, 'overlap': 64})
    def set_region_count(self, n):
        pass

h._scope_panel = _Panel()
before = len(cm.load_crop_templates())
h._on_scope_panel_closed(True)
print("accept", len(cm.load_crop_templates()) - before)
print("spec", h._window_spec)

# 4) 同样的参数再点一次确定：**不重复写**（防把增强侧真模板挤掉）
before = len(cm.load_crop_templates())
h._on_scope_panel_closed(True)
print("repeat", len(cm.load_crop_templates()) - before)

# 5) 参数变了再确定：应记一条新的
class _Panel2(_Panel):
    def selection(self):
        return (MODE_WINDOW, {'w': 1024, 'h': 1024, 'overlap': 128})

h._scope_panel = _Panel2()
before = len(cm.load_crop_templates())
h._on_scope_panel_closed(True)
print("changed", len(cm.load_crop_templates()) - before)
""")
    lines = _lines(out)
    assert lines[0] == "full 0", lines[0]
    assert lines[1] == "cancel 0", lines[1]
    assert lines[2] == "cancel mode full", lines[2]
    assert lines[3] == "accept 1", lines[3]
    assert lines[4] == "spec {'w': 512, 'h': 512, 'overlap': 64}", lines[4]
    # 关键：重复确定不再重复写（B-3）
    assert lines[5] == "repeat 0", lines[5]
    assert lines[6] == "changed 1", lines[6]


def test_recorded_template_does_not_fabricate_min_visible():
    """记进共享模板时**不伪造** `min_visible`（复审 B-3）。

    早先硬塞 `min_visible=0.5` 只为过校验，那是凭空的数。现在缺该字段时由
    `_normalize_crop_template` 用与数据增强默认一致的 0.7 补齐。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.core import config_manager as cm

app = QApplication([])
# 清空模板列表：直接写配置文件，避免受同一进程里别的用例影响
cfg = cm.load_config()
cfg['crop_templates'] = []
cm.save_config(cfg)
print(cm.load_crop_templates())

ok = cm.add_crop_template({'w': 512, 'h': 512, 'overlap': 64})
print(ok)
tpl = cm.load_crop_templates()
print(len(tpl))
print(tpl[0]['w'], tpl[0]['h'], tpl[0]['overlap'])
print(tpl[0]['min_visible'])
""")
    lines = _lines(out)
    assert lines[0] == "[]", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "1", lines[2]
    assert lines[3] == "512 512 64", lines[3]
    # 与数据增强侧输入框默认值一致
    assert lines[4] == "0.7", lines[4]


def test_real_mixin_panel_close_ends_session_on_every_path():
    """关面板一定收会话并关掉区域显示：确定 / 取消 / 直接关闭三条路径都一样。

    会话若漏收，标注模式的鼠标行为会被一直劫持，所以三条路径都要钉。
    用**真的** ScopePanel + 真的 `close()`/`accept()`/`reject()`：非模态后
    这些调用会立即返回，不再需要替身拦 `exec_()`；替身恰好会把新机制
    （`finished_ok` 信号回传）整条绕过去，测不到。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PyQt5.QtWidgets import QApplication

def scenario(close_call):
    h = _Host()
    h._on_model_loaded(_Detector(), (640, 640), "model")
    h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
    h._scope_mode = h._scope_mode  # noqa
    h._open_scope_panel()
    # 面板开着时开一个区域绘制会话，模拟用户点了「编辑区域」
    h._edit_regions_for_scope()
    during_session = h.canvas.region_session_active()
    during_show = h._show_scope_regions
    getattr(h._scope_panel, close_call)()
    QApplication.processEvents()
    return (during_session, during_show, h.canvas.region_session_active(),
            h._show_scope_regions, h.canvas.end_count)

for call in ('accept', 'reject', 'close'):
    print(call, scenario(call))
""")
    lines = _lines(out)
    for i, call in enumerate(("accept", "reject", "close")):
        # 开着的会话确实激活、轮廓确实在显示
        assert lines[i] == (f"{call} (True, True, False, False, 1)"), lines[i]


def test_real_mixin_scope_panel_closing_never_leaves_session_open():
    """无论以哪条路径关闭，都不能留下未收的会话（防鼠标行为被劫持）。"""
    out = _run_real_qt(_MIXIN_HOST + """
from PyQt5.QtWidgets import QApplication

h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")
h._open_scope_panel()
h._edit_regions_for_scope()
# 模拟异常路径：宿主在收尾途中抛异常，也必须已经收干净
panel = h._scope_panel
try:
    panel.reject()
    QApplication.processEvents()
    raise RuntimeError('boom')
except RuntimeError:
    pass
print("session", h.canvas.region_session_active())
print("open flag", getattr(h, '_scope_panel_open', None))
print("show", getattr(h, '_show_scope_regions', None))
""")
    lines = _lines(out)
    assert lines[0] == "session False", lines[0]
    assert lines[1] == "open flag False", lines[1]
    assert lines[2] == "show False", lines[2]


