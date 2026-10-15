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
    """真实控件：滑窗参数控件**始终显示**，区域按钮只在区域模式显示。

    用户要求「滑窗参数打开窗口就要存在」——参数区不再随模式显隐，一打开面板
    就能看到当前窗口尺寸与模板。能不能改仍由模式决定（见下一条用例）。
    """
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
    # 参数区三种模式都在（「打开窗口就要存在」）
    assert lines[0] == "full True False", lines[0]
    assert lines[1] == "window True False", lines[1]
    assert lines[2] == "roi True True", lines[2]
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
# 圆点刷新（`_refresh_background_item_status`）在列表 mixin 里；范围刷新用例要
# 走真实实现，别在替身里重写一遍「刷圆点」的语义。
from pastelabel.ui.mixins.background_list import BackgroundListMixin as _BackgroundListMixin
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
        # 圆点刷新要向 `background_list` 写 BG_ROLE_STATUS，`_sync_current_boxes_from_cache`
        # 还要 `update_label_list`。给真控件（QListWidget）而不是替身：状态图标是
        # 通过 `setData/setIcon` 落到 item 上的，替身会把断言变成「测替身」。
        from PyQt5.QtWidgets import QListWidget
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
                from PyQt5.QtCore import QRect
                return QRect(0, 0, self._r[2], self._r[3])

            def mapTo(self, parent, point):
                from PyQt5.QtCore import QPoint
                return QPoint(self._r[0] + point.x(), self._r[1] + point.y())

            def width(self):
                return self._r[2]

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
        from PyQt5.QtCore import QRect
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
    """锚点超出屏幕可用区时钳回屏内：不能把弹窗推到看不见的地方。

    量的是 `frameGeometry()`（**全局**坐标）：弹窗是顶层窗口，`geometry()`
    给的是父窗坐标，拿去和屏幕可用区比是两套坐标系，会假红。
    """
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

    场景取「区域」范围：这里要验的正是「面板开着 → 轮廓在显示 → 关掉 →
    会话收了、轮廓也收起」。轮廓的收起对应用户报的「点确认后画布上的区域框
    还在」——面板只是调范围时的参照，关掉就不该继续叠加在画布上。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PyQt5.QtWidgets import QApplication
from pastelabel.engine.inference_scope import MODE_ROI

def scenario(close_call):
    h = _Host()
    h._on_model_loaded(_Detector(), (640, 640), "model")
    h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
    h._scope_mode = MODE_ROI
    h._open_scope_panel()
    # 面板开着时开一个区域绘制会话，模拟用户点了「编辑区域」
    h._edit_regions_for_scope()
    during_session = h.canvas.region_session_active()
    during_show = h._show_scope_regions
    getattr(h._scope_panel, close_call)()
    QApplication.processEvents()
    # 关掉后：会话收了、面板标记清了、区域轮廓也收起（三条路径都一样）
    return (during_session, during_show, h.canvas.region_session_active(),
            h._scope_panel_open, h._show_scope_regions, h.canvas.end_count)

for call in ('accept', 'reject', 'close'):
    print(call, scenario(call))
""")
    lines = _lines(out)
    for i, call in enumerate(("accept", "reject", "close")):
        # 开着的会话确实激活、轮廓确实在显示；关掉后会话收了、面板标记清了、
        # 轮廓收起（用户报的「点确认后区域框还在」）
        assert lines[i] == (f"{call} (True, True, False, False, False, 1)"), lines[i]


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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap
from pastelabel.ui.main_window import ImageEditor
from pastelabel.core.config import REGION_CONFIG
from pastelabel.engine.inference_scope import MODE_WINDOW, MODE_ROI

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
    canvas.render(pnt)
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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

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
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_ROI

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
from PyQt5.QtWidgets import QApplication, QPushButton
from pastelabel.engine.inference_scope import MODE_ROI

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
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPainter, QImage, QColor, QPixmap
from pastelabel.ui.main_window import ImageEditor
from pastelabel.engine.inference_scope import MODE_ROI

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
    canvas.render(pnt)
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
w._scope_panel.ok_btn.click()
app.processEvents()
after_ok = count_edges()

w._open_scope_panel()
w._scope_panel.set_mode(MODE_ROI)
app.processEvents()
w._scope_panel.cancel_btn.click()
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
    """① 范围面板要浮在**右侧侧栏**上层、竖直居中，不能压到画布上。

    用户报的现象：「推理范围窗口怎么在屏幕左边了？如果我软件移过去不就遮挡
    画布了」。根因有两个，这里一并钉住：
      * 锚点曾经是 `QRect(0, ...)`——x 被写死成 0，而画布在主窗左侧，
        x=0 就是画布左端；
      * `move()` 对顶层窗口是**全局**坐标，早先在 `place_within` 里又减了一次
        父窗原点，等于把面板整体左移一个主窗 x 的量。

    量的是 `frameGeometry()`（全局）与侧栏的实际全局位置，两者同构。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication

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
side_center_y = h.mapToGlobal(side.center()).y()
frame = panel.frameGeometry()

# 面板要落在侧栏这块的左边界起、竖直居中；「不压画布」由锚点 x 不是 0 保证
# （画布在主窗左侧，x=0 就是画布左端）。
#
# 水平用「中心线对齐侧栏中心」而不是只判左右边界：锚点够宽时，把 gx 整体
# 平移某个量后左边界仍可能落在 side_left 右侧（被 `gx = left` 兜住或屏幕钳住），
# 只判边界的话『面板左移』这类变异会存活（实测）。
side_center_x = h.mapToGlobal(side.center()).x()
print(frame.x() >= side_left)
print(abs(side_left - frame.x()) < side.width())
print(abs(frame.center().y() - side_center_y) < 40)
print(abs(frame.center().x() - side_center_x) < 8)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        f"面板左边缘跑到侧栏左边之外了（{lines}）：用户报的「跑到屏幕左边」")
    assert lines[1] == "True", (
        f"面板整个跑出侧栏那一块了（{lines}）")
    assert lines[2] == "True", (
        f"面板没有对齐侧栏的竖直中心（{lines}）")
    assert lines[3] == "True", (
        f"面板没有对齐侧栏的水平中心（{lines}）：整体平移能被这条抓住")


def test_real_scope_panel_anchor_uses_the_real_side_panel_x():
    """① 的根因回归：锚点必须带侧栏的真实 x，不能再是 0。

    只钉「面板最终位置」的话，换个屏幕尺寸/窗口位置就可能测不出来；这里直接
    断言锚点矩形与侧栏左边界对齐。
    """
    out = _run_real_qt(_MIXIN_HOST + """
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication

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
side = h._side_panel_geometry()
bg_left = h.bg_list_group.mapTo(h, h.bg_list_group.rect().topLeft()).x()

print(anchor.x() > 0)
print(abs(anchor.x() - bg_left) <= 12)
print(anchor.width() > 0)
# 锚点要覆盖整个侧栏高度（竖直居中的依据），不是两个列表之间那条窄带
print(anchor.height() >= side.height() - 20)
""")
    lines = _lines(out)
    assert lines[0] == "True", (
        f"锚点 x 还是 0（{lines}）：这正是「面板跑到屏幕左边」的根因")
    assert lines[1] == "True", f"锚点左边界没和侧栏对齐（{lines}）"
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", (
        f"锚点高度没覆盖侧栏（{lines}）：竖直居中会偏")


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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

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

from PyQt5.QtGui import QImage
from pastelabel.engine.image_loader.mixin import BG_ROLE_STATUS
from pastelabel.engine.image_loader.status import (
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
from pastelabel.engine.image_loader.status import decorate_background_list_item
from PyQt5.QtWidgets import QListWidgetItem
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
from pastelabel.engine.inference_task import ImageResult
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

from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QListWidgetItem
from pastelabel.engine.image_loader.status import decorate_background_list_item
from pastelabel.engine.inference_task import ImageResult

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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
from pastelabel.ui import i18n
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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
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
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
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
