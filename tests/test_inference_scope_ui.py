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


def test_real_row_scope_summary_shows_mode_and_window_params():
    """真实控件：范围摘要显示模式；滑窗模式带窗口参数。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
from pastelabel.ui.i18n import set_lang

app = QApplication([])
row = ModelRow()
row.set_model('m.onnx', 0.25, 0.45, (640, 640), ['CPUExecutionProvider'])
row.set_scope_summary('全图')
print(row.scope_btn.text())
row.set_scope_summary('滑窗', ' 640×640/200')
print(row.scope_btn.text())
row.set_scope_summary('区域+滑窗', ' 512×512/100')
print(row.scope_btn.text())
row.set_scope_summary('区域')
print(row.scope_btn.text())
""")
    lines = _lines(out)
    assert lines[0] == "推理范围", lines[0]
    assert lines[1] == "推理范围: 滑窗 640×640/200", lines[1]
    assert lines[2] == "推理范围: 区域+滑窗 512×512/100", lines[2]
    assert lines[3] == "推理范围: 区域", lines[3]


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
    """真实控件：窗口按钮只在滑窗显示，区域按钮只在区域显示，全图都不显示。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    p.set_mode(mode)
    print(mode, p.window_btn.isHidden(), p.region_btn.isHidden())
""")
    lines = _lines(out)
    assert lines[0] == "full True True", lines[0]
    assert lines[1] == "window False True", lines[1]
    assert lines[2] == "roi True False", lines[2]


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
    """真实控件：窗口摘要带上尺寸与重叠。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 512, 'overlap': 100})
print(repr(p.window_summary()))
print(repr(ScopePanel().window_summary()))
""")
    lines = _lines(out)
    assert lines[0] == "'640×512 重叠 100'", lines[0]
    assert lines[1] == "''", lines[1]


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
print(h.model_row.scope_btn.text())
print(h._scope_for_run())

h.region_boxes = [{'x': 0, 'y': 0, 'width': 100, 'height': 100}]
h._scope_mode = MODE_ROI
h._refresh_scope_button()
print(h.model_row.scope_btn.text())
print(h._scope_for_run())

h._scope_mode = MODE_FULL
h._refresh_scope_button()
print(h.model_row.scope_btn.text(), h._scope_for_run())
""")
    lines = _lines(out)
    assert lines[0] == "请先加载模型", lines[0]
    assert lines[1] == "True 推理范围", lines[1]
    assert lines[2] == "(400, 300)", lines[2]
    assert lines[3] == "推理范围: 滑窗 100×100/50", lines[3]
    assert lines[4] == "{'mode': 'window', 'windows': {'w': 100, 'h': 100, 'overlap': 50}}", lines[4]
    assert lines[5] == "推理范围: 区域", lines[5]
    assert "'reference_size': (400, 300)" in lines[6], lines[6]
    assert lines[7] == "推理范围 None", lines[7]


def test_real_mixin_open_scope_panel_constructs_panel_with_loaded_model():
    """真实控件：加载模型后点「推理范围」要能真的把面板建起来。

    踩过的坑：`ScopePanel(self)` 把 editor 当成第一个位置参数 `windows`，
    运行时报 `TypeError: 'ImageEditor' object is not iterable`。原用例在
    **加载模型前**调 `_open_scope_panel()`，撞上「请先加载模型」的提前返回，
    根本没走到构造那行，所以一直是绿的。这里必须先加载模型，再拦下 `exec_()`
    不让它真的开窗口（否则测试会挂住）。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._on_model_loaded(_Detector(), (640, 640), "model")

import pastelabel.ui.scope_panel as sp
created = []
orig_exec = sp.ScopePanel.exec_
def fake_exec(self):
    created.append(self)
    return 0
sp.ScopePanel.exec_ = fake_exec
try:
    h._open_scope_panel()
finally:
    sp.ScopePanel.exec_ = orig_exec

print("面板已建:", len(created) == 1)
panel = created[0]
# 关键：parent 必须是宿主，windows 不能被塞成 editor
print("parent 是宿主:", panel.parent() is h)
print("windows 是字典:", isinstance(panel.windows(), dict), panel.windows())
""")
    lines = _lines(out)
    assert lines[0] == "面板已建: True", lines
    assert lines[1] == "parent 是宿主: True", lines
    assert lines[2].startswith("windows 是字典: True"), lines


def test_real_mixin_edit_regions_switches_to_paste_mode():
    """「编辑区域」必须真的切到贴图模式，否则用户没法画区域。

    踩过的坑：这里原来调 `_switch_to_paste_mode()`，**这个方法根本不存在**，
    `getattr(..., None)` 兜住后静默什么都不做——按钮点了没反应，也没有报错。
    真实入口是主窗口的 `_set_edit_mode('paste')`。
    """
    out = _run_real_qt(_MIXIN_HOST + """
h = _Host()
h._edit_regions_for_scope()
print(h.switched_to)
print(h.status_label.text_)
""")
    lines = _lines(out)
    assert lines[0] == "[('paste', True)]", lines[0]
    assert "贴图模式" in lines[1], lines[1]


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
print(h.model_row.scope_btn.text())
h._on_model_loaded(_Detector(), (640, 640), "model")
print(h.model_row.scope_btn.text(), h._scope_for_run())
""")
    lines = _lines(out)
    assert lines[0] == "推理范围: 区域", lines[0]
    assert lines[1] == "推理范围 None", lines[1]


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


def test_real_scope_panel_positions_over_side_panel():
    """真实控件：弹窗左上角落在右侧栏那块（全局坐标），而不是屏幕正中。

    不要求装进侧栏：范围弹窗最小宽 440px、侧栏固定 350px，装不下。
    用户要的是「盖在背景图列表上方、别盖画布」。
    锚点取「屏幕中部偏左」是为了避开钳制：800px 屏上 440px 弹窗若贴到
    中右部就会被钳回右边（那是另一个用例的事），这里单独验「跟着锚点走」。
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
# 用全局坐标比较：panel.geometry() 是相对父窗口的，边界值易混
panel_tl = panel.mapToGlobal(panel.rect().topLeft())
anchor_tl = host.mapToGlobal(anchor.topLeft())
print(anchor_tl.x() <= panel_tl.x() <= anchor_tl.x() + 40)
print(anchor_tl.y() <= panel_tl.y() <= anchor_tl.y() + 40)
panel.close()
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", lines[1]


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


def test_real_sliding_dialog_preview_follows_params():
    """真实控件：改参数 → 预览窗口数跟着变，且与引擎一致。

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
before = d.preview.window_count()
d._w.setValue(500)
d._ov.setValue(0)
after = d.preview.window_count()
print(before, after)
print(before != after)
print(d.preview.is_valid())
print(after == len(iter_windows(2000, 1000, d.spec())))
""")
    lines = _lines(out)
    assert lines[0] == "10 8", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", lines[3]


def test_real_sliding_dialog_preview_matches_engine_count():
    """预览窗口数必须等于 iter_windows()：画面上的网格就是实际会跑的窗口。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog
from pastelabel.engine.inference_scope import iter_windows

app = QApplication([])
d = SlidingWindowDialog(reference_size=(2000, 1000))
spec = d.spec()
print(d.preview.window_count() == len(iter_windows(2000, 1000, spec)))
""")
    assert _lines(out)[0] == "True", out


def test_real_sliding_dialog_preview_unusable_without_reference():
    """没有参考图时预览显示不可用，但弹窗仍能开、不崩。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.sliding_window_dialog import SlidingWindowDialog

app = QApplication([])
d = SlidingWindowDialog(reference_size=None)
print(d.preview.window_count(), d.preview.is_valid())
d.repaint()
print('ok')
""")
    lines = _lines(out)
    assert lines[0] == "None False", lines[0]
    assert lines[1] == "ok", lines[1]


def test_real_sliding_dialog_template_fills_and_refreshes_preview():
    """真实控件：点模板 → 回填输入框 + 预览窗口数**确实变了**。

    断言「变了」而不是「不为 None」：后者在预览完全不刷新的情况下也会过
    （旧 spec 还在，仍算得出窗口数）——变异验证时发现该断言杀不死
    「点击不回刷预览」这个变异体。
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
before = d.preview.window_count()
print(len(d.template_buttons()) > 0)
d.template_buttons()[0].click()
print(d.spec())
after = d.preview.window_count()
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
d.preview.window_count()
d.template_buttons()[0].click() if d.template_buttons() else None
after = len(config_manager.load_crop_templates())
print(before == after, before > 0)
""")
    lines = _lines(out)
    assert lines[0] == "True True", lines[0]


def test_real_scope_panel_records_template_only_on_accept():
    """点「确定」才记模板；「取消」与直接关闭都不记（两者等价）。

    这是 spec 第 7/8 条的落地断言。用替身弹窗控制 exec_ 的返回值，
    避免测试里真开模态窗。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication, QDialog
from pastelabel.core import config_manager as cm
from pastelabel.ui import scope_panel as sp
import pastelabel.ui.sliding_window_dialog as swd

app = QApplication([])
Real = swd.SlidingWindowDialog

class FakeAccepted(Real):
    def exec_(self):
        self._w.setValue(512)
        self._h.setValue(512)
        self._ov.setValue(64)
        return QDialog.Accepted

class FakeRejected(Real):
    def exec_(self):
        self._w.setValue(999)
        return QDialog.Rejected

panel = sp.ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200})

swd.SlidingWindowDialog = FakeAccepted
before = len(cm.load_crop_templates())
panel._edit_windows()
after = len(cm.load_crop_templates())
print(before, after)
print(panel._windows)

swd.SlidingWindowDialog = FakeRejected
before2 = len(cm.load_crop_templates())
panel._edit_windows()
after2 = len(cm.load_crop_templates())
print(before2, after2)
print(panel._windows)
""")
    lines = _lines(out)
    # 断言「增量」而不是绝对条数：配置路径是本进程共享的，同批其它用例
    # 可能已经写过模板，硬编码起始条数会变成顺序依赖的假红。
    accepted_before, accepted_after = (int(v) for v in lines[0].split())
    assert accepted_after == accepted_before + 1, lines[0]
    assert lines[1] == "{'w': 512, 'h': 512, 'overlap': 64}", lines[1]
    rejected_before, rejected_after = (int(v) for v in lines[2].split())
    assert rejected_after == rejected_before, lines[2]
    # 取消不该改参数：仍是确定后的值，不是替身设的 999
    assert lines[3] == "{'w': 512, 'h': 512, 'overlap': 64}", lines[3]
