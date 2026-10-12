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
    MODE_FULL, MODE_ROI, MODE_ROI_WINDOW, MODE_WINDOW, ScopeError,
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
    """真实控件：窗口按钮只在窗口相关模式显示，区域按钮只在区域相关模式显示。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.scope_panel import ScopePanel
from pastelabel.engine.inference_scope import (
    MODE_FULL, MODE_WINDOW, MODE_ROI, MODE_ROI_WINDOW)

app = QApplication([])
p = ScopePanel(windows={'w': 640, 'h': 640, 'overlap': 200}, region_count=2,
               reference_size=(3840, 2160))
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI, MODE_ROI_WINDOW):
    p.set_mode(mode)
    print(mode, p.window_btn.isHidden(), p.region_btn.isHidden())
""")
    lines = _lines(out)
    assert lines[0] == "full True True", lines[0]
    assert lines[1] == "window False True", lines[1]
    assert lines[2] == "roi True False", lines[2]
    assert lines[3] == "roi_window False False", lines[3]


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


def test_scope_for_run_roi_window_carries_both():
    """区域+滑窗同时带区域与窗口参数。"""
    host = _scope_host(mode=MODE_ROI_WINDOW,
                       regions=[{"x": 0, "y": 0, "width": 100, "height": 100}],
                       windows={"w": 64, "h": 64, "overlap": 0},
                       size=(100, 100))

    scope = host._scope_for_run()

    assert scope["mode"] == MODE_ROI_WINDOW
    assert scope["windows"] == {"w": 64, "h": 64, "overlap": 0}
    assert scope["regions"] and scope["reference_size"] == (100, 100)


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
    keys |= {"全图", "滑窗", "区域", "区域+滑窗"}

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
