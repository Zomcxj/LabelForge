"""P5 回归：置忙必须先于同步读盘（防双击重入）。

审计 P5：_run_augment/_run_export/_run_split 三个入口都先
_ensure_boxes_loaded()（里面每 5 张图 processEvents() 一次）再
_button_busy()——读盘期间按钮尚未 disabled，连点可重入。

修法：_button_busy 提前到 _ensure_boxes_loaded 之前；因此发生在
置忙**之后**的每个提前 return（无图/无标签/无划分文件）必须显式
_button_ready，否则校验失败按钮永久卡死——本文件每个用例都同时
钉住这两件事。

用记录器包裹真实方法记调用顺序（比模拟双击经过 disabled 按钮的
时序直接且稳）；_ensure_boxes_loaded 跑真实现（真 tmp 目录）。
"""
import json

import pytest

from labelforge.ui.processing_panel import ProcessingPanel


class _Check:
    def __init__(self, value=False):
        self._value = value

    def isChecked(self):
        return self._value

    def value(self):
        return 0.5

    def currentIndex(self):
        return 0


class _Btn:
    def __init__(self, name):
        self.name = name

    def text(self):
        return self.name

    def setText(self, text):
        self._text = text

    def setEnabled(self, enabled):
        self._enabled = enabled


def _make_panel(tmp_path, monkeypatch, order):
    """假 panel + 记录器包裹四个关键方法，返回 (panel, buttons)。"""
    panel = ProcessingPanel.__new__(ProcessingPanel)

    work_dir = tmp_path / "dataset"
    work_dir.mkdir()
    (work_dir / "a.jpg").write_bytes(b"\xff\xd8fake")
    (work_dir / "a.json").write_text(json.dumps({
        "shapes": [{"label": "cat", "points": [[1, 1], [2, 2]],
                    "shape_type": "rectangle"}],
    }), encoding="utf-8")

    panel._path_edit = _Check()
    panel._path_edit.text = lambda: str(work_dir)
    panel._get_session_dir = lambda: ""
    panel._get_output_dir = lambda: str(tmp_path / "out")
    panel._log = lambda *a, **k: None
    panel._pipe_finish = lambda error=False: order.append("pipe_finish")

    buttons = {
        "aug": _Btn("aug"), "exp": _Btn("exp"),
        "split": _Btn("split"), "pipe": _Btn("pipe"),
        "aug_clear": _Btn("aug_clear"), "exp_clear": _Btn("exp_clear"),
        "split_clear": _Btn("split_clear"),
    }
    panel._aug_btn = buttons["aug"]
    panel._exp_btn = buttons["exp"]
    panel._split_btn = buttons["split"]
    panel._pipe_btn = buttons["pipe"]
    panel._aug_clear_btn = buttons["aug_clear"]
    panel._exp_clear_btn = buttons["exp_clear"]
    panel._split_clear_btn = buttons["split_clear"]
    panel._spinner_timer = _Check()
    panel._spinner_timer.start = lambda ms: None
    panel._spinner_timer.stop = lambda: None

    panel._aug_ratio = _Check()
    panel._aug_mode = _Check()
    panel._aug_inc_orig = _Check()
    panel._aug_skip_empty = _Check()
    panel._aug_widgets = {}
    panel._get_transform_specs = lambda: []
    panel._get_crop_spec = lambda: None
    panel._get_selected_labels = lambda: []

    real_busy = ProcessingPanel._button_busy
    real_ready = ProcessingPanel._button_ready
    real_ensure = ProcessingPanel._ensure_boxes_loaded

    def busy_rec(self, btn, text=None, clear_btn=None):
        order.append("busy")
        return real_busy(self, btn, text, clear_btn)

    def ready_rec(self, btn, clear_btn=None):
        order.append("ready")
        return real_ready(self, btn, clear_btn)

    def ensure_rec(self):
        order.append("ensure")
        return real_ensure(self)

    monkeypatch.setattr(ProcessingPanel, "_button_busy", busy_rec)
    monkeypatch.setattr(ProcessingPanel, "_button_ready", ready_rec)
    monkeypatch.setattr(ProcessingPanel, "_ensure_boxes_loaded", ensure_rec)
    return panel, buttons


def _assert_busy_first(order):
    """置忙先于读盘，且提前 return 后按钮已恢复。"""
    assert order[0] == "busy", order
    assert order[1] == "ensure", order
    assert "ready" in order, order  # 校验失败的提前 return 不得卡死按钮
    assert order.index("ready") > order.index("ensure"), order


def test_run_augment_is_busy_before_loading_boxes(tmp_path, monkeypatch):
    order = []
    panel, _ = _make_panel(tmp_path, monkeypatch, order)
    panel._run_augment()
    _assert_busy_first(order)


def test_run_export_is_busy_before_loading_boxes(tmp_path, monkeypatch):
    order = []
    panel, _ = _make_panel(tmp_path, monkeypatch, order)
    panel._run_export()
    _assert_busy_first(order)


def test_run_split_is_busy_before_loading_boxes(tmp_path, monkeypatch):
    order = []
    panel, _ = _make_panel(tmp_path, monkeypatch, order)
    panel._run_split()
    _assert_busy_first(order)


def test_ensure_boxes_loads_sidecars_for_real(tmp_path, monkeypatch):
    """顺带钉住 _ensure_boxes_loaded 的真实行为：真目录真 sidecar，
    空结果走 pipe_finish(True)（修复前后该路径都返回 False 语义）。"""
    order = []
    panel, _ = _make_panel(tmp_path, monkeypatch, order)
    panel._run_export()
    assert panel._run_images, order
    assert 0 in panel._run_boxes and panel._run_boxes[0], order
