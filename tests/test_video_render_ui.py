"""④A 逐帧推理成片的 UI 接线：弹窗第三页、worker 透传、mixin 取色与分发。

约定同 test_video_compose_ui：进程内验证接线（源码/假 editor），真实 Qt
验证控件行为与显隐。
"""
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(script):
    """子进程跑真实 Qt（本项目约定：进程内 PyQt5 是假模块）。"""
    script = script.replace("@@ROOT@@", str(ROOT))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


# ---------- worker ----------

def test_render_worker_passes_everything_to_engine(monkeypatch):
    """worker.run() 真调引擎，且把范围/绘制规格/筛选原样传下去。

    不用源码字符串断言：那种测试在「接线绿、参数传丢」时照样通过。
    这里把引擎换成记录参数的替身，直接跑 run()（同步跑，省掉事件循环）。
    """
    from pastelabel.engine import video_io
    from pastelabel.ui.inference_workers import VideoRenderWorker

    seen = {}

    def fake_render_video(video_path, detector, **kwargs):
        seen["video_path"] = video_path
        seen["detector"] = detector
        seen.update(kwargs)
        return "RESULT-SENTINEL"

    monkeypatch.setattr(video_io, "render_video", fake_render_video)

    spec = object()
    filt = ["person"]
    worker = VideoRenderWorker("in.mp4", "DET", output_path="out.mp4",
                               scope={"mode": "full"}, render=spec,
                               class_filter=filt)
    # 进程内的假 pyqtSignal 只有 connect、没有 emit（conftest 的最薄替身），
    # 而 run() 要真的 emit。就地换成带 emit 的替身，避免动 conftest 影响
    # 全仓 2000+ 用例。真实信号接线由真实 Qt 子进程用例覆盖。
    done, failed = [], []
    worker.progress = _NoopSignal()
    worker.finished_ok = _NoopSignal()
    worker.failed = _NoopSignal()
    worker.finished_ok.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()

    assert failed == [], failed
    assert done == ["RESULT-SENTINEL"], done
    assert seen["video_path"] == "in.mp4"
    assert seen["detector"] == "DET"
    assert seen["output_path"] == "out.mp4"
    assert seen["scope"] == {"mode": "full"}, "范围没传下去"
    assert seen["render"] is spec, "绘制规格没传下去（框会画不出来）"
    assert seen["class_filter"] == filt, "类别筛选没传下去"
    # 取消回调要接到自己的标志上：cancel() 之后引擎能看到 True
    assert callable(seen["cancel"]) and callable(seen["progress"])
    worker.cancel()
    assert seen["cancel"]() is True, "cancel() 没传到引擎的取消回调"


def test_render_worker_reports_engine_failure(monkeypatch):
    """引擎抛错 → failed 信号带原因，不当成成功。"""
    from pastelabel.engine import video_io
    from pastelabel.ui.inference_workers import VideoRenderWorker

    def boom(*a, **kw):
        raise video_io.VideoError("编码器打不开")

    monkeypatch.setattr(video_io, "render_video", boom)
    worker = VideoRenderWorker("in.mp4", "DET")
    done, failed = [], []
    worker.progress = _NoopSignal()
    worker.finished_ok = _NoopSignal()
    worker.failed = _NoopSignal()
    worker.finished_ok.connect(done.append)
    worker.failed.connect(failed.append)
    worker.run()

    assert done == []
    assert failed == ["编码器打不开"], failed


def test_render_worker_signature_has_render_and_scope():
    """构造签名收 render/scope/class_filter（默认 None = 全图、不绘制）。"""
    from pastelabel.ui.inference_workers import VideoRenderWorker

    params = inspect.signature(VideoRenderWorker.__init__).parameters
    for name in ("video_path", "detector", "output_path", "resize_fn",
                 "scope", "render", "class_filter"):
        assert name in params, f"缺少参数 {name}"
    assert params["scope"].default is None
    assert params["render"].default is None
    assert params["class_filter"].default is None


def test_render_worker_has_cancel():
    """必须能取消：逐帧推理很慢，用户要能中途停。"""
    from pastelabel.ui.inference_workers import VideoRenderWorker

    assert hasattr(VideoRenderWorker, "cancel")
    src = inspect.getsource(VideoRenderWorker.cancel)
    assert "True" in src


# ---------- 弹窗第三页 ----------

def test_dialog_has_render_tab_and_action():
    """弹窗要有「推理成片」页与对应 action 常量。"""
    from pastelabel.ui import video_dialog

    assert hasattr(video_dialog, "ACTION_RENDER")
    src = inspect.getsource(video_dialog.VideoDialog._build_ui)
    assert "_build_render_tab" in src, "弹窗没有加第三页"


def test_render_tab_has_its_own_overlay_group():
    """推理成片页有自己的叠加控件（类别来自模型，与合成页的 sidecar 不同源）。"""
    from pastelabel.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog._build_render_tab)
    assert '_build_overlay_group("render")' in src, "没给第三页建独立的叠加组"
    # 合成页不能跟着换成 render 组：两页各自的列表要互相独立
    compose_src = inspect.getsource(VideoDialog._build_compose_tab)
    assert '_build_overlay_group("compose")' in compose_src


def test_dialog_settings_branches_on_render_action():
    """settings() 要按 action 分流到推理成片的参数。"""
    from pastelabel.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog.settings)
    assert "ACTION_RENDER" in src
    assert "video_path" in src and "output_path" in src


def test_dialog_validate_requires_model_and_source():
    """推理成片要校验：选了视频、有模型、输出路径不为空。"""
    from pastelabel.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog.validate)
    assert "ACTION_RENDER" in src
    assert "_render_class_names" in src, "校验要挡住「没加载模型」"


# ---------- mixin ----------

from pastelabel.ui.mixins.video import VideoMixin as _VideoMixinBase


class _RenderHost(_VideoMixinBase):
    """只喂 _start_video_render 要用的东西，不建真实窗口。

    继承 mixin 本体：这样 `_compose_render_spec` 等真方法参与测试，
    只把「需要真窗口」的几处换成替身。
    """

    def __init__(self, detector=None, scope_error=None, filter_value=None):
        self._detector = detector
        self._busy = False
        self.hints = []
        self._scope_error = scope_error
        self._filter_value = filter_value

    def _video_dialog_hint(self, message):
        self.hints.append(message)

    def _scope_for_run(self):
        if self._scope_error is not None:
            raise self._scope_error
        return None

    def _filter_for_run(self):
        return self._filter_value

    def _all_class_names(self):
        return list(getattr(self._detector, "class_names", None) or [])

    def get_label_color(self, label):
        return {"person": "#FF0000"}.get(label, "#00FF80")


class _Detector:
    class_names = ["person", "car"]

    def __init__(self):
        self.model_path = "m.onnx"


def _settings(**overrides):
    base = dict(video_path="in.mp4", output_path="out.mp4",
                render={"labels": ["person"], "show_labels": True,
                        "show_scores": False})
    base.update(overrides)
    return base


class _NoopSignal:
    """信号替身：connect 收下回调，emit 时逐个调用。"""

    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _CapturedWorker:
    """替掉 VideoRenderWorker：记下构造参数，不起真线程。"""

    def __init__(self, calls, *args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        self.started = False
        self.cancelled = False
        # 每个实例自己的信号替身：类属性会让不同用例连到同一个 slots 列表，
        # 后面的用例会收到前面用例的回调（跨用例污染）。
        self.progress = _NoopSignal()
        self.finished_ok = _NoopSignal()
        self.failed = _NoopSignal()
        self.finished = _NoopSignal()

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def deleteLater(self):
        pass


@pytest.fixture
def fake_worker(monkeypatch):
    calls = []
    from pastelabel.ui import inference_workers

    monkeypatch.setattr(
        inference_workers, "VideoRenderWorker",
        lambda *a, **kw: _CapturedWorker(calls, *a, **kw))
    return calls


def test_start_render_worker_gets_main_thread_colors(fake_worker):
    """类别色必须在主线程取好交给 worker（后台不能读 editor 取色）。"""
    from pastelabel.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector())
    assert VideoMixin._start_video_render(host, _settings()) is True
    assert len(fake_worker) == 1
    render = fake_worker[0]["kwargs"]["render"]
    assert render is not None, "勾了类别却没带绘制规格"
    assert render.labels == {"person"}
    assert render.label_colors["person"] == "#FF0000"
    assert host._busy is True, "起了任务要置忙"


def test_start_render_worker_passes_scope_and_filter(fake_worker):
    """范围与类别筛选都要原样交给后台，不能丢成 None。"""
    from pastelabel.ui.mixins.video import VideoMixin

    sentinel = ["person"]
    host = _RenderHost(detector=_Detector(), filter_value=sentinel)
    VideoMixin._start_video_render(host, _settings())
    kwargs = fake_worker[0]["kwargs"]
    assert kwargs["class_filter"] == sentinel
    assert "scope" in kwargs and kwargs["scope"] is None, "全图 = None"
    assert kwargs["output_path"] == "out.mp4"


def test_start_render_worker_without_detector_hints(fake_worker):
    """没加载模型：明确提示，且**不能**起线程。"""
    from pastelabel.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=None)
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert fake_worker == [], "没模型不能起任务"
    assert host.hints, "要说明为什么起不来"
    assert host._busy is False


def test_start_render_worker_blocks_second_run(fake_worker):
    """已经在跑（或忙）时不能再起一个。"""
    from pastelabel.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector())
    assert VideoMixin._start_video_render(host, _settings()) is True
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert len(fake_worker) == 1, "不能起第二个任务"


def test_start_render_worker_rejects_bad_scope_before_running(fake_worker):
    """范围非法（没画 ROI 等）在启动时拦下，不进后台跑出 0 框结果。"""
    from pastelabel.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector(), scope_error=ValueError("没画区域"))
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert fake_worker == []
    assert "没画区域" in " ".join(host.hints)


def test_start_render_worker_without_render_options(fake_worker):
    """关掉叠加时 render=None，引擎走「只推理不绘制」。"""
    from pastelabel.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector())
    VideoMixin._start_video_render(host, _settings(render=None))
    assert fake_worker[0]["kwargs"]["render"] is None


def test_dispatch_recognizes_render_action():
    """start_requested 分发到 ④A，不是掉进抽帧兜底分支。"""
    from pastelabel.ui import video_dialog
    from pastelabel.ui.mixins.video import VideoMixin

    calls = []
    host = _RenderHost()
    host._video_dialog = None
    host._busy = False
    host._start_video_render = lambda s: calls.append(("render", s)) or True
    host._start_video_compose = lambda s: calls.append(("compose", s)) or True
    host._start_video_extract = lambda s: calls.append(("extract", s)) or True

    VideoMixin._on_video_start_requested(
        host, video_dialog.ACTION_RENDER, _settings())
    assert calls and calls[0][0] == "render", calls


def test_cancel_covers_render_worker():
    """弹窗取消要能取消 ④A 任务（三个 worker 都要覆盖）。"""
    from pastelabel.ui.mixins.video import VideoMixin

    class _Worker:
        def __init__(self):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    worker = _Worker()
    host = _RenderHost()
    host._render_worker = worker
    host._video_worker = None
    host._compose_worker = None
    VideoMixin._cancel_video_task(host)
    assert worker.cancelled is True, "④A 任务没被取消"


# ---------- 真实 Qt：第三页的行为 ----------

_PRELUDE = '''
import os, sys, tempfile
sys.path.insert(0, r"@@ROOT@@")
from PyQt5.QtWidgets import QApplication
app = QApplication([])
import numpy as np
import cv2
from pastelabel.ui.video_dialog import VideoDialog
from pastelabel.engine import video_io as vio

TMP = tempfile.mkdtemp()
SRC = os.path.join(TMP, "源视频.mp4")
_vw = cv2.VideoWriter(SRC, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
for _ in range(20):
    _vw.write(np.zeros((48, 64, 3), np.uint8))
_vw.release()


def render_settings(dialog):
    """settings() 默认按抽帧动作返回；先按推理成片按钮切动作。"""
    dialog._on_start_render()
    return dialog.settings()
'''


def _run_prelude(script):
    return _run_qt(_PRELUDE.replace("@@ROOT@@", str(ROOT)) + script)


def test_real_qt_render_tab_lists_model_classes():
    """模型类别要列进第三页；没模型时按钮禁用并提示先加载。"""
    result = _run_prelude('''
d = VideoDialog()
before = (d.render_btn.isEnabled(), d.render_label_list.count(),
          d.render_model_label.text())
d.set_detector_info(["person", "car", "bus"], "best.onnx")
after = (d.render_label_list.count(), d.render_overlay_check.isChecked(),
         d._render_overlay.options()["labels"])
print("RESULT", repr((before, after)))
''')
    before, after = result
    assert before[0] is False, "没模型不能开始"
    assert before[1] == 0
    assert before[2], "要说明为什么不能开始"
    assert after[0] == 3, after
    assert after[1] is True, "有类别时主开关默认打开"
    assert after[2] == ["person", "car", "bus"], after


def test_real_qt_render_tab_needs_both_video_and_model():
    """源视频与模型缺一不可：只有模型、只有视频都不能开始。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_detector_info(["person"], "m.onnx")
only_model = d.render_btn.isEnabled()
d.set_video(SRC)
both = d.render_btn.isEnabled()
same_dir = os.path.dirname(d.render_out_edit.text()) == os.path.dirname(SRC)
print("RESULT", repr((only_model, both, d.render_out_edit.text(), same_dir)))
''')
    only_model, both, out_path, same_dir = result
    assert only_model is False, "只有模型没有视频不能开始"
    assert both is True, "两者齐了才能开始"
    assert out_path.endswith("源视频_render.mp4"), out_path
    assert same_dir, "输出必须与源视频同级"


def test_real_qt_render_output_avoids_overwriting_existing():
    """同级已有同名输出时加序号，不覆盖已有文件。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_detector_info(["person"], "m.onnx")
d.set_video(SRC)
first = d.render_out_edit.text()
open(first, "wb").close()          # 假装已经有一条产物
d.set_video(SRC)
second = d.render_out_edit.text()
print("RESULT", repr((os.path.basename(first), os.path.basename(second))))
''')
    first, second = result
    assert first == "源视频_render.mp4", first
    assert second == "源视频_render_2.mp4", second


def test_real_qt_render_settings_carry_render_options():
    """推理成片页的 settings：源视频 + 输出 + render 选项。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_detector_info(["person", "car"], "m.onnx")
d.set_video(SRC)
settings = render_settings(d)
d._render_overlay.show_label_check.setChecked(False)
changed = render_settings(d)
print("RESULT", repr((settings.get("video_path") is not None,
                      "output_path" in settings,
                      settings["render"]["labels"],
                      settings["render"]["show_labels"],
                      changed["render"]["show_labels"])))
''')
    has_video, has_out, labels, show_labels, changed = result
    assert has_video and has_out
    assert labels == ["person", "car"], labels
    assert show_labels is True and changed is False


def test_real_qt_render_validate_blocks_missing_model():
    """没模型时 validate 要给出原因（提示区显示），不是静默不动作。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_video(SRC)
d._on_start_render()
print("RESULT", repr((d.action, d.render_hint_label.text())))
''')
    action, hint = result
    assert action == "render"
    assert hint, "必须说明为什么不能开始"


def test_real_qt_render_tab_shares_the_selected_video():
    """两页共用同一个已选视频：抽帧页选完，第三页立刻可用。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_detector_info(["person"], "m.onnx")
d.set_video(SRC)
print("RESULT", repr((d.path_edit.text() == SRC,
                      d.render_path_edit.text() == SRC,
                      d.info_label.text() == d.render_info_label.text())))
''')
    same_extract, same_render, same_info = result
    assert same_extract and same_render, "两页应共用同一个源视频"
    assert same_info, "探测信息两页要一致"


def test_real_qt_running_locks_render_tab():
    """任务在跑时锁住第三页，避免改参数或重复起任务。"""
    result = _run_prelude('''
d = VideoDialog()
d.set_detector_info(["person"], "m.onnx")
d.set_video(SRC)
d.action = "render"
d.set_running(True)
locked = (d.render_btn.isEnabled(), d.render_pick_btn.isEnabled(),
          d.render_label_list.isEnabled())
d.set_running(False)
freed = (d.render_btn.isEnabled(), d.render_pick_btn.isEnabled())
print("RESULT", repr((locked, freed)))
''')
    locked, freed = result
    assert locked == (False, False, False), locked
    assert freed == (True, True), freed


def test_real_qt_cancel_button_cancels_running_render():
    """任务在跑时按「取消」是取消任务，不是关窗。"""
    result = _run_prelude('''
d = VideoDialog()
emitted = []
d.cancel_requested.connect(lambda: emitted.append(True))
d.set_running(True)
d._on_cancel_clicked()
d.set_running(False)
d._on_cancel_clicked()
print("RESULT", repr((emitted, d.isVisible())))
''')
    emitted, visible = result
    assert emitted == [True], "跑任务时取消要发取消信号"
    assert visible is False, "空闲时按取消才是关窗"
