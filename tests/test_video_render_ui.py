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
    """子进程跑真实 Qt（本项目约定：进程内 PySide6 是假模块）。"""
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
    from labelforge.engine import video_io
    from labelforge.ui.inference_workers import VideoRenderWorker

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
    # 进程内的假 Signal 只有 connect、没有 emit（conftest 的最薄替身），
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
    from labelforge.engine import video_io
    from labelforge.ui.inference_workers import VideoRenderWorker

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
    from labelforge.ui.inference_workers import VideoRenderWorker

    params = inspect.signature(VideoRenderWorker.__init__).parameters
    for name in ("video_path", "detector", "output_path", "resize_fn",
                 "scope", "render", "class_filter"):
        assert name in params, f"缺少参数 {name}"
    assert params["scope"].default is None
    assert params["render"].default is None
    assert params["class_filter"].default is None


def test_render_worker_has_cancel():
    """必须能取消：逐帧推理很慢，用户要能中途停。"""
    from labelforge.ui.inference_workers import VideoRenderWorker

    assert hasattr(VideoRenderWorker, "cancel")
    src = inspect.getsource(VideoRenderWorker.cancel)
    assert "True" in src


# ---------- 弹窗第三页 ----------

def test_dialog_has_render_tab_and_action():
    """弹窗要有「推理成片」页与对应 action 常量。"""
    from labelforge.ui import video_dialog

    assert hasattr(video_dialog, "ACTION_RENDER")
    src = inspect.getsource(video_dialog.VideoDialog._build_ui)
    assert "_build_render_tab" in src, "弹窗没有加第三页"


def test_render_tab_has_its_own_overlay_group():
    """推理成片页有自己的叠加控件（类别来自模型，与合成页的 sidecar 不同源）。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog._build_render_tab)
    assert '_build_overlay_group("render")' in src, "没给第三页建独立的叠加组"
    # 合成页不能跟着换成 render 组：两页各自的列表要互相独立
    compose_src = inspect.getsource(VideoDialog._build_compose_tab)
    assert '_build_overlay_group("compose")' in compose_src


def test_dialog_settings_branches_on_render_action():
    """settings() 要按 action 分流到推理成片的参数。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog.settings)
    assert "ACTION_RENDER" in src
    assert "video_path" in src and "output_path" in src


def test_dialog_validate_requires_model_and_source():
    """推理成片要校验：选了视频、有模型、输出路径不为空。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog.validate)
    assert "ACTION_RENDER" in src
    assert "_render_class_names" in src, "校验要挡住「没加载模型」"


# ---------- mixin ----------

from labelforge.ui.mixins.video import VideoMixin as _VideoMixinBase


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
    from labelforge.ui import inference_workers

    monkeypatch.setattr(
        inference_workers, "VideoRenderWorker",
        lambda *a, **kw: _CapturedWorker(calls, *a, **kw))
    return calls


def test_start_render_worker_gets_main_thread_colors(fake_worker):
    """类别色必须在主线程取好交给 worker（后台不能读 editor 取色）。"""
    from labelforge.ui.mixins.video import VideoMixin

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
    from labelforge.ui.mixins.video import VideoMixin

    sentinel = ["person"]
    host = _RenderHost(detector=_Detector(), filter_value=sentinel)
    VideoMixin._start_video_render(host, _settings())
    kwargs = fake_worker[0]["kwargs"]
    assert kwargs["class_filter"] == sentinel
    assert "scope" in kwargs and kwargs["scope"] is None, "全图 = None"
    assert kwargs["output_path"] == "out.mp4"


def test_start_render_worker_without_detector_hints(fake_worker):
    """没加载模型：明确提示，且**不能**起线程。"""
    from labelforge.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=None)
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert fake_worker == [], "没模型不能起任务"
    assert host.hints, "要说明为什么起不来"
    assert host._busy is False


def test_start_render_worker_blocks_second_run(fake_worker):
    """已经在跑（或忙）时不能再起一个。"""
    from labelforge.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector())
    assert VideoMixin._start_video_render(host, _settings()) is True
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert len(fake_worker) == 1, "不能起第二个任务"


def test_start_render_worker_rejects_bad_scope_before_running(fake_worker):
    """范围非法（没画 ROI 等）在启动时拦下，不进后台跑出 0 框结果。"""
    from labelforge.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector(), scope_error=ValueError("没画区域"))
    assert VideoMixin._start_video_render(host, _settings()) is False
    assert fake_worker == []
    assert "没画区域" in " ".join(host.hints)


def test_start_render_worker_without_render_options(fake_worker):
    """关掉叠加时 render=None，引擎走「只推理不绘制」。"""
    from labelforge.ui.mixins.video import VideoMixin

    host = _RenderHost(detector=_Detector())
    VideoMixin._start_video_render(host, _settings(render=None))
    assert fake_worker[0]["kwargs"]["render"] is None


def test_dispatch_recognizes_render_action():
    """start_requested 分发到 ④A，不是掉进抽帧兜底分支。"""
    from labelforge.ui import video_dialog
    from labelforge.ui.mixins.video import VideoMixin

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
    from labelforge.ui.mixins.video import VideoMixin

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
from PySide6.QtWidgets import QApplication
app = QApplication([])
import numpy as np
import cv2
from labelforge.ui.video_dialog import VideoDialog
from labelforge.engine import video_io as vio

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


def test_real_qt_render_tab_prompts_when_model_missing():
    """没加载模型时，推理成片页要有显眼且可照做的提示。

    用户实测反馈：打开第三页只看到开始按钮是灰的，不知道该怎么办。
    原来只在模型名那一行写「请先加载模型」，颜色与正文一样、也没有说
    去哪里加载，很容易被当成「这页坏了」。

    要求：提示用强调色（`videoHint`），并说清去哪加载；加载模型后
    提示消失，按钮恢复可用。
    """
    result = _run_prelude('''
d = VideoDialog()
d.show()
d.tabs.setCurrentIndex(2)
d.set_video(SRC)
before = (d.render_model_hint_label.isVisible(),
          d.render_model_hint_label.text(),
          d.render_model_hint_label.objectName(),
          d.render_btn.isEnabled())
d.set_detector_info(["person"], "m.onnx")
after = (d.render_model_hint_label.isVisible(), d.render_btn.isEnabled())
print("RESULT", repr((before, after)))
''')
    before, after = result
    visible, text, name, enabled = before
    assert visible is True, "没模型时提示必须显示出来"
    assert "加载模型" in text, f"提示要说清去哪加载：{text!r}"
    assert name == "videoHint", f"提示要用强调色（videoHint）：{name}"
    assert enabled is False, "没模型不能开始"
    assert after == (False, True), f"加载模型后提示要消失、按钮可用：{after}"


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


# ---------- 取消后的报告（计划 §4.5：保留可播放半成品） ----------

class _StatusStub:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _RenderFinishedStub:
    """只喂 _on_render_finished 要用的东西。"""

    def __init__(self):
        self.status_label = _StatusStub()
        self._busy = True
        self.infos = []
        self.warns = []
        self.dialog_texts = []

    def _close_video_dialog(self, text=None):
        self.dialog_texts.append(text)

    def _video_info(self, message):
        self.infos.append(message)

    def _video_warn(self, message):
        self.warns.append(message)


def _render_result(**overrides):
    from types import SimpleNamespace

    base = dict(cancelled=False, frame_count=40, source_frame_count=40,
                duration=4.0, output_fps=10.0, output_path="out.mp4",
                frames_written=40, detected_total=12, partial_path="")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_render_finished_reports_output_and_burn_in_warning():
    """正常完成：报输出路径 + 帧数/时长，并说清框已烧进画面。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _RenderFinishedStub()
    VideoMixin._on_render_finished(stub, _render_result())
    assert stub._busy is False
    assert stub.warns == []
    assert "40" in stub.status_label.text, stub.status_label.text
    assert stub.infos and "out.mp4" in stub.infos[0], stub.infos
    assert "烧进画面" in stub.infos[0], "要提醒框不能再改"
    assert "音轨" not in stub.infos[0], "不提音轨，用户没问过这个"


def test_render_cancelled_reports_kept_partial():
    """取消：报出半成品在哪、有多少帧、正式输出没生成。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _RenderFinishedStub()
    VideoMixin._on_render_finished(stub, _render_result(
        cancelled=True, frames_written=17, frame_count=17,
        partial_path=r"D:\v\.out.mp4.partial.mp4"))
    assert stub._busy is False, "取消也要清掉忙标志"
    assert stub.dialog_texts == ["已取消推理成片"], stub.dialog_texts
    assert stub.infos, "取消时也要告诉用户半成品在哪"
    assert "partial" in stub.infos[0], stub.infos
    assert "17" in stub.infos[0], stub.infos
    assert "17" in stub.status_label.text, stub.status_label.text


def test_render_cancelled_without_partial_is_quiet():
    """零帧取消：没有半成品可报，就不弹「保留了什么」。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _RenderFinishedStub()
    VideoMixin._on_render_finished(stub, _render_result(
        cancelled=True, frames_written=0, frame_count=0, partial_path=""))
    assert stub._busy is False
    assert stub.infos == [], stub.infos
    assert stub.status_label.text == "已取消推理成片", stub.status_label.text


def test_render_failed_reports_reason():
    """失败：报原因，不当成成功。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _RenderFinishedStub()
    VideoMixin._on_render_failed(stub, "编码器打不开")
    assert stub._busy is False
    assert stub.dialog_texts == ["推理成片失败"], stub.dialog_texts
    assert stub.warns == ["编码器打不开"], stub.warns


def test_full_chain_render_with_real_qt():
    """真实 Qt 全链路：弹窗勾选 → mixin 组 spec（含颜色） → 引擎逐帧推理
    → 回读成片确认每帧都有框、帧数与源一致。

    这条把三个环节串起来跑真 Qt + 真引擎 + 真编码：只在进程内用假替身验接线
    会漏掉「颜色取到了但没传到画面」「画了框但帧数不对」这类跨层问题。
    """
    result = _run_prelude('''
import cv2
from labelforge.ui.mixins.video import VideoMixin
from labelforge.engine import video_io as vio
from labelforge.engine.inference_image import resize_for_model

class Stub(VideoMixin):
    def get_label_color(self, label):
        return {"person": "#FF0000"}.get(label, "#00FF80")

class Det:
    """每帧都报一个 person 框（画面中央，够大以便抗住 mp4v 有损）。"""
    class_names = ["person"]
    model_path = "m.onnx"
    def detect(self, rgb, resize_fn):
        h, w = rgb.shape[:2]
        return [{"label": "person", "x": w * 0.25, "y": h * 0.25,
                 "width": w * 0.5, "height": h * 0.5, "score": 0.9,
                 "class_id": 0}]

d = VideoDialog()
d.set_detector_info(["person"], "m.onnx")
d.set_video(SRC)
settings = render_settings(d)
render = Stub()._compose_render_spec(settings["render"])
out = os.path.join(TMP, "chain_render.mp4")
res = vio.render_video(SRC, Det(), output_path=out,
                       resize_fn=resize_for_model, render=render)

cap = cv2.VideoCapture(out)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()

# 每帧都要有红色像素（框画上去了），且不能整帧变红（画坏了）
hits = 0
for f in frames:
    red = np.abs(f[:, :, ::-1].astype(int) - np.array([255, 0, 0])).max(axis=2)
    n = int((red <= 40).sum())
    if 0 < n < f.shape[0] * f.shape[1]:
        hits += 1

src_frames = 20
print("RESULT", repr((res.frame_count, res.detected_total, len(frames),
                      hits, src_frames, sorted(render.labels),
                      render.label_colors.get("person"),
                      res.output_path == out)))
''')
    (frame_count, detected, decoded, hits, src_frames, labels, color,
     is_final) = result
    assert frame_count == src_frames, f"帧数应与源一致：{frame_count}"
    assert decoded == src_frames, f"回读帧数不对：{decoded}"
    assert detected == src_frames, f"每帧都该检出一个框：{detected}"
    assert labels == ["person"], labels
    assert color == "#FF0000", color
    assert hits == src_frames, f"每帧都要画上框，实际只有 {hits}/{src_frames}"
    assert is_final is True, "正常完成要发布到正式名"


def test_full_chain_render_cancel_keeps_playable_partial_real_qt():
    """真实 Qt 全链路取消：半成品能播、帧数对得上、正式名不占用。"""
    result = _run_prelude('''
import cv2
from labelforge.engine import video_io as vio
from labelforge.engine.inference_image import resize_for_model
from labelforge.engine.video_render import RenderSpec

class Det:
    class_names = ["person"]
    def detect(self, rgb, resize_fn):
        h, w = rgb.shape[:2]
        return [{"label": "person", "x": 5.0, "y": 5.0,
                 "width": w * 0.3, "height": h * 0.3, "score": 0.9,
                 "class_id": 0}]

seen = {"n": 0}
out = os.path.join(TMP, "cancel_render.mp4")
res = vio.render_video(SRC, Det(), output_path=out,
                       resize_fn=resize_for_model,
                       render=RenderSpec(labels={"person"},
                                         show_labels=True,
                                         label_colors={"person": "#FF0000"}),
                       cancel=lambda: seen["n"] >= 5,
                       on_frame=lambda i: seen.__setitem__("n", i))

decoded = 0
if res.partial_path:
    cap = cv2.VideoCapture(res.partial_path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()

print("RESULT", repr((res.cancelled, res.frames_written, decoded,
                      bool(res.partial_path), os.path.exists(out),
                      "partial" in os.path.basename(res.partial_path or ""))))
''')
    cancelled, written, decoded, has_partial, final_exists, named_partial = result
    assert cancelled is True
    assert written == 5, written
    assert has_partial is True, "取消要保留半成品"
    assert named_partial is True, "半成品必须用 .partial 名，不能冒充正式产物"
    assert final_exists is False, "取消不能生成正式文件"
    assert decoded == written, f"半成品要能播且帧数对得上：{decoded} != {written}"


# ---------- 第 12 轮：合成范围的两个输入框 ----------

def test_compose_range_spins_default_to_first_and_last():
    """范围输入默认「0 到最后一张」，来源选定后自动填好。"""
    result = _run_qt('''
import sys, os, tempfile
sys.path.insert(0, r"@@ROOT@@")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
app = QApplication([])

# 用 tempfile.gettempdir()：Windows 有 TEMP，Linux 没有（只有 TMPDIR/RUNNER_TEMP）
work = os.path.join(tempfile.gettempdir(), "labelforge_range_ui")
os.makedirs(work, exist_ok=True)
for name in os.listdir(work):
    os.remove(os.path.join(work, name))
for i in range(7):
    img = QImage(320, 240, QImage.Format_RGB888)
    img.fill(0x223344)
    img.save(os.path.join(work, f"{i:03d}.png"))

from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
d.set_compose_dir(work)
app.processEvents()
print("RESULT", repr((d.compose_start_spin.value(), d.compose_end_spin.value(),
                     d.compose_start_spin.maximum(), d.compose_end_spin.maximum(),
                     d.compose_range())))
''')
    start, end, start_max, end_max, rng = result
    assert (start, end) == (0, 6), f"默认应为 0..6（7 张），实际 {start}..{end}"
    assert (start_max, end_max) == (6, 6), (start_max, end_max)
    assert rng == (0, 6)


def test_compose_range_is_carried_into_the_settings():
    """范围要进 `settings()`，否则界面上选了也不生效。"""
    result = _run_qt('''
import sys, os, tempfile
sys.path.insert(0, r"@@ROOT@@")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
app = QApplication([])

work = os.path.join(tempfile.gettempdir(), "labelforge_range_settings")
os.makedirs(work, exist_ok=True)
for name in os.listdir(work):
    os.remove(os.path.join(work, name))
for i in range(6):
    img = QImage(320, 240, QImage.Format_RGB888)
    img.fill(0x223344)
    img.save(os.path.join(work, f"{i:03d}.png"))

from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
d.set_compose_dir(work)
d.compose_start_spin.setValue(2)
d.compose_end_spin.setValue(4)
# settings() 按当前 action 分派，不设成 compose 会返回 None
from labelforge.ui.video_dialog import ACTION_COMPOSE
d.action = ACTION_COMPOSE
app.processEvents()
s = d.settings()
print("RESULT", repr((s.get("start_index"), s.get("end_index"))))
''')
    start_index, end_index = result
    assert (start_index, end_index) == (2, 4), (start_index, end_index)


def test_compose_range_corrects_a_reversed_input():
    """起始 > 结束 时纠回，不让用户带着非法范围点开始。"""
    result = _run_qt('''
import sys, os, tempfile
sys.path.insert(0, r"@@ROOT@@")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
app = QApplication([])

work = os.path.join(tempfile.gettempdir(), "labelforge_range_reverse")
os.makedirs(work, exist_ok=True)
for name in os.listdir(work):
    os.remove(os.path.join(work, name))
for i in range(6):
    img = QImage(320, 240, QImage.Format_RGB888)
    img.fill(0x223344)
    img.save(os.path.join(work, f"{i:03d}.png"))

from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
d.set_compose_dir(work)
app.processEvents()
# 先把起始拉到 4，再把结束拉回 1（越界）
d.compose_start_spin.setValue(4)
app.processEvents()
d.compose_end_spin.setValue(1)
app.processEvents()
print("RESULT", repr((d.compose_start_spin.value(), d.compose_end_spin.value(),
                     d.compose_range())))
''')
    start, end, rng = result
    assert start <= end, f"起始({start}) 不该大于结束({end})"
    assert rng == (start, end)
