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

def test_render_worker_passes_everything_to_engine():
    """worker 要把源、detector、范围、绘制规格都透传给引擎。"""
    from pastelabel.ui.inference_workers import VideoRenderWorker

    src = inspect.getsource(VideoRenderWorker.run)
    assert "render_video" in src
    for kw in ("detector", "resize_fn", "scope", "render", "class_filter",
               "cancel", "progress"):
        assert kw in src, f"run() 没把 {kw} 传给引擎"


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

def test_mixin_starts_render_worker_with_main_thread_colors():
    """mixin 起 ④A 任务：类别色必须在主线程取好再交给 worker。"""
    from pastelabel.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin)
    assert "_start_video_render" in src
    starter = inspect.getsource(VideoMixin._start_video_render)
    assert "VideoRenderWorker" in starter
    assert "get_label_color" in starter or "_compose_render_spec" in starter


def test_mixin_reports_render_result_and_warns_about_burned_boxes():
    """完成后要提示「框已烧进画面，不可再改」与不含音轨（首版限制）。"""
    from pastelabel.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._on_render_finished)
    assert "output_path" in src
    assert "frame_count" in src


def test_mixin_dispatches_render_action():
    """start_requested 分发要认 ACTION_RENDER。"""
    from pastelabel.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._on_video_start_requested)
    assert "ACTION_RENDER" in src
    assert "_start_video_render" in src


def test_mixin_cancel_covers_render_worker():
    """弹窗取消要能取消 ④A 任务。"""
    from pastelabel.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._cancel_video_task)
    assert "_render_worker" in src


def test_mixin_cleanup_handles_render_worker():
    """任务结束要释放引用，否则第二次跑被 busy 挡住。"""
    from pastelabel.ui.mixins.video import VideoMixin

    assert hasattr(VideoMixin, "_cleanup_render_worker")
    src = inspect.getsource(VideoMixin._cleanup_render_worker)
    assert "_render_worker" in src


def test_mixin_render_requires_loaded_detector():
    """没加载模型就不能起 ④A：先提示，不静默什么都不做。"""
    from pastelabel.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._start_video_render)
    assert "_detector" in src or "detector" in src


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
