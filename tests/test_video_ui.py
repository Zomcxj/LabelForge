"""视频抽帧 UI 接线测试。

约定（同 test_inference_ui）：
- 不实例化 Qt 窗口；源码字符串 / AST / i18n 字典断言 + 假 editor 直接调 mixin。
- 需要真实控件时用子进程 + 真实 Qt。
"""
import inspect
import os
import re
from pathlib import Path

import pytest

from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.mixins.inference import InferenceMixin
from labelforge.ui.mixins.translation import TranslationMixin
from labelforge.ui.mixins.video import VideoMixin

ROOT = Path(__file__).resolve().parents[1]

VIDEO_KEYS = (
    "视频抽帧", "视频抽帧:", "选择视频文件", "选择视频", "每秒抽几帧",
    "起始时间(秒)", "结束时间(秒)", "图片格式", "输出目录", "开始抽帧",
    "已按源帧率上限", "抽帧中", "抽帧完成", "抽帧失败", "正在取消...",
    "未安装 opencv（cv2），视频功能不可用", "请先选择视频文件",
    "已取消，保留已抽出的图片", "张图片", "清单未完成，已抽出的图片保留",
    "抽帧目录", "视频文件不存在", "帧率必须为正数",
    "时间范围非法", "全部", "jpg 质量",
)


# ---------- 入口 ----------

def test_video_button_sits_between_labels_and_model_icons():
    """视频按钮在工具栏「标签:」两个导入图标与「模型推理:」之间。"""
    from labelforge.ui.mixins.toolbar import ToolbarMixin

    src = inspect.getsource(ToolbarMixin._create_toolbar)
    label_idx = src.index("self.label_lbl = QLabel")
    paste_label_idx = src.index("self.upload_paste_label_btn = self._create_svg_button")
    video_idx = src.index("self.video_load_btn = self._create_svg_button")
    model_lbl_idx = src.index("self.model_lbl = QLabel")

    assert label_idx < paste_label_idx < video_idx < model_lbl_idx


def test_video_label_precedes_video_button():
    """视频按钮前有「视频:」文字标签，位置与「标签:」「模型推理:」一致。"""
    from labelforge.ui.mixins.toolbar import ToolbarMixin

    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "video_lbl" in src
    video_lbl_idx = src.index("self.video_lbl = QLabel")
    video_btn_idx = src.index("self.video_load_btn = self._create_svg_button")
    model_lbl_idx = src.index("self.model_lbl = QLabel")

    assert video_lbl_idx < video_btn_idx < model_lbl_idx


def test_video_label_is_translated():
    """「视频:」词条在两个语言表里都有。"""
    assert "视频:" in i18n._strings["zh"]
    assert "视频:" in i18n._strings["en"]


def test_video_label_refreshed_on_language_switch():
    from labelforge.ui.mixins.translation import TranslationMixin

    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "video_lbl" in src


# ---------- 模型行：加载按钮在模型名之前（与视频行顺序一致） ----------

def test_model_row_load_button_precedes_name_label():
    """模型行的顺序与视频行一致：先按钮、后名字。

    视频行是「选择视频」按钮在前、视频名在后；模型行原来是名字在前，
    两行并排时顺序看起来不一致。
    """
    from labelforge.ui.model_row import ModelRow

    src = inspect.getsource(ModelRow._build_ui)
    load_idx = src.index("self.load_btn = QPushButton")
    name_idx = src.index("self.name_label = QLabel")

    assert load_idx < name_idx, "加载模型按钮应在模型名之前"


def test_model_row_clear_model_still_works_after_reorder():
    """交换顺序后 clear_model 仍要填默认值、禁用控件（不能因顺序改动而漏掉）。"""
    from labelforge.ui.model_row import ModelRow

    src = inspect.getsource(ModelRow.clear_model)
    assert "尚未加载模型" in src
    assert "setEnabled(False)" in src


def test_video_button_greyed_when_cv2_missing():
    """cv2 缺失时按钮置灰并说明原因，不能点了才报错。"""
    from labelforge.ui.mixins.toolbar import ToolbarMixin

    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "cv2" in src
    assert "setEnabled(False)" in src


def test_video_button_is_not_in_options_menu():
    """入口在工具栏，不再重复放到「选项」行。"""
    from labelforge.ui.mixins.options_popup import OptionsPopupMixin

    src = inspect.getsource(OptionsPopupMixin._create_options_menu)
    assert "video_btn" not in src


# ---------- 弹窗（按钮在工具栏，点击打开配置窗口） ----------

def test_video_dialog_has_all_extraction_params():
    """弹窗里要有：选视频、帧率、起止时间、格式、质量、输出目录、开始按钮。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    for name in ("path_edit", "pick_btn", "fps_spin", "start_spin", "end_spin",
                 "format_combo", "quality_spin", "out_edit", "hint_label"):
        assert f"self.{name}" in src, name
    for key in ("每秒抽几帧", "起始时间(秒)", "结束时间(秒)", "图片格式",
                "jpg 质量", "输出目录", "开始抽帧", "选择视频"):
        assert key in src, key


def test_video_dialog_output_dir_is_readonly():
    """输出目录只读展示，由视频路径推导，用户不能另选位置。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "setReadOnly(True)" in src
    assert "output_dir_for" in src


def test_video_dialog_validates_before_signalling():
    """开始前用 probe 校验；非法输入不发 start_requested，只在提示区说明。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "probe(" in src
    assert "VideoError" in src
    assert "start_requested" in src
    # 校验失败必须在 emit 之前返回
    for handler in (VideoDialog._on_start_extract, VideoDialog._on_start_compose):
        handler_src = inspect.getsource(handler)
        assert handler_src.index("return") < handler_src.index("start_requested.emit")


def test_video_dialog_uses_theme_and_centers():
    """与其它弹窗一致：走 dialog_helpers 的主题标题栏与居中（别自己重写一遍）。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "sync_titlebar" in src
    assert "center_on_parent" in src
    # 这两个辅助函数本身负责 set_titlebar_dark 与居中计算
    from labelforge.ui import dialog_helpers

    helper_src = inspect.getsource(dialog_helpers)
    assert "set_titlebar_dark" in helper_src


def test_video_dialog_settings_reports_full_range_as_none():
    """全片按 None 传：全片抽帧要读至 EOF 核对帧数，显式范围会丢掉这层检查。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog.settings)
    assert "None" in src
    assert "duration" in src


def test_video_button_opens_dialog():
    """工具栏按钮打开弹窗（按钮位置不动）。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._open_video_extract)
    assert "VideoDialog" in src
    assert "exec_()" in src


def test_video_mixin_has_no_row_methods():
    """已改回弹窗，不再有参数行的开关方法。"""
    from labelforge.ui.mixins.video import VideoMixin

    for name in ("_toggle_video_row", "_ensure_video_row"):
        assert not hasattr(VideoMixin, name), name


def test_video_row_module_is_gone():
    """参数行模块已删除。"""
    assert not (ROOT / "labelforge" / "ui" / "video_row.py").exists()


def test_video_mixin_registered_on_image_editor():
    from labelforge.ui.mixins.video import VideoMixin

    assert VideoMixin in ImageEditor.__mro__
    assert callable(getattr(ImageEditor, "_open_video_extract"))


def test_video_mixin_contract():
    from labelforge.ui.mixins.video import VideoMixin

    for name in ("_open_video_extract", "_start_video_extract",
                 "_on_video_extract_finished", "_on_video_extract_failed",
                 "_cleanup_video_worker"):
        assert hasattr(VideoMixin, name), name


# ---------- 线程与互斥 ----------

def test_extraction_runs_in_background_thread():
    """抽帧必须走线程，否则大视频会冻住界面。"""
    src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    assert "class VideoExtractWorker" in src
    assert "QThread" in src

    mixin = (ROOT / "labelforge" / "ui" / "mixins" / "video.py").read_text(encoding="utf-8")
    assert "worker.start()" in mixin
    assert "extract_frames(" not in mixin, "抽帧只能在 worker 里执行"


def test_video_worker_supports_cancel():
    src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    block = src[src.index("class VideoExtractWorker"):]
    assert "def cancel(self)" in block
    assert "cancel=" in block


def test_video_extract_respects_busy_flag():
    """抽帧不能与其它任务并行：起任务时检查 _busy，结束时释放。"""
    from labelforge.ui.mixins.video import VideoMixin

    start_src = inspect.getsource(VideoMixin._start_video_extract)
    assert "_busy" in start_src
    assert "_video_worker" in start_src
    # 收尾必须释放忙碌标记，否则后续任务永远起不来
    assert "_busy = False" in inspect.getsource(VideoMixin._cleanup_video_worker)


def test_close_event_releases_video_worker():
    """关窗时必须释放抽帧线程，避免活线程访问已销毁控件。"""
    src = inspect.getsource(ImageEditor.closeEvent)

    assert "video" in src.lower()
    assert "cancel()" in src or "wait(" in src


# ---------- i18n ----------

def test_i18n_video_terms():
    for key in VIDEO_KEYS:
        assert key in i18n._strings["zh"], key
        assert key in i18n._strings["en"], key


@pytest.mark.parametrize("rel", [
    "labelforge/ui/video_dialog.py",
    "labelforge/ui/mixins/video.py",
])
def test_every_tr_key_in_video_code_is_translated(rel):
    """视频相关代码里每个 tr() 键都必须在两个语言表里有定义。

    用 AST 取键而不是正则：`tr("上半句" "下半句")` 这种跨行隐式拼接
    在源码里是**一个**字符串，正则却只会逮到第一个片段，于是整串
    从来没进过词条表也照样绿（推理成片页那句说明就这样漏掉了）。
    """
    import ast

    missing = []
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tr" and node.args):
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            continue
        for lang in ("zh", "en"):
            if arg.value not in i18n._strings[lang]:
                missing.append((rel, lang, arg.value, node.lineno))
    assert not missing, missing


def test_video_texts_refreshed_on_language_switch():
    """工具栏标签与按钮要随语言刷新；弹窗每次新建、自带当前语言，无需常驻刷新。"""
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "video_load_btn" in src
    assert "video_lbl" in src


# ---------- 弹窗与模型行互不影响 ----------

def test_dialog_and_model_row_are_independent():
    """视频走弹窗、模型走参数行：点视频不能顺手改模型行状态。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._open_video_extract)
    assert "model_row" not in src
    assert "_update_inference_buttons" not in src


def test_theme_refresh_does_not_reference_deleted_row():
    """主题刷新里不能留已删除的 video_row 引用。"""
    from labelforge.ui.mixins.theme import ThemeMixin

    src = inspect.getsource(ThemeMixin._apply_theme)
    assert "video_row" not in src


# ---------- 真实 Qt：弹窗行为 ----------

def test_dialog_end_to_end_with_real_qt(tmp_path):
    """真实 Qt 下走一遍：探测中文路径视频、默认值、只读输出目录、校验放行。

    中文路径必须真的跑一遍：cv2 在 Windows 上对中文路径的图片读写有坑，
    探测走的是 VideoCapture（正常），但要确保弹窗这一层没有自己拼路径。
    """
    import subprocess
    import sys

    script = '''
import os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication, QDialog
app = QApplication(sys.argv)
import cv2, numpy as np
from labelforge.ui.video_dialog import VideoDialog

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "我的 视频.mp4")
w, h, n = 96, 64, 50
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
for i in range(n):
    frame = np.zeros((h, w, 3), np.uint8)
    x = 4 + (i * 3) %% (w - 40)
    frame[8:24, x:x + 32] = (0, 0, 255)
    vw.write(frame)
vw.release()

d = VideoDialog()
before = d.start_btn.isEnabled()          # 未选视频：不能开始
started = []
d.start_requested.connect(lambda a, s: started.append((a, s)))
d._on_start_extract()
blocked = not started and d.result() != QDialog.Accepted
hint = d.hint_label.text()

d.set_video(video)
info = d.info_label.text()
readonly = d.path_edit.isReadOnly() and d.out_edit.isReadOnly()
outdir = d.out_edit.text()
fps_cap = d.fps_spin.maximum()
full = d.settings()

d.start_spin.setValue(4.0); d.end_spin.setValue(1.0)   # 起 > 止
d._on_start_extract()
bad_blocked = not started
bad_hint = d.hint_label.text()

d.start_spin.setValue(1.0); d.end_spin.setValue(3.0)
partial = d.settings()
spin_fps = d.fps_spin.value()
d._on_start_extract()
# 合法参数：发信号但不关窗（弹窗要留着显示进度条）
signalled = len(started) == 1 and started[0][0] == "extract"
not_closed = d.result() != QDialog.Accepted
run_action, run_settings = started[0] if started else (None, {})

bad = VideoDialog()
bad.set_video(os.path.join(tmp, "不存在.mp4"))
bad_ok = bool(bad.hint_label.text()) and not bad.start_btn.isEnabled()

print("RESULT", repr((before, blocked, hint, info, readonly,
                      os.path.basename(outdir), fps_cap,
                      full["start"], full["end"], full["image_format"],
                      bad_blocked, bad_hint,
                      partial["start"], partial["end"],
                      signalled, not_closed, run_action,
                      run_settings.get("target_fps"), spin_fps, bad_ok)))
''' % str(ROOT)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"

    (before, blocked, hint, info, readonly, outdir, fps_cap,
     start, end, fmt, bad_blocked, bad_hint, p_start, p_end,
     signalled, not_closed, run_action, run_fps, spin_fps, bad_ok) = \
        eval(line[0][len("RESULT"):].strip())

    assert before is False, "未选视频时不能开始"
    assert blocked is True and "请先选择视频文件" in hint, hint
    assert "96×64" in info and "10.000 FPS" in info and "50 帧" in info, info
    assert readonly is True, "路径与输出目录都必须只读"
    assert outdir == "我的 视频_frames", outdir
    assert abs(fps_cap - 10.0) < 1e-6, fps_cap
    # 全片按 None 传，才能触发读至 EOF 的帧数核对
    assert start is None and end is None, (start, end)
    assert fmt == "jpg", fmt
    assert bad_blocked is True and "时间范围非法" in bad_hint, bad_hint
    assert abs(p_start - 1.0) < 1e-6 and abs(p_end - 3.0) < 1e-6, (p_start, p_end)
    assert signalled is True, "合法参数必须发 start_requested"
    assert not_closed is True, "按下开始不能关窗，进度条要在弹窗里显示"
    assert run_action == "extract", run_action
    # 信号带的是按下那一刻的参数，与弹窗上的值一致
    assert abs(run_fps - spin_fps) < 1e-6, (run_fps, spin_fps)
    assert run_fps > 0, run_fps
    assert bad_ok is True, "探测失败要提示并禁用开始"


# ---------- 抽帧结果加载为数据集 ----------

def test_load_video_frames_reuses_folder_loader():
    """抽帧结果必须走项目自带的文件夹加载入口。

    手写一份「塞 background_images」是不够的：背景图列表（background_list）
    由 load_background_folder 负责填充，还要重置索引、拉起分批加载与标签重扫。
    自己拼一遍就会只显示图片、列表里没有路径。
    """
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._load_video_frames)
    assert "load_background_folder" in src
    # 不能再自己拼列表/写错属性名
    assert "current_index" not in src.replace("current_background_index", "")


def test_video_frames_appear_in_background_list_with_real_qt(tmp_path):
    """真实 Qt：抽帧完成后背景图列表要列出每一张，且能选中加载。

    回归：曾经只赋值 background_images，列表条目为 0，界面看得见图、
    列表里却没有路径。
    """
    import subprocess
    import sys

    script = '''
import os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import cv2, numpy as np
from labelforge.ui.main_window import ImageEditor
from labelforge.engine import video_io

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "我的 视频.mp4")
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
for i in range(50):
    f = np.zeros((64, 96, 3), np.uint8)
    f[8:24, 4:36] = (0, 0, 255)
    vw.write(f)
vw.release()

window = ImageEditor()
result = video_io.extract_frames(video, 2.0)
expected = len(result.image_paths)
window._load_video_frames(result)
# 分批加载是异步的，排空事件直到列表条数到位
for _ in range(4000):
    app.processEvents()
    if window.background_list.count() >= expected:
        break

count = window.background_list.count()
paths = [window.background_list.item(i).text() for i in range(count)]
index_ok = window.current_background_index == 0
pixmap_ok = window.current_background is not None and not window.current_background.isNull()
window.select_background(window.background_list.item(1))
for _ in range(200):
    app.processEvents()
moved_ok = window.current_background_index == 1
print("RESULT", repr((expected, count, os.path.basename(paths[0]) if paths else "",
                     paths == sorted(paths), index_ok, pixmap_ok, moved_ok)))
window.close()
for _ in range(200):
    app.processEvents()
sys.stdout.flush()
os._exit(0)
''' % str(ROOT)

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=300, cwd=ROOT, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"

    expected, count, first, sorted_ok, index_ok, pixmap_ok, moved_ok =         eval(line[0][len("RESULT"):].strip())
    assert count == expected, f"背景图列表条数 {count} != 抽出的 {expected}"
    assert count > 0, "列表里没有任何路径"
    assert first.endswith(".jpg"), first
    assert sorted_ok is True, "列表顺序应与文件名排序一致"
    assert index_ok is True, "加载后应选中第 0 张"
    assert pixmap_ok is True, "第 0 张要有图像内容"
    assert moved_ok is True, "应能切换列表项加载别的帧"


# ---------- 标签页弹窗（抽帧 / 合成） ----------

def test_dialog_has_two_tabs():
    """抽帧与合成放在同一个弹窗的两个标签页里。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "QTabWidget" in src
    for name in ("extract_tab", "compose_tab", "tabs"):
        assert name in src, name


def test_dialog_tab_titles_translated():
    """两个页签标题都有词条。"""
    for key in ("抽帧", "合成"):
        assert key in i18n._strings["zh"], key
        assert key in i18n._strings["en"], key


def test_dialog_extract_tab_keeps_existing_params():
    """抽帧页保留原参数：选视频、每秒抽几帧、起止时间、格式、质量、输出目录。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    for name in ("path_edit", "pick_btn", "fps_spin", "start_spin", "end_spin",
                 "format_combo", "quality_spin", "out_edit"):
        assert f"self.{name}" in src, name
    for key in ("每秒抽几帧", "起始时间(秒)", "结束时间(秒)", "图片格式",
                "jpg 质量", "输出目录", "选择视频"):
        assert key in src, key


def test_dialog_compose_tab_has_its_own_params():
    """合成页参数：图片来源、每秒几帧、输出视频路径。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    for name in ("compose_dir_edit", "compose_pick_btn", "compose_fps_spin",
                 "compose_out_edit", "compose_hint_label"):
        assert f"self.{name}" in src, name
    for key in ("图片目录", "每秒几帧", "输出视频"):
        assert key in src, key


def test_dialog_tabs_have_separate_action_buttons():
    """两页各有自己的动作按钮（开始抽帧 / 开始合成），不能共用一个。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "self.start_btn" in src          # 抽帧
    assert "self.compose_btn" in src        # 合成
    assert "开始抽帧" in src and "开始合成" in src


def test_dialog_reports_which_tab_was_accepted():
    """关窗后主窗口要知道用户做的是抽帧还是合成。"""
    from labelforge.ui.video_dialog import VideoDialog

    from labelforge.ui.video_dialog import ACTION_COMPOSE, ACTION_EXTRACT

    assert ACTION_EXTRACT == "extract"
    assert ACTION_COMPOSE == "compose"
    src = inspect.getsource(VideoDialog)
    assert "self.action = ACTION_EXTRACT" in src
    assert "self.action = ACTION_COMPOSE" in src


def test_dialog_compose_detects_manifest_automatically():
    """合成页选目录后自动判断有没有清单：有就按清单，没有才让填每秒几帧。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "MANIFEST_NAME" in src or "video_manifest.json" in src
    assert "read_manifest" in src
    # 有清单时不该再要求用户填帧率
    assert "compose_fps_spin.setEnabled" in src


def test_dialog_compose_derives_output_path():
    """输出视频路径由来源推导，不让用户乱填。"""
    from labelforge.ui.video_dialog import VideoDialog

    src = inspect.getsource(VideoDialog)
    assert "compose_out_edit" in src
    assert "setReadOnly(True)" in src
    assert ".mp4" in src


# ---------- 合成：worker 与主窗口分发 ----------

def test_compose_worker_runs_in_background_thread():
    """合成必须走线程，否则大序列会冻住界面。"""
    src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    assert "class VideoComposeWorker" in src

    block = src[src.index("class VideoComposeWorker"):]
    assert "QThread" in block
    assert "compose_video(" in block
    assert "def cancel(self)" in block
    assert "cancel=" in block


def test_mixin_dispatches_by_action():
    """弹窗按 action 发信号，mixin 据此分发到抽帧或合成。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._on_video_start_requested)
    assert "ACTION_COMPOSE" in src
    assert "_start_video_compose" in src
    assert "_start_video_extract" in src
    # 弹窗不关窗：只发信号，由 mixin 起任务
    dialog_src = inspect.getsource(
        __import__('labelforge.ui.video_dialog', fromlist=['VideoDialog']).VideoDialog)
    assert "start_requested.emit" in dialog_src


def test_mixin_has_compose_lifecycle():
    """合成要有起任务、进度、完成、失败、清理。"""
    from labelforge.ui.mixins.video import VideoMixin

    for name in ("_start_video_compose", "_on_compose_progress",
                 "_on_compose_finished", "_on_compose_failed",
                 "_cleanup_compose_worker"):
        assert hasattr(VideoMixin, name), name


def test_compose_respects_busy_flag():
    """合成不能与其它任务并行：起任务时检查 _busy，结束时释放。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._start_video_compose)
    assert "_busy" in src
    assert "_compose_worker" in src
    assert "_busy = False" in inspect.getsource(VideoMixin._cleanup_compose_worker)


def test_close_event_releases_compose_worker():
    """关窗时必须释放合成线程。"""
    src = inspect.getsource(ImageEditor.closeEvent)
    assert "compose" in src.lower()


def test_compose_does_not_load_dataset():
    """合成产出是视频，不能当成图片数据集加载。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._on_compose_finished)
    assert "load_background_folder" not in src
    assert "background_images" not in src


def test_dialog_settings_uses_real_attribute_names(tmp_path):
    """settings() 引用的控件属性必须真实存在。

    回归：曾经把 self.compose_out_edit 写成 self._compose_out_edit，
    直到真实点「开始合成」才炸。源码断言查不出来，这里用真实控件调用。
    """
    import subprocess
    import sys

    script = '''
import os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import cv2, numpy as np
from labelforge.ui.video_dialog import VideoDialog
from labelforge.engine import video_io as vio

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "v.mp4")
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
for i in range(20):
    vw.write(np.zeros((48, 64, 3), np.uint8))
vw.release()
res = vio.extract_frames(video, 2.0)

# 有清单：settings() 不能抛 AttributeError
d1 = VideoDialog()
d1.set_compose_dir(res.output_dir)
d1._on_start_compose()
s1 = d1.settings()

# 无清单：同样
seq = os.path.join(tmp, "seq")
os.makedirs(seq)
for i in range(3):
    f = np.full((48, 64, 3), 40, np.uint8)
    ok, buf = cv2.imencode(".png", f)
    open(os.path.join(seq, f"a{i}.png"), "wb").write(buf.tobytes())
d2 = VideoDialog()
d2.set_compose_dir(seq)
d2._on_start_compose()
s2 = d2.settings()

# 抽帧页也要能取参数
d3 = VideoDialog()
d3.set_video(video)
d3._on_start_extract()
s3 = d3.settings()

print("RESULT", repr((sorted(s1.keys()), sorted(s2.keys()), "target_fps" in s3)))
''' % str(ROOT)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"settings() 抛异常：{out[-1500:]}"
    keys1, keys2, has_fps = eval(line[0][len("RESULT"):].strip())
    assert "output_path" in keys1 and "manifest_path" in keys1, keys1
    assert "output_fps" not in keys1, "有清单时不该带帧率"
    assert "output_fps" in keys2 and "manifest_path" not in keys2, keys2
    assert has_fps is True


def test_compose_checks_busy_flag_before_start():
    """合成起任务前必须查 _busy 与 _compose_worker，避免与其它任务并行。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._start_video_compose)
    assert "if getattr(self, '_busy', False) or getattr(self, '_compose_worker', None):" in src
    # 守卫命中后必须直接返回，不能继续起线程
    guard = src.split("_compose_worker', None):", 1)[1].lstrip()
    assert guard.startswith("return")


def test_compose_fps_disabled_and_shows_manifest_rate(tmp_path):
    """有清单时「每秒几帧」禁用，且显示清单算出的实际帧率。

    禁用却留着默认的 5.0 会让用户以为输出被写死成 5 帧。
    """
    import subprocess
    import sys

    script = '''
import os, re, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import cv2, numpy as np
from labelforge.ui.video_dialog import VideoDialog
from labelforge.engine import video_io as vio

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "v.mp4")
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
for i in range(20):
    vw.write(np.zeros((48, 64, 3), np.uint8))
vw.release()
res = vio.extract_frames(video, 2.0)

with_manifest = VideoDialog()
with_manifest.set_compose_dir(res.output_dir)

seq = os.path.join(tmp, "seq")
os.makedirs(seq)
for i in range(3):
    ok, buf = cv2.imencode(".png", np.full((48, 64, 3), 40, np.uint8))
    open(os.path.join(seq, f"a{i}.png"), "wb").write(buf.tobytes())
without = VideoDialog()
without.set_compose_dir(seq)

label = with_manifest.compose_info_label.text()
shown = float(re.search(r"([0-9.]+) FPS", label).group(1))
print("RESULT", repr((with_manifest.compose_fps_spin.isEnabled(),
                      without.compose_fps_spin.isEnabled(),
                      with_manifest.compose_fps_spin.value(), shown,
                      without.compose_fps_spin.value())))
''' % str(ROOT)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出：{out[-1500:]}"
    (with_manifest_enabled, without_enabled, shown_value, label_fps,
     plain_default) = eval(line[0][len("RESULT"):].strip())
    assert with_manifest_enabled is False, "有清单时帧率必须禁用"
    assert without_enabled is True, "无清单时帧率必须可填"
    assert abs(shown_value - label_fps) < 0.001, \
        f"禁用状态下应显示清单实际帧率 {label_fps}，实际显示 {shown_value}"
    assert shown_value != 5.0, "不能留着默认 5.0 让人以为写死了"
    assert plain_default == 5.0, "无清单时仍应给一个可编辑的默认值"


def test_extract_fps_cap_follows_the_cached_video(tmp_path):
    """「每秒抽几帧」上限跟着源帧率走：换视频要重算上限并回落到合法值。

    用户实测反馈：上限跟源帧率这件事要落在用户真能看到的地方——换一个
    高帧率的源，上限要跟着抬；换一个低帧率的源，之前填写的高值要能被
    上限拉回来，而不是留着一个超过源帧率的数字。

    分别造 8 / 30 fps 两个源，因为同一弹窗里换视频才会暴露「上限没跟着
    重算」或「旧值越界」这类问题（mp4v 写不出奇数尺寸，这里用偶数）。
    """
    import subprocess
    import sys

    script = '''
import os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import cv2, numpy as np
from labelforge.ui.video_dialog import VideoDialog

tmp = tempfile.mkdtemp()
def make(fps, name):
    path = os.path.join(tmp, name)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for _ in range(12):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    return path

slow = make(8.0, "slow.mp4")
fast = make(30.0, "fast.mp4")

d = VideoDialog()
d.set_video(slow)
slow_cap = d.fps_spin.maximum()
d.fps_spin.setValue(8.0)                 # 用户调到上限
kept = d.fps_spin.value()
d.set_video(fast)                        # 换成高帧率源
fast_cap = d.fps_spin.maximum()
fast_value = d.fps_spin.value()
d.set_video(slow)                        # 再换回低帧率源
back_cap = d.fps_spin.maximum()
back_value = d.fps_spin.value()
print("RESULT", repr((slow_cap, kept, fast_cap, fast_value,
                      back_cap, back_value)))
''' % str(ROOT)

    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出：{out[-1500:]}"
    slow_cap, kept, fast_cap, fast_value, back_cap, back_value = \
        eval(line[0][len("RESULT"):].strip())

    assert abs(slow_cap - 8.0) < 1e-6, f"8fps 源的上限应为 8：{slow_cap}"
    assert abs(kept - 8.0) < 1e-6, kept
    assert abs(fast_cap - 30.0) < 1e-6, f"换 30fps 源后上限应抬到 30：{fast_cap}"
    assert fast_value <= fast_cap + 1e-6, (fast_value, fast_cap)
    assert abs(back_cap - 8.0) < 1e-6, f"换回 8fps 源后上限应回落：{back_cap}"
    assert back_value <= back_cap + 1e-6, \
        f"旧的高帧率值必须被上限拉回来：{back_value} > {back_cap}"


# ---------- 任务期间的弹窗行为 ----------

def _run_dialog_qt(script):
    """子进程 + 真实 Qt 跑一段脚本（调用方已填好路径），取回 RESULT 的值。"""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出：{out[-1500:]}"
    return eval(line[0][len("RESULT"):].strip())


def test_dialog_progress_bar_hidden_until_run():
    """空闲时不显示进度条；按下开始后亮起来，参数锁住。"""
    script = '''
import sys
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
# 用 isHidden 而不是 isVisible：弹窗没 show() 过，isVisible 恒为 False
idle = (not d.progress_bar.isHidden(), not d.progress_label.isHidden())

d.begin_run()
running = (not d.progress_bar.isHidden(), not d.progress_label.isHidden(),
           d.progress_bar.minimum(), d.progress_bar.maximum(),
           d.progress_label.text())

d.set_progress(3, 10)
mid = (d.progress_bar.maximum(), d.progress_bar.value(),
       d.progress_label.text())

d.set_running(False)
done = (d.progress_bar.value(), d.progress_label.text())

print("RESULT", repr((idle, running, mid, done)))
''' % str(ROOT)
    idle, running, mid, done = _run_dialog_qt(script)

    assert idle == (False, False), "空闲时不该占视觉"
    # 还没收到进度：不确定态（0,0）让用户看到「在跑」
    assert running[0] is True and running[1] is True, running
    assert (running[2], running[3]) == (0, 0), running
    assert running[4] == "抽帧中", running[4]
    assert mid[0] == 10 and mid[1] == 3, mid
    assert mid[2] == "抽帧中 3/10", mid[2]


def test_dialog_progress_uses_compose_verb_on_compose_tab():
    """合成任务显示「合成中」，不能沿用抽帧的文案。"""
    script = '''
import sys
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
d.action = "compose"
d.set_progress(1, 4)
print("RESULT", repr(d.progress_label.text()))
''' % str(ROOT)

    assert _run_dialog_qt(script) == "合成中 1/4"


def test_dialog_locks_params_while_running():
    """任务在跑时参数不能再改（改了也不生效，不如锁住避免误解）。"""
    script = '''
import sys, tempfile, os
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import cv2, numpy as np
from labelforge.ui.video_dialog import VideoDialog

tmp = tempfile.mkdtemp()
video = os.path.join(tmp, "v.mp4")
vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
for i in range(20):
    vw.write(np.zeros((48, 64, 3), np.uint8))
vw.release()

d = VideoDialog()
d.set_video(video)
before = (d.start_btn.isEnabled(), d.fps_spin.isEnabled(),
          d.start_spin.isEnabled(), d.pick_btn.isEnabled())
d.begin_run()
during = (d.start_btn.isEnabled(), d.fps_spin.isEnabled(),
          d.start_spin.isEnabled(), d.pick_btn.isEnabled())
d.set_running(False)
after = (d.start_btn.isEnabled(), d.fps_spin.isEnabled(),
         d.start_spin.isEnabled(), d.pick_btn.isEnabled())

print("RESULT", repr((before, during, after)))
''' % str(ROOT)
    before, during, after = _run_dialog_qt(script)

    assert before == (True, True, True, True), before
    assert during == (False, False, False, False), during
    assert after == (True, True, True, True), after


def test_dialog_cancel_button_cancels_while_running():
    """跑任务时「取消」是取消任务并留着窗口，空闲时才是关窗。"""
    script = '''
import sys
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication, QDialog
app = QApplication([])
from labelforge.ui.video_dialog import VideoDialog

d = VideoDialog()
cancels = []
d.cancel_requested.connect(lambda: cancels.append(1))
d.begin_run()
d._on_cancel_clicked()
during = (len(cancels), d.result(), d.progress_label.text())

d.set_running(False)
d._on_cancel_clicked()
idle = (len(cancels), d.result() != QDialog.Accepted)

print("RESULT", repr((during, idle)))
''' % str(ROOT)
    during, idle = _run_dialog_qt(script)

    assert during[0] == 1, "跑任务时取消要发信号"
    assert during[1] != 1, "跑任务时不能关窗（还要看进度）"
    assert during[2] == "正在取消...", during[2]
    assert idle[0] == 1, "空闲时点取消不该再发任务取消信号"
    assert idle[1] is True, "空闲时点取消要关窗"


class _DialogStub:
    def __init__(self):
        self.progress = []
        self.hints = []
        self.running = []
        self.accepted = 0
        self.text = ""

    @property
    def progress_label(self):
        return self

    def setText(self, text):
        self.text = text


    def set_progress(self, done, total):
        self.progress.append((done, total))

    def show_hint(self, message):
        self.hints.append(message)

    def set_running(self, running):
        self.running.append(running)

    def begin_run(self):
        self.running.append(True)

    def accept(self):
        self.accepted += 1


class _VideoStub(VideoMixin):
    """只喂弹窗生命周期要用的东西，不建真实窗口。"""

    def __init__(self):
        self.status_label = _DialogStub()
        self._busy = False
        self._video_worker = None
        self._compose_worker = None
        self._video_dialog = None
        self.started = []

    def _start_video_extract(self, settings):
        self.started.append(("extract", settings))
        return True

    def _start_video_compose(self, settings):
        self.started.append(("compose", settings))
        return True


def test_start_requested_dispatches_and_shows_progress_bar():
    """按下开始后：起任务 + 弹窗切到运行态（进度条亮起、参数锁住）。"""
    stub = _VideoStub()
    dialog = _DialogStub()
    stub._video_dialog = dialog

    stub._on_video_start_requested("extract", {"video_path": "a.mp4"})

    assert stub.started == [("extract", {"video_path": "a.mp4"})], stub.started
    assert dialog.running == [True], dialog.running


def test_start_requested_refuses_when_busy_and_explains_in_dialog():
    """已有任务时不能起第二个，并且要在弹窗里说明（弹窗还开着）。"""
    stub = _VideoStub()
    dialog = _DialogStub()
    stub._video_dialog = dialog
    stub._busy = True

    stub._on_video_start_requested("extract", {})

    assert stub.started == []
    assert dialog.hints and "已有任务在运行" in dialog.hints[0], dialog.hints
    assert dialog.running == [], "起不来就不该切成运行态"


def test_start_requested_reports_failure_in_dialog():
    """任务起不来（如线程没起来）也要在弹窗里说，别静默。"""
    class Failing(_VideoStub):
        def _start_video_compose(self, settings):
            return False

    stub = Failing()
    dialog = _DialogStub()
    stub._video_dialog = dialog

    stub._on_video_start_requested("compose", {})

    assert dialog.hints and "任务启动失败" in dialog.hints[0], dialog.hints
    assert dialog.running == [], dialog.running


def test_close_video_dialog_accepts_and_unlocks():
    """收尾必须真的关窗并解锁，否则参数永远改不了。"""
    stub = _VideoStub()
    dialog = _DialogStub()
    stub._video_dialog = dialog

    stub._close_video_dialog("抽帧完成")

    assert dialog.accepted == 1, "没关窗"
    assert dialog.running == [False], dialog.running
    assert dialog.text == "抽帧完成", dialog.text


def test_close_video_dialog_survives_manual_close():
    """用户自己先关掉弹窗时，收尾不能炸（此时 _video_dialog 已清空）。"""
    stub = _VideoStub()
    stub._video_dialog = None
    stub._close_video_dialog("抽帧完成")


def test_progress_updates_only_go_to_dialog():
    """进度只回灌弹窗：主窗口状态栏不再显示「抽帧中 x/x」。"""
    stub = _VideoStub()
    dialog = _DialogStub()
    stub._video_dialog = dialog
    stub.status_label.text = "原来的文字"

    stub._on_video_progress(3, 10)
    stub._on_compose_progress(4, 20)

    assert dialog.progress == [(3, 10), (4, 20)], dialog.progress
    assert stub.status_label.text == "原来的文字", \
        "状态栏那行是留给标签/类别信息的，别被进度覆盖"


def test_progress_without_dialog_does_not_crash():
    """弹窗已被关掉时进度还要能安全丢弃。"""
    stub = _VideoStub()
    stub._video_dialog = None
    stub._on_video_progress(1, 2)
    stub._on_compose_progress(1, 2)


def test_cancel_from_dialog_cancels_the_running_worker():
    """弹窗上点取消要真的取消任务，而不是只关窗。"""
    class Worker:
        def __init__(self):
            self.cancelled = 0

        def cancel(self):
            self.cancelled += 1

    stub = _VideoStub()
    compose = Worker()
    stub._compose_worker = compose

    stub._cancel_video_task()

    assert compose.cancelled == 1, "合成任务没被取消"

    stub2 = _VideoStub()
    extract = Worker()
    stub2._video_worker = extract
    stub2._cancel_video_task()
    assert extract.cancelled == 1, "抽帧任务没被取消"


def test_cancel_from_dialog_without_worker_is_safe():
    _VideoStub()._cancel_video_task()


# ---------- 进度条样式全项目一致 ----------

def test_all_progress_bars_share_one_rounded_style():
    """四个进度条（视频弹窗、推理进度窗、质检、QProgressDialog 工厂）同一套圆角。

    回归：视频弹窗的进度条一开始压根没写样式，是直角 12px 高，跟别处不像
    同一个控件；另外三处各抄了一份 8px 圆角 QSS，改一处就漂一处。
    """
    script = '''
import sys
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication, QProgressBar
app = QApplication([])
from labelforge.ui.theme import ThemeManager
from labelforge.ui import dialogs, inference_progress, quality_lint_dialog
from labelforge.ui.video_dialog import VideoDialog

base = ThemeManager.get_progress_style()

video = VideoDialog().styleSheet()
infer = inference_progress.InferenceProgressDialog().styleSheet()
lint = quality_lint_dialog._progress_qss()
# 留住 dialog 引用：不然它被回收，bar 变悬空对象
dlg = dialogs.ProgressDialogFactory.create_progress_dialog(None, "t", "l", 10)
factory = dlg.findChild(QProgressBar).styleSheet()

print("RESULT", repr((base.count("border-radius: 8px"),
                      "border-radius: 8px" in video,
                      "QProgressBar::chunk" in video,
                      "border-radius: 8px" in infer,
                      "border-radius: 8px" in lint,
                      "border-radius: 8px" in factory,
                      lint == base)))
''' % str(ROOT)

    out = _run_dialog_qt(script)
    (radius_count, video_rounded, video_chunk, infer_rounded,
     lint_rounded, factory_rounded, lint_is_base) = out

    assert radius_count == 2, "统一样式要同时给外层与 chunk 圆角"
    assert video_rounded and video_chunk, "视频弹窗进度条没走统一圆角样式"
    assert infer_rounded, "推理进度窗没走统一圆角样式"
    assert lint_rounded and lint_is_base, "质检进度条没用 ThemeManager 的统一实现"
    assert factory_rounded, "QProgressDialog 工厂没走统一圆角样式"


def test_video_progress_bar_matches_project_height():
    """视频进度条高度跟其它三处一致（统一样式给的 16px），不再是 12px 直角。"""
    src = (ROOT / "labelforge" / "ui" / "video_dialog.py").read_text(encoding="utf-8")
    assert "setFixedHeight(12)" not in src, "12px 直角是旧的、与别处不一致的高度"
    assert "get_progress_style" in src, "要用 ThemeManager 的统一进度条样式"
