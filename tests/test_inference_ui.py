"""推理 UI 接线回归测试。

约定（同 test_dataset_tools_ui / test_ui_regressions）：
- 不实例化 Qt 窗口；源码字符串 / AST / i18n 字典断言 + monkeypatch 派发。
- 结果应用的行为用假 editor 直接调 mixin 方法验证。
"""
import ast
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pastelabel.ui import i18n
from pastelabel.ui.main_window import ImageEditor
from pastelabel.ui.mixins.inference import InferenceMixin
from pastelabel.ui.mixins.options_popup import OptionsPopupMixin
from pastelabel.ui.mixins.toolbar import ToolbarMixin
from pastelabel.ui.mixins.translation import TranslationMixin

ROOT = Path(__file__).resolve().parents[1]

INFERENCE_KEYS = (
    "推理", "模型推理", "模型推理:", "显示模型参数行", "加载模型",
    "尚未加载模型", "加载 ONNX 模型", "推理当前图片",
    "从当前图片推理到末尾", "选择 ONNX 模型", "置信度", "NMS", "输入尺寸",
    "覆盖", "补充", "覆盖：替换上一轮推理框；补充：保留并追加",
    "动态轴模型需要指定；静态轴模型按模型自身尺寸推理",
    "关闭参数行（关闭后不能推理）", "模型参数行已关闭，无法推理",
    "模型缺少类别信息",
    "一行一个类别名；将保存到模型同级的同名 txt",
    "未提供类别信息，无法推理", "请先加载模型", "正在取消...",
    "正在加载模型...", "模型加载失败", "模型加载完成", "推理中",
    "推理完成", "推理失败", "未安装 onnxruntime，无法执行推理",
    "数据集已切换，推理结果已丢弃", "张图", "个框", "张已写入", "张写入失败",
)


# ---------- 工具栏入口 ----------

def test_model_icons_sit_right_of_label_import_buttons():
    """三个模型图标在「标签:」两个导入图标右边。"""
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    label_idx = src.index("self.label_lbl = QLabel")
    paste_label_idx = src.index("self.upload_paste_label_btn = self._create_svg_button")
    model_lbl_idx = src.index("self.model_lbl = QLabel")
    assert label_idx < paste_label_idx < model_lbl_idx


def test_three_model_icons_exist_and_are_wired():
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    for name, slot in (("model_row_btn", "_toggle_model_row"),
                       ("model_single_btn", "_run_inference_current"),
                       ("model_all_btn", "_run_inference_from_here")):
        assert f"self.{name} = self._create_svg_button(" in src, name
        assert f"self.{name}" in src and slot in src, name


def test_old_inference_button_is_gone_from_options_row():
    """「质检」右边那个推理按钮已删除，位置留给「格式」。"""
    src = inspect.getsource(OptionsPopupMixin._create_options_menu)
    assert "inference_btn" not in src
    assert src.index("self.lint_btn = QPushButton") < src.index("self.format_btn = QPushButton")


def test_toolbar_texts_refreshed_on_language_switch():
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "model_lbl" in src
    for name in ("model_row_btn", "model_single_btn", "model_all_btn"):
        assert name in src, name


# ---------- mixin 注册 ----------

def test_inference_mixin_registered_on_image_editor():
    assert InferenceMixin in ImageEditor.__mro__
    assert callable(getattr(ImageEditor, "_load_inference_model"))
    assert callable(getattr(ImageEditor, "_run_inference_current"))


def test_inference_mixin_contract():
    for name in ("_load_inference_model", "_run_inference_current",
                 "_run_inference_from_here", "_start_inference",
                 "_apply_inference_results", "_apply_inference_to_current",
                 "_on_model_thresholds_changed", "_on_model_size_changed",
                 "_inference_dataset_changed", "_ensure_model_row"):
        assert hasattr(InferenceMixin, name), name


def test_dialog_module_is_deleted():
    """弹窗已被图标 + 模型行取代。"""
    assert not (ROOT / "pastelabel" / "ui" / "inference_dialog.py").exists()


def test_inference_runs_in_background_threads():
    """模型加载与推理都必须走线程，否则大模型会冻住界面。"""
    src = (ROOT / "pastelabel" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    assert "QThread" in src
    assert src.count("def run(self)") >= 2, "加载与推理各一个 worker"
    mixin = (ROOT / "pastelabel" / "ui" / "mixins" / "inference.py").read_text(encoding="utf-8")
    assert "worker.start()" in mixin
    assert "detector.detect(" not in mixin, "推理只能在 worker 里执行"


def test_mixin_does_not_write_annotations_per_box():
    """结果应用不能逐框调 _create_detection_box，否则整批会产生上千次保存。"""
    src = inspect.getsource(InferenceMixin._apply_inference_to_current)
    assert "_create_detection_box" not in src
    assert "save_undo_state()" in src


# ---------- i18n ----------

def test_i18n_inference_terms():
    for key in INFERENCE_KEYS:
        assert key in i18n._strings["zh"], key
        assert key in i18n._strings["en"], key


@pytest.mark.parametrize("rel", [
    "pastelabel/ui/inference_workers.py",
    "pastelabel/ui/inference_progress.py",
    "pastelabel/ui/model_row.py",
    "pastelabel/ui/mixins/inference.py",
])
def test_every_tr_key_in_inference_code_is_translated(rel):
    """推理相关代码里用到的每个 tr() 键都必须在两个语言表里有定义。

    tr() 取不到键时回退成键本身，所以漏译不会报错，只会在英文界面漏出中文；
    这条测试把「键存在」变成显式契约。
    """
    import re
    missing = []
    src = (ROOT / rel).read_text(encoding="utf-8")
    for key in sorted(set(re.findall(r"tr\(['\"]([^'\"]+)['\"]\)", src))):
        for lang in ("zh", "en"):
            if key not in i18n._strings[lang]:
                missing.append((rel, lang, key))
    assert not missing, missing


def test_inference_keys_do_not_shadow_existing_translations():
    """新加的键不能与已有键重名：字典后值覆盖前值，会改掉别的界面文案。

    必须查源码而不是 i18n._strings —— 字典字面量在运行时已经去重，
    重名的前一份定义在内存里根本看不到。
    """
    import collections
    text = (ROOT / "pastelabel/ui/i18n.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    dicts = [n for n in ast.walk(tree)
             if isinstance(n, ast.Dict) and len(n.keys) > 100]
    assert len(dicts) == 2, "i18n 里应有 zh / en 两个语言表"
    for node in dicts:
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        dupes = {k for k, c in collections.Counter(keys).items() if c > 1}
        shadowed = sorted(dupes & set(INFERENCE_KEYS))
        assert not shadowed, shadowed


# ---------- 阈值与尺寸不烘焙 ----------

def test_thresholds_are_passed_per_call_not_baked_in():
    """detect 必须接受阈值参数，否则调参就得重建会话。"""
    from pastelabel.engine.onnx_inference import OnnxDetector
    sig = inspect.signature(OnnxDetector.detect)
    assert "conf_threshold" in sig.parameters
    assert "iou_threshold" in sig.parameters


def test_detector_exposes_live_setters():
    from pastelabel.engine.onnx_inference import OnnxDetector
    for name in ("set_thresholds", "set_input_size"):
        assert callable(getattr(OnnxDetector, name)), name


def test_changing_thresholds_does_not_reload_model():
    """模型行改阈值只写回 detector，不能再走一遍加载流程。"""
    src = inspect.getsource(InferenceMixin._on_model_thresholds_changed)
    assert "set_thresholds" in src
    assert "_load_inference_model" not in src
    assert "_load_model_with_names" not in src


def test_changing_input_size_does_not_reload_model():
    src = inspect.getsource(InferenceMixin._on_model_size_changed)
    assert "set_input_size" in src
    assert "_load_model_with_names" not in src


# ---------- 类别名解析 ----------

def test_class_names_prefer_sidecar_txt(tmp_path):
    from pastelabel.engine import inference_task

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    (tmp_path / "m.txt").write_text("cat\ndog\n", encoding="utf-8")

    names, source = inference_task.resolve_class_names(str(model), None)
    assert (names, source) == (["cat", "dog"], "file")


def test_class_names_fall_back_to_metadata(tmp_path):
    from pastelabel.engine import inference_task

    class _Meta:
        custom_metadata_map = {"names": "{0: 'person', 1: 'car'}"}

    class _Session:
        def get_modelmeta(self):
            return _Meta()

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    names, source = inference_task.resolve_class_names(str(model), _Session())
    assert names == ["person", "car"]
    assert source == "metadata"


def test_class_names_missing_reports_missing(tmp_path):
    """两处都没有时报 missing，不能编造 0 1 2 3 之类的假类名。"""
    from pastelabel.engine import inference_task

    class _Session:
        def get_modelmeta(self):
            return type("M", (), {"custom_metadata_map": {}})()

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    names, source = inference_task.resolve_class_names(str(model), _Session())
    assert (names, source) == ([], "missing")


def test_sidecar_class_names_path_is_same_stem(tmp_path):
    from pastelabel.engine import inference_task
    assert inference_task.sidecar_class_names_path(
        str(tmp_path / "yolov8n.onnx")) == str(tmp_path / "yolov8n.txt")


def test_write_class_names_creates_file(tmp_path):
    from pastelabel.engine import inference_task
    target = tmp_path / "sub" / "m.txt"

    assert inference_task.write_class_names(str(target), ["cat", "dog"]) is True
    assert target.read_text(encoding="utf-8").splitlines() == ["cat", "dog"]


def test_cancelling_class_prompt_disables_inference():
    """取消类别输入 = 不支持推理：不加载模型，图标保持禁用。"""
    src = inspect.getsource(InferenceMixin._on_class_names_needed)
    assert "class_names_prompt" in src
    assert "self._detector = None" in src
    assert "未提供类别信息，无法推理" in src
    assert "_update_inference_buttons" in src


def test_cancelling_prompt_does_not_fall_back_to_numbers(monkeypatch):
    """取消后不能退化成 0 1 2 3... 之类的占位类名。"""
    from pastelabel.ui import inference_workers

    monkeypatch.setattr(
        "pastelabel.ui.dialog_helpers.get_text", lambda *a, **k: ("", False))
    assert inference_workers.class_names_prompt(None, "m.onnx", "m.txt") is None


# ---------- 结果应用 ----------


class _FakeCanvas:
    def __init__(self):
        self.selected_box = 5
        self.selected_boxes = [5]
        self.updates = 0
        self.status_updates = 0

    def update(self):
        self.updates += 1

    def update_status_label(self):
        self.status_updates += 1


class _FakeEditor(InferenceMixin):
    def __init__(self, images=("a.png",), current=0):
        self.background_images = list(images)
        self.current_background_index = current
        self.detection_boxes = [{"label": "manual", "x": 0, "y": 0,
                                 "width": 5, "height": 5}]
        self.detection_boxes_dict = {current: list(self.detection_boxes)}
        self.canvas = _FakeCanvas()
        self.status_label = type("L", (), {
            "setText": lambda self, t: setattr(self, "text", t),
            "text": ""})()
        self.canvas_items = []
        self._is_delete_view = False
        self.undo_snapshots = []
        self.saved = []
        self.label_list_updates = 0
        self._inference_dataset = tuple(self.background_images)

    def save_undo_state(self):
        self.undo_snapshots.append(list(self.detection_boxes))

    def update_label_list(self):
        self.label_list_updates += 1

    def save_json(self, image_path, image_name, label_prefix, canvas_items=None,
                  image_width=None, image_height=None, current_index=None):
        self.saved.append((image_path, list(self.detection_boxes)))


def _result(index, path, boxes, error=None, width=100, height=80):
    from pastelabel.engine.inference_task import ImageResult
    return ImageResult(index, path, boxes, error, width, height)


def test_apply_results_takes_single_undo_and_single_save():
    """整批结果只产生一次撤销快照、一次列表刷新、一次保存。"""
    editor = _FakeEditor()
    results = {0: _result(0, "a.png", [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2},
        {"label": "dog", "x": 5, "y": 5, "width": 2, "height": 2},
    ])}

    editor._apply_inference_results(results, False)

    assert len(editor.undo_snapshots) == 1
    assert editor.label_list_updates == 1
    assert len(editor.saved) == 1


def test_apply_results_keeps_manual_annotations():
    editor = _FakeEditor()
    results = {0: _result(0, "a.png", [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}])}

    editor._apply_inference_results(results, False)

    labels = [b["label"] for b in editor.detection_boxes]
    assert labels == ["manual", "cat"]


def test_apply_results_marks_inference_source():
    editor = _FakeEditor()
    results = {0: _result(0, "a.png", [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}])}

    editor._apply_inference_results(results, False)

    auto = [b for b in editor.detection_boxes if b["label"] == "cat"][0]
    assert auto["flags"] == {"inference": True}
    assert "paste" not in auto["flags"]


def test_apply_results_clears_canvas_selection():
    editor = _FakeEditor()
    editor._apply_inference_results({0: _result(0, "a.png", [])}, False)
    assert editor.canvas.selected_box is None
    assert editor.canvas.selected_boxes == []
    assert editor.canvas.updates >= 1


def test_apply_results_replaces_previous_inference_only():
    editor = _FakeEditor()
    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "old", "x": 1, "y": 1, "width": 2, "height": 2}])}, False)

    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "new", "x": 3, "y": 3, "width": 2, "height": 2}])}, True)

    labels = [b["label"] for b in editor.detection_boxes]
    assert labels == ["manual", "new"]


def test_apply_results_appends_when_not_replacing():
    """补充模式：新结果追加在旧推理框之后，不删旧的。"""
    editor = _FakeEditor()
    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "old", "x": 1, "y": 1, "width": 2, "height": 2}])}, False)

    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "new", "x": 3, "y": 3, "width": 2, "height": 2}])}, False)

    labels = [b["label"] for b in editor.detection_boxes]
    assert labels == ["manual", "old", "new"]


def test_apply_results_discards_when_dataset_changed():
    """迟到结果不能写到另一个数据集上。"""
    editor = _FakeEditor()
    editor.background_images = ["other.png"]

    editor._apply_inference_results(
        {0: _result(0, "a.png", [{"label": "cat", "x": 1, "y": 1,
                                  "width": 2, "height": 2}])}, False)

    assert editor.undo_snapshots == []
    assert editor.saved == []
    assert "数据集已切换" in editor.status_label.text


def test_apply_results_ignores_delete_view():
    editor = _FakeEditor()
    editor._is_delete_view = True

    editor._apply_inference_results({0: _result(0, "a.png", [])}, False)

    assert editor.undo_snapshots == []
    assert editor.saved == []


def test_apply_results_writes_other_images_to_disk(monkeypatch):
    """非当前图直接写 sidecar，不依赖主窗口索引。"""
    from pastelabel.engine import inference_task

    calls = {}
    monkeypatch.setattr(
        inference_task, "write_results",
        lambda results, memory_boxes=None, replace_auto=False, skip_index=None:
        calls.update(results=set(results), skip_index=skip_index) or (1, []))

    editor = _FakeEditor(images=("a.png", "b.png"))
    results = {0: _result(0, "a.png", []), 1: _result(1, "b.png", [])}

    editor._apply_inference_results(results, False)

    assert calls["results"] == {0, 1}
    assert calls["skip_index"] == 0


def test_apply_results_reports_failures_in_status():
    editor = _FakeEditor()
    results = {0: _result(0, "a.png", [], "模型炸了")}

    editor._apply_inference_results(results, False)

    assert "推理完成" in editor.status_label.text
    assert editor.undo_snapshots == [], "失败图不应改动标注"


def test_apply_results_does_not_touch_current_on_error():
    editor = _FakeEditor()
    before = list(editor.detection_boxes)

    editor._apply_inference_results({0: _result(0, "a.png", [], "读取失败")}, False)

    assert editor.detection_boxes == before
    assert editor.saved == []


# ---------- 单图 / 一键推理入口 ----------

def test_single_inference_targets_only_current_image():
    src = inspect.getsource(InferenceMixin._run_inference_current)
    assert "current_background_index" in src
    assert "current_only=True" in src


def test_batch_inference_runs_from_current_to_end():
    """一键推理：当前位置 → 列表末尾。"""
    src = inspect.getsource(InferenceMixin._run_inference_from_here)
    assert "range(start, len(images))" in src
    assert "current_only=False" in src


def test_batch_inference_shows_non_modal_progress_dialog():
    src = inspect.getsource(InferenceMixin._start_inference)
    assert "_ensure_progress_dialog" in src
    assert "progress.show()" in src
    assert "progress.cancelled.connect(worker.cancel)" in src


def test_progress_dialog_syncs_titlebar_on_show():
    """小窗导航栏要跟主题一致：显示时同步一次系统标题栏。"""
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.showEvent)
    assert "sync_titlebar(self)" in src, src


def test_progress_dialog_refresh_theme_repaints_and_syncs_titlebar():
    """切主题后重新上色 + 再同步标题栏，否则标题栏会留在旧主题。"""
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    assert hasattr(InferenceProgressDialog, "refresh_theme")
    src = inspect.getsource(InferenceProgressDialog.refresh_theme)
    assert "_apply_style()" in src, src
    assert "sync_titlebar(self)" in src, src


def test_progress_dialog_style_uses_theme_colors():
    """样式表必须取自当前主题，不能写死颜色。"""
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog._apply_style)
    assert "ThemeManager.get_theme()" in src, src


def test_theme_switch_refreshes_inference_widgets():
    """参数行与小窗自带样式表，全局样式刷不到它们，必须显式刷新。"""
    from pastelabel.ui.mixins.theme import ThemeMixin
    src = inspect.getsource(ThemeMixin._apply_theme)
    assert "model_row" in src, src
    assert "_progress_dialog" in src, src
    assert "refresh_theme()" in src, src


def test_model_row_has_refresh_theme():
    from pastelabel.ui.model_row import ModelRow
    assert hasattr(ModelRow, "refresh_theme")
    src = inspect.getsource(ModelRow.refresh_theme)
    assert "_apply_style()" in src, src


def test_progress_dialog_is_cancellable_and_non_modal():
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    assert hasattr(InferenceProgressDialog, "cancelled")
    src = (ROOT / "pastelabel" / "ui" / "inference_progress.py").read_text(encoding="utf-8")
    # 非模态：用 show() 而不是 exec_()
    assert "exec_()" not in src


def test_inference_requires_loaded_model():
    src = inspect.getsource(InferenceMixin._start_inference)
    assert "请先加载模型" in src
    assert "detector is None" in src


def test_buttons_disabled_until_model_loaded():
    src = inspect.getsource(InferenceMixin._update_inference_buttons)
    assert "detector is not None" in src
    assert "model_single_btn" in src and "model_all_btn" in src


def test_first_icon_toggles_the_settings_row():
    """第一个图标只负责显示/隐藏参数行，加载模型在行内进行。"""
    src = inspect.getsource(InferenceMixin._toggle_model_row)
    assert "_ensure_model_row" in src
    assert "row.show()" in src and "row.hide()" in src
    toolbar = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "self.model_row_btn = self._create_svg_button(" in toolbar
    assert "_toggle_model_row" in toolbar


def test_model_row_has_its_own_load_button():
    """加载按钮搬进参数行，并通过信号请求主窗口加载。"""
    from pastelabel.ui.model_row import ModelRow
    src = (ROOT / "pastelabel" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert "self.load_btn = QPushButton(" in src
    assert "load_requested" in src
    mixin = inspect.getsource(InferenceMixin._ensure_model_row)
    assert "load_requested.connect(self._load_inference_model)" in mixin


def test_replace_button_uses_shared_button_style():
    """覆盖/补充按钮与工具栏其它按钮同一套样式，选中态交给主题的 :checked。"""
    from pastelabel.ui.model_row import ModelRow
    src = (ROOT / "pastelabel" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert 'self.replace_btn = QPushButton()' in src
    assert 'self.replace_btn.setObjectName("optionsBtn")' in src
    assert "QPushButton#optionsBtn:checked" in (
        ROOT / "pastelabel" / "ui" / "theme.py").read_text(encoding="utf-8")


# ---------- 模型行 ----------

def test_model_row_close_hides_and_signals():
    """关闭隐藏那一行并通知主窗口；由主窗口负责停用推理图标。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow._on_close)
    assert "hide()" in src
    assert "closed.emit()" in src


def test_closing_row_disables_inference():
    """× 关闭后不能推理：按钮可用性必须把「参数行是否可见」算进去。"""
    src = inspect.getsource(InferenceMixin._update_inference_buttons)
    assert "isVisible()" in src, src
    assert "row_open" in src, src
    assert "not busy and row_open" in src, src


def test_closed_row_blocks_inference_even_when_called_directly():
    """按钮禁用只挡鼠标；直接调用也要挡住，否则快捷键或代码路径能绕过。"""
    src = inspect.getsource(InferenceMixin._start_inference)
    assert "row.isVisible()" in src, src
    assert "模型参数行已关闭，无法推理" in src, src


def test_closing_row_keeps_detector_for_reopen():
    """关掉参数行只是停用推理，模型本身留着，重新点开即可继续用。"""
    src = inspect.getsource(InferenceMixin._update_inference_buttons)
    assert "detector is None" not in src
    assert "_detector = None" not in src


def test_model_row_exposes_required_controls():
    from pastelabel.ui.model_row import ModelRow
    src = (ROOT / "pastelabel" / "ui" / "model_row.py").read_text(encoding="utf-8")
    for name in ("name_label", "conf_spin", "iou_spin", "size_spin_w",
                 "size_spin_h", "replace_btn", "close_btn"):
        assert f"self.{name} =" in src, name
    for signal in ("thresholds_changed", "input_size_changed",
                   "replace_mode_changed", "closed"):
        assert signal in src, signal
    assert hasattr(ModelRow, "set_model")


def test_model_row_inserted_above_canvas():
    src = inspect.getsource(InferenceMixin._ensure_model_row)
    assert "insertWidget(0" in src


def test_default_thresholds_are_conf_025_nms_045():
    """用户指定的默认值：置信度 0.25、NMS 0.45。"""
    from pastelabel.engine.inference_task import DEFAULT_CONF, DEFAULT_IOU
    assert DEFAULT_CONF == 0.25
    assert DEFAULT_IOU == 0.45


def test_default_input_size_is_640():
    from pastelabel.engine.onnx_inference import DEFAULT_INPUT_SIZE
    assert DEFAULT_INPUT_SIZE == (640, 640)


# ---------- 配置持久化 ----------

def test_inference_settings_roundtrip(tmp_path, monkeypatch):
    from pastelabel.core import config_manager
    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_all(inference_settings={
        "model_path": "m.onnx", "conf_threshold": 0.6,
        "output_format": "v8", "input_size": [640, 640], "extra": "dropped",
    })

    loaded = config_manager.load_all()["inference_settings"]

    assert loaded["model_path"] == "m.onnx"
    assert loaded["conf_threshold"] == 0.6
    assert loaded["output_format"] == "v8"
    assert loaded["input_size"] == (640, 640)
    assert "extra" not in loaded


def test_inference_settings_tolerates_garbage(tmp_path, monkeypatch):
    from pastelabel.core import config_manager
    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_config({"inference_settings": "not-a-dict"})

    loaded = config_manager.load_all()["inference_settings"]

    assert loaded["model_path"] == ""
    assert loaded["conf_threshold"] == 0.25


# ---------- 当前图：无检出不写盘 ----------

def test_current_image_without_detection_skips_save(tmp_path):
    """当前图无检出且无贴图时不写盘，并顺手删掉空 json。"""
    editor = _FakeEditor(images=(str(tmp_path / "a.png"),))
    editor.detection_boxes = []
    editor.detection_boxes_dict = {0: []}
    (tmp_path / "a.json").write_text(
        '{"version": "5.0.1", "flags": {}, "imageData": null,'
        ' "imagePath": "a.png", "imageHeight": 80, "imageWidth": 100, "shapes": []}',
        encoding="utf-8")

    editor._apply_inference_to_current(_result(0, editor.background_images[0], []), False)

    assert editor.saved == []
    assert not (tmp_path / "a.json").exists()
    assert editor.undo_snapshots, "仍要留下撤销点，便于回退"


def test_current_image_without_detection_keeps_manual_boxes(tmp_path):
    """当前图有人工标注时照常保存，不能因为没检出就不写。"""
    editor = _FakeEditor(images=(str(tmp_path / "a.png"),))

    editor._apply_inference_to_current(_result(0, editor.background_images[0], []), False)

    assert len(editor.saved) == 1
    assert editor.detection_boxes[0]["label"] == "manual"


def test_current_image_without_detection_saves_when_paste_shapes_exist(tmp_path):
    """画布上有贴图时仍要写盘，否则贴图状态会丢。"""
    editor = _FakeEditor(images=(str(tmp_path / "a.png"),))
    editor.detection_boxes = []
    editor.detection_boxes_dict = {0: []}
    editor.canvas_items = [(None, object(), "pasted")]

    editor._apply_inference_to_current(_result(0, editor.background_images[0], []), False)

    assert len(editor.saved) == 1


# ---------- 设置持久化与动态轴 ----------

def test_saved_thresholds_are_restored_on_reload():
    """重新加载模型要沿用上次的阈值与尺寸，不能每次都回到默认。"""
    src = inspect.getsource(InferenceMixin._saved_inference_settings)
    assert "conf_threshold" in src
    assert "iou_threshold" in src
    assert "input_size" in src
    load = inspect.getsource(InferenceMixin._load_inference_model)
    assert "_saved_inference_settings" in load


def test_worker_passes_saved_thresholds_to_detector():
    src = (ROOT / "pastelabel" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    for key in ("conf_threshold", "iou_threshold", "input_size"):
        assert f"self._{key}" in src or key in src, key
    assert "conf_threshold=self._conf" in src
    assert "iou_threshold=self._iou" in src


def test_engine_defaults_are_single_source():
    """默认值只在引擎里定义一份，任务模块与界面都引用它。"""
    from pastelabel.engine import inference_task, onnx_inference
    assert inference_task.DEFAULT_CONF == onnx_inference.DEFAULT_CONF == 0.25
    assert inference_task.DEFAULT_IOU == onnx_inference.DEFAULT_IOU == 0.45


def test_detector_defaults_come_from_constants():
    """不传阈值时 detector 用引擎默认值，而不是各自写死一个数。"""
    from pastelabel.engine.onnx_inference import (
        DEFAULT_CONF, DEFAULT_IOU, OnnxDetector,
    )
    detector = OnnxDetector("m.onnx")
    assert detector.conf_threshold == DEFAULT_CONF
    assert detector.iou_threshold == DEFAULT_IOU


def test_static_model_size_field_is_read_only():
    """静态轴模型的尺寸由模型决定，输入框禁用，避免改了没效果。"""
    src = inspect.getsource(InferenceMixin._on_model_loaded)
    assert "dynamic=" in src
    from pastelabel.ui.model_row import ModelRow
    row_src = inspect.getsource(ModelRow.set_model)
    assert "setEnabled(dynamic)" in row_src


def test_class_prompt_writes_txt_next_to_model(tmp_path):
    """用户输入后要真的在模型同级写出同名 txt。"""
    from pastelabel.engine import inference_task
    from pastelabel.ui import inference_workers

    model = tmp_path / "yolov8n.onnx"
    model.write_bytes(b"x")
    suggested = inference_task.sidecar_class_names_path(str(model))

    written = inference_task.write_class_names(suggested, ["cat", "dog"])
    assert written is True
    assert (tmp_path / "yolov8n.txt").read_text(encoding="utf-8").splitlines() == ["cat", "dog"]
    assert inference_workers.read_saved_class_names(suggested) == ["cat", "dog"]


# ---------- 类别筛选 ----------

def test_first_icon_uses_sliders_glyph():
    """第一个图标是「参数行开关」，用滑杆比芯片更贴语义。"""
    from pastelabel.ui import icons
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "SVG_MODEL_SLIDERS, self._toggle_model_row" in src
    assert hasattr(icons, "SVG_MODEL_SLIDERS")
    assert "line" in icons.SVG_MODEL_SLIDERS, "滑杆图标应含横线"


def test_cross_mark_label_import_icons_unchanged():
    """另外两个图标（单图/批量）不动。"""
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "SVG_MODEL_SINGLE, self._run_inference_current" in src
    assert "SVG_MODEL_ALL, self._run_inference_from_here" in src


def test_model_row_has_class_filter_button():
    from pastelabel.ui.model_row import ModelRow
    src = (ROOT / "pastelabel" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert "self.filter_btn = QPushButton(" in src
    assert "filter_requested" in src
    assert 'self.filter_btn.setObjectName("optionsBtn")' in src


def test_filter_button_disabled_until_model_loaded():
    """没模型就没有类别可选，按钮要是灰的。"""
    from pastelabel.ui.model_row import ModelRow
    assert "self.filter_btn.setEnabled(False)" in inspect.getsource(ModelRow.clear_model)
    assert "self.filter_btn.setEnabled(True)" in inspect.getsource(ModelRow.set_model)


def test_filter_button_shows_selected_count():
    """不打开面板也能看出当前筛了几类。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_filter_summary)
    assert "len(selected)" in src and "total" in src


def test_mixin_opens_filter_panel_with_model_classes():
    src = inspect.getsource(InferenceMixin._open_class_filter)
    assert "ClassFilterPanel" in src
    assert "detector.class_names" in src, "应列出当前模型的全部类别"
    assert "请先加载模型" in src, "没模型要拦住"


def test_filter_survives_as_empty_means_no_filter():
    """面板「全不选」= 不过滤：空集合要变成 None 传给引擎。"""
    src = inspect.getsource(InferenceMixin._filter_for_run)
    assert "or None" in src, src


def test_class_filter_passed_into_worker():
    src = inspect.getsource(InferenceMixin._start_inference)
    assert "_filter_for_run()" in src, src
    worker_src = (ROOT / "pastelabel" / "ui" / "inference_workers.py").read_text(
        encoding="utf-8")
    assert "class_filter=self._class_filter" in worker_src


def test_changing_model_resets_class_filter():
    """换模型后类别表可能完全不同，旧勾选必须作废。"""
    src = inspect.getsource(InferenceMixin._on_model_loaded)
    assert "_class_filter = set()" in src, src


def test_filter_panel_has_select_all_and_clear():
    from pastelabel.ui.class_filter_panel import ClassFilterPanel
    src = (ROOT / "pastelabel" / "ui" / "class_filter_panel.py").read_text(
        encoding="utf-8")
    assert "全选" in src and "全不选" in src
    assert hasattr(ClassFilterPanel, "set_class_names")
    assert hasattr(ClassFilterPanel, "current_selection")
    assert hasattr(ClassFilterPanel, "set_selection")
    assert hasattr(ClassFilterPanel, "retranslate")


def test_filter_panel_is_scrollable():
    """COCO 80 类会很长，必须能滚。"""
    src = (ROOT / "pastelabel" / "ui" / "class_filter_panel.py").read_text(
        encoding="utf-8")
    assert "QScrollArea" in src


def test_filter_panel_rebuilds_keep_previous_selection():
    """重建列表（比如换模型后重开）要保留还存在的勾选。"""
    from pastelabel.ui.class_filter_panel import ClassFilterPanel
    src = inspect.getsource(ClassFilterPanel.set_class_names)
    assert "keep" in src, src


# ---------- 进度小窗日志 ----------

def test_progress_log_keeps_only_five_lines():
    from pastelabel.ui.inference_progress import LOG_LINES, InferenceProgressDialog
    assert LOG_LINES == 5
    src = inspect.getsource(InferenceProgressDialog._build_ui)
    assert "setMaximumBlockCount(LOG_LINES)" in src, src


def test_progress_log_line_has_time_boxes_and_duration():
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "beijing_timestamp()" in src, src
    assert "个框" in src, src
    assert "ms" in src, src


def test_progress_accumulates_total_box_count():
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "self._total_boxes += int(boxes)" in src, src


def test_progress_does_not_yank_scroll_when_user_scrolled_up():
    """用户往上翻历史时新结果不能把他拽回底部。"""
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog._append_log)
    assert "at_bottom" in src, src
    assert "if at_bottom" in src, src


def test_progress_uses_utc_plus_8_not_local_time():
    from pastelabel.ui import inference_progress
    src = (ROOT / "pastelabel" / "ui" / "inference_progress.py").read_text(
        encoding="utf-8")
    assert "beijing_timestamp" in src
    assert "datetime.now()" not in src, "不能用本机时间"


def test_progress_tolerates_three_arg_progress_calls():
    """单图路径也可能走这条连接，缺框数时不应报错。"""
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "if boxes is None" in src, src


def test_progress_reset_clears_log_and_counter():
    from pastelabel.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.reset)
    assert "log_area.clear()" in src
    assert "_total_boxes = 0" in src


# ---------- 端到端模型（v26）在界面上的表现 ----------

def test_model_row_disables_nms_for_end2end_model():
    """v26 已内置 NMS，界面上的 NMS 输入框必须禁用，否则用户改了没反应。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "self.iou_spin.setEnabled(not self._end2end)" in src, src


def test_model_row_explains_why_nms_is_disabled():
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "该模型已内置 NMS，此项不生效" in src, src


def test_model_row_tracks_end2end_flag():
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "self._end2end = bool(end2end)" in src
    # __init__ 里要有默认值，否则未加载模型时访问会 AttributeError
    init_src = inspect.getsource(ModelRow.__init__)
    assert "self._end2end = False" in init_src, init_src


def test_end2end_flag_passed_from_detector_format():
    """界面禁用与否取决于模型真实格式，不是靠用户勾选。"""
    src = inspect.getsource(InferenceMixin._on_model_loaded)
    assert "FORMAT_V26" in src, src
    assert "end2end=self._model_format == FORMAT_V26" in src, src


def test_conf_stays_enabled_for_end2end_model():
    """端到端模型仍要按置信度过滤，置信度框不能一起禁用。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "self.conf_spin.setEnabled(True)" in src, src


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
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_real_row_shows_defaults_before_any_model_is_loaded():
    """真实控件：还没加载模型时，参数行显示默认值而不是 spinbox 下限。

    这条是 0.01 / 0.05 / 32x32 那个 bug 的回归护栏——只读源码挡不住，
    必须看真实控件的数值。
    """
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow
from pastelabel.engine.onnx_inference import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_INPUT_SIZE

app = QApplication([])
row = ModelRow()
print(row.conf_spin.value(), row.iou_spin.value(),
      row.size_spin_w.value(), row.size_spin_h.value())
print(DEFAULT_CONF, DEFAULT_IOU, DEFAULT_INPUT_SIZE[0], DEFAULT_INPUT_SIZE[1])
""")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    shown = lines[0].split()
    expected = lines[1].split()
    assert shown == expected, f"参数行显示 {shown}，应为 {expected}"
    assert shown == ["0.25", "0.45", "640", "640"], shown


def test_real_row_populates_values_after_set_model():
    """真实控件：加载模型后填入的是传进来的值，不是占位数字。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_model('m.onnx', 0.31, 0.62, (512, 512), ['CPUExecutionProvider'], dynamic=False)
print(row.conf_spin.value(), row.iou_spin.value(),
      row.size_spin_w.value(), row.size_spin_h.value())
""")
    shown = out.strip().splitlines()[0].split()
    assert shown == ["0.31", "0.62", "512", "512"], shown


def test_real_row_clear_model_restores_defaults():
    """真实控件：清掉模型后回到默认值，不会残留上一个模型的值。"""
    out = _run_real_qt("""
from PyQt5.QtWidgets import QApplication
from pastelabel.ui.model_row import ModelRow

app = QApplication([])
row = ModelRow()
row.set_model('m.onnx', 0.77, 0.88, (320, 320), ['CPUExecutionProvider'], dynamic=False)
row.clear_model()
print(row.conf_spin.value(), row.iou_spin.value(),
      row.size_spin_w.value(), row.size_spin_h.value())
print(row.conf_spin.isEnabled(), row.filter_btn.isEnabled())
""")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0].split() == ["0.25", "0.45", "640", "640"], lines[0]
    assert lines[1].split() == ["False", "False"], lines[1]


def test_single_inference_icon_is_a_play_triangle():
    """单图推理用播放三角，与批量的闪电区分开。"""
    from pastelabel.ui import icons
    svg = icons.SVG_MODEL_SINGLE
    assert "<rect" not in svg, "旧图标是矩形相框，应已换成三角"
    assert "M7 4.5v15l12.5-7.5z" in svg, svg
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "SVG_MODEL_SINGLE, self._run_inference_current" in src


def test_single_and_batch_icons_are_distinct():
    """两个推理图标不能长得一样，否则分不清哪个是哪个。"""
    from pastelabel.ui import icons
    assert icons.SVG_MODEL_SINGLE != icons.SVG_MODEL_ALL
    assert "M13 2L4.5 13.5" in icons.SVG_MODEL_ALL, "批量应是闪电"


def test_toolbar_icons_are_all_different():
    """四个图标两两不同：滑杆 / 三角 / 闪电 / 漏斗。"""
    from pastelabel.ui import icons
    four = [icons.SVG_MODEL_SLIDERS, icons.SVG_MODEL_SINGLE,
            icons.SVG_MODEL_ALL, icons.SVG_MODEL_FILTER]
    assert len(set(four)) == 4


# ---------- 未加载模型时参数行的初值 ----------

def test_clear_model_fills_defaults_not_spinbox_minimums():
    """没模型时参数行要显示默认值。

    不显式填值的话 QDoubleSpinBox 会停在 setRange 的下限，
    显示成 0.01 / 0.05 / 32x32，看着像参数坏了。
    """
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "DEFAULT_CONF" in src and "DEFAULT_IOU" in src, src
    assert "DEFAULT_INPUT_SIZE" in src, src
    for field in ("self.conf_spin", "self.iou_spin",
                  "self.size_spin_w", "self.size_spin_h"):
        assert f"({field}, " in src, f"{field} 未在 clear_model 里填值"


def test_clear_model_does_not_emit_change_signals():
    """填初值时不能触发阈值变更回调（那时还没有 detector）。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    # 阈值一组、尺寸一组，每组各封一次信号
    assert src.count("blockSignals(True)") >= 2, src
    assert src.count("blockSignals(False)") >= 2, src
    assert src.count("blockSignals(True)") == src.count("blockSignals(False)")


def test_clear_model_resets_end2end_and_tooltip():
    """换回无模型状态后，端到端模型的 NMS 提示文案要清掉。"""
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "self._end2end = False" in src, src
    assert "NMS 去重阈值" in src, src


def test_clear_model_resets_filter_button_text():
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "self.set_filter_summary(set(), 0)" in src, src


def test_row_constructor_starts_in_the_no_model_state():
    from pastelabel.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.__init__)
    assert "self.clear_model()" in src, src
