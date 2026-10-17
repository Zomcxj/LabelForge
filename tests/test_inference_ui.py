"""推理 UI 接线回归测试。

约定（同 test_dataset_tools_ui / test_ui_regressions）：
- 不实例化 Qt 窗口；源码字符串 / AST / i18n 字典断言 + monkeypatch 派发。
- 结果应用的行为用假 editor 直接调 mixin 方法验证。
"""
import ast
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.mixins.inference import InferenceMixin
from labelforge.ui.mixins.options_popup import OptionsPopupMixin
from labelforge.ui.mixins.toolbar import ToolbarMixin
from labelforge.ui.mixins.translation import TranslationMixin

ROOT = Path(__file__).resolve().parents[1]

INFERENCE_KEYS = (
    "推理", "模型推理", "模型推理:", "显示模型参数行", "加载模型",
    "尚未加载模型", "加载 ONNX 模型", "推理当前图片",
    "从当前图片推理到末尾", "选择 ONNX 模型", "置信度", "conf", "NMS", "输入尺寸",
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
    assert not (ROOT / "labelforge" / "ui" / "inference_dialog.py").exists()


def test_inference_runs_in_background_threads():
    """模型加载与推理都必须走线程，否则大模型会冻住界面。"""
    src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    assert "QThread" in src
    assert src.count("def run(self)") >= 2, "加载与推理各一个 worker"
    mixin = (ROOT / "labelforge" / "ui" / "mixins" / "inference.py").read_text(encoding="utf-8")
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
    "labelforge/ui/inference_workers.py",
    "labelforge/ui/inference_progress.py",
    "labelforge/ui/model_row.py",
    "labelforge/ui/mixins/inference.py",
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
    text = (ROOT / "labelforge/ui/i18n.py").read_text(encoding="utf-8")
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
    from labelforge.engine.onnx_inference import OnnxDetector
    sig = inspect.signature(OnnxDetector.detect)
    assert "conf_threshold" in sig.parameters
    assert "iou_threshold" in sig.parameters


def test_detector_exposes_live_setters():
    from labelforge.engine.onnx_inference import OnnxDetector
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
    from labelforge.engine import inference_task

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    (tmp_path / "m.txt").write_text("cat\ndog\n", encoding="utf-8")

    names, source = inference_task.resolve_class_names(str(model), None)
    assert (names, source) == (["cat", "dog"], "file")


def test_class_names_fall_back_to_metadata(tmp_path):
    from labelforge.engine import inference_task

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
    from labelforge.engine import inference_task

    class _Session:
        def get_modelmeta(self):
            return type("M", (), {"custom_metadata_map": {}})()

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")

    names, source = inference_task.resolve_class_names(str(model), _Session())
    assert (names, source) == ([], "missing")


def test_sidecar_class_names_path_is_same_stem(tmp_path):
    from labelforge.engine import inference_task
    assert inference_task.sidecar_class_names_path(
        str(tmp_path / "yolov8n.onnx")) == str(tmp_path / "yolov8n.txt")


def test_write_class_names_creates_file(tmp_path):
    from labelforge.engine import inference_task
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
    from labelforge.ui import inference_workers

    monkeypatch.setattr(
        "labelforge.ui.dialog_helpers.get_multiline_text",
        lambda *a, **k: ("", False))
    assert inference_workers.class_names_prompt(None, "m.onnx", "m.txt") is None


def test_prompt_uses_multiline_input():
    """QInputDialog 只有单行，多个类别写不下——必须用多行输入。"""
    from labelforge.ui import inference_workers
    src = inspect.getsource(inference_workers.class_names_prompt)
    assert "get_multiline_text" in src
    assert "get_text(" not in src


def test_prompt_passes_expected_class_count(monkeypatch):
    """能把「模型需要几个类别」传进输入框，用户才知道该填几行。"""
    from labelforge.ui import inference_workers

    seen = {}

    def fake(parent, title, label, text="", expected_lines=None):
        seen["expected"] = expected_lines
        return "a\nb", True

    monkeypatch.setattr("labelforge.ui.dialog_helpers.get_multiline_text", fake)
    monkeypatch.setattr(inference_workers.inference_task, "write_class_names",
                        lambda path, names: True)
    out = inference_workers.class_names_prompt(None, "m.onnx", "m.txt",
                                               expected_count=4)
    assert seen["expected"] == 4
    assert out == "m.txt"


def test_prompt_writes_all_lines_not_just_first(monkeypatch):
    """多行都要写进 txt；只写第一行等于丢类别（原始 bug）。"""
    from labelforge.ui import inference_workers

    written = {}
    monkeypatch.setattr(
        "labelforge.ui.dialog_helpers.get_multiline_text",
        lambda *a, **k: ("car\nperson\nbus\ntruck", True))
    monkeypatch.setattr(
        inference_workers.inference_task, "write_class_names",
        lambda path, names: written.update(names=list(names)) or True)

    inference_workers.class_names_prompt(None, "m.onnx", "m.txt")
    assert written["names"] == ["car", "person", "bus", "truck"], written


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
        self.status_refreshes = []

    def _refresh_background_item_status(self, index, path):
        self.status_refreshes.append((index, path))

    def save_undo_state(self):
        self.undo_snapshots.append(list(self.detection_boxes))

    def update_label_list(self):
        self.label_list_updates += 1

    def save_json(self, image_path, image_name, label_prefix, canvas_items=None,
                  image_width=None, image_height=None, current_index=None):
        self.saved.append((image_path, list(self.detection_boxes)))


def _result(index, path, boxes, error=None, width=100, height=80):
    from labelforge.engine.inference_task import ImageResult
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


def test_apply_results_replaces_all_detection_boxes():
    """覆盖模式：本图检测框全清（含人工框），只留本轮结果。

    用户明确要求「手动绘制的没有覆盖，应该都覆盖」。早先这条叫
    `..._replaces_previous_inference_only`，断言人工框保留——那正是用户
    报的「看着像框重复」的来源。
    """
    editor = _FakeEditor()
    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "old", "x": 1, "y": 1, "width": 2, "height": 2}])}, False)

    editor._apply_inference_results({0: _result(0, "a.png", [
        {"label": "new", "x": 3, "y": 3, "width": 2, "height": 2}])}, True)

    labels = [b["label"] for b in editor.detection_boxes]
    assert labels == ["new"], labels


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


def test_result_ready_writes_non_current_image_to_disk(monkeypatch):
    """非当前图逐张写 sidecar，不依赖主窗口索引。"""
    from labelforge.engine import inference_task

    calls = {}

    def fake_write(result, existing=None, replace_auto=False, class_filter=None):
        calls.update(key=result.index, path=result.path, existing=existing)
        return True

    monkeypatch.setattr(inference_task, "write_one_result", fake_write)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor.detection_boxes_dict = {1: [{"label": "manual", "x": 0, "y": 0,
                                        "width": 5, "height": 5}]}

    editor._on_result_ready(1, _result(1, "b.png", []))

    assert calls["key"] == 1
    assert calls["path"] == "b.png"
    # 内存里的框比磁盘新，必须传下去，否则会覆盖掉用户刚做的编辑
    assert calls["existing"] == editor.detection_boxes_dict[1]


def test_result_ready_skips_current_image(monkeypatch):
    """当前显示图走 `_apply_inference_to_current`，逐张回调里不能再写一遍。"""
    from labelforge.engine import inference_task

    calls = []
    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: calls.append(a) or True)

    editor = _FakeEditor(images=("a.png", "b.png"), current=0)
    editor._on_result_ready(0, _result(0, "a.png", []))

    assert calls == []


def test_result_ready_refreshes_the_row_status_dot(monkeypatch):
    """写完盘要立刻刷新那一行的圆点，用户才看得到绿点一个个亮起来。"""
    from labelforge.engine import inference_task

    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: True)
    refreshed = []
    editor = _FakeEditor(images=("a.png", "b.png"))
    editor._refresh_background_item_status = \
        lambda index, path: refreshed.append((index, path))

    editor._on_result_ready(1, _result(1, "b.png", []))

    assert refreshed == [(1, "b.png")]


def test_result_ready_discards_when_dataset_changed(monkeypatch):
    """推理途中换了数据集，迟到的结果不能写进新数据集的磁盘。"""
    from labelforge.engine import inference_task

    calls = []
    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: calls.append(a) or True)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor._inference_dataset = ("other.png",)

    editor._on_result_ready(1, _result(1, "b.png", []))

    assert calls == []


def test_batch_finish_does_not_rewrite_already_written_images(monkeypatch):
    """收尾时跳过逐张写过的图，否则会把用户中途的新编辑覆盖掉。

    走完整流程：逐张回调记账 → 收尾消费，不能只在收尾时手工塞一个集合。
    """
    from labelforge.engine import inference_task

    calls = []
    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: calls.append(("one", a[0].index)) or True)
    monkeypatch.setattr(inference_task, "write_results",
                        lambda *a, **kw: calls.append(("batch",)) or (0, []))

    editor = _FakeEditor(images=("a.png", "b.png"))
    # 1 是「非当前图」，逐张回调会写它并记账
    editor._on_result_ready(1, _result(1, "b.png", []))
    assert editor._batch_written_keys == {1}, editor._batch_written_keys
    calls.clear()

    editor._on_batch_finished({0: _result(0, "a.png", []),
                               1: _result(1, "b.png", [])})

    # 收尾只处理当前图（走正常保存），不再有任何写盘调用
    assert calls == [], calls
    assert len(editor.saved) == 1
    assert editor._batch_written_keys == set(), "记账没清空，下一批会误跳过"


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
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.showEvent)
    assert "sync_titlebar(self)" in src, src


def test_progress_dialog_refresh_theme_repaints_and_syncs_titlebar():
    """切主题后重新上色 + 再同步标题栏，否则标题栏会留在旧主题。"""
    from labelforge.ui.inference_progress import InferenceProgressDialog
    assert hasattr(InferenceProgressDialog, "refresh_theme")
    src = inspect.getsource(InferenceProgressDialog.refresh_theme)
    assert "_apply_style()" in src, src
    assert "sync_titlebar(self)" in src, src


def test_progress_dialog_style_uses_theme_colors():
    """样式表必须取自当前主题，不能写死颜色。"""
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog._apply_style)
    assert "ThemeManager.get_theme()" in src, src


def test_theme_switch_refreshes_inference_widgets():
    """参数行与小窗自带样式表，全局样式刷不到它们，必须显式刷新。"""
    from labelforge.ui.mixins.theme import ThemeMixin
    src = inspect.getsource(ThemeMixin._apply_theme)
    assert "model_row" in src, src
    assert "_progress_dialog" in src, src
    assert "refresh_theme()" in src, src


def test_model_row_has_refresh_theme():
    from labelforge.ui.model_row import ModelRow
    assert hasattr(ModelRow, "refresh_theme")
    src = inspect.getsource(ModelRow.refresh_theme)
    assert "_apply_style()" in src, src


def test_progress_dialog_is_cancellable_and_non_modal():
    from labelforge.ui.inference_progress import InferenceProgressDialog
    assert hasattr(InferenceProgressDialog, "cancelled")
    src = (ROOT / "labelforge" / "ui" / "inference_progress.py").read_text(encoding="utf-8")
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
    from labelforge.ui.model_row import ModelRow
    src = (ROOT / "labelforge" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert "self.load_btn = QPushButton(" in src
    assert "load_requested" in src
    mixin = inspect.getsource(InferenceMixin._ensure_model_row)
    assert "load_requested.connect(self._load_inference_model)" in mixin


def test_replace_button_uses_shared_button_style():
    """覆盖/补充按钮与工具栏其它按钮同一套样式，选中态交给主题的 :checked。"""
    from labelforge.ui.model_row import ModelRow
    src = (ROOT / "labelforge" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert 'self.replace_btn = QPushButton()' in src
    assert 'self.replace_btn.setObjectName("optionsBtn")' in src
    assert "QPushButton#optionsBtn:checked" in (
        ROOT / "labelforge" / "ui" / "theme.py").read_text(encoding="utf-8")


# ---------- 模型行 ----------

def test_model_row_close_hides_and_signals():
    """关闭隐藏那一行并通知主窗口；由主窗口负责停用推理图标。"""
    from labelforge.ui.model_row import ModelRow
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
    from labelforge.ui.model_row import ModelRow
    src = (ROOT / "labelforge" / "ui" / "model_row.py").read_text(encoding="utf-8")
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
    from labelforge.engine.inference_task import DEFAULT_CONF, DEFAULT_IOU
    assert DEFAULT_CONF == 0.25
    assert DEFAULT_IOU == 0.45


def test_default_input_size_is_640():
    from labelforge.engine.onnx_inference import DEFAULT_INPUT_SIZE
    assert DEFAULT_INPUT_SIZE == (640, 640)


# ---------- 配置持久化 ----------

def test_inference_settings_roundtrip(tmp_path, monkeypatch):
    from labelforge.core import config_manager
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
    from labelforge.core import config_manager
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

def test_loading_a_model_does_not_restore_thresholds():
    """加载模型的入口仍要读配置，但读回来的只剩尺寸（阈值不恢复）。"""
    load = inspect.getsource(InferenceMixin._load_inference_model)
    assert "_saved_inference_settings" in load
    saved = inspect.getsource(InferenceMixin._saved_inference_settings)
    assert "input_size" in saved


def test_thresholds_are_not_remembered_across_launches():
    """阈值不跨启动记忆：每次加载模型都回到引擎默认值。

    用户实测反馈：调过置信度/NMS 之后重启，加载模型时又被旧值套上，
    而他早就忘了上次调过——结果框多框少说不清。阈值是「这次要怎么看」
    的临时旋钮，只有输入尺寸与窗口参数（`window_spec`）值得记住。

    断言分两层：存的时候不写这两个键，读的时候也不取值——只删一边的话
    旧配置文件里的残留值还是会被读回来。
    """
    save_src = inspect.getsource(InferenceMixin._save_inference_settings)
    saved_src = inspect.getsource(InferenceMixin._saved_inference_settings)
    assert "conf_threshold" not in save_src, "不该再存置信度阈值"
    assert "iou_threshold" not in save_src, "不该再存 NMS 阈值"
    assert "conf_threshold" not in saved_src, "不该再读回置信度阈值"
    assert "iou_threshold" not in saved_src, "不该再读回 NMS 阈值"
    # 该记的还要记
    for key in ("input_size", "window_spec", "model_path"):
        assert key in save_src, f"{key} 仍应持久化"


def test_real_launch_ignores_saved_thresholds():
    """真实控件：配置文件里躺着旧阈值，加载模型时也要用默认值。

    这是「不跨启动记忆」的落脚点——只查源码字符串的话，只要还有别的
    地方读配置就照样能恢复旧值。
    """
    out = _run_real_qt("""
import os, tempfile
from PySide6.QtWidgets import QApplication
app = QApplication([])

from labelforge.core import config_manager
from labelforge.engine.onnx_inference import DEFAULT_CONF, DEFAULT_IOU
config_manager.CONFIG_PATH = os.path.join(tempfile.mkdtemp(), "c.json")
config_manager.save_config({"inference_settings": {
    "model_path": "m.onnx", "conf_threshold": 0.77, "iou_threshold": 0.11,
    "input_size": [512, 512], "window_spec": {"w": 512, "h": 512, "overlap": 64},
}})

from labelforge.ui.mixins.inference import InferenceMixin
settings = InferenceMixin._saved_inference_settings(None)
print(settings.get("conf_threshold"), settings.get("iou_threshold"))
print(DEFAULT_CONF, DEFAULT_IOU)
print(settings.get("input_size"))
""")
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0] == "None None", f"旧配置里的阈值被读回来了：{lines[0]}"
    assert lines[1] == "0.25 0.45", lines[1]
    assert lines[2] == "(512, 512)", f"输入尺寸仍要记住：{lines[2]}"


def test_worker_passes_saved_thresholds_to_detector():
    src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(encoding="utf-8")
    for key in ("conf_threshold", "iou_threshold", "input_size"):
        assert f"self._{key}" in src or key in src, key
    assert "conf_threshold=self._conf" in src
    assert "iou_threshold=self._iou" in src


def test_engine_defaults_are_single_source():
    """默认值只在引擎里定义一份，任务模块与界面都引用它。"""
    from labelforge.engine import inference_task, onnx_inference
    assert inference_task.DEFAULT_CONF == onnx_inference.DEFAULT_CONF == 0.25
    assert inference_task.DEFAULT_IOU == onnx_inference.DEFAULT_IOU == 0.45


def test_detector_defaults_come_from_constants():
    """不传阈值时 detector 用引擎默认值，而不是各自写死一个数。"""
    from labelforge.engine.onnx_inference import (
        DEFAULT_CONF, DEFAULT_IOU, OnnxDetector,
    )
    detector = OnnxDetector("m.onnx")
    assert detector.conf_threshold == DEFAULT_CONF
    assert detector.iou_threshold == DEFAULT_IOU


def test_static_model_size_field_is_read_only():
    """静态轴模型的尺寸由模型决定，整组「输入尺寸」直接不显示。

    用户要求「静态模型直接不加载，动态的才显示」。只「禁用」不够：改不动的
    输入框留在那儿，用户会反复去点它。这里连标签与 "x" 一起收起。
    """
    src = inspect.getsource(InferenceMixin._on_model_loaded)
    assert "dynamic=" in src
    from labelforge.ui.model_row import ModelRow
    row_src = inspect.getsource(ModelRow.set_model)
    assert "set_size_fields_visible(bool(dynamic))" in row_src
    # 隐藏必须覆盖标签与两个输入框，不能只藏一个
    hide_src = inspect.getsource(ModelRow.set_size_fields_visible)
    for name in ("size_lbl", "size_spin_w", "size_sep_label", "size_spin_h"):
        assert name in hide_src, name
    assert "setVisible" in hide_src


def test_class_prompt_writes_txt_next_to_model(tmp_path):
    """用户输入后要真的在模型同级写出同名 txt。"""
    from labelforge.engine import inference_task
    from labelforge.ui import inference_workers

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
    from labelforge.ui import icons
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
    from labelforge.ui.model_row import ModelRow
    src = (ROOT / "labelforge" / "ui" / "model_row.py").read_text(encoding="utf-8")
    assert "self.filter_btn = QPushButton(" in src
    assert "filter_requested" in src
    assert 'self.filter_btn.setObjectName("optionsBtn")' in src


def test_filter_button_disabled_until_model_loaded():
    """没模型就没有类别可选，按钮要是灰的。"""
    from labelforge.ui.model_row import ModelRow
    assert "self.filter_btn.setEnabled(False)" in inspect.getsource(ModelRow.clear_model)
    assert "self.filter_btn.setEnabled(True)" in inspect.getsource(ModelRow.set_model)


def test_filter_button_shows_selected_count():
    """不打开面板也能看出当前筛了几类。"""
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_filter_summary)
    assert "len(selected)" in src and "total" in src


def test_mixin_opens_filter_panel_with_model_classes():
    src = inspect.getsource(InferenceMixin._open_class_filter)
    assert "ClassFilterPanel" in src
    assert "detector.class_names" in src, "应列出当前模型的全部类别"
    assert "请先加载模型" in src, "没模型要拦住"


def test_filter_none_when_all_classes_checked():
    """默认全选等于没有限制，这时要回 None 让引擎走原路径。"""
    src = inspect.getsource(InferenceMixin._filter_for_run)
    assert "return None" in src, src
    assert "selected >= set(names)" in src, src


def test_filter_returns_concrete_set_when_some_unchecked():
    """取消了某些类别才真的过滤，返回具体集合（空集也照传，表示全不选）。"""
    src = inspect.getsource(InferenceMixin._filter_for_run)
    assert "return selected" in src, src
    assert "or None" not in src, src


def test_checked_classes_defaults_to_all():
    """没设过筛选视为全选，不是空集。"""
    from labelforge.ui.mixins.inference import InferenceMixin
    src = inspect.getsource(InferenceMixin._checked_classes)
    assert "is None" in src, src


def test_filter_summary_shows_all_selected_after_model_load(monkeypatch):
    """加载模型后按钮应显示 (总数/总数)，而不是 (0/总数)。"""
    editor = _FakeEditor()
    editor._detector = type("D", (), {"class_names": ["car", "person", "bus"]})()
    editor._class_filter = None
    row = type("R", (), {})()
    shown = {}
    row.set_filter_summary = lambda sel, total: shown.update(
        selected=set(sel), total=total)
    editor.model_row = row

    editor._refresh_filter_button()

    assert shown["total"] == 3, shown
    assert shown["selected"] == {"car", "person", "bus"}, shown


def test_class_filter_passed_into_worker():
    src = inspect.getsource(InferenceMixin._start_inference)
    assert "_filter_for_run()" in src, src
    worker_src = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(
        encoding="utf-8")
    assert "class_filter=self._class_filter" in worker_src


def test_changing_model_resets_class_filter():
    """换模型后类别表可能完全不同，旧勾选必须作废（回到默认全选）。"""
    src = inspect.getsource(InferenceMixin._on_model_loaded)
    assert "_class_filter = None" in src, src


def test_filter_panel_has_select_all_and_clear():
    from labelforge.ui.class_filter_panel import ClassFilterPanel
    src = (ROOT / "labelforge" / "ui" / "class_filter_panel.py").read_text(
        encoding="utf-8")
    assert "全选" in src and "全不选" in src
    assert hasattr(ClassFilterPanel, "set_class_names")
    assert hasattr(ClassFilterPanel, "current_selection")
    assert hasattr(ClassFilterPanel, "set_selection")
    assert hasattr(ClassFilterPanel, "retranslate")


def test_filter_panel_is_scrollable():
    """COCO 80 类会很长，必须能滚。"""
    src = (ROOT / "labelforge" / "ui" / "class_filter_panel.py").read_text(
        encoding="utf-8")
    assert "QScrollArea" in src


def test_filter_panel_rebuilds_keep_previous_selection():
    """重建列表（比如换模型后重开）要保留还存在的勾选。"""
    from labelforge.ui.class_filter_panel import ClassFilterPanel
    src = inspect.getsource(ClassFilterPanel.set_class_names)
    assert "keep" in src, src


# ---------- 进度小窗日志 ----------

def test_progress_log_keeps_only_five_lines():
    from labelforge.ui.inference_progress import LOG_LINES, InferenceProgressDialog
    assert LOG_LINES == 5
    src = inspect.getsource(InferenceProgressDialog._build_ui)
    assert "setMaximumBlockCount(LOG_LINES)" in src, src


def test_progress_log_line_has_time_boxes_and_duration():
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "beijing_timestamp()" in src, src
    assert "个框" in src, src
    assert "ms" in src, src


def test_progress_accumulates_total_box_count():
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "self._total_boxes += int(boxes)" in src, src


def test_progress_does_not_yank_scroll_when_user_scrolled_up():
    """用户往上翻历史时新结果不能把他拽回底部。"""
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog._append_log)
    assert "at_bottom" in src, src
    assert "if at_bottom" in src, src


def test_progress_uses_utc_plus_8_not_local_time():
    from labelforge.ui import inference_progress
    src = (ROOT / "labelforge" / "ui" / "inference_progress.py").read_text(
        encoding="utf-8")
    assert "beijing_timestamp" in src
    assert "datetime.now()" not in src, "不能用本机时间"


def test_progress_tolerates_three_arg_progress_calls():
    """单图路径也可能走这条连接，缺框数时不应报错。"""
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.set_progress)
    assert "if boxes is None" in src, src


def test_real_log_line_shows_time_boxes_ms_and_no_date():
    """真实控件：日志行是 `[时:分:秒.毫秒] 文件名 · N 个框 · 耗时ms`，
    时刻**不带年月日**（用户要求：一次批量只跑一趟，日期是噪声）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.inference_progress import InferenceProgressDialog

app = QApplication([])
d = InferenceProgressDialog()
d.set_total(3)
d.set_progress(1, 3, "car.jpg", boxes=4, elapsed_ms=123)
print("RESULT", repr(d.log_area.toPlainText()))
""")
    line = out.strip().removeprefix("RESULT ").strip("'\"")
    assert re.match(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\] car\.jpg", line), line
    assert "car.jpg" in line and "123ms" in line, line
    assert not re.search(r"\d{4}-\d{2}-\d{2}", line), f"日志带上了年月日：{line}"


def test_progress_reset_clears_log_and_counter():
    from labelforge.ui.inference_progress import InferenceProgressDialog
    src = inspect.getsource(InferenceProgressDialog.reset)
    assert "log_area.clear()" in src
    assert "_total_boxes = 0" in src


# ---------- 端到端模型（v26）在界面上的表现 ----------

def test_model_row_disables_nms_for_end2end_model():
    """v26 已内置 NMS，界面上的 NMS 输入框必须禁用，否则用户改了没反应。"""
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "self.iou_spin.setEnabled(not self._end2end)" in src, src


def test_model_row_explains_why_nms_is_disabled():
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.set_model)
    assert "该模型已内置 NMS，此项不生效" in src, src


def test_model_row_tracks_end2end_flag():
    from labelforge.ui.model_row import ModelRow
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
    from labelforge.ui.model_row import ModelRow
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
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_real_row_shows_defaults_before_any_model_is_loaded():
    """真实控件：还没加载模型时，参数行显示默认值而不是 spinbox 下限。

    这条是 0.01 / 0.05 / 32x32 那个 bug 的回归护栏——只读源码挡不住，
    必须看真实控件的数值。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
from labelforge.engine.onnx_inference import DEFAULT_CONF, DEFAULT_IOU, DEFAULT_INPUT_SIZE

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
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

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
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow

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
    from labelforge.ui import icons
    svg = icons.SVG_MODEL_SINGLE
    assert "<rect" not in svg, "旧图标是矩形相框，应已换成三角"
    assert "M7 4.5v15l12.5-7.5z" in svg, svg
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "SVG_MODEL_SINGLE, self._run_inference_current" in src


def test_single_and_batch_icons_are_distinct():
    """两个推理图标不能长得一样，否则分不清哪个是哪个。"""
    from labelforge.ui import icons
    assert icons.SVG_MODEL_SINGLE != icons.SVG_MODEL_ALL
    assert "M13 2L4.5 13.5" in icons.SVG_MODEL_ALL, "批量应是闪电"


def test_toolbar_icons_are_all_different():
    """四个图标两两不同：滑杆 / 三角 / 闪电 / 漏斗。"""
    from labelforge.ui import icons
    four = [icons.SVG_MODEL_SLIDERS, icons.SVG_MODEL_SINGLE,
            icons.SVG_MODEL_ALL, icons.SVG_MODEL_FILTER]
    assert len(set(four)) == 4


# ---------- 未加载模型时参数行的初值 ----------

def test_clear_model_fills_defaults_not_spinbox_minimums():
    """没模型时参数行要显示默认值。

    不显式填值的话 QDoubleSpinBox 会停在 setRange 的下限，
    显示成 0.01 / 0.05 / 32x32，看着像参数坏了。
    """
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "DEFAULT_CONF" in src and "DEFAULT_IOU" in src, src
    assert "DEFAULT_INPUT_SIZE" in src, src
    for field in ("self.conf_spin", "self.iou_spin",
                  "self.size_spin_w", "self.size_spin_h"):
        assert f"({field}, " in src, f"{field} 未在 clear_model 里填值"


def test_clear_model_does_not_emit_change_signals():
    """填初值时不能触发阈值变更回调（那时还没有 detector）。"""
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    # 阈值一组、尺寸一组，每组各封一次信号
    assert src.count("blockSignals(True)") >= 2, src
    assert src.count("blockSignals(False)") >= 2, src
    assert src.count("blockSignals(True)") == src.count("blockSignals(False)")


def test_clear_model_resets_end2end_and_tooltip():
    """换回无模型状态后，端到端模型的 NMS 提示文案要清掉。"""
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "self._end2end = False" in src, src
    assert "NMS 去重阈值" in src, src


def test_clear_model_resets_filter_button_text():
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.clear_model)
    assert "self.set_filter_summary(set(), 0)" in src, src


def test_row_constructor_starts_in_the_no_model_state():
    from labelforge.ui.model_row import ModelRow
    src = inspect.getsource(ModelRow.__init__)
    assert "self.clear_model()" in src, src


# ---------- 显示置信度 ----------

def test_toolbar_has_show_scores_checkbox():
    src = (ROOT / 'labelforge' / 'ui' / 'mixins' / 'toolbar.py').read_text(
        encoding='utf-8')
    assert 'self.show_scores_checkbox = QCheckBox(tr("显示置信度"))' in src
    assert 'showScoreCb' in src, '要跟其它隐藏复选框一样给 objectName'
    assert 'self.show_scores_checkbox.setChecked(True)' in src
    assert ('self.show_scores_checkbox.stateChanged.connect('
            'self.on_labels_checkbox_changed)') in src, '切换后必须重绘'


def test_show_scores_has_shortcut():
    from labelforge.core.config import SHORTCUT_CONFIG
    assert SHORTCUT_CONFIG.get('toggle_scores') == 'Y'


def test_shortcut_does_not_collide_with_existing():
    """新快捷键不能和已有的撞车，否则会有一个不生效。"""
    from labelforge.core.config import SHORTCUT_CONFIG
    values = [v for k, v in SHORTCUT_CONFIG.items() if k != 'toggle_scores']
    assert 'Y' not in values


def test_toggle_scores_handler_is_wired():
    from labelforge.engine.event_handler import EventHandlerMixin
    src = inspect.getsource(EventHandlerMixin.setup_shortcuts)
    assert "'toggle_scores': self._toggle_scores" in src
    handler = inspect.getsource(EventHandlerMixin._toggle_scores)
    assert 'show_scores_checkbox' in handler
    assert 'setChecked(not' in handler, '应当是切换而不是置位'


def test_toggle_scores_tolerates_missing_checkbox():
    """没有该复选框的宿主（测试替身/精简窗口）不能抛异常。"""
    from labelforge.engine.event_handler import EventHandlerMixin
    handler = inspect.getsource(EventHandlerMixin._toggle_scores)
    assert 'hasattr' in handler


def test_options_menu_lists_show_scores():
    from labelforge.ui.mixins.options_popup import OptionsPopupMixin
    src = (ROOT / 'labelforge' / 'ui' / 'mixins' /
           'options_popup.py').read_text(encoding='utf-8')
    assert src.count('"toggle_scores"') == 2, '菜单与弹窗两处都要有'


def test_renderer_reads_the_show_scores_checkbox():
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin
    src = inspect.getsource(CanvasRendererMixin._show_scores)
    assert 'show_scores_checkbox' in src
    assert 'isChecked' in src


def test_renderer_passes_score_flag_at_all_three_sites():
    """矩形、多边形、关键点三条渲染路径都要带上置信度开关。"""
    from labelforge.canvas.canvas_renderer import CanvasRendererMixin
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    assert src.count('show_score=self._show_scores()') == 3, \
        '三处渲染都要传 show_score'


def test_rect_renderer_uses_box_display_label_for_score():
    """矩形路径原先用 format_label_display，拿不到 score，必须改走 box_display_label。"""
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    rect_block = src.split('def _draw_single_detection_box')[1][:2500]
    assert 'box_display_label' in rect_block
    assert '"score": score' in rect_block, '矩形路径要把 score 传进去'


def test_translation_refreshes_show_scores_label():
    from labelforge.ui.mixins.translation import TranslationMixin
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert 'show_scores_checkbox' in src
    assert '显示置信度' in src


def test_i18n_has_show_scores_in_both_languages():
    from labelforge.ui import i18n
    assert i18n._strings['zh']['显示置信度'] == '显示置信度'
    assert i18n._strings['en']['显示置信度'] == 'Show Score'
    assert i18n._strings['zh']['在类别名后显示推理置信度']
    assert i18n._strings['en']['在类别名后显示推理置信度']


def test_score_hidden_when_label_names_hidden():
    """显示Label 关掉后置信度也要一起隐藏（分数挂在类别名后面）。"""
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    assert src.count("show_label_names_checkbox.isChecked()") >= 3


def test_rect_draw_helper_does_not_use_undefined_box():
    """_draw_single_detection_box 收到的是散开的标量，作用域里没有 box 这个名字。

    之前在这里写 box.get("score") 会抛 NameError；paintEvent 有兜底吞掉异常，
    表现就是「画布空白且没有报错」，极难排查。这条测试专门守住这个坑。
    """
    import ast
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    tree = ast.parse(src)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)][0]
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef)
              and n.name == '_draw_single_detection_box')
    params = {a.arg for a in fn.args.args}
    assert 'box' not in params
    bound = set(params) | {'self'}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id == 'box':
            raise AssertionError(
                f'第 {node.lineno} 行引用了未定义的 box（签名里只有标量参数）')


def test_rect_draw_takes_score_as_parameter():
    """score 必须作为参数传进来，而不是在函数里现找 box。"""
    import ast
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    tree = ast.parse(src)
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef)][0]
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef)
              and n.name == '_draw_single_detection_box')
    params = {a.arg for a in fn.args.args}
    assert 'score' in params, 'score 应作为参数传入'
    assert fn.args.defaults, 'score 要有默认值，避免破坏既有调用'


def test_rect_caller_passes_score():
    """调用点必须把 box 的 score 传进绘制函数，否则框上永远不会显示置信度。"""
    import ast
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and getattr(n.func, 'attr', None) == '_draw_single_detection_box']
    assert len(calls) == 1
    kw = {k.arg for k in calls[0].keywords}
    assert 'score' in kw, f'调用点缺 score 关键字参数，实际有 {kw}'
    val = next(k.value for k in calls[0].keywords if k.arg == 'score')
    assert isinstance(val, ast.Call) and val.func.attr == 'get'
    assert val.args[0].value == 'score'


def test_paint_event_swallows_errors_so_tests_must_cover_draw_helpers():
    """paintEvent 会吞掉绘制异常，说明绘制函数必须有直接测试兜底。"""
    src = (ROOT / 'labelforge' / 'canvas' / 'canvas_renderer.py').read_text(
        encoding='utf-8')
    head = src.split('def paintEvent')[1][:900]
    assert 'except' in head, 'paintEvent 依赖兜底异常处理'


# ---------- 选项菜单文字/快捷键不能错位 ----------

def test_menu_actions_store_i18n_key_not_translation():
    """每项要记住自己的 i18n 键；存翻译后的文字会在切语言时二次翻译。"""
    src = (ROOT / 'labelforge' / 'ui' / 'mixins' / 'options_popup.py').read_text(
        encoding='utf-8')
    body = src.split('def _rebuild_options_popup')[1][:6000]
    assert 'text_key' in body
    # 第一段列表每条应是 4 元组：(显示文字, 动作键, 复选框, i18n键)
    assert '"显示置信度"' in body
    assert '显示置信度' in src


def test_translation_no_longer_hardcodes_menu_labels():
    """翻译模块曾按下标回填一份并列的硬编码菜单文字表，加项就整体错位。

    约定：菜单文字只能由 _refresh_options_menu_texts 一处产生。
    """
    src = (ROOT / 'labelforge' / 'ui' / 'mixins' / 'translation.py').read_text(
        encoding='utf-8')
    assert '_menu_actions' not in src, \
        'translation 不应再直接操作 _menu_actions，应由共享方法统一刷新'
    assert src.count('_refresh_options_menu_texts()') == 2, \
        '两处刷新（retranslate / _refresh_menu_shortcuts）都要走共享方法'


def test_refresh_options_menu_texts_uses_stored_key():
    from labelforge.ui.mixins.options_popup import OptionsPopupMixin
    src = inspect.getsource(OptionsPopupMixin._refresh_options_menu_texts)
    assert 'i18n.t(text_key)' in src
    assert '_get_shortcut' in src


def test_menu_actions_are_four_tuples_with_keys():
    """结构调整后所有追加点都要保持一致，混用长度会让解包炸掉。"""
    src = (ROOT / 'labelforge' / 'ui' / 'mixins' / 'options_popup.py').read_text(
        encoding='utf-8')
    import ast
    tree = ast.parse(src)
    appends = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == '_menu_actions'):
            appends.append(node)
    assert appends, '没找到 _menu_actions.append'
    for call in appends:
        tup = call.args[0]
        assert isinstance(tup, ast.Tuple), call.lineno
        assert len(tup.elts) == 4, (call.lineno, len(tup.elts))


def test_shortcut_Y_does_not_collide():
    from labelforge.core.config import SHORTCUT_CONFIG
    assert SHORTCUT_CONFIG['toggle_scores'] == 'Y'
    values = [v for k, v in SHORTCUT_CONFIG.items() if k != 'toggle_scores']
    assert 'Y' not in values


def test_settings_dialog_lists_toggle_scores():
    """设置对话框漏了这一行，用户就没法改或查看这个快捷键。"""
    src = (ROOT / 'labelforge' / 'ui' / 'settings_dialog.py').read_text(
        encoding='utf-8')
    body = src.split('shortcut_names = {')[1].split('}')[0]
    assert "'toggle_scores'" in body
    assert '显示置信度' in body


def test_settings_dialog_keeps_display_label_adjacent():
    """插在显示标签名与显示贴图名之间，与选项菜单顺序一致。"""
    src = (ROOT / 'labelforge' / 'ui' / 'settings_dialog.py').read_text(
        encoding='utf-8')
    body = src.split('shortcut_names = {')[1].split('}')[0]
    i_name = body.index("'toggle_label_names'")
    i_scores = body.index("'toggle_scores'")
    i_paste = body.index("'toggle_paste_names'")
    assert i_name < i_scores < i_paste


# ---------- 当前图这条路径必须真的把筛选传下去 ----------

def test_current_image_merge_receives_class_filter():
    """只勾 car 时，上一轮推理留下的 truck 必须从当前图上消失。

    这是用户实际走的那条路（当前图推理），只测引擎层不够。
    """
    editor = _FakeEditor()
    editor._class_filter = {"car"}
    editor._detector = type("D", (), {"class_names": ["car", "truck"]})()
    editor.detection_boxes = [
        {"label": "truck", "x": 0, "y": 0, "width": 5, "height": 5,
         "flags": {"inference": True}},
        {"label": "manual", "x": 1, "y": 1, "width": 5, "height": 5},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor._inference_dataset = tuple(editor.background_images)

    editor._apply_inference_to_current(
        _result(0, editor.background_images[0],
                [{"label": "car", "x": 9, "y": 9, "width": 4, "height": 4}]),
        False)

    labels = sorted(b["label"] for b in editor.detection_boxes)
    assert labels == ["car", "manual"], labels


def test_current_image_merge_keeps_everything_without_filter():
    """未设筛选（None，即默认全选）时不能顺手清理。"""
    editor = _FakeEditor()
    editor._class_filter = None
    editor._detector = type("D", (), {"class_names": ["car", "truck"]})()
    editor.detection_boxes = [
        {"label": "truck", "x": 0, "y": 0, "width": 5, "height": 5,
         "flags": {"inference": True}},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}

    editor._apply_inference_to_current(
        _result(0, editor.background_images[0],
                [{"label": "car", "x": 9, "y": 9, "width": 4, "height": 4}]),
        False)

    labels = sorted(b["label"] for b in editor.detection_boxes)
    assert labels == ["car", "truck"], labels


def test_result_ready_forwards_filter_to_disk_writer(monkeypatch):
    """逐张写盘那条路也要带上筛选，否则其它图还是脏的。"""
    from labelforge.engine import inference_task

    seen = {}

    def fake_write(result, existing=None, replace_auto=False, class_filter=None):
        seen["class_filter"] = class_filter
        return True

    monkeypatch.setattr(inference_task, "write_one_result", fake_write)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor._class_filter = {"car"}
    editor._detector = type("D", (), {"class_names": ["car", "truck"]})()
    editor._on_result_ready(1, _result(1, "b.png", []))

    assert seen["class_filter"] == {"car"}, seen


# ---------- 类别数提示 ----------

def test_guess_class_count_uses_detection_head_not_first_output():
    """带辅助输出的模型也要给出类别数：要挑出检测头再算，不能只认第 0 个输出。

    断言必须看真实行为，不能 grep 源码——docstring 里就写着
    pick_detection_output，字符串匹配在变异后会假通过。
    """
    from labelforge.ui.inference_workers import ModelLoadWorker

    class Out:
        def __init__(self, shape):
            self.shape = shape

    class FakeSession:
        """辅助头排在最前，检测头在第二个：只用 outputs[0] 会算出错的数。"""

        def get_outputs(self):
            return [Out([1, 3, 80, 80, 9]), Out([1, 25200, 9])]

    got = ModelLoadWorker._guess_class_count(FakeSession())
    assert got == 4, got          # [1, 25200, 9] -> 9-5 = 4

    class NoHead:
        def get_outputs(self):
            return [Out([1, 3, 80, 80, 9])]

    assert ModelLoadWorker._guess_class_count(NoHead()) is None

    class Broken:
        def get_outputs(self):
            raise RuntimeError("boom")

    assert ModelLoadWorker._guess_class_count(Broken()) is None


def test_guess_class_count_lives_on_model_load_worker():
    """算类别数要在加载 worker 上；放错类会让 need_class_names 直接抛异常。"""
    from labelforge.ui.inference_workers import InferenceWorker, ModelLoadWorker
    assert hasattr(ModelLoadWorker, "_guess_class_count")
    assert not hasattr(InferenceWorker, "_guess_class_count")


def test_need_class_names_signal_carries_three_args():
    from labelforge.ui.inference_workers import ModelLoadWorker
    src = inspect.getsource(ModelLoadWorker.run)
    assert "self._guess_class_count(session)" in src
    # 信号本身要是三元组，否则 emit 会丢参数
    sig = (ROOT / "labelforge" / "ui" / "inference_workers.py").read_text(
        encoding="utf-8")
    assert "need_class_names = Signal(str, str, object)" in sig


def test_guess_class_count_from_shapes():
    from labelforge.engine.onnx_inference import guess_class_count
    assert guess_class_count([1, 25200, 9]) == 4        # v5: 5+C
    assert guess_class_count([1, 25200, 85]) == 80      # v5: 5+C
    assert guess_class_count([1, 84, 8400]) == 80       # v8: 4+C（通道在 rows）
    assert guess_class_count([1, 300, 6]) is None       # 端到端推不出
    assert guess_class_count([1, 3, 80, 80, 9]) is None  # 辅助头


def test_guess_class_count_refuses_ambiguous_small_shapes():
    """小形状端到端与普通检测同形，猜不得。"""
    from labelforge.engine.onnx_inference import guess_class_count
    for shape in ([1, 5, 6], [1, 100, 6], [1, 6, 6]):
        assert guess_class_count(shape) is None, shape


def test_guess_class_count_tolerates_garbage():
    from labelforge.engine.onnx_inference import guess_class_count
    for bad in (None, [], [1], "abc", [1, "x", 5]):
        assert guess_class_count(bad) is None, bad


def test_on_model_loaded_accepts_expected_count():
    """回调签名要能吃下第三个参数，不然信号一发就崩。"""
    src = inspect.getsource(InferenceMixin._on_class_names_needed)
    assert "expected_count" in src


def test_filter_for_run_is_none_without_known_classes(tmp_save_target=None):
    """类别表为空时必须回 None。

    没有类别信息时「勾了什么」无从谈起；若当成空集合（全不选），
    已有推理框会被整批清掉——这是会丢标注的静默数据损坏。
    """
    editor = _FakeEditor()
    editor._class_filter = set()          # 曾被误当成「全不选」
    assert editor._all_class_names() == []
    assert editor._filter_for_run() is None

    editor2 = _FakeEditor()
    editor2._class_filter = {"car"}       # 面板选过，但模型没类别信息
    assert editor2._filter_for_run() is None


def test_default_all_selected_needs_a_detector():
    """默认全选语义要靠真实类别表比对，不能只看 _class_filter 是不是 None。"""
    from labelforge.engine.inference_task import filter_by_class

    names = ["car", "person", "bus"]
    editor = _FakeEditor()
    editor._detector = type("D", (), {"class_names": list(names)})()
    editor._class_filter = None
    assert editor._filter_for_run() is None
    # 全选与不过滤的结论要一致
    boxes = [{"label": n} for n in names + ["truck"]]
    assert filter_by_class(boxes, editor._filter_for_run()) == boxes

    editor._class_filter = set(names)     # 面板里全勾 == 没有限制
    assert editor._filter_for_run() is None

    editor._class_filter = {"car"}        # 取消了一部分才真过滤
    assert editor._filter_for_run() == {"car"}


def test_real_panel_defaults_to_everything_checked():
    """真实面板：首次填入类别表时默认全勾（用户要求「默认全选」）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.class_filter_panel import ClassFilterPanel
app = QApplication([])
names = ["car", "person", "bus", "truck"]
panel = ClassFilterPanel(None)
panel.set_class_names(names)
got = panel.current_selection()
print("CHECKED", len(got), "OF", len(panel._boxes))
print("ALL", got == set(names))
# 再设一次（二次填表）不该把用户的选择重置掉
panel.set_selection({"car"})
panel.set_class_names(["car", "person", "bus"])
print("AFTER_RESEED", panel.current_selection() == {"car"})
""")
    assert "CHECKED 4 OF 4" in out, out
    assert "ALL True" in out, out
    assert "AFTER_RESEED True" in out, out


def test_real_multiline_dialog_drops_blank_lines():
    """真实多行框：空行和首尾空格要被丢掉，不能变成空类名。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.dialog_helpers import ThemedMultiLineDialog
app = QApplication([])
blank = chr(10) + "   " + chr(10)
dlg = ThemedMultiLineDialog(None)
dlg.set_text("  car " + blank + " person" + chr(10) + blank + "bus  " + chr(10))
print("LINES", dlg.lines())
bad = chr(25968) + chr(37327) + chr(19981) + chr(19968) + chr(33268)
dlg2 = ThemedMultiLineDialog(None, expected_lines=4)
dlg2.set_text("a" + chr(10) + "b")
print("MISMATCH", bad in dlg2.count_label.text())
dlg3 = ThemedMultiLineDialog(None, expected_lines=2)
dlg3.set_text("a" + chr(10) + "b")
print("MATCH_OK", bad not in dlg3.count_label.text())
print("TIP", dlg3.count_label.text())
""")
    assert "LINES ['car', 'person', 'bus']" in out, out
    assert "MISMATCH True" in out, out
    assert "MATCH_OK True" in out, out


def test_real_replace_button_defaults_to_overwrite():
    """真实控件：覆盖/补充默认是「覆盖」。

    默认是补充的话，用户不点这个按钮跑第二遍就会把新旧框叠在一起。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.model_row import ModelRow
app = QApplication([])
row = ModelRow()
print("MODE", row.replace_mode())
print("TEXT", row.replace_btn.text())
print("CHECKED", row.replace_btn.isChecked())
row.replace_btn.setChecked(False)
print("AFTER_TOGGLE", row.replace_mode())
print("AFTER_TEXT", row.replace_btn.text())
""")
    assert "MODE True" in out, out
    assert "CHECKED True" in out, out
    assert "AFTER_TOGGLE False" in out, out
    # 未选中显示「补充」，切回来显示「覆盖」
    assert "AFTER_TEXT" in out, out


def test_summary_counts_only_images_actually_written(monkeypatch):
    """汇总里的「张已写入」要数真的写了盘的，没框可写的不算。

    0 是当前图（有手工框，走正常保存流程 → 写）、1 写了、2 没框可写。
    """
    from labelforge.engine import inference_task

    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: a[0].index == 1)

    editor = _FakeEditor(images=("a.png", "b.png", "c.png"))
    editor._on_result_ready(1, _result(1, "b.png", []))
    editor._on_result_ready(2, _result(2, "c.png", []))
    editor._on_batch_finished({0: _result(0, "a.png", []),
                               1: _result(1, "b.png", []),
                               2: _result(2, "c.png", [])})

    # 0（手工框）与 1 写了；2 返回 False → 不能算进去
    assert "2 张已写入" in editor.status_label.text, editor.status_label.text


def test_summary_reports_write_failures(monkeypatch):
    """写盘失败要单独报出来，不能混进「已写入」。"""
    from labelforge.engine.inference_task import InferenceError
    from labelforge.engine import inference_task

    def fake_write(result, **kwargs):
        raise InferenceError("写不进去")

    monkeypatch.setattr(inference_task, "write_one_result", fake_write)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor._on_result_ready(1, _result(1, "b.png", []))
    editor._on_batch_finished({0: _result(0, "a.png", []),
                               1: _result(1, "b.png", [])})

    text = editor.status_label.text
    # 当前图（手工框）走正常保存成功；1 失败 → 两个数字必须分开
    assert "1 张写入失败" in text, text
    assert "1 张已写入" in text, text


def test_summary_does_not_count_failed_write_as_written(monkeypatch):
    """失败的图不能同时被算成「已写入」。"""
    from labelforge.engine.inference_task import InferenceError
    from labelforge.engine import inference_task

    monkeypatch.setattr(
        inference_task, "write_one_result",
        lambda *a, **kw: (_ for _ in ()).throw(InferenceError("x")))

    # 当前图换成没有手工框的空编辑器 → 当前图也不写
    editor = _FakeEditor(images=("a.png", "b.png"))
    editor.detection_boxes = []
    editor.detection_boxes_dict = {0: []}
    editor._on_result_ready(1, _result(1, "b.png", []))
    editor._on_batch_finished({0: _result(0, "a.png", []),
                               1: _result(1, "b.png", [])})

    text = editor.status_label.text
    assert "0 张已写入" in text, text
    assert "1 张写入失败" in text, text


def test_write_failure_does_not_stop_the_batch(monkeypatch):
    """单张写失败不能影响后面的图（否则一批全废）。"""
    from labelforge.engine.inference_task import InferenceError
    from labelforge.engine import inference_task

    seen = []

    def fake_write(result, **kwargs):
        seen.append(result.index)
        if result.index == 1:
            raise InferenceError("这张写不进去")
        return True

    monkeypatch.setattr(inference_task, "write_one_result", fake_write)

    editor = _FakeEditor(images=("a.png", "b.png", "c.png"))
    editor._on_result_ready(1, _result(1, "b.png", []))
    editor._on_result_ready(2, _result(2, "c.png", []))

    assert seen == [1, 2], "失败后必须继续写后面的图"
    assert len(editor.status_refreshes) == 2, "失败那张也要刷新圆点"


def test_result_ready_writes_real_json_and_turns_the_dot_green(tmp_path):
    """真实文件走一遍：写完盘那一行的圆点状态必须从「未标注」变成「已标注」。

    逐张写盘的全部意义就在这里：用户要看着 json 一张张出来、绿点一个个亮。
    """
    import json

    from labelforge.engine import inference_task
    from labelforge.engine.image_loader.status import (
        STATUS_ANNOTATED, annotation_status_for_image)

    path = tmp_path / "a.png"
    path.write_bytes(b"")
    assert annotation_status_for_image(str(path)) != STATUS_ANNOTATED

    result = _result(0, str(path), [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}])

    wrote = inference_task.write_one_result(result, existing=[])
    assert wrote is True

    assert annotation_status_for_image(str(path)) == STATUS_ANNOTATED, \
        "写完盘状态还是「未标注」，绿点不会亮"
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert [s["label"] for s in data["shapes"]] == ["cat"]


def test_write_one_result_skips_empty_merge(tmp_path):
    """没框且原本没标注：不写盘，免得数据集里多出一堆空 json。"""
    from labelforge.engine import inference_task

    path = tmp_path / "b.png"
    path.write_bytes(b"")
    wrote = inference_task.write_one_result(_result(0, str(path), []), existing=[])
    assert wrote is False
    assert not path.with_suffix(".json").exists(), "空结果不该留下 json"


def test_write_one_result_reads_disk_when_caller_has_no_memory_copy(tmp_path):
    """existing=None 时要自己读盘，否则盘上的人工框会被当成不存在而丢掉。

    只有当前显示图在内存里；其它图的底稿只能从 sidecar 拿。
    """
    import json

    from labelforge.engine import inference_task

    path = tmp_path / "e.png"
    path.write_bytes(b"")
    # 先在盘上放一个人工框（模拟用户之前标注过、内存里没有）
    inference_task.write_sidecar_boxes(
        str(path),
        [{"label": "manual", "x": 0, "y": 0, "width": 5, "height": 5,
          "flags": {}}],
        image_width=100, image_height=80)

    wrote = inference_task.write_one_result(
        _result(0, str(path), [{"label": "cat", "x": 1, "y": 1,
                                "width": 2, "height": 2}]),
        existing=None)

    assert wrote is True
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sorted(s["label"] for s in data["shapes"]) == ["cat", "manual"], \
        "existing=None 时没读盘，人工框被冲掉了"


def test_write_one_result_keeps_existing_manual_annotations(tmp_path):
    """合并时要以内存里的标注为底，不能把人工框冲掉。"""
    import json

    from labelforge.engine import inference_task

    path = tmp_path / "c.png"
    path.write_bytes(b"")
    manual = [{"label": "manual", "x": 0, "y": 0, "width": 5, "height": 5,
               "flags": {}}]
    wrote = inference_task.write_one_result(
        _result(0, str(path), [{"label": "cat", "x": 1, "y": 1,
                                "width": 2, "height": 2}]),
        existing=manual)

    assert wrote is True
    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sorted(s["label"] for s in data["shapes"]) == ["cat", "manual"]


def test_write_one_result_replace_clears_all_detection_boxes(tmp_path):
    """覆盖模式：本图检测框全清（含人工框），贴图保留。"""
    import json

    from labelforge.engine import inference_task

    path = tmp_path / "d.png"
    path.write_bytes(b"")
    existing = [
        {"label": "old", "x": 0, "y": 0, "width": 5, "height": 5,
         "flags": {"inference": True}},
        {"label": "manual", "x": 0, "y": 0, "width": 5, "height": 5,
         "flags": {}},
        {"label": "logo", "x": 0, "y": 0, "width": 5, "height": 5,
         "flags": {"paste": True}},
    ]
    inference_task.write_one_result(
        _result(0, str(path), [{"label": "new", "x": 1, "y": 1,
                                "width": 2, "height": 2}]),
        existing=existing, replace_auto=True)

    data = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sorted(s["label"] for s in data["shapes"]) == ["logo", "new"]


def test_write_one_result_raises_on_unwritable_path(tmp_path):
    """写不进去必须抛 InferenceError，让调用方记成「这张失败」。"""
    from labelforge.engine.inference_task import InferenceError, write_one_result

    # 父路径是文件而不是目录 → 建目录必失败
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    result = _result(0, str(blocker / "sub" / "e.png"), [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}])

    with pytest.raises(InferenceError):
        write_one_result(result, existing=[])


def test_write_results_matches_write_one_result_per_image(tmp_path):
    """批量入口就是逐张调单张入口：两条路的结果必须一致。"""
    import json

    from labelforge.engine import inference_task

    batch_dir = tmp_path / "batch"
    single_dir = tmp_path / "single"
    for folder in (batch_dir, single_dir):
        folder.mkdir()
        for name in ("a.png", "b.png"):
            (folder / name).write_bytes(b"")

    results = {
        0: _result(0, str(batch_dir / "a.png"),
                   [{"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}]),
        1: _result(1, str(batch_dir / "b.png"), []),
    }
    written, failed = inference_task.write_results(results, memory_boxes={})
    assert (written, failed) == (1, [])

    for key, result in results.items():
        inference_task.write_one_result(
            _result(key, str(single_dir / os.path.basename(result.path)),
                    result.boxes), existing=[])

    for name in ("a.png", "b.png"):
        batch_json = (batch_dir / name).with_suffix(".json")
        single_json = (single_dir / name).with_suffix(".json")
        assert batch_json.exists() == single_json.exists(), name
        if batch_json.exists():
            assert (json.loads(batch_json.read_text(encoding="utf-8"))
                    == json.loads(single_json.read_text(encoding="utf-8"))), name


def test_worker_emits_result_ready_per_image_before_finishing(tmp_path):
    """真实线程：每张跑完就发 result_ready，且都在 finished_ok 之前。

    逐张写盘全靠这个信号，顺序错了就会「先出汇总、再一张张写」。
    """
    out = _run_real_qt("""
import sys, os, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import numpy as np
from labelforge.ui.inference_workers import InferenceWorker

tmp = tempfile.mkdtemp()
paths = []
for name in ("a.png", "b.png", "c.png"):
    p = os.path.join(tmp, name)
    open(p, "wb").close()
    paths.append(p)

class Detector:
    class_names = ["cat"]
    def detect(self, rgb, resize_fn):
        return [{"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}]

import labelforge.ui.inference_workers as workers
workers.load_rgb = lambda path: np.zeros((10, 10, 3), dtype="uint8")
workers.resize_for_model = lambda rgb: (rgb, None, None, 0, 0)

worker = InferenceWorker(paths, [0, 1, 2], Detector())
order = []
worker.result_ready.connect(lambda key, res: order.append(("ready", key)))
worker.finished_ok.connect(lambda res: order.append(("done", len(res))))
worker.failed.connect(lambda msg: order.append(("failed", msg)))
worker.run()          # 同步跑，省掉事件循环
print("RESULT", repr(order))
""" % str(ROOT))
    line = [l for l in out.splitlines() if l.startswith("RESULT")][0]
    order = eval(line[len("RESULT"):].strip())

    assert order == [("ready", 0), ("ready", 1), ("ready", 2), ("done", 3)], order


def test_first_image_of_a_batch_turns_green_immediately(tmp_path):
    """一键推理的第一张就是当前图，必须和其他图一样立刻变绿。

    回归：曾经把当前图整个跳过，留到整批结束的收尾里处理，用户看到的是
    「第一张一直不绿，跑到最后才绿」。
    """
    from labelforge.engine import inference_task
    from labelforge.engine.image_loader.status import (
        STATUS_ANNOTATED, annotation_status_for_image)

    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    for p in (a, b):
        p.write_bytes(b"")
    assert annotation_status_for_image(str(a)) != STATUS_ANNOTATED

    editor = _FakeEditor(images=(str(a), str(b)), current=0)
    # 当前图走正常保存流程 → 这里直接调真实的 save_json 等价物
    editor.save_json = lambda *args, **kw: inference_task.write_sidecar_boxes(
        str(a), editor.detection_boxes, 100, 80)

    editor._on_result_ready(0, _result(0, str(a), [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}]))

    assert annotation_status_for_image(str(a)) == STATUS_ANNOTATED, \
        "第一张（当前图）没有立刻写盘，绿点不会亮"
    assert 0 in editor._batch_written_keys, editor._batch_written_keys
    # 走的是正常保存流程：内存框与画布也要同步，不能只写盘
    assert [bx["label"] for bx in editor.detection_boxes] == ["manual", "cat"], \
        editor.detection_boxes
    assert editor.detection_boxes_dict[0] == editor.detection_boxes
    assert len(editor.undo_snapshots) == 1, "当前图要留撤销点"
    assert editor.canvas.updates >= 1, "画布没重绘"


def test_current_image_write_goes_through_the_normal_save_path(monkeypatch):
    """当前图必须走 `_apply_inference_to_current`，不能走非当前图的直接写盘。

    两条路的差别不只是写盘：正常保存流程还会更新内存框、画布与撤销快照。
    走错了就会出现「盘上有了、画布上还是旧框，切走再切回才更新」。
    """
    from labelforge.engine import inference_task

    direct = []
    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: direct.append(a[0].index) or True)

    editor = _FakeEditor(images=("a.png", "b.png"), current=0)
    editor._on_result_ready(0, _result(0, "a.png", [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}]))

    assert direct == [], "当前图不该走直接写盘那条路"
    assert len(editor.undo_snapshots) == 1
    assert len(editor.saved) == 1


def test_current_image_is_not_applied_twice_at_batch_end(monkeypatch):
    """收尾不能把当前图再处理一遍：补充模式会写重复框，覆盖模式会多压快照。"""
    from labelforge.engine import inference_task

    editor = _FakeEditor(images=("a.png", "b.png"), current=0)
    applied = []
    monkeypatch.setattr(
        _FakeEditor, "_apply_inference_to_current",
        lambda self, result, replace_auto: applied.append(result.index) or True,
        raising=False)

    editor._on_result_ready(0, _result(0, "a.png", []))
    assert applied == [0], applied

    editor._on_batch_finished({0: _result(0, "a.png", []),
                               1: _result(1, "b.png", [])})
    assert applied == [0], f"收尾又处理了一次当前图: {applied}"


def test_result_ready_updates_memory_cache_for_non_current_image(monkeypatch):
    """非当前图写盘后，内存缓存也要跟着更新。

    用户报的 bug：一键推理中途取消，圆点变绿了（盘上真有框），但切到那张
    图却还是旧标注。原因是逐张写盘只动了磁盘与圆点，`detection_boxes_dict`
    里仍是跑批之前的旧框；而切图时的 `_load_detection_boxes_for_index`
    只要发现缓存非空就直接用缓存，于是盘上的新结果永远看不到。
    """
    from labelforge.engine import inference_task

    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: True)

    editor = _FakeEditor(images=("a.png", "b.png"))
    # 第 1 张缓存里有旧框（非空 → 会盖过盘上的新内容，正是 bug 的前提）
    editor.detection_boxes_dict = {
        1: [{"label": "旧的", "x": 1, "y": 1, "width": 5, "height": 5}]
    }

    editor._on_result_ready(1, _result(
        1, "b.png", [{"label": "新", "x": 10, "y": 10, "width": 20,
                      "height": 20}]))

    cached = editor.detection_boxes_dict[1]
    labels = sorted(b["label"] for b in cached)
    assert labels == ["新", "旧的"], f"缓存没并进新结果：{labels}"


def test_result_ready_cache_sync_honours_replace_mode(monkeypatch):
    """覆盖模式下缓存要与盘上一致：旧推理框该删就删。"""
    from labelforge.engine import inference_task

    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: True)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor._is_delete_view = False
    # 旧框是上一轮推理留下的（flags.inference），覆盖模式应清掉
    editor.detection_boxes_dict = {
        1: [{"label": "旧推理", "x": 1, "y": 1, "width": 5, "height": 5,
             "flags": {"inference": True}}]
    }
    # 生产代码读的是 `_replace_mode()`（它再去问 model_row），所以这里
    # 换掉 `_replace_mode` 本身，而不是它下游的 `replace_mode`。
    editor._replace_mode = lambda: True

    editor._on_result_ready(1, _result(
        1, "b.png", [{"label": "新", "x": 10, "y": 10, "width": 20,
                      "height": 20}]))

    labels = sorted(b["label"] for b in editor.detection_boxes_dict[1])
    assert labels == ["新"], f"覆盖模式没清掉旧推理框：{labels}"


def test_result_ready_cache_sync_honours_class_filter(monkeypatch):
    """类别筛选也要作用到缓存，否则切回去会看到被筛掉的类别。"""
    from labelforge.engine import inference_task

    monkeypatch.setattr(inference_task, "write_one_result",
                        lambda *a, **kw: True)

    editor = _FakeEditor(images=("a.png", "b.png"))
    editor.detection_boxes_dict = {
        1: [{"label": "不要的", "x": 1, "y": 1, "width": 5, "height": 5,
             "flags": {"inference": True}}]
    }
    editor._filter_for_run = lambda: {"新"}

    editor._on_result_ready(1, _result(
        1, "b.png", [{"label": "新", "x": 10, "y": 10, "width": 20,
                      "height": 20}]))

    labels = sorted(b["label"] for b in editor.detection_boxes_dict[1])
    assert labels == ["新"], f"缓存没按类别筛选：{labels}"
