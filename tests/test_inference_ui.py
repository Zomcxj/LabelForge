"""推理 UI 接线回归测试。

约定（同 test_dataset_tools_ui / test_ui_regressions）：
- 不实例化 Qt 窗口；源码字符串 / AST / i18n 字典断言 + monkeypatch 派发。
- 结果应用的行为用假 editor 直接调 mixin 方法验证。
"""
import ast
import inspect
from pathlib import Path

import pytest

from pastelabel.ui import inference_dialog as dialog_mod
from pastelabel.ui import i18n
from pastelabel.ui.inference_dialog import InferenceDialog
from pastelabel.ui.main_window import ImageEditor
from pastelabel.ui.mixins.inference import InferenceMixin
from pastelabel.ui.mixins.options_popup import OptionsPopupMixin
from pastelabel.ui.mixins.translation import TranslationMixin

ROOT = Path(__file__).resolve().parents[1]

INFERENCE_KEYS = (
    "推理", "模型推理", "模型", "加载模型", "尚未加载模型", "模型加载失败",
    "输入尺寸", "输出格式:", "类别数", "置信度:", "设备:",
    "当前图片", "全部图片", "开始推理", "推理中", "推理完成", "推理失败",
    "替换上次推理结果", "人工标注始终保留", "类别筛选（不勾选表示全部）:",
    "未安装 onnxruntime，无法执行推理", "数据集已切换，推理结果已丢弃",
    "张图", "个框", "张已写入", "张写入失败", "浏览...", "设备", "输出格式",
)
def _options_popup_src():
    return inspect.getsource(OptionsPopupMixin._create_options_menu)


# ---------- 入口按钮 ----------

def test_inference_btn_created_between_lint_and_format():
    """推理按钮放在「质检」与「格式」之间，同款 optionsBtn 样式。"""
    src = _options_popup_src()
    lint_idx = src.index("self.lint_btn = QPushButton")
    inf_idx = src.index("self.inference_btn = QPushButton")
    fmt_idx = src.index("self.format_btn = QPushButton")
    assert lint_idx < inf_idx < fmt_idx
    assert 'self.inference_btn = QPushButton(tr("推理"))' in src
    assert "self.inference_btn.clicked.connect(self._open_inference)" in src


def test_inference_btn_text_refreshed_on_language_switch():
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "inference_btn" in src
    assert 'tr("推理")' in src


# ---------- mixin 注册 ----------

def test_inference_mixin_registered_on_image_editor():
    assert InferenceMixin in ImageEditor.__mro__
    assert callable(getattr(ImageEditor, "_open_inference"))
    assert callable(getattr(ImageEditor, "_apply_inference_results"))


def test_inference_mixin_contract():
    for name in ("_open_inference", "_apply_inference_results",
                 "_apply_inference_to_current", "_close_inference_dialog",
                 "_inference_dataset_changed"):
        assert hasattr(InferenceMixin, name), name


def test_dialog_exposes_expected_api():
    for name in ("set_images", "settings", "selected_classes",
                 "apply_requested", "showEvent", "closeEvent"):
        assert hasattr(InferenceDialog, name), name


def test_dialog_uses_background_threads_not_blocking_calls():
    """模型加载与推理都必须走线程，否则大模型会冻住界面。"""
    src = (ROOT / "pastelabel" / "ui" / "inference_dialog.py").read_text(encoding="utf-8")
    assert "QThread" in src
    assert "worker.start()" in src
    assert "detector.detect(" not in src, "推理只能在 worker 里执行"


def test_dialog_does_not_write_annotations_itself():
    """弹窗只发信号；标注写入由主窗口一次性完成。"""
    src = (ROOT / "pastelabel" / "ui" / "inference_dialog.py").read_text(encoding="utf-8")
    assert "apply_requested.emit" in src
    for forbidden in ("_create_detection_box", "save_json(", "update_label_list("):
        assert forbidden not in src, forbidden


# ---------- i18n ----------

def test_i18n_inference_terms():
    for key in INFERENCE_KEYS:
        assert key in i18n._strings["zh"], key
        assert key in i18n._strings["en"], key


def test_every_tr_key_in_inference_code_is_translated():
    """推理相关代码里用到的每个 tr() 键都必须在两个语言表里有定义。

    tr() 取不到键时回退成键本身，所以漏译不会报错，只会在英文界面漏出中文；
    这条测试把「键存在」变成显式契约。
    """
    import re
    missing = []
    for rel in ("pastelabel/ui/inference_dialog.py", "pastelabel/ui/mixins/inference.py"):
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
    tree = ast.parse((ROOT / "pastelabel/ui/i18n.py").read_text(encoding="utf-8"))
    dicts = [n for n in ast.walk(tree)
             if isinstance(n, ast.Dict) and len(n.keys) > 100]
    assert len(dicts) == 2, "i18n 里应有 zh / en 两个语言表"
    for node in dicts:
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        dupes = {k for k, c in collections.Counter(keys).items() if c > 1}
        shadowed = sorted(dupes & set(INFERENCE_KEYS))
        assert not shadowed, shadowed


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
    editor._apply_inference_results(
        {0: _result(0, "a.png", [])}, False)
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


def test_apply_results_writes_other_images_to_disk(tmp_path, monkeypatch):
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


# ---------- 派发 ----------


def test_open_inference_dispatches_dialog(monkeypatch):
    calls = {}

    class _Signal:
        def __init__(self):
            self.slots = []

        def connect(self, fn):
            self.slots.append(fn)

        def disconnect(self):
            self.slots.clear()

    class FakeDialog:
        def __init__(self, parent=None, settings=None):
            calls["parent"] = parent
            calls["settings"] = settings
            self.apply_requested = _Signal()
            self.rejected = _Signal()

        def set_images(self, entries, current_index=-1):
            calls["entries"] = entries
            calls["current_index"] = current_index

        def show(self):
            calls["shown"] = True

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(dialog_mod, "InferenceDialog", FakeDialog)

    class Owner(InferenceMixin):
        _busy = False

        def __init__(self):
            self.background_images = ["a.png", "b.png"]
            self.current_background_index = 1
            self.status_label = type("L", (), {
                "setText": lambda self, t: setattr(self, "text", t), "text": ""})()
            self._inference_dialog = None
            self._inference_dataset = None

    owner = Owner()
    owner._open_inference()

    assert calls["shown"] is True
    assert calls["entries"] == [(0, "a.png"), (1, "b.png")]
    assert calls["current_index"] == 1
    assert owner._inference_dataset == ("a.png", "b.png")


def test_open_inference_requires_dataset():
    class Owner(InferenceMixin):
        _busy = False

        def __init__(self):
            self.background_images = []
            self.status_label = type("L", (), {
                "setText": lambda self, t: setattr(self, "text", t), "text": ""})()

    owner = Owner()
    owner._open_inference()

    assert "请先加载数据集" in owner.status_label.text


def test_open_inference_refuses_while_busy():
    class Owner(InferenceMixin):
        _busy = True

        def __init__(self):
            self.background_images = ["a.png"]
            self.status_label = type("L", (), {
                "setText": lambda self, t: setattr(self, "text", t), "text": ""})()

    owner = Owner()
    owner._open_inference()

    assert getattr(owner, "_inference_dialog", None) is None
    assert owner.status_label.text == ""


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


def test_dialog_settings_returns_normalizable_dict():
    """settings() 的输出必须能直接被持久化与规整，不依赖窗口实例。"""
    tree = ast.parse(
        (ROOT / "pastelabel" / "ui" / "inference_dialog.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "InferenceDialog")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "settings")
    src = ast.get_source_segment(
        (ROOT / "pastelabel" / "ui" / "inference_dialog.py").read_text(encoding="utf-8"), fn)
    for key in ("model_path", "class_names_path", "output_format",
                "conf_threshold", "iou_threshold", "input_size", "providers"):
        assert f'"{key}"' in src, key


def test_run_batch_keys_follow_dataset_index():
    """推理 worker 必须把数据集索引传给 run_batch，否则结果会写错图。"""
    src = inspect.getsource(dialog_mod._InferenceWorker.run)
    assert "keys=self._keys" in src


# ---------- 当前图：无检出不写盘 ----------

def test_current_image_without_detection_skips_save(monkeypatch, tmp_path):
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


def test_current_image_without_detection_keeps_manual_boxes(monkeypatch, tmp_path):
    """当前图有人工标注时照常保存，不能因为没检出就不写。"""
    editor = _FakeEditor(images=(str(tmp_path / "a.png"),))

    editor._apply_inference_to_current(_result(0, editor.background_images[0], []), False)

    assert len(editor.saved) == 1
    assert editor.detection_boxes[0]["label"] == "manual"


def test_current_image_without_detection_saves_when_paste_shapes_exist(monkeypatch, tmp_path):
    """画布上有贴图时仍要写盘，否则贴图状态会丢。"""
    editor = _FakeEditor(images=(str(tmp_path / "a.png"),))
    editor.detection_boxes = []
    editor.detection_boxes_dict = {0: []}
    editor.canvas_items = [(None, object(), "pasted")]

    editor._apply_inference_to_current(_result(0, editor.background_images[0], []), False)

    assert len(editor.saved) == 1
