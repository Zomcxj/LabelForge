"""推理任务层测试：配置校验、类别表、结果合并、批量执行。

用假 detector / 假 loader，不依赖 onnxruntime 与 Qt。
"""
import numpy as np
import pytest

from pastelabel.engine.inference_task import (
    ImageResult, build_detector, failure_summary, filter_by_class,
    is_auto_box, mark_auto_boxes, merge_boxes, model_class_names,
    normalize_settings, run_batch, validate_settings,
)
from pastelabel.engine.onnx_inference import InferenceError


class _FakeSession:
    def __init__(self, meta=None):
        self._meta = meta or {}

    def get_modelmeta(self):
        return type("Meta", (), {"custom_metadata_map": self._meta})()


class _FakeDetector:
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or {}
        self.error = error
        self.calls = []

    def detect(self, rgb, resize_fn):
        self.calls.append(rgb.shape)
        if self.error:
            raise InferenceError(self.error)
        return [dict(b) for b in self.outputs.get(rgb.shape[0], [])]


# ---------- 配置规整 ----------

def test_normalize_settings_fills_defaults():
    s = normalize_settings({})
    assert s["output_format"] == "auto"
    assert s["conf_threshold"] == 0.25
    assert s["iou_threshold"] == 0.45
    assert s["input_size"] is None
    assert s["class_filter"] == []


def test_normalize_settings_clamps_thresholds():
    s = normalize_settings({"conf_threshold": 5, "iou_threshold": -1})
    assert s["conf_threshold"] == 1.0
    assert s["iou_threshold"] == 0.05


def test_normalize_settings_rejects_unknown_format_and_bad_size():
    s = normalize_settings({"output_format": "v9", "input_size": (0, 640)})
    assert s["output_format"] == "auto"
    assert s["input_size"] is None


def test_normalize_settings_keeps_valid_size_and_format():
    s = normalize_settings({"output_format": "V8", "input_size": [640, 480]})
    assert s["output_format"] == "v8"
    assert s["input_size"] == (640, 480)


def test_normalize_settings_coerces_numeric_strings():
    s = normalize_settings({"conf_threshold": "0.5", "input_size": ["320", "320"]})
    assert s["conf_threshold"] == 0.5
    assert s["input_size"] == (320, 320)


# ---------- 校验 ----------

def test_validate_requires_model_path():
    assert validate_settings({}) == "请选择 ONNX 模型文件"


def test_validate_rejects_missing_model_file(tmp_path):
    err = validate_settings({"model_path": str(tmp_path / "nope.onnx")})
    assert "不存在" in err


def test_validate_rejects_non_onnx_extension(tmp_path):
    bad = tmp_path / "m.pt"
    bad.write_bytes(b"x")
    err = validate_settings({"model_path": str(bad)})
    assert ".onnx" in err


def test_validate_rejects_missing_class_file(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    err = validate_settings({
        "model_path": str(model),
        "class_names_path": str(tmp_path / "nope.txt"),
    })
    assert "类别文件不存在" in err


def test_validate_passes_for_existing_files(tmp_path):
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    names = tmp_path / "c.txt"
    names.write_text("cat\ndog\n", encoding="utf-8")
    assert validate_settings({
        "model_path": str(model), "class_names_path": str(names)}) is None


# ---------- 类别表 ----------

def test_model_class_names_parses_dict_metadata_sorted_by_id():
    session = _FakeSession({"names": "{1: 'dog', 0: 'cat', 10: 'bird'}"})
    assert model_class_names(session) == ["cat", "dog", "bird"]


def test_model_class_names_parses_list_metadata():
    session = _FakeSession({"names": "['cat', 'dog']"})
    assert model_class_names(session) == ["cat", "dog"]


def test_model_class_names_returns_empty_for_garbage():
    assert model_class_names(_FakeSession({"names": "not-a-literal"})) == []
    assert model_class_names(_FakeSession({})) == []
    assert model_class_names(_FakeSession({"names": "{'a': 'b', 'x': 'y'}"})) == []


def test_model_class_names_tolerates_broken_session():
    class Broken:
        def get_modelmeta(self):
            raise RuntimeError("boom")
    assert model_class_names(Broken()) == []


# ---------- 类别过滤 ----------

def test_filter_by_class_empty_filter_passes_all():
    boxes = [{"label": "cat"}, {"label": "dog"}]
    assert filter_by_class(boxes, None) == boxes
    assert filter_by_class(boxes, []) == boxes


def test_filter_by_class_keeps_only_allowed():
    boxes = [{"label": "cat"}, {"label": "dog"}, {"label": "car"}]
    assert [b["label"] for b in filter_by_class(boxes, ["cat", "car"])] == ["cat", "car"]


# ---------- 合并与来源标记 ----------

def test_merge_boxes_keeps_manual_annotations():
    manual = [{"label": "cat", "x": 0, "y": 0, "width": 5, "height": 5}]
    merged = merge_boxes(manual, [{"label": "dog", "x": 1, "y": 1, "width": 5, "height": 5}])
    assert [b["label"] for b in merged] == ["cat", "dog"]


def test_merge_boxes_marks_inference_source_without_paste_flag():
    merged = merge_boxes([], [{"label": "dog", "x": 1, "y": 1, "width": 5, "height": 5}])
    flags = merged[0]["flags"]
    assert flags == {"inference": True}
    assert "paste" not in flags


def test_merge_boxes_preserves_existing_flags_and_group_id():
    detected = [{"label": "dog", "x": 1, "y": 1, "width": 5, "height": 5,
                 "group_id": 7, "flags": {"tiny_keep": True}}]
    merged = merge_boxes([], detected)
    assert merged[0]["flags"] == {"tiny_keep": True, "inference": True}
    assert merged[0]["group_id"] == 7


def test_merge_boxes_replace_auto_removes_previous_inference_only():
    manual = [{"label": "cat", "x": 0, "y": 0, "width": 5, "height": 5}]
    old_auto = [{"label": "dog", "x": 9, "y": 9, "width": 5, "height": 5}]
    existing = merge_boxes(manual, old_auto)

    merged = merge_boxes(existing, [{"label": "car", "x": 2, "y": 2, "width": 5, "height": 5}],
                         replace_auto=True)

    assert [b["label"] for b in merged] == ["cat", "car"]
    assert all(b["label"] != "dog" for b in merged)


def test_merge_boxes_without_replace_keeps_previous_inference():
    existing = merge_boxes([], [{"label": "dog", "x": 9, "y": 9, "width": 5, "height": 5}])
    merged = merge_boxes(existing, [{"label": "car", "x": 2, "y": 2, "width": 5, "height": 5}])
    assert [b["label"] for b in merged] == ["dog", "car"]


def test_is_auto_box_detects_only_inference_flag():
    assert is_auto_box({"flags": {"inference": True}})
    assert not is_auto_box({"flags": {"paste": True}})
    assert not is_auto_box({"flags": {}})
    assert not is_auto_box({})


def test_mark_auto_boxes_does_not_mutate_input():
    original = {"label": "cat", "flags": {"keep": 1}}
    marked = mark_auto_boxes([original])
    assert original["flags"] == {"keep": 1}
    assert marked[0]["flags"] == {"keep": 1, "inference": True}


# ---------- 批量执行 ----------

def _loader(shape_by_path):
    def load(path):
        shape = shape_by_path.get(path)
        if shape is None:
            return None
        return np.zeros(shape, dtype=np.uint8)
    return load


def test_run_batch_maps_results_by_index():
    detector = _FakeDetector({10: [{"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2}]})
    paths = ["a.png", "b.png"]
    results = run_batch(paths, detector, _loader({"a.png": (10, 10, 3), "b.png": (20, 20, 3)}),
                        lambda *a: None)
    assert set(results) == {0, 1}
    assert [b["label"] for b in results[0].boxes] == ["cat"]
    assert results[1].boxes == []
    assert results[1].error is None


def test_run_batch_records_unreadable_image_as_failure():
    detector = _FakeDetector()
    results = run_batch(["missing.png"], detector, _loader({}), lambda *a: None)
    assert results[0].boxes == []
    assert results[0].error
    assert detector.calls == []


def test_run_batch_records_detector_error_without_aborting():
    paths = ["a.png", "b.png"]
    detector = _FakeDetector(error="模型炸了")
    results = run_batch(paths, detector, _loader({"a.png": (4, 4, 3), "b.png": (4, 4, 3)}),
                        lambda *a: None)
    assert all(r.error == "模型炸了" for r in results.values())
    assert len(detector.calls) == 2


def test_run_batch_stops_on_interrupt_and_keeps_finished():
    paths = ["a.png", "b.png", "c.png"]
    detector = _FakeDetector()
    state = {"n": 0}

    def interrupted():
        state["n"] += 1
        return state["n"] > 2

    results = run_batch(paths, detector, _loader({"a.png": (4, 4, 3)}),
                        lambda *a: None, is_interrupted=interrupted)

    assert set(results) == {0, 1}
    assert len(detector.calls) == 1


def test_run_batch_reports_progress_for_every_image():
    paths = ["a.png", "b.png"]
    seen = []
    run_batch(paths, _FakeDetector(), _loader({"a.png": (4, 4, 3), "b.png": (4, 4, 3)}),
              lambda *a: None, progress_fn=lambda done, total, name: seen.append((done, total, name)))
    assert seen == [(1, 2, "a.png"), (2, 2, "b.png")]


def test_run_batch_applies_class_filter_to_detections():
    detector = _FakeDetector({4: [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2},
        {"label": "dog", "x": 1, "y": 1, "width": 2, "height": 2},
    ]})
    results = run_batch(["a.png"], detector, _loader({"a.png": (4, 4, 3)}),
                        lambda *a: None, class_filter=["dog"])
    assert [b["label"] for b in results[0].boxes] == ["dog"]


def test_run_batch_distinguishes_empty_result_from_missing():
    """空结果（error=None）与失败必须区分开。"""
    results = run_batch(["a.png", "b.png"], _FakeDetector(),
                        _loader({"a.png": (4, 4, 3)}), lambda *a: None)
    assert results[0].boxes == [] and results[0].error is None
    assert (results[0].width, results[0].height) == (4, 4)
    assert results[1].error


def test_run_batch_uses_dataset_keys_when_given():
    """批量结果按数据集索引归档，写入内存/磁盘时才不会串图。"""
    results = run_batch(["a.png", "b.png"], _FakeDetector(),
                        _loader({"a.png": (4, 4, 3), "b.png": (4, 4, 3)}),
                        lambda *a: None, keys=[7, 9])
    assert set(results) == {7, 9}
    assert results[7].index == 7 and results[7].path == "a.png"
    assert results[9].path == "b.png"


def test_failure_summary_empty_when_all_ok():
    results = {0: ImageResult(0, "a.png", [], None)}
    assert failure_summary(results) == ""


def test_failure_summary_mentions_count_and_first_reason():
    results = {
        0: ImageResult(0, "a.png", [], "坏文件"),
        1: ImageResult(1, "b.png", [], "坏文件"),
    }
    summary = failure_summary(results)
    assert "2 张失败" in summary
    assert "a.png" in summary and "坏文件" in summary


# ---------- 装配 ----------

def test_build_detector_uses_class_file_over_metadata(monkeypatch, tmp_path):
    import pastelabel.engine.inference_task as task
    from pastelabel.engine.onnx_inference import OnnxDetector

    class FakeDetector(OnnxDetector):
        def _ensure_session(self):
            self._input_name = "images"
            return object()
    monkeypatch.setattr(task, "OnnxDetector", FakeDetector)
    names = tmp_path / "c.txt"
    names.write_text("cat\ndog\n", encoding="utf-8")

    detector = task.build_detector({"model_path": "m.onnx", "class_names_path": str(names)})

    assert detector.class_names == ["cat", "dog"]


def test_build_detector_reads_names_from_file(tmp_path):
    names = tmp_path / "c.txt"
    names.write_text("cat\n\ndog\n", encoding="utf-8")
    from pastelabel.engine.onnx_inference import read_class_names
    assert read_class_names(str(names)) == ["cat", "dog"]


def test_build_detector_falls_back_to_model_metadata(monkeypatch):
    import pastelabel.engine.inference_task as task
    from pastelabel.engine.onnx_inference import OnnxDetector

    class FakeDetector(OnnxDetector):
        def _ensure_session(self):
            self._input_name = "images"
            return _FakeSession({"names": "{0: 'cat', 1: 'dog'}"})
    monkeypatch.setattr(task, "OnnxDetector", FakeDetector)

    detector = task.build_detector({"model_path": "m.onnx"})

    assert detector.class_names == ["cat", "dog"]


# ---------- sidecar 读写与批量写入 ----------

def _write_sidecar(tmp_path, stem, shapes, width=100, height=80, extra=None):
    import json
    img = tmp_path / f"{stem}.png"
    img.write_bytes(b"x")
    payload = {
        "version": "5.0.1", "flags": {"note": "keep"}, "imageData": None,
        "imagePath": f"{stem}.png", "imageHeight": height, "imageWidth": width,
        "shapes": shapes,
    }
    if extra:
        payload.update(extra)
    (tmp_path / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(img)


def test_read_sidecar_boxes_returns_none_when_missing(tmp_path):
    from pastelabel.engine.inference_task import read_sidecar_boxes
    img = tmp_path / "none.png"
    img.write_bytes(b"x")
    assert read_sidecar_boxes(str(img)) is None


def test_read_sidecar_boxes_returns_empty_list_for_empty_shapes(tmp_path):
    """空标注（shapes 为空）与缺文件是两种状态，不能都返回 None。"""
    from pastelabel.engine.inference_task import read_sidecar_boxes
    img = _write_sidecar(tmp_path, "empty", [])
    assert read_sidecar_boxes(img) == []


def test_read_sidecar_boxes_converts_shapes(tmp_path):
    from pastelabel.engine.inference_task import read_sidecar_boxes
    img = _write_sidecar(tmp_path, "one", [
        {"label": "cat", "points": [[10, 20], [40, 20], [40, 60], [10, 60]],
         "shape_type": "rectangle", "flags": {"paste": True}, "group_id": 3},
    ])
    boxes = read_sidecar_boxes(img)
    assert len(boxes) == 1
    assert (boxes[0]["x"], boxes[0]["y"]) == (10, 20)
    assert (boxes[0]["width"], boxes[0]["height"]) == (30, 40)
    assert boxes[0]["group_id"] == 3
    assert boxes[0]["is_paste"] is True


def test_read_sidecar_boxes_tolerates_corrupt_json(tmp_path):
    from pastelabel.engine.inference_task import read_sidecar_boxes
    img = tmp_path / "bad.png"
    img.write_bytes(b"x")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert read_sidecar_boxes(str(img)) is None


def test_write_sidecar_boxes_preserves_other_fields(tmp_path):
    from pastelabel.engine.inference_task import write_sidecar_boxes
    img = _write_sidecar(tmp_path, "keep", [])
    assert write_sidecar_boxes(img, [
        {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4}], 100, 80)
    import json
    data = json.loads((tmp_path / "keep.json").read_text(encoding="utf-8"))
    assert data["version"] == "5.0.1"
    assert data["flags"] == {"note": "keep"}
    assert data["shapes"][0]["points"] == [[1, 2], [4, 2], [4, 6], [1, 6]]


def test_write_sidecar_boxes_creates_file_when_missing(tmp_path):
    from pastelabel.engine.inference_task import write_sidecar_boxes
    img = tmp_path / "fresh.png"
    img.write_bytes(b"x")
    assert write_sidecar_boxes(img, [], 64, 48)
    import json
    data = json.loads((tmp_path / "fresh.json").read_text(encoding="utf-8"))
    assert data["shapes"] == []
    assert (data["imageWidth"], data["imageHeight"]) == (64, 48)
    assert data["imagePath"] == "fresh.png"


def test_write_sidecar_boxes_keeps_inference_flags(tmp_path):
    """推理来源标记必须落盘，重载后才能识别上一轮自动结果。"""
    from pastelabel.engine.inference_task import write_sidecar_boxes
    img = tmp_path / "flag.png"
    img.write_bytes(b"x")
    write_sidecar_boxes(img, [
        {"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1,
         "flags": {"inference": True}}], 10, 10)
    import json
    data = json.loads((tmp_path / "flag.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["flags"] == {"inference": True}
    assert "paste" not in data["shapes"][0]["flags"]


def test_write_results_merges_with_disk_boxes(tmp_path):
    from pastelabel.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "merge", [
        {"label": "manual", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
         "shape_type": "rectangle", "flags": {}},
    ])
    results = {0: ImageResult(0, img, [
        {"label": "auto", "x": 10, "y": 10, "width": 5, "height": 5}], None, 100, 80)}

    written, failed = write_results(results)

    assert (written, failed) == (1, [])
    import json
    labels = [s["label"] for s in
              json.loads((tmp_path / "merge.json").read_text(encoding="utf-8"))["shapes"]]
    assert labels == ["manual", "auto"]


def test_write_results_prefers_memory_over_disk(tmp_path):
    """已加载图以内存为准：磁盘滞后时不能把内存里的新标注当成不存在。"""
    from pastelabel.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "mem", [
        {"label": "stale", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
         "shape_type": "rectangle", "flags": {}},
    ])
    memory = {0: [{"label": "fresh", "x": 0, "y": 0, "width": 5, "height": 5}]}
    results = {0: ImageResult(0, img, [], None, 100, 80)}

    write_results(results, memory_boxes=memory)

    import json
    labels = [s["label"] for s in
              json.loads((tmp_path / "mem.json").read_text(encoding="utf-8"))["shapes"]]
    assert labels == ["fresh"]


def test_write_results_skips_current_index_and_failures(tmp_path):
    from pastelabel.engine.inference_task import ImageResult, write_results
    ok = _write_sidecar(tmp_path, "ok", [])
    bad = _write_sidecar(tmp_path, "bad", [])
    results = {
        0: ImageResult(0, ok, [], None, 10, 10),
        1: ImageResult(1, bad, [], "模型炸了", 10, 10),
    }

    written, failed = write_results(results, skip_index=0)

    assert (written, failed) == (0, [])


def test_write_results_reports_failure_paths(tmp_path, monkeypatch):
    from pastelabel.engine import inference_task
    from pastelabel.engine.inference_task import ImageResult, write_results
    img = tmp_path / "fail.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(inference_task, "write_sidecar_boxes",
                        lambda *a, **kw: False)
    results = {0: ImageResult(0, str(img), [], None, 10, 10)}

    written, failed = write_results(results)

    assert written == 0
    assert failed == [str(img)]


def test_write_results_replace_auto_drops_previous_inference(tmp_path):
    from pastelabel.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "repl", [
        {"label": "old", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
         "shape_type": "rectangle", "flags": {"inference": True}},
        {"label": "manual", "points": [[9, 9], [12, 9], [12, 12], [9, 12]],
         "shape_type": "rectangle", "flags": {}},
    ])
    results = {0: ImageResult(0, img, [
        {"label": "new", "x": 1, "y": 1, "width": 2, "height": 2}], None, 100, 80)}

    write_results(results, replace_auto=True)

    import json
    labels = [s["label"] for s in
              json.loads((tmp_path / "repl.json").read_text(encoding="utf-8"))["shapes"]]
    assert labels == ["manual", "new"]


def test_write_results_preserves_paste_shapes(tmp_path):
    """贴图在 sidecar 里靠 flags.paste 区分，写结果不能把它当检测框删掉。"""
    from pastelabel.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "paste", [
        {"label": "logo", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
         "shape_type": "rectangle", "flags": {"paste": True}},
    ])
    results = {0: ImageResult(0, img, [], None, 100, 80)}

    write_results(results)

    import json
    shapes = json.loads((tmp_path / "paste.json").read_text(encoding="utf-8"))["shapes"]
    assert [s["label"] for s in shapes] == ["logo"]
    assert shapes[0]["flags"] == {"paste": True}
