"""推理任务层测试：配置校验、类别表、结果合并、批量执行。

用假 detector / 假 loader，不依赖 onnxruntime 与 Qt。
"""
import os
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from labelforge.engine.inference_task import (
    ImageResult, build_detector, failure_summary, filter_by_class,
    is_auto_box, mark_auto_boxes, merge_boxes, model_class_names,
    normalize_settings, run_batch, validate_settings,
)
from labelforge.engine.onnx_inference import (
    DEFAULT_CONF, DEFAULT_IOU, InferenceError)


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
    assert s["conf_threshold"] == DEFAULT_CONF == 0.25
    assert s["iou_threshold"] == DEFAULT_IOU == 0.45
    assert s["input_size"] is None
    assert s["class_filter"] == []


def test_defaults_are_not_duplicated_elsewhere():
    """阈值只在 onnx_inference 定义一份，别处一律引用，避免界面与执行对不上。"""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for py in (root / "labelforge").rglob("*.py"):
        if py.name == "onnx_inference.py":
            continue
        for lineno, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"(conf_threshold|iou_threshold)\s*=\s*0\.(25|45)", line):
                offenders.append(f"{py.relative_to(root)}:{lineno}: {line.strip()}")
    assert not offenders, "阈值被写死在别处:" + "; ".join(offenders)


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

def test_filter_by_class_none_passes_all():
    """None = 不过滤（界面默认全选走这条）。"""
    boxes = [{"label": "cat"}, {"label": "dog"}]
    assert filter_by_class(boxes, None) == boxes


def test_filter_by_class_empty_set_discards_everything():
    """空集合 = 全不选，结果一定为空；不能再当成「不过滤」。"""
    boxes = [{"label": "cat"}, {"label": "dog"}]
    assert filter_by_class(boxes, set()) == []
    assert filter_by_class(boxes, []) == []


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


def test_merge_boxes_replace_auto_removes_all_detection_boxes():
    """覆盖 = 本图检测框完全重来：人工框与上一轮推理框一起清。

    用户报的是「手动绘制的没有覆盖，应该都覆盖」——早先只清推理框，
    手工框会与本轮结果叠在一起，看着像框重复。
    """
    manual = [{"label": "cat", "x": 0, "y": 0, "width": 5, "height": 5}]
    old_auto = [{"label": "dog", "x": 9, "y": 9, "width": 5, "height": 5}]
    existing = merge_boxes(manual, old_auto)

    merged = merge_boxes(existing, [{"label": "car", "x": 2, "y": 2, "width": 5, "height": 5}],
                         replace_auto=True)

    assert [b["label"] for b in merged] == ["car"], (
        "覆盖后应只剩本轮结果")
    assert all(b["label"] != "dog" for b in merged)


def test_merge_boxes_replace_auto_leaves_paste_shapes_alone():
    """贴图不是检测框：覆盖不清它（用户明确「只清检测框，贴图保留」）。

    贴图单独存进 json，丢了就是丢用户的素材。`existing` 来自 sidecar，里面
    混着贴图条目，覆盖时它要原样留回来。
    """
    merged = merge_boxes(
        [{"label": "logo", "x": 1, "y": 1, "width": 3, "height": 3,
          "flags": {"paste": True}},
         {"label": "cat", "x": 4, "y": 4, "width": 3, "height": 3}],
        [{"label": "car", "x": 2, "y": 2, "width": 5, "height": 5}],
        replace_auto=True)

    labels = [b["label"] for b in merged]
    assert "logo" in labels, "覆盖把贴图删掉了"
    assert "cat" not in labels, "覆盖没清掉人工检测框"
    assert "car" in labels, "覆盖后缺少本轮结果"


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
              lambda *a: None,
              progress_fn=lambda done, total, name, boxes, ms: seen.append(
                  (done, total, name, boxes, ms)))
    assert [(d, t, n) for d, t, n, _b, _m in seen] == [
        (1, 2, "a.png"), (2, 2, "b.png")]


def test_run_batch_reports_box_count_and_duration_per_image():
    """进度回调要带上这一张自己的框数与耗时，界面不用回查结果。"""
    detector = _FakeDetector({4: [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2},
        {"label": "dog", "x": 1, "y": 1, "width": 2, "height": 2},
    ]})
    seen = []
    run_batch(["a.png"], detector, _loader({"a.png": (4, 4, 3)}), lambda *a: None,
              progress_fn=lambda done, total, name, boxes, ms: seen.append((boxes, ms)))
    (boxes, ms), = seen
    assert boxes == 2, "应上报这一张的框数"
    assert isinstance(ms, int) and ms >= 0, ms


def test_run_batch_reports_zero_boxes_for_failed_image():
    """失败图框数为 0，但耗时仍然上报，否则日志会缺一行。"""
    seen = []
    run_batch(["missing.png"], _FakeDetector(), _loader({}), lambda *a: None,
              progress_fn=lambda done, total, name, boxes, ms: seen.append((boxes, ms)))
    (boxes, ms), = seen
    assert boxes == 0
    assert isinstance(ms, int)


def test_run_batch_class_filter_counts_filtered_boxes():
    """上报的框数必须是筛选后的，不是检出总数。"""
    detector = _FakeDetector({4: [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2},
        {"label": "dog", "x": 1, "y": 1, "width": 2, "height": 2},
    ]})
    seen = []
    run_batch(["a.png"], detector, _loader({"a.png": (4, 4, 3)}), lambda *a: None,
              progress_fn=lambda done, total, name, boxes, ms: seen.append(boxes),
              class_filter=["dog"])
    assert seen == [1]


def test_beijing_timestamp_is_time_only_utc_plus_8_with_milliseconds():
    """北京时间固定 UTC+8，不跟本机时区；只给时刻、不给年月日；精确到毫秒。"""
    from datetime import datetime, timezone
    from labelforge.engine.inference_task import beijing_timestamp

    # 2024-01-01 00:00:00 UTC -> 北京时间 08:00:00.123
    moment = datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc)
    assert beijing_timestamp(moment) == "08:00:00.123"


def test_beijing_timestamp_has_no_date():
    """日志行不带年月日：2026-10-12 这种日期串不该出现在时间戳里。"""
    import re
    from datetime import datetime, timezone
    from labelforge.engine.inference_task import beijing_timestamp

    moment = datetime(2024, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    got = beijing_timestamp(moment)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", got), got
    assert len(got) == len("08:00:00.000"), got  # HH:MM:SS.mmm


def test_beijing_timestamp_ignores_the_local_timezone():
    """本机时区改成别的，时间戳也必须还是 UTC+8。"""
    import subprocess
    import sys

    script = (
        "import os, sys;"
        "os.environ['TZ'] = 'America/New_York';"
        "import time; time.tzset() if hasattr(time, 'tzset') else None;"
        "from labelforge.engine.inference_task import beijing_timestamp;"
        "print(beijing_timestamp())"
    )
    env = dict(os.environ, TZ="America/New_York", PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, env=env, timeout=60,
                         encoding="utf-8", errors="replace")
    assert out.returncode == 0, out.stderr
    # 与当场算出的 UTC+8 时刻只差几分钟，说明用的是固定偏移而不是本机时区
    # （比到秒会偶发跨秒失败：子进程本身要花时间启动）
    from labelforge.engine.inference_task import beijing_timestamp

    expected = beijing_timestamp()
    got = out.stdout.strip()
    assert got[:5] == expected[:5], (got, expected)


def test_beijing_timestamp_truncates_not_rounds():
    """毫秒是截断，避免 .9996 进位到下一秒。"""
    from datetime import datetime, timezone
    from labelforge.engine.inference_task import beijing_timestamp

    moment = datetime(2024, 1, 1, 0, 0, 0, 999999, tzinfo=timezone.utc)
    assert beijing_timestamp(moment).endswith(".999")


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
    import labelforge.engine.inference_task as task
    from labelforge.engine.onnx_inference import OnnxDetector

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
    from labelforge.engine.onnx_inference import read_class_names
    assert read_class_names(str(names)) == ["cat", "dog"]


def test_build_detector_falls_back_to_model_metadata(monkeypatch):
    """不给类别文件时，类别表应由 OnnxDetector 从模型元数据里取。

    Session 是懒加载的，所以要显式取一次 session 才会回落读元数据。
    """
    import labelforge.engine.inference_task as task
    from labelforge.engine.onnx_inference import OnnxDetector

    class FakeDetector(OnnxDetector):
        def _ensure_session(self):
            self._input_name = "images"
            session = _FakeSession({"names": "{0: 'cat', 1: 'dog'}"})
            if not self.class_names:
                from labelforge.engine.onnx_inference import metadata_class_names
                self.class_names = metadata_class_names(session)
            return session
    monkeypatch.setattr(task, "OnnxDetector", FakeDetector)

    detector = task.build_detector({"model_path": "m.onnx"})
    detector.session      # 触发懒加载

    assert detector.class_names == ["cat", "dog"]


def test_build_detector_prefers_explicit_class_file_over_metadata():
    """给了类别文件就用文件，不被模型元数据覆盖。"""
    import labelforge.engine.inference_task as task

    detector = task.build_detector(
        {"model_path": "model/yolov5n.onnx", "class_names_path": ""})
    assert detector.class_names == []      # 未加载 session 前为空，不预读元数据


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
    from labelforge.engine.inference_task import read_sidecar_boxes
    img = tmp_path / "none.png"
    img.write_bytes(b"x")
    assert read_sidecar_boxes(str(img)) is None


def test_read_sidecar_boxes_returns_empty_list_for_empty_shapes(tmp_path):
    """空标注（shapes 为空）与缺文件是两种状态，不能都返回 None。"""
    from labelforge.engine.inference_task import read_sidecar_boxes
    img = _write_sidecar(tmp_path, "empty", [])
    assert read_sidecar_boxes(img) == []


def test_read_sidecar_boxes_converts_shapes(tmp_path):
    from labelforge.engine.inference_task import read_sidecar_boxes
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
    from labelforge.engine.inference_task import read_sidecar_boxes
    img = tmp_path / "bad.png"
    img.write_bytes(b"x")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert read_sidecar_boxes(str(img)) is None


def test_write_sidecar_boxes_preserves_other_fields(tmp_path):
    from labelforge.engine.inference_task import write_sidecar_boxes
    img = _write_sidecar(tmp_path, "keep", [])
    assert write_sidecar_boxes(img, [
        {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4}], 100, 80)
    import json
    data = json.loads((tmp_path / "keep.json").read_text(encoding="utf-8"))
    assert data["version"] == "5.0.1"
    assert data["flags"] == {"note": "keep"}
    assert data["shapes"][0]["points"] == [[1, 2], [4, 2], [4, 6], [1, 6]]


def test_write_sidecar_boxes_creates_file_when_missing(tmp_path):
    from labelforge.engine.inference_task import write_sidecar_boxes
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
    from labelforge.engine.inference_task import write_sidecar_boxes
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
    from labelforge.engine.inference_task import ImageResult, write_results
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
    from labelforge.engine.inference_task import ImageResult, write_results
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
    from labelforge.engine.inference_task import ImageResult, write_results
    ok = _write_sidecar(tmp_path, "ok", [])
    bad = _write_sidecar(tmp_path, "bad", [])
    results = {
        0: ImageResult(0, ok, [], None, 10, 10),
        1: ImageResult(1, bad, [], "模型炸了", 10, 10),
    }

    written, failed = write_results(results, skip_index=0)

    assert (written, failed) == (0, [])


def test_write_results_reports_failure_paths(tmp_path, monkeypatch):
    """写盘失败要上报路径；用有检出的图，否则空结果根本不会尝试写盘。"""
    from labelforge.engine import inference_task
    from labelforge.engine.inference_task import ImageResult, write_results
    img = tmp_path / "fail.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(inference_task, "write_sidecar_boxes",
                        lambda *a, **kw: False)
    results = {0: ImageResult(0, str(img), [
        {"label": "car", "x": 1, "y": 1, "width": 5, "height": 5}], None, 10, 10)}

    written, failed = write_results(results)

    assert written == 0
    assert failed == [str(img)]


def test_write_results_replace_auto_drops_previous_inference(tmp_path):
    """覆盖：上一轮推理框与人工框都清掉，只留本轮结果。"""
    from labelforge.engine.inference_task import ImageResult, write_results
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
    assert labels == ["new"], labels


def test_write_results_preserves_paste_shapes(tmp_path):
    """贴图在 sidecar 里靠 flags.paste 区分，写结果不能把它当检测框删掉。"""
    from labelforge.engine.inference_task import ImageResult, write_results
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


# ---------- 无检出时不写盘 ----------

def _labels_of(image_path):
    """读 sidecar 里的标签列表。"""
    import json
    from pathlib import Path
    json_path = Path(image_path).with_suffix(".json")
    return [s["label"] for s in json.loads(json_path.read_text(encoding="utf-8"))["shapes"]]


def test_no_detection_does_not_create_sidecar(tmp_path):
    """无检出的图不新建 json：否则数据集里全是 shapes 为 [] 的空文件。"""
    from labelforge.engine.inference_task import ImageResult, write_results
    img = tmp_path / "miss.png"
    img.write_bytes(b"x")

    written, failed = write_results(
        {0: ImageResult(0, str(img), [], None, 100, 80)}, memory_boxes={})

    assert (written, failed) == (0, [])
    assert not (tmp_path / "miss.json").exists()


def test_no_detection_removes_preexisting_empty_sidecar(tmp_path):
    """原本就是空 shapes 的 json 顺手删掉，不留无意义文件。"""
    from labelforge.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "empty", [])

    written, _ = write_results(
        {0: ImageResult(0, img, [], None, 100, 80)}, memory_boxes={})

    assert written == 0
    assert not (tmp_path / "empty.json").exists()


def test_no_detection_keeps_manual_annotations(tmp_path):
    """无检出但有人工标注：必须原样保留，不能因为没结果就清掉。"""
    from labelforge.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "manual", [
        {"label": "manual", "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
         "shape_type": "rectangle", "flags": {}},
    ])

    written, failed = write_results(
        {0: ImageResult(0, img, [], None, 100, 80)}, memory_boxes={})

    assert (written, failed) == (1, [])
    assert _labels_of(img) == ["manual"]


def test_replace_auto_clears_previous_inference_boxes_and_drops_file(tmp_path):
    """勾选替换且无检出：上一轮推理框被清掉，清空后连 json 一起删。"""
    from labelforge.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "old", [
        {"label": "oldauto", "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
         "shape_type": "rectangle", "flags": {"inference": True}},
    ])

    written, failed = write_results(
        {0: ImageResult(0, img, [], None, 100, 80)},
        memory_boxes={}, replace_auto=True)

    assert (written, failed) == (0, [])
    assert not (tmp_path / "old.json").exists()


def test_replace_auto_also_clears_manual_boxes(tmp_path):
    """勾选替换：人工标注也被清掉，只剩本轮结果（用户明确要求「都覆盖」）。

    早先这里叫 `..._keeps_manual_and_drops_only_inference`，断言的是「人工框
    保留」。用户报的正是那条语义不合预期：「手动绘制的没有覆盖，应该都覆盖」
    ——手工框与本轮结果叠在一起看着像框重复。贴图不在检测框这层，见下一条
    贴图用例。
    """
    from labelforge.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "mix", [
        {"label": "oldauto", "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
         "shape_type": "rectangle", "flags": {"inference": True}},
        {"label": "manual", "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
         "shape_type": "rectangle", "flags": {}},
    ])

    written, failed = write_results(
        {0: ImageResult(0, img, [], None, 100, 80)},
        memory_boxes={}, replace_auto=True)

    # 全部被清、又没有新检出 → 连 json 一起删（与纯推理框那条一致）
    assert (written, failed) == (0, [])
    assert not (tmp_path / "mix.json").exists()


def test_paste_shapes_survive_inference_write(tmp_path):
    """贴图与多边形必须原样保留：推理只追加检测框。"""
    from labelforge.engine.inference_task import ImageResult, write_results
    img = _write_sidecar(tmp_path, "shapes", [
        {"label": "poly", "points": [[10, 10], [20, 10], [15, 20]],
         "shape_type": "polygon", "flags": {}},
        {"label": "pasted", "points": [[30, 30], [40, 30], [40, 40], [30, 40]],
         "shape_type": "rectangle", "flags": {"paste": True}},
        {"label": "tiny_keep", "points": [[50, 50], [55, 50], [55, 55], [50, 55]],
         "shape_type": "rectangle", "flags": {"tiny_keep": True}},
    ])

    write_results({0: ImageResult(0, img, [
        {"label": "car", "x": 100, "y": 10, "width": 20, "height": 20}],
        None, 100, 80)}, memory_boxes={})

    import json
    data = json.loads((tmp_path / "shapes.json").read_text(encoding="utf-8"))
    by_label = {s["label"]: s for s in data["shapes"]}
    assert set(by_label) == {"poly", "pasted", "tiny_keep", "car"}
    assert by_label["poly"]["shape_type"] == "polygon"
    assert by_label["pasted"]["flags"]["paste"] is True
    assert by_label["tiny_keep"]["flags"]["tiny_keep"] is True
    assert by_label["car"]["flags"]["inference"] is True
    assert "paste" not in by_label["car"]["flags"]


def test_remove_empty_sidecar_keeps_file_with_shapes(tmp_path):
    """非空 json 绝不能被删。"""
    from labelforge.engine.inference_task import remove_empty_sidecar
    img = _write_sidecar(tmp_path, "keep", [
        {"label": "x", "points": [[1, 1], [5, 1], [5, 5], [1, 5]],
         "shape_type": "rectangle", "flags": {}},
    ])

    assert remove_empty_sidecar(img) is False
    assert (tmp_path / "keep.json").exists()


def test_remove_empty_sidecar_ignores_missing_and_broken(tmp_path):
    """文件不存在或内容损坏时不删、不抛错。"""
    from labelforge.engine.inference_task import remove_empty_sidecar
    img = tmp_path / "none.png"
    img.write_bytes(b"x")
    assert remove_empty_sidecar(str(img)) is False

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"x")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert remove_empty_sidecar(str(broken)) is False
    assert (tmp_path / "broken.json").exists()


# ---------- 严格筛选：勾选之外不留推理框 ----------

def _auto(label, x=0):
    return {"label": label, "x": x, "y": 0, "width": 10, "height": 10,
            "flags": {"inference": True}}


def test_merge_prunes_prior_auto_boxes_outside_filter():
    """勾了 car 之后，上一轮推理留下的 person 不能继续留在画布上。

    否则筛选只能「加」不能「减」，用户看到勾选外的类别还在，
    就会认为筛选没生效（这正是最初的 bug 报告）。
    """
    existing = [_auto("person"), _auto("car", 1), _auto("bus", 2)]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                      class_filter={"car"})
    assert sorted(b["label"] for b in out) == ["car", "car"], out


def test_merge_keeps_manual_boxes_of_any_class():
    """严格筛选只针对推理框；人工标注无论类别都必须留下。"""
    existing = [
        {"label": "person", "x": 0, "y": 0, "width": 10, "height": 10},
        _auto("bus", 1),
    ]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                      class_filter={"car"})
    labels = sorted(b["label"] for b in out)
    assert labels == ["car", "person"], labels


def test_merge_keeps_paste_items_outside_filter():
    """贴图不是推理结果，不能被类别筛选清掉。"""
    existing = [{"label": "logo", "x": 3, "y": 3, "width": 10, "height": 10,
                 "is_paste": True}]
    out = merge_boxes(existing, [_auto("car")], replace_auto=False,
                      class_filter={"car"})
    assert sorted(b["label"] for b in out) == ["car", "logo"], out


def test_merge_without_filter_keeps_prior_auto_boxes():
    """没有筛选时不能顺手清理上一轮结果，否则默认行为被悄悄改掉。"""
    existing = [_auto("person"), _auto("bus", 1)]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                      class_filter=None)
    assert len(out) == 3, out


def test_merge_empty_filter_clears_all_old_auto_boxes():
    """空集合 = 全不选：旧推理框全清，只并入本轮。"""
    existing = [_auto("person"), _auto("bus", 1)]
    for empty in (set(), [], ()):
        out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                          class_filter=empty)
        assert sorted(b["label"] for b in out) == ["car"], (empty, out)


def test_merge_none_filter_keeps_all_old_auto_boxes():
    """None = 不过滤，旧推理框照常保留。"""
    existing = [_auto("person"), _auto("bus", 1)]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                      class_filter=None)
    assert sorted(b["label"] for b in out) == ["bus", "car", "person"]


def test_merge_empty_filter_still_keeps_manual_boxes():
    """全不选也不能动人工标注。"""
    existing = [{"label": "hand", "x": 0, "y": 0, "width": 4, "height": 4},
                _auto("person")]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=False,
                      class_filter=set())
    assert sorted(b["label"] for b in out) == ["car", "hand"], out


def test_merge_replace_auto_still_works_with_filter():
    """replace_auto 与严格筛选同时给出时，语义叠加：先清掉本图全部检测框
    （人工的也清），再并入本轮结果。"""
    existing = [_auto("person"), _auto("car", 1),
                {"label": "hand", "x": 5, "y": 5, "width": 4, "height": 4}]
    out = merge_boxes(existing, [_auto("car", 9)], replace_auto=True,
                      class_filter={"car"})
    labels = sorted(b["label"] for b in out)
    assert labels == ["car"], labels


def test_write_results_passes_filter_through(tmp_path):
    """批量写盘那条路径也必须带上筛选，否则当前图对了、其它图还是脏的。"""
    import numpy as np
    from labelforge.engine.inference_task import ImageResult, write_results

    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    results = {0: ImageResult(0, str(img), [_auto("car")], None, 10, 10)}
    memory = {0: [_auto("person"), _auto("bus", 1)]}
    written, failed = write_results(results, memory_boxes=memory,
                                    class_filter={"car"})
    assert failed == []
    from labelforge.engine.inference_task import read_sidecar_boxes
    boxes = read_sidecar_boxes(str(img)) or []
    assert sorted(b["label"] for b in boxes) == ["car"], boxes



# ---------- ③ 多个重叠区域：推理框跨区域去重 ----------

def _region(x, y, w, h):
    return {"x": x, "y": y, "width": w, "height": h}


def test_detect_in_scope_dedupes_a_box_found_in_two_overlapping_regions():
    """两个**重叠**区域盖住同一个目标时，结果里只能留一个框。

    用户要求「多个区域框重叠部分推理框去重」：区域各裁各的，重叠那块被推理
    两遍，同一个目标就出两份。`detect_in_scope` 的 ROI 分支已经把各区域结果
    交给 `merge_scope_results` 按 IoU 合并，这里钉住端到端确实生效。
    """
    from labelforge.engine.inference_task import detect_in_scope

    # 区域 A=(0,0,200,200)、B=(100,100,200,200)，重叠区 100..200。
    # 目标放在重叠区正中：原图 (140,140) 尺寸 40x40 -> 两个区域都看得见。
    # 两个区域尺寸相同、按形状分不开，所以让 detect 自己数第几次调用，
    # 把「这次裁剪的左上角」对上去，再把目标换算成裁剪图的**局部**坐标
    # （detector 只看得到裁剪副本，返回的就是局部坐标）。
    #
    # 两次检出**故意差几像素**：真实模型对同一目标的两次推理不会逐像素一致，
    # 差一点点时 IoU 略小于 1，才真正考验阈值。若两次完全相同（IoU 恰为 1），
    # 连阈值 1.0 都能合并掉，测不出「IoU 阈值有没有被正确传下去」。
    calls = {"n": 0}
    origins = [(0, 0), (100, 100)]
    jitter = [0.0, 3.0]

    class Detector:
        def detect(self, rgb, resize_fn):
            i = min(calls["n"], len(origins) - 1)
            calls["n"] += 1
            ox, oy = origins[i]
            off = jitter[i]
            return [{"label": "cat", "x": 140.0 - ox + off, "y": 140.0 - oy + off,
                     "width": 40.0, "height": 40.0, "conf": 0.9}]

    scope = {"mode": "roi",
             "regions": [_region(0, 0, 200, 200), _region(100, 100, 200, 200)]}
    boxes = detect_in_scope(np.zeros((400, 400, 3), np.uint8), Detector(),
                            lambda *a, **k: None, scope)

    assert len(boxes) == 1, (
        f"重叠区域盖住的同一个目标没去重，出了 {len(boxes)} 个框：{boxes}")
    assert boxes[0]["label"] == "cat"
    # 去重后留下的应当是**原图坐标**的那一个
    assert boxes[0]["x"] == 140.0 and boxes[0]["y"] == 140.0, boxes[0]


def test_detect_in_scope_keeps_distinct_targets_in_overlapping_regions():
    """反向要求：重叠区里**两个不同**目标不能被误合成一个。"""
    from labelforge.engine.inference_task import detect_in_scope

    origins = [(0, 0), (100, 100)]
    calls = {"n": 0}

    class Detector:
        def detect(self, rgb, resize_fn):
            ox, oy = origins[min(calls["n"], 1)]
            calls["n"] += 1
            # 两个目标都落在两个区域的重叠区里，但彼此相距很远
            return [{"label": "cat", "x": 120.0 - ox, "y": 120.0 - oy,
                     "width": 20.0, "height": 20.0, "conf": 0.9},
                    {"label": "dog", "x": 180.0 - ox, "y": 180.0 - oy,
                     "width": 20.0, "height": 20.0, "conf": 0.9}]

    scope = {"mode": "roi",
             "regions": [_region(0, 0, 250, 250), _region(100, 100, 250, 250)]}
    boxes = detect_in_scope(np.zeros((400, 400, 3), np.uint8), Detector(),
                            lambda *a, **k: None, scope)

    labels = sorted(b["label"] for b in boxes)
    assert labels == ["cat", "dog"], (
        f"两个不同目标应当各留一个（共两个）：{labels}")


def test_detect_in_scope_drops_boxes_whose_center_is_outside_every_region():
    """区域外心的框要丢掉：区域之外是靠掩膜遮住的，那里的框不可信。

    这条与去重合起来才是「重叠区域」的完整语义：**先在各自区域内取框、再去重**。
    顺序反了（先合并后过滤）会把区域外的框留下来。
    """
    from labelforge.engine.inference_task import detect_in_scope

    class Detector:
        def detect(self, rgb, resize_fn):
            return [{"label": "in", "x": 10.0, "y": 10.0,
                     "width": 20.0, "height": 20.0, "conf": 0.9},
                    {"label": "out", "x": 300.0, "y": 300.0,
                     "width": 20.0, "height": 20.0, "conf": 0.9}]

    scope = {"mode": "roi", "regions": [_region(0, 0, 100, 100)]}
    boxes = detect_in_scope(np.zeros((400, 400, 3), np.uint8), Detector(),
                            lambda *a, **k: None, scope)

    labels = sorted(b["label"] for b in boxes)
    assert labels == ["in"], f"区域外的框没被丢掉：{labels}"
