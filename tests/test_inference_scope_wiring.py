"""推理范围的接线测试：run_batch 的 scope 参数、四种模式、失败保护与配置持久化。

引擎本身（窗口枚举、ROI 掩膜、合并去重）在 `test_inference_scope.py`；这里
只测「接进推理链路之后」的行为，尤其是 scope=None 必须与原来逐字节一致。
"""
import copy
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

from labelforge.engine.inference_scope import ScopeError
from labelforge.engine.inference_task import (
    _masked_crop, _normalize_window_spec, detect_in_scope, normalize_settings,
    run_batch,
)
from labelforge.engine.onnx_inference import LETTERBOX_FILL


class _FakeDetector:
    """每个裁剪图中心放一个框；记录每次收到的裁剪尺寸。"""

    def __init__(self, iou=0.45, label="person", class_id=0):
        self.iou_threshold = iou
        self.label = label
        self.class_id = class_id
        self.calls = []

    def detect(self, rgb, resize_fn):
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        self.calls.append((w, h))
        return [{"x": w / 2.0 - 5, "y": h / 2.0 - 5, "width": 10, "height": 10,
                 "label": self.label, "score": 0.9, "class_id": self.class_id}]


def _loader(shape=(100, 200, 3)):
    def load(path):
        return np.zeros(shape, dtype=np.uint8)
    return load


# ---------- scope=None 保持原行为 ----------

def test_scope_none_calls_detect_once_on_the_whole_image():
    """scope=None 必须与原路径完全一致：一次 detect、原图尺寸。"""
    detector = _FakeDetector()
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)

    boxes = detect_in_scope(rgb, detector, None, {"mode": "full"})

    assert detector.calls == [(200, 100)]
    assert boxes == [{"x": 95.0, "y": 45.0, "width": 10, "height": 10,
                      "label": "person", "score": 0.9, "class_id": 0}]


def test_run_batch_without_scope_matches_plain_detect():
    """run_batch(scope=None) 的结果与直接调 detector.detect 一致。"""
    detector = _FakeDetector()
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    expected = detector.detect(rgb, None)
    detector.calls.clear()

    results = run_batch(["a.jpg"], detector, _loader(), None, scope=None)

    assert detector.calls == [(200, 100)]
    assert results[0].boxes == expected
    assert results[0].error is None
    assert (results[0].width, results[0].height) == (200, 100)


def test_run_batch_scope_full_is_same_as_none():
    """显式传 full 也不能走滑窗路径。"""
    for scope in (None, {"mode": "full"}):
        detector = _FakeDetector()
        run_batch(["a.jpg"], detector, _loader(), None, scope=scope)
        assert detector.calls == [(200, 100)], scope


# ---------- 滑窗 ----------

def test_window_scope_runs_each_window_and_offsets_boxes():
    """滑窗逐窗推理，框偏移回原图坐标。"""
    detector = _FakeDetector()

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "window",
                                   "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert detector.calls == [(100, 100), (100, 100)]
    assert sorted(b["x"] for b in boxes) == [45.0, 145.0]
    assert all(b["y"] == 45.0 for b in boxes)


def test_window_scope_collapses_exact_duplicates():
    """同一个目标被多个范围检到、坐标完全相同时，合并后只留一个。"""
    class _FixedDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            self.calls.append((int(rgb.shape[1]), int(rgb.shape[0])))
            return [{"x": 20, "y": 20, "width": 50, "height": 50,
                     "label": "person", "score": 0.9, "class_id": 0}]

    # 两个完全相同的 ROI → 两次 detect 得到同一坐标的框，NMS 必须合成一个
    detector = _FixedDetector()
    boxes = detect_in_scope(np.zeros((100, 100, 3), dtype=np.uint8), detector,
                            None, {"mode": "roi",
                                   "regions": [{"x": 0, "y": 0, "width": 100,
                                                "height": 100},
                                               {"x": 0, "y": 0, "width": 100,
                                                "height": 100}]})

    assert len(detector.calls) == 2
    assert len(boxes) == 1
    assert boxes[0]["x"] == 20 and boxes[0]["y"] == 20


def test_window_scope_reduces_but_may_leave_low_iou_duplicates():
    """滑窗重叠很大时，合并能大幅减少重复，但低 IoU 重复会残留。

    101 个 50px 框、中心逐像素错开：贪心 NMS 抑制掉 IoU>0.45 的邻居后仍剩
    若干个（框错开 19px 时 IoU=31/69≈0.449 刚好不被抑制）。这是标准贪心 NMS
    的固有行为，计划 §3.6.3 明确写了「不承诺完全消除低 IoU 重复」，这里把
    实际行为固定下来，避免有人误以为合并会清到只剩一个。
    """
    class _SameTargetDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            self.calls.append((int(rgb.shape[1]), int(rgb.shape[0])))
            w, h = int(rgb.shape[1]), int(rgb.shape[0])
            return [{"x": w / 2.0 - 25, "y": h / 2.0 - 25,
                     "width": 50, "height": 50,
                     "label": "person", "score": 0.9, "class_id": 0}]

    detector = _SameTargetDetector()
    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "window",
                                   "windows": {"w": 100, "h": 100, "overlap": 99}})

    assert len(detector.calls) == 101
    assert 1 < len(boxes) < len(detector.calls), len(boxes)
    # 相邻保留框的间距大于抑制半径，说明确实做了 IoU 抑制
    xs = sorted(b["x"] for b in boxes)
    assert all(b - a >= 19 for a, b in zip(xs, xs[1:])), xs


def test_window_scope_crops_the_actual_window_region():
    """每个窗口裁的是自己那块像素，不是图的左上角。

    用带梯度的图：如果实现固定从 (0,0) 裁，所有窗口看到的都是左上角那块，
    检出内容就完全一样。这里让每个窗口的均值不同，从而能区分。
    """
    class _MeanDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            self.calls.append((int(rgb.shape[1]), int(rgb.shape[0])))
            mean = float(rgb[:, :, 0].mean())
            return [{"x": 5, "y": 5, "width": 10, "height": 10,
                     "label": "person", "score": round(mean, 3),
                     "class_id": 0}]

    # 横向渐变：左半 0，右半 200
    rgb = np.zeros((100, 200, 3), dtype=np.uint8)
    rgb[:, 100:, :] = 200

    detector = _MeanDetector()
    boxes = detect_in_scope(rgb, detector, None,
                            {"mode": "window",
                             "windows": {"w": 100, "h": 100, "overlap": 0}})

    scores = sorted(b["score"] for b in boxes)
    assert scores == [0.0, 200.0], scores


def test_window_scope_keeps_different_classes():
    """不同类别的高重合框必须都保留，不能被合并吃掉。"""
    class _TwoClassDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            w, h = int(rgb.shape[1]), int(rgb.shape[0])
            base = {"x": w / 2.0 - 5, "y": h / 2.0 - 5, "width": 10, "height": 10,
                    "score": 0.9}
            return [dict(base, label="person", class_id=0),
                    dict(base, label="car", class_id=1)]

    boxes = detect_in_scope(np.zeros((100, 100, 3), dtype=np.uint8),
                            _TwoClassDetector(), None,
                            {"mode": "window",
                             "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert sorted(b["label"] for b in boxes) == ["car", "person"]


# ---------- 区域 ----------

def test_roi_scope_uses_only_the_region_crop():
    """区域模式只在 ROI 外接矩形上推理。"""
    detector = _FakeDetector()

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "roi",
                                   "regions": [{"x": 100, "y": 0,
                                                "width": 100, "height": 100}]})

    assert detector.calls == [(100, 100)]
    assert boxes[0]["x"] == 145.0 and boxes[0]["y"] == 45.0


def test_roi_scope_offset_uses_the_integer_crop_origin():
    """ROI 是小数坐标时，偏移必须用**取整后的实际裁剪起点**。

    掩膜按整数像素判定，偏移却用原始小数坐标的话，两者差半像素：detect 在
    裁剪图坐标系里返回的框，被搬到原图时会整体偏 0.5。这里 ROI 起点取
    x=10.5，正确偏移是 10（不是 10.5）。
    """
    detector = _FakeDetector()

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "roi",
                                   "regions": [{"x": 10.5, "y": 0,
                                                "width": 50, "height": 50}]})

    assert detector.calls == [(50, 50)]
    # 裁剪图中心 (20,20) + 偏移 10 → 原图 30；用 10.5 会得到 30.5
    assert boxes[0]["x"] == 30.0, boxes[0]["x"]
    assert boxes[0]["y"] == 20.0, boxes[0]["y"]


def test_roi_scope_offset_is_clamped_for_negative_region():
    """ROI 起点为负时偏移取 0（裁剪被钳到图内），不能是负数。"""
    detector = _FakeDetector()

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "roi",
                                   "regions": [{"x": -30, "y": -20,
                                                "width": 100, "height": 100}]})

    assert detector.calls == [(70, 80)]
    assert boxes[0]["x"] == 30.0, boxes[0]["x"]
    assert boxes[0]["y"] == 35.0, boxes[0]["y"]


def test_roi_scope_drops_boxes_whose_center_is_outside():
    """框中心落在 ROI 外的一律丢弃（哪怕它和 ROI 有重叠）。

    框跨过 ROI 左边界：原图 x 90..100，中心 95 在 ROI(100..200) 之外，但框
    本身没退化，钳制阶段不会丢掉它——只有中心过滤能拦下。
    （若框整个在负坐标，钳制就把它丢了，那条路径测不到中心过滤。）
    """
    class _StraddleDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            self.calls.append((int(rgb.shape[1]), int(rgb.shape[0])))
            return [{"x": -10, "y": 30, "width": 10, "height": 10,
                     "label": "person", "score": 0.9, "class_id": 0}]

    detector = _StraddleDetector()
    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None,
                            {"mode": "roi",
                             "regions": [{"x": 100, "y": 0,
                                          "width": 100, "height": 100}]})

    assert detector.calls == [(100, 100)]
    assert boxes == [], boxes


def test_roi_scope_keeps_boxes_whose_center_is_inside():
    """中心在 ROI 内的框必须保留（不能把中心过滤做得太狠）。"""
    class _InsideDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            return [{"x": 10, "y": 30, "width": 10, "height": 10,
                     "label": "person", "score": 0.9, "class_id": 0}]

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8),
                            _InsideDetector(), None,
                            {"mode": "roi",
                             "regions": [{"x": 100, "y": 0,
                                          "width": 100, "height": 100}]})

    assert len(boxes) == 1
    assert boxes[0]["x"] == 110.0 and boxes[0]["y"] == 30.0


def test_roi_scope_center_on_boundary_is_kept():
    """框中心恰好在 ROI 边界上算在内（矩形含边界）。"""
    class _EdgeDetector(_FakeDetector):
        def detect(self, rgb, resize_fn):
            # 原图 x 95..105，中心 100 正好落在 ROI 左边界
            return [{"x": -5, "y": 30, "width": 10, "height": 10,
                     "label": "person", "score": 0.9, "class_id": 0}]

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8),
                            _EdgeDetector(), None,
                            {"mode": "roi",
                             "regions": [{"x": 100, "y": 0,
                                          "width": 100, "height": 100}]})

    assert len(boxes) == 1, boxes


def test_roi_scope_runs_every_region():
    """多 ROI 逐个跑，结果合并。"""
    detector = _FakeDetector()

    boxes = detect_in_scope(np.zeros((100, 200, 3), dtype=np.uint8), detector,
                            None, {"mode": "roi",
                                   "regions": [{"x": 0, "y": 0, "width": 100,
                                                "height": 100},
                                               {"x": 100, "y": 0, "width": 100,
                                                "height": 100}]})

    assert detector.calls == [(100, 100), (100, 100)]
    assert sorted(b["x"] for b in boxes) == [45.0, 145.0]


def test_roi_scope_rejects_missing_regions():
    """选了区域却没有 ROI：记为该图失败，不能「成功但 0 框」。

    批量里一张图的配置错误不该中断整批，所以是记 error 而不是抛出去。
    """
    detector = _FakeDetector()
    results = run_batch(["a.jpg"], detector, _loader(), None,
                        scope={"mode": "roi"})

    assert results[0].error and "必须提供 ROI" in results[0].error
    assert detector.calls == []


def test_roi_scope_rejects_region_without_intersection():
    """ROI 与图完全没有交集时该图失败，不能返回空结果。"""
    detector = _FakeDetector()
    results = run_batch(["a.jpg"], detector, _loader(), None,
                        scope={"mode": "roi",
                               "regions": [{"x": 500, "y": 500,
                                            "width": 50, "height": 50}]})

    assert results[0].error and "交集" in results[0].error
    assert detector.calls == []


def test_roi_scope_rejects_degenerate_region():
    """零面积 ROI 必须拦下。"""
    results = run_batch(["a.jpg"], _FakeDetector(), _loader(), None,
                        scope={"mode": "roi",
                               "regions": [{"x": 10, "y": 10, "width": 0,
                                            "height": 50}]})
    assert results[0].error


# ---------- 参考图尺寸 ----------

def test_reference_size_mismatch_fails_only_that_image():
    """ROI 按参考图量的：尺寸不符的图记失败，符合的照常推理。"""
    detector = _FakeDetector()

    def loader(path):
        return np.zeros((100 if path == "ok.jpg" else 300, 200, 3), dtype=np.uint8)

    results = run_batch(["ok.jpg", "bad.jpg"], detector, loader, None,
                        scope={"mode": "roi",
                               "regions": [{"x": 0, "y": 0, "width": 50,
                                            "height": 50}],
                               "reference_size": (200, 100)})

    assert results[0].error is None
    assert results[1].error and "参考图" in results[1].error


def test_reference_size_absent_accepts_any_size():
    """没带参考图尺寸时不做尺寸校验（老调用方/视频逐帧用）。"""
    results = run_batch(["a.jpg"], _FakeDetector(), _loader((300, 400, 3)), None,
                        scope={"mode": "roi",
                               "regions": [{"x": 0, "y": 0, "width": 50,
                                            "height": 50}]})
    assert results[0].error is None


def test_normalize_scope_rejects_bad_reference_size():
    from labelforge.engine.inference_scope import normalize_scope

    for bad in ((0, 100), (-1, 100), ("x", 100), (100,)):
        with pytest.raises(ScopeError):
            normalize_scope({"mode": "roi",
                             "regions": [{"x": 0, "y": 0, "width": 10,
                                          "height": 10}],
                             "reference_size": bad}, (100, 100))


# ---------- ROI 快照与掩膜 ----------

def test_scope_snapshot_is_deep_copied():
    """启动时的 scope 必须是快照：之后改原 ROI 不能影响已定稿的配置。"""
    from labelforge.engine.inference_scope import normalize_scope

    regions = [{"x": 10, "y": 10, "width": 50, "height": 50}]
    scope = normalize_scope({"mode": "roi", "regions": regions}, (100, 100))
    regions[0]["x"] = 9999
    regions.append({"x": 0, "y": 0, "width": 1, "height": 1})

    assert scope["regions"][0]["x"] == 10
    assert len(scope["regions"]) == 1


def test_polygon_roi_masks_outside_pixels():
    """多边形 ROI 把界外像素涂成 letterbox 填充色，且不改写原图。"""
    from labelforge.engine.inference_scope import Region

    points = [[20.0, 10.0], [180.0, 40.0], [60.0, 140.0]]
    region = Region(bbox=(20.0, 10.0, 180.0, 140.0), points=points)
    rgb = np.full((150, 200, 3), 200, dtype=np.uint8)
    original = rgb.copy()

    crop, ox, oy = _masked_crop(rgb, region, 20, 10, 180, 140)

    assert (ox, oy) == (20, 10)
    assert crop.shape == (130, 160, 3)
    assert (rgb == original).all(), "原图被改写了"
    # 顶点 (20,10) 落在边界上 → 算在内，保持原色（与框中心判定同口径）
    assert crop[0, 0, 0] == 200
    # (100,10) 在多边形外（该高度上多边形只覆盖顶点附近）→ 填充色
    assert crop[0, 80, 0] == LETTERBOX_FILL
    # 多边形内部 → 保持原色
    assert crop[60, 60, 0] == 200
    assert LETTERBOX_FILL in crop[:, :, 0]


def test_rectangular_roi_needs_no_mask():
    """矩形 ROI 不做掩膜：整块保留原像素。"""
    from labelforge.engine.inference_scope import Region

    region = Region(bbox=(0.0, 0.0, 50.0, 50.0), points=None)
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)

    crop, ox, oy = _masked_crop(rgb, region, 0, 0, 50, 50)

    assert (ox, oy) == (0, 0)
    assert (crop == 200).all()


def test_masked_crop_returns_integer_origin_for_fractional_roi():
    """ROI 坐标是小数时，返回的起点必须取整并与掩膜口径一致。"""
    from labelforge.engine.inference_scope import Region

    region = Region(bbox=(10.6, 20.4, 60.6, 70.4), points=None)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    crop, ox, oy = _masked_crop(rgb, region, 10.6, 20.4, 60.6, 70.4)

    assert (ox, oy) == (10, 20)
    assert crop.shape == (50, 50, 3)


def test_roi_crop_is_clamped_to_image_bounds():
    """ROI 越界部分被裁掉，不报错也不越界读取。"""
    from labelforge.engine.inference_scope import Region

    region = Region(bbox=(0.0, 0.0, 50.0, 50.0), points=None)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)

    crop, ox, oy = _masked_crop(rgb, region, -30, -30, 20, 20)

    assert (ox, oy) == (0, 0)
    assert crop.shape == (20, 20, 3)


# ---------- 每张图独立校验 ----------

def test_each_image_is_validated_independently():
    """逐图校验：坏图记 error，好图照常出结果（批量不能整批失败）。"""
    detector = _FakeDetector()

    def loader(path):
        if path == "bad.jpg":
            return None
        return np.zeros((100, 200, 3), dtype=np.uint8)

    results = run_batch(["ok.jpg", "bad.jpg", "ok2.jpg"], detector, loader, None,
                        scope={"mode": "window",
                               "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert results[0].error is None and results[2].error is None
    assert results[1].error and "无法读取" in results[1].error


def test_scope_error_does_not_stop_the_batch():
    """一张图的 scope 校验失败不中断整批。"""
    detector = _FakeDetector()

    def loader(path):
        # 第二张图尺寸与参考图不符 → 只有它失败
        return np.zeros((100, 200, 3) if path == "a.jpg" else (400, 400, 3),
                        dtype=np.uint8)

    results = run_batch(["a.jpg", "b.jpg"], detector, loader, None,
                        scope={"mode": "roi",
                               "regions": [{"x": 0, "y": 0, "width": 50,
                                            "height": 50}],
                               "reference_size": (200, 100)})

    assert len(results) == 2
    assert results[0].error is None
    assert results[1].error


# ---------- 取消 ----------

def test_cancel_stops_between_images():
    """取消时保留已完成的结果，不跑剩下的图。"""
    detector = _FakeDetector()
    seen = []

    def interrupted():
        return len(seen) >= 1

    def progress(position, total, name, count, elapsed):
        seen.append(name)

    results = run_batch(["a.jpg", "b.jpg", "c.jpg"], detector, _loader(), None,
                        progress_fn=progress, is_interrupted=interrupted,
                        scope={"mode": "window",
                               "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert len(results) == 1
    assert len(seen) == 1


# ---------- 窗口参数持久化 ----------

def test_normalize_settings_keeps_window_spec():
    """窗口参数必须过白名单，否则存了也会被丢掉。"""
    settings = normalize_settings({"model_path": "m.onnx",
                                   "window_spec": {"w": 640, "h": 512,
                                                   "overlap": 100}})
    spec = settings["window_spec"]
    # 三个尺寸键要原样留下来；`min_visible` 是后来加的合法窗口参数，
    # 所以这里按「这三个键在」断言，不钉死整份字典。
    assert (spec["w"], spec["h"], spec["overlap"]) == (640, 512, 100)


def test_normalize_settings_drops_broken_window_spec():
    """坏窗口参数整份丢弃，不留半份让下游猜。"""
    for bad in (None, {}, "x", {"w": 640}, {"w": "a", "h": 640, "overlap": 0},
                {"w": 0, "h": 640, "overlap": 0},
                {"w": 640, "h": -1, "overlap": 0},
                {"w": 640, "h": 640, "overlap": -5}):
        assert normalize_settings({"window_spec": bad})["window_spec"] == {}, bad


def test_normalize_window_spec_accepts_zero_overlap():
    """重叠 0 是合法值（不重叠的网格），不能被当成坏值丢掉。"""
    out = _normalize_window_spec({"w": 640, "h": 640, "overlap": 0})
    assert (out["w"], out["h"], out["overlap"]) == (640, 640, 0)


def test_normalize_window_spec_ignores_extra_keys():
    """未知字段不进结果（避免旧配置里的杂项一直留着）。

    `fill` 是数据增强侧的键，推理窗口不吃它；`min_visible` 则是**已知**的
    窗口参数，会被保留——用后者当反例才说明这是白名单而不是「什么都留」。
    """
    out = _normalize_window_spec({"w": 640, "h": 640, "overlap": 10, "fill": True})
    assert "fill" not in out, out
    assert (out["w"], out["h"], out["overlap"]) == (640, 640, 10)


# ---------- 与既有行为的兼容 ----------

def test_scope_does_not_change_other_settings():
    """新增 window_spec 不能影响既有字段的规整结果。"""
    base = {"model_path": "m.onnx", "conf_threshold": 0.5, "iou_threshold": 0.4}
    without = normalize_settings(base)
    with_spec = normalize_settings(dict(base, window_spec={"w": 640, "h": 640,
                                                           "overlap": 0}))
    without.pop("window_spec")
    with_spec.pop("window_spec")
    assert without == with_spec


def test_class_filter_still_applies_with_scope():
    """范围推理的结果仍要过类别筛选。"""
    detector = _FakeDetector(label="person")

    results = run_batch(["a.jpg"], detector, _loader(), None,
                        class_filter={"car"},
                        scope={"mode": "window",
                               "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert results[0].boxes == []


def test_progress_reports_per_image_not_per_window():
    """progress 仍是「每张图一次」，不能变成每窗一次。"""
    calls = []

    def progress(position, total, name, count, elapsed):
        calls.append((position, total, name, count))

    detector = _FakeDetector()
    run_batch(["a.jpg", "b.jpg"], detector, _loader(), None,
              progress_fn=progress,
              scope={"mode": "window",
                     "windows": {"w": 100, "h": 100, "overlap": 0}})

    assert [c[0] for c in calls] == [1, 2]
    assert all(c[1] == 2 for c in calls)
    # 每图 2 窗，每窗 1 框 → 合并后 2 框
    assert [c[3] for c in calls] == [2, 2]
    assert len(detector.calls) == 4


def test_unknown_scope_key_is_rejected():
    """范围配置的未知字段必须报错，不静默忽略。"""
    results = run_batch(["a.jpg"], _FakeDetector(), _loader(), None,
                        scope={"mode": "window", "typo": 1,
                               "windows": {"w": 100, "h": 100, "overlap": 0}})
    assert results[0].error and "未知字段" in results[0].error


def test_unknown_mode_is_rejected():
    results = run_batch(["a.jpg"], _FakeDetector(), _loader(), None,
                        scope={"mode": "diagonal"})
    assert results[0].error and "未知范围模式" in results[0].error


# ---------- 跨窗合并的 NMS 阈值必须取本次任务的模型行值 ----------

class _SameSpotDetector:
    """每个窗口都报出裁剪图正中心的同一个目标（故意制造跨窗重复）。

    两个窗口的裁剪中心在原图上相距 `窗口尺寸 - 重叠`；`side` 控制框边长，
    从而控制这两个重复框的 IoU，让阈值高低能真正改变保留数量。
    """

    def __init__(self, iou, side, label="person", class_id=0):
        self.iou_threshold = iou
        self.side = side
        self.label = label
        self.class_id = class_id
        self.calls = []

    def detect(self, rgb, resize_fn):
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        self.calls.append((w, h))
        half = self.side / 2.0
        return [{"x": w / 2.0 - half, "y": h / 2.0 - half,
                 "width": float(self.side), "height": float(self.side),
                 "label": self.label, "score": 0.9,
                 "class_id": self.class_id}]


# 150×100 的图切两个 100×100 窗口、重叠 50 → 恰好 2 个窗口，中心相距 50px。
# 框边长取 150px 时两个重复框的 IoU = (150-50)/(150+50) = 0.5，正好卡在
# 「严阈值 0.3 要抑制」与「松阈值 0.95 要保留」之间。
_TWO_WINDOW_RGB = np.zeros((100, 150, 3), dtype=np.uint8)
_TWO_WINDOW_SCOPE = {"mode": "window",
                     "windows": {"w": 100, "h": 100, "overlap": 50}}
_TWO_WINDOW_BOX_SIDE = 150


def test_window_merge_uses_detector_iou_threshold_not_a_constant():
    """跨窗合并的 NMS 阈值取 `detector.iou_threshold`，不是写死的常量。

    这是「阈值跟随模型行」这条接线的唯一钉子：`merge_scope_results` 自己
    尊重阈值已有 `test_inference_scope.py::test_merge_respects_iou_threshold`
    覆盖，但**没人钉住 `detect_in_scope` 真的把 detector 的值传了进去**。
    同一张图、同一批重复框（IoU 0.5），只改 detector 的阈值，保留数量必须
    跟着变；若把 `_scope_iou(detector)` 换成 `DEFAULT_IOU`（0.45），严的那
    一路会变成 1→1 看似通过，松的那一路则拿不到 2，本条即红。
    """
    tight = _SameSpotDetector(iou=0.3, side=_TWO_WINDOW_BOX_SIDE)
    boxes_tight = detect_in_scope(_TWO_WINDOW_RGB, tight, None,
                                  _TWO_WINDOW_SCOPE)

    loose = _SameSpotDetector(iou=0.95, side=_TWO_WINDOW_BOX_SIDE)
    boxes_loose = detect_in_scope(_TWO_WINDOW_RGB, loose, None,
                                  _TWO_WINDOW_SCOPE)

    assert len(tight.calls) == len(loose.calls) == 2, "两次都必须跑满 2 个窗口"
    assert len(boxes_tight) == 1, [b["x"] for b in boxes_tight]
    assert len(boxes_loose) == 2, [b["x"] for b in boxes_loose]


def test_window_merge_threshold_is_per_run_not_shared_state():
    """两次调用互不影响：阈值来自各自的 detector，没有跨调用的残留。

    `detect_in_scope` 是模块级函数，若阈值被缓存在模块变量里，先紧后松
    与先松后紧会得到同样的结果。两个方向都跑，结论必须对称。
    """

    def count(iou):
        return len(detect_in_scope(
            _TWO_WINDOW_RGB, _SameSpotDetector(iou=iou, side=_TWO_WINDOW_BOX_SIDE),
            None, _TWO_WINDOW_SCOPE))

    # 先紧后松
    assert (count(0.3), count(0.95)) == (1, 2)
    # 先松后紧（反过来）：仍各自正确 → 没有被上一次性污染
    assert (count(0.95), count(0.3)) == (2, 1)


# ---------- min_visible：跨窗低可见框过滤 ----------

class _EdgeBoxDetector:
    """在裁剪图右边缘放一个「一半在窗外」的框，用来试可见比例过滤。

    裁剪图宽 100，框宽 20、右边缘贴住裁剪图右边界 → 在**本窗**里完整可见。
    它真正的裁剪来源是原图，所以跨窗去重时同一个目标会在相邻窗各出一个框；
    这里只关心「本窗内的可见比例」这个判据。
    """

    def __init__(self, iou=0.45):
        self.iou_threshold = iou
        self.calls = []

    def detect(self, rgb, resize_fn):
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        self.calls.append((w, h))
        return [{"x": w - 10, "y": h / 2.0 - 5, "width": 10, "height": 10,
                 "label": "person", "score": 0.9, "class_id": 0}]


class _HalfOutDetector:
    """返回一个**只有一半落在裁剪图内**的框（左半在窗外，坐标可负）。"""

    def __init__(self, iou=0.45):
        self.iou_threshold = iou

    def detect(self, rgb, resize_fn):
        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        # 框中心在裁剪图左边界上：一半露在裁剪图外
        return [{"x": -5.0, "y": h / 2.0 - 10, "width": 10, "height": 20,
                 "label": "person", "score": 0.9, "class_id": 0}]


def test_window_inference_keeps_everything_at_zero_min_visible():
    """min_visible=0 时不做可见比例过滤（只有零交集才丢）。"""
    boxes = detect_in_scope(
        np.zeros((100, 100, 3), dtype=np.uint8), _HalfOutDetector(), None,
        {"mode": "window",
         "windows": {"w": 100, "h": 100, "overlap": 0, "min_visible": 0.0}})

    assert len(boxes) == 1, f"min_visible=0 不该丢掉任何有交集的框：{boxes}"


def test_window_inference_drops_boxes_below_min_visible():
    """min_visible 调高后，可见比例不够的框要被丢掉。"""
    boxes = detect_in_scope(
        np.zeros((100, 100, 3), dtype=np.uint8), _HalfOutDetector(), None,
        {"mode": "window",
         "windows": {"w": 100, "h": 100, "overlap": 0, "min_visible": 0.9}})

    assert boxes == [], f"可见比例只有约一半，min_visible=0.9 时该被丢掉：{boxes}"


def test_min_visible_of_one_drops_partially_visible_boxes():
    """100% 时只保留完全落在窗内的框。"""
    boxes = detect_in_scope(
        np.zeros((100, 100, 3), dtype=np.uint8), _HalfOutDetector(), None,
        {"mode": "window",
         "windows": {"w": 100, "h": 100, "overlap": 0, "min_visible": 1.0}})

    assert boxes == [], f"只露一半的框在 100% 下必须丢掉：{boxes}"


def test_min_visible_of_one_keeps_a_fully_visible_box():
    """100% 时完全落在窗内的框要保留——不能把整批结果清空。"""
    boxes = detect_in_scope(
        np.zeros((100, 100, 3), dtype=np.uint8), _EdgeBoxDetector(), None,
        {"mode": "window",
         "windows": {"w": 100, "h": 100, "overlap": 0, "min_visible": 1.0}})

    assert len(boxes) == 1, f"完全在窗内的框被误丢：{boxes}"


def test_min_visible_filter_does_not_apply_to_full_image_mode():
    """全图模式没有窗口，min_visible 不该参与（否则会误丢边缘框）。"""
    boxes = detect_in_scope(
        np.zeros((100, 100, 3), dtype=np.uint8), _HalfOutDetector(), None,
        {"mode": "full"})

    assert len(boxes) == 1, f"全图模式不该做跨窗可见比例过滤：{boxes}"
