"""ONNX 推理纯计算测试：预处理几何、v5/v8 解码、按类 NMS、格式拒绝。

不加载真实模型（项目里没有 ONNX 文件，当前环境也没有 onnxruntime），
因此这里只覆盖「输出张量 → 内部框」这段可独立验证的计算，模型精度与
推理速度必须用真实 YOLOv5/v8 模型对照，不能由这些用例代替。
"""
import numpy as np
import pytest

from pastelabel.engine.onnx_inference import (
    FORMAT_V5, FORMAT_V8, InferenceError, boxes_to_internal, decode_predictions,
    decode_yolov5, decode_yolov8, detect_output_format, letterbox_geometry,
    nms_per_class, unletterbox_boxes,
)


# ---------- letterbox 几何 ----------

def test_letterbox_square_into_square_no_padding():
    g = letterbox_geometry(640, 640, 640, 640)
    assert g.scale == 1.0
    assert (g.pad_x, g.pad_y) == (0, 0)
    assert (g.new_w, g.new_h) == (640, 640)


def test_letterbox_wide_image_pads_vertically_only():
    # 1280x640 → 640x640：scale=0.5，补在上下，左右无补边
    g = letterbox_geometry(1280, 640, 640, 640)
    assert g.scale == 0.5
    assert (g.new_w, g.new_h) == (640, 320)
    assert g.pad_x == 0
    assert g.pad_y == 160


def test_letterbox_tall_image_pads_horizontally_only():
    g = letterbox_geometry(320, 640, 640, 640)
    assert g.scale == 1.0
    assert (g.new_w, g.new_h) == (320, 640)
    assert g.pad_x == 160
    assert g.pad_y == 0


def test_letterbox_rounds_padding_so_scaled_size_plus_padding_fits_canvas():
    # 奇数缩放比例下补边必须取整，且 new + 2*pad 不超出画布
    g = letterbox_geometry(1000, 333, 640, 640)
    assert g.new_w + 2 * g.pad_x <= 640
    assert g.new_h + 2 * g.pad_y <= 640


def test_letterbox_rejects_invalid_sizes():
    with pytest.raises(InferenceError):
        letterbox_geometry(0, 100, 640, 640)
    with pytest.raises(InferenceError):
        letterbox_geometry(100, 100, 0, 640)


def test_unletterbox_roundtrip_restores_original_coords():
    # 1280x640 原图，模型输入 640x640
    g = letterbox_geometry(1280, 640, 640, 640)
    # 原图 (100, 50)-(300, 150) 在模型输入坐标下应为 (50, 185)-(150, 235)
    model_box = np.array([[50.0, 185.0, 150.0, 235.0]], dtype=np.float32)
    out, keep = unletterbox_boxes(model_box, g, 1280, 640)
    assert keep.tolist() == [True]
    assert np.allclose(out, [[100.0, 50.0, 300.0, 150.0]], atol=1e-3)


def test_unletterbox_clamps_to_image_bounds():
    # 模型框超出右下边界时钳到原图内，不产生越界坐标
    g = letterbox_geometry(640, 480, 640, 480)
    out, keep = unletterbox_boxes(
        np.array([[0.0, 0.0, 900.0, 900.0]], dtype=np.float32), g, 640, 480,
    )
    assert keep.tolist() == [True]
    assert np.allclose(out, [[0.0, 0.0, 640.0, 480.0]])


def test_unletterbox_drops_box_fully_inside_padding():
    # 框整体落在补边区内（模型输入的 0~160 行是灰边），钳制后为零面积 → 掩码为 False
    g = letterbox_geometry(1280, 640, 640, 640)
    out, keep = unletterbox_boxes(
        np.array([[10.0, 10.0, 100.0, 100.0]], dtype=np.float32), g, 1280, 640,
    )
    assert keep.tolist() == [False]
    # 掩码由调用方使用；本函数不改变行数，避免与分数、类别错位
    assert out.shape == (1, 4)


def test_unletterbox_empty_input():
    g = letterbox_geometry(100, 100, 100, 100)
    out, keep = unletterbox_boxes(
        np.zeros((0, 4), dtype=np.float32), g, 100, 100,
    )
    assert out.shape == (0, 4)
    assert keep.shape == (0,)


# ---------- YOLOv5 解码 ----------

def _v5_output(rows):
    """构造 [1, N, 5+C] 的 v5 输出。"""
    return np.array([rows], dtype=np.float32)


def test_yolov5_confidence_is_objectness_times_class_score():
    # 一行：cx=100, cy=100, w=40, h=20, obj=0.8, 两个类别分数 0.9 / 0.1
    out = decode_yolov5(_v5_output([[100, 100, 40, 20, 0.8, 0.9, 0.1]]), 0.01)
    assert len(out["boxes"]) == 1
    assert out["scores"][0] == pytest.approx(0.8 * 0.9, rel=1e-6)
    assert out["class_ids"][0] == 0


def test_yolov5_xywh_converted_to_xyxy():
    out = decode_yolov5(_v5_output([[100, 100, 40, 20, 1.0, 1.0]]), 0.01)
    assert np.allclose(out["boxes"][0], [80.0, 90.0, 120.0, 110.0])


def test_yolov5_filters_by_threshold():
    rows = [
        [100, 100, 40, 20, 0.9, 1.0],   # 0.9 → 保留
        [200, 200, 40, 20, 0.1, 1.0],   # 0.1 → 过滤
    ]
    out = decode_yolov5(_v5_output(rows), 0.25)
    assert len(out["boxes"]) == 1
    assert out["scores"][0] == pytest.approx(0.9, rel=1e-6)


def test_yolov5_empty_result_keeps_shapes():
    out = decode_yolov5(_v5_output([[100, 100, 40, 20, 0.01, 0.01]]), 0.9)
    assert out["boxes"].shape == (0, 4)
    assert out["scores"].shape == (0,)
    assert out["class_ids"].shape == (0,)


def test_yolov5_rejects_class_count_mismatch():
    # 输出只有 1 个类别，类别表有 3 个 → 明确报错，不静默按错位处理
    with pytest.raises(InferenceError):
        decode_yolov5(_v5_output([[100, 100, 40, 20, 0.9, 0.9]]), 0.25, num_classes=3)


def test_yolov5_rejects_batch_gt_one():
    raw = np.zeros((2, 5, 6), dtype=np.float32)
    with pytest.raises(InferenceError):
        decode_yolov5(raw, 0.25)


def test_yolov5_rejects_too_few_columns():
    with pytest.raises(InferenceError):
        decode_yolov5(_v5_output([[100, 100, 40, 20, 0.9]]), 0.25)


# ---------- YOLOv8 解码 ----------

def test_yolov8_channel_first_transposed_and_decoded():
    # [1, 4+C, N] = [1, 6, 1]：cx, cy, w, h, 两个类别分数
    raw = np.array([[[100.0], [100.0], [40.0], [20.0], [0.7], [0.1]]], dtype=np.float32)
    out = decode_yolov8(raw, 0.01)
    assert len(out["boxes"]) == 1
    assert out["scores"][0] == pytest.approx(0.7, rel=1e-6)
    assert out["class_ids"][0] == 0
    assert np.allclose(out["boxes"][0], [80.0, 90.0, 120.0, 110.0])


def test_yolov8_confidence_is_class_score_without_objectness():
    # 没有 objectness：分数就等于类别分数本身
    raw = np.array([[[10.0], [20.0], [4.0], [4.0], [0.42], [0.9]]], dtype=np.float32)
    out = decode_yolov8(raw, 0.01)
    assert out["scores"][0] == pytest.approx(0.9, rel=1e-6)
    assert out["class_ids"][0] == 1


def test_yolov8_accepts_transposed_layout():
    # 同一组数据写成 [1, N, 4+C]
    raw = np.array([[[10.0, 20.0, 4.0, 4.0, 0.42, 0.9]]], dtype=np.float32)
    out = decode_yolov8(raw, 0.01, num_classes=2)
    assert len(out["boxes"]) == 1
    assert out["class_ids"][0] == 1


def test_yolov8_ambiguous_layout_rejected():
    # num_classes=1 时 4+C=5，形状 5x5 无法判断是通道优先还是转置 → 必须报错
    raw = np.zeros((1, 5, 5), dtype=np.float32)
    with pytest.raises(InferenceError):
        decode_yolov8(raw, 0.25, num_classes=1)


def test_yolov8_rejects_shape_mismatch_with_class_table():
    raw = np.zeros((1, 7, 3), dtype=np.float32)   # 4+C=7 → C=3，与类别表 2 不符
    with pytest.raises(InferenceError):
        decode_yolov8(raw, 0.25, num_classes=2)


def test_yolov8_empty_result_keeps_shapes():
    raw = np.array([[[10.0], [20.0], [4.0], [4.0], [0.0], [0.0]]], dtype=np.float32)
    out = decode_yolov8(raw, 0.5)
    assert out["boxes"].shape == (0, 4)


# ---------- 格式识别与拒绝 ----------

def test_detect_format_v5_by_class_count():
    assert detect_output_format((1, 25200, 7), num_classes=2) == FORMAT_V5


def test_detect_format_v8_by_class_count():
    assert detect_output_format((1, 6, 8400), num_classes=2) == FORMAT_V8
    assert detect_output_format((1, 8400, 6), num_classes=2) == FORMAT_V8


def test_detect_format_returns_none_when_class_count_conflicts():
    # 类别表与输出对不上时不能猜，交给用户显式指定
    assert detect_output_format((1, 25200, 7), num_classes=80) is None


def test_detect_format_without_class_table_prefers_channel_first():
    assert detect_output_format((1, 6, 8400)) == FORMAT_V8


def test_detect_format_without_class_table_rejects_ambiguous_small_shapes():
    """无类别数时小 N 形状两种解释都成立，必须返回 None 让用户指定。

    真实回归：v5 模型 [1, 3, 8] 曾被猜成 v8，于是把 objectness 当类别分数、
    把列当行读，静默产出错误标注（分数 0.9 而非 obj×cls=0.81，类别名也错位）。
    """
    assert detect_output_format((1, 3, 8)) is None
    assert detect_output_format((1, 8, 3)) is None
    assert detect_output_format((1, 100, 3)) is None


def test_detect_format_without_class_table_needs_real_anchor_count():
    assert detect_output_format((1, 84, 8400)) == FORMAT_V8
    assert detect_output_format((1, 25200, 85)) is None, "末轴是通道的 v5 不能猜成 v8"


def test_decode_predictions_rejects_unknown_format():
    raw = np.zeros((1, 10, 7), dtype=np.float32)
    with pytest.raises(InferenceError):
        decode_predictions(raw, "obb", 0.25)
    with pytest.raises(InferenceError):
        decode_predictions(raw, "v7", 0.25)


def test_decode_predictions_dispatches_by_format():
    v5 = _v5_output([[100, 100, 40, 20, 1.0, 1.0]])
    assert len(decode_predictions(v5, FORMAT_V5, 0.01)["boxes"]) == 1
    v8 = np.array([[[100.0], [100.0], [40.0], [20.0], [1.0]]], dtype=np.float32)
    assert len(decode_predictions(v8, FORMAT_V8, 0.01)["boxes"]) == 1


# ---------- 按类别 NMS ----------

def test_nms_removes_overlapping_same_class():
    boxes = np.array([
        [0, 0, 100, 100],
        [5, 5, 105, 105],     # 与第 1 个高度重合
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)
    assert nms_per_class(boxes, scores, classes, 0.45) == [0]


def test_nms_keeps_overlapping_different_classes():
    # 不同类别即使几乎完全重合也要都保留——这是接入保存器时的关键约束
    boxes = np.array([
        [0, 0, 100, 100],
        [5, 5, 105, 105],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 1], dtype=np.int64)
    assert nms_per_class(boxes, scores, classes, 0.45) == [0, 1]


def test_nms_keeps_low_iou_same_class():
    boxes = np.array([
        [0, 0, 100, 100],
        [200, 200, 300, 300],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)
    assert nms_per_class(boxes, scores, classes, 0.45) == [0, 1]


def test_nms_orders_by_score_descending():
    boxes = np.array([
        [0, 0, 100, 100],
        [500, 500, 600, 600],
    ], dtype=np.float32)
    scores = np.array([0.3, 0.95], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)
    assert nms_per_class(boxes, scores, classes, 0.45) == [1, 0]


def test_nms_empty_input():
    assert nms_per_class(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
    ) == []


def test_nms_ignores_degenerate_zero_area_box():
    # 零面积框与任何框的 IoU 都是 0，不能被当成重复项抑制掉别的框
    boxes = np.array([
        [10, 10, 10, 10],
        [10, 10, 60, 60],
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    classes = np.array([0, 0], dtype=np.int64)
    assert nms_per_class(boxes, scores, classes, 0.45) == [0, 1]


# ---------- 转内部框格式 ----------

def test_boxes_to_internal_maps_label_by_class_id():
    out = boxes_to_internal(
        np.array([[10, 20, 110, 220]], dtype=np.float32),
        np.array([0.75], dtype=np.float32),
        np.array([1], dtype=np.int64),
        ["cat", "dog"],
    )
    assert len(out) == 1
    assert out[0]["label"] == "dog"
    assert out[0]["x"] == 10.0
    assert out[0]["y"] == 20.0
    assert out[0]["width"] == 100.0
    assert out[0]["height"] == 200.0
    assert out[0]["score"] == pytest.approx(0.75)


def test_boxes_to_internal_uses_empty_label_for_out_of_range_class():
    # 类别 id 越界时给空标签而不是抛错：整批任务不能因单帧元数据异常而中断
    out = boxes_to_internal(
        np.array([[0, 0, 10, 10]], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        np.array([7], dtype=np.int64),
        ["cat"],
    )
    assert out[0]["label"] == ""


def test_boxes_to_internal_skips_degenerate_boxes():
    out = boxes_to_internal(
        np.array([[10, 10, 10, 50]], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([0], dtype=np.int64),
        ["cat"],
    )
    assert out == []


def test_boxes_to_internal_empty_input():
    assert boxes_to_internal(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        ["cat"],
    ) == []
