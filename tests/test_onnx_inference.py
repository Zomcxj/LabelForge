"""ONNX 推理纯计算测试：预处理几何、v5/v8 解码、按类 NMS、格式拒绝。

不加载真实模型：这里覆盖「输出张量 → 内部框」这段可独立验证的计算。
真实模型精度必须用 model/ 下的 yolov5n/yolov8n 对照，不能由这些用例代替；
端到端链路（真实 Session、元数据、模糊形状拒绝）见 test_inference_e2e.py。
"""
import numpy as np
import pytest

from labelforge.engine.onnx_inference import (
    DEFAULT_MAX_DET, FORMAT_V5, FORMAT_V8, FORMAT_V26, InferenceError,
    boxes_to_internal, decode_end2end, decode_predictions,
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
    # 末轴是通道的 v5 要认成 v5，不能猜成 v8。25200x85 是 yolov5n 的形状、
    # 25200x9 是 4 类第三方导出的形状；没有类别表时也必须能判出来，否则这类
    # 模型会被「请显式选择格式」直接挡住。
    assert detect_output_format((1, 25200, 85)) == FORMAT_V5
    assert detect_output_format((1, 25200, 9)) == FORMAT_V5
    # 反向：候选数在末轴的 v8
    assert detect_output_format((1, 9, 25200)) == FORMAT_V8
    # 8400x84 只能读成 v5（C=79）：候选轴 8400 够大，而 84 当候选轴太小
    assert detect_output_format((1, 8400, 84)) == FORMAT_V5
    # 小形状两个方向都说不通，交回用户
    assert detect_output_format((1, 3, 8)) is None
    assert detect_output_format((1, 600, 700)) is None


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


# ---------- 逆变换精确性（注入已知模型输出，不依赖真实模型）----------

class _FakeSession:
    """替掉真实 Session，返回注入的常量输出。"""

    def __init__(self, raw):
        self._raw = raw

    def run(self, _outputs, _feeds):
        return [self._raw]

    def get_inputs(self):
        class _In:
            name = "images"
            shape = [1, 3, 640, 640]
            type = "tensor(float)"
        return [_In()]


def _detector_with_output(raw, num_classes=80):
    from labelforge.engine.onnx_inference import OnnxDetector
    det = OnnxDetector("injected.onnx", class_names=[f"c{i}" for i in range(num_classes)],
                       output_format=FORMAT_V5, conf_threshold=0.25, iou_threshold=0.45)
    det._session = _FakeSession(raw)
    return det


def _v5_raw(model_box, obj=0.9, cls_id=2, cls_score=0.9, num_classes=80):
    """模型空间的框 (x1, y1, x2, y2) → [1, 1, 5+C] 输出张量。"""
    x1, y1, x2, y2 = model_box
    raw = np.zeros((1, 1, 5 + num_classes), dtype=np.float32)
    raw[0, 0, 0:5] = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, obj)
    raw[0, 0, 5 + cls_id] = cls_score
    return raw


def _run(det, w, h):
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    return det.detect(blank, lambda src, geom, fill: np.zeros(
        (geom.dst_h, geom.dst_w, 3), dtype=np.uint8))


def _expected(w, h, model_box):
    """按 geometry 解析地逆变换，独立于生产代码的算法。"""
    g = letterbox_geometry(w, h, 640, 640)
    x1, y1 = (model_box[0] - g.pad_x) / g.scale, (model_box[1] - g.pad_y) / g.scale
    x2, y2 = (model_box[2] - g.pad_x) / g.scale, (model_box[3] - g.pad_y) / g.scale
    x1, x2 = sorted((max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))))
    y1, y2 = sorted((max(0.0, min(float(h), y1)), max(0.0, min(float(h), y2))))
    return x1, y1, x2, y2


@pytest.mark.parametrize("w,h", [(2048, 1080), (1920, 1080), (640, 640), (320, 240), (1080, 2048)])
def test_inverse_transform_is_exact_for_real_image_sizes(w, h):
    """注入已知模型框，逆变换结果必须与解析解逐位一致。

    真实图片尺寸（2048x1080 等）下 scale/pad 都非整数，任何取整错误
    都会在这里放大成可见偏差；用真实模型跑则会被模型自身方差掩盖。
    """
    g = letterbox_geometry(w, h, 640, 640)
    model_box = (g.pad_x + 20, g.pad_y + 30, g.pad_x + 60, g.pad_y + 90)
    det = _detector_with_output(_v5_raw(model_box))

    boxes = _run(det, w, h)

    assert len(boxes) == 1, boxes
    exp = _expected(w, h, model_box)
    got = boxes[0]
    assert got["x"] == pytest.approx(exp[0], abs=1e-3)
    assert got["y"] == pytest.approx(exp[1], abs=1e-3)
    assert got["width"] == pytest.approx(exp[2] - exp[0], abs=1e-3)
    assert got["height"] == pytest.approx(exp[3] - exp[1], abs=1e-3)


def test_box_crossing_top_padding_is_clamped_not_shifted():
    """跨越上补边的框：超出部分应被钳掉，剩下的位置不能被平移。"""
    w, h = 2048, 1080
    g = letterbox_geometry(w, h, 640, 640)
    assert g.pad_y > 40, g
    model_box = (100, g.pad_y - 40, 200, g.pad_y + 40)
    det = _detector_with_output(_v5_raw(model_box))

    boxes = _run(det, w, h)

    exp = _expected(w, h, model_box)
    assert exp[1] == 0.0
    assert boxes[0]["y"] == pytest.approx(0.0, abs=1e-3)
    assert boxes[0]["height"] == pytest.approx(exp[3], abs=1e-3)


def test_box_entirely_inside_padding_is_dropped_not_clamped():
    """全在补边内的框必须丢弃：钳位后会变成贴边的假框。"""
    w, h = 2048, 1080
    g = letterbox_geometry(w, h, 640, 640)
    det = _detector_with_output(_v5_raw((100, 10, 200, g.pad_y - 10)))

    assert _run(det, w, h) == []


def test_v5_score_is_objectness_times_class_score():
    """v5 置信度是 obj × cls；误用 obj 会高估，误用 cls 会漏检。"""
    det = _detector_with_output(_v5_raw((300, 300, 340, 340), obj=0.9, cls_score=0.9))

    boxes = _run(det, 640, 640)

    assert boxes[0]["score"] == pytest.approx(0.81, abs=1e-6)
    assert boxes[0]["label"] == "c2"


def test_low_objectness_suppresses_high_class_score():
    """obj 很低时即使类别分数很高也不该报出。"""
    det = _detector_with_output(_v5_raw((300, 300, 340, 340), obj=0.05, cls_score=1.0))

    assert _run(det, 640, 640) == []


def test_full_content_area_box_maps_to_whole_image():
    """占满内容区的框应映射成整张图，这是 letterbox 边界的上界检查。"""
    w, h = 2048, 1080
    g = letterbox_geometry(w, h, 640, 640)
    det = _detector_with_output(_v5_raw((0, g.pad_y, 640, g.pad_y + g.new_h)))

    boxes = _run(det, w, h)

    assert boxes[0]["x"] == pytest.approx(0.0, abs=1e-3)
    assert boxes[0]["width"] == pytest.approx(float(w), abs=1e-3)
    assert boxes[0]["height"] == pytest.approx(float(h), abs=1e-3)


# ---------- YOLO26 端到端输出 [x1,y1,x2,y2,score,cls] ----------

def _end2end(rows, pad=0):
    """构造 (1, N, 6) 输出；pad 行用全 0 补齐，模拟 300 定长输出。"""
    arr = np.zeros((1, len(rows) + pad, 6), dtype=np.float32)
    arr[0, :len(rows)] = np.array(rows, dtype=np.float32)
    return arr


def test_detect_format_recognizes_end2end_output():
    """[1, 300, 6] 是 YOLO26 的端到端导出，不是 v5/v8。"""
    assert detect_output_format((1, 300, 6)) == FORMAT_V26


def test_detect_format_treats_six_columns_as_v8_when_class_count_fits():
    """6 列也可能是 C=2 的 v8（4+2），类别数吻合时不能抢成端到端。"""
    assert detect_output_format((1, 6, 8400), num_classes=2) == FORMAT_V8
    assert detect_output_format((1, 8400, 6), num_classes=2) == FORMAT_V8


def test_detect_format_treats_six_columns_as_v5_when_class_count_fits():
    """6 列还可能是 C=1 的 v5（5+1）。"""
    assert detect_output_format((1, 300, 6), num_classes=1) == FORMAT_V5


def test_decode_end2end_reads_boxes_score_and_class():
    raw = _end2end([
        [10.0, 20.0, 30.0, 40.0, 0.9, 2.0],
        [1.0, 2.0, 3.0, 4.0, 0.3, 0.0],
    ])
    out = decode_end2end(raw, 0.25)
    assert out["boxes"].shape == (2, 4)
    assert list(out["boxes"][0]) == [10.0, 20.0, 30.0, 40.0]
    assert list(out["scores"]) == pytest.approx([0.9, 0.3])
    assert list(out["class_ids"]) == [2, 0]
    assert out["class_ids"].dtype == np.int64


def test_decode_end2end_keeps_xyxy_unchanged():
    """端到端输出已是 xyxy，不能再当中心宽高换算。"""
    raw = _end2end([[100.0, 200.0, 140.0, 260.0, 0.8, 1.0]])
    box = decode_end2end(raw, 0.1)["boxes"][0]
    assert list(box) == [100.0, 200.0, 140.0, 260.0]


def test_decode_end2end_filters_by_confidence():
    raw = _end2end([
        [0.0, 0.0, 1.0, 1.0, 0.80, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.20, 0.0],
    ])
    out = decode_end2end(raw, 0.25)
    assert len(out["scores"]) == 1 and out["scores"][0] == pytest.approx(0.80)


def test_decode_end2end_threshold_is_inclusive():
    raw = _end2end([[0.0, 0.0, 1.0, 1.0, 0.25, 0.0]])
    assert len(decode_end2end(raw, 0.25)["scores"]) == 1


def test_decode_end2end_drops_zero_padding_rows():
    """定长输出尾部常是全 0 的占位行，不能被当成 (0,0,0,0) 的框。"""
    raw = _end2end([[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]], pad=5)
    out = decode_end2end(raw, 0.25)
    assert len(out["scores"]) == 1
    assert out["scores"][0] == pytest.approx(0.9)


def test_decode_end2end_rounds_fractional_class_ids():
    """类别 id 偶尔以浮点写出，直接 int() 会截断错位。"""
    raw = _end2end([[0.0, 0.0, 1.0, 1.0, 0.9, 2.9999]])
    assert list(decode_end2end(raw, 0.1)["class_ids"]) == [3]


def test_decode_end2end_rejects_wrong_last_dim():
    """末维不是 6 说明不是端到端导出，必须报错而不是硬解。"""
    for bad in (5, 7):
        with pytest.raises(InferenceError):
            decode_end2end(np.zeros((1, 10, bad), dtype=np.float32), 0.25)


def test_decode_end2end_rejects_out_of_range_class_id():
    """类别 id 越界说明模型与类别表对不上，早点报错好过产出空标签。"""
    raw = _end2end([[0.0, 0.0, 1.0, 1.0, 0.9, 7.0]])
    with pytest.raises(InferenceError):
        decode_end2end(raw, 0.25, num_classes=3)


def test_decode_end2end_accepts_two_dimensional_input():
    raw = np.array([[10.0, 10.0, 20.0, 20.0, 0.9, 1.0]], dtype=np.float32)
    assert len(decode_end2end(raw, 0.25)["scores"]) == 1


def test_decode_end2end_truncates_to_max_det_by_score():
    """超出 max_det 时按分数保留最高的那些，不是按原始顺序。"""
    rows = [[0.0, 0.0, 1.0, 1.0, s / 100.0, 0.0] for s in range(1, 11)]
    out = decode_end2end(_end2end(rows), 0.0, max_det=3)
    assert len(out["scores"]) == 3
    assert list(out["scores"]) == pytest.approx([0.10, 0.09, 0.08])


def test_decode_end2end_keeps_all_when_under_max_det():
    rows = [[0.0, 0.0, 1.0, 1.0, 0.5, 0.0]] * 4
    assert len(decode_end2end(_end2end(rows), 0.1, max_det=10)["scores"]) == 4


def test_decode_end2end_default_max_det_matches_export():
    assert DEFAULT_MAX_DET == 300


def test_decode_predictions_dispatches_end2end():
    raw = _end2end([[5.0, 6.0, 7.0, 8.0, 0.9, 1.0]])
    out = decode_predictions(raw, FORMAT_V26, 0.25)
    assert list(out["boxes"][0]) == [5.0, 6.0, 7.0, 8.0]


def test_decode_predictions_rejects_v26_when_classes_out_of_range():
    """走 decode_predictions 也要带上类别数校验。"""
    raw = _end2end([[0.0, 0.0, 1.0, 1.0, 0.9, 9.0]])
    with pytest.raises(InferenceError):
        decode_predictions(raw, FORMAT_V26, 0.25, num_classes=2)


def test_decode_predictions_error_message_lists_all_formats():
    with pytest.raises(InferenceError) as e:
        decode_predictions(np.zeros((1, 4, 4), dtype=np.float32), "nope", 0.25)
    msg = str(e.value)
    assert FORMAT_V5 in msg and FORMAT_V8 in msg and FORMAT_V26 in msg


def test_supported_formats_includes_v26():
    from labelforge.engine.onnx_inference import SUPPORTED_FORMATS
    assert FORMAT_V26 in SUPPORTED_FORMATS


def _end2end_raw(rows, num_cols=6):
    """(1, N, 6) 端到端输出。"""
    arr = np.zeros((1, len(rows), num_cols), dtype=np.float32)
    for i, row in enumerate(rows):
        arr[0, i, :len(row)] = row
    return arr


def _detector_v26(raw, num_classes=80):
    from labelforge.engine.onnx_inference import OnnxDetector
    det = OnnxDetector("injected.onnx", class_names=[f"c{i}" for i in range(num_classes)],
                       output_format=FORMAT_V26, conf_threshold=0.25, iou_threshold=0.45)
    det._session = _FakeSession(raw)
    return det


def test_resolve_detect_keeps_all_end2end_boxes_without_nms():
    """端到端模型不再跑 NMS：同类重叠框要原样保留，与官方一致。

    若这里偷偷补一次 NMS，重叠框会被吞掉，结果与官方导出对不上。
    """
    # 三个同类框，后两个几乎完全重合（IoU > 1e-3）；不做 NMS 就该全部保留
    raw = _end2end_raw([
        [10.0, 10.0, 60.0, 60.0, 0.90, 1.0],
        [100.0, 100.0, 150.0, 150.0, 0.80, 1.0],
        [100.2, 100.1, 150.3, 150.2, 0.70, 1.0],
    ])
    det = _detector_v26(raw)
    assert det.resolve_format() == FORMAT_V26
    boxes = _run(det, 640, 640)
    assert len(boxes) == 3, f"端到端输出不该被 NMS 去重，实际剩 {len(boxes)}"


def test_resolve_detect_still_applies_conf_threshold_for_end2end():
    raw = _end2end_raw([
        [10.0, 10.0, 60.0, 60.0, 0.90, 1.0],
        [20.0, 20.0, 70.0, 70.0, 0.05, 1.0],
    ])
    assert len(_run(_detector_v26(raw), 640, 640)) == 1


def test_resolve_detect_uses_v8_nms_for_v8_models():
    """对照组：v8 走 NMS，同类重叠框应被去重，证明上一条不是普遍行为。"""
    from labelforge.engine.onnx_inference import OnnxDetector
    # [1, 4+C, N]，C=1 -> 末维 5
    raw = np.zeros((1, 5, 3), dtype=np.float32)
    for i, (cx, cy, s) in enumerate([(35.0, 35.0, 0.9), (35.2, 35.1, 0.8)]):
        raw[0, :4, i] = (cx, cy, 50.0, 50.0)
        raw[0, 4, i] = s
    det = OnnxDetector("injected.onnx", class_names=["cat"], output_format=FORMAT_V8,
                       conf_threshold=0.25, iou_threshold=0.45)
    det._session = _FakeSession(raw)
    assert len(_run(det, 640, 640)) == 1, "v8 的同类重叠框应被 NMS 去重"


def test_detect_format_requires_many_rows_for_end2end():
    """端到端的行数是 max_det（≤300），而 v8 布局的候选轴很大且排在前。

    形状同时能用两种方式解释时（6 行 vs 6 列）不能猜成端到端：
    v26 的框数是模型导出时定死的上限，行数不会只有个位数。
    """
    assert detect_output_format((1, 6, 6)) is None
    assert detect_output_format((1, 4, 6)) is None
    # 端到端的行数是导出时定死的 max_det，行数过大只可能是别的布局
    assert detect_output_format((1, 8400, 6)) is None
    # 有类别表时 6 列正好是 C=2 的 v8
    assert detect_output_format((1, 6, 8400), num_classes=2) == FORMAT_V8
    # 真正的端到端：6 列、行数在 max_det 以内
    assert detect_output_format((1, 300, 6)) == FORMAT_V26
    assert detect_output_format((1, 100, 6)) == FORMAT_V26
    assert detect_output_format((1, DEFAULT_MAX_DET, 6)) == FORMAT_V26


# ---------- 无类别表时按轴序判布局 ----------

def test_detect_format_v5_without_class_table():
    """v5 布局 [1, 候选数, 5+C] 在没有类别表时也要能判出来。

    这是 nonmotor.onnx 那个「无法从输出 shape (1, 25200, 9) 判断模型格式」的
    回归护栏：第三方导出没有 metadata，以前一律被拒，模型根本加载不了。
    """
    assert detect_output_format((1, 25200, 9)) == FORMAT_V5    # C=4
    assert detect_output_format((1, 25200, 85)) == FORMAT_V5   # C=80
    assert detect_output_format((1, 8400, 84)) == FORMAT_V5    # 84 当不了候选轴


def test_detect_format_v8_without_class_table():
    """反向也要认：通道轴在前。"""
    assert detect_output_format((1, 9, 25200)) == FORMAT_V8    # C=5
    assert detect_output_format((1, 84, 8400)) == FORMAT_V8    # C=80


def test_detect_format_refuses_when_no_reading_makes_sense():
    """两种解释都推不出合理类别数时返回 None，交回用户显式指定。"""
    # 600x700：无论哪轴当通道，解出的 C 都远超 80
    assert detect_output_format((1, 600, 700)) is None
    # 小形状：候选数太少，当不成检测头
    assert detect_output_format((1, 3, 8)) is None
    assert detect_output_format((1, 1, 6)) is None


def test_detect_format_end2end_row_width_is_not_v5():
    """末轴恰为端到端行宽时不认 v5，避免与 v26 在形状上撞车。"""
    # 行数超过 max_det，不是 v26；但 6 是端到端行宽，也不该当成 C=1 的 v5
    assert detect_output_format((1, 8400, 6)) is None
    assert detect_output_format((1, 25200, 6)) is None
    # 真正的端到端仍照旧
    assert detect_output_format((1, 300, 6)) == FORMAT_V26
    assert detect_output_format((1, 100, 6)) == FORMAT_V26


def test_detect_format_uses_class_table_when_available():
    """有类别表时以类别数为准，轴序判据不参与。"""
    assert detect_output_format((1, 25200, 9), num_classes=4) == FORMAT_V5
    assert detect_output_format((1, 9, 25200), num_classes=5) == FORMAT_V8
    # 与类别数对不上就不认
    assert detect_output_format((1, 25200, 9), num_classes=80) is None
    # 类别数已知时不把 [1, 6, 8400] 当端到端（6 正好是 C=2 的 v8）
    assert detect_output_format((1, 6, 8400), num_classes=2) == FORMAT_V8


def test_detect_format_class_count_has_an_upper_bound():
    """解出的 C 超过上限就不能认——否则候选数会被当成类别数。

    上限是 80（COCO 量级）。有了它，(1, 25200, 200) 解出 C=195 直接不成立；
    若把上限放到很大，rows-4=25196 也会满足「>=1」，v8 那条一并成立，于是
    误判成「分不清」而拒绝加载。
    """
    assert detect_output_format((1, 25200, 9)) == FORMAT_V5     # C=4
    assert detect_output_format((1, 25200, 85)) == FORMAT_V5    # C=80，正好到顶
    # 超过 80 类：判不出来，交回用户显式指定
    assert detect_output_format((1, 25200, 200)) is None        # C=195
    assert detect_output_format((1, 25200, 605)) is None        # C=600
    assert detect_output_format((1, 25200, 10006)) is None
    # 通道轴在前时同理
    assert detect_output_format((1, 84, 8400)) == FORMAT_V8     # C=80，正好到顶
    assert detect_output_format((1, 604, 8400)) is None         # C=600
