"""OBB 旋转框用多边形精确 IoU 做 NMS。

外接矩形 IoU 会把两根交叉的长条旋转框当成同一个（它们的 AABB 几乎重合），
导致其中一根被误抑制；跨分块合并时同一框的 AABB 稍差一点又去重不掉。
旋转框必须按真实几何算交并比。
"""
import math

import numpy as np
import pytest

from labelforge.engine.inference_scope import merge_scope_results
from labelforge.engine.onnx_inference import OnnxDetector, nms_per_class
from labelforge.engine.onnx_tasks import nms_rotated, obb_corners, rotated_iou

IMAGE_W, IMAGE_H = 220, 180


# --------------------------------------------------------------------- 替身
class _Meta:
    def __init__(self, mapping):
        self.custom_metadata_map = dict(mapping)


class _Output:
    def __init__(self, shape):
        self.name = "out"
        self.shape = tuple(shape)


class _Session:
    def __init__(self, shapes, metadata, outputs):
        self._shapes = list(shapes)
        self._meta = dict(metadata)
        self._outputs = list(outputs)

    def get_inputs(self):
        class _Input:
            name = "images"
            shape = [1, 3, IMAGE_H, IMAGE_W]
            type = "tensor(float)"
        return [_Input()]

    def get_outputs(self):
        return [_Output(shape) for shape in self._shapes]

    def get_modelmeta(self):
        return _Meta(self._meta)

    def run(self, _outputs, _feeds):
        return list(self._outputs)


def _resize(src, geometry, fill):
    return np.full((geometry.dst_h, geometry.dst_w, 3), fill, dtype=np.uint8)


def detect_obb(head):
    detector = OnnxDetector("injected.onnx", class_names=["person"])
    detector._session = _Session([(1, 6, 200)], {"task": "obb"}, [head])
    detector._input_name = "images"
    return detector.detect(np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8), _resize)


def rotation_item(points, score, class_id=0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {
        "x": min(xs), "y": min(ys),
        "width": max(xs) - min(xs), "height": max(ys) - min(ys),
        "shape_type": "rotation", "points": [list(p) for p in points],
        "score": score, "class_id": class_id, "label": "person",
    }


# --------------------------------------------------------------- rotated_iou
def test_identical_quad_iou_is_one():
    quad = obb_corners(50.0, 50.0, 40.0, 20.0, 0.3)
    assert rotated_iou(quad, quad) == pytest.approx(1.0)


def test_disjoint_quads_iou_is_zero():
    a = obb_corners(10.0, 10.0, 10.0, 10.0, 0.0)
    b = obb_corners(100.0, 100.0, 10.0, 10.0, 0.0)
    assert rotated_iou(a, b) == 0.0


def test_half_overlapping_squares_iou_is_one_third():
    """10x10 两个方块沿 x 错开 5：交 50、并 150，IoU = 1/3。"""
    a = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    b = [[5.0, 0.0], [15.0, 0.0], [15.0, 10.0], [5.0, 10.0]]
    assert rotated_iou(a, b) == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_zero_area_quad_returns_zero_without_raising():
    """退化成线段的框不能抛异常，也不能被算成交集。"""
    line = [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 0.0]]
    square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    assert rotated_iou(line, square) == 0.0
    assert rotated_iou(square, line) == 0.0


def test_iou_drops_as_same_square_rotates():
    """同中心同一方块：转 10° 的交集要大于转 45°。"""
    base = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

    def rotate(points, angle):
        cx = sum(p[0] for p in points) / 4.0
        cy = sum(p[1] for p in points) / 4.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return [[cx + (x - cx) * cos_a - (y - cy) * sin_a,
                 cy + (x - cx) * sin_a + (y - cy) * cos_a]
                for x, y in points]

    small = rotated_iou(base, rotate(base, math.radians(10)))
    large = rotated_iou(base, rotate(base, math.radians(45)))
    assert 0.0 < large < small < 1.0


# --------------------------------------------------------------- nms_rotated
def test_crossing_thin_boxes_are_not_suppressed():
    """判据：两根 ±45° 长条框交叉，AABB 几乎重合、实际 IoU 只有 ~0.02。

    外接矩形 NMS 会删掉其中一根（AABB IoU ≈ 1），旋转 IoU 必须都保留。
    """
    a = obb_corners(50.0, 50.0, 100.0, 4.0, math.pi / 4)
    b = obb_corners(50.0, 50.0, 100.0, 4.0, -math.pi / 4)
    items = [rotation_item(a, 0.9), rotation_item(b, 0.8)]

    # 旧口径的对照证据：外接矩形 IoU 高到会被当成同一个
    boxes = np.array([[it["x"], it["y"], it["x"] + it["width"],
                       it["y"] + it["height"]] for it in items], dtype=np.float32)
    assert len(nms_per_class(boxes, np.array([0.9, 0.8], np.float32),
                             np.array([0, 0], np.int64), 0.45)) == 1

    kept = nms_rotated(items, 0.45)
    assert sorted(kept) == [0, 1]


def test_duplicate_rotation_boxes_keep_only_the_best():
    a = obb_corners(50.0, 50.0, 40.0, 20.0, 0.3)
    items = [rotation_item(a, 0.9), rotation_item(a, 0.6)]
    kept = nms_rotated(items, 0.45)
    assert kept == [0]


def test_different_classes_never_suppress_each_other():
    a = obb_corners(50.0, 50.0, 40.0, 20.0, 0.3)
    items = [rotation_item(a, 0.9, class_id=0), rotation_item(a, 0.8, class_id=1)]
    assert sorted(nms_rotated(items, 0.45)) == [0, 1]


def test_non_quad_items_fall_back_to_axis_aligned_iou():
    """顶点缺失或不是四角时退回外接矩形，不能崩。"""
    items = [
        {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0,
         "shape_type": "rectangle", "score": 0.9, "class_id": 0},
        {"x": 1.0, "y": 1.0, "width": 10.0, "height": 10.0,
         "shape_type": "rectangle", "score": 0.8, "class_id": 0},
    ]
    assert nms_rotated(items, 0.45) == [0]


# ------------------------------------------------------- detect() 集成
def test_detect_obb_keeps_crossing_boxes():
    head = np.zeros((1, 6, 200), dtype=np.float32)
    head[0, :4, 0] = [110.0, 90.0, 100.0, 4.0]
    head[0, 4, 0] = 0.9
    head[0, 5, 0] = math.pi / 4
    head[0, :4, 1] = [110.0, 90.0, 100.0, 4.0]
    head[0, 4, 1] = 0.8
    head[0, 5, 1] = -math.pi / 4
    boxes = detect_obb(head)
    assert len(boxes) == 2, boxes
    assert all(b["shape_type"] == "rotation" for b in boxes)


def test_detect_obb_suppresses_duplicate_boxes():
    head = np.zeros((1, 6, 200), dtype=np.float32)
    head[0, :4, 0] = [110.0, 90.0, 100.0, 40.0]
    head[0, 4, 0] = 0.9
    head[0, 5, 0] = 0.3
    head[0, :4, 1] = [111.0, 90.0, 100.0, 40.0]
    head[0, 4, 1] = 0.85
    head[0, 5, 1] = 0.31
    assert len(detect_obb(head)) == 1


# --------------------------------------------------- 跨块合并（scope）
def test_merge_scope_keeps_crossing_rotations():
    a = rotation_item(obb_corners(50.0, 50.0, 100.0, 4.0, math.pi / 4), 0.9)
    b = rotation_item(obb_corners(50.0, 50.0, 100.0, 4.0, -math.pi / 4), 0.8)
    merged = merge_scope_results([(0, 0, [a]), (0, 0, [b])], 0.45)
    assert len(merged) == 2


def test_merge_scope_dedupes_same_rotation_across_chunks():
    """同一旋转框出现在两个分块（偏移一致）时只留一个。"""
    a = rotation_item(obb_corners(50.0, 50.0, 40.0, 20.0, 0.3), 0.9)
    b = rotation_item(obb_corners(50.0, 50.0, 40.0, 20.0, 0.3), 0.85)
    merged = merge_scope_results([(0, 0, [a]), (0, 0, [b])], 0.45)
    assert len(merged) == 1


def test_merge_scope_keeps_rotations_far_apart():
    a = rotation_item(obb_corners(50.0, 50.0, 40.0, 20.0, 0.3), 0.9)
    b = rotation_item(obb_corners(50.0, 50.0, 40.0, 20.0, 0.3), 0.85)
    merged = merge_scope_results([(0, 0, [a]), (200, 200, [b])], 0.45)
    assert len(merged) == 2


def test_merge_scope_still_dedupes_axis_aligned_boxes():
    a = {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0,
         "shape_type": "rectangle", "score": 0.9, "class_id": 0}
    b = {"x": 1.0, "y": 1.0, "width": 10.0, "height": 10.0,
         "shape_type": "rectangle", "score": 0.8, "class_id": 0}
    assert len(merge_scope_results([(0, 0, [a]), (0, 0, [b])], 0.45)) == 1
