# -*- coding: utf-8 -*-
"""seg / obb / pose 后处理的单元测试。

全部用 numpy 常量张量与假 Session，不需要 onnxruntime，也不需要真实模型。
需要 OpenCV 的掩膜用例单独 skip。
"""
import numpy as np
import pytest

from labelforge.engine.onnx_inference import OnnxDetector, unletterbox_points
from labelforge.engine.onnx_tasks import (
    DEFAULT_MAX_DET, MASK_DIM, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG,
    TaskError, channel_count, decode_obb, decode_pose, decode_rows, decode_seg,
    detect_task_from_outputs, end2end_from_metadata, kpt_shape_from_metadata,
    obb_corners, pose_keypoint_names, task_from_metadata,
)


class _FakeMeta:
    def __init__(self, mapping):
        self.custom_metadata_map = dict(mapping)


class _FakeOut:
    def __init__(self, shape):
        self.name = "out"
        self.shape = tuple(shape)


class _FakeSession:
    """只管元数据与输出形状，用于任务识别。"""

    def __init__(self, shapes=(), metadata=None):
        self._shapes = list(shapes)
        self._meta = dict(metadata or {})

    def get_outputs(self):
        return [_FakeOut(shape) for shape in self._shapes]

    def get_modelmeta(self):
        return _FakeMeta(self._meta)


# ---------------------------------------------------------------- 任务识别
def test_task_from_metadata_maps_ultralytics_values():
    assert task_from_metadata(_FakeSession(metadata={"task": "detect"})) == TASK_DET
    assert task_from_metadata(_FakeSession(metadata={"task": "segment"})) == TASK_SEG
    assert task_from_metadata(_FakeSession(metadata={"task": "obb"})) == TASK_OBB
    assert task_from_metadata(_FakeSession(metadata={"task": "pose"})) == TASK_POSE
    assert task_from_metadata(_FakeSession(metadata={"task": "classify"})) is None
    assert task_from_metadata(_FakeSession()) is None


def test_kpt_shape_and_end2end_and_names_from_metadata():
    session = _FakeSession(metadata={
        "kpt_shape": "[17, 3]",
        "end2end": "True",
        "kpt_names": "{0: ['nose', 'left_eye']}",
    })
    assert kpt_shape_from_metadata(session) == (17, 3)
    assert end2end_from_metadata(session) is True
    assert pose_keypoint_names(session) == ["nose", "left_eye"]

    assert kpt_shape_from_metadata(_FakeSession()) is None
    assert end2end_from_metadata(_FakeSession()) is None
    assert pose_keypoint_names(_FakeSession()) == []


def test_detect_task_from_outputs_uses_proto_for_seg():
    outputs = [_FakeOut((1, 116, 8400)), _FakeOut((1, MASK_DIM, 160, 160))]
    assert detect_task_from_outputs(outputs, num_classes=80) == TASK_SEG


def test_detect_task_from_outputs_reads_det_obb_pose_from_channels():
    assert detect_task_from_outputs([_FakeOut((1, 84, 8400))], num_classes=80) == TASK_DET
    assert detect_task_from_outputs([_FakeOut((1, 20, 21504))], num_classes=15) == TASK_OBB
    assert detect_task_from_outputs([_FakeOut((1, 56, 8400))], num_classes=1) == TASK_POSE


def test_detect_task_from_outputs_returns_none_when_ambiguous():
    # 20 列：15 类 OBB 或 1 类 5 关键点姿态，没元数据分不清
    assert detect_task_from_outputs([_FakeOut((1, 20, 21504))]) is None
    assert detect_task_from_outputs([_FakeOut((1, 900, 90))]) is None


def test_channel_count_takes_shorter_axis_and_handles_degenerate():
    assert channel_count((1, 84, 8400)) == 84
    assert channel_count((1, 8400, 84)) == 84
    assert channel_count((1, 300, 38)) == 38
    assert channel_count((1, 1, 85)) == 85
    with pytest.raises(TaskError):
        channel_count((84,))


# ------------------------------------------------------------ 布局与拆分
def _seg_head_nonend(num_classes=2, anchors=200, rows=None):
    """(1, 4+C+32, N) 的非端到端分割头，第 0 行为一个可辨识的目标。"""
    channels = 4 + num_classes + MASK_DIM
    raw = np.zeros((1, channels, anchors), dtype=np.float32)
    raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = 100, 100, 40, 20
    raw[0, 4 + 1, 0] = 0.9           # 第 2 类
    raw[0, 4 + num_classes:, 0] = 1.0
    return raw


def test_decode_rows_nonend2end_keeps_center_format():
    raw = _seg_head_nonend()
    box, scores, class_ids, tail, box_xyxy = decode_rows(
        raw, TASK_SEG, num_classes=2, conf_threshold=0.25)
    assert box_xyxy is False
    assert box.shape[0] == 1
    assert np.allclose(box[0], [100, 100, 40, 20])
    assert scores[0] == pytest.approx(0.9, abs=1e-6)
    assert int(class_ids[0]) == 1
    assert tail.shape[1] == MASK_DIM


def test_decode_rows_end2end_reads_xyxy_score_and_class():
    raw = np.zeros((1, DEFAULT_MAX_DET, 4 + 2 + MASK_DIM), dtype=np.float32)
    raw[0, 0, :] = [10, 20, 110, 220, 0.8, 3.0] + [0.5] * MASK_DIM
    box, scores, class_ids, tail, box_xyxy = decode_rows(
        raw, TASK_SEG, num_classes=80, end2end=True, conf_threshold=0.25)
    assert box_xyxy is True
    assert np.allclose(box[0], [10, 20, 110, 220])
    assert scores[0] == pytest.approx(0.8, abs=1e-6)
    assert int(class_ids[0]) == 3
    assert tail.shape[1] == MASK_DIM


def test_decode_rows_resolves_end2end_by_row_count_not_channels():
    # 3 类分割：非端到端 4+3+32=39，端到端 4+2+32=38，通道数不同但同一模型族
    dense = np.zeros((1, 4 + 3 + MASK_DIM, 8400), dtype=np.float32)
    _, _, _, _, dense_xyxy = decode_rows(dense, TASK_SEG, num_classes=3)
    assert dense_xyxy is False

    sparse = np.zeros((1, 4 + 2 + MASK_DIM, DEFAULT_MAX_DET), dtype=np.float32)
    _, _, _, _, sparse_xyxy = decode_rows(sparse, TASK_SEG, num_classes=3)
    assert sparse_xyxy is True


def test_decode_rows_prefers_nonend2end_when_shape_collides():
    # 2 类分割时两种布局都是 38 列：没有元数据标记就按非端到端解
    sparse = np.zeros((1, 4 + 2 + MASK_DIM, DEFAULT_MAX_DET), dtype=np.float32)
    _, _, _, _, xyxy = decode_rows(sparse, TASK_SEG, num_classes=2)
    assert xyxy is False
    _, _, _, _, forced = decode_rows(sparse, TASK_SEG, num_classes=2, end2end=True)
    assert forced is True


def test_decode_rows_rejects_channel_mismatch():
    raw = np.zeros((1, 3, 500), dtype=np.float32)
    with pytest.raises(TaskError):
        decode_rows(raw, TASK_SEG, num_classes=2)


# ------------------------------------------------------------------ OBB
def test_obb_corners_zero_angle_matches_axis_aligned_rect():
    corners = obb_corners(100, 50, 40, 20, 0.0)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert min(xs) == pytest.approx(80.0)
    assert max(xs) == pytest.approx(120.0)
    assert min(ys) == pytest.approx(40.0)
    assert max(ys) == pytest.approx(60.0)


def test_obb_corners_quarter_turn_swaps_extent():
    corners = obb_corners(0, 0, 40, 20, np.pi / 2)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert max(xs) - min(xs) == pytest.approx(20.0, abs=1e-5)
    assert max(ys) - min(ys) == pytest.approx(40.0, abs=1e-5)


def test_decode_obb_takes_angle_from_last_channel():
    raw = np.zeros((1, 4 + 15 + 1, 900), dtype=np.float32)
    raw[0, :4, 0] = [500, 500, 100, 50]
    raw[0, 4 + 4, 0] = 0.7                     # 第 5 类
    raw[0, -1, 0] = np.pi / 4
    dets = decode_obb(raw, num_classes=15, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 4
    assert dets[0]["score"] == pytest.approx(0.7, abs=1e-6)
    assert dets[0]["angle"] == pytest.approx(np.pi / 4, abs=1e-6)
    assert len(dets[0]["corners"]) == 4
    box = dets[0]["box"]
    assert box[2] - box[0] > 100           # 旋转后外接矩形比原宽大


def test_decode_obb_end2end_row_layout():
    raw = np.zeros((1, DEFAULT_MAX_DET, 4 + 2 + 1), dtype=np.float32)
    raw[0, 0] = [300, 300, 60, 30, 0.9, 2.0, 0.0]
    dets = decode_obb(raw, num_classes=15, end2end=True, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 2
    assert dets[0]["box"] == pytest.approx([270, 285, 330, 315], abs=1e-4)


# ------------------------------------------------------------------ pose
def _pose_head_nonend(kpt=3, num_classes=1, anchors=200):
    channels = 4 + num_classes + kpt * 3
    raw = np.zeros((1, channels, anchors), dtype=np.float32)
    raw[0, :4, 0] = [100, 100, 50, 80]
    raw[0, 4, 0] = 0.8
    base = 4 + num_classes
    values = [(10, 20, 0.9), (30, 40, 0.1), (50, 60, 0.6)][:kpt]
    for index, (kx, ky, kc) in enumerate(values):
        raw[0, base + index * 3: base + index * 3 + 3, 0] = [kx, ky, kc]
    return raw


def test_decode_pose_filters_keypoints_by_confidence():
    dets = decode_pose(_pose_head_nonend(), 3, num_classes=1, conf_threshold=0.25,
                       kpt_conf=0.5)
    assert len(dets) == 1
    kpts = dets[0]["keypoints"]
    assert len(kpts) == 3
    assert kpts[0][2] == pytest.approx(0.9, abs=1e-6)
    assert kpts[1][2] is None          # 低于阈值：保留坐标，标记为不可见
    assert kpts[2][2] == pytest.approx(0.6, abs=1e-6)


def test_decode_pose_box_is_xyxy_for_end2end_and_center_otherwise():
    dense = decode_pose(_pose_head_nonend(), 3, num_classes=1, conf_threshold=0.25)
    assert dense[0]["box"] == pytest.approx([75, 60, 125, 140], abs=1e-4)

    raw = np.zeros((1, DEFAULT_MAX_DET, 4 + 2 + 9), dtype=np.float32)
    raw[0, 0] = [10, 20, 60, 90, 0.9, 0.0, 1, 2, 0.9, 3, 4, 0.9, 5, 6, 0.9]
    sparse = decode_pose(raw, 3, num_classes=1, end2end=True, conf_threshold=0.25)
    assert sparse[0]["box"] == pytest.approx([10, 20, 60, 90], abs=1e-4)


def test_decode_pose_requires_keypoint_count():
    with pytest.raises(TaskError):
        decode_pose(_pose_head_nonend(), 0, num_classes=1)


# ------------------------------------------------------------------- seg
def test_decode_seg_falls_back_when_contour_missing():
    pytest.importorskip("cv2")
    raw = _seg_head_nonend()
    protos = np.zeros((1, MASK_DIM, 40, 40), dtype=np.float32)
    # 系数与原型全零：概率恒为 0.5，二值化后无前景，取不出轮廓
    dets = decode_seg(raw, protos, num_classes=2, input_size=(160, 160),
                      conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0]["points"] is None
    assert dets[0]["box"] == pytest.approx([80, 90, 120, 110], abs=1e-4)


def test_decode_seg_extracts_square_polygon():
    pytest.importorskip("cv2")
    raw = _seg_head_nonend(num_classes=2, anchors=200)
    protos = np.zeros((1, MASK_DIM, 40, 40), dtype=np.float32)
    protos[0, 0] = 1.0                 # 第 0 个原型全 1
    raw[0, 4 + 2:, 0] = 0.0
    raw[0, 4 + 2, 0] = 12.0            # 系数×原型 -> logits=12 -> 概率≈1
    dets = decode_seg(raw, protos, num_classes=2, input_size=(160, 160),
                      conf_threshold=0.25)
    assert len(dets) == 1
    points = dets[0]["points"]
    assert points is not None and len(points) >= 4
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    # 框是 [80, 90, 120, 110]（cxcywh 128..72? 见下）；掩膜全 1 时轮廓应铺满框
    assert min(xs) == pytest.approx(80, abs=2)
    assert max(xs) == pytest.approx(120, abs=2)
    assert min(ys) == pytest.approx(90, abs=2)
    assert max(ys) == pytest.approx(110, abs=2)


# --------------------------------------------------- 逆变换与内部 box 组装
def test_unletterbox_points_applies_same_transform_as_boxes():
    from labelforge.engine.onnx_inference import letterbox_geometry
    geometry = letterbox_geometry(1280, 720, 640, 640)
    points = unletterbox_points([[000, 000], [640, 640]], geometry, 1280, 720)
    assert points[0][0] == pytest.approx(0, abs=1e-3)
    assert points[1][0] == pytest.approx(1280, abs=1e-3)
    assert points[1][1] == pytest.approx(720, abs=1e-3)


class _TaskSession:
    """给 OnnxDetector 用：内置元数据 + 可注入输出。"""

    def __init__(self, shapes, metadata, outputs):
        self._shapes = list(shapes)
        self._meta = dict(metadata)
        self._outputs = list(outputs)

    def get_inputs(self):
        class _In:
            name = "images"
            shape = [1, 3, 640, 640]
            type = "tensor(float)"
        return [_In()]

    def get_outputs(self):
        return [_FakeOut(shape) for shape in self._shapes]

    def get_modelmeta(self):
        return _FakeMeta(self._meta)

    def run(self, _outputs, _feeds):
        return list(self._outputs)


def _resize(src, geometry, fill):
    return np.full((geometry.dst_h, geometry.dst_w, 3), fill, dtype=np.uint8)


def _detector(session, task="auto"):
    det = OnnxDetector("injected.onnx", class_names=["person"], task=task)
    det._session = session
    det._input_name = "images"
    return det


def test_resolve_task_prefers_metadata_over_shape():
    session = _TaskSession([(1, 56, 8400)], {"task": "pose", "kpt_shape": "[17, 3]"}, [])
    det = _detector(session)
    assert det.resolve_task() == TASK_POSE
    assert det.kpt_shape() == (17, 3)


def test_resolve_task_explicit_format_means_det():
    # 80 类检测 85 列，与「80 类 OBB」同形；显式选过格式就按检测解
    session = _TaskSession([(1, 85, 8400)], {}, [])
    det = OnnxDetector("injected.onnx", class_names=[f"c{i}" for i in range(80)],
                       output_format="v5")
    det._session = session
    assert det.resolve_task() == TASK_DET


def test_set_task_switches_postprocess_without_reload():
    session = _TaskSession([(1, 900, 900)], {}, [])
    det = _detector(session)
    assert det.resolve_task() == TASK_DET      # 形状上推不出 -> 退回 det
    det.set_task(TASK_POSE)
    assert det.resolve_task() == TASK_POSE
    det.set_task("auto")
    assert det.resolve_task() == TASK_DET


def test_set_task_rejects_unknown():
    det = OnnxDetector("injected.onnx")
    with pytest.raises(Exception):
        det.set_task("segment")


def test_detect_pose_emits_instance_and_grouped_keypoints():
    kpt = 2
    head = np.zeros((1, 4 + 1 + kpt * 3, 200), dtype=np.float32)
    head[0, :4, 0] = [320, 320, 64, 128]
    head[0, 4, 0] = 0.9
    head[0, 5:8, 0] = [300, 300, 0.9]
    head[0, 8:11, 0] = [340, 300, 0.8]
    session = _TaskSession([(1, head.shape[1], head.shape[2])],
                           {"task": "pose", "kpt_shape": "[2, 3]",
                            "kpt_names": "{0: ['nose', 'ear']}"},
                           [head])
    det = _detector(session)
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)

    rects = [b for b in boxes if b["shape_type"] == "rectangle"]
    points = [b for b in boxes if b["shape_type"] == "point"]
    assert len(rects) == 1 and len(points) == 2
    assert rects[0]["label"] == "person"
    assert rects[0]["pose_class"] == "person"
    assert rects[0]["flags"]["pose_class"] == "person"
    assert {b["label"] for b in points} == {"nose", "ear"}
    assert all(b["pose_group"] == rects[0]["pose_group"] for b in points)


def test_detect_pose_nms_keeps_keypoints_with_their_instance():
    kpt = 1
    head = np.zeros((1, 4 + 1 + kpt * 3, 200), dtype=np.float32)
    # 两个高度重合的实例：NMS 只该留一个，另一个的关键点不能变成孤儿
    for row, score in ((0, 0.9), (1, 0.8)):
        head[0, :4, row] = [320, 320, 64, 128]
        head[0, 4, row] = score
        head[0, 5:8, row] = [300, 300, 0.9]
    session = _TaskSession([(1, head.shape[1], head.shape[2])],
                           {"task": "pose", "kpt_shape": "[1, 3]"}, [head])
    det = _detector(session)
    det.iou_threshold = 0.5
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)
    rects = [b for b in boxes if b["shape_type"] == "rectangle"]
    points = [b for b in boxes if b["shape_type"] == "point"]
    assert len(rects) == 1
    assert len(points) == 1
    assert points[0]["pose_group"] == rects[0]["pose_group"]


def test_detect_seg_returns_polygon():
    pytest.importorskip("cv2")
    channels = 4 + 1 + MASK_DIM
    head = np.zeros((1, channels, 200), dtype=np.float32)
    head[0, :4, 0] = [320, 320, 128, 128]
    head[0, 4, 0] = 0.9
    protos = np.zeros((1, MASK_DIM, 40, 40), dtype=np.float32)
    protos[0, 0] = 1.0
    head[0, 5, 0] = 12.0
    session = _TaskSession([(1, channels, 200), (1, MASK_DIM, 40, 40)],
                           {"task": "segment"}, [head, protos])
    det = _detector(session)
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)
    assert len(boxes) == 1
    assert boxes[0]["shape_type"] == "polygon"
    assert len(boxes[0]["points"]) >= 4
    assert boxes[0]["label"] == "person"


def test_detect_obb_returns_rotation():
    channels = 4 + 1 + 1
    head = np.zeros((1, channels, 200), dtype=np.float32)
    head[0, :4, 0] = [320, 320, 100, 40]
    head[0, 4, 0] = 0.9
    head[0, -1, 0] = np.pi / 6
    session = _TaskSession([(1, channels, 200)], {"task": "obb"}, [head])
    det = _detector(session)
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)
    assert len(boxes) == 1
    assert boxes[0]["shape_type"] == "rotation"
    assert len(boxes[0]["points"]) == 4
    assert boxes[0]["width"] > 100        # 斜置后外接矩形变宽


def test_obb_corners_stay_perpendicular_at_any_angle():
    """四角必须是矩形，不是任意四边形。

    高向量漏掉负号会让四个角拼成斜平行四边形：45° 时还会退化成一条直线。
    用点积钉死「相邻边垂直 + 对边等长」，形状类用例只查外接矩形是查不出来的。
    """
    for angle in (0.0, 0.3, np.pi / 4, np.pi / 2, 1.1, 2.0, -0.7):
        corners = obb_corners(100.0, 100.0, 80.0, 40.0, angle)
        edges = [(corners[(i + 1) % 4][0] - corners[i][0],
                  corners[(i + 1) % 4][1] - corners[i][1]) for i in range(4)]
        for i in range(4):
            first, second = edges[i], edges[(i + 1) % 4]
            dot = first[0] * second[0] + first[1] * second[1]
            assert abs(dot) < 1e-6, f"角度 {angle} 下第 {i} 个角不是直角：点积 {dot}"
        lengths = [float(np.hypot(e[0], e[1])) for e in edges]
        # 输出顺序是「左上 → 左下 → 右下 → 右上」，所以边长序列是 高、宽、高、宽
        assert lengths[0] == pytest.approx(40.0, abs=1e-4), f"角度 {angle} 下首边 {lengths}"
        assert lengths[1] == pytest.approx(80.0, abs=1e-4), f"角度 {angle} 下次边 {lengths}"
        assert lengths[2] == pytest.approx(lengths[0], abs=1e-4)
        assert lengths[3] == pytest.approx(lengths[1], abs=1e-4)


def test_obb_corners_rotated_45_degrees_is_a_diamond_not_a_line():
    corners = obb_corners(0.0, 0.0, 100.0, 100.0, np.pi / 4)
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    assert len({(round(x, 4), round(y, 4)) for x, y in corners}) == 4, "四点重合了"
    assert max(xs) - min(xs) == pytest.approx(141.42, abs=0.1)
    assert max(ys) - min(ys) == pytest.approx(141.42, abs=0.1)


def test_unletterbox_points_can_skip_clipping_for_rotation():
    """旋转框不逐点钳制：贴边时钳制会把矩形剪成四边形。"""
    from labelforge.engine.onnx_inference import letterbox_geometry

    geometry = letterbox_geometry(100, 100, 100, 100)   # 1:1，无补边
    raw = [[-20.0, 10.0], [30.0, 10.0], [30.0, 20.0], [-20.0, 20.0]]
    clipped = unletterbox_points(raw, geometry, 100, 100)
    assert clipped[0][0] == pytest.approx(0.0), "默认仍要钳制"
    kept = unletterbox_points(raw, geometry, 100, 100, clip=False)
    assert kept[0][0] == pytest.approx(-20.0), "旋转框要保留图外坐标"


def test_detect_obb_keeps_right_angles_when_box_crosses_image_edge():
    """框有一角跑到图外时，四个角仍必须互相垂直。

    曾经的 bug：逆变换逐点钳制到图内，贴边的旋转框被剪成不规则四边形。
    """
    channels = 4 + 1 + 1
    head = np.zeros((1, channels, 200), dtype=np.float32)
    head[0, :4, 0] = [12, 12, 60, 40]      # 中心贴左上角
    head[0, 4, 0] = 0.9
    head[0, -1, 0] = 0.6                   # 约 34°，四角必然有负坐标
    session = _TaskSession([(1, channels, 200)], {"task": "obb"}, [head])
    det = _detector(session)
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)
    assert len(boxes) == 1
    corners = boxes[0]["points"]
    edges = [(corners[(i + 1) % 4][0] - corners[i][0],
              corners[(i + 1) % 4][1] - corners[i][1]) for i in range(4)]
    lengths = [float(np.hypot(e[0], e[1])) for e in edges]
    for i in range(4):
        nxt = edges[(i + 1) % 4]
        cos = abs(edges[i][0] * nxt[0] + edges[i][1] * nxt[1]) / (
            lengths[i] * lengths[(i + 1) % 4])
        assert cos < 1e-4, f"第 {i} 个角不是直角：cos={cos}"
    assert min(p[0] for p in corners) < 0, "这个用例必须真的把角点顶到图外"


def test_detect_obb_puts_the_top_left_corner_first():
    """推理出的旋转框，起点要与手绘一致（最靠左上），标签与手柄才对得上。"""
    channels = 4 + 1 + 1
    head = np.zeros((1, channels, 200), dtype=np.float32)
    head[0, :4, 0] = [320, 240, 160, 60]
    head[0, 4, 0] = 0.9
    head[0, -1, 0] = np.pi / 3          # 60°，起点必然不在视觉左上
    session = _TaskSession([(1, channels, 200)], {"task": "obb"}, [head])
    det = _detector(session)
    boxes = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)

    assert len(boxes) == 1
    corners = boxes[0]["points"]
    top = min(range(4), key=lambda i: (corners[i][1], corners[i][0]))
    assert top == 0, f"起点不是最靠左上的角：{corners}"

    # 摇杆画在矩形中心：与手绘旋转框同一位置
    from labelforge.engine.shape_io import rotation_center
    center = rotation_center(corners)
    cx = sum(p[0] for p in corners) / 4
    cy = sum(p[1] for p in corners) / 4
    assert center == pytest.approx((cx, cy)), f"中心算错：{center} vs {(cx, cy)}"


def test_detect_obb_matches_hand_drawn_winding_and_start():
    """水平框时，推理结果的四角要与手绘完全一致。

    手绘是 `[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]`（左上起、顺时针）。
    绕向反了、或起点错位，画布的标签与旋转手柄就会与手绘的对不上。
    """
    from labelforge.engine.shape_io import rect_to_rotation_points

    channels = 4 + 1 + 1
    head = np.zeros((1, channels, 200), dtype=np.float32)
    head[0, :4, 0] = [320, 240, 160, 60]
    head[0, 4, 0] = 0.9
    head[0, -1, 0] = 0.0
    session = _TaskSession([(1, channels, 200)], {"task": "obb"}, [head])
    det = _detector(session)
    corners = det.detect(np.zeros((480, 640, 3), dtype=np.uint8), _resize)[0]["points"]

    # 模型坐标 (320,240) 在 640x640 画布上、源图 640x480 补边 80 → 原图 (320,160)
    expected = rect_to_rotation_points(320 - 80, 160 - 30, 160, 60)
    for got, want in zip(corners, expected):
        assert got[0] == pytest.approx(want[0], abs=1e-4), f"{corners} != {expected}"
        assert got[1] == pytest.approx(want[1], abs=1e-4), f"{corners} != {expected}"
