# -*- coding: utf-8 -*-
"""四个任务的推理结果与手绘标注在**显示格式与 JSON 格式**上一致。

同样的几何走两条路径各产一份内部 box：

  手绘路径 —— `rect_to_rotation_points` / 手工字典 / 旋转后的顶点序列
  推理路径 —— `OnnxDetector.detect()`（注入的模型输出，坐标可直接对照）

比的是界面上真正出现的东西与落盘的字段：

  JSON   `labelme_shape_from_box()` 之后的 label / points / group_id /
         description / shape_type / flags
  显示   画布走哪条绘制分支、类别名文字、标签锚点（`points[0]`）、
         旋转摇杆位置、角度文字

不比像素：像素受字体光栅化影响（本机与 CI 镜像落点不同），推理坐标是
float32、手绘是 float64，差 1e-7 就会让文字整块移 1 像素，那是数值精度不是
不一致。坐标统一量化到 3 位小数（0.001px）后比。

`score` 与 `flags`（inference / pose_class）是推理专有字段，比对时剥掉。
"""
import math

import numpy as np
import pytest

from labelforge.engine.inference_task import merge_boxes
from labelforge.engine.onnx_inference import OnnxDetector
from labelforge.engine.shape_io import (
    box_display_label, labelme_shape_from_box, order_rotation_points,
    rect_to_rotation_points, rotate_points, rotation_center,
    rotation_angle_degrees, start_at_top_left,
)

IMAGE_W, IMAGE_H = 220, 180
DIGITS = 3          # 0.001px：够区分真实形状差异，又盖掉 float32/float64 噪声


# --------------------------------------------------------------------- 替身
class _Meta:
    def __init__(self, mapping):
        self.custom_metadata_map = dict(mapping)


class _Output:
    def __init__(self, shape):
        self.name = "out"
        self.shape = tuple(shape)


class _Session:
    """输入尺寸与原图相同，letterbox 退化成恒等，模型坐标可直接当原图坐标比。"""

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


def detect(shapes, metadata, head):
    detector = OnnxDetector("injected.onnx", class_names=["person"])
    detector._session = _Session(shapes, metadata, [head])
    detector._input_name = "images"
    return detector.detect(np.zeros((IMAGE_H, IMAGE_W, 3), np.uint8), _resize)


def zero_head(columns, rows=200):
    return np.zeros((1, columns, rows), dtype=np.float32)


# ----------------------------------------------------------------- 比对工具
def snap(box):
    """量化坐标，抹掉 float32 与 float64 的 1e-7 级差异。"""
    item = dict(box)
    for key in ("x", "y", "width", "height"):
        if isinstance(item.get(key), (int, float)):
            item[key] = round(float(item[key]), DIGITS)
    if item.get("points"):
        item["points"] = [[round(float(p[0]), DIGITS), round(float(p[1]), DIGITS)]
                          for p in item["points"]]
    return item


def json_form(box):
    """落盘形态：LabelMe shape，去掉推理专有字段。"""
    shape = labelme_shape_from_box(snap(box))
    shape.pop("score", None)
    flags = dict(shape.get("flags") or {})
    flags.pop("inference", None)
    flags.pop("pose_class", None)
    shape["flags"] = flags
    return shape


def draw_branch(box):
    """画布 `_draw_one_box` 的分流结果（决定手柄个数、画不画摇杆）。"""
    shape_type = box.get("shape_type") or "rectangle"
    if shape_type == "point":
        return "point"
    if shape_type == "polygon":
        return "polygon"
    if shape_type == "rotation" and len(box.get("points") or []) == 4:
        return "rotation"
    return "rectangle"


def display_form(box):
    """界面形态：标签文字、标签锚点、摇杆位置、角度文字。"""
    points = snap(box).get("points") or []
    branch = draw_branch(box)
    if points:
        anchor = list(points[0])
    else:
        anchor = [round(float(box.get("x", 0.0)), DIGITS),
                  round(float(box.get("y", 0.0)), DIGITS)]
    if branch == "rotation":
        joystick = [round(value, DIGITS) for value in rotation_center(points)]
        angle = f"{rotation_angle_degrees(points):.0f}°"
    else:
        joystick = "四角"
        angle = None
    return {
        "branch": branch,
        # 画布在左上角只画类别与置信度，角度单独画在中心
        "label_text": box_display_label(box, with_angle=False),
        "label_anchor": anchor,
        "joystick": joystick,
        "angle_text": angle,
    }


def assert_parity(manual, inferred):
    manual, inferred = snap(manual), snap(inferred)
    assert json_form(manual) == json_form(inferred), (
        f"JSON 格式不一致：\n手绘 {json_form(manual)}\n推理 {json_form(inferred)}")
    assert display_form(manual) == display_form(inferred), (
        f"显示格式不一致：\n手绘 {display_form(manual)}\n推理 {display_form(inferred)}")


# -------------------------------------------------------------------- det
def test_det_matches_hand_drawn_rectangle():
    head = zero_head(5)
    head[0, :4, 0] = [110, 90, 80, 40]          # cx, cy, w, h
    head[0, 4, 0] = 0.9
    boxes = detect([(1, 5, 200)], {"task": "detect"}, head)
    assert len(boxes) == 1
    manual = {"label": "person", "x": 70.0, "y": 70.0, "width": 80.0,
              "height": 40.0, "shape_type": "rectangle"}
    assert_parity(manual, boxes[0])


# -------------------------------------------------------------------- obb
@pytest.mark.parametrize("angle", [0.0, 0.35, math.pi / 4, 1.2, -0.7])
def test_obb_matches_hand_drawn_rotation(angle):
    """推理出的旋转框，四角顺序、标签锚点、摇杆位置都要与手绘一致。

    手绘是「先画水平框再旋转」，起点落在右下角之类的地方会很自然，所以两边
    都必须归一到「最靠左上的角起、顺时针」，否则同一个框的类别名与摇杆会跑到
    不同头上。
    """
    hand = rect_to_rotation_points(60.0, 60.0, 100.0, 60.0)
    rotated = order_rotation_points(
        rotate_points(hand, rotation_center(hand), angle))
    manual = {
        "label": "person",
        "x": min(p[0] for p in rotated),
        "y": min(p[1] for p in rotated),
        "width": max(p[0] for p in rotated) - min(p[0] for p in rotated),
        "height": max(p[1] for p in rotated) - min(p[1] for p in rotated),
        "shape_type": "rotation",
        "points": rotated,
    }

    head = zero_head(6)
    head[0, :4, 0] = [110, 90, 100, 60]
    head[0, 4, 0] = 0.9
    head[0, -1, 0] = angle
    boxes = detect([(1, 6, 200)], {"task": "obb"}, head)
    assert len(boxes) == 1
    assert_parity(manual, boxes[0])
    # 四角互相垂直（斜平行四边形会被 JSON 比对放过，这里按几何钉死）。
    # 用未量化的原始坐标 + 按边长归一的 cos：量化到 0.001px 后点积的绝对误差
    # 就有 0.05 量级，绝对阈值会误判。
    corners = boxes[0]["points"]
    edges = [(corners[(i + 1) % 4][0] - corners[i][0],
              corners[(i + 1) % 4][1] - corners[i][1]) for i in range(4)]
    lengths = [math.hypot(edge[0], edge[1]) for edge in edges]
    for i in range(4):
        nxt = edges[(i + 1) % 4]
        cos = abs(edges[i][0] * nxt[0] + edges[i][1] * nxt[1]) / (
            lengths[i] * lengths[(i + 1) % 4])
        assert cos < 1e-5, f"角度 {angle} 下第 {i} 个角不是直角：cos={cos}"
    assert sorted(lengths[:2]) == pytest.approx([60.0, 100.0], abs=1e-3), \
        f"角度 {angle} 下边长应为 60 与 100：{lengths}"
    assert lengths[2] == pytest.approx(lengths[0], abs=1e-3)
    assert lengths[3] == pytest.approx(lengths[1], abs=1e-3)


# -------------------------------------------------------------------- seg
def test_seg_polygon_anchor_matches_hand_drawn_polygon():
    """分割的多边形起点归一到最左上，标签锚点与手绘一致。

    轮廓来自 OpenCV，起点是任意一个顶点；画布把类别名画在 `points[0]`，
    不归一化就会让同类目标的标签位置随机。
    """
    contour = [[20.0, 20.0], [120.0, 30.0], [130.0, 120.0], [30.0, 110.0]]
    normalized = start_at_top_left(contour)
    manual = {"label": "person", "x": 20.0, "y": 20.0, "width": 110.0,
              "height": 100.0, "shape_type": "polygon", "points": normalized}
    # 推理侧的同一形状，只是顶点从别处开始
    inferred = {"label": "person", "x": 20.0, "y": 20.0, "width": 110.0,
                "height": 100.0, "shape_type": "polygon",
                "points": start_at_top_left(contour[2:] + contour[:2])}
    assert inferred["points"] == normalized, "起点归一化没生效"
    assert_parity(manual, inferred)


def test_start_at_top_left_tolerates_float_noise():
    """两个上角 y 只差 1e-16 时必须取更靠左的那个。

    直接比浮点会随机挑中右上角，同一个框每次旋转后标签左右跳。
    """
    points = [[10.0, 5.0 + 1e-15], [0.0, 5.0], [0.0, 15.0], [10.0, 15.0]]
    assert start_at_top_left(points)[0] == [0.0, 5.0]


# ------------------------------------------------------------------- pose
def test_pose_instance_and_keypoints_match_hand_drawn_group():
    """姿态的实例框与关键点分组号一致，标签显示成 `person [1]` / `nose [1]`。"""
    head = zero_head(11)
    head[0, :4, 0] = [110, 90, 80, 120]
    head[0, 4, 0] = 0.9
    head[0, 5:8, 0] = [90, 60, 0.9]
    head[0, 8:11, 0] = [130, 60, 0.8]
    raw = detect([(1, 11, 200)],
                 {"task": "pose", "kpt_shape": "[2, 3]",
                  "kpt_names": "{0: ['nose', 'eye']}"}, head)
    boxes = merge_boxes([], raw, replace_auto=True)

    rect = [b for b in boxes if b["shape_type"] == "rectangle"]
    points = [b for b in boxes if b["shape_type"] == "point"]
    assert len(rect) == 1 and len(points) == 2

    manual_rect = {"label": "person", "x": 70.0, "y": 30.0, "width": 80.0,
                   "height": 120.0, "shape_type": "rectangle", "group_id": 1}
    assert_parity(manual_rect, rect[0])

    for name, px in (("nose", 90.0), ("eye", 130.0)):
        got = [p for p in points if p["label"] == name]
        assert len(got) == 1, f"缺少关键点 {name}"
        manual_point = {"label": name, "x": px, "y": 60.0, "width": 0.0,
                        "height": 0.0, "shape_type": "point",
                        "points": [[px, 60.0]], "group_id": 1}
        assert_parity(manual_point, got[0])

    assert display_form(rect[0])["label_text"] == "person [1]"
    assert display_form(points[0])["label_text"] in ("nose [1]", "eye [1]")
