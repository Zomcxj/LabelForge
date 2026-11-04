"""seg / obb / pose 三种推理任务的后处理：布局识别、解码与几何转换。

只做纯计算：不导入 Qt，不碰会话，坐标一律停在**模型输入坐标系**（letterbox
之后）。逆变换、类别名映射与内部 box 组装由 `onnx_inference` 负责。

两条导出线的差异是硬事实，见 `decode_*` 的说明：
非端到端（v8 / v11）的框是 cx,cy,w,h，端到端（v26）是 x1,y1,x2,y2，
但 v26 的 OBB 例外——旋转框只能表达成中心 + 宽高 + 角度。
"""
import ast
import math
from typing import List, Optional, Tuple

import numpy as np

TASK_DET = "det"
TASK_SEG = "seg"
TASK_OBB = "obb"
TASK_POSE = "pose"
SUPPORTED_TASKS = (TASK_DET, TASK_SEG, TASK_OBB, TASK_POSE)

#: 分割头的掩膜系数通道数，ultralytics 导出固定为 32
MASK_DIM = 32
#: 关键点每点的维数（x, y, 可见度）
KPT_DIM = 3
#: 关键点低于该置信度不落盘
DEFAULT_KPT_CONF = 0.25
#: 掩膜二值化阈值
MASK_THRESHOLD = 0.5
#: 端到端模型的默认最大检出数
DEFAULT_MAX_DET = 300
#: 类别数未知时，非端到端 OBB 末尾预留的分数列数
TASK_SCORE_SLACK = 1

#: 元数据里的 task 取值到内部任务名的映射
_TASK_ALIASES = {
    "detect": TASK_DET,
    "detection": TASK_DET,
    "det": TASK_DET,
    "segment": TASK_SEG,
    "segmentation": TASK_SEG,
    "seg": TASK_SEG,
    "obb": TASK_OBB,
    "pose": TASK_POSE,
    "keypoint": TASK_POSE,
    "keypoints": TASK_POSE,
}


class TaskError(RuntimeError):
    """任务识别失败或输出布局与任务不符。"""


def _metadata(session):
    try:
        return session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        return {}


def task_from_metadata(session):
    """读模型自带的 task 字段；没有或不认识返回 None。"""
    raw = str(_metadata(session).get("task") or "").strip().lower()
    return _TASK_ALIASES.get(raw)


def kpt_shape_from_metadata(session):
    """读 kpt_shape（如 [17, 3]）；没有或格式不对返回 None。"""
    raw = _metadata(session).get("kpt_shape")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 1:
        return None
    try:
        count = int(parsed[0])
        dim = int(parsed[1]) if len(parsed) > 1 else KPT_DIM
    except (TypeError, ValueError):
        return None
    return (count, dim) if count > 0 else None


def end2end_from_metadata(session):
    """读 end2end 标记；没有该键返回 None（交给形状判断）。"""
    raw = _metadata(session).get("end2end")
    if raw is None:
        return None
    return str(raw).strip().lower() in ("true", "1", "yes")


def pose_keypoint_names(session):
    """读 kpt_names（如 {0: ['nose', ...]}）；没有返回空表。"""
    raw = _metadata(session).get("kpt_names")
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        parsed = next(iter(parsed.values()), None)
    if isinstance(parsed, (list, tuple)):
        return [str(name) for name in parsed]
    return []


def proto_index(outputs):
    """找掩膜原型输出：4 维且通道数等于 MASK_DIM。返回 (下标, shape) 或 None。"""
    for index, out in enumerate(outputs):
        shape = tuple(int(d) for d in out.shape)
        if len(shape) == 4 and shape[1] == MASK_DIM:
            return index, shape
    return None


def _head_shapes(outputs):
    return [(index, tuple(int(d) for d in out.shape))
            for index, out in enumerate(outputs) if len(out.shape) == 3]


def channel_count(shape):
    """3 维输出的通道数。

    非端到端是 [1, 4+C+tail, N]，端到端是 [1, N, 4+2+tail]；两种布局的通道轴
    都远小于候选轴（候选数通常 ≥ 300），取两轴中的小值即可，不必先判断谁是
    通道轴。任一轴为 1 时（单候选或单通道的退化输出）取另一个。
    """
    dims = [int(d) for d in shape]
    if len(dims) == 3 and dims[0] == 1:
        dims = dims[1:]
    if len(dims) != 2:
        raise TaskError(f"输出维度不支持：{shape}")
    first, second = dims
    if first == 1 or second == 1:
        return second if first == 1 else first
    return min(first, second)


def tail_width(task, kpt_count=0):
    """各任务在类别分数之后的额外通道数。"""
    if task == TASK_SEG:
        return MASK_DIM
    if task == TASK_OBB:
        return 1
    if task == TASK_POSE:
        return KPT_DIM * int(kpt_count)
    return 0


def detect_task_from_outputs(outputs, num_classes=None, kpt_shape=None):
    """按输出形状推任务；推不出返回 None（调用方须要求显式指定）。

    有 4 维原型输出就是分割；否则拿唯一的 3 维头，用「通道数 = 4 + 类别数 +
    额外通道」反推。

    **只有类别数已知才能判 OBB / 姿态**：4+C 与 4+C+1 只差一列（80 类检测
    85 列 == 80 类 OBB 85 列），类别数未知时这两者的候选会大面积撞车，猜错
    就是把整批标注解成旋转框。所以类别数未知时只认端到端形态，其余一律
    交给产出 None 或退回检测。
    """
    if proto_index(outputs) is not None:
        return TASK_SEG
    heads = _head_shapes(outputs)
    if len(heads) != 1:
        return None
    channels = channel_count(heads[0][1])
    if not (num_classes or 0) > 0:
        if kpt_shape:
            count = int(kpt_shape[0])
            if channels == 4 + 2 + KPT_DIM * count:
                return TASK_POSE
        if channels == 4 + 2 + MASK_DIM:
            return TASK_SEG
        return None
    if channels == 4 + num_classes:
        return TASK_DET
    if channels == 4 + num_classes + 1:
        return TASK_OBB
    rest = channels - 4 - num_classes
    if rest > 0 and rest % KPT_DIM == 0:
        return TASK_POSE
    return None


def _as_rows(raw, name):
    """(1, A, B) 或 (A, B) 统一成「行是检出」的 2D 布局。

    通道轴一定更短，所以「行数 < 列数」时转置。与检测头同一套判据。
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise TaskError(f"{name} 输出 batch={arr.shape[0]}，仅支持 batch=1")
        arr = arr[0]
    if arr.ndim != 2:
        raise TaskError(f"{name} 输出维度不支持：{arr.shape}")
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    return arr


def _resolve_end2end(end2end, channels, rows, num_classes, task, kpt_count):
    """端到端与否：显式标记优先，否则按通道数 + 行数分辨率判别。

    两种布局的通道数可能撞车（如 2 类分割：非端到端 4+2+32=38，端到端也是 38），
    但行数不会：端到端固定输出 max_det 行（300），非端到端是锚点数（数千）。
    """
    tail = tail_width(task, kpt_count)
    end2 = 4 + 2 + tail
    nonend = (4 + int(num_classes) + tail) if (num_classes or 0) > 0 else None
    if end2end is not None:
        return bool(end2end)
    if rows > DEFAULT_MAX_DET:
        return False
    if channels == end2 and channels != nonend:
        return True
    if nonend is not None and channels == nonend:
        return False
    return channels == end2


def decode_rows(raw, task, num_classes=None, kpt_count=0, end2end=None,
                conf_threshold=0.25, max_det=DEFAULT_MAX_DET):
    """按任务拆出 (框, 分数, 类别 id, 尾巴通道)。

    返回的框在**模型输入坐标系**：非端到端是 cx,cy,w,h，端到端是 x1,y1,x2,y2。
    尾巴通道：seg 是 32 个掩膜系数，obb 是 1 个弧度角，pose 是 K*3 个关键点。
    """
    arr = _as_rows(raw, task)
    channels = arr.shape[1]
    tail = tail_width(task, kpt_count)
    is_end2end = _resolve_end2end(end2end, channels, arr.shape[0],
                                  num_classes, task, kpt_count)

    if is_end2end:
        need = 4 + 2 + tail
        if channels != need:
            raise TaskError(
                f"{task} 端到端输出末维应为 {need}（4+分数+类别+{tail}），"
                f"实际 {channels}")
        box = arr[:, :4].astype(np.float32, copy=True)
        scores = arr[:, 4].astype(np.float32, copy=True)
        class_ids = np.rint(arr[:, 5]).astype(np.int64)
        tail_values = arr[:, 6:].astype(np.float32, copy=True)
        box_xyxy = True
    else:
        if (num_classes or 0) > 0:
            tail = channels - 4 - int(num_classes)
            if tail < 0:
                raise TaskError(
                    f"{task} 输出末维 {channels} 装不下 {num_classes} 个类别")
        elif task == TASK_OBB and channels >= 4 + TASK_SCORE_SLACK + 1:
            # 类别数未知时最末一列固定是角度，前 4 列是框，中间全是类别分数。
            # 按 a-z 顺序猜类别数在这类模型上必错（自定义类别名不排序），
            # 所以取「预留一列给分数」的上界，宁可能力弱一点也不能解错布局。
            tail = 1
        else:
            tail = tail_width(task, kpt_count)
        classes = channels - 4 - tail
        if classes <= 0:
            raise TaskError(f"{task} 输出末维 {channels} 至少要 4+1+{tail}")
        box = arr[:, :4].astype(np.float32, copy=True)
        cls_scores = arr[:, 4:4 + classes]
        class_ids = cls_scores.argmax(axis=1).astype(np.int64)
        scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
        tail_values = arr[:, 4 + classes:].astype(np.float32, copy=True)
        box_xyxy = False

    keep = scores >= float(conf_threshold)
    box, scores = box[keep], scores[keep]
    class_ids, tail_values = class_ids[keep], tail_values[keep]

    if int(max_det) > 0 and box.shape[0] > int(max_det) and is_end2end:
        top = np.argsort(-scores, kind="stable")[:int(max_det)]
        box, scores = box[top], scores[top]
        class_ids, tail_values = class_ids[top], tail_values[top]
    return box, scores, class_ids, tail_values, box_xyxy


def cxcywh_to_xyxy(box):
    """(N,4) 中心宽高 -> 左上右下；输入不被改写。"""
    out = np.array(box, dtype=np.float32, copy=True)
    if out.size:
        cx, cy = out[:, 0].copy(), out[:, 1].copy()
        w, h = out[:, 2].copy(), out[:, 3].copy()
        out[:, 0], out[:, 1] = cx - w / 2.0, cy - h / 2.0
        out[:, 2], out[:, 3] = cx + w / 2.0, cy + h / 2.0
    return out


def obb_corners(cx, cy, width, height, angle):
    """旋转框四角，**顺时针**（屏幕坐标，y 向下），起点在「左上」。

    宽向量是 (w/2·cos, w/2·sin)，高向量必须与它**垂直**，即 (−h/2·sin, h/2·cos)。
    写成 (h/2·sin, h/2·cos) 会得到一组斜平行四边形而不是矩形：45° 时四点还会
    退化成一条直线。

    输出顺序是「左上 → 左下 → 右下 → 右上」，与手绘旋转框
    （`shape_io.rect_to_rotation_points`：左上 → 右上 → 右下 → 左下）**同为顺时针**。
    绕向不能反：画布的旋转手柄沿 p0→p1 的外法线放，绕向反了手柄就落到框内。
    """
    cos_a, sin_a = math.cos(float(angle)), math.sin(float(angle))
    wx, wy = float(width) / 2.0 * cos_a, float(width) / 2.0 * sin_a
    hx, hy = -float(height) / 2.0 * sin_a, float(height) / 2.0 * cos_a
    return [
        [cx - wx + hx, cy - wy + hy],   # 左上
        [cx - wx - hx, cy - wy - hy],   # 左下
        [cx + wx - hx, cy + wy - hy],   # 右下
        [cx + wx + hx, cy + wy + hy],   # 右上
    ]


def mask_coeffs_to_polygon(coeffs, protos, box_xyxy, input_size,
                           threshold=MASK_THRESHOLD):
    """一组掩膜系数 -> 模型输入坐标系下的多边形顶点；取不出返回 None。

    做法与 ultralytics 一致：系数乘原型得到低分辨率概率图，按框裁切后放大到
    框尺寸再二值化，最后取最大外轮廓。邻近目标的掩膜不互相污染，全靠裁切。
    """
    try:
        import cv2
    except ImportError as exc:
        raise TaskError(
            "分割结果转多边形需要 OpenCV，请安装 opencv-python-headless 后重试"
        ) from exc

    proto = np.asarray(protos, dtype=np.float32)
    if proto.ndim == 4:
        proto = proto[0]
    if proto.ndim != 3:
        raise TaskError(f"掩膜原型维度不支持：{proto.shape}")
    count, ph, pw = proto.shape
    coeff = np.asarray(coeffs, dtype=np.float32).reshape(-1)
    if coeff.size != count:
        raise TaskError(f"掩膜系数长度 {coeff.size} 与原型通道数 {count} 不一致")

    logits = coeff @ proto.reshape(count, -1)
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = prob.reshape(ph, pw)

    input_w, input_h = float(input_size[0]), float(input_size[1])
    sx, sy = pw / input_w, ph / input_h
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    bx1 = max(0, int(np.floor(x1)))
    by1 = max(0, int(np.floor(y1)))
    bx2 = min(int(round(input_w)), int(np.ceil(x2)))
    by2 = min(int(round(input_h)), int(np.ceil(y2)))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return None

    px1, py1 = max(0, int(np.floor(bx1 * sx))), max(0, int(np.floor(by1 * sy)))
    px2 = min(pw, int(np.ceil(bx2 * sx)) + 1)
    py2 = min(ph, int(np.ceil(by2 * sy)) + 1)
    if px2 - px1 < 2 or py2 - py1 < 2:
        return None

    sub = prob[py1:py2, px1:px2]
    big = cv2.resize(sub, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_LINEAR)
    binary = (big > float(threshold)).astype(np.uint8)
    if not binary.any():
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3 or cv2.contourArea(contour) < 4.0:
        return None
    epsilon = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    return [[float(p[0][0]) + bx1, float(p[0][1]) + by1] for p in approx]


def rotated_iou(quad_a, quad_b):
    """两个凸四边形的精确交并比；顶点不足或面积为 0 时返回 0。

    缺 OpenCV 时退回外接矩形 IoU（NMS 是增强，不该让整个推理失败）。
    """
    pts_a = np.asarray(quad_a, dtype=np.float32).reshape(-1, 2)
    pts_b = np.asarray(quad_b, dtype=np.float32).reshape(-1, 2)
    if len(pts_a) < 3 or len(pts_b) < 3:
        return 0.0
    try:
        import cv2
    except ImportError:
        return _aabb_iou(_points_aabb(pts_a), _points_aabb(pts_b))
    try:
        area_a = abs(float(cv2.contourArea(pts_a)))
        area_b = abs(float(cv2.contourArea(pts_b)))
        if area_a <= 1e-6 or area_b <= 1e-6:
            return 0.0
        inter, _ = cv2.intersectConvexConvex(pts_a, pts_b, handleNested=True)
    except Exception:
        # 自交/退化多边形：算不了就按不重合处理，宁可多留一个框
        return 0.0
    inter = float(inter)
    if inter <= 0.0:
        return 0.0
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _points_aabb(points):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _box_aabb(box):
    points = box.get("points")
    if points:
        return _points_aabb(points)
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    return [x, y, x + float(box.get("width", 0.0)),
            y + float(box.get("height", 0.0))]


def _aabb_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _box_pair_iou(a, b):
    pa, pb = a.get("points"), b.get("points")
    if pa and pb and len(pa) >= 4 and len(pb) >= 4:
        return rotated_iou(pa[:4], pb[:4])
    return _aabb_iou(_box_aabb(a), _box_aabb(b))


def nms_rotated(items, iou_threshold=0.45):
    """旋转框 NMS：按分数贪心，同类别旋转 IoU 超阈值即抑制。

    返回保留下标（与 `nms_per_class` 同口径）。顶点不是四角的项退回外接矩形。
    用真实多边形算 IoU：两根交叉的长条框外接矩形几乎重合，轴对齐口径会误删。
    """
    if not items:
        return []
    threshold = float(iou_threshold)
    scores = [float(b.get("score", 0.0) or 0.0) for b in items]
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    kept = []
    for index in order:
        box = items[index]
        class_id = box.get("class_id", -1)
        if any(items[other].get("class_id", -1) == class_id
               and _box_pair_iou(box, items[other]) >= threshold
               for other in kept):
            continue
        kept.append(index)
    return kept


def decode_seg(head, protos, num_classes=None, end2end=None, input_size=None,
               conf_threshold=0.25, max_det=DEFAULT_MAX_DET):
    """分割：返回 [{box, score, class_id, points}, ...]，坐标为模型输入坐标系。

    轮廓取不出来的检出按矩形落盘（points 为 None），不丢结果。
    `input_size` 是模型输入宽高；不传时按原型是输入的 1/4 反推（ultralytics 惯例）。
    """
    box, scores, class_ids, coeffs, box_xyxy = decode_rows(
        head, TASK_SEG, num_classes=num_classes, end2end=end2end,
        conf_threshold=conf_threshold, max_det=max_det)
    boxes = box if box_xyxy else cxcywh_to_xyxy(box)
    proto = np.asarray(protos, dtype=np.float32)
    if proto.ndim == 4:
        proto = proto[0]
    if input_size is None:
        input_size = (proto.shape[2] * 4, proto.shape[1] * 4)

    out = []
    for index in range(boxes.shape[0]):
        points = mask_coeffs_to_polygon(coeffs[index], proto, boxes[index],
                                        input_size)
        out.append({
            "box": boxes[index],
            "score": float(scores[index]),
            "class_id": int(class_ids[index]),
            "points": points,
        })
    return out


def decode_obb(head, num_classes=None, end2end=None, conf_threshold=0.25,
               max_det=DEFAULT_MAX_DET):
    """OBB：返回 [{box, score, class_id, corners, angle}, ...]。

    两条线的框都按中心宽高 + 弧度解，末列是角度。外接矩形单独给出，供 NMS 用。
    """
    box, scores, class_ids, tail, _ = decode_rows(
        head, TASK_OBB, num_classes=num_classes, end2end=end2end,
        conf_threshold=conf_threshold, max_det=max_det)
    out = []
    for index in range(box.shape[0]):
        cx, cy, width, height = (float(v) for v in box[index])
        angle = float(tail[index][0]) if tail.shape[1] else 0.0
        corners = obb_corners(cx, cy, width, height, angle)
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        out.append({
            "box": np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32),
            "score": float(scores[index]),
            "class_id": int(class_ids[index]),
            "corners": corners,
            "angle": angle,
        })
    return out


def decode_pose(head, kpt_count, num_classes=None, end2end=None,
                conf_threshold=0.25, kpt_conf=DEFAULT_KPT_CONF,
                max_det=DEFAULT_MAX_DET):
    """姿态：返回 [{box, score, class_id, keypoints}, ...]。

    keypoints 为 [(x, y, conf), ...]，已按可见度置信度过滤；坐标为模型输入坐标系。
    """
    count = int(kpt_count)
    if count <= 0:
        raise TaskError("姿态任务需要关键点个数（kpt_shape），当前为 0")
    box, scores, class_ids, tail, box_xyxy = decode_rows(
        head, TASK_POSE, num_classes=num_classes, kpt_count=count,
        end2end=end2end, conf_threshold=conf_threshold, max_det=max_det)
    boxes = box if box_xyxy else cxcywh_to_xyxy(box)
    tail = tail.reshape(tail.shape[0], count, KPT_DIM) if tail.size else \
        np.zeros((box.shape[0], count, KPT_DIM), dtype=np.float32)

    out = []
    for index in range(boxes.shape[0]):
        points = []
        for k in range(count):
            kx, ky, kc = (float(v) for v in tail[index][k])
            if kc >= float(kpt_conf):
                points.append((kx, ky, kc))
            else:
                points.append((kx, ky, None))
        out.append({
            "box": boxes[index],
            "score": float(scores[index]),
            "class_id": int(class_ids[index]),
            "keypoints": points,
        })
    return out


def decode_task(task, head, protos=None, num_classes=None, kpt_shape=None,
                end2end=None, input_size=None, conf_threshold=0.25,
                kpt_conf=DEFAULT_KPT_CONF, max_det=DEFAULT_MAX_DET):
    """按任务分派解码；det 不在此处（仍由 onnx_inference 的检测解码器负责）。"""
    if task == TASK_SEG:
        if protos is None:
            raise TaskError("分割任务缺少掩膜原型输出")
        return decode_seg(head, protos, num_classes=num_classes,
                          end2end=end2end, input_size=input_size,
                          conf_threshold=conf_threshold, max_det=max_det)
    if task == TASK_OBB:
        return decode_obb(head, num_classes=num_classes, end2end=end2end,
                          conf_threshold=conf_threshold, max_det=max_det)
    if task == TASK_POSE:
        count = kpt_shape[0] if kpt_shape else 0
        return decode_pose(head, count, num_classes=num_classes,
                           end2end=end2end, conf_threshold=conf_threshold,
                           kpt_conf=kpt_conf, max_det=max_det)
    raise TaskError(f"不支持的任务：{task!r}")
