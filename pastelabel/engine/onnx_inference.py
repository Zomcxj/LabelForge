"""ONNX 检测推理：输入 RGB，只支持未内置 NMS 的 v5/v8 常规检测导出。

onnxruntime 延迟导入，缺少该依赖时本模块仍可导入，纯计算部分可独立测试。
"""
import ast
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

LETTERBOX_FILL = 114

FORMAT_V5 = "v5"
#: [1, 4+C, N]。YOLOv8 起的锚点无关检测头，YOLO11 沿用同一设计、导出后形状
#: 与 v8 完全相同（都是 [1, 4+C, N]），所以共用这一种格式，不单独建 v11。
FORMAT_V8 = "v8"
#: 端到端导出（YOLO26），输出已含 NMS 结果，无需再去重
FORMAT_V26 = "v26"
SUPPORTED_FORMATS = (FORMAT_V5, FORMAT_V8, FORMAT_V26)

#: 端到端输出每行 [x1, y1, x2, y2, score, class_id]
END2END_ROW = 6

#: 端到端模型的默认最大检出数（导出时写死在输出维度里）
DEFAULT_MAX_DET = 300

MIN_SCORE = 0.01
MAX_SCORE = 1.0
MIN_IOU = 0.05
MAX_IOU = 0.95

#: 动态轴模型的默认输入尺寸；模型行里可以随时改
DEFAULT_INPUT_SIZE = (640, 640)

#: 默认阈值。只在这里定义一份，界面与批量任务都引用它，避免多处写死后对不上
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45

# 无类别数时自动判定布局用的范围。判定方法是两头都看：把某一轴当通道轴解出的
# C 要合理（[_MIN_CLASSES, _MAX_PLAUSIBLE_CLASSES]），另一轴作为候选轴也不能
# 太少（_MIN_ANCHORS_FOR_AUTO）。用来排除把候选数当通道解出的荒谬结果。
_MIN_CLASSES = 1
# 类别数上限取 80（COCO 的量级）。自己训练的模型基本不会超过这个数，超了说明
# 多半是把候选数当成了通道轴。真有大类别数（如 OpenImages 600 类）的模型，
# 判不出来会提示显式选择格式，用户选一次即可，不会静默用错。
_MAX_PLAUSIBLE_CLASSES = 80
_MIN_ANCHORS_FOR_AUTO = 100


class InferenceError(RuntimeError):
    """可预期错误：依赖缺失、模型格式不支持、输出无法解析。"""


class LetterboxGeometry(NamedTuple):
    """letterbox 变换参数：缩放、补边偏移、缩放后尺寸、画布尺寸。

    缩放后尺寸与补边偏移一起返回，是为了让「绘制」和「逆变换」用同一组整数：
    各自独立取整会让坐标系统性偏移最多 1px。
    """
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    dst_w: int
    dst_h: int


def letterbox_geometry(src_w, src_h, dst_w, dst_h):
    """等比缩放居中补边的几何参数。

    源图乘 scale 后缩放到 (new_w, new_h)，再偏移 (pad_x, pad_y) 落在 (dst_w, dst_h)
    画布上；长宽比相同时补边为 0。补边向下取整，画布右下可能余 1px，由填充色补齐。
    """
    if src_w <= 0 or src_h <= 0:
        raise InferenceError(f"图像尺寸非法: {src_w}x{src_h}")
    if dst_w <= 0 or dst_h <= 0:
        raise InferenceError(f"模型输入尺寸非法: {dst_w}x{dst_h}")
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, min(dst_w, int(round(src_w * scale))))
    new_h = max(1, min(dst_h, int(round(src_h * scale))))
    return LetterboxGeometry(
        scale=scale,
        pad_x=(dst_w - new_w) // 2,
        pad_y=(dst_h - new_h) // 2,
        new_w=new_w,
        new_h=new_h,
        dst_w=dst_w,
        dst_h=dst_h,
    )


def unletterbox_boxes(boxes, geometry, src_w, src_h):
    """模型输入坐标的 xyxy 还原到原图坐标并钳制在图内。

    返回 (boxes, keep_mask)；落在补边区内、钳制后退化为零面积的框标为 False，
    调用方必须用同一掩码同步过滤分数与类别，否则三者错位。
    """
    if boxes.size == 0:
        return boxes.reshape(0, 4), np.zeros((0,), dtype=bool)
    if geometry.scale <= 0:
        raise InferenceError(f"缩放系数非法: {geometry.scale}")
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = np.clip((out[:, [0, 2]] - geometry.pad_x) / geometry.scale, 0, src_w)
    out[:, [1, 3]] = np.clip((out[:, [1, 3]] - geometry.pad_y) / geometry.scale, 0, src_h)
    keep = (out[:, 2] > out[:, 0]) & (out[:, 3] > out[:, 1])
    return out, keep


def _as_2d(raw, name):
    """(1, A, B) 或 (A, B) 统一成 2D；其它维度拒绝。"""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise InferenceError(f"{name} 输出 batch={arr.shape[0]}，仅支持 batch=1")
        arr = arr[0]
    if arr.ndim != 2:
        raise InferenceError(f"{name} 输出维度不支持: {arr.shape}")
    return arr


def _pack(box_xywh, scores, class_ids, conf_threshold):
    """按置信度过滤并把中心宽高转成 xyxy。"""
    keep = scores >= float(conf_threshold)
    box = box_xywh[keep].astype(np.float32, copy=True)
    scores = scores[keep].astype(np.float32, copy=True)
    class_ids = class_ids[keep].astype(np.int64, copy=True)
    if box.size:
        # 必须先复制：cx 等是 box 的视图，就地改写后再参与计算会拿到已更新的值
        cx, cy = box[:, 0].copy(), box[:, 1].copy()
        w, h = box[:, 2].copy(), box[:, 3].copy()
        box[:, 0], box[:, 1] = cx - w / 2.0, cy - h / 2.0
        box[:, 2], box[:, 3] = cx + w / 2.0, cy + h / 2.0
    else:
        box = box.reshape(0, 4)
    return {"boxes": box, "scores": scores, "class_ids": class_ids}


def decode_yolov5(raw, conf_threshold, num_classes=None):
    """解码 [1, N, 5+C]，行内 cx, cy, w, h, obj, cls...；置信度 = obj × 类别分数。

    坐标仍在模型输入坐标系下，需再做 unletterbox。
    """
    arr = _as_2d(raw, "YOLOv5")
    if arr.shape[1] < 6:
        raise InferenceError(f"YOLOv5 输出末维 {arr.shape[1]}，至少需 5+C（C>=1）")
    classes = arr.shape[1] - 5
    if num_classes is not None and classes != num_classes:
        raise InferenceError(f"YOLOv5 输出类别数 {classes} 与类别表 {num_classes} 不一致")
    cls_scores = arr[:, 5:]
    best_cls = cls_scores.argmax(axis=1)
    scores = arr[:, 4] * cls_scores[np.arange(cls_scores.shape[0]), best_cls]
    return _pack(arr[:, :4], scores, best_cls, conf_threshold)


def decode_yolov8(raw, conf_threshold, num_classes=None):
    """解码 [1, 4+C, N]（也接受转置导出），行内 cx, cy, w, h, cls...。

    对应 YOLOv8 及之后沿用同一检测头的版本（YOLO11 输出结构与 v8 完全一致，
    都走这里）。无 objectness，置信度直接取类别分数；两种布局无法区分时
    （N == 4+C）报错。
    """
    arr = _as_2d(raw, "YOLOv8/YOLO11")
    expected = None if num_classes is None else 4 + num_classes
    a, b = arr.shape
    if expected is not None:
        if a == expected and b == expected:
            raise InferenceError(
                f"YOLOv8/YOLO11 输出 {arr.shape} 两种布局无法区分，请显式指定格式")
        if a == expected:
            arr = arr.T
        elif b != expected:
            raise InferenceError(
                f"YOLOv8/YOLO11 输出 {arr.shape} 与类别表 {num_classes} 不匹配")
    elif a > b:
        arr = arr.T
    if arr.shape[1] < 5:
        raise InferenceError(
            f"YOLOv8/YOLO11 输出末维 {arr.shape[1]}，至少需 4+C（C>=1）")
    cls_scores = arr[:, 4:]
    best_cls = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), best_cls]
    return _pack(arr[:, :4], scores, best_cls, conf_threshold)


def decode_end2end(raw, conf_threshold, max_det=DEFAULT_MAX_DET, num_classes=None):
    """解码端到端导出 [1, N, 6]，行内 x1, y1, x2, y2, score, class_id。

    这类模型（YOLO26）导出时已内置 NMS，官方后处理也只做分数过滤、类别过滤
    与 max_det 截断，因此这里不再做一次 NMS —— 再做会与官方结果不一致。

    已知现象：实测 yolo26s 的输出里仍有同类高 IoU 重叠框（conf>=0.25 时最大
    约 0.96）。这是模型自身 NMS 阈值偏松，不是解码问题，别在这里补救。
    """
    arr = _as_2d(raw, "端到端")
    if arr.shape[1] != END2END_ROW:
        raise InferenceError(
            f"端到端输出末维应为 {END2END_ROW}（x1y1x2y2+score+cls），"
            f"实际 {arr.shape[1]}")
    cls_ids = np.rint(arr[:, 5]).astype(np.int64)
    if num_classes is not None and num_classes > 0:
        # 越界说明模型与类别表对不上，早点报错好过产出空标签
        bad = cls_ids[(cls_ids < 0) | (cls_ids >= num_classes)]
        if bad.size:
            raise InferenceError(
                f"端到端输出出现越界类别 id {int(bad.min())}~{int(bad.max())}，"
                f"类别表只有 {num_classes} 项")

    scores = arr[:, 4].astype(np.float32, copy=True)
    keep = scores >= float(conf_threshold)
    box = arr[keep, :4].astype(np.float32, copy=True)
    scores = scores[keep]
    cls_ids = cls_ids[keep]

    if int(max_det) > 0 and box.shape[0] > int(max_det):
        # 模型输出未保证有序，按分数取前 max_det
        top = np.argsort(-scores, kind="stable")[:int(max_det)]
        box, scores, cls_ids = box[top], scores[top], cls_ids[top]
    return {"boxes": box, "scores": scores, "class_ids": cls_ids}


def guess_class_count(shape):
    """从检测输出 shape 猜类别数，猜不出返回 None。

    只在能唯一解释时才给数：v5 的 [1, N, 5+C]、v8 的 [1, 4+C, N] 里的 C 是
    唯一答案；端到端 [1, N, 6] 不含类别数（分数与类别 id 都在列里），返回 None。
    这个值只用来提示用户「该填几个类别」，不参与解码，猜错也不会污染结果。
    """
    try:
        dims = [int(d) for d in shape if int(d) != 1]
    except (TypeError, ValueError):
        return None
    if len(dims) != 2:
        return None
    rows, cols = dims
    # 端到端与「类别很少的 v5/v8」在 shape 上无法区分（[1, N, 6] 两种解释
    # 都成立），这种小形状一律不给数，宁可让用户自己填。
    if cols <= END2END_ROW and rows <= DEFAULT_MAX_DET:
        return None
    if cols > rows:
        # v8 转置布局 [1, 4+C, N]：通道在 rows，4+C 就是它的长度
        c = rows - 4
    else:
        # v5 [1, N, 5+C]：通道在 cols
        c = cols - 5
    return c if c > 0 else None


def detect_output_format(shape, num_classes=None):
    """按输出 shape 推断格式，推断不出返回 None（调用方须让用户显式指定）。

    检测头的两个轴一个是候选（锚点）轴、一个是通道轴（5+C 或 4+C）。类别数
    未知时只能靠轴序：

      v5 是 [1, 候选数, 5+C] —— 候选在前、通道在后；
      v8/v11 是 [1, 4+C, 候选数] —— 通道在前、候选在后。

    两个方向都要认：只认 v8 那一侧会让没有类别表的 v5 第三方导出（如
    [1, 25200, 9]）被判成「无法判断格式」而加载不了。

    判据是「把某一轴当通道轴解出的 C 是否合理」，另一轴同时要够大像候选数。
    两边都推不出结果时返回 None，交回用户显式指定——猜错会把 objectness 当
    类别分数、把列当行读，静默产出错误标注。

    末维恰为 6、行数在 max_det 以内且远多于 6 时按端到端解释。但 6 也可能
    正好是 v5 的 5+C（C=1）或 v8 的 4+C（C=2），类别数已知且与之吻合时不认
    v26，继续按 v5/v8 判断，避免把类别数很少的普通检测导出误判成端到端。
    """
    dims = [int(d) for d in shape if int(d) != 1]
    if len(dims) != 2:
        return None
    known = num_classes is not None and num_classes > 0
    if not (known and dims[1] in (5 + num_classes, 4 + num_classes)):
        # 端到端导出的行数是 max_det（固定上限），列数只有 6；
        # 行数过大的只可能是别的布局（如转置的 v8），不能认成端到端。
        if (dims[1] == END2END_ROW and dims[1] < dims[0] <= DEFAULT_MAX_DET):
            return FORMAT_V26
    if known:
        if dims[1] == 5 + num_classes:
            return FORMAT_V5
        if dims[1] == 4 + num_classes or dims[0] == 4 + num_classes:
            return FORMAT_V8
        return None
    # 类别数未知：靠「哪一轴像通道轴」判。通道轴是 5+C（v5）或 4+C（v8），
    # 判据两头都要看：把某轴当通道解出的 C 要落在 [_MIN_CLASSES,
    # _MAX_PLAUSIBLE_CLASSES]，另一轴作为候选轴也不能太少。
    #
    #   [1, 25200, 9]  cols=9  -> C=4（合理），rows=25200 当通道 -> C=25196（超限）
    #                  => v5
    #   [1, 84, 8400]  rows=84 -> C=80（合理），cols 当通道 -> C=8395（超限）
    #                  => v8
    #   [1, 8400, 84]  cols=84 -> C=79（合理）=> v5；反向要 cols>=100 才成立，
    #                  84 不够，所以不是 v8
    #
    # 末轴恰为 END2END_ROW 时不认 v5：那是端到端的行宽，读成 v5 会解出 C=1，
    # 与端到端导出撞车，宁可让用户显式指定。
    #
    # 两边都成立（not as_v8 / not as_v5 那两处）在当前常量下不会发生——类别
    # 数上限 80 让 v5 要求 cols<=85、v8 要求 cols>=100，天然互斥。守卫留着，
    # 以后调常量时不必重新推一遍。
    rows, cols = dims
    as_v5 = (_MIN_CLASSES <= cols - 5 <= _MAX_PLAUSIBLE_CLASSES
             and rows >= _MIN_ANCHORS_FOR_AUTO
             and cols != END2END_ROW)
    as_v8 = (_MIN_CLASSES <= rows - 4 <= _MAX_PLAUSIBLE_CLASSES
             and cols >= _MIN_ANCHORS_FOR_AUTO)
    if as_v5 and not as_v8:
        return FORMAT_V5
    if as_v8 and not as_v5:
        return FORMAT_V8
    return None


def pick_detection_output(outputs):
    """从模型输出里挑出唯一的检测头，返回 (下标, shape)。

    YOLOv5 训练导出的第三方模型常带一堆辅助输出：主检测头是 3 维的
    [1, 锚点数, 5+C]，另外还挂出每个 stride 的原始头（5 维
    [1, 每格锚数, H, W, 5+C]，值还没解码）。这些辅助输出不参与推理，
    但只要多挂了一个，模型就变成"多输出"，直接报"仅支持单输出"会把
    这类型号一律拒之门外。

    判别只看形状，不靠输出名（名字在不同导出脚本里毫无规律）：
      3 维的只有一个 -> 就是它；
      3 维的一个都没有，或有多个 -> 说不清是谁，报错让用户显式指定，
      绝不瞎猜（挑错会静默产出错误标注）。
    """
    candidates = [
        (i, tuple(int(d) for d in o.shape))
        for i, o in enumerate(outputs)
        if len(o.shape) == 3
    ]
    if len(candidates) == 1:
        return candidates[0]
    shapes = [tuple(o.shape) for o in outputs]
    if not candidates:
        raise InferenceError(
            f"模型没有 3 维检测输出，实际输出形状 {shapes}；"
            "分割 / 姿态 / OBB 导出需各自的适配")
    raise InferenceError(
        f"模型有 {len(candidates)} 个 3 维输出，无法确定哪个是检测头，"
        f"实际输出形状 {shapes}")


def decode_predictions(raw, output_format, conf_threshold, num_classes=None,
                       max_det=DEFAULT_MAX_DET):
    """按指定格式解码；未知格式明确拒绝，不静默回退成普通检测。"""
    if output_format == FORMAT_V5:
        return decode_yolov5(raw, conf_threshold, num_classes)
    if output_format == FORMAT_V8:
        return decode_yolov8(raw, conf_threshold, num_classes)
    if output_format == FORMAT_V26:
        return decode_end2end(raw, conf_threshold, max_det, num_classes)
    raise InferenceError(
        "不支持的输出格式: "
        f"{output_format!r}，当前仅支持 {' / '.join(SUPPORTED_FORMATS)}")


def box_iou_matrix(boxes, box):
    """boxes(N,4) 与单个 box(4,) 的 IoU 数组。"""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    inter_w = np.clip(np.minimum(boxes[:, 2], box[2]) - np.maximum(boxes[:, 0], box[0]), 0, None)
    inter_h = np.clip(np.minimum(boxes[:, 3], box[3]) - np.maximum(boxes[:, 1], box[1]), 0, None)
    inter = inter_w * inter_h
    area_a = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    area_b = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def nms_per_class(boxes, scores, class_ids, iou_threshold=DEFAULT_IOU):
    """按类别分别做贪心 NMS，返回保留下标（按置信度降序）。

    类别参与去重：不同类别的框即使高度重合也都保留，不能按纯几何去重。
    """
    if boxes.size == 0:
        return []
    kept: List[int] = []
    for idx in np.argsort(-scores, kind="stable"):
        same = [k for k in kept if class_ids[k] == class_ids[idx]]
        if same and (box_iou_matrix(boxes[same], boxes[idx]) >= float(iou_threshold)).any():
            continue
        kept.append(int(idx))
    return kept


def boxes_to_internal(boxes, scores, class_ids, class_names):
    """转成内部 box 字典（x, y, width, height, label）。

    类别 id 越界时给空标签而不抛错：模型与类别表不匹配已在解码阶段拦过，
    这里再抛会打断整批任务。
    """
    out = []
    for box, score, cls in zip(boxes, scores, class_ids):
        cid = int(cls)
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        out.append({
            "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
            "label": class_names[cid] if 0 <= cid < len(class_names) else "",
            "score": float(score), "class_id": cid,
        })
    return out


def runtime_available():
    """是否装有 onnxruntime；UI 用它决定能否发起任务。"""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def available_providers():
    """当前环境可用的 provider；缺少依赖时返回空表。"""
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def read_class_names(path):
    """读取类别表，一行一个类别。

    严格保持文件行序：类别 id 由行号决定，排序或去重都会让 id 静默错位。
    """
    names = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names


def metadata_class_names(session):
    """从 ONNX 元数据取类别表，取不到返回空表。

    ultralytics 导出会把 names 写成 "{0: 'person', ...}" 这类字典字面量；
    能解析就按 key 排序还原 id 顺序，格式不认识时返回空表交给调用方兜底。
    放在本模块是为了让格式识别能用上模型自带的类别数。
    """
    try:
        meta = session.get_modelmeta().custom_metadata_map or {}
    except Exception:
        return []
    raw = meta.get("names")
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return []
    if isinstance(parsed, dict):
        try:
            return [str(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
        except (TypeError, ValueError):
            return []
    if isinstance(parsed, (list, tuple)):
        return [str(name) for name in parsed]
    return []


class OnnxDetector:
    """单个检测模型的会话封装，Session 延迟创建（构造时不加载模型）。"""

    def __init__(self, model_path, class_names=None, output_format="auto",
                 conf_threshold=None, iou_threshold=None,
                 input_size=None, providers=None):
        if output_format not in ("auto",) + SUPPORTED_FORMATS:
            raise InferenceError(f"不支持的输出格式: {output_format!r}")
        self.model_path = model_path
        self.class_names = list(class_names or [])
        self.output_format = output_format
        # 阈值默认值只在这里定义一份，界面与批量任务都从这里取
        self.conf_threshold = float(
            DEFAULT_CONF if conf_threshold is None else conf_threshold)
        self.iou_threshold = float(
            DEFAULT_IOU if iou_threshold is None else iou_threshold)
        self.input_size = input_size
        self.providers = list(providers) if providers else None
        self._session = None
        self._input_name = None
        self._resolved_format = None
        # 检测头在输出列表里的下标；带辅助输出的模型不是 0
        self._output_index = 0

    def set_thresholds(self, conf_threshold=None, iou_threshold=None):
        """改阈值；只影响后续 detect，不重建会话。"""
        if conf_threshold is not None:
            self.conf_threshold = float(conf_threshold)
        if iou_threshold is not None:
            self.iou_threshold = float(iou_threshold)

    def set_input_size(self, input_size):
        """改动态轴模型的输入尺寸；只影响后续 detect，不重建会话。"""
        self.input_size = input_size

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise InferenceError(
                "未安装 onnxruntime，无法执行 ONNX 推理。"
                "请安装 onnxruntime（CPU）或 onnxruntime-gpu（GPU）后重试。") from exc
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        try:
            self._session = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=self.providers)
        except Exception as exc:
            raise InferenceError(f"加载模型失败: {self.model_path}\n{exc}") from exc
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise InferenceError(f"模型输入个数为 {len(inputs)}，仅支持单输入检测模型")
        self._input_name = inputs[0].name
        # 没显式给类别表时用模型自带的：否则标签会静默变成空串，
        # 保存出去的标注看不出是哪一类。
        if not self.class_names:
            self.class_names = metadata_class_names(self._session)
        return self._session

    @property
    def session(self):
        return self._ensure_session()

    @property
    def active_providers(self):
        """实际生效的 provider。CPU/GPU 包都导入为 onnxruntime，只能问 Session。"""
        if self._session is None:
            return []
        try:
            return list(self._session.get_providers())
        except Exception:
            return []

    def model_input_size(self):
        """模型输入 (width, height)。

        静态轴直接取 shape；动态轴用用户指定的尺寸，没有就用 640×640 默认值，
        该值在模型行里可以随时改。
        """
        self._ensure_session()
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            raise InferenceError(f"模型输入维度不支持: {shape}")
        h, w = shape[2], shape[3]
        if all(isinstance(v, int) and v > 0 for v in (h, w)):
            return int(w), int(h)
        if self.input_size:
            return int(self.input_size[0]), int(self.input_size[1])
        return DEFAULT_INPUT_SIZE

    def is_dynamic_input(self):
        """输入宽高是否为动态轴；动态轴模型才需要用户指定尺寸。"""
        self._ensure_session()
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            return False
        h, w = shape[2], shape[3]
        return not all(isinstance(v, int) and v > 0 for v in (h, w))

    def resolve_format(self):
        """输出格式：用户指定优先，auto 时按输出 shape 推断，推断不出报错。"""
        if self.output_format in SUPPORTED_FORMATS:
            return self.output_format
        if self._resolved_format:
            return self._resolved_format
        self._ensure_session()
        outputs = self.session.get_outputs()
        index, shape = pick_detection_output(outputs)
        # 模型自带的类别表也算已知类别数：ultralytics 导出的 v5/v8 都写了 names，
        # 有它就能把 5+C 与 4+C 区分开，不必强求用户再选一次格式。
        num_classes = len(self.class_names) or len(metadata_class_names(self.session))
        fmt = detect_output_format(shape, num_classes or None)
        if fmt is None:
            raise InferenceError(
                f"无法从输出 shape {shape} 判断模型格式，"
                "请显式选择 YOLOv5 / YOLOv8 / YOLO11 / YOLO26")
        self._resolved_format = fmt
        self._output_index = index
        return fmt

    def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
        """对单张 RGB 图（H, W, 3 uint8）推理，返回原图坐标下的内部框列表。

        resize_fn(src_rgb, geometry, fill) 由调用方注入，返回 (dst_h, dst_w, 3)
        的 letterbox 画布；这样本模块不绑定具体图像库。

        阈值按调用传入，不烘焙进会话：界面改完阈值可以直接重跑，不必重建模型。
        """
        if rgb is None or getattr(rgb, "ndim", None) != 3 or rgb.shape[2] != 3:
            shape = None if rgb is None else getattr(rgb, "shape", None)
            raise InferenceError(f"输入必须是 (H, W, 3) 的 RGB 图，收到 {shape}")
        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou = self.iou_threshold if iou_threshold is None else float(iou_threshold)
        src_h, src_w = int(rgb.shape[0]), int(rgb.shape[1])
        dst_w, dst_h = self.model_input_size()
        geometry = letterbox_geometry(src_w, src_h, dst_w, dst_h)

        canvas = resize_fn(rgb, geometry, LETTERBOX_FILL)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

        outputs = self._ensure_session().run(None, {self._input_name: tensor})
        fmt = self.resolve_format()
        decoded = decode_predictions(
            outputs[self._output_index], fmt, conf,
            len(self.class_names) or None)

        boxes, keep = unletterbox_boxes(decoded["boxes"], geometry, src_w, src_h)
        scores, class_ids = decoded["scores"][keep], decoded["class_ids"][keep]

        # 端到端模型（v26）自带 NMS，再跑一遍会与官方结果不一致
        kept = list(range(len(boxes))) if fmt == FORMAT_V26 else \
            nms_per_class(boxes, scores, class_ids, iou)
        if not kept:
            return []
        return boxes_to_internal(boxes[kept], scores[kept], class_ids[kept], self.class_names)
