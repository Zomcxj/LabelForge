"""ONNX 检测推理：输入 RGB，只支持未内置 NMS 的 v5/v8 常规检测导出。

onnxruntime 延迟导入，缺少该依赖时本模块仍可导入，纯计算部分可独立测试。
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

LETTERBOX_FILL = 114

FORMAT_V5 = "v5"
FORMAT_V8 = "v8"
SUPPORTED_FORMATS = (FORMAT_V5, FORMAT_V8)

MIN_SCORE = 0.01
MAX_SCORE = 1.0
MIN_IOU = 0.05
MAX_IOU = 0.95

# 无类别数时认定「末轴是候选轴」所需的最少候选数。真实检测导出在 320px 以上
# 输入至少有 2100 个候选，100 是很保守的下限；小于它的形状（如 3x8）两种解释
# 都成立，必须让用户显式指定格式。
_MIN_ANCHORS_FOR_V8 = 100


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

    无 objectness，置信度直接取类别分数；两种布局无法区分时（N == 4+C）报错。
    """
    arr = _as_2d(raw, "YOLOv8")
    expected = None if num_classes is None else 4 + num_classes
    a, b = arr.shape
    if expected is not None:
        if a == expected and b == expected:
            raise InferenceError(f"YOLOv8 输出 {arr.shape} 两种布局无法区分，请显式指定格式")
        if a == expected:
            arr = arr.T
        elif b != expected:
            raise InferenceError(f"YOLOv8 输出 {arr.shape} 与类别表 {num_classes} 不匹配")
    elif a > b:
        arr = arr.T
    if arr.shape[1] < 5:
        raise InferenceError(f"YOLOv8 输出末维 {arr.shape[1]}，至少需 4+C（C>=1）")
    cls_scores = arr[:, 4:]
    best_cls = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), best_cls]
    return _pack(arr[:, :4], scores, best_cls, conf_threshold)


def detect_output_format(shape, num_classes=None):
    """按输出 shape 推断格式，推断不出返回 None（调用方须让用户显式指定）。

    没有类别数时只能靠轴序判断：v8 的候选数远大于通道数（如 8400 vs 84），
    只有这种明显情形才认。小 N 的形状两种解释都成立（v5 的 [1, N, 5+C]
    与 v8 的 [1, 4+C, N] 可能同形），此时必须返回 None —— 猜错会把 v5 的
    objectness 当成类别分数、把列当行读，静默产出错误标注。
    """
    dims = [int(d) for d in shape if int(d) != 1]
    if len(dims) != 2:
        return None
    if num_classes is not None and num_classes > 0:
        if dims[1] == 5 + num_classes:
            return FORMAT_V5
        if dims[1] == 4 + num_classes or dims[0] == 4 + num_classes:
            return FORMAT_V8
        return None
    if dims[1] > dims[0] and dims[1] >= _MIN_ANCHORS_FOR_V8:
        return FORMAT_V8
    return None


def decode_predictions(raw, output_format, conf_threshold, num_classes=None):
    """按指定格式解码；未知格式明确拒绝，不静默回退成普通检测。"""
    if output_format == FORMAT_V5:
        return decode_yolov5(raw, conf_threshold, num_classes)
    if output_format == FORMAT_V8:
        return decode_yolov8(raw, conf_threshold, num_classes)
    raise InferenceError(
        f"不支持的输出格式: {output_format!r}，当前仅支持 {FORMAT_V5} / {FORMAT_V8}")


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


def nms_per_class(boxes, scores, class_ids, iou_threshold=0.45):
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


class OnnxDetector:
    """单个检测模型的会话封装，Session 延迟创建（构造时不加载模型）。"""

    def __init__(self, model_path, class_names=None, output_format="auto",
                 conf_threshold=0.25, iou_threshold=0.45,
                 input_size=None, providers=None):
        if output_format not in ("auto",) + SUPPORTED_FORMATS:
            raise InferenceError(f"不支持的输出格式: {output_format!r}")
        self.model_path = model_path
        self.class_names = list(class_names or [])
        self.output_format = output_format
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.input_size = input_size
        self.providers = list(providers) if providers else None
        self._session = None
        self._input_name = None
        self._resolved_format = None

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

        静态轴直接取 shape；动态轴用用户指定的尺寸，都没有则报错，
        不擅自假设一个尺寸。
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
        raise InferenceError(f"模型输入尺寸为动态轴 {shape}，请指定输入宽高")

    def resolve_format(self):
        """输出格式：用户指定优先，auto 时按输出 shape 推断，推断不出报错。"""
        if self.output_format in SUPPORTED_FORMATS:
            return self.output_format
        if self._resolved_format:
            return self._resolved_format
        self._ensure_session()
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise InferenceError(
                f"模型输出个数为 {len(outputs)}，仅支持单输出检测模型；"
                "内置 NMS / 分割 / 姿态 / OBB 导出需各自的适配")
        fmt = detect_output_format(outputs[0].shape, len(self.class_names) or None)
        if fmt is None:
            raise InferenceError(
                f"无法从输出 shape {outputs[0].shape} 判断模型格式，"
                "请显式选择 YOLOv5 或 YOLOv8")
        self._resolved_format = fmt
        return fmt

    def detect(self, rgb, resize_fn):
        """对单张 RGB 图（H, W, 3 uint8）推理，返回原图坐标下的内部框列表。

        resize_fn(src_rgb, geometry, fill) 由调用方注入，返回 (dst_h, dst_w, 3)
        的 letterbox 画布；这样本模块不绑定具体图像库。
        """
        if rgb is None or getattr(rgb, "ndim", None) != 3 or rgb.shape[2] != 3:
            shape = None if rgb is None else getattr(rgb, "shape", None)
            raise InferenceError(f"输入必须是 (H, W, 3) 的 RGB 图，收到 {shape}")
        src_h, src_w = int(rgb.shape[0]), int(rgb.shape[1])
        dst_w, dst_h = self.model_input_size()
        geometry = letterbox_geometry(src_w, src_h, dst_w, dst_h)

        canvas = resize_fn(rgb, geometry, LETTERBOX_FILL)
        tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]

        outputs = self._ensure_session().run(None, {self._input_name: tensor})
        decoded = decode_predictions(
            outputs[0], self.resolve_format(), self.conf_threshold,
            len(self.class_names) or None)

        boxes, keep = unletterbox_boxes(decoded["boxes"], geometry, src_w, src_h)
        scores, class_ids = decoded["scores"][keep], decoded["class_ids"][keep]

        kept = nms_per_class(boxes, scores, class_ids, self.iou_threshold)
        if not kept:
            return []
        return boxes_to_internal(boxes[kept], scores[kept], class_ids[kept], self.class_names)
