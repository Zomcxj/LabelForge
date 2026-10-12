"""滑窗 / 区域推理的范围枚举、ROI 过滤与结果合并（纯逻辑，无 Qt 与 cv2）。

坐标约定：`OnnxDetector.detect()` 接收裁剪后的 RGB，返回的框已在**该裁剪图**
坐标下——它自己完成 letterbox 逆变换并钳到输入尺寸。所以 scope 只负责
「加原图像素偏移 + 按类别 NMS」，不能在除以 scale 前加偏移。

复用 `augmenter/crop.py` 的窗口数学（`normalized_window` / `window_starts`），
不另写一套；不复用 `crop_boxes()`——那是为数据集增强的**已有标注**写的，
按 `min_visible` 比例截断/丢弃框，套到 detector 的预测上会丢掉边缘目标。
"""
import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .augmenter.crop import MAX_WINDOW, normalized_window, window_starts
from .onnx_inference import DEFAULT_IOU, nms_per_class

MODE_FULL = "full"
MODE_WINDOW = "window"
MODE_ROI = "roi"
MODE_ROI_WINDOW = "roi_window"
MODES = (MODE_FULL, MODE_WINDOW, MODE_ROI, MODE_ROI_WINDOW)

_SCOPE_KEYS = {"mode", "windows", "regions"}
_WINDOW_KEYS = {"w", "h", "overlap"}
_EDGE_EPS = 1e-6            # 顶点/边命中容差（原图像素坐标）


class ScopeError(RuntimeError):
    """范围配置、ROI 数据或合并参数非法。"""


@dataclass
class Region:
    """一个 ROI：外接矩形（图内）与可选多边形顶点（原图坐标）。"""
    bbox: Tuple[float, float, float, float]      # (x1, y1, x2, y2)
    points: Optional[List[List[float]]] = None   # 多边形顶点；矩形为 None


def _finite(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ScopeError(f"{name} 不是有效数字：{value!r}")
    if not math.isfinite(number):
        raise ScopeError(f"{name} 必须是有限数字：{value!r}")
    return number


def iter_windows(iw, ih, spec):
    """枚举滑窗：返回 [(x, y, w, h), ...]，末窗贴边、不越界。

    窗口尺寸与重叠先经 `normalized_window()` 统一钳制（小图缩为整图、
    重叠不超过 min(cw,ch)-1），再交给 `window_starts()` 排布——数量以实际
    结果为准，不用 `size_for_count()` 反推（那个函数取整后可能多出一窗）。
    """
    try:
        iw, ih = int(iw), int(ih)
    except (TypeError, ValueError):
        raise ScopeError(f"图像尺寸不是整数：{iw!r} x {ih!r}")
    if iw <= 0 or ih <= 0:
        raise ScopeError(f"图像尺寸必须为正：{iw} x {ih}")

    if not isinstance(spec, dict):
        raise ScopeError("窗口参数必须是字典")
    unknown = set(spec) - _WINDOW_KEYS - {"fill"}
    if unknown:
        raise ScopeError(f"窗口参数含未知字段：{sorted(unknown)}")
    if "w" not in spec or "h" not in spec:
        raise ScopeError("窗口参数必须包含 w 与 h")

    try:
        raw_w, raw_h = int(spec["w"]), int(spec["h"])
    except (TypeError, ValueError):
        raise ScopeError(f"窗口尺寸必须是整数：{spec.get('w')!r} x {spec.get('h')!r}")
    if raw_w <= 0 or raw_h <= 0:
        raise ScopeError(f"窗口尺寸必须为正：{raw_w} x {raw_h}")
    if raw_w > MAX_WINDOW or raw_h > MAX_WINDOW:
        raise ScopeError(f"窗口尺寸超过上限 {MAX_WINDOW}：{raw_w} x {raw_h}")

    try:
        overlap = int(spec.get("overlap", 0))
    except (TypeError, ValueError):
        raise ScopeError(f"重叠必须是整数：{spec.get('overlap')!r}")
    if overlap < 0:
        raise ScopeError(f"重叠不能为负：{overlap}")

    # fill 不参与推理：范围推理一律贴边不补边，避免把填充色喂给模型
    cw, ch, ov = normalized_window(iw, ih, {"w": raw_w, "h": raw_h,
                                            "overlap": overlap})
    xs = window_starts(iw, cw, ov)
    ys = window_starts(ih, ch, ov)
    return [(x, y, cw, ch) for y in ys for x in xs]


def region_bbox(region):
    """ROI 的外接矩形 (x1, y1, x2, y2)；退化或非有限坐标报错。

    多边形以 `points` 为准，不读可能过期的 x/y/width/height（画布在编辑
    过程中可能只更新其中一份）。
    """
    if not isinstance(region, dict):
        raise ScopeError(f"ROI 必须是字典：{region!r}")
    shape_type = region.get("shape_type") or "rectangle"

    if shape_type == "polygon" or region.get("points"):
        points = region.get("points") or []
        if len(points) < 3:
            raise ScopeError(f"多边形 ROI 顶点不足：{len(points)}")
        xs, ys = [], []
        for point in points:
            try:
                px, py = point[0], point[1]
            except (TypeError, IndexError):
                raise ScopeError(f"多边形顶点格式不对：{point!r}")
            xs.append(_finite(px, "ROI 顶点 x"))
            ys.append(_finite(py, "ROI 顶点 y"))
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        x = _finite(region.get("x", 0), "ROI x")
        y = _finite(region.get("y", 0), "ROI y")
        width = _finite(region.get("width", 0), "ROI 宽")
        height = _finite(region.get("height", 0), "ROI 高")
        x1, y1, x2, y2 = x, y, x + width, y + height

    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ScopeError(f"ROI 面积必须为正：{(x1, y1, x2, y2)}")
    return (x1, y1, x2, y2)


def iter_regions(regions, image_size):
    """校验 ROI 列表并裁剪到图内；返回 [Region]。

    空列表、全部非法、与目标图没有有效交集都报错——不能返回空范围让
    调用方以为「成功但 0 框」（默认覆盖模式会因此删掉原有自动标注）。
    """
    if not isinstance(regions, (list, tuple)) or not regions:
        raise ScopeError("选择了区域范围，但没有可用的 ROI")
    try:
        iw, ih = int(image_size[0]), int(image_size[1])
    except (TypeError, ValueError, IndexError):
        raise ScopeError(f"图像尺寸无效：{image_size!r}")
    if iw <= 0 or ih <= 0:
        raise ScopeError(f"图像尺寸必须为正：{iw} x {ih}")

    out = []
    for region in regions:
        x1, y1, x2, y2 = region_bbox(region)
        # 裁到图内；交集为空说明该 ROI 与这张图无关
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(float(iw), x2), min(float(ih), y2)
        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
            raise ScopeError(
                f"ROI 与图像没有有效交集：ROI {(x1, y1, x2, y2)}，图像 {iw}x{ih}")
        points = None
        if region.get("shape_type") == "polygon" or region.get("points"):
            points = [[_finite(p[0], "ROI 顶点 x"), _finite(p[1], "ROI 顶点 y")]
                      for p in region["points"]]
        out.append(Region(bbox=(cx1, cy1, cx2, cy2), points=points))
    return out


def region_contains_box_center(region, box):
    """框中心是否在 ROI 内（矩形与多边形均含边界）。

    `shape_io.point_in_polygon()` 是纯 ray-casting，落在边或顶点上会判 False；
    这里先用 `point_line_distance()` 补边界判定，再退回 ray-casting，保持画布
    命中与贴图规则不变。
    """
    if region is None or not isinstance(box, dict):
        return False
    try:
        cx = float(box.get("x", 0)) + float(box.get("width", 0)) / 2.0
        cy = float(box.get("y", 0)) + float(box.get("height", 0)) / 2.0
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return False

    x1, y1, x2, y2 = region.bbox
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):
        return False
    if not region.points:
        return True

    from .shape_io import point_in_polygon, point_line_distance
    points = region.points
    if point_in_polygon(cx, cy, points):
        return True
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        if point_line_distance((cx, cy), (start, end)) <= _EDGE_EPS:
            return True
    return False


def normalize_scope(scope, image_size):
    """校验并归一化范围配置；None 表示全图（保持既有行为）。

    返回的字典是深拷贝：后台任务不能读到正在被编辑的 `region_boxes`。
    """
    if scope is None:
        return None
    if not isinstance(scope, dict):
        raise ScopeError(f"范围配置必须是字典：{scope!r}")

    unknown = set(scope) - _SCOPE_KEYS
    if unknown:
        raise ScopeError(f"范围配置含未知字段：{sorted(unknown)}")

    mode = scope.get("mode", MODE_FULL)
    if mode not in MODES:
        raise ScopeError(f"未知范围模式：{mode!r}")
    if mode == MODE_FULL:
        return {"mode": MODE_FULL}

    out = {"mode": mode}
    if mode in (MODE_WINDOW, MODE_ROI_WINDOW):
        windows = scope.get("windows")
        if not isinstance(windows, dict):
            raise ScopeError("滑窗范围必须提供窗口参数")
        # 立刻用真实图尺寸校验一次，非法参数在启动时就报错
        iter_windows(image_size[0], image_size[1], windows)
        out["windows"] = dict(windows)
    if mode in (MODE_ROI, MODE_ROI_WINDOW):
        regions = scope.get("regions")
        if not isinstance(regions, (list, tuple)) or not regions:
            raise ScopeError("区域范围必须提供 ROI")
        # 深拷贝：ROI 是临时区域，任务运行期间用户还可能继续编辑
        iter_regions(copy.deepcopy(regions), image_size)
        out["regions"] = copy.deepcopy(regions)
    return out


def merge_scope_results(chunks, iou_threshold=DEFAULT_IOU, image_size=None):
    """把各窗口/各 ROI 的结果合并到原图坐标，按类别 NMS 去重。

    chunks: [(offset_x, offset_y, [box, ...]), ...]，box 是 `detect()` 返回的
    裁剪图坐标框。偏移在缩放之后加（`detect()` 已完成去 padding 与缩放），
    每个框只加一次，再加 ROI 偏移一次（调用方把两者相加后传入）。
    """
    merged = []
    for chunk in chunks or []:
        if not chunk or len(chunk) != 3:
            continue
        offset_x, offset_y, boxes = chunk
        try:
            dx, dy = float(offset_x), float(offset_y)
        except (TypeError, ValueError):
            continue
        for box in boxes or []:
            if not isinstance(box, dict):
                continue
            try:
                x = float(box.get("x", 0)) + dx
                y = float(box.get("y", 0)) + dy
                width = float(box.get("width", 0))
                height = float(box.get("height", 0))
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x, y, width, height)):
                continue
            if width <= 0 or height <= 0:
                continue
            if image_size is not None:
                iw, ih = float(image_size[0]), float(image_size[1])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(iw, x + width), min(ih, y + height)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                x, y, width, height = x1, y1, x2 - x1, y2 - y1
            item = dict(box)
            item["x"], item["y"] = x, y
            item["width"], item["height"] = width, height
            merged.append(item)

    if not merged:
        return []

    import numpy as np
    boxes = np.array([[b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]]
                      for b in merged], dtype=np.float64)
    scores = np.array([float(b.get("score", 0.0)) for b in merged], dtype=np.float64)
    class_ids = np.array([int(b.get("class_id", -1)) for b in merged], dtype=np.int64)

    kept = nms_per_class(boxes, scores, class_ids, iou_threshold)
    out = [merged[i] for i in kept]
    out.sort(key=lambda b: float(b.get("score", 0.0)), reverse=True)
    return out
