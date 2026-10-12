"""推理任务的配置校验、模型装配与批量执行。

本模块不导入 Qt：图像读取与 letterbox 由调用方注入（loader / resize_fn），
这样批量逻辑可以用假模型和假图像直接测。
"""
import os
import time
from typing import Dict, List, NamedTuple, Optional

from pastelabel.engine.onnx_inference import (
    DEFAULT_CONF, DEFAULT_IOU, DEFAULT_INPUT_SIZE,
    MAX_IOU, MAX_SCORE, MIN_IOU, MIN_SCORE,
    InferenceError, OnnxDetector, SUPPORTED_FORMATS,
)

SOURCE_KEY = "inference"


class ImageResult(NamedTuple):
    """一张图的推理结果。

    boxes 为空且 error 为 None 表示「已推理但无目标」，
    这与「尚未推理」是不同状态，调用方不能把两者混为一谈。
    宽高随结果一起带出，写 sidecar 时不用再读一次图片。
    """
    index: int
    path: str
    boxes: List[dict]
    error: Optional[str] = None
    width: int = 0
    height: int = 0


def _normalize_window_spec(raw):
    """窗口参数：只留 w/h/overlap 三个整数，坏值整份丢弃。

    保存的是「上次用过的参数」，不校验图尺寸——图尺寸随数据集变，
    校验留给 `normalize_scope()` 在真正跑任务时做。
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    out = {}
    for key in ("w", "h", "overlap"):
        if key not in raw:
            return {}
        try:
            value = int(raw[key])
        except (TypeError, ValueError):
            return {}
        if value < 0 or (key != "overlap" and value <= 0):
            return {}
        out[key] = value
    return out


def normalize_settings(settings):
    """把界面/配置里的原始值规整成可用的模型配置，越界值钳到合法范围。"""
    settings = dict(settings or {})
    fmt = str(settings.get("output_format") or "auto").strip().lower()
    if fmt not in ("auto",) + SUPPORTED_FORMATS:
        fmt = "auto"
    size = settings.get("input_size")
    if size:
        try:
            size = (int(size[0]), int(size[1]))
        except (TypeError, ValueError, IndexError):
            size = None
        else:
            if size[0] <= 0 or size[1] <= 0:
                size = None
    else:
        size = None
    providers = settings.get("providers") or None
    if providers is not None:
        providers = [str(p) for p in providers if str(p).strip()] or None
    return {
        "model_path": str(settings.get("model_path") or ""),
        "class_names_path": str(settings.get("class_names_path") or ""),
        "output_format": fmt,
        "conf_threshold": _clamp_float(settings.get("conf_threshold"), DEFAULT_CONF, MIN_SCORE, MAX_SCORE),
        "iou_threshold": _clamp_float(settings.get("iou_threshold"), DEFAULT_IOU, MIN_IOU, MAX_IOU),
        "input_size": size,
        "providers": providers,
        "class_filter": [str(c) for c in (settings.get("class_filter") or [])],
        "replace_auto": bool(settings.get("replace_auto", False)),
        "window_spec": _normalize_window_spec(settings.get("window_spec")),
    }


def _clamp_float(value, default, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(low, min(high, number))


def validate_settings(settings):
    """返回错误文案；配置可用时返回 None。"""
    settings = normalize_settings(settings)
    path = settings["model_path"]
    if not path:
        return "请选择 ONNX 模型文件"
    if not os.path.isfile(path):
        return f"模型文件不存在: {path}"
    if path.lower().endswith(".onnx") is False:
        return "模型文件必须是 .onnx"
    names = settings["class_names_path"]
    if names and not os.path.isfile(names):
        return f"类别文件不存在: {names}"
    return None


def model_class_names(session):
    """从 ONNX 元数据里取类别表，取不到返回空表。"""
    from pastelabel.engine.onnx_inference import metadata_class_names
    return metadata_class_names(session)


def sidecar_class_names_path(model_path):
    """模型同级目录的同名类别表：模型.onnx → 模型.txt。"""
    return f"{os.path.splitext(model_path)[0]}.txt"


def resolve_class_names(model_path, session, read_names=None):
    """解析类别名，返回 (names, source)。

    source 为 "file"（同级同名 txt）、"metadata"（模型自带 names）或 "missing"。
    找不到就报 missing，由调用方决定是否让用户手输；这里不编造 0 1 2 3，
    否则标注里会写进一堆无意义的类名。
    """
    from pastelabel.engine.onnx_inference import metadata_class_names, read_class_names

    path = sidecar_class_names_path(model_path)
    if os.path.isfile(path):
        names = (read_names or read_class_names)(path)
        if names:
            return names, "file"
    names = metadata_class_names(session)
    if names:
        return names, "metadata"
    return [], "missing"


def write_class_names(path, names):
    """把类别名写成一行一个的 txt；失败返回 False。"""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(name) for name in names) + "\n")
    except OSError:
        return False
    return True


def build_detector(settings, read_names=None):
    """按配置装配 OnnxDetector；类别表优先用文件，其次用模型元数据。

    不给类别表时由 OnnxDetector 自己回落到元数据，这里不重复读取。
    """
    from pastelabel.engine.onnx_inference import read_class_names

    settings = normalize_settings(settings)
    names = []
    if settings["class_names_path"]:
        names = (read_names or read_class_names)(settings["class_names_path"])
    return OnnxDetector(
        settings["model_path"],
        class_names=names,
        output_format=settings["output_format"],
        conf_threshold=settings["conf_threshold"],
        iou_threshold=settings["iou_threshold"],
        input_size=settings["input_size"],
        providers=settings["providers"],
    )


def filter_by_class(boxes, class_filter):
    """按类别名白名单过滤。

    None 表示不过滤（界面默认全选就走这条）；具体集合表示只保留这些类别，
    空集合等于全不选（结果一定为空）。界面上「默认全选」由调用方翻译成 None，
    这里不去猜「空集合是不是没勾」。
    """
    if class_filter is None:
        return list(boxes)
    allowed = {str(c) for c in class_filter if str(c).strip()}
    return [b for b in boxes if b.get("label") in allowed]


def is_auto_box(box):
    """是否为上一次推理留下的结果（用 flags 记录来源，不标记为贴图）。"""
    flags = box.get("flags")
    return isinstance(flags, dict) and bool(flags.get(SOURCE_KEY))


def merge_boxes(existing, detected, replace_auto=False, class_filter=None):
    """把推理结果并入现有标注。

    默认保留人工标注；replace_auto=True 时先移除上一轮推理结果再并入，
    人工标注在任何情况下都不被删除。

    class_filter 给定时做严格筛选：上一轮推理留下的、类别不在白名单里的框
    也要一并丢掉。否则筛选只能「加」不能「减」——上一次没筛跑出的类别会
    一直留在画布上，看起来就像筛选没生效。人工标注不受影响。

    None 表示不过滤（界面默认全选），空集合表示全不选、清掉所有旧推理框。
    """
    kept = []
    for box in existing or []:
        if replace_auto and is_auto_box(box):
            continue
        # 严格筛选：只清推理框，人工框无论类别都保留
        if (class_filter is not None and is_auto_box(box)
                and box.get("label") not in class_filter):
            continue
        kept.append(box)
    for box in detected or []:
        merged = dict(box)
        flags = dict(merged.get("flags") or {})
        flags[SOURCE_KEY] = True
        merged["flags"] = flags
        kept.append(merged)
    return kept


def mark_auto_boxes(boxes):
    """给推理结果打来源标记，保持其它 flags 不变。"""
    out = []
    for box in boxes or []:
        item = dict(box)
        flags = dict(item.get("flags") or {})
        flags[SOURCE_KEY] = True
        item["flags"] = flags
        out.append(item)
    return out


def detect_in_scope(rgb, detector, resize_fn, scope):
    """按范围跑一张图：全图 / 滑窗 / ROI / ROI+滑窗，返回原图坐标的框。

    每个窗口都是「裁剪副本 → detect → 加偏移」，detect 自己完成 letterbox
    逆变换并钳到裁剪图，所以偏移在它返回之后加，且只加一次。

    ROI 的多边形掩膜只在裁剪副本上生成，不改写原图。
    """
    from .inference_scope import (
        MODE_ROI, MODE_ROI_WINDOW, MODE_WINDOW,
        iter_regions, iter_windows, merge_scope_results, region_bbox,
        region_contains_box_center,
    )

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    mode = scope.get("mode")
    windows = None
    if mode in (MODE_WINDOW, MODE_ROI_WINDOW):
        windows = iter_windows(width, height, scope.get("windows") or {})

    if mode == MODE_ROI:
        regions = iter_regions(scope.get("regions"), (width, height))
        chunks = []
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            crop, ox, oy = _masked_crop(rgb, region, x1, y1, x2, y2)
            boxes = detector.detect(crop, resize_fn)
            chunks.append((ox, oy, boxes))
        merged = merge_scope_results(chunks, _scope_iou(detector),
                                    image_size=(width, height))
        return [b for b in merged
                if any(region_contains_box_center(r, b) for r in regions)]

    if mode == MODE_ROI_WINDOW:
        regions = iter_regions(scope.get("regions"), (width, height))
        chunks = []
        for region in regions:
            x1, y1, x2, y2 = region.bbox
            for wx, wy, ww, wh in iter_windows(int(x2 - x1), int(y2 - y1),
                                               scope.get("windows") or {}):
                crop, ox, oy = _masked_crop(rgb, region, x1 + wx, y1 + wy,
                                            x1 + wx + ww, y1 + wy + wh)
                boxes = detector.detect(crop, resize_fn)
                chunks.append((ox, oy, boxes))
        merged = merge_scope_results(chunks, _scope_iou(detector),
                                    image_size=(width, height))
        return [b for b in merged
                if any(region_contains_box_center(r, b) for r in regions)]

    if mode == MODE_WINDOW:
        chunks = []
        for wx, wy, ww, wh in windows:
            crop = rgb[wy:wy + wh, wx:wx + ww]
            boxes = detector.detect(crop, resize_fn)
            chunks.append((wx, wy, boxes))
        return merge_scope_results(chunks, _scope_iou(detector),
                                   image_size=(width, height))

    return detector.detect(rgb, resize_fn)


def _scope_iou(detector):
    """跨窗合并用的 IoU：取本次任务的阈值，界面改过就跟着变。"""
    return float(getattr(detector, "iou_threshold", DEFAULT_IOU))


def _masked_crop(rgb, region, x1, y1, x2, y2):
    """裁出 [x1,y1,x2,y2)，返回 (裁剪图, 实际起点 x, 实际起点 y)。

    起点取整后**返回给调用方当偏移**：ROI 坐标可能是小数，掩膜按整数像素
    判定，两者若各算各的会差半个像素。
    多边形 ROI 把界外像素涂成 letterbox 背景色，只在副本上操作，原图不动；
    矩形 ROI 无需掩膜。掩膜用 numpy 向量化 ray-casting（逐像素调
    `point_in_polygon` 在 640x640 窗口上要 41 万次调用，慢到不可用）。
    """
    import numpy as np
    from .onnx_inference import LETTERBOX_FILL

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    cx1, cy1 = max(0, int(x1)), max(0, int(y1))
    cx2, cy2 = min(width, int(x2)), min(height, int(y2))
    crop = np.array(rgb[cy1:cy2, cx1:cx2], copy=True)
    if not region.points or crop.size == 0:
        return crop, cx1, cy1

    points = [(float(p[0]), float(p[1])) for p in region.points]
    rows, cols = crop.shape[0], crop.shape[1]
    ys = np.arange(cy1, cy1 + rows, dtype=np.float64)[:, None]
    xs = np.arange(cx1, cx1 + cols, dtype=np.float64)[None, :]

    inside = np.zeros((rows, cols), dtype=bool)
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if yi != yj:
            lower, upper = (xi, yi), (xj, yj)
            if yi > yj:
                lower, upper = upper, lower
            crossing = (xs < (upper[0] - lower[0]) * (ys - lower[1])
                        / (upper[1] - lower[1]) + lower[0])
            inside ^= crossing & (ys >= lower[1]) & (ys < upper[1])
        j = i

    # 边上的像素也算在内，与 region_contains_box_center 的边界口径一致
    for i, start in enumerate(points):
        end = points[(i + 1) % len(points)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0:
            distance = np.hypot(xs - start[0], ys - start[1])
        else:
            t = ((xs - start[0]) * dx + (ys - start[1]) * dy) / seg_len_sq
            np.clip(t, 0.0, 1.0, out=t)
            distance = np.hypot(xs - (start[0] + t * dx), ys - (start[1] + t * dy))
        inside |= distance <= 1e-6

    crop[~inside] = LETTERBOX_FILL
    return crop, cx1, cy1


def run_batch(paths, detector, loader, resize_fn, keys=None,
              progress_fn=None, is_interrupted=None, class_filter=None,
              scope=None, on_result=None):
    """顺序推理多张图，返回 {key: ImageResult}。

    key 默认是 paths 的下标；批量任务要按数据集索引写回内存与磁盘，
    所以调用方可以传入 dataset 索引作为 key（单张推理传当前索引）。

    class_filter 是允许保留的类别名集合；None 表示不过滤，空集合表示全不选
    （结果一定为空，界面据此提示）。它只作用于本次检出的框，不动已有标注。

    scope 是范围配置（全图/滑窗/ROI/ROI+滑窗）；None 表示全图，此时走原来的
    `detector.detect()` 单次调用，结果与不传 scope 完全一致。scope 由
    `inference_scope.normalize_scope()` 校验过，ROI 已是快照。

    on_result 每张跑完立刻调用一次，参数 (key, ImageResult)。**必须在这里
    落盘**，不能等整批结束：用户要看到 json 一张张出来、列表圆点逐个变绿，
    中途取消或崩溃时已完成的图也不能白跑。

    progress_fn 每张完成后调用一次，参数为
    (position, total, 文件名, 框数, 耗时毫秒)。框数与耗时都是这一张自己的，
    调用方不用再回查结果；失败时框数为 0，耗时仍然上报。

    单张失败只记录该图的 error，不中断整批；取消时保留已完成的结果。
    """
    from .inference_scope import MODE_FULL, check_reference_size, normalize_scope

    results: Dict[int, ImageResult] = {}
    total = len(paths)
    keys = list(keys) if keys is not None else list(range(total))
    scoped = scope is not None and scope.get("mode", MODE_FULL) != MODE_FULL
    for position, path in enumerate(paths):
        if is_interrupted and is_interrupted():
            break
        key = keys[position]
        started = time.perf_counter()
        try:
            rgb = loader(path)
            if rgb is None:
                raise InferenceError("无法读取图像")
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            if scoped:
                # 逐图校验：ROI 是相对参考图量的，尺寸不符只让这张失败
                check_reference_size(scope, (width, height))
                active = normalize_scope(scope, (width, height))
                boxes = detect_in_scope(rgb, detector, resize_fn, active)
            else:
                boxes = detector.detect(rgb, resize_fn)
            kept = filter_by_class(boxes, class_filter)
            results[key] = ImageResult(key, path, kept, None, width, height)
        except Exception as exc:
            kept = []
            results[key] = ImageResult(key, path, [], str(exc))
        if on_result:
            on_result(key, results[key])
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        if progress_fn:
            progress_fn(position + 1, total, os.path.basename(path),
                        len(kept), elapsed_ms)
    return results


def failure_summary(results):
    """失败帧摘要，供批量结束后的提示使用。"""
    failed = [r for r in results.values() if r.error]
    if not failed:
        return ""
    first = failed[0]
    return f"{len(failed)} 张失败，例如 {os.path.basename(first.path)}: {first.error}"


def beijing_timestamp(now=None):
    """北京时间戳，精确到毫秒。

    界面统一按 UTC+8 显示，不跟随本机时区——本机时区变了日志时间也不该变。
    """
    from datetime import datetime, timedelta, timezone
    tz = timezone(timedelta(hours=8))
    moment = now.astimezone(tz) if now is not None else datetime.now(tz)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def read_sidecar_boxes(image_path):
    """读 sidecar 里的检测框；文件缺失或损坏返回 None（与「空标注」区分）。"""
    import json
    from pastelabel.engine.shape_io import box_from_labelme_shape

    json_path = f"{os.path.splitext(image_path)[0]}.json"
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        return None
    boxes = []
    for shape in data["shapes"]:
        box = box_from_labelme_shape(shape)
        if box is not None:
            boxes.append(box)
    return boxes


def sidecar_path(image_path):
    return f"{os.path.splitext(image_path)[0]}.json"


def remove_empty_sidecar(image_path):
    """json 里没有任何 shape 就删掉它，返回是否删除。

    推理无检出时若留下 shapes 为 [] 的 json，数据集里会多出一堆空文件，
    后续扫描与训练都要额外过滤；只有确实无标注时才删，损坏的文件不动。
    """
    import json

    json_path = sidecar_path(image_path)
    if not os.path.exists(json_path):
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    shapes = data.get("shapes")
    if not isinstance(shapes, list) or shapes:
        return False
    try:
        os.remove(json_path)
    except OSError:
        return False
    return True


def write_sidecar_boxes(image_path, boxes, image_width=0, image_height=0):
    """把检测框写回 sidecar，保留 version/flags/imageData 等其它字段。

    贴图与多边形按 labelme_shape_from_box 序列化，不依赖当前主窗口索引，
    因此可以给任意图片写结果；先写临时文件再原子替换，中断不留半截 JSON。
    """
    import json
    from pastelabel.core.config import LABELME_VERSION
    from pastelabel.engine.shape_io import labelme_shape_from_box

    json_path = sidecar_path(image_path)
    data = None
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = None
    if data is None:
        data = {
            "version": LABELME_VERSION,
            "flags": {},
            "imageData": None,
            "imagePath": os.path.basename(image_path),
            "imageHeight": int(image_height),
            "imageWidth": int(image_width),
        }
    data["shapes"] = [labelme_shape_from_box(box) for box in (boxes or [])]
    if image_width:
        data["imageWidth"] = int(image_width)
    if image_height:
        data["imageHeight"] = int(image_height)

    tmp_path = f"{json_path}.tmp"
    try:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, json_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False
    return True


def write_one_result(result, existing=None, replace_auto=False,
                     class_filter=None):
    """把单张结果合并后写进它的 sidecar；返回是否真的写了盘。

    `existing` 给 None 表示从磁盘读（调用方有更新鲜的内存时应当传进来）。
    没检出且合并后仍无任何标注的不写盘：否则数据集里会多出一堆 shapes 为 []
    的空 json。已有标注（人工/贴图/多边形）的图照常写回，因为 replace_auto
    可能刚好删掉了上一轮的推理框。

    失败抛 `InferenceError`，让调用方按「这张写失败」处理，别把整批拖掉。
    """
    if result.error:
        return False
    if existing is None:
        existing = read_sidecar_boxes(result.path)
        if existing is None:
            existing = []
    merged = merge_boxes(existing, result.boxes, replace_auto=replace_auto,
                         class_filter=class_filter)
    if merged:
        if not write_sidecar_boxes(result.path, merged, result.width,
                                   result.height):
            raise InferenceError(f"写入失败: {result.path}")
        return True
    # 合并后没有标注：replace_auto 可能刚把上一轮的推理框删光，
    # 这时盘上还留着它们，必须先写出真实内容再按需删文件。
    if not existing:
        remove_empty_sidecar(result.path)
        return False
    if not write_sidecar_boxes(result.path, [], result.width, result.height):
        raise InferenceError(f"写入失败: {result.path}")
    remove_empty_sidecar(result.path)
    return False


def write_results(results, memory_boxes=None, replace_auto=False, skip_index=None,
                  class_filter=None):
    """批量把结果写进各自 sidecar，返回 (写入数, 失败路径列表)。

    逐张调用 `write_one_result`。**逐张写盘**（而不是攒到最后一起写）是硬要求：
    界面按每张完成就刷新列表圆点，用户也要能随时看到已写出的 json。

    memory_boxes 里已有的图以内存为准：内存可能比磁盘新，直接读盘会把
    用户刚做的编辑当成不存在，进而丢掉。skip_index 交给调用方用正常保存
    流程处理（当前显示图还涉及贴图与画布状态）。

    class_filter 会传给 merge_boxes，把上一轮不在白名单里的推理框也清掉。
    """
    memory_boxes = memory_boxes or {}
    written = 0
    failed = []
    for index, result in sorted(results.items()):
        if index == skip_index:
            continue
        try:
            if write_one_result(result, existing=memory_boxes.get(index),
                                replace_auto=replace_auto,
                                class_filter=class_filter):
                written += 1
        except InferenceError:
            failed.append(result.path)
    return written, failed
