"""推理任务的配置校验、模型装配与批量执行。

本模块不导入 Qt：图像读取与 letterbox 由调用方注入（loader / resize_fn），
这样批量逻辑可以用假模型和假图像直接测。
"""
import os
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
    """按类别名白名单过滤；白名单为空表示不过滤。"""
    allowed = {str(c) for c in (class_filter or []) if str(c).strip()}
    if not allowed:
        return list(boxes)
    return [b for b in boxes if b.get("label") in allowed]


def is_auto_box(box):
    """是否为上一次推理留下的结果（用 flags 记录来源，不标记为贴图）。"""
    flags = box.get("flags")
    return isinstance(flags, dict) and bool(flags.get(SOURCE_KEY))


def merge_boxes(existing, detected, replace_auto=False):
    """把推理结果并入现有标注。

    默认保留人工标注；replace_auto=True 时先移除上一轮推理结果再并入，
    人工标注在任何情况下都不被删除。
    """
    kept = []
    for box in existing or []:
        if replace_auto and is_auto_box(box):
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


def run_batch(paths, detector, loader, resize_fn, keys=None,
              progress_fn=None, is_interrupted=None, class_filter=None):
    """顺序推理多张图，返回 {key: ImageResult}。

    key 默认是 paths 的下标；批量任务要按数据集索引写回内存与磁盘，
    所以调用方可以传入 dataset 索引作为 key（单张推理传当前索引）。

    单张失败只记录该图的 error，不中断整批；取消时保留已完成的结果。
    """
    results: Dict[int, ImageResult] = {}
    total = len(paths)
    keys = list(keys) if keys is not None else list(range(total))
    for position, path in enumerate(paths):
        if is_interrupted and is_interrupted():
            break
        key = keys[position]
        try:
            rgb = loader(path)
            if rgb is None:
                raise InferenceError("无法读取图像")
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            boxes = detector.detect(rgb, resize_fn)
            results[key] = ImageResult(
                key, path, filter_by_class(boxes, class_filter), None, width, height)
        except Exception as exc:
            results[key] = ImageResult(key, path, [], str(exc))
        if progress_fn:
            progress_fn(position + 1, total, os.path.basename(path))
    return results


def failure_summary(results):
    """失败帧摘要，供批量结束后的提示使用。"""
    failed = [r for r in results.values() if r.error]
    if not failed:
        return ""
    first = failed[0]
    return f"{len(failed)} 张失败，例如 {os.path.basename(first.path)}: {first.error}"


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


def write_results(results, memory_boxes=None, replace_auto=False, skip_index=None):
    """批量把结果写进各自 sidecar，返回 (写入数, 失败路径列表)。

    memory_boxes 里已有的图以内存为准：内存可能比磁盘新，直接读盘会把
    用户刚做的编辑当成不存在，进而丢掉。skip_index 交给调用方用正常保存
    流程处理（当前显示图还涉及贴图与画布状态）。

    没检出且合并后仍无任何标注的图不写盘：否则数据集里会多出一堆
    shapes 为 [] 的空 json。已有标注（人工/贴图/多边形）的图照常写回，
    因为 replace_auto 可能刚好删掉了上一轮的推理框。
    """
    memory_boxes = memory_boxes or {}
    written = 0
    failed = []
    for index, result in sorted(results.items()):
        if index == skip_index or result.error:
            continue
        existing = memory_boxes.get(index)
        if existing is None:
            existing = read_sidecar_boxes(result.path)
            if existing is None:
                existing = []
        merged = merge_boxes(existing, result.boxes, replace_auto=replace_auto)
        if merged:
            if write_sidecar_boxes(result.path, merged, result.width, result.height):
                written += 1
            else:
                failed.append(result.path)
            continue
        # 合并后没有标注：replace_auto 可能刚把上一轮的推理框删光，
        # 这时盘上还留着它们，必须先写出真实内容再按需删文件。
        if not existing:
            remove_empty_sidecar(result.path)
            continue
        if write_sidecar_boxes(result.path, [], result.width, result.height):
            remove_empty_sidecar(result.path)
        else:
            failed.append(result.path)
    return written, failed
