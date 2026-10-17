"""数据集转换与校验适配层（自研实现，零外部依赖）

只依赖 Python 标准库：
- 图片尺寸通过文件头解析（PNG/JPEG/GIF/BMP/WebP/TIFF），不解码整张图；
- YOLO 的 data.yaml 只解析 names 与 train/val 路径，不引入 PyYAML；
- COCO / LabelMe 用 json，Pascal VOC 用 xml.etree。

设计约束（见 .superpowers/plans/2026-09-14-supervision-integration.md）：
- 不依赖 Qt，可在无 GUI 环境下测试；
- 默认不覆盖原始数据，输出目录非空时必须显式 overwrite；
- 转换前完成校验，失败时不写出半成品；
- 所有解析 / 写出异常统一转换为 DatasetToolError。

历史上本模块依赖 supervision 作为可选依赖，但它把 scipy 一并拖入 exe，
单文件包体积从 52MB 涨到 91MB。当前实现已完全自研，无任何外部依赖。
"""
import json
import os
import re
import shutil
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

#: 支持的输入 / 输出格式
INPUT_FORMATS = ("yolo", "coco", "voc", "labelme")
OUTPUT_FORMATS = ("yolo", "coco", "voc", "labelme")

#: COCO 标注文件名，与 labelforge/engine/coco_exporter.py 保持一致
COCO_ANNOTATION_NAME = "coco_detection.json"

#: 分块写出时的块大小，避免大图集长时间无进度回调
_CHUNK_SIZE = 32

#: 本自研实现版本
IMPL_VERSION = "1.0"

#: 数据集支持的图片扩展名（小写）
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff")


class DatasetToolError(Exception):
    """数据集工具的应用层错误，消息可直接展示给用户。"""


class ValidationReport:
    """数据集校验结果。

    errors 非空表示不可转换；warnings 仅提示，不阻塞转换。
    """

    def __init__(self, image_count: int = 0, annotation_count: int = 0,
                 classes: Optional[List[str]] = None,
                 errors: Optional[List[str]] = None,
                 warnings: Optional[List[str]] = None):
        self.image_count = image_count
        self.annotation_count = annotation_count
        self.classes = list(classes or [])
        self.errors = list(errors or [])
        self.warnings = list(warnings or [])

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "classes": list(self.classes),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass
class Detections:
    """单张图的检测结果，坐标统一为像素空间的左上-右下角 (x1, y1, x2, y2)。

    mask_points 与 class_id 等长：None 表示该框是纯矩形框，否则为多边形顶点
    （像素坐标）。YOLO 分割行（8 个及以上归一化坐标）会解析成多边形，
    保证转 LabelMe 时不静默降级成矩形。
    """

    class_id: List[int] = field(default_factory=list)
    box_xyxy: List[Tuple[float, float, float, float]] = field(default_factory=list)
    mask_points: Optional[List[Optional[List[Tuple[float, float]]]]] = None
    #: 与 class_id 等长的形状类型（rectangle/polygon/rotation/point）。
    #: 空列表表示未知，写出侧按 mask_points 推断（有多边形即 polygon）。
    shape_types: List[str] = field(default_factory=list)
    #: 与 class_id 等长的原始像素点集（rotation 存 4 角点，point 存 1 点，
    #: polygon 存全部顶点，rectangle 可空）。obb/pose 转换依赖它。
    points_list: List[Optional[List[Tuple[float, float]]]] = field(default_factory=list)
    #: pose 分组用；None 表示无分组。
    group_ids: List[Optional[int]] = field(default_factory=list)
    #: 与 class_id 等长的 LabelMe flags（贴图 flags.paste 等）；缺失为 {}。
    flags_list: List[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.class_id)

    def type_of(self, index: int) -> str:
        """第 index 个标注的形状类型；缺失时按 mask 推断。"""
        if index < len(self.shape_types) and self.shape_types[index]:
            return self.shape_types[index]
        masks = self.mask_points or []
        if index < len(masks) and masks[index]:
            return "polygon"
        return "rectangle"

    def points_of(self, index: int) -> Optional[List[Tuple[float, float]]]:
        if index < len(self.points_list) and self.points_list[index]:
            return self.points_list[index]
        masks = self.mask_points or []
        if index < len(masks) and masks[index]:
            return masks[index]
        return None

    def group_of(self, index: int) -> Optional[int]:
        if index < len(self.group_ids):
            return self.group_ids[index]
        return None

    def flags_of(self, index: int) -> dict:
        if index < len(self.flags_list) and isinstance(self.flags_list[index], dict):
            return dict(self.flags_list[index])
        return {}

    @property
    def mask(self):
        """存在任一分割标注时返回多边形顶点列表，否则返回 None。"""
        if self.mask_points and any(p is not None for p in self.mask_points):
            return self.mask_points
        return None

    def box_xywh(self) -> List[Tuple[float, float, float, float]]:
        return [(x1, y1, x2 - x1, y2 - y1)
                for x1, y1, x2, y2 in self.box_xyxy]


@dataclass
class Dataset:
    """一个已加载的数据集：类别表 + 图片路径 + 逐图标注 + 逐图尺寸。"""

    classes: List[str]
    image_paths: List[str]
    annotations: Dict[str, Detections] = field(default_factory=dict)
    image_sizes: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def size_of(self, path: str) -> Tuple[int, int]:
        """返回 (width, height)；缓存缺失时现场读文件头，避免静默写出错误坐标。"""
        size = self.image_sizes.get(path)
        if not size:
            size = read_image_size(path)
            if size:
                self.image_sizes[path] = size
        if not size:
            raise DatasetToolError(f"无法读取图片尺寸：{path}")
        return size


# ==========================================================================
# 图片尺寸：只读文件头，不解码
# ==========================================================================

def _png_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    return None


def _gif_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    return None


def _bmp_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return width, abs(height)
    return None


def _webp_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP" or len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 " and len(data) >= 30:
        width = struct.unpack("<H", data[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", data[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(data) >= 25:
        value = struct.unpack("<I", data[21:25])[0]
        return (value & 0x3FFF) + 1, ((value >> 14) & 0x7FFF) + 1
    if chunk == b"VP8X" and len(data) >= 30:
        # chunk data 自偏移 20 起：特性标志位 4 字节 + 宽-1(3) + 高-1(3)。
        # 宽高在 24/27；20..24 是 alpha/exif/xmp/anim 标志，不是尺寸。
        return (int.from_bytes(data[24:27], "little") + 1,
                int.from_bytes(data[27:30], "little") + 1)
    return None


def _jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    """扫描 JPEG SOF 段取尺寸；SOF 可能被大量量化表推后，未找到返回 None。"""
    if data[:2] != b"\xff\xd8":
        return None
    index, total = 2, len(data)
    while index < total - 9:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        length = struct.unpack(">H", data[index + 2:index + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5:index + 9])
            return width, height
        index += 2 + length
    return None


def _tiff_size(data: bytes) -> Optional[Tuple[int, int]]:
    if data[:2] not in (b"II", b"MM") or data[2:4] not in (b"*\x00", b"\x00*"):
        return None
    order = "<" if data[:2] == b"II" else ">"
    if len(data) < 8:
        return None
    ifd = struct.unpack(order + "L", data[4:8])[0]
    if ifd + 2 > len(data):
        return None
    entries = struct.unpack(order + "H", data[ifd:ifd + 2])[0]
    sizes: Dict[int, int] = {}
    for offset in range(entries):
        base = ifd + 2 + offset * 12
        if base + 12 > len(data):
            break
        tag, typ, _count = struct.unpack(order + "HHI", data[base:base + 8])
        if tag not in (256, 257):
            continue
        # IFD 值域固定 4 字节，但 SHORT(type 3) 只占前 2 字节。
        # 一律按 4 字节去解 SHORT 会抛 struct.error。
        if typ == 3:
            value = struct.unpack(order + "H", data[base + 8:base + 10])[0]
        elif typ == 4:
            value = struct.unpack(order + "I", data[base + 8:base + 12])[0]
        else:
            continue
        sizes[tag] = value
    if 256 in sizes and 257 in sizes:
        return sizes[256], sizes[257]
    return None


def read_image_size(path: str) -> Optional[Tuple[int, int]]:
    """读取图片宽高 (width, height)；无法识别返回 None。

    JPEG 的 SOF 段可能被大段量化表推后，因此逐步扩大读取窗口重试。
    """
    for limit in (4096, 65536, 1 << 20):
        try:
            with open(path, "rb") as fh:
                data = fh.read(limit)
        except OSError:
            return None
        if not data:
            return None
        for parser in (_png_size, _gif_size, _bmp_size, _webp_size,
                       _jpeg_size, _tiff_size):
            size = parser(data)
            if size:
                width, height = int(size[0]), int(size[1])
                if width > 0 and height > 0:
                    return width, height
                return None
    return None


# ==========================================================================
# data.yaml 子集解析（只取 names 与 train/val）
# ==========================================================================

def _strip_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        quote = text[0]
        return text[1:-1].replace(quote * 2, quote)
    return text


def _split_flow(text: str) -> List[str]:
    """按逗号切分 flow 序列，跳过引号内的逗号。"""
    parts, current, quote = [], "", None
    for char in text:
        if quote:
            current += char
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            current += char
        elif char == ",":
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def _parse_flow_or_scalar(text: str) -> List[str]:
    """把 flow 映射 / flow 列表 / 单个标量解析成字符串列表（保序）。"""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        values: Dict[str, str] = {}
        order: List[str] = []
        for item in _split_flow(text[1:-1]):
            item = item.strip()
            if not item or ":" not in item:
                continue
            key, _, value = item.partition(":")
            key = _strip_quotes(key)
            if not key or key not in values:
                order.append(key)
            values[key] = _strip_quotes(value)
        return [values[key] for key in order]
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
        result = []
        for item in _split_flow(text):
            item = item.strip()
            if not item:
                continue
            if ":" in item and not item.startswith(("'", '"')):
                result.append(_strip_quotes(item.partition(":")[2]))
            else:
                result.append(_strip_quotes(item))
        return result
    return [_strip_quotes(text)] if text else []


def _parse_names(yaml_text: str) -> List[str]:
    """解析 data.yaml 的 names，支持 flow 列表 / 块列表 / 数字键映射。"""
    lines = yaml_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        key = line.strip()[:6].rstrip(":").strip()
        if line.lstrip().startswith("#"):
            continue
        if key == "names":
            start = index
            break
    if start is None:
        return []
    inline = lines[start].split(":", 1)[1].strip()
    if inline:
        return _parse_flow_or_scalar(inline)

    values: List[str] = []
    keyed: Dict[int, str] = {}
    cursor = start + 1
    while cursor < len(lines):
        raw = lines[cursor]
        stripped = raw.strip()
        cursor += 1
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")):
            break
        if stripped.startswith("-"):
            values.append(_strip_quotes(stripped[1:]))
        elif ":" in stripped:
            token, _, value = stripped.partition(":")
            if token.strip().isdigit():
                keyed[int(token.strip())] = _strip_quotes(value)
        else:
            values.append(_strip_quotes(stripped))
    if keyed:
        return [keyed[key] for key in sorted(keyed)]
    return values


def _parse_paths(yaml_text: str, key: str) -> List[str]:
    """解析 data.yaml 里 train / val 等路径项，返回原始路径字符串列表。"""
    lines = yaml_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        token = stripped.split(":", 1)[0].strip()
        if token == key:
            start = index
            break
    if start is None:
        return []
    inline = lines[start].split(":", 1)[1].strip()
    if inline:
        return _parse_flow_or_scalar(inline)

    values: List[str] = []
    cursor = start + 1
    while cursor < len(lines):
        raw = lines[cursor]
        stripped = raw.strip()
        cursor += 1
        if not stripped or stripped.startswith("#"):
            continue
        if not raw.startswith((" ", "\t")):
            break
        values.append(_strip_quotes(stripped[1:] if stripped.startswith("-")
                                    else stripped))
    return values


# ==========================================================================
# 输入校验
# ==========================================================================

def _check_dir(path: str, what: str) -> str:
    if not path:
        raise DatasetToolError(f"未指定{what}")
    if not os.path.isdir(path):
        raise DatasetToolError(f"{what}不存在：{path}")
    return os.path.abspath(path)


def _check_file(path: str, what: str) -> str:
    if not path:
        raise DatasetToolError(f"未指定{what}")
    if not os.path.isfile(path):
        raise DatasetToolError(f"{what}不存在：{path}")
    return os.path.abspath(path)


def _list_images(directory: str) -> List[str]:
    names = sorted(name for name in os.listdir(directory)
                   if os.path.isfile(os.path.join(directory, name))
                   and os.path.splitext(name)[1].lower() in IMAGE_EXTS)
    return [os.path.join(os.path.abspath(directory), name) for name in names]


def _empty_detections() -> Detections:
    # mask_points 用空列表而不是 None，方便读者逐个 append
    return Detections(class_id=[], box_xyxy=[], mask_points=[],
                      shape_types=[], points_list=[], group_ids=[])


def _append_detection(det: "Detections", class_id, box_xyxy, *,
                      shape_type="rectangle", points=None, group_id=None,
                      mask=None, flags=None):
    """统一追加一个标注，保证各并行列表长度一致。"""
    det.class_id.append(class_id)
    det.box_xyxy.append(box_xyxy)
    det.mask_points.append(mask)
    det.shape_types.append(shape_type)
    det.points_list.append(points)
    det.group_ids.append(group_id)
    det.flags_list.append(flags if isinstance(flags, dict) else {})


def _bounding_box(points) -> Tuple[float, float, float, float]:
    """一组顶点的包围盒 (x1, y1, x2, y2)。"""
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


# ==========================================================================
# 读取
# ==========================================================================

def _read_yolo(images_dir: str, labels_dir: str, data_yaml_path: str,
               task: str = "auto") -> Dataset:
    yaml_path = _check_file(data_yaml_path, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8", errors="replace") as fh:
        yaml_text = fh.read()
    classes = _parse_names(yaml_text)
    if not classes:
        raise DatasetToolError("data.yaml 中没有 names 类别定义")

    base_dir = os.path.dirname(yaml_path)
    image_paths: List[str] = []
    for raw in _parse_paths(yaml_text, "train") + _parse_paths(yaml_text, "val"):
        candidate = raw if os.path.isabs(raw) else os.path.join(base_dir, raw)
        if os.path.isdir(candidate):
            image_paths.extend(_list_images(candidate))
        elif os.path.isfile(candidate):
            image_paths.append(os.path.abspath(candidate))
    if not image_paths:
        image_paths = _list_images(images_dir)

    dataset = Dataset(classes=classes, image_paths=image_paths)
    labels_abs = os.path.abspath(labels_dir)
    for path in image_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        label_path = os.path.join(labels_abs, stem + ".txt")
        annotations = _empty_detections()
        if os.path.isfile(label_path):
            annotations = _parse_yolo_labels(label_path, path, dataset, task)
        dataset.annotations[path] = annotations
    return dataset


def _parse_yolo_labels(label_path: str, image_path: str,
                       dataset: Dataset, task: str = "auto") -> Detections:
    """解析 YOLO 标注文件，按任务类型区分行格式：

    - det/hbb: `cls cx cy w h` -> rectangle
    - seg:     `cls x1 y1 x2 y2 ...`(>=3点) -> polygon
    - obb:     `cls x0 y0 x1 y1 x2 y2 x3 y3`(恰好4点) -> rotation
    - pose:    `cls cx cy w h (kx ky v)*` -> rectangle + point(按行分组)
    - auto:    4列=框，>=8列偶数=多边形（历史行为，无法区分 obb/seg）
    """
    width, height = dataset.size_of(image_path)
    det = _empty_detections()
    with open(label_path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            tokens = line.replace(",", " ").split()
            if len(tokens) < 5:
                continue
            try:
                class_id = int(float(tokens[0]))
                coords = [float(value) for value in tokens[1:]]
            except ValueError:
                continue
            if task in ("det", "hbb") and len(coords) >= 4:
                _yolo_rect(det, class_id, coords[:4], width, height)
            elif task == "seg" and len(coords) >= 6 and len(coords) % 2 == 0:
                _yolo_polygon(det, class_id, coords, width, height)
            elif task == "obb" and len(coords) == 8:
                pts = [(coords[i] * width, coords[i + 1] * height)
                       for i in range(0, 8, 2)]
                _append_detection(det, class_id, _bounding_box(pts),
                                  shape_type="rotation", points=pts)
            elif task == "pose" and len(coords) >= 4:
                _yolo_pose(det, class_id, coords, width, height, line_no)
            elif task == "auto":
                if len(coords) == 4:
                    _yolo_rect(det, class_id, coords, width, height)
                elif len(coords) >= 8 and len(coords) % 2 == 0:
                    _yolo_polygon(det, class_id, coords, width, height)
    return det


def _yolo_rect(det, class_id, coords, width, height):
    cx, cy, bw, bh = coords[:4]
    _append_detection(det, class_id,
                      ((cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height),
                      shape_type="rectangle")


def _yolo_polygon(det, class_id, coords, width, height):
    pts = [(coords[i] * width, coords[i + 1] * height)
           for i in range(0, len(coords) - 1, 2)]
    _append_detection(det, class_id, _bounding_box(pts),
                      shape_type="polygon", points=pts, mask=pts)


def _yolo_pose(det, class_id, coords, width, height, group_id):
    """一行 pose = 一个矩形 + 若干关键点，共享 group_id；关键点类名不可逆，
    统一用占位标签（导回 YOLO 仍按 group 聚合，语义不丢）。"""
    cx, cy, bw, bh = coords[:4]
    _append_detection(det, class_id,
                      ((cx - bw / 2) * width, (cy - bh / 2) * height,
                       (cx + bw / 2) * width, (cy + bh / 2) * height),
                      shape_type="rectangle", group_id=group_id)
    kpts = coords[4:]
    step = 3 if len(kpts) % 3 == 0 else 2
    for i in range(0, len(kpts) - (step - 1), step):
        if step == 3 and kpts[i + 2] == 0:
            continue
        kx, ky = kpts[i] * width, kpts[i + 1] * height
        if kx == 0 and ky == 0:
            continue
        _append_detection(det, class_id, (kx, ky, kx, ky),
                          shape_type="point", points=[(kx, ky)],
                          group_id=group_id)


def _read_coco(images_dir: str, annotations_path: str) -> Dataset:
    json_path = _check_file(annotations_path, "COCO 标注文件")
    with open(json_path, "r", encoding="utf-8", errors="replace") as fh:
        payload = json.load(fh)
    categories = sorted(payload.get("categories") or [],
                        key=lambda item: item.get("id", 0))
    classes = [str(category["name"]) for category in categories]
    category_to_index = {category.get("id"): index
                         for index, category in enumerate(categories)}
    dataset = Dataset(classes=classes, image_paths=[])
    for entry in payload.get("images") or []:
        name = os.path.basename(str(entry.get("file_name") or ""))
        if not name:
            continue
        path = os.path.join(os.path.abspath(images_dir), name)
        dataset.image_paths.append(path)
        width = entry.get("width")
        height = entry.get("height")
        if width and height:
            dataset.image_sizes[path] = (int(width), int(height))
    grouped: Dict[int, List[dict]] = {}
    for entry in payload.get("annotations") or []:
        grouped.setdefault(int(entry.get("image_id", -1)), []).append(entry)
    for path in dataset.image_paths:
        entry = next((item for item in payload.get("images") or []
                      if os.path.basename(str(item.get("file_name") or ""))
                      == os.path.basename(path)), None)
        annotations = _empty_detections()
        items = grouped.get(int(entry.get("id", -1))) if entry else None
        if items:
            for item in items:
                bbox = item.get("bbox") or []
                if len(bbox) != 4 or any(float(value) < 0 for value in bbox):
                    continue
                index = category_to_index.get(item.get("category_id"))
                if index is None:
                    continue
                x, y, width, height = (float(value) for value in bbox)
                _append_detection(annotations, index, (x, y, x + width, y + height),
                                  shape_type="rectangle")
        dataset.annotations[path] = annotations
    return dataset


# ==========================================================================
# XML 读取（纯标准库，不依赖 expat）
#
# 冻结环境没有打包 pyexpat，xml.etree.ElementTree.parse 会直接 ImportError。
# 写出侧用 ET.Element + ElementTree.write 是纯 Python 序列化，不需要 expat；
# 读取侧改用下面的极简解析器，节点表示为 (tag, text, children) 三元组。
# 支持属性、自闭合标签、注释、CDATA、声明与标准实体引用，
# 足够覆盖结构固定的 Pascal VOC 标注文件。
# ==========================================================================

_XML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _decode_entities(text: str) -> str:
    def repl(match):
        token = match.group(1)
        if token.startswith("#"):
            try:
                if token[1] in "xX":
                    return chr(int(token[2:], 16))
                return chr(int(token[1:]))
            except (ValueError, OverflowError):
                return match.group(0)
        return _XML_ENTITIES.get(token, match.group(0))

    return re.sub(r"&(#?[A-Za-z0-9]+);", repl, text)


class _XMLCursor:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def _find_gt(self, at: int, what: str) -> int:
        index = self.text.find(">", at)
        if index < 0:
            raise DatasetToolError(f"VOC 标注 XML 不合法：{what}未闭合")
        return index

    def parse_root(self):
        self._skip_junk()
        if self.pos >= len(self.text):
            return None
        return self._read_start_tag()

    def _skip_junk(self):
        """跳过空白、注释、处理指令与 DOCTYPE 声明。"""
        text = self.text
        while True:
            while self.pos < len(text) and text[self.pos] in " \t\r\n":
                self.pos += 1
            if self.pos >= len(text):
                return
            if text.startswith("<!--", self.pos):
                end = text.find("-->", self.pos)
                if end < 0:
                    raise DatasetToolError("VOC 标注 XML 不合法：注释未闭合")
                self.pos = end + 3
            elif text.startswith("<?", self.pos) or text.startswith("<!", self.pos):
                self.pos = self._find_gt(self.pos, "声明") + 1
            else:
                return

    def _read_start_tag(self):
        """读取当前标签，返回 (tag, text, children)；遇到结束标签返回 None。"""
        text = self.text
        end = self._find_gt(self.pos, "标签")
        head = text[self.pos + 1:end]
        self.pos = end + 1
        if head.startswith("/"):
            return None
        self_closing = head.rstrip().endswith("/")
        name_part = head[:-1] if self_closing else head
        tag = name_part.split(None, 1)[0].strip()
        if self_closing:
            return tag, "", []
        body, children = self._read_body(tag)
        return tag, body, children

    def _read_body(self, tag: str):
        """读取标签 tag 的正文，返回 (text, children)。"""
        text = self.text
        parts: List[str] = []
        children: List[Tuple[str, str, list]] = []
        while self.pos < len(text):
            lt = text.find("<", self.pos)
            if lt < 0:
                parts.append(text[self.pos:])
                break
            parts.append(text[self.pos:lt])
            self.pos = lt
            if text.startswith("</", lt):
                end = self._find_gt(lt, "结束标签") + 1
                closing = text[lt + 2:end - 1].split(None, 1)[0].strip()
                if closing != tag:
                    raise DatasetToolError(
                        f"VOC 标注 XML 不合法：标签不匹配 <{tag}> / </{closing}>")
                self.pos = end
                break
            if text.startswith("<!--", lt):
                end = text.find("-->", lt)
                if end < 0:
                    raise DatasetToolError("VOC 标注 XML 不合法：注释未闭合")
                self.pos = end + 3
                continue
            if text.startswith("<![CDATA[", lt):
                end = text.find("]]>", lt)
                if end < 0:
                    raise DatasetToolError("VOC 标注 XML 不合法：CDATA 未闭合")
                parts.append(text[lt + 9:end])
                self.pos = end + 3
                continue
            if text.startswith("<?", lt) or text.startswith("<!", lt):
                self.pos = self._find_gt(lt, "声明") + 1
                continue
            child = self._read_start_tag()
            if child is not None:
                children.append(child)
        return _decode_entities("".join(parts)), children


def parse_xml_root(xml_text: str):
    """把 XML 文本解析成 (tag, text, children) 树；无内容返回 None。"""
    return _XMLCursor(xml_text).parse_root()


def _find_child(node, tag: str):
    if node is None:
        return None
    for child in node[2]:
        if child[0] == tag:
            return child
    return None


def _child_text(node, tag: str) -> str:
    """子元素文本，去首尾空白；不存在返回空串。"""
    child = _find_child(node, tag)
    return (child[1] if child is not None else "").strip()


def _read_voc(images_dir: str, annotations_dir: str) -> Dataset:
    xml_files = sorted(os.path.join(annotations_dir, name)
                       for name in os.listdir(annotations_dir)
                       if os.path.splitext(name)[1].lower() == ".xml")
    classes: List[str] = []
    class_index: Dict[str, int] = {}
    rows: List[Tuple[str, str, Optional[Tuple[int, int]],
                       List[Tuple[str, Tuple[float, float, float, float]]]]] = []
    for xml_path in xml_files:
        try:
            with open(xml_path, "r", encoding="utf-8", errors="replace") as fh:
                root = parse_xml_root(fh.read())
        except DatasetToolError:
            raise
        except Exception as exc:
            raise DatasetToolError(f"读取 VOC 标注失败：{xml_path}: {exc}") from exc
        if root is None:
            continue
        filename = os.path.basename(_child_text(root, "filename"))
        size = None
        width = _child_text(_find_child(root, "size"), "width")
        height = _child_text(_find_child(root, "size"), "height")
        if width and height:
            size = (int(float(width)), int(float(height)))
        objects: List[Tuple[str, Tuple[float, float, float, float]]] = []
        for node in root[2]:
            if node[0] != "object":
                continue
            label = _child_text(node, "name")
            box_node = _find_child(node, "bndbox")
            if not label or box_node is None:
                continue
            try:
                xmin = float(_child_text(box_node, "xmin") or 0)
                ymin = float(_child_text(box_node, "ymin") or 0)
                xmax = float(_child_text(box_node, "xmax") or 0)
                ymax = float(_child_text(box_node, "ymax") or 0)
            except ValueError:
                continue
            objects.append((label, (xmin, ymin, xmax, ymax)))
        if filename:
            rows.append((filename, xml_path, size, objects))
            for label, _box in objects:
                if label not in class_index:
                    class_index[label] = len(classes)
                    classes.append(label)

    dataset = Dataset(classes=classes, image_paths=[])
    for filename, _xml_path, size, objects in rows:
        path = os.path.join(os.path.abspath(images_dir), filename)
        dataset.image_paths.append(path)
        if size:
            dataset.image_sizes[path] = size
        annotations = _empty_detections()
        for label, (xmin, ymin, xmax, ymax) in objects:
            index = class_index.get(label)
            if index is None:
                continue
            _append_detection(annotations, index, (xmin, ymin, xmax, ymax),
                              shape_type="rectangle")
        dataset.annotations[path] = annotations
    return dataset


def _read_labelme(images_dir: str, annotations_dir: str) -> Dataset:
    json_files = sorted(os.path.join(annotations_dir, name)
                        for name in os.listdir(annotations_dir)
                        if os.path.splitext(name)[1].lower() == ".json")
    classes: List[str] = []
    class_index: Dict[str, int] = {}
    # 每个 shape: (label, shape_type, points像素, polygon或None, bbox, group_id)
    rows = []
    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8", errors="replace") as fh:
            payload = json.load(fh)
        name = os.path.basename(str(payload.get("imagePath") or
                                    os.path.splitext(os.path.basename(json_path))[0]))
        width = payload.get("imageWidth")
        height = payload.get("imageHeight")
        size = (int(width), int(height)) if width and height else None
        shapes = []
        for shape in payload.get("shapes") or []:
            label = str(shape.get("label") or "")
            points = shape.get("points") or []
            shape_type = shape.get("shape_type") or "polygon"
            # point 只需 1 点，其余至少 2 点
            min_pts = 1 if shape_type == "point" else 2
            if not label or len(points) < min_pts:
                continue
            coords = [(float(point[0]), float(point[1])) for point in points]
            is_rectangle = shape_type == "rectangle"
            polygon = coords if shape_type == "polygon" else None
            box = _bounding_box(coords[:2] if is_rectangle and len(coords) == 2 else coords)
            group_id = shape.get("group_id")
            try:
                group_id = int(group_id) if group_id is not None else None
            except (TypeError, ValueError):
                group_id = None
            flags = shape.get("flags")
            flags = flags if isinstance(flags, dict) else {}
            shapes.append((label, shape_type, coords, polygon, box, group_id, flags))
            if label not in class_index:
                class_index[label] = len(classes)
                classes.append(label)
        rows.append((name, size, shapes))

    dataset = Dataset(classes=classes, image_paths=[])
    for name, size, shapes in rows:
        path = os.path.join(os.path.abspath(images_dir), name)
        dataset.image_paths.append(path)
        if size:
            dataset.image_sizes[path] = size
        annotations = _empty_detections()
        for label, shape_type, coords, polygon, box, group_id, flags in shapes:
            index = class_index.get(label)
            if index is None:
                continue
            _append_detection(annotations, index, box, shape_type=shape_type,
                              points=coords, group_id=group_id, mask=polygon,
                              flags=flags)
        dataset.annotations[path] = annotations
    return dataset


# ==========================================================================
# 写出
# ==========================================================================

def _quote_yaml(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _num(value: float) -> str:
    """像素坐标输出：整数不拖小数，否则保留两位。"""
    rounded = round(float(value), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _copy_image(path: str, images_dir: str) -> None:
    os.makedirs(images_dir, exist_ok=True)
    destination = os.path.join(images_dir, os.path.basename(path))
    if os.path.abspath(destination) != os.path.abspath(path):
        shutil.copy2(path, destination)


def _write_yolo(dataset: Dataset, layout: dict, first: bool,
                task: str = "det") -> None:
    if first:
        with open(layout["data_yaml"], "w", encoding="utf-8") as fh:
            fh.write("names: [" + ", ".join(_quote_yaml(name)
                                            for name in dataset.classes) + "]\n")
    for path in dataset.image_paths:
        _copy_image(path, layout["images"])
        width, height = dataset.size_of(path)
        annotations = dataset.annotations.get(path) or _empty_detections()
        stem = os.path.splitext(os.path.basename(path))[0]
        lines = _yolo_lines_for(annotations, task, width, height)
        label_path = os.path.join(layout["annotations"], stem + ".txt")
        os.makedirs(layout["annotations"], exist_ok=True)
        with open(label_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))


def _yolo_lines_for(det: "Detections", task: str, width: int, height: int) -> List[str]:
    """按任务类型把一张图的标注渲染成 YOLO 文本行，只输出与任务匹配的形状。"""
    from .shape_io import yolo_obb_line, yolo_seg_line, yolo_pose_line
    lines: List[str] = []
    if task == "pose":
        # 按 group_id 聚合：矩形 + 关键点
        groups: Dict[object, dict] = {}
        for i in range(len(det)):
            gid = det.group_of(i)
            if gid is None:
                continue
            slot = groups.setdefault(gid, {"box": None, "cls": 0, "kpts": []})
            if det.type_of(i) == "rectangle":
                slot["box"] = det.box_xyxy[i]
                slot["cls"] = det.class_id[i]
            elif det.type_of(i) == "point":
                pts = det.points_of(i)
                if pts:
                    slot["kpts"].append(pts[0])
        for slot in groups.values():
            if not slot["box"]:
                continue
            x1, y1, x2, y2 = slot["box"]
            line = yolo_pose_line(slot["cls"], (x1, y1, x2 - x1, y2 - y1),
                                  slot["kpts"], width, height)
            if line:
                lines.append(line)
        return lines
    for i in range(len(det)):
        cls = det.class_id[i]
        stype = det.type_of(i)
        pts = det.points_of(i)
        if task == "obb":
            if stype != "rotation" or not pts or len(pts) != 4:
                continue
            line = yolo_obb_line({"points": pts}, cls, width, height)
            if line:
                lines.append(line)
        elif task == "seg":
            if stype != "polygon" or not pts or len(pts) < 3:
                continue
            line = yolo_seg_line({"points": pts}, cls, width, height)
            if line:
                lines.append(line)
        else:  # det / hbb：所有形状取外接框
            x1, y1, x2, y2 = det.box_xyxy[i]
            if not width or not height:
                continue
            lines.append(f"{cls} {(x1 + x2) / 2 / width:.6f} "
                         f"{(y1 + y2) / 2 / height:.6f} "
                         f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")
    return lines


def _write_coco(dataset: Dataset, layout: dict) -> None:
    images_dir = layout["images"]
    os.makedirs(images_dir, exist_ok=True)
    images, annotations, counter = [], [], 0
    for index, path in enumerate(dataset.image_paths, start=1):
        _copy_image(path, images_dir)
        width, height = dataset.size_of(path)
        images.append({"id": index, "file_name": os.path.basename(path),
                       "width": width, "height": height})
        detections = dataset.annotations.get(path) or _empty_detections()
        for class_id, (x1, y1, x2, y2) in zip(detections.class_id,
                                              detections.box_xyxy):
            counter += 1
            annotations.append({
                "id": counter,
                "image_id": index,
                "category_id": class_id + 1,
                "bbox": [round(float(x1), 6), round(float(y1), 6),
                         round(float(x2 - x1), 6), round(float(y2 - y1), 6)],
                "area": round(float(x2 - x1) * float(y2 - y1), 6),
                "iscrowd": 0,
            })
    payload = {
        "info": {"description": "converted by LabelForge",
                 "version": IMPL_VERSION},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name, "supercategory": "none"}
                       for index, name in enumerate(dataset.classes)],
    }
    with open(layout["annotations"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_voc(dataset: Dataset, layout: dict) -> None:
    images_dir = layout["images"]
    annotations_dir = layout["annotations"]
    os.makedirs(annotations_dir, exist_ok=True)
    for path in dataset.image_paths:
        _copy_image(path, images_dir)
        width, height = dataset.size_of(path)
        detections = dataset.annotations.get(path) or _empty_detections()
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = os.path.basename(path)
        size_node = ET.SubElement(root, "size")
        ET.SubElement(size_node, "width").text = _num(width)
        ET.SubElement(size_node, "height").text = _num(height)
        ET.SubElement(size_node, "depth").text = "3"
        ET.SubElement(root, "segmented").text = "0"
        for class_id, (x1, y1, x2, y2) in zip(detections.class_id,
                                              detections.box_xyxy):
            node = ET.SubElement(root, "object")
            ET.SubElement(node, "name").text = dataset.classes[class_id]
            ET.SubElement(node, "pose").text = "Unspecified"
            ET.SubElement(node, "truncated").text = "0"
            ET.SubElement(node, "difficult").text = "0"
            box_node = ET.SubElement(node, "bndbox")
            ET.SubElement(box_node, "xmin").text = _num(x1)
            ET.SubElement(box_node, "ymin").text = _num(y1)
            ET.SubElement(box_node, "xmax").text = _num(x2)
            ET.SubElement(box_node, "ymax").text = _num(y2)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(annotations_dir,
                                os.path.splitext(os.path.basename(path))[0] + ".xml"),
                   encoding="utf-8", xml_declaration=True)


def _write_labelme(dataset: Dataset, layout: dict) -> None:
    target_dir = layout["annotations"]
    os.makedirs(target_dir, exist_ok=True)
    for path in dataset.image_paths:
        _copy_image(path, layout["images"])
        width, height = dataset.size_of(path)
        detections = dataset.annotations.get(path) or _empty_detections()
        shapes = []
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            stype = detections.type_of(i)
            pts = detections.points_of(i)
            group_id = detections.group_of(i)
            x1, y1, x2, y2 = detections.box_xyxy[i]
            if stype == "rotation" and pts and len(pts) == 4:
                shape = {"shape_type": "rotation",
                         "points": [[round(float(x), 6), round(float(y), 6)] for x, y in pts]}
            elif stype == "point" and pts:
                shape = {"shape_type": "point",
                         "points": [[round(float(pts[0][0]), 6), round(float(pts[0][1]), 6)]]}
            elif stype == "polygon" and pts and len(pts) >= 3:
                shape = {"shape_type": "polygon",
                         "points": [[round(float(x), 6), round(float(y), 6)] for x, y in pts]}
            else:
                shape = {"shape_type": "rectangle",
                         "points": [[round(float(x1), 6), round(float(y1), 6)],
                                    [round(float(x2), 6), round(float(y2), 6)]]}
            shape.update({"label": dataset.classes[class_id],
                          "group_id": group_id, "flags": detections.flags_of(i)})
            shapes.append(shape)
        payload = {
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": os.path.basename(path),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_path = os.path.join(target_dir,
                                 os.path.splitext(os.path.basename(path))[0] + ".json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)


def _write_chunk(chunk: "Dataset", output_format: str, layout: dict,
                 first: bool, task: str = "det") -> None:
    if output_format == "yolo":
        _write_yolo(chunk, layout, first, task)
    elif output_format == "voc":
        _write_voc(chunk, layout)
    else:
        _write_labelme(chunk, layout)


def _slice_dataset(dataset: Dataset, paths: List[str]) -> Dataset:
    """取子集，保持写出可分块（COCO 例外，标注是单个 json）。"""
    return Dataset(
        classes=list(dataset.classes),
        image_paths=list(paths),
        annotations={path: dataset.annotations[path]
                     for path in paths if path in dataset.annotations},
        image_sizes={path: dataset.image_sizes[path]
                     for path in paths if path in dataset.image_sizes},
    )


# ==========================================================================
# 对外入口
# ==========================================================================

def load_dataset(input_format: str, images_dir: str, annotations_path: str,
                 data_yaml_path: Optional[str] = None,
                 task: str = "auto") -> Dataset:
    """按指定格式加载数据集，返回 Dataset。

    annotations_path 对 yolo/voc/labelme 是目录，对 coco 是 json 文件。
    task 决定 YOLO 标注行的解析方式（det/seg/obb/pose/auto）。
    """
    fmt = (input_format or "").strip().lower()
    if fmt not in INPUT_FORMATS:
        raise DatasetToolError(f"不支持的输入格式：{input_format}")
    images_dir = _check_dir(images_dir, "图片目录")
    try:
        if fmt == "yolo":
            annotations_dir = _check_dir(annotations_path, "标注目录")
            yaml_path = _check_file(data_yaml_path, "data.yaml")
            return _read_yolo(images_dir, annotations_dir, yaml_path, task)
        if fmt == "coco":
            json_path = _check_file(annotations_path, "COCO 标注文件")
            return _read_coco(images_dir, json_path)
        if fmt == "voc":
            annotations_dir = _check_dir(annotations_path, "标注目录")
            return _read_voc(images_dir, annotations_dir)
        annotations_dir = _check_dir(annotations_path, "标注目录")
        return _read_labelme(images_dir, annotations_dir)
    except DatasetToolError:
        raise
    except Exception as exc:
        raise DatasetToolError(f"读取 {fmt} 数据集失败：{exc}") from exc


#: 任务类型 -> 该任务在 YOLO 输出时会保留的形状类型
_TASK_SHAPE = {
    "seg": {"polygon"},
    "obb": {"rotation"},
    "pose": {"rectangle", "point"},
    # det/hbb/auto 对所有形状取外接框，不丢弃
}


def validate_dataset(dataset, task: str = "auto") -> ValidationReport:
    """检查数据集的图片、标注、类别一致性。

    task 为 seg/obb/pose 时，额外统计与任务不匹配、转 YOLO 会被跳过的标注，
    并按形状类型给出明细提示（避免多类型混标时静默丢框）。
    """
    classes = list(getattr(dataset, "classes", []) or [])
    image_paths = list(getattr(dataset, "image_paths", []) or [])
    annotations = dict(getattr(dataset, "annotations", {}) or {})
    errors: List[str] = []
    warnings: List[str] = []

    if not image_paths:
        errors.append("数据集为空，没有可转换的图片")
    if not classes:
        warnings.append("数据集没有类别定义")

    missing_files = [path for path in image_paths if not os.path.isfile(path)]
    for path in missing_files[:5]:
        errors.append(f"图片文件缺失：{path}")
    if len(missing_files) > 5:
        errors.append(f"另有 {len(missing_files) - 5} 个图片文件缺失")

    seen: Dict[str, str] = {}
    for path in image_paths:
        name = os.path.basename(path).lower()
        if name in seen and seen[name] != path:
            errors.append(f"图片文件名重复，转换会互相覆盖：{name}")
        seen[name] = path

    annotation_count = 0
    max_class_id = len(classes) - 1
    bad_class = 0
    empty_images = 0
    shape_counts: Dict[str, int] = {}
    for path in image_paths:
        detections = annotations.get(path)
        if detections is None:
            warnings.append(f"图片没有对应标注：{os.path.basename(path)}")
            continue
        count = len(detections)
        annotation_count += count
        if count == 0:
            empty_images += 1
            continue
        for i in range(count):
            if hasattr(detections, "type_of"):
                stype = detections.type_of(i)
                shape_counts[stype] = shape_counts.get(stype, 0) + 1
        class_ids = getattr(detections, "class_id", None)
        if class_ids is None:
            bad_class += count
            continue
        for class_id in class_ids:
            if class_id is None or int(class_id) < 0 or int(class_id) > max_class_id:
                bad_class += 1
    if bad_class:
        errors.append(f"存在 {bad_class} 个标注的类别 ID 超出类别列表范围")
    if empty_images:
        warnings.append(f"{empty_images} 张图片没有标注框")

    # 多类型混标：转 YOLO 时只保留与任务匹配的形状，其余会被跳过
    keep_types = _TASK_SHAPE.get((task or "").lower())
    if keep_types and shape_counts:
        kept = sum(n for t, n in shape_counts.items() if t in keep_types)
        skipped = sum(n for t, n in shape_counts.items() if t not in keep_types)
        if skipped:
            detail = "、".join(f"{t}×{n}" for t, n in sorted(shape_counts.items()))
            warnings.append(
                f"当前任务[{task}]只导出 {'/'.join(sorted(keep_types))} 形状，"
                f"将跳过 {skipped} 个其它类型标注（本数据集形状分布：{detail}）")
        if kept == 0:
            errors.append(
                f"当前任务[{task}]没有可导出的标注：数据集里没有 "
                f"{'/'.join(sorted(keep_types))} 形状")
    if len(warnings) > 20:
        extra = len(warnings) - 20
        warnings = warnings[:20] + [f"另有 {extra} 条提示已省略"]

    return ValidationReport(
        image_count=len(image_paths),
        annotation_count=annotation_count,
        classes=classes,
        errors=errors,
        warnings=warnings,
    )


def _prepare_output_dir(output_dir: str, overwrite: bool) -> str:
    if not output_dir:
        raise DatasetToolError("未指定输出目录")
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise DatasetToolError(f"输出路径不是目录：{output_dir}")
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if not overwrite:
            raise DatasetToolError(
                f"输出目录不为空：{output_dir}\n请选择空目录，或确认覆盖后重试。")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def output_layout(output_format: str, output_dir: str) -> dict:
    """返回指定输出格式的目录结构，便于 UI 提示与测试断言。"""
    fmt = (output_format or "").strip().lower()
    output_dir = os.path.abspath(output_dir) if output_dir else ""
    if fmt == "yolo":
        return {
            "images": os.path.join(output_dir, "images"),
            "annotations": os.path.join(output_dir, "labels"),
            "data_yaml": os.path.join(output_dir, "data.yaml"),
        }
    if fmt == "coco":
        return {
            "images": os.path.join(output_dir, "images"),
            "annotations": os.path.join(output_dir, COCO_ANNOTATION_NAME),
        }
    if fmt == "voc":
        return {
            "images": os.path.join(output_dir, "images"),
            "annotations": os.path.join(output_dir, "annotations"),
        }
    if fmt == "labelme":
        # LabelMe 使用 sidecar json，与 LabelForge 主存储一致，输出到同一目录
        return {
            "images": output_dir,
            "annotations": output_dir,
        }
    raise DatasetToolError(f"不支持的输出格式：{output_format}")


def convert_dataset(dataset, output_format: str, output_dir: str,
                    overwrite: bool = False,
                    on_progress: Optional[Callable] = None,
                    is_interrupted: Optional[Callable] = None,
                    task: str = "det") -> dict:
    """把已加载的数据集写出为目标格式。

    返回 {"output_dir", "format", "layout", "image_count", "annotation_count",
    "cancelled"}。取消或失败时不会保留半成品目录。
    """
    fmt = (output_format or "").strip().lower()
    if fmt not in OUTPUT_FORMATS:
        raise DatasetToolError(f"不支持的输出格式：{output_format}")

    # 只有导出 YOLO 时任务才影响形状取舍；其它格式按 auto 校验（不误报跳过）
    report = validate_dataset(dataset, task=task if fmt == "yolo" else "auto")
    if not report.ok:
        raise DatasetToolError("数据集校验未通过：\n" + "\n".join(report.errors))

    is_interrupted = is_interrupted or (lambda: False)
    output_dir = _prepare_output_dir(output_dir, overwrite)
    layout = output_layout(fmt, output_dir)
    image_paths = list(dataset.image_paths)
    total = len(image_paths)
    done = 0

    try:
        if fmt == "coco":
            # COCO 标注是单个 json，必须整体写出，无法分块
            if is_interrupted():
                raise _Cancelled()
            _write_coco(dataset, layout)
            done = total
            if on_progress:
                on_progress(done, total)
        else:
            for start in range(0, total, _CHUNK_SIZE):
                if is_interrupted():
                    raise _Cancelled()
                paths = image_paths[start:start + _CHUNK_SIZE]
                _write_chunk(_slice_dataset(dataset, paths), fmt, layout,
                             first=(start == 0), task=task)
                done += len(paths)
                if on_progress:
                    on_progress(done, total)
    except _Cancelled:
        shutil.rmtree(output_dir, ignore_errors=True)
        return {
            "output_dir": output_dir,
            "format": fmt,
            "layout": layout,
            "image_count": 0,
            "annotation_count": 0,
            "cancelled": True,
        }
    except DatasetToolError:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise DatasetToolError(f"写出 {fmt} 数据集失败：{exc}") from exc

    return {
        "output_dir": output_dir,
        "format": fmt,
        "layout": layout,
        "image_count": report.image_count,
        "annotation_count": report.annotation_count,
        "cancelled": False,
    }


def convert_paths(input_format: str, images_dir: str, annotations_path: str,
                  output_format: str, output_dir: str,
                  data_yaml_path: Optional[str] = None,
                  overwrite: bool = False,
                  on_progress: Optional[Callable] = None,
                  is_interrupted: Optional[Callable] = None,
                  task: str = "auto") -> dict:
    """加载 + 校验 + 转换的一站式入口，供 UI 在后台线程调用。

    task 同时决定 YOLO 输入的解析方式与 YOLO 输出的行格式；
    输入为 auto（非 YOLO 时的默认）时输出退化为 det。
    """
    dataset = load_dataset(input_format, images_dir, annotations_path,
                           data_yaml_path=data_yaml_path, task=task)
    out_task = task if task in ("det", "seg", "obb", "pose") else "det"
    result = convert_dataset(dataset, output_format, output_dir,
                             overwrite=overwrite,
                             on_progress=on_progress,
                             is_interrupted=is_interrupted,
                             task=out_task)
    report_task = out_task if output_format.strip().lower() == "yolo" else "auto"
    result["report"] = validate_dataset(dataset, task=report_task).as_dict()
    return result


class _Cancelled(Exception):
    """内部信号：用户取消。"""
