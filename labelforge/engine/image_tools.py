"""图像编辑工具的操作清单与处理链。

对单张图按顺序执行若干变换，可选只在选区内生效。变换本身全部来自
`engine.augmenter` 的注册表，本模块只负责「清单 + 编排 + 蒙版合成」。
"""
import inspect
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from .augmenter import get_all_transforms
from .augmenter import spec as augmenter_spec
from .augmenter.base import (
    qimage_gray_to_array,
    qimage_to_rgb_array,
    rgb_array_to_qimage,
)
from .augmenter.spec import (
    GEOMETRIC_KEYS, OPERATION_ORDER, OP_SPECS, default_params, group_of,
    is_geometric, keys_in, label_of,
)
from .augmenter.spec import params_of as _params_of

# 预览降采样的长边上限；实时预览只在这个尺度上跑处理链
PREVIEW_MAX_SIDE = 1280

# 分组名（界面按此顺序渲染）
GROUP_GEOMETRY = augmenter_spec.GEOMETRY
GROUP_COLOR = augmenter_spec.COLOR
GROUP_FILTER = augmenter_spec.FILTER
GROUP_NOISE = augmenter_spec.NOISE

#: key -> (分组, 显示名, 参数五元组列表)。参数五元组是 (名字, 类型, 下限, 上限, 默认)，
#: 供界面按同一份规格生成控件；真正的默认值只有算子类里那一处。
OPERATION_META: dict = {
    key: (item.group, item.label, [p.as_tuple() for p in item.params])
    for key, item in OP_SPECS.items()
}

#: 预设文件的版本号；将来改格式时靠它区分
PRESET_VERSION = 1



def params_of(key: str) -> list:
    """参数五元组列表 (名字, 类型, 下限, 上限, 默认)，供界面生成控件。"""
    return [item.as_tuple() for item in _params_of(key)]


def build_transform(name: str, params: Optional[dict] = None,
                    seed: Optional[int] = None):
    """按 key 与参数构造变换实例；未知 key 或未知参数名抛 ValueError。"""
    registry = get_all_transforms()
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"未知的图像编辑操作: {name}")
    kwargs = dict(params or {})
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "rng"}
    unknown = set(kwargs) - accepted
    if unknown:
        raise ValueError(
            f"{name} 不接受参数 {sorted(unknown)}，可接受 {sorted(accepted)}")
    if seed is not None:
        kwargs["rng"] = random.Random(seed)
    return cls(**kwargs)


def apply_chain(image: QImage, specs: List[dict],
                seed: Optional[int] = None) -> QImage:
    """顺序执行处理链。

    每步用 `seed + 序号` 派生随机源：整条链可复现，且各步互不共享序列。
    """
    out = image
    for index, spec in enumerate(specs or []):
        step_seed = None if seed is None else seed + index
        transform = build_transform(spec["name"], spec.get("params"), step_seed)
        out, _ = transform.apply(out, [], out.width(), out.height())
    return out


def apply_chain_with_mask(image: QImage, specs: List[dict], mask: QImage,
                          seed: Optional[int] = None) -> QImage:
    """只在蒙版白色区域应用处理链，其余保留原图。

    处理链改变了画布尺寸（几何类）时无法逐像素合成，直接返回处理结果。
    """
    processed = apply_chain(image, specs, seed)
    if not specs:
        return processed
    original = qimage_to_rgb_array(image)
    result = qimage_to_rgb_array(processed)
    if result.shape != original.shape:
        return processed
    gray = qimage_gray_to_array(_scaled_mask(mask, image))
    inside = gray > 127
    return rgb_array_to_qimage(np.where(inside[:, :, None], result, original))


def _scaled_mask(mask: QImage, image: QImage) -> QImage:
    if mask.width() == image.width() and mask.height() == image.height():
        return mask
    return mask.scaled(image.width(), image.height(),
                       Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


def downsample_for_preview(image: QImage,
                           max_side: int = PREVIEW_MAX_SIDE) -> QImage:
    """长边超过 max_side 时等比缩小，预览用。"""
    if image.isNull():
        return image
    if max(image.width(), image.height()) <= max_side:
        return image
    return image.scaled(max_side, max_side,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)


def selection_mask_from_points(points: List[Tuple[float, float]],
                               width: int, height: int,
                               mode: str) -> QImage:
    """归一化选区点 → 灰度蒙版，白色为选区内。

    points 少于 2 个（单击未拖动）时返回全黑，处理链退化为不生效。
    """
    mask = QImage(width, height, QImage.Format_Grayscale8)
    mask.fill(0)
    if not points or len(points) < 2:
        return mask
    pts = [(_clamp01(p[0]) * width, _clamp01(p[1]) * height) for p in points]
    painter = QPainter(mask)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setPen(QPen(QColor(255, 255, 255), 1))
    painter.setBrush(QColor(255, 255, 255))
    if mode in ("rect", "ellipse"):
        x0, x1 = sorted((pts[0][0], pts[1][0]))
        y0, y1 = sorted((pts[0][1], pts[1][1]))
        rect = QRectF(x0, y0, x1 - x0, y1 - y0)
        if mode == "rect":
            painter.drawRect(rect)
        else:
            painter.drawEllipse(rect)
    else:
        painter.drawPolygon(QPolygonF([QPointF(p[0], p[1]) for p in pts]))
    painter.end()
    return mask


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


def save_result(image: QImage, path: str) -> bool:
    """把处理结果写成图片文件。

    jpg/bmp 不带透明通道，有 alpha 时先合成到白底，否则透明区会变成黑块。
    目标目录不存在时先建出来，返回值是 `QImage.save` 的结果。
    """
    if image.isNull():
        return False
    ext = os.path.splitext(path)[1].lower()
    target = _opaque_for_format(image, ext) if ext in (".jpg", ".jpeg", ".bmp") else image
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    return bool(target.save(path))


def _opaque_for_format(image: QImage, ext: str) -> QImage:
    if not image.hasAlphaChannel():
        return image
    canvas = QImage(image.width(), image.height(), QImage.Format_RGB32)
    canvas.fill(QColor(255, 255, 255))
    painter = QPainter(canvas)
    painter.drawImage(0, 0, image)
    painter.end()
    return canvas


def specs_to_preset(specs: List[dict]) -> dict:
    """处理项列表 → 预设字典。

    每一步同时带两份视图：`params` 是单值（图像编辑用），`ranges` 是区间
    （批量增强用）。只给单值时区间就是「上下限相同」，只给区间时单值取中点，
    这样两个面板存出来的预设彼此都能打开。
    """
    steps = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if name not in OP_SPECS:
            continue
        params, ranges = _clean_step(name, spec.get("params"), spec.get("ranges"))
        step = {"name": name, "params": params}
        if ranges:
            step["ranges"] = {key: list(pair) for key, pair in ranges.items()}
        steps.append(step)
    return {"version": PRESET_VERSION, "steps": steps}


def specs_from_preset(data) -> List[dict]:
    """预设字典 → 处理项列表（每项都补齐 params 与 ranges 两份）。

    预设是用户之间传来传去的文件，坏了不能崩：结构不对返回空列表，未知
    算子与算子不认识的参数被跳过，剩下的照常执行。
    """
    if not isinstance(data, dict):
        return []
    steps = data.get("steps")
    if not isinstance(steps, list):
        return []
    result = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        if name not in OP_SPECS:
            continue
        params, ranges = _clean_step(name, step.get("params"), step.get("ranges"))
        result.append({"name": name, "params": params, "ranges": ranges})
    return result


def _clean_step(name, raw_params, raw_ranges):
    """一步操作 → (单值参数, 区间参数)，都按算子规格收敛一遍。"""
    params, ranges = {}, {}
    for item in OP_SPECS[name].params:
        default = item.default
        numeric = isinstance(default, (int, float)) and not isinstance(default, bool)
        raw_range = (raw_ranges or {}).get(item.name) if isinstance(raw_ranges, dict) else None
        if numeric and raw_range is not None:
            pair = _as_pair(raw_range)
            if pair is None:
                continue
            ranges[item.name] = pair
            params[item.name] = _midpoint(pair, default)
            continue
        if not isinstance(raw_params, dict) or item.name not in raw_params:
            continue
        value = _coerce_param(raw_params[item.name], default)
        params[item.name] = value
        if numeric:
            ranges[item.name] = (value, value)
    return params, ranges


def _as_pair(value):
    """[下限, 上限] → (小, 大)；结构不对返回 None。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low, high = _as_number(value[0]), _as_number(value[1])
    if low is None or high is None:
        return None
    return (min(low, high), max(low, high))


def _midpoint(pair, default):
    """区间中点，按默认值决定要不要取整。"""
    middle = (pair[0] + pair[1]) / 2.0
    return int(round(middle)) if isinstance(default, int) else float(middle)


#: 预设槽位：A~E 五个，满了覆盖最旧的
PRESET_SLOTS = ("A", "B", "C", "D", "E")

#: 配置里存预设的那一段
PRESETS_KEY = "image_tools_presets"


def preset_name(slot: str) -> str:
    """槽位 → 界面上的名字。"""
    return f"预设{slot}"


def load_presets() -> List[dict]:
    """读配置里的预设，按槽位 A~E 返回。

    每项 `{"slot": "A", "steps": [...], "saved_at": "2026-11-21 12:00:00"}`。
    配置是手改得坏的普通 JSON：结构不对的整项丢掉，未知算子按步骤规则跳过。
    """
    from ..core.config_manager import load_config

    raw = load_config().get(PRESETS_KEY)
    return normalize_presets(raw)


def normalize_presets(raw) -> List[dict]:
    """配置里的预设 → 合法槽位列表（按 A~E 顺序、同一槽位只留一条）。"""
    if not isinstance(raw, list):
        return []
    seen = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip().upper()
        if slot not in PRESET_SLOTS or slot in seen:
            continue
        steps = specs_from_preset({"steps": item.get("steps")})
        if not steps:
            continue
        seen[slot] = {"slot": slot, "steps": steps,
                      "saved_at": str(item.get("saved_at") or "")}
    return [seen[slot] for slot in PRESET_SLOTS if slot in seen]


def preset_summary(steps) -> str:
    """预设内容的一句话摘要，给数据增强面板的下拉框用。"""
    labels = []
    for step in steps or []:
        spec = OP_SPECS.get(step.get("name"))
        if spec is not None:
            labels.append(spec.label)
    return "、".join(labels)


def store_preset(specs) -> str:
    """把当前设置存成预设，返回落到的槽位（A~E）。

    依次占空槽；五个都满了就覆盖最旧的那个——按 `saved_at` 比，时间相同时
    按槽位顺序，保证结果确定。
    """
    from ..core.config_manager import load_config, save_config
    from datetime import datetime

    steps = specs_to_preset(specs)["steps"]
    if not steps:
        return ""
    presets = load_presets()
    used = {item["slot"]: item for item in presets}
    free = next((slot for slot in PRESET_SLOTS if slot not in used), None)
    if free is None:
        free = min(presets, key=lambda item: (item["saved_at"], item["slot"]))["slot"]
    used[free] = {"slot": free, "steps": steps,
                  "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    config = load_config()
    config[PRESETS_KEY] = [used[slot] for slot in PRESET_SLOTS if slot in used]
    if not save_config(config):
        # 写不进去就当没存，别让界面报「已存为预设A」却什么都没落盘
        return ""
    return free


def preset_steps_of(slot: str) -> List[dict]:
    """某个槽位的处理项；槽位为空返回空列表。"""
    for item in load_presets():
        if item["slot"] == slot:
            return item["steps"]
    return []


def derived_output_path(source: str, suffix: str = "_edit",
                        folder: str = None) -> str:
    """按原图名派生一个不冲突的新文件名。

    `folder` 为空时落在原图旁边，给了就落在那个文件夹里。图像编辑不改原图：
    结果一律写成新文件，重名时依次用 `_2`、`_3`。
    """
    folder = folder or os.path.dirname(os.path.abspath(source))
    stem, ext = os.path.splitext(os.path.basename(source))
    ext = ext or ".png"
    candidate = os.path.join(folder, f"{stem}{suffix}{ext}")
    index = 2
    while os.path.exists(candidate) and index < 1000:
        candidate = os.path.join(folder, f"{stem}{suffix}_{index}{ext}")
        index += 1
    return candidate


#: 输出目录：与数据集同级的 exports
OUTPUT_SUBDIR = ("exports",)

#: 文件名里最多带多少个字符的操作名，超了就截断
_MAX_SUFFIX_CHARS = 60


def default_output_dir(image_path: str) -> str:
    """默认输出目录：`<数据集父目录>/exports`。

    数据集就是图片所在的那一层，输出目录与它同级，不写进数据集本身。
    """
    dataset = os.path.dirname(os.path.abspath(image_path or ""))
    base = os.path.dirname(dataset) or dataset
    return os.path.join(base, *OUTPUT_SUBDIR)


def ops_suffix(specs) -> str:
    """按处理顺序拼出文件名后缀，例如 `_rotate90_invert_gauss`。

    用算子键名而不是界面名称：文件名不随界面语言变化，也全是 ASCII。
    拼太长就截断并以「等」结尾。
    """
    names = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        key = spec.get("name")
        if key in OPERATION_META:
            names.append(str(key))
    if not names:
        return "_edit"
    suffix = ""
    for name in names:
        candidate = f"{suffix}_{name}"
        if len(candidate) > _MAX_SUFFIX_CHARS:
            return f"{suffix}_etc"
        suffix = candidate
    return suffix


#: 配置文件里存图像编辑参数的那一段
PARAMS_KEY = "image_tools_params"


def normalize_params(raw) -> dict:
    """配置里读回的参数：未知算子、未知参数、类型不对的一律丢掉。

    配置是手改得坏的普通 JSON，读进来先过一遍；面板只认这里剩下的东西。
    """
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, params in raw.items():
        if key not in OP_SPECS or not isinstance(params, dict):
            continue
        defaults = OP_SPECS[key].defaults()
        clean = {}
        for name, value in params.items():
            if name in defaults:
                clean[name] = _coerce_param(value, defaults[name])
        if clean:
            result[key] = clean
    return result


def _coerce_param(value, default):
    """按默认值的类型转一下；转不动就退回默认值。"""
    if isinstance(default, bool):
        return bool(value) if isinstance(value, (bool, int)) else default
    if isinstance(default, (int, float)):
        number = _as_number(value)
        if number is None:
            return default
        return int(number) if isinstance(default, int) else float(number)
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return default


def _as_number(value):
    """数字或数字字符串 -> float；其它返回 None（bool 不算数字）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def load_saved_params() -> dict:
    """读取上次用过的参数（不含勾选）。"""
    from ..core.config_manager import load_config

    return normalize_params(load_config().get(PARAMS_KEY))


def save_saved_params(entries) -> bool:
    """把参数写进配置，方便下次打开面板接着调。"""
    from ..core.config_manager import load_config, save_config

    config = load_config()
    config[PARAMS_KEY] = normalize_params(entries)
    return save_config(config)
