"""抠图编排：裁块、推理、裁边、命名、存盘。

本模块不依赖 Qt，也不读配置：目录、阈值、外扩比例都由调用方传进来，
因此可以只用 numpy + 一个注入的模型替身跑通全链路。
"""

import os
import re

import numpy as np

from .matting import alpha_bbox, apply_threshold, compose_rgba
from .paste_item import PasteItem

#: Windows 文件名里不允许出现的字符
INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]+')
#: 序号补零宽度：001 起，第 1000 张自然变成四位
INDEX_WIDTH = 3
#: 存盘临时文件后缀：先写它再 `os.replace`，中途失败/取消都不留半截 PNG
TEMP_SUFFIX = ".part"
#: 能靠顶点算外接框的形状（旋转框的 x/y/width/height 是轴对齐外接框）
POINT_SHAPES = ("polygon", "rotation")
#: 短边小于该像素的目标抠不出有意义素材，批量时跳过
MIN_TARGET_SHORT_SIDE = 8


def default_output_dir(background_dir):
    """默认输出目录：<背景图目录>_matting_output/。"""
    base = os.path.abspath(background_dir or ".")
    return _Path(base + "_matting_output")


def expand_box(box, image_size, margin_ratio=0.05):
    """按短边的 margin_ratio 外扩外接框，并夹紧到图内。box=(x, y, w, h)。"""
    x, y, w, h = box
    img_w, img_h = image_size
    margin = max(2, int(round(min(w, h) * margin_ratio)))
    left = max(0, int(x) - margin)
    top = max(0, int(y) - margin)
    right = min(img_w, int(x + w) + margin)
    bottom = min(img_h, int(y + h) + margin)
    return (left, top, right - left, bottom - top)


def sanitise_name(value, fallback):
    """替换文件名非法字符；清空后（含只有点号）回落到 fallback。"""
    cleaned = INVALID_NAME_CHARS.sub("_", str(value or "")).strip()
    cleaned = cleaned.strip(".")
    return cleaned or fallback


def should_skip_box(box):
    """短边小于 8px 的目标抠不出有意义素材，批量时跳过。"""
    _, _, w, h = box
    return min(w, h) < MIN_TARGET_SHORT_SIDE


def batch_key(source_name, label):
    """批量去重键 `<背景主名>_<标签>`：同一张图的同一标签只抠一次。"""
    stem = sanitise_name(
        os.path.splitext(os.path.basename(str(source_name or "")))[0], "image")
    return f"{stem}_{sanitise_name(label, 'object')}"


def next_matte_path(directory, label, source_name):
    """生成不覆盖已有文件的输出路径：<源图名>_<标签>_<三位序号>.png。

    序号从 001 起，占用就往上加；已存在的文件一律不覆盖。
    """
    os.makedirs(directory, exist_ok=True)
    stem = sanitise_name(
        os.path.splitext(os.path.basename(str(source_name or "")))[0], "image")
    safe_label = sanitise_name(label, "object")
    index = 1
    while True:
        candidate = os.path.join(os.fspath(directory),
                                 f"{stem}_{safe_label}_{index:03d}.png")
        if not os.path.exists(candidate):
            return _Path(candidate)
        index += 1


class _Path:
    """轻量路径包装：调用方要的只是名字、`fspath` 与写字节。"""

    def __init__(self, value):
        self.value = value
        self.name = os.path.basename(value)

    def __fspath__(self):
        return self.value

    def __str__(self):
        return self.value

    def write_bytes(self, data):
        with open(self.value, "wb") as handle:
            handle.write(data)

    def exists(self):
        return os.path.exists(self.value)


def paste_item_from_matte(pixmap, rect, label, path=None):
    """抠图产物一律建成默认 det 的贴图项，不继承来源标注的形状。

    path 记下产物落盘位置，后续「裁剪素材」要覆盖写回这个文件。
    """
    return PasteItem(pixmap=pixmap, rect=rect, label=label, path=path)


def target_box(target):
    """区域/标注 -> 外接框 (x, y, w, h)。

    多边形与旋转框按顶点算：它们的 x/y/width/height 未必被同步维护，
    顶点是唯一可信的来源。
    """
    points = target.get('points') or []
    if target.get('shape_type') in POINT_SHAPES and len(points) >= 2:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return (float(target['x']), float(target['y']),
            float(target['width']), float(target['height']))


def target_polygon(target):
    """区域/标注 -> 顶点列表；矩形没有顶点，返回 None。"""
    points = target.get('points') or []
    if target.get('shape_type') in POINT_SHAPES and len(points) >= 3:
        return [(float(point[0]), float(point[1])) for point in points]
    return None


def box_pixel_bounds(box, origin, shape):
    """框在裁块里的像素边界 (x0, y0, x1, y1)，夹紧到裁块范围内。"""
    height, width = int(shape[0]), int(shape[1])
    x0 = max(0, int(round(box[0] - origin[0])))
    y0 = max(0, int(round(box[1] - origin[1])))
    x1 = min(width, int(round(box[0] + box[2] - origin[0])))
    y1 = min(height, int(round(box[1] + box[3] - origin[1])))
    return x0, y0, x1, y1


def box_mask(shape, box, origin=(0, 0)):
    """矩形 -> (H, W) 的 0/1 蒙版；origin 是裁块左上角在原图中的坐标。"""
    x0, y0, x1, y1 = box_pixel_bounds(box, origin, shape)
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float32)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1.0
    return mask


def polygon_mask(shape, polygon, origin=(0, 0)):
    """多边形 -> (H, W) 的 0/1 蒙版；origin 是裁块左上角在原图中的坐标。"""
    import cv2

    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(
        [[point[0] - origin[0], point[1] - origin[1]] for point in polygon],
        dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(np.float32)


def save_matte_png(rgba, path):
    """RGBA -> PNG 落盘；先写临时文件再 `os.replace`，不留半截文件。

    不用 `cv2.imwrite`：Windows 上它按 ANSI 打开路径，中文目录写不进去。
    """
    import cv2

    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError(f"期望 (H, W, 4) 的 RGBA，实际 {arr.shape}")
    # 编码器按 BGR(A) 理解数组，PNG 文件里的通道顺序仍是 RGBA
    bgra = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
    ok, buffer = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError("PNG 编码失败")

    target = os.fspath(path)
    temp_path = target + TEMP_SUFFIX
    try:
        with open(temp_path, "wb") as handle:
            handle.write(buffer.tobytes())
        os.replace(temp_path, target)
    except OSError:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise
    return target


def matte_crop(model, rgb, box, threshold=0.0, margin_ratio=0.05,
               polygon=None, cancel=None):
    """对 rgb 的 box 区域抠图。

    返回 (rgba, (x, y, w, h))：rgba 已裁到目标外接框，元组是该框在原图中的位置；
    没有可抠像素时返回 (None, None)。polygon 给定时与模型输出相交，
    区域之外的像素一律透明。cancel 在步骤之间调用，返回真即中止。
    """
    if cancel is not None and cancel():
        return None, None

    x, y, w, h = expand_box(box, (rgb.shape[1], rgb.shape[0]), margin_ratio)
    patch = rgb[y:y + h, x:x + w]
    if patch.size == 0:
        return None, None

    alpha = apply_threshold(model.predict(patch), threshold)
    if cancel is not None and cancel():
        return None, None
    # 外扩只是让模型多看一圈上下文：落盘像素必须留在调用方给的框内，
    # 否则区域抠图会把框外的内容一起抠进素材。
    alpha = alpha * box_mask(alpha.shape, box, (x, y))
    if polygon:
        alpha = alpha * polygon_mask(alpha.shape, polygon, (x, y))

    bbox = alpha_bbox(alpha, pad=1)
    if bbox is None:
        return None, None

    top, left, bottom, right = bbox
    # 软边的 1px pad 同样不能溢出原框
    bx0, by0, bx1, by1 = box_pixel_bounds(box, (x, y), alpha.shape)
    left, top = max(left, bx0), max(top, by0)
    right, bottom = min(right, bx1), min(bottom, by1)
    if right - left <= 0 or bottom - top <= 0:
        return None, None

    rgba = compose_rgba(patch, alpha)[top:bottom, left:right]
    return rgba, (x + left, y + top, right - left, bottom - top)


class MatteResult:
    """一次抠图编排的产物：位图、它在原图中的位置、落盘路径、标签。"""

    __slots__ = ("rgba", "rect", "path", "label")

    def __init__(self, rgba, rect, path, label):
        self.rgba = rgba
        self.rect = rect
        self.path = path
        self.label = label

    def __repr__(self):
        x, y, w, h = self.rect
        return (f"MatteResult({self.path!r}, x={x}, y={y}, "
                f"w={w}, h={h})")


def matte_selection(model, rgb, box, label, source_name, output_dir,
                    threshold=0.0, margin_ratio=0.05, polygon=None,
                    cancel=None):
    """把 rgb 上 box 指的这块抠成素材并落盘，返回 MatteResult。

    目录由调用方定（配置值或默认的 `<背景图目录>_matting_output/`），
    不存在时随命名一起创建；取消或没有前景时返回 None 且不写任何文件。
    """
    rgba, rect = matte_crop(model, rgb, box, threshold=threshold,
                            margin_ratio=margin_ratio, polygon=polygon,
                            cancel=cancel)
    if rgba is None:
        return None

    path = next_matte_path(output_dir, label, source_name)
    save_matte_png(rgba, path)
    return MatteResult(rgba=rgba, rect=rect, path=os.fspath(path),
                       label=label or "object")


class BatchMatteReport:
    """一次批量抠图的账本：成功、跳过（小目标/重复/无前景）、失败。

    计数口径是「目标数」：不取消时 saved + skipped + failed 恒等于 total，
    这样状态栏报出来的三个数加起来就是批量处理了多少个目标。
    """

    __slots__ = ("saved", "skipped", "failed", "cancelled", "total")

    def __init__(self):
        self.saved = []
        self.skipped = []
        self.failed = []
        self.cancelled = False
        self.total = 0

    def counts(self):
        """(完成, 跳过, 失败) 三个数，供界面直接拼状态栏。"""
        return len(self.saved), len(self.skipped), len(self.failed)

    def __repr__(self):
        saved, skipped, failed = self.counts()
        return (f"BatchMatteReport(saved={saved}, skipped={skipped}, "
                f"failed={failed}, cancelled={self.cancelled}, "
                f"total={self.total})")


def matte_batch(model, jobs, output_dir, load_rgb, threshold=0.0,
                margin_ratio=0.05, cancel=None, progress=None):
    """按 jobs 串行抠图，返回 BatchMatteReport。

    jobs 是 `[(图片路径, [标注字典, ...]), ...]`：标注只读，本函数不写回任何
    JSON，只往 output_dir 落 PNG。同一张图的同一标签（`batch_key`）只抠一次，
    短边不足 `MIN_TARGET_SHORT_SIDE` 的目标直接跳过。单张失败记进
    `report.failed` 后继续下一个目标；取消时已完成的部分保留，未开始的
    目标不产生任何文件。progress 收到 `(已完成, 总数)`。
    """
    report = BatchMatteReport()
    report.total = sum(len(boxes or ()) for _, boxes in jobs)
    seen = set()
    cached_path, cached_rgb, cached_error = None, None, None
    done = 0

    for image_path, boxes in jobs:
        source_name = os.path.basename(str(image_path)) or "image.png"
        for box in boxes or ():
            if cancel is not None and cancel():
                report.cancelled = True
                return report

            label = (box.get("label") if isinstance(box, dict) else None) or "object"
            try:
                raw_box = target_box(box)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                report.failed.append((image_path, label, f"标注缺少尺寸: {exc}"))
                done += 1
                if progress is not None:
                    progress(done, report.total)
                continue

            if should_skip_box(raw_box):
                report.skipped.append((image_path, label, "小目标"))
            elif batch_key(source_name, label) in seen:
                report.skipped.append((image_path, label, "重复"))
            else:
                seen.add(batch_key(source_name, label))
                if cached_path != image_path:
                    cached_path, cached_rgb, cached_error = image_path, None, None
                    try:
                        cached_rgb = load_rgb(image_path)
                    except Exception as exc:
                        cached_error = str(exc)
                if cached_rgb is None:
                    report.failed.append(
                        (image_path, label, cached_error or "图片读取失败"))
                else:
                    try:
                        result = matte_selection(
                            model, cached_rgb, raw_box, label, source_name,
                            output_dir, threshold=threshold,
                            margin_ratio=margin_ratio,
                            polygon=target_polygon(box), cancel=cancel)
                    except Exception as exc:
                        report.failed.append((image_path, label, str(exc)))
                    else:
                        if result is not None:
                            report.saved.append(result)
                        elif cancel is not None and cancel():
                            # 推理途中被取消：不算「没有前景」，直接收工
                            report.cancelled = True
                            return report
                        else:
                            report.skipped.append((image_path, label, "无前景"))

            done += 1
            if progress is not None:
                progress(done, report.total)

    return report

