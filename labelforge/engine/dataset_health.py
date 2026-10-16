"""数据集健康诊断：纯逻辑，无 Qt 控件依赖。

扫描标注 sidecar 采集逐框几何，计算类别/尺寸/长宽比/IoU 分布并生成建议。
只读：不写盘、不改标注、不持久化结果。
"""
import json
import math
import os
import concurrent.futures

from .quality_lint import _bbox
from ..core.utils import calculate_iou, output_sidecar_paths, PathUtils
from ..core.config import QUALITY_LINT_CONFIG


def _finite_float(value):
    """转 float；非数值或非有限值（NaN/Inf）返回 None，调用方跳过该框。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _geometry_of(label, rect):
    x1, y1, x2, y2 = (_finite_float(v) for v in rect)
    if None in (x1, y1, x2, y2):
        return None
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None
    return {
        'label': str(label or ''),
        'x': float(x1), 'y': float(y1),
        'width': w, 'height': h,
        'area': w * h, 'aspect': w / h,
    }


def _read_json_shapes(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    shapes = data.get('shapes')
    if not isinstance(shapes, list):
        return None
    return shapes


def _shape_is_paste(shape):
    flags = shape.get("flags")
    return bool(isinstance(flags, dict) and flags.get("paste"))


def _shape_dedupe_key(shape):
    """跨两份 sidecar 的去重键：类别 + 几何 + paste 标志。

    save_json 会把当前图的检测框同时写进原目录与输出目录 sidecar，
    不按内容去重会把同一个框计两次。内存框用 is_paste、磁盘 shape 用
    flags.paste，两者必须视为同一标志，否则同几何贴图会计两次。
    """
    try:
        rect = _bbox(shape)
    except Exception:
        return None
    if rect is None:
        return None
    try:
        geom = tuple(round(float(v), 3) for v in rect)
    except (TypeError, ValueError):
        return None
    is_paste = _shape_is_paste(shape) or bool(shape.get('is_paste'))
    return (str(shape.get('label', '') or ''), geom, is_paste)


def _read_output_sidecar(path):
    """输出目录 sidecar（贴图实际落盘处）里的贴图 shapes。

    保存名是 `{prefix}_{stem}.json`（前缀用户可自由输入），按文件名匹配；
    非贴图框是保存时的副本，不作为来源，避免复活已删除/移动的框。
    无匹配文件或全部损坏时返回 None（未加载分支据此判定图片未被标注）。
    """
    output_dir = PathUtils.get_output_dir(path)
    found = None
    for sidecar in output_sidecar_paths(path, output_dir):
        loaded = _read_json_shapes(sidecar)
        if loaded is None:
            continue
        if found is None:
            found = []
        found.extend(s for s in loaded if isinstance(s, dict) and _shape_is_paste(s))
    return found


def _merge_shapes(primary, secondary):
    """原始 sidecar 全量保留；输出目录 sidecar 按主侧内容去重后追加。

    primary 内部不去重：同侧 sidecar 的重复框是真实标注数据，IoU 直方图
    正是要暴露这类重复；去重只用于消除同一框在两份 sidecar 的副本。
    """
    merged = []
    seen = set()
    for shape in primary or []:
        if not isinstance(shape, dict):
            continue
        key = _shape_dedupe_key(shape)
        if key is not None:
            seen.add(key)
        merged.append(shape)
    for shape in secondary or []:
        if not isinstance(shape, dict):
            continue
        key = _shape_dedupe_key(shape)
        if key is not None and key in seen:
            continue
        merged.append(shape)
    return merged


def collect_shape_geometry(image_paths, memory_boxes=None, is_interrupted=None):
    """采集逐框几何。已加载图以内存为准（空内存槽回读磁盘）；贴图单独归类。"""
    image_paths = list(image_paths or [])
    memory_boxes = memory_boxes or {}
    boxes = []
    paste_boxes = []
    images_scanned = 0

    def _one(index, path):
        json_path = f"{os.path.splitext(path)[0]}.json"
        loaded = index in memory_boxes
        memory_shapes = list(memory_boxes.get(index) or []) if loaded else None
        if memory_shapes:
            return index, _merge_shapes(memory_shapes, _read_output_sidecar(path))
        # 空内存槽只代表"未加载"，不代表"已清空"：磁盘有就回读
        original = _read_json_shapes(json_path)
        output = _read_output_sidecar(path)
        if original is None and output is None:
            return index, None
        return index, _merge_shapes(original, output)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    interrupted = False
    try:
        futures = [pool.submit(_one, i, p) for i, p in enumerate(image_paths)]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                interrupted = True
                break
            index, shapes = future.result()
            if shapes is None:
                continue
            images_scanned += 1
            for shape in shapes:
                if not isinstance(shape, dict):
                    continue
                if (shape.get('shape_type') or 'rectangle') == 'point':
                    continue
                try:
                    rect = _bbox(shape)
                    if rect is None:
                        continue
                    geo = _geometry_of(shape.get('label', ''), rect)
                except Exception:
                    continue
                if geo is None:
                    continue
                geo['image_index'] = index
                is_paste = _shape_is_paste(shape) or bool(shape.get('is_paste'))
                (paste_boxes if is_paste else boxes).append(geo)
    finally:
        # 中断时放弃排队中的读盘任务，避免 requestInterruption 后仍被拖住
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return {'boxes': boxes, 'paste_boxes': paste_boxes,
            'images_scanned': images_scanned}


def collect_paste_geometry(canvas_items_dict):
    """从 {index: [(pixmap, rect, label), ...]} 提取贴图几何。"""
    boxes = []
    for index, items in (canvas_items_dict or {}).items():
        for entry in items or []:
            if not entry or len(entry) < 3:
                continue
            _pixmap, rect, label = entry[0], entry[1], entry[2]
            try:
                x = _finite_float(rect.x())
                y = _finite_float(rect.y())
                w = _finite_float(rect.width())
                h = _finite_float(rect.height())
            except Exception:
                continue
            if None in (x, y, w, h) or w <= 0 or h <= 0:
                continue
            boxes.append({
                'label': str(label or ''),
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': w * h, 'aspect': w / h,
                'image_index': index,
            })
    return boxes


def merge_paste_geometry(disk_boxes, memory_boxes):
    """合并磁盘与内存贴图：同一位置以内存为准（内存含未保存的最新状态）。"""
    def _key(box):
        return (box.get('image_index'), round(float(box.get('x', 0)), 1),
                round(float(box.get('y', 0)), 1),
                round(float(box.get('width', 0)), 1),
                round(float(box.get('height', 0)), 1))

    merged = {}
    for box in disk_boxes or []:
        merged[_key(box)] = box
    for box in memory_boxes or []:
        merged[_key(box)] = box
    return list(merged.values())


def _bucket_count(n):
    if n <= 0:
        return 5
    return max(5, min(20, int(round(math.sqrt(n)))))


def _histogram(values, buckets):
    # 防御：NaN/Inf 参与 min/max 会让桶索引转换抛 ValueError，直接剔除
    values = [v for v in (_finite_float(v) for v in (values or []))
              if v is not None]
    if not values:
        return {'edges': [0.0, 1.0], 'counts': [0]}
    lo, hi = min(values), max(values)
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / buckets
    edges = [lo + i * width for i in range(buckets + 1)]
    counts = [0] * buckets
    for v in values:
        idx = int((v - lo) / width)
        if idx >= buckets:
            idx = buckets - 1
        counts[idx] += 1
    return {'edges': edges, 'counts': counts}


def _iou_counts(boxes):
    max_pairwise = QUALITY_LINT_CONFIG['max_boxes_pairwise']
    per_image = {}
    for b in boxes:
        idx = b.get('image_index')
        if idx is None:
            continue
        per_image.setdefault(idx, []).append(b)
    counts = [0] * 10
    for _idx, group in per_image.items():
        if len(group) > max_pairwise:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.get('label') != b.get('label'):
                    continue
                ax1 = _finite_float(a.get('x', 0.0))
                ay1 = _finite_float(a.get('y', 0.0))
                aw = _finite_float(a.get('width'))
                ah = _finite_float(a.get('height'))
                bx1 = _finite_float(b.get('x', 0.0))
                by1 = _finite_float(b.get('y', 0.0))
                bw = _finite_float(b.get('width'))
                bh = _finite_float(b.get('height'))
                if None in (ax1, ay1, aw, ah, bx1, by1, bw, bh):
                    continue
                iou = _finite_float(calculate_iou(
                    (ax1, ay1, ax1 + aw, ay1 + ah),
                    (bx1, by1, bx1 + bw, by1 + bh)))
                if iou is None:
                    continue
                idx_bucket = min(9, int(iou * 10))
                counts[idx_bucket] += 1
    edges = [i / 10 for i in range(11)]
    return {'edges': edges, 'counts': counts}


def compute_health(boxes):
    boxes = [b for b in (boxes or []) if isinstance(b, dict)]

    counts = {}
    for b in boxes:
        label = str(b.get('label', '') or '')
        counts[label] = counts.get(label, 0) + 1
    class_dist = [{'label': k, 'count': v}
                  for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]

    # 防御：单个坏框（NaN/Inf/非数值）不能让整库扫描失败
    areas = [v for v in (_finite_float(b.get('area') or 0) for b in boxes)
             if v is not None]
    aspects = [v for v in (_finite_float(b.get('aspect') or 0) for b in boxes)
               if v is not None]

    return {
        'class_dist': class_dist,
        'size_hist': _histogram(areas, _bucket_count(len(areas))),
        'aspect_hist': _histogram(aspects, _bucket_count(len(aspects))),
        'iou_hist': _iou_counts(boxes),
        'summary': {
            'total_boxes': len(boxes),
            'class_count': len(class_dist),
            'images_scanned': len({b.get('image_index') for b in boxes
                                   if b.get('image_index') is not None}),
        },
    }


def _advice_item(key, **params):
    """建议项：{key, params}，由 UI 端 tr(key).format(**params) 渲染。"""
    return {'key': key, 'params': params}


def health_advice(stats):
    advice = []
    class_dist = stats.get('class_dist') or []
    total = stats.get('summary', {}).get('total_boxes', 0) or 0
    if class_dist and total:
        for entry in class_dist:
            share = entry['count'] / total
            # count>=20 是噪声护栏：<20 个样本占比低属正常长尾，不宜报警
            if entry['count'] >= 20 and share < 0.05:
                advice.append(_advice_item(
                    "类别 {label} 样本偏少（{count} 个，占比 {share}%），建议多合成",
                    label=entry['label'], count=entry['count'],
                    share=f"{share * 100:.0f}"))
        if len(class_dist) >= 2:
            top, bottom = class_dist[0]['count'], class_dist[-1]['count']
            if bottom > 0 and top / bottom >= 10:
                advice.append(_advice_item(
                    "类别分布失衡（{top_label}:{bottom_label} = {top}:{bottom}）",
                    top_label=class_dist[0]['label'],
                    bottom_label=class_dist[-1]['label'],
                    top=top, bottom=bottom))

    aspect = stats.get('aspect_hist') or {}
    counts = aspect.get('counts') or []
    if counts and sum(counts) > 0:
        peak = max(counts) / sum(counts)
        if peak > 0.8:
            edges = aspect.get('edges') or []
            i = counts.index(max(counts))
            rng = f"{edges[i]:.2f}~{edges[i + 1]:.2f}" if len(edges) > i + 1 else "?"
            advice.append(_advice_item(
                "长宽比单一（集中于 {rng}），多样性不足", rng=rng))

    iou = stats.get('iou_hist') or {}
    iou_counts = iou.get('counts') or []
    if iou_counts and sum(iou_counts) > 0:
        high = sum(iou_counts[-2:]) / sum(iou_counts)
        if high >= 0.2:
            advice.append(_advice_item(
                "重复/高度重叠框偏多（{high}%）", high=f"{high * 100:.0f}"))

    return advice or [_advice_item("未发现明显失衡")]


from PyQt5.QtCore import QThread, pyqtSignal


class DatasetHealthWorker(QThread):
    """后台扫描数据集并计算健康统计。"""
    health_ready = pyqtSignal(object)

    def __init__(self, image_paths, memory_boxes=None, canvas_items_dict=None,
                 parent=None):
        super().__init__(parent)
        self._image_paths = tuple(image_paths or ())
        self._memory_boxes = dict(memory_boxes or {})
        self._canvas_items_dict = dict(canvas_items_dict or {})

    def run(self):
        try:
            geo = collect_shape_geometry(
                self._image_paths, memory_boxes=self._memory_boxes,
                is_interrupted=self.isInterruptionRequested)
            memory_paste = collect_paste_geometry(self._canvas_items_dict)
            paste = merge_paste_geometry(geo['paste_boxes'], memory_paste)
            annot_stats = compute_health(geo['boxes'])
            payload = {
                'annot': {'stats': annot_stats,
                          'advice': health_advice(annot_stats)},
                'paste': {},
                'images_scanned': geo['images_scanned'],
            }
            if paste:
                paste_stats = compute_health(paste)
                payload['paste'] = {'stats': paste_stats,
                                    'advice': health_advice(paste_stats)}
            else:
                payload['paste'] = {'stats': {}, 'advice': []}
        except Exception:
            payload = {'error': True, 'annot': {}, 'paste': {},
                       'images_scanned': 0}
        if not self.isInterruptionRequested():
            self.health_ready.emit(payload)
