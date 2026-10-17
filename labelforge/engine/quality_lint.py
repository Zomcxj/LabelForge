"""标注质检 lint：纯逻辑，无 Qt 控件依赖。

检查项：越界框 / 小目标不可检出 / 小目标需放大 / 重复框（同类高 IoU）/
异类重叠（高 IoU）/ 近似名 / group 不一致。缺 sidecar / 空标注由背景图列表状态
图标负责，质检不再上报。
"""
import bisect
import json
import math
import os
import concurrent.futures

from PySide6.QtCore import QThread, Signal

from ..core.config import QUALITY_LINT_CONFIG
from ..core.utils import calculate_iou


KIND_LABELS = {
    'out_of_bounds': '越界框',
    'tiny_unreachable': '小目标不可检出',
    'tiny_needs_p2': '小目标需放大',
    'duplicate': '重复框',
    'cross_label_overlap': '异类重叠',
    'similar_name': '近似名',
    'group_mismatch': 'group 不一致',
    'skipped_pairwise': '跳过两两比对',
}

#: 小目标判据的标定基准：所有图片与框先等比缩放（长边→该值、短边补边）到
#: 这个边长，再按 4px / 8px / 16px 判档。阈值本身就是按它定的，改这里等于
#: 重新标定三档，config 里的三个比例也一并失效。
TINY_INPUT_SIZE = 640

#: 短边比例的直方图分桶边界。刻意压在判档阈值上而不是等宽分桶：
#: 数据集里大框占多数时，等宽桶会把 0~2.5% 的整个决策区糊进第一桶，
#: 看不出"该不该上滑窗"。前两个桶是 🔴 区、第三个是 🟡 区、其余为 ✅。
TINY_RATIO_EDGES = (0.0, 0.003125, 0.00625, 0.0125, 0.025,
                    0.05, 0.1, 0.2, 0.5, 1.0)

#: 标注员人工确认「保留」的标记，写在 LabelMe shape 的 flags 里
TINY_KEEP_FLAG = 'tiny_keep'

#: 可被「标记保留」豁免的两类小目标问题
TINY_KINDS = ('tiny_unreachable', 'tiny_needs_p2')


def _bbox(box):
    """(x1, y1, x2, y2)；优先用 x/y/width/height，缺失时回退 points。

    坏数据（非数值坐标 / 点数不足）返回 None，由调用方跳过该 shape，
    不能让它中断整库扫描。
    """
    x = box.get("x")
    y = box.get("y")
    w = box.get("width")
    h = box.get("height")
    if None in (x, y, w, h):
        points = box.get("points") or []
        if not points:
            return None
        try:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        return (min(xs), min(ys), max(xs), max(ys))
    try:
        return (float(x), float(y), float(x) + float(w), float(y) + float(h))
    except (TypeError, ValueError):
        return None


def normalized_min_side(box, image_width, image_height):
    """框短边比例：min(w, h) / max(W, H)；不可计算时返回 None。

    口径是**把原图与框等比缩放（长边→640、短边不足处补边）到模型输入后，
    框短边占 640 的比例**，乘 `TINY_INPUT_SIZE` 就是它在模型输入里的像素数。
    阈值 4px / 8px / 16px @640 就是按这个前提定的，所以判档与训练分辨率无关。

    分母取**长边**而不是各轴各比各的：预处理是等比的，短边会被同比例缩小，
    各轴比自己的边会把这个缩水抹掉（1920x1080 上 100x100 按轴算是 5.2% 很
    "健康"，等比缩放到 640 后其实只剩 33px）。

    只看矩形检测框：关键点宽高恒为 0（比例会算成 0，等于把每个关键点判死），
    多边形的 bbox 也不代表目标本身，两者都跳过。
    """
    if not isinstance(box, dict):
        return None
    if (box.get('shape_type') or 'rectangle') != 'rectangle':
        return None
    if not image_width or not image_height:
        return None
    rect = _bbox(box)
    if rect is None:
        return None
    # NaN/Inf 必须显式挡掉：json.loads 默认接受 NaN 字面量，而 `nan <= 0`
    # 为 False 会绕过下面的退化守卫，一路算进报告，最后被当成"最大的健康
    # 目标"落进直方图末桶。dataset_health 用 _finite_float 防的是同一件事。
    try:
        w = float(rect[2] - rect[0])
        h = float(rect[3] - rect[1])
        iw = float(image_width)
        ih = float(image_height)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (w, h, iw, ih)):
        return None
    if w <= 0 or h <= 0 or iw <= 0 or ih <= 0:
        return None
    return min(w, h) / max(iw, ih)


def is_tiny_exempt(box):
    """标注员已人工确认「保留」：电线、栏杆这类有意为之的细长目标。"""
    if not isinstance(box, dict):
        return False
    flags = box.get('flags')
    return bool(isinstance(flags, dict) and flags.get(TINY_KEEP_FLAG))


def tiny_ratio_kind(ratio):
    """比例 → 问题 kind；通过返回 None。下界取闭区间：正好踩线算较好的那档。"""
    if ratio is None:
        return None
    if ratio < QUALITY_LINT_CONFIG['tiny_fail_ratio']:
        return 'tiny_unreachable'
    if ratio < QUALITY_LINT_CONFIG['tiny_warn_ratio']:
        return 'tiny_needs_p2'
    return None


def _ratio_bin(ratio):
    """ratio 落在哪个桶：恰好等于边界时归上界那个桶，与 < 判档口径一致。"""
    idx = bisect.bisect_right(TINY_RATIO_EDGES, ratio) - 1
    if idx < 0:
        return 0
    return min(idx, len(TINY_RATIO_EDGES) - 2)


def _ratio_histogram(ratios, edges=TINY_RATIO_EDGES):
    """按 edges 分桶计数；边界值归上界桶，避免直方图与判档自相矛盾。"""
    counts = [0] * (len(edges) - 1)
    for r in ratios:
        counts[_ratio_bin(r)] += 1
    return {'edges': list(edges), 'counts': counts}


def collect_tiny_ratios(shapes, image_width, image_height):
    """(ratios, exempt_count)：报告口径，与 lint_shapes 判档共用同一套规则。"""
    ratios, exempt = [], 0
    for box in (shapes or []):
        if not isinstance(box, dict):
            continue
        if (box.get('shape_type') or 'rectangle') != 'rectangle':
            continue
        if is_tiny_exempt(box):
            exempt += 1
            continue
        r = normalized_min_side(box, image_width, image_height)
        if r is not None:
            ratios.append(r)
    return ratios, exempt


def tiny_report(ratios, exempt_count=0):
    """数据集级小目标报告：三档计数 + 稳定线 + 分布直方图。

    ratios 只含未被豁免的矩形框；三档占比的分母是 measured，exempt 单列，
    这样「保留」不会虚增通过数。total = measured + exempt = 矩形框总数。
    """
    cfg = QUALITY_LINT_CONFIG
    fail_r = cfg['tiny_fail_ratio']
    warn_r = cfg['tiny_warn_ratio']
    stable_r = cfg['tiny_stable_ratio']
    # 公开函数自己设防：非有限值一律丢弃，不能只依赖调用方给干净数据
    ratios = [r for r in (ratios or [])
              if isinstance(r, (int, float)) and math.isfinite(r)]
    fail = sum(1 for r in ratios if r < fail_r)
    warn = sum(1 for r in ratios if fail_r <= r < warn_r)
    exempt = int(exempt_count or 0)
    return {
        'total': len(ratios) + exempt,
        'measured': len(ratios),
        'fail': fail,
        'warn': warn,
        'pass': len(ratios) - fail - warn,
        'stable': sum(1 for r in ratios if r >= stable_r),
        'exempt': exempt,
        'histogram': _ratio_histogram(ratios),
    }


def lint_shapes(shapes, image_width, image_height, label_index=None):
    """返回问题列表：每项 {kind, label, box_index, detail}。"""
    issues = []
    shapes = [s for s in (shapes or []) if isinstance(s, dict)]
    tol = QUALITY_LINT_CONFIG['bounds_tolerance']

    if image_width and image_height:
        for i, box in enumerate(shapes):
            rect = _bbox(box)
            if rect is None:
                continue
            x1, y1, x2, y2 = rect
            if (x1 < -tol or y1 < -tol or
                    x2 > image_width + tol or y2 > image_height + tol):
                issues.append({
                    'kind': 'out_of_bounds', 'label': box.get('label', ''),
                    'box_index': i,
                    'detail': f"({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                              f"vs {image_width}x{image_height}",
                })

    for i, box in enumerate(shapes):
        if (box.get('shape_type') or 'rectangle') == 'point':
            continue
        rect = _bbox(box)
        if rect is None:
            continue
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        if is_tiny_exempt(box):
            # 人工确认保留：尺寸问题不再报（1000x5 的电线在等比缩放到 640 后
            # 厚度只剩 1.6px，会被判 🔴，但标注员说了那是有意为之）。
            # 越界/重复等在别的循环里判，不受这里影响。
            continue
        ratio = normalized_min_side(box, image_width, image_height)
        kind = tiny_ratio_kind(ratio)
        if kind is not None:
            # min_side/both_below 给第 4 列提示用：标注员需要知道该拉宽、拉高
            # 还是整个框都得放大。等比缩放下宽高共用同一个系数，所以哪条边是
            # 短边就看原始像素，不能再拿各轴的占比去比（那是旧口径的残留，
            # 非方图上会把短边判反）。both_below 用命中档自己的线（fail 档对
            # fail 线、warn 档对 warn 线），与 ratio 的判档口径一致。
            threshold = (QUALITY_LINT_CONFIG['tiny_fail_ratio']
                         if kind == 'tiny_unreachable'
                         else QUALITY_LINT_CONFIG['tiny_warn_ratio'])
            scale = TINY_INPUT_SIZE / max(image_width, image_height)
            issues.append({
                'kind': kind, 'label': box.get('label', ''),
                'box_index': i, 'ratio': ratio,
                'min_side': 'width' if w <= h else 'height',
                'both_below': (w * scale < threshold * TINY_INPUT_SIZE
                               and h * scale < threshold * TINY_INPUT_SIZE),
                'detail': f"{ratio * 100:.2f}% W:{w:.0f} H:{h:.0f}",
            })
            continue

    if len(shapes) <= QUALITY_LINT_CONFIG['max_boxes_pairwise']:
        issues.extend(_pairwise_issues(shapes))
    issues.extend(_similar_name_issues(shapes))
    issues.extend(_group_issues(shapes))
    return issues


def _pairwise_issues(shapes):
    issues = []
    rects = [_bbox(b) for b in shapes]
    dup_iou = QUALITY_LINT_CONFIG['dup_iou']
    cross_iou = QUALITY_LINT_CONFIG['cross_iou']
    for i in range(len(shapes)):
        if rects[i] is None:
            continue
        for j in range(i + 1, len(shapes)):
            if rects[j] is None:
                continue
            iou = calculate_iou(rects[i], rects[j])
            if iou < min(dup_iou, cross_iou):
                continue
            label_i = shapes[i].get('label', '')
            label_j = shapes[j].get('label', '')
            if label_i == label_j:
                if iou >= dup_iou:
                    issues.append({
                        'kind': 'duplicate', 'label': label_i,
                        'box_index': j, 'detail': f"IoU:{iou:.2f} #{i}",
                    })
            elif iou >= cross_iou:
                issues.append({
                    'kind': 'cross_label_overlap', 'label': label_i,
                    'box_index': i,
                    'label_a': label_i, 'box_index_a': i,
                    'label_b': label_j, 'box_index_b': j,
                    'detail': f"IoU:{iou:.2f} #{i}({label_i}) vs ({label_j})",
                })
    return issues


def _edit_distance_at_most_one(a, b):
    """a、b 编辑距离是否 <= 1（等价于相差一个增删改）。"""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for ca, cb in zip(a, b) if ca != cb) == 1
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            if skipped:
                return False
            skipped = True
            j += 1
            continue
        i += 1
        j += 1
    return True


def _normalized(name):
    return " ".join(str(name or "").split()).lower()


def _similar_name_issues(shapes):
    """标签名疑似：忽略大小写/多余空白后相同，或编辑距离 <= 阈值。"""
    issues = []
    threshold = QUALITY_LINT_CONFIG['name_edit_distance']
    labels = []
    for box in shapes:
        label = str(box.get('label', '') or '')
        if label and label not in labels:
            labels.append(label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            norm_a, norm_b = _normalized(a), _normalized(b)
            if norm_a == norm_b or (
                    threshold >= 1 and _edit_distance_at_most_one(norm_a, norm_b)):
                issues.append({
                    'kind': 'similar_name', 'label': a, 'box_index': None,
                    'detail': f"{a} / {b}",
                })
    return issues


def _group_issues(shapes):
    """关键点 group 校验（复用 point_warning 语义）+ 有组框无关键点。"""
    from .shape_io import point_warning
    issues = []
    non_points = [
        (i, b) for i, b in enumerate(shapes)
        if (b.get('shape_type') or 'rectangle') != 'point'
    ]
    for i, box in enumerate(shapes):
        if (box.get('shape_type') or 'rectangle') == 'point':
            warning = point_warning(box, shapes)
            if warning:
                issues.append({
                    'kind': 'group_mismatch', 'label': box.get('label', ''),
                    'box_index': i, 'detail': warning,
                })
    # 只收集标量 group_id：坏数据里的 list/dict 不可哈希，会让集合构造 TypeError
    group_ids = {
        b.get('group_id') for _, b in non_points
        if isinstance(b.get('group_id'), (int, float, str))
    }
    for gid in group_ids:
        has_point = any(
            (b.get('shape_type') or 'rectangle') == 'point'
            and b.get('group_id') == gid
            for b in shapes
        )
        if not has_point:
            idx = next(i for i, b in non_points if b.get('group_id') == gid)
            issues.append({
                'kind': 'group_mismatch',
                'label': shapes[idx].get('label', ''),
                'box_index': idx, 'detail': f"组 {gid} 无关键点",
            })
    return issues


def filter_ignored_issues(issues, ignored_rules):
    """按忽略规则过滤问题：{kind: [detail 或 '*']}。

    detail 匹配为精确相等；'*' 表示整类忽略。
    """
    if not ignored_rules or not isinstance(ignored_rules, dict):
        return list(issues or [])
    filtered = []
    for issue in issues or []:
        keys = ignored_rules.get(issue.get('kind'))
        if not keys:
            filtered.append(issue)
            continue
        if '*' in keys or str(issue.get('detail', '')) in keys:
            continue
        filtered.append(issue)
    return filtered


def summarize_issues(issues):
    """统计各类型问题数量。"""
    summary = {}
    for issue in issues or []:
        kind = issue.get('kind')
        if kind:
            summary[kind] = summary.get(kind, 0) + 1
    return summary


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_json_shapes(json_path):
    """返回 (shapes, image_w, image_h)；文件缺失/损坏返回 None。"""
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return [], 0, 0
    if not isinstance(data, dict):
        return [], 0, 0
    shapes = data.get('shapes')
    if not isinstance(shapes, list):
        shapes = []
    return shapes, _safe_int(data.get('imageWidth')), _safe_int(data.get('imageHeight'))


def lint_image_shapes(image_path, shapes, image_width, image_height):
    """单图问题列表（含跳过两两比对提示），不含 image_index。

    shapes 为 None 表示 sidecar 缺失；[] 表示空标注。两种情况都不再
    上报（背景图列表状态图标已覆盖），直接返回空列表。
    """
    if not shapes:
        return []
    issues = []
    if len(shapes) > QUALITY_LINT_CONFIG['max_boxes_pairwise']:
        issues.append({
            'kind': 'skipped_pairwise', 'label': '', 'box_index': None,
            'detail': f"{len(shapes)} 框",
        })
    issues.extend(lint_shapes(shapes, image_width, image_height))
    return issues


def image_shapes_for_lint(index, path, memory_boxes=None):
    """(shapes_or_None, image_w, image_h)：已加载图以内存为准（含空列表）。

    内存中已加载的图（即使框为空）不再回读磁盘 shapes，避免磁盘滞后
    导致已修好的问题仍被报出；sidecar 是否存在仍以磁盘为准。
    """
    json_path = f"{os.path.splitext(path)[0]}.json"
    loaded = index in (memory_boxes or {})
    info = _read_json_shapes(json_path)
    if loaded:
        boxes = list((memory_boxes or {}).get(index) or [])
        if info is None and not boxes:
            return None, 0, 0
        return boxes, (info[1] if info else 0), (info[2] if info else 0)
    if info is None:
        return None, 0, 0
    return info[0], info[1], info[2]


def lint_single_image(index, path, memory_boxes=None):
    """单图实时质检：返回带 image_index/image_path 的问题列表。"""
    shapes, img_w, img_h = image_shapes_for_lint(index, path, memory_boxes)
    issues = lint_image_shapes(path, shapes, img_w, img_h)
    for issue in issues:
        issue['image_index'] = index
        issue['image_path'] = path
    return issues


def lint_dataset(image_paths, memory_boxes=None, progress_cb=None,
                 is_interrupted=None):
    """扫描数据集，返回 {'issues': [...], 'summary': {...}, 'tiny_report': {...}}。

    memory_boxes: {index: [box,...]}，已加载图以内存为准。
    """
    image_paths = list(image_paths or [])
    memory_boxes = memory_boxes or {}
    issues = []
    summary = {
        'scanned_images': 0,
        'out_of_bounds': 0, 'duplicate': 0,
        'tiny_unreachable': 0, 'tiny_needs_p2': 0,
        'cross_label_overlap': 0, 'similar_name': 0, 'group_mismatch': 0,
        'skipped_pairwise': 0,
    }
    tiny_ratios = []
    tiny_exempt = 0
    total = len(image_paths)
    done = 0

    def _one(index, path):
        shapes, img_w, img_h = image_shapes_for_lint(index, path, memory_boxes)
        # 报告要的比例在 worker 线程里顺手算掉：shapes 已在手，零额外 IO
        ratios, exempt = collect_tiny_ratios(shapes, img_w, img_h)
        return index, path, shapes, img_w, img_h, ratios, exempt

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    interrupted = False
    try:
        futures = [pool.submit(_one, i, p) for i, p in enumerate(image_paths)]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                interrupted = True
                break
            (index, path, shapes, img_w, img_h,
             ratios, exempt) = future.result()
            done += 1
            tiny_ratios.extend(ratios)
            tiny_exempt += exempt
            for issue in lint_image_shapes(path, shapes, img_w, img_h):
                issue['image_index'] = index
                issue['image_path'] = path
                issues.append(issue)
                summary[issue['kind']] = summary.get(issue['kind'], 0) + 1
            if shapes is not None:
                summary['scanned_images'] += 1
            if progress_cb is not None and (done % 50 == 0 or done == total):
                try:
                    progress_cb(done, total)
                except Exception:
                    pass
    finally:
        # 中断时放弃排队中的读盘任务，避免 wait() 被未开始的 future 拖住
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return {'issues': issues, 'summary': summary,
            'tiny_report': tiny_report(tiny_ratios, tiny_exempt)}


class QualityLintWorker(QThread):
    """后台质检扫描：进度 + 结果信号。"""
    lint_progress = Signal(int, int)
    lint_finished = Signal(object)

    def __init__(self, image_paths, memory_boxes=None, parent=None):
        super().__init__(parent)
        self._image_paths = tuple(image_paths)
        self._memory_boxes = dict(memory_boxes or {})

    def run(self):
        try:
            def _progress(done, total):
                if not self.isInterruptionRequested():
                    self.lint_progress.emit(done, total)

            result = lint_dataset(
                self._image_paths, memory_boxes=self._memory_boxes,
                progress_cb=_progress,
                is_interrupted=self.isInterruptionRequested,
            )
        except Exception:
            # 带 error 标记：UI 要显示"扫描失败"，不能伪装成"未发现问题"
            result = {'issues': [], 'summary': {}, 'error': True}
        if not self.isInterruptionRequested():
            self.lint_finished.emit(result)


def box_signature(box):
    """内容签名（类别 + 量化 bbox），用于跨线程同步内存框。"""
    if not isinstance(box, dict):
        return None
    rect = _bbox(box)
    if rect is None:
        return None
    try:
        return (str(box.get('label', '') or ''),) + tuple(
            round(float(v), 3) for v in rect)
    except (TypeError, ValueError):
        return None


def _write_json(json_path, data):
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        return False
    return True


def _remove_shapes_at_indices(json_path, indices):
    """按 shapes 索引删除并写回，保留其它字段。

    返回 (removed, ok)：ok=False 表示读取/写入失败。
    """
    index_set = set(indices or [])
    if not index_set:
        return 0, True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return 0, False
    if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
        return 0, False
    shapes = data['shapes']
    kept = [s for i, s in enumerate(shapes) if i not in index_set]
    removed = len(shapes) - len(kept)
    if removed <= 0:
        return 0, False
    data['shapes'] = kept
    if not _write_json(json_path, data):
        return 0, False
    return removed, True


def _remove_shapes_matching(json_path, boxes):
    """内存框与磁盘 JSON 对齐删除：按内容签名匹配，避免索引错位。

    返回 (removed, ok)：ok=False 表示磁盘未按预期删除（读取/写入失败，
    或签名失配导致磁盘与内存不一致），由调用方标记该图失败，避免仅删
    内存、磁盘残留导致重载后问题复现。
    文件不存在（内存独有）视为成功，由内存同步兜底。
    """
    pending = {}
    for box in boxes or []:
        sig = box_signature(box)
        if sig is not None:
            pending[sig] = pending.get(sig, 0) + 1
    if not pending or not os.path.exists(json_path):
        return 0, True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return 0, False
    if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
        return 0, False
    kept = []
    removed = 0
    for shape in data['shapes']:
        sig = box_signature(shape)
        if sig is not None and pending.get(sig, 0) > 0:
            pending[sig] -= 1
            removed += 1
            continue
        kept.append(shape)
    if removed <= 0:
        return 0, False
    data['shapes'] = kept
    if not _write_json(json_path, data):
        return 0, False
    return removed, True


def _set_keep_flag(shape):
    """就地给 shape 打上 tiny_keep，保留原有 flags；返回是否新打上的。"""
    flags = shape.get('flags')
    if not isinstance(flags, dict):
        flags = {}
        shape['flags'] = flags
    if flags.get(TINY_KEEP_FLAG):
        return False
    flags[TINY_KEEP_FLAG] = True
    return True


def _mark_shapes_at_indices(json_path, indices):
    """未加载图：按 shapes 索引写 flags.tiny_keep 并写回，保留其它字段。

    返回 (marked, ok)。ok=False 表示读取/写入失败或索引全部越界
    （说明磁盘与扫描结果已不一致，不能静默当成成功）。
    已标记过的框算命中但不算新增，重复标记是幂等的。
    """
    index_set = set(indices or [])
    if not index_set:
        return 0, True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return 0, False
    if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
        return 0, False
    shapes = data['shapes']
    matched = changed = 0
    for i in sorted(index_set):
        if not (0 <= i < len(shapes)) or not isinstance(shapes[i], dict):
            continue
        matched += 1
        if _set_keep_flag(shapes[i]):
            changed += 1
    if matched <= 0:
        return 0, False
    if changed and not _write_json(json_path, data):
        return 0, False
    return matched, True


def _mark_shapes_matching(json_path, boxes):
    """已加载图：按内容签名匹配磁盘 shape 后写标记，避免顺序不一致标错框。

    内存里 box_index 指向内存列表，而磁盘 shapes 顺序可能不同（磁盘滞后），
    直接按下标写会把豁免标到隔壁框上。返回 (matched, ok)。
    文件不存在（内存独有）视为成功，由内存同步兜底。
    """
    pending = {}
    for box in boxes or []:
        sig = box_signature(box)
        if sig is not None:
            pending[sig] = pending.get(sig, 0) + 1
    if not pending:
        return 0, True
    if not os.path.exists(json_path):
        return 0, True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return 0, False
    if not isinstance(data, dict) or not isinstance(data.get('shapes'), list):
        return 0, False
    matched = changed = 0
    for shape in data['shapes']:
        if not isinstance(shape, dict):
            continue
        sig = box_signature(shape)
        if sig is None or pending.get(sig, 0) <= 0:
            continue
        pending[sig] -= 1
        matched += 1
        if _set_keep_flag(shape):
            changed += 1
    if matched <= 0:
        return 0, False
    if changed and not _write_json(json_path, data):
        return 0, False
    return matched, True


def mark_tiny_keep(selected_issues, image_paths, memory_boxes=None):
    """把选中的小目标问题标记为「保留」：写 flags.tiny_keep 到 sidecar 与内存。

    返回 {'marked', 'images_changed', 'failed'}，failed 是路径列表。
    只处理 TINY_KINDS 两类 —— 它们的豁免语义是"这个框就是有意这么小"，
    与按 detail 字符串忽略不是一回事（框改 1px 后者就静默失配）。

    内存与磁盘必须一起改：已加载图重扫时以内存为准，只写盘豁免不生效；
    反之只写内存则重载后问题复现。磁盘写失败时不动内存，避免两边不一致。
    """
    memory_boxes = memory_boxes or {}
    image_paths = list(image_paths or [])
    result = {'marked': 0, 'images_changed': 0, 'failed': []}
    per_image = {}
    for issue in selected_issues or []:
        if not isinstance(issue, dict) or issue.get('kind') not in TINY_KINDS:
            continue
        index = issue.get('image_index')
        box_index = issue.get('box_index')
        if not isinstance(index, int) or not (0 <= index < len(image_paths)):
            continue
        if not isinstance(box_index, int) or box_index < 0:
            continue
        per_image.setdefault(index, []).append(box_index)
    for index in sorted(per_image):
        path = image_paths[index]
        json_path = f"{os.path.splitext(path)[0]}.json"
        box_indices = sorted(set(per_image[index]))
        targets = []
        if index in memory_boxes:
            memory = list(memory_boxes.get(index) or [])
            targets = [memory[bi] for bi in box_indices
                       if 0 <= bi < len(memory) and isinstance(memory[bi], dict)]
            marked, ok = _mark_shapes_matching(json_path, targets)
        else:
            marked, ok = _mark_shapes_at_indices(json_path, box_indices)
        if not ok:
            result['failed'].append(path)
            continue
        for box in targets:
            _set_keep_flag(box)
        result['marked'] += marked or len(targets)
        result['images_changed'] += 1
    return result


def _selected_removal_indices(shapes, selected, target_label):
    """按选中项匹配目标类框的 shapes 索引。

    每个 issue 携带两侧结构（label_a/box_index_a、label_b/box_index_b），
    取与 target_label 一致的一侧；索引与类别必须同时匹配，防止索引漂移误删。
    返回 (indices, boxes)。
    """
    boxes = [s for s in (shapes or []) if isinstance(s, dict)]
    target = str(target_label or '')
    indices = []
    if not target:
        return [], []
    for issue in selected or []:
        for index, label in (
                (issue.get('box_index_a'), issue.get('label_a')),
                (issue.get('box_index_b'), issue.get('label_b'))):
            if index is None or not (0 <= index < len(boxes)) or index in indices:
                continue
            if str(label or '') != target:
                continue
            if str(boxes[index].get('label', '') or '') != target:
                continue
            indices.append(index)
    indices.sort()
    return indices, [boxes[i] for i in indices]


def remove_selected_cross_overlaps(selected_issues, target_label, image_paths,
                                   memory_boxes=None, progress_cb=None,
                                   is_interrupted=None):
    """删除选中问题里 target_label 类的重叠框（外科式改 JSON）。

    未加载图按 shapes 索引删；已加载图按内容签名匹配删（避免索引错位）。
    返回 {'removed', 'images_changed', 'removed_boxes', 'refreshed_issues',
    'failed', 'interrupted'}。
    """
    image_paths = list(image_paths or [])
    memory_boxes = memory_boxes or {}
    result = {'removed': 0, 'images_changed': 0, 'removed_boxes': {},
              'refreshed_issues': {}, 'failed': [], 'interrupted': False}
    target = str(target_label or '')
    if not target or not image_paths:
        return result

    per_image = {}
    for issue in selected_issues or []:
        if issue.get('kind') != 'cross_label_overlap':
            continue
        index = issue.get('image_index')
        if index is None or not (0 <= index < len(image_paths)):
            continue
        per_image.setdefault(index, []).append(issue)
    if not per_image:
        return result

    total = len(per_image)
    done = 0
    for index in sorted(per_image):
        if is_interrupted and is_interrupted():
            result['interrupted'] = True
            break
        path = image_paths[index]
        json_path = f"{os.path.splitext(path)[0]}.json"
        loaded = index in memory_boxes
        if loaded:
            shapes = list(memory_boxes.get(index) or [])
            img_w = img_h = 0
            info = _read_json_shapes(json_path)
            if info:
                img_w, img_h = info[1], info[2]
        else:
            info = _read_json_shapes(json_path)
            if info is None:
                result['failed'].append(path)
                done += 1
                continue
            shapes, img_w, img_h = info[0], info[1], info[2]
        indices, boxes = _selected_removal_indices(
            shapes, per_image[index], target)
        if not indices:
            done += 1
            continue
        if loaded:
            _, ok = _remove_shapes_matching(json_path, boxes)
            if not ok:
                result['failed'].append(path)
                done += 1
                continue
        else:
            disk_removed, ok = _remove_shapes_at_indices(json_path, indices)
            if not ok or disk_removed <= 0:
                result['failed'].append(path)
                done += 1
                continue
        removed = [b for b in boxes if isinstance(b, dict)]
        result['removed_boxes'][index] = removed
        result['removed'] += len(removed)
        result['images_changed'] += 1
        index_set = set(indices)
        kept = [b for i, b in enumerate(shapes) if i not in index_set]
        refreshed = lint_image_shapes(path, kept, img_w, img_h)
        for issue in refreshed:
            issue['image_index'] = index
            issue['image_path'] = path
        result['refreshed_issues'][index] = refreshed
        done += 1
        if progress_cb is not None:
            try:
                progress_cb(done, total)
            except Exception:
                pass
    return result


class CrossLabelDeleteWorker(QThread):
    """后台删除选中的异类重叠框：进度 + 结果信号（中断也回报部分结果）。"""
    delete_progress = Signal(int, int)
    delete_finished = Signal(object)

    def __init__(self, selected_issues, target_label, image_paths,
                 memory_boxes=None, parent=None):
        super().__init__(parent)
        self._selected_issues = tuple(selected_issues or ())
        self._target_label = str(target_label or '')
        self._image_paths = tuple(image_paths)
        self._memory_boxes = dict(memory_boxes or {})

    def run(self):
        try:
            def _progress(done, total):
                if not self.isInterruptionRequested():
                    self.delete_progress.emit(done, total)

            result = remove_selected_cross_overlaps(
                self._selected_issues, self._target_label, self._image_paths,
                memory_boxes=self._memory_boxes,
                progress_cb=_progress,
                is_interrupted=self.isInterruptionRequested,
            )
        except Exception as e:
            result = {'removed': 0, 'images_changed': 0, 'removed_boxes': {},
                      'failed': [], 'interrupted': False, 'error': str(e)}
        self.delete_finished.emit(result)
