"""数据集侧车 JSON 扫描：标签/计数/状态，以及后台扫描 worker。"""
import os
import json
import concurrent.futures

from PySide6.QtCore import QThread, Signal

from .status import STATUS_UNANNOTATED, STATUS_ANNOTATED, STATUS_EMPTY
from ...core.utils import shape_task_type

__all__ = [
    "scan_dataset_labels", "scan_dataset_labels_with_counts", "scan_dataset_full",
    "collect_background_label_counts", "collect_background_label_tasks",
    "count_labels_in_json", "DatasetLabelScanWorker",
]


def _scan_single_json(json_path):
    """Parse a single LabelMe JSON and return its labels, or None on error."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        return None
    labels = set()
    for shape in shapes:
        label = shape.get("label") if isinstance(shape, dict) else None
        if isinstance(label, str) and label.strip():
            labels.add(label)
    return labels


def scan_dataset_labels(image_paths, is_interrupted=None):
    """Return valid LabelMe labels from the JSON files beside image paths."""
    labels = set()
    pending = []
    for image_path in image_paths:
        if is_interrupted and is_interrupted():
            break
        pending.append(f"{os.path.splitext(image_path)[0]}.json")
    if not pending:
        return labels

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_scan_single_json, jp) for jp in pending]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                break
            result = future.result()
            if result:
                labels.update(result)
    return labels


def scan_dataset_labels_with_counts(image_paths, is_interrupted=None):
    """Return (labels_set, {label: count}) across all background sidecar JSONs."""
    labels, counts, _ = scan_dataset_full(image_paths, is_interrupted)
    return labels, counts


def scan_dataset_full(image_paths, is_interrupted=None, progress_cb=None,
                      tasks_out=None):
    """Return (labels_set, {label: count}, {image_path: status}) in one pass.

    Parsing each sidecar once yields both the dataset label counts and the
    per-image annotated/empty/unannotated status, so populating a large
    dataset's list never needs a second full JSON scan.

    progress_cb: optional callable(batch_statuses: dict) invoked every
    PROGRESS_BATCH results so callers can stream partial results to the UI
    instead of waiting for the whole dataset (keeps the status circles filling
    in progressively on large datasets).

    tasks_out: optional dict filled with {label: set(task_types)} during the
    same pass, so the stats dialog can show 框类型 without rescanning.
    """
    PROGRESS_BATCH = 200
    counts = {}
    statuses = {}
    pending = []
    for image_path in image_paths:
        if is_interrupted and is_interrupted():
            break
        pending.append(image_path)
    if not pending:
        return set(), counts, statuses

    def _one(image_path):
        json_path = f"{os.path.splitext(image_path)[0]}.json"
        if not os.path.exists(json_path):
            return image_path, STATUS_UNANNOTATED, None, None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return image_path, STATUS_EMPTY, None, None
        shapes = data.get("shapes") if isinstance(data, dict) else None
        if not isinstance(shapes, list):
            return image_path, STATUS_EMPTY, None, None
        local = {}
        local_tasks = {}
        for shape in shapes:
            label = shape.get("label") if isinstance(shape, dict) else None
            if isinstance(label, str) and label.strip():
                label = label.strip()
                local[label] = local.get(label, 0) + 1
                if tasks_out is not None:
                    local_tasks.setdefault(label, set()).add(shape_task_type(shape))
        status = STATUS_ANNOTATED if local else STATUS_EMPTY
        return image_path, status, local, local_tasks

    batch = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one, ip) for ip in pending]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                break
            image_path, status, local, local_tasks = future.result()
            statuses[image_path] = status
            batch[image_path] = status
            if local:
                for lbl, n in local.items():
                    counts[lbl] = counts.get(lbl, 0) + int(n or 0)
            if tasks_out is not None and local_tasks:
                for lbl, ts in local_tasks.items():
                    tasks_out.setdefault(lbl, set()).update(ts)
            if progress_cb is not None and len(batch) >= PROGRESS_BATCH:
                try:
                    progress_cb(dict(batch))
                except Exception:
                    pass
                batch = {}
    if progress_cb is not None and batch:
        try:
            progress_cb(dict(batch))
        except Exception:
            pass
    labels = {lbl for lbl, n in counts.items() if n > 0}
    return labels, counts, statuses


def collect_background_label_stats_and_tasks(image_paths, is_interrupted=None):
    """一次扫描同时得到 {label: count} 与 {label: set(task_types)}。

    统计弹窗原先分别调 `collect_background_label_counts()` 与
    `collect_background_label_tasks()`，**把每个 sidecar json 解析了两遍**，
    而且都在 UI 线程上，数据集一大就要好几秒（用户报的「统计有时候需要 5s
    才能打开」）。这里复用 `scan_dataset_full()` 的单趟解析：它本来就在同一
    次读取里同时产出 counts 与 tasks。

    返回 `(counts, tasks)`；两者都是普通 dict（tasks 的值仍为 set）。
    """
    tasks = {}
    _labels, counts, _statuses = scan_dataset_full(
        image_paths, is_interrupted=is_interrupted, tasks_out=tasks)
    return counts, tasks


class DatasetStatsScanWorker(QThread):
    """在后台为统计弹窗扫一遍数据集（counts + 框类型）。

    与 `DatasetLabelScanWorker` 的区别：那个服务背景列表的标签/状态扫描、
    结果走 `labels_scanned` 缓存；这个只服务统计弹窗的一次性查询，结果直接
    回传，不改动任何缓存状态。
    """
    stats_ready = Signal(object, object)

    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self._image_paths = tuple(image_paths)

    def run(self):
        try:
            tasks = {}
            _labels, counts, _statuses = scan_dataset_full(
                self._image_paths, self.isInterruptionRequested,
                tasks_out=tasks,
            )
        except Exception:
            counts, tasks = {}, {}
        if not self.isInterruptionRequested():
            self.stats_ready.emit(counts, tasks)


def count_labels_in_json(json_path):
    """Parse a LabelMe JSON and return {label: shape_count}, or None on error."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        return None
    counts = {}
    for shape in shapes:
        label = shape.get("label") if isinstance(shape, dict) else None
        if isinstance(label, str) and label.strip():
            label = label.strip()
            counts[label] = counts.get(label, 0) + 1
    return counts

# 私名别名：保留为内部调用与向后兼容（collect_background_label_counts 仍在用）
_count_labels_in_json = count_labels_in_json


def collect_background_label_counts(image_paths, is_interrupted=None):
    """Count every shape occurrence across background sidecar JSONs."""
    counts = {}
    pending = []
    for image_path in image_paths:
        if is_interrupted and is_interrupted():
            break
        pending.append(f"{os.path.splitext(image_path)[0]}.json")
    if not pending:
        return counts
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_count_labels_in_json, jp) for jp in pending]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                break
            result = future.result()
            if result:
                for lbl, n in result.items():
                    counts[lbl] = counts.get(lbl, 0) + int(n or 0)
    return counts


def _scan_label_tasks_in_json(json_path):
    """Return {label: set(shape_task_type)} for one sidecar JSON, or None."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if not isinstance(shapes, list):
        return None
    from ...core.utils import shape_task_type
    tasks = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if not (isinstance(label, str) and label.strip()):
            continue
        tasks.setdefault(label.strip(), set()).add(shape_task_type(shape))
    return tasks


def collect_background_label_tasks(image_paths, is_interrupted=None):
    """Collect the set of task types (det/seg/pose/obb) per label across sidecars."""
    tasks = {}
    pending = []
    for image_path in image_paths:
        if is_interrupted and is_interrupted():
            break
        pending.append(f"{os.path.splitext(image_path)[0]}.json")
    if not pending:
        return tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_scan_label_tasks_in_json, jp) for jp in pending]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted and is_interrupted():
                break
            result = future.result()
            if result:
                for lbl, ts in result.items():
                    tasks.setdefault(lbl, set()).update(ts)
    return tasks


class DatasetLabelScanWorker(QThread):
    """Scan a fixed dataset snapshot outside the UI thread."""
    labels_scanned = Signal(int, tuple, object, object, object, object)
    statuses_progress = Signal(int, tuple, object)

    def __init__(self, generation, image_paths, parent=None):
        super().__init__(parent)
        self._generation = generation
        self._image_paths = tuple(image_paths)

    def run(self):
        def _emit_progress(batch):
            if not self.isInterruptionRequested():
                self.statuses_progress.emit(
                    self._generation, self._image_paths, batch
                )

        try:
            tasks = {}
            labels, counts, statuses = scan_dataset_full(
                self._image_paths, self.isInterruptionRequested,
                progress_cb=_emit_progress, tasks_out=tasks,
            )
            if not self.isInterruptionRequested():
                self.labels_scanned.emit(
                    self._generation, self._image_paths, labels, counts,
                    statuses, tasks,
                )
        except Exception:
            if not self.isInterruptionRequested():
                self.labels_scanned.emit(
                    self._generation, self._image_paths, set(), {}, {}, {}
                )
