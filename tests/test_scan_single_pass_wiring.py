"""三处 sidecar 双扫描必须接上现成的单趟扫描（Task 12 回归钉）。

`collect_background_label_stats_and_tasks()`（engine/image_loader/scan.py）
一次解析就能同时得到 {label: count} 与框类型，但 label_manager / stats /
memory_record 三处仍在分别调 `collect_background_label_counts()` +
`collect_background_label_tasks()`，把每个 sidecar json **解析两遍**。
数据集一大，统计打开与记忆快照就要慢一倍。

这里对同一组 sidecar 分别触发三条路径，统计 `.json` 的 open 次数钉住
「每个 json 只解析一次」；顺带钉住 counts 与框类型仍按旧口径产出
（计数口径由 test_stats_scan_async.py 单独钉，这里钉接线后的端到端结果）。
"""
import builtins
import json

import pytest


def _write_dataset(tmp_path, count=6):
    paths = []
    for i in range(count):
        img = tmp_path / f"img{i:02d}.png"
        img.write_bytes(b"x")
        (tmp_path / f"img{i:02d}.json").write_text(json.dumps({
            "shapes": [
                {"label": "cat", "points": [[0, 0], [5, 5]],
                 "shape_type": "rectangle"},
                {"label": "dog", "points": [[0, 0], [3, 3], [6, 3]],
                 "shape_type": "polygon"},
            ],
        }), encoding="utf-8")
        paths.append(str(img))
    return paths


class _JsonOpenCounter:
    """统计对数据集目录内 `.json` 的 open 次数（解析一次 = open 一次）。

    扫描在 8 线程池里并发打开，list.append 在 GIL 下是原子的，直接记即可。
    只统计 tmp_path 下的 json：全局配置（labelforge_test_* 目录）也是 json，
    但与被测路径无关，混进来会误报。
    """

    def __init__(self, root):
        self.root = str(root)
        self.json_opens = []
        self._real_open = builtins.open

    def __call__(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("file")
        text = str(target)
        if text.endswith(".json") and text.startswith(self.root):
            self.json_opens.append(text)
        return self._real_open(*args, **kwargs)


@pytest.fixture
def count_json_opens(tmp_path, monkeypatch):
    counter = _JsonOpenCounter(tmp_path)
    monkeypatch.setattr(builtins, "open", counter)
    return counter


def _assert_each_json_parsed_once(counter, paths):
    assert len(counter.json_opens) == len(paths), (
        f"{len(paths)} 份 sidecar 应当只解析 {len(paths)} 次，"
        f"实际 open 了 {len(counter.json_opens)} 次（每份被解析了两遍）"
    )


# ---------- 路径 1：LabelManager._seed_stats_cache_from_disk_and_memory ----------

def test_label_manager_seed_parses_each_json_once(count_json_opens, tmp_path):
    from labelforge.engine.label_manager import LabelManager

    paths = _write_dataset(tmp_path)
    editor = type("Editor", (), {})()
    editor.background_images = list(paths)
    # 空的内存框字典：只走磁盘扫描，不触发逐文件兜底（那是另一条已钉住的逻辑）
    editor.detection_boxes_dict = {}
    editor.label_color_map = {}
    editor._memory_background_path = ""
    editor.get_label_color = lambda label: "#000000"
    lm = LabelManager(editor)

    lm._seed_stats_cache_from_disk_and_memory()

    _assert_each_json_parsed_once(count_json_opens, paths)
    rows = {row["label"]: row for row in editor._cached_bg_label_stats}
    assert rows["cat"]["count"] == len(paths)
    assert rows["dog"]["count"] == len(paths)
    assert rows["cat"]["tasks"] == ["det"]
    assert rows["dog"]["tasks"] == ["seg"]


# ---------- 路径 2：StatsMixin._collect_bg_stats_for_dialog ----------

def _stats_host(paths):
    host = type("StatsHost", (), {})()
    host.background_images = list(paths)
    host.global_labels = set()
    host.background_dataset_labels = set()
    host.label_color_map = {}
    host._cached_bg_label_stats = []
    host._cached_bg_label_stats_path = ""
    host._memory_background_path = ""
    host._background_label_scan_completed = False
    host._dataset_stats_dirty = False
    host.get_label_color = lambda label: "#123456"
    return host


def test_stats_dialog_collector_parses_each_json_once(count_json_opens, tmp_path):
    from labelforge.ui.mixins.stats import StatsMixin

    paths = _write_dataset(tmp_path)
    host = _stats_host(paths)

    bg_stats = StatsMixin._collect_bg_stats_for_dialog(host)

    _assert_each_json_parsed_once(count_json_opens, paths)
    assert bg_stats == {"cat": len(paths), "dog": len(paths)}
    rows = {row["label"]: row for row in host._cached_bg_label_stats}
    assert rows["cat"]["tasks"] == ["det"]
    assert rows["dog"]["tasks"] == ["seg"]


# ---------- 路径 3：MemoryRecordMixin._build_bg_label_stats_snapshot ----------

class _MemoryHost:
    """最小化 host（与 test_dataset_progressive_scan._SnapshotHost 同套路）。"""

    background_images = []
    background_dataset_labels = set()
    global_labels = set()
    label_color_map = {}
    _cached_bg_label_stats = []
    _cached_bg_label_stats_path = ""
    _memory_background_path = ""
    _background_label_scan_completed = False
    _dataset_stats_dirty = False

    def _build_bg_label_stats_snapshot(self):
        from labelforge.ui.mixins.memory_record import MemoryRecordMixin
        return MemoryRecordMixin._build_bg_label_stats_snapshot(self)


def test_memory_snapshot_parses_each_json_once(count_json_opens, tmp_path):
    paths = _write_dataset(tmp_path)
    host = _MemoryHost()
    host.background_images = list(paths)

    stats = host._build_bg_label_stats_snapshot()

    _assert_each_json_parsed_once(count_json_opens, paths)
    rows = {row["label"]: row for row in stats}
    assert rows["cat"]["count"] == len(paths)
    assert rows["dog"]["count"] == len(paths)
    assert rows["cat"]["tasks"] == ["det"]
    assert rows["dog"]["tasks"] == ["seg"]
