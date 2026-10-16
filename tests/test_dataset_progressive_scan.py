"""回归测试：大数据集切换时的卡顿与状态图标延迟。

历史问题：
1. 状态只在整轮扫描结束时一次性应用，6700 张数据集期间所有状态圈一直
   是 pending（虚线灰圈），直到扫描完成才变绿 —— 需要分块增量回填。
2. 切换数据集时 `_build_bg_label_stats_snapshot()` 同步重扫整个旧数据集
   （O(n) JSON 解析），造成 UI 线程卡住 —— 缓存新鲜时必须复用。
"""
import json
import time

from labelforge.engine.image_loader import (
    STATUS_ANNOTATED,
    STATUS_UNANNOTATED,
    scan_dataset_full,
)


def _make_dataset(tmp_path, n, shapes_per=15):
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i:05d}.png"
        p.write_bytes(b"x")
        shapes = [{"label": f"c{k % 10}"} for k in range(shapes_per)]
        (tmp_path / f"img{i:05d}.json").write_text(
            json.dumps({"shapes": shapes}), encoding="utf-8")
        paths.append(str(p))
    return paths


def test_scan_full_streams_progress_batches(tmp_path):
    """状态必须分块流式回调，而不是只在结束时一次性返回。"""
    paths = _make_dataset(tmp_path, 500)

    batches = []
    labels, counts, statuses = scan_dataset_full(
        paths, None, progress_cb=lambda b: batches.append(dict(b)))

    assert len(statuses) == 500
    assert len(batches) >= 2, "500 张应至少分 2 批回调"
    assert sum(len(b) for b in batches) == 500, "批次总数必须等于图片数"
    merged = {}
    for b in batches:
        merged.update(b)
    assert merged == statuses


def test_scan_full_first_batch_arrives_before_completion(tmp_path):
    """首批必须在扫描全部完成之前到达，否则图标依旧会延迟。"""
    paths = _make_dataset(tmp_path, 600)

    first_at = {}
    start = time.time()

    def _cb(batch):
        if not first_at:
            first_at["t"] = time.time() - start
            first_at["n"] = len(batch)

    _, _, _ = scan_dataset_full(paths, None, progress_cb=_cb)
    total = time.time() - start

    assert first_at, "未收到任何进度回调"
    assert first_at["t"] < total, "首批不应在扫描结束后才到达"


def test_scan_full_without_progress_cb_still_returns_all(tmp_path):
    """不传 progress_cb 时行为保持不变（向后兼容）。"""
    paths = _make_dataset(tmp_path, 120)
    labels, counts, statuses = scan_dataset_full(paths)
    assert len(statuses) == 120
    assert labels == {"c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"}
    assert sum(counts.values()) == 120 * 15


def test_progress_cb_exception_does_not_abort_scan(tmp_path):
    """进度回调抛异常不应中断扫描。"""
    paths = _make_dataset(tmp_path, 450)

    def _bad_cb(batch):
        raise RuntimeError("boom")

    labels, counts, statuses = scan_dataset_full(paths, None, progress_cb=_bad_cb)
    assert len(statuses) == 450


class _SnapshotHost:
    """最小化 host，模拟 ImageEditor 上缓存复用所需属性。"""
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


def _make_snapshot_host(paths, dataset_path, **overrides):
    host = _SnapshotHost()
    host.background_images = list(paths)
    host._memory_background_path = dataset_path
    for k, v in overrides.items():
        setattr(host, k, v)
    return host


def test_snapshot_reuses_fresh_cache_without_rescan(tmp_path, monkeypatch):
    """扫描完成且未脏时，快照必须复用缓存，绝不重新解析 JSON。"""
    paths = _make_dataset(tmp_path, 300)
    host = _make_snapshot_host(
        paths, str(tmp_path),
        _cached_bg_label_stats=[{"label": "c0", "count": 300, "color": ""}],
        _cached_bg_label_stats_path=str(tmp_path),
        _background_label_scan_completed=True,
        _dataset_stats_dirty=False,
    )

    from labelforge.engine import image_loader

    def _boom(*a, **k):
        raise AssertionError("缓存新鲜时不应重扫磁盘")

    monkeypatch.setattr(image_loader, "collect_background_label_counts", _boom)
    stats = host._build_bg_label_stats_snapshot()
    assert stats == [{"label": "c0", "count": 300, "color": "", "tasks": []}]


def test_snapshot_rescans_when_dirty(tmp_path):
    """数据集被编辑（dirty）时必须重新扫描以反映最新计数。"""
    paths = _make_dataset(tmp_path, 300)
    host = _make_snapshot_host(
        paths, str(tmp_path),
        _cached_bg_label_stats=[{"label": "stale", "count": 1, "color": ""}],
        _cached_bg_label_stats_path=str(tmp_path),
        _background_label_scan_completed=True,
        _dataset_stats_dirty=True,
    )
    stats = host._build_bg_label_stats_snapshot()
    labels = {s["label"] for s in stats}
    assert "stale" not in labels
    assert labels == {f"c{k}" for k in range(10)}
    assert sum(s["count"] for s in stats) == 300 * 15


def test_snapshot_cache_reuse_is_fast(tmp_path):
    """缓存复用路径必须远快于全量重扫（回归守卫）。"""
    paths = _make_dataset(tmp_path, 1500)
    host = _make_snapshot_host(
        paths, str(tmp_path),
        _cached_bg_label_stats=[
            {"label": f"c{k}", "count": 1500, "color": ""} for k in range(10)],
        _cached_bg_label_stats_path=str(tmp_path),
        _background_label_scan_completed=True,
        _dataset_stats_dirty=False,
    )
    t = time.time()
    host._build_bg_label_stats_snapshot()
    cached = time.time() - t

    host._dataset_stats_dirty = True
    t = time.time()
    host._build_bg_label_stats_snapshot()
    rescan = time.time() - t

    assert cached < 0.1, f"缓存复用过慢: {cached:.3f}s"
    assert cached < rescan, "缓存复用应快于重扫"


def test_batch_loader_uses_time_budget_not_fixed_tiny_batch():
    """列表填充必须按时间预算推进，而不是固定 50 条/30ms 的小批次。

    历史问题：50 条/批 + 30ms 定时器，6748 张要 135 个 tick ≈ 4s 纯等待，
    而真正插入 6748 行只需 ~0.2s。
    """
    import inspect

    from labelforge.engine.image_loader import ImageLoaderMixin

    src = inspect.getsource(ImageLoaderMixin._load_next_background_batch)
    assert "perf_counter" in src, "应按时间预算推进批次"
    assert "TIME_BUDGET" in src
    assert "BATCH_SIZE = 50" not in src, "不应再用固定 50 条小批次"


def test_background_list_timer_interval_is_zero():
    """填充定时器间隔必须为 0，避免每 tick 人为 sleep 30ms。"""
    import inspect

    from labelforge.engine.image_loader import ImageLoaderMixin

    src = inspect.getsource(ImageLoaderMixin.load_background_folder)
    assert "setInterval(0)" in src
    assert "setInterval(30)" not in src


def test_batch_loader_disables_updates_while_filling():
    """填充期间必须关闭控件重绘，避免逐行重排/重绘。"""
    import inspect

    from labelforge.engine.image_loader import ImageLoaderMixin

    src = inspect.getsource(ImageLoaderMixin._load_next_background_batch)
    assert "setUpdatesEnabled(False)" in src
    assert "setUpdatesEnabled(True)" in src
