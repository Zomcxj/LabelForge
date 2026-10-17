"""质检 lint 纯逻辑测试。"""
import json

from labelforge.engine.quality_lint import lint_shapes
from labelforge.core.config import QUALITY_LINT_CONFIG


def _box(label="cat", x=10, y=10, w=100, h=50, shape_type=None, **extra):
    box = {"label": label, "x": x, "y": y, "width": w, "height": h}
    if shape_type:
        box["shape_type"] = shape_type
    box.update(extra)
    return box


def test_tiny_box_is_no_longer_a_kind():
    """「超小框」已删除：原来按原图短边 < 8px 判，与训练分辨率脱钩。

    同一像素尺寸在不同的图上结论天差地别（8px 在 480 图上缩放到 640 后
    是 10.7px、很健康；在 1920 图上只剩 2.7px、根本检不到），所以统一到
    640 空间的三档判据，不再有这一档。
    """
    from labelforge.engine.quality_lint import KIND_LABELS
    assert "tiny_box" not in KIND_LABELS


def test_clean_shapes_produce_no_issues():
    issues = lint_shapes([_box()], 200, 200)
    assert issues == []


def test_out_of_bounds_detected():
    issues = lint_shapes([_box(x=-5)], 200, 200)
    kinds = {i["kind"] for i in issues}
    assert "out_of_bounds" in kinds


def test_out_of_bounds_tolerance_one_pixel():
    assert lint_shapes([_box(x=-1, y=-1, w=101, h=51)], 100, 50) == []


def test_degenerate_box_on_small_image_is_flagged():
    """200x200 上的 5x50：等比缩放到 640 后短边 16px? 不 —— 5x3.2=16px 边界。

    5/200 = 2.5%，正好是「稳定检出」线 —— 通过。这条守住「小图上退化框
    也不该乱报」：旧口径按原图像素 5px 报了超小框，新口径按 640 空间
    判是 16px，属于能稳定检出。
    """
    assert lint_shapes([_box(w=5, h=50)], 200, 200) == []


def test_tiny_issue_records_width_as_small_side():
    """200x200 图上 1x2 框：宽 0.5% 踩 fail 线、高 1.0% 达标 —— 只有宽不足。"""
    issues = lint_shapes([_box(w=1, h=2)], 200, 200)
    tiny = [i for i in issues if i["kind"] == "tiny_unreachable"]
    assert len(tiny) == 1
    assert tiny[0]["min_side"] == "width"
    assert tiny[0]["both_below"] is False


def test_tiny_issue_records_height_as_small_side():
    issues = lint_shapes([_box(w=2, h=1)], 200, 200)
    tiny = [i for i in issues if i["kind"] == "tiny_unreachable"]
    assert tiny[0]["min_side"] == "height"
    assert tiny[0]["both_below"] is False


def test_tiny_issue_marks_both_sides_below_fail_line():
    """1x1 框：宽高都是 0.5%，都低于 0.625% fail 线。"""
    issues = lint_shapes([_box(w=1, h=1)], 200, 200)
    tiny = [i for i in issues if i["kind"] == "tiny_unreachable"]
    assert tiny[0]["both_below"] is True


def test_tiny_issue_warn_tier_compares_against_warn_line():
    """100x100 图上 0.8x0.8 框：0.8% 落 warn 档；两侧都低于 1.25% warn 线。"""
    issues = lint_shapes([_box(w=0.8, h=0.8)], 100, 100)
    tiny = [i for i in issues if i["kind"] == "tiny_needs_p2"]
    assert len(tiny) == 1
    assert tiny[0]["both_below"] is True


def test_tiny_issue_warn_tier_single_side_below():
    """0.8x1.4 框：min 0.8% 落 warn 档；高 1.4% ≥ 1.25% —— 只有宽不足。"""
    issues = lint_shapes([_box(w=0.8, h=1.4)], 100, 100)
    tiny = [i for i in issues if i["kind"] == "tiny_needs_p2"]
    assert tiny[0]["min_side"] == "width"
    assert tiny[0]["both_below"] is False


def test_duplicate_boxes_same_label_detected():
    a = _box(x=0, y=0, w=100, h=100)
    b = _box(x=2, y=2, w=100, h=100)
    issues = lint_shapes([a, b], 500, 500)
    kinds = {i["kind"] for i in issues}
    assert "duplicate" in kinds


def test_cross_label_overlap_detected():
    a = _box(label="cat", x=0, y=0, w=100, h=100)
    b = _box(label="dog", x=2, y=2, w=100, h=100)
    issues = lint_shapes([a, b], 500, 500)
    kinds = {i["kind"] for i in issues}
    assert "cross_label_overlap" in kinds
    assert "duplicate" not in kinds


def test_bbox_used_for_polygon_bounds_check():
    poly = {"label": "cat", "shape_type": "polygon",
            "points": [[0, 0], [500, 0], [500, 100], [0, 100]],
            "x": 0, "y": 0, "width": 500, "height": 100}
    issues = lint_shapes([poly], 200, 200)
    assert {i["kind"] for i in issues} == {"out_of_bounds"}


# --------------------------------------------------------------------------
# Task 2: 近似名 + group 不一致
# --------------------------------------------------------------------------
def test_similar_label_names_detected():
    a = _box(label="car", x=0, y=0, w=10, h=10)
    b = _box(label="Car", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    similar = [i for i in issues if i["kind"] == "similar_name"]
    assert len(similar) == 1
    assert "Car" in similar[0]["detail"] and "car" in similar[0]["detail"]


def test_distant_label_names_not_flagged():
    a = _box(label="car", x=0, y=0, w=10, h=10)
    b = _box(label="bicycle", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    assert not [i for i in issues if i["kind"] == "similar_name"]


def test_whitespace_only_label_difference_detected():
    a = _box(label="car", x=0, y=0, w=10, h=10)
    b = _box(label="car ", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    assert [i for i in issues if i["kind"] == "similar_name"]


def test_point_without_group_box_detected():
    point = {"label": "kp", "shape_type": "point", "points": [[10, 10]],
             "x": 10, "y": 10, "width": 0, "height": 0, "group_id": 1}
    issues = lint_shapes([point], 200, 200)
    kinds = {i["kind"] for i in issues}
    assert "group_mismatch" in kinds


def test_point_outside_group_box_detected():
    box = {"label": "car", "x": 0, "y": 0, "width": 50, "height": 50, "group_id": 1}
    point = {"label": "kp", "shape_type": "point", "points": [[200, 200]],
             "x": 200, "y": 200, "width": 0, "height": 0, "group_id": 1}
    issues = lint_shapes([box, point], 500, 500)
    kinds = {i["kind"] for i in issues}
    assert "group_mismatch" in kinds


def test_point_inside_group_box_ok():
    box = {"label": "car", "x": 0, "y": 0, "width": 100, "height": 100, "group_id": 1}
    point = {"label": "kp", "shape_type": "point", "points": [[50, 50]],
             "x": 50, "y": 50, "width": 0, "height": 0, "group_id": 1}
    issues = lint_shapes([box, point], 500, 500)
    assert not [i for i in issues if i["kind"] == "group_mismatch"]


def test_group_box_without_point_detected():
    box = {"label": "car", "x": 0, "y": 0, "width": 100, "height": 100, "group_id": 7}
    issues = lint_shapes([box], 500, 500)
    kinds = {i["kind"] for i in issues}
    assert "group_mismatch" in kinds


def test_group_mismatch_matches_point_warning():
    """一致性契约：group 判定必须与 point_warning 结论一致。"""
    from labelforge.engine.shape_io import point_warning
    point = {"label": "kp", "shape_type": "point", "points": [[10, 10]],
             "x": 10, "y": 10, "width": 0, "height": 0, "group_id": 1}
    assert point_warning(point, [point]) == "无同组框"
    issues = lint_shapes([point], 200, 200)
    assert [i for i in issues if i["kind"] == "group_mismatch"]


# --------------------------------------------------------------------------
# Task 3: 数据集扫描
# --------------------------------------------------------------------------
def _write_image_and_json(tmp_path, stem, shapes, image_size=(200, 200)):
    img = tmp_path / f"{stem}.png"
    img.write_bytes(b"x")
    payload = {
        "shapes": shapes,
        "imageWidth": image_size[0], "imageHeight": image_size[1],
    }
    (tmp_path / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(img)


def test_lint_dataset_ignores_missing_sidecar(tmp_path):
    """缺 sidecar 不再上报（背景图列表状态图标已覆盖）。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = tmp_path / "a.png"
    img.write_bytes(b"x")

    results = lint_dataset([str(img)])

    assert results['issues'] == []
    assert 'missing_sidecar' not in results['summary']


def test_lint_dataset_ignores_empty_annotation(tmp_path):
    """空标注不再上报（背景图列表状态图标已覆盖）。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "b", [])

    results = lint_dataset([img])

    assert results['issues'] == []
    assert 'empty_annotation' not in results['summary']


def test_lint_dataset_reports_geometry_issues(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "c", [
        {"label": "cat", "x": -50, "y": 0, "width": 100, "height": 100},
    ])

    results = lint_dataset([img])

    assert results['summary']['out_of_bounds'] == 1


def test_lint_dataset_progress_callback(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    paths = [_write_image_and_json(tmp_path, f"img{i}", []) for i in range(5)]

    seen = []
    lint_dataset(paths, progress_cb=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (5, 5)


def test_lint_dataset_uses_memory_override(tmp_path):
    """已加载内存图以内存框为准，不读磁盘。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "d", [
        {"label": "cat", "x": 0, "y": 0, "width": 10, "height": 10},
    ])
    memory = {0: [{"label": "cat", "x": -99, "y": 0, "width": 10, "height": 10}]}

    results = lint_dataset([img], memory_boxes=memory)

    assert results['summary']['out_of_bounds'] == 1


def test_lint_dataset_skips_pairwise_over_limit(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    limit = QUALITY_LINT_CONFIG['max_boxes_pairwise']
    shapes = [
        {"label": "cat", "x": 0, "y": 0, "width": 50, "height": 50}
        for _ in range(limit + 1)
    ]
    # 1000x1000 让 50px 框占 5%（比例健康）。早先写的 10000x10000 只有 0.5%，
    # 会额外产出 501 条小目标问题；本测试只断言 duplicate/skipped_pairwise 计数
    # 所以看不出被污染，但日后谁加一条 issue 总数断言就会莫名炸掉。
    img = _write_image_and_json(tmp_path, "e", shapes, image_size=(1000, 1000))

    results = lint_dataset([img])

    assert results['summary']['duplicate'] == 0
    assert results['summary']['skipped_pairwise'] == 1
    # 钉住"夹具尺寸是干净的"这一前提，被污染时直接失败而不是悄悄通过
    assert results['summary']['tiny_unreachable'] == 0
    assert results['summary']['tiny_needs_p2'] == 0


def test_lint_dataset_interruptible(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    paths = [_write_image_and_json(tmp_path, f"img{i}", []) for i in range(20)]

    results = lint_dataset(paths, is_interrupted=lambda: True)

    assert results['summary']['scanned_images'] == 0


def test_bbox_returns_none_for_non_numeric_coordinates():
    from labelforge.engine.quality_lint import _bbox
    assert _bbox({"label": "a", "x": "abc", "y": 0, "width": 10, "height": 10}) is None
    assert _bbox({"label": "a", "points": [[5]]}) is None
    assert _bbox({"label": "a", "points": []}) is None
    assert _bbox({"label": "a", "points": [["x", "y"]]}) is None
    assert _bbox({"label": "a", "x": 0, "y": 0, "width": 10, "height": 10}) == (0, 0, 10, 10)


def test_malformed_shape_does_not_abort_scan(tmp_path):
    """坏 shape 不能中断整库扫描，其它图的问题仍要报出来。"""
    from labelforge.engine.quality_lint import lint_dataset
    bad = _write_image_and_json(tmp_path, "bad", [
        {"label": "cat", "points": [[5]]},
        {"label": "cat", "x": "abc", "y": 0, "width": 10, "height": 10},
    ])
    good = _write_image_and_json(tmp_path, "good", [
        {"label": "cat", "x": -50, "y": 0, "width": 100, "height": 100},
    ])

    results = lint_dataset([bad, good])

    assert results['summary']['out_of_bounds'] == 1
    assert results['summary']['scanned_images'] == 2


def test_grouped_point_with_string_coords_does_not_abort_scan(tmp_path):
    """分组关键点坐标是字符串时不能 TypeError 中断扫描。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "strpt", [
        {"label": "car", "x": -50, "y": 0, "width": 100, "height": 100,
         "group_id": 1},
        {"label": "kp", "shape_type": "point", "points": [["x", "y"]],
         "x": "x", "y": "y", "width": 0, "height": 0, "group_id": 1},
    ])

    results = lint_dataset([img])

    assert results['summary']['out_of_bounds'] == 1
    assert 'error' not in results


def test_unhashable_group_id_does_not_abort_scan(tmp_path):
    """group_id 是 list 时集合构造不能 TypeError 中断扫描。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "badgid", [
        {"label": "cat", "x": -50, "y": 0, "width": 100, "height": 100,
         "group_id": []},
    ])

    results = lint_dataset([img])

    assert results['summary']['out_of_bounds'] == 1
    assert 'error' not in results


def test_lint_worker_does_not_report_error_for_malformed_group_data(tmp_path):
    """以上坏数据不能让 worker 报 error=True（UI 会误显示扫描失败）。"""
    from labelforge.engine.quality_lint import QualityLintWorker
    img = _write_image_and_json(tmp_path, "worker_bad", [
        {"label": "cat", "x": -50, "y": 0, "width": 100, "height": 100,
         "group_id": []},
        {"label": "kp", "shape_type": "point", "points": [["x", "y"]],
         "x": "x", "y": "y", "width": 0, "height": 0, "group_id": 1},
    ])
    worker = QualityLintWorker((img,))
    emitted = []
    worker.lint_finished = type("S", (), {
        "emit": staticmethod(lambda payload: emitted.append(payload))})()
    worker.isInterruptionRequested = lambda: False

    worker.run()

    assert emitted and not emitted[0].get('error')
    assert emitted[0]['summary']['out_of_bounds'] == 1


def test_read_json_shapes_tolerates_non_numeric_size(tmp_path):
    from labelforge.engine.quality_lint import _read_json_shapes
    img = tmp_path / "weird.png"
    img.write_bytes(b"x")
    (tmp_path / "weird.json").write_text(json.dumps({
        "shapes": [{"label": "cat", "x": -50, "y": 0, "width": 10, "height": 10}],
        "imageWidth": "abc", "imageHeight": None,
    }), encoding="utf-8")

    info = _read_json_shapes(str(tmp_path / "weird.json"))

    assert info is not None
    assert info[1] == 0 and info[2] == 0
    assert info[0][0]["label"] == "cat"


def test_filtered_lint_result_preserves_error_flag():
    """error 结果经忽略规则过滤后必须保留标记，UI 才能显示失败。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    editor = type("Editor", (QualityLintMixin,), {})()
    filtered = editor._filtered_lint_result(
        {"issues": [], "summary": {}, "error": True}, {})
    assert filtered.get("error") is True


def test_lint_dialog_shows_failure_not_clean_message():
    """error 结果不能显示『未发现问题』。"""
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog.set_result)
    assert "error" in src
    assert "扫描失败" in src
    from labelforge.ui.i18n import _strings
    assert "扫描失败" in _strings["zh"]
    assert "扫描失败" in _strings["en"]


def test_cleanup_lint_worker_waits_for_worker():
    """关闭质检弹窗必须等待 worker 结束，不能只发中断就丢引用（回归）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Worker:
        def __init__(self):
            self.interrupted = 0
            self.waited = []

        def isRunning(self):
            return True

        def requestInterruption(self):
            self.interrupted += 1

        def wait(self, timeout=0):
            self.waited.append(timeout)
            return True

    editor = type("Editor", (QualityLintMixin,), {})()
    worker = Worker()
    editor._lint_worker = worker

    editor._cleanup_lint_worker(worker)

    assert worker.interrupted == 1
    assert worker.waited and worker.waited[0] > 0
    assert editor._lint_worker is None


def test_cleanup_lint_worker_tolerates_missing_wait():
    """假 worker（无 wait）不能抛异常。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Worker:
        def isRunning(self):
            return False

        def requestInterruption(self):
            raise AssertionError("未运行不应中断")

    editor = type("Editor", (QualityLintMixin,), {})()
    editor._lint_worker = Worker()

    editor._cleanup_lint_worker()


def test_lint_dataset_abandons_pending_reads_on_interrupt(tmp_path, monkeypatch):
    """中断后不能等所有排队读盘 future 完成。"""
    import time
    from labelforge.engine import quality_lint

    def _slow_read(json_path):
        time.sleep(0.5)
        return [], 0, 0

    monkeypatch.setattr(quality_lint, "_read_json_shapes", _slow_read)
    paths = [str(tmp_path / f"p{i}.png") for i in range(40)]

    t0 = time.perf_counter()
    result = quality_lint.lint_dataset(paths, is_interrupted=lambda: True)
    elapsed = time.perf_counter() - t0

    assert result['summary']['scanned_images'] == 0
    # 8 workers 跑完全部 40 个 0.5s 任务要 2.5s；中断后只等已在跑的 8 个
    assert elapsed < 1.2, elapsed


def test_sync_boxes_after_delete_clears_undo_history():
    """跨类删除不可撤销：同步内存后必须清空撤销栈，否则 Ctrl+Z 会复活删掉的框。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Undo:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.detection_boxes_dict = {0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ]}
    editor.detection_boxes = list(editor.detection_boxes_dict[0])
    editor.current_background_index = 0
    editor.undo_manager = Undo()

    editor._sync_boxes_after_delete({0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ]})

    assert [b["label"] for b in editor.detection_boxes_dict[0]] == ["ZTruck"]
    assert editor.undo_manager.cleared == 1


def test_sync_boxes_after_delete_keeps_undo_when_nothing_removed():
    """没有实际删除时不能清空撤销栈。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Undo:
        def __init__(self):
            self.cleared = 0

        def clear(self):
            self.cleared += 1

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.detection_boxes_dict = {0: [
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ]}
    editor.detection_boxes = list(editor.detection_boxes_dict[0])
    editor.current_background_index = 0
    editor.undo_manager = Undo()

    editor._sync_boxes_after_delete({0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ]})

    assert editor.undo_manager.cleared == 0


def test_sync_boxes_after_delete_tolerates_missing_undo_manager():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.detection_boxes_dict = {0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ]}
    editor.detection_boxes = list(editor.detection_boxes_dict[0])
    editor.current_background_index = 0

    editor._sync_boxes_after_delete({0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ]})

    assert editor.detection_boxes_dict[0] == []


def test_worker_reports_error_field_on_failure(monkeypatch):
    """扫描异常必须带 error=True，UI 才能提示失败而不是『未发现问题』。"""
    from labelforge.engine import quality_lint

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(quality_lint, "lint_dataset", _boom)
    worker = quality_lint.QualityLintWorker(("a.png",))
    emitted = []
    worker.lint_finished = type("S", (), {
        "emit": staticmethod(lambda payload: emitted.append(payload))})()
    worker.isInterruptionRequested = lambda: False

    worker.run()

    assert emitted and emitted[0].get('error') is True
    assert emitted[0].get('issues') == []


# --------------------------------------------------------------------------
# Task 4: Worker
# --------------------------------------------------------------------------
def test_worker_exposes_signals_and_run_calls_lint():
    import inspect
    from labelforge.engine import quality_lint

    assert hasattr(quality_lint, "QualityLintWorker")
    source = inspect.getsource(quality_lint.QualityLintWorker.run)
    assert "lint_dataset" in source
    assert "memory_boxes" in source
    assert "isInterruptionRequested" in source


# --------------------------------------------------------------------------
# Task 5: i18n
# --------------------------------------------------------------------------
def test_i18n_quality_lint_terms():
    from labelforge.ui.i18n import _strings
    for key in ("质检", "标注质检", "越界框",
                "小目标不可检出", "小目标需放大", "重复框", "异类重叠", "近似名", "group 不一致",
                "跳过两两比对", "正在扫描", "未发现问题", "双击跳转到问题图"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


# --------------------------------------------------------------------------
# Task 6: UI 接线
# --------------------------------------------------------------------------
def test_options_popup_has_lint_button_source():
    import inspect
    from labelforge.ui.mixins.options_popup import OptionsPopupMixin
    source = inspect.getsource(OptionsPopupMixin._create_options_menu)
    assert "lint_btn" in source
    assert "_open_quality_lint" in source


def test_quality_lint_mixin_contract():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    for name in ("_open_quality_lint",
                 "_jump_to_lint_issue", "_cleanup_lint_worker"):
        assert hasattr(QualityLintMixin, name), name


def test_main_window_includes_quality_lint_mixin():
    from labelforge.ui.main_window import ImageEditor
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    assert issubclass(ImageEditor, QualityLintMixin)


def test_dialog_exposes_progress_and_result_api():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    for name in ("set_progress", "set_result", "issue_activated"):
        assert hasattr(QualityLintDialog, name), name


def test_jump_to_lint_issue_selects_box():
    """双击跳图逻辑：切图 + 选中问题框。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    calls = []
    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png", "b.png"]
    editor.current_background_index = 0
    editor.switch_background_to_index = lambda idx: calls.append(("switch", idx))
    editor.detection_boxes = [{"label": "cat"}]
    editor.canvas = type("Canvas", (), {
        "update_status_label": lambda self: calls.append(("status",)),
        "update": lambda self: calls.append(("update",)),
    })()

    editor._jump_to_lint_issue({
        "image_index": 1, "box_index": 0, "kind": "out_of_bounds"})

    assert ("switch", 1) in calls
    assert editor.canvas.selected_box == 0
    assert editor.canvas.selected_boxes == [0]


def test_jump_to_lint_issue_ignores_invalid_index():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png"]
    editor.current_background_index = 0
    editor.switch_background_to_index = lambda idx: (_ for _ in ()).throw(
        AssertionError("不应切图"))
    editor.detection_boxes = []

    editor._jump_to_lint_issue({"image_index": 99, "box_index": None})


def test_keypoint_not_reported_as_tiny_target():
    """关键点 width/height=0 是正常形态，不能报小目标两类档。"""
    point = {"label": "kp", "shape_type": "point", "points": [[50, 50]],
             "x": 50, "y": 50, "width": 0, "height": 0, "group_id": 1}
    box = {"label": "car", "x": 0, "y": 0, "width": 100, "height": 100,
           "group_id": 1}
    issues = lint_shapes([box, point], 500, 500)
    assert not [i for i in issues
                if i["kind"] in ("tiny_unreachable", "tiny_needs_p2")]


def test_edit_distance_one_typo_detected(monkeypatch):
    """编辑距离路径：阈值设为 1 时差一个字符（非大小写）要报。"""
    monkeypatch.setitem(QUALITY_LINT_CONFIG, 'name_edit_distance', 1)
    a = _box(label="car", x=0, y=0, w=10, h=10)
    b = _box(label="cat", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    similar = [i for i in issues if i["kind"] == "similar_name"]
    assert len(similar) == 1


def test_edit_distance_two_typos_not_detected(monkeypatch):
    monkeypatch.setitem(QUALITY_LINT_CONFIG, 'name_edit_distance', 1)
    a = _box(label="car", x=0, y=0, w=10, h=10)
    b = _box(label="abcd", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    assert not [i for i in issues if i["kind"] == "similar_name"]


# --------------------------------------------------------------------------
# 忽略规则 + 汇总
# --------------------------------------------------------------------------
def test_edit_distance_default_is_zero():
    """默认仅归一化差异，Truck/VTruck 这类业务命名不应误报。"""
    a = _box(label="Truck", x=0, y=0, w=10, h=10)
    b = _box(label="VTruck", x=50, y=50, w=10, h=10)
    issues = lint_shapes([a, b], 200, 200)
    assert not [i for i in issues if i["kind"] == "similar_name"]


def test_filter_ignored_issues_by_detail():
    from labelforge.engine.quality_lint import filter_ignored_issues
    issues = [
        {"kind": "similar_name", "detail": "car / Car"},
        {"kind": "similar_name", "detail": "cat / Car"},
        {"kind": "tiny_box", "detail": "W:1 H:1"},
    ]
    filtered = filter_ignored_issues(
        issues, {"similar_name": ["car / Car"]})
    assert len(filtered) == 2
    assert {i["detail"] for i in filtered} == {"cat / Car", "W:1 H:1"}


def test_filter_ignored_issues_by_star():
    from labelforge.engine.quality_lint import filter_ignored_issues
    issues = [
        {"kind": "similar_name", "detail": "car / Car"},
        {"kind": "tiny_box", "detail": "W:1 H:1"},
    ]
    filtered = filter_ignored_issues(issues, {"similar_name": ["*"]})
    assert len(filtered) == 1
    assert filtered[0]["kind"] == "tiny_box"


def test_filter_ignored_issues_empty_rules_passthrough():
    from labelforge.engine.quality_lint import filter_ignored_issues
    issues = [{"kind": "tiny_box", "detail": "x"}]
    assert filter_ignored_issues(issues, None) == issues
    assert filter_ignored_issues(issues, {}) == issues


def test_summarize_issues_counts_by_kind():
    from labelforge.engine.quality_lint import summarize_issues
    issues = [
        {"kind": "tiny_box"}, {"kind": "tiny_box"}, {"kind": "duplicate"},
    ]
    assert summarize_issues(issues) == {"tiny_box": 2, "duplicate": 1}


def test_config_manager_roundtrips_lint_ignored_rules():
    from labelforge.core import config_manager
    rules = {"similar_name": ["car / Car"], "tiny_box": ["*"]}
    config_manager.save_all(lint_ignored_rules=rules)
    loaded = config_manager.load_all()
    assert loaded['lint_ignored_rules'] == rules
    config_manager.save_all(lint_ignored_rules={})
    assert config_manager.load_all()['lint_ignored_rules'] == {}


def test_dialog_exposes_ignore_api():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    for name in ("ignore_requested", "ignore_many_requested",
                 "_remove_issues", "_on_context_menu",
                 "_selected_issues", "_on_filter_changed",
                 "_rebuild_filter_combo", "_apply_filter"):
        assert hasattr(QualityLintDialog, name), name


def test_mixin_exposes_ignore_helpers():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    for name in ("_ignore_lint_issue", "_ignore_lint_issues",
                 "_get_lint_ignored_rules", "_save_lint_ignored_rules"):
        assert hasattr(QualityLintMixin, name), name


def test_lint_dialog_is_non_modal_source():
    """质检弹窗必须非模态打开（show 而非 exec_），否则无法操作主窗口。"""
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    source = inspect.getsource(QualityLintMixin._open_quality_lint)
    assert "dialog.show()" in source
    assert "dialog.exec_()" not in source


def test_lint_dialog_syncs_titlebar_theme():
    """质检弹窗必须同步标题栏主题（对齐 stats 弹窗）。"""
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    source = inspect.getsource(QualityLintDialog.__init__)
    assert "set_titlebar_dark" in source


def test_lint_dialog_filter_combo_and_multi_select_source():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog)
    assert "ExtendedSelection" in src
    assert "filter_combo" in src
    assert "_kind_filter" in src


def test_i18n_ignore_terms():
    from labelforge.ui.i18n import _strings
    for key in ("右键忽略", "忽略此类问题", "忽略此项", "已忽略"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_remove_issues_matches_by_content_not_identity():
    """表格 UserRole 取回的是副本，忽略必须按内容匹配（回归）。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    import inspect
    src = inspect.getsource(QualityLintDialog._remove_issues)
    assert "id(" not in src, "不能按 id 匹配"
    assert "==" in src


# --------------------------------------------------------------------------
# 异类重叠按选中项删除：判定 / 写盘 / worker
# --------------------------------------------------------------------------
def _cross_issue(index, ia, la, ib, lb, path=None, iou=0.9):
    return {
        "kind": "cross_label_overlap", "label": la, "box_index": ia,
        "label_a": la, "box_index_a": ia, "label_b": lb, "box_index_b": ib,
        "detail": f"IoU:{iou:.2f} #{ia}({la}) vs ({lb})",
        "image_index": index, "image_path": path,
    }


def test_selected_removal_indices_matches_target_side():
    from labelforge.engine.quality_lint import _selected_removal_indices
    shapes = [
        _box(label="Truck", x=0, y=0, w=100, h=100),
        _box(label="ZTruck", x=2, y=2, w=100, h=100),
    ]
    issue = _cross_issue(0, 0, "Truck", 1, "ZTruck")
    assert _selected_removal_indices(shapes, [issue], "Truck")[0] == [0]
    assert _selected_removal_indices(shapes, [issue], "ZTruck")[0] == [1]


def test_selected_removal_indices_rejects_index_drift():
    """索引与类别不一致时不能删（防索引漂移误删）。"""
    from labelforge.engine.quality_lint import _selected_removal_indices
    shapes = [
        _box(label="Car", x=0, y=0, w=100, h=100),
        _box(label="ZTruck", x=2, y=2, w=100, h=100),
    ]
    issue = _cross_issue(0, 0, "Truck", 1, "ZTruck")
    assert _selected_removal_indices(shapes, [issue], "Truck")[0] == []
    assert _selected_removal_indices(shapes, [issue], "ZTruck")[0] == [1]


def test_selected_removal_indices_empty_target():
    from labelforge.engine.quality_lint import _selected_removal_indices
    shapes = [_box(label="Truck")]
    issue = _cross_issue(0, 0, "Truck", 1, "ZTruck")
    assert _selected_removal_indices(shapes, [issue], "")[0] == []


def test_remove_selected_deletes_only_selected_images(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img0 = _write_image_and_json(tmp_path, "sel0", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    img1 = _write_image_and_json(tmp_path, "sel1", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", img0)]

    result = remove_selected_cross_overlaps(
        selected, "Truck", [img0, img1])

    assert result["removed"] == 1
    assert result["images_changed"] == 1
    assert result["failed"] == []
    p0 = json.loads((tmp_path / "sel0.json").read_text(encoding="utf-8"))
    p1 = json.loads((tmp_path / "sel1.json").read_text(encoding="utf-8"))
    assert [s["label"] for s in p0["shapes"]] == ["ZTruck"]
    assert [s["label"] for s in p1["shapes"]] == ["Truck", "ZTruck"]


def test_remove_selected_multiple_images_and_labels(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img0 = _write_image_and_json(tmp_path, "m0", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    img1 = _write_image_and_json(tmp_path, "m1", [
        {"label": "Car", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "SUV", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    selected = [
        _cross_issue(0, 0, "Truck", 1, "ZTruck", img0),
        _cross_issue(1, 0, "Car", 1, "SUV", img1),
    ]

    result = remove_selected_cross_overlaps(selected, "Truck", [img0, img1])

    assert result["removed"] == 1
    assert result["images_changed"] == 1
    p1 = json.loads((tmp_path / "m1.json").read_text(encoding="utf-8"))
    assert [s["label"] for s in p1["shapes"]] == ["Car", "SUV"]


def test_remove_selected_preserves_other_json_fields(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = tmp_path / "keep2.png"
    img.write_bytes(b"x")
    payload = {
        "version": "5.0.1", "flags": {"note": "x"}, "imageData": None,
        "imagePath": "keep2.png", "imageHeight": 200, "imageWidth": 200,
        "shapes": [
            {"label": "Truck", "points": [[0, 0], [100, 0], [100, 100], [0, 100]],
             "shape_type": "rectangle", "flags": {}},
            {"label": "ZTruck", "points": [[2, 2], [102, 2], [102, 102], [2, 102]],
             "shape_type": "rectangle", "flags": {}},
        ],
    }
    (tmp_path / "keep2.json").write_text(json.dumps(payload), encoding="utf-8")
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", str(img))]

    remove_selected_cross_overlaps(selected, "Truck", [str(img)])

    out = json.loads((tmp_path / "keep2.json").read_text(encoding="utf-8"))
    assert out["version"] == "5.0.1"
    assert out["flags"] == {"note": "x"}
    assert [s["label"] for s in out["shapes"]] == ["ZTruck"]


def test_remove_selected_memory_wins(tmp_path):
    """已加载图以内存框为准：磁盘索引与内存不一致时按内容签名删。"""
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = _write_image_and_json(tmp_path, "mem2", [
        {"label": "ZTruck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "Truck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    memory = {0: [
        {"label": "Truck", "x": 2, "y": 2, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 0, "y": 0, "width": 100, "height": 100},
    ]}
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", img)]

    result = remove_selected_cross_overlaps(
        selected, "Truck", [img], memory_boxes=memory)

    assert result["removed"] == 1
    assert result["removed_boxes"][0][0]["label"] == "Truck"
    payload = json.loads((tmp_path / "mem2.json").read_text(encoding="utf-8"))
    assert [s["label"] for s in payload["shapes"]] == ["ZTruck"]


def test_remove_selected_reports_refreshed_issues(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = _write_image_and_json(tmp_path, "refresh2", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", img)]

    result = remove_selected_cross_overlaps(selected, "Truck", [img])

    refreshed = result["refreshed_issues"][0]
    assert not [i for i in refreshed if i["kind"] == "cross_label_overlap"]
    assert all(i["image_index"] == 0 for i in refreshed)


def test_remove_selected_refreshed_reports_remaining_issues(tmp_path):
    """删除后刷新结果要反映剩余框的真实问题（越界）。"""
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = _write_image_and_json(tmp_path, "empty2", [
        {"label": "Truck", "x": 0, "y": 0, "width": 300, "height": 300},
        {"label": "ZTruck", "x": -5, "y": -5, "width": 300, "height": 300},
    ])
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", img)]

    result = remove_selected_cross_overlaps(selected, "Truck", [img])

    refreshed = result["refreshed_issues"][0]
    kinds = {i["kind"] for i in refreshed}
    assert "out_of_bounds" in kinds
    assert "cross_label_overlap" not in kinds


def test_remove_selected_ignores_non_cross_issues(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = _write_image_and_json(tmp_path, "nocross", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ])
    selected = [{"kind": "tiny_box", "image_index": 0, "image_path": img,
                 "label": "Truck", "box_index": 0, "detail": "W:1 H:1"}]

    result = remove_selected_cross_overlaps(selected, "Truck", [img])

    assert result["removed"] == 0
    payload = json.loads((tmp_path / "nocross.json").read_text(encoding="utf-8"))
    assert len(payload["shapes"]) == 1


def test_remove_selected_missing_sidecar_recorded_as_failed(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    img = tmp_path / "lonely2.png"
    img.write_bytes(b"x")
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", str(img))]

    result = remove_selected_cross_overlaps(selected, "Truck", [str(img)])

    assert result["removed"] == 0
    assert result["failed"] == [str(img)]


def test_remove_selected_interruptible(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    paths = [_write_image_and_json(tmp_path, f"si{i}", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ]) for i in range(5)]
    selected = [
        _cross_issue(i, 0, "Truck", 1, "ZTruck", p) for i, p in enumerate(paths)]

    result = remove_selected_cross_overlaps(
        selected, "Truck", paths, is_interrupted=lambda: True)

    assert result["interrupted"] is True
    assert result["removed"] == 0


def test_remove_selected_progress_callback(tmp_path):
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps
    paths = [_write_image_and_json(tmp_path, f"sp{i}", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ]) for i in range(3)]
    selected = [
        _cross_issue(i, 0, "Truck", 1, "ZTruck", p) for i, p in enumerate(paths)]
    seen = []

    remove_selected_cross_overlaps(
        selected, "Truck", paths,
        progress_cb=lambda done, total: seen.append((done, total)))

    assert seen[-1] == (3, 3)


def test_cross_delete_worker_contract():
    import inspect
    from labelforge.engine import quality_lint
    assert hasattr(quality_lint, "CrossLabelDeleteWorker")
    source = inspect.getsource(quality_lint.CrossLabelDeleteWorker.run)
    assert "remove_selected_cross_overlaps" in source
    assert "delete_finished" in source
    init = inspect.getsource(quality_lint.CrossLabelDeleteWorker.__init__)
    assert "target_label" in init


def test_box_signature_quantizes_rect():
    from labelforge.engine.quality_lint import box_signature
    a = {"label": "Truck", "x": 1.0001, "y": 2, "width": 3, "height": 4}
    b = {"label": "Truck", "x": 1.0002, "y": 2, "width": 3, "height": 4}
    assert box_signature(a) == box_signature(b)
    assert box_signature({"label": "Truck"}) is None


# --------------------------------------------------------------------------
# 单图实时质检
# --------------------------------------------------------------------------
def test_lint_single_image_uses_memory_even_when_empty(tmp_path):
    """已加载图删空后不报旧问题，也不报空标注。"""
    from labelforge.engine.quality_lint import lint_single_image
    img = _write_image_and_json(tmp_path, "live", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    issues = lint_single_image(0, img, {0: []})
    assert issues == []


def test_lint_single_image_clears_fixed_problem(tmp_path):
    """把重叠框修好后，同一张图不再报异类重叠。"""
    from labelforge.engine.quality_lint import lint_single_image
    img = _write_image_and_json(tmp_path, "fixed", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    before = lint_single_image(0, img)
    assert any(i["kind"] == "cross_label_overlap" for i in before)

    memory = {0: [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 500, "y": 500, "width": 100, "height": 100},
    ]}
    after = lint_single_image(0, img, memory)
    assert not any(i["kind"] == "cross_label_overlap" for i in after)


def test_lint_single_image_missing_sidecar(tmp_path):
    """缺 sidecar 不再上报。"""
    from labelforge.engine.quality_lint import lint_single_image
    img = tmp_path / "noside.png"
    img.write_bytes(b"x")
    issues = lint_single_image(0, str(img))
    assert issues == []


def test_lint_single_image_sets_index_and_path(tmp_path):
    from labelforge.engine.quality_lint import lint_single_image
    img = _write_image_and_json(tmp_path, "meta", [
        {"label": "cat", "x": -5, "y": 0, "width": 10, "height": 10},
    ])
    issues = lint_single_image(3, img)
    assert issues and issues[0]["image_index"] == 3
    assert issues[0]["image_path"] == img


def test_lint_image_shapes_skips_pairwise_over_limit():
    from labelforge.engine.quality_lint import lint_image_shapes
    limit = QUALITY_LINT_CONFIG["max_boxes_pairwise"]
    shapes = [
        {"label": "cat", "x": 0, "y": 0, "width": 10, "height": 10}
        for _ in range(limit + 1)
    ]
    # 图取 500x500：10px 框占 2%，比例判为通过。这里要验的是"超框数就跳过
    # 两两比对"，若沿用早先随手写的 10000x10000，每个框只占 0.1% 会被小目标
    # 检查判成 🔴，501 条噪声会淹掉本测试真正断言的那一条。
    issues = lint_image_shapes("a.png", shapes, 500, 500)
    assert [i["kind"] for i in issues] == ["skipped_pairwise"]


# --------------------------------------------------------------------------
# 实时刷新（UI 契约）
# --------------------------------------------------------------------------
def test_mixin_exposes_live_refresh_helpers():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    for name in ("_notify_lint_boxes_changed", "_flush_lint_refresh"):
        assert hasattr(QualityLintMixin, name), name


def test_live_refresh_is_debounced_and_only_when_dialog_visible():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._notify_lint_boxes_changed)
    assert "isVisible" in src
    assert "QTimer" in src
    assert "setSingleShot" in src


def test_live_refresh_uses_single_image_lint_with_memory():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._flush_lint_refresh)
    assert "lint_single_image" in src
    assert "detection_boxes" in src
    assert "filter_ignored_issues" in src


def test_save_json_triggers_live_refresh():
    import inspect
    from labelforge.ui.main_window import ImageEditor
    src = inspect.getsource(ImageEditor.save_json)
    assert "_notify_lint_boxes_changed" in src


def test_live_refresh_flush_updates_dialog(tmp_path):
    """契约：刷新后弹窗该图旧问题被替换为新结果。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    calls = []

    class Dialog:
        def isVisible(self):
            return True

        def refresh_issues_for_images(self, indexes, issues):
            calls.append((set(indexes), list(issues)))

    img = _write_image_and_json(tmp_path, "liveui", [])
    editor = type("Editor", (QualityLintMixin,), {})()
    editor._lint_dialog = Dialog()
    editor.background_images = [img]
    editor.detection_boxes_dict = {0: []}
    editor.current_background_index = 0
    editor.detection_boxes = []

    editor._lint_refresh_pending = {0}
    editor._flush_lint_refresh()

    assert calls and calls[0][0] == {0}
    assert calls[0][1] == []


def test_live_refresh_uses_current_canvas_boxes(tmp_path):
    """当前图以画布内存框为准（detection_boxes 可能比 dict 更新）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    calls = []

    class Dialog:
        def isVisible(self):
            return True

        def refresh_issues_for_images(self, indexes, issues):
            calls.append(list(issues))

    img = _write_image_and_json(tmp_path, "canvaslive", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    editor = type("Editor", (QualityLintMixin,), {})()
    editor._lint_dialog = Dialog()
    editor.background_images = [img]
    editor.detection_boxes_dict = {0: []}
    editor.current_background_index = 0
    editor.detection_boxes = [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 500, "y": 500, "width": 100, "height": 100},
    ]
    editor._lint_refresh_pending = {0}

    editor._flush_lint_refresh()

    kinds = {i["kind"] for i in calls[0]}
    assert "cross_label_overlap" not in kinds


# --------------------------------------------------------------------------
# 表格：列 / 排序 / 角标 / 筛选宽度
# --------------------------------------------------------------------------
def test_dialog_table_has_no_box_column_and_shows_two_labels():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog.__init__)
    # 仍是"不要 Box 列"：四列是 框类型/类别/背景图/尺寸提示，没有坐标列。
    # 第 4 列是小目标审核的提示文案落点（需求要求审核界面直接显示提示）。
    assert "QTableWidget(0, 4)" in src
    assert '"Box"' not in src
    assert "尺寸/提示" in src
    label_src = inspect.getsource(QualityLintDialog._issue_label_text)
    assert "label_a" in label_src and "label_b" in label_src


def test_dialog_sorts_issues_by_image_then_kind_then_label():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog._sort_issues)
    assert "_image_order" in src
    assert "_kind_order" in src
    assert "casefold" in src
    assert "label_a" in src
    image_src = inspect.getsource(QualityLintDialog._image_order)
    assert "image_index" in image_src


def test_sort_groups_same_image_together():
    """同一张图的多个问题必须相邻（用户要求）。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    dialog = QualityLintDialog.__new__(QualityLintDialog)
    issues = [
        {"kind": "tiny_needs_p2", "label": "Zed", "image_index": 5},
        {"kind": "tiny_needs_p2", "label": "Aaa", "image_index": 1},
        {"kind": "out_of_bounds", "label": "", "image_index": 5},
        {"kind": "cross_label_overlap", "label": "Car",
         "label_a": "Car", "label_b": "SUV", "image_index": 1},
    ]
    ordered = dialog._sort_issues(issues)
    indexes = [i["image_index"] for i in ordered]
    assert indexes == [1, 1, 5, 5], indexes
    # 同图内按类型顺序：越界框（KIND_LABELS 第一）在小目标之前
    assert [i["kind"] for i in ordered if i["image_index"] == 5] == \
        ["out_of_bounds", "tiny_needs_p2"]


def test_dialog_corner_button_themed():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog.__init__)
    assert "QTableCornerButton::section" in src


def test_dialog_filter_combo_widened():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog.__init__)
    assert "filter_combo.setMinimumWidth" in src
    assert "setMinimumWidth(133)" in src


def test_dialog_delete_menu_uses_selection():
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog._on_context_menu)
    assert "delete_requested.emit(items, target)" in src
    assert "_cross_labels_in" in src


def test_dialog_cross_labels_in_dedupes_order():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    dialog = QualityLintDialog.__new__(QualityLintDialog)
    labels = dialog._cross_labels_in([
        _cross_issue(0, 0, "Truck", 1, "ZTruck"),
        _cross_issue(0, 0, "Truck", 2, "Car"),
    ])
    assert labels == ["Truck", "ZTruck", "Car"]


def test_mixin_delete_uses_selection_and_target():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._delete_cross_label_overlaps)
    assert "selected" in src
    assert "不可撤销" in src
    assert "IoU" not in src


def test_delete_worker_gets_only_related_memory_boxes():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._start_cross_label_delete)
    assert "image_index" in src
    assert "CrossLabelDeleteWorker" in src


def test_i18n_delete_terms():
    from labelforge.ui.i18n import _strings
    for key in ("删除 {n} 项中的 '{label}' 框", "删除进度", "正在删除",
                "删除已中断", "已删除", "个重叠框", "张图", "张写入失败",
                "确认删除"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_i18n_lint_button_refreshed_on_language_switch():
    import inspect
    from labelforge.ui.mixins.translation import TranslationMixin
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "lint_btn" in src
    assert 'tr("质检")' in src


# --------------------------------------------------------------------------
# 修完当前图问题后自动跳到下一张问题图
# --------------------------------------------------------------------------
class _AdvanceEditor:
    """把 mixin 方法绑到一个带状态的对象上，模拟主窗口。"""

    def __init__(self, current, image_count=3):
        from labelforge.ui.mixins.quality_lint import QualityLintMixin
        self.background_images = [f"img{i}.png" for i in range(image_count)]
        self.current_background_index = current
        self.switched = []
        self.status_text = None
        self._busy = False
        self._lint_dialog = None
        self.status_label = self

    def setText(self, text):
        self.status_text = text

    def switch_background_to_index(self, idx):
        self.switched.append(idx)
        self.current_background_index = idx

    def maybe_advance(self, issues, refreshed=None):
        from labelforge.ui.mixins.quality_lint import QualityLintMixin

        class Dialog:
            _all_issues = issues

            def isVisible(self):
                return True

        self._lint_dialog = Dialog()
        return QualityLintMixin._maybe_advance_lint_issue(self, Dialog(), refreshed)


def test_advance_jumps_to_next_image_with_issues():
    editor = _AdvanceEditor(current=0)
    moved = editor.maybe_advance([
        {"image_index": 2, "kind": "tiny_box"},
        {"image_index": 1, "kind": "out_of_bounds"},
    ])
    assert moved is True
    assert editor.switched == [1]
    assert editor.status_text == "已跳到下一张问题图"


def test_advance_skips_cleared_current_image():
    """当前图还有问题时不跳。"""
    editor = _AdvanceEditor(current=0)
    moved = editor.maybe_advance([
        {"image_index": 0, "kind": "tiny_box"},
        {"image_index": 1, "kind": "out_of_bounds"},
    ])
    assert moved is False
    assert editor.switched == []


def test_advance_wraps_to_first_issue_image():
    """后面没有问题时回绕到最前面的问题图。"""
    editor = _AdvanceEditor(current=2)
    moved = editor.maybe_advance([
        {"image_index": 0, "kind": "tiny_box"},
        {"image_index": 1, "kind": "out_of_bounds"},
    ])
    assert moved is True
    assert editor.switched == [0]


def test_advance_ignores_when_no_issues_left():
    editor = _AdvanceEditor(current=0)
    assert editor.maybe_advance([]) is False
    assert editor.switched == []


def test_advance_jumps_when_clearing_other_image_issues():
    """清理『其他图』的问题后，当前图已无问题也要跳过去（用户实测回归）。"""
    editor = _AdvanceEditor(current=0)
    moved = editor.maybe_advance(
        [{"image_index": 2, "kind": "cross_label_overlap"}])
    assert moved is True
    assert editor.switched == [2]


def test_advance_uses_visible_filtered_issues():
    """筛选某类问题时，只看表格可见问题：当前图该类清完就跳。

    全量里当前图还有其它类问题，也不能卡住跳转（用户实测回归）。
    """
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Dialog:
        _all_issues = [
            {"image_index": 0, "kind": "tiny_box"},
            {"image_index": 2, "kind": "cross_label_overlap"},
        ]

        def isVisible(self):
            return True

        def visible_issues(self):
            return [i for i in self._all_issues
                    if i.get("kind") == "cross_label_overlap"]

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png", "b.png", "c.png"]
    editor.current_background_index = 0
    editor.switched = []
    editor.switch_background_to_index = lambda idx: editor.switched.append(idx)
    editor._busy = False

    moved = QualityLintMixin._maybe_advance_lint_issue(editor, Dialog())

    assert moved is True
    assert editor.switched == [2]


def test_advance_focuses_dialog_row():
    """跳转后表格定位到目标图的第一条问题行。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    calls = []

    class Dialog:
        _all_issues = [{"image_index": 2, "kind": "tiny_box"}]

        def isVisible(self):
            return True

        def visible_issues(self):
            return list(self._all_issues)

        def focus_image(self, index):
            calls.append(index)
            return True

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png", "b.png", "c.png"]
    editor.current_background_index = 0
    editor.switch_background_to_index = lambda idx: None
    editor._busy = False

    assert QualityLintMixin._maybe_advance_lint_issue(editor, Dialog()) is True
    assert calls == [2]


def test_filter_preserved_after_rebuild():
    """删除后重建筛选器不能把用户的筛选重置回『全部』。"""
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog._rebuild_filter_combo)
    assert "previous" in src
    assert "itemData" in src


def test_advance_respects_refreshed_indexes():
    """只刷新了别的图（当前图未变）时不跳。"""
    editor = _AdvanceEditor(current=0)
    moved = editor.maybe_advance(
        [{"image_index": 1, "kind": "tiny_box"}], refreshed={1})
    assert moved is False
    assert editor.switched == []


def test_advance_skips_while_busy():
    editor = _AdvanceEditor(current=0)
    editor._busy = True
    moved = editor.maybe_advance([{"image_index": 1, "kind": "tiny_box"}])
    assert moved is False
    assert editor.switched == []


def test_advance_triggered_by_flush_lint_refresh(tmp_path):
    """修好当前图后 _flush_lint_refresh 自动跳下一张问题图。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    img0 = _write_image_and_json(tmp_path, "adv0", [])
    img1 = _write_image_and_json(tmp_path, "adv1", [])

    class Dialog:
        _all_issues = [{"image_index": 1, "kind": "tiny_box"}]

        def isVisible(self):
            return True

        def refresh_issues_for_images(self, indexes, issues):
            pass

    editor = type("Editor", (QualityLintMixin,), {})()
    editor._lint_dialog = Dialog()
    editor.background_images = [img0, img1]
    editor.detection_boxes_dict = {0: []}
    editor.current_background_index = 0
    editor.detection_boxes = []
    editor.switched = []
    editor.switch_background_to_index = lambda idx: editor.switched.append(idx)
    editor._lint_refresh_pending = {0}

    editor._flush_lint_refresh()

    assert editor.switched == [1]


def test_ignore_triggers_advance():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    for name in ("_ignore_lint_issue", "_ignore_lint_issues"):
        src = inspect.getsource(getattr(QualityLintMixin, name))
        assert "_maybe_advance_lint_issue" in src, name


def test_delete_finished_triggers_advance():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._on_cross_label_delete_finished)
    assert "_maybe_advance_lint_issue" in src
    assert "set_status=False" in src
    assert "jumped" in src
    # 清理其他图的问题时也要跳：不能按 refreshed.keys() 限制
    assert "refreshed.keys(), set_status" not in src


def test_ignore_triggers_advance_without_index_guard():
    """忽略其他图的问题后同样要跳（不能按被忽略问题的图限制）。"""
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    for name in ("_ignore_lint_issue", "_ignore_lint_issues"):
        src = inspect.getsource(getattr(QualityLintMixin, name))
        assert "_maybe_advance_lint_issue(dialog, set_status=False)" in src, name
        assert "已跳到下一张问题图" in src, name


def test_i18n_advance_status():
    from labelforge.ui.i18n import _strings
    assert "已跳到下一张问题图" in _strings["zh"]
    assert "已跳到下一张问题图" in _strings["en"]


# --------------------------------------------------------------------------
# 设置对话框：质检忽略规则管理
# --------------------------------------------------------------------------
def test_settings_dialog_manages_lint_ignored_rules():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "labelforge" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")
    assert "self.lint_rules_list = QListWidget()" in src
    assert "self.lint_remove_btn.clicked.connect(self._remove_selected_lint_rules)" in src
    assert "self.lint_clear_btn.clicked.connect(self._clear_lint_rules)" in src
    assert "lint_ignored_rules=self._lint_ignored_rules" in src
    # 保存后主窗口缓存同步，质检弹窗立即用新规则
    assert "self._editor._lint_ignored_rules" in src


def test_i18n_lint_ignore_settings_terms():
    from labelforge.ui.i18n import _strings
    for key in ("质检忽略", "质检忽略规则", "移除选中", "清空全部", "整类"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_settings_dialog_lint_ignored_rules_end_to_end(tmp_path):
    """真实 Qt（offscreen）：查看 / 移除 / 清空 / 保存忽略规则。"""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "config.json"
    script = '''
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from labelforge.core import config_manager
from labelforge.ui.settings_dialog import SettingsDialog

app = QApplication.instance() or QApplication([])
config_manager.save_all(lint_ignored_rules={"tiny_box": ["*"], "out_of_bounds": ["(1,2)"]})

dialog = SettingsDialog(None)
assert dialog.lint_rules_list.count() == 2, dialog.lint_rules_list.count()
data = [dialog.lint_rules_list.item(i).data(Qt.UserRole)
        for i in range(dialog.lint_rules_list.count())]
assert ("tiny_box", "*") in data, data
assert ("out_of_bounds", "(1,2)") in data, data

dialog.lint_rules_list.item(0).setSelected(True)
dialog._remove_selected_lint_rules()
assert dialog.lint_rules_list.count() == 1, dialog.lint_rules_list.count()

dialog._clear_lint_rules()
assert dialog.lint_rules_list.count() == 0

dialog._save_shortcuts()
rules = config_manager.load_all().get("lint_ignored_rules")
assert rules == {}, rules
print("OK")

# Qt 子进程退出阶段析构顺序不定，偶发崩溃/挂死；断言通过后 flush + os._exit(0)
# 绕开析构（os._exit 不冲刷缓冲，必须先 flush，否则标记丢失会误报失败）。
import os, sys
sys.stdout.flush()
os._exit(0)
'''
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(root),
        "LABELFORGE_CONFIG_PATH": str(config_path),
    }
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr


# --------------------------------------------------------------------------
# 审计修复回归：汇总计数重建 / 签名失配不静默成功 / 实时刷新挂钩 / 关窗清理
# --------------------------------------------------------------------------
def test_filtered_lint_result_rebuilds_summary_after_ignore_all():
    """整类忽略后汇总计数必须重建，不能残留被忽略类的旧计数（回归 §2-2）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    editor = type("Editor", (QualityLintMixin,), {})()
    result = {
        "issues": [
            {"kind": "tiny_box", "label": "a", "detail": "1"},
            {"kind": "tiny_box", "label": "b", "detail": "2"},
            {"kind": "out_of_bounds", "label": "c", "detail": "3"},
        ],
        "summary": {"scanned_images": 7, "tiny_box": 2, "out_of_bounds": 1},
    }
    filtered = editor._filtered_lint_result(result, {"tiny_box": ["*"]})

    assert [i["kind"] for i in filtered["issues"]] == ["out_of_bounds"]
    assert filtered["summary"]["tiny_box"] == 0
    assert filtered["summary"]["out_of_bounds"] == 1
    assert filtered["summary"]["scanned_images"] == 7


def test_remove_shapes_matching_signature_mismatch_reports_failure(tmp_path):
    """内存框与磁盘签名失配时必须报失败，不能静默成功（回归 §2-5）。"""
    from labelforge.engine.quality_lint import _remove_shapes_matching

    img = _write_image_and_json(tmp_path, "sigmiss", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
    ])
    json_path = str(tmp_path / "sigmiss.json")
    memory_box = {"label": "Truck", "x": 50, "y": 50, "width": 100, "height": 100}

    removed, ok = _remove_shapes_matching(json_path, [memory_box])

    assert removed == 0
    assert ok is False
    payload = json.loads((tmp_path / "sigmiss.json").read_text(encoding="utf-8"))
    assert len(payload["shapes"]) == 1


def test_remove_selected_signature_mismatch_marks_failed(tmp_path):
    """端到端：签名失配的已加载图进入 failed，且内存同步兜底不删磁盘。"""
    from labelforge.engine.quality_lint import remove_selected_cross_overlaps

    img = _write_image_and_json(tmp_path, "sigmiss2", [
        {"label": "Truck", "x": 0, "y": 0, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ])
    memory = {0: [
        {"label": "Truck", "x": 50, "y": 50, "width": 100, "height": 100},
        {"label": "ZTruck", "x": 2, "y": 2, "width": 100, "height": 100},
    ]}
    selected = [_cross_issue(0, 0, "Truck", 1, "ZTruck", img)]

    result = remove_selected_cross_overlaps(
        selected, "Truck", [img], memory_boxes=memory)

    assert result["failed"] == [img]
    assert result["removed"] == 0
    payload = json.loads((tmp_path / "sigmiss2.json").read_text(encoding="utf-8"))
    assert [s["label"] for s in payload["shapes"]] == ["Truck", "ZTruck"]


def test_close_lint_dialog_interrupts_and_clears():
    """关窗/切数据集：中断扫描并清空引用，旧结果不再跳错图（回归 §2-9）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Worker:
        def __init__(self):
            self.interrupted = False

        def isRunning(self):
            return True

        def requestInterruption(self):
            self.interrupted = True

    class Dialog:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    editor = type("Editor", (QualityLintMixin,), {})()
    worker, dialog = Worker(), Dialog()
    editor._lint_worker = worker
    editor._lint_dialog = dialog

    editor._close_lint_dialog()

    assert worker.interrupted is True
    assert dialog.closed is True
    assert editor._lint_worker is None
    assert editor._lint_dialog is None


def test_dialog_summary_shows_scanned_image_count():
    """汇总行需展示扫描图片数（回归 §2-3）。"""
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog._update_summary)
    assert "_scanned_images" in src
    assert "扫描图片" in src


def test_label_manager_notifies_lint_on_rename_and_delete():
    """标签重命名/删除后必须触发质检实时刷新挂钩（回归 §2-10）。"""
    import inspect
    from labelforge.engine import label_manager
    src = inspect.getsource(label_manager.LabelManager)
    assert src.count("_notify_lint_boxes_changed") >= 5


# --------------------------------------------------------------------------
# 审计修复回归：删除 worker 并发写盘 / 重复弹窗 / 刷新积压 / 进度上限 / i18n
# --------------------------------------------------------------------------
def _save_editor(busy=False):
    editor = type("Editor", (), {})()
    editor._is_delete_view = False
    editor._lint_removal_busy = busy
    editor._dataset_stats_dirty = False
    editor.canvas_items = []
    editor.current_background = None
    editor.current_background_index = 0
    editor.detection_boxes_dict = {0: []}
    editor.detection_boxes = []
    return editor


def test_save_json_rejected_while_lint_removal_busy(tmp_path):
    """删除 worker 改写 sidecar 期间主线程保存必须被拒绝（防并发截断）。"""
    from labelforge.engine.save_manager import SaveManager

    img = tmp_path / "busy.png"
    img.write_bytes(b"x")
    manager = SaveManager(_save_editor(busy=True))

    manager.save_json(str(img), "busy.png", "", image_width=10, image_height=10,
                      current_index=0)

    assert not (tmp_path / "busy.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_save_json_atomic_replace_leaves_no_tmp(tmp_path):
    """sidecar 写入先 tmp 后 os.replace：成功后无 .tmp 残留且 JSON 完整。"""
    from labelforge.engine.save_manager import SaveManager

    img = tmp_path / "atomic.png"
    img.write_bytes(b"x")
    manager = SaveManager(_save_editor())

    manager.save_json(str(img), "atomic.png", "", image_width=10, image_height=10,
                      current_index=0)

    assert (tmp_path / "atomic.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
    json.loads((tmp_path / "atomic.json").read_text(encoding="utf-8"))


def test_delete_worker_sets_lint_removal_busy_and_image_progress_max(monkeypatch):
    """启动删除 worker：置忙标记；进度条上限按涉及图片数而非问题条数。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    from labelforge.ui import dialogs as dialogs_mod
    from labelforge.engine import quality_lint as ql

    captured = {}

    class Progress:
        canceled = type("S", (), {"connect": lambda self, fn: None})()

        def show(self):
            pass

        def setValue(self, v):
            pass

        def setLabelText(self, text):
            pass

    def fake_create(parent, title, label_text, maximum):
        captured["max"] = maximum
        return Progress()

    class Signal:
        def connect(self, fn):
            pass

    class FakeWorker:
        def __init__(self, *a, **kw):
            self.delete_progress = Signal()
            self.delete_finished = Signal()

        def start(self):
            captured["started"] = True

        def requestInterruption(self):
            pass

        def isRunning(self):
            return False

    monkeypatch.setattr(dialogs_mod.ProgressDialogFactory,
                        "create_progress_dialog", staticmethod(fake_create))
    monkeypatch.setattr(ql, "CrossLabelDeleteWorker", FakeWorker)

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png", "b.png"]
    editor.detection_boxes_dict = {}
    editor.detection_boxes = []
    editor.current_background_index = -1
    editor.status_label = type("L", (), {"setText": lambda self, text: None})()
    selected = [
        _cross_issue(0, 0, "Truck", 1, "ZTruck", "a.png"),
        _cross_issue(0, 2, "Truck", 3, "ZTruck", "a.png"),
        _cross_issue(1, 0, "Car", 1, "SUV", "b.png"),
    ]

    editor._start_cross_label_delete(None, selected, "Truck", ["a.png", "b.png"])

    assert captured["started"] is True
    assert captured["max"] == 2
    assert editor._lint_removal_busy is True


def test_delete_finished_clears_lint_removal_busy():
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    class Progress:
        def close(self):
            pass

    worker = object()
    editor = type("Editor", (QualityLintMixin,), {})()
    editor._delete_worker = worker
    editor._delete_progress = None
    editor._busy = True
    editor._lint_removal_busy = True
    editor._lint_dialog = object()
    editor.detection_boxes_dict = {}
    editor.detection_boxes = []
    editor.current_background_index = -1
    editor.status_label = type("L", (), {"setText": lambda self, text: None})()

    editor._on_cross_label_delete_finished(
        None, Progress(), worker, "Truck",
        {"removed": 0, "images_changed": 0, "failed": []})

    assert editor._lint_removal_busy is False
    assert editor._busy is False


def test_jump_blocked_while_lint_removal_busy():
    """删除 worker 运行中双击问题行不得切图（切图会触发 sidecar 保存）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    editor = type("Editor", (QualityLintMixin,), {})()
    editor.background_images = ["a.png", "b.png"]
    editor.current_background_index = 0
    editor._lint_removal_busy = True
    editor.switch_background_to_index = lambda idx: (_ for _ in ()).throw(
        AssertionError("删除进行中不应切图"))

    editor._jump_to_lint_issue({"image_index": 1, "box_index": None})


def test_open_quality_lint_closes_existing_dialog_source():
    """重复点质检必须复用/关闭旧弹窗，不能叠加过期弹窗（回归 C2）。"""
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._open_quality_lint)
    assert "_close_lint_dialog" in src


def test_refresh_overflow_triggers_full_rescan(tmp_path):
    """刷新事件积压超过上限时整库重扫一次，而不是全部丢弃（回归 C3）。"""
    from labelforge.ui.mixins.quality_lint import QualityLintMixin

    calls = []

    class Dialog:
        def isVisible(self):
            return True

        def refresh_issues_for_images(self, indexes, issues):
            calls.append(set(indexes))

    class Timer:
        def start(self):
            pass

    img0 = _write_image_and_json(tmp_path, "full0", [])
    img1 = _write_image_and_json(tmp_path, "full1", [])
    editor = type("Editor", (QualityLintMixin,), {})()
    editor._lint_dialog = Dialog()
    editor._lint_refresh_timer = Timer()
    editor.background_images = [img0, img1]
    editor.detection_boxes_dict = {0: [], 1: []}
    editor.current_background_index = 0
    editor.detection_boxes = []

    for i in range(66):
        editor._notify_lint_boxes_changed(i)

    assert editor._lint_refresh_full is True
    editor._flush_lint_refresh()

    assert calls and calls[0] == {0, 1}


def test_i18n_scanned_images_key_has_en_translation():
    from labelforge.ui.i18n import _strings
    assert _strings["zh"]["扫描图片"] == "扫描图片"
    assert _strings["en"]["扫描图片"] == "Scanned images"


# ---- 小目标可检测性审核：等比缩放到 640 后的像素判据 ----
# 短边比例 = min(w, h) / max(W, H)：这是「原图与框等比缩放（长边→640、短边补边）
# 到模型输入后，框短边占 640 的比例」，乘 640 就是模型输入里的像素数。
# 阈值 4px / 8px / 16px @640。分母取长边而不是各轴各比，因为预处理是等比的。

def test_tensor_side_pixels_is_ratio_times_640():
    """核心口径：比例 × 640 = 框在模型输入里的短边像素。"""
    from labelforge.engine.quality_lint import normalized_min_side, TINY_INPUT_SIZE
    assert TINY_INPUT_SIZE == 640
    # 704x576 图上的 10x8 框：等比缩放到 640 后短边 640->524 那一档
    box = _box(x=0, y=0, w=10, h=8)
    ratio = normalized_min_side(box, 704, 576)
    assert ratio == 8 / 704              # 分母是长边 704，不是高 576
    assert abs(ratio * 640 - 7.27) < 0.01


def test_tiny_unreachable_below_fail_ratio():
    """3px @640 = 0.47% < 0.625%：常规配置根本检不到。"""
    assert {i["kind"] for i in lint_shapes([_box(w=3, h=3)], 640, 640)} == {
        "tiny_unreachable"}


def test_tiny_needs_p2_between_fail_and_warn():
    """6px @640 = 0.94%，落在 [0.625%, 1.25%)：需 P2 头或滑窗放大。"""
    assert {i["kind"] for i in lint_shapes([_box(w=6, h=6)], 640, 640)} == {
        "tiny_needs_p2"}


def test_fail_ratio_boundary_belongs_to_warn_tier():
    """正好 4px @640 = 0.625% 是 🟡 不是 🔴（下界取闭区间）。"""
    assert {i["kind"] for i in lint_shapes([_box(w=4, h=4)], 640, 640)} == {
        "tiny_needs_p2"}


def test_warn_ratio_boundary_passes_clean():
    """正好 8px @640 = 1.25% 通过。"""
    assert lint_shapes([_box(w=8, h=8)], 640, 640) == []


def test_square_image_matches_intuitive_pixel_reading():
    """方图是最容易心算的情形：480x480 上 6px 框缩放到 640 后正好 8px。"""
    assert lint_shapes([_box(w=6, h=6)], 480, 480) == []
    # 480 上 4px -> 640 上 5.33px，落 🟡
    assert {i["kind"] for i in lint_shapes([_box(w=4, h=4)], 480, 480)} == {
        "tiny_needs_p2"}
    # 480 上 3px -> 640 上 4px，正好踩 🟡 下界
    assert {i["kind"] for i in lint_shapes([_box(w=3, h=3)], 480, 480)} == {
        "tiny_needs_p2"}
    # 480 上 2px -> 640 上 2.67px，落 🔴
    assert {i["kind"] for i in lint_shapes([_box(w=2, h=2)], 480, 480)} == {
        "tiny_unreachable"}


def test_bigger_image_makes_same_box_worse():
    """同一像素尺寸的框，图越大缩放到 640 后越小 —— 结论必须跟着变。

    这正是旧口径（各轴比自己的边）的错误来源：1920x1080 上的 100x100 框
    各轴 5.2%/9.3% 都"健康"，但等比缩放到 640 后只剩 33px。
    """
    # 1920x1080 上 8x8：缩放到 640 后只有 2.67px -> 🔴
    assert {i["kind"] for i in lint_shapes([_box(w=8, h=8)], 1920, 1080)} == {
        "tiny_unreachable"}
    # 同一张图上的 24x24 -> 8px @640 -> 通过
    assert lint_shapes([_box(w=24, h=24)], 1920, 1080) == []
    # 320x320 上 4x4：缩放到 640 是放大到 8px -> 通过（旧口径会误报）
    assert lint_shapes([_box(w=4, h=4)], 320, 320) == []


def test_worse_axis_decides_ratio():
    """1000x5 的电线：等比缩放后厚度只剩 1.6px @640 -> 🔴。

    这也是不能用 sqrt(w*h) 定级的原因：等效面积 70px 看着正常，实际厚度 5px。
    """
    issues = lint_shapes([_box(x=0, y=0, w=1000, h=5)], 2000, 1000)
    assert {i["kind"] for i in issues} == {"tiny_unreachable"}
    # 5/2000 = 0.25% -> 1.6px @640
    assert issues[0]["ratio"] == 5 / 2000


def test_short_side_of_image_is_not_the_denominator():
    """分母必须是长边：竖图 576x704 与横图 704x576 上同一个框结论相同。

    等比缩放只看长边，取短边当分母会把竖图结果算错 22%。
    """
    for W, H in [(704, 576), (576, 704)]:
        assert {i["kind"] for i in
                lint_shapes([_box(x=0, y=0, w=10, h=8)], W, H)} == {
            "tiny_needs_p2"}, (W, H)


def test_no_tiny_box_kind_anymore():
    """「超小框」这一档已删除：它的语义被 640 空间的 4/8/16px 三档完全覆盖。

    没有 imageWidth/Height 时比例无从算起，此时也**不报**（旧口径会退回
    原图像素的 tiny_box，等于又冒出第二套口径）。
    """
    from labelforge.engine.quality_lint import KIND_LABELS
    assert "tiny_box" not in KIND_LABELS
    assert lint_shapes([_box(w=3, h=3)], 0, 0) == []
    assert lint_shapes([_box(w=5, h=50)], 200, 200) == []


def test_keypoint_never_reported_as_tiny_target():
    """关键点宽高恒为 0，比例算出来是 0，不跳过会把每个关键点判成 🔴。

    只断言不参与比例判档：孤立关键点本来就会被既有的 group_mismatch
    报「无同组框」，那是另一码事，不该在这里被掩盖或误伤。
    """
    kp = {"label": "nose", "shape_type": "point", "points": [[5, 5]],
          "x": 5, "y": 5, "width": 0, "height": 0}
    kinds = {i["kind"] for i in lint_shapes([kp], 640, 640)}
    assert "tiny_unreachable" not in kinds
    assert "tiny_needs_p2" not in kinds
    assert "tiny_box" not in kinds


def test_polygon_skipped_by_ratio_check():
    """只看 box：多边形的 bbox 不代表目标本身，不参与比例判档。"""
    poly = {"label": "road", "shape_type": "polygon",
            "points": [[0, 0], [3, 0], [3, 3]],
            "x": 0, "y": 0, "width": 3, "height": 3}
    kinds = {i["kind"] for i in lint_shapes([poly], 640, 640)}
    assert "tiny_unreachable" not in kinds
    assert "tiny_needs_p2" not in kinds


def test_exempt_flag_silences_both_size_checks():
    """标了「保留」→ 归一化与绝对像素两类尺寸问题都不再报。

    1000x5 的电线同时命中 🔴 和 tiny_box，只豁免其一会让「保留」形同虚设。
    """
    wire = _box(x=0, y=0, w=1000, h=5, flags={"tiny_keep": True})
    assert lint_shapes([wire], 2000, 1000) == []


def test_exempt_flag_does_not_silence_other_kinds():
    """豁免只管尺寸，越界这类真问题仍要报。"""
    box = _box(x=-50, y=0, w=3, h=3, flags={"tiny_keep": True})
    assert {i["kind"] for i in lint_shapes([box], 640, 640)} == {
        "out_of_bounds"}


def test_tiny_issue_carries_ratio_for_ui_column():
    """第 4 列要直接显示百分比，不能让 UI 去反解 detail 字符串。"""
    issues = lint_shapes([_box(w=6, h=6)], 640, 640)
    assert issues[0]["ratio"] == 6 / 640


def test_normalized_min_side_helper_returns_none_when_undefined():
    from labelforge.engine.quality_lint import normalized_min_side
    assert normalized_min_side(_box(x=0, y=0, w=10, h=20), 100, 100) == 0.1
    assert normalized_min_side(
        {"shape_type": "point", "points": [[1, 1]]}, 100, 100) is None
    assert normalized_min_side(_box(w=10, h=20), 0, 100) is None
    assert normalized_min_side(_box(w=10, h=20), 100, 0) is None
    assert normalized_min_side(_box(w=0, h=20), 100, 100) is None
    assert normalized_min_side(_box(w=-4, h=20), 100, 100) is None
    assert normalized_min_side("bogus", 100, 100) is None
    assert normalized_min_side({}, 100, 100) is None
    assert normalized_min_side(None, 100, 100) is None


def test_non_finite_geometry_is_not_judged_as_healthy():
    """损坏 sidecar 里的 NaN/Inf 不能算成"通过"，否则虚增通过率。

    json.loads 默认接受 NaN/Infinity 字面量，而 `nan <= 0` 为 False，
    会绕过退化宽高守卫一路算到比例里。dataset_health 早为同一类问题
    设过 _finite_float 防线，这里必须同样处理。
    """
    from labelforge.engine.quality_lint import normalized_min_side
    nan = float("nan")
    inf = float("inf")
    assert normalized_min_side(_box(w=nan, h=20), 640, 640) is None
    assert normalized_min_side(_box(w=inf, h=20), 640, 640) is None
    assert normalized_min_side(_box(w=20, h=20), nan, 640) is None
    assert normalized_min_side(_box(w=20, h=20), inf, 640) is None


def test_non_finite_width_excluded_from_report(tmp_path):
    """NaN 宽的框既不进三档计数也不进直方图，更不能落进"最大目标"那一桶。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "nan", [
        {"label": "broken", "x": 0, "y": 0, "width": float("nan"),
         "height": 20, "shape_type": "rectangle"},
        {"label": "car", "x": 0, "y": 100, "width": 100, "height": 100,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    report = lint_dataset([img])["tiny_report"]

    assert report["measured"] == 1        # 只有 car 参与统计
    assert report["pass"] == 1
    assert sum(report["histogram"]["counts"]) == 1
    assert report["histogram"]["counts"][-1] == 0   # 没被当成最大那一档


def test_tiny_report_drops_non_finite_ratios():
    """公开函数自己也要设防，不能只依赖调用方给干净数据。"""
    from labelforge.engine.quality_lint import tiny_report
    report = tiny_report([float("nan"), float("inf"), 0.5], 0)
    assert report["measured"] == 1
    assert report["pass"] == 1
    assert report["stable"] == 1
    assert sum(report["histogram"]["counts"]) == 1


# ---- 「标记保留」：写 flags.tiny_keep 到 sidecar 与内存 ----

def _tiny_issue(index, path, box_index, kind="tiny_unreachable"):
    return {"kind": kind, "label": "wire", "box_index": box_index,
            "ratio": 0.001, "detail": "0.10% W:3 H:3",
            "image_index": index, "image_path": path}


def test_mark_tiny_keep_writes_flag_to_sidecar(tmp_path):
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "m", [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
        {"label": "car", "x": 100, "y": 0, "width": 80, "height": 80,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    result = mark_tiny_keep([_tiny_issue(0, img, 0)], [img])

    assert result["marked"] == 1
    assert result["images_changed"] == 1
    assert result["failed"] == []
    data = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["flags"] == {"tiny_keep": True}
    assert "flags" not in data["shapes"][1]      # 没选中的框不能被动到
    assert data["imageWidth"] == 640             # 其它字段原样保留


def test_mark_tiny_keep_preserves_existing_flags(tmp_path):
    """贴图的 flags.paste 不能被覆盖掉，否则贴图标记丢失。"""
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "k", [
        {"label": "logo", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle", "flags": {"paste": True}},
    ], image_size=(640, 640))

    mark_tiny_keep([_tiny_issue(0, img, 0)], [img])

    shape = json.loads(
        (tmp_path / "k.json").read_text(encoding="utf-8"))["shapes"][0]
    assert shape["flags"] == {"paste": True, "tiny_keep": True}


def test_mark_tiny_keep_is_idempotent(tmp_path):
    """重复标记同一框不该报错，也不该重复计数成两次改动。"""
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "i", [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))
    issue = _tiny_issue(0, img, 0)

    mark_tiny_keep([issue], [img])
    second = mark_tiny_keep([issue], [img])

    assert second["failed"] == []
    shape = json.loads(
        (tmp_path / "i.json").read_text(encoding="utf-8"))["shapes"][0]
    assert shape["flags"] == {"tiny_keep": True}


def test_mark_tiny_keep_updates_memory_boxes(tmp_path):
    """已加载图重扫时以内存为准，只写盘的话豁免不会生效。"""
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "mem", [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))
    memory = {0: [{"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
                   "shape_type": "rectangle"}]}

    result = mark_tiny_keep([_tiny_issue(0, img, 0)], [img],
                            memory_boxes=memory)

    assert result["marked"] == 1
    assert memory[0][0]["flags"] == {"tiny_keep": True}


def test_mark_tiny_keep_matches_disk_by_signature_not_index(tmp_path):
    """已加载图的 box_index 指向内存列表，磁盘顺序可能不同 —— 必须按签名匹配。

    内存里目标框排在第 0 位，磁盘上排在第 1 位。若按下标写盘就会把
    tiny_keep 标到旁边那个大框上，豁免错对象。
    """
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "sig", [
        {"label": "car", "x": 100, "y": 0, "width": 80, "height": 80,
         "shape_type": "rectangle"},
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))
    memory = {0: [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
        {"label": "car", "x": 100, "y": 0, "width": 80, "height": 80,
         "shape_type": "rectangle"},
    ]}

    mark_tiny_keep([_tiny_issue(0, img, 0)], [img], memory_boxes=memory)

    shapes = json.loads(
        (tmp_path / "sig.json").read_text(encoding="utf-8"))["shapes"]
    by_label = {s["label"]: s for s in shapes}
    assert by_label["wire"]["flags"] == {"tiny_keep": True}
    assert "flags" not in by_label["car"]


def test_mark_tiny_keep_skips_other_kinds(tmp_path):
    """只有两个小目标 kind 能「保留」，重复框之类没有这个语义。"""
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "o", [
        {"label": "a", "x": 0, "y": 0, "width": 100, "height": 100,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    result = mark_tiny_keep(
        [_tiny_issue(0, img, 0, kind="duplicate"),
         _tiny_issue(0, img, 0, kind="out_of_bounds")], [img])

    assert result["marked"] == 0
    assert result["images_changed"] == 0
    data = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))
    assert "flags" not in data["shapes"][0]


def test_mark_tiny_keep_reports_missing_sidecar(tmp_path):
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = tmp_path / "gone.png"
    img.write_bytes(b"x")

    result = mark_tiny_keep([_tiny_issue(0, str(img), 0)], [str(img)])

    assert result["marked"] == 0
    assert result["failed"] == [str(img)]


def test_mark_tiny_keep_rejects_out_of_range_index(tmp_path):
    """扫描结果与磁盘不一致时不能越界写，也不能静默当成成功。"""
    from labelforge.engine.quality_lint import mark_tiny_keep
    img = _write_image_and_json(tmp_path, "r", [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    result = mark_tiny_keep([_tiny_issue(0, img, 9)], [img])

    assert result["marked"] == 0
    assert result["failed"] == [img]


def test_marked_box_disappears_from_rescan(tmp_path):
    """验收闭环：标记 → 重扫 → 问题消失且计入 exempt。"""
    from labelforge.engine.quality_lint import lint_dataset, mark_tiny_keep
    img = _write_image_and_json(tmp_path, "loop", [
        {"label": "wire", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
        {"label": "car", "x": 100, "y": 0, "width": 100, "height": 100,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    before = lint_dataset([img])
    assert before["summary"]["tiny_unreachable"] == 1

    mark_tiny_keep([dict(before["issues"][0])], [img])
    after = lint_dataset([img])

    assert after["issues"] == []
    assert after["tiny_report"]["exempt"] == 1
    assert after["tiny_report"]["fail"] == 0
    assert after["tiny_report"]["pass"] == 1


# ---- 质检弹窗：第 4 列提示 / 严重度着色 / 标记保留 / 报告区 ----

def test_size_hint_column_shows_ratio_and_advice():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    fail = d._size_hint_text({"kind": "tiny_unreachable", "ratio": 0.0047})
    assert fail.startswith("0.47%")
    assert "滑窗" in fail
    warn = d._size_hint_text({"kind": "tiny_needs_p2", "ratio": 0.0094})
    assert warn.startswith("0.94%")
    # 其它 kind 沿用 detail，第 4 列对它们不是新语义
    assert d._size_hint_text(
        {"kind": "duplicate", "detail": "IoU:0.90 #1"}) == "IoU:0.90 #1"


def test_size_hint_survives_missing_or_bad_ratio():
    """缺 ratio / NaN / None 都不能抛，UI 里一处异常会整表画不出来。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    assert d._size_hint_text({"kind": "tiny_needs_p2"})
    assert d._size_hint_text({"kind": "tiny_needs_p2", "ratio": float("nan")})
    assert d._size_hint_text({"kind": "tiny_needs_p2", "ratio": "bogus"})
    assert d._size_hint_text(None) == ""


def test_size_hint_distinguishes_the_two_tiers():
    """两档建议必须不同，否则标注员看不出该裁剪放大还是只需 P2。

    必须用相同的 ratio：文本前缀是百分比，若两档各取一个真实值，
    百分比本身就不同，建议文案即使写成一样也照样"通过"。
    """
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    ratio = 0.00625
    fail = d._size_hint_text({"kind": "tiny_unreachable", "ratio": ratio})
    warn = d._size_hint_text({"kind": "tiny_needs_p2", "ratio": ratio})
    assert fail != warn
    assert "滑窗" in fail
    assert "P2" in warn


def test_size_hint_names_the_small_side():
    """提示要说明是宽、高还是宽高都不满足 —— 标注员才知道该往哪个方向改。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    wide_only = d._size_hint_text({"kind": "tiny_unreachable", "ratio": 0.0045,
                                   "min_side": "width", "both_below": False})
    assert wide_only.startswith("宽不足 0.45%")
    assert "滑窗" in wide_only
    tall_only = d._size_hint_text({"kind": "tiny_needs_p2", "ratio": 0.008,
                                   "min_side": "height", "both_below": False})
    assert tall_only.startswith("高不足 0.80%")
    both = d._size_hint_text({"kind": "tiny_unreachable", "ratio": 0.0045,
                              "min_side": "width", "both_below": True})
    assert both.startswith("宽高均不足 0.45%")


def test_size_hint_both_below_wins_over_min_side():
    """两侧都低于命中档的线时，"宽高均不足"必须压过 min_side 的单侧说法。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    text = d._size_hint_text({"kind": "tiny_needs_p2", "ratio": 0.008,
                              "min_side": "height", "both_below": True})
    assert text.startswith("宽高均不足")
    assert "高不足" not in text


def test_progress_bar_hidden_after_successful_result():
    """扫描完成后进度条该藏起来，一直钉在 100% 只占地方。"""
    from labelforge.ui.quality_lint_dialog import QualityLintDialog

    class _Bar:
        def __init__(self):
            self.visible = True
            self.max = 100
        def setMaximum(self, m):
            self.max = m
        def setValue(self, v):
            pass
        def maximum(self):
            return self.max
        def setVisible(self, v):
            self.visible = bool(v)

    d = QualityLintDialog.__new__(QualityLintDialog)
    d._all_issues = []
    d._kind_filter = None
    d._scanned_images = 0
    d.status_label = type("L", (), {"setText": lambda self, s: None})()
    d.summary_label = type("L", (), {"setText": lambda self, s: None})()
    d.progress = _Bar()
    d._rebuild_filter_combo = lambda: None
    d._apply_filter = lambda: None
    d._update_summary = lambda summary: None
    d.set_result({"summary": {"scanned_images": 3}, "issues": []})
    assert d.progress.visible is False


def test_progress_bar_hidden_after_failed_result():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    d = QualityLintDialog.__new__(QualityLintDialog)
    d._all_issues = []
    d._kind_filter = None
    d._scanned_images = 0
    d.status_label = type("L", (), {"setText": lambda self, s: None})()
    d.summary_label = type("L", (), {"setText": lambda self, s: None})()
    d.progress = type("B", (), {
        "visible": True,
        "setMaximum": lambda self, m: None,
        "setValue": lambda self, v: None,
        "maximum": lambda self: 100,
        "setVisible": lambda self, v: setattr(self, "visible", bool(v)),
    })()
    d._rebuild_filter_combo = lambda: None
    d._apply_filter = lambda: None
    d.set_result({"error": True})
    assert d.progress.visible is False


def test_progress_bar_stylesheet_is_rounded():
    """质检进度条走 ThemeManager 的统一实现（8px 圆角，chunk 同圆角）。"""
    from labelforge.ui import quality_lint_dialog as mod

    qss = mod._progress_qss()
    assert qss.count("border-radius: 8px") == 2, qss
    assert "QProgressBar::chunk" in qss
    # 与全项目其它进度条是同一份样式
    from labelforge.ui.theme import ThemeManager

    assert qss == ThemeManager.get_progress_style()


def test_context_menu_gives_keep_not_detail_ignore_for_tiny_kinds():
    """新 kind 的 detail 含实时百分比，按 detail 忽略改框即失配。

    所以这两类只给「标记保留」（写 flags，跟框走）与「忽略此类」（整类，
    detail 无关），不给基于 detail 的「忽略此项」。
    """
    import inspect
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    src = inspect.getsource(QualityLintDialog._on_context_menu)
    assert "TINY_KINDS" in src
    assert "mark_keep_requested" in src


def test_dialog_exposes_mark_keep_signal():
    from labelforge.ui.quality_lint_dialog import QualityLintDialog
    assert hasattr(QualityLintDialog, "mark_keep_requested")


def _fake_table_env(monkeypatch):
    """给弹窗注入记录型假表格/假 item。

    conftest 把 PySide6 整个 mock 掉：QTableWidgetItem 记不住 text，
    表格内容只能靠记录型替身断言。弹窗里是模块级引用，
    必须 patch 模块属性才生效。（行不着色后不再需要记录型 QColor。)
    """
    from labelforge.ui import quality_lint_dialog as mod

    class _Item:
        def __init__(self, text=""):
            self.text = str(text)
            self.bg = None
            self.tooltip = None
            self.role_data = None

        def setBackground(self, color):
            self.bg = color

        def setToolTip(self, tip):
            self.tooltip = tip

        def setData(self, role, value):
            self.role_data = value

    class _Table:
        def __init__(self):
            self.cells = {}
            self.rows = 0

        def setRowCount(self, n):
            self.rows = n

        def setItem(self, row, col, item):
            self.cells[(row, col)] = item

        def item(self, row, col):
            return self.cells.get((row, col))

    monkeypatch.setattr(mod, "QTableWidgetItem", _Item)
    dialog = mod.QualityLintDialog.__new__(mod.QualityLintDialog)
    dialog.table = _Table()
    return dialog, dialog.table


def test_dialog_rows_have_no_severity_tint(monkeypatch):
    """表格一律不着色（用户裁决）：审核读第 4 列文字，底色反而与普通问题混淆。"""
    dialog, table = _fake_table_env(monkeypatch)
    issues = [
        {"kind": "tiny_unreachable", "label": "wire", "ratio": 0.0047,
         "image_path": "a.png"},
        {"kind": "tiny_needs_p2", "label": "dot", "ratio": 0.0094,
         "image_path": "b.png"},
        {"kind": "out_of_bounds", "label": "cat", "detail": "(1,1,9,9) vs 5x5",
         "image_path": "c.png"},
    ]
    dialog._issues = issues

    dialog._populate_table()

    assert table.rows == 3
    # 第 4 列确实是提示文案，不是坐标
    assert table.item(0, 3).text.startswith("0.47%")
    assert "滑窗" in table.item(0, 3).text
    assert table.item(1, 3).text.startswith("0.94%")
    # 任何 kind、任何格子都不给底色
    for row in range(3):
        for col in range(4):
            assert table.item(row, col).bg is None, (row, col)
    # 非小目标 kind 的第 4 列沿用 detail
    assert table.item(2, 3).text == "(1,1,9,9) vs 5x5"
    # 提示列有 tooltip，列宽不够时能悬停看全文
    assert table.item(0, 3).tooltip.startswith("0.47%")
    # 每个格子都挂着 issue，双击跳图/右键菜单靠它
    assert table.item(0, 0).role_data is issues[0]


def test_size_hint_handles_issue_without_ratio_gracefully_in_table(monkeypatch):
    """ratio 缺失时第 4 列退化成纯建议文字，不能让整表画不出来。"""
    dialog, table = _fake_table_env(monkeypatch)
    dialog._issues = [{"kind": "tiny_unreachable", "label": "wire",
                       "image_path": "a.png"}]

    dialog._populate_table()

    assert "滑窗" in table.item(0, 3).text
    assert "%" not in table.item(0, 3).text


def test_mixin_wires_mark_keep_to_sidecar_write():
    import inspect
    from labelforge.ui.mixins.quality_lint import QualityLintMixin
    src = inspect.getsource(QualityLintMixin._open_quality_lint)
    assert "mark_keep_requested" in src
    handler = inspect.getsource(QualityLintMixin._mark_lint_tiny_keep)
    assert "mark_tiny_keep" in handler
    # 已加载图重扫以内存为准，必须把内存框一起刷新
    assert "detection_boxes" in handler


def test_i18n_small_target_audit_terms():
    from labelforge.ui.i18n import _strings
    for key in (
            "小目标不可检出", "小目标需放大",
            "尺寸/提示", "小目标可检测性", "检测框",
            "不可检出", "需放大", "可通过", "稳定检出",
            "已标记保留", "标记保留", "标记保留选中的 {n} 项", "标记失败",
            "未找到可统计的检测框", "min(w,h) 占图长边比例", "检测框数",
            "无分布数据", "常规配置检不到，建议提高分辨率或滑窗切片",
            "需 P2 检测头或裁剪放大",
            "存在常规配置检不到的目标：建议提高训练分辨率或启用滑窗切片；"
            "确有意的细长目标可右键「标记保留」"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_every_literal_tr_key_in_lint_ui_is_translated():
    """扫描质检 UI 源码里的 tr("字面量")，逐条验证中英词典都有。

    显式列表测试会漏：新加文案时忘了同步 i18n，中文环境一切正常，
    切英文才显示中文。这条从源码自动提取，以后加文案不需要再维护列表。
    KIND_LABELS 的中文名也走 tr()，一并覆盖。

    用 ast 而不是正则：源码里的长文案常写成跨行相邻字面量（隐式拼接），
    正则只会截到第一段，提取出词典里根本不存在的"半个键"。
    ast 会把相邻字面量合并成一个 Constant，正好是运行时真正的键。
    """
    import ast
    from pathlib import Path
    from labelforge.ui.i18n import _strings
    from labelforge.engine.quality_lint import KIND_LABELS
    root = Path(__file__).resolve().parents[1] / "labelforge"
    sources = [
        root / "ui" / "quality_lint_dialog.py",
        root / "ui" / "mixins" / "quality_lint.py",
    ]
    keys = {v for v in KIND_LABELS.values()}
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            if not (isinstance(func, ast.Name) and func.id == "tr"):
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                keys.add(arg.value)
    assert len(keys) > 30, f"扫描到的键异常偏少，提取逻辑可能失效: {len(keys)}"
    for key in sorted(keys):
        assert key in _strings["zh"], f"zh 缺: {key}"
        assert key in _strings["en"], f"en 缺: {key}"


def test_kind_labels_cover_new_tiny_kinds():
    from labelforge.engine.quality_lint import KIND_LABELS
    assert "tiny_unreachable" in KIND_LABELS
    assert "tiny_needs_p2" in KIND_LABELS


def test_stable_line_is_not_the_warn_line(tmp_path):
    """12px @640（1.875%）落 warn 与 stable 之间：算通过，但**不算**稳定检出。

    这条守住 stable 线（16px）与 warn 线（8px）是两个数：把 stable 线误设成
    warn 线时，只有落在两线之间的框才会暴露 —— 现有用例的框要么 < 8px、
    要么 >= 16px，全都擦肩而过。stable 是报告参考线，把「勉强够」的框算进
    「稳定检出」会让用户以为这类目标已经稳了。
    """
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "stable", [
        {"label": "a", "x": 0, "y": 0, "width": 12, "height": 12,
         "shape_type": "rectangle"},
        {"label": "b", "x": 100, "y": 0, "width": 16, "height": 16,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    report = lint_dataset([img])["tiny_report"]

    assert report["pass"] == 2          # 两个都算通过
    assert report["stable"] == 1        # 只有 16px 那个达到稳定线
    assert report["warn"] == 0


def test_tiny_report_counts_three_tiers(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "t", [
        {"label": "a", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
        {"label": "b", "x": 100, "y": 0, "width": 6, "height": 6,
         "shape_type": "rectangle"},
        {"label": "c", "x": 200, "y": 0, "width": 20, "height": 20,
         "shape_type": "rectangle"},
        {"label": "d", "x": 300, "y": 0, "width": 40, "height": 40,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    report = lint_dataset([img])["tiny_report"]

    assert report["total"] == 4
    assert report["fail"] == 1        # 3px = 0.47%
    assert report["warn"] == 1        # 6px = 0.94%
    assert report["pass"] == 2        # 20px / 40px
    assert report["stable"] == 2      # 都 >= 16px(2.5%)
    assert report["exempt"] == 0


def test_tiny_report_histogram_bins_are_decision_aligned(tmp_path):
    """分桶边界必须压在 0.625%/1.25%/2.5% 上，等宽分桶会把决策区糊成一坨。"""
    from labelforge.engine.quality_lint import lint_dataset, TINY_RATIO_EDGES
    assert 0.00625 in TINY_RATIO_EDGES
    assert 0.0125 in TINY_RATIO_EDGES
    assert 0.025 in TINY_RATIO_EDGES

    img = _write_image_and_json(tmp_path, "h", [
        {"label": "a", "x": 0, "y": 0, "width": 1, "height": 1,
         "shape_type": "rectangle"},
        {"label": "b", "x": 100, "y": 0, "width": 300, "height": 300,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    hist = lint_dataset([img])["tiny_report"]["histogram"]

    assert hist["edges"] == list(TINY_RATIO_EDGES)
    assert len(hist["counts"]) == len(TINY_RATIO_EDGES) - 1
    assert sum(hist["counts"]) == 2
    assert hist["counts"][0] == 1     # 1px = 0.156% 落 [0, 0.3125%)
    assert hist["counts"][7] == 1     # 300px = 46.9% 落 [20%, 50%)


def test_tiny_report_histogram_boundary_value_goes_to_upper_bin(tmp_path):
    """正好踩在桶边界上的值归上界那个桶，必须与 < 判档口径一致。

    4px/640 = 0.625% 正是 🟡 档下界，也得落进 🟡 桶 [0.625%, 1.25%)。
    若归到下界那个桶，直方图会把"刚好不算 🔴"的框画进 🔴 区，
    图形与判档自相矛盾 —— 这条只有在值恰好等于边界时才暴露。
    """
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "b", [
        {"label": "a", "x": 0, "y": 0, "width": 4, "height": 4,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    hist = lint_dataset([img])["tiny_report"]["histogram"]

    assert hist["counts"][2] == 1     # bin 2 = [0.625%, 1.25%) = 🟡 区
    assert hist["counts"][1] == 0     # bin 1 = [0.3125%, 0.625%) = 🔴 区


def test_tiny_report_counts_exempt_separately(tmp_path):
    """豁免框单列，不混进三档占比，否则「保留」会虚增通过数。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "e", [
        {"label": "wire", "x": 0, "y": 0, "width": 600, "height": 3,
         "shape_type": "rectangle", "flags": {"tiny_keep": True}},
        {"label": "car", "x": 0, "y": 100, "width": 100, "height": 100,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    report = lint_dataset([img])["tiny_report"]

    assert report["exempt"] == 1
    assert report["total"] == 2
    assert report["measured"] == 1          # 占比分母只算未豁免的
    assert report["fail"] == 0
    assert report["pass"] == 1
    assert report["fail"] + report["warn"] + report["pass"] == report["measured"]
    assert sum(report["histogram"]["counts"]) == 1


def test_tiny_report_present_even_when_no_issues(tmp_path):
    """干净数据集也要有报告，否则直方图无从显示。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "ok", [
        {"label": "car", "x": 0, "y": 0, "width": 100, "height": 100,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    result = lint_dataset([img])

    assert result["issues"] == []
    assert result["tiny_report"]["total"] == 1
    assert result["tiny_report"]["pass"] == 1


def test_tiny_report_on_empty_dataset(tmp_path):
    from labelforge.engine.quality_lint import lint_dataset
    report = lint_dataset([])["tiny_report"]
    assert report["total"] == 0
    assert sum(report["histogram"]["counts"]) == 0


def test_summary_initializes_new_tiny_kinds(tmp_path):
    """UI 直接读 summary[kind]，键必须恒在（哪怕计数为 0）。"""
    from labelforge.engine.quality_lint import lint_dataset
    img = _write_image_and_json(tmp_path, "s", [
        {"label": "a", "x": 0, "y": 0, "width": 3, "height": 3,
         "shape_type": "rectangle"},
    ], image_size=(640, 640))

    summary = lint_dataset([img])["summary"]

    assert summary["tiny_unreachable"] == 1
    assert summary["tiny_needs_p2"] == 0
