"""滑窗 / 区域推理的范围枚举、ROI 过滤与结果合并（纯逻辑，不依赖 Qt 与 cv2）。

坐标约定：`OnnxDetector.detect()` 接收裁剪后的 RGB，返回的框已在**该裁剪图**
坐标下（它自己完成 letterbox 逆变换并钳到输入尺寸）。所以 scope 只负责
「加原图像素偏移 + 按类别 NMS」，不能在除以 scale 前加偏移。
"""
import numpy as np
import pytest

from labelforge.engine.augmenter.crop import MAX_WINDOW
from labelforge.engine.inference_scope import (
    MODES, ScopeError, iter_regions, iter_windows, merge_scope_results,
    normalize_scope, region_bbox, region_contains_box_center,
)


# ---------- 窗口枚举 ----------

def test_iter_windows_baseline_4k_grid():
    """3840×2160、640×640、重叠 200 → 9×5 = 45 窗（计划 §3.4 的验收数字）。"""
    windows = iter_windows(3840, 2160, {"w": 640, "h": 640, "overlap": 200})

    assert len(windows) == 45
    xs = sorted({w[0] for w in windows})
    ys = sorted({w[1] for w in windows})
    assert len(xs) == 9 and len(ys) == 5
    assert xs[0] == 0 and ys[0] == 0


def test_iter_windows_stay_inside_image():
    """末窗贴边、不越界：每个窗口都在图内。"""
    windows = iter_windows(1000, 700, {"w": 300, "h": 300, "overlap": 100})

    for x, y, w, h in windows:
        assert x >= 0 and y >= 0
        assert x + w <= 1000 and y + h <= 700


def test_iter_windows_last_window_flush_to_edge():
    """末窗贴边：最后一个窗口的右/下边正好落在图像边界。"""
    windows = iter_windows(1000, 700, {"w": 300, "h": 300, "overlap": 100})

    assert max(x + w for x, y, w, h in windows) == 1000
    assert max(y + h for x, y, w, h in windows) == 700


def test_iter_windows_window_larger_than_image_becomes_single_window():
    """窗口 ≥ 图边时缩为整图，只有一个窗口。"""
    windows = iter_windows(200, 150, {"w": 640, "h": 640, "overlap": 200})

    assert windows == [(0, 0, 200, 150)]


def test_iter_windows_zero_overlap():
    """重叠 0 时按步长整除铺开，不重叠。"""
    windows = iter_windows(900, 300, {"w": 300, "h": 300, "overlap": 0})

    assert sorted(windows) == [(0, 0, 300, 300), (300, 0, 300, 300), (600, 0, 300, 300)]


def test_iter_windows_uneven_remainder_is_distributed():
    """不整除时余量均摊，间距不超过窗口尺寸（保证无空隙）。"""
    windows = iter_windows(1000, 300, {"w": 300, "h": 300, "overlap": 0})

    xs = sorted({x for x, y, w, h in windows})
    assert xs == [0, 233, 467, 700]          # round(i*700/3)
    assert all(b - a <= 300 for a, b in zip(xs, xs[1:]))


def test_iter_windows_overlap_clamped_below_window_size():
    """重叠被钳到 min(cw,ch)-1，不能等于或超过窗口尺寸。

    用 400px 小图验证：重叠 999 钳到 299 后步长为 1，此时 101 个窗口仍在
    上限内，可观测到「钳制后起点逐像素推进」这一效果。
    """
    windows = iter_windows(400, 400, {"w": 300, "h": 300, "overlap": 999})

    for x, y, w, h in windows:
        assert w == 300 and h == 300
    xs = sorted({x for x, y, w, h in windows})
    assert xs[:3] == [0, 1, 2]


def test_iter_windows_rejects_window_count_above_cap():
    """窗口数超上限必须报错，不能把内存吃光。

    重叠逼近窗口尺寸时步长退化为 1，窗口数变成图面积量级：20000² 配 640
    窗口、重叠 639 → 3.7 亿窗；1000² 配 300 窗口、重叠 999 → 49 万窗。
    这类配置等于逐像素扫，没有实用价值。
    """
    for iw, spec in ((20000, {"w": 640, "h": 640, "overlap": 639}),
                     (1000, {"w": 300, "h": 300, "overlap": 999})):
        with pytest.raises(ScopeError) as excinfo:
            iter_windows(iw, iw, spec)
        assert "上限" in str(excinfo.value)


def test_iter_windows_allows_large_but_bounded_grids():
    """上限之内的网格照常放行，不能把正常配置也挡掉。"""
    windows = iter_windows(20000, 20000, {"w": 640, "h": 640, "overlap": 0})
    assert len(windows) == 1024


def test_iter_windows_rejects_non_positive_size():
    """窗口尺寸必须为正整数。"""
    for spec in ({"w": 0, "h": 640, "overlap": 0},
                 {"w": 640, "h": 0, "overlap": 0},
                 {"w": -10, "h": 640, "overlap": 0}):
        with pytest.raises(ScopeError):
            iter_windows(1000, 1000, spec)


def test_iter_windows_rejects_size_above_max_window():
    """窗口尺寸上限沿用 MAX_WINDOW。"""
    with pytest.raises(ScopeError):
        iter_windows(100000, 100000, {"w": MAX_WINDOW + 1, "h": 640, "overlap": 0})


def test_iter_windows_rejects_negative_overlap():
    """重叠不能为负。"""
    with pytest.raises(ScopeError):
        iter_windows(1000, 1000, {"w": 300, "h": 300, "overlap": -1})


def test_iter_windows_rejects_non_positive_image():
    """图像尺寸无效时报错，不返回空列表冒充成功。"""
    for iw, ih in ((0, 100), (100, 0), (-5, 100)):
        with pytest.raises(ScopeError):
            iter_windows(iw, ih, {"w": 300, "h": 300, "overlap": 0})


def test_iter_windows_reuses_the_augmenter_grid_math():
    """③ 推理窗口与数据增强裁剪共用同一套排布，不另写一份。

    用户要求「滑窗建议还是复用数据增强裁剪那块」。共用才不会有「增强里排 45 窗、
    推理里排 46 窗」这种对不上的事。这里把推理结果与直接用增强侧
    `normalized_window()` + `window_starts()`（fill=False）算出来的集合逐项比对。
    """
    from labelforge.engine.augmenter.crop import normalized_window, window_starts

    for iw, ih, spec in ((3840, 2160, {"w": 640, "h": 640, "overlap": 200}),
                         (1000, 700, {"w": 300, "h": 300, "overlap": 100}),
                         (913, 457, {"w": 256, "h": 256, "overlap": 37})):
        cw, ch, ov = normalized_window(iw, ih, spec)
        expected = [(x, y, cw, ch)
                    for y in window_starts(ih, ch, ov)
                    for x in window_starts(iw, cw, ov)]
        assert iter_windows(iw, ih, spec) == expected, (iw, ih, spec)


def test_iter_windows_never_pads_the_edge():
    """③ 推理一律贴边、不越界补边，即使传了增强侧的 fill=True。

    数据增强的填充口径（fill=True）会让末窗越出图像、由调用方补填充色；推理
    绝不能这样——补边色会被喂给模型。用户明确要求「不要边缘填充」。
    """
    iw, ih = 1000, 700
    spec = {"w": 300, "h": 300, "overlap": 100}

    plain = iter_windows(iw, ih, spec)
    # 显式带上 fill=True 也不能改变结果
    with_fill = iter_windows(iw, ih, dict(spec, fill=True))
    assert with_fill == plain

    for x, y, w, h in with_fill:
        assert x + w <= iw and y + h <= ih, (x, y, w, h)
    # 且确实覆盖到边缘（贴边），不是缩在中间
    assert max(x + w for x, y, w, h in with_fill) == iw
    assert max(y + h for x, y, w, h in with_fill) == ih


# ---------- ROI ----------

def test_region_bbox_for_rectangle():
    """矩形 ROI 的外接矩形就是它自己。"""
    bbox = region_bbox({"shape_type": "rectangle", "x": 10, "y": 20,
                        "width": 100, "height": 50})

    assert bbox == (10.0, 20.0, 110.0, 70.0)


def test_region_bbox_for_polygon_uses_points_not_stale_xy():
    """多边形取 points 的外接矩形，不用可能过期的 x/y/width/height。"""
    bbox = region_bbox({"shape_type": "polygon", "x": 0, "y": 0,
                        "width": 1, "height": 1,
                        "points": [[30, 40], [90, 45], [85, 120], [25, 110]]})

    assert bbox == (25.0, 40.0, 90.0, 120.0)


def test_region_bbox_rejects_degenerate_and_non_finite():
    """退化（面积 0）与非有限坐标拒绝。"""
    for region in (
        {"shape_type": "rectangle", "x": 10, "y": 10, "width": 0, "height": 50},
        {"shape_type": "rectangle", "x": 10, "y": 10, "width": 50, "height": 0},
        {"shape_type": "polygon", "points": [[0, 0], [10, 10]]},          # 顶点不足
        {"shape_type": "polygon", "points": [[0, 0], [0, 0], [0, 0]]},    # 零面积
        {"shape_type": "rectangle", "x": float("nan"), "y": 0,
         "width": 10, "height": 10},
        {"shape_type": "polygon", "points": [[0, 0], [float("inf"), 1], [1, 1]]},
    ):
        with pytest.raises(ScopeError):
            region_bbox(region)


def test_iter_regions_clips_to_image_and_keeps_polygon():
    """区域裁剪到图内；多边形保留顶点供掩膜与中心判定使用。"""
    regions = iter_regions(
        [{"shape_type": "rectangle", "x": -20, "y": -10, "width": 100, "height": 100}],
        image_size=(64, 48))

    assert len(regions) == 1
    region = regions[0]
    assert region.bbox == (0.0, 0.0, 64.0, 48.0)
    assert region.points is None


def test_iter_regions_rejects_empty_list():
    """选 ROI 范围却没有 ROI 必须报错，不能返回空列表。"""
    with pytest.raises(ScopeError):
        iter_regions([], image_size=(64, 48))


def test_iter_regions_rejects_no_intersection_with_image():
    """ROI 与目标图没有有效交集时报错（不能返回空范围）。"""
    with pytest.raises(ScopeError):
        iter_regions(
            [{"shape_type": "rectangle", "x": 500, "y": 500,
              "width": 10, "height": 10}],
            image_size=(64, 48))


def test_iter_regions_multiple_regions_kept_in_order():
    """多个 ROI 逐个保留，顺序与输入一致；超出图高的部分裁到图内。"""
    regions = iter_regions([
        {"shape_type": "rectangle", "x": 0, "y": 0, "width": 20, "height": 20},
        {"shape_type": "rectangle", "x": 30, "y": 30, "width": 20, "height": 20},
    ], image_size=(64, 48))

    assert [r.bbox for r in regions] == [(0.0, 0.0, 20.0, 20.0),
                                        (30.0, 30.0, 50.0, 48.0)]


# ---------- ROI 中心过滤 ----------

def test_region_contains_box_center_rectangle_includes_edges():
    """矩形 ROI 含边界：中心正好落在边上算在内。"""
    region = iter_regions(
        [{"shape_type": "rectangle", "x": 10, "y": 10, "width": 40, "height": 40}],
        image_size=(64, 48))[0]

    assert region_contains_box_center(region, {"x": 10, "y": 10, "width": 10, "height": 10})
    assert region_contains_box_center(region, {"x": 40, "y": 40, "width": 10, "height": 10})
    assert not region_contains_box_center(region, {"x": 0, "y": 0, "width": 10, "height": 10})


def test_region_contains_box_center_polygon_edges_and_vertices():
    """多边形 ROI 含边界与顶点（ray-casting 不含边界，须单独判）。"""
    region = iter_regions(
        [{"shape_type": "polygon",
          "points": [[0, 0], [60, 0], [60, 40], [0, 40]]}],
        image_size=(64, 48))[0]

    # 中心落在左边上、顶点上、上边中点上都算在内
    assert region_contains_box_center(region, {"x": -5, "y": 18, "width": 10, "height": 10})
    assert region_contains_box_center(region, {"x": -5, "y": -5, "width": 10, "height": 10})
    assert region_contains_box_center(region, {"x": 25, "y": -5, "width": 10, "height": 10})
    # 明确在外面的不算
    assert not region_contains_box_center(region, {"x": 100, "y": 100, "width": 10, "height": 10})


def test_region_contains_box_center_concave_polygon():
    """凹多边形：凹口里的点不算在内。"""
    region = iter_regions(
        [{"shape_type": "polygon",
          "points": [[0, 0], [60, 0], [60, 40], [30, 10], [0, 40]]}],
        image_size=(64, 48))[0]

    # 中心 (30, 30) 落在凹口内 → 不在 ROI
    assert not region_contains_box_center(region, {"x": 25, "y": 25, "width": 10, "height": 10})
    # 中心 (10, 10) 在实体部分 → 在内
    assert region_contains_box_center(region, {"x": 5, "y": 5, "width": 10, "height": 10})


# ---------- 范围归一化 ----------

def test_normalize_scope_none_means_full_image():
    """scope=None 表示全图，保持既有行为。"""
    assert normalize_scope(None, image_size=(64, 48)) is None


def test_normalize_scope_requires_regions_for_roi_mode():
    """区域范围必须带 ROI；缺失/空表都报错。"""
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "roi", "regions": []}, image_size=(64, 48))
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "roi"}, image_size=(64, 48))


def test_normalize_scope_window_mode_requires_window_spec():
    """滑窗范围必须带窗口参数。"""
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "window"}, image_size=(64, 48))


def test_normalize_scope_rejects_unknown_mode():
    """未知范围模式报错，不静默退回全图。"""
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "随便"}, image_size=(64, 48))


def test_roi_window_mode_is_gone():
    """已删除的「区域+滑窗」不再是合法模式。"""
    assert "roi_window" not in MODES
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "roi_window"}, image_size=(64, 48))


def test_normalize_scope_rejects_unknown_keys():
    """多余字段报错，避免把拼错的字段静默忽略。"""
    with pytest.raises(ScopeError):
        normalize_scope({"mode": "window", "windows": {"w": 640, "h": 640,
                                                       "overlap": 200},
                         "widnow": 1}, image_size=(64, 48))


def test_normalize_scope_deep_copies_regions():
    """ROI 深拷贝：后台任务不能读到正在被编辑的列表。"""
    source = [{"shape_type": "polygon",
               "points": [[0, 0], [60, 0], [60, 40]], "x": 0, "y": 0,
               "width": 60, "height": 40}]

    scope = normalize_scope({"mode": "roi", "regions": source}, image_size=(64, 48))

    source[0]["points"][0][0] = 999
    assert scope["regions"][0]["points"][0][0] == 0


# ---------- 合并与去重 ----------

def _box(x, y, w, h, label="cat", score=0.9, class_id=0):
    return {"x": float(x), "y": float(y), "width": float(w), "height": float(h),
            "label": label, "score": score, "class_id": class_id}


def test_merge_offsets_added_exactly_once():
    """偏移只加一次：窗口 (100, 200) 内的框平移到原图坐标。"""
    merged = merge_scope_results([(100, 200, [_box(10, 20, 30, 40)])])

    assert len(merged) == 1
    assert (merged[0]["x"], merged[0]["y"]) == (110.0, 220.0)
    assert (merged[0]["width"], merged[0]["height"]) == (30.0, 40.0)


def test_merge_offsets_are_not_scaled():
    """偏移在缩放之后加：不能出现 (x*scale + offset) 这类二次变换。

    窗口偏移 100、框坐标 50、scale 无关——结果必须是 150，不是 100+50*scale。
    """
    merged = merge_scope_results([(100, 0, [_box(50, 0, 10, 10)])])

    assert merged[0]["x"] == 150.0


def test_merge_roi_plus_window_offsets_accumulate():
    """ROI+滑窗：ROI 偏移与窗口偏移都要加，各加一次。"""
    merged = merge_scope_results([(300 + 100, 200 + 50, [_box(10, 10, 20, 20)])])

    assert (merged[0]["x"], merged[0]["y"]) == (410.0, 260.0)


def test_merge_drops_same_class_duplicates_across_windows():
    """跨窗同类重复被按类 NMS 去掉。"""
    merged = merge_scope_results([
        (0, 0, [_box(100, 100, 50, 50)]),
        (10, 0, [_box(90, 100, 50, 50)]),        # 偏移后与上一个几乎重合
    ])

    assert len(merged) == 1


def test_merge_keeps_different_classes_even_when_overlapping():
    """不同类的高重合框必须保留（类别参与去重）。"""
    merged = merge_scope_results([
        (0, 0, [_box(100, 100, 50, 50, label="cat", class_id=0)]),
        (0, 0, [_box(100, 100, 50, 50, label="dog", class_id=1)]),
    ])

    assert len(merged) == 2
    assert {b["label"] for b in merged} == {"cat", "dog"}


def test_merge_keeps_separate_same_class_boxes():
    """同类但不重合的框都保留。"""
    merged = merge_scope_results([
        (0, 0, [_box(0, 0, 20, 20)]),
        (0, 0, [_box(500, 500, 20, 20)]),
    ])

    assert len(merged) == 2


def test_merge_clips_to_image_bounds():
    """合并后钳到原图：窗口内贴近边缘的框不能越出图像。"""
    merged = merge_scope_results([(60, 40, [_box(0, 0, 20, 20)])],
                                 image_size=(64, 48))

    assert (merged[0]["x"], merged[0]["y"]) == (60.0, 40.0)
    assert merged[0]["width"] == 4.0 and merged[0]["height"] == 8.0


def test_merge_drops_boxes_entirely_outside_image():
    """完全落在图外的框丢弃。"""
    merged = merge_scope_results([(200, 200, [_box(0, 0, 20, 20)])],
                                 image_size=(64, 48))

    assert merged == []


def test_merge_empty_input():
    """空输入返回空列表。"""
    assert merge_scope_results([]) == []
    assert merge_scope_results([(0, 0, [])]) == []


def test_merge_sorted_by_score_descending():
    """结果按置信度降序，便于界面与写盘稳定。"""
    merged = merge_scope_results([
        (0, 0, [_box(0, 0, 10, 10, score=0.3)]),
        (100, 0, [_box(0, 0, 10, 10, score=0.9)]),
    ])

    assert [b["score"] for b in merged] == [0.9, 0.3]


def test_merge_respects_iou_threshold():
    """IoU 阈值生效：阈值极高时不合并。"""
    boxes = [
        (0, 0, [_box(100, 100, 50, 50)]),
        (0, 0, [_box(105, 100, 50, 50)]),
    ]
    assert len(merge_scope_results(boxes, iou_threshold=0.99)) == 2
    assert len(merge_scope_results(boxes, iou_threshold=0.45)) == 1


# ---------- min_visible 随窗口参数传递 ----------

def test_iter_windows_accepts_min_visible_in_the_spec():
    """③ 加了 min_visible 的 window_spec 不能被当成未知字段拒掉。

    `_WINDOW_KEYS` 是白名单，漏加会让界面「点确定」后直接报错，
    所以这条测试同时钉住白名单与真正枚举两条路。
    """
    windows = iter_windows(1000, 1000, {"w": 640, "h": 640, "overlap": 200,
                                        "min_visible": 0.7})
    assert windows


def test_iter_windows_still_rejects_a_genuinely_unknown_key():
    """白名单不能因为要放行 min_visible 就整块放开。"""
    with pytest.raises(ScopeError):
        iter_windows(1000, 1000, {"w": 640, "h": 640, "overlap": 200,
                                  "bogus": 1})


def test_normalize_window_spec_keeps_min_visible():
    from labelforge.engine.inference_task import _normalize_window_spec
    assert _normalize_window_spec(
        {"w": 640, "h": 640, "overlap": 200, "min_visible": 0.8}
    )["min_visible"] == 0.8


def test_normalize_window_spec_clamps_out_of_range_min_visible():
    """越界钳到合法区间，而不是整份丢弃。

    与模板侧（越界即丢）策略不同：模板是用户手填的一条记录，丢了没损失；
    这里是「上次用过的参数」，丢了会让上次的设置莫名其妙消失。
    """
    from labelforge.engine.inference_task import _normalize_window_spec
    assert _normalize_window_spec(
        {"w": 640, "h": 640, "overlap": 200, "min_visible": 5}
    )["min_visible"] == 1.0
    assert _normalize_window_spec(
        {"w": 640, "h": 640, "overlap": 200, "min_visible": 0}
    )["min_visible"] == 0.5


def test_normalize_window_spec_defaults_missing_min_visible():
    """旧版本存的 spec 没有这个键 → 补默认值，不能让整份 spec 失效。"""
    from labelforge.core.config_manager import DEFAULT_CROP_MIN_VISIBLE
    from labelforge.engine.inference_task import _normalize_window_spec
    spec = _normalize_window_spec({"w": 640, "h": 640, "overlap": 200})
    assert spec.get("min_visible") == DEFAULT_CROP_MIN_VISIBLE
    # 其余三个键仍要在，别把整份丢弃了
    assert (spec["w"], spec["h"], spec["overlap"]) == (640, 640, 200)


def test_normalize_window_spec_still_drops_a_broken_size():
    """min_visible 的宽容不能顺带让坏尺寸蒙混过关。"""
    from labelforge.engine.inference_task import _normalize_window_spec
    assert _normalize_window_spec(
        {"w": 0, "h": 640, "overlap": 200, "min_visible": 0.7}) == {}
    assert _normalize_window_spec(
        {"w": "x", "h": 640, "overlap": 200}) == {}


# ---------- ② 带行列号的滑窗枚举（画布左下角要标 r1c1）----------

def test_iter_windows_with_index_matches_iter_windows():
    """带序号版必须与 `iter_windows()` **逐项同序同值**（只多两个序号）。"""
    from labelforge.engine.inference_scope import iter_windows_with_index

    spec = {"w": 300, "h": 300, "overlap": 100}
    plain = iter_windows(1000, 700, spec)
    indexed = iter_windows_with_index(1000, 700, spec)

    assert [t[:4] for t in indexed] == plain
    assert len(indexed) == len(plain)


def test_iter_windows_with_index_numbers_rows_and_cols_from_one():
    """行列号从 1 开始、行优先：`r1c1`、`r1c2`、`r2c1`……

    数据增强里就是这么标的（`enumerate(window_starts(...), 1)`），画布要
    跟它一致，否则同一个窗口在两处显示不同的号。
    """
    from labelforge.engine.inference_scope import iter_windows_with_index

    # 1000x700、300 窗、100 重叠 → 5 列 x 3 行（共 15 窗）
    indexed = iter_windows_with_index(1000, 700, {"w": 300, "h": 300,
                                                  "overlap": 100})
    labels = [(r, c) for (_x, _y, _w, _h, r, c) in indexed]

    assert len(indexed) == 15, len(indexed)
    assert labels[0] == (1, 1), labels[0]
    assert labels[1] == (1, 2), labels[1]
    # 第 6 个应该换到第二行第一列（每行 5 个）
    assert labels[5] == (2, 1), labels[5]
    assert labels[-1] == (3, 5), labels[-1]


def test_iter_windows_with_index_single_window_is_r1c1():
    """窗口比图还大 → 只有一窗，号就是 `r1c1`。"""
    from labelforge.engine.inference_scope import iter_windows_with_index

    indexed = iter_windows_with_index(200, 150, {"w": 640, "h": 640,
                                                 "overlap": 200})
    assert len(indexed) == 1
    assert indexed[0][4:] == (1, 1)


def test_iter_windows_with_index_propagates_scope_errors():
    """参数非法时和 `iter_windows()` 一样抛 `ScopeError`，不静默给空表。"""
    from labelforge.engine.inference_scope import iter_windows_with_index

    with pytest.raises(ScopeError):
        iter_windows_with_index(0, 100, {"w": 300, "h": 300, "overlap": 0})


def test_window_grid_shape_agrees_with_the_augmenter():
    """行×列必须等于窗口数，且列数与数据增强的 `window_starts()` 一致。

    这是「画布上的号」与「数据增强里的号」同源的依据。
    """
    from labelforge.engine.augmenter.crop import normalized_window, window_starts
    from labelforge.engine.inference_scope import iter_windows_with_index

    iw, ih, spec = 3840, 2160, {"w": 640, "h": 640, "overlap": 200}
    indexed = iter_windows_with_index(iw, ih, spec)
    cw, ch, ov = normalized_window(iw, ih, dict(spec))
    cols = len(window_starts(iw, cw, ov))
    rows = len(window_starts(ih, ch, ov))

    assert rows * cols == len(indexed)
    assert max(c for *_r, c in indexed) == cols
    assert max(r for *_x, r, _c in indexed) == rows
