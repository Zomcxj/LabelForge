"""LabelMe shape <-> internal box conversion (polygon / group_id / compat)."""
from labelforge.engine.shape_io import (
    box_display_label,
    box_from_labelme_shape,
    format_label_display,
    format_score_display,
    labelme_shape_from_box,
    yolo_seg_line,
)


def test_old_json_without_shape_type_loads_as_rectangle():
    box = box_from_labelme_shape({
        "label": "cat",
        "points": [[10, 20], [110, 20], [110, 80], [10, 80]],
    })
    assert box["shape_type"] == "rectangle"
    assert box["x"] == 10
    assert box["y"] == 20
    assert box["width"] == 100
    assert box["height"] == 60
    assert box["label"] == "cat"
    assert box.get("group_id") is None


def test_polygon_shape_keeps_points_and_bbox():
    pts = [[10, 10], [50, 10], [40, 40]]
    box = box_from_labelme_shape({
        "label": "person",
        "shape_type": "polygon",
        "points": pts,
        "group_id": 2,
    })
    assert box["shape_type"] == "polygon"
    assert box["points"] == pts
    assert box["x"] == 10
    assert box["y"] == 10
    assert box["width"] == 40
    assert box["height"] == 30
    assert box["group_id"] == 2


def test_labelme_roundtrip_polygon_preserves_points_and_group():
    box = {
        "label": "dog",
        "shape_type": "polygon",
        "points": [[1, 2], [3, 4], [5, 1]],
        "x": 1, "y": 1, "width": 4, "height": 3,
        "group_id": 3,
    }
    shape = labelme_shape_from_box(box)
    assert shape["shape_type"] == "polygon"
    assert shape["points"] == box["points"]
    assert shape["group_id"] == 3
    assert shape["label"] == "dog"


def test_labelme_roundtrip_rectangle_still_four_corner_points():
    box = {"label": "cat", "x": 10, "y": 20, "width": 30, "height": 40}
    shape = labelme_shape_from_box(box)
    assert shape["shape_type"] == "rectangle"
    assert shape["points"] == [[10, 20], [40, 20], [40, 60], [10, 60]]
    assert shape["group_id"] is None


def test_format_label_display_shows_group_brackets():
    assert format_label_display("nose") == "nose"
    assert format_label_display("nose", None) == "nose"
    assert format_label_display("nose", 1) == "nose [1]"


def test_yolo_seg_line_normalizes_polygon_points():
    box = {
        "label": "person",
        "shape_type": "polygon",
        "points": [[50, 25], [150, 25], [100, 75]],
    }
    line = yolo_seg_line(box, class_id=0, image_width=200, image_height=100)
    assert line.startswith("0 ")
    parts = line.split()
    assert parts[0] == "0"
    assert len(parts) == 1 + 6
    assert parts[1:3] == ["0.250000", "0.250000"]


def test_point_in_polygon_inside_and_outside():
    from labelforge.engine.shape_io import point_in_polygon
    pts = [[10, 10], [40, 10], [40, 40]]
    assert point_in_polygon(30, 20, pts) is True
    assert point_in_polygon(11, 39, pts) is False


def test_point_line_distance_perpendicular_and_clamped():
    from labelforge.engine.shape_io import point_line_distance
    assert point_line_distance((15, 20), ((10, 10), (40, 10))) == 10.0
    assert point_line_distance((0, 0), ((10, 10), (40, 10))) == 200 ** 0.5


def test_nearest_polygon_edge_insert_index():
    from labelforge.engine.shape_io import nearest_polygon_edge
    pts = [[10, 10], [40, 10], [40, 40], [10, 40]]
    assert nearest_polygon_edge((25, 10), pts, 3.0) == 1
    assert nearest_polygon_edge((40, 25), pts, 3.0) == 2
    assert nearest_polygon_edge((25, 40), pts, 3.0) == 3
    assert nearest_polygon_edge((10, 25), pts, 3.0) == 0
    assert nearest_polygon_edge((70, 25), pts, 3.0) is None


def test_point_shape_roundtrips_through_labelme():
    shape = {"label": "nose", "points": [[12.5, 30.0]],
             "group_id": 2, "shape_type": "point", "flags": {}}
    box = box_from_labelme_shape(shape)
    assert box is not None
    assert box["shape_type"] == "point"
    assert box["points"] == [[12.5, 30.0]]
    assert box["x"] == 12.5 and box["y"] == 30.0

    back = labelme_shape_from_box(box)
    assert back["shape_type"] == "point"
    assert back["points"] == [[12.5, 30.0]]


def test_yolo_seg_line_skips_fewer_than_three_points():
    box = {"shape_type": "polygon", "points": [[1, 1], [2, 2]]}
    assert yolo_seg_line(box, 0, 100, 100) is None


def test_paste_flag_round_trip():
    from labelforge.engine.shape_io import labelme_shape_from_box, box_from_labelme_shape
    box = {"label": "logo", "x": 1, "y": 2, "width": 3, "height": 4}
    shape = labelme_shape_from_box(box)
    assert shape["flags"] == {}
    assert box_from_labelme_shape(shape)["is_paste"] is False
    shape["flags"] = {"paste": True}
    assert box_from_labelme_shape(shape)["is_paste"] is True


def test_labelme_shape_from_box_preserves_flags():
    from labelforge.engine.shape_io import labelme_shape_from_box
    shape = labelme_shape_from_box(
        {"label": "logo", "x": 0, "y": 0, "width": 1, "height": 1,
         "flags": {"paste": True}})
    assert shape["flags"] == {"paste": True}


def test_labelme_shape_from_box_rebuilds_paste_flag_from_is_paste():
    """磁盘载入的贴图 box（只有 is_paste 无 flags）再保存不丢标记。"""
    from labelforge.engine.shape_io import labelme_shape_from_box
    shape = labelme_shape_from_box(
        {"label": "logo", "x": 0, "y": 0, "width": 1, "height": 1,
         "is_paste": True})
    assert shape["flags"] == {"paste": True}


# ---- flags 透传：审核豁免等自定义标记不能在载入时被丢弃 ----

def test_box_from_labelme_shape_preserves_unknown_flags():
    """只挑 paste 会丢掉 tiny_keep，豁免标记保存一次就失效。"""
    box = box_from_labelme_shape({
        "label": "wire", "shape_type": "rectangle",
        "points": [[0, 0], [10, 0], [10, 2], [0, 2]],
        "flags": {"tiny_keep": True},
    })
    assert box["flags"] == {"tiny_keep": True}


def test_unknown_flag_survives_full_roundtrip():
    """回归护栏：载入 → 保存 → 再载入，自定义 flag 必须仍在。"""
    shape = {
        "label": "wire", "shape_type": "rectangle",
        "points": [[0, 0], [10, 0], [10, 2], [0, 2]],
        "flags": {"tiny_keep": True},
    }
    again = box_from_labelme_shape(
        labelme_shape_from_box(box_from_labelme_shape(shape)))
    assert again["flags"] == {"tiny_keep": True}


def test_point_shape_preserves_flags():
    """point 分支同样要透传，否则关键点的豁免标记会丢。"""
    box = box_from_labelme_shape({
        "label": "nose", "shape_type": "point", "points": [[12.5, 30.0]],
        "flags": {"tiny_keep": True},
    })
    assert box is not None
    assert box["flags"] == {"tiny_keep": True}


def test_box_without_flags_has_no_flags_key():
    """无 flags 的 JSON 载入后不塞空字典，保持与改动前完全一致。"""
    box = box_from_labelme_shape({
        "label": "cat",
        "points": [[10, 20], [110, 20], [110, 80], [10, 80]],
    })
    assert "flags" not in box


def test_empty_flags_dict_does_not_add_key():
    box = box_from_labelme_shape({
        "label": "cat", "flags": {},
        "points": [[10, 20], [110, 20], [110, 80], [10, 80]],
    })
    assert "flags" not in box


def test_paste_and_custom_flags_coexist_without_conflict():
    """透传改动不能让 is_paste 与 flags 打架。"""
    box = box_from_labelme_shape({
        "label": "logo",
        "points": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "flags": {"paste": True, "tiny_keep": True},
    })
    assert box["is_paste"] is True
    assert labelme_shape_from_box(box)["flags"] == {
        "paste": True, "tiny_keep": True}


def test_non_dict_flags_ignored_not_crash():
    box = box_from_labelme_shape({
        "label": "cat", "flags": "bogus",
        "points": [[10, 20], [110, 20], [110, 80], [10, 80]],
    })
    assert box is not None
    assert "flags" not in box
    assert box["is_paste"] is False


# ---------- 置信度存盘（参考 X-AnyLabeling 的顶层 score 字段）----------

def test_labelme_shape_stores_score_at_top_level():
    """置信度按 X-AnyLabeling 约定存顶层 "score"，不是塞进 flags。"""
    shape = labelme_shape_from_box({
        "label": "car", "x": 1, "y": 2, "width": 3, "height": 4,
        "score": 0.8675})
    assert shape["score"] == 0.8675
    assert "score" not in shape.get("flags", {})


def test_labelme_shape_score_follows_label():
    """score 紧跟 label，读 JSON 时一眼能看到。"""
    shape = labelme_shape_from_box({"label": "car", "x": 0, "y": 0,
                                    "width": 1, "height": 1, "score": 0.5})
    assert list(shape.keys())[:2] == ["label", "score"]


def test_manual_box_writes_null_score():
    """手工框没有置信度，写 null 占位，让字段始终存在。"""
    shape = labelme_shape_from_box({"label": "person", "x": 0, "y": 0,
                                    "width": 5, "height": 5})
    assert "score" in shape and shape["score"] is None


def test_box_from_shape_drops_null_score():
    """手工框的 null 不该在内部 box 里留下 score=None。"""
    box = box_from_labelme_shape({
        "label": "person", "points": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "score": None})
    assert "score" not in box


def test_score_round_trips_through_shape():
    """存盘再载入后置信度不丢，这是「保存到 json」的核心要求。"""
    original = {"label": "car", "x": 10, "y": 20, "width": 30, "height": 40,
                "score": 0.9087275862693787}
    back = box_from_labelme_shape(labelme_shape_from_box(original))
    assert back["score"] == original["score"]


def test_score_accepted_from_integer_json_value():
    """JSON 里 score 写成整数 1 也要能读，不能要求必须是 float。"""
    box = box_from_labelme_shape({
        "label": "car", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "score": 1})
    assert box["score"] == 1.0
    assert isinstance(box["score"], float)


def test_boolean_score_rejected():
    """bool 是 int 的子类，不能被当成置信度收进来。"""
    box = box_from_labelme_shape({
        "label": "car", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "score": True})
    assert "score" not in box
    shape = labelme_shape_from_box({"label": "car", "x": 0, "y": 0,
                                    "width": 1, "height": 1, "score": True})
    assert shape["score"] is None


def test_garbage_score_rejected_not_crash():
    """score 是奇怪类型时忽略，不能让载入整批失败。"""
    for bad in ("abc", [], {}):
        box = box_from_labelme_shape({
            "label": "car", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "score": bad})
        assert box is not None and "score" not in box


def test_score_display_is_two_decimals():
    assert format_score_display(0.8675) == "0.87"
    assert format_score_display(0.9) == "0.90"
    assert format_score_display(1.0) == "1.00"


def test_score_display_empty_when_absent():
    """没有分数必须返回空串，界面才不会显示 "None"。"""
    for none_like in (None, "0.9", True, [], {}):
        assert format_score_display(none_like) == ""


def test_box_display_label_appends_score_only_when_asked():
    box = {"label": "car", "score": 0.8675}
    assert box_display_label(box, show_score=True) == "car 0.87"
    assert box_display_label(box, show_score=False) == "car"


def test_box_display_label_defaults_to_hiding_score():
    """默认不显示分数，避免既有调用点行为被悄悄改掉。"""
    assert box_display_label({"label": "car", "score": 0.9}) == "car"


def test_box_display_label_keeps_angle_before_score():
    """旋转框显示成「类别 角度 置信度」，角度不能丢。"""
    box = {"label": "car", "shape_type": "rotation", "score": 0.75,
           "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    text = box_display_label(box, show_score=True)
    assert text.startswith("car ")
    assert "0.75" in text


def test_box_display_label_without_score_key():
    """老数据没有 score，显示时不能冒出 "None"。"""
    assert box_display_label({"label": "car"}, show_score=True) == "car"
