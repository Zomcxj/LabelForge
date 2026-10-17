"""
utils.py 单元测试 - 纯函数测试，不需要 PySide6
"""
import sys
import os
import importlib.util

# 直接从文件加载模块，绕过 __init__.py 的 PySide6 导入
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_utils_path = os.path.join(os.path.dirname(__file__), '..', 'labelforge', 'core', 'utils.py')
_utils = _load_module('labelforge.core.utils', os.path.abspath(_utils_path))

calculate_iou = _utils.calculate_iou
natural_sort_key = _utils.natural_sort_key
extract_label_name = _utils.extract_label_name


class TestCalculateIou:
    """calculate_iou 测试"""

    def test_no_overlap(self):
        assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_complete_overlap(self):
        assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_partial_overlap(self):
        iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.0 < iou < 1.0
        assert abs(iou - 25 / 175) < 1e-6

    def test_one_inside_other(self):
        iou = calculate_iou((0, 0, 10, 10), (2, 2, 8, 8))
        assert iou == 36 / 100

    def test_touching_edges(self):
        assert calculate_iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_touching_corners(self):
        assert calculate_iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0

    def test_identical_boxes(self):
        assert calculate_iou((5, 5, 15, 15), (5, 5, 15, 15)) == 1.0

    def test_zero_area_box(self):
        assert calculate_iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0

    def test_symmetric(self):
        box1 = (0, 0, 10, 10)
        box2 = (5, 5, 15, 15)
        assert calculate_iou(box1, box2) == calculate_iou(box2, box1)


class TestNaturalSortKey:
    """natural_sort_key 测试"""

    def test_numeric_ordering(self):
        items = ["img10", "img2", "img1"]
        assert sorted(items, key=natural_sort_key) == ["img1", "img2", "img10"]

    def test_mixed_alphanumeric(self):
        items = ["file_10.txt", "file_2.txt", "file_1.txt"]
        assert sorted(items, key=natural_sort_key) == [
            "file_1.txt", "file_2.txt", "file_10.txt"
        ]

    def test_same_prefix(self):
        items = ["a10b", "a2b", "a1b"]
        assert sorted(items, key=natural_sort_key) == ["a1b", "a2b", "a10b"]

    def test_pure_alpha(self):
        items = ["banana", "apple", "cherry"]
        assert sorted(items, key=natural_sort_key) == ["apple", "banana", "cherry"]

    def test_empty_string(self):
        items = ["", "a", "1"]
        result = sorted(items, key=natural_sort_key)
        assert result == ["", "1", "a"]


class TestExtractLabelName:
    """extract_label_name 测试"""

    def test_with_count(self):
        assert extract_label_name("cat (5)") == "cat"

    def test_without_count(self):
        assert extract_label_name("dog") == "dog"

    def test_empty_string(self):
        assert extract_label_name("") == ""

    def test_multiple_spaces(self):
        assert extract_label_name("car (10)") == "car"

    def test_no_parens(self):
        assert extract_label_name("person_01") == "person_01"

    def test_strips_group_brackets(self):
        assert extract_label_name("nose [1]") == "nose"
        assert extract_label_name("nose [1] (3)") == "nose"

    def test_strips_warning_suffix(self):
        assert extract_label_name("nose [1] ⚠无det") == "nose"
        assert extract_label_name("nose ⚠不在框内") == "nose"
        assert extract_label_name("nose [2] ⚠无det (3)") == "nose"

    def test_strips_obb_angle_suffix(self):
        assert extract_label_name("car 33°") == "car"
        assert extract_label_name("car 30.0°") == "car"
        assert extract_label_name("car [1] 45°") == "car"
        # 非角度的度数不误删
        assert extract_label_name("90°car") == "90°car"


class TestBoxMatchesFilter:
    """任务/分组筛选（仅显示用）。"""

    def test_empty_filters_pass_all(self):
        from labelforge.core.utils import box_matches_filter
        assert box_matches_filter({"shape_type": "rectangle"})
        assert box_matches_filter({"shape_type": "point", "group_id": 5})

    def test_task_filter_matches_shape_type(self):
        from labelforge.core.utils import box_matches_filter
        rect = {"shape_type": "rectangle"}
        poly = {"shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}
        assert box_matches_filter(rect, task_filter={"det"})
        assert not box_matches_filter(rect, task_filter={"seg"})
        assert box_matches_filter(poly, task_filter={"seg", "pose"})

    def test_group_filter_matches_group_id(self):
        from labelforge.core.utils import box_matches_filter
        assert box_matches_filter({"group_id": 1}, group_filter={1})
        assert not box_matches_filter({"group_id": 2}, group_filter={1})
        assert not box_matches_filter({}, group_filter={1})


class TestPathUtils:
    """PathUtils 测试"""

    def test_normalize_path(self):
        assert _utils.PathUtils.normalize_path("a/b/../c") == os.path.normpath("a/b/../c")

    def test_normalize_path_empty(self):
        assert _utils.PathUtils.normalize_path("") == ""

    def test_to_display_path(self):
        result = _utils.PathUtils.to_display_path("some/path")
        assert isinstance(result, str)

    def test_to_file_path(self):
        result = _utils.PathUtils.to_file_path("some\\path")
        assert isinstance(result, str)

    def test_join_path(self):
        assert _utils.PathUtils.join_path("a", "b") == os.path.join("a", "b")

    def test_join_path_single(self):
        assert _utils.PathUtils.join_path("a") == "a"

    def test_get_path_separator(self):
        assert _utils.PathUtils.get_path_separator() == os.sep


class TestOutputSidecarPaths:
    """output_sidecar_paths：前缀可变，精确匹配优先于 `_{stem}` 后缀匹配。"""

    def test_exact_stem_wins_over_suffix_collision(self, tmp_path):
        image = str(tmp_path / "cat.png")
        (tmp_path / "cat.json").write_text("{}", encoding="utf-8")
        (tmp_path / "paste_big_cat.json").write_text("{}", encoding="utf-8")
        (tmp_path / "paste_cat.json").write_text("{}", encoding="utf-8")

        found = [os.path.basename(p)
                 for p in _utils.output_sidecar_paths(image, str(tmp_path))]

        assert found == ["cat.json"]

    def test_suffix_match_used_when_no_exact_match(self, tmp_path):
        image = str(tmp_path / "cat.png")
        (tmp_path / "paste_cat.json").write_text("{}", encoding="utf-8")
        (tmp_path / "sess2_cat.json").write_text("{}", encoding="utf-8")

        found = sorted(os.path.basename(p)
                       for p in _utils.output_sidecar_paths(image, str(tmp_path)))

        assert found == ["paste_cat.json", "sess2_cat.json"]

    def test_missing_output_dir_returns_empty(self, tmp_path):
        assert _utils.output_sidecar_paths(
            str(tmp_path / "cat.png"), str(tmp_path / "nope")) == []
