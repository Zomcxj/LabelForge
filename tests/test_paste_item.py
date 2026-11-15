"""贴图项数据模型：默认字段、派生属性、旧三元组归一化。"""
import pytest

from labelforge.engine.paste_item import PasteItem, as_paste_item


class _Pixmap:
    """最小替身：只需要能被粘贴项持有。"""


def test_defaults_are_det_without_a_source_path():
    item = PasteItem(pixmap=_Pixmap(), rect=(10, 20, 30, 40), label="car")
    assert item.angle == 0.0
    assert item.path is None
    assert item.shape_type == "rectangle"
    for gone in ('feather', 'color_match', 'shadow'):
        assert not hasattr(item, gone)


def test_source_path_is_kept_for_cropping():
    item = PasteItem(pixmap=_Pixmap(), rect=(0, 0, 10, 10), label="car",
                     path="C:/matte/car_001.png")
    assert item.path == "C:/matte/car_001.png"


def test_angle_turns_shape_into_rotation():
    item = PasteItem(pixmap=_Pixmap(), rect=(0, 0, 10, 10), label="car", angle=30.0)
    assert item.shape_type == "rotation"


def test_corners_are_axis_aligned_when_angle_is_zero():
    item = PasteItem(pixmap=_Pixmap(), rect=(10, 20, 30, 40), label="car")
    assert item.corners() == [(10, 20), (40, 20), (40, 60), (10, 60)]


def test_corners_rotate_about_center():
    item = PasteItem(pixmap=_Pixmap(), rect=(0, 0, 10, 10), label="car", angle=90.0)
    corners = item.corners()
    # 绕中心 (5,5) 顺时针 90°：(0,0) -> (10,0)
    assert corners[0] == pytest.approx((10, 0), abs=1e-6)
    assert corners[1] == pytest.approx((10, 10), abs=1e-6)
    assert corners[2] == pytest.approx((0, 10), abs=1e-6)
    assert corners[3] == pytest.approx((0, 0), abs=1e-6)


def test_as_paste_item_wraps_legacy_tuple():
    pixmap = _Pixmap()
    item = as_paste_item((pixmap, (1, 2, 3, 4), "dog"))
    assert item.pixmap is pixmap
    assert item.label == "dog"
    assert item.angle == 0.0


def test_as_paste_item_returns_same_instance():
    item = PasteItem(pixmap=_Pixmap(), rect=(0, 0, 1, 1), label="a")
    assert as_paste_item(item) is item


def test_as_paste_item_rejects_unknown_shape():
    with pytest.raises(TypeError):
        as_paste_item(42)


def test_every_canvas_item_is_a_paste_item_after_random_paste():
    """随机贴图产出的每一项都必须是 PasteItem，不得再产出裸三元组。"""
    import inspect
    from labelforge.engine import paste_engine

    source = inspect.getsource(paste_engine)
    assert "canvas_items.append((" not in source, (
        "随机贴图仍在追加裸三元组，应改为 PasteItem")
