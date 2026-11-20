"""色彩类变换的像素行为测试（纯 numpy，不经 Qt）。"""
import numpy as np
import pytest

from labelforge.engine.augmenter import color2 as C


def test_invert_complements():
    arr = np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8)
    out = C.invert_array(arr)
    assert out[0, 0, 0] == 255 and out[0, 1, 0] == 0


def test_gamma_midpoint_stays():
    arr = np.full((4, 4, 3), 128, np.uint8)
    out = C.gamma_array(arr, 1.0)
    assert abs(int(out[0, 0, 0]) - 128) <= 1


def test_grayscale_channels_equal():
    arr = np.array([[[10, 200, 90]]], np.uint8)
    out = C.grayscale_array(arr)
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]


def test_channel_extract_keeps_one_channel():
    arr = np.array([[[10, 200, 90]]], np.uint8)
    out = C.channel_array(arr, "g")
    assert out[0, 0, 1] == 200
    assert out[0, 0, 0] == 0 and out[0, 0, 2] == 0


def test_levels_clamps_endpoints():
    arr = np.array([[[0, 0, 0], [255, 255, 255], [128, 128, 128]]], np.uint8)
    out = C.levels_array(arr, 10, 245)
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255


def test_levels_inverted_range_is_noop():
    arr = np.array([[[40, 50, 60]]], np.uint8)
    assert np.array_equal(C.levels_array(arr, 200, 100), arr)


def test_hist_equal_on_flat_image_is_safe():
    arr = np.full((8, 8, 3), 77, np.uint8)
    assert C.hist_equal_array(arr).shape == arr.shape


def test_white_balance_zero_is_noop():
    arr = np.full((4, 4, 3), 128, np.uint8)
    assert np.array_equal(C.white_balance_array(arr, 0), arr)


def test_white_balance_warm_boosts_red():
    arr = np.full((2, 2, 3), 100, np.uint8)
    out = C.white_balance_array(arr, 100)
    assert out[0, 0, 0] > 100 and out[0, 0, 2] < 100


@pytest.mark.parametrize("name", ["invert", "gamma", "white_balance",
                                  "hist_equal", "grayscale", "levels",
                                  "channel"])
def test_each_colour_transform_is_registered(name):
    from labelforge.engine.augmenter import get_all_transforms
    assert name in get_all_transforms()
