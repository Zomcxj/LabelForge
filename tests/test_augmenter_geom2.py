"""几何类变换的像素行为测试（纯 numpy，不经 Qt）。"""
import numpy as np
import pytest

from labelforge.engine.augmenter import geom2 as G


def _img(h=8, w=6):
    return np.zeros((h, w, 3), np.uint8)


def test_rotate90_swaps_dimensions_once():
    assert G.rotate90_array(_img(8, 6), 90).shape == (6, 8, 3)
    assert G.rotate90_array(_img(8, 6), 180).shape == (8, 6, 3)


def test_rotate90_rejects_other_angles():
    assert G.Rotate90(angle=45)._angle == 90
    assert G.rotate90_array(_img(8, 6), 45).shape == (6, 8, 3)


def test_skew_zero_is_noop():
    arr = _img()
    assert np.array_equal(G.skew_array(arr, 0, 0), arr)


def test_skew_keeps_shape():
    arr = _img()
    assert G.skew_array(arr, 10, 0).shape == arr.shape
    assert G.skew_array(arr, 0, 10).shape == arr.shape


def test_perspective_zero_is_noop_shape():
    arr = _img()
    assert G.perspective_array(arr, 0).shape == arr.shape


def test_perspective_offset_keeps_shape():
    arr = _img()
    assert G.perspective_array(arr, 1).shape == arr.shape


def test_crop_region_none_is_noop():
    arr = _img()
    assert np.array_equal(G.crop_region_array(arr, None), arr)


def test_crop_region_slices():
    arr = _img(8, 8)
    assert G.crop_region_array(arr, (2, 2, 4, 3)).shape == (3, 4, 3)


def test_crop_region_coerces_out_of_bounds():
    arr = _img(8, 8)
    out = G.crop_region_array(arr, (-5, -5, 100, 100))
    assert out.shape == (8, 8, 3)


def test_crop_region_zero_size_is_noop():
    arr = _img(8, 8)
    assert np.array_equal(G.crop_region_array(arr, (4, 4, 0, 0)), arr)


def test_pad_adds_border():
    out = G.pad_array(_img(8, 8), 2, 3, 1, 4, (114, 114, 114))
    assert out.shape == (13, 13, 3)
    assert tuple(out[0, 0]) == (114, 114, 114)


def test_pad_zero_is_noop():
    arr = _img()
    assert np.array_equal(G.pad_array(arr, 0, 0, 0, 0), arr)


def test_single_pixel_survives_geometry():
    arr = np.array([[[10, 20, 30]]], np.uint8)
    assert G.skew_array(arr, 10, 10).shape == arr.shape
    assert G.perspective_array(arr, 1).shape == arr.shape
    assert G.crop_region_array(arr, (0, 0, 1, 1)).shape == arr.shape
    assert G.pad_array(arr, 1, 1, 1, 1, (0, 0, 0)).shape == (3, 3, 3)


@pytest.mark.parametrize("name", ["rotate90", "skew", "perspective",
                                  "crop_region", "pad"])
def test_each_geometry_transform_is_registered(name):
    from labelforge.engine.augmenter import get_all_transforms
    assert name in get_all_transforms()
