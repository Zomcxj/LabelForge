"""滤镜类变换的像素行为测试（纯 numpy，不经 Qt）。"""
import numpy as np
import pytest

from labelforge.engine.augmenter import filters as F


def _img(values, h=8, w=8):
    return np.full((h, w, 3), values, np.uint8)


def test_ksize_is_forced_odd():
    assert F.GaussianBlur(ksize=4)._ksize % 2 == 1
    assert F.MedianFilter(ksize=0)._ksize == 1
    assert F.AdaptiveThreshold(block_size=4)._block % 2 == 1


def test_threshold_binarises():
    arr = np.array([[[100, 100, 100], [200, 200, 200]]], np.uint8)
    out = F.threshold_array(arr, 128, False)
    assert set(np.unique(out).tolist()) <= {0, 255}
    out_inv = F.threshold_array(arr, 128, True)
    assert not np.array_equal(out, out_inv)


def test_blur_reduces_variance():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    out = F.gaussian_blur_array(arr, 5)
    assert out.std() < arr.std()


def test_sharpen_keeps_shape():
    arr = _img(120)
    assert F.sharpen_array(arr, 1.0).shape == arr.shape


def test_dilate_erode_roundtrip_shape():
    arr = _img(120)
    assert F.dilate_array(arr, 3).shape == arr.shape
    assert F.erode_array(arr, 3).shape == arr.shape


def test_single_pixel_survives_all_filters():
    arr = np.array([[[10, 20, 30]]], np.uint8)
    for fn in (F.gaussian_blur_array, F.box_blur_array, F.median_array,
               F.sharpen_array, F.sobel_array, F.emboss_array,
               F.dilate_array, F.erode_array):
        assert fn(arr, 3).shape == arr.shape


def test_sobel_and_emboss_keep_three_channels():
    arr = _img(120)
    assert F.sobel_array(arr, 1.0).shape == (8, 8, 3)
    assert F.emboss_array(arr, 1.0).shape == (8, 8, 3)


@pytest.mark.parametrize("name", ["gauss_blur", "box_blur", "median",
                                  "sharpen", "sobel", "emboss", "dilate",
                                  "erode", "threshold", "adaptive_thresh"])
def test_each_filter_is_registered(name):
    from labelforge.engine.augmenter import get_all_transforms
    assert name in get_all_transforms()
