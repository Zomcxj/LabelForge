"""抠图后处理：概率归一、阈值、包围盒裁剪、RGBA 合成。"""
import numpy as np
import pytest

from labelforge.engine.matting import (
    alpha_bbox, alpha_from_output, apply_threshold, crop_to_alpha, compose_rgba,
)


def test_probabilities_pass_through():
    raw = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = alpha_from_output(raw)
    assert pytest.approx(out[0, 0], abs=1e-6) == [0.0, 0.5, 1.0]


def test_logits_get_sigmoid():
    raw = np.array([[[-10.0, 0.0, 10.0]]], dtype=np.float32)
    out = alpha_from_output(raw)
    assert out[0, 0, 0] < 1e-3
    assert pytest.approx(out[0, 0, 1], abs=1e-6) == 0.5
    assert out[0, 0, 2] > 0.999


def test_threshold_zero_keeps_soft_alpha():
    alpha = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    assert np.array_equal(apply_threshold(alpha, 0.0), alpha)


def test_threshold_turns_alpha_binary():
    alpha = np.array([[0.1, 0.5, 0.9]], dtype=np.float32)
    out = apply_threshold(alpha, 0.5)
    assert out.tolist() == [[0.0, 1.0, 1.0]]


def test_crop_to_alpha_removes_transparent_border():
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[3:6, 4:7] = 1.0
    cropped = crop_to_alpha(alpha, pad=1)
    assert cropped.shape == (5, 5)          # 3x3 + 1px 边


def test_crop_to_alpha_returns_none_when_empty():
    assert crop_to_alpha(np.zeros((5, 5), dtype=np.float32)) is None


def test_alpha_bbox_reports_unpadded_bounds():
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[3:6, 4:7] = 1.0
    assert alpha_bbox(alpha, pad=0) == (3, 4, 6, 7)


def test_alpha_bbox_clamps_padding_at_the_border():
    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[0:2, 0:2] = 1.0
    assert alpha_bbox(alpha, pad=2) == (0, 0, 4, 4)


def test_crop_to_alpha_measures_the_extent_after_eight_bit_quantisation():
    """默认软 alpha 下，包围盒必须按落盘的 8 位 alpha 判，不能按 float 判。

    sigmoid 输出永远不会精确为 0，按 float `> 0` 判会让包围盒恒等于整块裁窗，
    于是 det 框比物体大一圈（这正是本用例要防住的缺陷）。1e-4 量化后是 0，
    也就是 PNG 里真正的透明像素。
    """
    alpha = np.full((40, 40), 1e-4, dtype=np.float32)
    alpha[10:14, 20:24] = 0.9
    alpha[14, 20:24] = 0.5                  # 软边：量化后仍 > 0

    assert alpha_bbox(alpha, pad=0) == (10, 20, 15, 24)

    cropped = crop_to_alpha(alpha, pad=1)
    assert cropped.shape == (7, 6)          # 5x4 + 1px 边，不是整块 40x40
    # 软边保留：裁出来的 alpha 是原值，没有被二值化
    assert np.isclose(cropped.max(), 0.9, atol=1e-6)
    assert np.any(np.isclose(cropped, 0.5, atol=1e-6))
    assert np.any(np.isclose(cropped, 1e-4, atol=1e-9))


def test_quantised_crop_still_rejects_a_fully_transparent_input():
    """量化后整块都是 0（例如 alpha 恒为 1e-4）同样算没有前景。"""
    assert crop_to_alpha(np.zeros((8, 8), dtype=np.float32)) is None
    assert crop_to_alpha(np.full((40, 40), 1e-4, dtype=np.float32)) is None


def test_compose_rgba_sets_alpha_channel():
    rgb = np.full((4, 4, 3), 200, dtype=np.uint8)
    alpha = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = compose_rgba(rgb, alpha)
    assert out.shape == (4, 4, 4)
    assert out[0, 0, 3] == 0
    assert out[3, 3, 3] == 255
    assert out[3, 3, 0] == 200


def test_empty_alpha_raises_with_a_clear_message():
    """全透明输入要给出可读错误，不能把 None 传到下游当数组用。"""
    from labelforge.engine.matting import crop_to_alpha_or_raise

    with pytest.raises(ValueError) as excinfo:
        crop_to_alpha_or_raise(np.zeros((5, 5), dtype=np.float32))
    assert "全透明" in str(excinfo.value)


def test_crop_to_alpha_or_raise_returns_the_crop():
    from labelforge.engine.matting import crop_to_alpha_or_raise

    alpha = np.zeros((10, 10), dtype=np.float32)
    alpha[3:6, 4:7] = 1.0
    assert crop_to_alpha_or_raise(alpha, pad=1).shape == (5, 5)
