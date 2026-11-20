"""向量化内核测试。

color/noise 的内核是纯 numpy，进程内直接跑；base 的 QImage 转换助手要看真实
像素与步长，`tests/conftest.py` 的 mock 没有，只能走 offscreen 子进程
（与 test_augmenter 同套路）。
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from labelforge.engine.augmenter.color import (
    brightness_array, contrast_array, hsv_adjust_array)
from labelforge.engine.augmenter.noise import (
    gaussian_noise_array, salt_pepper_array)

ROOT = Path(__file__).resolve().parents[1]

# 退出用 sys.exit（正常 finalize）而不是 os._exit：后者仍会经 ExitProcess 卸载 DLL
# 触发 Qt 静态析构，实测 150 次崩 12 次（0xC0000005/0xC0000409）；sys.exit 崩 0 次。
# 断言失败时脚本先抛异常，两种写法返回码都是 1，不会掩盖真实失败。
_QT_EXIT_GUARD = '''
import sys
sys.stdout.flush()
sys.exit(0)
'''


def _run_real_qt(script, timeout=120):
    """在真实 Qt（offscreen）子进程里跑断言，返回 stdout。"""
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    r = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=timeout,
        encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_brightness_clamps_both_ends():
    arr = np.array([[[10, 10, 10], [250, 250, 250], [100, 100, 100]]], np.uint8)
    out = brightness_array(arr, 30)
    assert out[0, 0, 0] == 40
    assert out[0, 1, 0] == 255
    assert out[0, 2, 0] == 130


def test_contrast_scales_around_128():
    arr = np.array([[[128, 128, 128], [160, 160, 160], [96, 96, 96]]], np.uint8)
    out = contrast_array(arr, 1.5)
    assert out[0, 0, 0] == 128
    assert out[0, 1, 0] == 176
    assert out[0, 2, 0] == 80


def test_hsv_adjust_identity_is_near_noop():
    arr = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    out = hsv_adjust_array(arr, 0, 1.0, 1.0)
    # float32 通路误差为 0，留 1 的余量只为浮点边界
    assert np.abs(out.astype(np.int16) - arr.astype(np.int16)).max() <= 1


def test_hsv_hue_shift_is_in_degrees():
    """色相偏移按角度计：纯红 +120 度应为绿。"""
    red = np.array([[[255, 0, 0]]], np.uint8)
    out = hsv_adjust_array(red, 120, 1.0, 1.0)[0, 0]
    assert abs(int(out[1]) - 255) <= 2 and int(out[0]) <= 2 and int(out[2]) <= 2


def test_hsv_zero_saturation_is_grey():
    arr = np.array([[[200, 100, 50]]], np.uint8)
    out = hsv_adjust_array(arr, 0, 0.0, 1.0)[0, 0]
    assert int(out[0]) == int(out[1]) == int(out[2])


def test_hsv_zero_value_is_black():
    arr = np.array([[[200, 100, 50]]], np.uint8)
    out = hsv_adjust_array(arr, 0, 1.0, 0.0)[0, 0]
    assert int(out[0]) == int(out[1]) == int(out[2]) == 0


def test_gaussian_noise_same_seed_same_output():
    import random
    arr = np.full((8, 8, 3), 100, np.uint8)
    a = gaussian_noise_array(arr, 0, 20, random.Random(5))
    b = gaussian_noise_array(arr, 0, 20, random.Random(5))
    c = gaussian_noise_array(arr, 0, 20, random.Random(6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_salt_pepper_only_extremes_and_keeps_input():
    import random
    arr = np.full((20, 20, 3), 100, np.uint8)
    out = salt_pepper_array(arr, 0.2, random.Random(3))
    assert set(np.unique(out).tolist()) <= {0, 100, 255}
    assert arr.min() == 100 and arr.max() == 100, "不应修改输入数组"


def test_single_pixel_image_survives():
    arr = np.array([[[7, 8, 9]]], np.uint8)
    assert brightness_array(arr, 10)[0, 0, 0] == 17
    assert contrast_array(arr, 1.2).shape == arr.shape
    assert hsv_adjust_array(arr, 30, 1.1, 0.9).shape == arr.shape


# ---------- QImage 转换助手（真实 Qt）----------

def test_qimage_helpers_handle_stride_gray_and_alpha():
    """任意格式都要能转成 RGB888 数组：灰度补三通道、alpha 直接丢弃。"""
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.base import (
    qimage_to_rgb_array, rgb_array_to_qimage, qimage_gray_to_array)

# 5px 宽的 RGB888 会逼出每行对齐填充，按 width*3 读会逐行错位
img = QImage(5, 3, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
img.setPixelColor(4, 2, QColor(10, 20, 30))
arr = qimage_to_rgb_array(img)
assert arr.shape == (3, 5, 3), arr.shape
assert arr.dtype.name == "uint8", arr.dtype
assert arr[2, 4].tolist() == [10, 20, 30], arr[2, 4]
assert arr[0, 0].tolist() == [0, 0, 0], arr[0, 0]

# 灰度输入补成三通道
gray = QImage(2, 2, QImage.Format_Grayscale8)
gray.fill(QColor(128, 128, 128))
g = qimage_to_rgb_array(gray)
assert g.shape == (2, 2, 3), g.shape
assert g[0, 0].tolist() == [128, 128, 128], g[0, 0]

# 带 alpha 的输入只取 RGB，且不是 BGRA 的通道序
argb = QImage(2, 2, QImage.Format_ARGB32)
argb.fill(QColor(10, 20, 30, 128))
a = qimage_to_rgb_array(argb)
assert a[0, 0].tolist() == [10, 20, 30], a[0, 0]

# 回写后是 RGB888 且像素不变
res = rgb_array_to_qimage(arr)
assert res.format() == QImage.Format_RGB888, res.format()
assert qimage_to_rgb_array(res).tolist() == arr.tolist()

# 任意格式取灰度：Qt 用 qGray(r,g,b) = (11r + 16g + 5b) / 32
ga = qimage_gray_to_array(img)
assert ga.shape == (3, 5), ga.shape
assert ga[0, 0] == 0 and ga[2, 4] == 18, (ga[0, 0], ga[2, 4])
assert qimage_gray_to_array(gray).tolist() == [[128, 128], [128, 128]]
print("OK")
''')
    assert "OK" in out
