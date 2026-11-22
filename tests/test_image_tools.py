"""图像编辑工具的操作清单与处理链测试。"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from labelforge.engine import image_tools as it

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_KEYS = {
    "fliph", "flipv", "trans", "rotate", "scale", "rotate90", "skew",
    "perspective", "crop_region", "pad",
    "bright", "contrast", "hue", "saturation", "value", "invert", "gamma",
    "white_balance", "hist_equal", "grayscale", "levels", "channel",
    "gauss_blur", "box_blur", "median", "sharpen", "sobel", "emboss",
    "dilate", "erode", "threshold", "adaptive_thresh",
    "gauss", "saltpepper",
    # 后续补充：几何、色彩、滤镜各一批
    "resize", "clahe", "auto_levels", "posterize",
    "morph_open", "morph_close", "morph_gradient", "tophat", "blackhat",
    "usm", "laplacian", "bilateral", "nlmeans", "motion_blur", "mosaic",
}


def _run_real_qt(script):
    """在真实 Qt（offscreen）子进程里跑断言，返回 stdout。"""
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen",
                        "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c",
         script + "\nimport sys\nsys.stdout.flush()\nsys.exit(0)\n"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=180,
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_operation_meta_covers_every_operator():
    assert set(it.OPERATION_META) == EXPECTED_KEYS
    assert len(it.OPERATION_ORDER) == len(EXPECTED_KEYS)
    assert set(it.OPERATION_ORDER) == EXPECTED_KEYS


def test_operation_order_groups_are_contiguous():
    groups = [it.OPERATION_META[k][0] for k in it.OPERATION_ORDER]
    seen = []
    for g in groups:
        if not seen or seen[-1] != g:
            assert g not in seen, f"分组 {g} 不连续"
            seen.append(g)


def test_every_operation_is_registered():
    from labelforge.engine.augmenter import get_all_transforms
    missing = EXPECTED_KEYS - set(get_all_transforms())
    assert not missing, missing


def test_geometric_keys_are_the_geometry_group():
    assert it.GEOMETRIC_KEYS == {
        k for k, v in it.OPERATION_META.items() if v[0] == it.GROUP_GEOMETRY}
    assert it.is_geometric("fliph") and not it.is_geometric("invert")


def test_every_meta_param_name_is_accepted_by_the_constructor():
    """meta 的参数名必须与构造器形参一致，否则界面走到那一步才炸。"""
    from labelforge.engine.augmenter import get_all_transforms
    registry = get_all_transforms()
    import inspect
    for key in it.OPERATION_ORDER:
        accepted = set(inspect.signature(
            registry[key].__init__).parameters) - {"self", "rng"}
        declared = {spec[0] for spec in it.params_of(key)}
        assert declared <= accepted, f"{key}: {declared - accepted}"


def test_default_params_builds_every_operation():
    for key in it.OPERATION_ORDER:
        params = it.default_params(key)
        assert it.build_transform(key, params) is not None


# 变换构造器会把参数夹到自己的区间；界面范围必须落在该区间内，否则用户拖动
# 到两端时数值被无声截断，界面值与实际效果对不上。
_TRANSFORM_CLAMPS = {
    ("bright", "delta"): (-100, 100),
    ("contrast", "factor"): (0.5, 2.0),
    ("hue", "delta"): (-180, 180),
    ("saturation", "factor"): (0.0, 2.0),
    ("value", "delta"): (-100, 100),
    ("trans", "max_dx"): (0, 100),
    ("trans", "max_dy"): (0, 100),
    ("rotate", "max_angle"): (0, 45),
    ("gauss", "mean"): (0, 255),
    ("gauss", "sigma"): (1, 100),
    ("saltpepper", "prob"): (0.0, 0.5),
}


def test_meta_ranges_stay_within_transform_clamps():
    for (key, name), (lo, hi) in _TRANSFORM_CLAMPS.items():
        spec = next(s for s in it.params_of(key) if s[0] == name)
        assert lo <= spec[2], f"{key}.{name} 下限 {spec[2]} 低于夹取下限 {lo}"
        assert spec[3] <= hi, f"{key}.{name} 上限 {spec[3]} 高于夹取上限 {hi}"


def test_build_transform_rejects_unknown():
    with pytest.raises(ValueError):
        it.build_transform("nope", {})


def test_build_transform_rejects_unknown_param():
    with pytest.raises(ValueError):
        it.build_transform("bright", {"nosuch": 1})


def test_build_transform_injects_seed():
    a = it.build_transform("gauss", {"sigma": 10}, seed=1)
    b = it.build_transform("gauss", {"sigma": 10}, seed=1)
    assert a.rng.random() == b.rng.random()


def test_apply_chain_is_sequential_and_masked():
    out = _run_real_qt('''
import numpy as np
from labelforge.engine import image_tools as it
from labelforge.engine.augmenter.base import (
    rgb_array_to_qimage, qimage_to_rgb_array)

arr = np.full((8, 8, 3), 100, np.uint8)
img = rgb_array_to_qimage(arr)
out = it.apply_chain(img, [
    {"name": "bright", "params": {"delta": 50}},
    {"name": "bright", "params": {"delta": -50}},
], seed=0)
assert int(qimage_to_rgb_array(out)[0, 0, 0]) == 100

mask_arr = np.zeros((8, 8), np.uint8)
mask_arr[:, :4] = 255
mask = rgb_array_to_qimage(np.repeat(mask_arr[:, :, None], 3, axis=2))
res = it.apply_chain_with_mask(img, [{"name": "invert", "params": {}}], mask)
a = qimage_to_rgb_array(res)
assert a[0, 0, 0] == 155, a[0, 0, 0]
assert a[0, 7, 0] == 100, a[0, 7, 0]
print("OK")
''')
    assert "OK" in out


def test_empty_chain_returns_input_unchanged():
    out = _run_real_qt('''
import numpy as np
from labelforge.engine import image_tools as it
from labelforge.engine.augmenter.base import (
    rgb_array_to_qimage, qimage_to_rgb_array)
img = rgb_array_to_qimage(np.full((4, 4, 3), 33, np.uint8))
same = it.apply_chain(img, [])
assert qimage_to_rgb_array(same)[0, 0, 0] == 33
print("OK")
''')
    assert "OK" in out


def test_downsample_caps_long_side():
    out = _run_real_qt('''
from PySide6.QtGui import QImage
from labelforge.engine import image_tools as it
img = QImage(4000, 2000, QImage.Format_RGB888)
img.fill(0)
small = it.downsample_for_preview(img, 1000)
assert max(small.width(), small.height()) <= 1000, small.size()
assert small.height() == 500, small.height()
print("OK")
''')
    assert "OK" in out


def test_mask_from_zero_area_points_is_all_black():
    out = _run_real_qt('''
from labelforge.engine import image_tools as it
m = it.selection_mask_from_points([(0.5, 0.5)], 100, 100, "rect")
assert m.pixelColor(0, 0).red() == 0
assert m.pixelColor(50, 50).red() == 0
print("OK")
''')
    assert "OK" in out


def test_mask_from_two_points_covers_the_rect():
    out = _run_real_qt('''
from labelforge.engine import image_tools as it
m = it.selection_mask_from_points([(0.0, 0.0), (0.5, 0.5)], 100, 100, "rect")
assert m.pixelColor(10, 10).red() == 255
assert m.pixelColor(80, 80).red() == 0
e = it.selection_mask_from_points([(0.0, 0.0), (0.9, 0.9)], 100, 100, "ellipse")
assert e.pixelColor(50, 50).red() == 255
assert e.pixelColor(2, 2).red() == 0
p = it.selection_mask_from_points(
    [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)], 100, 100, "polygon")
assert p.pixelColor(50, 40).red() == 255
assert p.pixelColor(2, 98).red() == 0
print("OK")
''')
    assert "OK" in out


def test_grayscale_and_alpha_input_survive_the_chain():
    """Review Focus：灰度图与带 alpha 的 PNG 都要能被处理。"""
    out = _run_real_qt('''
import numpy as np
from PySide6.QtGui import QImage
from labelforge.engine import image_tools as it
from labelforge.engine.augmenter.base import qimage_to_rgb_array

gray = QImage(8, 8, QImage.Format_Grayscale8)
gray.fill(120)
res = it.apply_chain(gray, [{"name": "bright", "params": {"delta": 10}}])
assert qimage_to_rgb_array(res).shape == (8, 8, 3)
assert int(qimage_to_rgb_array(res)[0, 0, 0]) == 130

alpha = QImage(8, 8, QImage.Format_ARGB32)
alpha.fill(0x80FFFFFF)
res2 = it.apply_chain(alpha, [{"name": "invert", "params": {}}])
assert qimage_to_rgb_array(res2)[0, 0, 0] == 0
print("OK")
''')
    assert "OK" in out


def test_one_pixel_image_survives_every_operation():
    """Review Focus：1×1 图不能被卷积核、透视或缩放打崩。"""
    out = _run_real_qt('''
import numpy as np
from labelforge.engine import image_tools as it
from labelforge.engine.augmenter.base import (
    rgb_array_to_qimage, qimage_to_rgb_array)
img = rgb_array_to_qimage(np.full((1, 1, 3), 90, np.uint8))
bad = []
for key in it.OPERATION_ORDER:
    try:
        t = it.build_transform(key, it.default_params(key), seed=0)
        res, boxes = t.apply(img, [], 1, 1)
        if res.isNull() or boxes != []:
            bad.append(key)
    except Exception as exc:
        bad.append("%s: %s" % (key, exc))
assert not bad, bad
print("OK")
''')
    assert "OK" in out
