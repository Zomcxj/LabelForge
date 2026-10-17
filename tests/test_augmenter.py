"""数据增强测试。

`tests/conftest.py` 把 PySide6 换成了 mock，像素级变换必须在真实 Qt 里跑，
所以图像相关断言走 offscreen 子进程（与 test_theme_style_consolidation 同套路）；
纯参数构造与 JSON 序列化可以直接测。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.base import BaseTransform, get_all_transforms
from labelforge.engine.augmenter.color import (
    Brightness, Contrast, Hue, Saturation, Value,
)
from labelforge.engine.augmenter.flipt import HorizontalFlip, VerticalFlip
from labelforge.engine.augmenter.noise import GaussianNoise, SaltPepper
from labelforge.engine.augmenter.rotate import RandomRotation
from labelforge.engine.augmenter.scale import RandomScale
from labelforge.engine.augmenter.translate import RandomTranslate

ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死（exit code 0xC0000409 /
# 0xC0000005，或无限超时）。断言全部通过、标记打印之后立即 flush + os._exit(0)，
# 绕开不可控的析构阶段。os._exit 不冲刷缓冲，必须先 flush，否则标记会丢。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_real_qt(script):
    """在真实 Qt（offscreen）子进程里跑断言，返回 stdout。"""
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
    }
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


# ---------- 注册表 ----------

def test_all_twelve_transforms_are_registered():
    names = get_all_transforms()
    assert sorted(names) == [
        "bright", "contrast", "fliph", "flipv", "gauss", "hue",
        "rotate", "saltpepper", "saturation", "scale", "trans", "value",
    ]


def test_registered_classes_subclass_base_transform():
    for name, cls in get_all_transforms().items():
        assert issubclass(cls, BaseTransform), name
        assert cls is not BaseTransform
        assert cls.name == name


# ---------- 参数构造（纯逻辑）----------

def test_build_kwargs_passes_rng_to_transform():
    aug = Augmenter("out", seed=1)

    kwargs = aug._build_kwargs(Brightness, {"delta": (10, 40)}, "all")

    assert kwargs["rng"] is aug._rng


def test_build_kwargs_is_reproducible_for_same_seed():
    a = Augmenter("out", seed=7)._build_kwargs(Brightness, {"delta": (10, 40)}, "all")
    b = Augmenter("out", seed=7)._build_kwargs(Brightness, {"delta": (10, 40)}, "all")

    assert a["delta"] == b["delta"]


def test_build_kwargs_maps_angle_to_max_angle():
    kwargs = Augmenter("out", seed=1)._build_kwargs(
        RandomRotation, {"angle": (5, 20)}, "all")

    assert "angle" not in kwargs
    assert 5 <= kwargs["max_angle"] <= 20


def test_build_kwargs_maps_scale_to_min_max_around_one():
    kwargs = Augmenter("out", seed=1)._build_kwargs(
        RandomScale, {"scale": (0.8, 1.2)}, "all")

    assert "scale" not in kwargs
    assert kwargs["min"] <= 1.0 <= kwargs["max"]


def test_build_kwargs_maps_offset_to_dx_dy():
    kwargs = Augmenter("out", seed=1)._build_kwargs(
        RandomTranslate, {"offset": (5, 20)}, "all")

    assert "offset" not in kwargs
    assert kwargs["max_dx"] == kwargs["max_dy"]
    assert 5 <= kwargs["max_dx"] <= 20


def test_build_kwargs_truncates_delta_and_sigma_to_int():
    delta = Augmenter("out", seed=1)._build_kwargs(Brightness, {"delta": (10, 40)}, "all")
    sigma = Augmenter("out", seed=1)._build_kwargs(GaussianNoise, {"sigma": (10, 30)}, "all")

    assert isinstance(delta["delta"], int)
    assert isinstance(sigma["sigma"], int)


def test_build_kwargs_keeps_float_factor_for_contrast():
    kwargs = Augmenter("out", seed=1)._build_kwargs(
        Contrast, {"factor": (1.1, 1.5)}, "all")

    assert isinstance(kwargs["factor"], float)


# ---------- LabelMe JSON 序列化（纯逻辑）----------

def test_save_labelme_json_writes_rectangle_shape(tmp_path):
    aug = Augmenter(str(tmp_path))
    out = tmp_path / "a.json"

    aug._save_labelme_json(str(out), "a.png",
                           [{"label": "cat", "x": 1, "y": 2,
                             "width": 3, "height": 4}], 100, 50)

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["imagePath"] == "a.png"
    assert data["imageWidth"] == 100
    assert data["imageHeight"] == 50
    shape = data["shapes"][0]
    assert shape["label"] == "cat"
    assert shape["shape_type"] == "rectangle"
    assert shape["points"] == [[1, 2], [4, 2], [4, 6], [1, 6]]


def test_save_labelme_json_handles_no_boxes(tmp_path):
    aug = Augmenter(str(tmp_path))
    out = tmp_path / "empty.json"

    aug._save_labelme_json(str(out), "a.png", [], 10, 10)

    assert json.loads(out.read_text(encoding="utf-8"))["shapes"] == []


def test_save_labelme_json_preserves_unicode_labels(tmp_path):
    aug = Augmenter(str(tmp_path))
    out = tmp_path / "u.json"

    aug._save_labelme_json(str(out), "a.png",
                           [{"label": "猫", "x": 0, "y": 0,
                             "width": 1, "height": 1}], 10, 10)

    assert json.loads(out.read_text(encoding="utf-8"))["shapes"][0]["label"] == "猫"


# ---------- 图像变换（真实 Qt）----------

def test_horizontal_and_vertical_flip_mirror_boxes():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.flipt import HorizontalFlip, VerticalFlip

img = QImage(100, 50, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
boxes = [{"label": "cat", "x": 10, "y": 5, "width": 20, "height": 10}]

_, hb = HorizontalFlip().apply(img, boxes, 100, 50)
assert hb[0]["x"] == 70, hb
assert hb[0]["y"] == 5
assert hb[0]["width"] == 20 and hb[0]["height"] == 10
assert hb[0]["label"] == "cat"

_, vb = VerticalFlip().apply(img, boxes, 100, 50)
assert vb[0]["x"] == 10
assert vb[0]["y"] == 35, vb
assert vb[0]["width"] == 20 and vb[0]["height"] == 10

# 翻转两次应回到原状
_, back = HorizontalFlip().apply(img, hb, 100, 50)
assert back[0]["x"] == 10, back
print("OK")
''')
    assert "OK" in out


def test_translate_clips_boxes_and_drops_degenerate():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.translate import RandomTranslate

img = QImage(100, 100, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))

# max_dx=max_dy=0 => 不平移；越界框应被裁剪
boxes = [{"label": "cat", "x": -20, "y": 50, "width": 40, "height": 40},
         {"label": "dog", "x": 95, "y": 0, "width": 40, "height": 40}]
_, out_boxes = RandomTranslate(max_dx=0, max_dy=0).apply(img, boxes, 100, 100)
assert len(out_boxes) == 2, out_boxes
assert out_boxes[0]["x"] == 0 and out_boxes[0]["width"] == 20, out_boxes[0]
assert out_boxes[1]["x"] == 95 and out_boxes[1]["width"] == 5, out_boxes[1]

# 完全在画布外的框必须丢弃
outside = [{"label": "cat", "x": 200, "y": 200, "width": 10, "height": 10}]
_, dropped = RandomTranslate(max_dx=0, max_dy=0).apply(img, outside, 100, 100)
assert dropped == [], dropped
print("OK")
''')
    assert "OK" in out


def test_brightness_adds_delta_and_clamps():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.color import Brightness

img = QImage(4, 1, QImage.Format_RGB888)
img.setPixelColor(0, 0, QColor(10, 10, 10))
img.setPixelColor(1, 0, QColor(250, 250, 250))
img.setPixelColor(2, 0, QColor(100, 100, 100))
img.setPixelColor(3, 0, QColor(128, 128, 128))

res, boxes = Brightness(delta=30).apply(img, [], 4, 1)
assert res.pixelColor(0, 0).red() == 40, res.pixelColor(0, 0).red()
assert res.pixelColor(1, 0).red() == 255, "应在上界截断"
assert res.pixelColor(2, 0).red() == 130
assert boxes == []

neg, _ = Brightness(delta=-30).apply(img, [], 4, 1)
assert neg.pixelColor(0, 0).red() == 0, "应在下界截断"
print("OK")
''')
    assert "OK" in out


def test_contrast_scales_around_midpoint():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.color import Contrast

img = QImage(3, 1, QImage.Format_RGB888)
img.setPixelColor(0, 0, QColor(128, 128, 128))
img.setPixelColor(1, 0, QColor(160, 160, 160))
img.setPixelColor(2, 0, QColor(96, 96, 96))

res, _ = Contrast(factor=1.5).apply(img, [], 3, 1)
assert res.pixelColor(0, 0).red() == 128, "中点不应变"
assert res.pixelColor(1, 0).red() == 176, res.pixelColor(1, 0).red()
assert res.pixelColor(2, 0).red() == 80, res.pixelColor(2, 0).red()
print("OK")
''')
    assert "OK" in out


def test_saturation_and_value_keep_boxes_untouched():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.color import Saturation, Value, Hue

img = QImage(2, 2, QImage.Format_RGB888)
img.fill(QColor(200, 100, 50))
boxes = [{"label": "cat", "x": 1, "y": 1, "width": 1, "height": 1}]

for t in (Saturation(factor=0.5), Value(delta=-50), Hue(delta=90)):
    _, out_boxes = t.apply(img, boxes, 2, 2)
    assert out_boxes == boxes, (type(t).__name__, out_boxes)

# 灰度图（饱和度为 0）不应崩
gray = QImage(2, 2, QImage.Format_RGB888)
gray.fill(QColor(128, 128, 128))
Saturation(factor=1.5).apply(gray, [], 2, 2)
print("OK")
''')
    assert "OK" in out


def test_gaussian_noise_is_deterministic_with_injected_rng():
    out = _run_real_qt('''
import random
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.noise import GaussianNoise

img = QImage(8, 8, QImage.Format_RGB888)
img.fill(QColor(100, 100, 100))

a, _ = GaussianNoise(sigma=20, rng=random.Random(5)).apply(img, [], 8, 8)
b, _ = GaussianNoise(sigma=20, rng=random.Random(5)).apply(img, [], 8, 8)
c, _ = GaussianNoise(sigma=20, rng=random.Random(6)).apply(img, [], 8, 8)

pixels = lambda im: [im.pixelColor(x, y).red() for y in range(8) for x in range(8)]
assert pixels(a) == pixels(b), "同 seed 必须一致"
assert pixels(a) != pixels(c), "不同 seed 应不同"
print("OK")
''')
    assert "OK" in out


def test_salt_pepper_is_deterministic_with_injected_rng():
    out = _run_real_qt('''
import random
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.noise import SaltPepper

img = QImage(20, 20, QImage.Format_RGB888)
img.fill(QColor(100, 100, 100))

a, _ = SaltPepper(prob=0.2, rng=random.Random(3)).apply(img, [], 20, 20)
b, _ = SaltPepper(prob=0.2, rng=random.Random(3)).apply(img, [], 20, 20)
c, _ = SaltPepper(prob=0.2, rng=random.Random(4)).apply(img, [], 20, 20)

pixels = lambda im: [im.pixelColor(x, y).red() for y in range(20) for x in range(20)]
assert pixels(a) == pixels(b)
assert pixels(a) != pixels(c)
vals = set(pixels(a))
assert vals <= {0, 100, 255}, vals
print("OK")
''')
    assert "OK" in out


def test_rotation_enlarges_canvas_and_transforms_boxes():
    out = _run_real_qt('''
import random
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.rotate import RandomRotation

img = QImage(100, 100, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
boxes = [{"label": "cat", "x": 40, "y": 40, "width": 20, "height": 20}]

res, out_boxes = RandomRotation(max_angle=30, rng=random.Random(1)).apply(img, boxes, 100, 100)
assert res.width() >= 100 and res.height() >= 100, (res.width(), res.height())
assert len(out_boxes) == 1
b = out_boxes[0]
assert b["width"] >= 20 and b["height"] >= 20, b
assert b["label"] == "cat"

# 0 度旋转应基本保持原框
res0, b0 = RandomRotation(max_angle=0, rng=random.Random(1)).apply(img, boxes, 100, 100)
assert abs(b0[0]["x"] - 40) < 1e-6 and abs(b0[0]["width"] - 20) < 1e-6, b0
print("OK")
''')
    assert "OK" in out


def test_scale_shrinks_centers_and_keeps_canvas_size():
    out = _run_real_qt('''
import random
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.scale import RandomScale

img = QImage(100, 100, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
boxes = [{"label": "cat", "x": 10, "y": 10, "width": 20, "height": 20}]

# min=max 固定缩放比，便于断言
small, sb = RandomScale(min=0.5, max=0.5, rng=random.Random(1)).apply(img, boxes, 100, 100)
assert small.width() == 100 and small.height() == 100, (small.width(), small.height())
assert sb[0]["width"] == 10 and sb[0]["height"] == 10, sb
assert sb[0]["x"] == 5 + 25, sb

big, bb = RandomScale(min=2.0, max=2.0, rng=random.Random(1)).apply(img, boxes, 100, 100)
assert big.width() == 200 and big.height() == 200, (big.width(), big.height())
assert bb[0]["x"] == 20 and bb[0]["width"] == 40, bb
print("OK")
''')
    assert "OK" in out


# ---------- 端到端可复现性（真实 Qt）----------

def test_augmenter_is_reproducible_for_same_seed_in_all_mode():
    out = _run_real_qt('''
import hashlib, json, os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness
from labelforge.engine.augmenter.rotate import RandomRotation
from labelforge.engine.augmenter.scale import RandomScale

src_dir = tempfile.mkdtemp()
img = QImage(64, 48, QImage.Format_RGB888)
img.fill(QColor(120, 130, 140))
src = os.path.join(src_dir, "a.png")
img.save(src)
boxes = {0: [{"label": "cat", "x": 5, "y": 5, "width": 20, "height": 15}]}
specs = [(Brightness, {"delta": (10, 40)}),
         (RandomRotation, {"angle": (5, 20)}),
         (RandomScale, {"scale": (0.8, 1.2)})]

def run(seed):
    out = tempfile.mkdtemp()
    res = Augmenter(out, seed=seed).run([src], boxes, specs, mode="all",
                                        image_ratio=1.0, skip_empty=False)
    payload = json.dumps([{"stem": r["stem"], "boxes": r["boxes"]} for r in res],
                         sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()

assert run(1) == run(1), "同 seed 必须一致"
assert run(1) != run(2), "不同 seed 应不同"
print("OK")
''')
    assert "OK" in out


def test_augmenter_does_not_pollute_global_random():
    """seed 必须只作用于内部 RNG，不能影响调用方的全局 random 状态。"""
    out = _run_real_qt('''
import os, random, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness

src_dir = tempfile.mkdtemp()
img = QImage(32, 32, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
src = os.path.join(src_dir, "a.png")
img.save(src)
boxes = {0: [{"label": "cat", "x": 1, "y": 1, "width": 5, "height": 5}]}

random.seed(7)
before = random.random()
random.seed(7)
Augmenter(tempfile.mkdtemp(), seed=99).run(
    [src], boxes, [(Brightness, {"delta": (10, 40)})],
    mode="all", image_ratio=1.0, skip_empty=False)
after = random.random()
assert before == after, (before, after)
print("OK")
''')
    assert "OK" in out


def test_augmenter_run_respects_skip_empty_and_include_original():
    out = _run_real_qt('''
import os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness

src_dir = tempfile.mkdtemp()
img = QImage(32, 32, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
src = os.path.join(src_dir, "a.png")
img.save(src)

specs = [(Brightness, {"delta": (10, 40)})]

# skip_empty=True：没有框的图片应被跳过
out1 = tempfile.mkdtemp()
res1 = Augmenter(out1, seed=1).run([src], {0: []}, specs, mode="all",
                                   image_ratio=1.0, skip_empty=True)
assert res1 == [], res1
assert os.listdir(os.path.join(out1, "images")) == [], os.listdir(os.path.join(out1, "images"))

# include_original=True：额外产出一份原图 + json
out2 = tempfile.mkdtemp()
res2 = Augmenter(out2, seed=1).run([src], {0: [{"label": "cat", "x": 1, "y": 1,
                                                "width": 5, "height": 5}]},
                                   specs, mode="all", image_ratio=1.0,
                                   skip_empty=False, include_original=True)
stems = sorted(r["stem"] for r in res2)
assert stems == ["a", "a_bright"], stems
files = sorted(f for f in os.listdir(os.path.join(out2, "images"))
               if f.endswith(".png"))
assert files == ["a.png", "a_bright.png"], files
assert os.path.exists(os.path.join(out2, "images", "a.json"))
assert os.path.exists(os.path.join(out2, "images", "a_bright.json"))
print("OK")
''')
    assert "OK" in out


def test_augmenter_image_ratio_zero_skips_random_transforms():
    out = _run_real_qt('''
import os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness

src_dir = tempfile.mkdtemp()
img = QImage(32, 32, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
src = os.path.join(src_dir, "a.png")
img.save(src)
boxes = {0: [{"label": "cat", "x": 1, "y": 1, "width": 5, "height": 5}]}

# random 模式 + image_ratio=0 => 没有任何变换生效，回落为保存原图
out = tempfile.mkdtemp()
res = Augmenter(out, seed=1).run([src], boxes, [(Brightness, {"delta": (10, 40)})],
                                 mode="random", image_ratio=0.0, skip_empty=False)
assert [r["stem"] for r in res] == ["a"], res
print("OK")
''')
    assert "OK" in out


def test_augmenter_stops_when_interrupted():
    out = _run_real_qt('''
import os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness

src_dir = tempfile.mkdtemp()
img = QImage(32, 32, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
src = os.path.join(src_dir, "a.png")
img.save(src)
boxes = {0: [{"label": "cat", "x": 1, "y": 1, "width": 5, "height": 5}]}

out = tempfile.mkdtemp()
res = Augmenter(out, seed=1, is_interrupted=lambda: True).run(
    [src], boxes, [(Brightness, {"delta": (10, 40)})],
    mode="all", image_ratio=1.0, skip_empty=False)
assert res == [], res
print("OK")
''')
    assert "OK" in out


# ---------- 形状感知（保留 OBB/多边形/关键点/分组）----------

def test_transforms_preserve_shape_metadata():
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter.flipt import HorizontalFlip
from labelforge.engine.augmenter.scale import RandomScale
from labelforge.engine.augmenter.translate import RandomTranslate

img = QImage(100, 100, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))

rot = {"label": "car", "shape_type": "rotation", "group_id": 2,
       "x": 10, "y": 10, "width": 20, "height": 10,
       "points": [[10, 10], [30, 10], [30, 20], [10, 20]]}
poly = {"label": "road", "shape_type": "polygon", "group_id": None,
        "x": 5, "y": 5, "width": 40, "height": 30,
        "points": [[5, 5], [45, 5], [30, 35]]}
pt = {"label": "nose", "shape_type": "point", "group_id": 1,
      "x": 50, "y": 60, "width": 0, "height": 0, "points": [[50, 60]]}

_, flipped = HorizontalFlip().apply(img, [dict(rot), dict(poly), dict(pt)], 100, 100)
fr, fp, fpt = flipped
assert fr["shape_type"] == "rotation" and fr["group_id"] == 2, fr
assert fr["points"] == [[90, 10], [70, 10], [70, 20], [90, 20]], fr["points"]
assert fr["x"] == 70 and fr["width"] == 20, fr
assert fp["shape_type"] == "polygon" and len(fp["points"]) == 3, fp
assert fpt["shape_type"] == "point" and fpt["points"] == [[50, 60]], fpt
assert fpt["x"] == 50 and fpt["y"] == 60, fpt

_, scaled = RandomScale(min=0.5, max=0.5).apply(
    img, [dict(rot)], 100, 100)
s = scaled[0]
assert s["shape_type"] == "rotation" and s["group_id"] == 2, s
assert s["points"] == [[30, 30], [40, 30], [40, 35], [30, 35]], s["points"]

_, moved = RandomTranslate(max_dx=0, max_dy=0).apply(img, [dict(rot)], 100, 100)
m = moved[0]
assert m["shape_type"] == "rotation" and m["points"] == rot["points"], m
print("OK")
''')
    assert "OK" in out


def test_augmented_labelme_json_roundtrips_shape_types(tmp_path):
    out = _run_real_qt('''
import json, os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness
from labelforge.engine.shape_io import box_from_labelme_shape

src_dir = tempfile.mkdtemp()
img = QImage(80, 60, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
src = os.path.join(src_dir, "a.png")
img.save(src)

boxes = {0: [
    {"label": "car", "shape_type": "rotation", "group_id": 2,
     "x": 10, "y": 10, "width": 20, "height": 10,
     "points": [[10, 10], [30, 10], [30, 20], [10, 20]]},
    {"label": "nose", "shape_type": "point", "group_id": 1,
     "x": 40, "y": 40, "width": 0, "height": 0, "points": [[40, 40]]},
    {"label": "road", "shape_type": "polygon", "group_id": None,
     "x": 5, "y": 5, "width": 30, "height": 30,
     "points": [[5, 5], [35, 5], [20, 35]]},
]}
out_dir = tempfile.mkdtemp()
Augmenter(out_dir, seed=1).run(
    [src], boxes, [(Brightness, {"delta": (10, 40)})],
    mode="all", image_ratio=1.0, skip_empty=False)

jp = os.path.join(out_dir, "images", "a_bright.json")
with open(jp, encoding="utf-8") as f:
    shapes = json.load(f)["shapes"]
by_label = {s["label"]: s for s in shapes}
assert by_label["car"]["shape_type"] == "rotation", by_label["car"]
assert by_label["car"]["group_id"] == 2, by_label["car"]
assert by_label["nose"]["shape_type"] == "point", by_label["nose"]
assert by_label["nose"]["group_id"] == 1, by_label["nose"]
assert by_label["road"]["shape_type"] == "polygon", by_label["road"]
assert [list(p) for p in by_label["road"]["points"]] == [[5, 5], [35, 5], [20, 35]]

# 读回后 box 转换仍保留形状
back = {b["label"]: b for b in (box_from_labelme_shape(s) for s in shapes)}
assert back["car"]["shape_type"] == "rotation", back["car"]
assert back["nose"]["points"] == [[40, 40]], back["nose"]
print("OK")
''')
    assert "OK" in out


# ---------- 构造函数边界（纯逻辑）----------

@pytest.mark.parametrize("cls,attr,kwargs,expected", [
    (Brightness, "delta", {"delta": 999}, 100),
    (Brightness, "delta", {"delta": -999}, -100),
    (Contrast, "factor", {"factor": 99.0}, 2.0),
    (Contrast, "factor", {"factor": 0.0}, 0.5),
    (Hue, "delta", {"delta": 999}, 180),
    (Saturation, "factor", {"factor": 99.0}, 2.0),
    (GaussianNoise, "sigma", {"sigma": 999}, 100),
    (GaussianNoise, "sigma", {"sigma": 0}, 1),
    (SaltPepper, "prob", {"prob": 99.0}, 0.5),
    (RandomRotation, "max_angle", {"max_angle": 999}, 45),
    (RandomTranslate, "max_dx", {"max_dx": 999}, 100),
])
def test_transform_params_are_clamped(cls, attr, kwargs, expected):
    assert getattr(cls(**kwargs), attr) == expected


def test_transform_defaults_use_global_random_when_no_rng_given():
    """不注入 rng 时回退全局 random，保持既有调用方行为。"""
    import random

    assert Brightness().rng is random


# ---------- include_original 中断回归（当前应失败）----------

def test_augmenter_include_original_stops_when_interrupted():
    """include_original 循环必须响应中断：标志恒 True 时不应写入任何原图/JSON。"""
    out = _run_real_qt('''
import os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter

src_dir = tempfile.mkdtemp()
srcs = []
for i in range(4):
    img = QImage(32, 32, QImage.Format_RGB888)
    img.fill(QColor(10 * i, 20, 30))
    p = os.path.join(src_dir, f"img{i}.png")
    img.save(p)
    srcs.append(p)
boxes = {i: [{"label": "cat", "x": 1, "y": 1, "width": 5, "height": 5}]
         for i in range(4)}

out_dir = tempfile.mkdtemp()
res = Augmenter(out_dir, is_interrupted=lambda: True).run(
    srcs, boxes, [], mode="all", image_ratio=1.0,
    include_original=True, skip_empty=False)
assert res == [], res
files = os.listdir(os.path.join(out_dir, "images"))
assert files == [], files
print("OK")
''')
    assert "OK" in out


def test_augmenter_include_original_stops_midway_when_interrupted():
    """复制中途中断：第 3 次检查起返回 True 时，写入数量应小于总图数。"""
    out = _run_real_qt('''
import os, tempfile
from PySide6.QtGui import QImage, QColor
from labelforge.engine.augmenter import Augmenter

src_dir = tempfile.mkdtemp()
srcs = []
for i in range(4):
    img = QImage(32, 32, QImage.Format_RGB888)
    img.fill(QColor(10 * i, 20, 30))
    p = os.path.join(src_dir, f"img{i}.png")
    img.save(p)
    srcs.append(p)
boxes = {i: [{"label": "cat", "x": 1, "y": 1, "width": 5, "height": 5}]
         for i in range(4)}

state = {"checks": 0}
def is_interrupted():
    state["checks"] += 1
    return state["checks"] >= 3

out_dir = tempfile.mkdtemp()
res = Augmenter(out_dir, is_interrupted=is_interrupted).run(
    srcs, boxes, [], mode="all", image_ratio=1.0,
    include_original=True, skip_empty=False)
n_imgs = len([f for f in os.listdir(os.path.join(out_dir, "images"))
               if f.endswith(".png")])
assert n_imgs < len(srcs), (n_imgs, len(srcs))
assert len(res) < len(srcs), (len(res), len(srcs))
print("OK")
''')
    assert "OK" in out
