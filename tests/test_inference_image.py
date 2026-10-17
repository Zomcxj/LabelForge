"""QImage ↔ 推理内核的桥接测试。

`tests/conftest.py` 把 PySide6 换成了 mock，像素与步长必须在真实 Qt 里跑，
所以断言走 offscreen 子进程（与 test_augmenter 同套路）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死；
# 断言全部通过、标记打印之后立即 flush + os._exit(0) 绕开析构阶段。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_real_qt(script, timeout=120):
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        r = subprocess.run(
            [sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=ROOT, env=env,
            text=True, capture_output=True, timeout=timeout,
            encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"Qt 子进程 {timeout}s 超时\nstdout: {e.stdout}") from None
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_qimage_to_rgb_array_strips_stride_padding():
    """宽度非 4 倍数时 Qt 每行有填充，取图必须剥掉，否则逐行错位。"""
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.inference_image import qimage_to_rgb_array

# 7px 宽 ARGB32：bytesPerLine=28 无填充；用 5px 宽 RGB888 逼出对齐填充
img = QImage(5, 3, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
img.setPixelColor(4, 2, QColor(10, 20, 30))

arr = qimage_to_rgb_array(img)
assert arr.shape == (3, 5, 3), arr.shape
assert arr.dtype.name == "uint8"
# 关键断言：最后一行最后一列是设置的颜色，说明没有错位
assert arr[2, 4].tolist() == [10, 20, 30], arr[2, 4]
assert arr[0, 0].tolist() == [0, 0, 0]
print("OK_STRIDE")
''')
    assert "OK_STRIDE" in out


def test_qimage_to_rgb_array_converts_argb32_to_rgb_channel_order():
    """显示缓冲是 ARGB32（小端 BGRA），取图必须转成 RGB 而不是直接读内存。"""
    out = _run_real_qt('''
from PySide6.QtGui import QImage, QColor
from labelforge.engine.inference_image import qimage_to_rgb_array

img = QImage(4, 2, QImage.Format_ARGB32)
img.fill(QColor(255, 128, 64))

arr = qimage_to_rgb_array(img)
assert arr.shape == (2, 4, 3), arr.shape
assert arr[0, 0].tolist() == [255, 128, 64], arr[0, 0]
print("OK_ORDER")
''')
    assert "OK_ORDER" in out


def test_qimage_to_rgb_array_returns_none_for_null_image():
    out = _run_real_qt('''
from PySide6.QtGui import QImage
from labelforge.engine.inference_image import qimage_to_rgb_array

assert qimage_to_rgb_array(None) is None
assert qimage_to_rgb_array(QImage()) is None
print("OK_NULL")
''')
    assert "OK_NULL" in out


def test_letterbox_to_array_pads_with_fill_and_keeps_size():
    out = _run_real_qt('''
import numpy as np
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import letterbox_to_array

# 100x50 → 64x64：scale=0.64，缩放后 64x32，上下各补 16
g = letterbox_geometry(100, 50, 64, 64)
rgb = np.zeros((50, 100, 3), dtype=np.uint8)
rgb[:, :] = (200, 100, 50)

canvas = letterbox_to_array(rgb, g, LETTERBOX_FILL)
assert canvas.shape == (64, 64, 3), canvas.shape
assert canvas[0, 0].tolist() == [114, 114, 114], canvas[0, 0]
assert canvas[63, 63].tolist() == [114, 114, 114]
# 内容区域是原图颜色（缩放后仍是纯色）
assert canvas[32, 32].tolist() == [200, 100, 50], canvas[32, 32]
# 补边边界：pad_y-1 是灰，pad_y 起是内容
assert canvas[15, 32].tolist() == [114, 114, 114]
assert canvas[16, 32].tolist() == [200, 100, 50]
print("OK_PAD")
''')
    assert "OK_PAD" in out


def test_letterbox_to_array_no_resize_when_already_model_size():
    out = _run_real_qt('''
import numpy as np
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import letterbox_to_array

g = letterbox_geometry(32, 32, 32, 32)
rgb = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)

canvas = letterbox_to_array(rgb, g, LETTERBOX_FILL)

# 不缩放时必须是逐像素原样拷贝，不能有重采样插值
assert np.array_equal(canvas, rgb), "同尺寸不应重采样"
print("OK_SAME")
''')
    assert "OK_SAME" in out


def test_letterbox_canvas_matches_geometry_for_odd_ratios():
    """画布尺寸必须严格等于 geometry.dst_*，否则送模型前张量形状就错了。"""
    out = _run_real_qt('''
import numpy as np
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import letterbox_to_array

for src_w, src_h in [(1000, 333), (333, 1000), (7, 5), (1, 1), (1920, 1080)]:
    g = letterbox_geometry(src_w, src_h, 64, 64)
    rgb = np.full((src_h, src_w, 3), 7, dtype=np.uint8)
    canvas = letterbox_to_array(rgb, g, LETTERBOX_FILL)
    assert canvas.shape == (g.dst_h, g.dst_w, 3), (src_w, src_h, g, canvas.shape)
    # 内容区必须完整落在画布内
    assert g.pad_x + g.new_w <= g.dst_w and g.pad_y + g.new_h <= g.dst_h, g
print("OK_ALL_SIZES")
''')
    assert "OK_ALL_SIZES" in out


def test_resize_for_model_is_usable_as_detector_resize_fn():
    """桥接函数必须能被 OnnxDetector 直接当作 resize_fn 使用。"""
    out = _run_real_qt('''
import numpy as np
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import resize_for_model

g = letterbox_geometry(80, 40, 64, 64)
rgb = np.zeros((40, 80, 3), dtype=np.uint8)
canvas = resize_for_model(rgb, g, LETTERBOX_FILL)
assert canvas.shape == (64, 64, 3), canvas.shape
# 形状可直接转成 NCHW 张量
t = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
assert t.shape == (1, 3, 64, 64), t.shape
print("OK_RESIZE_FN")
''')
    assert "OK_RESIZE_FN" in out


def test_letterbox_to_array_matches_reference_geometry():
    """画布内容必须与几何参数一致：内容块位置/尺寸和 geometry 完全对应。

    绘制与逆变换各自实现，一旦取整方式不同，坐标就会系统性偏移；
    这里用非对称图（左上角标红）钉住内容块的落点。
    """
    out = _run_real_qt('''
import numpy as np
from PySide6.QtGui import QImage, QColor
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import qimage_to_rgb_array, letterbox_to_array

# 100x50 的图，左上角 10x5 红块
img = QImage(100, 50, QImage.Format_RGB888)
img.fill(QColor(0, 0, 0))
for y in range(5):
    for x in range(10):
        img.setPixelColor(x, y, QColor(255, 0, 0))
rgb = qimage_to_rgb_array(img)

g = letterbox_geometry(100, 50, 64, 64)
canvas = letterbox_to_array(rgb, g, LETTERBOX_FILL)

# 补边区域严格是填充色
assert canvas[0, 32].tolist() == [114, 114, 114], canvas[0, 32]
assert canvas[g.pad_y - 1, 32].tolist() == [114, 114, 114], g
# 内容区第一行第一列必须是原图左上角（红），说明落点没偏移
assert canvas[g.pad_y, g.pad_x].tolist() == [255, 0, 0], canvas[g.pad_y, g.pad_x]
# 内容区右下角之后回到填充色（该方向确实存在补边时）
if g.pad_y + g.new_h < g.dst_h:
    assert canvas[g.pad_y + g.new_h, 32].tolist() == [114, 114, 114], g
if g.pad_x + g.new_w < g.dst_w:
    assert canvas[g.pad_y, g.pad_x + g.new_w].tolist() == [114, 114, 114], g
print("OK_ALIGN")
''')
    assert "OK_ALIGN" in out


def test_letterbox_to_array_scales_non_square_input_preserving_aspect():
    """等比缩放：4:1 的图缩到 64x64 画布后内容高度应为 16，上下各补 24。"""
    out = _run_real_qt('''
import numpy as np
from PySide6.QtGui import QImage, QColor
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import qimage_to_rgb_array, letterbox_to_array

img = QImage(256, 64, QImage.Format_RGB888)
img.fill(QColor(10, 20, 30))
rgb = qimage_to_rgb_array(img)
g = letterbox_geometry(256, 64, 64, 64)
assert (g.new_w, g.new_h, g.pad_y) == (64, 16, 24), g

canvas = letterbox_to_array(rgb, g, LETTERBOX_FILL)
assert canvas.shape == (64, 64, 3)
# 内容区整块都是原色（原图纯色，缩放后仍纯色）
assert np.all(canvas[24:40, :] == [10, 20, 30]), canvas[24:40, :].reshape(-1, 3)[:3]
# 补边区
assert np.all(canvas[:24] == LETTERBOX_FILL)
assert np.all(canvas[40:] == LETTERBOX_FILL)
print("OK_ASPECT")
''')
    assert "OK_ASPECT" in out


def test_qimage_and_letterbox_roundtrip_through_detector_resize_fn():
    """生产路径：QImage → RGB → letterbox → NCHW 张量形状正确。"""
    out = _run_real_qt('''
import numpy as np
from PySide6.QtGui import QImage, QColor
from labelforge.engine.onnx_inference import letterbox_geometry, LETTERBOX_FILL
from labelforge.engine.inference_image import qimage_to_rgb_array, resize_for_model

img = QImage(256, 64, QImage.Format_ARGB32)
img.fill(QColor(200, 100, 50))
rgb = qimage_to_rgb_array(img)
assert rgb.shape == (64, 256, 3)
assert rgb[0, 0].tolist() == [200, 100, 50]

g = letterbox_geometry(256, 64, 64, 64)
canvas = resize_for_model(rgb, g, LETTERBOX_FILL)
tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
assert tensor.shape == (1, 3, 64, 64), tensor.shape
assert tensor.min() >= 0.0 and tensor.max() <= 1.0
print("OK_TENSOR")
''')
    assert "OK_TENSOR" in out
