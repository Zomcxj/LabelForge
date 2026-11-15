"""贴图位图转换：RGBA <-> QImage/QPixmap 与 BGR 通道顺序。

转换函数要真 Qt（QPixmap/QImage 的格式常量），所以整段在子进程里跑。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUARD = "\nimport os, sys\nsys.stdout.flush()\nos._exit(0)\n"

_QT_EXIT_CRASH = {3221225477, 3221226505}


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    last = None
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", script + _GUARD],
            cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=180, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout
        last = result
        if result.returncode not in _QT_EXIT_CRASH:
            break
    assert last is not None and last.returncode == 0, (
        f"exit={None if last is None else last.returncode} "
        + (last.stderr if last is not None else "子进程没有启动"))
    return last.stdout


_SETUP = '''
import numpy as np
from PySide6.QtWidgets import QApplication
from labelforge.engine.paste_fusion import (
    bgr_to_qimage, pixmap_to_rgba, qimage_to_bgr, rgba_to_pixmap,
)

app = QApplication([])


def solid_rgba(h=8, w=6):
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 1] = 30
    rgba[..., 2] = 60
    rgba[..., 3] = 255
    return rgba
'''


def test_rgba_to_pixmap_keeps_size_and_pixels():
    out = _run_real_qt(_SETUP + '''
rgba = solid_rgba()
pixmap = rgba_to_pixmap(rgba)
print(pixmap.width(), pixmap.height())
print(np.array_equal(pixmap_to_rgba(pixmap, 6, 8), rgba))
''')
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0] == "6 8", lines[0]
    assert lines[1] == "True", lines[1]


def test_pixmap_to_rgba_scales_to_the_requested_size():
    out = _run_real_qt(_SETUP + '''
pixmap = rgba_to_pixmap(solid_rgba(8, 6))
scaled = pixmap_to_rgba(pixmap, 12, 16)
print(scaled.shape)
print(tuple(int(v) for v in scaled[0, 0][:3]))
''')
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0] == "(16, 12, 4)", lines[0]
    assert lines[1] == "(200, 30, 60)", lines[1]


def test_bgr_helpers_swap_channel_order():
    out = _run_real_qt(_SETUP + '''
bgr = np.zeros((4, 5, 3), dtype=np.uint8)
bgr[..., 0] = 10      # B
bgr[..., 1] = 20      # G
bgr[..., 2] = 30      # R
image = bgr_to_qimage(bgr)
print(image.width(), image.height())
print(tuple(int(v) for v in qimage_to_bgr(image)[0, 0]))
''')
    lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
    assert lines[0] == "5 4", lines[0]
    assert lines[1] == "(10, 20, 30)", lines[1]
