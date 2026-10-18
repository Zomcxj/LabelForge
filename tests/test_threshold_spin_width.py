# -*- coding: utf-8 -*-
"""conf / NMS 的数字必须完整显示，末位小数不能被裁掉。

用户报的「conf 和 NMS 缺了一位小数」：输入框宽 58px 时可编辑区只剩 34px，
而 "0.01" 需要 48px，Qt 直接把末位截掉，看起来就是 "0.0"。

量的是真实控件，不是裸 QDoubleSpinBox：项目 QSS 的内边距比默认样式宽，
用裸控件量会得出偏小的结论（74px 在裸控件上绰绰有余，在真实主题下才刚好）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_QT_EXIT_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""

_SCRIPT = r"""
from PySide6.QtWidgets import QApplication, QStyle, QStyleOptionSpinBox
app = QApplication.instance() or QApplication([])

from labelforge.ui.main_window import ImageEditor
from labelforge.ui.model_row import _THRESHOLD_SPIN_WIDTH

# 走真实窗口，才会套上项目的 QSS
ed = ImageEditor()
ed.show()
app.processEvents()
ed._ensure_model_row()
app.processEvents()
row = ed.model_row

for name, spin in (("conf", row.conf_spin), ("nms", row.iou_spin)):
    spin.ensurePolished()
    opt = QStyleOptionSpinBox()
    spin.initStyleOption(opt)
    field = spin.style().subControlRect(
        QStyle.CC_SpinBox, opt, QStyle.SC_SpinBoxEditField, spin)
    worst, worst_text = 0, ""
    for value in (spin.minimum(), spin.maximum(),
                  (spin.minimum() + spin.maximum()) / 2):
        text = spin.textFromValue(value)
        need = spin.fontMetrics().horizontalAdvance(text)
        if need > worst:
            worst, worst_text = need, text
    print("RESULT", name, field.width(), worst, worst_text)
print("WIDTHCONST", _THRESHOLD_SPIN_WIDTH)
"""


def _run_real_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=180, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _results(out):
    rows = {}
    for line in out.strip().splitlines():
        parts = line.split()
        if parts and parts[0] == "RESULT":
            _tag, name, available, needed, text = parts
            rows[name] = (int(available), int(needed), text)
    return rows


def test_conf_and_nms_show_their_full_decimal():
    """可编辑区必须装得下最长的取值文本，否则末位被裁。"""
    rows = _results(_run_real_qt(_SCRIPT))
    assert set(rows) == {"conf", "nms"}, rows
    for name, (available, needed, text) in rows.items():
        assert available >= needed, (
            f"{name} 的可编辑区只有 {available}px，而 \"{text}\" 需要 "
            f"{needed}px —— 末位小数会被裁掉")


def test_threshold_spin_width_is_not_narrower_than_the_cut_value():
    """宽度常量别被再收窄回 58px —— 那一档会把 "0.01" 裁成 "0.0"。

    只守下限，不守精确值：真正该保证的是上面那条「装得下文本」，
    这里只是拦住「又改回被报过的宽度」这一种回归。
    """
    out = _run_real_qt(_SCRIPT)
    width = None
    for line in out.strip().splitlines():
        if line.startswith("WIDTHCONST"):
            width = int(line.split()[1])
    assert width is not None, out
    assert width > 58, (
        f"_THRESHOLD_SPIN_WIDTH={width} 回到了会裁掉末位小数的宽度")
