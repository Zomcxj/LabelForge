# -*- coding: utf-8 -*-
"""Clicking the paste-mode segment must actually switch the mode.

conftest replaces PySide6 with stubs at import time, so anything that needs
real widgets has to run in a subprocess with real PySide6. That is the same
pattern the canvas regression tests use.

Builds the real ImageEditor, then drives the mode segment the way a user does
and reads back `edit_mode`.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 子进程里 import 完就退，避免真 Qt 的事件循环挂在退出阶段
_GUARD = """
import sys
sys.stdout.flush()
sys.exit(0)
"""

_SCRIPT = r"""
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

app = QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor()
w.show()
app.processEvents()

print("mode_initial", w.edit_mode)

# 用 .click() 驱动，和按钮自身的 clicked 信号是同一条路径
w.btn_paste_mode.click()
app.processEvents()
print("after_paste_click", w.edit_mode)
print("paste_checked", w.btn_paste_mode.isChecked())
print("annotate_checked", w.btn_annotate_mode.isChecked())

# 指示器应该挪到第二个按钮下面。动画 180ms，采样前要等它跑完，
# 否则量到的是动画起点，测试会误报成「指示器没动」。
from PySide6.QtCore import QEventLoop, QTimer
w.btn_paste_mode.click()
_loop = QEventLoop()
QTimer.singleShot(320, _loop.quit)
_loop.exec()
print("after_paste_click_settled", w.edit_mode)
ctrl = getattr(w, "mode_seg_ctrl", None)
if ctrl is not None:
    print("indicator_x", ctrl._indicator.x())
    print("paste_x", w.btn_paste_mode.x())

# 再切回去
w.btn_annotate_mode.click()
app.processEvents()
print("after_annotate_click", w.edit_mode)
print("annotate_checked2", w.btn_annotate_mode.isChecked())

# 空模式下点击是否真的会覆盖到按钮：看点位命中的控件
holder = w.btn_paste_mode.parentWidget()
center = holder.mapTo(holder.parentWidget(), holder.rect().center())
hit = holder.parentWidget().childAt(center)
print("hit_at_segment_center", type(hit).__name__ if hit is not None else "None")
"""


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _values(out):
    pairs = {}
    for line in out.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            pairs[parts[0]] = parts[1]
    return pairs


def test_clicking_paste_segment_switches_mode():
    """A click on 贴图 must set edit_mode to paste and check that button."""
    out = _values(_run_real_qt(_SCRIPT))
    assert out.get("mode_initial") == "annotate", out
    assert out.get("after_paste_click") == "paste", (
        f"clicking 贴图 left edit_mode={out.get('after_paste_click')}")
    assert out.get("paste_checked") == "True", "贴图 button did not stay checked"
    assert out.get("annotate_checked") == "False", "标注 button stayed checked too"


def test_clicking_annotate_segment_switches_back():
    out = _values(_run_real_qt(_SCRIPT))
    assert out.get("after_annotate_click") == "annotate", (
        f"clicking 标注 left edit_mode={out.get('after_annotate_click')}")
    assert out.get("annotate_checked2") == "True"


def test_indicator_follows_the_checked_segment():
    """The sliding indicator must sit under the second segment after switching.

    The indicator animates over 180 ms, so the assertion has to wait for the
    animation rather than sampling one frame after the click.
    """
    out = _values(_run_real_qt(_SCRIPT))
    if "indicator_x" not in out:
        return  # 控件不存在就没什么可量的
    indicator_x = int(out["indicator_x"])
    paste_x = int(out["paste_x"])
    assert indicator_x >= paste_x - 2, (
        f"indicator stayed at x={indicator_x}, 贴图 starts at x={paste_x}")


def test_nothing_covers_the_segment_buttons():
    """The segment control must not sit on top of the buttons and eat clicks.

    AnimatedSegmentedControl is parented to the container's parent. If it gets
    sized over the segment area it swallows mouse events and neither button
    fires, which looks exactly like "clicking does nothing".
    """
    out = _values(_run_real_qt(_SCRIPT))
    hit = out.get("hit_at_segment_center")
    assert hit != "AnimatedSegmentedControl", (
        "an AnimatedSegmentedControl covers the segment centre and eats clicks")
