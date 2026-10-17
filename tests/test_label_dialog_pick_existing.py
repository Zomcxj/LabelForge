# -*- coding: utf-8 -*-
"""「修改标签」对话框：从已有标签里挑一个改不动（点列表没用）的回归测试。

用户报的场景：画布右键「修改标签」和背景图标签列表「修改标签」两处，
想把 `drone` 改成已存在的 `Drone`——**改不动**。

根因在对话框本体的取值逻辑，而不是改名引擎（引擎一直支持合并）：

1. 对话框会把**当前标签名预填进输入框**（`initial_text`）并全选；
2. `get_selected_label()` 走「输入框非空就用输入框」，于是用户点了列表里的
   `Drone`，返回的仍是输入框里预填的旧名 `drone`；
3. 调用方拿到 `new_label == current_label`，判定为没变化，什么都不做——
   表现就是「怎么点都没反应」。

所以只要「从列表里挑已有标签」这条路，从预填那一刻起就是废的。画布右键和
标签列表共用同一个对话框，两处一起坏。

修法是按「用户最后动过谁」取值：输入框被真正编辑过（`textEdited`）就听输入
框，否则听列表选中。这里锁住这个优先级，包括「什么都不动直接确定」仍返回
预填旧名（不能把「只改分组」这种用法弄坏）。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_PICK_LIST = """
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
import json
app = QApplication([])
from labelforge.ui.dialogs import LabelSelectionDialog

d = LabelSelectionDialog(None, ["drone", "Drone", "car"], title="修改标签",
                         initial_text="drone", show_group=True,
                         current_group_id=None)
print("prefill", repr(d.new_label_input.text()))
row = next(i for i in range(d.label_list.count())
           if d.label_list.item(i).text() == "@@PICK@@")
d.label_list.setCurrentRow(row)
# 模拟用户真的去点列表项（触发 itemClicked，而不是只改选中行）
QTest.mouseClick(d.label_list.viewport(), Qt.LeftButton,
                 pos=d.label_list.visualItemRect(d.label_list.item(row)).center())
app.processEvents()
print("picked", repr(d.get_selected_label()))
"""


def test_picking_an_existing_label_in_the_list_is_honoured():
    """在列表里点了 `Drone`，取到的就该是 `Drone`，不该被输入框里的旧名盖掉。

    这是用户那个 bug 的直接回归：修复前这里返回 'drone'，于是改名「没反应」。
    """
    out = _run_real_qt(_PICK_LIST.replace("@@PICK@@", "Drone"))
    lines = _lines(out)
    assert lines[0] == "prefill 'drone'", lines
    assert lines[1] == "picked 'Drone'", (
        f"点列表里的 Drone 应返回 Drone，实际 {lines[1]}")


def test_picking_the_list_keeps_what_was_typed_in_the_box():
    """点列表不该把用户已输入的内容抹掉——只是让列表这一票作数。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
app = QApplication([])
from labelforge.ui.dialogs import LabelSelectionDialog

d = LabelSelectionDialog(None, ["drone", "Drone", "car"], title="修改标签",
                         initial_text="drone", show_group=True,
                         current_group_id=None)
d.new_label_input.clear()
QTest.keyClicks(d.new_label_input, "truck")
app.processEvents()
row = next(i for i in range(d.label_list.count())
           if d.label_list.item(i).text() == "car")
d.label_list.setCurrentRow(row)
QTest.mouseClick(d.label_list.viewport(), Qt.LeftButton,
                 pos=d.label_list.visualItemRect(d.label_list.item(row)).center())
app.processEvents()
print("box", repr(d.new_label_input.text()))
print("picked", repr(d.get_selected_label()))
""")
    lines = _lines(out)
    assert lines[0] == "box 'truck'", (
        f"点列表不应抹掉用户输入，实际输入框变成 {lines[0]}")
    assert lines[1] == "picked 'car'", (
        f"点列表后应以列表为准，实际 {lines[1]}")


def test_typing_in_the_box_still_wins_over_the_list():
    """反过来也要成立：用户在输入框里打的字优先于列表选中（打新标签）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
app = QApplication([])
from labelforge.ui.dialogs import LabelSelectionDialog

d = LabelSelectionDialog(None, ["drone", "Drone", "car"], title="修改标签",
                         initial_text="drone", show_group=True,
                         current_group_id=None)
d.new_label_input.clear()
QTest.keyClicks(d.new_label_input, "truck")
app.processEvents()
print("typed", repr(d.get_selected_label()))
""")
    lines = _lines(out)
    assert lines[0] == "typed 'truck'", (
        f"输入框打的新标签应优先，实际 {lines[0]}")


def test_confirming_without_touching_anything_keeps_the_current_label():
    """什么都不动直接确定 → 仍返回预填的旧名。

    这条守的是「只改分组、不改标签」的用法：调用方会给当前标签预选列表项，
    此时列表选中与预填内容一致，返回值不变，不能因为修 bug 把它弄坏。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
app = QApplication([])
from labelforge.ui.dialogs import LabelSelectionDialog

d = LabelSelectionDialog(None, ["drone", "Drone", "car"], title="修改标签",
                         initial_text="drone", show_group=True,
                         current_group_id=None)
print("untouched", repr(d.get_selected_label()))
""")
    lines = _lines(out)
    assert lines[0] == "untouched 'drone'", (
        f"没动过任何控件时应保持原名，实际 {lines[0]}")

