# -*- coding: utf-8 -*-
"""画布右键「修改标签」只能改**那一个**框的回归测试。

用户报的：「画布上右键修改标签怎么是改全部？我只是改一个啊」。

根因：`_modify_box_label()` 直接调 `LabelManager.rename_detection_label()`，
而那是**按标签名全局改名**——把当前图和整个数据集里所有同名框一起改掉。
而同一个右键菜单里「点已有标签直接改」（`_change_box_label()`）本来就是
「只改一个」，两条路径口径不一致。

修法是让 `_modify_box_label()` 复用 `_change_box_label()`（颜色迁移、统计
缓存 delta=1、写盘都已在里面），分组（`group_id`）单独处理。

这里锁住的行为：
1. 只改右键那一个框，同图其它同名框不动；
2. **其它图片**里的同名框也不动（全局改名的老行为会把它们一起改掉）；
3. 磁盘同步（只改这一个）；
4. 改成**已存在**的标签也是只改这一个（不是合并整类）；
5. 只改分组、不改标签时也要落盘；
6. 取消对话框什么都不改。
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


# 数据集宿主：图0 有 drone,drone,Drone；图1 有一个 drone。
_HOST = """
import os, json, shutil
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from unittest.mock import patch
app = QApplication([])
from labelforge.ui.main_window import ImageEditor

WORK = r"@@WORK@@"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

def _box(label, y, gid=None):
    return {"label": label, "points": [[10, y], [80, y + 40]], "group_id": gid,
            "shape_type": "rectangle", "flags": {}, "description": ""}

def _write(idx, shapes):
    QImage(300, 200, QImage.Format_RGB888).save(os.path.join(WORK, f"{idx:03d}.png"))
    with open(os.path.join(WORK, f"{idx:03d}.json"), "w", encoding="utf-8") as fh:
        json.dump({"version": "1.0", "flags": {}, "shapes": shapes,
                   "imagePath": f"{idx:03d}.png", "imageData": None,
                   "imageHeight": 200, "imageWidth": 300}, fh)

def _disk(idx):
    with open(os.path.join(WORK, f"{idx:03d}.json"), encoding="utf-8") as fh:
        return [s["label"] for s in json.load(fh)["shapes"]]

def _disk_groups(idx):
    with open(os.path.join(WORK, f"{idx:03d}.json"), encoding="utf-8") as fh:
        return [s.get("group_id") for s in json.load(fh)["shapes"]]

def build(shapes0, shapes1=None):
    _write(0, shapes0)
    if shapes1 is not None:
        _write(1, shapes1)
    w = ImageEditor(); w.resize(1200, 900); w.show(); app.processEvents()
    w.load_background_folder(WORK); app.processEvents()
    return w, patch

def mem(w):
    return [b["label"] for b in w.detection_boxes]
"""


def test_canvas_rename_only_touches_the_clicked_box():
    """右键第 0 个框改名，同图的其它同名框不能被动。"""
    out = _run_real_qt(
        _HOST.replace("@@WORK@@", "D:/tmp/_menu_one") + """
w, patch = build([_box("drone", 10), _box("drone", 80), _box("Drone", 140)],
                 [_box("drone", 10)])
print("before", json.dumps(mem(w)))
with patch("labelforge.ui.dialogs.LabelSelectionDialog.select_label",
           staticmethod(lambda *a, **k: ("cat", None))):
    w.canvas._modify_box_label(0)
app.processEvents()
print("after", json.dumps(mem(w)))
""")
    lines = _lines(out)
    assert lines[0] == 'before ["drone", "drone", "Drone"]', lines
    assert lines[1] == 'after ["cat", "drone", "Drone"]', (
        f"只应改第 0 个框，实际 {lines[1]}")


def test_canvas_rename_does_not_touch_other_images():
    """别的图片里的同名框也不能被改（全局改名的老行为会殃及它们）。"""
    out = _run_real_qt(
        _HOST.replace("@@WORK@@", "D:/tmp/_menu_other") + """
w, patch = build([_box("drone", 10), _box("drone", 80)], [_box("drone", 10)])
with patch("labelforge.ui.dialogs.LabelSelectionDialog.select_label",
           staticmethod(lambda *a, **k: ("cat", None))):
    w.canvas._modify_box_label(0)
app.processEvents()
print("img0", json.dumps(_disk(0)))
print("img1", json.dumps(_disk(1)))
""")
    lines = _lines(out)
    assert lines[0] == 'img0 ["cat", "drone"]', lines
    assert lines[1] == 'img1 ["drone"]', (
        f"另一张图的同名框不该被动，实际 {lines[1]}")


def test_canvas_rename_into_an_existing_label_touches_only_that_box():
    """改成已存在的标签时也只改这一个框，不是把整个类别合并过去。"""
    out = _run_real_qt(
        _HOST.replace("@@WORK@@", "D:/tmp/_menu_merge") + """
w, patch = build([_box("drone", 10), _box("Drone", 80), _box("drone", 140)])
with patch("labelforge.ui.dialogs.LabelSelectionDialog.select_label",
           staticmethod(lambda *a, **k: ("Drone", None))):
    w.canvas._modify_box_label(0)
app.processEvents()
print("mem", json.dumps(mem(w)))
print("disk", json.dumps(_disk(0)))
""")
    lines = _lines(out)
    assert lines[0] == 'mem ["Drone", "Drone", "drone"]', (
        f"只有第 0 个框该变，实际 {lines[0]}")
    assert lines[1] == 'disk ["Drone", "Drone", "drone"]', (
        f"磁盘也应只有第 0 个框变，实际 {lines[1]}")


def test_canvas_group_only_change_is_saved():
    """只改分组、标签不变：标签不能被动，但分组要落盘（否则切图就丢）。"""
    out = _run_real_qt(
        _HOST.replace("@@WORK@@", "D:/tmp/_menu_group") + """
w, patch = build([_box("drone", 10, 3), _box("drone", 80)])
with patch("labelforge.ui.dialogs.LabelSelectionDialog.select_label",
           staticmethod(lambda *a, **k: ("drone", 9))):
    w.canvas._modify_box_label(0)
app.processEvents()
print("mem", json.dumps(mem(w)))
print("gids", json.dumps(_disk_groups(0)))
""")
    lines = _lines(out)
    assert lines[0] == 'mem ["drone", "drone"]', (
        f"只改分组时标签不该变，实际 {lines[0]}")
    assert lines[1] == "gids [9, null]", (
        f"分组应落盘，实际 {lines[1]}")


def test_canvas_rename_cancelled_changes_nothing():
    """取消对话框（返回 None）→ 一个框都不动。"""
    out = _run_real_qt(
        _HOST.replace("@@WORK@@", "D:/tmp/_menu_cancel") + """
w, patch = build([_box("drone", 10), _box("drone", 80)])
with patch("labelforge.ui.dialogs.LabelSelectionDialog.select_label",
           staticmethod(lambda *a, **k: None)):
    w.canvas._modify_box_label(0)
app.processEvents()
print("mem", json.dumps(mem(w)))
print("disk", json.dumps(_disk(0)))
""")
    lines = _lines(out)
    assert lines[0] == 'mem ["drone", "drone"]', lines
    assert lines[1] == 'disk ["drone", "drone"]', lines


def test_canvas_rename_through_the_real_dialog_still_hits_one_box():
    """走**真实对话框**（不 patch `select_label`）：点列表里的 Drone 再确定。

    这条把两件事串起来验证——对话框能正确返回点中的标签（上个 bug），
    且只改右键的那一个框（这个 bug）。两头都修了才通得过。
    """
    script = _HOST.replace("@@WORK@@", "D:/tmp/_menu_real") + """
from PySide6.QtWidgets import QDialog
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest

# 这条用真实对话框，不 patch select_label
w, _unused = build([_box("drone", 10), _box("drone", 80), _box("Drone", 140)])

def drive():
    d = next((c for c in app.topLevelWidgets()
              if isinstance(c, QDialog) and "Label" in c.__class__.__name__
              and c.isVisible()), None)
    if d is None:
        print("err no dialog"); app.quit(); return
    row = next((i for i in range(d.label_list.count())
                if d.label_list.item(i).text() == "Drone"), None)
    d.label_list.setCurrentRow(row)
    QTest.mouseClick(d.label_list.viewport(), Qt.LeftButton,
                     pos=d.label_list.visualItemRect(d.label_list.item(row)).center())
    app.processEvents()
    d.accept()

QTimer.singleShot(2200, drive)
w.canvas._modify_box_label(0)
app.processEvents()
print("mem", json.dumps(mem(w)))
print("disk", json.dumps(_disk(0)))
"""
    out = _run_real_qt(script)
    lines = _lines(out)
    assert lines[0] == 'mem ["Drone", "drone", "Drone"]', (
        f"真实对话框里点 Drone，只有第 0 个框该变，实际 {lines[0]}")
    assert lines[1] == 'disk ["Drone", "drone", "Drone"]', (
        f"磁盘应同步，实际 {lines[1]}")
