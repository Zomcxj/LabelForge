# -*- coding: utf-8 -*-
"""统计弹窗「改标签」时把一类合并进另一类的回归测试。

用户报的场景：背景图标签表里同时有 `drone` 和 `Drone`，想把 `drone` 双击改成
`Drone`（**目标是已存在的标签**）——改不了，一改就弹回原名。

根因不是底层不支持合并（`rename_detection_label` / `rename_paste_label` 本来就
显式处理 `target_exists`，把两组框合并、数量相加、保留目标颜色），而是统计窗口
的 `itemChanged` 回调里各有一道「重名就拒绝」的拦截，把用户要的合并提前挡掉了。
主窗口「修改标签」从来不做这种拦截，所以同一个操作在主窗口能成、在统计窗口不成。

这里锁两件事：
1. 底层确实是**合并**语义（数量相加、旧标签消失）；
2. 统计窗口的两张表（背景图 / 贴图）都**放行**重名改名——两处都要测，
   否则「只修了背景图、贴图还是不能合并」这种半修状态会再次出现。
"""
import json
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


# 造一个数据集：既有 drone 也有 Drone，用来复现「改成已存在的标签」
_DATASET_HELPER = """
import os, json, shutil
from PySide6.QtGui import QImage

WORK = r"@@WORK@@"
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

def _box(label, y):
    return {"label": label, "points": [[10, y], [60, y + 30]], "group_id": None,
            "shape_type": "rectangle", "flags": {}, "description": ""}

def write_sidecar(idx, shapes):
    QImage(200, 150, QImage.Format_RGB888).save(os.path.join(WORK, f"{idx:03d}.png"))
    data = {"version": "1.0", "flags": {}, "shapes": shapes,
            "imagePath": f"{idx:03d}.png", "imageData": None,
            "imageHeight": 150, "imageWidth": 200}
    with open(os.path.join(WORK, f"{idx:03d}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh)

# 背景图：3 个 drone + 3 个 Drone
write_sidecar(0, [_box("drone", 10), _box("drone", 50), _box("Drone", 90)])
write_sidecar(1, [_box("drone", 10)])
write_sidecar(2, [_box("Drone", 10), _box("Drone", 50)])
"""


def test_rename_merges_into_an_existing_label():
    """底层：把 `drone` 改成已存在的 `Drone` = 合并，数量相加、旧标签消失。"""
    out = _run_real_qt(_DATASET_HELPER.replace("@@WORK@@", "D:/tmp/_merge_label_a") + """
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
app = QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor(); w.resize(1200, 900); w.show(); app.processEvents()
w.load_background_folder(WORK); app.processEvents()
w._cache_bg_stats({"drone": 3, "Drone": 3}, {})
print("before", sorted(getattr(w, "global_labels", set())))
print("call", w.label_manager.rename_detection_label("drone", "Drone"))
print("labels", sorted(getattr(w, "global_labels", set())))
stats = {s["label"]: s["count"] for s in (getattr(w, "_cached_bg_label_stats", None) or [])}
print("stats", json.dumps(stats, sort_keys=True))
""")
    lines = _lines(out)
    assert lines[0] == "before ['Drone', 'drone']", lines
    assert lines[1] == "call True", lines
    assert lines[2] == "labels ['Drone']", (
        f"合并后应只剩 Drone，实际 {lines[2]}")
    stats = json.loads(lines[3].split(" ", 1)[1])
    assert stats == {"Drone": 6}, (
        f"合并应把数量相加（3+3=6），实际 {stats}")


def test_stats_dialog_rename_into_an_existing_label_merges():
    """统计弹窗里双击把 `drone` 改成 `Drone`，两行合并成一行、数量相加。

    这条是用户报的那个 bug 的直接回归：修复前表格回调会把重名改回原名，
    表格纹丝不动。
    """
    out = _run_real_qt(_DATASET_HELPER.replace("@@WORK@@", "D:/tmp/_merge_label_b") + """
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtGui import QImage
from PySide6.QtCore import QTimer
app = QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor(); w.resize(1200, 900); w.show(); app.processEvents()
w.load_background_folder(WORK); app.processEvents()

out = {}
def drive():
    d = next((c for c in app.topLevelWidgets()
              if isinstance(c, QDialog) and c.isVisible()), None)
    if d is None:
        out["err"] = "no dialog"; app.quit(); return
    t = getattr(d, "_bg_table", None)
    row = next((r for r in range(t.rowCount())
                if t.item(r, 0) and t.item(r, 0).text() == "drone"), None)
    if row is None:
        out["err"] = "no drone row"; d.accept(); return
    t.item(row, 0).setText("Drone")
    app.processEvents()
    t2 = d._bg_table
    out["rows"] = [(t2.item(r, 0).text() if t2.item(r, 0) else "",
                    t2.item(r, 1).text() if t2.item(r, 1) else "")
                   for r in range(t2.rowCount())]
    d.accept()

QTimer.singleShot(2500, drive)
w._show_label_stats()

print("err", out.get("err"))
print("rows", json.dumps(out.get("rows")))
""")
    lines = _lines(out)
    assert lines[0] == "err None", lines
    rows = json.loads(lines[1].split(" ", 1)[1])
    assert rows == [["Drone", "6"]], (
        f"统计弹窗里 drone→Drone 应合并成一行、数量 6，实际 {rows}")


def test_stats_dialog_rename_still_blocks_an_empty_name():
    """放行重名不等于放行空名：清空标签名仍要回退，不能让标签变无名。"""
    out = _run_real_qt(_DATASET_HELPER.replace("@@WORK@@", "D:/tmp/_merge_label_c") + """
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtGui import QImage
from PySide6.QtCore import QTimer
app = QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor(); w.resize(1200, 900); w.show(); app.processEvents()
w.load_background_folder(WORK); app.processEvents()

out = {}
def drive():
    d = next((c for c in app.topLevelWidgets()
              if isinstance(c, QDialog) and c.isVisible()), None)
    if d is None:
        out["err"] = "no dialog"; app.quit(); return
    t = getattr(d, "_bg_table", None)
    row = next((r for r in range(t.rowCount())
                if t.item(r, 0) and t.item(r, 0).text() == "drone"), None)
    t.item(row, 0).setText("   ")
    app.processEvents()
    t2 = d._bg_table
    out["first"] = t2.item(row, 0).text() if t2.item(row, 0) else None
    d.accept()

QTimer.singleShot(2500, drive)
w._show_label_stats()
print("first", repr(out.get("first")))
""")
    lines = _lines(out)
    assert lines[0] == "first 'drone'", (
        f"空名应回退为原名，实际 {lines[0]}")


def test_paste_table_also_merges_into_an_existing_label():
    """贴图表必须与背景图表同一口径：双击把贴图标签改成已存在的名字也要合并。

    这条**必须真驱动贴图表的 `itemChanged` 回调**，不能只调底层
    `rename_paste_label`——那样即使贴图表里的重名拦截原样还在，测试也照样
    是绿的（变异验证抓到过这个漏网）。用户批评过「修一半」，所以这条要能
    单独发现「只修了背景图、贴图没修」。
    """
    script = """
import os, json
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtGui import QPixmap, QImage, QPainter
from PySide6.QtCore import QTimer
app = QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor(); w.resize(1200, 900); w.show(); app.processEvents()
# 手工摆出「会话里有 logo 和 badge 两种贴图」的状态：贴图表的数据源就是
# canvas_items_dict 里的 (pixmap, rect, label)。
w.background_images = ["a.png", "b.png"]
w.current_background_index = 0
pm = QPixmap.fromImage(QImage(20, 20, QImage.Format_RGB888))
from PySide6.QtCore import QRect
w.canvas_items = [(pm, QRect(0, 0, 20, 20), "logo"),
                  (pm, QRect(30, 0, 20, 20), "badge")]
w.canvas_items_dict = {0: list(w.canvas_items), 1: []}

out = {}
def drive():
    d = next((c for c in app.topLevelWidgets()
              if isinstance(c, QDialog) and c.isVisible()), None)
    if d is None:
        out["err"] = "no dialog"; app.quit(); return
    t = getattr(d, "_paste_table", None)
    if t is None:
        out["err"] = "no paste_table"; d.accept(); return
    out["before"] = [(t.item(r, 0).text() if t.item(r, 0) else "")
                     for r in range(t.rowCount())]
    row = next((r for r in range(t.rowCount())
                if t.item(r, 0) and t.item(r, 0).text() == "logo"), None)
    if row is None:
        out["err"] = "no logo row"; d.accept(); return
    t.item(row, 0).setText("badge")      # 改成**已存在**的贴图标签
    app.processEvents()
    t2 = d._paste_table
    out["after"] = [(t2.item(r, 0).text() if t2.item(r, 0) else "",
                     t2.item(r, 1).text() if t2.item(r, 1) else "")
                    for r in range(t2.rowCount())]
    labels = set()
    for items in (w.canvas_items_dict or {}).values():
        for _pm, _rect, lab in items or []:
            labels.add(lab)
    for _pm, _rect, lab in w.canvas_items or []:
        labels.add(lab)
    out["canvas_labels"] = sorted(labels)
    d.accept()

QTimer.singleShot(2500, drive)
w._show_label_stats()

print("err", out.get("err"))
print("before", json.dumps(out.get("before")))
print("after", json.dumps(out.get("after")))
print("labels", json.dumps(out.get("canvas_labels")))
"""
    out = _run_real_qt(script)
    lines = _lines(out)
    assert lines[0] == "err None", lines
    before = json.loads(lines[1].split(" ", 1)[1])
    assert sorted(before) == ["badge", "logo"], before
    after = json.loads(lines[2].split(" ", 1)[1])
    assert after == [["badge", "2"]], (
        f"贴图表里 logo→badge 应合并成一行、数量 2，实际 {after}")
    labels = json.loads(lines[3].split(" ", 1)[1])
    assert labels == ["badge"], (
        f"画布上的贴图标签应全部改名，实际 {labels}")
