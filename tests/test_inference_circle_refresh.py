"""一键推理无检出后，磁盘与图标都要**当场**落到最终状态。

用户报的现象（两次描述合起来）：一键推理没检出东西时，列表里该图仍是旧的
「有检测框」圆圈，切到别的图再切回来才变；本质是**磁盘上还留着上一轮的
自动框**，没被清掉。

根因：当前图这条路径在「无框且无贴图」时调 `remove_empty_sidecar()` 删 json
就 `return False` 了——可 `remove_empty_sidecar` 只删「原文 shapes 为空」的
文件，看到上一轮的旧框就拒绝删，于是磁盘一直停在「已标注」。必须先按本轮
结果写盘（把清空后的状态落到 json、此时 shapes 才是空），再删空文件、再刷新
圆点。

图标语义按既有设计（`200ace1`「无检出不留 sidecar」）：删掉 json 后是
「未标注」灰色空心环，**不是**「空标签」橙色环。用户已确认灰色空心是对的，
这里就照这个口径断言，别再把空标签当成期望值。
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

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
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def test_one_key_inference_without_detections_settles_disk_and_circle_immediately():
    """一键推理无检出 → 旧自动框从磁盘清掉、圆点当场变「未标注」，不需切图。

    场景要搭对：磁盘上那条必须是**上一轮推理留下的自动框**（flags.inference），
    覆盖模式下这一轮无检出才会真的清掉它。这是用户报的原始场景。
    """
    out = _run_real_qt("""
import json, os, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.image_loader.status import (
    BG_ROLE_STATUS, STATUS_UNANNOTATED, STATUS_ANNOTATED)
from labelforge.engine.inference_task import ImageResult, sidecar_path

app = QApplication([])
tmp = tempfile.mkdtemp(prefix="pl-circle-")
img_path = os.path.join(tmp, "a.jpg")
QImage(80, 60, QImage.Format_RGB888).save(img_path)
# 磁盘上：上一轮推理留下的自动框 → 状态「已标注」
with open(sidecar_path(img_path), "w", encoding="utf-8") as f:
    json.dump({"shapes": [{"label": "old", "points": [[1, 1], [5, 5]],
                           "shape_type": "rectangle",
                           "flags": {"inference": True}}],
               "imagePath": "a.jpg"}, f)

w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()
w.background_images = [img_path]
w.background_list.clear()
w.background_list.addItem(QListWidgetItem("a.jpg"))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(img_path))
w.canvas_items = []
w.detection_boxes_dict = {0: [{"label": "old", "x": 1, "y": 1,
                               "width": 4, "height": 4,
                               "shape_type": "rectangle",
                               "flags": {"inference": True}}]}
w.detection_boxes = list(w.detection_boxes_dict[0])
w._refresh_background_item_status(0, img_path)
app.processEvents()
print('before', w.background_list.item(0).data(BG_ROLE_STATUS))

# 一键推理：无检出、覆盖模式 → 旧自动框该被清掉
w._apply_inference_to_current(
    ImageResult(index=0, path=img_path, boxes=[]), True)
app.processEvents()
after = w.background_list.item(0).data(BG_ROLE_STATUS)
print('after', after)
# 磁盘上旧框必须真的没了（否则切图回来又会显示旧框）
still_old = False
if os.path.exists(sidecar_path(img_path)):
    data = json.load(open(sidecar_path(img_path), encoding="utf-8"))
    still_old = bool(data.get("shapes") or [])
print('磁盘无旧框', not still_old)
print('圆点当场更新', after == STATUS_UNANNOTATED)
""")
    lines = _lines(out)
    assert lines[0] == "before annotated", lines
    assert lines[2] == "磁盘无旧框 True", (
        f"无检出后磁盘仍留着上一轮的推理框（{lines}）：切图回来会又看到旧框")
    assert lines[3] == "圆点当场更新 True", (
        f"无检出后圆点没当场刷新（{lines}）：用户会看到旧的检测框圆圈，"
        "要切图才变")


def test_one_key_inference_without_detections_never_leaves_a_stale_sidecar():
    """原文件里只有上一轮自动框、这一轮无检出 → 那份 json 该消失。

    对照上一条：不只看图标，直接把「磁盘上还有没有 shapes」也钉住，
    避免只靠图标（图标是 json 的派生量）把问题绕过去。
    """
    out = _run_real_qt("""
import json, os, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_task import ImageResult, sidecar_path

app = QApplication([])
tmp = tempfile.mkdtemp(prefix="pl-stale-")
img_path = os.path.join(tmp, "b.jpg")
QImage(80, 60, QImage.Format_RGB888).save(img_path)
with open(sidecar_path(img_path), "w", encoding="utf-8") as f:
    json.dump({"shapes": [{"label": "auto", "points": [[2, 2], [9, 9]],
                           "shape_type": "rectangle",
                           "flags": {"inference": True}}]}, f)

w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()
w.background_images = [img_path]
w.background_list.clear()
w.background_list.addItem(QListWidgetItem("b.jpg"))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(img_path))
w.canvas_items = []
w.detection_boxes_dict = {0: [{"label": "auto", "x": 2, "y": 2,
                               "width": 7, "height": 7,
                               "shape_type": "rectangle",
                               "flags": {"inference": True}}]}
w.detection_boxes = list(w.detection_boxes_dict[0])
print('推理前有 json', os.path.exists(sidecar_path(img_path)))

w._apply_inference_to_current(
    ImageResult(index=0, path=img_path, boxes=[]), True)
app.processEvents()
print('推理后 shapes 为空或文件已删',
      (not os.path.exists(sidecar_path(img_path)))
      or not (json.load(open(sidecar_path(img_path),
                             encoding="utf-8")).get("shapes") or []))
""")
    lines = _lines(out)
    assert lines[0] == "推理前有 json True", lines
    assert lines[1] == "推理后 shapes 为空或文件已删 True", (
        f"无检出后磁盘仍留着旧自动框（{lines}）")


def test_inference_replace_mode_clears_manual_annotations_too():
    """勾了覆盖时，无检出也会把这一图**所有检测框**清掉（含手工画的）。

    这条原先叫 `test_inference_keeps_manual_annotations_when_nothing_is_detected`，
    断言的是「人工标注不能被无检出清掉」。用户明确要求「手动绘制的没有覆盖，
    应该都覆盖」，所以覆盖模式下这条反转：清理是本意，不是事故。

    **贴图不在其中**：贴图是单独存进 json 的素材，不属于检测框，见下一条。
    """
    out = _run_real_qt("""
import json, os, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_task import ImageResult, sidecar_path

app = QApplication([])
tmp = tempfile.mkdtemp(prefix="pl-manual-")
img_path = os.path.join(tmp, "c.jpg")
QImage(80, 60, QImage.Format_RGB888).save(img_path)
# 磁盘上只有**人工**标注（没有 flags.inference）
with open(sidecar_path(img_path), "w", encoding="utf-8") as f:
    json.dump({"shapes": [{"label": "hand", "points": [[1, 1], [7, 7]],
                           "shape_type": "rectangle"}]}, f)

w = ImageEditor()
w.resize(1200, 800)
w.show()
app.processEvents()
w.background_images = [img_path]
w.background_list.clear()
w.background_list.addItem(QListWidgetItem("c.jpg"))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(img_path))
w.canvas_items = []
w.detection_boxes_dict = {0: [{"label": "hand", "x": 1, "y": 1,
                               "width": 6, "height": 6,
                               "shape_type": "rectangle"}]}
w.detection_boxes = list(w.detection_boxes_dict[0])

w._apply_inference_to_current(
    ImageResult(index=0, path=img_path, boxes=[]), True)
app.processEvents()
kept = []
if os.path.exists(sidecar_path(img_path)):
    data = json.load(open(sidecar_path(img_path), encoding="utf-8"))
    kept = [s for s in (data.get("shapes") or [])
            if s.get("label") == "hand"]
print('手工框已被覆盖清掉', len(kept) == 0)
print('画布上也清了', [b['label'] for b in w.detection_boxes])
""")
    lines = _lines(out)
    assert lines[0] == "手工框已被覆盖清掉 True", (
        f"覆盖模式下无检出没清掉手工框（{lines}）")
    assert lines[1] == "画布上也清了 []", lines[1]


def test_inference_never_deletes_paste_shapes():
    """贴图绝不能因为推理被删掉——它单独存 json，是用户的素材。

    这条是上一条的反向约束：清检测框不能扩大成「清空一切」，贴图必须原样在。
    """
    out = _run_real_qt("""
import json, os, tempfile
from PySide6.QtWidgets import QApplication
from labelforge.engine.inference_task import ImageResult, write_one_result

tmp = tempfile.mkdtemp(prefix="pl-paste-")
img_path = os.path.join(tmp, "p.jpg")
open(img_path, "wb").close()

existing = [
    {"label": "hand", "x": 1, "y": 1, "width": 5, "height": 5, "flags": {}},
    {"label": "logo", "x": 2, "y": 2, "width": 5, "height": 5,
     "flags": {"paste": True}},
]
write_one_result(
    ImageResult(index=0, path=img_path, boxes=[
        {"label": "new", "x": 3, "y": 3, "width": 2, "height": 2}],
        width=80, height=60),
    existing=existing, replace_auto=True)

from labelforge.engine.inference_task import sidecar_path
data = json.load(open(sidecar_path(img_path), encoding="utf-8"))
labels = sorted(s["label"] for s in data["shapes"])
print(labels)
""")
    lines = _lines(out)
    assert lines[0] == "['logo', 'new']", (
        f"覆盖时贴图被删了或手工框没清：{lines[0]}")
