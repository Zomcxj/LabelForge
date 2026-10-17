"""标签缓存槽复制/粘贴要带上置信度（以及形状类型），不能只剩坐标和类别。

用户报的问题：检测框缓存（缓存槽）里的框粘回来之后置信度没了。

根因在粘贴侧：`copy_selected_labels_to_active_cache_slot` 用 `dict(box)` 存，
置信度 `score` 其实**已经**进了缓存槽；但 `paste_label_cache_slot` 把框压成
`(QRectF, label)` 二元组再重建，只写回 x/y/width/height/label——`score` 和
`shape_type` 就在这一步被丢掉。`_offset_overlapping_paste_group` 也只透传
两元组，全重合走偏移分支时同样丢。

这里用真 ImageEditor + 真缓存槽走完整「复制 → 粘贴」，断言粘贴出来的框字段。
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


_HOST = """
import os, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor

app = QApplication([])

def make_editor(size=200, with_list=True):
    tmp = tempfile.mkdtemp(prefix="pl-cache-")
    p = os.path.join(tmp, "a.jpg")
    QImage(size, size, QImage.Format_RGB888).save(p)
    w = ImageEditor()
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.background_images = [p]
    if with_list:
        w.background_list.clear()
        w.background_list.addItem(QListWidgetItem("a.jpg"))
    w.current_background_index = 0
    w.current_background = QPixmap.fromImage(QImage(p))
    w.canvas_items = []
    w.detection_boxes_dict = {0: []}
    return w

def roundtrip(w, boxes, paste_times=1):
    \"\"\"把 boxes 复制进缓存槽，清空画布，再粘贴回来；返回粘贴后的框列表。\"\"\"
    w.detection_boxes = [dict(b) for b in boxes]
    w.detection_boxes_dict = {0: [dict(b) for b in boxes]}
    w.canvas.selected_box = 0
    w.canvas.selected_boxes = []
    w.copy_selected_labels_to_active_cache_slot()
    slot = w.active_label_cache_slot
    w.detection_boxes = []
    w.detection_boxes_dict = {0: []}
    for _ in range(paste_times):
        w.paste_label_cache_slot(slot)
    return list(w.detection_boxes)
"""


def test_paste_keeps_the_confidence():
    """复制再粘贴，置信度必须还在。"""
    out = _run_real_qt(_HOST + """
w = make_editor()
box = {"label": "person", "score": 0.87, "x": 10, "y": 10,
       "width": 30, "height": 30, "shape_type": "rectangle"}
pasted = roundtrip(w, [box])
print('框数', len(pasted))
print('score', pasted[0].get('score'))
print('shape_type', pasted[0].get('shape_type'))
print('label', pasted[0].get('label'))
""")
    lines = _lines(out)
    assert lines[0] == "框数 1", lines
    assert lines[1] == "score 0.87", (
        f"粘贴后置信度丢了（{lines}）：缓存槽存了 score，粘回来却没了")
    assert lines[2] == "shape_type rectangle", (
        f"粘贴后形状类型丢了（{lines}）")
    assert lines[3] == "label person", lines


def test_paste_does_not_invent_a_confidence_for_manual_boxes():
    """手工框本来没有 score，粘贴后也不能凭空长出一个。"""
    out = _run_real_qt(_HOST + """
w = make_editor()
box = {"label": "hand", "x": 5, "y": 5, "width": 10, "height": 10,
       "shape_type": "rectangle"}
pasted = roundtrip(w, [box])
print('无 score 键', 'score' not in pasted[0])
print('字段', sorted(pasted[0].keys()))
""")
    lines = _lines(out)
    assert lines[0] == "无 score 键 True", (
        f"粘贴给手工框凭空补了 score（{lines}）")


def test_paste_keeps_the_confidence_on_the_overlap_offset_path():
    """完全重合触发偏移分支时，置信度同样不能丢。

    这条盯的是 `_offset_overlapping_paste_group`：它原来只透传 `(rect, label)`，
    即使粘贴侧把 box 带下去了，经过这一步也会被削掉。所以必须单独钉住。
    """
    out = _run_real_qt(_HOST + """
w = make_editor()
box = {"label": "person", "score": 0.87, "x": 10, "y": 10,
       "width": 30, "height": 30, "shape_type": "rectangle"}
w.detection_boxes = [dict(box)]
w.detection_boxes_dict = {0: [dict(box)]}
w.canvas.selected_box = 0
w.canvas.selected_boxes = []
w.copy_selected_labels_to_active_cache_slot()
# 故意不清空原框 → 粘贴与原框完全重合，必须走偏移
w.paste_label_cache_slot(w.active_label_cache_slot)
print('框数', len(w.detection_boxes))
print('都保住了 score',
      all(b.get('score') == 0.87 for b in w.detection_boxes))
print('确实偏移了', w.detection_boxes[1]['x'] != w.detection_boxes[0]['x'])
""")
    lines = _lines(out)
    assert lines[0] == "框数 2", lines
    assert lines[1] == "都保住了 score True", (
        f"偏移分支把置信度削掉了（{lines}）")
    assert lines[2] == "确实偏移了 True", (
        f"完全重合时没走偏移，这条测不到目标分支（{lines}）")


def test_multiple_boxes_keep_their_own_confidences():
    """多选复制：每个框各自的置信度不能串（都变成同一个或者丢掉）。"""
    out = _run_real_qt(_HOST + """
w = make_editor()
boxes = [
    {"label": "person", "score": 0.91, "x": 1, "y": 1,
     "width": 10, "height": 10, "shape_type": "rectangle"},
    {"label": "car", "score": 0.42, "x": 50, "y": 50,
     "width": 12, "height": 12, "shape_type": "rectangle"},
]
w.detection_boxes = [dict(b) for b in boxes]
w.detection_boxes_dict = {0: [dict(b) for b in boxes]}
w.canvas.selected_boxes = [0, 1]
w.canvas.selected_box = None
w.copy_selected_labels_to_active_cache_slot()
slot = w.active_label_cache_slot
w.detection_boxes = []
w.detection_boxes_dict = {0: []}
w.paste_label_cache_slot(slot)
print('框数', len(w.detection_boxes))
print('置信度', sorted(b.get('score') for b in w.detection_boxes))
""")
    lines = _lines(out)
    assert lines[0] == "框数 2", lines
    assert lines[1] == "置信度 [0.42, 0.91]", (
        f"多选粘贴把置信度弄丢了或串了（{lines}）")


def test_paste_offset_helper_still_accepts_bare_two_tuples():
    """旧调用方给 `(rect, label)` 两元组时不能炸（保持向后兼容）。"""
    out = _run_real_qt("""
from PySide6.QtCore import QRectF
from labelforge.engine.paste_engine import PasteEngineMixin

class FakeBg:
    def width(self):
        return 10
    def height(self):
        return 10

class Editor(PasteEngineMixin):
    def __init__(self):
        self.current_background = FakeBg()
        self.detection_boxes = [{"x": 7, "y": 7, "width": 3, "height": 3,
                                 "label": "cat"}]

e = Editor()
out = e._offset_overlapping_paste_group([(QRectF(7, 7, 3, 3), "cat")])
print('项数', len(out))
print('仍是两元组', len(out[0]) == 2)
print('标签', out[0][1])
print('已偏移', out[0][0].x() > 7)
""")
    lines = _lines(out)
    assert lines[0] == "项数 1", lines
    assert lines[1] == "仍是两元组 True", (
        f"两元组输入被改成了别的形状（{lines}）")
    assert lines[2] == "标签 cat", lines
    assert lines[3] == "已偏移 True", lines
