"""贴图外接框的两种表示都要能读：QRectF 与抠图产物给的 (x, y, w, h)。

抠图结果 `MatteResult.rect` 是四元组，直接进 `PasteItem.rect`。画布与统计
原先直接调 `rect.x()`，鼠标一移到画布上就抛
`AttributeError: 'tuple' object has no attribute 'x'`。统一改走
`rect_tuple()`，这里把几条读取路径钉住。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUARD = "\nimport os, sys\nsys.stdout.flush()\nos._exit(0)\n"


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_CANVAS_FIXTURE = '''
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage
from PySide6.QtCore import QPoint, QRectF
from labelforge.canvas.canvas import Canvas
from labelforge.engine.paste_item import PasteItem

app = QApplication([])

class Editor:
    current_background = QImage(200, 200, QImage.Format_RGB888)
    current_background_index = 0
    canvas_items = [PasteItem(pixmap=object(), rect=(10, 20, 30, 40), label="car")]
    selected_item = None
    edit_mode = "paste"
    background_scale = 1.0
    background_offset = QPoint(0, 0)
    detection_boxes = []
    _max_labels = 3

    class _Label:
        def setText(self, text):
            pass

    status_label = _Label()

    def get_image_info(self):
        return {"box_count": 0, "paste_count": len(self.canvas_items),
                "width": 200, "height": 200, "name": "t.png"}

    def get_label_stats(self):
        return {}

    def __getattr__(self, name):
        return False

ed = Editor()
canvas = Canvas()
canvas._editor = ed
canvas.resize(400, 400)
canvas.show()
app.processEvents()
canvas.reset_view()
app.processEvents()
'''


def test_hit_test_accepts_a_tuple_rect():
    """抠图产物的 rect 是四元组，命中测试不能崩。"""
    out = _run_real_qt(_CANVAS_FIXTURE + '''
from PySide6.QtCore import QPoint
bg = canvas.get_background_rect()
sx = bg.left() + (10 + 15) * canvas.background_scale
sy = bg.top() + (20 + 20) * canvas.background_scale
print(canvas.find_item_at_position(QPoint(int(sx), int(sy))))
print(canvas.find_item_at_position(QPoint(0, 0)))
''')
    lines = _lines(out)
    assert lines[0] == "0", f"元组 rect 的贴图没被命中：{lines[0]}"
    assert lines[1] == "None", lines[1]


def test_resize_handle_lookup_accepts_a_tuple_rect():
    """右下角手柄查找同样只吃 rect_tuple()。"""
    out = _run_real_qt(_CANVAS_FIXTURE + '''
from PySide6.QtCore import QPoint, QRectF
from labelforge.engine.paste_item import as_paste_item
item = as_paste_item(ed.canvas_items[0])
rect = QRectF(*item.rect_tuple())
bg = canvas.get_background_rect()
# 右下角在画布上的位置
br_x = bg.left() + (item.rect_tuple()[0] + item.rect_tuple()[2]) * canvas.background_scale
br_y = bg.top() + (item.rect_tuple()[1] + item.rect_tuple()[3]) * canvas.background_scale
print(canvas._item_handle_at_pos(QPoint(int(br_x), int(br_y)), rect))
''')
    lines = _lines(out)
    assert lines[0] != "None", "右下角手柄没被识别"


def test_painting_a_tuple_rect_item_does_not_crash():
    """渲染一次画布：贴图绘制路径也要接受四元组。"""
    out = _run_real_qt(_CANVAS_FIXTURE + '''
pixmap = canvas.grab()
print(pixmap.width(), pixmap.height())
''')
    lines = _lines(out)
    assert lines[0].split()[0].isdigit(), lines[0]


def test_paste_geometry_counts_a_tuple_rect_item():
    """数据集统计不能把四元组 rect 的贴图项当成坏数据跳过。"""
    from labelforge.engine.dataset_health import collect_paste_geometry
    from labelforge.engine.paste_item import PasteItem

    items = {0: [PasteItem(pixmap=object(), rect=(10, 20, 30, 40), label="car")]}
    result = collect_paste_geometry(items)

    assert len(result) == 1, result
    geo = result[0]
    assert (geo["x"], geo["y"], geo["width"], geo["height"]) == (10, 20, 30, 40)
