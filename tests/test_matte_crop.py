"""抠图素材的现场裁剪：拖框 → 裁剪 → 覆盖写回原 PNG。

脱框交互在真 Qt 下标量（QPixmap/QImage 与画家都要真货），所以整段在子进程里跑。
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


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_SETUP = '''
import os
import shutil
import tempfile

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

app = QApplication([])

from labelforge.engine.matting_service import save_matte_png
from labelforge.engine.paste_fusion import pixmap_to_rgba, rgba_to_pixmap
from labelforge.engine.paste_item import PasteItem
from labelforge.ui.main_window import ImageEditor

WORK = os.path.join(tempfile.gettempdir(), "lf_matte_crop_case")
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

editor = ImageEditor()
editor.resize(1200, 900)
editor.show()
app.processEvents()


def make_rgba(w=40, h=30):
    """左半边不透明红、右半边全透明，便于验证裁剪位置。"""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :w // 2] = (200, 30, 30, 255)
    return rgba


def new_material(rgba, rect, label="car", angle=0.0, name="car_001.png"):
    path = os.path.join(WORK, name)
    save_matte_png(rgba, path)
    item = PasteItem(pixmap=rgba_to_pixmap(rgba), rect=rect, label=label,
                     angle=angle, path=path)
    editor.canvas_items = [item]
    editor.selected_item = None
    editor.save_undo_state = lambda: None
    editor._refresh_export_task_label = lambda: None
    editor._refresh_small_list_view = lambda: None
    return item, path


def read(path):
    from PySide6.QtGui import QImage
    image = QImage(path).convertToFormat(QImage.Format_RGBA8888)
    arr = np.frombuffer(image.bits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine()
    )[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()
    return arr
'''


def test_crop_trims_the_item_and_writes_the_file_back():
    out = _run_real_qt(_SETUP + '''
item, path = new_material(make_rgba(), (100, 100, 40, 30))

print(editor.crop_matte_material(0, (110, 105, 10, 10)))

cropped = editor.canvas_items[0]
print([float(v) for v in cropped.rect_tuple()])
print(cropped.pixmap.width(), cropped.pixmap.height())
print(os.path.basename(cropped.path) == os.path.basename(path))

arr = read(path)
print(arr.shape)
# 原素材左半边不透明：裁 x=110..120 仍落在不透明区，alpha 应为满值
print(int(arr[:, :, 3].max()))
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "[110.0, 105.0, 10.0, 10.0]", lines[1]
    assert lines[2] == "10 10", "画布上的贴图没跟着裁"
    assert lines[3] == "True", "没写回原文件"
    assert lines[4] == "(10, 10, 4)", lines[4]
    assert lines[5] == "255", "裁剪后的像素应原样来自源素材"
    assert lines[6].startswith("已裁剪素材"), lines[6]


def test_crop_keeps_the_pixels_inside_the_opaque_half():
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30))

print(editor.crop_matte_material(0, (100, 100, 10, 10)))

arr = read(os.path.join(WORK, "car_001.png"))
print(arr.shape)
print(int(arr[0, 0, 0]), int(arr[0, 0, 3]))
''')
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "(10, 10, 4)", lines[1]
    assert lines[2] == "200 255", "不透明区的颜色与 alpha 要原样保留"


def test_crop_clamps_to_the_item_bounds():
    """裁剪框超出贴图范围时，只取交集。"""
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30))

print(editor.crop_matte_material(0, (90, 100, 200, 5)))

cropped = editor.canvas_items[0]
print([float(v) for v in cropped.rect_tuple()])
''')
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "[100.0, 100.0, 40.0, 5.0]", lines[1]


def test_crop_refuses_a_rotated_item():
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30), angle=45.0)

print(editor.crop_matte_material(0, (110, 105, 10, 10)))
print(editor.status_label.text() == "请先重置旋转再裁剪")
print(int(read(os.path.join(WORK, "car_001.png")).shape[0]))
''')
    lines = _lines(out)
    assert lines[0] == "False", "旋转过的贴图不能裁剪"
    assert lines[1] == "True", lines[1]
    assert lines[2] == "30", "拒绝时文件不能动"


def test_crop_refuses_an_item_without_a_source_file():
    out = _run_real_qt(_SETUP + '''
from labelforge.engine.paste_fusion import rgba_to_pixmap

editor.canvas_items = [PasteItem(pixmap=rgba_to_pixmap(make_rgba()),
                                 rect=(100, 100, 40, 30), label="car")]
editor.save_undo_state = lambda: None

print(editor.crop_matte_material(0, (110, 105, 10, 10)))
print(editor.status_label.text() == "该贴图不是抠图素材，无法写回文件")
''')
    lines = _lines(out)
    assert lines[0] == "False", lines[0]
    assert lines[1] == "True", lines[1]


def test_crop_refuses_an_empty_area():
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30))

print(editor.crop_matte_material(0, (100, 100, 0.5, 0.5)))
print(editor.status_label.text() == "裁剪范围太小")
print(int(read(os.path.join(WORK, "car_001.png")).shape[0]))
''')
    lines = _lines(out)
    assert lines[0] == "False", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "30", "拒绝时文件不能动"


def test_crop_entry_puts_the_canvas_into_crop_mode():
    """右键「裁剪素材…」进来时不能抛异常；裁剪框先等于贴图本身。"""
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30))
editor.selected_item = 0

editor.canvas._start_paste_crop(0)
print(editor.canvas._crop_paste_active())
print(editor.canvas.cursor().shape() == Qt.CrossCursor)
print([float(v) for v in editor.canvas._crop_rect.getRect()])
print(editor.status_label.text() == "裁剪素材：拖动边框调整范围，回车应用，Esc 取消")

class _Key:
    def __init__(self, key):
        self._key = key
        self.accepted = False
    def key(self):
        return self._key
    def accept(self):
        self.accepted = True

editor.canvas.keyPressEvent(_Key(Qt.Key_Escape))
print(editor.canvas._crop_paste_active())
print(editor.status_label.text() == "已取消裁剪")
''')
    lines = _lines(out)
    assert lines[0] == "True", f"没进入裁剪模式：{lines}"
    assert lines[1] == "True", "光标应变成十字"
    assert lines[2] == "[100.0, 100.0, 40.0, 30.0]", lines[2]
    assert lines[3] == "True", lines[3]
    assert lines[4] == "False", "Esc 应退出裁剪模式"
    assert lines[5] == "True", lines[5]


def test_crop_drags_the_right_edge_and_applies_on_enter():
    """拖右边框收窄 → 回车应用：文件与画布上的贴图都跟着变。"""
    out = _run_real_qt(_SETUP + '''
new_material(make_rgba(), (100, 100, 40, 30))
editor.canvas._start_paste_crop(0)

# 没有背景图时画布算不出图像坐标：直接把「屏幕坐标=图像坐标」钉住
editor.canvas._canvas_to_image_point = lambda pos, rect=None: [pos.x(), pos.y()]

class _Pos:
    def __init__(self, x, y):
        self._x, self._y = x, y
    def x(self):
        return self._x
    def y(self):
        return self._y

print(editor.canvas._crop_handle_at(_Pos(140, 115)))
editor.canvas._crop_handle = "r"
editor.canvas._update_crop_rect(_Pos(120, 115))
print([float(v) for v in editor.canvas._crop_rect.getRect()])

class _Key:
    def __init__(self, key):
        self._key = key
    def key(self):
        return self._key
    def accept(self):
        pass

editor.canvas.keyPressEvent(_Key(Qt.Key_Return))
print([float(v) for v in editor.canvas_items[0].rect_tuple()])
print(int(read(os.path.join(WORK, "car_001.png")).shape[1]))
print(editor.status_label.text().startswith("已裁剪素材"))
''')
    lines = _lines(out)
    assert lines[0] == "r", f"右边框没命中：{lines[0]}"
    assert lines[1] == "[100.0, 100.0, 20.0, 30.0]", lines[1]
    assert lines[2] == "[100.0, 100.0, 20.0, 30.0]", lines[2]
    assert lines[3] == "20", lines[3]
    assert lines[4] == "True", lines[4]



def test_material_placed_from_the_list_keeps_its_file_and_crops_back():
    """从贴图列表点到画布的素材也要能裁剪：写回列表里那张图，缩略图同步换新。"""
    out = _run_real_qt(_SETUP + '''
path = os.path.join(WORK, "list_001.png")
save_matte_png(make_rgba(), path)
editor.small_images = [(path, rgba_to_pixmap(make_rgba()))]
editor.current_background = rgba_to_pixmap(np.zeros((200, 200, 4), np.uint8))
editor.canvas_items = []
editor._busy = False
editor._is_delete_view = False


class _Item:
    def data(self, role):
        return 0


editor.add_small_to_canvas(_Item())
item = editor.canvas_items[0]
print(getattr(item, "path", None) == path)
editor.canvas._start_paste_crop(0)
print(editor.canvas._crop_paste_active())
x, y, w, h = item.rect_tuple()
print(editor.crop_matte_material(0, (x, y, w / 2.0, h)))
print(int(read(path).shape[1]))
print(int(editor.small_images[0][1].width()))
''')
    lines = _lines(out)
    assert lines[0] == "True", f"列表点进来的素材没记住源文件：{lines}"
    assert lines[1] == "True", f"裁剪入口被拒：{lines}"
    assert lines[2] == "True", f"裁剪失败：{lines}"
    assert lines[3] == "20", f"磁盘 PNG 没写回：{lines}"
    assert lines[4] == "20", f"列表缩略图没同步：{lines}"


def test_random_paste_also_records_the_source_file():
    """随机贴图建出的项同样带源文件，否则那批素材一样裁不了。"""
    from pathlib import Path

    source = (Path(ROOT) / "labelforge" / "engine" / "paste_engine.py").read_text(
        encoding="utf-8")
    assert "PasteItem(pixmap, rect, paste_label, path=source_path)" in source
    assert "PasteItem(pixmap, rect, paste_label, path=file_path)" in source
