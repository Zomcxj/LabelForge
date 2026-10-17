"""对话框回归测试。"""
from labelforge.ui import i18n
from labelforge.ui.dialog_helpers import center_on_parent
from labelforge.ui.dialogs import ProgressDialogFactory


def test_progress_dialog_cancel_button_uses_current_language():
    original = i18n.get_lang()
    i18n.set_lang("en")
    try:
        dialog = ProgressDialogFactory.create_progress_dialog(None, "保存进度", "正在保存所有图片...", 1)

        assert dialog.args[1] == "Cancel"
    finally:
        i18n.set_lang(original)


class Geometry:
    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self): return self._x
    def y(self): return self._y
    def width(self): return self._w
    def height(self): return self._h


class Widget:
    def __init__(self, geometry, parent=None):
        self._geometry = geometry
        self._parent = parent
        self.moved_to = None

    def geometry(self): return self._geometry
    def parent(self): return self._parent
    def move(self, x, y): self.moved_to = (x, y)


def test_center_on_parent_uses_parent_window_geometry():
    parent = Widget(Geometry(100, 200, 800, 600))
    dialog = Widget(Geometry(0, 0, 200, 100), parent)

    center_on_parent(dialog)

    assert dialog.moved_to == (400, 450)


def test_label_stats_dialog_centers_after_show_event():
    import inspect
    from pathlib import Path
    from labelforge.ui.main_window import ImageEditor

    stats_block = inspect.getsource(ImageEditor._show_label_stats)

    assert 'class _StatsDialog(QDialog):' in stats_block
    assert 'def showEvent(self, event):' in stats_block
    assert 'center_on_parent(self, self.parent())' in stats_block
    assert 'dialog.setMinimumSize(1200, 640)' in stats_block


def test_label_stats_aggregates_in_memory_pastes_from_other_images():
    from labelforge.ui.main_window import ImageEditor

    editor = ImageEditor.__new__(ImageEditor)
    editor.background_images = ["current.png", "other.png"]
    editor.current_background_index = 0
    editor.canvas_items = []
    editor.canvas_items_dict = {
        0: [(None, None, "cached-current")],
        1: [(None, None, "paste")],
    }

    assert editor._get_session_paste_stats() == {"paste": 1}


def test_label_color_uses_all_open_image_paste_labels_when_switching_current_image():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor

app = QApplication.instance() or QApplication([])
editor = ImageEditor()
editor.background_images = ["zebra.png", "apple.png"]
editor.label_colors = ["#111111", "#222222"]
editor.current_background_index = 0
editor.canvas_items = [(None, None, "zebra")]
editor.canvas_items_dict = {1: [(None, None, "apple")]}

assert editor.get_label_color("zebra") == "#222222"

editor.current_background_index = 1
editor.canvas_items = [(None, None, "apple")]
editor.canvas_items_dict = {0: [(None, None, "zebra")]}

assert editor.get_label_color("zebra") == "#222222"
assert editor.get_label_color("apple") == "#111111"

# Qt 子进程退出阶段析构顺序不定，偶发崩溃/挂死；断言通过后 flush + os._exit(0)
# 绕开析构（os._exit 不冲刷缓冲，必须先 flush）。
import os, sys
sys.stdout.flush()
os._exit(0)
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
