"""贴图面板布局与状态栏导出任务的显示条件。

- 贴图参数只剩「个数 + 短边尺寸」这一行（羽化/调色/投影已整行删除）；
- 右下角的「导出任务」只有贴图模式下才有意义，标注模式下隐藏。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Qt 退出时偶发崩溃，用 os._exit 跳过清理；先把 stdout 刷出去，否则
# 子进程的打印会丢在缓冲区里
_QT_EXIT_GUARD = "\nimport os, sys\nsys.stdout.flush()\nos._exit(0)\n"


def _run_real_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen",
                        "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


def test_paste_params_are_one_row_without_the_fusion_controls():
    """贴图参数只剩一行：个数 + 短边尺寸；融合三项与「应用到全部」已删除。"""
    out = _run_real_qt('''
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
from labelforge.ui.mixins.lists import ListsMixin

app = QApplication([])
host = ListsMixin.__new__(ListsMixin)
host._on_min_size_changed = lambda v: None
host._on_max_size_changed = lambda v: None

page = QWidget()
layout = QVBoxLayout(page)
host._create_paste_params(layout)
page.show()
app.processEvents()

def y(widget):
    return widget.mapTo(page, QPoint(0, 0)).y()

print(y(host.paste_count_spin), y(host.min_size_spin), y(host.max_size_spin))
for gone in ("feather_spin", "color_match_spin", "shadow_spin",
             "fusion_apply_all_btn"):
    print(hasattr(host, gone))
''')
    lines = _lines(out)
    first = [int(v) for v in lines[0].split()]
    assert len(set(first)) == 1, f"同一行没对齐：{first}"
    assert lines[1:] == ["False"] * 4, lines[1:]


class _Label:
    """记录 setText / setVisible 的替身。"""

    def __init__(self):
        self.text = ""
        self.visible = None
        self.visible_calls = []

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = visible
        self.visible_calls.append(visible)


class _Editor:
    def __init__(self, mode, boxes=None):
        self.edit_mode = mode
        self.export_task_label = _Label()
        self.canvas_items = []
        self.detection_boxes = boxes or []


def test_export_task_label_is_hidden_in_annotate_mode():
    from labelforge.ui.main_window import ImageEditor

    editor = _Editor("annotate", boxes=[{"x": 1, "y": 1, "width": 5,
                                        "height": 5, "label": "car"}])
    ImageEditor._refresh_export_task_label(editor)

    assert editor.export_task_label.visible is False, (
        "标注模式下不该显示导出任务")
    assert editor.export_task_label.text == "", (
        "隐藏时不该再算任务类型")


def test_export_task_label_shows_in_paste_mode():
    from labelforge.ui.i18n import t as tr
    from labelforge.ui.main_window import ImageEditor

    editor = _Editor("paste")
    ImageEditor._refresh_export_task_label(editor)

    assert editor.export_task_label.visible is True
    assert editor.export_task_label.text.startswith(f"{tr('导出任务')}: "), (
        editor.export_task_label.text)
