"""加载流程和滚轮优先级回归测试。"""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

_QT_EXIT_GUARD = """
import sys
sys.stdout.flush()
sys.exit(0)
"""


def _run_real_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=180, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_load_background_folder_initializes_first_image_inside_loop():
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")

    assert "if load_first:" in source
    assert "self.current_background = self._get_cached_pixmap(first_path)" in source


def test_load_background_folder_does_not_block_background_list_during_label_scan():
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")
    method_source = source[source.index("    def load_background_folder("):source.index("    def upload_small_images(")]

    assert "self._show_loading_spinner()" not in method_source
    assert "self._hide_loading_spinner()" not in method_source
    assert "self.background_list.setUpdatesEnabled(False)" not in method_source
    assert "self.background_list.setUpdatesEnabled(True)" not in method_source


def test_ctrl_wheel_scales_background_before_selected_objects():
    """Ctrl+滚轮必须先给背景，不能被选中的形状或检测框截走。

    这条原来靠读源码里三行的先后顺序判断，改一点结构就断，也证明不了运行时
    真的按这个顺序走。现在实际构造画布、发一次 Ctrl+滚轮，看哪个处理函数被调用。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent

app = QApplication([])
from labelforge.canvas.canvas import Canvas

calls = []
canvas = Canvas.__new__(Canvas)

class _Ed:
    current_background = object()
    selected_item = object()
    detection_boxes = [{"shape_type": "rectangle", "x": 0, "y": 0,
                        "width": 10, "height": 10}]

canvas._editor = _Ed()
canvas.selected_box = 0
canvas.mouse_pos = QPoint(5, 5)
canvas.background_scale = 1.0

for name in ("_scale_background", "_scale_selected_item", "_scale_selected_box",
             "_wheel_region", "_adjust_selected_box_edge"):
    setattr(canvas, name, (lambda n: lambda *a, **k: calls.append(n))(name))
canvas._region_editable = lambda: True
canvas._wheel_target_region = lambda: 0
canvas.find_item_at_position = lambda pos: None
canvas._selected_box_takes_wheel = lambda: False
canvas.update_status_label = lambda: None
canvas.update = lambda: None
canvas._notify_region_changed = lambda: None

ev = QWheelEvent(QPoint(5, 5), QPoint(5, 5), QPoint(0, 0), QPoint(0, 120),
                 Qt.NoButton, Qt.ControlModifier, Qt.NoScrollPhase, False)
canvas.wheelEvent(ev)
print("CALLS", ",".join(calls) or "NONE")
""")
    calls = None
    for line in out.strip().splitlines():
        if line.startswith("CALLS"):
            calls = line.split()[1]
    assert calls == "_scale_background", (
        f"Ctrl+滚轮应只动背景，实际调用: {calls}")
