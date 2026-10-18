"""统计窗图表 dock 的自动纠正：宽度、**高度**、左右比例。

用户报：「左侧可以 1:1 放置，右侧每次都是 1:2 或者 2:1」→ 指同一列上下两个
图表的高度。「调整比例，点击后拖动就变成了这个组件被拿走」→ 拖的是标题栏，
面板被搬走，比例跟着乱。期望：事后**全部自动纠正**回 1:1 与默认比例。

两条真 bug：
  1. 等高归一化只在「四个 dock 全部堆成一列」时跑，默认 2×2 是两列，所以
     同一列上下被拖成 1:2 后再也没人纠正。
  2. 拖分隔条只改几何、不发 dockLocationChanged，归一化根本等不到触发。

mock 的 PySide6 没有真实 dock 几何，必须真 Qt（offscreen 子进程）。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''

_SCRIPT = '''
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new, rewrite_disk=False):
        return False


class FakeEditor(StatsMixin, QDialog):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {
            0: [{"label": "bg_cls", "x": 0, "y": 0, "width": 10, "height": 10}]}
        self.canvas_items_dict = {0: [(None, None, "paste_cls")]}
        self.canvas_items = []
        self.current_background_index = 0
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = "x.png"
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor()
captured = {}


def _capture_exec(self):
    captured["dialog"] = self
    return 0


QDialog.exec_ = _capture_exec
editor._show_label_stats()
dialog = captured["dialog"]
host = dialog._dock_host
docks = dialog._health_docks

dialog.resize(1400, 820)
dialog.show()
QTest.qWait(200)
for _ in range(20):
    app.processEvents()

assert dialog._dock_norm_timer is not None, "防抖归一化定时器没装上"
assert dialog._dock_geometry_filter is not None, "几何过滤器没装上"


def settle():
    """等防抖定时器（120ms）跑完并让布局落定。"""
    QTest.qWait(260)
    for _ in range(20):
        app.processEvents()


def groups(axis):
    """按 y（行）或 x（列）分组，只留同组 ≥2 个的。"""
    out = {}
    for v in docks.values():
        key = v.y() if axis == "row" else v.x()
        out.setdefault(key, []).append(v)
    return [g for g in out.values() if len(g) > 1]


def check_one_to_one(what):
    """每一行等宽、每一列等高。

    不能全局比：重排后某个面板可能独占一列（那时它本来就该更高），要纠正的
    只是「同一行 / 同一列里互不相等」。
    """
    for g in groups("row"):
        w = [v.width() for v in g]
        assert max(w) - min(w) <= 3, (
            what + " 行内不等宽", [v.objectName() for v in g], w)
    for g in groups("col"):
        h = [v.height() for v in g]
        assert max(h) - min(h) <= 3, (
            what + " 列内不等高", [v.objectName() for v in g], h)


def check_ratio(what):
    """表格区 : 图表区 应维持默认 1:2。"""
    table = host.centralWidget().width()
    charts = host.width() - table
    assert 1.7 <= charts / table <= 2.3, (what, charts, table)
    return round(charts / table, 2)


# 初始：2×2，行内等宽、列内等高，表格:图表 ≈ 1:2
check_one_to_one("初始")
print("  初始 OK  表格:图表=", check_ratio("初始比例"))

# ① 同一列上下两个拖成高度 1:2 —— 用户报的「右侧总是 1:2 / 2:1」
col = sorted([v for v in docks.values()
              if v.x() == min(w.x() for w in docks.values())],
             key=lambda w: w.y())
assert len(col) == 2, [w.objectName() for w in col]
host.resizeDocks(col, [300, 100], Qt.Vertical)
for _ in range(10):
    app.processEvents()
settle()
check_one_to_one("拖高后")

# ② 同一行左右两个拖成宽度 2:1 —— 同时表格:图表 应回到默认 1:2
top = sorted([v for v in docks.values()
              if v.y() == min(w.y() for w in docks.values())],
             key=lambda w: w.x())
assert len(top) == 2, [w.objectName() for w in top]
host.resizeDocks(top, [200, 500], Qt.Horizontal)
for _ in range(10):
    app.processEvents()
settle()
check_one_to_one("拖宽后")
print("  拖过宽高 OK  表格:图表=", check_ratio("拖动后比例没回默认"))

# ③ 面板被搬走（水平方向重排）后，也应全部回到 1:1 + 默认比例
host.splitDockWidget(docks['aspect'], docks['size'], Qt.Horizontal)
for _ in range(10):
    app.processEvents()
settle()
check_one_to_one("重排后")
print("  重排后 OK  表格:图表=", check_ratio("重排后比例没回默认"))

editor._close_health_worker()
print("OK")
'''


def _run():
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_dock_layout_recovers_one_to_one_and_default_ratio():
    assert "OK" in _run()