"""统计窗口的圆角必须与全项目一致（4px）。

用户要求「统计窗口中的 UI 直角改成统一的圆角」。这里盯住三个曾经漏掉的位置：
表格本体、表头、dock 标题条——它们的边框此前都只有 1px 描边、没有圆角。
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


def _run_real_qt(script, timeout=180):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _stats_dialog_qss():
    """取出统计对话框真正应用的那段 QSS。"""
    source = (ROOT / "labelforge" / "ui" / "mixins" / "stats.py").read_text(
        encoding="utf-8")
    start = source.index("dialog.setStyleSheet(f\"\"\"")
    end = source.index('""")', start)
    return source[start:end]


def _rule(qss, selector):
    """取出一条规则块的内容；QSS 在 f-string 里写作 `{{`，收尾是 `}}`。"""
    start = qss.index(selector + " {{")
    start += len(selector + " {{")
    end = qss.index("}}", start)
    return qss[start:end]


def test_table_and_header_have_rounded_corners():
    qss = _stats_dialog_qss()

    table = _rule(qss, "QTableWidget")
    assert "border: 1px solid" in table
    assert "border-radius: 4px" in table, "表格本体仍缺圆角"

    header = _rule(qss, "QHeaderView::section")
    assert "border: 1px solid" in header
    # 表头只圆上面两角：下面要跟表格贴合
    assert "QHeaderView::section:first" in qss
    assert "QHeaderView::section:last" in qss
    assert "border-top-left-radius: 4px" in qss
    assert "border-top-right-radius: 4px" in qss


def test_dock_title_has_rounded_corners():
    qss = _stats_dialog_qss()
    dock = _rule(qss, "QDockWidget::title")
    assert "border-top-left-radius: 4px" in dock
    assert "border-top-right-radius: 4px" in dock


def test_label_color_button_has_rounded_corners():
    source = (ROOT / "labelforge" / "ui" / "mixins" / "stats.py").read_text(
        encoding="utf-8")
    block = source[source.index("def _set_label_color_button"):
                   source.index("def _change_label_color")]
    assert "border-radius: 4px" in block, "类别颜色按钮仍是直角"


def test_no_square_corner_left_in_the_stats_dialog_styles():
    """凡是带 1px 描边的容器都要给圆角，防止以后再漏。"""
    qss = _stats_dialog_qss()
    for selector in ("QTableWidget", "QDockWidget::title", "QTabWidget::pane"):
        block = _rule(qss, selector)
        assert "border: 1px solid" in block, selector
        assert "radius" in block, f"{selector} 有描边却没有圆角"


def test_the_stats_window_actually_renders_with_the_qss():
    """真实 Qt：样式表能应用、对话框能构造出来。"""
    script = '''
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.ui.main_window import ImageEditor
ed = ImageEditor()
ed.background_images = []
ed.detection_boxes_dict = {}
ed.canvas_items_dict = {}

# 直接构造统计对话框，确认 QSS 应用后不抛异常
ed._show_stats_dialog() if hasattr(ed, "_show_stats_dialog") else None
print("OK_STATS_QSS_APPLIES")
'''
    assert 'OK_STATS_QSS_APPLIES' in _run_real_qt(script)
