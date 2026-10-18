"""日志写入在真实 Qt 下必须不炸。

用户报过：数据增强、标签导出一点就
`AttributeError: 'QTextCursor' object has no attribute 'End'`。
根因是 `cursor.movePosition(cursor.End)` —— 在**实例**上取枚举成员，
PyQt5 允许、PySide6 不允许（枚举得走类 `QTextCursor.MoveOperation.End`）。
两个日志框都踩了，而且旧测试完全没覆盖 `_log`，所以迁移后一直没暴露。

mock 的 PySide6 不会复现这个错误（替身没有 `End` 这个概念），必须真 Qt。
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
from PySide6.QtWidgets import QApplication, QTextEdit

from labelforge.ui.dataset_tools_dialog import DatasetToolsDialog
from labelforge.ui.processing_panel import ProcessingPanel

app = QApplication([])


class _Bare:
    """只带一个日志框的替身：够跑 `_log`，不用搭整个面板。"""

    def __init__(self):
        self._log_area = QTextEdit()


for cls in (ProcessingPanel, DatasetToolsDialog):
    obj = _Bare()
    for i in range(3):
        cls._log(obj, f"line{i}")
    lines = obj._log_area.toPlainText().splitlines()
    assert len(lines) == 3, (cls.__name__, lines)
    # 光标停在末尾、日志按时间戳前缀追加
    assert lines[-1].endswith("line2"), (cls.__name__, lines[-1])
    assert lines[-1].startswith("["), (cls.__name__, lines[-1])

print("OK")
'''


def _run():
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_panel_log_writes_without_crashing():
    assert "OK" in _run()