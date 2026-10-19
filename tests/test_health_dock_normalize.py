"""统计界面右侧 dock 的等宽归一化测试。

`tests/conftest.py` 把 PySide6 换成了 mock，而这里要验的正是真实几何
（splitDockWidget / resizeDocks 的结果），所以走 offscreen 子进程。

背景：用户报「统计界面调整比例……无法一比一划分」。原因在记忆键 ——
原实现按**目标宽度**记 `last_norm[key] = width`，于是把一行拖成 200/400
时平均值仍是原来的 300，被判成「已经归一化过」而永远跳过。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 退出用 sys.exit（正常 finalize）而不是 os._exit：后者仍会经 ExitProcess 卸载 DLL
# 触发 Qt 静态析构，实测 150 次崩 12 次（0xC0000005/0xC0000409）；sys.exit 崩 0 次。
# 断言失败时脚本先抛异常，两种写法返回码都是 1，不会掩盖真实失败。
_QT_EXIT_GUARD = '''
import sys
sys.stdout.flush()
sys.exit(0)
'''

_SCRIPT = '''
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from labelforge.ui.mixins.stats import _equalize_dock_sizes, _side_by_side

app = QApplication([])
host = QMainWindow()
host.setDockNestingEnabled(True)
host.setCentralWidget(QWidget())
host.resize(1200, 700)
a, b = QDockWidget("a"), QDockWidget("b")
host.addDockWidget(Qt.RightDockWidgetArea, a)
host.addDockWidget(Qt.RightDockWidgetArea, b)
host.splitDockWidget(a, b, Qt.Horizontal)
host.show()


def settle(times=8):
    for _ in range(times):
        app.processEvents()


settle()
docks = [a, b]
key = tuple(id(d) for d in docks)
memo = {}
widths = lambda: [d.width() for d in docks]

# 1) 拖偏成 220/380，归一化必须拉回 1:1
host.resizeDocks(docks, [220, 380], Qt.Horizontal)
settle()
_equalize_dock_sizes(host, docks, memo, Qt)
settle()
first = widths()
assert max(first) - min(first) <= 2, ("首次归一化失败", first)

# 2) 几何生效后再跑一轮：读到已均衡，记忆要清掉，否则下次拖偏会被当成
#    「同一形态、已处理过」而跳过
_equalize_dock_sizes(host, docks, memo, Qt)
settle()
assert key not in memo, ("已均衡却留着记忆", memo)

# 3) 关键回归：把这一行拖偏，且**平均宽度不变**（300 → 240/360）。
#    旧实现按目标宽度记记忆，这一轮会被跳过，行永远回不到 1:1。
target = sum(first) // 2
host.resizeDocks(docks, [target - 60, target + 60], Qt.Horizontal)
settle()
_equalize_dock_sizes(host, docks, memo, Qt)
settle()
after = widths()
assert max(after) - min(after) <= 2, ("平均值不变时纠正失败", first, after)

# 4) 上下堆叠时 _side_by_side 为假 —— apply_ratio 就是用这个判据决定
#    要不要按行改宽度（否则会把用户重排好的堆叠弄乱）
host.addDockWidget(Qt.RightDockWidgetArea, b)
host.splitDockWidget(a, b, Qt.Vertical)
settle()
assert not _side_by_side(a.geometry(), b.geometry()), (
    "上下堆叠被判成并排", a.geometry(), b.geometry())

print("OK")
'''


def _run():
    env = os.environ | {
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": str(ROOT),
        # 子进程按 UTF-8 写、父进程按 UTF-8 读，不看本机 locale：CI 的 Windows
        # 是 cp1252，用 text=True 时子进程输出的中文会让读取线程抛
        # UnicodeDecodeError，stdout 直接变成 None（本地是 cp936 解得了，所以
        # 这个坑只在 CI 暴露，报出来还是个看不懂的 TypeError）。
        "PYTHONIOENCODING": "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, capture_output=True, timeout=120,
    )
    out = result.stdout.decode("utf-8", "replace")
    err = result.stderr.decode("utf-8", "replace")
    assert result.returncode == 0, out + err
    return out


def test_dock_row_can_be_normalized_back_to_one_to_one():
    assert "OK" in _run()