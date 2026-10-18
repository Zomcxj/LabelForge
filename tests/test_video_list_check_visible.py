"""视频窗「叠加标签框」的类别列表：勾选状态必须看得见。

用户报「叠加标签框列表不显示选中」。列表项是 `ItemIsUserCheckable`，`fill()`
把每一项都设成 Checked，但主题只给了 `QListWidget::item` 的样式、没给
`::indicator`，于是勾选框走平台原生绘制 —— Windows 下是缩在左上角、溢出色块
外的小勾，暗色主题里几乎看不出勾没勾上（截图里就是一片空框）。

这里用真实 Qt 抓图来判断：勾选态的指示器区域必须出现主题强调色，`全不选`
之后必须消失。只看「两态像素不同」不够 —— 原来那圈原生勾也让像素不同。
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
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from labelforge.ui.theme import ThemeManager
from labelforge.ui.video_dialog import VideoDialog

app = QApplication([])
app.setStyleSheet(ThemeManager.get_stylesheet())
t = ThemeManager.get_theme()
accent = QColor(t['accent'])


def band_colours(lst, index):
    """某个列表项指示器区域（左侧 22px）用到的颜色。"""
    item = lst.item(index)
    rect = lst.visualItemRect(item)
    off = lst.viewport().mapTo(lst, QPoint(0, 0))
    img = lst.grab().toImage()
    colours = set()
    for y in range(rect.top(), rect.bottom() + 1):
        for x in range(rect.left(), min(rect.left() + 22, rect.right()) + 1):
            px = QColor(img.pixel(x + off.x(), y + off.y()))
            colours.add((px.red(), px.green(), px.blue()))
    return colours


def near(c1, c2, tol=28):
    return all(abs(a - b) <= tol for a, b in zip(c1, c2))


def has_accent(colours):
    return any(near(c, (accent.red(), accent.green(), accent.blue()))
               for c in colours)


d = VideoDialog()
d._compose_images = []
# 让合成页进入「就绪」：ready 为假时子控件是禁用的，禁用的按钮 click() 不生效
import tempfile
d._compose_dir = tempfile.mkdtemp()
d.compose_info_label.setText("6 张图 / 0.2 秒")
d._fill_compose_labels({'person': 109, 'bicycle': 62, 'bird': 13})
d._sync_compose_enabled()
d.show()
for _ in range(10):
    app.processEvents()

lst = d.compose_label_list
assert lst.count() == 3, lst.count()
assert all(lst.item(i).checkState() == Qt.Checked for i in range(3)), "默认应全勾"
assert d.compose_none_btn.isEnabled(), "全不选按钮应可用（否则测不到点击）"

checked_bands = [band_colours(lst, i) for i in range(3)]
assert all(has_accent(b) for b in checked_bands), (
    "勾选态的指示器没有用主题强调色（看不出勾上了）", t['accent'],
    [{c for c in b if c[0] + c[1] + c[2] > 300} for b in checked_bands])

# 「全不选」之后同一个位置不该再有强调色
d.compose_none_btn.click()
for _ in range(10):
    app.processEvents()
assert all(lst.item(i).checkState() == Qt.Unchecked for i in range(3)), "全不选没生效"

unchecked_bands = [band_colours(lst, i) for i in range(3)]
assert not any(has_accent(b) for b in unchecked_bands), (
    "未勾选态仍然显示成勾上的样子", t['accent'])

print("ACCENT", t['accent'])
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


def test_checked_list_items_use_the_theme_accent():
    assert "OK" in _run()