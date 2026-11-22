"""图像编辑工具的入口与文案测试（真实 Qt 子进程）。"""
import os
import subprocess
import sys
from pathlib import Path

from labelforge.engine import image_tools as it
from labelforge.ui import i18n

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen",
                        "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c",
         script + "\nimport sys\nsys.stdout.flush()\nsys.exit(0)\n"],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=240,
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_numeric_inputs_are_right_aligned():
    """参数输入框右对齐，且整组贴着面板右边（中英文都不漂）。"""
    out = _run_qt('''
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QSpinBox)
from labelforge.ui.image_tools_panel import ImageToolsPanel

app = QApplication([])
panel = ImageToolsPanel(None)
want = int(Qt.AlignRight | Qt.AlignVCenter)
checked = 0
for key, widgets in panel._param_widgets.items():
    for name, widget, _getter in widgets:
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            checked += 1
            assert int(widget.alignment()) == want, (key, name, widget.alignment())
assert checked >= 30, f"应当有几十个数字输入框，实际 {checked}"

def right_gaps():
    """每一行最后一个参数框的右边缘，距面板右边还有多远。"""
    gaps = []
    for key, widgets in panel._param_widgets.items():
        if not widgets:
            continue
        last = widgets[-1][1]
        pos = last.mapTo(panel, QPoint(0, 0))
        gaps.append(panel.width() - (pos.x() + last.width()))
    return gaps

# 布局还没走完时几何是旧的，先让它算一遍
panel.resize(380, 720)
panel.show()
app.processEvents()
gaps = right_gaps()
# 面板左右各有 10px 边距 + 滚动条，最后一个输入框应当落在 10~40px 内
assert all(g <= 45 for g in gaps), f"输入框没贴右边: {sorted(gaps)[-5:]}"
print("OK")
''')
    assert "OK" in out


def test_panel_follows_a_language_switch_while_open():
    """面板开着切语言，所有静态文字都要跟着换，宽度也跟着变。"""
    out = _run_qt('''
import os, re, tempfile
from PySide6.QtGui import QColor, QFontMetrics, QImage
from PySide6.QtWidgets import (QApplication, QPushButton, QLabel, QCheckBox,
                               QComboBox, QWidget, QGroupBox)
from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor

CJK = re.compile(r"[\u4e00-\u9fff]")

def chinese_texts(panel):
    found = []
    for w in panel.findChildren(QWidget):
        if isinstance(w, QGroupBox):
            values = [w.title()]
        elif isinstance(w, (QPushButton, QLabel, QCheckBox)):
            values = [w.text(), w.toolTip()]
        elif isinstance(w, QComboBox):
            values = [w.itemText(i) for i in range(w.count())]
        else:
            continue
        found.extend(v for v in values if v and CJK.search(v))
    return found

app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "a.png")
img = QImage(80, 60, QImage.Format_RGB32)
img.fill(QColor(30, 30, 30))
img.save(path)

window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
window._open_image_tools()
app.processEvents()
panel = window._image_tools_panel

assert i18n.get_lang() == "zh", i18n.get_lang()
assert chinese_texts(panel), "中文模式下本该有中文"
zh_width = panel._checks["crop_region"].width()
assert "标注已隐藏" in panel._status_base, panel._status_base

try:
    window.toggle_language()
    app.processEvents()
    assert i18n.get_lang() == "en"
    left = chinese_texts(panel)
    assert not left, f"切英文后仍有中文: {left[:8]}"
    assert panel._save_btn.text() == "Save", panel._save_btn.text()
    # 状态行按宽度省略，比对未截断的原文字
    assert "hidden" in panel._status_base, panel._status_base

    # 英文文案更长，勾选框要跟着变宽，否则「Crop to selection」被截断
    check = panel._checks["crop_region"]
    en_width = check.width()
    assert en_width > zh_width, (zh_width, en_width)
    need = QFontMetrics(check.font()).horizontalAdvance(check.text())
    assert en_width >= need + 20, (en_width, need, check.text())
    assert panel._mode_combo.width() >= panel._mode_combo.sizeHint().width()
finally:
    if i18n.get_lang() != "zh":
        window.toggle_language()
        app.processEvents()

assert panel._save_btn.text() == "保存图片", panel._save_btn.text()
assert "标注已隐藏" in panel._status_base, panel._status_base
assert panel._checks["crop_region"].width() == zh_width, "切回中文宽度要还原"
print("OK")
''')
    assert "OK" in out


def test_export_menu_lists_image_tools():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication, QMenu
from labelforge.ui.main_window import ImageEditor

captured = {}
def fake_exec(self, *_args, **_kwargs):
    captured["texts"] = [a.text() for a in self.actions()]
    return None

QMenu.exec_ = fake_exec
app = QApplication([])
window = ImageEditor()
window._show_export_menu()
texts = captured["texts"]
assert any("图像编辑" in text for text in texts), texts
# 顺序：增强划分 / 图像编辑 / 数据集分组
assert texts.index("增强划分") < texts.index("图像编辑") < texts.index("数据集分组"), texts
print("OK")
''')
    assert "OK" in out


def test_open_image_tools_creates_the_dialog():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
assert hasattr(window, "_open_image_tools")
window._open_image_tools()
assert window._image_tools_panel is not None
assert window._image_tools_panel.windowTitle() == "图像编辑"
window._open_image_tools()
assert window._image_tools_panel is not None
print("OK")
''')
    assert "OK" in out


def test_every_operation_label_and_group_has_english():
    """操作名与分组名都要有英文，否则英文界面会显示中文。"""
    missing = []
    for key in it.OPERATION_ORDER:
        group, label, _params = it.OPERATION_META[key]
        for text in (group, label):
            if i18n._strings["en"].get(text, text) == text and not text.isascii():
                missing.append(text)
    assert not missing, sorted(set(missing))


def test_dialog_strings_have_english():
    keys = [
        "图像编辑", "未打开图片", "选区:", "无选区", "矩形", "椭圆", "多边形",
        "清除选区", "打开图片", "保存结果", "无法读取该图片",
        "选择图片", "请先打开一张图片", "没有勾选任何处理项", "已保存", "保存失败",
        "几何类操作会改变画布尺寸, 标注不再适用",
        "图片文件 (*.png *.jpg *.jpeg *.bmp)",
    ]
    for key in keys:
        assert key in i18n._strings["zh"], key
        assert i18n._strings["en"].get(key, key) != key, key


def test_preview_swaps_the_canvas_and_restores_on_close():
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "a.png")
img = QImage(64, 48, QImage.Format_RGB32)
img.fill(QColor(10, 20, 30))
img.save(path)
window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
before = window.current_background.toImage().pixelColor(32, 24).name()
assert window._image_tools_active is False
window._open_image_tools()
assert window._image_tools_active is True
assert window._image_tools_original is not None
window._image_tools_panel._checks["invert"].setChecked(True)
window._image_tools_apply()
after = window.current_background.toImage().pixelColor(32, 24).name()
assert after != before, (before, after)
window._image_tools_close()
restored = window.current_background.toImage().pixelColor(32, 24).name()
assert restored == before, (before, restored)
assert window._image_tools_active is False
print("OK")
''')
    assert "OK" in out


def test_open_panel_without_a_dataset_does_not_crash():
    out = _run_qt('''
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
window = ImageEditor()
window._open_image_tools()
assert window._image_tools_original is None
window._image_tools_apply()
window._image_tools_close()
assert window._image_tools_active is False
print("OK")
''')
    assert "OK" in out


def test_switch_background_does_not_overrun_a_single_image_dataset():
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
app = QApplication([])
folder = tempfile.mkdtemp()
path = os.path.join(folder, "only.png")
img = QImage(16, 16, QImage.Format_RGB32)
img.fill(QColor(9, 9, 9))
img.save(path)
window = ImageEditor()
window.background_images = [path]
window.current_background_index = 0
window.load_image_by_index(0)
window.switch_background(1)
assert window.current_background_index == 0
window.switch_background(-1)
assert window.current_background_index == 0
print("OK")
''')
    assert "OK" in out
