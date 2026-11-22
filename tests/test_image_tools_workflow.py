"""图像编辑的落盘、对比与预设（真实 Qt 子进程）。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SETUP = '''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton
import labelforge.ui.dialog_helpers as dh
from labelforge.ui.main_window import ImageEditor

def build(size=(200, 150), gray=20):
    root = tempfile.mkdtemp()
    folder = os.path.join(root, "ds")
    os.makedirs(folder)
    path = os.path.join(folder, "a.png")
    img = QImage(size[0], size[1], QImage.Format_RGB32)
    img.fill(QColor(gray, gray, gray))
    img.save(path)
    window = ImageEditor()
    window.background_images = [path]
    window.background_list.addItem("a.png")
    window.current_background_index = 0
    window.load_image_by_index(0)
    window._open_image_tools()
    return window, folder


def export_dir(folder):
    """默认输出目录：与数据集同级的 exports。"""
    return os.path.join(os.path.dirname(os.path.abspath(folder)), "exports")

app = QApplication([])
'''


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


def test_generate_new_image_leaves_the_original_untouched():
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
panel._checks["bright"].setChecked(True)
panel._param_widgets["bright"][0][1].setValue(80)
path = window.background_images[0]
before = open(path, "rb").read()
window._image_tools_save_new()
# 处理不改原图：新图写到同级的 export/edit，原文件一个字节都不变
assert open(path, "rb").read() == before, "原图被改动了"
out_dir = export_dir(folder)
new_path = os.path.join(out_dir, "a_bright.png")
assert os.path.exists(new_path), os.listdir(out_dir)
assert sorted(os.listdir(folder)) == ["a.png"], "不该写进数据集目录"
assert QImage(new_path).pixelColor(10, 10).lightness() > 60
window._image_tools_save_new()
assert os.path.exists(os.path.join(out_dir, "a_bright_2.png")), "重名要顺延编号"
print("OK")
''')
    assert "OK" in out


def test_generate_new_image_allows_geometric_operations():
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
panel._checks["rotate90"].setChecked(True)
window._image_tools_save_new()
new_path = os.path.join(export_dir(folder), "a_rotate90.png")
saved = QImage(new_path)
assert (saved.width(), saved.height()) == (150, 200), saved.size()
assert QImage(window.background_images[0]).size() == QImage(200, 150, QImage.Format_RGB32).size()
print("OK")
''')
    assert "OK" in out


def test_new_image_name_carries_the_operations_in_order():
    """文件名按处理顺序拼算子键名，重名再顺延编号。"""
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
panel._checks["fliph"].setChecked(True)
panel._checks["invert"].setChecked(True)
window._image_tools_save_new()
out_dir = export_dir(folder)
assert os.path.exists(os.path.join(out_dir, "a_fliph_invert.png")), os.listdir(out_dir)
window._image_tools_save_new()
assert os.path.exists(os.path.join(out_dir, "a_fliph_invert_2.png")), os.listdir(out_dir)
print("OK")
''')
    assert "OK" in out


def test_output_dir_is_only_remembered_for_this_run():
    """另存为选的目录只在本次运行内生效，不写进配置文件。"""
    out = _run_qt(_SETUP + '''
from labelforge.engine import image_tools as it

window, folder = build()
panel = window._image_tools_panel
panel._checks["invert"].setChecked(True)
default_dir = export_dir(folder)
assert window._image_tools_output_dir() == default_dir, window._image_tools_output_dir()

# 另存为换到别处，本次运行内的默认目录跟着变
other = os.path.join(folder, "elsewhere")
os.makedirs(other)
window._image_tools_ask_save_path = lambda default: os.path.join(other, "x.png")
window._image_tools_save_copy()
assert window._image_tools_output_dir() == other, window._image_tools_output_dir()
window._image_tools_save_new()
assert os.path.exists(os.path.join(other, "a_invert.png")), os.listdir(other)

# 配置里不该出现输出目录这一项
import json
cfg_path = os.environ["LABELFORGE_CONFIG_PATH"]
cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
assert "image_tools_output_dir" not in cfg, cfg.keys()
assert not hasattr(it, "save_output_dir"), "引擎里不该再有写配置的输出目录"
print("OK")
''')
    assert "OK" in out


def test_parameter_values_are_remembered_without_the_checkboxes():
    out = _run_qt(_SETUP + '''
from labelforge.engine import image_tools as it
from labelforge.ui.image_tools_panel import ImageToolsPanel
saved = {}
it.save_saved_params = lambda entries: saved.update(entries) or True
it.load_saved_params = lambda: {"bright": {"delta": -40},
                                "clahe": {"clip": 4.5},
                                "nope": {"x": 1}}
panel = ImageToolsPanel(None)
assert panel._checks["bright"].isChecked() is False, "勾选状态不该恢复"
assert panel._param_widgets["bright"][0][2]() == -40, "参数要沿用上次的值"
assert panel._param_widgets["clahe"][0][2]() == 4.5
panel._param_widgets["contrast"][0][1].setValue(1.75)
panel._persist_timer.stop()
panel._persist_params()
assert saved["contrast"]["factor"] == 1.75, saved
assert "nope" not in saved
print("OK")
''')
    assert "OK" in out


def test_save_copy_writes_a_full_size_file():
    out = _run_qt(_SETUP + '''
window, folder = build(size=(3000, 40))
panel = window._image_tools_panel
target = os.path.join(folder, "out.png")
window._image_tools_ask_save_path = lambda default: target
panel._checks["invert"].setChecked(True)
window._image_tools_save_copy()
assert os.path.exists(target), "另存为应当真的写出文件"
# 另存为选过的文件夹成为本次运行后续「生成新图」的位置
assert os.path.normcase(os.path.abspath(window._image_tools_output_dir())) == \
    os.path.normcase(os.path.abspath(folder)), window._image_tools_output_dir()
# 预览是降采样的，落盘必须按原图尺寸重跑，不能被预览尺寸带走
assert QImage(target).size() == QImage(window.background_images[0]).size()
assert QImage(target).pixelColor(5, 5).lightness() > 200
print("OK")
''')
    assert "OK" in out


def test_compare_shows_the_untouched_image():
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
panel._checks["invert"].setChecked(True)
window._image_tools_apply()
assert window.current_background.toImage().pixelColor(10, 10).lightness() > 200
window._image_tools_compare(True)
assert window.current_background.toImage().pixelColor(10, 10).lightness() < 60
window._image_tools_compare(False)
assert window.current_background.toImage().pixelColor(10, 10).lightness() > 200
print("OK")
''')
    assert "OK" in out


def test_preset_slots_go_from_a_to_e_then_overwrite_the_oldest():
    """最多五个预设（A~E），第六个覆盖最旧的那个。"""
    out = _run_qt('''
import os, tempfile
os.environ["LABELFORGE_CONFIG_PATH"] = os.path.join(tempfile.mkdtemp(), "c.json")
from PySide6.QtWidgets import QApplication
app = QApplication([])
from labelforge.engine import image_tools as it
from labelforge.ui.image_tools_panel import ImageToolsPanel

panel = ImageToolsPanel(None)
panel._checks["bright"].setChecked(True)
saved = []
for value in (-10, -20, -30, -40, -50, -60):
    panel._param_widgets["bright"][0][1].setValue(value)
    panel._on_save_preset()
    saved.append(panel._status.text())
assert saved[0].endswith("预设A"), saved
assert saved[4].endswith("预设E"), saved
assert saved[5].endswith("预设A"), saved + ["第六个应当覆盖最旧的"]
slots = [item["slot"] for item in it.load_presets()]
assert slots == ["A", "B", "C", "D", "E"], slots
# A 槽被第六次覆盖成最新那个值
assert it.preset_steps_of("A")[0]["params"]["delta"] == -60
assert it.preset_steps_of("B")[0]["params"]["delta"] == -20
print("OK")
''')
    assert "OK" in out


def test_broken_preset_config_is_ignored():
    out = _run_qt('''
import os, tempfile
os.environ["LABELFORGE_CONFIG_PATH"] = os.path.join(tempfile.mkdtemp(), "c.json")
from labelforge.core import config_manager
from labelforge.engine import image_tools as it

config = config_manager.load_config()
config["image_tools_presets"] = "不是列表"
config_manager.save_config(config)
assert it.load_presets() == []

config["image_tools_presets"] = [
    {"slot": "Z", "steps": [{"name": "bright", "params": {"delta": 10}}]},
    {"slot": "A", "steps": [{"name": "不存在的算子", "params": {}}]},
    {"slot": "B", "steps": [{"name": "bright", "params": {"delta": 10}}]},
    {"slot": "B", "steps": [{"name": "contrast", "params": {"factor": 1.2}}]},
    "坏条目",
]
config_manager.save_config(config)
regions = it.load_presets()
assert [item["slot"] for item in regions] == ["B"], regions
assert it.preset_steps_of("B")[0]["name"] == "bright", regions
print("OK")
''')
    assert "OK" in out


def test_save_result_flattens_alpha_for_jpeg():
    out = _run_qt('''
import os, tempfile
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
from labelforge.engine import image_tools as it
app = QApplication([])
folder = tempfile.mkdtemp()
src = QImage(8, 8, QImage.Format_ARGB32)
src.fill(QColor(0, 0, 0, 0))
target = os.path.join(folder, "a.jpg")
assert it.save_result(src, target) is True
saved = QImage(target)
assert saved.pixelColor(3, 3).lightness() > 200, saved.pixelColor(3, 3).name()
assert it.save_result(QImage(), os.path.join(folder, "b.png")) is False
print("OK")
''')
    assert "OK" in out


def test_panel_has_no_batch_augment_button():
    """批量增强与存为预设重复，只留「存为预设」。"""
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
names = [btn.text() for btn in panel.findChildren(QPushButton)]
assert "存为预设" in names, names
assert "批量增强" not in names, names
assert not hasattr(panel, "_to_augment_btn")
assert not hasattr(panel, "send_to_augment_requested")
print("OK")
''')
    assert "OK" in out


def test_compare_restores_the_view_for_a_resized_preview():
    """几何类操作改尺寸后按住看原图，图必须仍完整落在画布内。"""
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
panel._checks["rotate"].setChecked(True)
window._image_tools_apply()
rect_before = window.canvas.get_background_rect()
assert rect_before.top() >= 0, rect_before
window._image_tools_compare(True)
rect = window.canvas.get_background_rect()
assert rect.top() >= 0, f"原图顶出上边缘: {rect}"
assert rect.bottom() <= window.canvas.height(), f"原图超出下边缘: {rect}"
window._image_tools_compare(False)
assert window.canvas.get_background_rect().top() >= 0
print("OK")
''')
    assert "OK" in out


def test_save_notice_is_not_overwritten_by_the_preview_hint():
    """保存提示优先：预览流程随后写状态也不能把它冲掉，到点才回落。"""
    out = _run_qt(_SETUP + '''
from PySide6.QtTest import QTest
window, folder = build()
panel = window._image_tools_panel
panel.set_status("预览中, 标注已隐藏")
panel.flash_status("已生成新图: a_edit.png")
assert panel._status.text() == "已生成新图: a_edit.png", panel._status.text()
panel.set_status("预览中, 标注已隐藏")
assert panel._status.text() == "已生成新图: a_edit.png", panel._status.text()
panel._flash_timer.start(60)
QTest.qWait(200)
assert panel._status.text() == "预览中, 标注已隐藏", panel._status.text()

# 真实链路：存完新图，预览再跑一遍也盖不掉提示
panel._checks["rotate90"].setChecked(True)
app.processEvents()
window._image_tools_apply()
window._image_tools_save_new()
notice = panel._status.text()
assert notice.startswith("已生成新图"), notice
window._image_tools_apply()
assert panel._status.text() == notice, panel._status.text()

# 用户一改参数，上一条提示立刻作废
panel._checks["rotate90"].setChecked(False)
app.processEvents()
assert panel._status.text() != notice, panel._status.text()
print("OK")
''')
    assert "OK" in out


def test_status_line_keeps_the_buttons_in_place():
    """状态文字出现、变长、消失都不能把上面的按钮顶上去。"""
    out = _run_qt(_SETUP + '''
window, folder = build()
panel = window._image_tools_panel
before = (panel._save_btn.y(), panel._reset_btn.y(), panel._preset_save_btn.y())
for text in ("原图", "预览中, 标注已隐藏",
             "已存为 预设A，数据增强面板的下拉框里可以直接选到它，再点那边的增强"):
    panel.set_status(text)
    app.processEvents()
    now = (panel._save_btn.y(), panel._reset_btn.y(), panel._preset_save_btn.y())
    assert now == before, f"{text!r} 把按钮顶动了: {before} -> {now}"
panel.set_status("")
app.processEvents()
assert (panel._save_btn.y(), panel._reset_btn.y()) == before[:2]
# 三个按钮平分整行，不是各自缩到文字宽度
widths = {panel._save_btn.width(), panel._reset_btn.width(), panel._preset_save_btn.width()}
assert len(widths) == 1, widths
assert panel._save_btn.width() > panel._save_btn.sizeHint().width() + 20, widths
print("OK")
''')
    assert "OK" in out


def test_augment_runs_on_original_or_crop_even_without_a_preset():
    """原图与滑窗裁剪不依赖预设，单独勾选也要能跑。"""
    # 配置路径必须在任何 labelforge 导入之前改掉：CONFIG_PATH 是导入时算的，
    # 而同一 session 里别的用例已经往共享配置里存过预设。
    out = _run_qt("""
import os, tempfile
os.environ["LABELFORGE_CONFIG_PATH"] = os.path.join(tempfile.mkdtemp(), "c.json")
from PySide6.QtWidgets import QApplication
app = QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel

editor = type("E", (), {"background_images": []})()
panel = ProcessingPanel(editor)
panel._refresh_augment_presets()
assert panel._get_transform_specs() == [], "这个配置里应当没有预设"
# 默认勾着「包含原图」：没有预设也有活可干
assert panel._aug_inc_orig.isChecked()
assert panel._augment_has_work([], None, True) is True
assert panel._augment_has_work([], None, False) is False
assert panel._augment_has_work([("x", {})], None, False) is True
# 勾上滑窗裁剪也算有活可干，且能算出规格
panel._crop_check.setChecked(True)
crop_spec = panel._get_crop_spec()
assert crop_spec is not None, "滑窗裁剪应当有默认规格"
assert panel._augment_has_work([], crop_spec, False) is True
print("OK")
""")
    assert "OK" in out
