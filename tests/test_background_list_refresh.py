"""背景图列表「刷新」按钮：就地重扫文件夹增删列表项，不重新导入。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(script, timeout=180):
    """真实 Qt 子进程执行；显式 UTF-8 + 超时，避免 gbk 解码崩溃与无限挂起。"""
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                           text=True, capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"Qt 子进程 {timeout}s 超时\nstdout: {e.stdout}\nstderr: {e.stderr}") from None
    assert r.returncode == 0, r.stderr
    return r.stdout


_BOOT = '''
import os, time
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from pastelabel.ui.main_window import ImageEditor

d = r"{d}"
folder = os.path.join(d, "imgs")
os.makedirs(folder, exist_ok=True)
for name in ("a.png", "b.png"):
    QImage(40, 30, QImage.Format_RGB888).save(os.path.join(folder, name))
window = ImageEditor()
window.load_background_folder(folder)
for _ in range(4000):
    if len(window.background_images) >= 2:
        break
    app.processEvents()
assert [os.path.basename(p) for p in window.background_images] == ["a.png", "b.png"], window.background_images
'''


def test_refresh_button_syncs_added_and_removed_files(tmp_path):
    # 文件夹增删后点刷新：列表就地更新，未保存的框按路径保住，选中不丢。
    script = _BOOT.format(d=tmp_path.as_posix()) + '''
# 模拟未保存的标注（只在内存里）
window.detection_boxes_dict[0] = [{"label": "cat", "shape_type": "rectangle",
                                   "x": 1, "y": 1, "width": 5, "height": 5}]
window.detection_boxes = list(window.detection_boxes_dict[0])
window.current_background_index = 0

# 本地增 c.png、删 b.png
QImage(40, 30, QImage.Format_RGB888).save(os.path.join(folder, "c.png"))
os.remove(os.path.join(folder, "b.png"))

changed, added, removed = window.refresh_background_list()
assert changed is True and added == 1 and removed == 1, (changed, added, removed)
assert [os.path.basename(p) for p in window.background_images] == ["a.png", "c.png"], window.background_images
assert window.background_list.count() == 2, window.background_list.count()
# 未保存的框按路径迁移到新索引，没有被清掉
assert window.detection_boxes_dict[0] and window.detection_boxes_dict[0][0]["label"] == "cat", window.detection_boxes_dict
assert window.detection_boxes_dict[1] == []
# 选中仍停在 a.png
assert window.current_background_index == 0
assert os.path.basename(window.background_images[window.current_background_index]) == "a.png"
print("OK_REFRESH_SYNC")
'''
    assert "OK_REFRESH_SYNC" in _run_qt(script)


def test_refresh_button_exists_right_of_filter_and_click_works(tmp_path):
    # 刷新图标在背景列表表头最右（筛选按钮右侧），点击即触发同步。
    script = _BOOT.format(d=tmp_path.as_posix()) + '''
btn = window.bg_refresh_btn
assert btn.parentWidget() is window.bg_filter_btn.parentWidget()
assert btn.x() > window.bg_filter_btn.x(), (btn.x(), window.bg_filter_btn.x())

QImage(40, 30, QImage.Format_RGB888).save(os.path.join(folder, "z.png"))
btn.click()
assert [os.path.basename(p) for p in window.background_images] == ["a.png", "b.png", "z.png"], window.background_images
assert window.background_list.count() == 3
print("OK_REFRESH_BUTTON")
'''
    assert "OK_REFRESH_BUTTON" in _run_qt(script)


def test_refresh_without_folder_or_without_change_is_noop(tmp_path):
    # 没加载过文件夹 → 提示且不崩；文件夹没变化 → 空操作，不动内存里的框。
    script = f'''
import os
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from pastelabel.ui.main_window import ImageEditor

window = ImageEditor()
changed, added, removed = window.refresh_background_list()
assert (changed, added, removed) == (False, 0, 0), (changed, added, removed)

d = r"{tmp_path.as_posix()}"
folder = os.path.join(d, "imgs2")
os.makedirs(folder, exist_ok=True)
QImage(40, 30, QImage.Format_RGB888).save(os.path.join(folder, "a.png"))
window.load_background_folder(folder)
for _ in range(2000):
    if len(window.background_images) >= 1:
        break
    app.processEvents()
window.detection_boxes_dict[0] = [{{"label": "keep"}}]
changed, added, removed = window.refresh_background_list()
assert (changed, added, removed) == (False, 0, 0)
assert window.detection_boxes_dict[0] == [{{"label": "keep"}}]
print("OK_REFRESH_NOOP")
'''
    assert "OK_REFRESH_NOOP" in _run_qt(script)
