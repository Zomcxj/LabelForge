"""背景图列表「刷新」按钮：就地重扫文件夹增删列表项，不重新导入。"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# load_background_folder 会拉起后台标签扫描线程。不关窗就退出，解释器销毁
# ImageEditor 时线程还活着，Windows 上约 10% 概率在收尾阶段崩掉
# （0xC0000409 fail-fast / 0xC0000005 访问违例），退出码非零但 stdout/stderr
# 全空，看起来像"测试失败了"，实际业务断言早已通过并打印了标记。
# 处置分两层：
# 1) window.close() 走真实关闭路径（closeEvent 会请求中断扫描线程并
#    wait(2000)），顺便验证关闭流程本身不挂死；再排空事件让删除落地。
# 2) 做完上面这些后 flush + os._exit(0)：解释器退出阶段 Qt 对象析构顺序
#    不定，即使关窗干净仍有约 1/60 概率崩在析构里，与被测逻辑无关，
#    不值得继续追。必须先 flush —— os._exit 不冲刷缓冲，标记会丢，
#    断言就误报失败。不能只用 os._exit 跳过关闭，那会掩盖真实的挂死。
_TEARDOWN = (
    "\nwindow.close()\n"
    "for _ in range(200):\n"
    "    app.processEvents()\n"
    "import sys\n"
    "sys.stdout.flush()\n"
    "os._exit(0)\n"
)


def _run_qt(script, timeout=180):
    """真实 Qt 子进程执行；显式 UTF-8 + 超时，避免 gbk 解码崩溃与无限挂起。"""
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([sys.executable, "-c", script + _TEARDOWN], cwd=ROOT,
                           env=env,
                           text=True, capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"Qt 子进程 {timeout}s 超时\nstdout: {e.stdout}\nstderr: {e.stderr}") from None
    assert r.returncode == 0, r.stderr
    return r.stdout


_BOOT = '''
import os, time
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.main_window import ImageEditor

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


def test_nav_arrow_buttons_support_press_and_hold(tmp_path):
    # 长按 ◀/▶ 连续切图：Qt 内置 autoRepeat —— 按住不松手时 clicked 会
    # 先按 delay 再按 interval 持续重发，走既有的 switch_background 接线。
    script = _BOOT.format(d=tmp_path.as_posix()) + '''
for btn in (window.prev_img_btn, window.next_img_btn):
    assert btn.autoRepeat() is True, btn.text()
    assert 250 <= btn.autoRepeatDelay() <= 700, btn.autoRepeatDelay()
    assert 60 <= btn.autoRepeatInterval() <= 250, btn.autoRepeatInterval()
print("OK_NAV_REPEAT")
'''
    assert "OK_NAV_REPEAT" in _run_qt(script)


def test_drop_folder_onto_canvas_loads_it_as_backgrounds(tmp_path):
    # 文件夹拖到画布/窗口 = 直接打开该数据集（替换语义，与文件夹按钮一致）；
    # 拖入单张图片才是追加。dragEnter 也要接受文件夹（否则光标显示禁止）。
    script = _BOOT.format(d=tmp_path.as_posix()) + '''
from PySide6.QtCore import QUrl, QPoint, Qt, QMimeData
from PySide6.QtGui import QImage, QDragEnterEvent, QDropEvent

other = os.path.join(d, "imgs2")
os.makedirs(other, exist_ok=True)
QImage(40, 30, QImage.Format_RGB888).save(os.path.join(other, "x.png"))
QImage(40, 30, QImage.Format_RGB888).save(os.path.join(other, "y.png"))

mime = QMimeData()
mime.setUrls([QUrl.fromLocalFile(other)])
enter = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime,
                        Qt.LeftButton, Qt.NoModifier)
enter.ignore()
window.dragEnterEvent(enter)
assert enter.isAccepted(), "拖文件夹必须被接受，否则光标是禁止符号"

event = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime,
                   Qt.LeftButton, Qt.NoModifier)
window.dropEvent(event)
for _ in range(4000):
    if len(window.background_images) >= 2:
        break
    app.processEvents()
assert [os.path.basename(p) for p in window.background_images] == ["x.png", "y.png"], \\
    window.background_images
assert window.current_background is not None

# 只拖非图片文件（.txt）→ 不接受、不改动当前列表
txt = os.path.join(d, "readme.txt")
open(txt, "w", encoding="utf-8").close()
mime2 = QMimeData()
mime2.setUrls([QUrl.fromLocalFile(txt)])
enter2 = QDragEnterEvent(QPoint(10, 10), Qt.CopyAction, mime2,
                         Qt.LeftButton, Qt.NoModifier)
enter2.ignore()
window.dragEnterEvent(enter2)
assert not enter2.isAccepted(), "txt 不该被接受"
before = list(window.background_images)
drop2 = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime2,
                   Qt.LeftButton, Qt.NoModifier)
window.dropEvent(drop2)
assert list(window.background_images) == before, "txt 拖入不能改列表"
print("OK_DROP_FOLDER")
'''
    assert "OK_DROP_FOLDER" in _run_qt(script)


def test_refresh_without_folder_or_without_change_is_noop(tmp_path):
    # 没加载过文件夹 → 提示且不崩；文件夹没变化 → 空操作，不动内存里的框。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.main_window import ImageEditor

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
