"""贴图列表的刷新、素材重命名、Shift+滚轮旋转，以及抠图后的自动选中。

四条都是实机使用中提出的：
- 列表里的文件在外部被改过/删过，需要一个刷新入口，不必重新选文件夹；
- 素材要能改名，改的是磁盘上的文件名；
- 贴图要能转任意角度，不再只有 90° 两档；
- 抠图产物要立刻选中，否则紧接着调羽化/调色画面不动，像是参数没用。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUARD = "\nimport os, sys\nsys.stdout.flush()\nos._exit(0)\n"


# Windows 上 Qt 子进程偶发的退出异常码：0xC0000005 访问冲突、
# 0xC0000409 栈缓冲溢出。CPU 被占满时更容易出现，属于环境问题。
_QT_EXIT_CRASH = {3221225477, 3221226505}


def _run_real_qt(script):
    """跑一段真实 Qt 脚本；退出崩溃时重试，最多三次。"""
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    last = None
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", script + _GUARD],
            cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=180, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout
        last = result
        if result.returncode not in _QT_EXIT_CRASH:
            break
    assert last is not None and last.returncode == 0, (
        f"exit={None if last is None else last.returncode} "
        + (last.stderr if last is not None else "子进程没有启动"))
    return last.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_SETUP = '''
import os, shutil, tempfile
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap

app = QApplication([])
from labelforge.ui.main_window import ImageEditor

WORK = os.path.join(tempfile.gettempdir(), "lf_paste_list_case")
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)

def make_png(name, color):
    path = os.path.join(WORK, name)
    img = QImage(40, 30, QImage.Format_RGB888)
    img.fill(color)
    img.save(path)
    return path

first = make_png("a.png", 0xFF0000)
second = make_png("b.png", 0x00FF00)

editor = ImageEditor()
editor.resize(1200, 900)
editor.show()
app.processEvents()
editor.small_images = [(first, QPixmap(first)), (second, QPixmap(second))]
editor.refresh_list_items()
app.processEvents()
'''


def test_refresh_drops_files_that_disappeared():
    out = _run_real_qt(_SETUP + '''
os.remove(first)
editor.refresh_paste_files()
print(len(editor.small_images))
print(os.path.basename(editor.small_images[0][0]))
print(editor.small_list.count())
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "1", f"失效文件没被剔除：{lines[0]}"
    assert lines[1] == "b.png", lines[1]
    assert lines[2] == "1", f"列表没跟着重建：{lines[2]}"
    assert "刷新" in lines[3], lines[3]


def test_refresh_sees_an_externally_modified_file():
    """外部覆盖了同名文件：刷新后缩略图换成新内容。"""
    out = _run_real_qt(_SETUP + '''
from PySide6.QtGui import QImage
img = QImage(40, 30, QImage.Format_RGB888)
img.fill(0x0000FF)
img.save(first)
editor.refresh_paste_files()
path, pixmap = editor.small_images[0]
print(pixmap.toImage().pixelColor(0, 0).blue() > 200)
''')
    assert _lines(out)[0] == "True"


def test_rename_moves_the_file_on_disk():
    out = _run_real_qt(_SETUP + '''
from labelforge.ui import dialog_helpers
dialog_helpers.get_text = lambda *a, **k: ("car_001", True)
editor.rename_paste_file(0)
print(os.path.basename(editor.small_images[0][0]))
print(os.path.exists(os.path.join(WORK, "car_001.png")))
print(editor.small_list.item(0).text())
''')
    lines = _lines(out)
    assert lines[0] == "car_001.png", lines[0]
    assert lines[1] == "True", "磁盘上的文件没改名"
    assert lines[2] == "car_001.png", "列表项没跟着改"


def test_refresh_picks_up_files_written_into_the_folder():
    """抠图会把新素材写进文件夹：文件夹加载的列表刷新后要能看到它们。"""
    out = _run_real_qt(_SETUP + '''
from PySide6.QtGui import QImage

editor.load_paste_folder(WORK)
print(editor.small_list.count())

img = QImage(40, 30, QImage.Format_RGB888)
img.fill(0x0000FF)
img.save(os.path.join(WORK, "c.png"))

editor.refresh_paste_files()
print(len(editor.small_images))
print([os.path.basename(path) for path, _ in editor.small_images])
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "2", lines[0]
    assert lines[1] == "3", f"新写入的文件没被扫进来：{lines[1]}"
    assert lines[2] == "['a.png', 'b.png', 'c.png']", lines[2]
    assert "新增 1" in lines[3], lines[3]


def test_refresh_does_not_pull_in_siblings_of_hand_picked_files():
    """逐张选出来的列表只重读这几张，不会把同目录其它图拉进来。"""
    out = _run_real_qt(_SETUP + '''
from PySide6.QtGui import QImage

# _SETUP 里是手工塞进去的两张：标记为「不是整个文件夹装进来的」
editor._paste_list_from_folder = False
editor._memory_paste_path = WORK

img = QImage(40, 30, QImage.Format_RGB888)
img.fill(0x0000FF)
img.save(os.path.join(WORK, "c.png"))

editor.refresh_paste_files()
print(len(editor.small_images))
print([os.path.basename(path) for path, _ in editor.small_images])
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "2", f"不该把 c.png 拉进来：{lines}"
    assert lines[1] == "['a.png', 'b.png']", lines[1]
    assert lines[2] == "贴图列表已刷新", lines[2]


def test_rename_uses_the_themed_dialog_with_chinese_buttons():
    """改名弹窗走 dialog_helpers.get_text：按钮是「确定/取消」，不是 OK/Cancel。"""
    out = _run_real_qt(_SETUP + '''
from PySide6.QtWidgets import QPushButton
from labelforge.ui import dialog_helpers

seen = {}

def fake_get_text(parent, title, label, text="", anchor_pos=None):
    seen["title"] = title
    seen["label"] = label
    seen["text"] = text
    return text, False

dialog_helpers.get_text = fake_get_text
editor.rename_paste_file(0)
print(seen["title"])
print(seen["label"])
print(seen["text"])

# 真对话框按钮文案
dlg = dialog_helpers.ThemedInputDialog(editor)
dlg.setOkButtonText(dialog_helpers.i18n.t("确定"))
dlg.setCancelButtonText(dialog_helpers.i18n.t("取消"))
print([b.text() for b in dlg.findChildren(QPushButton)][:2])
''')
    lines = _lines(out)
    assert lines[0] == "重命名素材", lines[0]
    assert lines[1] == "新文件名", lines[1]
    assert lines[2] == "a", lines[2]
    assert "确定" in lines[3] and "取消" in lines[3], lines[3]


def test_rename_refuses_an_existing_name():
    out = _run_real_qt(_SETUP + '''
from labelforge.ui import dialog_helpers
from labelforge.ui import dialog_helpers
warnings = []
dialog_helpers.warning = lambda *a, **k: warnings.append(a)
dialog_helpers.get_text = lambda *a, **k: ("b", True)
editor.rename_paste_file(0)
print(len(warnings))
print(os.path.basename(editor.small_images[0][0]))
''')
    lines = _lines(out)
    assert lines[0] == "1", "重名时应该提示而不是覆盖"
    assert lines[1] == "a.png", "重名时不能改名"


def test_rename_refuses_illegal_characters():
    out = _run_real_qt(_SETUP + '''
from labelforge.ui import dialog_helpers
from labelforge.ui import dialog_helpers
warnings = []
dialog_helpers.warning = lambda *a, **k: warnings.append(a)
dialog_helpers.get_text = lambda *a, **k: ("bad:name", True)
editor.rename_paste_file(0)
print(len(warnings))
print(os.path.basename(editor.small_images[0][0]))
''')
    lines = _lines(out)
    assert lines[0] == "1", "非法字符应该被拦下"
    assert lines[1] == "a.png", lines[1]


def test_shift_wheel_rotates_the_selected_item_by_one_degree():
    out = _run_real_qt(_SETUP + '''
from PySide6.QtGui import QPixmap
from labelforge.engine.paste_item import PasteItem, as_paste_item

editor.canvas_items = [PasteItem(pixmap=QPixmap(40, 30), rect=(10, 10, 40, 30),
                                 label="car")]
editor.selected_item = 0

class _Delta:
    def __init__(self, value):
        self._value = value
    def y(self):
        return self._value

class _Event:
    def __init__(self, value):
        self._value = value
    def angleDelta(self):
        return _Delta(self._value)

editor.canvas._rotate_selected_item_by_wheel(_Event(120))
print(as_paste_item(editor.canvas_items[0]).angle)
editor.canvas._rotate_selected_item_by_wheel(_Event(-120))
print(as_paste_item(editor.canvas_items[0]).angle)
''')
    lines = _lines(out)
    assert lines[0] == "1.0", f"滚轮一格应转 1°：{lines[0]}"
    assert lines[1] == "0.0", f"反向应转回 0°：{lines[1]}"


def test_matte_result_is_selected_after_it_lands():
    """抠图产物落到画布后立刻成为选中项，融合参数才作用得到它。"""
    out = _run_real_qt(_SETUP + '''
import numpy as np
from labelforge.engine.matting_service import MatteResult

rgba = np.zeros((20, 30, 4), dtype=np.uint8)
rgba[..., :3] = 200
rgba[..., 3] = 255
result = MatteResult(rgba=rgba, rect=(5, 6, 30, 20), path="x.png", label="car")

editor.canvas_items = []
editor.selected_item = None
editor.edit_mode = "paste"
editor._apply_matte_result(result)
print(editor.selected_item)
print(editor.selected_item_size)
''')
    lines = _lines(out)
    assert lines[0] == "0", f"抠图后没有选中新贴图：{lines[0]}"
    assert lines[1] == "(30.0, 20.0)", lines[1]


def test_delete_removes_the_file_after_confirmation():
    """删除素材：确认后删磁盘文件，并把它移出列表。"""
    out = _run_real_qt(_SETUP + '''
from PySide6.QtWidgets import QMessageBox
from labelforge.ui import dialog_helpers

asked = []
dialog_helpers.question = lambda parent, title, text, **kw: (
    asked.append(text) or QMessageBox.Yes)

editor.delete_paste_file(0)
print(len(asked))
print(os.path.exists(first))
print(len(editor.small_images))
print([os.path.basename(path) for path, _ in editor.small_images])
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "1", "应该弹确认框"
    assert lines[1] == "False", "文件没被删掉"
    assert lines[2] == "1", lines[2]
    assert lines[3] == "['b.png']", lines[3]
    assert "已删除素材" in lines[4] and "a.png" in lines[4], lines[4]


def test_delete_keeps_the_file_when_the_user_says_no():
    out = _run_real_qt(_SETUP + '''
from PySide6.QtWidgets import QMessageBox
from labelforge.ui import dialog_helpers

asked = []
dialog_helpers.question = lambda parent, title, text, **kw: (
    asked.append(text) or QMessageBox.No)

editor.delete_paste_file(0)
print(len(asked))
print(os.path.exists(first))
print(len(editor.small_images))
''')
    lines = _lines(out)
    assert lines[0] == "1", lines[0]
    assert lines[1] == "True", "取消确认后不该删文件"
    assert lines[2] == "2", lines[2]


def test_delete_confirm_text_names_the_file():
    out = _run_real_qt(_SETUP + '''
from PySide6.QtWidgets import QMessageBox
from labelforge.ui import dialog_helpers

asked = []
dialog_helpers.question = lambda parent, title, text, **kw: (
    asked.append(text) or QMessageBox.No)

editor.delete_paste_file(0)
print(asked[0].splitlines()[-1])
''')
    assert _lines(out)[0] == "a.png", _lines(out)


def test_list_menu_survives_a_refresh_while_it_is_open():
    """菜单开着时列表被重建（clear 会删掉 QListWidgetItem）也不能崩。

    实机上报过：右键「删除素材」→ 菜单期间列表被刷新 → 点确定时抛
    `RuntimeError: Internal C++ object already deleted`。
    """
    out = _run_real_qt(_SETUP + '''
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMenu, QMessageBox
from labelforge.ui import dialog_helpers
from labelforge.ui.mixins import lists as lists_module

dialog_helpers.question = lambda *a, **kw: QMessageBox.Yes

# 菜单一打开就把列表重建一次，再返回「删除素材…」
real_menu = QMenu

class RefreshingMenu(real_menu):
    def exec_(self, *args, **kwargs):
        target = None
        for action in self.actions():
            if action.text() == "删除素材…":
                target = action
        editor.refresh_list_items()
        return target

lists_module.QMenu = RefreshingMenu

# 命中测试用替身，位置不重要
editor.small_list.itemAt = lambda pos: editor.small_list.item(0)
editor._show_paste_list_menu(QPoint(0, 0))

print(os.path.exists(first))
print(len(editor.small_images))
print(editor.status_label.text())
''')
    lines = _lines(out)
    assert lines[0] == "False", "确认后文件应被删掉（说明没有在已删除的 item 上取值）"
    assert lines[1] == "1", lines[1]
    assert "已删除素材" in lines[2], lines[2]
