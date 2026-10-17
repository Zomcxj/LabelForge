"""加载过的背景文件夹被删掉后，点「刷新」要清空列表与画布。

用户报的场景：软件运行期间把已加载的文件夹删了（或拔了盘），再点背景图列表
的刷新按钮。改之前 `refresh_background_list()` 只把 `_refresh_notice` 设成
「当前没有已加载的背景文件夹，无法刷新」就 `return False`——列表里那几行还
留着，全都指向不存在的文件；画布也还停在最后一张图上。用户点任何一行都失败。

现在区分两种情况：
  * 从来没加载过（`_memory_background_path` 为空）→ 保持原提示，没东西可清；
  * 加载过但现在不存在（目录没了）→ **清空列表、画布与按索引缓存的标注**，
    并给一句说清「文件夹已不存在」的提示。

这里全用真 ImageEditor + 真文件/真目录，因为要断言的是「列表行数、画布状态
这些真实控件状态」，替身测不出来。
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_LOADED = """
import os, shutil, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor

app = QApplication([])

def make_editor(names=("a.jpg", "b.jpg")):
    tmp = tempfile.mkdtemp(prefix="pl-gone-")
    paths = []
    for n in names:
        p = os.path.join(tmp, n)
        QImage(80, 60, QImage.Format_RGB888).save(p)
        paths.append(p)
    w = ImageEditor()
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.background_images = list(paths)
    w.background_list.clear()
    for p in paths:
        w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
    w.current_background_index = 0
    w.current_background = QPixmap.fromImage(QImage(paths[0]))
    box = {"label": "x", "x": 1, "y": 1, "width": 4, "height": 4,
           "shape_type": "rectangle"}
    w.detection_boxes = [dict(box)]
    w.canvas_items = []
    w.detection_boxes_dict = {i: [dict(box)] for i in range(len(paths))}
    w.canvas_items_dict = {i: [] for i in range(len(paths))}
    w._memory_background_path = tmp
    return w, tmp, paths
"""


def test_refresh_clears_the_list_and_canvas_when_the_folder_is_gone():
    """文件夹被删 → 刷新后列表为空、画布空白、按索引缓存清空。"""
    out = _run_real_qt(_LOADED + """
import shutil
w, tmp, paths = make_editor()
print('刷新前 行数', w.background_list.count())
print('刷新前 图数', len(w.background_images))

shutil.rmtree(tmp)
changed, added, removed = w.refresh_background_list()
print('返回', changed, added, removed)
print('行数', w.background_list.count())
print('图数', len(w.background_images))
print('当前图没了', w.current_background is None)
print('索引 -1', w.current_background_index == -1)
print('检测框清空', (w.detection_boxes or []) == [])
print('画布项清空', (w.canvas_items or []) == [])
print('缓存清空', w.detection_boxes_dict == {} and w.canvas_items_dict == {})
print('路径记忆清空', w._memory_background_path == '')
""")
    lines = _lines(out)
    assert lines[0] == "刷新前 行数 2", lines
    assert lines[1] == "刷新前 图数 2", lines
    assert lines[3] == "行数 0", (
        f"文件夹已删，刷新后列表还是满的（{lines}）：用户点哪行都打不开")
    assert lines[4] == "图数 0", lines
    assert lines[5] == "当前图没了 True", (
        f"文件夹已删，画布还停在已不存在的图上（{lines}）")
    assert lines[6] == "索引 -1 True", lines
    assert lines[7] == "检测框清空 True", lines
    assert lines[8] == "画布项清空 True", lines
    assert lines[9] == "缓存清空 True", lines
    assert lines[10] == "路径记忆清空 True", (
        f"路径记忆还指着被删的文件夹（{lines}）")


def test_refresh_reports_the_folder_is_missing_not_never_loaded():
    """提示要区分「文件夹没了」与「从没加载过」——后者不该被说成文件夹不存在。"""
    out = _run_real_qt(_LOADED + """
import shutil
w, tmp, paths = make_editor()
shutil.rmtree(tmp)
w.refresh_background_list()
print('gone', w._refresh_notice)

w2 = ImageEditor()
w2.resize(1200, 800)
w2.show()
app.processEvents()
w2._memory_background_path = ''
w2.refresh_background_list()
print('never', w2._refresh_notice)
""")
    lines = _lines(out)
    assert lines[0].startswith("gone "), lines
    assert "不存在" in lines[0], (
        f"文件夹被删后的提示没说明文件夹没了（{lines}）")
    assert lines[1] == "never 当前没有已加载的背景文件夹，无法刷新", (
        f"从没加载过的提示被改坏了（{lines}）")


def test_refresh_still_works_when_the_folder_exists():
    """文件夹还在时不能误清：就地增删，未变则原样保留。

    这是上面那些用例的反向约束——「清空」只在文件夹真的没了时才允许触发。
    """
    out = _run_real_qt(_LOADED + """
import os
from PySide6.QtGui import QImage
w, tmp, paths = make_editor()

# 1) 无变化：不动
w.refresh_background_list()
print('无变化 行数', w.background_list.count())
print('无变化 当前图还在', w.current_background is not None)

# 2) 新增一张：就地加行，不清空
QImage(80, 60, QImage.Format_RGB888).save(os.path.join(tmp, 'c.jpg'))
changed, added, removed = w.refresh_background_list()
print('新增 返回', changed, added, removed)
print('新增 行数', w.background_list.count())
print('新增 图数', len(w.background_images))
print('新增后当前图还在', w.current_background is not None)
""")
    lines = _lines(out)
    assert lines[0] == "无变化 行数 2", lines
    assert lines[1] == "无变化 当前图还在 True", lines
    assert lines[2] == "新增 返回 True 1 0", lines
    assert lines[3] == "新增 行数 3", (
        f"文件夹还在却被清空了（{lines}）")
    assert lines[4] == "新增 图数 3", lines
    assert lines[5] == "新增后当前图还在 True", lines


def test_refresh_clears_everything_when_the_folder_becomes_a_file():
    """文件夹位置变成一个同名文件（不是目录）→ 同样按「没了」处理并清空。

    这条例外比「目录被删」更阴：路径存在，但不是目录，`os.path.isdir` 为假。
    原来 `not os.path.isdir(folder)` 与「从未加载」共用一条分支，现在要一起
    走到清空那条。
    """
    out = _run_real_qt(_LOADED + """
import os, shutil
w, tmp, paths = make_editor()
shutil.rmtree(tmp)
with open(tmp, 'w', encoding='utf-8') as f:
    f.write('not a folder')
w.refresh_background_list()
print('行数', w.background_list.count())
print('当前图没了', w.current_background is None)
""")
    lines = _lines(out)
    assert lines[0] == "行数 0", (
        f"路径变成文件时没清空列表（{lines}）")
    assert lines[1] == "当前图没了 True", lines
