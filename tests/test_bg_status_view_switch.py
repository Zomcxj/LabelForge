"""工作路径 / 移除路径来回切换时，已标注的图必须保持绿点。

根因（修复前）：`_show_work_view()` 重建列表项时用 `light=True`，只做
「sidecar 文件在不在」的廉价判断——文件在就返回 STATUS_PENDING（灰圈
「扫描中」）。真实状态要等后台扫描 worker 回报后才由
`_apply_background_statuses()` 填回。

首次加载数据集时后台扫描会跑，状态会被补上；但切换视图只是重建列表，
不会重新触发扫描，于是已标注的图停在灰圈上，直到切图（触发按图加载）
才恢复正常。修复方式是把扫描/保存得到的状态按路径缓存下来，重建列表时
直接复用。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from labelforge.engine.image_loader.status import (  # noqa: E402
    STATUS_ANNOTATED, STATUS_PENDING, annotation_status_for_image,
    annotation_status_for_image_light, decorate_background_list_item,
)

# Qt 子进程退出阶段析构顺序不定，断言通过后立即 flush + os._exit(0) 绕开。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_real_qt(script, timeout=180):
    """真实 Qt 子进程执行；必须显式 UTF-8，否则中文 Windows 下按 gbk 解码。"""
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


def _make_annotated(tmp_path):
    """造一张有 sidecar 且有 shapes 的图。"""
    img = tmp_path / 'a.png'
    img.write_bytes(b'\x89PNG\r\n\x1a\n')
    (tmp_path / 'a.json').write_text(
        '{"shapes": [{"label": "car", "points": [[0, 0], [1, 1]]}]}',
        encoding='utf-8')
    return str(img)


def test_full_scan_reports_annotated_but_light_scan_reports_pending(tmp_path):
    """两种判定的差异就是这个 bug 的根源。"""
    img = _make_annotated(tmp_path)

    assert annotation_status_for_image(img) == STATUS_ANNOTATED
    # 廉价判定只看文件在不在，于是报「扫描中」
    assert annotation_status_for_image_light(img) == STATUS_PENDING


def test_light_scan_leaves_the_row_pending(tmp_path):
    """列表重建走 light 分支，图标必然停在待扫描状态。"""
    img = _make_annotated(tmp_path)

    class FakeItem:
        def __init__(self):
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def setIcon(self, icon):
            self._icon_set = True

        def setToolTip(self, text):
            self._tip = text

    item = FakeItem()
    status = decorate_background_list_item(item, img, 0, light=True)
    assert status == STATUS_PENDING, '前提不成立：light 判定本应报待扫描'


def test_status_survives_a_view_switch(tmp_path):
    """真实 Qt：切到移除路径再切回工作路径，绿点不能变灰圈。

    修复前这里必然失败——`_show_work_view` 只做廉价判定，缓存不存在。
    """
    img = _make_annotated(tmp_path)
    work = tmp_path / 'work'
    work.mkdir()
    moved = work / 'a.png'
    moved.write_bytes((tmp_path / 'a.png').read_bytes())
    (work / 'a.json').write_bytes((tmp_path / 'a.json').read_bytes())

    script = f'''
import os
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.ui.main_window import ImageEditor
from labelforge.engine.image_loader import BG_ROLE_STATUS

ed = ImageEditor()
ed.background_images = [r"{moved.as_posix()}"]
ed.detection_boxes_dict = {{0: []}}
ed.canvas_items_dict = {{0: []}}
ed.current_background_index = 0

# 模拟后台扫描回报的真实状态（这就是扫描完成后做的事）
ed._bg_status_by_path = {{r"{moved.as_posix()}": "annotated"}}

# 切到移除路径，再切回工作路径
ed._is_delete_view = True
ed._show_delete_view()
ed._is_delete_view = False
ed._show_work_view()

item = ed.background_list.item(0)
assert item is not None, "工作路径列表没有条目"
status = item.data(BG_ROLE_STATUS)
assert status == "annotated", f"切回后状态退化成了 {{status!r}}，应为 annotated"
print("OK_STATUS_KEPT")
'''
    assert 'OK_STATUS_KEPT' in _run_real_qt(script)


def test_saving_a_row_updates_the_cached_status(tmp_path):
    """保存后再切视图，缓存必须反映新状态，不能回退。"""
    work = tmp_path / 'w'
    work.mkdir()
    img = work / 'a.png'
    img.write_bytes(b'\x89PNG\r\n\x1a\n')
    (work / 'a.json').write_text(
        '{"shapes": [{"label": "car", "points": [[0, 0], [1, 1]]}]}',
        encoding='utf-8')

    script = f'''
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.ui.main_window import ImageEditor
from labelforge.engine.image_loader import BG_ROLE_STATUS

ed = ImageEditor()
ed.background_images = [r"{img.as_posix()}"]
ed.detection_boxes_dict = {{0: []}}
ed.canvas_items_dict = {{0: []}}
ed.current_background_index = 0
ed._bg_status_by_path = {{r"{img.as_posix()}": "pending"}}
ed._show_work_view()

from PySide6.QtWidgets import QListWidgetItem
ed._refresh_background_item_status(0, r"{img.as_posix()}")
assert ed._bg_status_by_path[r"{img.as_posix()}"] == "annotated", \
    ed._bg_status_by_path

ed._show_work_view()
item = ed.background_list.item(0)
assert item.data(BG_ROLE_STATUS) == "annotated", item.data(BG_ROLE_STATUS)
print("OK_SAVE_UPDATES_CACHE")
'''
    assert 'OK_SAVE_UPDATES_CACHE' in _run_real_qt(script)
