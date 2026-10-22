"""B5 回归：一键贴图的缓存判据是「下标在不在字典里」，不是 len > 0。

审计 B5：paste_engine.batch_paste_images 的循环里
`if i in self.detection_boxes_dict and len(self.detection_boxes_dict[i]) > 0`
把空列表（0 框，也是**有效缓存**——用户刚清空过这张图）当成「没加载过」，
从磁盘 sidecar 重读旧框并写回缓存，已删的框被复活。

image_loader/mixin._load_detection_boxes_for_index 的注释早已写明同一判据
（「空列表现在是有意义的缓存值」），本测试把一键贴图对齐到同一语义。

conftest 在 import 时已把 PySide6 换成桩（QPixmap.isNull 恒 False，
QApplication.processEvents 为空操作），因此纯 pytest 即可驱动真实循环；
只把真正干活的 random_paste_images 打桩为记录器，其余全走真代码。
"""
import pytest

from labelforge.engine.paste_engine import PasteEngineMixin


class _FakeDialog:
    def __init__(self):
        self.values = []
        self.labels = []

    def show(self):
        pass

    def setValue(self, value):
        self.values.append(value)

    def setLabelText(self, text):
        self.labels.append(text)

    def wasCanceled(self):
        return False


class _FakeFactory:
    @staticmethod
    def create_progress_dialog(parent, title, text, count):
        return _FakeDialog()


class _FakeCanvas:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1


def _make_editor(background_images, cache, sidecar_boxes):
    """最小可用的一键贴图环境。

    load_detection_boxes 模拟「磁盘 sidecar 里有框」：谁被调用谁被记录，
    返回 sidecar_boxes 的新副本（与真实读盘语义一致——每次返回新列表）。
    """
    loads = []
    pastes = []

    editor = PasteEngineMixin.__new__(PasteEngineMixin)
    editor._busy = False
    editor.small_images = ["stub_small.png"]
    editor.background_images = list(background_images)
    editor.current_background_index = 0
    editor.current_background = None
    editor.detection_boxes = []
    editor.canvas_items = []
    editor.detection_boxes_dict = cache
    editor.canvas_items_dict = {}
    editor.canvas = _FakeCanvas()

    def load_detection_boxes(file_path):
        loads.append(file_path)
        return [dict(b) for b in sidecar_boxes]

    def random_paste_images(background=None, detection_boxes=None, seed=None):
        pastes.append([dict(b) for b in (detection_boxes or [])])

    editor.load_detection_boxes = load_detection_boxes
    editor.random_paste_images = random_paste_images
    return editor, loads, pastes


def test_empty_cache_entry_is_valid_and_disk_is_not_reread(monkeypatch):
    """image 0 的缓存是 []（用户清空过）：不得从磁盘重读旧框，
    缓存不得被旧内容覆盖，贴图收到的也必须是 0 框。"""
    monkeypatch.setattr("labelforge.ui.dialogs.ProgressDialogFactory", _FakeFactory)

    disk_box = {"label": "ghost", "x": 1, "y": 1, "width": 5, "height": 5}
    images = ["img_0.png", "img_1.png"]
    editor, loads, pastes = _make_editor(
        images, cache={0: []}, sidecar_boxes=[disk_box])

    editor.batch_paste_images()

    # image 0：空缓存是有效值——磁盘一次都不许读
    assert loads == [images[1]], loads
    # image 0 的缓存仍是空列表，不是被磁盘旧框覆盖
    assert editor.detection_boxes_dict[0] == []
    # image 0 的贴图收到 0 框（复活即 bug）
    assert pastes[0] == []
    # image 1：无缓存条目，正常读盘并写回缓存
    assert editor.detection_boxes_dict[1] == [disk_box]
    assert pastes[1] == [disk_box]
