"""事件处理回归测试。"""
from pathlib import Path

from labelforge.engine.event_handler import EventHandlerMixin


class FakeEditor(EventHandlerMixin):
    def __init__(self):
        self._is_delete_view = False
        self.background_images = ["0.png", "1.png", "2.png", "3.png"]
        self.current_background_index = 0
        self._nav_step = 2
        self.switched_to = None

    def switch_background_to_index(self, index):
        self.switched_to = index


def test_work_path_navigation_uses_step():
    editor = FakeEditor()

    editor.switch_background(1)

    assert editor.switched_to == 2


def test_setup_shortcuts_binds_ctrl_c_to_copy_selected_labels_cache_action():
    source = Path("labelforge/engine/event_handler.py").read_text(encoding="utf-8")

    assert "copy_selected_labels_to_active_cache_slot" in source
    assert "Ctrl+C" in source


def test_setup_shortcuts_binds_ctrl_v_to_paste_active_label_cache_slot():
    source = Path("labelforge/engine/event_handler.py").read_text(encoding="utf-8")

    assert "active_label_cache_slot" in source
    assert "paste_label_cache_slot(self.active_label_cache_slot)" in source
    assert "Ctrl+V" in source


def test_switch_background_to_index_clears_stale_box_selection():
    class FakePixmap:
        def isNull(self):
            return False

    class FakeCanvas:
        def __init__(self):
            self.selected_box = 2
            self.selected_boxes = [0, 2]
            self.reset_count = 0
            self.update_count = 0

        def reset_view(self):
            self.reset_count += 1

        def update(self):
            self.update_count += 1

    class FakeList:
        def __init__(self):
            self.row = None

        def setCurrentRow(self, row):
            self.row = row

    class Editor(EventHandlerMixin):
        def __init__(self):
            self.background_images = ["0.png", "1.png"]
            self.current_background_index = 0
            self.current_background = FakePixmap()
            self.canvas_items = []
            self.canvas_items_dict = {0: [], 1: []}
            self.detection_boxes = [{"label": "old"}]
            self.detection_boxes_dict = {0: [{"label": "old"}], 1: [{"label": "new"}]}
            self.canvas = FakeCanvas()
            self.background_list = FakeList()
            self.selected_item = object()
            self.updated = 0

        def _get_cached_pixmap(self, file_path):
            return FakePixmap()

        def _load_detection_boxes_for_index(self, new_index, file_path):
            self.detection_boxes = self.detection_boxes_dict[new_index].copy()

        def update_label_list(self):
            self.updated += 1

        def update_file_count(self):
            self.updated += 1

        def save_current_json(self):
            pass

    editor = Editor()

    editor.switch_background_to_index(1)

    assert editor.canvas.selected_box is None
    assert editor.canvas.selected_boxes == []
    assert editor.selected_item is None
    assert editor.background_list.row == 1


def test_switch_and_close_save_current_json_after_syncing_current_objects():
    source = Path('labelforge/engine/event_handler.py').read_text(encoding='utf-8')

    switch_block = source.split('def switch_background_to_index', 1)[1].split('def on_labels_checkbox_changed', 1)[0]
    close_block = source.split('def closeEvent', 1)[1]
    assert switch_block.index('self.detection_boxes_dict[self.current_background_index] = self.detection_boxes.copy()') < switch_block.index('self.save_current_json()')
    assert 'self.save_current_json()' in close_block
