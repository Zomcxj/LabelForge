"""标签缓存多选复制的 RED 回归测试。"""
from pathlib import Path

from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.mixins.cache_menu import CacheMenuMixin
from labelforge.ui.mixins.label_cache_slot import LabelCacheSlotMixin


def test_canvas_uses_selected_boxes_state_for_multi_selection():
    source = Path("labelforge/canvas/canvas.py").read_text(encoding="utf-8")

    assert "self.selected_boxes" in source


def test_get_selected_detection_boxes_returns_all_selected_boxes():
    editor = type("Editor", (), {})()
    editor.detection_boxes = [
        {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4},
        {"label": "dog", "x": 5, "y": 6, "width": 7, "height": 8},
        {"label": "bird", "x": 9, "y": 10, "width": 11, "height": 12},
    ]
    editor.canvas = type("Canvas", (), {"selected_boxes": [0, 2]})()

    selected = ImageEditor._get_selected_detection_boxes(editor)

    assert selected == [
        dict(editor.detection_boxes[0]),
        dict(editor.detection_boxes[2]),
    ]


def test_copy_selected_labels_triggers_hover_sync_when_selection_is_empty():
    detection_box = {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4}

    class Canvas:
        def __init__(self):
            self.selected_box = None
            self.selected_boxes = []

        def _check_hover(self):
            self.selected_box = 0
            self.selected_boxes = [0]

    class StatusLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    editor = type("Editor", (), {})()
    editor.detection_boxes = [detection_box]
    editor.canvas = Canvas()
    editor.status_label = StatusLabel()
    editor.label_cache_slots = [{"name": "1", "items": [], "locked": False}]
    editor.active_label_cache_slot = 0
    editor.saved = 0
    editor.rebuilt = 0
    editor._save_label_cache_slots = lambda: setattr(editor, "saved", editor.saved + 1)
    editor._rebuild_label_cache_menu = lambda: setattr(editor, "rebuilt", editor.rebuilt + 1)
    editor._get_selected_detection_boxes = lambda: ImageEditor._get_selected_detection_boxes(editor)

    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)

    assert editor.canvas.selected_box == 0
    assert editor.canvas.selected_boxes == [0]
    assert editor.label_cache_slots[0]["items"] == [dict(detection_box)]


def test_copy_selected_labels_falls_back_to_first_unlocked_slot_when_active_slot_locked():
    detection_box = {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4}

    class Canvas:
        def __init__(self):
            self.selected_box = 0
            self.selected_boxes = [0]

    class StatusLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    editor = type("Editor", (), {})()
    editor.detection_boxes = [detection_box]
    editor.canvas = Canvas()
    editor.status_label = StatusLabel()
    editor.label_cache_slots = [
        {"name": "1", "items": [], "locked": True},
        {"name": "2", "items": [], "locked": False},
        {"name": "3", "items": [], "locked": False},
    ]
    editor.active_label_cache_slot = 0
    editor.saved = 0
    editor.rebuilt = 0
    editor._save_label_cache_slots = lambda: setattr(editor, "saved", editor.saved + 1)
    editor._rebuild_label_cache_menu = lambda: setattr(editor, "rebuilt", editor.rebuilt + 1)
    editor._get_selected_detection_boxes = lambda: ImageEditor._get_selected_detection_boxes(editor)

    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)

    assert editor.active_label_cache_slot == 1
    assert editor.label_cache_slots[0]["items"] == []
    assert editor.label_cache_slots[1]["items"] == [dict(detection_box)]


def test_reset_label_cache_slots_clears_items_and_locked_state():
    i18n.set_lang("zh")
    editor = type("Editor", (), {})()
    editor.label_cache_slots = [
        {"name": "自定义1", "items": [{"label": "cat"}], "locked": True, "copied_at": "10:00:00", "shortcut": "1"},
        {"name": "自定义2", "items": [{"label": "dog"}], "locked": True, "copied_at": "10:00:01", "shortcut": "2"},
        {"name": "自定义3", "items": [], "locked": False, "copied_at": "", "shortcut": "3"},
    ]
    editor.active_label_cache_slot = 2

    ImageEditor._reset_label_cache_slots(editor)

    assert editor.active_label_cache_slot == 0
    assert [slot["name"] for slot in editor.label_cache_slots] == ["缓存槽1", "缓存槽2", "缓存槽3"]
    assert [slot["items"] for slot in editor.label_cache_slots] == [[], [], []]
    assert [slot["locked"] for slot in editor.label_cache_slots] == [False, False, False]
    assert [slot["copied_at"] for slot in editor.label_cache_slots] == ["", "", ""]


def test_copy_selected_labels_rotates_to_oldest_unlocked_slot_instead_of_sticking_to_first_slot():
    detection_box = {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4}

    class Canvas:
        def __init__(self):
            self.selected_box = 0
            self.selected_boxes = [0]

    class StatusLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    editor = type("Editor", (), {})()
    editor.detection_boxes = [detection_box]
    editor.canvas = Canvas()
    editor.status_label = StatusLabel()
    editor.label_cache_slots = [
        {"items": [{"label": "old-1"}], "locked": False, "copied_at": "10:00:00", "copy_order": 1, "shortcut": "1"},
        {"items": [{"label": "old-2"}], "locked": False, "copied_at": "10:00:01", "copy_order": 2, "shortcut": "2"},
        {"items": [{"label": "old-3"}], "locked": False, "copied_at": "10:00:02", "copy_order": 3, "shortcut": "3"},
    ]
    editor.active_label_cache_slot = 0
    editor._label_cache_copy_counter = 3
    editor.saved = 0
    editor.rebuilt = 0
    editor._save_label_cache_slots = lambda: setattr(editor, "saved", editor.saved + 1)
    editor._rebuild_label_cache_menu = lambda: setattr(editor, "rebuilt", editor.rebuilt + 1)
    editor._get_selected_detection_boxes = lambda: ImageEditor._get_selected_detection_boxes(editor)

    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)

    assert editor.active_label_cache_slot == 0
    assert editor.label_cache_slots[0]["items"] == [dict(detection_box)]

    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)

    assert editor.active_label_cache_slot == 1
    assert editor.label_cache_slots[1]["items"] == [dict(detection_box)]


def test_cache_slot_row_click_only_switches_active_slot():
    editor = type("Editor", (), {})()
    editor.active = None
    editor.pasted = 0
    editor.cache_menu = None
    editor.set_active_label_cache_slot = lambda idx: setattr(editor, "active", idx)
    editor.paste_label_cache_slot = lambda idx: setattr(editor, "pasted", editor.pasted + 1)

    CacheMenuMixin._handle_cache_slot_row_click(editor, 2)

    assert editor.active == 2
    assert editor.pasted == 0


def _make_cache_editor():
    """最小 editor：够走「复制 → 粘贴 → 拖动」这条链。"""
    editor = type("Editor", (LabelCacheSlotMixin,), {})()
    editor.detection_boxes = [{"label": "cat", "x": 10, "y": 10,
                               "width": 20, "height": 20}]
    editor.canvas = type("Canvas", (), {
        "selected_box": 0, "selected_boxes": [0], "update": lambda s: None})()

    class _Status:
        text = ""

        def setText(self, t):
            self.text = t

    editor.status_label = _Status()
    editor.label_cache_slots = [{"name": "1", "items": [], "locked": False,
                                 "copy_order": 0, "copied_at": ""}]
    editor.active_label_cache_slot = 0
    editor._label_cache_copy_counter = 0
    editor.saved = 0
    editor._save_label_cache_slots = lambda: setattr(
        editor, "saved", editor.saved + 1)
    editor._rebuild_label_cache_menu = lambda: None
    editor._is_delete_view = False
    editor.current_background = object()
    editor.current_background_index = 0
    editor.detection_boxes_dict = {0: []}
    editor.canvas_items = []
    editor._last_paste_slot = None
    editor._last_paste_start = -1
    editor._last_paste_count = 0
    editor.save_undo_state = lambda: None
    editor.update_label_list = lambda: None
    editor._offset_overlapping_paste_group = lambda group: group
    return editor



def test_sync_writes_back_only_the_box_that_came_from_the_slot():
    """回写只该动「从这个槽粘出来的框」，不该牵连无关的框。

    用户报的场景：槽里只有一份标签，拖动一个**没被复制过**的框之后，
    缓存槽却被那个框的内容覆盖了。原因是回写按**下标区间**取框，而删掉/
    新增过框之后下标会整体平移，区间就落到别的框上了。
    """
    editor = _make_cache_editor()
    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)
    template = [dict(b) for b in editor.label_cache_slots[0]["items"]]

    # 再手画两个与缓存无关的框
    editor.detection_boxes.append({"label": "无关A", "x": 300, "y": 300,
                                   "width": 10, "height": 10})
    editor.detection_boxes.append({"label": "无关B", "x": 400, "y": 400,
                                   "width": 10, "height": 10})

    # 拖动「无关B」——它从没被复制进任何槽
    editor.detection_boxes[-1]["x"] = 999
    ImageEditor._sync_pasted_boxes_to_cache(editor)

    assert editor.label_cache_slots[0]["items"] == template, (
        "拖动一个没被复制过的框，却把缓存槽改写了：\n"
        f"  期望 {template}\n  实际 {editor.label_cache_slots[0]['items']}")


def test_sync_keeps_the_slot_intact_when_an_earlier_box_is_deleted():
    """删掉前面的框导致下标平移后，回写不能落到别的框上。"""
    editor = _make_cache_editor()
    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)
    template = [dict(b) for b in editor.label_cache_slots[0]["items"]]

    ImageEditor.paste_label_cache_slot(editor, 0)
    # 删掉最前面那个（原本被复制进槽的）框，后面的下标整体 -1
    del editor.detection_boxes[0]

    ImageEditor._sync_pasted_boxes_to_cache(editor)

    assert editor.label_cache_slots[0]["items"] == template, (
        "删框导致下标平移后，回写落到了别的框上：\n"
        f"  期望 {template}\n  实际 {editor.label_cache_slots[0]['items']}")


def test_sync_only_fires_for_the_box_that_was_pasted():
    """回写必须只认「这个被拖的框是不是从槽里粘出来的」。

    `_sync_detection_box_to_dict(box_index)` 是带着**被拖框的下标**进来的，
    但 `_sync_pasted_boxes_to_cache()` 把它丢了、只按当初记下的粘贴区间回写。
    于是拖动任何一个别的框，也会把「粘贴副本当初被挪到哪」写回槽里——
    用户看到的正是「拖动一个没被复制的框，缓存却被改写了」。
    """
    editor = _make_cache_editor()
    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)
    template = [dict(b) for b in editor.label_cache_slots[0]["items"]]

    ImageEditor.paste_label_cache_slot(editor, 0)
    pasted_index = editor._last_paste_start
    assert pasted_index >= 0, "粘贴没记下区间，测试前提不成立"

    # 再放一个与缓存完全无关的框，并拖动它
    editor.detection_boxes.append({"label": "无关", "x": 500, "y": 500,
                                   "width": 30, "height": 30})
    unrelated_index = len(editor.detection_boxes) - 1
    editor.detection_boxes[unrelated_index]["x"] = 777

    # 画布拖动无关框时，传进来的就是它的下标
    ImageEditor._sync_pasted_boxes_to_cache(editor,
                                            box_index=unrelated_index)

    assert editor.label_cache_slots[0]["items"] == template, (
        "拖动无关的框也回写了缓存槽：\n"
        f"  期望 {template}\n  实际 {editor.label_cache_slots[0]['items']}")


def test_sync_still_writes_back_when_the_pasted_box_is_moved():
    """反向要求：拖的**正是**粘贴出来的那个框时，回写要照常生效。"""
    editor = _make_cache_editor()
    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)
    ImageEditor.paste_label_cache_slot(editor, 0)
    pasted_index = editor._last_paste_start

    editor.detection_boxes[pasted_index]["x"] = 321
    ImageEditor._sync_pasted_boxes_to_cache(editor, box_index=pasted_index)

    items = editor.label_cache_slots[0]["items"]
    assert len(items) == 1 and items[0]["x"] == 321, (
        f"拖动粘贴出来的框没能回写槽：{items}")


def test_sync_without_a_box_index_still_writes_back():
    """不带下标调用时保持旧行为（回写整个粘贴区间）。

    画布是会带下标的主路径；不带下标留给「整组调整完统一回写」这类调用，
    所以默认不能变成「什么都不做」。
    """
    editor = _make_cache_editor()
    ImageEditor.copy_selected_labels_to_active_cache_slot(editor)
    ImageEditor.paste_label_cache_slot(editor, 0)
    start = editor._last_paste_start
    editor.detection_boxes[start]["x"] = 55

    ImageEditor._sync_pasted_boxes_to_cache(editor)

    items = editor.label_cache_slots[0]["items"]
    assert len(items) == 1 and items[0]["x"] == 55, (
        f"不带下标时应当照旧回写：{items}")
