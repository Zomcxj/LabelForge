"""Ctrl+Shift+Z 直接删除标签（无确认框）回归测试。"""
import inspect
from pathlib import Path

from labelforge.core import config as config_module
from labelforge.engine.label_manager import LabelManager

ROOT = Path(__file__).resolve().parents[1]
LABEL_MANAGER_SRC = ROOT / "labelforge" / "engine" / "label_manager.py"
EVENT_HANDLER_SRC = ROOT / "labelforge" / "engine" / "event_handler.py"
SETTINGS_SRC = ROOT / "labelforge" / "ui" / "settings_dialog.py"


class FakeSignal:
    def emit(self):
        pass


class FakeItem:
    def __init__(self, text, box_index=None):
        self._text = text
        self._box_index = box_index

    def text(self):
        return self._text

    def data(self, role):
        return self._box_index


class FakeList:
    def __init__(self, selected):
        self._selected = selected

    def selectedItems(self):
        return self._selected
    
    def setCurrentRow(self, row):
        pass


class FakeCanvas:
    """画布替身：选中项用**真下标**（旧的那版塞的是字符串 'x'，
    没法拿来索引 detection_boxes，一测就露馅）。"""

    def __init__(self, selected_box=None, selected_boxes=None):
        self.selected_box = selected_box
        self.selected_boxes = list(selected_boxes or [])
        self.updated = 0

    def update(self):
        self.updated += 1


class FakeEditor:
    def __init__(self, selected, boxes, mode='stats',
                 canvas_selected_box=None, canvas_selected_boxes=None):
        self.label_list = FakeList(selected)
        self.background_list = FakeList([])
        self._bg_label_list_mode = mode
        self.detection_boxes = list(boxes)
        self.detection_boxes_dict = {0: [dict(b) for b in boxes]}
        self.global_labels = {b['label'] for b in boxes}
        self.background_images = ['a.png']
        self.current_background_index = 0
        self.canvas = FakeCanvas(canvas_selected_box, canvas_selected_boxes)
        self.saved = []
    
    def update_label_list(self):
        pass
    
    def switch_background_to_index(self, idx):
        pass
    
    def _find_bg_list_row_for_index(self, idx):
        return 0


def _manager(editor):
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()
    manager._save_detection_json_for_index = lambda index: editor.saved.append(index)
    return manager


# ---------- 配置与注册 ----------

def test_shortcut_is_configured_as_ctrl_shift_z():
    assert config_module.SHORTCUT_CONFIG['delete_label'] == 'Ctrl+Shift+Z'


def test_shortcut_sits_right_below_restore_image_in_settings():
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    restore = source.index("'restore_image': tr(\"恢复图片\"),")
    delete = source.index("'delete_label': tr(\"删除标签\"),")
    assert restore < delete
    # 中间不能插入其它快捷键项
    between = source[restore:delete]
    assert between.count("': tr(") == 1


def test_shortcut_is_registered_in_event_handler():
    source = EVENT_HANDLER_SRC.read_text(encoding="utf-8")
    assert "'delete_label': self.delete_selected_label," in source


def test_shortcut_is_not_handled_by_the_confirming_delete_label():
    source = EVENT_HANDLER_SRC.read_text(encoding="utf-8")
    handlers = source[source.index("action_handlers = {"):source.index("self._shortcuts = []")]
    assert "'delete_label': self.delete_selected_label," in handlers
    assert "'delete_label': self.delete_label," not in handlers


# ---------- 无确认删除 ----------

def test_delete_selected_label_skips_the_confirmation_dialog(monkeypatch):
    """核心：快捷键路径不能弹任何对话框。"""
    import labelforge.engine.label_manager as lm
    calls = []
    monkeypatch.setattr(lm.dialog_helpers, "question",
                        lambda *a, **k: calls.append(a) or lm.dialog_helpers.QMessageBox.No)

    editor = FakeEditor([FakeItem('cat', box_index=0)], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}], mode='all')
    _manager(editor).delete_selected_label()

    assert calls == []
    assert editor.detection_boxes == []
    assert editor.global_labels == set()


def test_delete_label_still_confirms(monkeypatch):
    """右键菜单路径必须保留确认框。"""
    import labelforge.engine.label_manager as lm
    calls = []
    monkeypatch.setattr(lm.dialog_helpers, "question",
                        lambda *a, **k: calls.append(a) or lm.dialog_helpers.QMessageBox.No)

    editor = FakeEditor([FakeItem('cat')], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    _manager(editor).delete_label()

    assert len(calls) == 1
    assert editor.detection_boxes  # 回答 No，未删除


def test_confirming_no_leaves_everything_untouched(monkeypatch):
    import labelforge.engine.label_manager as lm
    monkeypatch.setattr(lm.dialog_helpers, "question",
                        lambda *a, **k: lm.dialog_helpers.QMessageBox.No)

    editor = FakeEditor([FakeItem('cat')], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    _manager(editor).delete_label()

    assert editor.detection_boxes_dict == {0: [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}]}
    assert editor.global_labels == {'cat'}
    assert editor.saved == []


def test_stats_mode_deletes_every_box_with_that_label(monkeypatch):
    """Stats 模式快捷键删除当前图片该标签的所有框（不跨图片）。"""
    import labelforge.engine.label_manager as lm
    monkeypatch.setattr(lm.dialog_helpers, "question", lambda *a, **k: lm.dialog_helpers.QMessageBox.Yes)

    boxes = [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
             {'label': 'cat', 'x': 5, 'y': 5, 'width': 1, 'height': 1},
             {'label': 'dog', 'x': 9, 'y': 9, 'width': 1, 'height': 1}]
    editor = FakeEditor([FakeItem('cat')], boxes)
    _manager(editor).delete_selected_label()

    # 当前图片的 cat 框全部删除，dog 保留
    assert [b['label'] for b in editor.detection_boxes] == ['dog']
    assert editor.global_labels == {'dog'}


def test_all_mode_deletes_only_the_selected_box():
    boxes = [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
             {'label': 'cat', 'x': 5, 'y': 5, 'width': 1, 'height': 1}]
    editor = FakeEditor([FakeItem('cat', box_index=1)], boxes, mode='all')
    _manager(editor).delete_selected_label()

    assert len(editor.detection_boxes) == 1
    # 仍有一个 cat 框在用，全局标签不能被移除
    assert editor.global_labels == {'cat'}


def test_no_selection_now_deletes_the_whole_image(monkeypatch):
    """① 空选区**不再是无操作**：改成删当前图全部标签框。

    这条原先是 `test_no_selection_is_a_noop`，钉的是「没选中就什么都不做」。
    用户明确要求「支持焦点在画布上就能删除」，所以旧契约被推翻、改成新契约。
    仍然要保证不弹确认框。
    """
    import labelforge.engine.label_manager as lm
    calls = []
    monkeypatch.setattr(lm.dialog_helpers, "question",
                        lambda *a, **k: calls.append(a) or lm.dialog_helpers.QMessageBox.Yes)

    editor = FakeEditor([], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    _manager(editor).delete_selected_label()

    assert calls == [], "快捷键删除不该弹确认框"
    assert editor.detection_boxes == [], (
        "焦点在画布上却没删掉标签框（用户报的就是这个）")


def test_deleted_label_is_written_back_to_json():
    editor = FakeEditor([FakeItem('cat')], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    _manager(editor).delete_selected_label()

    assert editor.saved == [0]


def test_canvas_selection_is_cleared_after_delete():
    editor = FakeEditor([FakeItem('cat')], [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1}])
    _manager(editor).delete_selected_label()

    assert editor.canvas.selected_box is None
    assert editor.canvas.selected_boxes == []


# ---------- 主窗口转发 ----------

def test_main_window_forwards_to_the_non_confirming_variant():
    from labelforge.ui.main_window import ImageEditor
    source = inspect.getsource(ImageEditor.delete_selected_label)
    assert "label_manager.delete_selected_label()" in source
    # 不能误转发到带确认的 delete_label
    assert "label_manager.delete_label()" not in source


def test_label_manager_exposes_both_variants():
    assert hasattr(LabelManager, "delete_selected_label")
    assert hasattr(LabelManager, "delete_label")


def test_shortcut_and_menu_have_different_scope():
    """快捷键和右键菜单的删除范围不同。
    
    - 快捷键 delete_selected_label：stats 模式只删当前图片
    - 右键菜单 delete_label：stats 模式跨图片删除（带确认）
    """
    source = LABEL_MANAGER_SRC.read_text(encoding="utf-8")
    assert "def delete_selected_label(self)" in source
    assert "def delete_label(self)" in source
    # 快捷键不调用 _delete_label_from_item（独立实现）
    delete_selected_src = source.split("def delete_selected_label(self)")[1].split("\n    def ")[0]
    assert "_delete_label_from_item" not in delete_selected_src


# ---------- ① 焦点在画布上也能删（不再要求先选中侧栏项）----------

def test_canvas_focus_deletes_all_boxes_when_nothing_is_selected():
    """① 画布上有焦点、侧栏一个都没选中时，Ctrl+Shift+Z 删掉**当前图全部**标签框。

    用户要求「要选中一个才能删除」改成「支持焦点在画布上就能删除」——
    原判据是 `label_list.selectedItems()` 为空就直接 return，于是焦点明明在
    画布上、快捷键也照常送达（QShortcut 与焦点无关），却什么都不发生。
    """
    editor = FakeEditor([], [
        {'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
        {'label': 'dog', 'x': 5, 'y': 5, 'width': 1, 'height': 1},
    ])
    _manager(editor).delete_selected_label()

    assert editor.detection_boxes == [], "画布焦点下没能删掉全部标签框"


def test_canvas_focus_writeback_and_selection_reset():
    """① 画布焦点删除后：缓存同步、清选中态、落盘、发信号——与选中项路径一致。"""
    editor = FakeEditor([], [
        {'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
    ])
    _manager(editor).delete_selected_label()

    assert editor.detection_boxes_dict[0] == []
    assert editor.canvas.selected_box is None
    assert editor.canvas.selected_boxes == []
    assert editor.saved == [0], "画布焦点删除也必须落盘"
    assert editor.global_labels == set(), "没有任何框在用，全局标签应清掉"


def test_canvas_selection_takes_precedence_over_sidebar_selection():
    """① 画布上选中了具体的框时，只删**选中的那些**，不是删全部。

    这是「焦点在画布上就能删」与「先选中一个再删」的合流：画布选中了就用
    画布的选择（对齐标注员的直觉——选哪删哪），画布没选中才删全部。
    """
    editor = FakeEditor(
        [FakeItem('cat')],   # 侧栏选中的是 cat
        [{'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
         {'label': 'dog', 'x': 5, 'y': 5, 'width': 1, 'height': 1}],
        canvas_selected_box=1,      # 画布上选中的是第 1 个（dog）
        canvas_selected_boxes=[1],
    )
    _manager(editor).delete_selected_label()

    remaining = [b['label'] for b in editor.detection_boxes]
    assert remaining == ['cat'], f"应当只删画布选中的那个 dog：{remaining}"


def test_multi_canvas_selection_deletes_every_selected_box():
    """① 画布上多选（框选）时，选中的那几个全删掉，其余保留。"""
    editor = FakeEditor(
        [],
        [{'label': 'a', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
         {'label': 'b', 'x': 2, 'y': 0, 'width': 1, 'height': 1},
         {'label': 'c', 'x': 4, 'y': 0, 'width': 1, 'height': 1}],
        canvas_selected_boxes=[0, 2],
    )
    _manager(editor).delete_selected_label()

    remaining = [b['label'] for b in editor.detection_boxes]
    assert remaining == ['b'], f"应当只留下未选中的 b：{remaining}"


def test_out_of_range_canvas_selection_does_not_crash():
    """① 画布选中项越界（列表刚被换过）时不能崩，也不能误删。

    切图/重新推理会重建 `detection_boxes`，而 `selected_box` 可能还停在旧
    下标上。这种脏状态下按快捷键必须安静地什么都不做。
    """
    editor = FakeEditor([], [
        {'label': 'cat', 'x': 0, 'y': 0, 'width': 1, 'height': 1},
    ], canvas_selected_box=99, canvas_selected_boxes=[99])
    _manager(editor).delete_selected_label()

    assert len(editor.detection_boxes) == 1, "越界的选中项不该删掉任何东西"
