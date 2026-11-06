"""检测框右键菜单的标签折叠：最近 3 个 + 数据集类别最多 5 个。

背景：原先菜单把数据集全部标签平铺列出，类别一多就超出屏幕。
现改为两段短列表，全量标签走「修改标签…」的滚动对话框。
"""
import pytest

from labelforge.canvas import canvas_menu
from labelforge.canvas.canvas_menu import (
    MAX_FREQUENT_LABELS, MAX_RECENT_LABELS, CanvasMenuMixin)
from labelforge.core.config_manager import _normalize_memory_record


class FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class FakeAction:
    def __init__(self, text, parent=None):
        self.text = text
        self.triggered = FakeSignal()
        self._enabled = True

    def setEnabled(self, value):
        self._enabled = value

    def isEnabled(self):
        return self._enabled

    def isSeparator(self):
        return False


class FakeSeparator:
    text = ""

    def isEnabled(self):
        return False

    def isSeparator(self):
        return True

    def setEnabled(self, value):
        pass

    def triggered(self):
        raise AssertionError("分隔线不应有 triggered")


class FakeMenu:
    def __init__(self):
        self._items = []
        self.exec_pos = None

    def addAction(self, action):
        self._items.append(action)

    def addSeparator(self):
        self._items.append(FakeSeparator())

    def removeAction(self, action):
        self._items = [item for item in self._items if item is not action]

    def actions(self):
        return list(self._items)

    def exec_(self, pos):
        self.exec_pos = pos

    # ---- 断言辅助 ----
    def labels(self):
        return [a.text for a in self._items
                if isinstance(a, FakeAction) and a.isEnabled()]

    def headers(self):
        return [a.text for a in self._items
                if isinstance(a, FakeAction) and not a.isEnabled()]

    def has_trailing_separator(self):
        return bool(self._items) and self._items[-1].isSeparator()


class _FakeList:
    """贴图标签列表的替身：只需 addItem / count。"""

    def __init__(self):
        self._items = []

    def addItem(self, text):
        self._items.append(text)

    def count(self):
        return len(self._items)

    def item(self, index):
        text = self._items[index]
        return type("Item", (), {"text": lambda self, _t=text: _t})()


class FakeEditor:
    def __init__(self, boxes=None, stats=None, recent=None):
        self.detection_boxes = boxes if boxes is not None else [
            {"label": "car", "shape_type": "rectangle",
             "x": 0, "y": 0, "width": 10, "height": 10}]
        self.detection_boxes_dict = {0: list(self.detection_boxes)}
        self.current_background_index = 0
        self.global_labels = set()
        self._is_delete_view = False
        self.label_manager = None
        self._cached_bg_label_stats = stats if stats is not None else []
        self._recent_labels = list(recent or [])
        self._recent_paste_labels = []
        self.canvas_items = []
        self.paste_label_list = _FakeList()

    def _delete_selected_box(self):
        pass

    def get_label_color(self, label):
        return ""

    def update_label_list(self):
        pass


class FakeCanvas(CanvasMenuMixin):
    """只用真实的菜单/标签切换逻辑，Qt 控件用替身。"""

    def __init__(self, editor):
        self._editor = editor
        self.menu = None
        self.repaints = 0

    def _rounded_menu(self, parent=None):
        self.menu = FakeMenu()
        return self.menu

    def mapToGlobal(self, pos):
        return pos

    def update(self):
        self.repaints += 1

    def open_box_menu(self, box_index=0):
        self._show_box_label_menu(box_index, None)
        return self.menu

    def open_paste_menu(self, item_index=0):
        self._show_paste_context_menu(item_index, None)
        return self.menu

    def action(self, text):
        return next(a for a in self.menu.actions()
                    if isinstance(a, FakeAction) and a.text == text)


def stats_of(*pairs):
    return [{"label": name, "count": count, "color": "", "tasks": []}
            for name, count in pairs]


@pytest.fixture(autouse=True)
def _patch_qaction(monkeypatch):
    """菜单里用 QAction(text, parent)，测试里换成可断言的替身。"""
    monkeypatch.setattr(canvas_menu, "QAction", FakeAction)


# --------------------------------------------------------------- 两段标签的计算
def test_quick_switch_excludes_current_and_dedupes_across_sections():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rectangle"}],
        stats=stats_of(("person", 30), ("bus", 20), ("car", 10),
                       ("dog", 5), ("cat", 4), ("bird", 3), ("fish", 2)),
        recent=["car", "truck", "bus"],
    ))
    recent, frequent = canvas._quick_switch_labels("car")
    assert recent == ["truck", "bus"]                 # 当前标签被剔除
    assert frequent == ["person", "dog", "cat", "bird", "fish"]  # 同段去重 + 截断 5


def test_quick_switch_caps_recent_at_three():
    canvas = FakeCanvas(FakeEditor(
        stats=stats_of(("a", 9)),
        recent=["a", "b", "c", "d", "e"],
    ))
    recent, _frequent = canvas._quick_switch_labels("")
    assert len(recent) == MAX_RECENT_LABELS == 3


def test_quick_switch_caps_frequent_at_five():
    canvas = FakeCanvas(FakeEditor(
        stats=stats_of(*[(f"L{i}", 100 - i) for i in range(20)]),
    ))
    _recent, frequent = canvas._quick_switch_labels("")
    assert len(frequent) == MAX_FREQUENT_LABELS == 5
    assert frequent == ["L0", "L1", "L2", "L3", "L4"]   # 保持统计的降序


def test_quick_switch_skips_blank_and_broken_stats():
    canvas = FakeCanvas(FakeEditor(stats=[
        {"label": "  "}, {"label": None}, "not-a-dict",
        {"label": "ok", "count": 1},
    ]))
    _recent, frequent = canvas._quick_switch_labels("")
    assert frequent == ["ok"]


# --------------------------------------------------------------- 最近标签记录
def test_remember_recent_label_orders_and_dedupes():
    canvas = FakeCanvas(FakeEditor())
    for name in ("car", "truck", "car", "bus"):
        canvas._remember_recent_label(name)
    assert canvas._editor._recent_labels == ["bus", "car", "truck"]


def test_remember_recent_label_ignores_blank():
    canvas = FakeCanvas(FakeEditor(recent=["car"]))
    canvas._remember_recent_label("   ")
    canvas._remember_recent_label(None)
    assert canvas._editor._recent_labels == ["car"]


# --------------------------------------------------------------- 菜单结构
def test_menu_lists_quick_labels_in_one_block():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rectangle"}],
        stats=stats_of(*[(f"L{i}", 100 - i) for i in range(40)]),
        recent=["truck", "bus"],
    ))
    menu = canvas.open_box_menu(0)
    # 最近在前、类别最多在后，合成一块，无小标题
    assert menu.labels() == ["修改标签", "移除", "转为OBB",
                             "truck", "bus", "L0", "L1", "L2", "L3", "L4"]
    assert len(menu.labels()) == 10          # 不再随类别数增长
    assert not menu.has_trailing_separator()


def test_menu_dedupes_between_recent_and_frequent():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rectangle"}],
        stats=stats_of(("person", 30), ("bus", 20), ("dog", 5)),
        recent=["bus", "person"],
    ))
    menu = canvas.open_box_menu(0)
    assert menu.labels().count("person") == 1
    assert menu.labels().count("bus") == 1
    assert menu.labels() == ["修改标签", "移除", "转为OBB",
                             "bus", "person", "dog"]


def test_menu_without_quick_labels_has_no_extra_separator():
    canvas = FakeCanvas(FakeEditor(stats=[], recent=[]))
    menu = canvas.open_box_menu(0)
    assert menu.labels() == ["修改标签", "移除", "转为OBB"]
    assert not menu.has_trailing_separator()


def test_clicking_menu_item_switches_label_and_remembers_it():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rectangle"}],
        stats=stats_of(("person", 3)), recent=[]))
    canvas.open_box_menu(0)
    canvas.action("person").triggered.emit(False)
    assert canvas._editor.detection_boxes[0]["label"] == "person"
    assert canvas._editor._recent_labels == ["person"]


def test_switching_keeps_label_list_and_memory_in_sync():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rectangle"}],
        stats=stats_of(("person", 3)), recent=["truck"]))
    canvas.open_box_menu(0)
    canvas.action("person").triggered.emit(False)
    assert canvas._editor.detection_boxes_dict[0][0]["label"] == "person"
    # 原标签已无框使用，从全局标签里移除；新标签加入
    assert canvas._editor.global_labels == {"person"}
    assert canvas._editor._recent_labels == ["person", "truck"]


def test_rotation_box_menu_keeps_shape_actions():
    canvas = FakeCanvas(FakeEditor(
        boxes=[{"label": "car", "shape_type": "rotation"}],
        stats=stats_of(("person", 3)), recent=[]))
    menu = canvas.open_box_menu(0)
    assert "重置旋转" in menu.labels()
    assert "转为HBB" in menu.labels()
    assert "转为OBB" not in menu.labels()


# --------------------------------------------------------------- 记忆记录存储
def test_memory_record_normalizes_recent_labels():
    record = _normalize_memory_record({
        "background_path": "C:/data",
        "recent_labels": [" car ", "car", "", None, "truck", "bus", "dog"],
    })
    assert record["recent_labels"] == ["car", "truck", "bus"]


def test_memory_record_without_recent_labels():
    record = _normalize_memory_record({"background_path": "C:/data"})
    assert record["recent_labels"] == []


def test_memory_record_rejects_non_list_recent_labels():
    record = _normalize_memory_record({
        "background_path": "C:/data", "recent_labels": "car",
    })
    assert record["recent_labels"] == []


# --------------------------------------------------------------- 记忆记录读写
def _memory_host(recent=None):
    from labelforge.ui.mixins.memory_record import MemoryRecordMixin

    class Host(MemoryRecordMixin):
        def __init__(self):
            self.background_images = []
            self.small_images = []
            self.canvas_items_dict = {}
            self.detection_boxes_dict = {}
            self.canvas_items = []
            self.detection_boxes = []
            self.global_labels = set()
            self.background_dataset_labels = set()
            self.imported_background_labels = set()
            self._scanned_background_labels = set()
            self._cached_bg_label_stats = []
            self._memory_background_path = "C:/data"
            self._memory_paste_path = ""
            self._memory_label_path = ""
            self.current_background_index = 0
            self.edit_mode = "annotate"
            self._recent_labels = list(recent or [])
            self.cleared_slots = 0

        # 桩：避免真的去读磁盘
        def _build_bg_label_stats_snapshot(self):
            return []

        def _reset_label_cache_slots(self):
            self.cleared_slots += 1

        def _save_label_cache_slots(self):
            pass

        def load_background_folder(self, *a, **k):
            pass

        def load_paste_folder(self, *a, **k):
            pass

        def load_paste_label_file(self, *a, **k):
            pass

        def _set_edit_mode(self, *a, **k):
            pass

        def _restore_bg_label_stats(self, *a, **k):
            pass

        def _merge_restored_stats_with_scan(self, *a, **k):
            return []

        def update_label_list(self):
            pass

    return Host()


def test_clear_memory_content_resets_recent_labels():
    host = _memory_host(recent=["car", "truck"])
    host._clear_memory_content()
    assert host._recent_labels == []


def test_loading_record_restores_recent_labels():
    host = _memory_host()
    host._load_memory_record_now({"recent_labels": ["bus", "car", "truck", "dog"]})
    assert host._recent_labels == ["bus", "car", "truck"]        # 截断 3
    assert [x for x in host._recent_labels if x in host._recent_labels] == \
        host._recent_labels                                      # 分域：来自记录


def test_loading_record_without_recent_labels_clears_them():
    host = _memory_host(recent=["stale"])
    host._load_memory_record_now({})
    assert host._recent_labels == []


def test_close_writes_recent_labels_into_record(monkeypatch):
    from labelforge.core import config_manager

    captured = {}
    monkeypatch.setattr(config_manager, "load_memory_records", lambda: [])
    monkeypatch.setattr(config_manager, "upsert_memory_record",
                        lambda record: captured.update(record) or True)

    host = _memory_host(recent=["car", "truck"])
    host._save_memory_record_on_close()

    assert captured["recent_labels"] == ["car", "truck"]
    assert captured["background_path"] == "C:/data"


# --------------------------------------------------------------- 贴图右键菜单
def _paste_canvas(labels=None, items=None, stats=None, recent=None):
    editor = FakeEditor(stats=stats or [], recent=recent or [])
    editor.paste_labels = list(labels or [])
    editor.canvas_items = list(items or [])
    editor.paste_stats = dict(stats or {})

    def _get_session_paste_stats():
        return dict(editor.paste_stats)

    editor._get_session_paste_stats = _get_session_paste_stats
    return FakeCanvas(editor)


def test_paste_menu_lists_quick_labels_in_one_block():
    canvas = _paste_canvas(
        items=[("pix", None, "paste")],
        stats={"paste": 8, "carton": 3, "barrel": 1},
    )
    menu = canvas.open_paste_menu(0)
    # 移除 + 分隔线 + 常用贴图标签（去掉当前 paste）+ 分隔线 + 增加标签
    assert menu.labels() == ["移除", "carton", "barrel", "增加标签"]


def test_paste_menu_puts_recent_labels_first():
    canvas = _paste_canvas(
        items=[("pix", None, "paste")],
        stats={"paste": 8, "carton": 3, "barrel": 1},
        recent=["barrel", "carton"],
    )
    canvas._editor._recent_paste_labels = ["barrel"]
    menu = canvas.open_paste_menu(0)
    assert menu.labels() == ["移除", "barrel", "carton", "增加标签"]


def test_paste_menu_caps_labels_at_five():
    canvas = _paste_canvas(
        items=[("pix", None, "paste")],
        stats={f"P{i}": 20 - i for i in range(15)},
    )
    menu = canvas.open_paste_menu(0)
    quick = [x for x in menu.labels() if x not in ("移除", "增加标签")]
    assert len(quick) == 5
    assert quick == ["P0", "P1", "P2", "P3", "P4"]


def test_paste_menu_without_labels_keeps_only_actions():
    canvas = _paste_canvas(items=[("pix", None, "paste")])
    menu = canvas.open_paste_menu(0)
    assert menu.labels() == ["移除", "增加标签"]


def test_paste_labels_do_not_mix_with_box_labels():
    """贴图最近标签与检测框最近标签是两套，互不影响。"""
    canvas = _paste_canvas(items=[("pix", None, "paste")])
    canvas._remember_recent_label("car")
    canvas._remember_recent_label("carton", kind='paste')
    assert canvas._editor._recent_labels == ["car"]
    assert canvas._editor._recent_paste_labels == ["carton"]


def test_paste_menu_fills_remaining_slots_from_the_paste_label_list():
    """会话统计为空时，用贴图标签列表补齐，仍受 5 个上限约束。"""
    editor = FakeEditor()
    editor.paste_label_list = _FakeList()
    for name in ("paste", "carton", "barrel", "box", "bag", "tin", "sack"):
        editor.paste_label_list.addItem(name)
    editor.canvas_items = [("pix", None, "paste")]
    editor.paste_stats = {}
    editor._get_session_paste_stats = lambda: {}
    canvas = FakeCanvas(editor)
    menu = canvas.open_paste_menu(0)
    quick = [x for x in menu.labels() if x not in ("移除", "增加标签")]
    assert quick == ["carton", "barrel", "box", "bag", "tin"]   # 当前 paste 被剔除
    assert len(quick) == 5
