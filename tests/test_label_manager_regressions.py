"""标签管理回归测试。"""
from labelforge.engine.label_manager import LabelManager


class FakeSignal:
    def emit(self):
        pass


class FakeList:
    def __init__(self, selected=None):
        self.items = []
        self.selected = selected or []

    def clear(self):
        self.items.clear()

    def addItem(self, item):
        self.items.append(item)

    def selectedItems(self):
        return self.selected


class FakeItem:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class FakeEditor:
    def __init__(self):
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_color_map = {}
        self.label_colors = []
        self._cached_bg_label_stats = []
        self._bg_label_list_mode = 'stats'
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        self.canvas_items = []
        self.canvas_items_dict = {}
        self.paste_label_list = FakeList()
        self.current_background = object()
        self.current_background_index = 0
        self.label_list = FakeList()
        self.background_images = []
        self.canvas = type("Canvas", (), {"selected_box": None})()
        self.saved = []

    def save_json(self, *args, **kwargs):
        self.saved.append((args, kwargs))
    
    def update_label_list(self):
        pass


def test_add_label_ignores_qaction_checked_bool():
    editor = FakeEditor()
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()

    manager.add_label(False)

    assert editor.global_labels == set()


def test_update_label_list_ignores_non_string_labels():
    editor = FakeEditor()
    editor.global_labels = {False, "cat"}
    editor.detection_boxes = [{"label": True}, {"label": "dog"}]

    manager = LabelManager(editor)

    manager.update_label_list()

    assert len(editor.label_list.items) == 2


def test_update_label_list_flags_point_without_matching_box_group():
    editor = FakeEditor()
    editor._bg_label_list_mode = 'all'
    editor.detection_boxes = [
        {"label": "person", "shape_type": "rectangle",
         "x": 0, "y": 0, "width": 10, "height": 10, "group_id": 1},
        {"label": "nose", "shape_type": "point",
         "points": [[5, 5]], "x": 5, "y": 5, "width": 1, "height": 1,
         "group_id": 1},
        {"label": "eye", "shape_type": "point",
         "points": [[8, 8]], "x": 8, "y": 8, "width": 1, "height": 1,
         "group_id": 2},
    ]

    manager = LabelManager(editor)
    import labelforge.engine.label_manager as lm
    lm.QListWidgetItem = FakeItem
    try:
        manager.update_label_list()
    finally:
        import importlib
        importlib.reload(lm)

    texts = [item.text() for item in editor.label_list.items]
    assert any(t.startswith("nose [1]") for t in texts)
    orphan = next(t for t in texts if t.startswith("eye [2]"))
    assert "无同组框" in orphan
    person = next(t for t in texts if t.startswith("person"))
    assert "⚠" not in person


def test_point_inside_group_box_has_no_warning():
    """关键点在同组框内部时不警告。"""
    editor = FakeEditor()
    editor._bg_label_list_mode = 'all'
    editor.detection_boxes = [
        {"label": "person", "shape_type": "rectangle",
         "x": 0, "y": 0, "width": 10, "height": 10, "group_id": 1},
        {"label": "nose", "shape_type": "point",
         "points": [[5, 5]], "x": 5, "y": 5, "width": 1, "height": 1,
         "group_id": 1},
    ]
    manager = LabelManager(editor)
    import labelforge.engine.label_manager as lm
    lm.QListWidgetItem = FakeItem
    try:
        manager.update_label_list()
    finally:
        import importlib
        importlib.reload(lm)

    nose = next(t for t in (i.text() for i in editor.label_list.items)
                if t.startswith("nose"))
    assert "⚠" not in nose


def test_point_outside_group_box_warns_not_inside():
    """关键点有同组框但落在框外时提示'不在框内'。"""
    editor = FakeEditor()
    editor._bg_label_list_mode = 'all'
    editor.detection_boxes = [
        {"label": "person", "shape_type": "rectangle",
         "x": 0, "y": 0, "width": 10, "height": 10, "group_id": 1},
        {"label": "nose", "shape_type": "point",
         "points": [[50, 50]], "x": 50, "y": 50, "width": 1, "height": 1,
         "group_id": 1},
    ]
    manager = LabelManager(editor)
    import labelforge.engine.label_manager as lm
    lm.QListWidgetItem = FakeItem
    try:
        manager.update_label_list()
    finally:
        import importlib
        importlib.reload(lm)

    nose = next(t for t in (i.text() for i in editor.label_list.items)
                if t.startswith("nose"))
    assert "不在框内" in nose


def test_delete_label_does_not_pass_detection_boxes_as_canvas_items(monkeypatch):
    editor = FakeEditor()
    editor._bg_label_list_mode = "stats"
    editor.label_list = FakeList([FakeItem("cat (1)")])
    editor.global_labels = {"cat"}
    editor.background_images = ["cat.png"]
    editor.detection_boxes = [
        {"label": "cat", "x": 1, "y": 2, "width": 3, "height": 4},
        {"label": "dog", "x": 5, "y": 6, "width": 7, "height": 8},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}

    from labelforge.engine import label_manager
    monkeypatch.setattr(
        label_manager.dialog_helpers,
        "question",
        lambda *a, **kw: label_manager.dialog_helpers.QMessageBox.Yes,
    )
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.delete_label()

    assert editor.saved[0][1]["canvas_items"] == []


def test_rename_to_new_label_keeps_old_color():
    editor = FakeEditor()
    editor.global_labels = {"cat"}
    editor.background_dataset_labels = {"cat"}
    editor.label_color_map = {"cat": "#ff0000"}
    editor._cached_bg_label_stats = [
        {"label": "cat", "count": 3, "color": "#ff0000"},
        {"label": "dog", "count": 1, "color": "#00ff00"},
    ]
    editor.detection_boxes = [{"label": "cat"}]
    editor.detection_boxes_dict = {0: [{"label": "cat"}]}
    editor.background_images = []
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "kitty", rewrite_disk=False) is True
    assert editor.label_color_map.get("kitty") == "#ff0000"
    assert "cat" not in editor.label_color_map
    stats = {s["label"]: s for s in editor._cached_bg_label_stats}
    assert "kitty" in stats
    assert stats["kitty"]["color"] == "#ff0000"
    assert stats["kitty"]["count"] == 3


def test_rename_seeds_stats_cache_when_empty():
    """Empty stats cache must be seeded from in-memory boxes on rename."""
    editor = FakeEditor()
    editor.global_labels = {"cat", "dog"}
    editor.background_dataset_labels = {"cat", "dog"}
    editor.label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
    editor._cached_bg_label_stats = []  # never opened stats yet
    # current image index 0; boxes must match dict[0] (rename syncs from detection_boxes).
    editor.current_background_index = 0
    editor.detection_boxes = [{"label": "cat"}, {"label": "cat"}]
    editor.detection_boxes_dict = {
        0: list(editor.detection_boxes),
        1: [{"label": "dog"}, {"label": "cat"}],
    }
    editor.background_images = []
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "kitty", rewrite_disk=False) is True
    stats = {s["label"]: s for s in editor._cached_bg_label_stats}
    assert stats["kitty"]["count"] == 3
    assert stats["dog"]["count"] == 1
    assert stats["kitty"]["color"] == "#ff0000"
    assert "cat" not in stats


def test_rename_to_new_label_moves_color_for_canvas_lookup():
    """After rename, get_label_color(new) must return old color (box redraw)."""
    editor = FakeEditor()
    editor.global_labels = {"cat"}
    editor.background_dataset_labels = {"cat"}
    editor.label_color_map = {"cat": "#ff0000"}
    editor.label_colors = ["#ff0000", "#111111", "#222222"]
    editor._cached_bg_label_stats = [{"label": "cat", "count": 1, "color": "#ff0000"}]
    editor.detection_boxes = [{"label": "cat"}]
    editor.detection_boxes_dict = {0: [{"label": "cat"}]}
    editor.background_images = []

    def get_label_color(label):
        if label in editor.label_color_map:
            return editor.label_color_map[label]
        return "#999999"

    editor.get_label_color = get_label_color
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "kitty", rewrite_disk=False) is True
    assert editor.get_label_color("kitty") == "#ff0000"
    assert editor.detection_boxes[0]["label"] == "kitty"


def test_rename_to_existing_label_uses_target_color():
    editor = FakeEditor()
    editor.global_labels = {"cat", "dog"}
    editor.background_dataset_labels = {"cat", "dog"}
    editor.label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
    editor._cached_bg_label_stats = [
        {"label": "cat", "count": 2, "color": "#ff0000"},
        {"label": "dog", "count": 3, "color": "#00ff00"},
    ]
    editor.detection_boxes = [{"label": "cat"}, {"label": "dog"}]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.background_images = []
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "dog", rewrite_disk=False) is True
    assert editor.label_color_map.get("dog") == "#00ff00"
    assert "cat" not in editor.label_color_map
    stats = {s["label"]: s for s in editor._cached_bg_label_stats}
    assert "cat" not in stats
    assert stats["dog"]["count"] == 5
    assert stats["dog"]["color"] == "#00ff00"


def test_rename_to_existing_keeps_target_color_from_stats_when_not_in_map():
    """Target only has color in stats cache — must not reallocate / take old color."""
    editor = FakeEditor()
    editor.global_labels = {"cat", "dog"}
    editor.background_dataset_labels = {"cat", "dog"}
    # dog color only lives in stats, not color_map (common after scan).
    editor.label_color_map = {"cat": "#ff0000"}
    editor._cached_bg_label_stats = [
        {"label": "cat", "count": 2, "color": "#ff0000"},
        {"label": "dog", "count": 3, "color": "#00ff00"},
    ]
    editor.detection_boxes = [{"label": "cat"}, {"label": "dog"}]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.background_images = []
    editor.label_colors = ["#111111", "#222222", "#333333", "#00ff00", "#ff0000"]

    def _fake_get_label_color(label):
        # Simulate allocation that would differ from stats dog color.
        return {"cat": "#ff0000", "dog": "#abcdef"}.get(label, "#abcdef")

    editor.get_label_color = _fake_get_label_color
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "dog", rewrite_disk=False) is True
    assert editor.label_color_map.get("dog") == "#00ff00"
    assert "cat" not in editor.label_color_map
    stats = {s["label"]: s for s in editor._cached_bg_label_stats}
    assert stats["dog"]["color"] == "#00ff00"
    assert stats["dog"]["count"] == 5


def test_modify_label_all_mode_to_existing_keeps_target_color(monkeypatch):
    class RoleItem(FakeItem):
        def __init__(self, text, box_index):
            super().__init__(text)
            self._data = {0x0100: box_index}

        def data(self, role):
            return self._data.get(role)

    editor = FakeEditor()
    editor._bg_label_list_mode = "all"
    editor.label_list = FakeList([RoleItem("cat", 0)])
    editor.global_labels = {"cat", "dog"}
    editor.background_images = ["img.png"]
    editor.detection_boxes = [
        {"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1},
        {"label": "dog", "x": 2, "y": 2, "width": 1, "height": 1},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
    editor._cached_bg_label_stats = [
        {"label": "cat", "count": 1, "color": "#ff0000"},
        {"label": "dog", "count": 1, "color": "#00ff00"},
    ]

    monkeypatch.setattr(
        "labelforge.ui.dialogs.LabelSelectionDialog.select_label",
        staticmethod(lambda *a, **kw: "dog"),
    )
    monkeypatch.setattr(
        "labelforge.core.config_manager.save_all",
        lambda **kwargs: None,
    )
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.modify_label()

    assert editor.detection_boxes[0]["label"] == "dog"
    assert editor.label_color_map.get("dog") == "#00ff00"
    stats = {s["label"]: s for s in editor._cached_bg_label_stats}
    assert "cat" not in stats
    assert stats["dog"]["count"] == 2
    assert stats["dog"]["color"] == "#00ff00"


def test_modify_label_all_mode_renames_only_selected_box(monkeypatch):
    class RoleItem(FakeItem):
        def __init__(self, text, box_index):
            super().__init__(text)
            self._data = {0x0100: box_index}

        def data(self, role):
            return self._data.get(role)

    editor = FakeEditor()
    editor._bg_label_list_mode = "all"
    editor.label_list = FakeList([RoleItem("Car", 1)])
    editor.global_labels = {"Car", "dog"}
    editor.background_images = ["img.png"]
    editor.detection_boxes = [
        {"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1},
        {"label": "Car", "x": 2, "y": 2, "width": 1, "height": 1},
        {"label": "dog", "x": 3, "y": 3, "width": 1, "height": 1},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.label_color_map = {"Car": "#111", "dog": "#222"}

    monkeypatch.setattr(
        "labelforge.ui.dialogs.LabelSelectionDialog.select_label",
        staticmethod(lambda *a, **kw: "Truck"),
    )
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.modify_label()

    assert editor.detection_boxes == [
        {"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1},
        {"label": "Truck", "x": 2, "y": 2, "width": 1, "height": 1},
        {"label": "dog", "x": 3, "y": 3, "width": 1, "height": 1},
    ]
    assert "Truck" in editor.global_labels
    assert "Car" in editor.global_labels


def test_modify_label_all_mode_changes_group_without_label_change(monkeypatch):
    """只改 group（标签不变）也要写回内存/磁盘。"""
    class RoleItem(FakeItem):
        def __init__(self, text, box_index):
            super().__init__(text)
            self._data = {0x0100: box_index}

        def data(self, role):
            return self._data.get(role)

    editor = FakeEditor()
    editor._bg_label_list_mode = "all"
    editor.label_list = FakeList([RoleItem("nose [1]", 0)])
    editor.global_labels = {"nose"}
    editor.background_images = ["img.png"]
    editor.detection_boxes = [
        {"label": "nose", "shape_type": "point", "points": [[5, 5]],
         "x": 5, "y": 5, "width": 1, "height": 1, "group_id": 1},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.label_color_map = {"nose": "#111"}

    monkeypatch.setattr(
        "labelforge.ui.dialogs.LabelSelectionDialog.select_label",
        staticmethod(lambda *a, **kw: ("nose", 3)),
    )
    monkeypatch.setattr("labelforge.core.config_manager.save_all", lambda **kw: None)
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.modify_label()

    assert editor.detection_boxes[0]["group_id"] == 3
    assert editor.detection_boxes[0]["label"] == "nose"
    assert editor.saved, "group-only change must be persisted"


def test_modify_label_all_mode_clears_group_to_none(monkeypatch):
    """分组留空（None）也要生效，不能被 None 判断吞掉。"""
    class RoleItem(FakeItem):
        def __init__(self, text, box_index):
            super().__init__(text)
            self._data = {0x0100: box_index}

        def data(self, role):
            return self._data.get(role)

    editor = FakeEditor()
    editor._bg_label_list_mode = "all"
    editor.label_list = FakeList([RoleItem("nose [3]", 0)])
    editor.global_labels = {"nose"}
    editor.background_images = ["img.png"]
    editor.detection_boxes = [
        {"label": "nose", "shape_type": "point", "points": [[5, 5]],
         "x": 5, "y": 5, "width": 1, "height": 1, "group_id": 3},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}
    editor.label_color_map = {"nose": "#111"}

    monkeypatch.setattr(
        "labelforge.ui.dialogs.LabelSelectionDialog.select_label",
        staticmethod(lambda *a, **kw: ("nose", None)),
    )
    monkeypatch.setattr("labelforge.core.config_manager.save_all", lambda **kw: None)
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.modify_label()

    assert editor.detection_boxes[0].get("group_id") is None
    assert editor.saved


def test_delete_label_all_mode_removes_only_selected_box(monkeypatch):
    class RoleItem(FakeItem):
        def __init__(self, text, box_index):
            super().__init__(text)
            self._data = {0x0100: box_index}

        def data(self, role):
            return self._data.get(role)

    editor = FakeEditor()
    editor._bg_label_list_mode = "all"
    # Select the second Car (box index 1)
    editor.label_list = FakeList([RoleItem("Car", 1)])
    editor.global_labels = {"Car", "dog"}
    editor.background_images = ["img.png"]
    editor.detection_boxes = [
        {"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1},
        {"label": "Car", "x": 2, "y": 2, "width": 1, "height": 1},
        {"label": "dog", "x": 3, "y": 3, "width": 1, "height": 1},
    ]
    editor.detection_boxes_dict = {0: list(editor.detection_boxes)}

    from labelforge.engine import label_manager
    monkeypatch.setattr(
        label_manager.dialog_helpers,
        "question",
        lambda *a, **kw: label_manager.dialog_helpers.QMessageBox.Yes,
    )
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.delete_label()

    assert editor.detection_boxes == [
        {"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1},
        {"label": "dog", "x": 3, "y": 3, "width": 1, "height": 1},
    ]
    assert editor.global_labels == {"Car", "dog"}
    assert len(editor.saved) == 1


def test_delete_label_saves_each_background_with_its_own_index(monkeypatch):
    editor = FakeEditor()
    editor._bg_label_list_mode = "stats"
    editor.label_list = FakeList([FakeItem("cat (1)")])
    editor.global_labels = {"cat", "dog"}
    editor.background_images = ["img0.png", "img1.png"]
    editor.current_background_index = 1
    editor.detection_boxes = [
        {"label": "cat", "x": 1, "y": 1, "width": 2, "height": 2},
        {"label": "dog", "x": 2, "y": 2, "width": 3, "height": 3},
    ]
    editor.detection_boxes_dict = {
        0: [{"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1}],
        1: list(editor.detection_boxes),
    }

    from labelforge.engine import label_manager
    monkeypatch.setattr(
        label_manager.dialog_helpers,
        "question",
        lambda *a, **kw: label_manager.dialog_helpers.QMessageBox.Yes,
    )
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    manager.delete_label()

    assert editor.detection_boxes == [
        {"label": "dog", "x": 2, "y": 2, "width": 3, "height": 3}
    ]
    assert editor.detection_boxes_dict == {0: [], 1: editor.detection_boxes}
    assert [call[1]["current_index"] for call in editor.saved] == [0, 1]
    assert [call[1]["canvas_items"] for call in editor.saved] == [[], []]


def test_rename_detection_label_updates_memory_sets_and_color_map(tmp_path, monkeypatch):
    editor = FakeEditor()
    editor.global_labels = {"cat"}
    editor.background_dataset_labels = {"cat"}
    editor.label_color_map = {"cat": "#abc"}
    editor.label_colors = ["#abc"]
    editor._cached_bg_label_stats = [{"label": "cat", "count": 2, "color": "#abc"}]
    editor.detection_boxes = [{"label": "cat"}]
    editor.detection_boxes_dict = {0: [{"label": "cat"}], 1: [{"label": "cat"}]}
    editor.current_background_index = 0
    img0 = tmp_path / "a.png"
    img1 = tmp_path / "b.png"
    img0.write_bytes(b"x")
    img1.write_bytes(b"x")
    (tmp_path / "a.json").write_text(
        '{"shapes":[{"label":"cat","points":[[0,0],[1,1]]},{"label":"dog","points":[[0,0],[1,1]]}]}',
        encoding="utf-8",
    )
    (tmp_path / "b.json").write_text(
        '{"shapes":[{"label":"cat","points":[[0,0],[1,1]]}]}',
        encoding="utf-8",
    )
    editor.background_images = [str(img0), str(img1)]
    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()
    monkeypatch.setattr(
        "labelforge.core.config_manager.save_all",
        lambda **kwargs: None,
    )

    assert manager.rename_detection_label("cat", "kitty") is True
    assert editor.global_labels == {"kitty"}
    assert editor.background_dataset_labels == {"kitty"}
    assert editor.label_color_map == {"kitty": "#abc"}
    assert editor.detection_boxes[0]["label"] == "kitty"
    assert editor.detection_boxes_dict[0][0]["label"] == "kitty"
    assert editor.detection_boxes_dict[1][0]["label"] == "kitty"
    assert editor._cached_bg_label_stats[0]["label"] == "kitty"
    import json
    data0 = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    data1 = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert [s["label"] for s in data0["shapes"]] == ["kitty", "dog"]
    assert [s["label"] for s in data1["shapes"]] == ["kitty"]
    assert manager.rename_detection_label("kitty", "kitty") is False


def test_update_label_list_stats_mode_aggregates_task_types(monkeypatch):
    """统计模式下同一标签跨任务时，行上要聚合出多个任务徽标。"""
    from labelforge.engine import label_manager as lm
    from labelforge.core.utils import TASK_DATA_ROLE

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "stats"
    editor.detection_boxes = [
        {"label": "car", "shape_type": "rectangle"},
        {"label": "car", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    row = next(i for i in editor.label_list.items if i.text().startswith("car"))
    assert row.data(TASK_DATA_ROLE) == "det seg"


def test_update_label_list_all_mode_stores_task_type_per_row(monkeypatch):
    """每框一行时，行上要带 det/seg/pose/obb 任务类型供右侧徽标绘制。"""
    from labelforge.engine import label_manager as lm
    from labelforge.core.utils import TASK_DATA_ROLE

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "all"
    editor.detection_boxes = [
        {"label": "car", "shape_type": "rectangle"},
        {"label": "road", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        {"label": "nose", "shape_type": "point", "points": [[0, 0]], "group_id": 1},
        {"label": "sign", "shape_type": "rotation"},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    # 先无组别（det/seg/obb），再组别 1（pose）
    assert [item.data(TASK_DATA_ROLE) for item in editor.label_list.items] == [
        "det", "seg", "obb", "pose",
    ]


def test_update_label_list_all_mode_sorts_by_group(monkeypatch):
    """每框一行：先无组别，再按 group_id 从小到大。"""
    from labelforge.engine import label_manager as lm

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "all"
    editor.detection_boxes = [
        {"label": "c2", "group_id": 2},
        {"label": "none1"},
        {"label": "c1", "group_id": 1},
        {"label": "none2"},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = [item.text() for item in editor.label_list.items]
    assert texts == ["none1", "none2", "c1 [1]", "c2 [2]"]


def test_update_label_list_all_mode_one_row_per_box(monkeypatch):
    from labelforge.engine import label_manager as lm

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "all"
    editor.global_labels = {"zebra", "apple"}
    editor.background_dataset_labels = {"mango"}
    editor.detection_boxes = [
        {"label": "Car"},
        {"label": "apple"},
        {"label": "Car"},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = [item.text() for item in editor.label_list.items]
    assert texts == ["Car", "apple", "Car"]
    # UserRole (0x0100) stores box index for click-to-select.
    assert [item.data(0x0100) for item in editor.label_list.items] == [0, 1, 2]


def test_update_label_list_empty_image_shows_dataset_labels(monkeypatch):
    """Empty current image must still list all dataset labels (count 0)."""
    from labelforge.engine import label_manager as lm

    monkeypatch.setattr(lm, "QListWidgetItem", FakeItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "stats"
    editor.global_labels = set()  # scan may only fill bg set first
    editor.background_dataset_labels = {"zebra", "apple"}
    editor.detection_boxes = []
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = sorted(item.text() for item in editor.label_list.items)
    assert texts == ["apple (0)", "zebra (0)"]


def test_rename_paste_label_updates_list_and_canvas_items():
    class MutableItem:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

        def setText(self, text):
            self._text = text

    class PasteList:
        def __init__(self, items):
            self._items = items

        def count(self):
            return len(self._items)

        def item(self, index):
            return self._items[index]

    editor = FakeEditor()
    editor.paste_label_list = PasteList([MutableItem("paste"), MutableItem("logo")])
    # 真实 canvas_items 是 (pixmap, rect, label) 三元组
    editor.canvas_items = [(None, object(), "logo")]
    editor.canvas_items_dict = {0: [(None, object(), "logo")]}
    editor.label_color_map = {"logo": "#123"}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()

    assert manager.rename_paste_label("logo", "badge") is True
    assert editor.paste_label_list.item(1).text() == "badge"
    assert editor.canvas_items[0][2] == "badge"
    assert editor.canvas_items_dict[0][0][2] == "badge"
    assert editor.label_color_map == {"badge": "#123"}


def test_rename_paste_label_renames_paste_boxes_only():
    """磁盘载入的贴图在 detection_boxes_dict 里带 is_paste，必须同步改名；
    普通检测框（无 is_paste）不能被动。"""
    class PasteList:
        def count(self):
            return 0

        def item(self, i):
            return None

    editor = FakeEditor()
    editor.paste_label_list = PasteList()
    editor.canvas_items = []
    editor.canvas_items_dict = {}
    editor.label_color_map = {"logo": "#123"}
    paste_box = {"label": "logo", "x": 0, "y": 0, "width": 10, "height": 10,
                 "is_paste": True}
    det_box = {"label": "logo", "x": 50, "y": 50, "width": 10, "height": 10}
    editor.detection_boxes = [paste_box, det_box]
    editor.detection_boxes_dict = {0: [paste_box, det_box]}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()

    assert manager.rename_paste_label("logo", "badge") is True
    assert editor.detection_boxes[0]["label"] == "badge"
    assert editor.detection_boxes[1]["label"] == "logo"
    assert editor.detection_boxes_dict[0][0]["label"] == "badge"
    assert editor.detection_boxes_dict[0][1]["label"] == "logo"


def test_update_label_list_all_mode_hides_filtered_out_boxes(monkeypatch):
    """每框一行时，被任务筛选排除的框不出现在列表。"""
    from labelforge.engine import label_manager as lm

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "all"
    editor._task_filter = {"det"}
    editor.detection_boxes = [
        {"label": "car", "shape_type": "rectangle"},
        {"label": "road", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
        {"label": "sign", "shape_type": "rectangle"},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = [item.text() for item in editor.label_list.items]
    assert texts == ["car", "sign"]
    # 隐藏的是显示，不是删除：底层数据仍在
    assert len(editor.detection_boxes) == 3


def test_update_label_list_all_mode_hides_filtered_out_groups(monkeypatch):
    """按分组筛选：只保留选中分组的框。"""
    from labelforge.engine import label_manager as lm

    class RoleItem(FakeItem):
        def __init__(self, text):
            super().__init__(text)
            self._data = {}

        def setData(self, role, value):
            self._data[role] = value

        def data(self, role):
            return self._data.get(role)

    monkeypatch.setattr(lm, "QListWidgetItem", RoleItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "all"
    editor._group_filter = {1}
    editor.detection_boxes = [
        {"label": "a", "group_id": 1},
        {"label": "b", "group_id": 2},
        {"label": "c", "group_id": 1},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = [item.text() for item in editor.label_list.items]
    assert texts == ["a [1]", "c [1]"]


def test_update_label_list_stats_mode_hides_filtered_out_labels(monkeypatch):
    """统计模式下，筛选后不并入无框的数据集标签。"""
    from labelforge.engine import label_manager as lm

    monkeypatch.setattr(lm, "QListWidgetItem", FakeItem)
    editor = FakeEditor()
    editor.current_background = object()
    editor._bg_label_list_mode = "stats"
    editor._task_filter = {"det"}
    editor.global_labels = {"zebra"}
    editor.background_dataset_labels = {"zebra"}
    editor.detection_boxes = [
        {"label": "car", "shape_type": "rectangle"},
        {"label": "road", "shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    ]
    manager = LabelManager(editor)

    manager.update_label_list()

    texts = [item.text() for item in editor.label_list.items]
    assert any(t.startswith("car") for t in texts)
    assert not any(t.startswith("road") for t in texts)
    assert not any(t.startswith("zebra") for t in texts)


def test_rename_paste_label_rewrite_disk_updates_flagged_shapes(tmp_path):
    import json
    img = tmp_path / "p.png"
    img.write_bytes(b"x")
    sidecar = tmp_path / "p.json"
    sidecar.write_text(json.dumps({"shapes": [
        {"label": "logo", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
         "shape_type": "rectangle", "flags": {"paste": True}},
        {"label": "logo", "points": [[2, 2], [3, 2], [3, 3], [2, 3]],
         "shape_type": "rectangle", "flags": {}},
    ]}), encoding="utf-8")

    class PasteList:
        def count(self): return 0
        def item(self, i): return None

    editor = FakeEditor()
    editor.background_images = [str(img)]
    editor.paste_label_list = PasteList()
    editor.canvas_items = []
    editor.canvas_items_dict = {}
    editor.label_color_map = {}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()

    assert manager.rename_paste_label("logo", "badge", rewrite_disk=True) is True
    shapes = json.loads(sidecar.read_text(encoding="utf-8"))["shapes"]
    assert shapes[0]["label"] == "badge"
    assert shapes[1]["label"] == "logo"


def test_rename_paste_label_rewrite_disk_updates_output_sidecar(tmp_path):
    """贴图实际保存在输出目录 sidecar，磁盘改名必须覆盖那里。"""
    import json
    import os
    from labelforge.core.utils import PathUtils
    img = tmp_path / "q.png"
    img.write_bytes(b"x")
    out_dir = PathUtils.get_output_dir(str(img))
    os.makedirs(out_dir, exist_ok=True)
    out_sidecar = os.path.join(out_dir, "q.json")
    with open(out_sidecar, "w", encoding="utf-8") as f:
        json.dump({"shapes": [
            {"label": "logo", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
             "shape_type": "rectangle", "flags": {"paste": True}},
            {"label": "cat", "points": [[2, 2], [3, 2], [3, 3], [2, 3]],
             "shape_type": "rectangle", "flags": {}},
        ]}, f)

    class PasteList:
        def count(self): return 0
        def item(self, i): return None

    editor = FakeEditor()
    editor.background_images = [str(img)]
    editor.paste_label_list = PasteList()
    editor.canvas_items = []
    editor.canvas_items_dict = {}
    editor.label_color_map = {}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()

    assert manager.rename_paste_label("logo", "badge", rewrite_disk=True) is True
    with open(out_sidecar, encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]
    assert shapes[0]["label"] == "badge"
    assert shapes[1]["label"] == "cat"


def test_rename_paste_label_rewrite_disk_updates_prefixed_output_sidecar(tmp_path):
    """默认前缀开启时贴图落在 paste_{stem}.json，改名必须覆盖该文件。"""
    import json
    import os
    from labelforge.core.utils import PathUtils
    img = tmp_path / "r.png"
    img.write_bytes(b"x")
    out_dir = PathUtils.get_output_dir(str(img))
    os.makedirs(out_dir, exist_ok=True)
    out_sidecar = os.path.join(out_dir, "paste_r.json")
    with open(out_sidecar, "w", encoding="utf-8") as f:
        json.dump({"shapes": [
            {"label": "logo", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
             "shape_type": "rectangle", "flags": {"paste": True}},
        ]}, f)

    class PasteList:
        def count(self): return 0
        def item(self, i): return None

    editor = FakeEditor()
    editor.background_images = [str(img)]
    editor.paste_label_list = PasteList()
    editor.canvas_items = []
    editor.canvas_items_dict = {}
    editor.label_color_map = {}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()

    assert manager.rename_paste_label("logo", "badge", rewrite_disk=True) is True
    with open(out_sidecar, encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]
    assert shapes[0]["label"] == "badge"


def test_modify_paste_label_rewrites_disk(tmp_path, monkeypatch):
    """列表右键改名（modify_paste_label）也要落盘，否则重启后统计回到旧名。"""
    import json
    import os
    from labelforge.core.utils import PathUtils
    from labelforge.engine import label_manager as lm
    img = tmp_path / "m.png"
    img.write_bytes(b"x")
    out_dir = PathUtils.get_output_dir(str(img))
    os.makedirs(out_dir, exist_ok=True)
    out_sidecar = os.path.join(out_dir, "m.json")
    with open(out_sidecar, "w", encoding="utf-8") as f:
        json.dump({"shapes": [
            {"label": "logo", "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
             "shape_type": "rectangle", "flags": {"paste": True}},
        ]}, f)

    class MutableItem:
        def __init__(self, text):
            self._text = text

        def text(self):
            return self._text

        def setText(self, text):
            self._text = text

    class PasteList:
        def __init__(self, selected):
            self._selected = selected

        def selectedItems(self):
            return self._selected

        def count(self):
            return len(self._selected)

        def item(self, i):
            return self._selected[i]

    item = MutableItem("logo")
    editor = FakeEditor()
    editor.background_images = [str(img)]
    editor.paste_label_list = PasteList([item])
    editor.canvas_items = [(None, object(), "logo")]
    editor.canvas_items_dict = {}
    editor.label_color_map = {}
    manager = LabelManager(editor)
    manager.data_changed = FakeSignal()
    monkeypatch.setattr(lm.dialog_helpers, "get_text", lambda *a, **kw: ("badge", True))

    manager.modify_paste_label()

    assert item.text() == "badge"
    assert editor.canvas_items[0][2] == "badge"
    with open(out_sidecar, encoding="utf-8") as f:
        shapes = json.load(f)["shapes"]
    assert shapes[0]["label"] == "badge"
