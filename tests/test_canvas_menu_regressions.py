"""画布右键菜单回归测试。"""
from labelforge.canvas.canvas_menu import CanvasMenuMixin


class FakeCheckbox:
    def __init__(self, checked):
        self.checked = checked

    def isChecked(self):
        return self.checked


class FakeEditor:
    def __init__(self):
        self.show_labels_checkbox = FakeCheckbox(False)
        self._is_delete_view = False
        self.current_background = object()


class FakeCanvas(CanvasMenuMixin):
    def __init__(self):
        self._editor = FakeEditor()
        self.calls = []

    def find_item_at_position(self, mouse_pos):
        return None

    def _find_detection_box_at(self, mouse_pos):
        self.calls.append("box_lookup")
        return 0

    def _show_box_label_menu(self, box_index, mouse_pos):
        self.calls.append("box_menu")

    def _show_background_context_menu(self, mouse_pos):
        self.calls.append("background_menu")


def test_find_polygon_at_mouse_without_qpolygon():
    class Editor:
        detection_boxes = [{
            "label": "p", "shape_type": "polygon",
            "points": [[10, 10], [40, 10], [40, 40]],
            "x": 10, "y": 10, "width": 30, "height": 30,
        }]

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.background_scale = 1

        def get_background_rect(self):
            class R:
                def left(self): return 0
                def top(self): return 0
            return R()

    class Pos:
        def x(self): return 30
        def y(self): return 20

    assert Canvas()._find_detection_box_at(Pos()) == 0


def test_find_polygon_misses_outside():
    class Editor:
        detection_boxes = [{
            "label": "p", "shape_type": "polygon",
            "points": [[10, 10], [40, 10], [40, 40]],
            "x": 10, "y": 10, "width": 30, "height": 30,
        }]

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.background_scale = 1

        def get_background_rect(self):
            class R:
                def left(self): return 0
                def top(self): return 0
            return R()

    class Pos:
        def x(self): return 11
        def y(self): return 39

    assert Canvas()._find_detection_box_at(Pos()) is None


def test_right_click_on_polygon_vertex_deletes_instead_of_label_menu():
    class Editor:
        detection_boxes = [{
            "label": "p", "shape_type": "polygon",
            "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
            "x": 10, "y": 10, "width": 30, "height": 30,
        }]
        show_labels_checkbox = FakeCheckbox(True)
        _is_delete_view = False
        current_background = object()

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.background_scale = 1
            self.calls = []
            self.deleted = []

        def find_item_at_position(self, mouse_pos):
            return None

        def get_background_rect(self):
            class R:
                def left(self): return 0
                def top(self): return 0
            return R()

        def _box_handle_at_pos(self, mouse_pos, box_index):
            return "v1"

        def _show_polygon_vertex_menu(self, box_index, vertex_index, mouse_pos):
            self.calls.append(("vertex_menu", box_index, vertex_index))

        def _show_box_label_menu(self, box_index, mouse_pos):
            self.calls.append("box_menu")

        def _show_background_context_menu(self, mouse_pos):
            self.calls.append("background_menu")

    class Pos:
        def x(self): return 40
        def y(self): return 10

    canvas = Canvas()
    assert canvas._handle_right_click(Pos()) is True
    assert canvas.calls == [("vertex_menu", 0, 1)]


def test_right_click_ignores_boxes_when_labels_hidden():
    canvas = FakeCanvas()

    assert canvas._handle_right_click(object()) is True
    assert canvas.calls == ["background_menu"]


def test_collect_dataset_labels_includes_bg_and_global():
    class Editor:
        background_dataset_labels = {"zebra", "apple"}
        global_labels = {"mango"}
        detection_boxes = [{"label": "banana"}]
        detection_boxes_dict = {0: [{"label": "car"}]}
        label_list = None

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()

    labels = Canvas()._collect_dataset_labels()
    assert labels == ["apple", "banana", "car", "mango", "zebra"]


def test_draw_label_choices_use_dataset_when_label_list_empty():
    """Annotate draw on empty image must offer dataset labels, not only label_list."""
    from labelforge.canvas.canvas_drawing import CanvasDrawingMixin

    class FakeList:
        def count(self):
            return 0

        def item(self, i):
            return None

    class Editor:
        label_list = FakeList()
        background_dataset_labels = {"dog", "cat"}
        global_labels = set()
        detection_boxes = []
        detection_boxes_dict = {}

    class Canvas(CanvasDrawingMixin, CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()

    labels = Canvas()._label_choices_for_draw()
    assert set(labels) >= {"cat", "dog"}


def test_canvas_modify_box_label_updates_stats_like_label_manager(monkeypatch):
    """Canvas「修改标签」只改**右键的那一个框**，统计缓存按 delta=1 修正。

    这条原先断言的是「按标签名全局改名」的旧语义（三个 cat 一起变 kitty、
    `kitty` 计数 3）。用户明确要求画布右键「我只是改一个」，所以现在是
    「只改这一个」：当前图那一个 cat 变 kitty，另外两个 cat 保持不动。
    想整类改名请走背景图标签列表的「修改标签」。
    """
    from labelforge.engine.label_manager import LabelManager
    from labelforge.ui.main_window import ImageEditor

    class FakeSignal:
        def emit(self):
            pass

    class Editor:
        def __init__(self):
            self._is_delete_view = False
            self.current_background_index = 0
            self.global_labels = {"cat", "dog"}
            self.background_dataset_labels = {"cat", "dog"}
            self.label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
            self.label_colors = []
            self._memory_background_path = r"D:\data"
            self._cached_bg_label_stats = [
                {"label": "cat", "count": 3, "color": "#ff0000"},
                {"label": "dog", "count": 1, "color": "#00ff00"},
            ]
            self._cached_bg_label_stats_path = r"D:\data"
            self.detection_boxes = [
                {"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1},
            ]
            self.detection_boxes_dict = {
                0: list(self.detection_boxes),
                1: [
                    {"label": "cat"},
                    {"label": "cat"},
                    {"label": "dog"},
                ],
            }
            self.background_images = ["a.png", "b.png"]
            self.label_manager = None

        def update_label_list(self):
            pass

        def get_label_color(self, label):
            return self.label_color_map.get(label, "#ffffff")

        def save_json(self, *a, **k):
            pass

    class Canvas(CanvasMenuMixin):
        def __init__(self, editor):
            self._editor = editor

        def update(self):
            pass

    monkeypatch.setattr(
        "labelforge.core.config_manager.save_all",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "labelforge.ui.dialogs.LabelSelectionDialog.select_label",
        staticmethod(lambda *a, **k: "kitty"),
    )
    editor = Editor()
    lm = LabelManager(editor)
    lm.label_list_changed = FakeSignal()
    lm.data_changed = FakeSignal()
    editor.label_manager = lm

    Canvas(editor)._modify_box_label(0)

    stats = ImageEditor._collect_bg_stats_for_dialog(editor)
    # 只有当前图那一个框改了：cat 3 -> 2，新增 kitty 1
    assert stats.get("kitty") == 1
    assert stats.get("cat") == 2
    assert stats.get("dog") == 1
    # 别的图片里的框不能被殃及
    assert [b.get("label") for b in editor.detection_boxes_dict[1]] == \
        ["cat", "cat", "dog"]


def test_change_box_label_syncs_stats_cache_and_keeps_target_color(monkeypatch):
    """Canvas switch-to-existing must update stats and not steal target color."""
    class FakeLM:
        def __init__(self, editor):
            self.editor = editor
            self.saved = []

        def _save_detection_json_for_index(self, index):
            self.saved.append(index)

        def _label_already_exists(self, label):
            from labelforge.engine.label_manager import LabelManager
            return LabelManager._label_already_exists(self, label)

        def _apply_rename_color(self, *a, **kw):
            from labelforge.engine.label_manager import LabelManager
            return LabelManager._apply_rename_color(self, *a, **kw)

        def _sync_cached_stats_label_rename(self, *a, **kw):
            from labelforge.engine.label_manager import LabelManager
            return LabelManager._sync_cached_stats_label_rename(self, *a, **kw)

        def _seed_stats_cache_from_disk_and_memory(self, *a, **kw):
            from labelforge.engine.label_manager import LabelManager
            return LabelManager._seed_stats_cache_from_disk_and_memory(self, *a, **kw)

    class Editor:
        _is_delete_view = False
        current_background_index = 0
        global_labels = {"cat", "dog"}
        background_dataset_labels = {"cat", "dog"}
        label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
        label_colors = []
        _cached_bg_label_stats = [
            {"label": "cat", "count": 1, "color": "#ff0000"},
            {"label": "dog", "count": 2, "color": "#00ff00"},
        ]
        detection_boxes = [
            {"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1},
            {"label": "dog", "x": 1, "y": 1, "width": 1, "height": 1},
        ]
        detection_boxes_dict = {}
        background_images = []
        lists_updated = 0

        def __init__(self):
            self.detection_boxes_dict = {0: list(self.detection_boxes)}
            self.label_manager = FakeLM(self)

        def update_label_list(self):
            self.lists_updated += 1

        def get_label_color(self, label):
            return self.label_color_map.get(label, "#ffffff")

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.updated = False

        def update(self):
            self.updated = True

    monkeypatch.setattr(
        "labelforge.core.config_manager.save_all",
        lambda **kwargs: None,
    )
    canvas = Canvas()
    canvas._change_box_label(0, "dog")

    assert canvas._editor.detection_boxes[0]["label"] == "dog"
    assert canvas._editor.label_color_map.get("dog") == "#00ff00"
    stats = {s["label"]: s for s in canvas._editor._cached_bg_label_stats}
    assert "cat" not in stats or stats.get("cat", {}).get("count", 0) == 0
    assert stats["dog"]["count"] == 3
    assert stats["dog"]["color"] == "#00ff00"
    assert canvas._editor.lists_updated == 1
    assert canvas.updated is True


def test_canvas_change_box_stats_include_unloaded_images(tmp_path, monkeypatch):
    """Canvas single-box change must not drop counts for images only on disk."""
    import json
    from labelforge.engine.label_manager import LabelManager
    from labelforge.ui.main_window import ImageEditor

    img0 = tmp_path / "a.png"
    img1 = tmp_path / "b.png"
    img0.write_bytes(b"x")
    img1.write_bytes(b"x")
    (tmp_path / "a.json").write_text(json.dumps({
        "shapes": [
            {"label": "cat", "points": [[0, 0], [1, 1]]},
            {"label": "dog", "points": [[0, 0], [1, 1]]},
        ],
    }), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "shapes": [
            {"label": "cat", "points": [[0, 0], [1, 1]]},
            {"label": "cat", "points": [[0, 0], [1, 1]]},
            {"label": "cat", "points": [[0, 0], [1, 1]]},
        ],
    }), encoding="utf-8")

    class FakeSignal:
        def emit(self):
            pass

    class Editor:
        def __init__(self):
            self._is_delete_view = False
            self.current_background_index = 0
            self.global_labels = {"cat", "dog"}
            self.background_dataset_labels = {"cat", "dog"}
            self.label_color_map = {"cat": "#ff0000", "dog": "#00ff00"}
            self.label_colors = []
            self._memory_background_path = str(tmp_path)
            self._cached_bg_label_stats = []  # never opened stats
            self._cached_bg_label_stats_path = ""
            self.detection_boxes = [
                {"label": "cat", "x": 0, "y": 0, "width": 1, "height": 1},
                {"label": "dog", "x": 0, "y": 0, "width": 1, "height": 1},
            ]
            # Only image 0 loaded; image 1 lives on disk only.
            self.detection_boxes_dict = {0: list(self.detection_boxes)}
            self.background_images = [str(img0), str(img1)]
            self.label_manager = None

        def update_label_list(self):
            pass

        def get_label_color(self, label):
            return self.label_color_map.get(label, "#ffffff")

        def save_json(self, *args, **kwargs):
            import os
            idx = kwargs.get("current_index", self.current_background_index)
            path = self.background_images[idx]
            jp = os.path.splitext(path)[0] + ".json"
            boxes = self.detection_boxes_dict.get(idx, self.detection_boxes)
            with open(jp, "w", encoding="utf-8") as f:
                json.dump({
                    "shapes": [
                        {"label": b["label"], "points": [[0, 0], [1, 1]]}
                        for b in boxes
                    ],
                }, f)

    class Canvas(CanvasMenuMixin):
        def __init__(self, editor):
            self._editor = editor

        def update(self):
            pass

    monkeypatch.setattr(
        "labelforge.core.config_manager.save_all",
        lambda **kwargs: None,
    )
    editor = Editor()
    lm = LabelManager(editor)
    lm.label_list_changed = FakeSignal()
    lm.data_changed = FakeSignal()
    editor.label_manager = lm

    Canvas(editor)._change_box_label(0, "dog")  # one cat → dog on image 0

    # img0: dog, dog; img1: cat×3 → cat=3, dog=2
    stats = ImageEditor._collect_bg_stats_for_dialog(editor)
    assert stats.get("cat") == 3
    assert stats.get("dog") == 2


def test_delete_current_label_file_removes_json_and_clears_boxes(tmp_path, monkeypatch):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    json_path = tmp_path / "a.json"
    json_path.write_text('{"shapes":[{"label":"Car","points":[[0,0],[1,1]]}]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        current_background_index = 0
        background_images = [str(img)]
        detection_boxes = [{"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1}]
        detection_boxes_dict = {0: [{"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1}]}
        refreshed = []
        jumped_to = []

        def update_label_list(self):
            pass

        def _refresh_background_item_status(self, idx, path):
            self.refreshed.append((idx, path))

        def switch_background_to_index(self, index):
            self.jumped_to.append(index)

        def _find_bg_list_row_for_index(self, index):
            return index

        background_list = type("List", (), {"setCurrentRow": lambda self, r: None})()

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()
            self.selected_box = 0
            self.selected_boxes = [0]
            self.updated = False

        def update(self):
            self.updated = True

    from labelforge.canvas import canvas_menu as cm
    monkeypatch.setattr(
        cm.dialog_helpers,
        "question",
        lambda *a, **kw: cm.dialog_helpers.QMessageBox.Yes,
    )
    canvas = Canvas()
    canvas._delete_current_label_file()

    assert canvas._editor.detection_boxes == []
    assert canvas._editor.detection_boxes_dict[0] == []
    assert not json_path.exists()
    assert canvas._editor.refreshed == [(0, str(img))]
    assert canvas.updated is True
    assert canvas._editor.jumped_to == [0]


def test_remove_current_background_closes_lint_dialog(tmp_path):
    """移除图片会重排索引，质检弹窗必须关闭（否则指向错图）。"""
    img0 = tmp_path / "r0.png"
    img1 = tmp_path / "r1.png"
    img0.write_bytes(b"x")
    img1.write_bytes(b"x")

    class Editor:
        _is_delete_view = False
        current_background_index = 0
        background_images = [str(img0), str(img1)]
        detection_boxes = []
        detection_boxes_dict = {0: [], 1: []}
        canvas_items_dict = {0: [], 1: []}
        canvas_items = []
        current_background = None
        closed = []

        def _close_lint_dialog(self):
            self.closed.append(True)

        def update_file_count(self):
            pass

        def update_label_list(self):
            pass

        def _show_work_view(self):
            pass

    class Canvas(CanvasMenuMixin):
        def __init__(self):
            self._editor = Editor()

        def update(self):
            pass

        def reset_view(self):
            pass

    canvas = Canvas()
    canvas._remove_current_background()

    assert canvas._editor.closed == [True]
