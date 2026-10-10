"""保存前同类框去重回归测试。"""
from pastelabel.engine.save_manager import SaveManager


class Rect:
    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self): return self._x
    def y(self): return self._y
    def width(self): return self._width
    def height(self): return self._height


def test_deduplicate_rectangles_keeps_first_item_when_later_item_iou_is_at_least_98_percent():
    items = [
        (None, Rect(10, 10, 100, 100), 'first'),
        (None, Rect(11, 10, 100, 100), 'later'),
    ]

    kept = SaveManager._deduplicate_rectangles(items, lambda item: item[1])

    assert kept == [items[0]]


def test_deduplicate_rectangles_keeps_different_labels_with_same_geometry():
    """不同类别即使几何几乎重合也都要保留。

    旧行为按纯几何去重、忽略 label，会把「不同类高度重合框」静默删掉一份；
    推理结果里这种情况很常见（模型对同一区域给出多个类别候选），
    保存/重载后标注数量就对不上了。
    """
    boxes = [
        {'label': 'first', 'x': 10, 'y': 10, 'width': 100, 'height': 100},
        {'label': 'later', 'x': 11, 'y': 10, 'width': 100, 'height': 100},
    ]

    kept = SaveManager._deduplicate_rectangles(
        boxes, lambda box: Rect(box['x'], box['y'], box['width'], box['height'])
    )

    assert kept == boxes


def test_deduplicate_rectangles_drops_same_label_duplicate():
    boxes = [
        {'label': 'cat', 'x': 10, 'y': 10, 'width': 100, 'height': 100},
        {'label': 'cat', 'x': 11, 'y': 10, 'width': 100, 'height': 100},
    ]

    kept = SaveManager._deduplicate_rectangles(
        boxes, lambda box: Rect(box['x'], box['y'], box['width'], box['height'])
    )

    assert kept == [boxes[0]]


def test_deduplicate_rectangles_without_label_still_dedupes_geometry():
    """无 label 的条目（贴图 canvas_items）保持原有纯几何去重。"""
    items = [
        (None, Rect(10, 10, 100, 100), 'first'),
        (None, Rect(11, 10, 100, 100), 'later'),
    ]

    kept = SaveManager._deduplicate_rectangles(items, lambda item: item[1])

    assert kept == [items[0]]


def test_deduplicate_rectangles_keeps_rectangles_below_98_percent_iou():
    boxes = [
        {'label': 'first', 'x': 10, 'y': 10, 'width': 100, 'height': 100},
        {'label': 'later', 'x': 20, 'y': 10, 'width': 100, 'height': 100},
    ]

    kept = SaveManager._deduplicate_rectangles(
        boxes, lambda box: Rect(box['x'], box['y'], box['width'], box['height'])
    )

    assert kept == boxes


class _FakeBg:
    def width(self):
        return 100

    def height(self):
        return 80


def test_save_json_marks_paste_shapes_with_flags(tmp_path):
    import json
    from pastelabel.engine.save_manager import SaveManager

    img = tmp_path / "d.png"
    img.write_bytes(b"x")

    class Editor:
        _is_delete_view = False
        edit_mode = "paste"
        canvas_items = [(None, Rect(10, 20, 30, 40), "logo")]
        detection_boxes = [{"label": "cat", "x": 0, "y": 0, "width": 5, "height": 5}]
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        current_background = _FakeBg()
        background_images = [str(img)]

    editor = Editor()
    mgr = SaveManager(editor)
    mgr.save_json(str(img), "d.png", "", canvas_items=editor.canvas_items,
                  image_width=100, image_height=80, current_index=0)
    data = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    by_label = {s["label"]: s for s in data["shapes"]}
    assert by_label["logo"]["flags"] == {"paste": True}
    assert by_label["cat"]["flags"] == {}


def test_save_current_json_annotate_skips_never_annotated(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        detection_boxes = []
        canvas_items = []
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        saved = []

        def __init__(self):
            pass

    editor = Editor()
    mgr = SaveManager(editor)
    mgr.save_json = lambda *a, **k: editor.saved.append((a, k))
    mgr.save_current_json()
    assert editor.saved == []


def test_save_current_json_annotate_saves_when_has_boxes(tmp_path):
    img = tmp_path / "b.jpg"
    img.write_bytes(b"x")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        detection_boxes = [{"label": "Car", "x": 0, "y": 0, "width": 1, "height": 1}]
        canvas_items = []
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        saved = []

    editor = Editor()
    mgr = SaveManager(editor)
    mgr.save_json = lambda *a, **k: editor.saved.append((a, k))
    mgr.save_current_json()
    assert len(editor.saved) == 1
    assert editor.saved[0][0][0] == str(img)


def test_save_current_json_annotate_saves_empty_when_json_exists(tmp_path):
    img = tmp_path / "c.jpg"
    img.write_bytes(b"x")
    (tmp_path / "c.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        detection_boxes = []
        canvas_items = []
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        saved = []

    editor = Editor()
    mgr = SaveManager(editor)
    mgr.save_json = lambda *a, **k: editor.saved.append((a, k))
    mgr.save_current_json()
    assert len(editor.saved) == 1
