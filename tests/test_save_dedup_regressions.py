"""保存前同类框去重回归测试。"""
import json

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


def test_save_current_json_annotate_deletes_json_when_cleared(tmp_path):
    """空画布 + 已有 json → 删掉 json（默认行为，见「选项」里的开关）。

    这条取代了原先「写一份空 json」的断言：留着空 json 会让「没标过」与
    「标过但全删了」两种状态看起来一样。要保留旧行为请关掉选项开关，
    那条路径由 test_empty_canvas_switch_keeps_json_when_toggle_off 覆盖。
    """
    img = tmp_path / "c.jpg"
    img.write_bytes(b"x")
    (tmp_path / "c.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        detection_boxes = []
        canvas_items = []
        detection_boxes_dict = {0: []}
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        saved = []

    editor = Editor()
    mgr = SaveManager(editor)
    mgr.save_json = lambda *a, **k: editor.saved.append((a, k))
    mgr.save_current_json()

    assert editor.saved == []
    assert not (tmp_path / "c.json").exists()


# ---------- 画布空了 → 切图时删掉 json ----------

def _empty_editor(tmp_path, boxes, extra_json=None, enabled=True):
    """构造一个「当前图已空」的编辑器与 SaveManager。"""
    img = tmp_path / "e.jpg"
    img.write_bytes(b"x")
    json_path = tmp_path / "e.json"
    if extra_json is not None:
        json_path.write_text(extra_json, encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = boxes
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = enabled

    editor = Editor()
    return editor, SaveManager(editor), img, json_path


def test_empty_canvas_switch_deletes_existing_json(tmp_path):
    """删除标签后切图：json 被真删除（不是写成空内容）。"""
    editor, mgr, img, json_path = _empty_editor(
        tmp_path, [], '{"shapes":[{"label":"cat"}]}')

    mgr.save_current_json()

    assert not json_path.exists()


def test_empty_canvas_switch_keeps_json_when_toggle_off(tmp_path):
    """开关关掉后维持旧行为：写空 json、不删文件。"""
    editor, mgr, img, json_path = _empty_editor(
        tmp_path, [], '{"shapes":[{"label":"cat"}]}', enabled=False)

    mgr.save_current_json()

    assert json_path.exists()


def test_empty_canvas_switch_skips_when_no_json(tmp_path):
    """本来就没有 json：不新建、也不报错（「从未标注过」的图不该产生文件）。"""
    editor, mgr, img, json_path = _empty_editor(tmp_path, [], None)

    mgr.save_current_json()

    assert not json_path.exists()


def test_canvas_with_boxes_still_saves_normally(tmp_path):
    """还有框时不删：正常走保存路径（防「误删有内容的 json」）。"""
    editor, mgr, img, json_path = _empty_editor(
        tmp_path,
        [{"label": "car", "x": 0, "y": 0, "width": 5, "height": 5}],
        '{"shapes":[{"label":"cat"}]}')

    mgr.save_current_json()

    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert [s["label"] for s in data["shapes"]] == ["car"]


def test_polygon_only_canvas_counts_as_not_empty(tmp_path):
    """只有多边形、没有矩形框：**不算空**，不能删 json。"""
    path = tmp_path / "p.jpg"
    path.write_bytes(b"x")
    (tmp_path / "p.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = [{"label": "seg", "shape_type": "polygon",
                            "points": [[0, 0], [5, 0], [5, 5]]}]
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(path)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "p.json").exists()


def test_rotation_only_canvas_counts_as_not_empty(tmp_path):
    """只有旋转框：也不算空。"""
    path = tmp_path / "r.jpg"
    path.write_bytes(b"x")
    (tmp_path / "r.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = [{"label": "obb", "shape_type": "rotation",
                            "points": [[0, 0], [5, 0], [5, 5], [0, 5]]}]
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(path)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "r.json").exists()


def test_all_deleted_but_box_list_has_hidden_filter_entries(tmp_path):
    """框被类别筛选隐藏时仍算「有内容」：筛掉的框不属于「删除后为空」的语义。"""
    path = tmp_path / "h.jpg"
    path.write_bytes(b"x")
    (tmp_path / "h.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        # 有框但都被筛掉了（visible=False）
        detection_boxes = [{"label": "cat", "x": 0, "y": 0, "width": 5,
                            "height": 5, "visible": False}]
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(path)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "h.json").exists()


def test_delete_view_never_touches_json(tmp_path):
    """删除视图里切图不该删 json（那是浏览模式，不是编辑结果）。"""
    path = tmp_path / "d.jpg"
    path.write_bytes(b"x")
    (tmp_path / "d.json").write_text('{"shapes":[]}', encoding="utf-8")

    class Editor:
        _is_delete_view = True
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = []
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(path)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "d.json").exists()


# ---------- 选项开关：默认开启且能持久化 ----------

def test_delete_empty_json_default_is_on(tmp_path, monkeypatch):
    """默认开启：配置里没这一项时也读成 True（用户要求默认开启）。"""
    from pastelabel.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    loaded = config_manager.load_all()
    assert loaded["delete_empty_json_on_switch"] is True


def test_delete_empty_json_toggle_persists(tmp_path, monkeypatch):
    """关掉后写盘、再读回来仍是关的。"""
    from pastelabel.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_all(delete_empty_json_on_switch=False)
    assert config_manager.load_all()["delete_empty_json_on_switch"] is False

    config_manager.save_all(delete_empty_json_on_switch=True)
    assert config_manager.load_all()["delete_empty_json_on_switch"] is True


def test_delete_empty_json_explicit_false_is_not_overridden(tmp_path, monkeypatch):
    """显式存过 False 就必须尊重它，不能被默认值 True 盖掉。"""
    from pastelabel.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_all(delete_empty_json_on_switch=False)
    for _ in range(3):
        assert config_manager.load_all()["delete_empty_json_on_switch"] is False
