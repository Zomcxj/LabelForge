"""保存前同类框去重回归测试。"""
import json
import os
import pathlib
import subprocess
import sys

from labelforge.engine.save_manager import SaveManager

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死。断言通过后立即
# flush + os._exit(0) 绕开析构；os._exit 不冲刷缓冲，必须先 flush。
_QT_EXIT_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    """在真实 Qt（offscreen）子进程里跑断言，返回 stdout。

    conftest 装的是进程内假 PySide6（没有真实布局/像素），凡是涉及
    「真实控件长什么样」的断言都必须走这里。
    """
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


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
    from labelforge.engine.save_manager import SaveManager

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
    """删除标签后切图：原文 shapes 为空 → json 被真删除（不是写成空内容）。

    注意这里给的是 `{"shapes": []}`。早先这条用例给的是
    `{"shapes":[{"label":"cat"}]}`（一条真实标注）却断言被删——那正是本
    轮修掉的误删行为：判空必须看**原文**，有内容就不能删。
    """
    editor, mgr, img, json_path = _empty_editor(
        tmp_path, [], '{"shapes": []}')

    mgr.save_current_json()

    assert not json_path.exists()


def test_empty_canvas_switch_keeps_json_when_toggle_off(tmp_path):
    """开关关掉后维持旧行为：写空 json、不删文件。"""
    editor, mgr, img, json_path = _empty_editor(
        tmp_path, [], '{"shapes": []}', enabled=False)

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
    from labelforge.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    loaded = config_manager.load_all()
    assert loaded["delete_empty_json_on_switch"] is True


def test_delete_empty_json_toggle_persists(tmp_path, monkeypatch):
    """关掉后写盘、再读回来仍是关的。"""
    from labelforge.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_all(delete_empty_json_on_switch=False)
    assert config_manager.load_all()["delete_empty_json_on_switch"] is False

    config_manager.save_all(delete_empty_json_on_switch=True)
    assert config_manager.load_all()["delete_empty_json_on_switch"] is True


def test_delete_empty_json_explicit_false_is_not_overridden(tmp_path, monkeypatch):
    """显式存过 False 就必须尊重它，不能被默认值 True 盖掉。"""
    from labelforge.core import config_manager

    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(tmp_path / "c.json"))
    config_manager.save_all(delete_empty_json_on_switch=False)
    for _ in range(3):
        assert config_manager.load_all()["delete_empty_json_on_switch"] is False


# ---------- 判空以 json 原文的 shapes 为准（防误删用户数据） ----------
#
# 之前判空只看内存里的 detection_boxes，导致两条误删路径（均已复现）：
#   1. 贴图模式下贴了图 → 切到标注模式（不画框）→ 切图：贴图在 canvas_items
#      里、detection_boxes 为空，于是含贴图标注的 json 被删。
#   2. json 里有 loader 解析不出的 shape（如 points 为空的多边形）→ 只浏览
#      一下再切图：解析后 detection_boxes 为空，文件被删。
# 两条都是**删用户数据**。现在改为读 json 原文：shapes 非空就一律保留。

def _switch_editor(tmp_path, json_obj, *, canvas_items=(), boxes=(),
                   enabled=True):
    """构造切图场景的编辑器；json_obj 为 None 表示没有 json 文件。"""
    img = tmp_path / "s.jpg"
    img.write_bytes(b"x")
    json_path = tmp_path / "s.json"
    if json_obj is not None:
        json_path.write_text(json_obj, encoding="utf-8")

    # 类体里看不到闭包变量，先把参数绑成普通名字再在 class body 里引用
    _items = list(canvas_items)
    _boxes = list(boxes)

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = _items
        detection_boxes = _boxes
        detection_boxes_dict = {0: _boxes}
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = enabled

    return Editor(), SaveManager(Editor()), img, json_path


def test_paste_items_keep_json_even_with_no_boxes(tmp_path):
    """贴图模式下贴图、切到标注模式再切图：含贴图标注的 json 必须保留。

    贴图写在 `canvas_items`，而判空当初只看 `detection_boxes`。贴图是被
    写进 json 的 `shapes`（带 `flags.paste`）的真实标注，不能当空画布删掉。
    """
    editor, mgr, img, json_path = _switch_editor(
        tmp_path,
        json.dumps({"shapes": [{"label": "logo",
                                "flags": {"paste": True}}]}),
        canvas_items=[(None, Rect(10, 10, 30, 30), "logo")],
        boxes=[])

    mgr.save_current_json()

    assert json_path.exists(), "贴图还在画布上，json 不该被删"


def test_unparseable_shape_keeps_json(tmp_path):
    """json 里有工具解析不出的 shape（空 points 多边形）：非空就不能删。

    这是「只是浏览一下就把文件删了」的路径：`box_from_labelme_shape` 对
    `points=[]` 返回 None，载入后 `detection_boxes` 为空，但文件里明明
    有内容。判空必须看**原文**，不能看解析结果。
    """
    editor, mgr, img, json_path = _switch_editor(
        tmp_path,
        json.dumps({"shapes": [{"label": "cat", "shape_type": "polygon",
                                "points": [], "flags": {}}]}),
        canvas_items=[], boxes=[])

    mgr.save_current_json()

    assert json_path.exists(), "原文 shapes 非空，不该删"


def test_truly_empty_shapes_is_deleted(tmp_path):
    """原文 shapes 真的是空数组 → 删（这才是「空标签」）。"""
    editor, mgr, img, json_path = _switch_editor(
        tmp_path, json.dumps({"shapes": []}), canvas_items=[], boxes=[])

    mgr.save_current_json()

    assert not json_path.exists()


def test_corrupt_json_is_kept_not_deleted(tmp_path):
    """json 损坏读不出来：**保留**，不删。

    读不出内容就无法证明它是空的。删用户数据这种事，宁可留下一个坏文件
    （用户能自己处理），也不能因为解析失败就当空文件删掉。
    """
    editor, mgr, img, json_path = _switch_editor(
        tmp_path, "{ this is not valid json", canvas_items=[], boxes=[])

    mgr.save_current_json()

    assert json_path.exists(), "解析失败不能当成空文件删掉"


def test_json_without_shapes_key_is_kept(tmp_path):
    """没有 shapes 键的 json（其它工具写的）也算有内容，不删。"""
    editor, mgr, img, json_path = _switch_editor(
        tmp_path, json.dumps({"version": "5.0.1", "other": [1, 2]}),
        canvas_items=[], boxes=[])

    mgr.save_current_json()

    assert json_path.exists()


def test_empty_shapes_respects_toggle_off(tmp_path):
    """开关关掉：空 shapes 也不删，维持写空 json 的旧行为。"""
    editor, mgr, img, json_path = _switch_editor(
        tmp_path, json.dumps({"shapes": []}), canvas_items=[], boxes=[],
        enabled=False)

    mgr.save_current_json()

    assert json_path.exists()


def test_delete_refreshes_background_list_icon(tmp_path):
    """删掉 json 后要刷新背景列表行的状态图标。

    踩过的坑：删除路径漏了刷新，列表行一直显示删除前的橙色「空标签」圆环，
    用户以为没删掉。保存路径（`save_json`）本来就有刷新，删除路径要对齐。
    """
    calls = []

    img = tmp_path / "r.jpg"
    img.write_bytes(b"x")
    (tmp_path / "r.json").write_text(json.dumps({"shapes": []}),
                                     encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = []
        detection_boxes_dict = {0: []}
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

        def _refresh_background_item_status(self, index=None, path=None):
            calls.append((index, path))

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert not (tmp_path / "r.json").exists()
    assert calls, "删除后必须刷新该行的状态图标"


def test_nonempty_json_is_saved_not_deleted(tmp_path):
    """原文 shapes 非空 → 走保存、不删，且文件里的内容仍是那条标注。

    这条防的是「把非空 json 当空画布删掉」。保存路径本身会刷新图标
    （`save_json` 里的既有行为），所以这里不断言「没刷新」，只断言
    **没被删、且内容保住了**。
    """
    img = tmp_path / "n.jpg"
    img.write_bytes(b"x")
    (tmp_path / "n.json").write_text(
        json.dumps({"shapes": [{"label": "cat"}]}), encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = [{"label": "cat", "x": 0, "y": 0,
                            "width": 5, "height": 5}]
        detection_boxes_dict = {0: detection_boxes}
        current_background_index = 0
        background_images = [str(img)]
        current_background = _FakeBg()
        _delete_empty_json_on_switch = True

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "n.json").exists()
    data = json.loads((tmp_path / "n.json").read_text(encoding="utf-8"))
    assert [sh["label"] for sh in data["shapes"]] == ["cat"]


def test_delete_path_refresh_is_not_triggered_by_save(tmp_path):
    """删除路径的刷新只属于「真的删掉了」；没删就不该由删除路径刷新。

    用一个只在删除成功时才置位的探针，避免和 `save_json` 里既有的刷新混淆
    （那条是保存路径的行为，与本条无关）。
    """
    events = []
    img = tmp_path / "m.jpg"
    img.write_bytes(b"x")
    (tmp_path / "m.json").write_text(json.dumps({"shapes": []}),
                                     encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = []
        detection_boxes_dict = {0: []}
        current_background_index = 0
        background_images = [str(img)]
        current_background = None  # 让保存路径提前返回，隔离出删除路径
        _delete_empty_json_on_switch = True

        def _refresh_background_item_status(self, index=None, path=None):
            events.append("refresh")

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    # 删掉了 → 必然刷新一次（且只有一次）
    assert not (tmp_path / "m.json").exists()
    assert events == ["refresh"], events


def test_shapes_not_a_list_is_kept(tmp_path):
    """`shapes` 不是列表（别的工具写成对象/字符串）：保留，不当空删掉。

    只有 `shapes: []` 才算空。类型不对说明这不是我们认识的标注文件，
    删它等于替用户决定「这个文件没用了」。
    """
    for bad in ('{"shapes": {}}', '{"shapes": "cat"}', '{"shapes": null}',
                '[1, 2, 3]', '"just a string"'):
        editor, mgr, img, json_path = _switch_editor(tmp_path, bad,
                                                    canvas_items=[], boxes=[])
        mgr.save_current_json()
        assert json_path.exists(), bad
        json_path.unlink()


def test_no_extra_refresh_when_delete_is_disabled(tmp_path):
    """开关关掉时不删、也不触发删除路径的刷新（防刷新写成无条件的）。"""
    events = []
    img = tmp_path / "d.jpg"
    img.write_bytes(b"x")
    (tmp_path / "d.json").write_text(json.dumps({"shapes": []}),
                                     encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = []
        detection_boxes_dict = {0: []}
        current_background_index = 0
        current_background = None  # 让保存路径提前返回，隔离删除路径
        background_images = [str(img)]
        _delete_empty_json_on_switch = False

        def _refresh_background_item_status(self, index=None, path=None):
            events.append("refresh")

    mgr = SaveManager(Editor())
    mgr.save_current_json()

    assert (tmp_path / "d.json").exists()
    assert events == [], events


def test_no_refresh_when_delete_fails(tmp_path, monkeypatch):
    """删除失败（只读盘/被占用）时不刷新图标：文件还在，图标不该变。

    这条把「刷新」精确地绑到「真的删掉了」上。若刷新写成无条件的，
    删除失败后图标会变成「未标注」，用户以为文件没了，其实还在。
    """
    events = []
    img = tmp_path / "f.jpg"
    img.write_bytes(b"x")
    (tmp_path / "f.json").write_text(json.dumps({"shapes": []}),
                                     encoding="utf-8")

    class Editor:
        _is_delete_view = False
        edit_mode = "annotate"
        canvas_items = []
        detection_boxes = []
        detection_boxes_dict = {0: []}
        current_background_index = 0
        current_background = None
        background_images = [str(img)]
        _delete_empty_json_on_switch = True

        def _refresh_background_item_status(self, index=None, path=None):
            events.append("refresh")

    mgr = SaveManager(Editor())

    # 真实的 `_delete_sidecar_json` 删不掉时返回 False（只记日志、不抛）。
    # 这里直接模拟那个返回值，专测调用方对 False 的处理。
    monkeypatch.setattr(mgr, "_delete_sidecar_json", lambda p: False)

    mgr.save_current_json()

    assert (tmp_path / "f.json").exists()
    assert events == [], events


def test_delete_toggle_label_is_renamed_in_both_languages():
    """选项菜单里的名字是「空标签文件删除」，且中英两套词条都在。

    改名要同时改三处（action 文本、items 表、i18n 两份），漏一处会出现
    「中文界面正常、英文界面显示中文」这类只在切语言后才暴露的问题。
    """
    from labelforge.ui import i18n

    src = (ROOT / "labelforge" / "ui" / "mixins" / "options_popup.py").read_text(
        encoding="utf-8")
    assert "空标签文件删除" in src
    assert "画布清空时删除标注文件" not in src

    for lang in ("zh", "en"):
        table = i18n._strings[lang]
        assert "空标签文件删除" in table, lang
        assert table["空标签文件删除"] != "", lang
        # 旧名字不该留在词条表里
        assert "画布清空时删除标注文件" not in table, lang

    # 英文那份必须是真英文，不能是中文占位
    assert "画布" not in i18n._strings["en"]["空标签文件删除"]


def test_delete_toggle_menu_text_resolves_after_language_switch():
    """真实控件：切到英文后菜单项显示英文，切回中文显示中文。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui import i18n

app = QApplication([])
for lang in ("zh", "en", "zh"):
    i18n.set_lang(lang)
    print(lang, i18n.t("空标签文件删除"))
""")
    lines = _lines(out)
    assert lines[0] == "zh 空标签文件删除", lines[0]
    assert lines[1] == "en Delete empty label file", lines[1]
    assert lines[2] == "zh 空标签文件删除", lines[2]
