"""B6 回归：删框/拖框保存 sidecar 时不得抹掉贴图 shape。

审计 B6：三条保存路径把 `canvas_items=[]` 传给 save_json——非 None
的空列表被当成「0 个贴图」写盘，sidecar 里的贴图 shape 被瞬时抹掉；
下次加载这张图，贴图全部消失。

对照证据：ui/mixins/inference.py:989,995 特意传 `canvas_items=self.canvas_items`
并注释「避免重写 sidecar 时把贴图 shape 丢掉」。正确修法是传 None——
save_json 的回退语义就是「用 editor 当前的画布贴图」。

SaveManager 不依赖真 Qt（conftest 桩已覆盖 QRectF 等），这里用
`SaveManager.__new__` 绑一个最小假 editor，让**真实的 save_json** 落盘，
直接读回 sidecar 断言。
"""
import json
from types import SimpleNamespace

from labelforge.engine.save_manager import SaveManager
from labelforge.engine.event_handler import EventHandlerMixin
from labelforge.canvas.canvas_drawing import CanvasDrawingMixin


class _Rect:
    """save_json 只需要 x()/y()/width()/height()。"""

    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h


def _make(tmp_path):
    """editor + SaveManager（真 save_json）+ 一张贴图 + 一个检测框。"""
    img = tmp_path / "img.png"
    sidecar = tmp_path / "img.json"

    save_mgr = SaveManager.__new__(SaveManager)
    editor = EventHandlerMixin.__new__(EventHandlerMixin)
    editor._is_delete_view = False
    editor.canvas_items = [(None, _Rect(10, 20, 30, 40), "cat")]
    editor.current_background = SimpleNamespace(  # 删框分支要求真值；save_json 取其宽高
        width=lambda: 100, height=lambda: 80)
    editor.current_background_index = 0
    editor.background_images = [str(img)]
    editor.detection_boxes = [{
        "label": "dog", "x": 0, "y": 0, "width": 10, "height": 10,
        "shape_type": "rectangle",
    }]
    editor.detection_boxes_dict = {0: [dict(editor.detection_boxes[0])]}
    editor.canvas = SimpleNamespace(
        selected_box=None, selected_boxes=[], update=lambda: None)
    editor.selected_item = None  # _delete_selected_box 尾部的贴图选择分支
    editor.save_undo_state = lambda: None
    editor.update_label_list = lambda: None
    save_mgr.editor = editor
    return editor, save_mgr, str(img), sidecar


def _paste_shape_labels(sidecar):
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    return [s for s in data["shapes"] if s.get("flags", {}).get("paste")]


def test_drag_save_keeps_paste_shapes(tmp_path):
    """拖框松手的保存路径（canvas._save_current_detection_boxes）
    写出的 sidecar 必须仍含贴图 shape。"""
    editor, save_mgr, img, sidecar = _make(tmp_path)
    canvas = CanvasDrawingMixin.__new__(CanvasDrawingMixin)
    canvas._editor = SimpleNamespace(
        current_background=object(),  # 真值即可进入保存分支
        current_background_index=0,
        background_images=editor.background_images,
        save_json=save_mgr.save_json,
    )

    canvas._save_current_detection_boxes()

    pastes = _paste_shape_labels(sidecar)
    assert len(pastes) == 1, sidecar.read_text(encoding="utf-8")
    assert pastes[0]["label"] == "cat"


def test_delete_box_keeps_paste_shapes(tmp_path):
    """删框路径（editor._delete_selected_box）写出的 sidecar
    必须仍含贴图 shape（框本身被删掉）。"""
    editor, save_mgr, img, sidecar = _make(tmp_path)
    editor.save_json = save_mgr.save_json  # 编辑器入口同走真 save_json
    editor.canvas.selected_box = 0

    editor._delete_selected_box()

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    pastes = [s for s in data["shapes"] if s.get("flags", {}).get("paste")]
    boxes = [s for s in data["shapes"] if not s.get("flags", {}).get("paste")]
    assert len(pastes) == 1, sidecar.read_text(encoding="utf-8")
    assert pastes[0]["label"] == "cat"
    assert boxes == []  # 框确实被删了
