"""标签磁盘改名回归测试（D2 输出目录 listdir 缓存 / D3 共享改名 helper）。"""
import json
import os
from pathlib import Path

from labelforge.core.utils import PathUtils
from labelforge.engine.label_manager import LabelManager


def _paste_shape(label):
    return {"label": label, "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "shape_type": "rectangle", "flags": {"paste": True}}


def _det_shape(label):
    return {"label": label, "points": [[2, 2], [3, 2], [3, 3], [2, 3]],
            "shape_type": "rectangle", "flags": {}}


def _read_labels(path):
    return [s["label"] for s in json.loads(Path(path).read_text(encoding="utf-8"))["shapes"]]


def test_rewrite_paste_label_on_disk_lists_output_dir_once(tmp_path, monkeypatch):
    """整库贴图改名：同一输出目录只 listdir 一次，且全部 sidecar 仍正确改名。"""
    import labelforge.engine.label_manager as lm

    imgs = []
    for stem in ("a", "b", "c"):
        img = tmp_path / f"{stem}.png"
        img.write_bytes(b"x")
        (tmp_path / f"{stem}.json").write_text(json.dumps({"shapes": [
            _paste_shape("logo"), _det_shape("logo"),
        ]}), encoding="utf-8")
        imgs.append(str(img))
    out_dir = PathUtils.get_output_dir(imgs[0])
    os.makedirs(out_dir, exist_ok=True)
    out_sidecars = {}
    for stem in ("a", "b", "c"):
        out_sidecars[stem] = os.path.join(out_dir, f"{stem}.json")
        with open(out_sidecars[stem], "w", encoding="utf-8") as f:
            json.dump({"shapes": [_paste_shape("logo")]}, f)

    calls = []
    real_listdir = lm.os.listdir

    def counting_listdir(path):
        calls.append(path)
        return real_listdir(path)

    monkeypatch.setattr(lm.os, "listdir", counting_listdir)

    editor = type("E", (), {})()
    editor.background_images = imgs
    LabelManager(editor)._rewrite_paste_label_on_disk("logo", "badge")

    assert calls.count(out_dir) == 1
    for stem in ("a", "b", "c"):
        assert _read_labels(tmp_path / f"{stem}.json") == ["badge", "logo"]
        assert _read_labels(out_sidecars[stem]) == ["badge"]


def test_rename_shapes_in_file_match_predicate(tmp_path):
    """共享 helper：match 过滤只改贴图 shape，缺省改全部 shape。"""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"shapes": [
        _paste_shape("cat"), _det_shape("cat"), {"label": "cat"},
        _paste_shape("dog"),
    ]}), encoding="utf-8")

    def is_paste(shape):
        flags = shape.get("flags")
        return isinstance(flags, dict) and flags.get("paste")

    LabelManager._rename_shapes_in_file(str(path), "cat", "dog", is_paste)
    assert _read_labels(path) == ["dog", "cat", "cat", "dog"]

    LabelManager._rename_shapes_in_file(str(path), "dog", "wolf")
    assert _read_labels(path) == ["wolf", "cat", "cat", "wolf"]


def test_rename_detection_label_disk_rewrite_still_works(tmp_path, monkeypatch):
    """背景标签整库改名走重构后的共享 helper，旧名 shape 全部落盘改名。"""
    import labelforge.core.config_manager as config_manager

    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    (tmp_path / "a.json").write_text(json.dumps({"shapes": [
        _det_shape("cat"), _paste_shape("cat"), _det_shape("dog"),
    ]}), encoding="utf-8")

    editor = type("E", (), {})()
    editor.global_labels = {"cat"}
    editor.background_dataset_labels = {"cat"}
    editor.label_color_map = {}
    editor.label_colors = []
    editor._cached_bg_label_stats = [{"label": "cat", "count": 2, "color": "#abc"}]
    editor.detection_boxes = [_det_shape("cat")]
    editor.detection_boxes_dict = {0: [_det_shape("cat")]}
    editor.current_background_index = 0
    editor.background_images = [str(img)]
    editor._memory_background_path = str(img)
    editor._cached_bg_label_stats_path = ""
    editor._notify_lint_boxes_changed = None
    monkeypatch.setattr(config_manager, "save_all", lambda **kwargs: None)

    class FakeSignal:
        def connect(self, *a):
            pass

        def emit(self, *a):
            pass

    manager = LabelManager(editor)
    manager.label_list_changed = FakeSignal()
    manager.data_changed = FakeSignal()

    assert manager.rename_detection_label("cat", "kitty", rewrite_disk=True) is True
    assert _read_labels(tmp_path / "a.json") == ["kitty", "kitty", "dog"]
