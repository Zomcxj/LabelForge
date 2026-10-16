"""回归测试：加载大数据集时首图/列表不得在 UI 线程解析全部 JSON。

历史问题：列表批量填充时对每张图片同步调用
`annotation_status_for_image()`（json.load 整个 sidecar），1903 张数据集
在 UI 线程上要花 15s+ 才能显示图片。
"""
import json
import time

from labelforge.engine.image_loader import (
    STATUS_ANNOTATED,
    STATUS_EMPTY,
    STATUS_PENDING,
    STATUS_UNANNOTATED,
    annotation_status_for_image_light,
    decorate_background_list_item,
    scan_dataset_full,
)


class FakeItem:
    def __init__(self, text=""):
        self._data = {}
        self._text = text
        self.icon = None
        self.tooltip = None

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def setIcon(self, icon):
        self.icon = icon

    def setToolTip(self, text):
        self.tooltip = text

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text


def test_light_status_never_parses_json(monkeypatch, tmp_path):
    """light 模式只做存在性检查，绝不 open/json.load。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    (tmp_path / "a.json").write_text(
        json.dumps({"shapes": [{"label": "cat"}]}), encoding="utf-8")

    from labelforge.engine import image_loader

    def _boom(*a, **k):
        raise AssertionError("light 模式不应解析 JSON")

    monkeypatch.setattr(image_loader.json, "load", _boom)
    assert annotation_status_for_image_light(str(img)) == STATUS_PENDING


def test_light_status_unannotated_without_sidecar(tmp_path):
    img = tmp_path / "b.png"
    img.write_bytes(b"x")
    assert annotation_status_for_image_light(str(img)) == STATUS_UNANNOTATED


def test_decorate_light_marks_pending_without_parse(monkeypatch, tmp_path):
    img = tmp_path / "c.png"
    img.write_bytes(b"x")
    (tmp_path / "c.json").write_text(
        json.dumps({"shapes": [{"label": "cat"}]}), encoding="utf-8")
    item = FakeItem()

    from labelforge.engine import image_loader

    def _boom(*a, **k):
        raise AssertionError("light 装饰不应解析 JSON")

    monkeypatch.setattr(image_loader.json, "load", _boom)
    status = decorate_background_list_item(item, str(img), 0, light=True)
    assert status == STATUS_PENDING


def test_scan_dataset_full_returns_statuses_and_counts(tmp_path):
    annotated = tmp_path / "a.png"
    empty = tmp_path / "b.png"
    unannotated = tmp_path / "c.png"
    for p in (annotated, empty, unannotated):
        p.write_bytes(b"x")
    (tmp_path / "a.json").write_text(json.dumps({"shapes": [
        {"label": "cat"}, {"label": "cat"}, {"label": "dog"}]}), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")

    labels, counts, statuses = scan_dataset_full(
        [str(annotated), str(empty), str(unannotated)])

    assert labels == {"cat", "dog"}
    assert counts == {"cat": 2, "dog": 1}
    assert statuses[str(annotated)] == STATUS_ANNOTATED
    assert statuses[str(empty)] == STATUS_EMPTY
    assert statuses[str(unannotated)] == STATUS_UNANNOTATED


def test_bulk_populate_is_fast_and_off_json_parser(monkeypatch, tmp_path):
    """批量填充 300 张带 JSON 的图片必须远快于逐张解析（回归守卫）。"""
    from labelforge.engine import image_loader

    N = 300
    paths = []
    for i in range(N):
        p = tmp_path / f"img{i:04d}.png"
        p.write_bytes(b"x")
        (tmp_path / f"img{i:04d}.json").write_text(json.dumps({
            "shapes": [{"label": f"c{i % 10}"} for _ in range(15)]}), encoding="utf-8")
        paths.append(str(p))

    def _boom(*a, **k):
        raise AssertionError("批量填充不得在 UI 线程解析 JSON")

    monkeypatch.setattr(image_loader.json, "load", _boom)

    t = time.time()
    for i, path in enumerate(paths):
        item = FakeItem()
        decorate_background_list_item(item, path, i, light=True)
    elapsed = time.time() - t

    assert elapsed < 1.0, f"批量填充过慢: {elapsed:.2f}s"
