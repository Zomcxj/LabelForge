"""Background annotation status + filter helpers."""
import json
import os

from labelforge.engine.image_loader import (
    STATUS_ANNOTATED,
    STATUS_EMPTY,
    STATUS_UNANNOTATED,
    annotation_status_for_image,
)


def test_annotation_status_unannotated(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    assert annotation_status_for_image(str(img)) == STATUS_UNANNOTATED


def test_annotation_status_empty_json(tmp_path):
    img = tmp_path / "b.jpg"
    img.write_bytes(b"x")
    (tmp_path / "b.json").write_text(
        json.dumps({"shapes": []}), encoding="utf-8"
    )
    assert annotation_status_for_image(str(img)) == STATUS_EMPTY


def test_annotation_status_annotated(tmp_path):
    img = tmp_path / "c.jpg"
    img.write_bytes(b"x")
    (tmp_path / "c.json").write_text(
        json.dumps({
            "shapes": [{
                "label": "Car",
                "points": [[0, 0], [10, 10]],
            }]
        }),
        encoding="utf-8",
    )
    assert annotation_status_for_image(str(img)) == STATUS_ANNOTATED


def test_annotation_status_invalid_json_is_empty(tmp_path):
    img = tmp_path / "d.jpg"
    img.write_bytes(b"x")
    (tmp_path / "d.json").write_text("{not-json", encoding="utf-8")
    assert annotation_status_for_image(str(img)) == STATUS_EMPTY


def test_filter_cycle_order():
    order = ('all', 'annotated', 'unannotated', 'empty')
    current = 'all'
    seen = [current]
    for _ in range(4):
        idx = order.index(current)
        current = order[(idx + 1) % len(order)]
        seen.append(current)
    assert seen == ['all', 'annotated', 'unannotated', 'empty', 'all']
