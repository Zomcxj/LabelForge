"""导出器行为测试（YOLO / VOC / COCO）。

这些模块此前零测试覆盖，而导出失败是静默的：文件写错也不抛异常。
测试真写文件后读回断言，不走源码字符串。
"""
import json
import os
import struct
import zlib
from pathlib import Path
from xml.etree import ElementTree

import pytest

from labelforge.engine.base_exporter import BaseExporter
from labelforge.engine.coco_exporter import CocoExporter
from labelforge.engine.voc_exporter import VocExporter
from labelforge.engine.yolo_exporter import YoloExporter


def _png_bytes(width, height):
    """最小 PNG，供 shutil 复制用；内容不重要，只要文件存在。"""
    def chunk(kind, payload):
        return (len(payload).to_bytes(4, "big") + kind + payload +
                zlib.crc32(kind + payload).to_bytes(4, "big"))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"\x00" * ((width * 3 + 1) * height)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def src_image(tmp_path):
    path = tmp_path / "src" / "a.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(200, 100))
    return str(path)


def _item(stem, boxes, width=200, height=100, img_path=None):
    return {
        "stem": stem,
        "boxes": boxes,
        "width": width,
        "height": height,
        "image": None,
        "img_path": img_path if img_path is not None else "",
    }


def _box(label, x, y, w, h):
    return {"label": label, "x": x, "y": y, "width": w, "height": h}


# ---------- BaseExporter ----------

def test_run_skips_empty_items_when_skip_empty(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", []), _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], skip_empty=True, input_data=items)

    labels = sorted(p.name for p in (tmp_path / "labels").iterdir())
    assert labels == ["b.txt"]


def test_run_keeps_empty_items_when_skip_empty_false(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", []), _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], skip_empty=False, input_data=items)

    labels = sorted(p.name for p in (tmp_path / "labels").iterdir())
    assert labels == ["a.txt", "b.txt"]
    assert (tmp_path / "labels" / "a.txt").read_text() == ""


def test_run_stops_early_when_interrupted(tmp_path):
    exp = YoloExporter(str(tmp_path), is_interrupted=lambda: True)
    items = [_item("a", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert not any((tmp_path / "labels").iterdir())


def test_run_filters_boxes_by_selected_labels(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10),
                         _box("dog", 20, 20, 10, 10)])]

    exp.run([], {}, ["dog"], input_data=items)

    text = (tmp_path / "labels" / "a.txt").read_text().strip()
    assert text.startswith("0 "), "只应写出 dog，且重映射为类别 0"
    assert len(text.splitlines()) == 1


def test_get_classes_is_sorted_without_selection(tmp_path):
    exp = BaseExporter(str(tmp_path))
    items = [_item("a", [_box("zebra", 0, 0, 5, 5), _box("ant", 0, 0, 5, 5)])]

    assert exp._get_classes(items) == ["ant", "zebra"]


def test_get_classes_prepends_existing_classes_file(tmp_path):
    """增量导出时沿用已有序号，避免同一目录下类别 id 漂移。"""
    (tmp_path / "classes.txt").write_text("dog\ncat\n")
    exp = BaseExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 5, 5), _box("bird", 0, 0, 5, 5)])]

    assert exp._get_classes(items) == ["dog", "cat", "bird"]


def test_copy_image_uses_original_extension(tmp_path, src_image):
    exp = BaseExporter(str(tmp_path))
    exp._ensure_dirs()

    exp._copy_image(_item("a", [], img_path=src_image))

    copied = tmp_path / "images" / "a.png"
    assert copied.exists()
    assert copied.read_bytes() == Path(src_image).read_bytes()


def test_copy_image_ignores_source_equal_to_destination(tmp_path):
    """同路径导出时不得先删后拷，否则源文件会消失。"""
    out_dir = tmp_path / "out"
    images = out_dir / "images"
    images.mkdir(parents=True)
    src = images / "a.png"
    src.write_bytes(_png_bytes(20, 10))
    exp = BaseExporter(str(out_dir))
    exp._ensure_dirs()

    exp._copy_image(_item("a", [], img_path=str(src)))

    assert src.exists()
    assert src.read_bytes() == _png_bytes(20, 10)


def test_copy_image_skips_when_img_path_is_empty(tmp_path):
    """空 img_path 不是「有源图」：normpath("") 得到 "."，会去复制工作目录。"""
    exp = BaseExporter(str(tmp_path))
    exp._ensure_dirs()

    exp._copy_image(_item("a", [], img_path=""))

    assert not any((tmp_path / "images").iterdir())


def test_copy_image_skips_when_img_path_missing(tmp_path):
    exp = BaseExporter(str(tmp_path))
    exp._ensure_dirs()

    exp._copy_image(_item("a", []))

    assert not any((tmp_path / "images").iterdir())


# ---------- YOLO ----------

def test_yolo_writes_normalized_center_coordinates(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 20, 10, 100, 50)])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert parts[0] == "0"
    x_c, y_c, w_n, h_n = (float(p) for p in parts[1:])
    assert (x_c, y_c) == pytest.approx((70 / 200, 35 / 100))
    assert (w_n, h_n) == pytest.approx((0.5, 0.5))


def test_yolo_clamps_boxes_to_image_bounds(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", -50, -50, 400, 300)])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    x_c, y_c, w_n, h_n = (float(p) for p in parts[1:])
    assert (x_c, y_c, w_n, h_n) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_yolo_skips_degenerate_boxes(tmp_path):
    """越界裁剪后不足 1px 的框必须丢弃，否则会产生零面积标注。"""
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 200, 0, 10, 10),
                         _box("cat", 0, 100, 10, 10),
                         _box("cat", 0, 0, 0, 0)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert (tmp_path / "labels" / "a.txt").read_text() == ""


def test_yolo_seg_writes_normalized_polygon_points(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="seg")
    items = [_item("a", [{
        "label": "cat",
        "shape_type": "polygon",
        "points": [[50, 25], [150, 25], [100, 75]],
        "x": 50, "y": 25, "width": 100, "height": 50,
    }])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert parts[0] == "0"
    assert len(parts) == 7
    assert [float(p) for p in parts[1:3]] == pytest.approx([0.25, 0.25])


def test_yolo_obb_writes_eight_normalized_corner_coords(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="obb")
    items = [_item("a", [{
        "label": "car",
        "shape_type": "rotation",
        "group_id": 3,
        "points": [[20, 10], [120, 10], [120, 60], [20, 60]],
        "x": 20, "y": 10, "width": 100, "height": 50,
    }])]

    exp.run([], {}, ["car"], input_data=items)

    parts = (tmp_path / "labels" / "a.txt").read_text().split()
    assert parts[0] == "0"
    assert len(parts) == 9, parts
    coords = [float(p) for p in parts[1:]]
    assert coords[0:2] == pytest.approx([0.1, 0.1])
    assert coords[2:4] == pytest.approx([0.6, 0.1])
    assert coords[4:6] == pytest.approx([0.6, 0.6])
    assert coords[6:8] == pytest.approx([0.1, 0.6])


def test_yolo_pose_groups_bbox_and_keypoints_by_group_id(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="pose")
    items = [_item("a", [
        {"label": "person", "shape_type": "rectangle", "group_id": 1,
         "x": 40, "y": 20, "width": 40, "height": 80},
        {"label": "person", "shape_type": "point", "group_id": 1,
         "x": 60, "y": 30, "width": 0, "height": 0, "points": [[60, 30]]},
        {"label": "person", "shape_type": "point", "group_id": 1,
         "x": 60, "y": 90, "width": 0, "height": 0, "points": [[60, 90]]},
        {"label": "person", "shape_type": "rectangle", "group_id": None,
         "x": 0, "y": 0, "width": 10, "height": 10},
    ])]

    exp.run([], {}, ["person"], input_data=items)

    lines = (tmp_path / "labels" / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 1, lines
    parts = lines[0].split()
    assert parts[0] == "0"
    assert [float(p) for p in parts[1:5]] == pytest.approx([0.3, 0.6, 0.2, 0.8])
    assert [float(p) for p in parts[5:7]] == pytest.approx([0.3, 0.3])
    assert parts[7] == "2"
    assert [float(p) for p in parts[8:10]] == pytest.approx([0.3, 0.9])


def test_yolo_writes_classes_file_in_class_id_order(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("zebra", 0, 0, 10, 10)])]

    exp.run([], {}, ["zebra", "ant"], input_data=items)

    assert (tmp_path / "classes.txt").read_text().splitlines() == ["ant", "zebra"]
    assert (tmp_path / "labels" / "a.txt").read_text().startswith("1 ")


def test_yolo_skips_item_with_zero_size(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)], width=0, height=0)]

    exp.run([], {}, ["cat"], input_data=items)

    assert not (tmp_path / "labels" / "a.txt").exists()


def test_yolo_reports_progress(tmp_path):
    seen = []
    exp = YoloExporter(str(tmp_path), on_progress=lambda c, t: seen.append((c, t)))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)]),
             _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert seen == [(1, 2), (2, 2)]


# ---------- VOC ----------

def test_voc_writes_well_formed_xml(tmp_path):
    exp = VocExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 20, 10, 100, 50)])]

    exp.run([], {}, ["cat"], input_data=items)

    root = ElementTree.parse(str(tmp_path / "labels" / "a.xml")).getroot()
    assert root.findtext("filename") == "a"
    assert root.findtext("size/width") == "200"
    assert root.findtext("size/height") == "100"
    obj = root.find("object")
    assert obj.findtext("name") == "cat"
    assert obj.findtext("bndbox/xmin") == "20"
    assert obj.findtext("bndbox/ymin") == "10"
    assert obj.findtext("bndbox/xmax") == "120"
    assert obj.findtext("bndbox/ymax") == "60"


def test_voc_clamps_and_skips_degenerate_boxes(tmp_path):
    exp = VocExporter(str(tmp_path))
    items = [_item("a", [_box("cat", -50, -50, 400, 300),
                         _box("dog", 200, 0, 10, 10)])]

    exp.run([], {}, ["cat", "dog"], input_data=items)

    root = ElementTree.parse(str(tmp_path / "labels" / "a.xml")).getroot()
    objects = root.findall("object")
    assert len(objects) == 1
    assert objects[0].findtext("name") == "cat"
    assert objects[0].findtext("bndbox/xmin") == "0"
    assert objects[0].findtext("bndbox/xmax") == "200"


# ---------- COCO ----------

def _read_coco(tmp_path):
    with open(tmp_path / "coco_detection.json", encoding="utf-8") as f:
        return json.load(f)


def test_coco_writes_categories_images_and_annotations(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 20, 10, 100, 50)])]

    exp.run([], {}, ["cat"], input_data=items)

    data = _read_coco(tmp_path)
    assert data["categories"] == [{"id": 1, "name": "cat", "supercategory": ""}]
    assert data["images"][0]["file_name"] == "a"
    assert data["images"][0]["width"] == 200
    ann = data["annotations"][0]
    assert ann["image_id"] == 1
    assert ann["category_id"] == 1
    assert ann["bbox"] == [20, 10, 100, 50]
    assert ann["area"] == 5000


def test_coco_annotation_ids_are_unique_across_images(tmp_path):
    """id 由调用方按「写了几个框」推进；跳过的框不能让下一个 id 撞车。"""
    exp = CocoExporter(str(tmp_path))
    items = [
        _item("a", [_box("cat", 0, 0, 10, 10)]),
        _item("b", [_box("cat", 200, 0, 10, 10),
                    _box("cat", 0, 0, 10, 10)]),
        _item("c", [_box("cat", 0, 0, 10, 10)]),
    ]

    exp.run([], {}, ["cat"], input_data=items)

    ids = [a["id"] for a in _read_coco(tmp_path)["annotations"]]
    assert len(ids) == len(set(ids)), f"id 重复: {ids}"


def test_coco_image_ids_are_unique_and_reference_written_images(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)]),
             _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    data = _read_coco(tmp_path)
    image_ids = [i["id"] for i in data["images"]]
    assert len(image_ids) == len(set(image_ids))
    assert {a["image_id"] for a in data["annotations"]} <= set(image_ids)


def test_coco_skips_image_with_zero_size_but_keeps_others(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)], width=0, height=0),
             _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    data = _read_coco(tmp_path)
    assert [i["file_name"] for i in data["images"]] == ["b"]


def test_coco_clamps_boxes(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", -50, -50, 400, 300)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert _read_coco(tmp_path)["annotations"][0]["bbox"] == [0, 0, 200, 100]


def test_coco_is_valid_json_with_utf8_labels(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("猫", 0, 0, 10, 10)])]

    exp.run([], {}, ["猫"], input_data=items)

    data = _read_coco(tmp_path)
    assert data["categories"][0]["name"] == "猫"


# ---------- 一致性 ----------

@pytest.mark.parametrize("exporter_cls", [YoloExporter, VocExporter, CocoExporter])
def test_every_exporter_creates_images_and_labels_dirs(tmp_path, exporter_cls):
    exp = exporter_cls(str(tmp_path / exporter_cls.__name__))

    exp.run([], {}, ["cat"], input_data=[_item("a", [_box("cat", 0, 0, 10, 10)])])

    root = tmp_path / exporter_cls.__name__
    assert (root / "images").is_dir()
    assert (root / "labels").is_dir()


@pytest.mark.parametrize("exporter_cls", [YoloExporter, VocExporter, CocoExporter])
def test_every_exporter_copies_the_source_image(tmp_path, exporter_cls, src_image):
    root = tmp_path / exporter_cls.__name__
    exp = exporter_cls(str(root))

    exp.run([], {}, ["cat"],
            input_data=[_item("a", [_box("cat", 0, 0, 10, 10)],
                              img_path=src_image)])

    assert (root / "images" / "a.png").read_bytes() == Path(src_image).read_bytes()
