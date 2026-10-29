"""导出器行为测试（YOLO / VOC / COCO）。

导出失败是静默的：文件写错也不抛异常。测试真写文件后读回断言，不走源码
字符串。导出器的序列化由 ``dataset_converter`` 的写出层负责，这里测的是
「端到端产物是否正确」——包括坐标钳制、退化框跳过、类别重映射与 VOC 转义。
"""
import json
import os
import struct
import zlib
from xml.etree import ElementTree

import pytest

from labelforge.engine.coco_exporter import CocoExporter
from labelforge.engine.voc_exporter import VocExporter
from labelforge.engine.yolo_exporter import YoloExporter


def _png_bytes(width, height):
    """最小 PNG，供复制用；内容不重要，只要文件存在且文件头可读。"""
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


def _labels_dir(tmp_path):
    return tmp_path / "labels"


# ---------- 通用的筛选 / 跳过 / 中断 ----------

def test_run_skips_empty_items_when_skip_empty(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", []), _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], skip_empty=True, input_data=items)

    labels = sorted(p.name for p in _labels_dir(tmp_path).iterdir())
    assert labels == ["b.txt"]


def test_run_keeps_empty_items_when_skip_empty_false(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", []), _item("b", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], skip_empty=False, input_data=items)

    labels = sorted(p.name for p in _labels_dir(tmp_path).iterdir())
    assert labels == ["a.txt", "b.txt"]
    assert (_labels_dir(tmp_path) / "a.txt").read_text() == ""


def test_run_stops_early_when_interrupted(tmp_path):
    exp = YoloExporter(str(tmp_path), is_interrupted=lambda: True)
    items = [_item("a", [_box("cat", 0, 0, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    labels_dir = _labels_dir(tmp_path)
    assert not labels_dir.exists() or not any(labels_dir.iterdir())


def test_run_filters_boxes_by_selected_labels(tmp_path):
    """只写勾选的类别，并把它重映射成类别 0。"""
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10),
                         _box("dog", 20, 20, 10, 10)])]

    exp.run([], {}, ["dog"], input_data=items)

    text = (_labels_dir(tmp_path) / "a.txt").read_text().strip()
    assert text.startswith("0 "), "只应写出 dog，且重映射为类别 0"
    assert len(text.splitlines()) == 1


def test_run_reports_progress_per_chunk(tmp_path):
    """进度按块上报，最终一定要到 (total, total)。"""
    seen = []
    exp = YoloExporter(str(tmp_path), on_progress=lambda c, t: seen.append((c, t)))
    items = [_item(f"img{i}", [_box("cat", 0, 0, 10, 10)]) for i in range(3)]

    exp.run([], {}, ["cat"], input_data=items)

    assert seen[-1] == (3, 3)


# ---------- YOLO ----------

def test_yolo_writes_normalized_center_coordinates(tmp_path):
    exp = YoloExporter(str(tmp_path))
    # 200x100 的图上，框 (20,10)-(120,60)
    items = [_item("a", [_box("cat", 20, 10, 100, 50)])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (_labels_dir(tmp_path) / "a.txt").read_text().split()
    cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
    assert cls == 0
    assert xc == pytest.approx(0.35, abs=1e-6)   # (20+120)/2 / 200
    assert yc == pytest.approx(0.35, abs=1e-6)   # (10+60)/2 / 100
    assert bw == pytest.approx(0.5, abs=1e-6)
    assert bh == pytest.approx(0.5, abs=1e-6)


def test_yolo_clamps_boxes_to_image_bounds(tmp_path):
    exp = YoloExporter(str(tmp_path))
    # 框越出右/下边界
    items = [_item("a", [_box("cat", 150, 80, 100, 100)])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (_labels_dir(tmp_path) / "a.txt").read_text().split()
    xc, yc, bw, bh = map(float, parts[1:])
    for value in (xc, yc, bw, bh):
        assert 0.0 <= value <= 1.0


def test_yolo_skips_degenerate_boxes(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 10, 10, 0, 0),
                         _box("cat", 20, 20, 10, 10)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert len((_labels_dir(tmp_path) / "a.txt").read_text().strip().splitlines()) == 1


def test_yolo_seg_writes_normalized_polygon_points(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="seg")
    polygon = {"label": "cat", "shape_type": "polygon",
               "points": [[0, 0], [100, 0], [100, 50]], "x": 0, "y": 0,
               "width": 100, "height": 50}
    items = [_item("a", [polygon])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (_labels_dir(tmp_path) / "a.txt").read_text().split()
    assert len(parts) == 1 + 6, "cls + 3 个归一化点"
    assert parts[0] == "0"


def test_yolo_obb_writes_eight_normalized_corner_coords(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="obb")
    obb = {"label": "cat", "shape_type": "rotation",
           "points": [[10, 10], [110, 10], [110, 60], [10, 60]],
           "x": 10, "y": 10, "width": 100, "height": 50}
    items = [_item("a", [obb])]

    exp.run([], {}, ["cat"], input_data=items)

    parts = (_labels_dir(tmp_path) / "a.txt").read_text().split()
    assert len(parts) == 1 + 8, "cls + 4 个归一化角点"


def test_yolo_pose_groups_bbox_and_keypoints_by_group_id(tmp_path):
    exp = YoloExporter(str(tmp_path), mode="pose")
    rect = {"label": "person", "x": 10, "y": 10, "width": 50, "height": 80,
            "group_id": 1}
    kpt = {"label": "person", "shape_type": "point", "points": [[30, 30]],
           "x": 30, "y": 30, "width": 0, "height": 0, "group_id": 1}
    items = [_item("a", [rect, kpt], width=200, height=200)]

    exp.run([], {}, ["person"], input_data=items)

    lines = (_labels_dir(tmp_path) / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 1, "同 group 的框与关键点应合成一行"


def test_yolo_writes_classes_file_in_class_id_order(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("zebra", 0, 0, 5, 5), _box("ant", 0, 0, 5, 5)])]

    exp.run([], {}, ["zebra", "ant"], input_data=items)

    names = (tmp_path / "data.yaml").read_text(encoding="utf-8")
    # 类别表按勾选顺序（这里已排序）写进 data.yaml
    assert "'ant'" in names and "'zebra'" in names
    assert names.index("'ant'") < names.index("'zebra'")


# ---------- VOC ----------

def test_voc_writes_well_formed_xml(tmp_path):
    exp = VocExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 10, 20, 30, 40)])]

    exp.run([], {}, ["cat"], input_data=items)

    xml_path = _labels_dir(tmp_path) / "a.xml"
    root = ElementTree.parse(str(xml_path)).getroot()
    assert root.findtext("filename") == "a.jpg" or root.findtext("filename") == "a.png"
    obj = root.find("object")
    assert obj is not None
    assert obj.findtext("name") == "cat"
    bndbox = obj.find("bndbox")
    assert float(bndbox.findtext("xmin")) == 10
    assert float(bndbox.findtext("ymax")) == 60


def test_voc_escapes_special_chars_in_label_and_filename(tmp_path):
    """标签里的 & < > 必须被转义，否则 XML 解析直接炸。"""
    exp = VocExporter(str(tmp_path))
    items = [_item("a&b", [_box('cat & "dog" <x>', 10, 10, 20, 20)])]

    exp.run([], {}, ['cat & "dog" <x>'], input_data=items)

    xml_path = _labels_dir(tmp_path) / "a&b.xml"
    root = ElementTree.parse(str(xml_path)).getroot()
    assert root.find("object").findtext("name") == 'cat & "dog" <x>'


def test_voc_clamps_and_skips_degenerate_boxes(tmp_path):
    exp = VocExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 150, 80, 100, 100),
                         _box("cat", 10, 10, 0, 0)])]

    exp.run([], {}, ["cat"], input_data=items)

    root = ElementTree.parse(str(_labels_dir(tmp_path) / "a.xml")).getroot()
    objects = root.findall("object")
    assert len(objects) == 1, "退化框应被跳过"
    box = objects[0].find("bndbox")
    assert float(box.findtext("xmax")) <= 200
    assert float(box.findtext("ymax")) <= 100


# ---------- COCO ----------

def _coco(tmp_path):
    return json.loads((tmp_path / "coco_detection.json").read_text(encoding="utf-8"))


def test_coco_writes_categories_images_and_annotations(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 10, 20, 30, 40)])]

    exp.run([], {}, ["cat"], input_data=items)

    payload = _coco(tmp_path)
    assert [c["name"] for c in payload["categories"]] == ["cat"]
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    ann = payload["annotations"][0]
    assert ann["category_id"] == 1
    assert ann["image_id"] == payload["images"][0]["id"]


def test_coco_annotation_ids_are_unique_across_images(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [
        _item("a", [_box("cat", 0, 0, 10, 10), _box("cat", 20, 20, 10, 10)]),
        _item("b", [_box("cat", 0, 0, 10, 10)]),
    ]

    exp.run([], {}, ["cat"], input_data=items)

    ids = [a["id"] for a in _coco(tmp_path)["annotations"]]
    assert len(ids) == len(set(ids)) == 3


def test_coco_clamps_boxes(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 150, 80, 100, 100)])]

    exp.run([], {}, ["cat"], input_data=items)

    bbox = _coco(tmp_path)["annotations"][0]["bbox"]
    x, y, w, h = bbox
    assert x + w <= 200 and y + h <= 100


def test_coco_is_valid_json_with_utf8_labels(tmp_path):
    exp = CocoExporter(str(tmp_path))
    items = [_item("a", [_box("猫", 0, 0, 10, 10)])]

    exp.run([], {}, ["猫"], input_data=items)

    payload = _coco(tmp_path)
    assert payload["categories"][0]["name"] == "猫"


# ---------- 源图复制 ----------

def test_export_copies_source_image_with_original_extension(tmp_path, src_image):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)], img_path=src_image)]

    exp.run([], {}, ["cat"], input_data=items)

    images = sorted(p.name for p in (tmp_path / "images").iterdir())
    assert images == ["a.jpg"] or images == ["a.png"]


def test_export_skips_copy_when_no_source_image(tmp_path):
    """没有源图（画布内合成、或 img_path 为空）时不该抛异常。"""
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)], img_path="")]

    exp.run([], {}, ["cat"], input_data=items)

    assert (_labels_dir(tmp_path) / "a.txt").exists()


def test_export_skips_copy_when_source_missing(tmp_path):
    exp = YoloExporter(str(tmp_path))
    items = [_item("a", [_box("cat", 0, 0, 10, 10)],
                   img_path=str(tmp_path / "missing.png"))]

    exp.run([], {}, ["cat"], input_data=items)

    assert (_labels_dir(tmp_path) / "a.txt").exists()


# ---------- 三种格式都创建输出目录 ----------

@pytest.mark.parametrize("exporter_cls", [YoloExporter, VocExporter, CocoExporter])
def test_every_exporter_creates_output_dirs(tmp_path, exporter_cls):
    exp = exporter_cls(str(tmp_path))
    items = [_item("a", [_box("cat", 10, 10, 20, 20)])]

    exp.run([], {}, ["cat"], input_data=items)

    assert (tmp_path / "images").is_dir()
