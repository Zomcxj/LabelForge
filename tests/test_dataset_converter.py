"""Tests for labelforge/engine/dataset_converter.py (数据集转换/校验适配层).

转换实现完全自研、零外部依赖，因此所有用例在任何环境都必须真实执行，
不存在可选依赖跳过。

不访问网络、Roboflow 或任何模型下载；全部数据在 tmp_path 里生成。
"""
import json
import os
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from labelforge.engine import dataset_converter as ST
from labelforge.engine.dataset_converter import (
    COCO_ANNOTATION_NAME,
    IMAGE_EXTS,
    DatasetToolError,
    ValidationReport,
    convert_dataset,
    convert_paths,
    load_dataset,
    output_layout,
    read_image_size,
    validate_dataset,
)


def _png_chunk(chunk_type, payload):
    return (len(payload)).to_bytes(4, "big") + chunk_type + payload + \
        zlib.crc32(chunk_type + payload).to_bytes(4, "big")


def make_png(width, height, color_type=2):
    """构造最小 PNG 字节，只要求文件头能被尺寸解析器读出宽高。"""
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = b"\x00" * ((width * 3 + 1) * height)
    return (b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) +
            _png_chunk(b"IDAT", zlib.compress(raw)) +
            _png_chunk(b"IEND", b""))


def write_png(path, width, height):
    Path(path).write_bytes(make_png(width, height))


def _jpeg_segment(marker, payload):
    return marker + (2 + len(payload)).to_bytes(2, "big") + payload


def make_jpeg(width, height, quant_table_bytes=0):
    """最小 JPEG；quant_table_bytes > 0 时插入伪造量化表以推后 SOF 段。"""
    app0 = _jpeg_segment(b"\xff\xe0",
                         b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    dqt = b""
    if quant_table_bytes > 0:
        dqt = _jpeg_segment(bytes([0xFF, 0xDB]),
                            bytes([0, 0x10]) * (quant_table_bytes - 5))
    sof0 = _jpeg_segment(b"\xff\xc0",
                         b"\x08" + struct.pack(">HH", height, width) +
                         b"\x03\x01\x22\x00\x01\x11\x01\x01\x11\x01")
    return (bytes([0xFF, 0xD8]) + app0 + dqt + sof0 +
            bytes([0xFF, 0xD9]))


def _gif_bytes(width, height):
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 16


def _bmp_bytes(width, height):
    return (b"BM" + (54).to_bytes(4, "little") + b"\x00" * 8 +
            (40).to_bytes(4, "little") + struct.pack("<ii", width, height))


def _webp_bytes(width, height):
    """VP8L WebP：宽高编码在 32 位位域里。"""
    value = ((height - 1) << 14) | (width - 1)
    body = b"VP8L" + (30).to_bytes(4, "little") + b"\x2f" + \
        struct.pack("<I", value) + b"\x00" * 12
    return (b"RIFF" + (4 + len(body)).to_bytes(4, "little") +
            b"WEBP" + body)


def _tiff_bytes(width, height, short=False):
    """TIFF：每项 12 字节。

    short=True 时宽高写成 SHORT(type 3)，否则用 LONG(type 4)。
    两者都是 TIFF 规范允许的写法，解析器都得读对。
    """
    header = b"II" + b"*" + bytes([0]) + (8).to_bytes(4, "little")
    entries = b""
    if short:
        for tag, value in ((256, width), (257, height)):
            entries += struct.pack("<HHI", tag, 3, 1) + \
                struct.pack("<H", value) + b"\x00" * 2
    else:
        for tag, value in ((256, width), (257, height)):
            entries += struct.pack("<HHI", tag, 4, 1) + \
                struct.pack("<I", value)
    return header + struct.pack("<H", 2) + entries + (0).to_bytes(4, "little")


def _webp_vp8x_bytes(width, height):
    """VP8X WebP 容器：chunk data 为 flags(4) + 宽-1(3) + 高-1(3)。

    宽高在绝对偏移 24/27，不在 20/23 —— 20..24 是特性标志位。
    """
    chunk = b"VP8X" + (10).to_bytes(4, "little") + \
        bytes([0x00, 0x00, 0x00, 0x00]) + \
        (width - 1).to_bytes(3, "little") + (height - 1).to_bytes(3, "little")
    return b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WEBP" + chunk


def SV(func):
    """历史上用于在缺少 supervision 时跳过用例；实现已内置，现全部执行。"""
    return func


# --------------------------------------------------------------------------
# 轻量假对象：validate_dataset 只依赖这三个属性
# --------------------------------------------------------------------------
class StubDetections:
    """支持 len() 与 class_id 的最小检测结果。"""

    def __init__(self, class_ids=None, with_class_id=True, size=None):
        if with_class_id:
            self.class_id = list(class_ids or [])
        self._size = size

    def __len__(self):
        if self._size is not None:
            return self._size
        return len(getattr(self, "class_id", []))


class StubDataset:
    def __init__(self, classes=(), image_paths=(), annotations=None):
        self.classes = list(classes)
        self.image_paths = list(image_paths)
        self.annotations = dict(annotations or {})


def _make_stub_dataset(base_dir, images=("a.png", "b.png", "c.png")):
    """生成一个能通过校验的假数据集（真实文件存在）。"""
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in images:
        path = images_dir / name
        path.write_bytes(b"\x89PNG-fake")
        paths.append(str(path))
    annotations = {
        paths[0]: StubDetections([0, 1]),
        paths[1]: StubDetections([1]),
        paths[2]: StubDetections([]),
    }
    return StubDataset(classes=["cat", "dog"], image_paths=paths, annotations=annotations)


class _Slice:
    """_slice_dataset 的替身，只需能被 len()。"""

    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size


@pytest.fixture
def offline_convert(monkeypatch):
    """把 supervision 相关内部实现替换掉，使 convert_dataset 可离线验证。"""
    calls = []
    monkeypatch.setattr(ST, "_slice_dataset", lambda dataset, paths: _Slice(len(paths)))

    def fake_write(chunk, output_format, layout, first, task="det"):
        calls.append({"format": output_format, "chunk": len(chunk), "first": first,
                      "layout": layout, "task": task})
        return None

    monkeypatch.setattr(ST, "_write_chunk", fake_write)
    return calls


# --------------------------------------------------------------------------
# 导入与依赖探测（无条件）
# --------------------------------------------------------------------------
def test_module_imports_and_exposes_public_api():
    assert ST is not None
    for name in ("load_dataset", "validate_dataset", "convert_dataset",
                 "convert_paths", "output_layout", "read_image_size"):
        assert callable(getattr(ST, name))
    assert issubclass(DatasetToolError, Exception)
    assert ST.INPUT_FORMATS == ("yolo", "coco", "voc", "labelme")
    assert ST.OUTPUT_FORMATS == ST.INPUT_FORMATS
    assert all(ext.startswith(".") for ext in IMAGE_EXTS)


# --------------------------------------------------------------------------
# 图片尺寸：只读文件头，不解码整图
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ext,expected", [
    (".png", (150, 200)),
    (".jpg", (640, 480)),
    (".gif", (32, 64)),
    (".bmp", (200, 100)),
    (".webp", (150, 250)),
])
def test_read_image_size_covers_common_formats(tmp_path, ext, expected):
    path = tmp_path / ("size" + ext)
    width, height = expected
    if ext == ".png":
        path.write_bytes(make_png(width, height))
    elif ext == ".jpg":
        path.write_bytes(make_jpeg(width, height))
    elif ext == ".gif":
        path.write_bytes(_gif_bytes(width, height))
    elif ext == ".bmp":
        path.write_bytes(_bmp_bytes(width, height))
    elif ext == ".webp":
        path.write_bytes(_webp_bytes(width, height))
    assert read_image_size(str(path)) == expected


def test_read_image_size_accepts_tiff(tmp_path):
    path = tmp_path / "size.tif"
    path.write_bytes(_tiff_bytes(123, 45))
    assert read_image_size(str(path)) == (123, 45)


def test_read_image_size_accepts_tiff_with_short_typed_dimensions(tmp_path):
    """宽高写成 SHORT(type 3) 时不能抛 struct.error。

    IFD 值域固定 4 字节，但 SHORT 只占其中前 2 字节；
    按 4 字节去解 2 字节格式会抛「requires a buffer of 2 bytes」。
    """
    path = tmp_path / "size.tif"
    path.write_bytes(_tiff_bytes(123, 45, short=True))
    assert read_image_size(str(path)) == (123, 45)


@pytest.mark.parametrize("width,height", [(150, 250), (1, 1), (32767, 4096)])
def test_read_image_size_reads_webp_vp8x_canvas(tmp_path, width, height):
    """VP8X 容器宽高在偏移 24/27，不是 20/23（那里是特性标志位）。"""
    path = tmp_path / "anim.webp"
    path.write_bytes(_webp_vp8x_bytes(width, height))
    assert read_image_size(str(path)) == (width, height)


def test_read_image_size_returns_none_on_unknown_or_missing(tmp_path):
    assert read_image_size(str(tmp_path / "nope.png")) is None
    garbage = tmp_path / "garbage.bin"
    garbage.write_bytes(b"this is not an image at all" * 40)
    assert read_image_size(str(garbage)) is None


def test_read_image_size_scans_jpeg_past_quantization_tables(tmp_path):
    """JPEG 的 SOF 段位于文件中部，必须跳过量化表才能读到尺寸。"""
    path = tmp_path / "big.jpg"
    path.write_bytes(make_jpeg(1024, 768, quant_table_bytes=8192))
    assert read_image_size(str(path)) == (1024, 768)


# --------------------------------------------------------------------------
# data.yaml 子集解析（不引入 PyYAML）
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("names: ['cat', 'dog']\n", ["cat", "dog"]),
    ("names: [cat, dog]\n", ["cat", "dog"]),
    ("names:\n  - cat\n  - dog\n", ["cat", "dog"]),
    ("names:\n  1: cat\n  0: dog\n", ["dog", "cat"]),
    ("# comment\ntrain: images\nnames: ['a']\n", ["a"]),
    ("names: {'0': 'cat', '1': 'dog'}\n", ["cat", "dog"]),
    ("names: ['it''s', 'dog']\n", ["it's", "dog"]),
])
def test_parse_names_handles_common_shapes(text, expected):
    assert ST._parse_names(text) == expected


def test_parse_names_empty_when_absent():
    assert ST._parse_names("train: images\nval: images\n") == []


@pytest.mark.parametrize("text,expected", [
    ("train: images\n", ["images"]),
    ("train: [/abs/a, /abs/b]\n", ["/abs/a", "/abs/b"]),
    ("train:\n  - /a\n  - /b\n", ["/a", "/b"]),
])
def test_parse_paths_reads_target_key_only(text, expected):
    assert ST._parse_paths(text, "train") == expected


def test_parse_paths_empty_for_unknown_key():
    assert ST._parse_paths("train: images\n", "val") == []


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------
def test_dataset_size_of_reads_from_disk_on_cache_miss(tmp_path):
    path = tmp_path / "a.png"
    write_png(path, 30, 20)
    dataset = ST.Dataset(classes=["cat"], image_paths=[str(path)])
    assert dataset.size_of(str(path)) == (30, 20)
    assert dataset.size_of(str(path)) == (30, 20)


def test_dataset_size_of_raises_on_unreadable(tmp_path):
    dataset = ST.Dataset(classes=["cat"], image_paths=[])
    with pytest.raises(DatasetToolError, match="无法读取图片尺寸"):
        dataset.size_of(str(tmp_path / "missing.png"))


def test_detections_mask_is_none_without_segmentation():
    det = ST.Detections(class_id=[0], box_xyxy=[(1, 2, 3, 4)], mask_points=[None])
    assert det.mask is None
    assert det.box_xywh() == [(1, 2, 2, 2)]


def test_detections_mask_returns_polygons_when_present():
    det = ST.Detections(class_id=[0], box_xyxy=[(1, 1, 3, 3)],
                        mask_points=[[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]])
    assert det.mask == [[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0)]]


def test_bounding_box_of_point_cloud():
    assert ST._bounding_box([(1, 4), (5, 2), (3, 8)]) == (1, 2, 5, 8)


# --------------------------------------------------------------------------
# XML 读取（纯标准库，冻结环境无 expat）
# --------------------------------------------------------------------------
def test_parse_xml_root_structure(tmp_path):
    xml = """<?xml version="1.0" encoding="utf-8"?>
<!-- 注释 -->
<annotation>
  <folder>images</folder>
  <filename>a.png</filename>
  <size><width>64</width><height>32</height><depth>3</depth></size>
  <segmented>0</segmented>
</annotation>
"""
    root = ST.parse_xml_root(xml)
    assert root[0] == "annotation"
    assert ST._child_text(root, "filename") == "a.png"
    size = ST._find_child(root, "size")
    assert ST._child_text(size, "width") == "64"
    assert ST._child_text(size, "height") == "32"
    assert ST._child_text(root, "missing") == ""


def test_parse_xml_root_attributes_and_self_closing():
    root = ST.parse_xml_root('<a x="1" y="2"><b/><c>hi</c></a>')
    assert root[0] == "a"
    tags = [child[0] for child in root[2]]
    assert tags == ["b", "c"]
    assert root[2][0][1] == ""
    assert root[2][1][1] == "hi"


def test_parse_xml_root_decodes_entities():
    root = ST.parse_xml_root('<n>&amp;lt;&quot;cat&apos;s&quot;&#60;&#x30; &#x4E2D;</n>')
    assert root[0] == "n"
    assert root[1] == '&lt;"cat\'s"<0 中'


def test_parse_xml_root_handles_cdata_and_doctype():
    xml = ('<?xml version="1.0"?>\n<!DOCTYPE annotation SYSTEM "voc.dtd">\n'
           '<r>before<![CDATA[not <tag>]]></r>')
    root = ST.parse_xml_root(xml)
    assert root[0] == "r"
    assert root[1] == "beforenot <tag>"


def test_parse_xml_root_raises_on_malformed_xml():
    for bad in ("<a x=", "<!-- unclosed", "<![CDATA[unclosed"):
        with pytest.raises(DatasetToolError, match="不合法"):
            ST.parse_xml_root(bad)


def test_parse_xml_root_reports_mismatched_tag_name():
    with pytest.raises(DatasetToolError, match="标签不匹配"):
        ST.parse_xml_root("<a><b></a>")


def test_parse_xml_root_matches_elementtree(tmp_path):
    """与 xml.etree.ElementTree 的结果逐项比对，保证解析语义一致。"""
    import xml.etree.ElementTree as ET

    xml = ('<?xml version="1.0"?>\n'
           '<annotation><folder>images</folder><filename>x.png</filename>\n'
           '  <size><width>64</width><height>64</height><depth>3</depth></size>\n'
           '  <object><name>cat</name><pose>Unspecified</pose><truncated>0</truncated>\n'
           '    <difficult>0</difficult>\n'
           '    <bndbox><xmin>12.8</xmin><ymin>12.8</ymin>'
           '<xmax>25.6</xmax><ymax>25.6</ymax></bndbox></object>\n'
           '  <object><name>&amp;dog</name>'
           '    <bndbox><xmin>1</xmin><ymin>1</ymin>'
           '<xmax>9</xmax><ymax>9</ymax></bndbox></object>\n'
           '</annotation>')
    mine = ST.parse_xml_root(xml)
    tree = ET.fromstring(xml)

    def _txt(value):
        """ET 把纯空白文本记为 None，统一成去空白字符串再比。"""
        return "" if value is None else value.strip()

    assert mine[0] == tree.tag
    assert _txt(mine[1]) == _txt(tree.text)
    assert len(mine[2]) == len(list(tree))
    assert [child[0] for child in mine[2]] == [child.tag for child in tree]
    assert [_txt(child[1]) for child in mine[2] if child[0] == "object"] == \
        [_txt(child.text) for child in tree if child.tag == "object"]
    objects = [child for child in tree if child.tag == "object"]
    assert ST._child_text(mine[2][3], "name") == objects[0].findtext("name")
    assert ST._child_text(mine[2][4], "name") == objects[1].findtext("name")


def test_voc_reader_handles_real_world_variants(tmp_path):
    """属性、注释、多行缩进、实体转义都不应影响读取。"""
    images = tmp_path / "images"
    xmls = tmp_path / "xml"
    images.mkdir()
    xmls.mkdir()
    write_png(images / "x.png", 64, 32)
    (xmls / "x.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- generated by labelImg -->\n"
        '<annotation verified="1">\n'
        "    <folder>images</folder>\n"
        "    <filename>x.png</filename>\n"
        "    <source><database>Unknown</database></source>\n"
        "    <size>\n        <width>64</width>\n        <height>32</height>\n"
        "        <depth>3</depth>\n    </size>\n"
        "    <segmented>0</segmented>\n"
        "    <object>\n        <name>cat&amp;dog</name>\n"
        "        <pose>Unspecified</pose>\n        <truncated>0</truncated>\n"
        "        <difficult>0</difficult>\n"
        "        <bndbox>\n            <xmin>4.5</xmin>\n            <ymin>2</ymin>\n"
        "            <xmax>30</xmax>\n            <ymax>18.75</ymax>\n        </bndbox>\n"
        "    </object>\n</annotation>\n",
        encoding="utf-8")
    dataset = load_dataset("voc", str(images), str(xmls))
    assert list(dataset.classes) == ["cat&dog"]
    assert [os.path.basename(p) for p in dataset.image_paths] == ["x.png"]
    det = dataset.annotations[dataset.image_paths[0]]
    assert det.class_id == [0]
    assert det.box_xyxy == [(4.5, 2.0, 30.0, 18.75)]
    assert det.mask_points == [None]
    assert validate_dataset(dataset).ok is True


def test_voc_class_names_keep_first_seen_order_across_files(tmp_path):
    images = tmp_path / "images"
    xmls = tmp_path / "xml"
    images.mkdir()
    xmls.mkdir()
    for name in ("a.png", "b.png"):
        write_png(images / name, 10, 10)
    (xmls / "a.xml").write_text(
        "<annotation><filename>a.png</filename>"
        "<object><name>zebra</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
        "<xmax>1</xmax><ymax>1</ymax></bndbox></object></annotation>",
        encoding="utf-8")
    (xmls / "b.xml").write_text(
        "<annotation><filename>b.png</filename>"
        "<object><name>apple</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
        "<xmax>1</xmax><ymax>1</ymax></bndbox></object>"
        "<object><name>zebra</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
        "<xmax>1</xmax><ymax>1</ymax></bndbox></object></annotation>",
        encoding="utf-8")
    dataset = load_dataset("voc", str(images), str(xmls))
    assert list(dataset.classes) == ["zebra", "apple"]
    second = dataset.annotations[dataset.image_paths[1]]
    assert second.class_id == [1, 0]


# --------------------------------------------------------------------------
# load_dataset 的输入校验（无条件，替换掉 supervision 调用）
# --------------------------------------------------------------------------
def test_load_dataset_rejects_unsupported_input_format(monkeypatch, tmp_path):
    with pytest.raises(DatasetToolError, match="不支持的输入格式"):
        load_dataset("xyz", str(tmp_path), str(tmp_path))
    with pytest.raises(DatasetToolError):
        load_dataset("", str(tmp_path), str(tmp_path))
    with pytest.raises(DatasetToolError):
        load_dataset(None, str(tmp_path), str(tmp_path))


def test_load_dataset_rejects_missing_images_dir_without_creating_anything(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist"
    labels = tmp_path / "labels"
    labels.mkdir()
    with pytest.raises(DatasetToolError, match="图片目录不存在"):
        load_dataset("yolo", str(missing), str(labels))
    assert not missing.exists(), "加载失败时不得创建不存在的目录"


def test_load_dataset_rejects_missing_annotation_dir(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(DatasetToolError, match="标注目录不存在"):
        load_dataset("yolo", str(images), str(tmp_path / "labels"))


def test_load_dataset_rejects_missing_data_yaml(monkeypatch, tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    with pytest.raises(DatasetToolError, match="data.yaml"):
        load_dataset("yolo", str(images), str(labels))
    with pytest.raises(DatasetToolError, match="data.yaml"):
        load_dataset("yolo", str(images), str(labels), data_yaml_path=None)


def test_load_dataset_rejects_missing_coco_json(monkeypatch, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(DatasetToolError, match="COCO 标注文件不存在"):
        load_dataset("coco", str(images), str(tmp_path / "nope.json"))


# --------------------------------------------------------------------------
# output_layout（无条件）
# --------------------------------------------------------------------------
def test_output_layout_yolo(tmp_path):
    layout = output_layout("yolo", str(tmp_path))
    assert layout == {
        "images": os.path.join(str(tmp_path), "images"),
        "annotations": os.path.join(str(tmp_path), "labels"),
        "data_yaml": os.path.join(str(tmp_path), "data.yaml"),
    }


def test_output_layout_coco_uses_fixed_annotation_name(tmp_path):
    layout = output_layout("coco", str(tmp_path))
    assert COCO_ANNOTATION_NAME == "coco_detection.json"
    assert layout["annotations"] == os.path.join(str(tmp_path), COCO_ANNOTATION_NAME)
    assert layout["images"] == os.path.join(str(tmp_path), "images")


def test_output_layout_voc(tmp_path):
    layout = output_layout("voc", str(tmp_path))
    assert layout["annotations"] == os.path.join(str(tmp_path), "annotations")
    assert layout["images"] == os.path.join(str(tmp_path), "images")


def test_output_layout_labelme_shares_images_and_annotations_dir(tmp_path):
    layout = output_layout("labelme", str(tmp_path))
    assert layout["images"] == layout["annotations"] == os.path.abspath(str(tmp_path))


@pytest.mark.parametrize("fmt", ["yolo", "coco", "voc", "labelme"])
def test_output_layout_accepts_case_and_whitespace(fmt, tmp_path):
    layout = output_layout(f"  {fmt.upper()}  ", str(tmp_path))
    assert output_layout(fmt, str(tmp_path)) == layout


def test_output_layout_rejects_unsupported_format(tmp_path):
    with pytest.raises(DatasetToolError, match="不支持的输出格式"):
        output_layout("xyz", str(tmp_path))
    with pytest.raises(DatasetToolError):
        output_layout("", str(tmp_path))


# --------------------------------------------------------------------------
# ValidationReport（无条件）
# --------------------------------------------------------------------------
def test_validation_report_defaults_to_ok():
    report = ValidationReport()
    assert report.ok is True
    assert report.image_count == 0
    assert report.annotation_count == 0
    assert report.classes == []
    assert report.errors == []
    assert report.warnings == []


def test_validation_report_ok_flips_when_errors_exist():
    assert ValidationReport(errors=["boom"]).ok is False
    assert ValidationReport(warnings=["noisy"]).ok is True


def test_validation_report_as_dict_shape_and_copies():
    report = ValidationReport(image_count=2, annotation_count=3, classes=["cat"],
                              errors=["bad"], warnings=["noisy"])
    data = report.as_dict()
    assert data == {
        "ok": False,
        "image_count": 2,
        "annotation_count": 3,
        "classes": ["cat"],
        "errors": ["bad"],
        "warnings": ["noisy"],
    }
    data["classes"].append("tamper")
    data["errors"].append("tamper")
    assert report.classes == ["cat"] and report.errors == ["bad"]


def test_validation_report_defends_against_mutation_of_inputs():
    classes = ["cat"]
    report = ValidationReport(classes=classes)
    classes.append("dog")
    assert report.classes == ["cat"]


# --------------------------------------------------------------------------
# validate_dataset（无条件，用假对象）
# --------------------------------------------------------------------------
def test_validate_dataset_empty_dataset_is_error():
    report = validate_dataset(StubDataset())
    assert report.ok is False
    assert any("为空" in e for e in report.errors)
    assert report.image_count == 0
    assert report.annotation_count == 0


def test_validate_dataset_missing_attributes_treated_as_empty():
    report = validate_dataset(object())
    assert report.ok is False
    assert report.image_count == 0


def test_validate_dataset_missing_classes_is_only_a_warning():
    paths = [str(Path("/tmp/no/such.png"))]
    report = validate_dataset(StubDataset(classes=[], image_paths=paths,
                                          annotations={paths[0]: StubDetections([])}))
    assert any("没有类别定义" in w for w in report.warnings)


def test_validate_dataset_flags_missing_image_files(tmp_path):
    paths = [str(tmp_path / "images" / f"{i}.png") for i in range(3)]
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=paths,
                                          annotations={p: StubDetections([0]) for p in paths}))
    assert report.ok is False
    assert sum("图片文件缺失" in e for e in report.errors) == 3


def test_validate_dataset_summarizes_many_missing_files(tmp_path):
    paths = [str(tmp_path / "images" / f"{i}.png") for i in range(8)]
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=paths, annotations={}))
    summary = [e for e in report.errors if "另有" in e]
    detailed = [e for e in report.errors if "图片文件缺失" in e and "另有" not in e]
    assert len(detailed) == 5
    assert len(summary) == 1 and "3" in summary[0]


def test_validate_dataset_rejects_duplicate_basenames(tmp_path):
    first = tmp_path / "up" / "IMG.png"
    second = tmp_path / "down" / "img.PNG"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"x")
    second.write_bytes(b"x")
    paths = [str(first), str(second)]
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=paths,
                                          annotations={p: StubDetections([0]) for p in paths}))
    assert report.ok is False
    assert any("重复" in e for e in report.errors)


def test_validate_dataset_allows_identical_path_listed_twice(tmp_path):
    path = tmp_path / "images" / "a.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    paths = [str(path), str(path)]
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=paths,
                                          annotations={paths[0]: StubDetections([0])}))
    assert not any("重复" in e for e in report.errors)


@pytest.mark.parametrize("class_id", [-1, 2, 100])
def test_validate_dataset_rejects_class_id_out_of_range(class_id):
    path = str(Path("/tmp/existing_but_never_used.png"))
    report = validate_dataset(StubDataset(classes=["cat", "dog"], image_paths=[path],
                                          annotations={path: StubDetections([class_id])}))
    assert report.ok is False
    assert any("类别 ID" in e for e in report.errors)


def test_validate_dataset_rejects_none_class_id():
    path = "/tmp/never_exists.png"
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=[path],
                                          annotations={path: StubDetections([0, None])}))
    assert any("类别 ID" in e for e in report.errors)


def test_validate_dataset_rejects_detections_without_class_id_attribute():
    path = "/tmp/never_exists.png"
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=[path],
                                          annotations={path: StubDetections(with_class_id=False,
                                                                             size=2)}))
    assert report.ok is False
    assert any("类别 ID" in e for e in report.errors)


def test_validate_dataset_counts_annotations_and_classes():
    path = "/tmp/never_exists.png"
    report = validate_dataset(StubDataset(classes=["cat", "dog"], image_paths=[path],
                                          annotations={path: StubDetections([0, 1, 1])}))
    assert report.annotation_count == 3
    assert report.classes == ["cat", "dog"]
    assert report.image_count == 1


def test_validate_dataset_missing_annotation_is_warning_not_error(tmp_path):
    path = tmp_path / "images" / "a.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=[str(path)], annotations={}))
    assert report.ok is True
    assert any("没有对应标注" in w for w in report.warnings)


def test_validate_dataset_empty_detections_is_warning_not_error(tmp_path):
    path = tmp_path / "images" / "a.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=[str(path)],
                                          annotations={str(path): StubDetections([])}))
    assert report.ok is True
    assert report.annotation_count == 0
    assert any("没有标注框" in w for w in report.warnings)


def test_validate_dataset_truncates_warning_list():
    paths = [f"/tmp/never_exists_{i}.png" for i in range(25)]
    report = validate_dataset(StubDataset(classes=["cat"], image_paths=paths, annotations={}))
    assert len(report.warnings) == 21
    assert "省略" in report.warnings[-1]


def test_validate_dataset_ok_on_clean_dataset(tmp_path):
    dataset = _make_stub_dataset(tmp_path)
    report = validate_dataset(dataset)
    assert report.ok is True
    assert report.errors == []
    assert report.image_count == 3
    assert report.annotation_count == 3
    assert report.classes == ["cat", "dog"]


# --------------------------------------------------------------------------
# convert_dataset 的离线路径：输出目录保护 / 进度 / 取消（无条件）
# --------------------------------------------------------------------------
def test_convert_dataset_rejects_unsupported_output_format(monkeypatch, tmp_path):
    dataset = _make_stub_dataset(tmp_path)
    with pytest.raises(DatasetToolError, match="不支持的输出格式"):
        convert_dataset(dataset, "xyz", str(tmp_path / "out"))


def test_convert_dataset_rejects_invalid_dataset_without_touching_output(
        monkeypatch, tmp_path, offline_convert):
    out = tmp_path / "should_not_exist"
    with pytest.raises(DatasetToolError, match="校验未通过"):
        convert_dataset(StubDataset(), "yolo", str(out))
    assert not out.exists(), "校验失败时不得创建输出目录"


@pytest.mark.parametrize("output_format", ["yolo", "voc", "labelme"])
def test_convert_dataset_refuses_nonempty_output_dir(
        tmp_path, offline_convert, output_format):
    dataset = _make_stub_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    victim = out / "precious.txt"
    victim.write_text("precious", encoding="utf-8")
    with pytest.raises(DatasetToolError, match="输出目录不为空"):
        convert_dataset(dataset, output_format, str(out))
    assert victim.read_text(encoding="utf-8") == "precious", "原目录内容必须保持不动"
    assert out.is_dir()
    assert offline_convert == [], "未通过校验前不得写出任何内容"


def test_convert_dataset_rejects_missing_output_dir(monkeypatch, tmp_path):
    dataset = _make_stub_dataset(tmp_path)
    with pytest.raises(DatasetToolError, match="未指定输出目录"):
        convert_dataset(dataset, "yolo", "")
    with pytest.raises(DatasetToolError):
        convert_dataset(dataset, "yolo", None)


def test_convert_dataset_rejects_output_path_that_is_a_file(monkeypatch, tmp_path):
    dataset = _make_stub_dataset(tmp_path)
    file_path = tmp_path / "afile"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(DatasetToolError, match="不是目录"):
        convert_dataset(dataset, "yolo", str(file_path))
    assert file_path.is_file()


def test_convert_dataset_overwrite_true_cleans_dir_and_writes(tmp_path, offline_convert):
    dataset = _make_stub_dataset(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    victim = out / "precious.txt"
    victim.write_text("precious", encoding="utf-8")
    result = convert_dataset(dataset, "yolo", str(out), overwrite=True)
    assert not victim.exists()
    assert out.is_dir()
    assert result["cancelled"] is False
    assert result["format"] == "yolo"
    assert result["image_count"] == 3
    assert result["annotation_count"] == 3
    assert result["output_dir"] == os.path.abspath(str(out))
    assert result["layout"] == output_layout("yolo", str(out))


def test_convert_dataset_reports_progress_done_equals_total(tmp_path, monkeypatch, offline_convert):
    monkeypatch.setattr(ST, "_CHUNK_SIZE", 1)
    dataset = _make_stub_dataset(tmp_path, images=("a.png", "b.png", "c.png"))
    events = []
    convert_dataset(dataset, "voc", str(tmp_path / "out"),
                    on_progress=lambda done, total: events.append((done, total)))
    assert events == [(1, 3), (2, 3), (3, 3)]
    assert events[-1][0] == events[-1][1]


def test_convert_dataset_progress_reflects_chunking(tmp_path, monkeypatch, offline_convert):
    monkeypatch.setattr(ST, "_CHUNK_SIZE", 2)
    dataset = _make_stub_dataset(tmp_path, images=("a.png", "b.png", "c.png"))
    convert_dataset(dataset, "voc", str(tmp_path / "out"))
    assert [call["chunk"] for call in offline_convert] == [2, 1]
    assert [call["first"] for call in offline_convert] == [True, False]


def test_convert_dataset_cancelled_returns_flag_and_removes_output(tmp_path, offline_convert):
    dataset = _make_stub_dataset(tmp_path)
    out = tmp_path / "cancelled"
    result = convert_dataset(dataset, "yolo", str(out), is_interrupted=lambda: True)
    assert result["cancelled"] is True
    assert result["image_count"] == 0
    assert result["annotation_count"] == 0
    assert not out.exists(), "取消后不得残留半成品目录"
    assert offline_convert == [], "取消前不得写出任何内容"


def test_convert_dataset_cancel_after_partial_chunk_removes_everything(tmp_path, monkeypatch,
                                                                       offline_convert):
    monkeypatch.setattr(ST, "_CHUNK_SIZE", 1)
    dataset = _make_stub_dataset(tmp_path, images=("a.png", "b.png", "c.png"))
    out = tmp_path / "partial"
    interrupted = {"n": 0}

    def maybe_interrupted():
        interrupted["n"] += 1
        return interrupted["n"] > 1

    result = convert_dataset(dataset, "yolo", str(out), is_interrupted=maybe_interrupted)
    assert result["cancelled"] is True
    assert not out.exists()
    assert len(offline_convert) == 1, "只写出了一个分块"


# --------------------------------------------------------------------------
# 真实转换测试（自研实现，无外部依赖，全部环境真实执行）
# --------------------------------------------------------------------------
@pytest.fixture
def yolo_dataset(tmp_path):
    """最小 YOLO 数据集：3 张图片、2 个类别、1 张空标注图片。"""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        write_png(images_dir / name, 64, 64)
    (labels_dir / "a.txt").write_text(
        "0 0.30 0.30 0.20 0.20\n1 0.70 0.70 0.15 0.15\n", encoding="utf-8")
    (labels_dir / "b.txt").write_text("1 0.50 0.50 0.40 0.40\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: ['cat', 'dog']\n", encoding="utf-8")
    return SimpleNamespace(images_dir=str(images_dir), labels_dir=str(labels_dir),
                           data_yaml=str(data_yaml), images=images_dir, labels=labels_dir)


def test_implementation_has_no_external_dependency():
    """转换实现必须自研：不得 import supervision / scipy / yaml。"""
    source = Path(ST.__file__).read_text(encoding="utf-8")
    for forbidden in ("import supervision", "from supervision",
                      "import scipy", "from scipy",
                      "import yaml", "from yaml", "import cv2", "import av"):
        assert forbidden not in source, forbidden


def test_yolo_dataset_fixture_is_readable(yolo_dataset):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    assert list(dataset.classes) == ["cat", "dog"]
    assert len(dataset.image_paths) == 3


def _load_output(fmt, layout):
    if fmt == "yolo":
        return load_dataset("yolo", layout["images"], layout["annotations"],
                            layout["data_yaml"])
    if fmt == "coco":
        return load_dataset("coco", layout["images"], layout["annotations"])
    if fmt == "voc":
        return load_dataset("voc", layout["images"], layout["annotations"])
    return load_dataset("labelme", layout["images"], layout["annotations"])


@SV
def test_load_dataset_yolo_reads_images_classes_and_boxes(yolo_dataset):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    assert list(dataset.classes) == ["cat", "dog"]
    assert len(dataset.image_paths) == 3
    assert sorted(os.path.basename(p) for p in dataset.image_paths) == ["a.png", "b.png", "c.png"]
    assert all(p in dataset.annotations for p in dataset.image_paths)
    report = validate_dataset(dataset)
    assert report.ok is True
    assert report.annotation_count == 3
    assert report.image_count == 3


@SV
def test_convert_paths_end_to_end(tmp_path, yolo_dataset):
    out = str(tmp_path / "converted")
    result = convert_paths("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           "yolo", out, data_yaml_path=yolo_dataset.data_yaml)
    assert result["cancelled"] is False
    assert result["image_count"] == 3
    assert result["annotation_count"] == 3
    assert result["format"] == "yolo"
    assert result["report"]["ok"] is True
    assert len(list(Path(result["layout"]["images"]).iterdir())) == 3
    assert Path(result["layout"]["data_yaml"]).is_file()
    assert len(list(Path(result["layout"]["annotations"]).glob("*.txt"))) == 3


@SV
def test_roundtrip_yolo_coco_voc_labelme_yolo_keeps_counts(tmp_path, yolo_dataset):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    baseline = validate_dataset(dataset)
    assert baseline.ok is True
    assert baseline.image_count == 3
    assert baseline.annotation_count == 3
    assert list(baseline.classes) == ["cat", "dog"]

    current = dataset
    for fmt in ("coco", "voc", "labelme", "yolo"):
        out = str(tmp_path / f"out_{fmt}")
        result = convert_dataset(current, fmt, out)
        assert result["cancelled"] is False
        assert result["image_count"] == baseline.image_count
        assert result["annotation_count"] == baseline.annotation_count
        assert result["format"] == fmt
        layout = output_layout(fmt, out)
        image_files = [p for p in Path(layout["images"]).glob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        assert len(image_files) == baseline.image_count, fmt
        next_dataset = _load_output(fmt, layout)
        report = validate_dataset(next_dataset)
        assert report.ok is True, f"{fmt} 回读校验失败：{report.errors}"
        assert report.image_count == baseline.image_count, fmt
        assert report.annotation_count == baseline.annotation_count, fmt
        assert list(report.classes) == list(baseline.classes), fmt
        current = next_dataset


@SV
def test_coco_output_uses_fixed_json_name_and_schema(tmp_path, yolo_dataset):
    out = str(tmp_path / "out_coco")
    convert_dataset(load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                                 yolo_dataset.data_yaml), "coco", out)
    layout = output_layout("coco", out)
    assert os.path.basename(layout["annotations"]) == "coco_detection.json"
    payload = json.loads(Path(layout["annotations"]).read_text(encoding="utf-8"))
    assert len(payload["images"]) == 3
    assert len(payload["annotations"]) == 3
    assert [c["name"] for c in payload["categories"]] == ["cat", "dog"]


@SV
def test_labelme_output_shapes_are_rectangles(tmp_path, yolo_dataset):
    out = tmp_path / "out_labelme"
    convert_dataset(load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                                 yolo_dataset.data_yaml), "labelme", str(out))
    files = {p.name: json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(out.glob("*.json"))}
    assert sorted(files) == ["a.json", "b.json", "c.json"]
    for name, payload in files.items():
        assert "shapes" in payload, name
        assert payload["imageHeight"] == 64
        assert payload["imageWidth"] == 64
        assert os.path.basename(payload["imagePath"]) == name.replace(".json", ".png")
        for shape in payload["shapes"]:
            assert shape["shape_type"] == "rectangle", name
            assert len(shape["points"]) == 2
            assert shape["label"] in ("cat", "dog")
    assert files["a.json"]["shapes"] and files["b.json"]["shapes"]
    assert files["c.json"]["shapes"] == []


@SV
def test_labelme_output_keeps_polygon_shapes(tmp_path):
    """含分割标注的数据集转 LabelMe 时不能静默降级成矩形框。"""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    write_png(images_dir / "a.png", 64, 64)
    (labels_dir / "a.txt").write_text(
        "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names: ['cat']\n", encoding="utf-8")
    dataset = load_dataset("yolo", str(images_dir), str(labels_dir), str(data_yaml))
    assert dataset.annotations[dataset.image_paths[0]].mask is not None

    out = tmp_path / "out"
    convert_dataset(dataset, "labelme", str(out))
    payload = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert payload["shapes"][0]["shape_type"] == "polygon"
    assert len(payload["shapes"][0]["points"]) == 4


@SV
def test_convert_dataset_reports_progress_to_end(tmp_path, yolo_dataset):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    events = []
    convert_dataset(dataset, "voc", str(tmp_path / "out"),
                    on_progress=lambda done, total: events.append((done, total)))
    assert events, "on_progress 必须被调用"
    assert events[-1][0] == events[-1][1] == 3


@SV
@pytest.mark.parametrize("output_format", ["yolo", "voc", "labelme", "coco"])
def test_cancel_removes_output_dir(tmp_path, yolo_dataset, output_format):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    out = tmp_path / f"cancelled_{output_format}"
    result = convert_dataset(dataset, output_format, str(out),
                             is_interrupted=lambda: True)
    assert result["cancelled"] is True
    assert result["image_count"] == 0
    assert not out.exists(), f"{output_format} 取消后不应残留目录"


@SV
def test_cancel_mid_conversion_removes_partial_output(tmp_path, yolo_dataset, monkeypatch):
    monkeypatch.setattr(ST, "_CHUNK_SIZE", 1)
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    out = tmp_path / "partial"
    state = {"calls": 0}

    def interrupted():
        state["calls"] += 1
        return state["calls"] > 1

    result = convert_dataset(dataset, "yolo", str(out),
                             is_interrupted=interrupted, on_progress=lambda d, t: None)
    assert result["cancelled"] is True
    assert state["calls"] > 1
    assert not out.exists()


@SV
def test_convert_dataset_overwrite_true_replaces_existing_output(tmp_path, yolo_dataset):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    out = tmp_path / "out"
    out.mkdir()
    victim = out / "stale.txt"
    victim.write_text("stale", encoding="utf-8")
    with pytest.raises(DatasetToolError, match="输出目录不为空"):
        convert_dataset(dataset, "yolo", str(out))
    assert victim.read_text(encoding="utf-8") == "stale"

    result = convert_dataset(dataset, "yolo", str(out), overwrite=True)
    assert result["cancelled"] is False
    assert not victim.exists()
    assert (out / "data.yaml").is_file()
    assert len(list((out / "labels").glob("*.txt"))) == 3
    assert len(list((out / "images").glob("*"))) == 3


@SV
def test_convert_dataset_raises_and_cleans_on_write_failure(tmp_path, yolo_dataset, monkeypatch):
    dataset = load_dataset("yolo", yolo_dataset.images_dir, yolo_dataset.labels_dir,
                           yolo_dataset.data_yaml)
    out = tmp_path / "out"

    def boom(chunk, output_format, layout, first, task="det"):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(ST, "_write_chunk", boom)
    with pytest.raises(DatasetToolError, match="写出 yolo 数据集失败"):
        convert_dataset(dataset, "yolo", str(out))
    assert not out.exists(), "写出失败后不得残留半成品目录"


# --------------------------------------------------------------------------
# OBB / pose / seg 类型保留（LabelMe <-> YOLO）
# --------------------------------------------------------------------------

def _labelme_dir_with_shapes(tmp_path, shapes):
    images = tmp_path / "img"
    images.mkdir()
    write_png(images / "a.png", 100, 100)
    payload = {
        "version": "5.3.1", "flags": {}, "shapes": shapes,
        "imagePath": "a.png", "imageData": None,
        "imageHeight": 100, "imageWidth": 100,
    }
    (images / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(images)


def test_labelme_preserves_rotation_roundtrip(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [{
        "label": "car", "shape_type": "rotation",
        "points": [[10, 10], [40, 10], [40, 30], [10, 30]],
        "group_id": None, "flags": {},
    }])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "labelme", str(out),
                     overwrite=True, task="obb")
    data = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["shape_type"] == "rotation"
    assert len(data["shapes"][0]["points"]) == 4


def test_labelme_labelme_keeps_paste_flags(tmp_path):
    """贴图靠 flags.paste 区分，LabelMe->LabelMe 转换不能把它丢掉。"""
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "logo", "shape_type": "rectangle",
         "points": [[10, 10], [40, 30]],
         "group_id": None, "flags": {"paste": True}},
        {"label": "car", "shape_type": "rectangle",
         "points": [[50, 50], [80, 80]],
         "group_id": None, "flags": {}},
    ])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "labelme", str(out),
                     overwrite=True, task="auto")
    shapes = json.loads((out / "a.json").read_text(encoding="utf-8"))["shapes"]
    by_label = {s["label"]: s for s in shapes}
    assert by_label["logo"]["flags"] == {"paste": True}
    assert by_label["car"]["flags"] == {}


def test_labelme_labelme_keeps_custom_audit_flags(tmp_path):
    """质检豁免标记 flags.tiny_keep 不是 paste，转换同样不能丢。

    转换器读的是整个 flags 字典（不是只挑 paste），所以这条本来就成立；
    钉住它，避免以后有人把 flags 处理收窄成只认 paste。
    """
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "wire", "shape_type": "rectangle",
         "points": [[10, 10], [400, 12]],
         "group_id": None, "flags": {"tiny_keep": True}},
        {"label": "car", "shape_type": "rectangle",
         "points": [[50, 50], [80, 80]],
         "group_id": None, "flags": {"paste": True, "tiny_keep": True}},
    ])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "labelme", str(out),
                     overwrite=True, task="auto")
    shapes = json.loads((out / "a.json").read_text(encoding="utf-8"))["shapes"]
    by_label = {s["label"]: s for s in shapes}
    assert by_label["wire"]["flags"] == {"tiny_keep": True}
    assert by_label["car"]["flags"] == {"paste": True, "tiny_keep": True}


def test_labelme_rotation_to_yolo_obb(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [{
        "label": "car", "shape_type": "rotation",
        "points": [[10, 10], [40, 10], [40, 30], [10, 30]],
        "group_id": None, "flags": {},
    }])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "yolo", str(out),
                     overwrite=True, task="obb")
    txt = (out / "labels" / "a.txt").read_text(encoding="utf-8").strip()
    parts = txt.split()
    assert parts[0] == "0"
    assert len(parts) == 9  # cls + 8 coords


def test_yolo_obb_back_to_labelme_rotation(tmp_path):
    root = tmp_path / "yolo"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    write_png(root / "images" / "a.png", 100, 100)
    (root / "labels" / "a.txt").write_text(
        "0 0.1 0.1 0.4 0.1 0.4 0.3 0.1 0.3\n", encoding="utf-8")
    (root / "data.yaml").write_text("names: ['car']\n", encoding="utf-8")
    out = tmp_path / "out"
    ST.convert_paths("yolo", str(root / "images"), str(root / "labels"),
                     "labelme", str(out), data_yaml_path=str(root / "data.yaml"),
                     overwrite=True, task="obb")
    data = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["shape_type"] == "rotation"
    assert len(data["shapes"][0]["points"]) == 4


def test_labelme_polygon_to_yolo_seg(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [{
        "label": "leaf", "shape_type": "polygon",
        "points": [[10, 10], [40, 10], [25, 40]],
        "group_id": None, "flags": {},
    }])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "yolo", str(out),
                     overwrite=True, task="seg")
    txt = (out / "labels" / "a.txt").read_text(encoding="utf-8").strip()
    parts = txt.split()
    assert parts[0] == "0"
    assert len(parts) == 7  # cls + 3 points * 2


def test_labelme_pose_to_yolo_pose(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "person", "shape_type": "rectangle",
         "points": [[10, 10], [50, 60]], "group_id": 0, "flags": {}},
        {"label": "person", "shape_type": "point",
         "points": [[20, 20]], "group_id": 0, "flags": {}},
        {"label": "person", "shape_type": "point",
         "points": [[30, 40]], "group_id": 0, "flags": {}},
    ])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "yolo", str(out),
                     overwrite=True, task="pose")
    txt = (out / "labels" / "a.txt").read_text(encoding="utf-8").strip()
    parts = txt.split()
    assert parts[0] == "0"
    # cls + cx cy w h + 2 keypoints * 3
    assert len(parts) == 5 + 2 * 3


def test_labelme_point_preserved_roundtrip(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [{
        "label": "kp", "shape_type": "point",
        "points": [[25, 25]], "group_id": 1, "flags": {},
    }])
    out = tmp_path / "out"
    ST.convert_paths("labelme", src, src, "labelme", str(out),
                     overwrite=True, task="pose")
    data = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert data["shapes"][0]["shape_type"] == "point"
    assert data["shapes"][0]["group_id"] == 1


def test_validate_warns_on_mixed_types_for_obb_task(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "car", "shape_type": "rotation",
         "points": [[10, 10], [40, 10], [40, 30], [10, 30]], "group_id": None, "flags": {}},
        {"label": "car", "shape_type": "rectangle",
         "points": [[5, 5], [20, 20]], "group_id": None, "flags": {}},
    ])
    dataset = ST.load_dataset("labelme", src, src, task="obb")
    report = ST.validate_dataset(dataset, task="obb")
    assert report.ok  # 仍可转换（有 rotation）
    assert any("跳过" in w for w in report.warnings)


def test_validate_errors_when_task_has_no_matching_shape(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "car", "shape_type": "rectangle",
         "points": [[5, 5], [20, 20]], "group_id": None, "flags": {}},
    ])
    dataset = ST.load_dataset("labelme", src, src, task="obb")
    report = ST.validate_dataset(dataset, task="obb")
    assert not report.ok
    assert any("没有可导出" in e for e in report.errors)


def test_validate_det_task_keeps_all_shapes(tmp_path):
    src = _labelme_dir_with_shapes(tmp_path, [
        {"label": "car", "shape_type": "rotation",
         "points": [[10, 10], [40, 10], [40, 30], [10, 30]], "group_id": None, "flags": {}},
        {"label": "car", "shape_type": "polygon",
         "points": [[5, 5], [20, 5], [12, 20]], "group_id": None, "flags": {}},
    ])
    dataset = ST.load_dataset("labelme", src, src, task="det")
    report = ST.validate_dataset(dataset, task="det")
    assert report.ok
    assert not any("跳过" in w for w in report.warnings)
