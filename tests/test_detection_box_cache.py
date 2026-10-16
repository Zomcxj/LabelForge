"""切图时检测框缓存的回归测试。

用户报的 ③：一键推理跑完「完全不更新结果」，7–8 次只有 1 次生效。
真凶是 `_load_detection_boxes_for_index()` 的缓存判据用了 `len(...) > 0`，
于是**空结果（0 框）不算有效缓存**，切到那张图时重读磁盘旧 json，并把
`detection_boxes_dict[index]` 覆盖成磁盘内容——覆盖模式刚清空的框又回来了。
"""
import json
import os

from pastelabel.engine.image_loader import ImageLoaderMixin


class _Loader(ImageLoaderMixin):
    """只带缓存字典的替身；不碰 Qt。"""

    def __init__(self):
        self.detection_boxes_dict = {}
        self.detection_boxes = []


def _sidecar(tmp_path, name, labels):
    img = tmp_path / name
    img.write_bytes(b"")
    shapes = [
        {"label": lbl, "points": [[i, i], [i + 4, i + 4]],
         "shape_type": "rectangle", "flags": {"inference": True}}
        for i, lbl in enumerate(labels)
    ]
    (tmp_path / (os.path.splitext(name)[0] + ".json")).write_text(
        json.dumps({"shapes": shapes}), encoding="utf-8")
    return str(img)


def test_empty_cache_is_a_valid_cache_and_is_not_reread_from_disk(tmp_path):
    """空列表是**有效缓存**：覆盖模式刚清空的图，切回来不能又冒出旧框。

    这是 ③ 的核心：`len(...) > 0` 把「已加载且结果为空」与「还没加载过」
    混为一谈，于是磁盘上的旧 json 被重读并写回缓存。
    """
    img = _sidecar(tmp_path, "a.jpg", ["OLD"])
    loader = _Loader()
    # 覆盖推理把这图清成 0 框（新语义下正常发生）
    loader.detection_boxes_dict[0] = []

    loader._load_detection_boxes_for_index(0, img)

    assert loader.detection_boxes == [], (
        f"空缓存被当成「没缓存」，又读回磁盘旧框："
        f"{[b['label'] for b in loader.detection_boxes]}")
    assert loader.detection_boxes_dict[0] == [], (
        "缓存被磁盘内容覆盖回去了")


def test_non_empty_cache_still_wins_over_disk(tmp_path):
    """非空缓存仍然优先于磁盘（原有行为不能被改坏）。"""
    img = _sidecar(tmp_path, "b.jpg", ["DISK"])
    loader = _Loader()
    loader.detection_boxes_dict[0] = [
        {"label": "MEM", "x": 1, "y": 1, "width": 2, "height": 2}]

    loader._load_detection_boxes_for_index(0, img)

    assert [b["label"] for b in loader.detection_boxes] == ["MEM"]


def test_index_absent_from_cache_reads_disk(tmp_path):
    """**没出现在缓存里**才是「没加载过」，这时读盘。"""
    img = _sidecar(tmp_path, "c.jpg", ["DISK"])
    loader = _Loader()

    loader._load_detection_boxes_for_index(0, img)

    assert [b["label"] for b in loader.detection_boxes] == ["DISK"]
    assert loader.detection_boxes_dict[0], "读盘后应当写进缓存"


def test_empty_cache_does_not_resurrect_a_deleted_sidecar(tmp_path):
    """磁盘上 json 已被删（无检出时删空文件）时，空缓存也不该读回任何东西。"""
    img = tmp_path / "d.jpg"
    img.write_bytes(b"")
    loader = _Loader()
    loader.detection_boxes_dict[0] = []

    loader._load_detection_boxes_for_index(0, str(img))

    assert loader.detection_boxes == []
