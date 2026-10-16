"""切图时检测框缓存的回归测试。

用户报的 ③：一键推理跑完「完全不更新结果」，7–8 次只有 1 次生效。
真凶是 `_load_detection_boxes_for_index()` 的缓存判据用了 `len(...) > 0`，
于是**空结果（0 框）不算有效缓存**，切到那张图时重读磁盘旧 json，并把
`detection_boxes_dict[index]` 覆盖成磁盘内容——覆盖模式刚清空的框又回来了。
"""
import json
import os

from labelforge.engine.image_loader import ImageLoaderMixin


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


def test_progressive_loader_does_not_poison_the_box_cache(tmp_path, monkeypatch):
    """流式加载**不许**预置 `detection_boxes_dict[idx] = []`。

    用户报的「加载数据集把标签吃掉」：一个大文件夹加载后，几乎所有图的绿点
    变成橙点再变成空心。真凶是流式加载（大文件夹走这条路径）每加一行就写
    `detection_boxes_dict[idx] = []`——缓存判据是「下标在不在字典里」，空列表
    也是有效缓存，于是等于告诉 `_load_detection_boxes_for_index()`「这张已经
    加载过、结果就是 0 框」，磁盘上的框**永远不会被读进来**。

    批量加载器与首次加载器都已有注释明确不许这么做，只有流式加载漏了。
    这里调用**真实的** `_load_next_background_batch()`，只把「装饰列表行」和
    Qt 列表控件换掉，所以源码真被改回预置空缓存时本测试会红。
    """
    import labelforge.engine.image_loader.mixin as mixin_mod

    img0 = _sidecar(tmp_path, "p0.jpg", ["A", "B"])
    img1 = _sidecar(tmp_path, "p1.jpg", ["C"])

    fake_list = _FakeList()
    loader = _Progressive([img0, img1], fake_list)
    monkeypatch.setattr(mixin_mod, "QListWidgetItem", _FakeItem)
    monkeypatch.setattr(mixin_mod, "decorate_background_list_item",
                        lambda item, path, idx, light=True: None)

    # 真实方法：一次 tick 的时间预算足够跑完两张
    loader._load_next_background_batch()

    assert loader._pending_image_index == 2, "两张图都应被加入"
    assert len(fake_list.items) == 2
    assert loader.detection_boxes_dict == {}, (
        "流式加载预置了空缓存，会把「还没加载」伪装成「已加载且为空」："
        f"{ {k: len(v) for k, v in loader.detection_boxes_dict.items()} }")

    # 切到第 1 张：磁盘上有 1 个框，必须读进来
    loader._load_detection_boxes_for_index(1, img1)
    assert [b["label"] for b in loader.detection_boxes] == ["C"], (
        "流式加载后切图读不到磁盘上的框——正是用户报的「标签被吃掉」")


class _Progressive(_Loader):
    """带流式加载所需状态的替身；只补 Qt 相关的那几个依赖。"""

    def __init__(self, files, fake_list):
        super().__init__()
        self.canvas_items_dict = {}
        self.background_images = []
        self._pending_image_files = list(files)
        self._pending_image_index = 0
        self._bg_annotation_filter = 'all'
        self.background_list = fake_list
        self._display_path_for = lambda p: os.path.basename(p)
        self.update_file_count = lambda: None
        # 循环收尾分支会碰到的几个钩子，测试里只求不炸
        self._background_load_timer = _FakeTimer()
        self._start_dataset_label_scan = lambda: None
        self._finalize_dataset_load = lambda: None


class _FakeTimer:
    def stop(self):
        pass


class _FakeItem:
    """Qt 列表行替身：`_load_next_background_batch()` 只用到这几个方法。"""

    def __init__(self, text=""):
        self._text = text

    def data(self, role):
        return None

    def setHidden(self, flag):
        pass

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text


class _FakeList:
    def __init__(self):
        self.items = []

    def addItem(self, item):
        self.items.append(item)

    def setUpdatesEnabled(self, flag):
        pass

    def viewport(self):
        return self

    def update(self):
        pass

    def count(self):
        return len(self.items)

