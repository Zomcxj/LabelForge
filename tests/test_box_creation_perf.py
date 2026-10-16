"""回归测试：新建检测框不得触发全数据集重扫描。

历史问题：`_create_detection_box` 每次调用都会执行
`_seed_stats_cache_from_disk_and_memory()`，该函数会读取整个数据集的
sidecar JSON（O(n) 磁盘 IO）。在数千张图片的数据集上，每画一个框都要
卡顿数百毫秒到 1 秒以上。

修复后：改为增量更新统计缓存（`_bump_cached_stats_for_box`），只做
内存操作。
"""
from labelforge.canvas.canvas_drawing import CanvasDrawingMixin


class Rect:
    def __init__(self, x=0, y=0, w=100, h=100):
        self._x, self._y, self._w, self._h = x, y, w, h

    def left(self):
        return self._x

    def top(self):
        return self._y

    def contains(self, point):
        return True


class Background:
    def width(self):
        return 400

    def height(self):
        return 300


class Editor:
    def __init__(self):
        self._is_delete_view = False
        self.background_images = ["a.png"]
        self.current_background = Background()
        self.current_background_index = 0
        self.detection_boxes = []
        self.detection_boxes_dict = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_color_map = {}
        self.label_manager = None

    def save_undo_state(self):
        pass

    def update_label_list(self):
        pass

    def get_label_color(self, _label):
        return "#00FF00"


class RecordingLabelManager:
    """记录是否触发了全量重扫。"""

    def __init__(self):
        self.full_seed_calls = 0
        self.bump_calls = 0
        self.stats = []

    def _seed_stats_cache_from_disk_and_memory(self):
        self.full_seed_calls += 1

    def _bump_cached_stats_for_box(self, box):
        self.bump_calls += 1


class Canvas(CanvasDrawingMixin):
    def __init__(self):
        self._editor = Editor()
        self.background_scale = 1
        self.updated = 0

    def get_background_rect(self):
        return Rect()

    def _sync_all_detection_boxes_to_dict(self):
        idx = self._editor.current_background_index
        self._editor.detection_boxes_dict[idx] = self._editor.detection_boxes.copy()

    def _save_current_detection_boxes(self):
        pass

    def update(self):
        self.updated += 1


def test_create_box_does_not_trigger_full_dataset_rescan():
    canvas = Canvas()
    lm = RecordingLabelManager()
    canvas._editor.label_manager = lm

    canvas._create_detection_box(10, 10, 50, 40, "brand_new_label")

    assert lm.full_seed_calls == 0, "建框不应触发全数据集重扫描"
    assert lm.bump_calls == 1


def test_create_box_falls_back_to_color_without_bump_api():
    """label_manager 没有增量 API 时，走 get_label_color，且不触发全量重扫。"""
    canvas = Canvas()

    class LegacyLabelManager:
        def __init__(self):
            self.seed_calls = 0

        def _seed_stats_cache_from_disk_and_memory(self):
            self.seed_calls += 1

    color_calls = []

    def _get_label_color(label):
        color_calls.append(label)
        return "#00FF00"

    lm = LegacyLabelManager()
    canvas._editor.label_manager = lm
    canvas._editor.get_label_color = _get_label_color

    canvas._create_detection_box(10, 10, 50, 40, "x")

    assert lm.seed_calls == 0, "旧 label_manager 也不应触发全量重扫描"
    assert color_calls == ["x"]
