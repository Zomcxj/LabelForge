"""后台背景标签扫描回归测试。"""
import json
import pytest

from labelforge.engine import image_loader
from labelforge.engine.image_loader import ImageLoaderMixin, scan_dataset_labels
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.processing_panel import ProcessingPanel


def test_collect_background_label_counts_counts_each_shape(tmp_path):
    """Stats must count every shape, not unique labels per file."""
    from labelforge.engine.image_loader import collect_background_label_counts

    img = tmp_path / "a.png"
    img.touch()
    (tmp_path / "a.json").write_text(json.dumps({
        "shapes": [
            {"label": "cat"}, {"label": "cat"}, {"label": "dog"},
        ],
    }), encoding="utf-8")

    counts = collect_background_label_counts([str(img)])
    assert counts == {"cat": 2, "dog": 1}


def test_collect_bg_stats_for_dialog_uses_nonempty_cache_without_path():
    """Live cache updates must show in stats even if path gate fails."""
    editor = type("E", (), {})()
    editor._cached_bg_label_stats = [
        {"label": "kitty", "count": 7, "color": "#abc"},
        {"label": "dog", "count": 1, "color": "#def"},
    ]
    editor._cached_bg_label_stats_path = ""
    editor._memory_background_path = r"D:\dataset"
    editor.background_images = []
    editor.global_labels = set()
    editor.background_dataset_labels = set()
    editor.label_color_map = {}
    editor.get_label_color = lambda label: "#000"

    result = ImageEditor._collect_bg_stats_for_dialog(editor)
    assert result == {"kitty": 7, "dog": 1}


def test_scan_dataset_labels_collects_only_non_empty_string_labels(tmp_path):
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.jpg"
    invalid_image = tmp_path / "invalid.bmp"
    missing_image = tmp_path / "missing.gif"
    for image in (first_image, second_image, invalid_image, missing_image):
        image.touch()

    (tmp_path / "first.json").write_text(json.dumps({"shapes": [
        {"label": "cat"}, {"label": ""}, {"label": 3}, {},
    ]}), encoding="utf-8")
    (tmp_path / "second.json").write_text(json.dumps({"shapes": [
        {"label": "dog"}, {"label": "cat"}, {"label": "  "},
    ]}), encoding="utf-8")
    (tmp_path / "invalid.json").write_text("not json", encoding="utf-8")

    assert scan_dataset_labels((str(first_image), str(second_image), str(invalid_image), str(missing_image))) == {"cat", "dog"}


def test_apply_dataset_labels_ignores_stale_generation_and_path_snapshot():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 2
    editor.background_images = ["current.png"]
    editor.global_labels = {"old"}
    editor.background_dataset_labels = {"old"}
    editor.update_calls = 0
    editor.panel_calls = 0
    editor.update_label_list = lambda: setattr(editor, "update_calls", editor.update_calls + 1)
    editor._processing_panel = type("Panel", (), {"isVisible": lambda self: True})()
    editor._update_processing_panel_labels = lambda: setattr(editor, "panel_calls", editor.panel_calls + 1)

    editor._apply_dataset_labels(1, ("current.png",), {"stale"})
    editor._apply_dataset_labels(2, ("other.png",), {"wrong"})

    assert editor.global_labels == {"old"}
    assert editor.background_dataset_labels == {"old"}
    assert editor.update_calls == 0
    assert editor.panel_calls == 0


def test_start_background_replacement_invalidates_scans_and_clears_labels():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 2
    editor.global_labels = set()
    editor.background_dataset_labels = {"old"}
    editor._background_label_scan_completed = True

    editor._start_background_replacement()

    assert editor._background_label_scan_generation == 3
    assert editor.global_labels == set()
    assert editor.background_dataset_labels == set()
    assert editor._background_label_scan_completed is False


def test_start_background_replacement_refreshes_visible_processing_panel():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 2
    editor.global_labels = {"old"}
    editor.background_dataset_labels = {"old"}
    editor._background_label_scan_completed = True
    editor.panel_calls = 0
    editor._processing_panel = type("Panel", (), {"isVisible": lambda self: True})()
    editor._update_processing_panel_labels = lambda: setattr(
        editor, "panel_calls", editor.panel_calls + 1
    )

    editor._start_background_replacement()

    assert editor.global_labels == set()
    assert editor.background_dataset_labels == set()
    assert editor.panel_calls == 1


def test_start_background_replacement_interrupts_running_worker_without_waiting():
    class Worker:
        def __init__(self):
            self.interrupted = False
            self.wait_calls = []

        def requestInterruption(self):
            self.interrupted = True

        def isRunning(self):
            return True

        def wait(self, timeout):
            self.wait_calls.append(timeout)
            return False

    editor = type("Editor", (ImageLoaderMixin,), {})()
    worker = Worker()
    editor._background_label_scan_generation = 2
    editor.global_labels = {"old"}
    editor.background_dataset_labels = {"old"}
    editor._background_label_scan_worker = worker
    editor._background_label_scan_workers = {worker}
    editor._background_label_scan_pending = True

    editor._start_background_replacement()

    assert worker.interrupted
    assert worker.wait_calls == []
    assert editor._background_label_scan_worker is None
    assert editor._background_label_scan_workers == {worker}
    assert editor._background_label_scan_pending is True
    assert editor._background_label_scan_generation == 3
    assert editor.global_labels == set()
    assert editor.background_dataset_labels == set()


def test_apply_dataset_labels_updates_current_dataset_and_visible_panel_only():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 3
    editor.background_images = ["current.png"]
    editor.global_labels = set()
    editor.background_dataset_labels = set()
    editor.detection_boxes_dict = {0: [{"label": "untouched"}]}
    editor.update_calls = 0
    editor.panel_calls = 0
    editor.update_label_list = lambda: setattr(editor, "update_calls", editor.update_calls + 1)
    editor._processing_panel = type("Panel", (), {"isVisible": lambda self: True})()
    editor._update_processing_panel_labels = lambda: setattr(editor, "panel_calls", editor.panel_calls + 1)

    editor._apply_dataset_labels(3, ("current.png",), {"cat", "dog"})

    assert editor.global_labels == {"cat", "dog"}
    assert editor.background_dataset_labels == {"cat", "dog"}
    assert editor.update_calls == 1
    assert editor.panel_calls == 1
    assert editor.detection_boxes_dict == {0: [{"label": "untouched"}]}


def test_apply_dataset_labels_preserves_labels_added_while_scan_is_pending():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 3
    editor._background_label_scan_pending = True
    editor.background_images = ["current.png"]
    editor.global_labels = {"Manual"}
    editor.background_dataset_labels = set()
    editor.update_label_list = lambda: None
    editor._processing_panel = None

    editor._apply_dataset_labels(3, ("current.png",), {"Disk"})

    assert editor.global_labels == {"Disk", "Manual"}
    assert editor.background_dataset_labels == {"Disk"}


def test_apply_dataset_labels_clears_pending_state_and_refreshes_panel():
    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._background_label_scan_generation = 3
    editor._background_label_scan_pending = True
    editor.background_images = ["current.png"]
    editor.global_labels = set()
    editor.background_dataset_labels = set()
    editor.update_calls = 0
    editor.panel_calls = 0
    editor.update_label_list = lambda: setattr(editor, "update_calls", editor.update_calls + 1)
    editor._processing_panel = type("Panel", (), {"isVisible": lambda self: True})()
    editor._update_processing_panel_labels = lambda: setattr(editor, "panel_calls", editor.panel_calls + 1)

    editor._apply_dataset_labels(3, ("current.png",), {"cat"})

    assert editor._background_label_scan_pending is False
    assert editor._background_label_scan_completed is True
    assert editor.update_calls == 1
    assert editor.panel_calls == 1


def test_finished_non_interrupted_worker_leaves_pending_until_queued_result_is_applied():
    class Worker:
        def isInterruptionRequested(self):
            return False

    editor = type("Editor", (ImageLoaderMixin,), {})()
    worker = Worker()
    editor._background_label_scan_worker = worker
    editor._background_label_scan_workers = {worker}
    editor._background_label_scan_pending = True
    editor._background_label_scan_in_progress = True

    editor._on_dataset_label_scan_finished(worker)

    assert editor._background_label_scan_worker is None
    assert editor._background_label_scan_pending is True


def test_upload_background_closes_lint_dialog(monkeypatch):
    """上传背景图会换数据集，质检弹窗必须关闭（否则指向旧图）。"""

    class Pixmap:
        def __init__(self, path):
            self.path = path

        def isNull(self):
            return False

    class List:
        def clear(self):
            pass

        def addItem(self, item):
            pass

    class Item:
        def __init__(self, text):
            pass

        def setData(self, role, value):
            pass

    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._memory_background_path = ""
    editor._background_label_scan_generation = 0
    editor.global_labels = set()
    editor.background_images = []
    editor.background_list = List()
    editor.current_background = None
    editor.detection_boxes_dict = {}
    editor.canvas_items_dict = {}
    editor.canvas_items = []
    editor.canvas = type("Canvas", (), {
        "background_scale": 0, "is_manual_scale": True, "update": lambda self: None})()
    editor.load_detection_boxes = lambda path: []
    editor.update_label_list = lambda: None
    editor.update_file_count = lambda: None
    editor._start_dataset_label_scan = lambda: None
    closed = []
    editor._close_lint_dialog = lambda: closed.append(True)
    monkeypatch.setattr(image_loader.QFileDialog, "getOpenFileNames",
                        lambda *args: (["first.png"], ""))
    monkeypatch.setattr(image_loader.mixin, "QPixmap", Pixmap)
    monkeypatch.setattr(image_loader.mixin, "QListWidgetItem", Item)
    monkeypatch.setattr(image_loader.mixin, "Qt", type("Qt", (), {"UserRole": 32})())

    editor.upload_background()

    assert closed == [True]


def test_upload_background_starts_dataset_label_scan_after_loading_first_image(monkeypatch):
    class Pixmap:
        def __init__(self, path):
            self.path = path

        def isNull(self):
            return False

    class List:
        def clear(self):
            pass

        def addItem(self, item):
            pass

    class Item:
        def __init__(self, text):
            pass

        def setData(self, role, value):
            pass

    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._memory_background_path = ""
    editor._background_label_scan_generation = 0
    editor.global_labels = set()
    editor.background_images = []
    editor.background_list = List()
    editor.current_background = None
    editor.detection_boxes_dict = {}
    editor.canvas_items_dict = {}
    editor.canvas_items = []
    editor.canvas = type("Canvas", (), {"background_scale": 0, "is_manual_scale": True, "update": lambda self: None})()
    loaded_detection_boxes = []
    editor.load_detection_boxes = lambda path: loaded_detection_boxes.append(path) or []
    editor.update_label_list = lambda: None
    editor.update_file_count = lambda: None
    editor.scan_starts = 0
    editor._start_dataset_label_scan = lambda: setattr(editor, "scan_starts", editor.scan_starts + 1)
    monkeypatch.setattr(image_loader.QFileDialog, "getOpenFileNames", lambda *args: (["first.png", "second.png"], ""))
    monkeypatch.setattr(image_loader.mixin, "QPixmap", Pixmap)
    monkeypatch.setattr(image_loader.mixin, "QListWidgetItem", Item)
    monkeypatch.setattr(image_loader.mixin, "Qt", type("Qt", (), {"UserRole": 32})())

    editor.upload_background()

    assert editor.current_background.path == "first.png"
    assert loaded_detection_boxes == ["first.png"]
    # 只有**被加载过**的那张在缓存里：另一张等真正切过去时再读盘。
    # （早先这里断言 `{0: [], 1: []}`，那是「预置空列表当占位」的旧行为；
    #  空列表现在是有意义的缓存值，预置它会让磁盘上的框永远读不进来。）
    assert editor.detection_boxes_dict == {0: []}
    assert editor.scan_starts == 1


def test_upload_background_finishes_replacement_when_no_selected_image_is_valid(monkeypatch):
    class Pixmap:
        def __init__(self, path):
            pass

        def isNull(self):
            return True

    class List:
        def clear(self):
            pass

    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._memory_background_path = ""
    editor._background_label_scan_generation = 0
    editor._background_label_scan_worker = None
    editor._background_label_scan_pending = False
    editor.global_labels = {"old"}
    editor.background_images = []
    editor.background_list = List()
    editor.current_background = None
    editor.detection_boxes_dict = {}
    editor.canvas_items_dict = {}
    editor.canvas_items = []
    editor.update_file_count = lambda: None
    monkeypatch.setattr(image_loader.QFileDialog, "getOpenFileNames", lambda *args: (["invalid.png"], ""))
    monkeypatch.setattr(image_loader.mixin, "QPixmap", Pixmap)

    editor.upload_background()

    assert editor._background_label_scan_pending is False
    assert editor._background_label_scan_in_progress is False
    assert editor._background_label_scan_completed is False


@pytest.mark.parametrize(
    ("listdir", "raises"),
    [(lambda path: [], False), (lambda path: (_ for _ in ()).throw(OSError("cannot enumerate")), True)],
)
def test_load_background_folder_finishes_replacement_when_no_scan_can_start(monkeypatch, listdir, raises):
    class List:
        def clear(self):
            pass

        def setUpdatesEnabled(self, value):
            pass

        def viewport(self):
            return type("Viewport", (), {"update": lambda self: None})()

    editor = type("Editor", (ImageLoaderMixin,), {})()
    editor._memory_background_path = ""
    editor._background_label_scan_generation = 0
    editor._background_label_scan_worker = None
    editor.global_labels = {"old"}
    editor._background_label_scan_pending = False
    editor.background_images = []
    editor.background_list = List()
    editor.current_background = None
    editor.detection_boxes_dict = {}
    editor.canvas_items_dict = {}
    editor.canvas_items = []
    editor.canvas = type("Canvas", (), {})()
    editor._show_loading_spinner = lambda: None
    editor._hide_loading_spinner = lambda: None
    editor.update_file_count = lambda: None
    editor.update_label_list = lambda: None
    monkeypatch.setattr(image_loader.os, "listdir", listdir)
    monkeypatch.setattr(image_loader.QMessageBox, "warning", lambda *args: None)

    if raises:
        editor.load_background_folder("images")
    else:
        editor.load_background_folder("images")

    assert editor._background_label_scan_pending is False
    assert editor._background_label_scan_in_progress is False
    assert editor._background_label_scan_completed is False


def test_processing_panel_uses_existing_labels_while_background_scan_is_running(monkeypatch):
    panel = ProcessingPanel.__new__(ProcessingPanel)
    panel._editor = type("Editor", (), {
        "detection_boxes_dict": {}, "global_labels": set(),
        "background_dataset_labels": set(),
        "background_images": ["missing.png"],
        "_background_label_scan_worker": type("Worker", (), {"isRunning": lambda self: True})(),
    })()
    panel._path_edit = type("PathEdit", (), {"text": lambda self: ""})()
    panel._get_session_dir = lambda: ""
    monkeypatch.setattr("labelforge.ui.processing_panel.os.path.exists", lambda path: True)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: pytest.fail("must not scan JSON on UI thread"))

    assert panel._scan_labels_from_json() == []


def test_processing_panel_does_not_scan_json_while_background_result_is_pending(monkeypatch):
    panel = ProcessingPanel.__new__(ProcessingPanel)
    panel._editor = type("Editor", (), {
        "detection_boxes_dict": {}, "global_labels": set(),
        "background_dataset_labels": set(),
        "background_images": ["missing.png"],
        "_background_label_scan_pending": True,
    })()
    panel._path_edit = type("PathEdit", (), {"text": lambda self: ""})()
    panel._get_session_dir = lambda: ""
    monkeypatch.setattr(panel, "_scan_labels_from_json", lambda: pytest.fail("must not scan JSON while result is pending"))
    panel._clear_grid = lambda: None
    panel._exp_scan_label = type("Spinner", (), {"setVisible": lambda self, value: None, "setLabel": lambda self, value: None})()

    panel._update_labels_list()


def test_processing_panel_does_not_scan_json_after_empty_background_result(monkeypatch):
    panel = ProcessingPanel.__new__(ProcessingPanel)
    panel._editor = type("Editor", (), {
        "detection_boxes_dict": {}, "global_labels": set(),
        "background_dataset_labels": set(),
        "background_images": ["missing.png"],
        "_background_label_scan_completed": True,
    })()
    panel._path_edit = type("PathEdit", (), {"text": lambda self: ""})()
    panel._get_session_dir = lambda: ""
    panel._clear_grid = lambda: None
    spinner = type("Spinner", (), {
        "visible": [], "label": None,
        "setVisible": lambda self, value: self.visible.append(value),
        "setLabel": lambda self, value: setattr(self, "label", value),
    })()
    panel._exp_scan_label = spinner
    panel._exp_label_grid = type("Grid", (), {"addWidget": lambda self, *args: None})()
    monkeypatch.setattr(panel, "_scan_labels_from_json", lambda: pytest.fail(
        "must not scan JSON after an empty background result is applied"
    ))

    panel._update_labels_list()

    assert spinner.visible == [False]


def test_processing_panel_fast_labels_ignores_detection_boxes_and_uses_background_dataset_only():
    panel = ProcessingPanel.__new__(ProcessingPanel)
    panel._editor = type("Editor", (), {
        "detection_boxes_dict": {0: [{"label": "paste_pollution"}]},
        "global_labels": {"paste_pollution", "cat"},
        "background_dataset_labels": {"cat"},
        "_background_label_scan_completed": True,
    })()

    assert panel._fast_labels() == ["cat"]


def test_processing_panel_fast_labels_empty_after_completed_scan():
    panel = ProcessingPanel.__new__(ProcessingPanel)
    panel._editor = type("Editor", (), {
        "detection_boxes_dict": {0: [{"label": "box"}]},
        "global_labels": {"box"},
        "background_dataset_labels": set(),
        "_background_label_scan_completed": True,
    })()

    assert panel._fast_labels() == []


def test_cleanup_background_label_scan_worker_interrupts_without_waiting_finished_worker():
    class Worker:
        def __init__(self):
            self.interrupted = False
            self.wait_calls = []

        def isRunning(self):
            return False

        def requestInterruption(self):
            self.interrupted = True

        def wait(self, timeout):
            self.wait_calls.append(timeout)

    editor = ImageEditor.__new__(ImageEditor)
    worker = Worker()
    editor._background_label_scan_worker = worker

    editor._cleanup_background_label_scan_worker()

    assert worker.interrupted
    assert worker.wait_calls == []
    assert editor._background_label_scan_worker is None


def test_stats_health_payload_discarded_after_rename_bump_real_qt():
    """D1：弹窗内改名/改色推进扫描代数后，迟到的 payload 必须被丢弃——
    不覆盖 _health_payload、不按旧类名重建贴图表；代数一致的新扫描仍正常应用。"""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtCore import QObject, Qt
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel,
                             QTableWidget, QTableWidgetItem)
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new, rewrite_disk=False):
        return False


class FakeEditor(StatsMixin, QObject):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {}
        self.canvas_items_dict = {}
        self.canvas_items = []
        self.current_background_index = -1
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = "x.png"
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


def payload_with(label):
    return {
        "annot": {"stats": {}, "advice": []},
        "paste": {"stats": {
            "class_dist": [{"label": label, "count": 4}],
            "size_hist": {"edges": [0, 1], "counts": [4]},
            "aspect_hist": {"edges": [0, 1], "counts": [4]},
            "iou_hist": {"edges": [0.0, 1.0], "counts": [9]},
        }, "advice": ["a"]},
        "images_scanned": 1,
    }


def make_dialog(editor):
    dialog = QDialog()
    layout = QVBoxLayout(dialog)
    editor._build_health_section(dialog, layout)
    editor._close_health_worker()  # 无事件循环，queued payload 不会投递
    table = QTableWidget(1, 3)
    item = QTableWidgetItem("new_name")
    item.setData(Qt.UserRole, "new_name")
    table.setItem(0, 0, item)
    dialog._paste_table = table
    dialog._total_label = QLabel()
    return dialog


editor = FakeEditor()
editor._health_scan_generation = 0

# 改名推进代数后，迟到的 payload 被丢弃
dialog = make_dialog(editor)
assert editor._health_scan_generation == 0
editor._health_scan_generation += 1  # 模拟弹窗内改名/改色
dialog._on_health_payload(payload_with("old_name"))
assert dialog._health_payload is None
assert dialog._paste_table.item(0, 0).text() == "new_name"

# 代数一致的新扫描 payload 正常应用（防过度丢弃）
dialog2 = make_dialog(editor)
assert editor._health_scan_generation == 1
dialog2._on_health_payload(payload_with("fresh_name"))
assert dialog2._health_payload is not None
assert dialog2._paste_table.item(0, 0).text() == "fresh_name"

editor._close_health_worker()
print("OK")

# Qt 子进程退出阶段析构顺序不定，偶发崩溃/挂死；断言通过后 flush + os._exit(0)
# 绕开析构（os._exit 不冲刷缓冲，必须先 flush，否则标记丢失会误报失败）。
import os, sys
sys.stdout.flush()
os._exit(0)
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
