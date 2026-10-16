"""背景图标签导入测试。

需求：导航栏「标签:」处新增背景标签导入按钮，读 txt（一行一类），
与扫描结果取**并集**——手动导入的类别不能被后续扫描覆盖。
"""
import ast
from pathlib import Path

from labelforge.engine.image_loader import ImageLoaderMixin
from labelforge.ui import i18n

ROOT = Path(__file__).resolve().parents[1]
TOOLBAR_SRC = ROOT / "labelforge" / "ui" / "mixins" / "toolbar.py"
LOADER_SRC = ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py"
ICONS_SRC = ROOT / "labelforge" / "ui" / "icons.py"


class _Editor(ImageLoaderMixin):
    def __init__(self):
        self.global_labels = set()
        self.background_dataset_labels = set()
        self._scanned_background_labels = set()
        self.imported_background_labels = set()
        self.update_calls = 0

    def update_label_list(self):
        self.update_calls += 1


def _editor():
    return _Editor()


# ---------- 按钮存在、背景在前、图标区分 ----------

def test_toolbar_has_background_label_import_button_before_paste_label_button():
    source = TOOLBAR_SRC.read_text(encoding="utf-8")
    assert "self.upload_bg_label_btn = self._create_svg_button(" in source
    assert "self.upload_background_labels" in source
    assert 'tr("导入背景标签文件")' in source
    # 用户要求「先背景再贴图」：背景标签按钮必须排在贴图标签按钮之前
    bg_idx = source.index("self.upload_bg_label_btn = self._create_svg_button(")
    paste_idx = source.index("self.upload_paste_label_btn = self._create_svg_button(")
    assert bg_idx < paste_idx


def test_toolbar_uses_two_distinct_icons_for_the_two_label_imports():
    """两个导入按钮必须长得不一样，且都不再是通用文件图标。"""
    source = TOOLBAR_SRC.read_text(encoding="utf-8")
    bg_block = source[source.index("self.upload_bg_label_btn = self._create_svg_button("):]
    bg_block = bg_block[:bg_block.index(")")]
    paste_block = source[source.index("self.upload_paste_label_btn = self._create_svg_button("):]
    paste_block = paste_block[:paste_block.index(")")]
    assert "SVG_IMPORT" in bg_block
    assert "SVG_PASTE_LABEL" in paste_block
    assert "SVG_FILE," not in bg_block
    assert "SVG_FILE," not in paste_block

    icons = ICONS_SRC.read_text(encoding="utf-8")
    assert "SVG_IMPORT = '<svg" in icons
    assert "SVG_PASTE_LABEL = '<svg" in icons
    # 两个 SVG 常量内容必须不同
    import re
    def svg(name):
        m = re.search(name + r" = '(<svg.*?</svg>)'", icons)
        return m.group(1)
    assert svg("SVG_IMPORT") != svg("SVG_PASTE_LABEL")
    assert svg("SVG_IMPORT") != svg("SVG_FILE")


def test_import_button_label_is_translated_in_both_languages():
    original_lang = i18n.get_lang()
    try:
        assert i18n.t("导入背景标签文件") == "导入背景标签文件"
        assert i18n.t("导入贴图标签文件") == "导入贴图标签文件"
        i18n.set_lang("en")
        assert i18n.t("导入背景标签文件") == "Import Background Label File"
        assert i18n.t("导入贴图标签文件") == "Import Paste Label File"
        assert i18n.t("选择背景标签文件") == "Select Background Label File"
    finally:
        i18n.set_lang(original_lang)


# ---------- 导入行为 ----------

def test_load_background_label_file_reads_one_label_per_line(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("cat\ndog\nbird\n", encoding="utf-8")
    editor = _editor()

    editor.load_background_label_file(str(path))

    assert editor.imported_background_labels == {"cat", "dog", "bird"}
    assert editor.background_dataset_labels == {"cat", "dog", "bird"}


def test_load_background_label_file_ignores_blank_lines_and_extra_columns(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("cat 1 2\ndog\n\n   \nbird extra\n", encoding="utf-8")
    editor = _editor()

    editor.load_background_label_file(str(path))

    assert editor.imported_background_labels == {"cat", "dog", "bird"}


def test_load_background_label_file_merges_with_existing_scanned_labels(tmp_path):
    """并集：导入不能覆盖扫描结果，扫描结果也不能挤掉导入。"""
    path = tmp_path / "labels.txt"
    path.write_text("manual\n", encoding="utf-8")
    editor = _editor()
    editor._scanned_background_labels = {"scanned"}
    editor.background_dataset_labels = {"scanned"}

    editor.load_background_label_file(str(path))

    assert editor.background_dataset_labels == {"scanned", "manual"}


def test_imported_labels_survive_a_subsequent_dataset_scan(tmp_path):
    """核心回归：扫描是整体赋值，导入的类别必须活下来。"""
    path = tmp_path / "labels.txt"
    path.write_text("manual\n", encoding="utf-8")
    editor = _editor()
    editor.load_background_label_file(str(path))

    editor.background_images = ["a.png"]
    editor._background_label_scan_generation = 1
    editor._background_label_scan_worker = None
    editor._processing_panel = None
    editor._apply_dataset_labels(1, ("a.png",), {"fromdisk"})

    assert editor.background_dataset_labels == {"manual", "fromdisk"}


def test_scan_then_import_keeps_both(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("manual\n", encoding="utf-8")
    editor = _editor()
    editor.background_images = ["a.png"]
    editor._background_label_scan_generation = 1
    editor._background_label_scan_worker = None
    editor._processing_panel = None
    editor._apply_dataset_labels(1, ("a.png",), {"fromdisk"})

    editor.load_background_label_file(str(path))

    assert editor.background_dataset_labels == {"fromdisk", "manual"}


def test_importing_twice_accumulates_without_duplicates(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("cat\n", encoding="utf-8")
    second = tmp_path / "b.txt"
    second.write_text("cat\ndog\n", encoding="utf-8")
    editor = _editor()

    editor.load_background_label_file(str(first))
    editor.load_background_label_file(str(second))

    assert editor.imported_background_labels == {"cat", "dog"}


def test_import_adds_labels_to_global_labels(tmp_path):
    """导入的类别要进入 global_labels，否则画布取色拿不到。"""
    path = tmp_path / "labels.txt"
    path.write_text("cat\n", encoding="utf-8")
    editor = _editor()
    editor.global_labels = {"existing"}

    editor.load_background_label_file(str(path))

    assert editor.global_labels == {"existing", "cat"}


def test_import_refreshes_label_list(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("cat\n", encoding="utf-8")
    editor = _editor()

    editor.load_background_label_file(str(path))

    assert editor.update_calls == 1


def test_empty_file_warns_and_changes_nothing(tmp_path, monkeypatch):
    path = tmp_path / "empty.txt"
    path.write_text("\n  \n", encoding="utf-8")
    editor = _editor()
    warnings = []
    import labelforge.engine.image_loader as loader
    monkeypatch.setattr(loader.QMessageBox, "warning",
                        lambda *a, **k: warnings.append(a), raising=False)

    editor.load_background_label_file(str(path))

    assert warnings, "空文件应提示"
    assert editor.imported_background_labels == set()
    assert editor.background_dataset_labels == set()


def test_unreadable_file_reports_error_and_changes_nothing(tmp_path, monkeypatch):
    editor = _editor()
    errors = []
    import labelforge.engine.image_loader as loader
    monkeypatch.setattr(loader.QMessageBox, "critical",
                        lambda *a, **k: errors.append(a), raising=False)

    editor.load_background_label_file(str(tmp_path / "does-not-exist.txt"))

    assert errors, "读取失败应报错"
    assert editor.imported_background_labels == set()
    assert editor.update_calls == 0


# ---------- 生命周期 ----------

def test_replacing_dataset_clears_imported_labels():
    """换数据集 = 换一套类别，导入的也要清掉。"""
    editor = _editor()
    editor.imported_background_labels = {"manual"}
    editor.background_dataset_labels = {"manual"}
    editor._background_label_scan_generation = 1
    editor._background_label_scan_completed = True
    editor._processing_panel = None

    editor._start_background_replacement()

    assert editor.imported_background_labels == set()
    assert editor.background_dataset_labels == set()


def test_clearing_memory_content_resets_imported_labels():
    from labelforge.ui.mixins.memory_record import MemoryRecordMixin

    class Host(MemoryRecordMixin):
        def __init__(self):
            self.background_images = []
            self.small_images = []
            self.canvas_items_dict = {}
            self.detection_boxes_dict = {}
            self.canvas_items = []
            self.detection_boxes = []
            self.global_labels = {"old"}
            self.background_dataset_labels = {"old"}
            self.imported_background_labels = {"manual"}
            self._scanned_background_labels = {"old"}
            self._cached_bg_label_stats = []

    host = Host()
    host._clear_memory_content()

    assert host.imported_background_labels == set()
    assert host._scanned_background_labels == set()


def test_restored_memory_stats_survive_a_later_scan():
    """记忆记录恢复的类别等同于手动导入，重新扫描不能丢。"""
    from labelforge.ui.mixins.memory_record import MemoryRecordMixin

    class Host(MemoryRecordMixin):
        def __init__(self):
            self.global_labels = set()
            self.background_dataset_labels = set()
            self.imported_background_labels = set()
            self._scanned_background_labels = set()
            self.label_color_map = {}
            self.update_calls = 0

        def update_label_list(self):
            self.update_calls += 1

    host = Host()
    host._restore_bg_label_stats(
        [{"label": "fromrecord", "count": 1, "color": ""}]
    )

    assert host.imported_background_labels == {"fromrecord"}
    assert host.background_dataset_labels == {"fromrecord"}
    assert host.update_calls == 1


def test_restored_memory_stats_merge_scan_tasks():
    """旧记录 tasks 为 None 时用扫描缓存的 tasks 补齐（框类型列不再为空）。"""
    from labelforge.ui.mixins.memory_record import MemoryRecordMixin

    class Host(MemoryRecordMixin):
        def __init__(self):
            self._cached_bg_label_stats = [
                {"label": "SCar", "count": 161, "color": "", "tasks": ["det"]},
                {"label": "BCar", "count": 84, "color": "",
                 "tasks": ["det", "seg", "pose", "obb"]},
            ]

    host = Host()
    merged = host._merge_restored_stats_with_scan([
        {"label": "SCar", "count": 161, "color": "", "tasks": None},
        {"label": "BCar", "count": 84, "color": "", "tasks": []},
        {"label": "Rider", "count": 43, "color": "", "tasks": None},
    ])

    by_label = {m["label"]: m for m in merged}
    assert by_label["SCar"]["tasks"] == ["det"]
    assert by_label["BCar"]["tasks"] == ["det", "obb", "pose", "seg"]
    assert by_label["Rider"]["tasks"] == []
    assert by_label["SCar"]["count"] == 161


def test_restored_memory_stats_keep_recorded_tasks():
    """记录里已有 tasks 时以记录为准，不被扫描缓存覆盖。"""
    from labelforge.ui.mixins.memory_record import MemoryRecordMixin

    class Host(MemoryRecordMixin):
        def __init__(self):
            self._cached_bg_label_stats = [
                {"label": "a", "count": 5, "color": "", "tasks": ["det"]},
            ]

    host = Host()
    merged = host._merge_restored_stats_with_scan([
        {"label": "a", "count": 5, "color": "", "tasks": ["obb"]},
    ])
    assert merged[0]["tasks"] == ["obb"]


def test_main_window_initializes_the_two_new_label_sets():
    source = (ROOT / "labelforge" / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "self._scanned_background_labels = set()" in source
    assert "self.imported_background_labels = set()" in source


# ---------- 扫描路径必须保留导入 ----------

def test_apply_dataset_labels_records_scanned_labels_separately():
    source = LOADER_SRC.read_text(encoding="utf-8")
    assert "self._scanned_background_labels = set(labels)" in source
    assert "self.background_dataset_labels = labels | imported" in source


def test_processing_panel_scan_preserves_imported_labels():
    """处理面板另有一条扫描路径，同样不能覆盖手动导入。"""
    source = (ROOT / "labelforge" / "ui" / "processing_panel.py").read_text(encoding="utf-8")
    assert "_scanned_background_labels = set(labels)" in source
    assert "imported_background_labels" in source
