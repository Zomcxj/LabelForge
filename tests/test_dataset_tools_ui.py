"""数据集工具（supervision 格式转换）UI 回归测试。

约定（同 test_ui_regressions / test_i18n_regressions / test_ui_layout_regressions）：
- 不实例化 Qt 窗口；源码字符串 / AST / i18n 字典断言 + monkeypatch 派发。
"""
import ast
import inspect
import os
import re

from pathlib import Path

from labelforge.ui import i18n
from labelforge.ui import dataset_tools_dialog
from labelforge.ui import icons
from labelforge.ui.dataset_tools_dialog import DatasetToolsDialog
from labelforge.ui.mixins import toolbar as toolbar_mod
from labelforge.ui.mixins.dataset_tools import DatasetToolsMixin
from labelforge.ui.mixins.options_popup import OptionsPopupMixin
from labelforge.ui.mixins.toolbar import ToolbarMixin
from labelforge.ui.mixins.translation import TranslationMixin

ROOT = Path(__file__).resolve().parents[1]
DIALOG_SRC = ROOT / "labelforge" / "ui" / "dataset_tools_dialog.py"
THEME_SRC = ROOT / "labelforge" / "ui" / "theme.py"
POPUP_SRC = ROOT / "labelforge" / "ui" / "mixins" / "options_popup.py"

DATASET_KEYS = (
    "格式", "格式转换", "输入格式:", "输出格式:", "图片目录:",
    "标注目录:", "标注文件:", "输出目录:", "覆盖输出目录", "校验", "开始转换",
    "关闭", "选择图片目录", "选择标注目录", "选择标注文件", "选择输出目录",
    "正在转换数据集...", "正在校验数据集...", "操作日志将显示在这里...",
    "log_sv_validate_ok", "log_sv_validate_fail",
    "log_sv_warn", "log_sv_err", "log_sv_start", "log_sv_done",
    "log_sv_cancelled", "sv_need_images_dir", "sv_need_annotations",
    "sv_need_output", "sv_same_dir",
)


def _dialog_method_src(name):
    tree = ast.parse(DIALOG_SRC.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DatasetToolsDialog"
    )
    fn = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(DIALOG_SRC.read_text(encoding="utf-8"), fn)


def _format_btn_src():
    """_create_options_menu 中 format_btn 的创建代码块。"""
    src = inspect.getsource(OptionsPopupMixin._create_options_menu)
    start = src.index("self.format_btn = QPushButton")
    end = src.index("self.process_btn = QPushButton")
    return src[start:end]


# ---------- 1. 按钮位置：统计 与 导出 之间 ----------

def test_format_btn_created_between_stats_and_export():
    """格式按钮位于「统计」与「导出」之间，与它们同款 optionsBtn 样式。"""
    src = inspect.getsource(OptionsPopupMixin._create_options_menu)
    stats_idx = src.index("self.view_stats_btn = QPushButton")
    fmt_idx = src.index("self.format_btn = QPushButton")
    exp_idx = src.index("self.process_btn = QPushButton")
    assert stats_idx < fmt_idx < exp_idx
    assert 'self.format_btn = QPushButton(tr("格式"))' in src
    assert 'setObjectName("optionsBtn")' in src
    assert "self.format_btn.clicked.connect(self._open_dataset_tools)" in src
    assert "self.format_btn.setToolTip(tr(" in src


def test_format_btn_not_in_top_toolbar():
    """格式按钮不再放在右上角工具栏（原 dataset_tool_btn 已移除）。"""
    src = inspect.getsource(ToolbarMixin._create_toolbar)
    assert "format_btn" not in src
    assert "dataset_tool_btn" not in src
    assert "SVG_CONVERT" not in src
    assert "_open_dataset_tools" not in src


# ---------- 2. 图标常量 ----------

def test_format_btn_is_text_only_without_icon():
    """格式按钮是纯文字按钮，和统计/导出一致，不挂图标。"""
    block = _format_btn_src()
    assert "setIcon" not in block
    assert "SVG_CONVERT" not in block


def test_svg_convert_constant_removed():
    """文字按钮不再需要转换箭头图标，常量已删除。"""
    assert not hasattr(icons, "SVG_CONVERT")
    icons_src = (ROOT / "labelforge" / "ui" / "icons.py").read_text(encoding="utf-8")
    assert "SVG_CONVERT" not in icons_src


def test_toolbar_module_does_not_import_svg_convert():
    toolbar_src = (ROOT / "labelforge" / "ui" / "mixins" / "toolbar.py").read_text(
        encoding="utf-8")
    assert "SVG_CONVERT" not in toolbar_src
    tree = ast.parse(toolbar_src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module == "icons":
            imported.update(alias.name for alias in node.names)
    assert "SVG_CONVERT" not in imported
    assert "SVG_CONVERT" not in dir(toolbar_mod)


# ---------- 3. mixin 注册 ----------

def test_dataset_tools_mixin_registered_on_image_editor():
    from labelforge.ui.main_window import ImageEditor
    assert DatasetToolsMixin in ImageEditor.__mro__
    assert OptionsPopupMixin in ImageEditor.__mro__
    assert callable(getattr(ImageEditor, "_open_dataset_tools"))
    assert callable(getattr(ImageEditor, "_get_dataset_tools_folder"))


# ---------- 4. 派发 ----------

def test_open_dataset_tools_dispatches_dialog(monkeypatch):
    calls = {}

    class FakeDialog:
        def __init__(self, parent, default_folder=None):
            calls["parent"] = parent
            calls["default_folder"] = default_folder

        def exec_(self):
            calls["exec_"] = True

    monkeypatch.setattr(dataset_tools_dialog, "DatasetToolsDialog", FakeDialog)

    class Owner(DatasetToolsMixin):
        def __init__(self):
            self.background_images = []

    owner = Owner()
    owner._open_dataset_tools()

    assert calls["parent"] is owner
    assert calls["default_folder"] == ""
    assert calls["exec_"] is True


# ---------- 5. 默认目录逻辑 ----------

def test_default_folder_empty_without_background_images():
    """页面未加载数据集时不默认到工作目录，路径留空。"""
    class Owner(DatasetToolsMixin):
        background_images = []

    assert DatasetToolsMixin._get_dataset_tools_folder(Owner()) == ""


def test_default_folder_empty_when_background_images_attr_absent():
    class Owner(DatasetToolsMixin):
        pass

    assert DatasetToolsMixin._get_dataset_tools_folder(Owner()) == ""


def test_default_folder_prefers_first_background_image_dir():
    """取第一张背景图所在目录，路径分隔符跟随当前平台。"""
    folder = os.path.join("D", "data", "shots")

    class Owner(DatasetToolsMixin):
        background_images = [
            os.path.join(folder, "a.png"),
            os.path.join(folder, "b.png"),
        ]

    assert DatasetToolsMixin._get_dataset_tools_folder(Owner()) == folder


def test_dialog_default_folder_is_empty_not_cwd():
    """对话框自身也不回退到 os.getcwd()，未传目录时各路径为空。"""
    src = DIALOG_SRC.read_text(encoding="utf-8")
    assert "os.getcwd()" not in src
    assert 'default_folder = default_folder or ""' in src


# ---------- 6. 主题样式 ----------

def test_theme_no_longer_styles_removed_dataset_tool_btn():
    src = THEME_SRC.read_text(encoding="utf-8")
    assert "#datasetToolBtn" not in src


def test_format_btn_reuses_options_btn_theme_style():
    """格式按钮复用 optionsBtn 选择器，与其他选项按钮外观一致。"""
    assert 'setObjectName("optionsBtn")' in _format_btn_src()
    src = THEME_SRC.read_text(encoding="utf-8")
    assert "QPushButton#optionsBtn {" in src
    assert "QPushButton#optionsBtn:hover {" in src


# ---------- 7. i18n 双语 ----------

def test_dataset_keys_present_in_both_language_tables():
    for lang in ("zh", "en"):
        table = i18n._strings[lang]
        for key in DATASET_KEYS:
            assert key in table, (lang, key)
            assert table[key].strip(), (lang, key)


def test_dataset_keys_resolve_through_i18n_t():
    original_lang = i18n.get_lang()
    try:
        for lang in ("zh", "en"):
            i18n.set_lang(lang)
            for key in DATASET_KEYS:
                assert isinstance(i18n.t(key), str)
                assert i18n.t(key).strip()
        # 英文侧必须是真翻译（值 != 键）
        i18n.set_lang("en")
        for key in DATASET_KEYS:
            assert i18n.t(key) != key, key
    finally:
        i18n.set_lang(original_lang)


def test_english_dataset_tool_values_are_sensible():
    original_lang = i18n.get_lang()
    try:
        i18n.set_lang("en")
        assert i18n.t("格式") == "Format"
        assert i18n.t("格式转换") == "Dataset Format Conversion"
        assert i18n.t("校验") == "Validate"
        assert i18n.t("开始转换") == "Convert"
        assert i18n.t("sv_same_dir") == "Output directory must differ from the input directory"
    finally:
        i18n.set_lang(original_lang)


def test_removed_keys_absent_from_both_language_tables():
    """按钮改名与「已就绪」提示下线后，旧键不应残留。"""
    for lang in ("zh", "en"):
        for key in ("数据集工具", "sv_ready"):
            assert key not in i18n._strings[lang], (lang, key)


# ---------- 8. 对话框源码关键约束 ----------

def test_run_validate_and_run_convert_refresh_availability():
    for name in ("_run_validate", "_run_convert"):
        src = _dialog_method_src(name)
        assert "_refresh_availability()" in src, name


def test_run_convert_uses_convert_paths_and_handles_cancelled():
    src = _dialog_method_src("_run_convert")
    assert "convert_paths" in src
    assert "result.get(" in src and "cancelled" in src


def test_collect_inputs_protects_against_same_directory():
    src = _dialog_method_src("_collect_inputs")
    assert 'tr("sv_same_dir")' in src
    assert "normcase" in src


def test_format_choices_cover_four_formats():
    source = DIALOG_SRC.read_text(encoding="utf-8")
    choices = ast.literal_eval(
        next(
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "FORMAT_CHOICES" for t in node.targets)
        )
    )
    assert tuple(key for _, key in choices) == ("yolo", "coco", "voc", "labelme")


def test_default_conversion_direction_is_labelme_to_yolo():
    """最常见的转换方向是 LabelMe -> YOLO，打开对话框即为该组合。"""
    assert dataset_tools_dialog.DEFAULT_INPUT_FORMAT == "labelme"
    assert dataset_tools_dialog.DEFAULT_OUTPUT_FORMAT == "yolo"
    assert dataset_tools_dialog._format_index("labelme") == 3
    assert dataset_tools_dialog._format_index("yolo") == 0
    src = DIALOG_SRC.read_text(encoding="utf-8")
    assert "self._in_combo.setCurrentIndex(_format_index(DEFAULT_INPUT_FORMAT))" in src
    assert "self._out_combo.setCurrentIndex(_format_index(DEFAULT_OUTPUT_FORMAT))" in src


def test_default_index_is_set_after_dependent_widgets_exist():
    """_on_input_format_changed 依赖 _ann_lbl/_yaml_*；默认值必须在其后设置。

    否则信号会在半初始化状态下触发，抛 AttributeError。
    """
    tree = ast.parse(DIALOG_SRC.read_text(encoding="utf-8"))
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DatasetToolsDialog"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    set_line = None
    for node in ast.walk(init):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setCurrentIndex"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_in_combo"):
            set_line = node.lineno
    assert set_line is not None, "未找到输入格式默认值设置"

    for widget in ("_ann_lbl", "_yaml_lbl", "_yaml_edit"):
        create_line = min(
            node.lineno for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == widget
        )
        assert create_line < set_line, f"{widget} 必须在设置默认值之前创建"

    # 默认值设置必须先于 _on_input_format_changed() 调用，否则刷新的是旧状态
    call_line = min(
        node.lineno for node in ast.walk(init)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_on_input_format_changed"
    )
    assert call_line > set_line


def test_format_index_rejects_unknown_key():
    import pytest
    with pytest.raises(ValueError):
        dataset_tools_dialog._format_index("does-not-exist")


def test_input_format_changed_toggles_coco_file_label_and_gates_yaml_row():
    """非 YOLO 时 data.yaml 整行禁用（不隐藏，避免布局跳动）。"""
    src = _dialog_method_src("_on_input_format_changed")
    assert 'tr("标注文件:")' in src
    assert 'tr("标注目录:")' in src
    assert 'fmt == "coco"' in src
    assert 'fmt == "yolo"' in src
    for widget in ("_yaml_lbl", "_yaml_edit", "_yaml_btn"):
        assert f"{widget}.setEnabled(is_yolo)" in src, widget
    assert ".setVisible(is_yolo)" not in src
    assert "_yaml_edit.setVisible" not in src


def test_yaml_row_gated_by_input_format_not_hidden():
    """行为断言：四个格式下 data.yaml 三件套的启用态正确，且始终可见。"""
    class Flag:
        def __init__(self):
            self.enabled = True
            self.visible = True
            self.text = None

        def setEnabled(self, value):
            self.enabled = value

        def setVisible(self, value):
            self.visible = value

        def setText(self, value):
            self.text = value

    class Combo:
        def __init__(self, index):
            self._index = index

        def currentIndex(self):
            return self._index

    class Owner:
        def __init__(self, index):
            self._in_combo = Combo(index)
            self._ann_lbl = Flag()
            self._yaml_lbl = Flag()
            self._yaml_edit = Flag()
            self._yaml_btn = Flag()

        def _input_format(self):
            return dataset_tools_dialog.FORMAT_CHOICES[
                self._in_combo.currentIndex()][1]

    for index, fmt, expected in ((0, "yolo", True), (1, "coco", False),
                                 (2, "voc", False), (3, "labelme", False)):
        owner = Owner(index)
        DatasetToolsDialog._on_input_format_changed(owner)
        for widget in (owner._yaml_lbl, owner._yaml_edit, owner._yaml_btn):
            assert widget.enabled is expected, fmt
            assert widget.visible is True, fmt
        label_key = "标注文件:" if fmt == "coco" else "标注目录:"
        assert owner._ann_lbl.text == i18n.t(label_key), fmt


def test_add_path_row_binds_button_when_button_attr_given():
    """button_attr 给定时把「选择」按钮挂到实例属性上，便于整行禁用。"""
    src = _dialog_method_src("_add_path_row")
    assert "button_attr=None" in src
    assert "setattr(self, button_attr, btn)" in src
    dialog_src = DIALOG_SRC.read_text(encoding="utf-8")
    assert 'button_attr="_yaml_btn"' in dialog_src


def test_log_area_reuses_processing_panel_log_area_style():
    """日志框圆角样式与「导出→增强划分」的日志框一致：同一 objectName。"""
    dialog_src = DIALOG_SRC.read_text(encoding="utf-8")
    panel_src = (ROOT / "labelforge" / "ui" / "processing_panel.py").read_text(
        encoding="utf-8")
    assert 'self._log_area.setObjectName("logArea")' in dialog_src
    assert 'self._log_area.setObjectName("logArea")' in panel_src
    theme_src = THEME_SRC.read_text(encoding="utf-8")
    block = theme_src.split("QTextEdit#logArea")[1].split("}}", 1)[0]
    assert "border-radius" in block, "logArea 样式必须带圆角"


def test_theme_stylesheet_is_applied_app_wide():
    """logArea 样式来自全局 app stylesheet，对话框才能继承到。"""
    theme_mixin = (ROOT / "labelforge" / "ui" / "mixins" / "theme.py").read_text(
        encoding="utf-8")
    assert "app.setStyleSheet(ThemeManager.get_stylesheet())" in theme_mixin


def test_refresh_availability_always_enables_actions():
    """转换实现已内置，不再探测外部依赖，按钮始终可用。"""
    src = _dialog_method_src("_refresh_availability")
    assert "supervision" not in src.lower()
    assert "_validate_btn.setEnabled(True)" in src
    assert "_convert_btn.setEnabled(True)" in src
    assert "return True" in src


def test_refresh_availability_hides_status_label_when_ready():
    """无外部依赖后没有「已就绪/未安装」状态可显示，状态标签清空并隐藏。"""
    src = _dialog_method_src("_refresh_availability")
    assert 'tr("sv_ready")' not in src
    assert "_status_lbl.setVisible(False)" in src
    assert "_status_lbl.setVisible(True)" not in src
    clear_idx = src.index('self._status_lbl.setText("")')
    hide_idx = src.index("_status_lbl.setVisible(False)")
    assert clear_idx < hide_idx


# ---------- 9. 不得回归 ----------

def test_dialog_does_not_import_supervision_directly():
    source = DIALOG_SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] != "supervision" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module.split(".")[0] != "supervision"
            assert node.level == 0 or module.split(".")[0] != "supervision"


def test_dialog_imports_dataset_converter_adapter_at_module_top_level():
    source = DIALOG_SRC.read_text(encoding="utf-8")
    assert "from ..engine import dataset_converter as st" in source
    tree = ast.parse(source)
    top_level = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 2
        and node.module == "engine"
        and any(alias.name == "dataset_converter" and alias.asname == "st"
                for alias in node.names)
        for node in top_level
    )


def test_dialog_has_no_bare_supervision_calls():
    source = DIALOG_SRC.read_text(encoding="utf-8")
    for forbidden in ("supervision.yolo", "supervision.coco", "sv.DetectionDataset"):
        assert forbidden not in source


# ---------- 10. 语言切换后刷新 format_btn ----------

def test_refresh_ui_texts_source_refreshes_format_btn_with_hasattr_guard():
    """_refresh_ui_texts 必须刷新 format_btn 的文字与 tooltip，且用 hasattr 守卫。"""
    src = inspect.getsource(TranslationMixin._refresh_ui_texts)
    assert "format_btn" in src
    assert 'self.format_btn.setText(tr("格式"))' in src
    assert 'self.format_btn.setToolTip(tr("格式转换"))' in src
    assert "hasattr(self, 'format_btn')" in src
    # 必须紧跟 theme_btn 的 tooltip 刷新之后（新增语句的实际位置）
    theme_idx = src.index('self.theme_btn.setToolTip(tr("切换深色/浅色主题"))')
    guard_idx = src.index("hasattr(self, 'format_btn')")
    refresh_idx = src.index('self.format_btn.setText(tr("格式"))')
    assert theme_idx < guard_idx < refresh_idx


def test_refresh_ui_texts_switches_format_btn_language():
    """行为断言：切 en 后按钮变 "Format" / tooltip 变 "Dataset Format Conversion"，
    切回 zh 变 "格式" / "格式转换"。

    不构造 ImageEditor（offscreen 下会挂住），改用最小假对象只挂上
    _refresh_ui_texts 无条件访问的属性；其余分支均被 hasattr 守卫，
    缺失属性时自动跳过，不会 AttributeError。
    """
    class Stub:
        def __init__(self):
            self.text = None
            self.tooltip = None

        def setText(self, value):
            self.text = value

        def setToolTip(self, value):
            self.tooltip = value

    class Owner(TranslationMixin):
        def __init__(self):
            for name in (
                "auto_save_b_checkbox", "auto_save_p_checkbox",
                "show_labels_checkbox", "show_label_names_checkbox",
                "auto_label_checkbox", "prefix_checkbox",
                "show_paste_names_checkbox", "show_grid_checkbox",
                "random_paste_btn", "batch_paste_btn",
                "toggle_view_btn", "clear_btn", "save_btn", "save_all_btn",
                "lang_btn", "theme_btn",
            ):
                setattr(self, name, Stub())
            self.format_btn = Stub()
            self.process_btn = Stub()
            self.is_thumbnail_mode = False

    owner = Owner()
    original_lang = i18n.get_lang()
    try:
        i18n.set_lang("en")
        owner._refresh_ui_texts()
        assert owner.format_btn.text == "Format"
        assert owner.format_btn.tooltip == "Dataset Format Conversion"
        assert owner.format_btn.tooltip != owner.process_btn.tooltip

        i18n.set_lang("zh")
        owner._refresh_ui_texts()
        assert owner.format_btn.text == "格式"
        assert owner.format_btn.tooltip == "格式转换"
    finally:
        i18n.set_lang(original_lang)


# ---------- 11. 对话框文案随语言刷新 ----------

def test_show_event_refreshes_ui_texts():
    """showEvent 必须刷新文案，避免语言在构造之后变化时标题等整体过期。"""
    src = _dialog_method_src("showEvent")
    assert "_refresh_ui_texts()" in src
    # 刷新要先于居中与标题栏配色
    refresh_idx = src.index("_refresh_ui_texts()")
    center_idx = src.index("center_on_parent")
    assert refresh_idx < center_idx


def test_refresh_ui_texts_source_refreshes_title_and_registry():
    src = _dialog_method_src("_refresh_ui_texts")
    assert 'self.setWindowTitle(tr("格式转换"))' in src
    assert "for widget, key in self._text_widgets" in src
    assert "widget.setText(tr(key))" in src
    # 标注行标签的 key 随输入格式变化，交给 _on_input_format_changed 处理
    assert "_on_input_format_changed()" in src


def test_refresh_ui_texts_switches_title_and_registered_widgets():
    """行为断言：绑定真实方法，验证标题与全部登记控件都切到目标语言。"""
    class Widget:
        def __init__(self):
            self.text = None

        def setText(self, value):
            self.text = value

    class LogArea:
        def __init__(self):
            self.placeholder = None

        def setPlaceholderText(self, value):
            self.placeholder = value

    class Owner:
        def __init__(self, widgets, keys):
            self.title = None
            self._text_widgets = list(zip(widgets, keys))
            self.format_changed = 0
            self._log_area = LogArea()

        def setWindowTitle(self, value):
            self.title = value

        def _on_input_format_changed(self):
            self.format_changed += 1

    keys = ("校验", "开始转换", "关闭", "选择", "输入格式:", "输出格式:",
            "覆盖输出目录", "图片目录:", "输出目录:")
    zh = {k: i18n._strings["zh"][k] for k in keys}
    en = {k: i18n._strings["en"][k] for k in keys}
    assert all(v != k for k, v in en.items()), "英文侧必须是真翻译，否则测不出切换"

    original_lang = i18n.get_lang()
    try:
        i18n.set_lang("en")
        widgets = [Widget() for _ in keys]
        owner = Owner(widgets, keys)
        DatasetToolsDialog._refresh_ui_texts(owner)
        assert owner.title == "Dataset Format Conversion"
        assert [w.text for w in widgets] == list(en.values())
        assert owner._log_area.placeholder == "Operation logs will appear here..."
        assert owner.format_changed == 1

        i18n.set_lang("zh")
        DatasetToolsDialog._refresh_ui_texts(owner)
        assert owner.title == "格式转换"
        assert [w.text for w in widgets] == list(zh.values())
        assert owner._log_area.placeholder == "操作日志将显示在这里..."
        assert owner.format_changed == 2
    finally:
        i18n.set_lang(original_lang)


def _registered_key_literals(node):
    """收集 _text_widgets 注册项与 _add_path_row 调用里的 i18n key 字面量。"""
    keys = set()
    for node in ast.walk(node):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr in ("append", "extend"):
            for pair in ast.walk(node):
                if isinstance(pair, ast.Tuple) and len(pair.elts) == 2:
                    literal = pair.elts[1]
                    if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                        keys.add(literal.value)
        if node.func.attr == "_add_path_row" and len(node.args) >= 2:
            literal = node.args[1]
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                keys.add(literal.value)
    return keys


def test_registered_widget_keys_all_exist_in_i18n():
    """登记文案的 key 必须真实存在于双语表，否则 t() 会静默回退成 key 本身。"""
    tree = ast.parse(DIALOG_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "DatasetToolsDialog")
    keys = _registered_key_literals(cls)
    assert {"选择", "校验", "开始转换", "关闭", "输入格式:", "输出格式:",
            "覆盖输出目录", "图片目录:", "输出目录:"} <= keys, sorted(keys)
    for lang in ("zh", "en"):
        for key in keys:
            assert key in i18n._strings[lang], (lang, key)


def test_add_path_row_registers_key_not_translated_text():
    """注册表存 key 而非 tr() 结果，否则英文切回中文会静默失效。"""
    src = _dialog_method_src("_add_path_row")
    assert "QLabel(tr(label))" in src
    assert "self._text_widgets.append((lbl, label))" in src
    # __init__ 调用点必须传 key，不能传 tr(...)
    init_src = _dialog_method_src("__init__")
    for call in re.findall(r"self\._add_path_row\(\s*layout,\s*([^,]+),", init_src):
        assert "tr(" not in call, call.strip()
