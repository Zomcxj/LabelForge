"""背景列表相对路径显示 + 右键复制图片路径 测试。"""
import os
import sys
from pathlib import Path

import pytest

from labelforge.core import config as config_module
from labelforge.core import config_manager
from labelforge.core.utils import PathUtils
from labelforge.ui import i18n

ROOT = Path(__file__).resolve().parents[1]
OPTIONS_SRC = ROOT / "labelforge" / "ui" / "mixins" / "options_popup.py"
BACKGROUND_LIST_SRC = ROOT / "labelforge" / "ui" / "mixins" / "background_list.py"
LISTS_SRC = ROOT / "labelforge" / "ui" / "mixins" / "lists.py"
TRANSLATION_SRC = ROOT / "labelforge" / "ui" / "mixins" / "translation.py"

# 相对路径显示依赖当前操作系统的路径语义（Windows 反斜杠/盘符，POSIX 正斜杠），
# 因此断言必须随平台取不同的输入与期望值，否则在 Linux CI 上必然失败。
IS_WINDOWS = sys.platform == "win32"


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    path = tmp_path / "labelforge.json"
    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(path))
    yield path


# ---------- 路径转换纯函数 ----------

def test_relative_path_uses_the_parent_of_the_loaded_folder(tmp_path):
    base = str(tmp_path)
    path = os.path.join(base, "images", "1.png")
    assert PathUtils.to_relative_display_path(path, base) == os.path.join("images", "1.png")


def test_relative_path_handles_nested_folders(tmp_path):
    base = str(tmp_path)
    path = os.path.join(base, "images", "sub", "a.png")
    assert PathUtils.to_relative_display_path(path, base) == os.path.join("images", "sub", "a.png")


@pytest.mark.skipif(not IS_WINDOWS, reason="跨盘符 relpath 仅在 Windows 抛 ValueError")
def test_relative_path_falls_back_to_absolute_on_a_different_drive():
    """Windows 下跨盘符 relpath 会抛 ValueError，必须回退而不是崩溃。"""
    result = PathUtils.to_relative_display_path(r"E:\other\1.png", r"D:\data")
    assert result == PathUtils.to_display_path(r"E:\other\1.png")


def test_relative_path_falls_back_when_base_is_empty():
    result = PathUtils.to_relative_display_path(r"D:\data\images\1.png", "")
    assert result == PathUtils.to_display_path(r"D:\data\images\1.png")


def test_relative_path_falls_back_when_base_is_none():
    result = PathUtils.to_relative_display_path(r"D:\data\images\1.png", None)
    assert result == PathUtils.to_display_path(r"D:\data\images\1.png")


def test_relative_path_returns_input_for_empty_path():
    assert PathUtils.to_relative_display_path("", r"D:\data") == ""


def test_relative_path_output_has_no_dot_dot_for_a_child_folder(tmp_path):
    """基准是父目录，图片在其子目录里，所以结果不应出现 .."""
    base = str(tmp_path)
    result = PathUtils.to_relative_display_path(
        os.path.join(base, "images", "1.png"), base)
    assert ".." not in result


# ---------- 配置读写 ----------

def test_relative_path_defaults_to_off(temp_config):
    assert config_manager.load_all()["relative_path_display"] is False


def test_relative_path_round_trips_through_config(temp_config):
    config_manager.save_all(relative_path_display=True)
    assert config_manager.load_all()["relative_path_display"] is True

    config_manager.save_all(relative_path_display=False)
    assert config_manager.load_all()["relative_path_display"] is False


def test_save_all_does_not_clobber_other_flags_when_setting_relative_path(temp_config):
    config_manager.save_all(canvas_image_copy_enabled=True)
    config_manager.save_all(relative_path_display=True)

    loaded = config_manager.load_all()
    assert loaded["canvas_image_copy_enabled"] is True
    assert loaded["relative_path_display"] is True


def test_reset_all_returns_relative_path_to_default(temp_config):
    config_manager.save_all(relative_path_display=True)

    config_manager.reset_all()

    assert config_manager.load_all()["relative_path_display"] is False


# ---------- 选项菜单接线 ----------

def test_options_menu_has_the_relative_path_toggle_at_the_bottom():
    source = OPTIONS_SRC.read_text(encoding="utf-8")
    # 文字现在由 _refresh_options_menu_texts 按登记的 key 回填，创建时留空
    assert "self.relative_path_action = self.options_menu.addAction(\"\")" in source
    assert "self.relative_path_action.triggered.connect(self._on_relative_path_menu_changed)" in source
    # 必须排在「窗口放大器」之后（选项最下面）
    magnifier = source.index("self.magnifier_action = self.options_menu.addAction")
    relative = source.index("self.relative_path_action = self.options_menu.addAction")
    assert magnifier < relative
    assert source.index("self.options_btn.setMenu(self.options_menu)", relative) > relative
    # 登记的 i18n 键要跟别人一样紧跟在这个 append 里
    tail = source[relative:source.index("self.options_btn.setMenu", relative)]
    assert '"相对路径显示"' in tail, tail[:300]


def test_relative_path_toggle_is_checkable_and_synced():
    source = OPTIONS_SRC.read_text(encoding="utf-8")
    block = source[source.index("self.relative_path_action = "):]
    block = block[:block.index("self.options_btn.setMenu")]
    assert "setCheckable(True)" in block
    assert "setChecked(getattr(self, '_relative_path_display', False))" in block


def test_handler_persists_and_refreshes_the_list():
    source = OPTIONS_SRC.read_text(encoding="utf-8")
    handler = source[source.index("def _on_relative_path_menu_changed"):]
    assert "config_manager.save_all(relative_path_display=" in handler
    assert "_refresh_background_path_texts()" in handler


def test_menu_text_is_translated_in_both_languages():
    original = i18n.get_lang()
    try:
        assert i18n.t("相对路径显示") == "相对路径显示"
        i18n.set_lang("en")
        assert i18n.t("相对路径显示") == "Show relative paths"
    finally:
        i18n.set_lang(original)


def test_translation_refreshes_the_new_menu_item():
    source = TRANSLATION_SRC.read_text(encoding="utf-8")
    # 菜单文字统一由 _refresh_options_menu_texts 刷新；两处刷新都要调它，
    # 且翻译模块不再自己维护菜单文字表（曾因按下标回填导致整体错位）。
    assert source.count('_refresh_options_menu_texts()') == 2
    assert '_menu_actions' not in source


def test_relative_path_menu_item_carries_its_i18n_key():
    """新项必须把自己的 i18n 键登记进 _menu_actions，否则刷新时会没文字。"""
    source = OPTIONS_SRC.read_text(encoding="utf-8")
    assert '"相对路径显示")' in source
    tail = source.split('self.relative_path_action =')[1][:500]
    assert '"相对路径显示"' in tail, tail[:200]


# ---------- 列表项文本统一走 helper ----------

def test_all_list_item_texts_go_through_the_helper():
    loader = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")
    main = (ROOT / "labelforge" / "ui" / "main_window.py").read_text(encoding="utf-8")
    bg_list = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")

    # 每个生成 item 文本的地方都必须用 _display_path_for
    assert "display_path = self._display_path_for(file)" in loader
    assert "display_path = self._display_path_for(first_path)" in loader
    assert "display_path = self._display_path_for(file_path)" in loader
    assert "display_path = self._display_path_for(file)" in main
    assert "QListWidgetItem(self._display_path_for(path))" in bg_list
    assert "addItem(self._display_path_for(fp))" in bg_list


def test_helper_lives_on_the_image_loader_mixin():
    """两个 mixin 都用到它，放在 ImageLoaderMixin 上才能被共享。"""
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")
    assert "def _display_path_for(self, path):" in source
    assert "def _relative_path_base(self):" in source


def test_base_is_the_parent_of_the_loaded_folder():
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")
    block = source[source.index("def _relative_path_base"):source.index("def _display_path_for")]
    assert "os.path.dirname(folder)" in block


def test_refresh_updates_existing_items_without_reloading():
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = source[source.index("def _refresh_background_path_texts"):source.index("def _bg_path_for_item")]
    assert "item.setText(" in block
    assert "_display_path_for(" in block


def test_refresh_handles_both_work_and_delete_views():
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = source[source.index("def _refresh_background_path_texts"):source.index("def _bg_path_for_item")]
    assert "_is_delete_view" in block
    assert "_delete_files" in block
    assert "background_images" in block


# ---------- 右键菜单 ----------

def test_background_list_has_a_context_menu():
    source = LISTS_SRC.read_text(encoding="utf-8")
    assert "self.background_list.setContextMenuPolicy(Qt.CustomContextMenu)" in source
    assert "self.background_list.customContextMenuRequested.connect(self.show_background_context_menu)" in source


def test_context_menu_offers_copy_path():
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = source[source.index("def show_background_context_menu"):]
    assert 'menu.addAction(tr("复制图片路径"))' in block


def test_copy_puts_the_absolute_path_on_the_clipboard():
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = source[source.index("def show_background_context_menu"):]
    assert "QGuiApplication.clipboard().setText(os.path.abspath(path))" in block


def test_copy_path_label_is_translated_in_both_languages():
    original = i18n.get_lang()
    try:
        assert i18n.t("复制图片路径") == "复制图片路径"
        i18n.set_lang("en")
        assert i18n.t("复制图片路径") == "Copy image path"
    finally:
        i18n.set_lang(original)


def _method_body(source, name):
    """截取单个方法体（到下一个同缩进的 def 为止）。"""
    start = source.index(f"def {name}")
    rest = source[start:]
    marker = "\n    def "
    return rest[:rest.index(marker)] if marker in rest[1:] else rest


def test_context_menu_reads_the_path_from_item_metadata_not_the_text():
    """开关打开后 item 文本是相对路径，不能拿文本当路径。"""
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = _method_body(source, "show_background_context_menu")
    assert "_bg_path_for_item(" in block
    assert "item.text()" not in block


def test_bg_path_for_item_prefers_stored_path_role():
    source = BACKGROUND_LIST_SRC.read_text(encoding="utf-8")
    block = source[source.index("def _bg_path_for_item"):]
    assert "BG_ROLE_PATH" in block
    assert "BG_ROLE_INDEX" in block


# ---------- 生命周期 ----------

def test_reset_settings_returns_relative_path_to_default():
    from labelforge.ui.main_window import ImageEditor
    import inspect
    source = inspect.getsource(ImageEditor.reset_settings_to_default)
    assert "self._relative_path_display = False" in source
    assert "relative_path_action.setChecked(False)" in source


def test_main_window_loads_the_flag():
    source = (ROOT / "labelforge" / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "self._relative_path_display = bool(settings.get('relative_path_display', False))" in source
    assert "self._relative_path_display = False" in source  # _init_data 兜底
