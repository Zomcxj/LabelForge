"""④ B 组借鉴功能测试：一键重置设置 + CLI 入口。"""
import inspect
import json
import sys
from pathlib import Path

import pytest

from labelforge.core import config as config_module
from labelforge.core import config_manager
from labelforge.ui import i18n
from labelforge.ui.main_window import _parse_args

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_SRC = ROOT / "labelforge" / "ui" / "settings_dialog.py"
MAIN_WINDOW_SRC = ROOT / "labelforge" / "ui" / "main_window.py"
MAIN_SRC = ROOT / "labelforge" / "main.py"

_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    """在真 Qt offscreen 子进程里跑脚本：控件几何/文本断言必须走真 Qt。

    进程内的假 PySide6 没有布局与真实几何，`mapTo` 之类的比较没有意义。
    """
    import os
    import subprocess

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


@pytest.fixture
def temp_config(tmp_path, monkeypatch):
    path = tmp_path / "labelforge.json"
    monkeypatch.setattr(config_manager, "CONFIG_PATH", str(path))
    yield path


# ---------- B-1 config.reset_defaults ----------

def test_reset_defaults_restores_mutated_config_dicts():
    original = dict(config_module.GRID_CONFIG)
    try:
        config_module.GRID_CONFIG["line_width"] = 99
        config_module.GRID_CONFIG["alpha"] = 1
        config_module.DETECTION_BOX_CONFIG["label_font_size"] = 42
        config_module.NUDGE_CONFIG["step"] = 5

        config_module.reset_defaults()

        assert config_module.GRID_CONFIG == original
        assert config_module.DETECTION_BOX_CONFIG["label_font_size"] == 9
        assert config_module.NUDGE_CONFIG["step"] == 1
    finally:
        config_module.reset_defaults()


def test_reset_defaults_snapshot_is_not_aliased():
    """快照必须和运行时字典是两份对象，否则重置无效。"""
    for name, snapshot in config_module._DEFAULT_SNAPSHOT.items():
        target = getattr(config_module, name)
        assert snapshot is not target, name


def test_reset_defaults_covers_the_mutated_config_dicts():
    for name in ("GRID_CONFIG", "DETECTION_BOX_CONFIG", "NUDGE_CONFIG",
                 "MAGNIFIER_CONFIG", "CROSSHAIR_CONFIG", "BOX_BORDER_CONFIG",
                 "DETECTION_BOX_WHEEL_CONFIG", "SHORTCUT_CONFIG"):
        assert name in config_module._DEFAULT_SNAPSHOT, name


# ---------- B-1 config_manager.reset_all ----------

def test_reset_all_deletes_the_config_file(temp_config):
    temp_config.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")

    assert config_manager.reset_all() is True
    assert not temp_config.exists()


def test_reset_all_succeeds_when_config_file_is_absent(temp_config):
    assert not temp_config.exists()

    assert config_manager.reset_all() is True


def test_reset_all_restores_in_memory_defaults(temp_config):
    config_module.GRID_CONFIG["line_width"] = 77
    temp_config.write_text(json.dumps({"grid_line_width": 77}), encoding="utf-8")

    config_manager.reset_all()

    assert config_module.GRID_CONFIG["line_width"] == 1


def test_reset_all_drops_custom_theme_language_and_shortcuts(temp_config):
    config_manager.save_all(theme="dark", language="en")
    config_manager.save_shortcuts({"undo": "Ctrl+Q"})

    config_manager.reset_all()

    loaded = config_manager.load_all()
    assert loaded["theme"] == "light"
    assert loaded["language"] == "zh"
    assert loaded["shortcuts"]["undo"] == "Ctrl+Z"


def test_reset_all_removes_memory_records(temp_config):
    config_manager.upsert_memory_record({"background_path": "C:/data", "paste_path": "", "label_path": ""})
    assert config_manager.load_memory_records()

    config_manager.reset_all()

    assert config_manager.load_memory_records() == []


# ---------- B-1 settings dialog button ----------

def test_settings_dialog_has_a_reset_button_with_confirmation():
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    assert "self.reset_all_btn = QPushButton(tr(\"恢复默认设置\"))" in source
    assert "self.reset_all_btn.clicked.connect(self._reset_all_settings)" in source
    # 必须二次确认，避免误触清空配置
    assert "question(" in source
    assert "QMessageBox.Yes" in source


def test_reset_button_uses_the_danger_style():
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    block = source[source.index("self.reset_all_btn = QPushButton("):]
    block = block[:block.index("self.reset_all_btn.clicked.connect")]
    assert 'setObjectName("dangerBtn")' in block


def test_reset_button_sits_immediately_left_of_save():
    """「恢复默认设置」必须紧邻「保存」左边（用户要求挪过去）。

    布局：`[版本号] ←stretch→ [恢复默认设置] [保存] [取消]`。
    顺序断言要同时看「在保存之前」和「中间不夹 stretch」——只看前者的话，
    把它留在左下角（stretch 之后、保存之前）也会过。
    """
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    block = source[source.index("btn_layout = QHBoxLayout()"):source.index("layout.addLayout(btn_layout)")]
    reset_at = block.index("btn_layout.addWidget(self.reset_all_btn)")
    save_at = block.index("btn_layout.addWidget(save_btn)")
    stretch_at = block.index("btn_layout.addStretch()")
    assert reset_at < save_at, "恢复默认设置要在保存左边"
    assert stretch_at < reset_at, "stretch 必须在两者之前，否则按钮还在左下角"
    # 版本号仍在左下角（stretch 之前）
    assert block.index("left_col.addWidget(self.version_label)") < stretch_at
    # 它已不再属于「左下角竖列」
    assert "left_col.addWidget(self.reset_all_btn)" not in block


def test_settings_pages_align_their_first_row():
    """快捷键页、参数页、质检忽略页的第一行必须处在同一高度。"""
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    # 三个页 group 的边距都要显式设为同一组值
    assert source.count("setContentsMargins(19, 14, 9, 9)") == 3
    # 快捷键页里 scroll 内容不能再叠加一层边距，否则首行会下移
    assert "scroll_layout.setContentsMargins(0, 0, 0, 0)" in source


def test_editor_exposes_reset_settings_to_default():
    from labelforge.ui.main_window import ImageEditor
    assert hasattr(ImageEditor, "reset_settings_to_default")


def test_reset_settings_to_default_reloads_and_refreshes(temp_config, monkeypatch):
    """reset 后必须重新加载设置并刷新界面，不能只删文件。"""
    source = inspect.getsource(
        __import__("labelforge.ui.main_window", fromlist=["ImageEditor"]).ImageEditor.reset_settings_to_default
    )
    assert "config_manager.reset_all()" in source
    assert "self._load_settings()" in source
    assert "self._apply_theme()" in source
    assert "self._refresh_ui_texts()" in source


# ---------- B-2 CLI ----------

def test_cli_parses_directory_positional():
    args = _parse_args(["D:/data"])
    assert args.path == "D:/data"
    assert args.labels is None


def test_cli_parses_labels_option():
    args = _parse_args(["D:/data", "--labels", "labels.txt"])
    assert args.path == "D:/data"
    assert args.labels == "labels.txt"


def test_cli_accepts_labels_without_directory():
    args = _parse_args(["--labels", "labels.txt"])
    assert args.path is None
    assert args.labels == "labels.txt"


def test_cli_defaults_are_empty():
    args = _parse_args([])
    assert args.path is None
    assert args.labels is None
    assert args.version is False


def test_cli_version_flag():
    args = _parse_args(["--version"])
    assert args.version is True


def test_cli_rejects_unknown_option():
    with pytest.raises(SystemExit):
        _parse_args(["--nope"])


def test_main_window_applies_startup_args():
    from labelforge.ui.main_window import ImageEditor
    source = inspect.getsource(ImageEditor.apply_startup_args)
    assert "load_background_folder" in source
    assert "load_background_label_file" in source
    # 路径不存在时给状态栏提示，不能静默失败
    assert "路径不存在" in source


def test_apply_startup_args_ignores_missing_paths():
    class Editor:
        def __init__(self):
            self.status_label = type("L", (), {"texts": [], "setText": lambda self, t: self.texts.append(t)})()
            self.folder_calls = []
            self.label_calls = []

        def load_background_folder(self, p):
            self.folder_calls.append(p)

        def load_background_label_file(self, p):
            self.label_calls.append(p)

    from labelforge.ui.main_window import ImageEditor

    editor = Editor()
    ImageEditor.apply_startup_args(editor, "Z:/definitely/missing/dir", "Z:/missing.txt")

    assert editor.folder_calls == []
    assert editor.label_calls == []
    assert len(editor.status_label.texts) == 2


def test_apply_startup_args_loads_existing_paths(tmp_path):
    folder = tmp_path / "bg"
    folder.mkdir()
    labels = tmp_path / "labels.txt"
    labels.write_text("cat\n", encoding="utf-8")

    class Editor:
        def __init__(self):
            self.status_label = type("L", (), {"setText": lambda self, t: None})()
            self.folder_calls = []
            self.label_calls = []

        def load_background_folder(self, p):
            self.folder_calls.append(p)

        def load_background_label_file(self, p):
            self.label_calls.append(p)

    from labelforge.ui.main_window import ImageEditor

    editor = Editor()
    ImageEditor.apply_startup_args(editor, str(folder), str(labels))

    assert editor.folder_calls == [str(folder)]
    assert editor.label_calls == [str(labels)]


def test_main_entry_point_parses_args_before_creating_qapplication():
    source = MAIN_WINDOW_SRC.read_text(encoding="utf-8")
    block = source[source.index("def main():"):]
    assert "_parse_args(sys.argv[1:])" in block
    assert block.index("_parse_args(sys.argv[1:])") < block.index("QApplication(")
    assert "editor.apply_startup_args(args.path, args.labels)" in block


def test_main_module_still_delegates_to_ui_main():
    source = MAIN_SRC.read_text(encoding="utf-8")
    assert "from labelforge.ui.main_window import main" in source


def test_cli_help_text_is_not_empty():
    import argparse
    from labelforge.ui.main_window import _parse_args
    parser_source = inspect.getsource(_parse_args)
    assert "argparse" in parser_source
    assert "--labels" in parser_source


# ---------- 版本号展示（左下角）与底部按钮顺序 ----------

def test_real_settings_dialog_shows_version_at_bottom_left():
    """真实控件：左下角显示 `LabelForge v<版本>`；重置按钮紧邻保存左边。

    不断言具体版本号：那会让每次发版都红一条测试。版本串的正确性由
    `test_real_settings_dialog_version_tracks_package_constant` 保证
    （它比对的就是包常量），这里只管「位置与格式」。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.settings_dialog import SettingsDialog
from labelforge import __version__

app = QApplication.instance() or QApplication([])
# 用真 ImageEditor 当宿主：SettingsDialog 的构造会读宿主的快捷方式输入框
# （prefix_input 等），裸 QWidget 不够。
window = ImageEditor()
d = SettingsDialog(window)
# 必须 show()：不显示时 Qt 不做布局，mapTo 拿到的坐标没有意义。
d.show()
app.processEvents()
label = d.version_label
# 不比对具体版本号（发版时会变），只比对「格式」与「取自包常量」
print(label.text().startswith("v"))
print(label.text() == f"v{__version__}")
# 版本号贴对话框左边缘（仍在左下角，没被挪走）
label_tl = label.mapTo(d, label.rect().topLeft())
print(label_tl.x() <= 12)
# 「恢复默认设置」紧邻「保存」左边：右边界不超过保存的左边界
reset_tl = d.reset_all_btn.mapTo(d, d.reset_all_btn.rect().topLeft())
save_btn = d.findChild(type(d.reset_all_btn), "successBtn")
save_tl = save_btn.mapTo(d, save_btn.rect().topLeft())
print(reset_tl.x() + d.reset_all_btn.width() <= save_tl.x())
# 且确实在保存左边（单独看一眼，防「顺序对了但被 stretch 顶到右边」）
print(reset_tl.x() < save_tl.x())
""")
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == "True", lines[1]
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", lines[3]
    assert lines[4] == "True", lines[4]


def test_real_settings_dialog_version_tracks_package_constant():
    """版本号的唯一来源是 `labelforge.__version__`：改常量即改显示。"""
    out = _run_real_qt("""
import labelforge
from PySide6.QtWidgets import QApplication
from labelforge.ui import settings_dialog

app = QApplication([])
original = labelforge.__version__
try:
    labelforge.__version__ = '9.9.9-test'
    d = settings_dialog.SettingsDialog(None)
    print(d.version_label.text())
finally:
    labelforge.__version__ = original
""")
    assert _lines(out)[0] == "v9.9.9-test", out
