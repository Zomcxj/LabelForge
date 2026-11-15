"""设置页「抠图」与「贴图融合」：读写走既有配置机制，界面是真的。

分两层：配置读写用进程内的假 PySide6 直接验；控件接线在真 Qt offscreen
子进程里验（假桩没有布局与真控件值，`currentData()` 之类拿不到东西）。
"""
import json
import os
import sys
from pathlib import Path

import pytest

from labelforge.core import config as config_module
from labelforge.core import config_manager
from labelforge.core.config import MATTING_CONFIG, PASTE_FUSION_CONFIG

ROOT = Path(__file__).resolve().parents[1]
SETTINGS_SRC = ROOT / "labelforge" / "ui" / "settings_dialog.py"
MAIN_WINDOW_SRC = ROOT / "labelforge" / "ui" / "main_window.py"

_GUARD = """
import sys
sys.stdout.flush()
sys.exit(0)
"""


def _run_real_qt(script):
    """在真 Qt offscreen 子进程里跑脚本（与 test_settings_reset_and_cli 同法）。"""
    import subprocess

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180,
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


# ---------- 配置读写 ----------

def test_load_all_matte_defaults_match_the_config_dicts(temp_config):
    loaded = config_manager.load_all()

    assert loaded['matte_model_path'] == MATTING_CONFIG['model_path']
    assert loaded['matte_preset'] == MATTING_CONFIG['preset']
    assert loaded['matte_input_size'] == MATTING_CONFIG['input_size']
    assert loaded['matte_threshold'] == MATTING_CONFIG['threshold']
    assert loaded['matte_output_dir'] == MATTING_CONFIG['output_dir']
    assert loaded['seamless_paste'] is PASTE_FUSION_CONFIG['seamless']


def test_save_all_round_trips_the_matte_settings(temp_config, tmp_path):
    out_dir = str(tmp_path / "matte_out")

    assert config_manager.save_all(
        matte_model_path="C:/models/rmbg14.onnx",
        matte_preset="rmbg14",
        matte_input_size=512,
        matte_threshold=0.4,
        matte_output_dir=out_dir,
        seamless_paste=True,
    )

    loaded = config_manager.load_all()
    assert loaded['matte_model_path'] == "C:/models/rmbg14.onnx"
    assert loaded['matte_preset'] == "rmbg14"
    assert loaded['matte_input_size'] == 512
    assert loaded['matte_threshold'] == 0.4
    assert loaded['matte_output_dir'] == out_dir
    assert loaded['seamless_paste'] is True
    # 真的写进了文件，不只是内存
    on_disk = json.loads(temp_config.read_text(encoding="utf-8"))
    assert on_disk['matte_threshold'] == 0.4
    assert on_disk['seamless_paste'] is True


def test_save_all_clamps_the_matte_values(temp_config):
    config_manager.save_all(matte_threshold=-3.0, matte_input_size=-64,
                            matte_model_path="  ", matte_output_dir="  ")
    loaded = config_manager.load_all()
    assert loaded['matte_threshold'] == 0.0
    assert loaded['matte_input_size'] == 0
    assert loaded['matte_model_path'] == ''
    assert loaded['matte_output_dir'] == ''

    config_manager.save_all(matte_threshold=9.5)
    assert config_manager.load_all()['matte_threshold'] == 1.0


def test_save_all_drops_an_unknown_matte_preset(temp_config):
    """未知预设会让模型构造抛 ValueError、入口置灰，读/写时就应纠正。"""
    config_manager.save_all(matte_preset="not-a-preset")

    assert config_manager.load_all()['matte_preset'] == MATTING_CONFIG['preset']


def test_seamless_paste_is_read_from_the_configured_file(temp_config):
    temp_config.write_text(json.dumps({"seamless_paste": True}),
                           encoding="utf-8")

    assert config_manager.load_all()['seamless_paste'] is True


def test_load_settings_applies_matte_and_fusion_config(temp_config):
    """主窗口启动时把配置回填到模块字典——推理与保存读到的就是这一份。"""
    from types import SimpleNamespace

    from labelforge.ui.main_window import ImageEditor

    temp_config.write_text(json.dumps({
        "matte_model_path": "C:/models/rmbg14.onnx",
        "matte_preset": "rmbg14",
        "matte_input_size": 768,
        "matte_threshold": 0.25,
        "matte_output_dir": "C:/matte_out",
        "seamless_paste": True,
    }), encoding="utf-8")

    snapshot_matte = dict(MATTING_CONFIG)
    snapshot_fusion = dict(PASTE_FUSION_CONFIG)
    try:
        editor = SimpleNamespace()
        ImageEditor._load_settings(editor)

        assert MATTING_CONFIG['model_path'] == "C:/models/rmbg14.onnx"
        # 配置里存的是已删除的预设：加载时归一到 rmbg14
        assert MATTING_CONFIG['preset'] == "rmbg14"
        assert MATTING_CONFIG['input_size'] == 768
        assert MATTING_CONFIG['threshold'] == 0.25
        assert MATTING_CONFIG['output_dir'] == "C:/matte_out"
        assert PASTE_FUSION_CONFIG['seamless'] is True
    finally:
        MATTING_CONFIG.clear()
        MATTING_CONFIG.update(snapshot_matte)
        PASTE_FUSION_CONFIG.clear()
        PASTE_FUSION_CONFIG.update(snapshot_fusion)


# ---------- 设置页接线（源码级：控件存在且绑定到配置） ----------

def test_settings_dialog_has_a_matting_page():
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    assert 'self.matte_page_btn = QPushButton(tr("抠图"))' in source
    assert "self.matte_page_btn.clicked.connect(lambda: self._switch_page(3))" in source
    assert "self.matte_page_btn.setChecked(index == 3)" in source
    assert "self.stack.addWidget(matte_page)" in source


def test_matting_page_carries_the_required_controls():
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    for attribute in ("self.matte_path_input", "self.matte_path_btn",
                      "self.matte_download_btn", "self.matte_input_size_spin",
                      "self.matte_threshold_spin",
                      "self.matte_output_dir_input",
                      "self.matte_output_dir_btn",
                      "self.matte_benchmark_btn"):
        assert attribute in source, attribute
    assert '"ONNX (*.onnx)"' in source
    # 只支持 RMBG-1.4：预设下拉已经删掉
    assert "matte_preset_combo" not in source
    assert "BiRefNet" not in source


def test_seamless_checkbox_is_bound_to_the_global_fusion_switch():
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    assert "self.seamless_checkbox = QCheckBox(tr(\"泊松融合\"))" in source
    assert "PASTE_FUSION_CONFIG['seamless'] = matte_settings['seamless_paste']" in source
    assert "seamless_paste" in config_manager.save_all.__code__.co_varnames


def test_settings_save_rebuilds_the_matting_model():
    source = SETTINGS_SRC.read_text(encoding="utf-8")
    main_source = MAIN_WINDOW_SRC.read_text(encoding="utf-8")

    assert "self._editor._rebuild_matting_model()" in source
    assert "def _rebuild_matting_model(self):" in main_source
    # 重建后要么有模型、要么清成 None（入口据此置灰）
    assert "self._matting_model = None" in main_source


def test_matting_defaults_in_the_dialog_come_from_the_config_dicts():
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    assert "MATTING_CONFIG.get('model_path')" in source
    assert "MATTING_CONFIG.get('output_dir')" in source
    assert "PASTE_FUSION_CONFIG.get('seamless')" in source


# ---------- 真 Qt：控件值 ↔ 配置文件来回一趟 ----------

def test_real_settings_dialog_round_trips_the_matting_page(tmp_path):
    weight = tmp_path / "rmbg14.onnx"
    weight.write_bytes(b"")
    config_path = tmp_path / "config.json"
    out_dir = tmp_path / "matte_out"
    out_dir.mkdir()
    config_path.write_text(json.dumps({
        "matte_model_path": str(weight),
        "matte_preset": "rmbg14",
        "matte_input_size": 640,
        "matte_threshold": 0.35,
        "matte_output_dir": str(out_dir),
        "seamless_paste": True,
    }), encoding="utf-8")

    out = _run_real_qt(f"""
import os
os.environ['LABELFORGE_CONFIG_PATH'] = r"{config_path}"
from PySide6.QtWidgets import QApplication
from labelforge.core import config_manager
from labelforge.core.config import MATTING_CONFIG, PASTE_FUSION_CONFIG
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.settings_dialog import SettingsDialog

app = QApplication.instance() or QApplication([])
window = ImageEditor()
d = SettingsDialog(window)
d.show()
app.processEvents()

# 页面存在且能切过去
d._switch_page(3)
print(d.stack.currentIndex() == 3)
print(d.matte_page_btn.isChecked())

# 控件值来自配置文件（而不是出厂默认）
print(d.matte_threshold_spin.value() == 0.35)
print(d.matte_input_size_spin.value() == 640)
print(d.matte_path_input.text() == r"{weight}")
print(d.matte_output_dir_input.text() == r"{out_dir}")
print(d.seamless_checkbox.isChecked() is True)

# 改控件 -> 保存 -> 配置字典与文件都要跟着变
d.matte_threshold_spin.setValue(0.6)
d.matte_output_dir_input.setText("")
d.seamless_checkbox.setChecked(False)
d._save_shortcuts()

print(MATTING_CONFIG['threshold'] == 0.6)
print(MATTING_CONFIG['output_dir'] == "")
print(PASTE_FUSION_CONFIG['seamless'] is False)
loaded = config_manager.load_all()
print(loaded['matte_threshold'] == 0.6)
print(loaded['seamless_paste'] is False)
# 权重文件存在 -> 保存后模型实例已重建，抠图入口不再是灰的
print(window.matting_model is not None)
""")

    lines = _lines(out)
    assert lines == ["True"] * 13, lines


def test_matting_path_rows_are_wide_enough_to_show_a_file_name():
    """权重路径与输出目录要看得见文件名：输入框最小宽度放宽、标签不再按比例挤压。"""
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    assert "self.matte_path_input.setMinimumWidth(300)" in source
    assert "self.matte_output_dir_input.setMinimumWidth(300)" in source
    # 标签不再带 2 的伸缩因子，输入框占满剩余宽度
    assert 'matte_path_row.addWidget(QLabel(tr("权重路径") + ":"))\n' in source
    assert 'matte_out_row.addWidget(QLabel(tr("输出目录") + ":"))\n' in source
    assert "matte_path_row.addWidget(self.matte_path_input, 1)" in source
    assert "matte_out_row.addWidget(self.matte_output_dir_input, 1)" in source


def test_fusion_controls_are_gone_from_the_paste_panel():
    """羽化/调色/投影已整行删除：面板源码里不应再出现这三个控件。"""
    lists_src = (ROOT / "labelforge" / "ui" / "mixins" / "lists.py").read_text(
        encoding="utf-8")

    for gone in ("feather_spin", "color_match_spin", "shadow_spin",
                 "fusion_apply_all_btn", "_apply_fusion_to_selection"):
        assert gone not in lists_src, gone


def test_matting_page_offers_a_weight_download_button():
    """权重不随包分发：设置页要有一键下载。"""
    source = SETTINGS_SRC.read_text(encoding="utf-8")

    assert 'self.matte_download_btn = QPushButton(tr("下载权重"))' in source
    assert "self._download_matte_weight" in source
    assert "WeightDownloadWorker" in source
    # 下载目标与校验值来自配置表，不写死在界面里
    assert "WEIGHT_DOWNLOAD.get('rmbg14')" in source
    assert "preset_weight_path('rmbg14')" in source


def test_weight_download_points_at_the_official_mirror():
    from labelforge.engine.matting_model import WEIGHT_DOWNLOAD, MATTING_PRESETS

    info = WEIGHT_DOWNLOAD["rmbg14"]
    assert info["url"].startswith("https://hf-mirror.com/briaai/RMBG-1.4/")
    assert info["url"].endswith("onnx/model.onnx")
    assert info["size"] == 176153355
    assert len(info["sha256"]) == 64
    # 校验值必须和内置预设对应（同一个模型）
    assert set(WEIGHT_DOWNLOAD) == set(MATTING_PRESETS)


def test_download_button_does_not_refetch_an_existing_weight(tmp_path):
    """已填路径或默认位置上有权重时，点「下载权重」只提示，不重新下载。"""
    weight = tmp_path / "rmbg14.onnx"
    weight.write_bytes(b"")
    missing = tmp_path / "absent" / "rmbg14.onnx"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"matte_model_path": str(weight)}),
                           encoding="utf-8")

    out = _run_real_qt(f"""
import os
os.environ['LABELFORGE_CONFIG_PATH'] = r"{config_path}"
from PySide6.QtWidgets import QApplication
import labelforge.engine.matting_model as matting_model
from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.settings_dialog import SettingsDialog

app = QApplication.instance() or QApplication([])
window = ImageEditor()
d = SettingsDialog(window)
d.show()
app.processEvents()

# 1) 输入框里的路径已存在，默认位置不存在：不该发起下载
d.matte_path_input.setText(r"{weight}")
matting_model.preset_weight_path = lambda preset, model_dir=None: r"{missing}"
d._download_matte_weight()
app.processEvents()
print(d.matte_status_label.text() == i18n.t("权重已就位"))
print(getattr(d, "_weight_worker", None) is None)

# 2) 输入框为空、默认位置已存在：填进输入框并提示，同样不下载
d.matte_path_input.setText("")
matting_model.preset_weight_path = lambda preset, model_dir=None: r"{weight}"
d._download_matte_weight()
app.processEvents()
print(d.matte_status_label.text() == i18n.t("权重已就位"))
print(d.matte_path_input.text() == r"{weight}")
print(getattr(d, "_weight_worker", None) is None)
""")
    lines = _lines(out)
    assert lines == ["True"] * 5, lines
