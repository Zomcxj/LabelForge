"""全局异常提示回归测试。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_exception_dialog_uses_localized_ok_and_show_details_buttons():
    source = (ROOT / "labelforge" / "core" / "exception_hook.py").read_text(encoding="utf-8")
    helper_source = (ROOT / "labelforge" / "ui" / "dialog_helpers.py").read_text(encoding="utf-8")

    assert 'from ..ui.dialog_helpers import ThemedMessageBox' in source
    assert 'msg_box = ThemedMessageBox()' in source
    assert 'msg_box.setStandardButtons(QMessageBox.Ok)' in source
    assert 'msg_box.button(QMessageBox.Ok).setText(i18n.t("确定"))' in source
    assert 'msg_box.setDetailedText(error_msg)' in source
    assert 'btn.setText(i18n.t("显示详情"))' in helper_source
    assert 'btn.setText(i18n.t("隐藏详情"))' in helper_source


def test_exception_log_uses_user_config_directory_and_labelforge_name():
    source = (ROOT / "labelforge" / "core" / "exception_hook.py").read_text(encoding="utf-8")

    assert 'config_manager.CONFIG_PATH' in source
    assert '"labelforge.log"' in source
    assert 'crash_log.txt' not in source
