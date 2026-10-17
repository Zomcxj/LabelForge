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


def test_benign_qt6_pixel_font_warning_is_dropped_but_others_are_logged():
    """QSS 的 `font-size: Npx` 让 Qt6 每次启动报一条 setPointSize 警告。

    那条是无害的（字号本身正确，Qt 只是拒绝了内部算出的 -1），但会写进用户
    报障时发给我们的那份日志。这里钉住：它被丢掉，其它 Qt 消息照记。
    """
    from labelforge.core import exception_hook as eh

    logged = []
    original = eh._write_log
    eh._write_log = logged.append
    try:
        eh._qt_message_handler(
            1, None,
            "QFont::setPointSize: Point size <= 0 (-1), must be greater than 0")
        eh._qt_message_handler(1, None, "Something else went wrong")
    finally:
        eh._write_log = original

    assert len(logged) == 1, logged
    assert "Something else went wrong" in logged[0]
