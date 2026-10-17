"""
全局异常捕获模块 - 捕获所有未处理的 Python 异常并写入日志文件
"""
import sys
import os
import traceback
from datetime import datetime
from . import config_manager


LOG_FILE = os.path.join(os.path.dirname(config_manager.CONFIG_PATH), "labelforge.log")


def _write_log(message):
    """写入日志文件"""
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass  # 无法写入日志时忽略


# 已知无害、但每次启动都会写一行的 Qt6 告警：像素字号（QSS 的 `font-size: Npx`）
# 会让 QFont.pointSize() 返回 -1，Qt6 内部再拿它调 setPointSize(-1) 并告警。
# 字号本身是对的（pixelSize 正常、渲染不变），Qt 只是拒绝了非法值。PyQt5 时代没有。
# 每次进程只发生一次；不记日志是因为这份日志是用户报障时发给我们的。
_IGNORED_QT_MESSAGES = (
    "QFont::setPointSize: Point size <= 0",
)


def _qt_message_handler(mode, context, message):
    """Qt 消息处理器 - 捕获 Qt 级别的警告/错误"""
    if any(s in message for s in _IGNORED_QT_MESSAGES):
        return
    _write_log(f"Qt [{mode}]: {message}")


def install_exception_hook():
    """安装全局异常钩子"""
    sys.excepthook = exception_hook
    # 安装 Qt 消息处理器
    try:
        from PySide6.QtCore import qInstallMessageHandler
        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        pass


def exception_hook(exctype, value, tb):
    """全局异常处理"""
    try:
        error_msg = "".join(traceback.format_exception(exctype, value, tb))
    except Exception:
        error_msg = f"{exctype.__name__}: {value}"
    _write_log(f"未捕获的异常:\n{error_msg}")
    
    # 尝试使用 QMessageBox 显示错误
    try:
        from PySide6.QtWidgets import QMessageBox, QApplication
        from ..ui import i18n
        from ..ui.dialog_helpers import ThemedMessageBox
        app = QApplication.instance()
        if app:
            msg_box = ThemedMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle(i18n.t("程序错误"))
            msg_box.setText(i18n.t("程序遇到意外错误，错误信息已保存到 labelforge.log"))
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.button(QMessageBox.Ok).setText(i18n.t("确定"))
            msg_box.setDetailedText(error_msg)
            msg_box.exec_()
    except Exception:
        pass
    
    # 调用原始的 excepthook
    try:
        sys.__excepthook__(exctype, value, tb)
    except Exception:
        pass
