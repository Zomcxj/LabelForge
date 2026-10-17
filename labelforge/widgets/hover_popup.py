from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QApplication, QLineEdit


class HoverDismissPopup(QWidget):
    """鼠标移出后自动收起的轻量弹层。

    圆角必须在这里开 `WA_TranslucentBackground`：只写 QSS 的
    `border-radius` 是不够的，Qt 会给顶层窗口铺一块不透明的方角背景，
    圆角被盖住（实测四角 alpha 仍是 255，底边看着就是直角）。
    透明背景关掉那块方底，QSS 画的圆角才真的切得出来——与
    `canvas_menu._rounded_menu()` 处理 QMenu 的做法一致。
    """

    def __init__(self, parent=None, flags=None):
        # `flags` 默认值不能直接写 `Qt.Popup`：默认参数在**类定义时**求值，
        # 而测试桩（`tests/conftest.py` 的假 Qt）只注册了用到的那几个枚举名，
        # 没有 `Popup`，会以 `AttributeError` 让导入就失败。放到函数体里取。
        if flags is None:
            flags = Qt.Popup
        super().__init__(parent, flags)
        # 透明背景：关掉 Qt 给顶层窗口铺的那块不透明方角底，QSS 的圆角才切得出来。
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 样式背景：裸 QWidget 默认不画 QSS 的 background-color（会变成全透明），
        # 必须显式打开才会让 `QWidget#optionsPopup` 那条规则真的着色。
        self.setAttribute(Qt.WA_StyledBackground, True)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, QLineEdit) and self.isAncestorOf(focus_widget):
            return
        self.hide()
