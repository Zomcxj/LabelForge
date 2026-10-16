from PyQt5.QtWidgets import QMenu


class HoverKeepMenu(QMenu):
    """点击菜单项不关闭，鼠标移开后关闭。

    first_action_momentary=True 时首项视为瞬时动作（点击只发 triggered(False)
    而不切换勾选），供"绘制BOX"这类动作使用；勾选类菜单传 False。
    """

    def __init__(self, parent=None, first_action_momentary=True):
        super().__init__(parent)
        self._first_action_momentary = first_action_momentary

    def mouseReleaseEvent(self, event):
        action = self.actionAt(event.pos())
        if action and action.isEnabled():
            if self._first_action_momentary and action == self.actions()[0]:
                action.triggered.emit(False)
                event.accept()
                return
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.hide()
