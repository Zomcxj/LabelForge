"""分段按钮动画控件（PyQt5版本，参考ScanVerse）"""

from PyQt5.QtWidgets import QFrame, QWidget, QPushButton
from PyQt5.QtCore import Qt, QPropertyAnimation, QEasingCurve, QRect


class AnimatedSegmentedControl(QFrame):
    """带滑动指示器的分段按钮控件"""

    def __init__(self, container, *buttons):
        super().__init__(container.parent())
        self._container = container
        self._buttons = list(buttons)
        self._accent = "#2196F3"

        self._indicator = QWidget(container)
        self._indicator.lower()
        self._indicator.setStyleSheet(
            f"background: {self._accent}; border-radius: 4px;")

        self._anim = QPropertyAnimation(self._indicator, b"geometry")
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        for btn in self._buttons:
            btn.clicked.connect(lambda: self.update_position(animated=True))

    def set_accent(self, color):
        self._accent = color
        self._indicator.setStyleSheet(
            f"background: {color}; border-radius: 4px;")

    def update_position(self, animated=True):
        checked = None
        for i, btn in enumerate(self._buttons):
            if btn.isChecked():
                checked = btn
                break
        if checked is None:
            if not self._buttons:
                return
            checked = self._buttons[0]

        cr = self._container.contentsRect()
        y_offset = cr.y()
        content_w = cr.width() or sum(btn.width() for btn in self._buttons)
        w = content_w // len(self._buttons)
        h = cr.height() or checked.height() or self._container.height() or 24

        index = self._buttons.index(checked)
        x = cr.x() + index * w
        if index == len(self._buttons) - 1:
            w = cr.right() - x + 1
        y = y_offset
        target = QRect(x, y, w, h)
        cur = self._indicator.geometry()

        if animated and cur.isValid() and cur != target:
            self._anim.stop()
            self._anim.setStartValue(cur)
            self._anim.setEndValue(target)
            self._anim.start()
        else:
            self._indicator.setGeometry(target)
