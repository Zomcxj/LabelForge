"""标签列表行右侧的标注任务徽标（det / seg / pose / obb）。"""
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QColor, QPainter, QPen, QFontMetrics
from PyQt5.QtWidgets import QStyledItemDelegate, QStyle

from ..core.utils import TASK_DATA_ROLE

TASK_BADGE_COLORS = {
    "det": "#1E88E5",
    "seg": "#43A047",
    "pose": "#FB8C00",
    "obb": "#8E24AA",
}


class TaskBadgeDelegate(QStyledItemDelegate):
    """在列表项文字右侧绘制任务徽标，不改动原文本。"""

    PADDING = 4
    MARGIN = 4

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        raw = index.data(TASK_DATA_ROLE)
        if not raw:
            return
        tasks = str(raw).split()
        if not tasks:
            return

        painter.save()
        font = painter.font()
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_h = metrics.height()
        rect = option.rect
        painter.setRenderHint(QPainter.Antialiasing, True)

        right = rect.right() - self.MARGIN
        for text in reversed(tasks):
            color = QColor(TASK_BADGE_COLORS.get(text, "#757575"))
            text_w = metrics.horizontalAdvance(text)
            badge = QRect(
                right - text_w - self.PADDING * 2,
                rect.center().y() - text_h // 2,
                text_w + self.PADDING * 2,
                text_h,
            )
            if badge.left() < rect.left():
                break
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(badge, 3, 3)
            painter.setPen(QPen(QColor("#FFFFFF")))
            painter.drawText(badge, Qt.AlignCenter, text)
            right = badge.left() - self.MARGIN
        painter.restore()

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        raw = index.data(TASK_DATA_ROLE)
        if raw:
            return QSize(size.width() + 40 * len(str(raw).split()), max(size.height(), 20))
        return size
