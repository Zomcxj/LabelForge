import os
from PyQt5.QtWidgets import QListWidget
from PyQt5.QtGui import QDrag
from PyQt5.QtCore import Qt, QMimeData, QUrl


class DragOutListWidget(QListWidget):
    """支持拖出文件的列表控件"""
    _drag_start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start_pos and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._drag_start_pos
            if abs(delta.x()) > 20 or abs(delta.y()) > 20:
                item = self.itemAt(self._drag_start_pos)
                if item:
                    file_path = item.data(Qt.UserRole + 1)
                    if file_path and os.path.isfile(file_path):
                        drag = QDrag(self)
                        mime = QMimeData()
                        mime.setUrls([QUrl.fromLocalFile(file_path)])
                        drag.setMimeData(mime)
                        drag.exec_(Qt.CopyAction)
                        self._drag_start_pos = None
                        return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)
