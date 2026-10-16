from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer, QRectF


class _SpinnerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._text = ""
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._rotate)
        self.setFixedSize(180, 60)

    def setText(self, text):
        self._text = text
        self.adjustSize()

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _rotate(self):
        self._angle = (self._angle + 10) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx, cy, r = 18, 30, 14
        pen = QPen(QColor(255, 255, 255), 3, Qt.SolidLine, Qt.RoundCap)
        p.setPen(pen)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2),
                  self._angle * 16, 90 * 16)
        bg = QColor(0, 0, 0, 160)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(40, 6, self.width() - 44, self.height() - 12), 10, 10)
        p.setPen(QColor(255, 255, 255))
        p.setFont(p.font())
        p.drawText(QRectF(42, 6, self.width() - 48, self.height() - 12),
                   Qt.AlignCenter, self._text)
        p.end()

    def adjustSize(self):
        fm = self.fontMetrics()
        tw = fm.horizontalAdvance(self._text) if hasattr(fm, 'horizontalAdvance') else fm.width(self._text)
        w = max(180, 50 + tw + 20)
        self.setFixedSize(w, 60)
