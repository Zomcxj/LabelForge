from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget


class _ScanSpinnerWidget(QLabel):
    """内嵌 braille 旋转动画的扫描提示组件"""
    _CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._idx = 0
        self._raw_text = ""
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._tick)
        self.setStyleSheet("color: gray; font-size: 11px; padding: 2px;")
        self.setFixedHeight(20)

    def setLabel(self, t):
        self._raw_text = t
        self._idx = 0
        self._refresh()

    def setVisible(self, v):
        super().setVisible(v)
        if v:
            self._idx = 0
            self._timer.start()
            self._refresh()
        else:
            self._timer.stop()

    def _tick(self):
        self._idx += 1
        self._refresh()

    def _refresh(self):
        c = self._CHARS[self._idx % len(self._CHARS)]
        super().setText(f"{c} {self._raw_text}" if self._raw_text else c)


class Worker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int, str)
    result_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func):
        super().__init__()
        self._func = func
        self.result = None
        self._error = None

    def run(self):
        try:
            self.result = self._func(self.log.emit, self.progress.emit)
            self.result_ready.emit(self.result)
        except Exception as e:
            self._error = str(e)
            self.error.emit(str(e))


class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent=None, color: str = "#4CAF50"):
        super().__init__(parent)
        self._expanded = True
        self._title = title
        self._color = color
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(0)
        bar = QWidget()
        bar.setFixedWidth(3)
        bar.setStyleSheet(f"background: {color};")
        outer.addWidget(bar)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(0)
        self._header = QPushButton(f"▼ {title}")
        self._header.setObjectName("collapseHeader")
        self._header.setFlat(True)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setFixedHeight(24)
        self._header.setStyleSheet("border: none; text-align: left; padding: 2px 8px;")
        self._header.clicked.connect(self._toggle)
        self._content = QWidget()
        self._content.setObjectName("collapseContent")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 2, 8, 4)
        inner_layout.addWidget(self._header)
        inner_layout.addWidget(self._content)
        outer.addWidget(inner, 1)

    def _toggle(self):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._header.setText(f"{'▼' if self._expanded else '▶'} {self._title}")
        self._update_margins()

    def _update_margins(self):
        outer = self.layout()
        if outer:
            if self._expanded:
                outer.setContentsMargins(0, 2, 0, 2)
                outer.setSpacing(0)
            else:
                outer.setContentsMargins(0, 0, 0, 0)
                outer.setSpacing(0)

    def set_title(self, title: str):
        self._title = title
        self._header.setText(f"{'▼' if self._expanded else '▶'} {title}")

    def content_layout(self):
        return self._content_layout
