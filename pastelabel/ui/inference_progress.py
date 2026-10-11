"""一键推理的进度小窗：不阻塞主窗口，可取消。

取消后已跑完的图结果仍然保留（worker 收到 cancel 后停止取下一张）。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
)

from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager


class InferenceProgressDialog(QDialog):
    """批量推理进度；取消按钮发 cancelled 信号。"""

    cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("推理中"))
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(380)
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_btn = QPushButton(tr("取消"))
        self.cancel_btn.clicked.connect(self._on_cancel)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['widget_bg']}; color: {theme['text_primary']}; }}
            QLabel {{ color: {theme['text_primary']}; }}
            QProgressBar {{ border: none; border-radius: 8px;
                background-color: {theme['scrollbar_bg']}; text-align: center;
                color: {theme['text_primary']}; height: 16px; }}
            QProgressBar::chunk {{ background-color: {theme['accent']}; border-radius: 8px; }}
            QPushButton {{ background-color: {theme['panel_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px; padding: 4px 10px; }}
            QPushButton:hover {{ border-color: {theme['accent']}; }}
        """)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())
        sync_titlebar(self)

    def refresh_theme(self):
        """主窗口切换主题后重新上色，并同步系统标题栏。"""
        self._apply_style()
        sync_titlebar(self)

    def set_total(self, total):
        self.progress.setMaximum(max(1, int(total)))
        self.progress.setValue(0)
        self.status_label.setText(f"0/{total}")
        self.cancel_btn.setEnabled(True)

    def set_progress(self, done, total, name):
        self.progress.setMaximum(max(1, int(total)))
        self.progress.setValue(int(done))
        self.status_label.setText(f"{done}/{total} · {name}")

    def reset(self):
        self.progress.setValue(0)
        self.status_label.setText("")
        self.cancel_btn.setEnabled(True)

    def _on_cancel(self):
        self.cancel_btn.setEnabled(False)
        self.status_label.setText(tr("正在取消..."))
        self.cancelled.emit()

    def closeEvent(self, event):
        """点窗口的 × 等同于取消，避免任务在后台无声跑完。"""
        if self.cancel_btn.isEnabled():
            self._on_cancel()
        super().closeEvent(event)
