"""一键推理的进度小窗：不阻塞主窗口，可取消。

取消后已跑完的图结果仍然保留（worker 收到 cancel 后停止取下一张）。
日志滚动显示，只留最近若干行；用户往上翻看历史时新结果不会把他拽回底部。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTextEdit,
)

from ..engine.inference_task import beijing_timestamp
from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager

#: 日志区保留的行数；超出丢弃最旧的
LOG_LINES = 5


class InferenceProgressDialog(QDialog):
    """批量推理进度；取消按钮发 cancelled 信号。"""

    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("推理中"))
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setMinimumWidth(520)
        self._total_boxes = 0
        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.summary_label = QLabel("")
        self.summary_label.setObjectName("inferenceSummary")
        layout.addWidget(self.summary_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setObjectName("logArea")
        # 只保留最后几行，避免整批推理把内存和重绘都拖垮
        self.log_area.document().setMaximumBlockCount(LOG_LINES)
        self.log_area.setFixedHeight(96)
        layout.addWidget(self.log_area)

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
            QLabel#inferenceSummary {{ color: {theme['text_secondary']}; }}
            {ThemeManager.get_progress_style()}
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
        self._total_boxes = 0
        self.status_label.setText(f"0/{total}")
        self.summary_label.setText("")
        self.log_area.clear()
        self.cancel_btn.setEnabled(True)

    def set_progress(self, done, total, name, boxes=None, elapsed_ms=None):
        """更新进度并追加一行日志。

        boxes/elapsed_ms 是这一张自己的结果，缺省时只更新进度条
        （单张推理没有日志区需求，走同一条路径也不会出错）。
        """
        self.progress.setMaximum(max(1, int(total)))
        self.progress.setValue(int(done))
        self.status_label.setText(f"{done}/{total} · {name}")
        if boxes is None:
            return
        self._total_boxes += int(boxes)
        self.summary_label.setText(
            f"{tr('累计检出')}: {self._total_boxes} {tr('个框')}")
        cost = "" if elapsed_ms is None else f" · {int(elapsed_ms)}ms"
        self._append_log(
            f"[{beijing_timestamp()}] {name} · {int(boxes)} {tr('个框')}{cost}")

    def _append_log(self, line):
        """追加一行；用户已在翻历史时不把他拽回底部。"""
        bar = self.log_area.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 2
        self.log_area.append(line)
        if at_bottom:
            bar.setValue(bar.maximum())

    def reset(self):
        self.progress.setValue(0)
        self.status_label.setText("")
        self.summary_label.setText("")
        self.log_area.clear()
        self._total_boxes = 0
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
