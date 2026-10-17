"""类别筛选面板：勾选要保留的类别，供批量推理过滤使用。

默认全选 = 全部保留；没勾选的会被过滤掉。「全不选」表示什么都不要。
面板始终回传一个具体集合，是否真的过滤由调用方结合类别总数判断——
全部勾选等于没有限制，不必走过滤逻辑。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager


class ClassFilterPanel(QDialog):
    """类别勾选面板；确定后 selection_changed 带上选中的类别名。"""

    selection_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("类别筛选"))
        self.setMinimumSize(360, 460)
        self._boxes = []
        self._build_ui()
        self._apply_style()
        self.selected = set()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.hint_label = QLabel(tr("保留勾选的类别；没勾选的会被过滤掉"))
        self.hint_label.setWordWrap(True)
        self.hint_label.setObjectName("classFilterHint")
        layout.addWidget(self.hint_label)

        self._area = QScrollArea()
        self._area.setWidgetResizable(True)
        self._content = QWidget()
        self._list_layout = QVBoxLayout(self._content)
        self._list_layout.setContentsMargins(2, 2, 2, 2)
        self._list_layout.setSpacing(2)
        self._list_layout.addStretch()
        self._area.setWidget(self._content)
        layout.addWidget(self._area, 1)

        top_row = QHBoxLayout()
        self.all_btn = QPushButton(tr("全选"))
        self.all_btn.clicked.connect(lambda: self._set_all(True))
        self.none_btn = QPushButton(tr("全不选"))
        self.none_btn.clicked.connect(lambda: self._set_all(False))
        top_row.addWidget(self.all_btn)
        top_row.addWidget(self.none_btn)
        top_row.addStretch()
        layout.addLayout(top_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self)
        self.ok_btn = buttons.button(QDialogButtonBox.Ok)
        self.cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        self.ok_btn.setText(tr("确定"))
        self.cancel_btn.setText(tr("取消"))
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['widget_bg']}; color: {theme['text_primary']}; }}
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#classFilterHint {{ color: {theme['text_secondary']}; }}
            QScrollArea {{ border: 1px solid {theme['border_color']}; border-radius: 4px; }}
            QCheckBox {{ color: {theme['text_primary']}; padding: 2px; }}
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

    def set_class_names(self, names):
        """重建勾选框列表，尽量保留当前选择；首次加载默认全选（= 不过滤）。"""
        keep = {b.text() for b in self._boxes if b.isChecked()}
        first_fill = not self._boxes
        for box in self._boxes:
            self._list_layout.removeWidget(box)
            box.deleteLater()
        self._boxes = []
        for name in names or []:
            box = QCheckBox(str(name))
            box.setChecked(True if first_fill else str(name) in keep)
            self._boxes.append(box)
            self._list_layout.insertWidget(len(self._boxes) - 1, box)

    def current_selection(self):
        return {b.text() for b in self._boxes if b.isChecked()}

    def set_selection(self, names):
        wanted = {str(n) for n in (names or [])}
        for box in self._boxes:
            box.setChecked(box.text() in wanted)

    def retranslate(self):
        """切语言后刷新面板文字（勾选项是类别名，不翻译）。"""
        self.setWindowTitle(tr("类别筛选"))
        self.hint_label.setText(tr("保留勾选的类别；没勾选的会被过滤掉"))
        self.all_btn.setText(tr("全选"))
        self.none_btn.setText(tr("全不选"))
        self.ok_btn.setText(tr("确定"))
        self.cancel_btn.setText(tr("取消"))

    def _set_all(self, checked):
        for box in self._boxes:
            box.setChecked(checked)

    def _on_accept(self):
        self.selected = self.current_selection()
        self.selection_changed.emit(set(self.selected))
        self.accept()
