"""记忆记录弹窗。"""
import os

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QPushButton, QLabel
from PyQt5.QtCore import QTimer

from ..core import config_manager
from . import i18n
from .dialog_helpers import center_on_parent, get_text
from .dwm import set_titlebar_dark
from .theme import ThemeManager


class MemoryRecordsDialog(QDialog):
    """管理并加载最近的素材路径组合。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._editor = parent
        self.setWindowTitle(i18n.t("记忆记录"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(360)
        self._records = []

        layout = QVBoxLayout(self)
        self.info_label = QLabel(i18n.t("最多保留10组记录"))
        layout.addWidget(self.info_label)

        self.record_list = QListWidget()
        layout.addWidget(self.record_list)

        buttons = QHBoxLayout()
        self.load_btn = QPushButton(i18n.t("加载"))
        self.note_btn = QPushButton(i18n.t("修改备注"))
        self.delete_btn = QPushButton(i18n.t("删除"))
        self.close_btn = QPushButton(i18n.t("关闭"))
        for btn in (self.load_btn, self.note_btn, self.delete_btn, self.close_btn):
            btn.setFixedHeight(24)
            btn.setStyleSheet(ThemeManager.get_dialog_button_style())
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self.load_btn.clicked.connect(self._load_selected)
        self.note_btn.clicked.connect(self._edit_note)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.close_btn.clicked.connect(self.accept)
        self._refresh()

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self)
        is_dark = ThemeManager.get_mode().value == "dark"
        set_titlebar_dark(int(self.winId()), is_dark)

    def _refresh(self):
        self._records = config_manager.load_memory_records()
        self.record_list.clear()
        for record in self._records:
            note = record.get('note') or i18n.t("未备注")
            bg = self._format_memory_path(record.get('background_path'), os.path.isdir)
            paste = self._format_memory_path(record.get('paste_path'), os.path.isdir)
            label = self._format_memory_path(record.get('label_path'), os.path.isfile)
            image_index = int(record.get('background_index', 0) or 0) + 1
            edit_mode = record.get('edit_mode', 'paste')
            mode_text = i18n.t("贴图模式") if edit_mode == 'paste' else i18n.t("标注模式")
            self.record_list.addItem(
                f"{note}  [{mode_text}]\n"
                f"{i18n.t('当前图片')}: {image_index}\n"
                f"{i18n.t('背景图')}: {bg}\n"
                f"{i18n.t('贴图')}: {paste}\n"
                f"{i18n.t('标签文件')}: {label}"
            )

    def _format_memory_path(self, path, validator):
        """格式化记忆路径，并在无效路径右侧标记状态。"""
        path = path or ''
        if not path:
            return i18n.t("空")
        return path if validator(path) else f"{path}  <{i18n.t('路径不存在')}>"

    def _selected_index(self):
        row = self.record_list.currentRow()
        return row if 0 <= row < len(self._records) else -1

    def _load_selected(self):
        idx = self._selected_index()
        if idx >= 0 and self._editor:
            record = self._records[idx]
            self.accept()
            QTimer.singleShot(0, lambda: self._editor.load_memory_record(record))

    def _edit_note(self):
        idx = self._selected_index()
        if idx < 0:
            return
        record = self._records[idx].copy()
        note, ok = get_text(self, "修改备注", "请输入备注:", text=record.get('note', ''))
        if ok:
            record['note'] = note.strip()
            config_manager.upsert_memory_record(record)
            self._refresh()
            self.record_list.setCurrentRow(0)

    def _delete_selected(self):
        idx = self._selected_index()
        if idx >= 0:
            config_manager.delete_memory_record(idx)
            self._refresh()
