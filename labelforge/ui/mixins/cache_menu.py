"""Cache Menu UI building mixin."""
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QLineEdit, QWidgetAction
from PyQt5.QtCore import Qt
from ..i18n import t as tr
from ...widgets.hover_menu import HoverKeepMenu


class CacheMenuMixin:
    """Cache Menu widgets and behavior."""

    def _rebuild_label_cache_menu(self):
        if not hasattr(self, 'cache_btn'):
            return

        if getattr(self, 'cache_menu', None) is None:
            menu = HoverKeepMenu(self)
            menu.setObjectName("cacheMenu")
            menu.setMinimumWidth(200)
            self.cache_menu = menu
            self.cache_btn.setMenu(menu)
        else:
            menu = self.cache_menu
            menu.clear()

        for index, slot in enumerate(getattr(self, 'label_cache_slots', [])):
            action = QWidgetAction(menu)
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(8, 4, 8, 4)
            row_layout.setSpacing(8)

            lock_btn = QPushButton(tr("上锁") if slot.get('locked') else tr("解锁"))
            lock_btn.setCursor(Qt.PointingHandCursor)
            lock_btn.clicked.connect(lambda checked=False, idx=index, btn=lock_btn: self._toggle_cache_slot_lock_from_popup(idx, btn))
            row_layout.addWidget(lock_btn, 0)

            copied_at = slot.get('copied_at') or '--:--:--'
            middle_widget = QWidget()
            middle_layout = QHBoxLayout(middle_widget)
            middle_layout.setContentsMargins(0, 0, 0, 0)
            middle_layout.setSpacing(2)

            slot_name_input = QLineEdit(slot.get('name', f"{tr('缓存槽')}{index + 1}"))
            slot_name_input.setObjectName("cacheSlotName")
            slot_name_input.setFrame(False)
            slot_name_input.setAttribute(Qt.WA_InputMethodEnabled, True)
            slot_name_input.setFixedWidth(slot_name_input.fontMetrics().horizontalAdvance("测" * 9) + 24)
            slot_name_input.setProperty("active", index == getattr(self, 'active_label_cache_slot', 0))
            slot_name_input.editingFinished.connect(
                lambda idx=index, field=slot_name_input: self._commit_cache_slot_name(idx, field)
            )
            middle_layout.addWidget(slot_name_input, 1)

            time_label = QLabel(copied_at)
            middle_layout.addWidget(time_label, 0)
            row_layout.addWidget(middle_widget, 1)

            shortcut_label = QLabel(slot.get('shortcut', ''))
            shortcut_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row_layout.addWidget(shortcut_label, 0)

            action.setDefaultWidget(row_widget)
            menu.addAction(action)

    def _handle_cache_slot_row_click(self, slot_index):
        self.set_active_label_cache_slot(slot_index)

    def _commit_cache_slot_name(self, slot_index, field):
        if slot_index < 0 or slot_index >= len(getattr(self, 'label_cache_slots', [])):
            return
        text = str(field.text() or '').strip()
        if not text:
            text = f"{tr('缓存槽')}{slot_index + 1}"
            field.setText(text)
        self.label_cache_slots[slot_index]['name'] = text
        self._save_label_cache_slots()

    def _toggle_cache_slot_lock_from_popup(self, slot_index, btn):
        if slot_index < 0 or slot_index >= len(getattr(self, 'label_cache_slots', [])):
            return
        self.label_cache_slots[slot_index]['locked'] = not self.label_cache_slots[slot_index].get('locked')
        self._save_label_cache_slots()
        btn.setText(tr("上锁") if self.label_cache_slots[slot_index].get('locked') else tr("解锁"))
