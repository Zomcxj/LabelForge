"""Label cache slot behavior for the main window."""

from datetime import datetime

from PySide6.QtCore import QRectF

from ..i18n import t as tr


class LabelCacheSlotMixin:
    def _save_label_cache_slots(self):
        from ...core import config_manager
        config_manager.save_all(label_cache_slots=self.label_cache_slots)

    def _get_next_writable_label_cache_slot_index(self):
        writable_slots = [
            (index, slot) for index, slot in enumerate(self.label_cache_slots)
            if not slot.get('locked')
        ]
        if not writable_slots:
            return None
        return min(
            writable_slots,
            key=lambda item: int(item[1].get('copy_order', 0) or 0),
        )[0]

    def _reset_label_cache_slots(self):
        for index, slot in enumerate(self.label_cache_slots):
            default_shortcut = str(index + 1)
            default_name = f"{tr('缓存槽')}{index + 1}"
            slot['name'] = default_name
            slot['locked'] = False
            slot['items'] = []
            slot['copied_at'] = ''
            slot['copy_order'] = 0
            slot['shortcut'] = str(slot.get('shortcut') or default_shortcut)
        self.active_label_cache_slot = 0
        self._label_cache_copy_counter = 0

    def set_active_label_cache_slot(self, slot_index):
        if slot_index < 0 or slot_index >= len(self.label_cache_slots):
            return
        self.active_label_cache_slot = slot_index
        self._rebuild_label_cache_menu()

    def _get_selected_detection_boxes(self):
        multi_indexes = [
            index for index in getattr(self.canvas, 'selected_boxes', [])
            if 0 <= index < len(self.detection_boxes)
        ]
        if multi_indexes:
            return [dict(self.detection_boxes[index]) for index in multi_indexes]

        index = getattr(self.canvas, 'selected_box', None)
        if index is None or index < 0 or index >= len(self.detection_boxes):
            return []
        return [dict(self.detection_boxes[index])]

    def copy_selected_labels_to_active_cache_slot(self):
        items = self._get_selected_detection_boxes()
        if not items and getattr(self, 'canvas', None):
            # hover 选中态可能还没同步到 selected_box，需要补一次同步。
            check_hover = getattr(self.canvas, '_check_hover', None)
            if callable(check_hover):
                check_hover()
                items = self._get_selected_detection_boxes()
        if not items:
            self.status_label.setText(tr("无可复制标签"))
            return
        # Keep direct calls on lightweight test doubles compatible with the old host-class method.
        slot_index = LabelCacheSlotMixin._get_next_writable_label_cache_slot_index(self)
        if slot_index is None:
            self.status_label.setText(tr("没有可写入的缓存槽"))
            return
        self.active_label_cache_slot = slot_index
        slot = self.label_cache_slots[slot_index]
        self._label_cache_copy_counter = getattr(self, '_label_cache_copy_counter', 0) + 1
        slot['items'] = items
        slot['copied_at'] = datetime.now().strftime('%H:%M:%S')
        slot['copy_order'] = self._label_cache_copy_counter
        self._last_paste_slot = None
        self._save_label_cache_slots()
        self._rebuild_label_cache_menu()

    def paste_label_cache_slot(self, slot_index):
        if self._is_delete_view or self.current_background is None:
            return
        if slot_index < 0 or slot_index >= len(self.label_cache_slots):
            return
        slot = self.label_cache_slots[slot_index]
        if not slot.get('items'):
            self.status_label.setText(tr("缓存槽为空"))
            return
        pasted_group = []
        for box in slot['items']:
            # 三元组：把原始检测框字典一起带下去，否则粘贴出来只剩坐标和类别，
            # 置信度（score）和形状类型（shape_type）都会在这一步丢掉。
            pasted_group.append((
                QRectF(box['x'], box['y'], box['width'], box['height']),
                box['label'],
                box,
            ))
        adjusted_group = self._offset_overlapping_paste_group(pasted_group)
        if adjusted_group:
            self.save_undo_state()
            self._last_paste_start = len(self.detection_boxes)
        for item in adjusted_group:
            rect, label = item[0], item[1]
            source = item[2] if len(item) > 2 else {}
            new_box = dict(source) if isinstance(source, dict) else {}
            new_box.update({
                'x': rect.x(),
                'y': rect.y(),
                'width': rect.width(),
                'height': rect.height(),
                'label': label,
            })
            self.detection_boxes.append(new_box)
        if adjusted_group:
            self._last_paste_slot = slot_index
            self._last_paste_count = len(adjusted_group)
        if adjusted_group and self.current_background_index >= 0:
            self.detection_boxes_dict[self.current_background_index] = self.detection_boxes.copy()
        self.update_label_list()
        self.canvas.update()

    def _sync_pasted_boxes_to_cache(self, box_index=None):
        """把「从缓存槽粘出来的那几个框」的当前位置回写进槽，当新的模板。

        只在**被拖的正是粘贴副本**时才回写（用户要求）：`box_index` 是画布
        传进来的被拖框下标，不属于粘贴区间就什么都不做。

        原来这里不看 `box_index`，只要粘过一次，之后拖动**任何**框都会把
        粘贴区间里的内容写回槽——于是「拖动一个没被复制过的框，缓存槽却被
        改写了」（用户报的 bug）。而且粘贴时副本会被 `_offset_overlapping_
        paste_group` 挪开避让，写回去的是**挪过的位置**，模板就这么悄悄变了。
        """
        if self._last_paste_slot is None or self._last_paste_count <= 0:
            return
        start = self._last_paste_start
        end = start + self._last_paste_count
        if start < 0 or end > len(self.detection_boxes):
            return
        # 被拖的框不在粘贴区间内 → 与这个槽无关，不动它
        if box_index is not None and not (start <= box_index < end):
            return
        slot = self.label_cache_slots[self._last_paste_slot]
        slot['items'] = [dict(self.detection_boxes[i]) for i in range(start, end)]
        self._save_label_cache_slots()

    def toggle_label_cache_slot_lock(self, slot_index):
        if slot_index < 0 or slot_index >= len(self.label_cache_slots):
            return
        self.label_cache_slots[slot_index]['locked'] = not self.label_cache_slots[slot_index].get('locked')
        self._save_label_cache_slots()
        self._rebuild_label_cache_menu()

    def clear_label_cache_slot(self, slot_index):
        if slot_index < 0 or slot_index >= len(self.label_cache_slots):
            return
        if self.label_cache_slots[slot_index].get('locked'):
            return
        self.label_cache_slots[slot_index]['items'] = []
        self.label_cache_slots[slot_index]['copied_at'] = ''
        self.label_cache_slots[slot_index]['copy_order'] = 0
        self._save_label_cache_slots()
        self._rebuild_label_cache_menu()

    def rename_label_cache_slot(self, slot_index, name):
        if slot_index < 0 or slot_index >= len(self.label_cache_slots):
            return
        text = str(name or '').strip()
        if not text:
            text = f"{tr('缓存槽')}{slot_index + 1}"
        self.label_cache_slots[slot_index]['name'] = text
        self._save_label_cache_slots()
        self._rebuild_label_cache_menu()
