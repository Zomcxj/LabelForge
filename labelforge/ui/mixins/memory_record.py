"""Memory record behavior for the main window."""

import os
from datetime import datetime

from PyQt5.QtWidgets import QApplication

from ..i18n import t as tr


class MemoryRecordMixin:
    def _connect_manager_signals(self):
        """连接管理器信号 → 编辑器 UI 刷新（需在 init_ui 之后调用）"""
        self.label_manager.data_changed.connect(self.canvas.update)
        self.label_manager.label_list_changed.connect(self.update_label_list)
        self.save_manager.save_completed.connect(self._on_save_completed)
        self.save_manager.label_list_changed.connect(self.update_label_list)
        self._is_delete_view = False

    def _build_bg_label_stats_snapshot(self):
        """Build background label stats snapshot for memory / stats restore.

        Reuses the live scan cache when it is fresh (scan finished, dataset
        unchanged, no edits since). Only falls back to a full disk rescan when
        the cache cannot be trusted — that rescan is O(n) JSON parsing and
        must not run on every dataset switch.
        """
        cached = getattr(self, '_cached_bg_label_stats', None) or []
        current_path = getattr(self, '_memory_background_path', '') or ''
        cache_path = getattr(self, '_cached_bg_label_stats_path', '') or ''
        scan_done = getattr(self, '_background_label_scan_completed', False)
        dirty = getattr(self, '_dataset_stats_dirty', False)
        cache_fresh = (
            bool(cached)
            and scan_done
            and not dirty
            and (not cache_path or not current_path or cache_path == current_path)
        )
        if cache_fresh:
            counts = {
                str(item.get('label', '')).strip(): int(item.get('count', 0) or 0)
                for item in cached
                if isinstance(item, dict) and str(item.get('label', '')).strip()
            }
            tasks_by_label = {
                str(item.get('label', '')).strip(): sorted(item.get('tasks') or [])
                for item in cached
                if isinstance(item, dict) and str(item.get('label', '')).strip()
            }
        else:
            from ...engine.image_loader import (
                collect_background_label_counts, collect_background_label_tasks,
            )
            paths = list(self.background_images or [])
            counts = collect_background_label_counts(paths)
            tasks_by_label = {
                str(label).strip(): sorted(ts)
                for label, ts in collect_background_label_tasks(paths).items()
                if str(label).strip()
            }
        for lbl in getattr(self, 'background_dataset_labels', set()) or set():
            counts.setdefault(lbl, 0)
        for lbl in getattr(self, 'global_labels', set()) or set():
            if lbl in counts or lbl in (getattr(self, 'background_dataset_labels', set()) or set()):
                counts.setdefault(lbl, 0)
        stats = []
        for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            color = ''
            if hasattr(self, 'get_label_color'):
                try:
                    color = self.get_label_color(label)
                except Exception:
                    color = self.label_color_map.get(label, '') if hasattr(self, 'label_color_map') else ''
            elif hasattr(self, 'label_color_map'):
                color = self.label_color_map.get(label, '')
            stats.append({
                'label': label, 'count': int(count), 'color': color or '',
                'tasks': tasks_by_label.get(label, []),
            })
        return stats

    def _save_memory_record_on_close(self):
        """关闭时保存当前素材来源路径组合。"""
        from ...core import config_manager

        bg_stats = self._build_bg_label_stats_snapshot()
        self._cached_bg_label_stats = bg_stats
        self._cached_bg_label_stats_path = self._memory_background_path or ''
        record = {
            'note': '',
            'background_path': self._memory_background_path,
            'paste_path': self._memory_paste_path,
            'label_path': self._memory_label_path,
            'background_index': self.current_background_index if self.current_background_index >= 0 else 0,
            'edit_mode': getattr(self, 'edit_mode', 'paste'),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'bg_label_stats': bg_stats,
        }
        for existing in config_manager.load_memory_records():
            if all(existing.get(k) == record[k] for k in ('background_path', 'paste_path', 'label_path')):
                record['note'] = existing.get('note', '')
                break
        config_manager.upsert_memory_record(record)
        self._reset_label_cache_slots()
        self._save_label_cache_slots()

    def load_memory_record(self, record):
        """用记忆记录替换当前打开的背景图、贴图和标签来源。"""
        # 记忆弹窗关闭后直接等待加载完成，不再在画布上显示加载动画。
        QApplication.processEvents()
        self._load_memory_record_now(record)

    def _load_memory_record_now(self, record):
        """实际加载记忆记录。"""
        if hasattr(self, '_save_memory_record_on_close'):
            self._save_memory_record_on_close()
        self._clear_memory_content()
        missing = []

        bg_path = record.get('background_path') or ''
        paste_path = record.get('paste_path') or ''
        label_path = record.get('label_path') or ''
        saved_index = int(record.get('background_index', 0) or 0)

        if bg_path:
            if os.path.isdir(bg_path):
                self.load_background_folder(
                    bg_path, load_first=False, restore_index=max(0, int(saved_index))
                )
                QApplication.processEvents()
            else:
                missing.append(bg_path)
        if paste_path:
            if os.path.isdir(paste_path):
                self.load_paste_folder(paste_path)
                QApplication.processEvents()
            else:
                missing.append(paste_path)
        if label_path:
            if os.path.isfile(label_path):
                self.load_paste_label_file(label_path)
                QApplication.processEvents()
            else:
                missing.append(label_path)

        edit_mode = record.get('edit_mode')
        if edit_mode not in ('paste', 'annotate'):
            edit_mode = 'annotate' if label_path and not paste_path else 'paste'
        self._set_edit_mode(edit_mode, animated=False)

        stats = record.get('bg_label_stats') or []
        if isinstance(stats, list):
            self._restore_bg_label_stats(stats)
            merged = self._merge_restored_stats_with_scan(stats)
            self._cached_bg_label_stats = merged
            self._cached_bg_label_stats_path = bg_path or ''

        if missing and hasattr(self, 'status_label'):
            self.status_label.setText(f"{tr('路径不存在')}: {missing[0]}")

    def _merge_restored_stats_with_scan(self, restored_stats):
        """合并记忆记录统计与扫描缓存：count 以记录为准，tasks 缺失时用扫描值。

        旧版记录没有 tasks 字段（None），直接覆盖会让「框类型」列在切图前
        一直为空；扫描缓存里同一类别的 tasks 更可信，用来补齐。
        """
        scanned = {}
        for item in getattr(self, '_cached_bg_label_stats', None) or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip()
            if label:
                scanned[label] = item
        merged = []
        for item in restored_stats or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip()
            if not label:
                continue
            entry = dict(item)
            entry['label'] = label
            if not entry.get('tasks'):
                entry['tasks'] = sorted(
                    (scanned.get(label) or {}).get('tasks') or [])
            merged.append(entry)
        return merged

    def _restore_bg_label_stats(self, stats):
        """把记忆记录里的类别恢复成可用的标签集合。"""
        restored = {
            str(item.get('label', '')).strip()
            for item in stats
            if isinstance(item, dict) and str(item.get('label', '')).strip()
        }
        if not restored:
            return
        if not hasattr(self, 'background_dataset_labels'):
            self.background_dataset_labels = set()
        if not hasattr(self, 'imported_background_labels'):
            self.imported_background_labels = set()
        # 记录里的类别不算「扫描结果」，放进 imported 才能扛住之后的重新扫描
        self.imported_background_labels.update(restored)
        self.background_dataset_labels.update(restored)
        self.global_labels.update(restored)
        for item in stats:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '')).strip()
            color = str(item.get('color', '') or '').strip()
            if label and color and hasattr(self, 'label_color_map'):
                self.label_color_map.setdefault(label, color)
        if hasattr(self, 'update_label_list'):
            self.update_label_list()

    def _clear_memory_content(self):
        """加载记录前清空当前素材，避免新旧内容混在一起。"""
        self.background_images.clear()
        self.small_images.clear()
        self.canvas_items_dict.clear()
        self.detection_boxes_dict.clear()
        self.canvas_items.clear()
        self.detection_boxes.clear()
        self.global_labels.clear()
        if hasattr(self, 'background_dataset_labels'):
            self.background_dataset_labels.clear()
        else:
            self.background_dataset_labels = set()
        if hasattr(self, 'imported_background_labels'):
            self.imported_background_labels.clear()
        else:
            self.imported_background_labels = set()
        self._scanned_background_labels = set()
        self._cached_bg_label_stats = []
        self._cached_bg_label_stats_path = ""
        self._background_label_scan_pending = False
        self._background_label_scan_completed = False
        self._background_label_scan_in_progress = False
        self.current_background = None
        self.current_background_index = -1
        self.selected_item = None
        self._memory_background_path = ""
        self._memory_paste_path = ""
        self._memory_label_path = ""
        for widget_name in ('background_list', 'small_list', 'label_list'):
            if hasattr(self, widget_name):
                getattr(self, widget_name).clear()
        if hasattr(self, 'paste_label_list'):
            self.paste_label_list.clear()
            self.paste_label_list.addItem('paste')
        if hasattr(self, 'canvas'):
            self.canvas.update()
