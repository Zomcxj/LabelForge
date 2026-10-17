"""Background image list and deleted-image view behavior for the main window."""

import os

from PySide6.QtCore import Qt

from ..i18n import t as tr
from ...engine.image_loader import BG_ROLE_INDEX, BG_ROLE_PATH, BG_ROLE_STATUS


class BackgroundListMixin:
    TASK_FILTER_ORDER = ("det", "seg", "pose", "obb")

    def _setup_filter_menus(self):
        """T/G 用与顶部选项一致的不自动关闭菜单（点击切换，移开收起）。"""
        from ...widgets.hover_menu import HoverKeepMenu
        self._task_filter_menu = HoverKeepMenu(self, first_action_momentary=False)
        self._task_filter_menu.setObjectName("optionsMenu")
        self._task_filter_menu.aboutToShow.connect(self._rebuild_task_filter_menu)
        self.task_filter_btn.setMenu(self._task_filter_menu)

        self._group_filter_menu = HoverKeepMenu(self, first_action_momentary=False)
        self._group_filter_menu.setObjectName("optionsMenu")
        self._group_filter_menu.aboutToShow.connect(self._rebuild_group_filter_menu)
        self.group_filter_btn.setMenu(self._group_filter_menu)

    def _rebuild_task_filter_menu(self):
        from PySide6.QtGui import QAction
        menu = self._task_filter_menu
        menu.clear()
        current = set(getattr(self, '_task_filter', set()))
        for task in self.TASK_FILTER_ORDER:
            act = QAction(tr(task), self)
            act.setCheckable(True)
            act.setChecked(task in current)
            act.triggered.connect(lambda checked, t=task: self._toggle_task_filter(t, checked))
            menu.addAction(act)
        menu.addSeparator()
        clear = QAction(f"{'√ ' if not current else ''}{tr('全部')}", self)
        clear.triggered.connect(self._clear_task_filter)
        self._task_filter_clear_action = clear
        menu.addAction(clear)

    def _rebuild_group_filter_menu(self):
        from PySide6.QtGui import QAction
        menu = self._group_filter_menu
        menu.clear()
        current = set(getattr(self, '_group_filter', set()))
        for gid in self._current_group_ids():
            text = tr("未分组") if gid is None else str(gid)
            act = QAction(text, self)
            act.setCheckable(True)
            act.setChecked(gid in current)
            act.triggered.connect(lambda checked, g=gid: self._toggle_group_filter(g, checked))
            menu.addAction(act)
        menu.addSeparator()
        clear = QAction(f"{'√ ' if not current else ''}{tr('全部')}", self)
        clear.triggered.connect(self._clear_group_filter)
        self._group_filter_clear_action = clear
        menu.addAction(clear)

    def _current_group_ids(self):
        """当前图出现过的 group_id（None 表示未分组），无分组时也保留 None 选项。"""
        gids = set()
        has_ungrouped = False
        for box in getattr(self, 'detection_boxes', []) or []:
            gid = box.get("group_id")
            if gid is None:
                has_ungrouped = True
            else:
                gids.add(gid)
        ordered = sorted(gids)
        if has_ungrouped:
            ordered.append(None)
        return ordered

    def _toggle_task_filter(self, task, checked):
        current = set(getattr(self, '_task_filter', set()))
        if checked:
            current.add(task)
        else:
            current.discard(task)
        self._task_filter = current
        self._after_filter_changed()

    def _clear_task_filter(self):
        self._task_filter = set()
        self._after_filter_changed()

    def _toggle_group_filter(self, gid, checked):
        current = set(getattr(self, '_group_filter', set()))
        if checked:
            current.add(gid)
        else:
            current.discard(gid)
        self._group_filter = current
        self._after_filter_changed()

    def _clear_group_filter(self):
        self._group_filter = set()
        self._after_filter_changed()

    def _after_filter_changed(self):
        self._refresh_filter_buttons()
        self.update_label_list()
        self.canvas.update()
        self._refresh_filter_menu_checks()

    def _refresh_filter_menu_checks(self):
        """菜单不自动关闭，就地刷新"全部"行的勾选标记。"""
        for attr, current in (
            ('_task_filter_clear_action', getattr(self, '_task_filter', set())),
            ('_group_filter_clear_action', getattr(self, '_group_filter', set())),
        ):
            act = getattr(self, attr, None)
            if act is not None:
                act.setText(f"{'√ ' if not current else ''}{tr('全部')}")

    def _refresh_filter_buttons(self):
        task_btn = getattr(self, 'task_filter_btn', None)
        if task_btn is not None:
            n = len(getattr(self, '_task_filter', set()))
            task_btn.setText(str(n) if n else "T")
            task_btn.setToolTip(tr("按任务筛选") if not n
                                else f"{tr('按任务筛选')}: {', '.join(sorted(getattr(self, '_task_filter', set())))}")
        group_btn = getattr(self, 'group_filter_btn', None)
        if group_btn is not None:
            n = len(getattr(self, '_group_filter', set()))
            group_btn.setText(str(n) if n else "G")
            names = [tr("未分组") if g is None else str(g)
                     for g in sorted(getattr(self, '_group_filter', set()),
                                     key=lambda v: (v is None, v))]
            group_btn.setToolTip(tr("按分组筛选") if not n
                                 else f"{tr('按分组筛选')}: {', '.join(names)}")

    def _toggle_bg_label_list_mode(self):
        """Switch background label list between stats counts and per-box rows."""
        current = getattr(self, '_bg_label_list_mode', 'stats')
        self._bg_label_list_mode = 'all' if current == 'stats' else 'stats'
        self._refresh_bg_label_mode_button()
        self.update_label_list()

    def _refresh_bg_label_mode_button(self):
        btn = getattr(self, 'bg_label_mode_btn', None)
        if btn is None:
            return
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QIcon
        mode = getattr(self, '_bg_label_list_mode', 'stats')
        # stats: show "rows" icon (click → list); list: show "bars" icon (click → stats)
        if mode == 'all':
            btn.setIcon(QIcon(self._bg_label_mode_icon('stats')))
            btn.setToolTip(tr("切换到统计计数"))
        else:
            btn.setIcon(QIcon(self._bg_label_mode_icon('list')))
            btn.setToolTip(tr("切换到每框一行"))
        btn.setText("")
        btn.setIconSize(QSize(14, 14))

    def _bg_label_mode_icon(self, kind):
        """Paint a tiny mode glyph: list rows or count bars."""
        from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
        from PySide6.QtCore import Qt
        size = 14
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#c0c4cc")
        if hasattr(self, 'theme_manager') and getattr(self, 'is_dark_theme', True) is False:
            color = QColor("#5a5e66")
        p.setPen(QPen(color, 1.6))
        if kind == 'list':
            # three equal horizontal rows
            for y in (3, 7, 11):
                p.drawLine(2, y, 12, y)
        else:
            # three vertical bars of different height (stats)
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawRect(2, 8, 2, 5)
            p.drawRect(6, 4, 2, 9)
            p.drawRect(10, 6, 2, 7)
        p.end()
        return pm

    def _cycle_bg_annotation_filter(self):
        """Cycle background list filter: all → annotated → unannotated → empty."""
        order = ('all', 'annotated', 'unannotated', 'empty')
        current = getattr(self, '_bg_annotation_filter', 'all')
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        next_mode = order[(idx + 1) % len(order)]
        if current == 'all' and next_mode != 'all':
            self._bg_filter_saved_index = self.current_background_index
        self._bg_annotation_filter = next_mode
        self._refresh_bg_filter_button()
        self._apply_bg_annotation_filter(navigate=True)

    def _refresh_bg_filter_button(self):
        btn = getattr(self, 'bg_filter_btn', None)
        if btn is None:
            return
        mode = getattr(self, '_bg_annotation_filter', 'all')
        tips = {
            'all': tr("筛选：全部"),
            'annotated': tr("筛选：已标注"),
            'unannotated': tr("筛选：未标注"),
            'empty': tr("筛选：空标签"),
        }
        from PySide6.QtCore import QSize
        from ...engine.image_loader import _status_icon
        icon_key = mode if mode in ('all', 'annotated', 'unannotated', 'empty') else 'all'
        btn.setText("")
        btn.setIcon(_status_icon(icon_key, size=14))
        btn.setIconSize(QSize(14, 14))
        btn.setToolTip(tips.get(mode, tips['all']))
        in_delete = getattr(self, '_is_delete_view', False)
        btn.setVisible(not in_delete)

    def _apply_bg_annotation_filter(self, navigate=False):
        """Hide list rows that do not match the current annotation filter."""
        if getattr(self, '_is_delete_view', False):
            return
        bg_list = getattr(self, 'background_list', None)
        if bg_list is None:
            return
        mode = getattr(self, '_bg_annotation_filter', 'all')
        first_visible_row = None
        for row in range(bg_list.count()):
            item = bg_list.item(row)
            if item is None:
                continue
            status = item.data(BG_ROLE_STATUS)
            if status is None:
                path = item.data(BG_ROLE_PATH)
                if not path:
                    idx = item.data(BG_ROLE_INDEX)
                    if isinstance(idx, int) and 0 <= idx < len(self.background_images):
                        path = self.background_images[idx]
                from ...engine.image_loader import decorate_background_list_item
                if path:
                    status = decorate_background_list_item(
                        item, path, item.data(BG_ROLE_INDEX), light=True)
                else:
                    status = 'unannotated'
            visible = (mode == 'all') or (status == mode)
            item.setHidden(not visible)
            if visible and first_visible_row is None:
                first_visible_row = row

        if not navigate:
            return

        if mode in ('unannotated', 'empty'):
            target_row = first_visible_row
        else:
            # all / annotated: restore remembered work position when possible
            saved = getattr(self, '_bg_filter_saved_index', self.current_background_index)
            target_row = self._find_bg_list_row_for_index(saved)
            if target_row is None or (bg_list.item(target_row) and bg_list.item(target_row).isHidden()):
                if mode == 'annotated':
                    target_row = first_visible_row
                else:
                    target_row = self._find_bg_list_row_for_index(saved)
                    if target_row is None:
                        target_row = first_visible_row

        if target_row is None:
            return
        item = bg_list.item(target_row)
        if item is None:
            return
        bg_list.setCurrentRow(target_row)
        self.select_background(item)

    def _find_bg_list_row_for_index(self, image_index):
        bg_list = getattr(self, 'background_list', None)
        if bg_list is None or not isinstance(image_index, int):
            return None
        for row in range(bg_list.count()):
            item = bg_list.item(row)
            if item is None:
                continue
            if item.data(BG_ROLE_INDEX) == image_index:
                return row
        if 0 <= image_index < bg_list.count():
            return image_index
        return None

    def _refresh_background_item_status(self, image_index=None, image_path=None):
        """Refresh status icon for one background list row after save/load."""
        if getattr(self, '_is_delete_view', False):
            return
        bg_list = getattr(self, 'background_list', None)
        if bg_list is None:
            return
        from ...engine.image_loader import decorate_background_list_item
        if image_path is None and isinstance(image_index, int):
            if 0 <= image_index < len(self.background_images):
                image_path = self.background_images[image_index]
        if not image_path:
            return
        target_row = self._find_bg_list_row_for_index(image_index) if isinstance(image_index, int) else None
        if target_row is None:
            for row in range(bg_list.count()):
                item = bg_list.item(row)
                if item and item.data(BG_ROLE_PATH) == image_path:
                    target_row = row
                    break
        if target_row is None:
            return
        item = bg_list.item(target_row)
        if item is None:
            return
        idx = item.data(BG_ROLE_INDEX)
        if idx is None:
            idx = image_index
        decorate_background_list_item(item, image_path, idx)
        mode = getattr(self, '_bg_annotation_filter', 'all')
        status = item.data(BG_ROLE_STATUS)
        # 同步状态缓存，否则切视图时会把刚存好的图又显示成扫描中。
        cache = getattr(self, '_bg_status_by_path', None)
        if cache is None:
            cache = self._bg_status_by_path = {}
        if status is not None:
            cache[image_path] = status
        visible = (mode == 'all') or (status == mode)
        item.setHidden(not visible)

    def _toggle_view_path(self):
        """切换工作路径/移除路径视图"""
        from .. import i18n
        _tr = i18n.t
        self._is_delete_view = not self._is_delete_view
        if self._is_delete_view:
            self.view_toggle_btn.setText(_tr("移除路径"))
            self._saved_work_index = self.current_background_index
            self._show_delete_view()
            self._refresh_bg_filter_button()
            saved_del = getattr(self, '_saved_delete_idx', 0)
            if self._delete_files and saved_del < len(self._delete_files):
                self._delete_current_idx = saved_del
                self._load_delete_image(saved_del)
                self.background_list.setCurrentRow(saved_del)
            disabled_keys = {
                'W', 'Q', 'P', 'K', 'O',
                self._get_shortcut('delete_selected'),
                self._get_shortcut('draw_polygon'),
                self._get_shortcut('draw_point'),
                self._get_shortcut('draw_obb'),
            }
            for sc in getattr(self, '_shortcuts', []):
                key = sc.key().toString()
                if key in disabled_keys:
                    sc.setEnabled(False)
            if hasattr(self, 'draw_box_btn'):
                self.draw_box_btn.setEnabled(False)
            self.canvas._clear_selection()
            self.canvas.is_drawing_box = False
            if hasattr(self.canvas, '_reset_drawing_state'):
                self.canvas._reset_drawing_state()
            self.canvas.draw_start_pos = None
            self.canvas.temp_draw_box = None
        else:
            self.view_toggle_btn.setText(_tr("工作路径"))
            self._saved_delete_idx = getattr(self, '_delete_current_idx', 0)
            saved = getattr(self, '_saved_work_index', 0)
            self._show_work_view()
            self._refresh_bg_filter_button()
            if self.background_images and saved < len(self.background_images):
                self.current_background_index = saved
                from PySide6.QtGui import QPixmap
                pixmap = QPixmap(self.background_images[saved])
                if not pixmap.isNull():
                    self.current_background = pixmap
                    self.detection_boxes = self.detection_boxes_dict.get(saved, []).copy()
                    self.canvas_items = self.canvas_items_dict.get(saved, [])
                    self.canvas.reset_view()
                    self.canvas.repaint()
                    self.update_label_list()
                self.update_file_count()
                row = self._find_bg_list_row_for_index(saved)
                if row is not None:
                    self.background_list.setCurrentRow(row)
            for sc in getattr(self, '_shortcuts', []):
                sc.setEnabled(True)
            if hasattr(self, 'draw_box_btn'):
                self.draw_box_btn.setEnabled(True)
        from PySide6.QtCore import QTimer
        mode_text = "Removed" if self._is_delete_view else "Work"
        self.status_label.setText(f"Path: {mode_text}")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _on_refresh_bg_list(self):
        """刷新按钮：就地重扫文件夹增删列表项，不重新导入、不丢未保存标注。"""
        from PySide6.QtCore import QTimer
        changed, added, removed = self.refresh_background_list()
        if not hasattr(self, 'status_label'):
            return
        notice = getattr(self, '_refresh_notice', '') or ''
        text = notice or tr("refresh_done").format(added=added, removed=removed)
        self.status_label.setText(text)
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _refresh_background_path_texts(self):
        """开关切换后原地刷新列表项文本，无需重载数据集。"""
        from ...core.utils import PathUtils
        try:
            if getattr(self, '_is_delete_view', False):
                files = list(getattr(self, '_delete_files', []) or [])
            else:
                files = list(getattr(self, 'background_images', []) or [])
            lst = getattr(self, 'background_list', None)
            if lst is None:
                return
            for row in range(lst.count()):
                item = lst.item(row)
                if item is None:
                    continue
                path = self._bg_path_for_item(item, row, files)
                if path:
                    item.setText(self._display_path_for(path))
        except Exception as e:
            from ...core.exception_hook import _write_log
            _write_log(f"刷新背景路径显示失败: {e}")

    def _bg_path_for_item(self, item, row, files):
        """取列表项对应的真实文件路径（优先用索引/路径元数据，回退按行号）。"""
        from ...engine.image_loader import BG_ROLE_INDEX, BG_ROLE_PATH
        stored = item.data(BG_ROLE_PATH) if hasattr(item, 'data') else None
        if stored:
            return stored
        index = item.data(BG_ROLE_INDEX) if hasattr(item, 'data') else None
        if isinstance(index, int) and 0 <= index < len(files):
            return files[index]
        if 0 <= row < len(files):
            return files[row]
        return ''

    def show_background_context_menu(self, position):
        """背景图列表右键菜单：复制图片全局路径。"""
        lst = getattr(self, 'background_list', None)
        if lst is None:
            return
        item = lst.itemAt(position)
        if item is None:
            return
        files = list(getattr(self, '_delete_files', []) or []) \
            if getattr(self, '_is_delete_view', False) \
            else list(getattr(self, 'background_images', []) or [])
        path = self._bg_path_for_item(item, lst.row(item), files)
        if not path:
            return
        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QGuiApplication
        from ..i18n import t as tr
        menu = QMenu(self)
        copy_action = menu.addAction(tr("复制图片路径"))
        chosen = menu.exec_(lst.mapToGlobal(position))
        if chosen == copy_action:
            QGuiApplication.clipboard().setText(os.path.abspath(path))

    def _show_work_view(self):
        """显示工作路径列表"""
        self.background_list.clear()
        self._bg_item_index_cache = None
        from ...core.config import SUPPORTED_IMAGE_EXTENSIONS
        from ...engine.image_loader import (
            apply_background_status_to_item, decorate_background_list_item,
        )
        status_cache = getattr(self, '_bg_status_by_path', None) or {}
        for i, path in enumerate(self.background_images):
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                from PySide6.QtWidgets import QListWidgetItem
                item = QListWidgetItem(self._display_path_for(path))
                # 扫描过的图直接用已知状态；只对没扫过的做廉价判定，
                # 否则切回工作路径时已标注的图会退回「扫描中」灰圈。
                decorate_background_list_item(item, path, i, light=True)
                resolved = status_cache.get(path)
                if resolved is not None:
                    apply_background_status_to_item(item, resolved)
                self.background_list.addItem(item)
        self._refresh_bg_filter_button()
        self._apply_bg_annotation_filter(navigate=False)
        if 0 <= self.current_background_index < self.background_list.count():
            row = self._find_bg_list_row_for_index(self.current_background_index)
            if row is not None:
                self.background_list.setCurrentRow(row)
            self.update_file_count()
        elif self.background_images:
            self.current_background_index = 0
            self.background_list.setCurrentRow(0)
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(self.background_images[0])
            if not pixmap.isNull():
                self.current_background = pixmap
                self.detection_boxes = self.detection_boxes_dict.get(0, []).copy()
                self.canvas_items = self.canvas_items_dict.get(0, [])
                self.canvas.reset_view()
                self.canvas.repaint()
                self.update_label_list()
            self.update_file_count()

    def _show_delete_view(self):
        """显示移除路径列表"""
        self.background_list.clear()
        self._bg_item_index_cache = None
        from ...core.config import SUPPORTED_IMAGE_EXTENSIONS
        self._delete_files = []
        if self.background_images:
            delete_dir = os.path.join(
                os.path.dirname(self.background_images[0]), '_delete_')
            if os.path.isdir(delete_dir):
                for f in sorted(os.listdir(delete_dir)):
                    fp = os.path.join(delete_dir, f)
                    ext = os.path.splitext(f)[1].lower()
                    if os.path.isfile(fp) and ext in SUPPORTED_IMAGE_EXTENSIONS:
                        self._delete_files.append(fp)
                        self.background_list.addItem(self._display_path_for(fp))
        if self._delete_files:
            target = getattr(self, '_saved_delete_idx', 0)
            target = min(target, len(self._delete_files) - 1)
            self._delete_current_idx = target
            self._load_delete_image(target)
            self.background_list.blockSignals(True)
            self.background_list.setCurrentRow(target)
            self.background_list.blockSignals(False)
            filename = os.path.basename(self._delete_files[target])
            total = len(self._delete_files)
            if self.current_background:
                w = self.current_background.width()
                h = self.current_background.height()
                self.setWindowTitle(f"LabelForge - {filename} [{w} x {h}] [{target + 1} / {total}]")
            else:
                self.setWindowTitle(f"LabelForge - {filename} [{target + 1} / {total}]")
        else:
            self.setWindowTitle("LabelForge")
            self.current_background = None
            self.detection_boxes = []
            self.canvas_items = []
            self.canvas.repaint()
        self.update_label_list()

    def _load_delete_image(self, idx):
        """加载移除路径图片到画布"""
        from PySide6.QtGui import QPixmap
        if 0 <= idx < len(self._delete_files):
            pixmap = QPixmap(self._delete_files[idx])
            if not pixmap.isNull():
                self.current_background = pixmap
                self.canvas.background_scale = 1.0
                self.canvas.is_manual_scale = False
                self.canvas.reset_view()
                self.canvas.selected_box = None
                self.canvas.selected_boxes = []
                self.selected_item = None
                self.canvas_items = []
                self.detection_boxes = self.load_detection_boxes(self._delete_files[idx])
                self.update_label_list()
                self.canvas.repaint()

    def _remove_to_delete(self, idx):
        """移除文件到 _delete_ 文件夹"""
        import shutil
        if idx < 0 or idx >= len(self.background_images):
            return
        file_path = self.background_images[idx]

        delete_dir = os.path.join(os.path.dirname(file_path), '_delete_')
        os.makedirs(delete_dir, exist_ok=True)

        shutil.move(file_path, os.path.join(delete_dir, os.path.basename(file_path)))
        json_path = os.path.splitext(file_path)[0] + '.json'
        if os.path.isfile(json_path):
            shutil.move(json_path, os.path.join(delete_dir, os.path.basename(json_path)))

        self.background_images.pop(idx)
        if idx in self.canvas_items_dict:
            del self.canvas_items_dict[idx]
        if idx in self.detection_boxes_dict:
            del self.detection_boxes_dict[idx]

        new_idx = min(idx, len(self.background_images) - 1)
        if self.background_images:
            self.current_background_index = new_idx
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(self.background_images[new_idx])
            if not pixmap.isNull():
                self.current_background = pixmap
                self.canvas.reset_view()
                self.canvas.update()
        else:
            self.current_background = None
            self.current_background_index = -1

        self._show_delete_view()
        self.update_file_count()

    def _restore_from_delete(self, idx):
        """从 _delete_ 恢复文件"""
        import shutil
        if idx < 0 or idx >= self.background_list.count():
            return
        item = self.background_list.item(idx)
        text = item.text()
        delete_dir = os.path.join(
            os.path.dirname(self.background_images[0]) if self.background_images else '', '_delete_')

        for f in os.listdir(delete_dir):
            if f in text or text.endswith(f):
                src = os.path.join(delete_dir, f)
                dst = os.path.join(os.path.dirname(delete_dir), f)
                shutil.move(src, dst)
                self.background_images.append(dst)
                self.canvas_items_dict[len(self.background_images) - 1] = []
                self.detection_boxes_dict[len(self.background_images) - 1] = []
                break

        self.current_background_index = len(self.background_images) - 1
        self._show_delete_view()
        self.update_file_count()
