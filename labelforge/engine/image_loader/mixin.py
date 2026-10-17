"""ImageLoaderMixin：背景图/贴图/标签文件加载与选图切换。"""
import os
import json
import time

from PySide6.QtWidgets import (
    QApplication, QFileDialog, QListWidgetItem, QMessageBox
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QSize, QTimer

from ...core.config import SUPPORTED_IMAGE_EXTENSIONS
from ...core.utils import PathUtils, natural_sort_key, create_thumbnail
from ...ui.i18n import t as tr

from .status import (
    BG_ROLE_INDEX, BG_ROLE_PATH, BG_ROLE_STATUS,
    decorate_background_list_item, apply_background_status_to_item,
)
from .scan import DatasetLabelScanWorker

__all__ = ["ImageLoaderMixin"]


class ImageLoaderMixin:
    """图片加载混入类 - 加载背景图、贴图、检测框"""

    def _start_background_replacement(self):
        """Invalidate scans and labels before replacing the background dataset."""
        self._pending_memory_index = None
        if hasattr(self, '_background_load_timer') and self._background_load_timer:
            self._background_load_timer.stop()
            self._background_load_timer = None
            self._pending_image_files = None
        self._background_label_scan_generation += 1
        self.global_labels.clear()
        if hasattr(self, 'background_dataset_labels'):
            self.background_dataset_labels.clear()
        else:
            self.background_dataset_labels = set()
        # 换数据集等于换一套类别，手动导入的也要清掉
        if hasattr(self, 'imported_background_labels'):
            self.imported_background_labels.clear()
        else:
            self.imported_background_labels = set()
        self._scanned_background_labels = set()
        self._cached_bg_label_stats = []
        self._cached_bg_label_stats_path = ""
        self._bg_item_index_cache = None
        if hasattr(self, '_dataset_stats_dirty'):
            self._dataset_stats_dirty = False
        self._background_label_scan_pending = True
        self._background_label_scan_completed = False
        if (hasattr(self, '_processing_panel') and self._processing_panel
                and self._processing_panel.isVisible()):
            self._update_processing_panel_labels()
        worker = getattr(self, '_background_label_scan_worker', None)
        if worker is not None:
            worker.requestInterruption()
            self._background_label_scan_worker = None

    def _finish_background_replacement_without_scan(self):
        """Clear scan state when replacement produced no dataset to scan."""
        self._background_label_scan_in_progress = False
        self._background_label_scan_pending = False
        self._background_label_scan_completed = False
        if (hasattr(self, '_processing_panel') and self._processing_panel
                and self._processing_panel.isVisible()):
            self._update_processing_panel_labels()

    def _get_cached_pixmap(self, file_path):
        """获取缓存的pixmap，缓存最近10张"""
        if not hasattr(self, '_pixmap_cache'):
            self._pixmap_cache = {}
            self._pixmap_cache_order = []
        cache_limit = 10
        if file_path in self._pixmap_cache:
            return self._pixmap_cache[file_path]
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self._pixmap_cache[file_path] = pixmap
            self._pixmap_cache_order.append(file_path)
            while len(self._pixmap_cache_order) > cache_limit:
                old = self._pixmap_cache_order.pop(0)
                self._pixmap_cache.pop(old, None)
        return pixmap

    def _relative_path_base(self):
        """相对路径基准：已加载背景图文件夹的父目录。

        没有基准（未加载文件夹）时返回空串，调用方回退为绝对路径。
        """
        folder = getattr(self, '_memory_background_path', '') or ''
        if not folder:
            return ''
        return os.path.dirname(folder)

    def _display_path_for(self, path):
        """按「相对路径显示」开关决定列表项显示的路径文本。"""
        if not getattr(self, '_relative_path_display', False):
            return PathUtils.to_display_path(path)
        return PathUtils.to_relative_display_path(path, self._relative_path_base())

    def upload_background(self):
        """上传背景图片"""
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("选择背景图片"), "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if files:
            close_lint = getattr(self, '_close_lint_dialog', None)
            if callable(close_lint):
                close_lint()
            self._memory_background_path = os.path.dirname(files[0])
            self._start_background_replacement()
            self.background_images.clear()
            self.background_list.clear()
            self.current_background = None
            self.detection_boxes_dict.clear()
            self.canvas_items_dict.clear()

            for file in files:
                pixmap = QPixmap(file)
                if not pixmap.isNull():
                    new_index = len(self.background_images)
                    self.background_images.append(file)
                    display_path = self._display_path_for(file)
                    item = QListWidgetItem(display_path)
                    decorate_background_list_item(item, file, new_index, light=True)
                    self.background_list.addItem(item)

                    self.canvas_items_dict[new_index] = []
                    # **不要**在这里预置 `detection_boxes_dict[new_index] = []`：
                    # 缓存判据是「下标在不在字典里」（空列表也是有效缓存），预置
                    # 一个空列表就等于告诉后面那行「这张已经加载过了、结果就是
                    # 空的」，于是磁盘上的框永远不会被读进来。
                    # 缓存由 `_load_detection_boxes_for_index()` 自己填。

                    if self.current_background is None:
                        self.current_background = pixmap
                        self.current_background_index = new_index
                        self.canvas_items = []
                        self._load_detection_boxes_for_index(new_index, file)
                        self.update_label_list()
                        self.canvas.background_scale = 1.0
                        self.canvas.is_manual_scale = False
                        self.canvas.update()

            if self.background_images:
                self._start_dataset_label_scan()
            else:
                self._finish_background_replacement_without_scan()

        self.update_file_count()

    def load_folder_images(self):
        """从文件夹加载背景图"""
        folder_path = QFileDialog.getExistingDirectory(self, tr("选择图片文件夹"), "")
        if not folder_path:
            return
        self.load_background_folder(folder_path)

    def load_background_folder(self, folder_path, load_first=True, restore_index=None):
        """从指定文件夹加载背景图，供文件夹按钮和记忆记录复用。

        restore_index: 数据集完整加载后要切换到的索引（记忆记录恢复用）。
        """
        if hasattr(self, '_save_memory_record_on_close'):
            self._save_memory_record_on_close()
        close_lint = getattr(self, '_close_lint_dialog', None)
        if callable(close_lint):
            close_lint()
        self._memory_background_path = folder_path
        self._start_background_replacement()
        self._pending_memory_index = restore_index

        self.background_images.clear()
        self.background_list.clear()
        self.current_background = None
        self.detection_boxes_dict.clear()
        self.canvas_items_dict.clear()
        self.canvas_items.clear()

        try:
            all_files = sorted(os.listdir(folder_path), key=natural_sort_key)
        except OSError:
            QMessageBox.warning(self, tr("警告"), tr("无法读取文件夹"))
            self._finish_background_replacement_without_scan()
            self.update_file_count()
            return

        image_files = []
        for file_name in all_files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.append(os.path.join(folder_path, file_name))

        if not image_files:
            QMessageBox.warning(self, tr("警告"), tr("该文件夹中没有找到支持的图片文件"))
            self._finish_background_replacement_without_scan()
            self.update_file_count()
            return

        first_path = image_files[0]
        self.background_images.append(first_path)
        display_path = self._display_path_for(first_path)
        item = QListWidgetItem(display_path)
        decorate_background_list_item(item, first_path, 0, light=True)
        self.background_list.addItem(item)
        self.canvas_items_dict[0] = []
        # 同理不预置 `detection_boxes_dict[0] = []`：那会把「还没加载」伪装成
        # 「已加载且为空」，下面 `_load_detection_boxes_for_index()` 就读不到盘了。

        if load_first:
            self.current_background_index = 0
            self.current_background = self._get_cached_pixmap(first_path)
            self.canvas_items = []
            self._load_detection_boxes_for_index(0, first_path)
            self.background_list.setCurrentRow(0)
            self.canvas.reset_view()
            self.canvas.update()
        self.update_label_list()

        QApplication.processEvents()

        if len(image_files) > 1:
            self._pending_image_files = image_files
            self._pending_image_index = 1
            if not hasattr(self, '_background_load_timer') or not self._background_load_timer:
                self._background_load_timer = QTimer(self)
                self._background_load_timer.timeout.connect(self._load_next_background_batch)
            self._background_load_timer.setInterval(0)
            self._background_load_timer.start()
        else:
            self._start_dataset_label_scan()
            self.update_file_count()
            self.background_list.viewport().update()
            self._finalize_dataset_load()

    def _clear_background_dataset(self):
        """清空背景图列表、画布与按索引缓存的标注（数据集不可用时用）。

        文件夹被删掉后必须走到这里：列表行留着会指向不存在的文件，画布还停在
        最后一张图上，用户点任何一行都会失败。与「选择文件夹」的导入前清场保持
        同一套字段，避免两处遗漏不同步。
        """
        if hasattr(self, 'background_images'):
            self.background_images.clear()
        self.detection_boxes_dict = {}
        self.canvas_items_dict = {}
        self.detection_boxes = []
        self.canvas_items = []
        self.current_background_index = -1
        self.current_background = None
        bg_list = getattr(self, 'background_list', None)
        if bg_list is not None:
            bg_list.clear()
        # 相对路径显示的基准就是这个文件夹，没了就别再留着
        self._memory_background_path = ""
        canvas = getattr(self, 'canvas', None)
        if canvas is not None:
            canvas.reset_view()
            canvas.update()
        if hasattr(self, 'update_label_list'):
            self.update_label_list()
        if hasattr(self, 'update_file_count'):
            self.update_file_count()
        if bg_list is not None:
            bg_list.viewport().update()

    def refresh_background_list(self):
        """重扫当前背景文件夹，就地增删列表项。返回 (changed, added, removed)。

        与「选择文件夹」重新导入的区别：不弹对话框、不清空内存中未保存的检测框/
        画布项（按路径迁移到新索引）、不重读首图，只重建列表行（light 装饰仅做
        一次 exists 检查，不解析 JSON），因此快得多且不丢未保存的标注。
        类别集合与状态图标交给后台扫描补齐，不阻塞界面。
        异常情况（无文件夹/读不了/无变化）不弹模态框，只把提示写进
        _refresh_notice 交给状态栏显示——刷新是高频轻操作，模态框太重。
        """
        self._refresh_notice = ""
        folder = getattr(self, '_memory_background_path', '') or ''
        if not folder:
            self._refresh_notice = tr("refresh_need_folder")
            return False, 0, 0
        if not os.path.isdir(folder):
            # 文件夹在运行期间被删掉（或整个盘符拔了）：列表里那些行都指向不存在
            # 的文件，留着只会让人点开报错。此时**清空画布与列表**，让界面如实
            # 反映「现在没有数据集」，而不是假装还能刷新。
            # 区分「从来没加载过」（folder 为空，上面那条）与「加载过但现在没了」：
            # 后者要清，前者没东西可清，提示也更有意义。
            self._clear_background_dataset()
            self._refresh_notice = tr("refresh_folder_missing")
            return False, 0, 0
        try:
            names = sorted(os.listdir(folder), key=natural_sort_key)
        except OSError:
            self._refresh_notice = tr("refresh_cannot_read")
            return False, 0, 0
        new_paths = [os.path.join(folder, n) for n in names
                     if os.path.splitext(n)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS]

        old_paths = list(self.background_images)
        if new_paths == old_paths:
            self._refresh_notice = tr("refresh_no_change")
            return False, 0, 0
        old_set, new_set = set(old_paths), set(new_paths)
        added = [p for p in new_paths if p not in old_set]
        removed = [p for p in old_paths if p not in new_set]

        # 先把当前图的实时编辑落回按索引缓存，再整体按路径取走
        cur_idx = self.current_background_index
        if 0 <= cur_idx < len(old_paths):
            self.canvas_items_dict[cur_idx] = list(getattr(self, 'canvas_items', []) or [])
            self.detection_boxes_dict[cur_idx] = list(getattr(self, 'detection_boxes', []) or [])
            cur_path = old_paths[cur_idx]
        else:
            cur_path = None
        boxes_by_path = {old_paths[i]: list(self.detection_boxes_dict.get(i, []))
                         for i in range(len(old_paths))}
        canvas_by_path = {old_paths[i]: list(self.canvas_items_dict.get(i, []))
                          for i in range(len(old_paths))}

        self.background_images = new_paths
        self.detection_boxes_dict = {i: boxes_by_path.get(p, [])
                                     for i, p in enumerate(new_paths)}
        self.canvas_items_dict = {i: canvas_by_path.get(p, [])
                                  for i, p in enumerate(new_paths)}

        bg_list = self.background_list
        bg_list.setUpdatesEnabled(False)
        try:
            bg_list.clear()
            mode = getattr(self, '_bg_annotation_filter', 'all')
            for i, path in enumerate(new_paths):
                item = QListWidgetItem(self._display_path_for(path))
                decorate_background_list_item(item, path, i, light=True)
                if mode != 'all' and item.data(BG_ROLE_STATUS) != mode:
                    item.setHidden(True)
                bg_list.addItem(item)
        finally:
            bg_list.setUpdatesEnabled(True)

        if new_paths:
            if cur_path in new_set:
                target = new_paths.index(cur_path)
            else:
                target = max(0, min(cur_idx, len(new_paths) - 1))
            self.current_background_index = target
            self.current_background = self._get_cached_pixmap(new_paths[target])
            self.detection_boxes = list(self.detection_boxes_dict.get(target, []))
            self.canvas_items = list(self.canvas_items_dict.get(target, []))
            row = (self._find_bg_list_row_for_index(target)
                   if hasattr(self, '_find_bg_list_row_for_index') else target)
            if row is not None:
                bg_list.setCurrentRow(row)
            self.canvas.reset_view()
            self.canvas.update()
        else:
            self.current_background_index = -1
            self.current_background = None
            self.detection_boxes = []
            self.canvas_items = []
            self.canvas.update()
        self.update_label_list()
        self.update_file_count()
        bg_list.viewport().update()
        self._start_dataset_label_scan()
        return True, len(added), len(removed)

    def _finalize_dataset_load(self):
        """Dataset fully loaded: apply a pending saved background index (memory restore)."""
        pending = getattr(self, '_pending_memory_index', None)
        if pending is None:
            return
        self._pending_memory_index = None
        if not self.background_images:
            return
        target = max(0, min(int(pending), len(self.background_images) - 1))
        if target != getattr(self, 'current_background_index', -1):
            self.switch_background_to_index(target)
        else:
            file_path = self.background_images[target]
            pixmap = self._get_cached_pixmap(file_path)
            if pixmap and not pixmap.isNull():
                self.current_background = pixmap
                self._load_detection_boxes_for_index(target, file_path)
                self.canvas_items = self.canvas_items_dict.get(target, []).copy()
                self.update_label_list()
                self.canvas.reset_view()
                self.canvas.update()
        row = self._find_bg_list_row_for_index(target) if hasattr(self, '_find_bg_list_row_for_index') else None
        if row is not None:
            self.background_list.setCurrentRow(row)
        self.update_file_count()

    def _load_next_background_batch(self):
        """Load next batch of background images into the list.

        Decoration is now cheap (no JSON parse), so instead of a fixed small
        batch we fill rows for a short time budget per tick. This keeps the UI
        responsive while populating tens of thousands of rows in a few ticks
        rather than sleeping 30ms per 50 items (which dominated load time).
        """
        TIME_BUDGET = 0.012  # seconds of work per tick
        deadline = time.perf_counter() + TIME_BUDGET
        bg_list = self.background_list
        bg_list.setUpdatesEnabled(False)
        try:
            while self._pending_image_index < len(self._pending_image_files):
                file_path = self._pending_image_files[self._pending_image_index]
                idx = self._pending_image_index
                self.background_images.append(file_path)
                display_path = self._display_path_for(file_path)
                item = QListWidgetItem(display_path)
                decorate_background_list_item(item, file_path, idx, light=True)
                mode = getattr(self, '_bg_annotation_filter', 'all')
                status = item.data(BG_ROLE_STATUS)
                if mode != 'all' and status != mode:
                    item.setHidden(True)
                bg_list.addItem(item)
                self.canvas_items_dict[idx] = []
                # **不要**在这里预置 `detection_boxes_dict[idx] = []`：缓存判据是
                # 「下标在不在字典里」（空列表也是有效缓存），预置空列表等于告诉
                # `_load_detection_boxes_for_index()`「这张已加载过、结果就是空的」，
                # 磁盘上的框永远读不进来——用户报的「加载数据集把标签吃掉、
                # 绿点变橙点再变空心」就是这个（大文件夹走这条流式加载路径）。
                # 缓存由 `_load_detection_boxes_for_index()` 自己填。
                self._pending_image_index += 1
                if time.perf_counter() >= deadline:
                    break
        finally:
            bg_list.setUpdatesEnabled(True)

        if self._pending_image_index >= len(self._pending_image_files):
            self._background_load_timer.stop()
            self._pending_image_files = None
            self._start_dataset_label_scan()
            self.update_file_count()
            apply_filter = getattr(self, '_apply_bg_annotation_filter', None)
            if callable(apply_filter):
                apply_filter(navigate=False)
            bg_list.viewport().update()
            self._finalize_dataset_load()
            return
        QApplication.processEvents()

    def upload_small_images(self):
        """上传贴图"""
        files, _ = QFileDialog.getOpenFileNames(
            self, tr("选择贴图片"), "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if files:
            self._memory_paste_path = os.path.dirname(files[0])
            self.small_images.clear()
            self.small_list.clear()

            for file in files:
                pixmap = QPixmap(file)
                if not pixmap.isNull():
                    self.small_images.append((file, pixmap))
                    file_name = os.path.basename(file)
                    self.add_list_item(file_name, pixmap)

        self._update_paste_count_spin()
        self._refresh_small_list_view()

    def load_small_folder_images(self):
        """从文件夹加载贴图"""
        folder_path = QFileDialog.getExistingDirectory(self, tr("选择贴图文件夹"), "")
        if not folder_path:
            return
        self.load_paste_folder(folder_path)

    def load_paste_folder(self, folder_path):
        """从指定文件夹加载贴图，供文件夹按钮和记忆记录复用。"""
        self._memory_paste_path = folder_path

        self.small_images.clear()
        self.small_list.clear()

        loaded_count = 0
        for file_name in sorted(os.listdir(folder_path), key=natural_sort_key):
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                file_path = os.path.join(folder_path, file_name)
                pixmap = QPixmap(file_path)
                if not pixmap.isNull():
                    self.small_images.append((file_path, pixmap))
                    self.add_list_item(file_name, pixmap)
                    loaded_count += 1

        if loaded_count == 0:
            QMessageBox.warning(self, tr("警告"), tr("该文件夹中没有找到支持的图片文件"))

        self._update_paste_count_spin()
        self._refresh_small_list_view()

    def _update_paste_count_spin(self):
        """更新贴图个数输入框"""
        if hasattr(self, 'paste_count_spin'):
            if len(self.small_images) <= 5:
                self.paste_count_spin.setValue(len(self.small_images))
                self.paste_count_spin.setMaximum(len(self.small_images))
            else:
                self.paste_count_spin.setMaximum(len(self.small_images))

    def _refresh_small_list_view(self):
        """刷新贴图列表视图"""
        if hasattr(self, 'is_thumbnail_mode'):
            if self.is_thumbnail_mode:
                self._configure_small_list()
            else:
                self._set_list_mode()

            self.refresh_list_items()
            self.small_list.scrollToTop()
            self.small_list.updateGeometry()
            self.small_list.repaint()

    def add_list_item(self, file_name, pixmap):
        """添加列表项"""
        item = QListWidgetItem(file_name)

        if self.is_thumbnail_mode:
            thumb_pixmap = create_thumbnail(pixmap, self.thumbnail_grid_width, self.thumbnail_grid_height)
            item.setIcon(QIcon(thumb_pixmap))
            item.setSizeHint(QSize(self.thumbnail_grid_width, self.thumbnail_grid_height + 20))

        item.setData(Qt.UserRole, len(self.small_images) - 1)
        self.small_list.addItem(item)

    def refresh_list_items(self):
        """刷新列表项"""
        self.small_list.clear()
        for idx, (file_path, pixmap) in enumerate(self.small_images):
            file_name = os.path.basename(file_path)
            item = QListWidgetItem(file_name)

            if self.is_thumbnail_mode:
                thumb_pixmap = create_thumbnail(pixmap, self.thumbnail_grid_width, self.thumbnail_grid_height)
                item.setIcon(QIcon(thumb_pixmap))
                item.setSizeHint(QSize(self.thumbnail_grid_width, self.thumbnail_grid_height + 20))

            item.setData(Qt.UserRole, idx)
            self.small_list.addItem(item)

    def upload_paste_labels(self):
        """上传贴图标签文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("导入贴图标签文件"), "", "Text Files (*.txt)"
        )
        if file_path:
            self.load_paste_label_file(file_path)

    def upload_background_labels(self):
        """上传背景标签文件（一行一个类别）。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("选择背景标签文件"), "", "Text Files (*.txt)"
        )
        if file_path:
            self.load_background_label_file(file_path)

    def load_background_label_file(self, file_path):
        """从 txt 读背景图类别，与扫描结果取并集（不覆盖已有类别）。

        导入的类别记在 imported_background_labels 里，因为
        background_dataset_labels 每次扫描都是整体赋值，单独存一份才不会丢。
        """
        if not hasattr(self, 'imported_background_labels'):
            self.imported_background_labels = set()
        try:
            labels = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            labels.append(parts[0])
        except Exception as e:
            QMessageBox.critical(self, tr("错误"), f"{tr('读取标签文件失败：')}{e}")
            return

        labels = [label for label in labels if label]
        if not labels:
            QMessageBox.warning(self, tr("警告"), tr("未找到有效的标签"))
            return

        self.imported_background_labels.update(labels)
        self._sync_background_labels()
        self._memory_label_path = file_path

    def _sync_background_labels(self):
        """把导入的类别并进扫描结果，并刷新标签列表。"""
        if not hasattr(self, 'background_dataset_labels'):
            self.background_dataset_labels = set()
        scanned = set(getattr(self, '_scanned_background_labels', None) or ())
        imported = set(getattr(self, 'imported_background_labels', None) or ())
        self.background_dataset_labels = scanned | imported
        self.global_labels.update(imported)
        if hasattr(self, 'update_label_list'):
            self.update_label_list()

    def load_paste_label_file(self, file_path):
        """从指定标签文件加载贴图标签，供文件按钮和记忆记录复用。"""
        self._memory_label_path = file_path
        try:
            labels = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            labels.append(parts[0])

            if labels:
                self.paste_label_list.clear()
                for label in labels:
                    self.paste_label_list.addItem(label)
            else:
                QMessageBox.warning(self, tr("警告"), tr("未找到有效的标签"))
        except Exception as e:
            QMessageBox.critical(self, tr("错误"), f"{tr('读取标签文件失败：')}{e}")

    def _load_detection_boxes_for_index(self, index, file_path):
        """加载指定索引的检测框（优先从缓存，否则从文件）

        缓存判据是「**这个下标在不在字典里**」，不能写成 `len(...) > 0`。
        空列表（0 框）也是有效缓存：覆盖模式刚把这张图清空时，若把空缓存
        当成「没加载过」而重读磁盘，磁盘上的旧 json 会把刚清掉的框又搬回来，
        连带 `detection_boxes_dict[index]` 也被旧内容覆盖——用户报的
        「一键推理跑完不更新结果，切到那张图又恢复原样」就是这个。
        """
        from_cache = index in self.detection_boxes_dict
        if from_cache:
            self.detection_boxes = self.detection_boxes_dict[index].copy()
        else:
            self.detection_boxes = self.load_detection_boxes(file_path)
            self.detection_boxes_dict[index] = self.detection_boxes.copy()

        from ...core import json_trace
        json_trace.trace_load_result(index, file_path, len(self.detection_boxes),
                                     from_cache)

    def load_detection_boxes(self, file_path):
        """加载检测框 JSON 文件"""
        from ..shape_io import box_from_labelme_shape
        base_name = os.path.splitext(file_path)[0]
        json_path = f"{base_name}.json"
        detection_boxes = []

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "shapes" in data:
                        shapes = data["shapes"]
                        if isinstance(shapes, list):
                            for shape in shapes:
                                box = box_from_labelme_shape(shape)
                                if box is not None:
                                    detection_boxes.append(box)
            except Exception as e:
                from ...core.exception_hook import _write_log
                _write_log(f"加载检测框文件失败：{e}")

        return detection_boxes

    def load_image_by_index(self, index):
        """加载指定索引的图片"""
        if 0 <= index < len(self.background_images):
            file_path = self.background_images[index]
            pixmap = self._get_cached_pixmap(file_path)
            if not pixmap.isNull():
                self.current_background = pixmap
                self._load_detection_boxes_for_index(index, file_path)
                self.canvas.reset_view()
                self.update_label_list()
                self.canvas.update()
            else:
                from ...core.exception_hook import _write_log
                _write_log(f"警告: 图片加载失败或为空: {file_path}")

    def select_background(self, item):
        """选择背景图"""
        if self._busy:
            return
        try:
            if item is None:
                return

            if self._is_delete_view:
                idx = self.background_list.row(item)
                if 0 <= idx < len(self._delete_files):
                    self._delete_current_idx = idx
                    self._load_delete_image(idx)
                    filename = os.path.basename(self._delete_files[idx])
                    total = len(self._delete_files)
                    if self.current_background:
                        w = self.current_background.width()
                        h = self.current_background.height()
                        self.setWindowTitle(f"LabelForge - {filename} [{w} x {h}] [{idx + 1} / {total}]")
                return

            index = item.data(BG_ROLE_INDEX)
            if index is None:
                return

            if self.current_background_index >= 0:
                self.canvas_items_dict[self.current_background_index] = self.canvas_items.copy()
                self.detection_boxes_dict[self.current_background_index] = self.detection_boxes.copy()
                # 诊断：切走之前先记下**上一张**的 json 状态。若它随后消失，
                # 下面这次 save_current_json() 就是第一嫌疑人。
                from ...core import json_trace
                if 0 <= self.current_background_index < len(self.background_images):
                    json_trace.trace(
                        '切走前(即将 save_current_json)',
                        f"{os.path.splitext(self.background_images[self.current_background_index])[0]}.json",
                        idx=self.current_background_index,
                        memory_boxes=len(self.detection_boxes))
                self.save_current_json()

            self.current_background_index = index

            if index not in self.canvas_items_dict:
                self.canvas_items_dict[index] = []
            self.canvas_items = self.canvas_items_dict[index].copy()

            if 0 <= index < len(self.background_images):
                file_path = self.background_images[index]
                pixmap = self._get_cached_pixmap(file_path)
                if not pixmap.isNull():
                    self.current_background = pixmap
                    self._load_detection_boxes_for_index(index, file_path)
                else:
                    self.current_background = None
                    self.detection_boxes = []
                # 诊断：切到的新图，它盘上的 json 在不在、有几条 shape
                from ...core import json_trace
                json_trace.trace('切到', f"{os.path.splitext(file_path)[0]}.json",
                                 index=index,
                                 loaded_boxes=len(self.detection_boxes))

            self.update_label_list()
            self.canvas.reset_view()
            self.selected_item = None
            self.canvas.update()
            self.update_file_count()
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self._log_error(f"select_background 错误: {e}\n{error_msg}")

    def _start_dataset_label_scan(self):
        """Start a label-only scan for the current immutable image list."""
        image_paths = tuple(self.background_images)
        worker = DatasetLabelScanWorker(
            self._background_label_scan_generation, image_paths, self
        )
        worker.labels_scanned.connect(self._apply_dataset_labels)
        worker.statuses_progress.connect(self._apply_statuses_progress)
        worker.finished.connect(lambda worker=worker: self._on_dataset_label_scan_finished(worker))
        worker.finished.connect(worker.deleteLater)
        if not hasattr(self, '_background_label_scan_workers'):
            self._background_label_scan_workers = set()
        self._background_label_scan_workers.add(worker)
        self._background_label_scan_worker = worker
        self._background_label_scan_in_progress = True
        self._background_label_scan_pending = True
        self._background_label_scan_completed = False
        worker.start()

    def _on_dataset_label_scan_finished(self, worker):
        """Release finished workers and clear pending state after cancellation."""
        self._background_label_scan_workers.discard(worker)
        if worker is not self._background_label_scan_worker:
            return
        self._background_label_scan_worker = None
        self._background_label_scan_in_progress = False
        if (worker.isInterruptionRequested()
                and getattr(self, '_background_label_scan_pending', False)):
            self._background_label_scan_pending = False
            self._background_label_scan_completed = False
            if (hasattr(self, '_processing_panel') and self._processing_panel
                    and self._processing_panel.isVisible()):
                self._update_processing_panel_labels()

    def _apply_dataset_labels(self, generation, image_paths, labels, counts=None,
                              statuses=None, tasks=None):
        """Apply only the latest worker result for the unchanged dataset."""
        if (generation != self._background_label_scan_generation
                or tuple(self.background_images) != tuple(image_paths)):
            return
        self._background_label_scan_in_progress = False
        self._background_label_scan_pending = False
        labels = set(labels or ())
        # 扫描结果单独存一份；对外暴露的是「扫描 ∪ 手动导入」，手动导入不丢
        self._scanned_background_labels = set(labels)
        imported = set(getattr(self, 'imported_background_labels', None) or ())
        if not hasattr(self, 'background_dataset_labels'):
            self.background_dataset_labels = set()
        self.background_dataset_labels = labels | imported
        self.global_labels.update(labels)
        if isinstance(counts, dict):
            color_map = getattr(self, 'label_color_map', None) or {}
            task_map = tasks if isinstance(tasks, dict) else {}
            stats = []
            for label, count in sorted(counts.items(), key=lambda x: (-int(x[1] or 0), x[0])):
                if int(count or 0) <= 0:
                    continue
                color = ''
                if isinstance(color_map, dict) and color_map.get(label):
                    color = color_map[label]
                elif hasattr(self, 'get_label_color'):
                    try:
                        color = self.get_label_color(label)
                    except Exception:
                        color = ''
                stats.append({
                    'label': label, 'count': int(count), 'color': color or '',
                    'tasks': sorted(task_map.get(label, set()) or set()),
                })
            self._cached_bg_label_stats = stats
            if not self._cached_bg_label_stats_path:
                self._cached_bg_label_stats_path = getattr(self, '_memory_background_path', '') or ''
            if hasattr(self, '_dataset_stats_dirty'):
                self._dataset_stats_dirty = False
        self._background_label_scan_completed = True
        if isinstance(statuses, dict) and statuses:
            self._apply_background_statuses(statuses)
        self.update_label_list()
        if (hasattr(self, '_processing_panel') and self._processing_panel
                and self._processing_panel.isVisible()):
            self._update_processing_panel_labels()

    def _apply_statuses_progress(self, generation, image_paths, batch):
        """Stream partial scan results into the list as they arrive."""
        if (generation != self._background_label_scan_generation
                or tuple(self.background_images) != tuple(image_paths)):
            return
        if isinstance(batch, dict) and batch:
            self._apply_background_statuses(batch)

    def _bg_item_index_map(self):
        """Cached {image_path: QListWidgetItem} for the current background list."""
        bg_list = getattr(self, 'background_list', None)
        if bg_list is None:
            return {}
        count = bg_list.count()
        cached = getattr(self, '_bg_item_index_cache', None)
        if cached is not None and cached[0] == count:
            return cached[1]
        mapping = {}
        for row in range(count):
            item = bg_list.item(row)
            if item is None:
                continue
            path = item.data(BG_ROLE_PATH)
            if path:
                mapping[path] = item
        self._bg_item_index_cache = (count, mapping)
        return mapping

    def _apply_background_statuses(self, statuses):
        """Fill resolved per-image statuses into already-built list rows."""
        bg_list = getattr(self, 'background_list', None)
        if bg_list is None:
            return
        # 记下解析结果：切视图重建列表项时只能做廉价判定（只看 sidecar 在不在），
        # 没有这份缓存就会把已标注的图重新显示成「扫描中」的灰圈，直到切图才恢复。
        cache = getattr(self, '_bg_status_by_path', None)
        if cache is None:
            cache = self._bg_status_by_path = {}
        cache.update({p: s for p, s in statuses.items() if s is not None})
        mode = getattr(self, '_bg_annotation_filter', 'all')
        mapping = self._bg_item_index_map()
        for image_path, status in statuses.items():
            if status is None:
                continue
            item = mapping.get(image_path)
            if item is None:
                continue
            apply_background_status_to_item(item, status)
            if mode != 'all':
                item.setHidden(status != mode)
        viewport = getattr(bg_list, 'viewport', None)
        if callable(viewport):
            viewport().update()

    def _cleanup_background_label_scan_worker(self):
        """Request a bounded shutdown before the window releases its worker."""
        if hasattr(self, '_background_load_timer') and self._background_load_timer:
            self._background_load_timer.stop()
            self._background_load_timer = None
            self._pending_image_files = None
        workers = set(getattr(self, '_background_label_scan_workers', set()))
        worker = getattr(self, '_background_label_scan_worker', None)
        if worker is not None:
            workers.add(worker)
        for worker in workers:
            worker.requestInterruption()
        for worker in workers:
            if worker.isRunning():
                worker.wait(2000)
        running_workers = {worker for worker in workers if worker.isRunning()}
        self._background_label_scan_workers = running_workers
        if running_workers:
            return False
        self._background_label_scan_worker = None
        self._background_label_scan_in_progress = False
        self._background_label_scan_pending = False
        self._background_label_scan_completed = False
        return True

    def update_file_count(self):
        """更新文件计数显示和标题栏"""
        import os as _os
        total = len(self.background_images)
        if total > 0 and self.current_background is not None:
            current = self.current_background_index + 1 if self.current_background_index >= 0 else 1
            if 0 <= self.current_background_index < total:
                filename = _os.path.basename(self.background_images[self.current_background_index])
                w = self.current_background.width()
                h = self.current_background.height()
                self.setWindowTitle(f"LabelForge - {filename} [{w} x {h}] [{current} / {total}]")
            else:
                self.setWindowTitle(f"LabelForge")
        else:
            self.setWindowTitle("LabelForge")

    def _log_error(self, message):
        """记录错误信息"""
        try:
            from ...core.exception_hook import _write_log
            _write_log(message)
        except Exception:
            pass
