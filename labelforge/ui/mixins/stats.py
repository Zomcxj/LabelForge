"""Label statistics behavior for the main window."""


class StatsMixin:
    def _get_session_labels(self):
        """收集当前打开文件夹中用于共享色板的全部标签。"""
        labels = [box.get('label', '') for boxes in self.detection_boxes_dict.values() for box in boxes]
        labels.extend(box.get('label', '') for box in self.detection_boxes)
        labels.extend(item[2] for item in self.canvas_items)
        labels.extend(
            item[2] for idx, items in self.canvas_items_dict.items()
            if idx != self.current_background_index for item in items
        )
        return labels

    def _get_session_paste_stats(self):
        """按图片聚合当前会话的贴图标签。"""
        stats = {}
        for idx in range(len(self.background_images)):
            items = self.canvas_items if idx == self.current_background_index else self.canvas_items_dict.get(idx, [])
            for _, _, label in items:
                if label:
                    stats[label] = stats.get(label, 0) + 1
        return stats

    def _collect_bg_stats_for_dialog(self):
        """Background label counts: reuse full-dataset cache when fresh, else seed disk+memory."""
        cached = getattr(self, '_cached_bg_label_stats', None) or []
        current_path = getattr(self, '_memory_background_path', '') or ''
        cache_path = getattr(self, '_cached_bg_label_stats_path', '') or ''
        scan_done = getattr(self, '_background_label_scan_completed', False)
        dirty = getattr(self, '_dataset_stats_dirty', False)
        if not (cached and scan_done and not dirty and cache_path == current_path):
            if hasattr(self, 'label_manager') and hasattr(self.label_manager, '_seed_stats_cache_from_disk_and_memory'):
                self.label_manager._seed_stats_cache_from_disk_and_memory()
            self._dataset_stats_dirty = False
            cached = getattr(self, '_cached_bg_label_stats', None) or []
        current_path = getattr(self, '_memory_background_path', '') or ''
        # Prefer any non-empty live cache (renames/canvas edits update it without path).
        if cached:
            bg_stats = {}
            for item in cached:
                if not isinstance(item, dict):
                    continue
                label = str(item.get('label', '') or '').strip()
                if not label:
                    continue
                try:
                    count = max(0, int(item.get('count', 0) or 0))
                except (TypeError, ValueError):
                    count = 0
                bg_stats[label] = count
                color = str(item.get('color', '') or '').strip()
                if color and hasattr(self, 'label_color_map'):
                    self.label_color_map.setdefault(label, color)
            if bg_stats:
                return bg_stats

        from ...engine.image_loader import collect_background_label_counts, collect_background_label_tasks
        bg_stats = collect_background_label_counts(list(self.background_images or []))
        bg_tasks = collect_background_label_tasks(list(self.background_images or []))
        for lbl in getattr(self, 'background_dataset_labels', set()) or set():
            bg_stats.setdefault(lbl, 0)
        for lbl in self.global_labels:
            if lbl in bg_stats or lbl in (getattr(self, 'background_dataset_labels', set()) or set()):
                bg_stats.setdefault(lbl, 0)
        self._cached_bg_label_stats = [
            {'label': label, 'count': count, 'color': self.get_label_color(label),
             'tasks': sorted(bg_tasks.get(label, set()))}
            for label, count in sorted(bg_stats.items(), key=lambda x: (-x[1], x[0]))
        ]
        self._cached_bg_label_stats_path = current_path
        return bg_stats

    def _collect_bg_label_tasks_for_dialog(self):
        """{label: [task,...]} 用于统计界面「框类型」列。"""
        from ...core.utils import shape_task_type
        cached = getattr(self, '_cached_bg_label_stats', None) or []
        tasks = {}
        for item in cached:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip()
            if not label:
                continue
            for t in item.get('tasks') or []:
                tasks.setdefault(label, set()).add(t)
        # 实时内存框（改名/新建后缓存可能滞后）
        for boxes in list(getattr(self, 'detection_boxes_dict', {}).values()) + [getattr(self, 'detection_boxes', [])]:
            for box in boxes or []:
                if not isinstance(box, dict):
                    continue
                label = str(box.get('label', '') or '').strip()
                if label:
                    tasks.setdefault(label, set()).add(shape_task_type(box))
        return {label: sorted(ts) for label, ts in tasks.items()}

    def _cache_bg_stats(self, counts, tasks):
        """把扫描结果写回统计缓存（`_cached_bg_label_stats`）。

        沿用与 `_collect_bg_stats_for_dialog()` 相同的条目结构，这样重命名/改色
        走的 `_reload_stats_bg_table()` 读到的是同一份数据，不会出现「表格显示的
        是扫描结果、重命名后却退回旧缓存」。
        """
        rows = [
            {'label': label, 'count': count,
             'color': self.get_label_color(label),
             'tasks': sorted(tasks.get(label, []))}
            for label, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        self._cached_bg_label_stats = rows
        self._cached_bg_label_stats_path = (
            getattr(self, '_memory_background_path', '') or '')
        self._dataset_stats_dirty = False

    def _cached_bg_stats_only(self):
        """只读**内存缓存**的背景标签计数；不做任何磁盘扫描。

        统计弹窗现在先弹窗、再后台扫描（见 `_show_label_stats`）。首次打开时
        这份缓存通常还没建立，返回空表、由后台扫描填充；已经有缓存时（列表已
        扫过一轮）立刻显示，用户不用等。
        """
        cached = getattr(self, '_cached_bg_label_stats', None) or []
        out = {}
        for item in cached:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip()
            if not label:
                continue
            try:
                out[label] = max(0, int(item.get('count', 0) or 0))
            except (TypeError, ValueError):
                out[label] = 0
        return out

    def _cached_bg_tasks_only(self):
        """只读内存缓存里的「框类型」；与 `_cached_bg_stats_only()` 配套。"""
        cached = getattr(self, '_cached_bg_label_stats', None) or []
        tasks = {}
        for item in cached:
            if not isinstance(item, dict):
                continue
            label = str(item.get('label', '') or '').strip()
            if not label:
                continue
            for t in item.get('tasks') or []:
                tasks.setdefault(label, set()).add(t)
        return tasks

    def _start_stats_scan(self, on_done, on_error=None):
        """后台扫一遍数据集，完成后回主线程调 `on_done(counts, tasks)`。

        数据集为空时同步回一次空结果（不必起线程）。同一时刻只允许一个统计
        扫描：新的会先请掉上一个，避免用户连点两次时两组结果互相覆盖。
        """
        previous = getattr(self, '_stats_scan_worker', None)
        if previous is not None:
            try:
                previous.requestInterruption()
            except Exception:
                pass
            self._stats_scan_worker = None

        images = list(getattr(self, 'background_images', None) or [])
        if not images:
            on_done({}, {})
            return None
        from ...engine.image_loader.scan import DatasetStatsScanWorker

        worker = DatasetStatsScanWorker(images, parent=self)
        self._stats_scan_worker = worker
        worker.stats_ready.connect(on_done)
        if on_error is not None:
            worker.finished.connect(
                lambda w=worker: self._on_stats_scan_finished(w))
        worker.start()
        return worker

    def _on_stats_scan_finished(self, worker):
        if worker is getattr(self, '_stats_scan_worker', None):
            self._stats_scan_worker = None
        worker.deleteLater()

    def _show_label_stats(self):
        """显示标签统计弹窗"""
        self._close_health_worker()
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
            QHeaderView, QPushButton, QWidget, QAbstractItemView,
            QMainWindow, QTabWidget,
        )
        from PySide6.QtCore import Qt as QtCore
        from ..dialog_helpers import center_on_parent
        from ..theme import ThemeManager
        from .. import i18n
        from PySide6.QtCore import QTimer
        tr = i18n.t

        class _StatsDialog(QDialog):
            def showEvent(self, event):
                super().showEvent(event)
                center_on_parent(self, self.parent())

        t = ThemeManager.get_theme()
        dialog = _StatsDialog(self)
        dialog.setWindowTitle(tr("标签统计"))
        dialog.setMinimumSize(1200, 640)
        dialog.resize(1360, 760)
        def _sync():
            hwnd = int(dialog.winId())
            from ..dwm import set_titlebar_dark
            set_titlebar_dark(hwnd, is_dark)
        is_dark = ThemeManager.get_mode().value == "dark"
        QTimer.singleShot(30, _sync)
        dialog.setStyleSheet(f"""
            QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QTableWidget {{ background-color: {t['widget_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px;
                gridline-color: {t['border_color']}; }}
            QTableWidget::item {{ padding: 4px; }}
            QTableWidget::item:selected {{ background-color: {t['accent_light']}; color: {t['accent']}; }}
            QHeaderView::section {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; padding: 4px; font-weight: bold; }}
            QHeaderView::section:first {{ border-top-left-radius: 4px; }}
            QHeaderView::section:last {{ border-top-right-radius: 4px; }}
            QTableWidget QTableCornerButton::section {{ background-color: {t['panel_bg']};
                border: 1px solid {t['border_color']}; border-top-left-radius: 4px; }}
            QTabWidget::pane {{ background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']}; border-radius: 4px;
                top: -1px; }}
            QTabBar::tab {{ background-color: {t['panel_bg']}; color: {t['text_secondary']};
                border: 1px solid {t['border_color']}; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                padding: 5px 16px; }}
            QTabBar::tab:selected {{ background-color: {t['widget_bg']};
                color: {t['text_primary']}; }}
            QTabBar::tab:hover {{ background-color: {t['list_hover']}; }}
            QDockWidget {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QDockWidget::title {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-top-left-radius: 4px;
                border-top-right-radius: 4px; padding: 4px 8px; font-weight: bold; }}
        """)
        layout = QVBoxLayout(dialog)
        from PySide6.QtWidgets import QMainWindow

        class _DockHost(QMainWindow):
            """dock 容器：尺寸稳定后重新应用左右比例。

            resizeEvent 触发时 QMainWindow 的 dock 布局尚未完成，直接
            resizeDocks 会被随后的布局覆盖；排到事件循环下一轮再应用。
            """

            def resizeEvent(self, event):
                parent_resize = getattr(super(), 'resizeEvent', None)
                if parent_resize is not None:
                    try:
                        parent_resize(event)
                    except Exception:
                        pass
                apply_ratio = getattr(self, '_ratio_fn', None)
                if apply_ratio is None or getattr(self, '_ratio_pending', False):
                    return
                self._ratio_pending = True

                def _run():
                    self._ratio_pending = False
                    try:
                        apply_ratio()
                    except Exception:
                        pass

                try:
                    QTimer.singleShot(0, _run)
                except Exception:
                    _run()

        host = _DockHost()
        host.setDockNestingEnabled(True)
        dialog._dock_host = host
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        host.setCentralWidget(central)
        layout.addWidget(host)
        # ⑦ 先在 UI 线程拿**内存缓存**（有就是现成的，立刻显示），磁盘扫描交给
        # 后台线程：原先 `_collect_bg_stats_for_dialog()` 会在主线程把每个
        # sidecar json 解析两遍，数据集一大就要好几秒（用户报的「统计有时候
        # 需要 5s 才能打开」）。现在弹窗先出来，扫描完成再填充。
        bg_stats = self._cached_bg_stats_only()
        bg_tasks = {}
        bg_table = QTableWidget(len(bg_stats), 4)
        bg_table.setHorizontalHeaderLabels([tr("类别"), tr("数量"), tr("框类型"), tr("颜色")])
        bg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bg_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)

        def _fill_bg_table(stats, tasks):
            """把（可能来自后台的）结果填进表里；可反复调用（扫描完再填一次）。"""
            bg_table.blockSignals(True)
            bg_table.setRowCount(len(stats))
            for row, (label, count) in enumerate(
                    sorted(stats.items(), key=lambda x: (-x[1], x[0]))):
                label_item = QTableWidgetItem(label)
                label_item.setFlags(label_item.flags() | QtCore.ItemIsEditable)
                label_item.setData(QtCore.UserRole, label)
                bg_table.setItem(row, 0, label_item)
                count_item = QTableWidgetItem(str(count))
                count_item.setFlags(count_item.flags() & ~QtCore.ItemIsEditable)
                bg_table.setItem(row, 1, count_item)
                task_item = QTableWidgetItem(" ".join(sorted(tasks.get(label, []))))
                task_item.setFlags(task_item.flags() & ~QtCore.ItemIsEditable)
                bg_table.setItem(row, 2, task_item)
                color_button = QPushButton()
                self._set_label_color_button(
                    color_button, self.get_label_color(label))
                color_button.clicked.connect(
                    lambda _, value=label, button=color_button:
                    self._change_label_color(value, dialog, button))
                bg_table.setCellWidget(row, 3, color_button)
            bg_table.blockSignals(False)

        def _mark_bg_scanning():
            """扫描期间的空态：**不加假行**，改用总计标签说一句「正在统计…」。

            早先试过在表里插一行「正在统计…」，但那行会被「背景表求和」看到
            （总计标签与表是同源的，测试也这么钉），于是出现 `int('')` 之类的
            口径错位。背景表本身保持真正的空（0 行、求和为 0），加载中这件事由
            总计标签表达。
            """
            bg_table.blockSignals(True)
            bg_table.setRowCount(0)
            bg_table.blockSignals(False)

        _fill_bg_table(bg_stats, bg_tasks)

        def _on_bg_label_changed(item):
            if item is None or item.column() != 0:
                return
            old_label = item.data(QtCore.UserRole) or ''
            new_label = (item.text() or '').strip()
            if not old_label:
                return
            if not new_label or new_label == old_label:
                bg_table.blockSignals(True)
                item.setText(old_label)
                bg_table.blockSignals(False)
                return
            # 故意**不拦重名**：改成已存在的标签 = 把这一类合并进那一类
            # （用户场景：把 `drone` 并进 `Drone`）。底层
            # `rename_detection_label` 本来就支持合并——它显式取
            # `target_exists`，把两组的框合到一起、数量相加、保留目标颜色。
            # 这里原先挡了一道「重名就弹回原名」，于是用户在统计窗口里怎么改都
            # 改不动，还以为功能坏了。
            if self.label_manager.rename_detection_label(old_label, new_label):
                item.setData(QtCore.UserRole, new_label)
                self._health_scan_generation = getattr(self, '_health_scan_generation', 0) + 1
                self._reload_stats_bg_table(bg_table, dialog)
                if hasattr(self, 'update_label_list'):
                    self.update_label_list()
                if (hasattr(self, '_processing_panel') and self._processing_panel
                        and self._processing_panel.isVisible()):
                    self._update_processing_panel_labels()
                self.canvas.update()
            else:
                bg_table.blockSignals(True)
                item.setText(old_label)
                bg_table.blockSignals(False)

        bg_table.itemChanged.connect(_on_bg_label_changed)
        dialog._bg_table = bg_table
        dialog._on_bg_label_changed = _on_bg_label_changed
        paste_stats = self._get_session_paste_stats()
        paste_table = QTableWidget(len(paste_stats), 3)
        paste_table.setHorizontalHeaderLabels([tr("类别"), tr("数量"), tr("颜色")])
        paste_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        paste_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        for row, (label, count) in enumerate(sorted(paste_stats.items(), key=lambda x: -x[1])):
            label_item = QTableWidgetItem(label)
            label_item.setFlags(label_item.flags() | QtCore.ItemIsEditable)
            label_item.setData(QtCore.UserRole, label)
            paste_table.setItem(row, 0, label_item)
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~QtCore.ItemIsEditable)
            paste_table.setItem(row, 1, count_item)
            color_button = QPushButton()
            self._set_label_color_button(color_button, self.get_label_color(label))
            color_button.clicked.connect(lambda _, value=label, button=color_button: self._change_label_color(value, dialog, button))
            paste_table.setCellWidget(row, 2, color_button)

        def _on_paste_label_changed(item):
            if item is None or item.column() != 0:
                return
            old_label = item.data(QtCore.UserRole) or ''
            new_label = (item.text() or '').strip()
            if not old_label:
                return
            if not new_label or new_label == old_label:
                paste_table.blockSignals(True)
                item.setText(old_label)
                paste_table.blockSignals(False)
                return
            # 与背景图表同一口径：不拦重名，重名走底层合并（`rename_paste_label`
            # 会把旧名的全部项改成新名并归并颜色）。两处必须一致，否则「背景图
            # 能合并、贴图不能合并」这种半修状态又要被用户抓到。
            if self.label_manager.rename_paste_label(old_label, new_label, rewrite_disk=True):
                self._health_scan_generation = getattr(self, '_health_scan_generation', 0) + 1
                # 必须**重建整张表**，不能只改当前行：改名到已存在的标签是
                # 合并，两行要并成一行（旧的只改本行，于是出现两行同名的
                # `badge` 各带一个计数——用户看到的「改不过去」的另一半）。
                # 背景图表早就这么做了（`_reload_stats_bg_table`），贴图这边
                # 先前漏了。
                self._reload_stats_paste_table_from_session(paste_table, dialog)
                self.canvas.update()
            else:
                paste_table.blockSignals(True)
                item.setText(old_label)
                paste_table.blockSignals(False)

        paste_table.itemChanged.connect(_on_paste_label_changed)
        dialog._paste_table = paste_table
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.tabBar().setDrawBase(False)
        bg_tab = QWidget()
        bg_tl = QVBoxLayout(bg_tab)
        bg_tl.setContentsMargins(0, 0, 0, 0)
        bg_tl.addWidget(bg_table)
        paste_tab = QWidget()
        paste_tl = QVBoxLayout(paste_tab)
        paste_tl.setContentsMargins(0, 0, 0, 0)
        paste_tl.addWidget(paste_table)
        tabs.addTab(bg_tab, tr('背景图标签'))
        tabs.addTab(paste_tab, tr('贴图标签_list'))
        central_layout.addWidget(tabs)
        dialog._tabs = tabs
        self._build_health_section(dialog, central_layout)
        total = QLabel()
        total.setStyleSheet("font-size: 12px; margin-top: 8px;")
        central_layout.addWidget(total)
        dialog._total_label = total
        self._update_stats_total_label(
            dialog, bg_count=sum(bg_stats.values()),
            paste_count=sum(paste_stats.values()))

        # ⑦ 立刻显示弹窗，再后台扫盘填充背景表：扫描期间先摆一句「正在统计…」，
        # 免得用户对着空表以为没数据。`exec_()` 之前就起线程，弹窗一出来扫描
        # 已经在跑了。
        def _on_stats_ready(counts, tasks):
            # 结果一律先落缓存（重命名/改色走的是同一份缓存），再决定要不要刷新表格。
            # **不能**用 `dialog.isVisible()` 当「用户是否已关掉」的判据：首帧
            # 还没来得及显示时它就是 False，那样表格会永远停在「正在统计…」。
            # 关掉弹窗的信号是 `finished`（见下方 _on_closed）。
            self._cache_bg_stats(counts, tasks)
            if getattr(dialog, '_stats_closed', False):
                return
            dialog._scan_applied = True
            _fill_bg_table(counts, tasks)
            self._update_stats_total_label(
                dialog, bg_count=sum(counts.values()),
                paste_count=sum(paste_stats.values()))

        dialog._stats_closed = False
        dialog.finished.connect(
            lambda _r: setattr(dialog, '_stats_closed', True))
        dialog._scan_applied = False
        if bg_stats:
            # 已有缓存：先显示缓存，后台仍扫一遍以保证新鲜
            _fill_bg_table(bg_stats, self._cached_bg_tasks_only())
        else:
            # 还没扫过：表先空着（0 行，求和不含假数据）。加载中这件事不改总计
            # 标签的**格式**（它是 `总计: ... | ...`，别处按 `|` 拆开读），而是
            # 在标签前面挂一句提示，扫描完再被正常的总计覆盖。
            _mark_bg_scanning()
            # 格式必须与正常总计**完全一致**（`总计: ... | ...`，两段）：别处按
            # `|` 拆开读，多一段就会 unpack 失败。加载中只是在前面挂一句提示。
            dialog._total_label.setText(
                f"{tr('正在统计…')} · {tr('总计')}: "
                f"{tr('背景图标签')} 0 {tr('个')} | "
                f"{tr('贴图标签_list')} {sum(paste_stats.values())} {tr('个')}")
        self._start_stats_scan(_on_stats_ready)
        dialog.exec_()

    def _update_stats_total_label(self, dialog, bg_count=None, paste_count=None):
        """更新统计弹窗底部总计标签。

        计数来源：背景 = 统计缓存 `_cached_bg_stats_only()`（与背景表同源，
        勿传 payload 的 annot 计数，它排除贴图/关键点；缓存未建立时按 0，
        后台扫描完成会再刷）；贴图 = payload 整库统计优先，缺省回退会话内存
        `_get_session_paste_stats()`。
        """
        from ..i18n import t as tr
        label = getattr(dialog, '_total_label', None)
        if label is None:
            return
        if not isinstance(bg_count, int) or isinstance(bg_count, bool):
            # 用**缓存**而不是 `_collect_bg_stats_for_dialog()`：后者会同步扫盘，
            # 正是 ⑦ 要避免的那件事。这里只是补一个缺省值，缓存没有就按 0 算，
            # 后台扫描完成时 `_on_stats_ready` 会带着真实数字再刷一次。
            cached = self._cached_bg_stats_only()
            bg_count = sum(cached.values())
        if not isinstance(paste_count, int) or isinstance(paste_count, bool):
            paste_count = sum(self._get_session_paste_stats().values())
        label.setText(
            f"{tr('总计')}: {tr('背景图标签')} {bg_count} {tr('个')} | "
            f"{tr('贴图标签_list')} {paste_count} {tr('个')}"
        )

    def _close_health_worker(self):
        """中断并清理健康扫描 worker。"""
        worker = getattr(self, '_health_worker', None)
        self._health_worker = None
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    worker.wait(3000)
                if worker.isRunning():
                    # 超时未停：先断开信号再丢引用，避免 worker 向已销毁的弹窗发结果
                    try:
                        worker.health_ready.disconnect()
                    except Exception:
                        pass
            except Exception:
                pass

    def _build_health_section(self, dialog, layout):
        """构建「数据集健康」区：4 个可拖拽 dock 面板 + 建议，切换标签页切换数据源。"""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (
            QLabel, QWidget, QVBoxLayout, QDockWidget)
        from ..i18n import t as tr
        from ..theme import ThemeManager
        from ..widgets.health_charts import HealthBarChart

        title_label = QLabel(tr('数据集健康'))
        title_label.setStyleSheet(
            "font-weight: bold; font-size: 13px; padding: 2px 0;")
        layout.addWidget(title_label)

        source_label = QLabel(tr('数据源：背景图标签（切换标签页）'))
        source_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(source_label)

        class_chart = HealthBarChart()
        size_chart = HealthBarChart()
        aspect_chart = HealthBarChart()
        iou_chart = HealthBarChart()
        charts = {
            'class': class_chart, 'size': size_chart, 'aspect': aspect_chart,
            'iou': iou_chart,
        }
        panels = (
            ('class', tr('类别分布'), tr('各类别占比应接近均衡；长尾类别建议多合成')),
            ('size', tr('尺寸分布'), tr('框尺寸应覆盖多种尺度，避免集中于单一范围')),
            ('aspect', tr('长宽比分布'), tr('长宽比多样化更贴近真实场景')),
            ('iou', tr('IoU 重叠分布'), tr('高 IoU 区间框多说明重复标注偏多')),
        )
        host = getattr(dialog, '_dock_host', None) or dialog
        dialog._health_docks = {}
        _last_norm = {}

        def _normalize_dock_sizes():
            """同一行的 dock 等宽（1:1）：拖动重排或窗口变化后恢复均衡。"""
            docks = list(dialog._health_docks.values())
            if len(docks) < 2:
                return
            try:
                items = [(dock, dock.geometry()) for dock in docks]
            except Exception:
                return
            rows = []
            for dock, g in items:
                placed = False
                for row in rows:
                    g0 = row[0][1]
                    v_overlap = min(g0.bottom(), g.bottom()) - max(g0.top(), g.top())
                    x_disjoint = g.right() < g0.left() or g0.right() < g.left()
                    if x_disjoint and v_overlap > min(g0.height(), g.height()) * 0.5:
                        row.append((dock, g))
                        placed = True
                        break
                if not placed:
                    rows.append([(dock, g)])
            for row in rows:
                if len(row) < 2:
                    continue
                row.sort(key=lambda t: t[1].x())
                widths = [g.width() for _, g in row]
                if max(widths) - min(widths) <= 1:
                    continue
                width = int(sum(widths) / len(widths))
                key = tuple(id(dock) for dock, _ in row)
                if width <= 0 or _last_norm.get(key) == width:
                    continue
                _last_norm[key] = width
                host.resizeDocks([dock for dock, _ in row],
                                 [width] * len(row), Qt.Horizontal)

            # 单列（全部竖向堆叠）→ 高度自动 1:1
            columns = []
            for dock, g in items:
                placed = False
                for col in columns:
                    g0 = col[0][1]
                    h_overlap = min(g0.right(), g.right()) - max(g0.left(), g.left())
                    y_disjoint = g.bottom() < g0.top() or g0.bottom() < g.top()
                    if y_disjoint and h_overlap > min(g0.width(), g.width()) * 0.5:
                        col.append((dock, g))
                        placed = True
                        break
                if not placed:
                    columns.append([(dock, g)])
            if len(columns) == 1 and len(columns[0]) >= 2:
                col = sorted(columns[0], key=lambda t: t[1].y())
                heights = [g.height() for _, g in col]
                if max(heights) - min(heights) > 1:
                    height = int(sum(heights) / len(heights))
                    key = tuple(id(dock) for dock, _ in col) + ("v",)
                    if height > 0 and _last_norm.get(key) != height:
                        _last_norm[key] = height
                        host.resizeDocks([dock for dock, _ in col],
                                         [height] * len(col), Qt.Vertical)

        if hasattr(host, 'addDockWidget'):
            for key, title, desc in panels:
                panel = QWidget()
                pl = QVBoxLayout(panel)
                pl.setContentsMargins(6, 4, 6, 4)
                desc_label = QLabel(desc)
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: gray; font-size: 11px;")
                pl.addWidget(desc_label)
                pl.addWidget(charts[key], 1)
                dock = QDockWidget(title, host)
                dock.setObjectName(f"health_{key}")
                dock.setWidget(panel)
                dock.setFeatures(QDockWidget.DockWidgetMovable)
                dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
                host.addDockWidget(Qt.RightDockWidgetArea, dock)
                dialog._health_docks[key] = dock
                charts[key].set_placeholder(tr('正在分析'))
            # 2×2 布局（右侧）：左列 class/size，右列 iou/aspect；拖动后自动让位
            d = dialog._health_docks
            host.splitDockWidget(d['class'], d['iou'], Qt.Horizontal)
            host.splitDockWidget(d['iou'], d['aspect'], Qt.Vertical)
            host.splitDockWidget(d['class'], d['size'], Qt.Vertical)

            def _apply_dock_ratio():
                """右侧图表区占窗口 2/3，左侧表格区占 1/3。

                不能在此同步调用 _normalize_dock_sizes：resizeDocks 的结果
                要等下一轮布局才反映到 dock.width()，此时读到的是旧宽度，
                归一化会用旧平均值再次 resizeDocks，把刚设好的比例覆盖掉。
                行内等宽由 dockLocationChanged 触发的防抖归一化负责。
                """
                total = host.width()
                if total <= 0 or not dialog._health_docks:
                    return
                docks_w = int(total * 2 / 3)
                host.resizeDocks(
                    [d['class'], d['iou']],
                    [docks_w // 2, docks_w - docks_w // 2], Qt.Horizontal)

            dialog._apply_dock_ratio = _apply_dock_ratio
            host._ratio_fn = _apply_dock_ratio

            # 拖动重排后恢复行内 1:1（去抖；mock 下 QTimer 不可用时直接跳过）
            try:
                from PySide6.QtCore import QTimer
                norm_timer = QTimer(host)
                norm_timer.setSingleShot(True)
                norm_timer.setInterval(120)
                norm_timer.timeout.connect(_normalize_dock_sizes)
            except Exception:
                norm_timer = None
            dialog._dock_norm_timer = norm_timer

            def _schedule_normalize(*_a):
                if norm_timer is not None:
                    norm_timer.start()
                else:
                    _normalize_dock_sizes()

            for dock in dialog._health_docks.values():
                signal = getattr(dock, 'dockLocationChanged', None)
                if signal is not None:
                    signal.connect(_schedule_normalize)

            # dock 浮动时是独立顶层窗口，原生标题栏不受弹窗 QSS 控制。
            # 拖动中 Qt 会重建顶层窗口（WinIdChange 多次触发），而
            # topLevelChanged 只在最初触发一次且此时 winId 还是旧句柄，
            # 所以监听 WinIdChange/Show，每次都用当前 winId 重设深色。
            try:
                from PySide6.QtCore import QObject, QEvent

                class _DockTitlebarFilter(QObject):
                    def eventFilter(self, obj, event):
                        try:
                            if event.type() in (QEvent.WinIdChange, QEvent.Show):
                                if getattr(obj, 'isFloating', lambda: False)():
                                    from ..dwm import set_titlebar_dark
                                    set_titlebar_dark(
                                        int(obj.winId()),
                                        ThemeManager.get_mode().value == "dark",
                                        force_refresh=True)
                        except Exception:
                            pass
                        return False

                titlebar_filter = _DockTitlebarFilter(host)
                for dock in dialog._health_docks.values():
                    dock.installEventFilter(titlebar_filter)
                dialog._dock_titlebar_filter = titlebar_filter
            except Exception:
                dialog._dock_titlebar_filter = None
        else:
            for key in charts:
                charts[key].set_placeholder(tr('正在分析'))
        advice_label = QLabel(f"{tr('建议')}: {tr('正在分析')}")
        advice_label.setWordWrap(True)
        layout.addWidget(advice_label)

        dialog._health_charts = charts
        dialog._health_advice_label = advice_label
        dialog._health_source_label = source_label
        dialog._health_source = 'annot'
        dialog._health_payload = None

        def _render_source(source, payload=None):
            payload = payload if payload is not None else dialog._health_payload
            dialog._health_source = source
            source_label.setText(
                tr('数据源：背景图标签（切换标签页）') if source == 'annot'
                else tr('数据源：贴图标签（切换标签页）'))
            if not payload:
                return
            if payload.get('error'):
                for chart in charts.values():
                    chart.set_placeholder(tr('分析失败'))
                advice_label.setText(tr('分析失败'))
                return
            section = payload.get(source) or {}
            stats = section.get('stats') or {}
            if source == 'paste' and not stats:
                for key in ('class', 'size', 'aspect', 'iou'):
                    charts[key].set_placeholder(tr('暂无贴图数据'))
                advice_label.setText("")
                return
            class_dist = stats.get('class_dist') or []
            if class_dist:
                charts['class'].set_data(
                    [{'label': c['label'], 'value': c['count'],
                      'color': self.get_label_color(c['label'])}
                     for c in class_dist],
                    horizontal=True, highlight_extremes=True)
            else:
                charts['class'].set_placeholder(tr('未发现明显失衡'))
            size = stats.get('size_hist') or {}
            charts['size'].set_histogram(size.get('edges'), size.get('counts'),
                                         xlabel=tr('面积 (px²)'),
                                         ylabel=tr('框数'))
            aspect = stats.get('aspect_hist') or {}
            charts['aspect'].set_histogram(aspect.get('edges'),
                                           aspect.get('counts'),
                                           xlabel=tr('长宽比'), ylabel=tr('框数'))
            iou = stats.get('iou_hist') or {}
            charts['iou'].set_histogram(iou.get('edges'), iou.get('counts'),
                                        xlabel=tr('IoU'), ylabel=tr('框对数'))
            advice = section.get('advice') or [tr('未发现明显失衡')]
            advice_label.setText(f"{tr('建议')}: " + "；".join(
                tr(a['key']).format(**a['params']) if isinstance(a, dict) else a
                for a in advice))

        def _set_health_source(source):
            _render_source(source)

        dialog._set_health_source = _set_health_source
        tabs = getattr(dialog, '_tabs', None)
        if tabs is not None:
            tabs.currentChanged.connect(
                lambda idx: _set_health_source('annot' if idx == 0 else 'paste'))

        if not getattr(self, 'background_images', None):
            for chart in charts.values():
                chart.set_placeholder(tr('请先加载数据集'))
            advice_label.setText("")
            return

        from ...engine.dataset_health import DatasetHealthWorker
        self._close_health_worker()
        # 扫描期间弹窗内改名/改色会推进 _health_scan_generation，
        # 迟到的过期 payload 直接丢弃（见 _on_payload 守卫）。
        gen = getattr(self, '_health_scan_generation', 0)
        memory_boxes = {
            idx: list(boxes) for idx, boxes in
            (getattr(self, 'detection_boxes_dict', None) or {}).items()}
        canvas_items = dict(getattr(self, 'canvas_items_dict', None) or {})
        current = getattr(self, 'current_background_index', -1)
        if current >= 0:
            canvas_items[current] = list(getattr(self, 'canvas_items', None) or [])
        worker = DatasetHealthWorker(
            tuple(self.background_images), memory_boxes, canvas_items, self)
        worker.finished.connect(worker.deleteLater)

        def _on_payload(payload):
            if getattr(self, '_health_scan_generation', 0) != gen:
                return
            dialog._health_payload = payload
            paste_class = ((payload.get('paste') or {}).get('stats') or {}).get(
                'class_dist') or []
            if paste_class:
                self._reload_stats_paste_table(dialog, paste_class)
            paste_total = (payload.get('paste') or {}).get('stats', {}).get(
                'summary', {}).get('total_boxes')
            if not isinstance(paste_total, int) or isinstance(paste_total, bool):
                paste_total = None
            # 背景计数不取 payload：annot 统计排除贴图/关键点，与背景表（sidecar 全量）
            # 不同源；交由 helper 回退到 _collect_bg_stats_for_dialog()，保证与表一致。
            self._update_stats_total_label(dialog, paste_count=paste_total)
            _render_source(dialog._health_source, payload)

        dialog._on_health_payload = _on_payload
        worker.health_ready.connect(_on_payload)
        self._health_worker = worker
        worker.start()

        def _on_close(*_a):
            self._close_health_worker()
        dialog.finished.connect(_on_close)

    def _reload_stats_bg_table(self, bg_table, dialog):
        """Rebuild stats dialog background table from live cache after rename/color."""
        from PySide6.QtWidgets import QTableWidgetItem, QPushButton
        from PySide6.QtCore import Qt as QtCore
        bg_stats = self._collect_bg_stats_for_dialog()
        bg_tasks = self._collect_bg_label_tasks_for_dialog()
        rows = sorted(bg_stats.items(), key=lambda x: (-x[1], x[0]))
        bg_table.blockSignals(True)
        bg_table.setRowCount(len(rows))
        for row, (label, count) in enumerate(rows):
            label_item = QTableWidgetItem(label)
            label_item.setFlags(label_item.flags() | QtCore.ItemIsEditable)
            label_item.setData(QtCore.UserRole, label)
            bg_table.setItem(row, 0, label_item)
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~QtCore.ItemIsEditable)
            bg_table.setItem(row, 1, count_item)
            task_item = QTableWidgetItem(" ".join(bg_tasks.get(label, [])))
            task_item.setFlags(task_item.flags() & ~QtCore.ItemIsEditable)
            bg_table.setItem(row, 2, task_item)
            color_button = QPushButton()
            self._set_label_color_button(color_button, self.get_label_color(label))
            color_button.clicked.connect(lambda _, value=label, button=color_button: self._change_label_color(value, dialog, button))
            bg_table.setCellWidget(row, 3, color_button)
        bg_table.blockSignals(False)

    def _reload_stats_paste_table(self, dialog, class_dist):
        """用整库贴图统计重建贴图表格（worker payload 到达后调用）。"""
        from PySide6.QtWidgets import QTableWidgetItem, QPushButton
        from PySide6.QtCore import Qt as QtCore
        table = getattr(dialog, '_paste_table', None)
        if table is None:
            return
        rows = [(str(e.get('label', '') or ''), int(e.get('count', 0) or 0))
                for e in class_dist if isinstance(e, dict)]
        rows.sort(key=lambda x: (-x[1], x[0]))
        table.blockSignals(True)
        table.setRowCount(len(rows))
        for row, (label, count) in enumerate(rows):
            label_item = QTableWidgetItem(label)
            label_item.setFlags(label_item.flags() | QtCore.ItemIsEditable)
            label_item.setData(QtCore.UserRole, label)
            table.setItem(row, 0, label_item)
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~QtCore.ItemIsEditable)
            table.setItem(row, 1, count_item)
            color_button = QPushButton()
            self._set_label_color_button(color_button, self.get_label_color(label))
            color_button.clicked.connect(
                lambda _, value=label, button=color_button: self._change_label_color(value, dialog, button))
            table.setCellWidget(row, 2, color_button)
        table.blockSignals(False)

    def _reload_stats_paste_table_from_session(self, table, dialog):
        """按当前会话的贴图标签重建贴图表（改名/合并后调用）。

        与 `_reload_stats_paste_table`（吃 worker 的 `class_dist`）分开：改名的
        时候手上没有 worker payload，数据源就是 `canvas_items_dict` 里的贴图
        标签，`_get_session_paste_stats()` 已经在按这个聚合了。
        """
        from PySide6.QtWidgets import QTableWidgetItem, QPushButton
        from PySide6.QtCore import Qt as QtCore
        stats = self._get_session_paste_stats()
        rows = sorted(stats.items(), key=lambda x: (-x[1], x[0]))
        table.blockSignals(True)
        table.setRowCount(len(rows))
        for row, (label, count) in enumerate(rows):
            label_item = QTableWidgetItem(label)
            label_item.setFlags(label_item.flags() | QtCore.ItemIsEditable)
            label_item.setData(QtCore.UserRole, label)
            table.setItem(row, 0, label_item)
            count_item = QTableWidgetItem(str(count))
            count_item.setFlags(count_item.flags() & ~QtCore.ItemIsEditable)
            table.setItem(row, 1, count_item)
            color_button = QPushButton()
            self._set_label_color_button(color_button, self.get_label_color(label))
            color_button.clicked.connect(
                lambda _, value=label, button=color_button: self._change_label_color(value, dialog, button))
            table.setCellWidget(row, 2, color_button)
        table.blockSignals(False)

    def _set_label_color_button(self, button, color):
        button.setText(color)
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: #FFFFFF;"
            f" border: 1px solid {color}; border-radius: 4px; }}"
        )

    def _change_label_color(self, label, parent, color_button=None):
        """修改指定类别的颜色。"""
        from ...core import config_manager
        from PySide6.QtGui import QColor
        from ..dialog_helpers import ThemedColorDialog
        from ..i18n import t as tr
        if not label:
            return
        dialog = ThemedColorDialog(parent)
        dialog.setWindowTitle(tr("颜色"))
        dialog.setCurrentColor(QColor(self.get_label_color(label)))
        if dialog.exec_() != 1:
            return
        color = dialog.currentColor()
        if not color.isValid():
            return
        self.label_color_map[label] = color.name()
        config_manager.save_all(label_colors=self.label_colors, label_color_map=self.label_color_map)
        self._health_scan_generation = getattr(self, '_health_scan_generation', 0) + 1
        cached = getattr(self, '_cached_bg_label_stats', None)
        if isinstance(cached, list):
            for item in cached:
                if isinstance(item, dict) and item.get('label') == label:
                    item['color'] = color.name()
        if color_button is not None:
            self._set_label_color_button(color_button, color.name())
        self.canvas.update()
