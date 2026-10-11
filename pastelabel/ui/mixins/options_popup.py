"""Options Popup UI building mixin."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QFrame, QLineEdit
from PyQt5.QtCore import Qt, QTimer, QPoint
from ..theme import ThemeManager
from ..i18n import t as tr
from ...widgets.hover_popup import HoverDismissPopup
from ...widgets.hover_menu import HoverKeepMenu
from ..segmented_control import AnimatedSegmentedControl


class OptionsPopupMixin:
    """Options Popup widgets and behavior."""

    def _create_options_menu(self, layout):
        """创建选项下拉菜单按钮"""

        self.mode_seg = QFrame()
        self.mode_seg.setObjectName("modeSeg")
        self.mode_seg.setFixedWidth(150)
        self.mode_seg.setFixedHeight(24)
        self.mode_seg.setContentsMargins(0, 0, 0, 0)
        mode_layout = QHBoxLayout(self.mode_seg)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(0)

        self.btn_annotate_mode = QPushButton(tr("标注"))
        self.btn_annotate_mode.setObjectName("modeSegBtn")
        self.btn_annotate_mode.setCheckable(True)
        self.btn_annotate_mode.setChecked(True)
        self.btn_annotate_mode.setFixedWidth(74)
        self.btn_annotate_mode.setFixedHeight(22)
        self.btn_annotate_mode.clicked.connect(lambda: self._toggle_edit_mode())
        mode_layout.addWidget(self.btn_annotate_mode)

        self.btn_paste_mode = QPushButton(tr("贴图"))
        self.btn_paste_mode.setObjectName("modeSegBtn")
        self.btn_paste_mode.setCheckable(True)
        self.btn_paste_mode.setFixedWidth(74)
        self.btn_paste_mode.setFixedHeight(22)
        self.btn_paste_mode.clicked.connect(lambda: self._toggle_edit_mode())
        mode_layout.addWidget(self.btn_paste_mode)

        self.mode_seg_ctrl = AnimatedSegmentedControl(self.mode_seg, self.btn_annotate_mode, self.btn_paste_mode)
        self.mode_seg_ctrl.set_accent(ThemeManager.get_theme()["interaction_active"])
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.mode_seg_ctrl.update_position(animated=False))
        layout.addWidget(self.mode_seg)

        layout.addSpacing(4)

        self.memory_btn = QPushButton(tr("记忆"))
        self.memory_btn.setObjectName("optionsBtn")
        self.memory_btn.setFixedWidth(70)
        self.memory_btn.setFixedHeight(24)
        self.memory_btn.setToolTip(tr("记忆记录"))
        self.memory_btn.clicked.connect(self._show_memory_records)
        layout.addWidget(self.memory_btn)

        layout.addSpacing(4)

        self.options_btn = QPushButton(tr("选项"))
        self.options_btn.setObjectName("optionsBtn")
        self.options_btn.setFixedHeight(24)
        self.options_btn.setFixedWidth(70)
        self.options_btn.setToolTip(tr("选项设置"))
        self.options_menu = HoverKeepMenu(first_action_momentary=False)
        self.options_menu.setObjectName("optionsMenu")
        self.options_menu.setMinimumWidth(200)

        items = [
            (tr("显示BOX"), "toggle_labels", self.show_labels_checkbox),
            (tr("显示Label"), "toggle_label_names", self.show_label_names_checkbox),
            (tr("显示贴图名"), "toggle_paste_names", self.show_paste_names_checkbox),
            (tr("自动保存B"), "auto_save_b", self.auto_save_b_checkbox),
            (tr("自动保存P"), "auto_save_p", self.auto_save_p_checkbox),
            (tr("显示网格线"), "toggle_grid", self.show_grid_checkbox),
        ]

        self._menu_actions = []
        for text, shortcut_action, checkbox in items:
            sc = self._get_shortcut(shortcut_action) if shortcut_action else ''
            label = f"{text}\t{sc}" if sc else text
            action = self.options_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(checkbox.isChecked())
            action.triggered.connect(lambda checked, cb=checkbox: cb.setChecked(checked))
            checkbox.stateChanged.connect(lambda state, a=action: a.setChecked(state == Qt.Checked))
            self._menu_actions.append((action, checkbox, shortcut_action))

        self.options_menu.addSeparator()

        prefix_action = self.options_menu.addAction(tr("添加文件名前缀"))
        prefix_action.setCheckable(True)
        prefix_action.setChecked(self.prefix_checkbox.isChecked())
        prefix_action.triggered.connect(lambda checked, cb=self.prefix_checkbox: cb.setChecked(checked))
        self.prefix_checkbox.stateChanged.connect(lambda state, a=prefix_action: a.setChecked(state == Qt.Checked))
        self._menu_actions.append((prefix_action, self.prefix_checkbox, None))

        self.canvas_copy_action = self.options_menu.addAction(tr("画布图片复制"))
        self.canvas_copy_action.setCheckable(True)
        self.canvas_copy_action.setChecked(getattr(self, '_canvas_image_copy_enabled', False))
        self.canvas_copy_action.triggered.connect(self._on_canvas_copy_menu_changed)
        self._menu_actions.append((self.canvas_copy_action, None, None))

        self.magnifier_action = self.options_menu.addAction(tr("窗口放大器"))
        self.magnifier_action.setCheckable(True)
        self.magnifier_action.setChecked(getattr(self, '_magnifier_enabled', False))
        self.magnifier_action.triggered.connect(self._on_magnifier_menu_changed)
        self._menu_actions.append((self.magnifier_action, None, None))

        self.relative_path_action = self.options_menu.addAction(tr("相对路径显示"))
        self.relative_path_action.setCheckable(True)
        self.relative_path_action.setChecked(getattr(self, '_relative_path_display', False))
        self.relative_path_action.triggered.connect(self._on_relative_path_menu_changed)
        self._menu_actions.append((self.relative_path_action, None, None))

        self.options_btn.setMenu(self.options_menu)
        layout.addWidget(self.options_btn)

        layout.addSpacing(4)

        self.cache_btn = QPushButton(tr("缓存"))
        self.cache_btn.setObjectName("optionsBtn")
        self.cache_btn.setFixedWidth(70)
        self.cache_btn.setFixedHeight(24)
        self.cache_btn.setToolTip(tr("复制缓存管理"))
        self.cache_menu = None
        layout.addWidget(self.cache_btn)
        self._rebuild_label_cache_menu()

        layout.addSpacing(4)

        self.view_stats_btn = QPushButton(tr("统计"))
        self.view_stats_btn.setObjectName("optionsBtn")
        self.view_stats_btn.setFixedHeight(24)
        self.view_stats_btn.setFixedWidth(70)
        self.view_stats_btn.setToolTip(tr("标签统计"))
        self.view_stats_btn.clicked.connect(self._show_label_stats)
        layout.addWidget(self.view_stats_btn)

        layout.addSpacing(4)

        self.lint_btn = QPushButton(tr("质检"))
        self.lint_btn.setObjectName("optionsBtn")
        self.lint_btn.setFixedHeight(24)
        self.lint_btn.setFixedWidth(70)
        self.lint_btn.setToolTip(tr("标注质检"))
        self.lint_btn.clicked.connect(self._open_quality_lint)
        layout.addWidget(self.lint_btn)

        layout.addSpacing(4)

        self.format_btn = QPushButton(tr("格式"))
        self.format_btn.setObjectName("optionsBtn")
        self.format_btn.setFixedWidth(70)
        self.format_btn.setFixedHeight(24)
        self.format_btn.setToolTip(tr("格式转换"))
        self.format_btn.clicked.connect(self._open_dataset_tools)
        layout.addWidget(self.format_btn)

        layout.addSpacing(4)

        self.process_btn = QPushButton(tr("导出"))
        self.process_btn.setObjectName("optionsBtn")
        self.process_btn.setFixedWidth(70)
        self.process_btn.setFixedHeight(24)
        self.process_btn.setToolTip(tr("数据处理"))
        self.process_btn.clicked.connect(self._show_export_menu)
        layout.addWidget(self.process_btn)

    def _rebuild_options_popup(self):
        if getattr(self, 'options_menu', None) is not None:
            self.options_menu.deleteLater()

        popup_flags = Qt.Popup | Qt.FramelessWindowHint
        no_shadow_flag = getattr(Qt, 'NoDropShadowWindowHint', None)
        if no_shadow_flag is not None:
            popup_flags |= no_shadow_flag
        popup = HoverDismissPopup(self, popup_flags)
        popup.setObjectName("optionsPopup")
        popup_layout = QVBoxLayout(popup)
        popup_layout.setContentsMargins(4, 4, 4, 4)
        popup_layout.setSpacing(2)

        self._option_popup_rows = []
        items = [
            (tr("显示BOX"), "toggle_labels", self.show_labels_checkbox, lambda cb=self.show_labels_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.show_labels_checkbox: cb.isChecked()),
            (tr("显示Label"), "toggle_label_names", self.show_label_names_checkbox, lambda cb=self.show_label_names_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.show_label_names_checkbox: cb.isChecked()),
            (tr("显示贴图名"), "toggle_paste_names", self.show_paste_names_checkbox, lambda cb=self.show_paste_names_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.show_paste_names_checkbox: cb.isChecked()),
            (tr("自动保存B"), "auto_save_b", self.auto_save_b_checkbox, lambda cb=self.auto_save_b_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.auto_save_b_checkbox: cb.isChecked()),
            (tr("自动保存P"), "auto_save_p", self.auto_save_p_checkbox, lambda cb=self.auto_save_p_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.auto_save_p_checkbox: cb.isChecked()),
            (tr("显示网格线"), "toggle_grid", self.show_grid_checkbox, lambda cb=self.show_grid_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.show_grid_checkbox: cb.isChecked()),
            (tr("添加文件名前缀"), None, self.prefix_checkbox, lambda cb=self.prefix_checkbox: cb.setChecked(not cb.isChecked()), lambda cb=self.prefix_checkbox: cb.isChecked()),
            (tr("画布图片复制"), None, None, lambda: self._on_canvas_copy_menu_changed(not getattr(self, '_canvas_image_copy_enabled', False)), lambda: getattr(self, '_canvas_image_copy_enabled', False)),
            (tr("窗口放大器"), None, None, lambda: self._on_magnifier_menu_changed(not getattr(self, '_magnifier_enabled', False)), lambda: getattr(self, '_magnifier_enabled', False)),
            (tr("相对路径显示"), None, None, lambda: self._on_relative_path_menu_changed(not getattr(self, '_relative_path_display', False)), lambda: getattr(self, '_relative_path_display', False)),
        ]
        for text, shortcut_action, checkbox, handler, getter in items:
            button = QPushButton()
            button.clicked.connect(handler)
            if checkbox is not None:
                checkbox.stateChanged.connect(lambda state: self._refresh_options_popup_texts())
            popup_layout.addWidget(button)
            self._option_popup_rows.append((button, text, shortcut_action, getter))

        self.options_menu = popup
        self._refresh_options_popup_texts()

    def _refresh_options_popup_texts(self):
        for button, text, shortcut_action, getter in getattr(self, '_option_popup_rows', []):
            sc = self._get_shortcut(shortcut_action) if shortcut_action else ''
            prefix = "√ " if getter() else ""
            button.setText(f"{prefix}{text}    {sc}" if sc else f"{prefix}{text}")

    def _toggle_options_popup(self):
        if getattr(self, 'options_menu', None) is None:
            self._rebuild_options_popup()
        if self.options_menu.isVisible():
            self.options_menu.hide()
            return
        self._refresh_options_popup_texts()
        global_pos = self.options_btn.mapToGlobal(QPoint(0, self.options_btn.height()))
        self.options_menu.move(global_pos)
        self.options_menu.show()

    def _on_canvas_copy_menu_changed(self, checked):
        """切换画布图片复制功能（仅保留在顶部选项菜单中）。"""
        self._canvas_image_copy_enabled = bool(checked)
        if hasattr(self, 'canvas_copy_action'):
            self.canvas_copy_action.setChecked(self._canvas_image_copy_enabled)
        from ...core import config_manager
        config_manager.save_all(canvas_image_copy_enabled=self._canvas_image_copy_enabled)

    def _on_magnifier_menu_changed(self, checked):
        """切换窗口放大器，直接重绘画布即可生效。"""
        self._magnifier_enabled = bool(checked)
        if hasattr(self, 'magnifier_action'):
            self.magnifier_action.setChecked(self._magnifier_enabled)
        from ...core import config_manager
        config_manager.save_all(magnifier_enabled=self._magnifier_enabled)
        if hasattr(self, 'canvas'):
            self.canvas.update()

    def _on_relative_path_menu_changed(self, checked):
        """切换背景列表的相对路径显示，并立即刷新列表文本。"""
        self._relative_path_display = bool(checked)
        if hasattr(self, 'relative_path_action'):
            self.relative_path_action.setChecked(self._relative_path_display)
        from ...core import config_manager
        config_manager.save_all(relative_path_display=self._relative_path_display)
        if hasattr(self, '_refresh_background_path_texts'):
            self._refresh_background_path_texts()

    def _show_memory_records(self):
        """显示记忆记录弹窗"""
        from ..memory_dialog import MemoryRecordsDialog
        MemoryRecordsDialog(self).exec_()

    def _validate_size_range(self):
        """验证尺寸范围，确保最小值不大于最大值"""
        min_size = self.min_size_spin.value()
        max_size = self.max_size_spin.value()
        if min_size > max_size:
            self.status_label.setText("Min cannot exceed Max")
            QTimer.singleShot(2000, lambda: self.status_label.setText(""))
            self.min_size_spin.setValue(max_size)

    def _on_min_size_changed(self, value):
        """最小值变化时更新最大值的范围"""
        self.max_size_spin.setMinimum(max(value, 30))

    def _on_max_size_changed(self, value):
        """最大值变化时更新最小值的范围"""
        self.min_size_spin.setMaximum(min(value, 100))
