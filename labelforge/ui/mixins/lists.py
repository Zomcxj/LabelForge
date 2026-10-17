"""Lists UI building mixin."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QCheckBox, QSpinBox, QFrame
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt, QSize, QTimer
from ...core.config import PASTE_PARAMS
from ..i18n import t as tr
from ..icons import SVG_REFRESH, _load_svg_icon
from ...widgets.drag_out_list import DragOutListWidget


class ListsMixin:
    """Lists widgets and behavior."""

    def _create_background_list_section(self, layout):
        """创建背景图列表区域"""
        group, group_layout, header = self._make_side_collapsible("背景图列表")
        self.bg_list_group = group
        self.bg_list_header = header

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self.prev_img_btn = QPushButton("◀")
        self.prev_img_btn.setObjectName("navBtn")
        self.prev_img_btn.setFixedWidth(35)
        self.prev_img_btn.setFixedHeight(22)
        # 长按连切：按住 400ms 后每 120ms 重发一次 clicked，走既有的
        # switch_background 接线；单击行为不变（第一次点击立即触发）。
        self.prev_img_btn.setAutoRepeat(True)
        self.prev_img_btn.setAutoRepeatDelay(400)
        self.prev_img_btn.setAutoRepeatInterval(120)
        header_layout.addWidget(self.prev_img_btn)

        self.next_img_btn = QPushButton("▶")
        self.next_img_btn.setObjectName("navBtn")
        self.next_img_btn.setFixedWidth(35)
        self.next_img_btn.setFixedHeight(22)
        self.next_img_btn.setAutoRepeat(True)
        self.next_img_btn.setAutoRepeatDelay(400)
        self.next_img_btn.setAutoRepeatInterval(120)
        header_layout.addWidget(self.next_img_btn)

        header_layout.addSpacing(4)

        self.step_label = QLabel(tr("步长："))
        self.step_label.setFixedHeight(22)
        header_layout.addWidget(self.step_label)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 9)
        self.step_spin.setValue(1)
        self.step_spin.setFixedWidth(48)
        self.step_spin.setFixedHeight(22)
        self.step_spin.setAlignment(Qt.AlignCenter)
        self.step_spin.setButtonSymbols(QSpinBox.PlusMinus)
        header_layout.addWidget(self.step_spin)

        header_layout.addSpacing(4)

        self.view_toggle_btn = QPushButton(tr("工作路径"))
        self.view_toggle_btn.setObjectName("warningBtn")
        self.view_toggle_btn.setMinimumWidth(52)
        self.view_toggle_btn.setFixedHeight(22)
        self.view_toggle_btn.clicked.connect(self._toggle_view_path)
        header_layout.addWidget(self.view_toggle_btn, 1)

        self.bg_filter_btn = QPushButton("")
        self.bg_filter_btn.setObjectName("navBtn")
        self.bg_filter_btn.setFixedSize(28, 22)
        self.bg_filter_btn.setToolTip(tr("筛选：全部"))
        self.bg_filter_btn.clicked.connect(self._cycle_bg_annotation_filter)
        header_layout.addWidget(self.bg_filter_btn)
        if hasattr(self, '_refresh_bg_filter_button'):
            self._refresh_bg_filter_button()

        self.bg_refresh_btn = QPushButton("")
        self.bg_refresh_btn.setObjectName("navBtn")
        self.bg_refresh_btn.setFixedSize(28, 22)
        self.bg_refresh_btn.setIcon(QIcon(_load_svg_icon(SVG_REFRESH, 14)))
        self.bg_refresh_btn.setToolTip(tr("refresh_bg_hint"))
        self.bg_refresh_btn.clicked.connect(self._on_refresh_bg_list)
        header_layout.addWidget(self.bg_refresh_btn)

        self.prev_img_btn.clicked.connect(lambda: self.switch_background(-1))
        self.next_img_btn.clicked.connect(lambda: self.switch_background(1))
        self.step_spin.valueChanged.connect(lambda v: setattr(self, '_nav_step', v))

        group_layout.addLayout(header_layout)

        self.background_list = DragOutListWidget()
        self.background_list.setObjectName("bgList")
        self.background_list.setUniformItemSizes(True)
        self.background_list.itemClicked.connect(self.select_background)
        self.background_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.background_list.customContextMenuRequested.connect(self.show_background_context_menu)
        self.background_list.setMinimumHeight(0)
        group_layout.addWidget(self.background_list, 1)

        layout.addWidget(group, 1)

    def _create_label_list_section(self, layout):
        """创建标签列表区域"""
        group, group_layout, header = self._make_side_collapsible("标签管理")
        self.label_group = group
        self.label_group_header = header

        label_layout = QHBoxLayout()

        original_label_layout = QVBoxLayout()
        self.bg_label_header_lbl = QLabel(tr("背景图标签"))
        original_label_header = QHBoxLayout()
        original_label_header.setContentsMargins(0, 0, 0, 0)
        original_label_header.setSpacing(2)
        original_label_header.addWidget(self.bg_label_header_lbl)
        self.bg_label_mode_btn = QPushButton("")
        self.bg_label_mode_btn.setObjectName("navBtn")
        self.bg_label_mode_btn.setFixedSize(22, 20)
        self.bg_label_mode_btn.setToolTip(tr("切换到每框一行"))
        self.bg_label_mode_btn.clicked.connect(self._toggle_bg_label_list_mode)
        original_label_header.addWidget(self.bg_label_mode_btn)
        self.task_filter_btn = QPushButton("")
        self.task_filter_btn.setObjectName("navBtn")
        self.task_filter_btn.setFixedSize(22, 20)
        self.task_filter_btn.setToolTip(tr("按任务筛选"))
        original_label_header.addWidget(self.task_filter_btn)
        self.group_filter_btn = QPushButton("")
        self.group_filter_btn.setObjectName("navBtn")
        self.group_filter_btn.setFixedSize(22, 20)
        self.group_filter_btn.setToolTip(tr("按分组筛选"))
        original_label_header.addWidget(self.group_filter_btn)
        self._setup_filter_menus()
        original_label_header.addStretch()
        original_label_layout.addLayout(original_label_header)
        if hasattr(self, '_refresh_bg_label_mode_button'):
            self._refresh_bg_label_mode_button()
        if hasattr(self, '_refresh_filter_buttons'):
            self._refresh_filter_buttons()

        self.label_list = QListWidget()
        self.label_list.setObjectName("labelList")
        self.label_list.setMinimumHeight(0)
        self.label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.label_list.customContextMenuRequested.connect(self.label_manager.show_label_context_menu)
        self.label_list.itemPressed.connect(self.label_list_item_pressed)
        self.label_list.itemClicked.connect(self.label_list_item_clicked)
        self._attach_task_badge_delegate(self.label_list)
        self.pressed_label = None
        self.pressed_box_index = None
        original_label_layout.addWidget(self.label_list)

        self.paste_label_column = QWidget()
        paste_label_layout = QVBoxLayout(self.paste_label_column)
        paste_label_layout.setContentsMargins(0, 0, 0, 0)
        self.paste_label_header_lbl = QLabel(tr("贴图标签_list"))
        paste_label_header = QHBoxLayout()
        paste_label_header.addWidget(self.paste_label_header_lbl)
        paste_label_header.addStretch()
        paste_label_layout.addLayout(paste_label_header)

        self.paste_label_list = QListWidget()
        self.paste_label_list.setObjectName("pasteLabelList")
        self.paste_label_list.setMinimumHeight(0)
        self.paste_label_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.paste_label_list.customContextMenuRequested.connect(
            self.label_manager.show_paste_label_context_menu
        )
        default_item = QListWidgetItem("paste")
        self.paste_label_list.addItem(default_item)
        self.paste_label_list.itemPressed.connect(self.label_list_item_pressed)
        self.paste_label_list.itemClicked.connect(self.label_list_item_clicked)
        paste_label_layout.addWidget(self.paste_label_list)

        label_layout.addLayout(original_label_layout, 1)
        label_layout.addWidget(self.paste_label_column, 1)
        group_layout.addLayout(label_layout)

        layout.addWidget(group, 1)

    def _attach_task_badge_delegate(self, list_widget):
        """给标签列表挂上任务徽标绘制器（Qt 缺失时静默跳过）。"""
        try:
            from ...widgets.task_badge import TaskBadgeDelegate
            list_widget.setItemDelegate(TaskBadgeDelegate(list_widget))
        except Exception:
            pass

    def _create_small_list_section(self, layout):
        """创建贴图列表区域"""
        group, group_layout, header = self._make_side_collapsible("贴图列表")
        self.paste_group = group
        self.paste_group_header = header

        small_list_layout = QHBoxLayout()

        self.random_paste_btn = QPushButton(tr("随机贴图"))
        self.random_paste_btn.setObjectName("accentBtn")
        self.random_paste_btn.setFixedHeight(22)
        self.random_paste_btn.clicked.connect(self.random_paste_images)
        self.random_paste_btn.setToolTip(tr("随机贴图"))
        small_list_layout.addWidget(self.random_paste_btn, 1)

        self.batch_paste_btn = QPushButton(tr("一键贴图"))
        self.batch_paste_btn.setObjectName("accentBtn")
        self.batch_paste_btn.setFixedHeight(22)
        self.batch_paste_btn.clicked.connect(self.batch_paste_images)
        self.batch_paste_btn.setToolTip(tr("一键贴图"))
        small_list_layout.addWidget(self.batch_paste_btn, 1)

        self.toggle_view_btn = QPushButton(tr("列表视图"))
        self.toggle_view_btn.setObjectName("warningBtn")
        self.toggle_view_btn.setFixedHeight(22)
        self.toggle_view_btn.clicked.connect(self.toggle_view_mode)
        small_list_layout.addWidget(self.toggle_view_btn, 1)

        small_list_layout.addStretch()
        group_layout.addLayout(small_list_layout)

        self._create_paste_params(group_layout)

        self.small_list = QListWidget()
        self.small_list.setObjectName("smallList")
        self.small_list.setMinimumHeight(0)
        self.small_list.itemClicked.connect(self.add_small_to_canvas)
        self._configure_small_list()
        group_layout.addWidget(self.small_list, 1)

        self._create_bottom_buttons(group_layout)

        layout.addWidget(group, 1)

    def _create_paste_params(self, layout):
        """创建贴图参数设置"""
        paste_params_layout = QHBoxLayout()
        paste_params_layout.setContentsMargins(0, 5, 0, 5)

        self.paste_count_lbl = QLabel(tr("贴图个数:"))
        paste_params_layout.addWidget(self.paste_count_lbl)
        self.paste_count_spin = QSpinBox()
        self.paste_count_spin.setObjectName("paramSpin")
        self.paste_count_spin.setMinimum(PASTE_PARAMS['min_count'])
        self.paste_count_spin.setMaximum(PASTE_PARAMS['max_count'])
        self.paste_count_spin.setValue(PASTE_PARAMS['default_count'])
        self.paste_count_spin.setMinimumWidth(50)
        paste_params_layout.addWidget(self.paste_count_spin)
        paste_params_layout.addSpacing(10)

        self.size_lbl = QLabel(tr("短边尺寸:"))
        paste_params_layout.addWidget(self.size_lbl)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setObjectName("paramSpin")
        self.min_size_spin.setMinimum(15)
        self.min_size_spin.setMaximum(100)
        self.min_size_spin.setValue(30)
        self.min_size_spin.setMinimumWidth(55)
        paste_params_layout.addWidget(self.min_size_spin)

        paste_params_layout.addWidget(QLabel("-"))

        self.max_size_spin = QSpinBox()
        self.max_size_spin.setObjectName("paramSpin")
        self.max_size_spin.setMinimum(30)
        self.max_size_spin.setMaximum(200)
        self.max_size_spin.setValue(60)
        self.max_size_spin.setMinimumWidth(55)
        paste_params_layout.addWidget(self.max_size_spin)

        self.min_size_spin.valueChanged.connect(self._on_min_size_changed)
        self.max_size_spin.valueChanged.connect(self._on_max_size_changed)
        self._on_min_size_changed(self.min_size_spin.value())
        self._on_max_size_changed(self.max_size_spin.value())

        paste_params_layout.addStretch()
        layout.addLayout(paste_params_layout)

    def _configure_small_list(self):
        """配置贴图列表"""
        if self.is_thumbnail_mode:
            self.small_list.setViewMode(QListWidget.IconMode)
            self.small_list.setIconSize(QSize(
                self.thumbnail_grid_width, self.thumbnail_grid_height
            ))
            self.small_list.setGridSize(QSize(
                self.thumbnail_grid_width, self.thumbnail_grid_height + 20
            ))
            self.small_list.setSpacing(self.thumbnail_spacing)
            self.small_list.setWrapping(True)
            self.small_list.setFlow(QListWidget.LeftToRight)
            self.small_list.setResizeMode(QListWidget.Adjust)
            self.small_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
            self.small_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)

    def _show_loading_spinner(self, text=""):
        self._loading_spinner.setText(text or tr("识别中..."))
        self._loading_spinner.adjustSize()
        parent = self._loading_spinner.parentWidget()
        if parent:
            pw, ph = parent.width(), parent.height()
            sw, sh = self._loading_spinner.width(), self._loading_spinner.height()
            self._loading_spinner.move((pw - sw) // 2, (ph - sh) // 2)
        self._loading_spinner.show()
        self._loading_spinner.raise_()
        self._loading_spinner.start()

    def _hide_loading_spinner(self):
        self._loading_spinner.stop()
        self._loading_spinner.hide()

    def _create_bottom_buttons(self, layout):
        """创建底部按钮"""
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(0, 4, 0, 0)

        self.clear_btn = QPushButton(tr("清空画布"))
        self.clear_btn.setObjectName("dangerBtn")
        self.clear_btn.setFixedHeight(22)
        self.clear_btn.clicked.connect(self.clear_canvas)
        self.clear_btn.setToolTip(tr("清空画布"))

        self.save_btn = QPushButton(tr("保存图片"))
        self.save_btn.setObjectName("successBtn")
        self.save_btn.setFixedHeight(22)
        self.save_btn.clicked.connect(self.save_canvas)
        self.save_btn.setToolTip(tr("保存图片"))

        self.save_all_btn = QPushButton(tr("全部保存"))
        self.save_all_btn.setObjectName("successBtn")
        self.save_all_btn.setFixedHeight(22)
        self.save_all_btn.clicked.connect(self.save_all_canvas)
        self.save_all_btn.setToolTip(tr("全部保存"))

        button_layout.addWidget(self.clear_btn, 1)
        button_layout.addWidget(self.save_btn, 1)
        button_layout.addWidget(self.save_all_btn, 1)
        layout.addLayout(button_layout)

    def toggle_view_mode(self):
        """切换视图模式"""
        self.is_thumbnail_mode = not self.is_thumbnail_mode

        if self.is_thumbnail_mode:
            self.toggle_view_btn.setText(tr("列表视图"))
        else:
            self.toggle_view_btn.setText(tr("缩略视图"))

        self.small_list.clear()
        self._configure_small_list() if self.is_thumbnail_mode else self._set_list_mode()
        self.refresh_list_items()
        self.small_list.scrollToTop()
        self.small_list.updateGeometry()
        self.small_list.repaint()

        mode_text = "Thumbnail" if self.is_thumbnail_mode else "List"
        self.status_label.setText(f"View: {mode_text}")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _set_list_mode(self):
        """设置列表模式"""
        self.small_list.setViewMode(QListWidget.ListMode)
        self.small_list.setIconSize(QSize())
        self.small_list.setGridSize(QSize())
        self.small_list.setSpacing(0)
        self.small_list.setWrapping(False)
        self.small_list.setFlow(QListWidget.TopToBottom)
        self.small_list.setVerticalScrollMode(QListWidget.ScrollPerItem)
