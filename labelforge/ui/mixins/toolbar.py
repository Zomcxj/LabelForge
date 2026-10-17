"""Toolbar UI building mixin."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QFrame
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt
from ...core.config import WINDOW_CONFIG, PASTE_PARAMS, THUMBNAIL_CONFIG, DEFAULT_PREFIX
from ..icons import (
    _load_svg_icon, SVG_FILE, SVG_FOLDER, SVG_IMPORT, SVG_PASTE_LABEL, SUN_SVG,
    SVG_MODEL_SLIDERS, SVG_MODEL_SINGLE, SVG_MODEL_ALL, SVG_VIDEO,
)
from ..i18n import t as tr
from ..segmented_control import AnimatedSegmentedControl


class ToolbarMixin:
    """Toolbar widgets and behavior."""

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self._create_toolbar(main_layout)

        self.shortcut_status_label = QLabel("")
        self.shortcut_status_label.setObjectName("shortcutStatusLabel")
        self.shortcut_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.shortcut_status_label.setMinimumHeight(24)
        main_layout.addWidget(self.shortcut_status_label)

        splitter = self._create_splitter()
        main_layout.addWidget(splitter)

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 1)

        self.setCentralWidget(central_widget)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().show()

    def _create_toolbar(self, layout):
        """创建工具栏"""
        toolbar_widget = QWidget()
        toolbar_widget.setObjectName("toolbar")
        upload_layout = QHBoxLayout(toolbar_widget)
        upload_layout.setSpacing(4)
        upload_layout.setContentsMargins(4, 4, 4, 4)

        bg_color = "#2196F3"
        paste_color = "#4CAF50"
        label_color = "#FF9800"

        self.bg_lbl = QLabel(tr("背景图:"))
        upload_layout.addWidget(self.bg_lbl)
        self.load_folder_btn = self._create_svg_button(
            SVG_FOLDER, self.load_folder_images, tr("加载文件夹图片"), bg_color, "bgBtn"
        )
        upload_layout.addWidget(self.load_folder_btn)

        self.upload_a_btn = self._create_svg_button(
            SVG_FILE, self.upload_background, tr("选择背景图片"), bg_color, "bgBtn"
        )
        upload_layout.addWidget(self.upload_a_btn)

        upload_layout.addSpacing(2)
        self.paste_lbl = QLabel(tr("贴图:"))
        upload_layout.addWidget(self.paste_lbl)
        self.load_small_folder_btn = self._create_svg_button(
            SVG_FOLDER, self.load_small_folder_images, tr("加载贴图文件夹"), paste_color, "pasteBtn"
        )
        upload_layout.addWidget(self.load_small_folder_btn)

        self.upload_b_btn = self._create_svg_button(
            SVG_FILE, self.upload_small_images, tr("选择贴图"), paste_color, "pasteBtn"
        )
        upload_layout.addWidget(self.upload_b_btn)

        upload_layout.addSpacing(2)
        self.label_lbl = QLabel(tr("标签:"))
        upload_layout.addWidget(self.label_lbl)
        # 背景图标签导入：读 txt 后并入背景标签集合（与扫描结果取并集）
        self.upload_bg_label_btn = self._create_svg_button(
            SVG_IMPORT, self.upload_background_labels,
            tr("导入背景标签文件"), label_color, "labelBtn"
        )
        upload_layout.addWidget(self.upload_bg_label_btn)
        self.upload_paste_label_btn = self._create_svg_button(
            SVG_PASTE_LABEL, self.upload_paste_labels, tr("导入贴图标签文件"), label_color, "labelBtn"
        )
        upload_layout.addWidget(self.upload_paste_label_btn)

        upload_layout.addSpacing(2)
        # 视频抽帧：在「标签:」两个导入图标与「模型推理:」之间；点击打开配置弹窗
        self.video_lbl = QLabel(tr("视频:"))
        upload_layout.addWidget(self.video_lbl)
        self.video_load_btn = self._create_svg_button(
            SVG_VIDEO, self._open_video_extract,
            tr("视频抽帧:"), label_color, "labelBtn"
        )
        # cv2 缺失时置灰并说明原因，不让用户点了才看到报错
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.video_load_btn.setEnabled(False)
            self.video_load_btn.setToolTip(tr("未安装 opencv（cv2），视频功能不可用"))
        upload_layout.addWidget(self.video_load_btn)

        upload_layout.addSpacing(2)
        self.model_lbl = QLabel(tr("模型推理:"))
        upload_layout.addWidget(self.model_lbl)
        # 第一个图标只负责显示/隐藏参数行，加载模型在参数行里做
        self.model_row_btn = self._create_svg_button(
            SVG_MODEL_SLIDERS, self._toggle_model_row,
            tr("显示模型参数行"), label_color, "labelBtn"
        )
        upload_layout.addWidget(self.model_row_btn)
        self.model_single_btn = self._create_svg_button(
            SVG_MODEL_SINGLE, self._run_inference_current,
            tr("推理当前图片"), label_color, "labelBtn"
        )
        # 未加载模型或参数行关闭时不能推理，由 _update_inference_buttons 打开
        self.model_single_btn.setEnabled(False)
        upload_layout.addWidget(self.model_single_btn)
        self.model_all_btn = self._create_svg_button(
            SVG_MODEL_ALL, self._run_inference_from_here,
            tr("从当前图片推理到末尾"), label_color, "labelBtn"
        )
        self.model_all_btn.setEnabled(False)
        upload_layout.addWidget(self.model_all_btn)

        upload_layout.addSpacing(8)
        self._add_separator(upload_layout)
        upload_layout.addSpacing(8)

        self._init_checkboxes()
        self._init_prefix_checkbox()
        self._create_options_menu(upload_layout)

        upload_layout.addStretch()

        self.lang_btn = QPushButton("中/EN")
        self.lang_btn.setObjectName("langBtn")
        self.lang_btn.setFixedSize(42, 28)
        self.lang_btn.setToolTip(tr("切换中英文"))
        self.lang_btn.clicked.connect(self.toggle_language)
        upload_layout.addWidget(self.lang_btn)

        self.theme_btn = QPushButton("")
        self.theme_btn.setObjectName("themeBtn")
        self.theme_btn.setFixedSize(28, 28)
        self.theme_btn.setToolTip(tr("切换深色/浅色主题"))
        self.theme_btn.setIcon(QIcon(_load_svg_icon(SUN_SVG, 16, "#D4AF37")))
        self.theme_btn.clicked.connect(self.toggle_theme)
        upload_layout.addWidget(self.theme_btn)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setObjectName("settingsBtn")
        self.settings_btn.setFixedSize(28, 28)
        self.settings_btn.setToolTip("设置快捷键")
        self.settings_btn.clicked.connect(self.open_settings)
        upload_layout.addWidget(self.settings_btn)

        layout.addWidget(toolbar_widget)

    def _create_svg_button(self, svg_data, slot, tooltip, color, obj_name=None):
        """创建 SVG 图标按钮"""
        btn = QPushButton("")
        btn.setObjectName(obj_name or "iconBtn")
        btn.setIcon(QIcon(_load_svg_icon(svg_data, 14, color)))
        btn.clicked.connect(slot)
        btn.setFixedSize(24, 24)
        btn.setToolTip(tooltip)
        return btn

    def _add_separator(self, layout):
        """添加垂直分隔线"""
        sep = QFrame()
        sep.setObjectName("toolbarSep")
        sep.setFrameShape(QFrame.VLine)
        sep.setFrameShadow(QFrame.Sunken)
        sep.setFixedHeight(20)
        layout.addWidget(sep)

    def _on_grid_changed(self):
        """网格复选框状态变化"""
        if hasattr(self, 'canvas'):
            self.canvas.update()

    def _init_checkboxes(self):
        """初始化隐藏的复选框"""
        self.show_labels_checkbox = QCheckBox(tr("显示BOX"))
        self.show_labels_checkbox.setObjectName("showBoxCb")
        self.show_labels_checkbox.setChecked(True)
        self.show_labels_checkbox.stateChanged.connect(self.on_labels_checkbox_changed)

        self.show_label_names_checkbox = QCheckBox(tr("显示Label"))
        self.show_label_names_checkbox.setObjectName("showLabelCb")
        self.show_label_names_checkbox.setChecked(True)
        self.show_label_names_checkbox.stateChanged.connect(self.on_labels_checkbox_changed)

        self.show_scores_checkbox = QCheckBox(tr("显示置信度"))
        self.show_scores_checkbox.setObjectName("showScoreCb")
        self.show_scores_checkbox.setChecked(True)
        self.show_scores_checkbox.setToolTip(tr("在类别名后显示推理置信度"))
        self.show_scores_checkbox.stateChanged.connect(self.on_labels_checkbox_changed)

        self.auto_label_checkbox = QCheckBox(tr("贴图标签"))
        self.auto_label_checkbox.setObjectName("autoLabelCb")
        self.auto_label_checkbox.setChecked(True)

        self.auto_save_b_checkbox = QCheckBox(tr("自动保存B"))
        self.auto_save_b_checkbox.setObjectName("autoSaveBCb")
        self.auto_save_b_checkbox.setChecked(False)
        self.auto_save_p_checkbox = QCheckBox(tr("自动保存P"))
        self.auto_save_p_checkbox.setObjectName("autoSavePCb")
        self.auto_save_p_checkbox.setChecked(False)

        self.show_paste_names_checkbox = QCheckBox(tr("显示贴图名"))
        self.show_paste_names_checkbox.setObjectName("pasteNameCb")
        self.show_paste_names_checkbox.setChecked(True)
        self.show_paste_names_checkbox.stateChanged.connect(self._on_grid_changed)

        self.show_grid_checkbox = QCheckBox(tr("显示网格线"))
        self.show_grid_checkbox.setObjectName("gridCb")
        self.show_grid_checkbox.setChecked(False)
        self.show_grid_checkbox.stateChanged.connect(self._on_grid_changed)

    def _init_prefix_checkbox(self):
        """初始化前缀复选框"""
        self.prefix_checkbox = QCheckBox(tr("添加文件名前缀"))
        self.prefix_checkbox.setObjectName("prefixCb")
        self.prefix_checkbox.setChecked(True)
