"""ProcessingPanel 的区块构建方法。"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget,
)

from ..i18n import t as tr
from ...widgets.processing import _ScanSpinnerWidget

SPLIT_KEYS = ["train", "val", "test"]

#: 类别勾选区的高度上限；超出后滚动，不再压缩每一行
_LABEL_GRID_MAX_HEIGHT = 132

#: 类别勾选框之间的间距，算滚动区高度时要用同一个值
_LABEL_GRID_SPACING = 8


def _style_pipe_step(wrap, checkbox):
    """流水线步骤块的选中样式：勾上就点亮，没勾就压暗。"""
    on = bool(checkbox.isChecked())
    bg = "#1E88E5" if on else "#3A3A3A"
    fg = "#FFFFFF" if on else "#9E9E9E"
    wrap.setStyleSheet(
        f"QWidget {{ background-color: {bg}; border-radius: 4px; }}"
        f" QLabel {{ color: {fg}; background: transparent; }}"
        f" QCheckBox {{ color: {fg}; background: transparent; }}")


class ProcessingPanelBuilderMixin:
    """构建数据处理面板的四个操作区块。"""

    def _build_augment_section(self, parent_layout):
        """数据增强：只选预设，不在这里配参数。

        几何/色彩/噪声那三组算子都在图像编辑面板里，参数也在那里调；
        这里只负责挑一个存好的预设去批量跑。
        """
        section = self._collapsible_section("", parent_layout, "#F44336")
        layout = section.content_layout()

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 2, 0, 2)
        preset_row.setSpacing(6)
        self._aug_preset_lbl = QLabel()
        preset_row.addWidget(self._aug_preset_lbl)
        self._aug_preset_combo = QComboBox()
        self._aug_preset_combo.setMinimumWidth(240)
        self._aug_preset_combo.currentIndexChanged.connect(
            lambda _index: self._update_augment_preset_summary())
        preset_row.addWidget(self._aug_preset_combo)
        self._aug_edit_btn = QPushButton()
        self._aug_edit_btn.setObjectName("accentBtn")
        self._aug_edit_btn.setToolTip(tr("打开图像编辑面板调整参数并存成预设"))
        self._aug_edit_btn.clicked.connect(self._open_image_editor_for_preset)
        preset_row.addWidget(self._aug_edit_btn)
        preset_row.addStretch()
        layout.addLayout(preset_row)

        self._aug_preset_summary = QLabel()
        self._aug_preset_summary.setWordWrap(True)
        self._aug_preset_summary.setStyleSheet(
            "color: #9E9E9E; font-size: 11px; background: transparent;")
        layout.addWidget(self._aug_preset_summary)

        crop_row = QHBoxLayout()
        crop_row.setContentsMargins(0, 2, 0, 2)
        crop_row.setSpacing(2)
        self._crop_check = QCheckBox(tr("滑窗裁剪"))
        self._crop_check.toggled.connect(lambda _checked: self._update_crop_summary())
        crop_row.addWidget(self._crop_check)
        crop_row.addSpacing(6)
        self._crop_config_btn = QPushButton(tr("配置…"))
        # 只保底不封顶：英文 "Configure…" 比中文宽，固定 56px 会把文字截断
        self._crop_config_btn.setMinimumWidth(56)
        self._crop_config_btn.clicked.connect(self._open_crop_config)
        crop_row.addWidget(self._crop_config_btn)
        self._crop_summary_lbl = QLabel()
        crop_row.addWidget(self._crop_summary_lbl, 1)
        layout.addLayout(crop_row)

        rl = QHBoxLayout()
        self._ratio_lbl = QLabel()
        rl.addWidget(self._ratio_lbl)
        self._aug_ratio = QDoubleSpinBox()
        self._aug_ratio.setRange(0.05, 1.0)
        self._aug_ratio.setSingleStep(0.05)
        self._aug_ratio.setDecimals(2)
        self._aug_ratio.setValue(1.0)
        self._aug_ratio.setFixedWidth(70)
        rl.addWidget(self._aug_ratio)
        rl.addSpacing(10)
        self._mode_lbl = QLabel()
        rl.addWidget(self._mode_lbl)
        self._aug_mode = QComboBox()
        self._aug_mode.addItems(["", ""])
        self._aug_mode.setFixedWidth(100)
        rl.addWidget(self._aug_mode)
        rl.addSpacing(10)
        self._aug_inc_orig = QCheckBox()
        self._aug_inc_orig.setChecked(True)
        rl.addWidget(self._aug_inc_orig)
        rl.addSpacing(10)
        self._aug_skip_empty = QCheckBox()
        self._aug_skip_empty.setChecked(True)
        rl.addWidget(self._aug_skip_empty)
        rl.addStretch()
        layout.addLayout(rl)

        btn_layout = QHBoxLayout()
        self._aug_btn = QPushButton()
        self._aug_btn.setObjectName("successBtn")
        self._aug_btn.clicked.connect(lambda: self._run_augment())
        btn_layout.addWidget(self._aug_btn, 5)
        btn_layout.addWidget(self._make_btn_with_stop(self._aug_btn, lambda: self._do_interrupt()), 1)
        self._aug_clear_btn = QPushButton()
        self._aug_clear_btn.setObjectName("accentBtn")
        self._aug_clear_btn.clicked.connect(lambda: self._clear_aug())
        btn_layout.addWidget(self._aug_clear_btn, 1)
        layout.addLayout(btn_layout)
        self._aug_section = section

    def _build_export_section(self, parent_layout):
        section = self._collapsible_section("", parent_layout, "#FFC107")
        layout = section.content_layout()
        # 本面板是「YOLO 训练集生产流水线」：导出 → 划分 → 训练。
        # 导出格式固定为 YOLO——划分模块读的是 labels/*.txt，导出成
        # VOC/COCO 后下游没人能接，所以不提供格式选择。
        # 需要 COCO/VOC 时用工具栏的「格式」按钮，那里支持四格式互转。
        self._exp_label_title = QLabel()
        layout.addWidget(self._exp_label_title)
        # 类别多时按行自然高度铺开，超过上限就滚动。不能给容器设最大高度硬
        # 压：行被压到 12px 而指示块是 18px，圆角被裁掉、看着成了方角，行与
        # 行之间还会叠在一起。
        self._exp_label_grid = QGridLayout()
        self._exp_label_grid.setSpacing(_LABEL_GRID_SPACING)
        self._exp_label_grid.setContentsMargins(0, 2, 0, 2)
        self._exp_label_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._exp_label_checkboxes = {}
        self._exp_label_scroll = QScrollArea()
        self._exp_label_scroll.setWidgetResizable(True)
        self._exp_label_scroll.setFrameShape(QScrollArea.NoFrame)
        self._exp_label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._exp_label_scroll.setMaximumHeight(_LABEL_GRID_MAX_HEIGHT)
        self._exp_label_grid_container = QWidget()
        self._exp_label_grid_container.setLayout(self._exp_label_grid)
        self._exp_label_scroll.setWidget(self._exp_label_grid_container)
        layout.addWidget(self._exp_label_scroll, 0)
        self._exp_scan_label = _ScanSpinnerWidget()
        self._exp_scan_label.setVisible(False)
        layout.addWidget(self._exp_scan_label)
        opts_layout = QHBoxLayout()
        opts_layout.setContentsMargins(0, 0, 0, 0)
        self._exp_skip_empty = QCheckBox()
        self._exp_skip_empty.setChecked(True)
        opts_layout.addWidget(self._exp_skip_empty)
        opts_layout.addStretch()
        layout.addLayout(opts_layout)
        btn_layout = QHBoxLayout()
        self._exp_btn = QPushButton()
        self._exp_btn.setObjectName("successBtn")
        self._exp_btn.clicked.connect(self._run_export)
        btn_layout.addWidget(self._exp_btn, 5)
        btn_layout.addWidget(self._make_btn_with_stop(self._exp_btn, lambda: self._do_interrupt()), 1)
        self._exp_clear_btn = QPushButton()
        self._exp_clear_btn.setObjectName("accentBtn")
        self._exp_clear_btn.clicked.connect(lambda: self._clear_exp())
        btn_layout.addWidget(self._exp_clear_btn, 1)
        layout.addLayout(btn_layout)
        self._exp_section = section

    def _build_split_section(self, parent_layout):
        section = self._collapsible_section("", parent_layout, "#4CAF50")
        layout = section.content_layout()
        sl = QHBoxLayout()
        for key in SPLIT_KEYS:
            self._split_lbls.append(QLabel())
            sl.addWidget(self._split_lbls[-1])
            s = QDoubleSpinBox()
            s.setRange(0.0, 1.0)
            s.setSingleStep(0.05)
            s.setValue({"train": 0.8, "val": 0.1, "test": 0.1}[key])
            s.setFixedWidth(60)
            s.valueChanged.connect(lambda val, k=key: self._on_split_changed(k))
            sl.addWidget(s)
            setattr(self, f"_split_{key}", s)
        # 三个比例加起来必须是 1，否则划分结果会漏掉或多出样本。
        # 实时显示合计并在不合法时变红，比等到点「划分」再报错更好。
        self._split_total_lbl = QLabel()
        self._split_total_lbl.setStyleSheet("background: transparent;")
        sl.addWidget(self._split_total_lbl)
        sl.addStretch()
        layout.addLayout(sl)
        btn_layout = QHBoxLayout()
        self._split_btn = QPushButton()
        self._split_btn.setObjectName("successBtn")
        self._split_btn.clicked.connect(lambda: self._run_split())
        btn_layout.addWidget(self._split_btn, 5)
        btn_layout.addWidget(self._make_btn_with_stop(self._split_btn, lambda: self._do_interrupt()), 1)
        self._split_clear_btn = QPushButton()
        self._split_clear_btn.setObjectName("accentBtn")
        self._split_clear_btn.clicked.connect(lambda: self._clear_split())
        btn_layout.addWidget(self._split_clear_btn, 1)
        layout.addLayout(btn_layout)
        self._split_section = section

    def _build_pipeline_section(self, parent_layout):
        section = self._collapsible_section("", parent_layout, "#FFFFFF")
        layout = section.content_layout()
        hl = QHBoxLayout()
        self._pipe_aug = QCheckBox()
        self._pipe_aug.setChecked(True)
        self._pipe_exp = QCheckBox()
        self._pipe_exp.setChecked(True)
        self._pipe_split = QCheckBox()
        self._pipe_split.setChecked(True)
        # 三个步骤做成有色块的步骤条：勾选时块变强调色，未勾选时淡灰。
        # 只用三个复选框 + 两个 → 的话，勾选状态不够醒目，观感也不完整。
        # 文字由 QCheckBox 自己带（_refresh_texts 里 setText），这里不再
        # 另加 QLabel —— 两处都给文字，界面上会显示成「增强 增强」。
        _steps = (self._pipe_aug, self._pipe_exp, self._pipe_split)
        for _i, _cb in enumerate(_steps):
            if _i:
                _arrow = QLabel("→")
                _arrow.setStyleSheet(
                    "color: #9E9E9E; background: transparent; font-size: 13px;")
                hl.addWidget(_arrow)
            _wrap = QWidget()
            _wl = QHBoxLayout(_wrap)
            _wl.setContentsMargins(6, 2, 8, 2)
            _wl.setSpacing(4)
            _wl.addWidget(_cb)
            hl.addWidget(_wrap)
            _cb.toggled.connect(
                lambda on, w=_wrap, c=_cb: _style_pipe_step(w, c))
            _style_pipe_step(_wrap, _cb)
        hl.addStretch()
        layout.addLayout(hl)
        btn_layout = QHBoxLayout()
        self._pipe_btn = QPushButton()
        self._pipe_btn.setObjectName("accentBtn")
        self._pipe_btn.clicked.connect(self._run_pipeline)
        btn_layout.addWidget(self._pipe_btn)
        btn_layout.addWidget(self._make_btn_with_stop(self._pipe_btn, lambda: self._do_interrupt()))
        layout.addLayout(btn_layout)
        self._pipe_section = section
