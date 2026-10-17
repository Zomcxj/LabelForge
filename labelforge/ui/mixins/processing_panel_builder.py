"""Section construction methods for ProcessingPanel."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QWidget

from ..i18n import t as tr
from ...engine.augmenter import get_all_transforms
from ...widgets.processing import _ScanSpinnerWidget

SPLIT_KEYS = ["train", "val", "test"]

TRANSFORM_META = {
    "fliph":       ("水平翻转", []),
    "flipv":       ("垂直翻转", []),
    "bright":      ("亮度", [("delta", QSpinBox, -100, 100, -30, 30)]),
    "contrast":    ("对比度", [("factor", QDoubleSpinBox, 0.5, 2.0, 1.2, 1.8)]),
    "hue":         ("色相(H)", [("delta", QSpinBox, -180, 180, -30, 30)]),
    "saturation":  ("饱和度(S)", [("factor", QDoubleSpinBox, 0.0, 2.0, 1.2, 1.8)]),
    "value":       ("明度(V)", [("delta", QSpinBox, -100, 100, -30, 30)]),
    "gauss":       ("高斯噪声", [("sigma", QSpinBox, 1, 100, 10, 40)]),
    "saltpepper":  ("椒盐噪声", [("prob", QDoubleSpinBox, 0.0, 0.5, 0.02, 0.08)]),
    "trans":       ("随机平移", [("offset", QSpinBox, 0, 100, 10, 30)]),
    "rotate":      ("随机旋转", [("angle", QSpinBox, 0, 45, 5, 25)]),
    "scale":       ("随机缩放", [("scale", QDoubleSpinBox, 0.5, 1.5, 0.8, 1.2)]),
    "crop":        ("滑窗裁剪", []),
}

_TRANSFORM_ORDER = ["fliph", "flipv", "bright", "trans", "rotate", "scale",
                    "contrast", "gauss", "saltpepper", "hue", "saturation", "value"]


class ProcessingPanelBuilderMixin:
    """Build the processing panel's four operation sections."""

    def _build_augment_section(self, parent_layout):
        section = self._collapsible_section("", parent_layout, "#F44336")
        layout = section.content_layout()
        transforms = get_all_transforms()
        order = [n for n in _TRANSFORM_ORDER if n in transforms]
        self._aug_widgets = {}
        grid = QGridLayout()
        grid.setSpacing(6)
        self._aug_inc_orig = QCheckBox()
        self._aug_inc_orig.setChecked(True)
        self._aug_widgets["__orig__"] = (self._aug_inc_orig, {}, "原图")
        orig_cell = QWidget()
        orig_cell.setMinimumWidth(140)
        orig_cl = QHBoxLayout(orig_cell)
        orig_cl.setContentsMargins(2, 1, 2, 1)
        orig_cl.setSpacing(2)
        orig_cl.addWidget(self._aug_inc_orig)
        orig_cl.addStretch()
        grid.addWidget(orig_cell, 0, 0)
        for idx, name in enumerate(order):
            row, col = divmod(idx + 1, 3)
            meta = TRANSFORM_META.get(name, (name, []))
            display_name, param_defs = meta
            cell = QWidget()
            cell.setMinimumWidth(140)
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(2, 1, 2, 1)
            cl.setSpacing(2)
            cb = QCheckBox(display_name)
            cb.setChecked(name in ("fliph", "flipv", "bright", "rotate", "trans"))
            cl.addWidget(cb)
            cl.addStretch()
            param_spins = {}
            for pname, ptype, pmin, pmax, dmin, dmax in param_defs:
                lbl = QLabel("~")
                lbl.setFixedWidth(10)
                spin_min = self._make_spin(ptype, pmin, pmax, dmin)
                spin_max = self._make_spin(ptype, pmin, pmax, dmax)
                cl.addWidget(spin_min)
                cl.addWidget(lbl)
                cl.addWidget(spin_max)
                param_spins[pname] = (spin_min, spin_max)
            grid.addWidget(cell, row, col)
            self._aug_widgets[name] = (cb, param_spins, display_name)
        crop_cell = QWidget()
        crop_cell.setMinimumWidth(140)
        crop_cl = QHBoxLayout(crop_cell)
        crop_cl.setContentsMargins(2, 1, 2, 1)
        crop_cl.setSpacing(2)
        self._crop_check = QCheckBox(tr("滑窗裁剪"))
        self._crop_check.toggled.connect(lambda _checked: self._update_crop_summary())
        crop_cl.addWidget(self._crop_check)
        crop_cl.addSpacing(6)
        self._crop_config_btn = QPushButton(tr("配置…"))
        # 只保底不封顶：英文 "Configure…" 比中文宽，固定 56px 会把文字截断
        self._crop_config_btn.setMinimumWidth(56)
        self._crop_config_btn.clicked.connect(self._open_crop_config)
        crop_cl.addWidget(self._crop_config_btn)
        self._crop_summary_lbl = QLabel()
        crop_cl.addWidget(self._crop_summary_lbl, 1)
        crop_row_idx, crop_col_idx = divmod(len(order) + 1, 3)
        grid.addWidget(crop_cell, crop_row_idx, crop_col_idx, 1, 2)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        layout.addLayout(grid)
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
        self._aug_skip_empty = QCheckBox()
        self._aug_skip_empty.setChecked(True)
        rl.addWidget(self._aug_skip_empty)
        rl.addSpacing(10)
        self._aug_uncheck_all_btn = QPushButton()
        self._aug_uncheck_all_btn.setObjectName("accentBtn")
        self._aug_uncheck_all_btn.setToolTip(tr("uncheck_all_hint"))
        self._aug_uncheck_all_btn.clicked.connect(self._uncheck_all_transforms)
        rl.addWidget(self._aug_uncheck_all_btn)
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
        fmt_layout = QHBoxLayout()
        fmt_layout.setContentsMargins(0, 0, 0, 0)
        self._exp_fmt_label = QLabel()
        fmt_layout.addWidget(self._exp_fmt_label)
        self._exp_format = QComboBox()
        self._exp_format.addItems([
            "YOLO Detection", "YOLO Seg", "YOLO Pose", "YOLO OBB",
            "VOC Detection", "COCO Detection"
        ])
        self._exp_format.setFixedWidth(150)
        fmt_layout.addWidget(self._exp_format)
        fmt_layout.addStretch()
        layout.addLayout(fmt_layout)
        self._exp_label_title = QLabel()
        layout.addWidget(self._exp_label_title)
        self._exp_label_grid_container = QWidget()
        self._exp_label_grid_container.setMaximumHeight(130)
        self._exp_label_grid = QGridLayout(self._exp_label_grid_container)
        self._exp_label_grid.setSpacing(8)
        self._exp_label_grid.setContentsMargins(0, 2, 0, 2)
        self._exp_label_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._exp_label_checkboxes = {}
        layout.addWidget(self._exp_label_grid_container, 0, Qt.AlignLeft)
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
        hl.addWidget(self._pipe_aug)
        hl.addWidget(QLabel(" → "))
        hl.addWidget(self._pipe_exp)
        hl.addWidget(QLabel(" → "))
        hl.addWidget(self._pipe_split)
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
