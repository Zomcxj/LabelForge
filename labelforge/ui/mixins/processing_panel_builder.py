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

# 按语义分成三组再排：几何（翻转/平移/旋转/缩放）→ 色彩（亮度/对比度/
# 色相/饱和度/明度）→ 噪声。原来的顺序是历史堆积，色彩与噪声、几何混在
# 一起，扫一眼看不出哪几项是一类。
_TRANSFORM_ORDER = [
    # 几何
    "fliph", "flipv", "trans", "rotate", "scale",
    # 色彩
    "bright", "contrast", "hue", "saturation", "value",
    # 噪声
    "gauss", "saltpepper",
]

# 数据增强网格的列数。4 列比 3 列更省纵向空间，且 12 个变换正好铺满 3 行。
_COLS = 4

# 分组标题：key 是组内第一个变换名，插入到它前面。
_TRANSFORM_GROUPS = {
    "fliph": "几何",
    "bright": "色彩",
    "gauss": "噪声",
}


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
        # 先把「每个变换落在第几行第几列」算完，再统一放置。插分组标题会让
        # 后续行号顺延，边算边放会把后面的格子挤错位。
        _slots = {}
        _heads = {}                # 分组标题单独记，不能和变换共用 key：
                                   # 「几何」组的第一项就是 fliph，共用会让
                                   # fliph 自己被当成标题，复选框永远建不出来
        _row, _col = 0, 1          # 第 0 行第 0 列已被「原图」占用
        for name in order:
            group = _TRANSFORM_GROUPS.get(name)
            if group is not None:
                if _col != 0:
                    _row, _col = _row + 1, 0
                _heads[group] = (_row, _col)
                _row, _col = _row + 1, 0
            _slots[name] = (_row, _col)
            _col += 1
            if _col >= _COLS:
                _row, _col = _row + 1, 0
        for group, (grow, gcol) in _heads.items():
            # 分组标题独占一整行，不占列宽
            head = QLabel(tr(group))
            head.setStyleSheet(
                "color: #9E9E9E; font-size: 11px; background: transparent;")
            grid.addWidget(head, grow, 0, 1, _COLS)
        for name in order:
            row, col = _slots[name]
            meta = TRANSFORM_META.get(name, (name, []))
            display_name, param_defs = meta
            cell = QWidget()
            cell.setMinimumWidth(140)
            cl = QHBoxLayout(cell)
            cl.setContentsMargins(2, 1, 2, 1)
            cl.setSpacing(2)
            cb = QCheckBox(display_name)
            cb.setChecked(name in ("fliph", "flipv", "bright", "rotate", "trans"))
            # 勾选框里直接带文字，不再用 QLabel + QCheckBox 两件套：
            # 那样文字与方框的基线差 1~2px，整列看上去是歪的。
            cl.addWidget(cb)
            cl.addStretch()
            param_spins = {}
            for pname, ptype, pmin, pmax, dmin, dmax in param_defs:
                sep = QLabel("~")
                sep.setFixedWidth(10)
                # 居中而不是默认靠上，否则 ~ 会比两侧输入框的基线高出一截
                sep.setAlignment(Qt.AlignCenter)
                spin_min = self._make_spin(ptype, pmin, pmax, dmin)
                spin_max = self._make_spin(ptype, pmin, pmax, dmax)
                # 两侧输入框统一右/左对齐，数字位数不同时也能对齐小数点
                for _sp, _al in ((spin_min, Qt.AlignRight), (spin_max, Qt.AlignLeft)):
                    _sp.setAlignment(_al)
                    _sp.setFixedWidth(56)
                cl.addWidget(spin_min)
                cl.addWidget(sep)
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
        # 滑窗裁剪跟噪声组之后，单独占一整行（它带「配置…」按钮和摘要）
        _last_row = max(v[0] for v in _slots.values()) if _slots else 0
        grid.addWidget(crop_cell, _last_row + 1, 0, 1, _COLS)
        for c in range(_COLS):
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
        # 原来只是三个复选框 + 两个 →，勾没勾一眼看不出来，像没做完的控件。
        # 文字由 QCheckBox 自己带（_refresh_texts 里 setText），这里**不再**
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
