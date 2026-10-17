"""
设置对话框 - 快捷键自定义
"""
import json
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QScrollArea, QWidget, QSpinBox, QStackedWidget,
    QComboBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QAbstractItemView
)
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QEvent

from ..core.config import SHORTCUT_CONFIG, LABEL_CACHE_SLOTS, BOX_BORDER_CONFIG
from .theme import ThemeManager
from .dwm import set_titlebar_dark
from .dialog_helpers import center_on_parent, ThemedColorDialog
from ..core import config_manager
from . import i18n


class SettingsDialog(QDialog):
    """设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from . import i18n
        tr = i18n.t
        self._editor = parent
        self._crosshair_color = "#00FF80"
        self.setWindowTitle(tr("设置"))
        self.setMinimumWidth(560)
        self.setMinimumHeight(600)
        self.setObjectName("settingsDialog")

        self.shortcut_inputs = {}
        self._lint_ignored_rules = {}
        self._init_ui()
        self._load_shortcuts()
        self._load_options()
        self.installEventFilter(self)
        if self._editor and hasattr(self._editor, '_shortcuts'):
            for sc in self._editor._shortcuts:
                sc.setEnabled(False)

    def showEvent(self, event):
        """窗口显示后设置标题栏颜色"""
        super().showEvent(event)
        center_on_parent(self)
        self._sync_titlebar()

    def closeEvent(self, event):
        """关闭时恢复主窗口快捷键"""
        super().closeEvent(event)
        if self._editor and hasattr(self._editor, '_shortcuts'):
            for sc in self._editor._shortcuts:
                sc.setEnabled(True)

    def hideEvent(self, event):
        """隐藏时也恢复快捷键"""
        super().hideEvent(event)
        if self._editor and hasattr(self._editor, '_shortcuts'):
            for sc in self._editor._shortcuts:
                sc.setEnabled(True)

    def _sync_titlebar(self):
        """同步标题栏颜色"""
        is_dark = ThemeManager.get_mode().value == "dark"
        hwnd = int(self.winId())
        set_titlebar_dark(hwnd, is_dark)

    def _init_ui(self):
        from . import i18n
        tr = i18n.t
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)

        nav_layout = QVBoxLayout()
        nav_layout.setSpacing(8)

        self.shortcut_page_btn = QPushButton(tr("快捷键设置"))
        self.shortcut_page_btn.setCheckable(True)
        self.shortcut_page_btn.setChecked(True)
        self.shortcut_page_btn.setFixedWidth(108)
        self.shortcut_page_btn.setMinimumHeight(38)
        self.shortcut_page_btn.clicked.connect(lambda: self._switch_page(0))
        nav_layout.addWidget(self.shortcut_page_btn)

        self.options_page_btn = QPushButton(tr("参数设置"))
        self.options_page_btn.setCheckable(True)
        self.options_page_btn.setFixedWidth(108)
        self.options_page_btn.setMinimumHeight(38)
        self.options_page_btn.clicked.connect(lambda: self._switch_page(1))
        nav_layout.addWidget(self.options_page_btn)

        self.lint_page_btn = QPushButton(tr("质检忽略"))
        self.lint_page_btn.setCheckable(True)
        self.lint_page_btn.setFixedWidth(108)
        self.lint_page_btn.setMinimumHeight(38)
        self.lint_page_btn.clicked.connect(lambda: self._switch_page(2))
        nav_layout.addWidget(self.lint_page_btn)
        nav_layout.addStretch()

        self.stack = QStackedWidget()

        group = QGroupBox(tr("快捷键设置"))
        group_layout = QVBoxLayout(group)
        # 与「参数设置」页保持同一首行位置，两页切换时行不会跳动
        group_layout.setContentsMargins(19, 14, 9, 9)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        shortcut_names = {
            'undo': tr("撤销"),
            'redo': tr("重做"),
            'toggle_grid': tr("显示网格线"),
            'toggle_labels': tr("显示BOX"),
            'toggle_label_names': tr("显示标签名"),
            'toggle_scores': tr("显示置信度"),
            'toggle_paste_names': tr("显示贴图名"),
            'auto_save_b': tr("自动保存B"),
            'auto_save_p': tr("自动保存P"),
            'draw_box': tr("绘制检测框"),
            'draw_polygon': tr("绘制多边形"),
            'draw_point': tr("标注关键点"),
            'draw_obb': tr("绘制旋转框"),
            'rotate_cw': tr("顺时针旋转"),
            'rotate_ccw': tr("逆时针旋转"),
            'quit_draw': tr("退出绘制"),
            'next_image': tr("下一张"),
            'prev_image': tr("上一张"),
            'delete_selected': tr("删除选中"),
            'fit_view': tr("适应视图"),
            'zoom_in': tr("放大"),
            'zoom_out': tr("缩小"),
            'remove_image': tr("移除图片"),
            'restore_image': tr("恢复图片"),
            'delete_label': tr("删除标签"),
            'label_cache_slot_1': tr("缓存槽1"),
            'label_cache_slot_2': tr("缓存槽2"),
            'label_cache_slot_3': tr("缓存槽3"),
        }

        for key, name in shortcut_names.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"), 2)

            input_field = QLineEdit()
            input_field.setObjectName("shortcutInput")
            input_field.setText(SHORTCUT_CONFIG.get(key, ''))
            input_field.setReadOnly(True)
            input_field.setMinimumWidth(150)
            input_field.installEventFilter(self)
            input_field.keyPressEvent = lambda event, field=input_field: self._capture_key(event, field)
            self.shortcut_inputs[key] = input_field
            row.addWidget(input_field, 1)

            reset_btn = QPushButton(tr("重置"))
            reset_btn.setObjectName("resetBtn")
            reset_btn.setFixedWidth(70)
            reset_btn.clicked.connect(lambda _, k=key, f=input_field: self._reset_shortcut(k, f))
            row.addWidget(reset_btn)

            scroll_layout.addLayout(row)

        scroll.setWidget(scroll_content)
        group_layout.addWidget(scroll)
        self.stack.addWidget(group)

        opt_group = QGroupBox(tr("参数设置"))
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setContentsMargins(19, 14, 9, 9)

        prefix_row = QHBoxLayout()
        prefix_label = QLabel(tr("添加文件名前缀") + ":")
        prefix_row.addWidget(prefix_label, 2)

        self.prefix_input = QLineEdit()
        self.prefix_input.setObjectName("shortcutInput")
        self.prefix_input.setMinimumWidth(150)
        prefix_row.addWidget(self.prefix_input)
        prefix_row.addStretch()
        opt_layout.addLayout(prefix_row)

        from ..core.config import GRID_CONFIG, DETECTION_BOX_CONFIG, MAGNIFIER_CONFIG, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG
        magnifier_zoom_row = QHBoxLayout()
        magnifier_zoom_label = QLabel(tr("窗口放大器倍率") + ":")
        magnifier_zoom_row.addWidget(magnifier_zoom_label, 2)
        self.magnifier_zoom_spin = QDoubleSpinBox()
        self.magnifier_zoom_spin.setObjectName("paramSpin")
        self.magnifier_zoom_spin.setRange(0.8, 3.0)
        self.magnifier_zoom_spin.setSingleStep(0.1)
        self.magnifier_zoom_spin.setDecimals(1)
        self.magnifier_zoom_spin.setMinimumWidth(150)
        self.magnifier_zoom_spin.setValue(max(0.8, min(3.0, float(MAGNIFIER_CONFIG.get('zoom', 2.0)))))
        magnifier_zoom_row.addWidget(self.magnifier_zoom_spin)
        magnifier_zoom_row.addStretch()
        opt_layout.addLayout(magnifier_zoom_row)

        magnifier_size_row = QHBoxLayout()
        magnifier_size_label = QLabel(tr("窗口放大器尺寸") + ":")
        magnifier_size_row.addWidget(magnifier_size_label, 2)
        self.magnifier_size_spin = QSpinBox()
        self.magnifier_size_spin.setObjectName("paramSpin")
        self.magnifier_size_spin.setRange(80, 400)
        self.magnifier_size_spin.setSingleStep(10)
        self.magnifier_size_spin.setSuffix("")
        self.magnifier_size_spin.setMinimumWidth(150)
        self.magnifier_size_spin.setValue(max(80, min(400, int(MAGNIFIER_CONFIG.get('size', 160)))))
        magnifier_size_row.addWidget(self.magnifier_size_spin)
        magnifier_size_row.addStretch()
        opt_layout.addLayout(magnifier_size_row)

        magnifier_pos_row = QHBoxLayout()
        magnifier_pos_label = QLabel(tr("放大器显示位置") + ":")
        magnifier_pos_row.addWidget(magnifier_pos_label, 2)
        self.magnifier_pos_combo = QComboBox()
        self.magnifier_pos_combo.addItem(tr("侧边"), "side")
        self.magnifier_pos_combo.addItem(tr("鼠标中心"), "center")
        idx = self.magnifier_pos_combo.findData(MAGNIFIER_CONFIG.get('position', 'side'))
        if idx >= 0:
            self.magnifier_pos_combo.setCurrentIndex(idx)
        self.magnifier_pos_combo.setMinimumWidth(150)
        magnifier_pos_row.addWidget(self.magnifier_pos_combo)
        magnifier_pos_row.addStretch()
        opt_layout.addLayout(magnifier_pos_row)

        grid_width_row = QHBoxLayout()
        grid_width_label = QLabel(tr("网格线粗细") + ":")
        grid_width_row.addWidget(grid_width_label, 2)
        self.grid_width_spin = QSpinBox()
        self.grid_width_spin.setRange(1, 10)
        self.grid_width_spin.setValue(GRID_CONFIG.get('line_width', 1))
        self.grid_width_spin.setMinimumWidth(150)
        grid_width_row.addWidget(self.grid_width_spin)
        grid_width_row.addStretch()
        opt_layout.addLayout(grid_width_row)

        grid_alpha_row = QHBoxLayout()
        grid_alpha_label = QLabel(tr("网格线灰度") + ":")
        grid_alpha_row.addWidget(grid_alpha_label, 2)
        self.grid_alpha_spin = QSpinBox()
        self.grid_alpha_spin.setRange(0, 255)
        self.grid_alpha_spin.setValue(GRID_CONFIG.get('alpha', 120))
        self.grid_alpha_spin.setMinimumWidth(150)
        grid_alpha_row.addWidget(self.grid_alpha_spin)
        grid_alpha_row.addStretch()
        opt_layout.addLayout(grid_alpha_row)

        from ..core.config import STATUSBAR_CONFIG
        max_labels_row = QHBoxLayout()
        max_labels_label = QLabel(tr("状态栏类别数") + ":")
        max_labels_row.addWidget(max_labels_label, 2)
        self.max_labels_spin = QSpinBox()
        self.max_labels_spin.setRange(1, 20)
        self.max_labels_spin.setValue(getattr(self._editor, '_max_labels', STATUSBAR_CONFIG.get('max_labels', 3)))
        self.max_labels_spin.setMinimumWidth(150)
        max_labels_row.addWidget(self.max_labels_spin)
        max_labels_row.addStretch()
        opt_layout.addLayout(max_labels_row)

        label_font_size_row = QHBoxLayout()
        label_font_size_label = QLabel(tr("类别名字号") + ":")
        label_font_size_row.addWidget(label_font_size_label, 2)
        self.label_font_size_spin = QSpinBox()
        self.label_font_size_spin.setRange(5, 15)
        self.label_font_size_spin.setValue(DETECTION_BOX_CONFIG.get('label_font_size', 9))
        self.label_font_size_spin.setMinimumWidth(150)
        label_font_size_row.addWidget(self.label_font_size_spin)
        label_font_size_row.addStretch()
        opt_layout.addLayout(label_font_size_row)

        label_position_row = QHBoxLayout()
        label_position_label = QLabel(tr("标签名位置") + ":")
        label_position_row.addWidget(label_position_label, 2)
        self.label_position_combo = QComboBox()
        self.label_position_combo.setMinimumWidth(150)
        self.label_position_combo.addItem(tr("框外侧"), "outside")
        self.label_position_combo.addItem(tr("框内侧"), "inside")
        label_position = DETECTION_BOX_CONFIG.get('label_position', 'outside')
        index = self.label_position_combo.findData(label_position)
        self.label_position_combo.setCurrentIndex(index if index >= 0 else 0)
        label_position_row.addWidget(self.label_position_combo)
        label_position_row.addStretch()
        opt_layout.addLayout(label_position_row)

        handle_size_row = QHBoxLayout()
        handle_size_label = QLabel(tr("缩放句柄大小") + ":")
        handle_size_row.addWidget(handle_size_label, 2)
        self.handle_size_spin = QSpinBox()
        self.handle_size_spin.setRange(7, 15)
        self.handle_size_spin.setValue(DETECTION_BOX_CONFIG.get('resize_handle_size', 8))
        self.handle_size_spin.setMinimumWidth(150)
        handle_size_row.addWidget(self.handle_size_spin)
        handle_size_row.addStretch()
        opt_layout.addLayout(handle_size_row)

        max_poly_row = QHBoxLayout()
        max_poly_row.addWidget(QLabel(tr("多边形最大点数") + ":"), 2)
        self.max_polygon_points_spin = QSpinBox()
        self.max_polygon_points_spin.setRange(3, 64)
        self.max_polygon_points_spin.setValue(DETECTION_BOX_CONFIG.get('max_polygon_points', 32))
        self.max_polygon_points_spin.setMinimumWidth(150)
        max_poly_row.addWidget(self.max_polygon_points_spin)
        max_poly_row.addStretch()
        opt_layout.addLayout(max_poly_row)

        rotate_step_row = QHBoxLayout()
        rotate_step_row.addWidget(QLabel(tr("旋转角度步长") + ":"), 2)
        self.rotate_step_spin = QSpinBox()
        self.rotate_step_spin.setObjectName("paramSpin")
        self.rotate_step_spin.setRange(1, 6)
        self.rotate_step_spin.setMinimumWidth(150)
        from ..core.config import OBB_CONFIG
        self.rotate_step_spin.setValue(max(1, min(6, int(OBB_CONFIG.get('rotate_step', 1)))))
        rotate_step_row.addWidget(self.rotate_step_spin)
        rotate_step_row.addStretch()
        opt_layout.addLayout(rotate_step_row)

        box_border_width_row = QHBoxLayout()
        box_border_width_label = QLabel(tr("框线粗细") + ":")
        box_border_width_row.addWidget(box_border_width_label, 2)
        self.box_border_width_spin = QDoubleSpinBox()
        self.box_border_width_spin.setObjectName("paramSpin")
        self.box_border_width_spin.setRange(1, 4)
        self.box_border_width_spin.setSingleStep(0.5)
        self.box_border_width_spin.setDecimals(1)
        self.box_border_width_spin.setMinimumWidth(150)
        self.box_border_width_spin.setValue(max(1, min(4, float(BOX_BORDER_CONFIG['width']))))
        box_border_width_row.addWidget(self.box_border_width_spin)
        box_border_width_row.addStretch()
        opt_layout.addLayout(box_border_width_row)

        nudge_step_row = QHBoxLayout()
        nudge_step_label = QLabel(tr("移动步长") + ":")
        nudge_step_row.addWidget(nudge_step_label, 2)
        self.nudge_step_spin = QSpinBox()
        self.nudge_step_spin.setRange(1, 5)
        self.nudge_step_spin.setValue(NUDGE_CONFIG.get('step', 1))
        self.nudge_step_spin.setMinimumWidth(150)
        nudge_step_row.addWidget(self.nudge_step_spin)
        nudge_step_row.addStretch()
        opt_layout.addLayout(nudge_step_row)

        for label, attribute, config_key in (
            ("检测框缩放步长", "detection_box_scale_step_spin", "detection_box_scale_step"),
            ("贴图缩放步长", "paste_item_scale_step_spin", "paste_item_scale_step"),
        ):
            scale_step_row = QHBoxLayout()
            scale_step_row.addWidget(QLabel(tr(label) + ":"), 2)
            scale_step_spin = QDoubleSpinBox()
            scale_step_spin.setObjectName("paramSpin")
            scale_step_spin.setRange(0.01, 0.30)
            scale_step_spin.setSingleStep(0.01)
            scale_step_spin.setDecimals(2)
            scale_step_spin.setMinimumWidth(150)
            scale_step_spin.setValue(max(0.01, min(0.30, float(DETECTION_BOX_WHEEL_CONFIG[config_key]))))
            setattr(self, attribute, scale_step_spin)
            scale_step_row.addWidget(scale_step_spin)
            scale_step_row.addStretch()
            opt_layout.addLayout(scale_step_row)

        detection_box_wheel_edge_step_row = QHBoxLayout()
        detection_box_wheel_edge_step_label = QLabel(tr("单侧位移像素") + ":")
        detection_box_wheel_edge_step_row.addWidget(detection_box_wheel_edge_step_label, 2)
        self.detection_box_wheel_edge_step_spin = QSpinBox()
        self.detection_box_wheel_edge_step_spin.setRange(1, 50)
        self.detection_box_wheel_edge_step_spin.setMinimumWidth(150)
        self.detection_box_wheel_edge_step_spin.setValue(max(1, min(50, int(DETECTION_BOX_WHEEL_CONFIG.get('edge_step', 5)))))
        detection_box_wheel_edge_step_row.addWidget(self.detection_box_wheel_edge_step_spin)
        detection_box_wheel_edge_step_row.addStretch()
        opt_layout.addLayout(detection_box_wheel_edge_step_row)

        crosshair_width_row = QHBoxLayout()
        crosshair_width_label = QLabel(tr("十字线粗细") + ":")
        crosshair_width_row.addWidget(crosshair_width_label, 2)
        self.crosshair_width_spin = QDoubleSpinBox()
        self.crosshair_width_spin.setRange(0.5, 3.0)
        self.crosshair_width_spin.setSingleStep(0.5)
        self.crosshair_width_spin.setDecimals(1)
        self.crosshair_width_spin.setMinimumWidth(150)
        self.crosshair_width_spin.setValue(max(0.5, min(3.0, float(CROSSHAIR_CONFIG.get('width', 1.0)))))
        crosshair_width_row.addWidget(self.crosshair_width_spin)
        crosshair_width_row.addStretch()
        opt_layout.addLayout(crosshair_width_row)

        crosshair_alpha_row = QHBoxLayout()
        crosshair_alpha_label = QLabel(tr("十字线透明度") + ":")
        crosshair_alpha_row.addWidget(crosshair_alpha_label, 2)
        self.crosshair_alpha_spin = QSpinBox()
        self.crosshair_alpha_spin.setRange(0, 255)
        self.crosshair_alpha_spin.setMinimumWidth(150)
        self.crosshair_alpha_spin.setValue(max(0, min(255, int(CROSSHAIR_CONFIG.get('alpha', 160)))))
        crosshair_alpha_row.addWidget(self.crosshair_alpha_spin)
        crosshair_alpha_row.addStretch()
        opt_layout.addLayout(crosshair_alpha_row)

        crosshair_color_row = QHBoxLayout()
        crosshair_color_label = QLabel(tr("十字线颜色") + ":")
        crosshair_color_row.addWidget(crosshair_color_label, 2)
        self.crosshair_color_btn = QPushButton()
        self.crosshair_color_btn.setMinimumWidth(150)
        self.crosshair_color_btn.clicked.connect(self._choose_crosshair_color)
        crosshair_color_row.addWidget(self.crosshair_color_btn)
        crosshair_color_row.addStretch()
        opt_layout.addLayout(crosshair_color_row)

        opt_layout.addStretch()
        self.stack.addWidget(opt_group)

        lint_group = QGroupBox(tr("质检忽略规则"))
        lint_layout = QVBoxLayout(lint_group)
        lint_layout.setContentsMargins(19, 14, 9, 9)

        lint_hint = QLabel(tr("被忽略的问题不再出现在质检结果中；移除后将重新报告。"))
        lint_hint.setWordWrap(True)
        lint_layout.addWidget(lint_hint)

        self.lint_rules_list = QListWidget()
        self.lint_rules_list.setObjectName("lintRulesList")
        self.lint_rules_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lint_rules_list.setMinimumWidth(150)
        lint_layout.addWidget(self.lint_rules_list, 1)

        lint_btn_row = QHBoxLayout()
        self.lint_remove_btn = QPushButton(tr("移除选中"))
        self.lint_remove_btn.clicked.connect(self._remove_selected_lint_rules)
        lint_btn_row.addWidget(self.lint_remove_btn)
        self.lint_clear_btn = QPushButton(tr("清空全部"))
        self.lint_clear_btn.setObjectName("dangerBtn")
        self.lint_clear_btn.clicked.connect(self._clear_lint_rules)
        lint_btn_row.addWidget(self.lint_clear_btn)
        lint_btn_row.addStretch()
        lint_layout.addLayout(lint_btn_row)

        self.stack.addWidget(lint_group)

        content_layout.addLayout(nav_layout, 0)
        content_layout.addWidget(self.stack, 1)

        layout.addLayout(content_layout, 1)

        btn_layout = QHBoxLayout()
        self.reset_all_btn = QPushButton(tr("恢复默认设置"))
        self.reset_all_btn.setObjectName("dangerBtn")
        self.reset_all_btn.setToolTip(tr("重置所有设置"))
        self.reset_all_btn.clicked.connect(self._reset_all_settings)

        # 左下角只放版本号；「恢复默认设置」挪到保存左边（用户要求）。
        # 版本取自 `labelforge.__version__`，不硬编码——发版时只需改包常量，
        # 不会两处漂移。
        left_col = QVBoxLayout()
        left_col.setSpacing(2)
        self.version_label = QLabel(self._version_text())
        self.version_label.setObjectName("versionLabel")
        left_col.addWidget(self.version_label)
        btn_layout.addLayout(left_col)
        btn_layout.addStretch()

        # 顺序：恢复默认设置 → 保存 → 取消。危险动作紧邻保存但仍在左侧，
        # 且与保存之间不夹 stretch，避免它被推到行中间。
        btn_layout.addWidget(self.reset_all_btn)

        save_btn = QPushButton(tr("保存"))
        save_btn.setObjectName("successBtn")
        save_btn.clicked.connect(self._save_shortcuts)
        btn_layout.addWidget(save_btn)

        cancel_btn = QPushButton(tr("取消"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)

    @staticmethod
    def _version_text():
        """左下角版本串；取不到 `__version__` 时不显示假版本号。

        只显示 `v1.5.3`，不带应用名前缀——对话框标题里已经有应用名了，重复。
        """
        from .. import __version__

        return f"v{__version__}" if __version__ else ""

    def _reset_all_settings(self):
        """恢复出厂设置（需二次确认）。"""
        from .dialog_helpers import question, warning
        from PySide6.QtWidgets import QMessageBox
        tr = i18n.t
        answer = question(
            self, tr("恢复默认设置"),
            tr("将删除所有自定义配置（快捷键、主题、语言、参数与记忆记录），并恢复出厂默认。此操作不可撤销，是否继续？"),
        )
        if answer != QMessageBox.Yes:
            return
        if not self._editor or not hasattr(self._editor, 'reset_settings_to_default'):
            warning(self, tr("警告"), tr("已恢复默认设置"))
            self.accept()
            return
        if not self._editor.reset_settings_to_default():
            warning(self, tr("警告"), tr("无法写入配置文件"))
            return
        self.accept()
        if hasattr(self._editor, 'status_label'):
            self._editor.status_label.setText(tr("已恢复默认设置"))

    def _switch_page(self, index):
        """切换左侧按钮对应的右侧设置页。"""
        self.stack.setCurrentIndex(index)
        self.shortcut_page_btn.setChecked(index == 0)
        self.options_page_btn.setChecked(index == 1)
        self.lint_page_btn.setChecked(index == 2)

    def eventFilter(self, obj, event):
        """拦截快捷键输入框的按键事件（含Ctrl+组合键）"""
        if event.type() == QEvent.KeyPress:
            focused = self.focusWidget()
            if focused is not None:
                for key, field in self.shortcut_inputs.items():
                    if focused is field or obj is field:
                        self._capture_key(event, field)
                        return True
        return super().eventFilter(obj, event)

    def _capture_key(self, event, field):
        """捕获按键"""
        key = event.key()
        modifiers = event.modifiers()

        key_name = ""
        if modifiers & Qt.ControlModifier:
            key_name += "Ctrl+"
        if modifiers & Qt.AltModifier:
            key_name += "Alt+"
        if modifiers & Qt.ShiftModifier:
            key_name += "Shift+"

        key_map = {
            Qt.Key_A: 'A', Qt.Key_B: 'B', Qt.Key_C: 'C', Qt.Key_D: 'D',
            Qt.Key_E: 'E', Qt.Key_F: 'F', Qt.Key_G: 'G', Qt.Key_H: 'H',
            Qt.Key_I: 'I', Qt.Key_J: 'J', Qt.Key_K: 'K', Qt.Key_L: 'L',
            Qt.Key_M: 'M', Qt.Key_N: 'N', Qt.Key_O: 'O', Qt.Key_P: 'P',
            Qt.Key_Q: 'Q', Qt.Key_R: 'R', Qt.Key_S: 'S', Qt.Key_T: 'T',
            Qt.Key_U: 'U', Qt.Key_V: 'V', Qt.Key_W: 'W', Qt.Key_X: 'X',
            Qt.Key_Y: 'Y', Qt.Key_Z: 'Z',
            Qt.Key_0: '0', Qt.Key_1: '1', Qt.Key_2: '2', Qt.Key_3: '3',
            Qt.Key_4: '4', Qt.Key_5: '5', Qt.Key_6: '6', Qt.Key_7: '7',
            Qt.Key_8: '8', Qt.Key_9: '9',
            Qt.Key_Delete: 'Delete', Qt.Key_Space: 'Space',
            Qt.Key_F1: 'F1', Qt.Key_F2: 'F2', Qt.Key_F3: 'F3', Qt.Key_F4: 'F4',
            Qt.Key_F5: 'F5', Qt.Key_F6: 'F6', Qt.Key_F7: 'F7', Qt.Key_F8: 'F8',
            Qt.Key_F9: 'F9', Qt.Key_F10: 'F10', Qt.Key_F11: 'F11', Qt.Key_F12: 'F12',
        }

        if key in key_map:
            key_name += key_map[key]
            field.setText(key_name)

    def _reset_shortcut(self, key, field):
        """重置单个快捷键"""
        if key.startswith('label_cache_slot_'):
            slot_index = int(key.rsplit('_', 1)[-1]) - 1
            field.setText(LABEL_CACHE_SLOTS[slot_index].get('shortcut', ''))
            return
        field.setText(SHORTCUT_CONFIG.get(key, ''))

    def _load_shortcuts(self):
        """加载保存的快捷键"""
        shortcuts = config_manager.load_shortcuts()
        label_cache_slots = config_manager.load_all().get('label_cache_slots', LABEL_CACHE_SLOTS)
        for key, field in self.shortcut_inputs.items():
            if key.startswith('label_cache_slot_'):
                slot_index = int(key.rsplit('_', 1)[-1]) - 1
                if slot_index < len(label_cache_slots):
                    field.setText(label_cache_slots[slot_index].get('shortcut', ''))
            elif key in shortcuts:
                field.setText(shortcuts[key])

    def _load_options(self):
        """加载选项状态"""
        if self._editor:
            self.prefix_input.setText(self._editor.prefix_input.text())
        from ..core.config import GRID_CONFIG, DETECTION_BOX_CONFIG, MAGNIFIER_CONFIG, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG, BOX_BORDER_CONFIG, OBB_CONFIG
        self.grid_width_spin.setValue(GRID_CONFIG.get('line_width', 1))
        self.grid_alpha_spin.setValue(GRID_CONFIG.get('alpha', 120))
        self.handle_size_spin.setValue(max(7, min(15, DETECTION_BOX_CONFIG.get('resize_handle_size', 8))))
        self.max_polygon_points_spin.setValue(max(3, min(64, int(DETECTION_BOX_CONFIG.get('max_polygon_points', 32)))))
        self.rotate_step_spin.setValue(max(1, min(6, int(OBB_CONFIG.get("rotate_step", 1)))))
        self.box_border_width_spin.setValue(max(1, min(4, float(BOX_BORDER_CONFIG['width']))))
        self.label_font_size_spin.setValue(max(5, min(15, DETECTION_BOX_CONFIG.get('label_font_size', 9))))
        label_position = DETECTION_BOX_CONFIG.get('label_position', 'outside')
        index = self.label_position_combo.findData(label_position)
        self.label_position_combo.setCurrentIndex(index if index >= 0 else 0)
        self.magnifier_zoom_spin.setValue(max(0.8, min(3.0, float(MAGNIFIER_CONFIG.get('zoom', 2.0)))))
        self.magnifier_size_spin.setValue(max(80, min(400, int(MAGNIFIER_CONFIG.get('size', 160)))))
        self.nudge_step_spin.setValue(NUDGE_CONFIG.get('step', 1))
        self.detection_box_scale_step_spin.setValue(max(0.01, min(0.30, float(DETECTION_BOX_WHEEL_CONFIG['detection_box_scale_step']))))
        self.paste_item_scale_step_spin.setValue(max(0.01, min(0.30, float(DETECTION_BOX_WHEEL_CONFIG['paste_item_scale_step']))))
        self.detection_box_wheel_edge_step_spin.setValue(max(1, min(50, int(DETECTION_BOX_WHEEL_CONFIG.get('edge_step', 5)))))
        self.crosshair_width_spin.setValue(max(0.5, min(3.0, float(CROSSHAIR_CONFIG.get('width', 1.0)))))
        color = str(CROSSHAIR_CONFIG.get('color', '#00FF80'))
        self._crosshair_color = color if len(color) == 7 and color.startswith('#') else '#00FF80'
        self.crosshair_alpha_spin.setValue(max(0, min(255, int(CROSSHAIR_CONFIG.get('alpha', 160)))))
        self._update_crosshair_color_button()
        self._load_lint_ignored_rules()

    def _load_lint_ignored_rules(self):
        """从配置加载质检忽略规则并刷新列表。"""
        rules = config_manager.load_all().get('lint_ignored_rules') or {}
        self._lint_ignored_rules = {k: list(v) for k, v in rules.items()}
        self._refresh_lint_rules_list()

    def _refresh_lint_rules_list(self):
        """按类型顺序重建忽略规则列表；历史遗留类型也保留可移除。"""
        from ..engine.quality_lint import KIND_LABELS
        tr = i18n.t
        self.lint_rules_list.clear()
        kinds = [k for k in KIND_LABELS if k in self._lint_ignored_rules]
        kinds += [k for k in sorted(self._lint_ignored_rules) if k not in KIND_LABELS]
        for kind in kinds:
            label = tr(KIND_LABELS.get(kind, kind))
            for key in self._lint_ignored_rules.get(kind) or []:
                text = f"{label} · {tr('整类')}" if key == '*' else f"{label} · {key}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, (kind, key))
                self.lint_rules_list.addItem(item)

    def _remove_selected_lint_rules(self):
        """移除选中的忽略规则（点「保存」后生效）。"""
        items = self.lint_rules_list.selectedItems()
        if not items:
            return
        for item in items:
            kind, key = item.data(Qt.UserRole)
            keys = self._lint_ignored_rules.get(kind)
            if not keys:
                continue
            if key in keys:
                keys.remove(key)
            if not keys:
                self._lint_ignored_rules.pop(kind, None)
        self._refresh_lint_rules_list()

    def _clear_lint_rules(self):
        """清空全部忽略规则（点「保存」后生效）。"""
        self._lint_ignored_rules = {}
        self._refresh_lint_rules_list()

    def _update_crosshair_color_button(self):
        self.crosshair_color_btn.setText(self._crosshair_color)
        self.crosshair_color_btn.setStyleSheet(
            f"QPushButton {{ background-color: {self._crosshair_color}; color: white; border: 1px solid {self._crosshair_color}; border-radius: 4px; padding: 3px 8px; font-size: 12px; font-weight: bold; }}"
        )

    def _choose_crosshair_color(self):
        dialog = ThemedColorDialog(self)
        dialog.setWindowTitle(i18n.t("十字线颜色"))
        dialog.setCurrentColor(QColor(self._crosshair_color))
        if dialog.exec_() != QDialog.Accepted:
            return
        color = dialog.currentColor()
        self._crosshair_color = color.name().upper()
        self._update_crosshair_color_button()

    def _save_shortcuts(self):
        """保存快捷键到文件并立即生效"""
        shortcuts = {}
        label_cache_slots = []
        if self._editor and hasattr(self._editor, 'label_cache_slots'):
            label_cache_slots = [dict(slot) for slot in self._editor.label_cache_slots]
        else:
            label_cache_slots = [dict(slot) for slot in config_manager.load_all().get('label_cache_slots', LABEL_CACHE_SLOTS)]
        for key, field in self.shortcut_inputs.items():
            text = field.text().strip()
            if key.startswith('label_cache_slot_'):
                slot_index = int(key.rsplit('_', 1)[-1]) - 1
                if slot_index < len(label_cache_slots):
                    label_cache_slots[slot_index]['shortcut'] = text or str(slot_index + 1)
            elif text:
                shortcuts[key] = text

        if self._editor:
            self._editor.prefix_input.setText(self.prefix_input.text())
            self._editor.shortcut_config = shortcuts
            self._editor.label_cache_slots = label_cache_slots
            if hasattr(self._editor, 'update_shortcuts'):
                self._editor.update_shortcuts()
            if hasattr(self._editor, '_refresh_menu_shortcuts'):
                self._editor._refresh_menu_shortcuts()
            if hasattr(self._editor, '_rebuild_label_cache_menu'):
                self._editor._rebuild_label_cache_menu()

        from ..core.config import GRID_CONFIG, DETECTION_BOX_CONFIG, PASTE_ITEM_CONFIG, MAGNIFIER_CONFIG, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG, BOX_BORDER_CONFIG, OBB_CONFIG
        GRID_CONFIG['line_width'] = self.grid_width_spin.value()
        GRID_CONFIG['alpha'] = self.grid_alpha_spin.value()
        handle_size = max(7, min(15, self.handle_size_spin.value()))
        DETECTION_BOX_CONFIG['resize_handle_size'] = handle_size
        max_polygon_points = max(3, min(64, self.max_polygon_points_spin.value()))
        DETECTION_BOX_CONFIG['max_polygon_points'] = max_polygon_points
        rotate_step = max(1, min(6, int(self.rotate_step_spin.value())))
        OBB_CONFIG['rotate_step'] = rotate_step
        PASTE_ITEM_CONFIG['handle_size'] = handle_size
        box_border_width = max(1, min(4, float(self.box_border_width_spin.value())))
        BOX_BORDER_CONFIG['width'] = box_border_width
        label_font_size = max(5, min(15, self.label_font_size_spin.value()))
        label_position = self.label_position_combo.currentData() or 'outside'
        if label_position not in ('outside', 'inside'):
            label_position = 'outside'
        DETECTION_BOX_CONFIG['label_font_size'] = label_font_size
        DETECTION_BOX_CONFIG['label_position'] = label_position
        magnifier_zoom = max(0.8, min(3.0, float(self.magnifier_zoom_spin.value())))
        MAGNIFIER_CONFIG['zoom'] = magnifier_zoom
        magnifier_size = max(80, min(400, self.magnifier_size_spin.value()))
        MAGNIFIER_CONFIG['size'] = magnifier_size
        MAGNIFIER_CONFIG['position'] = self.magnifier_pos_combo.currentData() or 'side'
        nudge_step = max(1, min(5, self.nudge_step_spin.value()))
        NUDGE_CONFIG['step'] = nudge_step
        detection_box_scale_step = max(0.01, min(0.30, float(self.detection_box_scale_step_spin.value())))
        paste_item_scale_step = max(0.01, min(0.30, float(self.paste_item_scale_step_spin.value())))
        detection_box_wheel_edge_step = max(1, min(50, self.detection_box_wheel_edge_step_spin.value()))
        crosshair_width = max(0.5, min(3.0, float(self.crosshair_width_spin.value())))
        crosshair_alpha = max(0, min(255, self.crosshair_alpha_spin.value()))
        DETECTION_BOX_WHEEL_CONFIG['detection_box_scale_step'] = detection_box_scale_step
        DETECTION_BOX_WHEEL_CONFIG['paste_item_scale_step'] = paste_item_scale_step
        DETECTION_BOX_WHEEL_CONFIG['edge_step'] = detection_box_wheel_edge_step
        CROSSHAIR_CONFIG['width'] = crosshair_width
        CROSSHAIR_CONFIG['color'] = self._crosshair_color
        CROSSHAIR_CONFIG['alpha'] = crosshair_alpha

        from ..core import config_manager as cm
        max_labels = self.max_labels_spin.value()
        cm.save_all(
            shortcuts=shortcuts,
            max_labels=max_labels,
            grid_line_width=GRID_CONFIG['line_width'],
            grid_alpha=GRID_CONFIG['alpha'],
            resize_handle_size=handle_size,
            max_polygon_points=max_polygon_points,
            rotate_step=rotate_step,
            label_font_size=label_font_size,
            label_position=label_position,
            magnifier_zoom=magnifier_zoom,
            magnifier_size=magnifier_size,
            magnifier_position=MAGNIFIER_CONFIG['position'],
            label_cache_slots=label_cache_slots,
            nudge_step=nudge_step,
            detection_box_scale_step=detection_box_scale_step,
            paste_item_scale_step=paste_item_scale_step,
            detection_box_wheel_edge_step=detection_box_wheel_edge_step,
            crosshair_width=crosshair_width,
            crosshair_color=self._crosshair_color,
            crosshair_alpha=crosshair_alpha,
            box_border_width=box_border_width,
            lint_ignored_rules=self._lint_ignored_rules,
        )
        if self._editor:
            self._editor._max_labels = max_labels
            self._editor._lint_ignored_rules = {
                k: list(v) for k, v in self._lint_ignored_rules.items()}
            if hasattr(self._editor, 'canvas'):
                self._editor.canvas.update()
        self.accept()


def load_shortcuts():
    """加载快捷键配置"""
    return config_manager.load_shortcuts()
