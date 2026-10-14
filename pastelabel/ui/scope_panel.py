"""推理范围面板：全图 / 滑窗 / 区域。

窗口参数走 `SlidingWindowDialog`，区域沿用画布上的临时区域（`region_boxes`）。
面板只负责收集选择，校验与快照由 `mixins/inference.py` 在启动任务前完成。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QRadioButton, QVBoxLayout,
)

from ..engine.inference_scope import (
    MODE_FULL, MODE_ROI, MODE_WINDOW,
)
from .dialog_helpers import place_within, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager

_MODE_KEYS = (
    (MODE_FULL, "全图", "整张图推理一次"),
    (MODE_WINDOW, "滑窗", "按窗口切分后逐窗推理，结果合并去重"),
    (MODE_ROI, "区域", "只在画布上画好的区域内推理"),
)


class ScopePanel(QDialog):
    """范围选择；确定后 `selection()` 返回 (mode, windows)。"""

    edit_regions_requested = pyqtSignal()

    def __init__(self, windows=None, mode=MODE_FULL, region_count=0,
                 reference_size=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("推理范围"))
        self.setMinimumWidth(440)
        self._windows = dict(windows or {})
        self._region_count = int(region_count or 0)
        self._reference = reference_size
        self._anchor_rect = None
        self._build_ui()
        self._apply_style()
        self.set_mode(mode)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self._buttons = {}
        for mode, name, hint in _MODE_KEYS:
            row = QHBoxLayout()
            button = QRadioButton(tr(name))
            button.setObjectName("scopeMode")
            button.toggled.connect(self._refresh)
            self._buttons[mode] = button
            row.addWidget(button)
            label = QLabel(tr(hint))
            label.setObjectName("scopeHint")
            row.addWidget(label)
            row.addStretch()
            layout.addLayout(row)

        layout.addSpacing(4)

        row = QHBoxLayout()
        self.window_btn = QPushButton("")
        self.window_btn.setObjectName("optionsBtn")
        self.window_btn.clicked.connect(self._edit_windows)
        row.addWidget(self.window_btn)
        self.region_label = QLabel("")
        self.region_label.setObjectName("scopeHint")
        row.addWidget(self.region_label)
        row.addStretch()
        self.region_btn = QPushButton(tr("编辑区域"))
        self.region_btn.setObjectName("optionsBtn")
        self.region_btn.setToolTip(tr("区域在贴图模式里绘制，切过去后不会丢失当前标注"))
        self.region_btn.clicked.connect(self.edit_regions_requested.emit)
        row.addWidget(self.region_btn)
        layout.addLayout(row)

        self.warn_label = QLabel("")
        self.warn_label.setObjectName("scopeWarn")
        self.warn_label.setWordWrap(True)
        layout.addWidget(self.warn_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.ok_btn = QPushButton(tr("确定"))
        self.ok_btn.setObjectName("successBtn")
        self.cancel_btn = QPushButton(tr("取消"))
        self.cancel_btn.setObjectName("accentBtn")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self.ok_btn)
        buttons.addWidget(self.cancel_btn)
        layout.addLayout(buttons)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['widget_bg']}; color: {theme['text_primary']}; }}
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#scopeHint {{ color: {theme['text_secondary']}; }}
            QLabel#scopeWarn {{ color: {theme['accent']}; }}
            QRadioButton {{ color: {theme['text_primary']}; padding: 2px; }}
            QPushButton {{
                background-color: {theme['panel_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                padding: 4px 10px;
            }}
            QPushButton:hover {{ border-color: {theme['accent']}; }}
        """)

    def refresh_theme(self):
        self._apply_style()
        sync_titlebar(self)

    def showEvent(self, event):
        super().showEvent(event)
        place_within(self, self._anchor_rect)
        sync_titlebar(self)

    # ---------- 对外接口 ----------

    def mode(self):
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return MODE_FULL

    def set_mode(self, mode):
        button = self._buttons.get(mode) or self._buttons[MODE_FULL]
        button.setChecked(True)
        self._refresh()

    def windows(self):
        return dict(self._windows)

    def set_windows(self, windows):
        self._windows = dict(windows or {})
        self._refresh()

    def set_region_count(self, count):
        self._region_count = int(count or 0)
        self._refresh()

    def set_reference(self, size):
        """参考图尺寸：窗口数预估要用它；没有参考图时预估显示为不可用。"""
        self._reference = size

    def set_anchor_rect(self, rect):
        """右侧侧栏那块位置；None 表示居中显示（拿不到侧栏时）。"""
        self._anchor_rect = rect

    def selection(self):
        """(mode, windows)；窗口参数只在滑窗模式返回。"""
        mode = self.mode()
        if mode == MODE_WINDOW:
            return mode, dict(self._windows)
        return mode, None

    def window_summary(self):
        """窗口参数摘要，如「640×640 重叠 200」；无参数时返回空串。"""
        if not self._windows:
            return ""
        return (f"{self._windows.get('w')}×{self._windows.get('h')} "
                f"{tr('重叠')} {self._windows.get('overlap')}")

    # ---------- 内部 ----------

    def _refresh(self, _checked=None):
        mode = self.mode()
        needs_window = mode == MODE_WINDOW
        needs_region = mode == MODE_ROI
        self.window_btn.setVisible(needs_window)
        self.window_btn.setText(f"{tr('滑窗参数')}: {self.window_summary()}"
                                if self._windows else tr("滑窗参数"))
        self.region_btn.setVisible(needs_region)
        self.region_label.setVisible(needs_region)
        self.region_label.setText(f"{self._region_count} {tr('个区域')}")

        warnings = []
        if needs_region and not self._region_count:
            warnings.append(tr("没有可用区域，请先画好区域"))
        if needs_region:
            warnings.append(tr("区域推理只保留框中心落在区域内的结果"))
        self.warn_label.setText("\n".join(warnings))
        # 没有区域时不让确定：否则会「成功但 0 框」，覆盖模式会删掉原有自动标注
        self.ok_btn.setEnabled(not (needs_region and not self._region_count))

    def _edit_windows(self):
        from .sliding_window_dialog import SlidingWindowDialog

        dialog = SlidingWindowDialog(reference_size=self._reference,
                                     spec=self._windows, parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._windows = dialog.spec()
        self._refresh()

    def retranslate(self):
        """切语言后刷新面板文字。"""
        self.setWindowTitle(tr("推理范围"))
        for mode, name, hint in _MODE_KEYS:
            self._buttons[mode].setText(tr(name))
        self.ok_btn.setText(tr("确定"))
        self.cancel_btn.setText(tr("取消"))
        self.region_btn.setText(tr("编辑区域"))
        self._refresh()
