"""推理范围面板：全图 / 滑窗 / 区域。

滑窗参数**内嵌在本面板里**（宽/高/重叠 + 原图网格预览 + 模板），点「滑窗」
直接就能调参、直接能看到窗口落在原图哪里，不再叠开第二个窗口。
区域沿用画布上的临时区域（`region_boxes`），「编辑区域」直接在背景图上画。

面板是**非模态**的：弹窗开着时用户仍要点画布画区域，模态会吃掉发往画布的
鼠标事件（Qt 只把输入送给模态窗及其子窗口），那样「点编辑区域直接在图上画」
根本用不了。面板只负责收集选择，校验与快照由 `mixins/inference.py` 在启动
任务前完成。
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
from .widgets.window_params import WindowParamsWidget

_MODE_KEYS = (
    (MODE_FULL, "全图", "整张图推理一次"),
    (MODE_WINDOW, "滑窗", "按窗口切分后逐窗推理，结果合并去重"),
    (MODE_ROI, "区域", "只在画布上画好的区域内推理"),
)


class ScopePanel(QDialog):
    """范围选择；确定后 `selection()` 返回 (mode, windows)。"""

    edit_regions_requested = pyqtSignal()
    # 确定/取消/直接关闭都要让宿主收尾（结束区域绘制会话）。非模态后不再有
    # `exec_()` 的返回值可看，改由信号通知，每条关闭路径都会发。
    finished_ok = pyqtSignal(bool)

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
        # 非模态：`exec_()` 的应用级模态会吃掉发往画布的鼠标事件，使
        # 「点编辑区域直接在背景图上画」不可用。
        self.setModal(False)

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

        # 滑窗参数内嵌（只有滑窗模式可见）：点「滑窗」就在同一处调参，
        # 不再弹第二个窗口。
        self.window_params = WindowParamsWidget(
            reference_size=self._reference, spec=self._windows)
        layout.addWidget(self.window_params)

        row = QHBoxLayout()
        self.region_label = QLabel("")
        self.region_label.setObjectName("scopeHint")
        row.addWidget(self.region_label)
        row.addStretch()
        self.region_btn = QPushButton(tr("编辑区域"))
        self.region_btn.setObjectName("optionsBtn")
        self.region_btn.setToolTip(tr("在图上拖动即可添加区域，不切换模式"))
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

        # 参数变化要重算「窗口数非法就禁用确定」
        self.window_params.changed.connect(self._refresh)

    def done(self, result):
        """所有关闭路径（确定/取消/直接关闭）都经此：先收尾再发信号。"""
        super().done(result)
        self.finished_ok.emit(result == QDialog.Accepted)

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
        return dict(self._windows) if self._windows else {}

    def set_windows(self, windows):
        self._windows = dict(windows or {})
        self.window_params.set_spec(self._windows)
        self._refresh()

    def set_region_count(self, count):
        self._region_count = int(count or 0)
        self._refresh()

    def set_reference(self, size):
        """参考图尺寸：窗口数预估要用它；没有参考图时预估显示为不可用。"""
        self._reference = size
        self.window_params.set_reference(size)

    def set_background_pixmap(self, pixmap):
        """当前背景图：内嵌预览把它画在网格底下（原图上叠加）。"""
        self.window_params.set_pixmap(pixmap)

    def refresh_templates(self):
        """重新读取共用模板（数据增强那边改过模板后要能刷到）。"""
        self.window_params.refresh_templates()

    def window_params_widget(self):
        """内嵌的滑窗参数控件（测试与调用方取用）。"""
        return self.window_params

    def set_anchor_rect(self, rect):
        """右侧侧栏那块位置；None 表示居中显示（拿不到侧栏时）。"""
        self._anchor_rect = rect

    def selection(self):
        """(mode, windows)；窗口参数只在滑窗模式返回。"""
        mode = self.mode()
        if mode == MODE_WINDOW:
            return mode, dict(self.window_params.spec())
        return mode, None

    def window_summary(self):
        """窗口参数摘要，如「640×640 重叠 200」；无参数时返回空串。"""
        spec = self.window_params.spec()
        if not spec:
            return ""
        return (f"{spec.get('w')}×{spec.get('h')} "
                f"{tr('重叠')} {spec.get('overlap')}")

    # ---------- 内部 ----------

    def _refresh(self, _checked=None):
        mode = self.mode()
        needs_window = mode == MODE_WINDOW
        needs_region = mode == MODE_ROI
        # 参数控件整个跟着模式显隐：全图/区域时它没有意义，留着只会让人
        # 以为「这些参数也在生效」。
        self.window_params.setVisible(needs_window)
        self.region_btn.setVisible(needs_region)
        self.region_label.setVisible(needs_region)
        self.region_label.setText(f"{self._region_count} {tr('个区域')}")

        warnings = []
        if needs_region and not self._region_count:
            warnings.append(tr("没有可用区域，请先画好区域"))
        if needs_region:
            warnings.append(tr("区域推理只保留框中心落在区域内的结果"))
        self.warn_label.setText("\n".join(warnings))
        # 没有区域、或滑窗参数非法时不让确定：否则会「成功但 0 框」，
        # 覆盖模式会删掉原有自动标注。
        valid = True
        if needs_region and not self._region_count:
            valid = False
        if needs_window and not self.window_params.is_valid():
            valid = False
        self.ok_btn.setEnabled(valid)

    def retranslate(self):
        """切语言后刷新面板文字。"""
        self.setWindowTitle(tr("推理范围"))
        for mode, name, hint in _MODE_KEYS:
            self._buttons[mode].setText(tr(name))
        self.ok_btn.setText(tr("确定"))
        self.cancel_btn.setText(tr("取消"))
        self.region_btn.setText(tr("编辑区域"))
        self.region_btn.setToolTip(tr("在图上拖动即可添加区域，不切换模式"))
        self.window_params.retranslate()
        self._refresh()
