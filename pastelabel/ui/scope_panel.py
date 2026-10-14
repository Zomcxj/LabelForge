"""推理范围面板：全图 / 滑窗 / 区域。

滑窗参数**内嵌在本面板里**（宽/高/重叠 + 窗口数预估 + 模板按钮），点「滑窗」
直接就能调参。但窗口网格**不在面板里画**——它直接叠画在编辑区的**主画布**上
（用户要求「直接在画布展示，不要单独小窗显示全图」）；区域框同理。本面板只
发 `overlay_changed` 让宿主重画，面板自身没有任何缩略图。

区域沿用画布上的临时区域（`region_boxes`），「编辑区域」直接在背景图上画。

**只有选中「滑窗」时滑窗参数才可用**，选中「区域」时同理只有区域相关可用；
未选中的那一组禁用且隐藏（用户要求「未选中滑窗时不能修改点击」）。

面板是**非模态**的：弹窗开着时用户仍要点画布画区域，模态会吃掉发往画布的
鼠标事件（Qt 只把输入送给模态窗及其子窗口），那样「点编辑区域直接在图上画」
根本用不了。面板只负责收集选择，校验与快照由 `mixins/inference.py` 在启动
任务前完成。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QRadioButton, QVBoxLayout, QWidget,
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
    # 面板状态变了（切模式/改窗口参数/区域数变化）→ 宿主重画主画布上的
    # 网格与区域轮廓。网格不再有面板内的小预览图，只能靠这条信号驱动。
    overlay_changed = pyqtSignal()
    # 点了某条区域右侧的删除：带区域下标回宿主（由宿主删真身并刷新画布）。
    # 面板不直接改 `region_boxes`——它拿到的只是一份用于显示的拷贝，
    # 自己删改会和画布上的真身脱节。
    region_delete_requested = pyqtSignal(int)

    def __init__(self, windows=None, mode=MODE_FULL, region_count=0,
                 reference_size=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("推理范围"))
        self.setMinimumWidth(440)
        self._windows = dict(windows or {})
        self._region_count = int(region_count or 0)
        self._regions = []
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
            button.toggled.connect(self._on_mode_toggled)
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

        # 已画区域的明细：每条显示坐标 + 尺寸，右侧带删除按钮。用户要求
        # 「区域画完后在推理范围窗口显示该框的坐标 加一个删除按钮」——
        # 之前这里只有一个「N 个区域」的计数，看不出框在哪，也没法删。
        self.region_list = QListWidget()
        self.region_list.setObjectName("scopeRegionList")
        self.region_list.setMaximumHeight(110)
        self.region_list.setSelectionMode(QListWidget.NoSelection)
        self.region_list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.region_list)

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
            QListWidget#scopeRegionList {{
                background-color: {theme['panel_bg']};
                color: {theme['text_primary']};
                border: 1px solid {theme['border_color']};
                border-radius: 4px;
            }}
            QLabel#scopeRegionText {{ color: {theme['text_primary']}; }}
            /* 每条区域右侧的删除按钮：做小一点，别和底部的确定/取消抢视觉 */
            QPushButton#scopeRegionDelete {{
                padding: 1px 8px; font-size: 11px;
            }}
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

    def _on_mode_toggled(self, _checked=False):
        """把三个模式按钮维持成「单选」。

        不能依赖 QRadioButton 的自动互斥：它只在**同一个父控件**内生效，而这里
        每个按钮各自放进一行布局、父控件不同，于是多个按钮能同时选中，
        `mode()` 只会返回字典序里第一个选中项（永远是「全图」），切到滑窗/区域
        会静默失效。

        也不用 QButtonGroup：本项目的 PyQt5 在某些导入路径下 `QtWidgets` 是
        半初始化模块（`QButtonGroup` 取不到），显式写反而更稳。
        """
        checked = [m for m, b in self._buttons.items() if b.isChecked()]
        # 用户可能把当前项再点一次（QRadioButton 默认不允许取消，这里兜底），
        # 保证任何时候恰好有一个选中项。
        if not checked:
            self._buttons[MODE_FULL].setChecked(True)
            return
        if len(checked) > 1:
            for mode in checked[1:]:
                button = self._buttons[mode]
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
        self._refresh()

    def set_mode(self, mode):
        button = self._buttons.get(mode) or self._buttons[MODE_FULL]
        # 先全部清掉再选目标：只 setChecked(True) 无法取消别的已选项。
        for other in self._buttons.values():
            if other is not button and other.isChecked():
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
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

    def regions(self):
        """当前用于显示的区域快照（宿主传入的那一份，不含删除操作）。"""
        return list(getattr(self, '_regions', []))

    def set_regions(self, regions):
        """填充区域明细列表（坐标 + 删除按钮）。

        入参是 `region_boxes` 的**快照副本**：面板只用来显示，删改一律经
        `region_delete_requested` 回宿主，避免面板持有列表引用后自己 pop，
        和画布上的真身对不上。坐标取 `region_bbox()` 的外接矩形——多边形
        只有一个点集，直接显示顶点第一个点会误导人。
        """
        self._regions = list(regions or [])
        self._region_count = len(self._regions)
        self._rebuild_region_list()
        self._refresh()

    def _rebuild_region_list(self):
        self.region_list.clear()
        for index, region in enumerate(getattr(self, '_regions', [])):
            item = QListWidgetItem()
            self.region_list.addItem(item)
            widget = self._build_region_row(index, region)
            item.setSizeHint(widget.sizeHint())
            self.region_list.setItemWidget(item, widget)

    def _build_region_row(self, index, region):
        """单条区域：`#N  (x, y) 宽×高` + 删除按钮。"""
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(2, 1, 2, 1)
        line.setSpacing(6)

        text = QLabel(f"#{index + 1}  {_region_text(region)}")
        text.setObjectName("scopeRegionText")
        line.addWidget(text)
        line.addStretch()

        delete_btn = QPushButton(tr("删除"))
        delete_btn.setObjectName("scopeRegionDelete")
        delete_btn.setToolTip(tr("删除这个区域"))
        # 用默认参数绑住 index：闭包直接引用循环变量的话，所有按钮都会删最后一条。
        delete_btn.clicked.connect(
            lambda _checked=False, i=index: self.region_delete_requested.emit(i))
        line.addWidget(delete_btn)
        return row

    def set_reference(self, size):
        """参考图尺寸：窗口数预估要用它；没有参考图时预估显示为不可用。"""
        self._reference = size
        self.window_params.set_reference(size)

    def window_params_widget(self):
        """内嵌的滑窗参数控件（测试与调用方取用）。"""
        return self.window_params

    def set_anchor_rect(self, rect):
        """面板竖直位置要对齐的那块区域（由宿主按两个列表算出来）。

        用户澄清「上面是层级不是位置」：面板在层级/顺序上要落在背景图列表
        与标签列表**中间**，所以这里收到的是那两个列表之间的那条带，而不是
        整个侧栏——只对齐侧栏顶部会一直盖住背景图列表。
        """
        self._anchor_rect = rect

    def window_spec(self):
        """当前窗口参数（引擎口径）。网格绘制要用它，与是否选中滑窗无关。"""
        return dict(self.window_params.spec())

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
        # 参数控件**始终可见**：用户要求「打开窗口它就要存在」，一打开面板就能
        # 看到这次用的是多大的窗口、有几个模板，不必先切到滑窗再回来核对。
        self.window_params.setVisible(True)
        # 但只有选中「滑窗」时才可编辑：用户要求「未选中滑窗就无法编辑」。
        # 显示但置灰——既能看到当前值，又点不动、改不了。
        self.window_params.set_editable(needs_window)
        self.region_btn.setVisible(needs_region)
        self.region_label.setVisible(needs_region)
        self.region_label.setText(f"{self._region_count} {tr('个区域')}")
        # 区域明细只在选中「区域」时显示（与按钮同步）
        self.region_list.setVisible(needs_region)
        # 区域按钮也只有在选中「区域」时可用（与滑窗参数对称）
        self.region_btn.setEnabled(needs_region)

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
        # 网格/区域画在主画布上，由宿主监听这个信号重画
        self.overlay_changed.emit()

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
        # 明细行里的「删除」也是 tr() 出来的，切语言要重建才跟着变
        self._rebuild_region_list()
        self._refresh()


def _region_text(region):
    """区域的一行文字：`(x, y) 宽×高`（外接矩形，多边形也适用）。

    取整显示：ROI 坐标是浮点（缩放换算得来），界面上小数点后一串没有意义，
    反而挤得看不清。取不到外接矩形（空/退化区域）时退化成「(未知)」，
    不让一行坏数据把整个面板构造炸掉。
    """
    from ..engine.inference_scope import region_bbox

    try:
        x1, y1, x2, y2 = region_bbox(region)
    except Exception:
        return tr("(未知)")
    return (f"({int(round(x1))}, {int(round(y1))}) "
            f"{int(round(x2 - x1))}×{int(round(y2 - y1))}")
