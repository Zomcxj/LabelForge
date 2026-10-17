"""推理范围面板：全图 / 滑窗 / 区域。

滑窗参数**内嵌在本面板里**（宽/高/重叠 + 窗口数预估），点「滑窗」
直接就能调参。但窗口网格**不在面板里画**——它直接叠画在编辑区的**主画布**上
（用户要求「直接在画布展示，不要单独小窗显示全图」）；区域框同理。本面板只
发 `overlay_changed` 让宿主重画，面板自身没有任何缩略图。

区域沿用画布上的临时区域（`region_boxes`），「编辑区域」直接在背景图上画。

**只有选中「滑窗」时滑窗参数才可用**，选中「区域」时同理只有区域相关可用；
未选中的那一组**置灰但不隐藏**——用户要的是「未选中滑窗时不能修改点击」，
同时还能一眼看到另外两种模式当前的参数与区域，不必来回切换核对。

面板是**非模态**的：弹窗开着时用户仍要点画布画区域，模态会吃掉发往画布的
鼠标事件（Qt 只把输入送给模态窗及其子窗口），那样「点编辑区域直接在图上画」
根本用不了。面板只负责收集选择，校验与快照由 `mixins/inference.py` 在启动
任务前完成。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QRadioButton, QVBoxLayout, QWidget,
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
    """范围选择；`selection()` 返回 (mode, windows)。

    面板是**实时**的：没有确定/取消按钮，任何改动都立刻经 `overlay_changed`
    写回宿主；关窗（右上角 X）不回滚，只通知宿主收尾。
    """

    # 关闭（只剩直接关这一条路径）时让宿主收尾：结束区域绘制会话、撤掉画布
    # 上的范围提示。非模态后不再有 `exec_()` 的返回值可看，改由信号通知。
    finished_ok = Signal(bool)
    # 面板状态变了（切模式/改窗口参数/区域数变化）→ 宿主重画主画布上的
    # 网格与区域轮廓。网格不再有面板内的小预览图，只能靠这条信号驱动。
    overlay_changed = Signal()
    # 点了某条区域右侧的删除：带区域下标回宿主（由宿主删真身并刷新画布）。
    # 面板不直接改 `region_boxes`——它拿到的只是一份用于显示的拷贝，
    # 自己删改会和画布上的真身脱节。
    region_delete_requested = Signal(int)
    # 点了「添加区域」：请宿主在画布上进入一次性绘制态（点一下画一个）。
    # 面板自己不画——绘制要拿画布坐标、还要临时框预览，只有画布能做。
    add_region_requested = Signal()
    # 某条区域的锁被切换：(区域下标, 是否锁定)。同样回宿主落到真身上。
    region_lock_toggled = Signal(int, bool)

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

        # 三段（全图/滑窗/区域）都在**同一个窗口**里，不拆弹窗；但各自有自己
        # 的分组框，选中的那段可编辑、其余置灰。用户要求「还在一个窗口，补全
        # 内容，分割一下」+「全部显示各自内容」——所以用置灰而不是隐藏，用户
        # 一眼能看到另外两种模式当前的参数/区域，不必来回切换核对。
        #
        # 每段拆成「标题行 + 内容体」两部分：`setEnabled` 只作用于**内容体**。
        # 整段禁用会连标题行里的单选按钮一起点不动，用户就切不过去了
        # （整段禁用时点别的模式没反应）——所以按钮必须一直可点。
        self._buttons = {}
        self.sections = {}
        self._section_bodies = {}
        self._hints = {}

        for mode, name, hint in _MODE_KEYS:
            section = QGroupBox(tr(name))
            section.setObjectName("scopeSection")
            section_layout = QVBoxLayout(section)
            section_layout.setSpacing(6)

            row = QHBoxLayout()
            button = QRadioButton(tr(name))
            button.setObjectName("scopeMode")
            button.toggled.connect(self._on_mode_toggled)
            self._buttons[mode] = button
            row.addWidget(button)
            label = QLabel(tr(hint))
            label.setObjectName("scopeHint")
            self._hints[mode] = label
            row.addWidget(label)
            row.addStretch()
            section_layout.addLayout(row)

            body = QWidget()
            body.setObjectName("scopeSectionBody")
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(0, 0, 0, 0)
            body_layout.setSpacing(6)
            section_layout.addWidget(body)
            self._section_bodies[mode] = body

            self.sections[mode] = section
            layout.addWidget(section)

        # 滑窗参数内嵌（不再弹第二个窗口）：点「滑窗」就在同一处调参。
        self.window_params = WindowParamsWidget(
            reference_size=self._reference, spec=self._windows)
        self._section_bodies[MODE_WINDOW].layout().addWidget(self.window_params)

        # 区域段改成**按钮驱动**：用户要求「画区域也加一个按钮啊 点一下画一个」。
        # 原先的「选中『区域』就自动进入可编辑态」要拆掉——那个自动进入的会话
        # 会独占左键，标签框就再也点不到了；现在只有点了这个按钮，下一次在图上
        # 拖动才画区域，画完立刻退出，其余时间画布与平时完全一样。
        row = QHBoxLayout()
        self.add_region_btn = QPushButton(tr("添加区域"))
        self.add_region_btn.setObjectName("scopeAddRegion")
        self.add_region_btn.setToolTip(tr("在图上拖动即可添加区域"))
        self.add_region_btn.clicked.connect(self.add_region_requested.emit)
        row.addWidget(self.add_region_btn)
        self.region_label = QLabel("")
        self.region_label.setObjectName("scopeHint")
        row.addWidget(self.region_label)
        row.addStretch()
        self._section_bodies[MODE_ROI].layout().addLayout(row)

        # 已画区域的明细：每条显示坐标 + 尺寸，右侧带删除按钮。用户要求
        # 「区域画完后在推理范围窗口显示该框的坐标 加一个删除按钮」——
        # 之前这里只有一个「N 个区域」的计数，看不出框在哪，也没法删。
        self.region_list = QListWidget()
        self.region_list.setObjectName("scopeRegionList")
        self.region_list.setMaximumHeight(110)
        self.region_list.setSelectionMode(QListWidget.NoSelection)
        self.region_list.setFocusPolicy(Qt.NoFocus)
        self._section_bodies[MODE_ROI].layout().addWidget(self.region_list)

        self.warn_label = QLabel("")
        self.warn_label.setObjectName("scopeWarn")
        self.warn_label.setWordWrap(True)
        layout.addWidget(self.warn_label)

        # 用户要求「选中某推理方式立马刷新不需要点击确定，可以把确定取消
        # 按钮删除」：面板不再是「编辑-确认」的两段式，改成纯实时面板。
        # 关窗只剩右上角 X，且**关窗不回滚**——所有改动都已在 `_refresh`
        # 里经 `overlay_changed` 即时写回宿主。

        # 参数变化要重算窗口数与提示，并把新值即时写回宿主
        self.window_params.changed.connect(self._refresh)

    def done(self, result):
        """关闭面板（只剩右上角 X 这条路径）时收尾并发信号。

        没有「确定/取消」了，`result` 只可能是 `QDialog.Rejected`；保留这个
        覆写是因为 `QDialog.close()` 也走 `done()`，宿主靠 `finished_ok` 得知
        面板没了（要收区域会话、撤掉画布上的范围提示）。
        """
        super().done(result)
        self.finished_ok.emit(result == QDialog.Accepted)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['widget_bg']}; color: {theme['text_primary']}; }}
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#scopeHint {{ color: {theme['text_secondary']}; }}
            QLabel#scopeWarn {{ color: {theme['accent']}; }}
            /* 三段分组框：一眼分得清全图/滑窗/区域各自的内容。置灰时
               Qt 会自动把标题和内容一起变淡，不需要额外的禁用配色。 */
            QGroupBox#scopeSection {{
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                margin-top: 8px; padding: 8px 6px 6px 6px;
                color: {theme['text_primary']};
            }}
            QGroupBox#scopeSection::title {{
                subcontrol-origin: margin; left: 8px; padding: 0 3px;
                color: {theme['text_secondary']};
            }}
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
            /* 每条区域右侧的锁定/删除按钮：做小一点，别抢视觉 */
            QPushButton#scopeRegionDelete, QPushButton#scopeRegionLock {{
                padding: 1px 8px; font-size: 11px;
            }}
            QPushButton#scopeAddRegion {{
                padding: 2px 10px;
            }}
        """)

    def refresh_theme(self):
        self._apply_style()
        sync_titlebar(self)

    def showEvent(self, event):
        super().showEvent(event)
        # 左上角对齐锚点（=「标签管理」区左上角），用户要求「始终对齐」。
        place_within(self, self._anchor_rect, align="top-left")
        sync_titlebar(self)

    # ---------- 对外接口 ----------

    def mode(self):
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return MODE_FULL

    def _on_mode_toggled(self, checked=False):
        """把三个模式按钮维持成「单选」。

        不能依赖 QRadioButton 的自动互斥：它只在**同一个父控件**内生效。原先
        三个按钮各占一行、父控件不同就已经不自动互斥了；改成三段分组框后更是
        各在各自的 `QGroupBox` 里，互斥彻底失效。

        也不用 QButtonGroup：测试桩（`tests/conftest.py` 的假 QtWidgets）没注册
        `QButtonGroup`，所以显式维护三个按钮的选中状态。

        这里**不能**按「保留 checked[0]、清掉其余」来写：`checked` 是按字典序
        （full, window, roi）收集的，从「滑窗」点「区域」时得到
        `['window', 'roi']`，清掉 `checked[1:]` 正好把用户刚点的那个清掉，
        表现为「点了没反应」。所以要以**信号带进来的那个按钮**为准：
        它 `checked=True` 时就保留它、取消其余；它变回 `False`（用户把当前项
        又点了一下）时兜底回「全图」，保证任何时候恰好有一个选中项。
        """
        source = self.sender()
        if checked and source is not None:
            self._keep_only(source)
            return
        # 没有 source（直接调用、测试里模拟）或按钮被取消选中：按字典序兜底
        if not checked and source is None:
            self._keep_only(self._buttons.get(self._selected_mode())
                            or self._buttons[MODE_FULL])
            return
        self._keep_only(self._buttons[MODE_FULL])

    def _selected_mode(self):
        """当前选中的模式（按字典序取第一个选中项）。"""
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return None

    def _keep_only(self, button):
        """只让 `button` 保持选中，其余一律取消。"""
        for other in self._buttons.values():
            if other is button or not other.isChecked():
                continue
            other.blockSignals(True)
            other.setChecked(False)
            other.blockSignals(False)
        if not button.isChecked():
            button.setChecked(True)
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
        # 行控件是 `setItemWidget` 挂上去的，`clear()` **不会**回收它：不显式
        # 收走的话旧行会一直挂在面板上（切语言重建几轮就攒一堆，而且它们还能
        # 被 findChildren 找到、文本停在上一门语言）。所以先把旧行摘下来。
        for index in range(self.region_list.count()):
            old = self.region_list.itemWidget(self.region_list.item(index))
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        self.region_list.clear()
        for index, region in enumerate(getattr(self, '_regions', [])):
            item = QListWidgetItem()
            self.region_list.addItem(item)
            widget = self._build_region_row(index, region)
            item.setSizeHint(widget.sizeHint())
            self.region_list.setItemWidget(item, widget)

    def _build_region_row(self, index, region):
        """单条区域：`#N  (x, y) 宽×高` + 锁按钮 + 删除按钮。

        锁按钮是**每条各自**的（用户要求「区域列表增加锁定按钮，不锁定默认
        最上层与检测框一样可修改，锁定后最底层不可修改」）。按钮文字直接显示
        当前状态的反操作，比一个勾选框更直白：写着「锁定」就是还能锁。
        """
        row = QWidget()
        line = QHBoxLayout(row)
        line.setContentsMargins(2, 1, 2, 1)
        line.setSpacing(6)

        locked = bool(region.get('locked', False))
        text = QLabel(f"#{index + 1}  {_region_text(region)}")
        text.setObjectName("scopeRegionText")
        line.addWidget(text)
        line.addStretch()

        lock_btn = QPushButton(tr("解锁") if locked else tr("锁定"))
        lock_btn.setObjectName("scopeRegionLock")
        lock_btn.setToolTip(tr("解锁这个区域（放回检测框上层，可以修改）") if locked
                            else tr("锁定这个区域（放到检测框下层，不能修改）"))
        # 用默认参数绑住 index：闭包直接引用循环变量的话，所有按钮都会改最后一条。
        lock_btn.clicked.connect(
            lambda _checked=False, i=index, want=not locked:
            self.region_lock_toggled.emit(i, want))
        line.addWidget(lock_btn)

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
        # 三段始终可见、各自显示自己的内容（用户要求「全部显示各自内容」），
        # 只有选中那段可编辑、其余置灰——置灰既能看到当前值又改不动。
        for _mode, body in self._section_bodies.items():
            body.setEnabled(_mode == mode)
        # 组框的 `setEnabled(False)` 会连带禁用里面的输入框，所以滑窗参数
        # 不必再单独 `set_editable`：未选中滑窗时那段的内容体就是灰的。
        self.window_params.set_editable(needs_window)
        self.region_label.setText(f"{self._region_count} {tr('个区域')}")

        warnings = []
        if needs_region and not self._region_count:
            warnings.append(tr("没有可用区域，请先画好区域"))
        if needs_window and not self.window_params.is_valid():
            warnings.append(tr("窗口参数非法，请调整"))
        if needs_region:
            warnings.append(tr("区域推理只保留框中心落在区域内的结果"))
        self.warn_label.setText("\n".join(warnings))

        # 网格/区域画在主画布上，由宿主监听这个信号重画。**这条信号也是
        # 「即时生效」的唯一通道**：宿主在 `_on_scope_overlay_changed` 里把
        # 模式与窗口参数写回自己的属性（原来这些只在点「确定」时发生）。
        # 所以它必须在每次状态变化后都发出去，不能只在参数变化时发。
        self.overlay_changed.emit()

    def retranslate(self):
        """切语言后刷新面板文字。"""
        self.setWindowTitle(tr("推理范围"))
        for mode, name, hint in _MODE_KEYS:
            self._buttons[mode].setText(tr(name))
            self._hints[mode].setText(tr(hint))
            # 段标题也要跟着切，否则中文界面下段标题还是上一门语言
            self.sections[mode].setTitle(tr(name))
        self.window_params.retranslate()
        self.add_region_btn.setText(tr("添加区域"))
        self.add_region_btn.setToolTip(tr("在图上拖动即可添加区域"))
        # 明细行里的「锁定/解锁」「删除」也是 tr() 出来的，切语言要重建才跟着变
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
