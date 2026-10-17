"""画布上方的模型行：加载按钮、模型名、阈值、输入尺寸、覆盖/补充与关闭。

工具栏第一个图标负责显示/隐藏这一行；关闭后两个推理图标会禁用，重新点开即恢复。
阈值与尺寸改动即时写回 detector，不重建会话，所以调参不用重新加载模型。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QToolButton,
)

from .i18n import t as tr
from .theme import ThemeManager

# 置信度 / NMS 两个输入框的宽度。原来 72px，按「减少 1/5」收到 58px——
# 两位小数（0.25）在这个宽度里还看得全，参数行则宽松不少。
_THRESHOLD_SPIN_WIDTH = 58


class ModelRow(QWidget):
    """模型信息行；控件改动通过信号交给主窗口写回 detector。"""

    thresholds_changed = Signal(float, float)
    input_size_changed = Signal(int, int)
    replace_mode_changed = Signal(bool)
    load_requested = Signal()
    filter_requested = Signal()
    scope_requested = Signal()
    batch_to_changed = Signal(int)
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modelRow")
        self._end2end = False
        self._build_ui()
        self._apply_style()
        self.clear_model()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        # 顺序与视频抽帧行一致：先按钮、后名字
        self.load_btn = QPushButton(tr("加载模型"))
        self.load_btn.setObjectName("optionsBtn")
        self.load_btn.setFixedHeight(24)
        self.load_btn.setToolTip(tr("加载 ONNX 模型"))
        self.load_btn.clicked.connect(self.load_requested.emit)
        layout.addWidget(self.load_btn)

        self.name_label = QLabel("")
        self.name_label.setObjectName("modelRowName")
        self._model_name = ""
        layout.addWidget(self.name_label)

        # 功能之间用 '-' 隔开（用户要求）：7 组 = 加载模型+模型名 | 置信度+NMS
        # | 输入尺寸 | 覆盖/补充 | 类别筛选 | 推理范围+摘要 | 目标张号。
        # 置信度与 NMS 同组，中间**不**隔。
        self._add_param_separator(layout)

        self.conf_lbl = QLabel(tr("conf"))
        layout.addWidget(self.conf_lbl)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        # 步长 0.01：阈值是逐点精调的旋钮，0.05 跳得太粗，微调时够不着想要的值
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(_THRESHOLD_SPIN_WIDTH)
        self.conf_spin.valueChanged.connect(self._emit_thresholds)
        layout.addWidget(self.conf_spin)

        self.iou_lbl = QLabel(tr("NMS"))
        layout.addWidget(self.iou_lbl)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setFixedWidth(_THRESHOLD_SPIN_WIDTH)
        self.iou_spin.valueChanged.connect(self._emit_thresholds)
        layout.addWidget(self.iou_spin)

        self._size_sep_before = self._add_param_separator(layout)

        # 输入尺寸只对**动态轴**模型有意义（静态轴模型的尺寸由模型自身决定）。
        # 静态模型下整组控件（标签 + 两个输入框 + "x"）直接不显示，而不是
        # 「显示但禁用」——用户要求「静态模型直接不加载，动态的才显示」，
        # 摆一组改不动的控件只会让人反复去点它。
        self.size_lbl = QLabel(tr("输入尺寸"))
        layout.addWidget(self.size_lbl)
        self.size_spin_w = QSpinBox()
        self.size_spin_w.setRange(32, 8192)
        # 与 conf / NMS 用同一个宽度常量：它们在同一行里并排，宽度不一会显得
        # 参差不齐（用户要求「输入尺寸的按钮宽度和 conf、NMS 保持一致」）。
        self.size_spin_w.setFixedWidth(_THRESHOLD_SPIN_WIDTH)
        self.size_spin_w.setToolTip(tr("动态轴模型需要指定；静态轴模型按模型自身尺寸推理"))
        self.size_spin_w.valueChanged.connect(self._emit_size)
        layout.addWidget(self.size_spin_w)
        self.size_sep_label = QLabel("x")
        layout.addWidget(self.size_sep_label)
        self.size_spin_h = QSpinBox()
        self.size_spin_h.setRange(32, 8192)
        self.size_spin_h.setFixedWidth(_THRESHOLD_SPIN_WIDTH)
        self.size_spin_h.valueChanged.connect(self._emit_size)
        layout.addWidget(self.size_spin_h)

        self._size_sep_after = self._add_param_separator(layout)

        # 与其他工具栏按钮同一套样式（optionsBtn），选中态由主题的 :checked 规则给出
        self.replace_btn = QPushButton()
        self.replace_btn.setObjectName("optionsBtn")
        self.replace_btn.setCheckable(True)
        # 默认覆盖：推理结果直接替换上一轮，不叠加。这个选择不写配置，
        # 每次启动都回到覆盖，避免上次改过补充的用户拿到叠出来的结果。
        self.replace_btn.setChecked(True)
        self.replace_btn.setFixedHeight(24)
        self.replace_btn.setToolTip(tr("覆盖：替换上一轮推理框；补充：保留并追加"))
        self.replace_btn.clicked.connect(self._on_replace_toggled)
        layout.addWidget(self.replace_btn)

        self._add_param_separator(layout)

        self.filter_btn = QPushButton(tr("类别筛选"))
        self.filter_btn.setObjectName("optionsBtn")
        self.filter_btn.setFixedHeight(24)
        self.filter_btn.setEnabled(False)
        self.filter_btn.clicked.connect(self.filter_requested.emit)
        layout.addWidget(self.filter_btn)

        self._add_param_separator(layout)

        self.scope_btn = QPushButton(tr("推理范围"))
        self.scope_btn.setObjectName("optionsBtn")
        self.scope_btn.setFixedHeight(24)
        self.scope_btn.setEnabled(False)
        self.scope_btn.setToolTip(tr("全图 / 滑窗 / 区域"))
        self.scope_btn.clicked.connect(self.scope_requested.emit)
        layout.addWidget(self.scope_btn)

        # 摘要放按钮右侧：按钮管「选哪种范围」，这里管「现在是哪种」。
        # 必须在 stretch 之前紧跟按钮，否则会被推到行尾。
        self.scope_summary_label = QLabel("")
        self.scope_summary_label.setObjectName("modelRowScope")
        layout.addWidget(self.scope_summary_label)

        # 批量的「张数范围」放在推理范围右侧（用户要求）。语义是**结束张号**：
        # 当前图是第 200 张时填 300 → 跑 200~300；填 100 → 跑 100~200。所以
        # 它表达的是「从当前图到第 N 张」这条区间，方向由两边大小自动决定。
        #
        # 前缀名加在输入框**自己左边**：用户说的是「范围推理右侧的序号加一个
        # 前缀名」——即推理范围按钮右侧那个序号，光一个数字看不出是什么。
        self._add_param_separator(layout)
        self.batch_to_label = QLabel(tr("目标张号"))
        layout.addWidget(self.batch_to_label)
        self.batch_to_spin = QSpinBox()
        self.batch_to_spin.setObjectName("modelRowBatchTo")
        self.batch_to_spin.setFixedHeight(24)
        self.batch_to_spin.setFixedWidth(63)
        self.batch_to_spin.setRange(0, 999999)
        self.batch_to_spin.setToolTip(
            tr("一键推理的目标张号（含）：当前第 200 张时填 300 跑 200~300，"
               "填 100 跑 100~200；0 表示不限制、跑到末尾"))
        self.batch_to_spin.setKeyboardTracking(False)
        self.batch_to_spin.valueChanged.connect(self._emit_batch_to)
        # 用户是否手动填过张号；填过就不再被「默认末张」覆盖。
        self._batch_to_user_set = False
        layout.addWidget(self.batch_to_spin)

        layout.addStretch()

        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setObjectName("modelRowClose")
        self.close_btn.setToolTip(tr("关闭参数行（关闭后不能推理）"))
        self.close_btn.clicked.connect(self._on_close)
        layout.addWidget(self.close_btn)

        self._refresh_replace_text()

    def _add_param_separator(self, layout):
        """插入一个 `-` 分隔符（参数行各功能组之间用）。

        用 QLabel 而不是 `addSpacing`：分隔符要**看得见**，纯间距只是空白。
        颜色由 `#modelRow QLabel` 的 `text_secondary` 给出，明暗主题都可读。
        """
        sep = QLabel("-")
        sep.setObjectName("modelRowSep")
        layout.addWidget(sep)
        return sep

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            #modelRow {{
                background-color: {theme['panel_bg']};
                border-bottom: 1px solid {theme['border_color']};
            }}
            #modelRow QLabel {{ color: {theme['text_secondary']}; }}
            #modelRowName {{ color: {theme['text_primary']}; font-weight: bold; }}
            #modelRowScope {{ color: {theme['accent']}; font-weight: bold; }}
            #modelRow QDoubleSpinBox, #modelRow QSpinBox {{
                background-color: {theme['widget_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                padding: 1px 4px;
            }}
            #modelRowClose {{
                background: transparent; border: none;
                color: {theme['text_secondary']}; font-size: 16px;
            }}
            #modelRowClose:hover {{ color: {theme['text_primary']}; }}
        """)

    # ---------- 对外接口 ----------

    def refresh_theme(self):
        """主窗口切换主题后重新上色（自带样式表不会随全局样式刷新）。"""
        self._apply_style()

    def set_model(self, name, conf, iou, size, active_providers=None, dynamic=False,
                  end2end=False):
        """加载完成后填入模型信息。

        **动态轴**模型的尺寸必须由用户指定，所以整组「输入尺寸」控件显示且可编辑；
        **静态轴**模型的尺寸由模型自身决定，整组直接**不显示**（用户要求「静态
        模型直接不加载，动态的才显示」）——摆一组改不动的输入框只会让人反复去点它。

        端到端模型（v26）导出时已内置 NMS，界面上的 NMS 值改了不起作用，
        因此禁用并标注，免得用户以为调了没反应。
        """
        self.name_label.setText(name)
        self._model_name = name
        self._end2end = bool(end2end)
        self.name_label.setToolTip(self._tooltip(name, size, active_providers))
        for widget, value in ((self.conf_spin, conf), (self.iou_spin, iou)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.conf_spin.setEnabled(True)
        self.iou_spin.setEnabled(not self._end2end)
        self.iou_spin.setToolTip(
            tr("该模型已内置 NMS，此项不生效") if self._end2end
            else tr("NMS 去重阈值，越小保留的框越少"))
        for widget, value in ((self.size_spin_w, size[0]), (self.size_spin_h, size[1])):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.setEnabled(bool(dynamic))
            widget.blockSignals(False)
        self.set_size_fields_visible(bool(dynamic))
        self.replace_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.scope_btn.setEnabled(True)

    def set_size_fields_visible(self, visible):
        """整组「输入尺寸」控件的显隐（含标签与 "x"）。

        隐藏而不是禁用：静态模型的尺寸不可改，留着只会误导。

        两侧的分隔符跟着一起藏**掉一个**：这组如果看不见，它左右那两条 `-`
        就会挨在一起，显示成 `NMS [0.45] - - 覆盖`，看着像多打了一个。
        组收起时只留左边那条，于是读起来仍是「… NMS - 覆盖 …」。
        """
        for widget in (self.size_lbl, self.size_spin_w, self.size_sep_label,
                       self.size_spin_h):
            widget.setVisible(bool(visible))
        # 组收起时两条分隔符相邻，藏掉前面那条，只留后面那条。
        self._size_sep_before.setVisible(bool(visible))

    def size_fields_visible(self):
        """供测试/调用方核对当前是否显示了输入尺寸。"""
        return bool(self.size_lbl.isVisible())

    def clear_model(self):
        """没有模型时的样子：显示引擎默认值，只留加载按钮可点。

        必须显式填默认值：QDoubleSpinBox 初值是 setRange 的下限，
        不填就会显示成 0.01 / 0.05 / 32x32 这些占位数字，看着像参数被改坏了。
        """
        from ..engine.onnx_inference import (
            DEFAULT_CONF, DEFAULT_INPUT_SIZE, DEFAULT_IOU,
        )

        self._model_name = ""
        self._end2end = False
        self.name_label.setText(tr("尚未加载模型"))
        self.name_label.setToolTip("")
        for widget, value in ((self.conf_spin, DEFAULT_CONF),
                              (self.iou_spin, DEFAULT_IOU)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.setEnabled(False)
            widget.blockSignals(False)
        for widget, value in ((self.size_spin_w, DEFAULT_INPUT_SIZE[0]),
                              (self.size_spin_h, DEFAULT_INPUT_SIZE[1])):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.setEnabled(False)
            widget.blockSignals(False)
        # 没加载模型时还不知道是静态还是动态，整组先收起；加载后按类型决定。
        self.set_size_fields_visible(False)
        self.replace_btn.setEnabled(False)
        self.filter_btn.setEnabled(False)
        self.scope_btn.setEnabled(False)
        self.iou_spin.setToolTip(tr("NMS 去重阈值，越小保留的框越少"))
        self.set_filter_summary(set(), 0)
        self.set_scope_summary(tr("全图"))
        self._refresh_replace_text()

    def set_filter_summary(self, selected, total):
        """在按钮上显示当前筛了几类，便于不打开面板也知道状态。"""
        if selected:
            self.filter_btn.setText(f"{tr('类别筛选')}({len(selected)}/{total})")
        else:
            self.filter_btn.setText(tr("类别筛选"))

    def set_scope_summary(self, name, detail=""):
        """在按钮右侧显示当前范围。

        `detail` 保留给以后放参数摘要：本轮只显示模式名（三种模式的参数在
        弹窗里都能看到，摘要在行上不必再挤一行数字）。给 detail 时放进
        tooltip，hover 仍能看到。
        """
        text = name if name and name != tr("全图") else tr("全图")
        self.scope_summary_label.setText(text)
        self.scope_summary_label.setToolTip(
            f"{tr('推理范围')}: {text}{detail}" if detail else text)

    def scope_summary_text(self):
        return self.scope_summary_label.text()

    def retranslate(self):
        """切语言后刷新行内文字。"""
        self.load_btn.setText(tr("加载模型"))
        self.load_btn.setToolTip(tr("加载 ONNX 模型"))
        self.close_btn.setToolTip(tr("关闭参数行（关闭后不能推理）"))
        self.replace_btn.setToolTip(tr("覆盖：替换上一轮推理框；补充：保留并追加"))
        self.size_spin_w.setToolTip(tr("动态轴模型需要指定；静态轴模型按模型自身尺寸推理"))
        self.conf_lbl.setText(tr("conf"))
        self.iou_lbl.setText(tr("NMS"))
        self.size_lbl.setText(tr("输入尺寸"))
        self.batch_to_label.setText(tr("目标张号"))
        self.name_label.setText(self._model_name or tr("尚未加载模型"))
        self._refresh_replace_text()
        # 筛选按钮的计数文案由主窗口按当前选择重设
        self.filter_btn.setToolTip(tr("保留勾选的类别；没勾选的会被过滤掉"))
        self.scope_btn.setText(tr("推理范围"))
        self.scope_btn.setToolTip(tr("全图 / 滑窗 / 区域"))
        # 摘要文字要由主窗口按当前模式重设（`_refresh_scope_button`），
        # 这里只把 tooltip 与当前文字对齐，避免切语言后 tooltip 与实际不符。
        self.scope_summary_label.setToolTip(self.scope_summary_label.text())

    def _tooltip(self, name, size, providers):
        parts = [name, f"{tr('输入尺寸')}: {size[0]}x{size[1]}"]
        if providers:
            parts.append(f"{tr('设备')}: {', '.join(providers)}")
        else:
            parts.append(f"{tr('设备')}: {tr('尚未加载模型')}")
        return "\n".join(parts)

    def thresholds(self):
        return self.conf_spin.value(), self.iou_spin.value()

    def input_size(self):
        return self.size_spin_w.value(), self.size_spin_h.value()

    def replace_mode(self):
        return self.replace_btn.isChecked()

    # ---------- 内部 ----------

    def _emit_thresholds(self, _value=None):
        conf, iou = self.thresholds()
        self.thresholds_changed.emit(conf, iou)

    def _emit_size(self, _value=None):
        width, height = self.input_size()
        self.input_size_changed.emit(width, height)

    def _emit_batch_to(self, value=None):
        # 只有**用户真手输**才会走到这里（`set_batch_to` 屏蔽信号，程序化设置
        # 不算）。标记下来，后续换数据集时不再用默认值覆盖用户的意图。
        self._batch_to_user_set = True
        self.batch_to_changed.emit(self.batch_to_target())

    def batch_to_target(self):
        """「一键推理」的目标张号；0 表示不限制。"""
        return int(self.batch_to_spin.value())

    def set_batch_to(self, value):
        """外部（切图/换数据集后）同步目标张号，不再回发信号造成循环。"""
        spin = self.batch_to_spin
        spin.blockSignals(True)
        spin.setValue(int(value or 0))
        spin.blockSignals(False)

    def set_batch_to_range(self, total):
        # 按数据集张数收窄可选范围，避免填出根本不存在的张号。
        total = int(total or 0)
        spin = self.batch_to_spin
        spin.setMaximum(max(0, total))
        # 默认值 = 最后一张（用户要求）。用户改过就不再覆盖，否则他一填完
        # 换数据集，输入的意图就被抹掉了。
        if not self._batch_to_user_set:
            spin.blockSignals(True)
            spin.setValue(max(0, total))
            spin.blockSignals(False)
        spin.setToolTip(
            tr("一键推理的目标张号（含）：当前第 200 张时填 300 跑 200~300，"
               "填 100 跑 100~200；0 表示不限制、跑到末尾"))

    def _on_replace_toggled(self):
        self._refresh_replace_text()
        self.replace_mode_changed.emit(self.replace_btn.isChecked())

    def _refresh_replace_text(self):
        self.replace_btn.setText(tr("覆盖") if self.replace_btn.isChecked() else tr("补充"))

    def _on_close(self):
        self.hide()
        self.closed.emit()
