"""画布上方的模型行：加载按钮、模型名、阈值、输入尺寸、覆盖/补充与关闭。

工具栏第一个图标负责显示/隐藏这一行；关闭后两个推理图标会禁用，重新点开即恢复。
阈值与尺寸改动即时写回 detector，不重建会话，所以调参不用重新加载模型。
"""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QDoubleSpinBox, QSpinBox,
    QToolButton,
)

from .i18n import t as tr
from .theme import ThemeManager


class ModelRow(QWidget):
    """模型信息行；控件改动通过信号交给主窗口写回 detector。"""

    thresholds_changed = pyqtSignal(float, float)
    input_size_changed = pyqtSignal(int, int)
    replace_mode_changed = pyqtSignal(bool)
    load_requested = pyqtSignal()
    filter_requested = pyqtSignal()
    scope_requested = pyqtSignal()
    closed = pyqtSignal()

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

        self.conf_lbl = QLabel(tr("置信度"))
        layout.addWidget(self.conf_lbl)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setFixedWidth(72)
        self.conf_spin.valueChanged.connect(self._emit_thresholds)
        layout.addWidget(self.conf_spin)

        self.iou_lbl = QLabel(tr("NMS"))
        layout.addWidget(self.iou_lbl)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setFixedWidth(72)
        self.iou_spin.valueChanged.connect(self._emit_thresholds)
        layout.addWidget(self.iou_spin)

        self.size_lbl = QLabel(tr("输入尺寸"))
        layout.addWidget(self.size_lbl)
        self.size_spin_w = QSpinBox()
        self.size_spin_w.setRange(32, 8192)
        self.size_spin_w.setFixedWidth(72)
        self.size_spin_w.setToolTip(tr("动态轴模型需要指定；静态轴模型按模型自身尺寸推理"))
        self.size_spin_w.valueChanged.connect(self._emit_size)
        layout.addWidget(self.size_spin_w)
        layout.addWidget(QLabel("x"))
        self.size_spin_h = QSpinBox()
        self.size_spin_h.setRange(32, 8192)
        self.size_spin_h.setFixedWidth(72)
        self.size_spin_h.valueChanged.connect(self._emit_size)
        layout.addWidget(self.size_spin_h)

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

        self.filter_btn = QPushButton(tr("类别筛选"))
        self.filter_btn.setObjectName("optionsBtn")
        self.filter_btn.setFixedHeight(24)
        self.filter_btn.setEnabled(False)
        self.filter_btn.clicked.connect(self.filter_requested.emit)
        layout.addWidget(self.filter_btn)

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

        layout.addStretch()

        self.close_btn = QToolButton()
        self.close_btn.setText("×")
        self.close_btn.setObjectName("modelRowClose")
        self.close_btn.setToolTip(tr("关闭参数行（关闭后不能推理）"))
        self.close_btn.clicked.connect(self._on_close)
        layout.addWidget(self.close_btn)

        self._refresh_replace_text()

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

        静态轴模型的尺寸由模型自身决定，输入框只作展示（禁用），避免用户改了
        却没有任何效果；动态轴模型必须靠它指定，保持可编辑。

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
            widget.setEnabled(dynamic)
            widget.blockSignals(False)
        self.replace_btn.setEnabled(True)
        self.filter_btn.setEnabled(True)
        self.scope_btn.setEnabled(True)

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
        self.conf_lbl.setText(tr("置信度"))
        self.iou_lbl.setText(tr("NMS"))
        self.size_lbl.setText(tr("输入尺寸"))
        self.name_label.setText(self._model_name or tr("尚未加载模型"))
        self._refresh_replace_text()
        # 筛选按钮的计数文案由主窗口按当前选择重设
        self.filter_btn.setToolTip(tr("保留勾选的类别；没勾选的会被过滤掉"))
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

    def _on_replace_toggled(self):
        self._refresh_replace_text()
        self.replace_mode_changed.emit(self.replace_btn.isChecked())

    def _refresh_replace_text(self):
        self.replace_btn.setText(tr("覆盖") if self.replace_btn.isChecked() else tr("补充"))

    def _on_close(self):
        self.hide()
        self.closed.emit()
