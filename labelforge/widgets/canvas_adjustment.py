"""A compact panel docked at the bottom-left of the canvas.

It groups three sliders that let the user fine tune how the annotations and the
underlying image are displayed:

- ``Opacity``    : transparency of the labels/shapes (0-100%, default 100%).
- ``Brightness`` : brightness of the image (slider 0-150, displayed as a
  0.00-3.00 factor, neutral 1.00).
- ``Contrast``   : contrast of the image, same range/mapping as brightness.
"""

from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QToolButton
)
from ..ui.theme import ThemeManager
from ..ui.i18n import t as tr


class CanvasAdjustmentWidget(QWidget):
    """Three-slider panel for label opacity, image brightness and contrast."""

    opacity_changed = Signal(int)
    brightness_contrast_changed = Signal(int, int)
    geometry_changed = Signal()

    SLIDER_WIDTH = 120
    BC_UPDATE_INTERVAL_MS = 33

    OPACITY_MIN = 0
    OPACITY_MAX = 100
    OPACITY_DEFAULT = 100

    BC_MIN = 0
    BC_MAX = 150
    BC_DEFAULT = 50

    def _label_css(self, bold=False):
        weight = " font-weight: 600;" if bold else ""
        return f"QLabel {{ color: {ThemeManager.get_theme()['text_primary']}; font-size: 11px;{weight} background: transparent; }}"

    def _reset_css(self):
        text = ThemeManager.get_theme()['text_primary']
        return (
            f"QPushButton {{ background: rgba(120, 120, 120, 60);"
            f" border-radius: 4px; font-size: 11px; color: {text}; padding: 0; }}"
            "QPushButton:hover { background: rgba(120, 120, 120, 110); }"
            "QPushButton:pressed { background: rgba(120, 120, 120, 150); }"
        )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvas_adjustment")
        self._collapsed = False
        self.setProperty("collapsed", False)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(self._build_stylesheet())

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        self.title_label = QLabel(tr("画布显示"))
        self.title_label.setStyleSheet(self._label_css(bold=True))
        self.toggle_button = QToolButton()
        self.toggle_button.setText("▲")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setToolTip(tr("收起调节"))
        self.toggle_button.setFocusPolicy(Qt.NoFocus)
        self.toggle_button.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            " border-radius: 4px; padding: 4px; }"
            "QToolButton:hover { background: rgba(120, 120, 120, 60); }"
            "QToolButton:pressed { background: rgba(120, 120, 120, 100); }"
        )
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.toggle_button)
        layout.addLayout(header_layout)

        self.content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)
        self.content_widget.setLayout(content_layout)

        self.opacity_slider, self.opacity_value_label, self.opacity_name_label = self._build_row(
            content_layout,
            tr("标签透明度"),
            self.OPACITY_MIN,
            self.OPACITY_MAX,
            self.OPACITY_DEFAULT,
            tr("调整标注形状与遮罩的透明度，标签文字保持完全可见。"),
            display="percent",
        )
        self.brightness_slider, self.brightness_value_label, self.brightness_name_label = self._build_row(
            content_layout,
            tr("画面亮度"),
            self.BC_MIN,
            self.BC_MAX,
            self.BC_DEFAULT,
            tr("调整底层图像的亮度。"),
            display="factor",
        )
        self.contrast_slider, self.contrast_value_label, self.contrast_name_label = self._build_row(
            content_layout,
            tr("对比度"),
            self.BC_MIN,
            self.BC_MAX,
            self.BC_DEFAULT,
            tr("调整底层图像的对比度。"),
            display="factor",
        )
        layout.addWidget(self.content_widget)

        self.setLayout(layout)
        self.adjustSize()

        self._bc_update_timer = QTimer(self)
        self._bc_update_timer.setSingleShot(True)
        self._bc_update_timer.setInterval(self.BC_UPDATE_INTERVAL_MS)
        self._bc_update_timer.timeout.connect(self._emit_bc_changed)

        self.toggle_button.clicked.connect(self._toggle_collapsed)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        self.brightness_slider.valueChanged.connect(self._on_bc_changed)
        self.contrast_slider.valueChanged.connect(self._on_bc_changed)

    def _build_stylesheet(self):
        primary = "#0071e3"
        theme = ThemeManager.get_theme()
        return f"""
        #canvas_adjustment {{
            background: {theme['widget_bg']};
            border: none;
            border-radius: 4px;
        }}
        #canvas_adjustment[collapsed="true"] {{
            background: transparent;
        }}
        #canvas_adjustment QSlider::groove:horizontal {{
            height: 4px;
            background: rgba(0, 0, 0, 55);
            border-radius: 2px;
        }}
        #canvas_adjustment QSlider::handle:horizontal {{
            background: {primary};
            border: none;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        #canvas_adjustment QSlider::sub-page:horizontal {{
            background: {primary};
            border-radius: 2px;
        }}
        """

    def _build_row(self, layout, title, minimum, maximum, default,
                   tooltip, display="percent"):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        name_label = QLabel(title)
        name_label.setFixedWidth(64)
        name_label.setStyleSheet(self._label_css())
        name_label.setToolTip(tooltip)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(default)
        slider.setFixedWidth(self.SLIDER_WIDTH)
        slider.setTracking(True)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setToolTip(tooltip)

        value_label = QLabel()
        value_label.setProperty("display", display)
        value_label.setFixedWidth(36)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value_label.setStyleSheet(self._label_css())
        value_label.setToolTip(tooltip)
        self._set_value_text(value_label, default)

        reset_btn = QPushButton("↺")
        reset_btn.setFixedSize(22, 20)
        reset_btn.setStyleSheet(self._reset_css())
        reset_btn.setToolTip(tr("恢复默认"))
        reset_btn.setFocusPolicy(Qt.NoFocus)
        reset_btn.clicked.connect(
            lambda _=False, s=slider: s.setValue(default)
        )

        row.addWidget(name_label)
        row.addWidget(slider)
        row.addWidget(value_label)
        row.addWidget(reset_btn)
        layout.addLayout(row)
        return slider, value_label, name_label

    @staticmethod
    def _set_value_text(label, value):
        if label.property("display") == "factor":
            label.setText(f"{value / 50:.2f}")
        else:
            label.setText(f"{value}%")

    def _on_opacity_changed(self, value):
        self._set_value_text(self.opacity_value_label, value)
        self.opacity_changed.emit(value)

    def _on_bc_changed(self, _=None):
        self._set_value_text(
            self.brightness_value_label, self.brightness_slider.value()
        )
        self._set_value_text(
            self.contrast_value_label, self.contrast_slider.value()
        )
        if not self._bc_update_timer.isActive():
            self._bc_update_timer.start()

    def _emit_bc_changed(self):
        self.brightness_contrast_changed.emit(
            self.brightness_slider.value(), self.contrast_slider.value()
        )

    def _toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self.setProperty("collapsed", self._collapsed)
        self.title_label.setVisible(not self._collapsed)
        self.content_widget.setVisible(not self._collapsed)
        margins = (4, 4, 4, 4) if self._collapsed else (10, 8, 10, 8)
        self.layout().setContentsMargins(*margins)
        icon = "▼" if self._collapsed else "▲"
        self.toggle_button.setText(icon)
        tooltip = (
            tr("展开调节")
            if self._collapsed
            else tr("收起调节")
        )
        self.toggle_button.setToolTip(tooltip)
        self.style().unpolish(self)
        self.style().polish(self)
        self.layout().activate()
        self.adjustSize()
        self.geometry_changed.emit()

    def set_opacity(self, value):
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(value)
        self.opacity_slider.blockSignals(False)
        self._set_value_text(self.opacity_value_label, value)

    def set_brightness_contrast(self, brightness, contrast):
        self._bc_update_timer.stop()
        for slider, value, label in (
            (self.brightness_slider, brightness, self.brightness_value_label),
            (self.contrast_slider, contrast, self.contrast_value_label),
        ):
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)
            self._set_value_text(label, value)

    def retranslate_ui(self):
        """语言切换时刷新面板全部文本"""
        self.title_label.setText(tr("画布显示"))
        self.toggle_button.setToolTip(
            tr("展开调节") if self._collapsed else tr("收起调节")
        )
        rows = (
            (self.opacity_name_label, "标签透明度",
             "调整标注形状与遮罩的透明度，标签文字保持完全可见。",
             self.opacity_slider, self.opacity_value_label),
            (self.brightness_name_label, "画面亮度",
             "调整底层图像的亮度。",
             self.brightness_slider, self.brightness_value_label),
            (self.contrast_name_label, "对比度",
             "调整底层图像的对比度。",
             self.contrast_slider, self.contrast_value_label),
        )
        for name_label, name_key, tip_key, slider, value_label in rows:
            name_label.setText(tr(name_key))
            name_label.setToolTip(tr(tip_key))
            slider.setToolTip(tr(tip_key))
            value_label.setToolTip(tr(tip_key))
            self._set_value_text(value_label, slider.value())
        for button in self.findChildren(QPushButton):
            button.setToolTip(tr("恢复默认"))

    def refresh_theme(self):
        self.setStyleSheet(self._build_stylesheet())
        self.title_label.setStyleSheet(self._label_css(bold=True))
        for label in self.findChildren(QLabel):
            if label is not self.title_label:
                label.setStyleSheet(self._label_css())
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(self._reset_css())
