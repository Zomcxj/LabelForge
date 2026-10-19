"""Canvas display controls (bottom-left overlay).

Three vertical tracks side by side — label opacity, image brightness and
contrast. Each track is a vertical slider with a caption at its foot and the
live value above it. A row-level reset sits at the card's bottom-right corner.

The panel only emits signals; the owner applies the values to the canvas.
"""

from PySide6.QtCore import Qt, QTimer, Signal, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
)

from ..ui.theme import ThemeManager
from ..ui.i18n import t as tr
from ..ui.icons import SVG_BURST, _load_svg_icon

TONE_MIN = 0
TONE_MAX = 150
TONE_NEUTRAL = 50

OPACITY_MIN = 0
OPACITY_MAX = 100
OPACITY_NEUTRAL = 100

TRACK_HEIGHT = 96
TONE_DEBOUNCE_MS = 33


class _VerticalTrack(QWidget):
    """One vertical slider column: value, track, caption."""

    def __init__(self, caption, tooltip, minimum, maximum, default,
                 display, parent=None):
        super().__init__(parent)
        self._caption = caption
        self._tooltip = tooltip
        self._display = display
        self._reset_tooltip = ""

        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(3)
        col.setAlignment(Qt.AlignHCenter)

        self.value = QLabel()
        self.value.setAlignment(Qt.AlignCenter)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(default)
        self.slider.setFixedHeight(TRACK_HEIGHT)
        self.slider.setFixedWidth(26)
        self.slider.setInvertedAppearance(False)
        self.slider.setTracking(True)
        self.slider.setFocusPolicy(Qt.NoFocus)

        self.caption = QLabel()
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(False)
        # 英文（Brightness/Contrast）比中文宽很多：列宽按内容撑开，不截断

        # 每行自己的复位按钮：恢复该行默认值
        self.reset = QToolButton()
        self.reset.setText("↺")
        self.reset.setFixedSize(16, 14)
        self.reset.setFocusPolicy(Qt.NoFocus)
        self._default = default
        self.reset.clicked.connect(lambda: self.slider.setValue(self._default))

        col.addWidget(self.value)
        col.addWidget(self.slider, 0, Qt.AlignHCenter)
        col.addWidget(self.caption)
        col.addWidget(self.reset, 0, Qt.AlignHCenter)

        self._refresh_text()

    def _set_reset_style(self, color):
        self.reset.setStyleSheet(
            f"QToolButton {{ background: transparent; border: none;"
            f" color: {color}; font-size: 11px; padding: 0; }}"
        )

    def _set_fonts(self, secondary, primary):
        self.value.setStyleSheet(
            f"QLabel {{ color: {primary}; font-size: 10px; font-weight: 600;"
            " background: transparent; }"
        )
        self.caption.setStyleSheet(
            f"QLabel {{ color: {secondary}; font-size: 10px;"
            " background: transparent; }"
        )
        self._set_reset_style(secondary)

    def _set_tooltip(self, text):
        self.caption.setToolTip(text)
        self.value.setToolTip(text)
        self.slider.setToolTip(text)
        self.reset.setToolTip(self._reset_tooltip)

    def _refresh_text(self):
        self.caption.setText(self._caption)
        self._set_tooltip(self._tooltip)
        self._render_value()

    def _render_value(self):
        v = self.slider.value()
        if self._display == "factor":
            self.value.setText(f"{v / TONE_NEUTRAL:.2f}x")
        else:
            self.value.setText(f"{v}")


class CanvasDisplayPanel(QWidget):
    """Vertical display controls: label opacity, brightness, contrast."""

    label_opacity_changed = Signal(int)
    image_tone_changed = Signal(int, int)
    geometry_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("canvasDisplayPanel")
        self._collapsed = True
        self.setProperty("collapsed", True)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(self._sheet())

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 6, 10, 6)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(0)
        self._title = QLabel()
        self._chevron = QToolButton()
        self._chevron.setFixedSize(18, 18)
        self._chevron.setFocusPolicy(Qt.NoFocus)
        self._bar = bar
        self._bar_stretch = 1
        bar.addWidget(self._title)
        bar.addStretch(1)
        bar.addWidget(self._chevron)
        root.addLayout(bar)

        self._body = QWidget()
        body = QHBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(14)

        self._opacity_row = _VerticalTrack(
            tr("标签透明度"), tr("调整标注形状与遮罩的透明度，标签文字保持完全可见。"),
            OPACITY_MIN, OPACITY_MAX, OPACITY_NEUTRAL, "percent",
        )
        self._bright_row = _VerticalTrack(
            tr("画面亮度"), tr("调整底层图像的亮度。"),
            TONE_MIN, TONE_MAX, TONE_NEUTRAL, "factor",
        )
        self._contrast_row = _VerticalTrack(
            tr("对比度"), tr("调整底层图像的对比度。"),
            TONE_MIN, TONE_MAX, TONE_NEUTRAL, "factor",
        )
        body.addWidget(self._opacity_row)
        body.addWidget(self._bright_row)
        body.addWidget(self._contrast_row)
        root.addWidget(self._body)

        self._tone_timer = QTimer(self)
        self._tone_timer.setSingleShot(True)
        self._tone_timer.setInterval(TONE_DEBOUNCE_MS)
        self._tone_timer.timeout.connect(self._emit_tone)

        self._chevron.clicked.connect(self._toggle)
        self._opacity_row.slider.valueChanged.connect(self._on_opacity)
        self._bright_row.slider.valueChanged.connect(self._on_tone_input)
        self._contrast_row.slider.valueChanged.connect(self._on_tone_input)

        self._apply_theme()
        self._refresh_texts()
        self.adjustSize()
        # 默认收起：只显示爆炸图标，点开才展开滑轨
        self._apply_collapsed_state()
        self.adjustSize()

    # ---------- 样式 ----------

    @staticmethod
    def _card_bg(theme):
        """卡片背景：在原主题底色上叠加半透明，浮在画布上不遮挡内容。

        widget_bg 可能是 #RRGGBB 或 rgba(...)，统一转成带透明度的 rgba。
        """
        raw = theme.get('widget_bg', '#2b2b2b')
        alpha = 170
        try:
            if raw.startswith('#') and len(raw) in (7, 9):
                r = int(raw[1:3], 16)
                g = int(raw[3:5], 16)
                b = int(raw[5:7], 16)
                return f"rgba({r}, {g}, {b}, {alpha})"
            if raw.startswith('rgba'):
                parts = raw[raw.index('(') + 1:raw.rindex(')')].split(',')
                r, g, b = (int(float(x)) for x in parts[:3])
                return f"rgba({r}, {g}, {b}, {alpha})"
            if raw.startswith('rgb'):
                parts = raw[raw.index('(') + 1:raw.rindex(')')].split(',')
                r, g, b = (int(float(x)) for x in parts[:3])
                return f"rgba({r}, {g}, {b}, {alpha})"
        except (ValueError, IndexError):
            pass
        return raw

    def _sheet(self):
        theme = ThemeManager.get_theme()
        accent = theme.get("accent", "#0071e3")
        return f"""
        #canvasDisplayPanel {{
            background: {self._card_bg(theme)};
            border: 1px solid rgba(128, 128, 128, 45);
            border-radius: 8px;
        }}
        #canvasDisplayPanel[collapsed="true"] {{
            background: transparent;
            border: none;
        }}
        /* 内部容器与子控件一律不画背景，整卡只有一层半透明底 */
        #canvasDisplayPanel QWidget {{
            background: transparent;
        }}
        #canvasDisplayPanel QSlider::groove:vertical {{
            width: 6px;
            background: rgba(128, 128, 128, 55);
            border-radius: 3px;
        }}
        #canvasDisplayPanel QSlider::sub-page:vertical {{
            background: rgba(128, 128, 128, 55);
            border-radius: 3px;
        }}
        #canvasDisplayPanel QSlider::add-page:vertical {{
            background: {accent};
            border-radius: 3px;
        }}
        #canvasDisplayPanel QSlider::handle:vertical {{
            background: #ffffff;
            border: 2px solid {accent};
            width: 18px;
            height: 8px;
            margin: 0 -7px;
            border-radius: 5px;
        }}
        """

    def _apply_theme(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(self._sheet())
        self._title.setStyleSheet(
            f"QLabel {{ color: {theme['text_secondary']}; font-size: 10px;"
            " font-weight: 600; background: transparent; }"
        )
        for row in (self._opacity_row, self._bright_row, self._contrast_row):
            row._set_fonts(theme['text_secondary'], theme['text_primary'])
        self._chevron.setStyleSheet(
            "QToolButton { background: transparent; border: none;"
            " outline: none; padding: 0; margin: 0; }"
            "QToolButton:hover { background: transparent; border: none; }"
            "QToolButton:pressed { background: transparent; border: none; }"
            "QToolButton:focus { background: transparent; border: none; }"
        )
        self._chevron.setIcon(
            QIcon(_load_svg_icon(
                SVG_BURST, 16, theme.get("text_primary", "#333")))
        )
        self._chevron.setIconSize(QSize(16, 16))

    def refresh_theme(self):
        self._apply_theme()

    # ---------- 文本 ----------

    def _refresh_texts(self):
        self._title.setText(tr("画布显示"))
        self._chevron.setToolTip(
            tr("展开调节") if self._collapsed else tr("收起调节")
        )

        for row, caption, tip in (
            (self._opacity_row, tr("标签透明度"),
             tr("调整标注形状与遮罩的透明度，标签文字保持完全可见。")),
            (self._bright_row, tr("画面亮度"), tr("调整底层图像的亮度。")),
            (self._contrast_row, tr("对比度"), tr("调整底层图像的对比度。")),
        ):
            row._caption = caption
            row._tooltip = tip
            row._reset_tooltip = tr("恢复默认")
            row._refresh_text()

    def retranslate_ui(self):
        self._refresh_texts()

    # ---------- 交互 ----------

    def _on_opacity(self, value):
        self._opacity_row._render_value()
        self.label_opacity_changed.emit(value)

    def _on_tone_input(self, _=None):
        self._bright_row._render_value()
        self._contrast_row._render_value()
        if not self._tone_timer.isActive():
            self._tone_timer.start()

    def _emit_tone(self):
        self.image_tone_changed.emit(
            self._bright_row.slider.value(),
            self._contrast_row.slider.value(),
        )

    def _apply_collapsed_state(self):
        """按当前 _collapsed 更新可见性与尺寸（构造时和切换时共用）。"""
        self.setProperty("collapsed", self._collapsed)
        # 收起时只留爆炸图标：标题与滑轨区都隐藏
        self._title.setVisible(not self._collapsed)
        self._body.setVisible(not self._collapsed)
        # 收起时整卡收缩到只剩爆炸图标
        self._body.setMaximumWidth(16777215 if not self._collapsed else 0)
        if self._collapsed:
            # 只剩图标：整卡收缩，四周留出边距避免图标被裁
            pad = 6
            self.setFixedSize(self._chevron.width() + pad * 2,
                              self._chevron.height() + pad * 2)
            self._bar.setContentsMargins(0, 0, 0, 0)
            self._bar.setStretch(self._bar_stretch, 0)
        else:
            self._bar.setAlignment(self._chevron, Qt.AlignRight)
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)
        self.setContentsMargins(*(0, 0, 0, 0) if self._collapsed else (10, 6, 10, 6))
        self._chevron.setToolTip(
            tr("展开调节") if self._collapsed else tr("收起调节")
        )
        self.style().unpolish(self)
        self.style().polish(self)
        self.layout().activate()
        self.adjustSize()

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._apply_collapsed_state()
        self.geometry_changed.emit()

    # ---------- 外部同步 ----------

    def set_opacity(self, value):
        self._opacity_row.slider.blockSignals(True)
        self._opacity_row.slider.setValue(value)
        self._opacity_row.slider.blockSignals(False)
        self._opacity_row._render_value()

    def set_brightness_contrast(self, brightness, contrast):
        self._tone_timer.stop()
        for row, value in (
            (self._bright_row, brightness),
            (self._contrast_row, contrast),
        ):
            row.slider.blockSignals(True)
            row.slider.setValue(value)
            row.slider.blockSignals(False)
            row._render_value()
