"""滑窗推理的窗口参数配置：窗口宽高、重叠像素、实时预估窗口数。

只负责参数，不画预览图：推理的窗口与增强裁剪的「空窗丢弃」口径不同
（推理必须跑每一个窗口），照搬 `crop_dialog` 的红绿预览会误导。
窗口数量以 `iter_windows()` 的实际结果为准，不用 `size_for_count()` 反推
——后者取整后可能多出一窗。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
)

from ..engine.augmenter.crop import MAX_WINDOW
from ..engine.inference_scope import ScopeError, iter_windows
from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager

DEFAULT_WINDOW = 640
DEFAULT_OVERLAP = 200
MIN_WINDOW = 8                  # 与 crop_dialog 的窗口下限一致
MANY_WINDOWS = 1000             # 超过就提醒一句：每张图要跑这么多窗


class SlidingWindowDialog(QDialog):
    """窗口参数配置；确定后取 `spec()` 得到引擎口径的窗口字典。"""

    def __init__(self, reference_size=None, spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("滑窗参数"))
        self.setMinimumWidth(420)
        self._reference = reference_size
        spec = dict(spec or {})
        self._build_ui()
        self._apply_style()
        # 配置里的值可能是旧版本写坏的字符串/None，逐项兜底，不整个崩掉
        for spin, key, fallback in ((self._w, "w", DEFAULT_WINDOW),
                                    (self._h, "h", DEFAULT_WINDOW),
                                    (self._ov, "overlap", DEFAULT_OVERLAP)):
            spin.setValue(_as_int(spec.get(key), fallback))
        self._refresh_estimate()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("裁剪尺寸（像素）")))
        self._w = QSpinBox()
        self._w.setRange(MIN_WINDOW, MAX_WINDOW)
        self._w.setFixedWidth(84)
        self._h = QSpinBox()
        self._h.setRange(MIN_WINDOW, MAX_WINDOW)
        self._h.setFixedWidth(84)
        row.addWidget(self._w)
        row.addWidget(QLabel("×"))
        row.addWidget(self._h)
        row.addStretch()
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("重叠长度（像素）")))
        self._ov = QSpinBox()
        self._ov.setRange(0, MAX_WINDOW)
        self._ov.setFixedWidth(84)
        row.addWidget(self._ov)
        row.addStretch()
        layout.addLayout(row)

        self._hint = QLabel(tr("窗口尺寸与模型输入尺寸独立：窗口是原图上的裁剪范围"))
        self._hint.setWordWrap(True)
        self._hint.setObjectName("windowHint")
        layout.addWidget(self._hint)

        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        self._estimate.setObjectName("windowEstimate")
        layout.addWidget(self._estimate)

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

        for spin in (self._w, self._h, self._ov):
            spin.valueChanged.connect(self._refresh_estimate)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {theme['widget_bg']}; color: {theme['text_primary']}; }}
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#windowHint, QLabel#windowEstimate {{ color: {theme['text_secondary']}; }}
            QSpinBox {{
                background-color: {theme['panel_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                padding: 1px 4px;
            }}
            QPushButton {{
                background-color: {theme['panel_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{ border-color: {theme['accent']}; }}
        """)

    def refresh_theme(self):
        self._apply_style()
        sync_titlebar(self)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())
        sync_titlebar(self)

    # ---------- 对外接口 ----------

    def spec(self):
        """引擎口径的窗口字典（`iter_windows()` 直接吃）。"""
        return {"w": self._w.value(), "h": self._h.value(),
                "overlap": self._ov.value()}

    def window_count(self):
        """按参考图算出的实际窗口数；没有参考图或参数非法时返回 None。"""
        if not self._reference:
            return None
        try:
            return len(iter_windows(self._reference[0], self._reference[1],
                                    self.spec()))
        except ScopeError:
            return None

    # ---------- 内部 ----------

    def _refresh_estimate(self, _value=None):
        """预估窗口数：参数非法就明说，并禁用确定，不能让非法值进任务。"""
        if not self._reference:
            self._estimate.setText(tr("未加载图片，无法预估窗口数"))
            self.ok_btn.setEnabled(True)
            return
        count = self.window_count()
        if count is None:
            self._estimate.setText(tr("窗口参数非法，请检查尺寸与重叠"))
            self.ok_btn.setEnabled(False)
            return
        self.ok_btn.setEnabled(True)
        rows, cols = self.window_grid()
        text = f"{tr('每张图')} {rows} × {cols} = {count} {tr('个窗口')}"
        if count > MANY_WINDOWS:
            text += "\n" + tr("窗口数很多，耗时约为全图的 {n} 倍").format(n=count)
        self._estimate.setText(text)

    def window_grid(self):
        """(行数, 列数)；与 `iter_windows()` 用同一份窗口数学。"""
        from ..engine.augmenter.crop import normalized_window, window_starts

        iw, ih = int(self._reference[0]), int(self._reference[1])
        cw, ch, ov = normalized_window(iw, ih, self.spec())
        return (len(window_starts(ih, ch, ov)), len(window_starts(iw, cw, ov)))


def _as_int(value, fallback):
    """配置里的整数；坏值（None/字符串/超范围）回退到默认。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)
