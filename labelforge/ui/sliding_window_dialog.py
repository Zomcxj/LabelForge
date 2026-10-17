"""滑窗推理的窗口参数配置（独立窗口形态）。

参数控件本体在 `widgets/window_params.py`（`WindowParamsWidget`），因为
`ScopePanel` 会把它**内嵌**在范围弹窗里用；本模块只提供「独立窗口」这个
宿主，保留给需要单独开窗的路径与既有调用方。

窗口数量以 `iter_windows()` 的实际结果为准，不用 `size_for_count()` 反推
——后者取整后可能多出一窗。

模板与数据增强**共用同一份** `crop_templates`（双向同步，最多 3 条）。
点击模板只回填输入框并刷新预览，**不生效**：与面板的确定/取消语义一致，
必须点「确定」才写回范围。本窗口自身不写模板——写模板由调用方在确定之后做。
"""
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QPushButton, QVBoxLayout,
)

from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager
from .widgets.window_params import (
    DEFAULT_OVERLAP, DEFAULT_WINDOW, MANY_WINDOWS, MIN_WINDOW,
    WindowParamsWidget,
)

__all__ = ["SlidingWindowDialog", "DEFAULT_WINDOW", "DEFAULT_OVERLAP",
           "MIN_WINDOW", "MANY_WINDOWS"]


class SlidingWindowDialog(QDialog):
    """窗口参数配置；确定后取 `spec()` 得到引擎口径的窗口字典。"""

    def __init__(self, reference_size=None, spec=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("滑窗参数"))
        self.setMinimumWidth(420)
        self._reference = reference_size
        self._build_ui()
        self._apply_style()
        self.params = WindowParamsWidget(reference_size=reference_size, spec=spec)
        self._body.addWidget(self.params, 1)
        # 参数非法（窗口数超上限等）时禁用确定：宿主内嵌时由宿主做这件事，
        # 独立窗口这一形态要自己接上，否则非法参数能点确定进任务。
        self.params.changed.connect(self._sync_ok_enabled)

    def _sync_ok_enabled(self):
        self.ok_btn.setEnabled(self.params.is_valid())

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        self._body = QVBoxLayout()
        layout.addLayout(self._body)

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

    # ---------- 对外接口（转发给参数控件） ----------

    def spec(self):
        """引擎口径的窗口字典（`iter_windows()` 直接吃）。"""
        return self.params.spec()

    def window_count(self):
        """按参考图算出的实际窗口数；没有参考图或参数非法时返回 None。"""
        return self.params.window_count()

    def window_grid(self):
        """(行数, 列数)；与 `iter_windows()` 用同一份窗口数学。"""
        return self.params.window_grid()

    # 输入框直接暴露出去，既有测试与调用方按这些名字取用
    @property
    def _w(self):
        return self.params._w

    @property
    def _h(self):
        return self.params._h

    @property
    def _ov(self):
        return self.params._ov

    @property
    def _estimate(self):
        return self.params._estimate

    @property
    def _hint(self):
        return self.params._hint

    def _refresh_estimate(self, _value=None):
        self.params._refresh_estimate(_value)
