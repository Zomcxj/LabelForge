"""滑窗参数控件：宽 / 高 / 重叠 + 窗口数预估 + 模板按钮。

抽成控件而不是留在某个对话框里，是因为它有两个宿主：
- `ScopePanel` 把它**内嵌**在范围弹窗里；
- `SlidingWindowDialog` 仍单独存在，供需要独立窗口的路径使用。

**这里没有预览图**：窗口网格直接画在编辑区的**主画布**上（用户明确要求
「直接在画布展示，不要单独小窗显示全图」）。控件只负责参数与「现在会切出
多少窗」的文字，网格的绘制由 `mixins/inference.py` 交给画布完成；本控件把
`changed` 发出去，宿主据此重画网格。

窗口数一律以 `iter_windows()` 的实际结果为准，不用 `size_for_count()` 反推
——后者取整后可能多出一窗。非法参数的口径也与引擎一致（`重叠 >= 窗口尺寸`
不算错，引擎会钳；真失败是尺寸越界/非正/超 `MAX_WINDOWS`）。

模板与数据增强**共用同一份** `crop_templates`（双向同步，最多 3 条）。点击
模板只回填输入框，**不生效**：必须由宿主的「确定」才写回。控件自身不写模板
——写模板由调用方在确定之后做。

**只有选中「滑窗」时才可编辑**（用户要求）：`set_editable(False)` 会把输入框
一并禁用，既点不动也改不动。

**没有模板**：早先这里有一排「模板」按钮，与数据增强共用 `crop_templates`
（双向同步）。用户要求「推理滑窗不要模板了，也不与数据增强同步模板」，那排
按钮与同步逻辑都已删除；滑窗参数只认当前输入框里的值。
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from ...engine.augmenter.crop import MAX_WINDOW
from ...engine.inference_scope import ScopeError, iter_windows
from ..i18n import t as tr
from ..theme import ThemeManager

DEFAULT_WINDOW = 640
DEFAULT_OVERLAP = 200
MIN_WINDOW = 8                  # 与 crop_dialog 的窗口下限一致
MANY_WINDOWS = 1000             # 超过就提醒一句：每张图要跑这么多窗
# 最低可见比例：数据增强侧也有同名概念，但**推理侧不再跟它同步**（用户
# 明确「推理滑窗不要模板了，也不与数据增强同步模板」），所以这个默认值只
# 服务推理侧自己的输入框。0.8 是用户指定的默认值。
MIN_VISIBLE = 0.5
DEFAULT_MIN_VISIBLE = 0.8


class WindowParamsWidget(QWidget):
    """窗口参数 + 预估 + 原图网格预览 + 模板按钮。"""

    # 参数或预览发生变化（宿主据此重画/校验）；也让宿主能挂自己的联动
    changed = Signal()

    def __init__(self, reference_size=None, spec=None, parent=None):
        super().__init__(parent)
        self._reference = reference_size
        spec = dict(spec or {})
        self._build_ui()
        self._apply_style()
        # 配置里的值可能是旧版本写坏的字符串/None，逐项兜底，不整个崩掉
        for spin, key, fallback in ((self._w, "w", DEFAULT_WINDOW),
                                    (self._h, "h", DEFAULT_WINDOW),
                                    (self._ov, "overlap", DEFAULT_OVERLAP)):
            spin.setValue(_as_int(spec.get(key), fallback))
        self.min_visible_spin.setValue(
            _as_visible_percent(spec.get("min_visible")))
        self._editable = True
        self._refresh_estimate()

    # ---------- 构建 ----------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        body = QHBoxLayout()

        params = QVBoxLayout()

        size_row = QHBoxLayout()
        self.size_label = QLabel(tr("裁剪尺寸（像素）"))
        size_row.addWidget(self.size_label)
        self._w = QSpinBox()
        self._w.setRange(MIN_WINDOW, MAX_WINDOW)
        self._w.setFixedWidth(84)
        self._h = QSpinBox()
        self._h.setRange(MIN_WINDOW, MAX_WINDOW)
        self._h.setFixedWidth(84)
        size_row.addWidget(self._w)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(self._h)
        size_row.addStretch()
        params.addLayout(size_row)

        ov_row = QHBoxLayout()
        self.overlap_label = QLabel(tr("重叠长度（像素）"))
        ov_row.addWidget(self.overlap_label)
        self._ov = QSpinBox()
        self._ov.setRange(0, MAX_WINDOW)
        self._ov.setFixedWidth(84)
        ov_row.addWidget(self._ov)
        ov_row.addStretch()
        params.addLayout(ov_row)

        # 最低可见比例：跨窗合并前丢掉「只露一角」的框。百分比显示，存的是小数。
        # 与数据增强侧模板里的 `min_visible` 是**同一个字段**，方便两边对照。
        vis_row = QHBoxLayout()
        self.min_visible_label = QLabel(tr("最低可见比例"))
        vis_row.addWidget(self.min_visible_label)
        self.min_visible_spin = QSpinBox()
        self.min_visible_spin.setRange(int(MIN_VISIBLE * 100), 100)
        self.min_visible_spin.setSuffix("%")
        self.min_visible_spin.setFixedWidth(84)
        self.min_visible_spin.setValue(int(round(DEFAULT_MIN_VISIBLE * 100)))
        self.min_visible_spin.setToolTip(
            tr("跨窗去重前丢弃可见比例低于此值的框；只影响滑窗模式"))
        vis_row.addWidget(self.min_visible_spin)
        vis_row.addStretch()
        params.addLayout(vis_row)

        self._hint = QLabel(tr("窗口尺寸与模型输入尺寸独立：窗口是原图上的裁剪范围"))
        self._hint.setWordWrap(True)
        self._hint.setObjectName("windowHint")
        params.addWidget(self._hint)

        self._estimate = QLabel("")
        self._estimate.setWordWrap(True)
        self._estimate.setObjectName("windowEstimate")
        params.addWidget(self._estimate)
        params.addStretch()
        body.addLayout(params, 1)

        # 曾经这里有个缩略预览小图，已删除：网格改画在主画布上。
        # 曾经这里还有一排「模板」按钮（与数据增强共用 `crop_templates`），
        # 也已删除：用户要求「推理滑窗不要模板了，也不与数据增强同步模板」。
        # 滑窗参数就是当前输入框里的值，不再被另一处的模板列表影响。

        root.addLayout(body)

        for spin in (self._w, self._h, self._ov, self.min_visible_spin):
            spin.valueChanged.connect(self._refresh_estimate)

    def _apply_style(self):
        theme = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QLabel {{ color: {theme['text_primary']}; }}
            QLabel#windowHint, QLabel#windowEstimate {{ color: {theme['text_secondary']}; }}
            QSpinBox {{
                background-color: {theme['panel_bg']}; color: {theme['text_primary']};
                border: 1px solid {theme['border_color']}; border-radius: 4px;
                padding: 1px 4px;
            }}
        """)

    # ---------- 对外接口 ----------

    def spec(self):
        """引擎口径的窗口字典（`iter_windows()` 直接吃）。"""
        return {"w": self._w.value(), "h": self._h.value(),
                "overlap": self._ov.value(),
                "min_visible": self.min_visible_spin.value() / 100.0}

    def set_spec(self, spec):
        """外部写入参数（如点模板）会**保留**未给出的字段。"""
        spec = dict(spec or {})
        for spin, key, fallback in ((self._w, "w", None),
                                    (self._h, "h", None),
                                    (self._ov, "overlap", None)):
            if key in spec:
                spin.setValue(_as_int(spec.get(key), spin.value()))
        if "min_visible" in spec:
            # 存的是小数（0.5~1.0），界面按百分比；坏值退回默认而不是崩
            pct = _as_visible_percent(spec.get("min_visible"))
            self.min_visible_spin.setValue(pct)
        self._refresh_estimate()

    def set_reference(self, size):
        """参考图尺寸：窗口数预估要用它。"""
        self._reference = size
        self._refresh_estimate()

    def set_editable(self, editable):
        """是否可编辑：只有选中「滑窗」时才允许改参数与点模板。

        禁用而不是仅隐藏——用户要求「未选中滑窗时不能修改点击」。禁用后
        输入框仍显示当前值，用户还能核对「这次会跑多大窗口」。
        """
        self._editable = bool(editable)
        for widget in self._editable_widgets():
            widget.setEnabled(self._editable)
        return self._editable

    def is_editable(self):
        return self._editable

    def spins_enabled(self):
        """各个输入框是否**实际可用**（测试与调用方用来核对禁用生效）。"""
        return all(spin.isEnabled()
                   for spin in (self._w, self._h, self._ov,
                                self.min_visible_spin))

    def _editable_widgets(self):
        return [self._w, self._h, self._ov, self.min_visible_spin]

    def window_count(self):
        """按参考图算出的实际窗口数；没有参考图或参数非法时返回 None。"""
        if not self._reference:
            return None
        try:
            return len(iter_windows(self._reference[0], self._reference[1],
                                    self.spec()))
        except ScopeError:
            return None

    def window_grid(self):
        """(行数, 列数)；与 `iter_windows()` 用同一份窗口数学。"""
        from ...engine.augmenter.crop import normalized_window, window_starts

        iw, ih = int(self._reference[0]), int(self._reference[1])
        cw, ch, ov = normalized_window(iw, ih, self.spec())
        return (len(window_starts(ih, ch, ov)), len(window_starts(iw, cw, ov)))

    def is_valid(self):
        """参数是否可用（无参考图时视为可用：还不知道要跑多少窗）。"""
        return not self._reference or self.window_count() is not None

    def estimate_text(self):
        return self._estimate.text()

    def retranslate(self):
        self.size_label.setText(tr("裁剪尺寸（像素）"))
        self.overlap_label.setText(tr("重叠长度（像素）"))
        self.min_visible_label.setText(tr("最低可见比例"))
        self.min_visible_spin.setToolTip(
            tr("跨窗去重前丢弃可见比例低于此值的框；只影响滑窗模式"))
        self._hint.setText(tr("窗口尺寸与模型输入尺寸独立：窗口是原图上的裁剪范围"))
        self._refresh_estimate()

    # ---------- 内部 ----------

    def _refresh_estimate(self, _value=None):
        """预估窗口数：参数非法就明说，并让宿主禁用确定，非法值不能进任务。

        `changed` 一定要发：主画布上的网格由宿主监听这个信号来重画，漏发
        就会出现「文字说 45 窗、画面画着别的网格」。
        """
        if not self._reference:
            self._estimate.setText(tr("未加载图片，无法预估窗口数"))
            self.changed.emit()
            return
        count = self.window_count()
        if count is None:
            self._estimate.setText(tr("窗口参数非法，请检查尺寸与重叠"))
            self.changed.emit()
            return
        rows, cols = self.window_grid()
        text = f"{tr('每张图')} {rows} × {cols} = {count} {tr('个窗口')}"
        if count > MANY_WINDOWS:
            text += "\n" + tr("窗口数很多，耗时约为全图的 {n} 倍").format(n=count)
        self._estimate.setText(text)
        self.changed.emit()

def _as_int(value, fallback):
    """配置里的整数；坏值（None/字符串/超范围）回退到默认。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _as_visible_percent(value):
    """最低可见比例（存的是 0.5~1.0 的小数）→ 界面用的百分比整数。

    坏值、越界值都退回默认，而不是让整个控件崩掉：这个值来自配置或模板，
    可能是旧版本写的、也可能被人手改坏。
    """
    if value is None:
        return int(round(DEFAULT_MIN_VISIBLE * 100))
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return int(round(DEFAULT_MIN_VISIBLE * 100))
    if ratio != ratio:          # NaN
        return int(round(DEFAULT_MIN_VISIBLE * 100))
    pct = int(round(ratio * 100))
    low, high = int(MIN_VISIBLE * 100), 100
    return max(low, min(high, pct))
