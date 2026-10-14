"""滑窗推理的窗口参数配置：窗口宽高、重叠像素、原图叠加预览、模板。

预览是原图缩略 + 窗口网格叠加（`WindowOverlayPreview`），不是 `crop_dialog`
的分离小图：推理的窗口与增强裁剪的「空窗丢弃」口径不同（推理必须跑每一个
窗口），分离小图在大图/多窗时也辨认不出落位。
窗口数量以 `iter_windows()` 的实际结果为准，不用 `size_for_count()` 反推
——后者取整后可能多出一窗。

模板与数据增强**共用同一份** `crop_templates`（双向同步，最多 3 条）。
点击模板只回填输入框并刷新预览，**不生效**：与面板的确定/取消语义一致，
必须点「确定」才写回范围。弹窗自身不写模板——写模板由调用方在确定之后做。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout,
)

from ..core.config_manager import load_crop_templates
from ..engine.augmenter.crop import MAX_WINDOW
from ..engine.inference_scope import ScopeError, iter_windows
from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager
from .widgets.window_overlay_preview import WindowOverlayPreview

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
        # 预览要等参数填好再设图与参数，否则会画一次 640 默认值的网格
        self.preview.set_image(reference_size)
        self._refresh_estimate()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        body = QHBoxLayout()
        params = QVBoxLayout()

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel(tr("裁剪尺寸（像素）")))
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
        ov_row.addWidget(QLabel(tr("重叠长度（像素）")))
        self._ov = QSpinBox()
        self._ov.setRange(0, MAX_WINDOW)
        self._ov.setFixedWidth(84)
        ov_row.addWidget(self._ov)
        ov_row.addStretch()
        params.addLayout(ov_row)

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

        # 原图叠加预览：窗口落在原图哪里，一眼可见。
        # 尺寸在 __init__ 里填（要等 _w/_h/_ov 建好才能取 spec()）。
        self.preview = WindowOverlayPreview()
        self.preview.setMinimumWidth(220)
        body.addWidget(self.preview, 1)

        templates = QVBoxLayout()
        self._template_buttons = []
        records = load_crop_templates()
        if records:
            templates.addWidget(QLabel(tr("模板")))
            for item in records:
                # 与 crop_dialog 用同一个 `template_fmt`：两份模板列表是同一
                # 份数据，标签格式该一致，也省得再添一条中英词条。
                txt = tr("template_fmt").format(
                    w=item["w"], h=item["h"], ov=item["overlap"],
                    vis=int(round(item["min_visible"] * 100)))
                btn = QPushButton(txt)
                btn.clicked.connect(
                    lambda _c=False, spec=item: self._apply_template(spec))
                self._template_buttons.append(btn)
                templates.addWidget(btn)
        templates.addStretch()
        body.addLayout(templates, 0)

        layout.addLayout(body)

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
        """预估窗口数：参数非法就明说，并禁用确定，不能让非法值进任务。

        预览与这段文字同时更新：两者都是「当前参数的后果」，任何一条路径漏掉
        更新，用户就会看到文字说 45 窗、画面画着别的网格。
        """
        self.preview.set_spec(self.spec())
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

    # ---------- 模板 ----------

    def template_buttons(self):
        """模板按钮列表（无模板时为空表）；点击即回填参数并刷预览。"""
        return list(self._template_buttons)

    def _apply_template(self, spec):
        """回填模板的 w/h/重叠到输入框并刷新预览；不改变确定/取消语义。

        只搬三个推理用得到的字段：`crop_templates` 里还有 `min_visible` /
        `square` / `fill` / `fill_rgb`，那些是数据增强侧的口径，推理侧不读。
        回填后输入框仍可编辑——模板只是起点，不是锁。
        """
        for spin, key, fallback in ((self._w, "w", DEFAULT_WINDOW),
                                    (self._h, "h", DEFAULT_WINDOW),
                                    (self._ov, "overlap", DEFAULT_OVERLAP)):
            spin.blockSignals(True)
            spin.setValue(_as_int(spec.get(key), fallback))
            spin.blockSignals(False)
        self._refresh_estimate()


def _as_int(value, fallback):
    """配置里的整数；坏值（None/字符串/超范围）回退到默认。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)
