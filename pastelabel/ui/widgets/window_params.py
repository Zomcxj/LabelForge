"""滑窗参数控件：宽 / 高 / 重叠 + 窗口数预估 + 原图网格预览 + 模板按钮。

抽成控件而不是留在某个对话框里，是因为它有两个宿主：
- `ScopePanel` 把它**内嵌**在范围弹窗里（用户要求：点「滑窗」直接看到参数与
  原图网格，不再叠开第二个窗口）；
- `SlidingWindowDialog` 仍单独存在，供需要独立窗口的路径使用。

窗口数一律以 `iter_windows()` 的实际结果为准，不用 `size_for_count()` 反推
——后者取整后可能多出一窗。非法参数的口径也与引擎一致（`重叠 >= 窗口尺寸`
不算错，引擎会钳；真失败是尺寸越界/非正/超 `MAX_WINDOWS`）。

模板与数据增强**共用同一份** `crop_templates`（双向同步，最多 3 条）。点击
模板只回填输入框并刷新预览，**不生效**：必须由宿主的「确定」才写回。控件
自身不写模板——写模板由调用方在确定之后做。

预览画**真实原图**（宿主传入 `QPixmap`）再叠网格，让用户看清窗口落在画面
的哪个部分；拿不到图时退回尺寸占位并给文字提示。
"""
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from ...core.config_manager import load_crop_templates
from ...engine.augmenter.crop import MAX_WINDOW
from ...engine.inference_scope import ScopeError, iter_windows
from ..i18n import t as tr
from ..theme import ThemeManager
from .window_overlay_preview import WindowOverlayPreview

DEFAULT_WINDOW = 640
DEFAULT_OVERLAP = 200
MIN_WINDOW = 8                  # 与 crop_dialog 的窗口下限一致
MANY_WINDOWS = 1000             # 超过就提醒一句：每张图要跑这么多窗


class WindowParamsWidget(QWidget):
    """窗口参数 + 预估 + 原图网格预览 + 模板按钮。"""

    # 参数或预览发生变化（宿主据此重画/校验）；也让宿主能挂自己的联动
    changed = pyqtSignal()

    def __init__(self, reference_size=None, spec=None, parent=None):
        super().__init__(parent)
        self._reference = reference_size
        self._template_buttons = []
        spec = dict(spec or {})
        self._build_ui()
        self._apply_style()
        # 配置里的值可能是旧版本写坏的字符串/None，逐项兜底，不整个崩掉
        for spin, key, fallback in ((self._w, "w", DEFAULT_WINDOW),
                                    (self._h, "h", DEFAULT_WINDOW),
                                    (self._ov, "overlap", DEFAULT_OVERLAP)):
            spin.setValue(_as_int(spec.get(key), fallback))
        # 预览要等参数填好再设，否则会画一次 640 默认值的网格
        self.preview.set_image(reference_size)
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
        # 图与参数在 __init__ 里填（要等 _w/_h/_ov 建好才能取 spec()）。
        self.preview = WindowOverlayPreview()
        self.preview.setMinimumWidth(220)
        body.addWidget(self.preview, 1)

        self._templates = QVBoxLayout()
        body.addLayout(self._templates, 0)
        self._rebuild_templates()

        root.addLayout(body)

        for spin in (self._w, self._h, self._ov):
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

    def _rebuild_templates(self):
        """按当前配置重建模板按钮（配置在别处被改过时也要能刷到）。"""
        while self._templates.count():
            item = self._templates.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._template_buttons = []
        records = load_crop_templates()
        if records:
            self.templates_label = QLabel(tr("模板"))
            self._templates.addWidget(self.templates_label)
        else:
            self.templates_label = None
        for item in records:
            # 与 crop_dialog 用同一个 `template_fmt`：两份模板列表是同一份
            # 数据，标签格式该一致，也省得再添一条中英词条。
            txt = tr("template_fmt").format(
                w=item["w"], h=item["h"], ov=item["overlap"],
                vis=int(round(item["min_visible"] * 100)))
            btn = QPushButton(txt)
            btn.setObjectName("templateBtn")
            btn.clicked.connect(
                lambda _c=False, spec=item: self._apply_template(spec))
            self._template_buttons.append(btn)
            self._templates.addWidget(btn)
        self._templates.addStretch()

    # ---------- 对外接口 ----------

    def spec(self):
        """引擎口径的窗口字典（`iter_windows()` 直接吃）。"""
        return {"w": self._w.value(), "h": self._h.value(),
                "overlap": self._ov.value()}

    def set_spec(self, spec):
        """外部写入参数（如点模板）会**保留**未给出的字段。"""
        spec = dict(spec or {})
        for spin, key, fallback in ((self._w, "w", None),
                                    (self._h, "h", None),
                                    (self._ov, "overlap", None)):
            if key in spec:
                spin.setValue(_as_int(spec.get(key), spin.value()))
        self._refresh_estimate()

    def set_reference(self, size):
        """参考图尺寸与图：窗口数预估与预览都要用。"""
        self._reference = size
        self.preview.set_image(size)
        self._refresh_estimate()

    def set_pixmap(self, pixmap):
        """当前背景图（QPixmap）；预览把它画在网格底下。"""
        self.preview.set_pixmap(pixmap)

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

    def template_buttons(self):
        """模板按钮列表（无模板时为空表）；点击即回填参数并刷预览。"""
        return list(self._template_buttons)

    def refresh_templates(self):
        self._rebuild_templates()

    def retranslate(self):
        self.size_label.setText(tr("裁剪尺寸（像素）"))
        self.overlap_label.setText(tr("重叠长度（像素）"))
        self._hint.setText(tr("窗口尺寸与模型输入尺寸独立：窗口是原图上的裁剪范围"))
        if self.templates_label is not None:
            self.templates_label.setText(tr("模板"))
        self._refresh_estimate()

    # ---------- 内部 ----------

    def _refresh_estimate(self, _value=None):
        """预估窗口数：参数非法就明说，并让宿主禁用确定，非法值不能进任务。

        预览与这段文字同时更新：两者都是「当前参数的后果」，任何一条路径漏掉
        更新，用户就会看到文字说 45 窗、画面画着别的网格。
        """
        self.preview.set_spec(self.spec())
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
