"""图像处理面板：只收集操作项与参数，预览由宿主画在主画布上。

这里不读图、不存图、不显示图像——图片由主界面决定，处理链与操作清单位于
`engine/image_tools.py`。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
)

from ..engine import image_tools as it
from .dialog_helpers import sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager

_SELECTION_MODES = (
    ("无选区", "none"),
    ("矩形", "rect"),
    ("多边形", "polygon"),
)


def _flatten_alpha(image):
    """带 alpha 的图合成到白底。

    处理链只认 RGB；不先合成的话，透明像素经 RGB888 转换会变成黑块。
    """
    if not image.hasAlphaChannel():
        return image
    canvas = QImage(image.width(), image.height(), QImage.Format_RGB32)
    canvas.fill(QColor(255, 255, 255))
    painter = QPainter(canvas)
    painter.drawImage(0, 0, image)
    painter.end()
    return canvas


class ImageToolsPanel(QWidget):
    """导出菜单里的「图像处理」入口，浮在主窗口右侧。"""

    specs_changed = Signal()
    closed = Signal()
    previous_requested = Signal()
    next_requested = Signal()

    def __init__(self, editor, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowCloseButtonHint)
        self._editor = editor
        self.setWindowTitle(tr("图像处理"))
        self.setStyleSheet(ThemeManager.get_dialog_style())
        self.setMinimumWidth(380)
        self.resize(380, 720)

        self._checks = {}
        self._param_widgets = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_operations())
        layout.addWidget(scroll, 1)

        layout.addLayout(self._build_selection_bar())
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        sync_titlebar(self)
        self._enter_region_mode(self.region_mode())

    # ---------- 界面构建 ----------

    def _build_operations(self):
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)

        current_group = None
        group_layout = None
        for key in it.OPERATION_ORDER:
            group, label, params = it.OPERATION_META[key]
            if group != current_group:
                current_group = group
                group_box = QGroupBox(tr(group))
                group_layout = QVBoxLayout(group_box)
                box.addWidget(group_box)
            group_layout.addWidget(self._build_operation_row(key, label, params))
        box.addStretch()
        return holder

    def _build_operation_row(self, key, label, params):
        """一个操作占一行；参数多于两个时换到第二行，避免超出面板宽度。"""
        check = QCheckBox(tr(label))
        check.setFixedWidth(110)
        self._checks[key] = check
        check.stateChanged.connect(
            lambda _state, k=key: self._on_operation_toggled(k))

        row = QWidget()
        if len(params) <= 2:
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(check)
            widgets = [self._make_param_widget(layout, spec) for spec in params]
            layout.addStretch()
        else:
            layout = QVBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            head = QHBoxLayout()
            head.addWidget(check)
            head.addStretch()
            layout.addLayout(head)
            widgets = []
            for start in range(0, len(params), 2):
                line = QHBoxLayout()
                line.setContentsMargins(18, 0, 0, 0)
                for spec in params[start:start + 2]:
                    widgets.append(self._make_param_widget(line, spec))
                line.addStretch()
                layout.addLayout(line)
        self._param_widgets[key] = widgets
        return row

    def _make_param_widget(self, parent_layout, spec):
        """按 meta 的 kind 生成参数控件，返回 (取值名, 控件, 取值函数)。"""
        name, kind = spec[0], spec[1]
        if kind == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(spec[2]))
            widget.stateChanged.connect(self._emit_specs_changed)
            getter = widget.isChecked
        elif kind == "choice":
            widget = QComboBox()
            for option in spec[2]:
                widget.addItem(str(option), option)
            widget.setCurrentIndex(max(0, widget.findData(spec[3])))
            widget.currentIndexChanged.connect(self._emit_specs_changed)
            getter = widget.currentData
        elif kind == "int":
            widget = QSpinBox()
            widget.setRange(int(spec[2]), int(spec[3]))
            widget.setValue(int(spec[4]))
            widget.valueChanged.connect(self._emit_specs_changed)
            getter = widget.value
        else:
            widget = QDoubleSpinBox()
            widget.setRange(float(spec[2]), float(spec[3]))
            widget.setDecimals(2)
            widget.setSingleStep(0.1)
            widget.setValue(float(spec[4]))
            widget.valueChanged.connect(self._emit_specs_changed)
            getter = widget.value
        widget.setFixedWidth(74)
        widget.setFixedHeight(22)
        parent_layout.addWidget(widget)
        return (name, widget, getter)

    def _build_selection_bar(self):
        bar = QHBoxLayout()
        bar.addWidget(QLabel(tr("选区:")))

        self._mode_combo = QComboBox()
        for text, key in _SELECTION_MODES:
            self._mode_combo.addItem(tr(text), key)
        # 打开面板就能直接拉选区：默认停在矩形，别让「无选区」变成隐形门槛
        self._mode_combo.setCurrentIndex(1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # 同一项再点一次也要能重新进入绘制态（画完一处想再画一处）
        self._mode_combo.activated.connect(self._on_mode_activated)
        bar.addWidget(self._mode_combo)

        self._add_btn = QPushButton(tr("＋选区"))
        self._add_btn.clicked.connect(self._on_add_region)
        bar.addWidget(self._add_btn)

        self._clear_btn = QPushButton(tr("清除选区"))
        self._clear_btn.clicked.connect(self.clear_selection)
        bar.addWidget(self._clear_btn)

        self._mask_check = QCheckBox(tr("仅选区内生效"))
        self._mask_check.setChecked(True)
        self._mask_check.stateChanged.connect(self._emit_specs_changed)
        bar.addWidget(self._mask_check)

        bar.addStretch()
        return bar

    # ---------- 对外接口 ----------

    def operation_keys(self):
        return list(it.OPERATION_ORDER)

    def region_mode(self):
        return self._mode_combo.currentData()

    def mask_enabled(self):
        return self._mask_check.isChecked()

    def selection_mode(self):
        """选区实际的形状，取自区域本身；没有区域返回 None。

        下拉框只决定「接下来画什么」，不能决定「已画的算不算数」——两者一致
        过一次就对不上，用户画了选区却不生效正是这么来的。
        """
        region = self._first_region()
        if not isinstance(region, dict):
            return None
        if (region.get('shape_type') or 'rectangle') == 'polygon':
            return "polygon"
        if float(region.get('width', 0.0)) <= 0 or float(region.get('height', 0.0)) <= 0:
            return None
        return "rect"

    def selected_keys(self):
        return [k for k in it.OPERATION_ORDER if self._checks[k].isChecked()]

    def checked_specs(self, width, height):
        """按界面顺序收集勾选的处理项，坐标换算到 width×height 空间。"""
        rect = self._selection_rect(width, height)
        specs = []
        for key in it.OPERATION_ORDER:
            if not self._checks[key].isChecked():
                continue
            params = {name: getter()
                      for name, _widget, getter in self._param_widgets[key]}
            if key == "crop_region":
                if rect is None:
                    continue
                params = {"rect": rect}
            specs.append({"name": key, "params": params})
        return specs

    def selection_points(self):
        """第一处选区的归一化顶点；没有选区返回空列表。"""
        selections = self.selections()
        return selections[0][0] if selections else []

    def set_status(self, text):
        self._status.setText(text)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)

    def clear_selection(self):
        regions = getattr(self._editor, 'image_tools_regions', None)
        if isinstance(regions, list):
            regions.clear()
        canvas = getattr(self._editor, 'canvas', None)
        if canvas is not None:
            canvas.update()
        self._emit_specs_changed()

    # ---------- 选区 ----------

    def regions(self):
        regions = getattr(self._editor, 'image_tools_regions', None)
        return list(regions) if isinstance(regions, list) else []

    def _first_region(self):
        regions = self.regions()
        return regions[0] if regions else None

    def selections(self):
        """每个选区一条 (归一化顶点, 形状)；形状取自区域本身。"""
        image = getattr(self._editor, 'current_background', None)
        if image is None:
            return []
        width, height = image.width(), image.height()
        result = []
        for region in self.regions():
            points = self._region_points(region, width, height)
            mode = 'polygon' if (region.get('shape_type') or 'rectangle') == 'polygon' else 'rect'
            if len(points) >= (3 if mode == 'polygon' else 2):
                result.append((points, mode))
        return result

    def _region_points(self, region, width, height):
        """单个区域 → 归一化顶点。

        矩形只返回对角两点：`selection_mask_from_points` 的 rect 模式只取前两点
        当对角，给四个角会画出一个零高度的蒙版。
        """
        if not isinstance(region, dict) or width <= 0 or height <= 0:
            return []
        if (region.get('shape_type') or 'rectangle') == 'polygon':
            raw = [(float(p[0]), float(p[1])) for p in region.get('points') or []]
        else:
            x = float(region.get('x', 0.0))
            y = float(region.get('y', 0.0))
            w = float(region.get('width', 0.0))
            h = float(region.get('height', 0.0))
            raw = [(x, y), (x + w, y + h)]
        return [(min(1.0, max(0.0, px / width)), min(1.0, max(0.0, py / height)))
                for px, py in raw]

    def _selection_rect(self, width, height):
        """所有选区的外接矩形 → (x, y, w, h)；无效时返回 None。"""
        selections = self.selections()
        if not selections:
            return None
        xs = [p[0] * width for points, _mode in selections for p in points]
        ys = [p[1] * height for points, _mode in selections for p in points]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def _on_mode_changed(self, _index):
        if self._editor is not None:
            self._editor.image_tools_region_mode = self.region_mode()
        self._emit_specs_changed()

    def _on_mode_activated(self, _index):
        self._enter_region_mode(self.region_mode())

    def _on_add_region(self):
        """再画一处选区：画完自动退出绘制态，方便接着拖动已画的那些。"""
        self._enter_region_mode(self.region_mode())

    def _enter_region_mode(self, mode):
        """把画布切到对应的选区绘制态。"""
        editor = self._editor
        if editor is None:
            return
        editor.image_tools_region_mode = mode
        canvas = getattr(editor, 'canvas', None)
        if canvas is None:
            return
        if mode == "rect":
            canvas._start_add_region()
        elif mode == "polygon":
            canvas._start_add_region_polygon()
        else:
            canvas._reset_region_drawing_state()
        canvas.update()

    # ---------- 内部 ----------

    def _on_operation_toggled(self, _key):
        self._update_mask_state()
        self._emit_specs_changed()

    def _update_mask_state(self):
        """几何类操作会改变画布尺寸，无法做选区合成，勾选时禁用蒙版。

        禁用期间把用户原来的勾选记下来，取消几何项后恢复——否则蒙版会被悄悄
        关掉，之后再怎么拉选区都不生效。
        """
        geometric = any(it.is_geometric(k) and self._checks[k].isChecked()
                        for k in it.OPERATION_ORDER)
        if geometric:
            self._mask_wanted = self._mask_check.isChecked()
            self._mask_check.setEnabled(False)
            if self._mask_check.isChecked():
                self._mask_check.setChecked(False)
            return
        self._mask_check.setEnabled(True)
        if getattr(self, '_mask_wanted', True):
            self._mask_check.setChecked(True)

    def _emit_specs_changed(self, *_args):
        self.specs_changed.emit()
