"""图像编辑面板：只收集操作项与参数，预览由宿主画在主画布上。

这里不读图、不存图、不显示图像——图片由主界面决定，处理链与操作清单位于
`engine/image_tools.py`。
"""
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QImage, QPainter
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSpinBox, QDoubleSpinBox, QVBoxLayout, QWidget,
)

from ..engine import image_tools as it
from .dialog_helpers import sync_titlebar
from .icons import SVG_ARROW_LEFT, SVG_ARROW_RIGHT, SVG_EYE, _load_svg_icon
from .i18n import t as tr
from .theme import ThemeManager

_SELECTION_MODES = (
    ("无选区", "none"),
    ("矩形", "rect"),
    ("多边形", "polygon"),
)

# 参数改动写进配置的延迟；连续拖滚动条时只落一次
_PERSIST_DELAY_MS = 800

# 勾选框宽度按当前语言量出来：英文文案明显比中文长，写死宽度会把英文截断。
# 宽度 = 最长文案像素宽 + 复选框方框与间距
_CHECK_TEXT_PADDING = 28

# 保存结果这类一次性提示在状态行停留的时长
_FLASH_MS = 4000


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
    """导出菜单里的「图像编辑」入口，浮在主窗口右侧。"""

    specs_changed = Signal()
    closed = Signal()
    previous_requested = Signal()
    next_requested = Signal()
    # 「按住看原图」按下与松开；保存都交给宿主，面板不碰文件
    compare_started = Signal()
    compare_finished = Signal()
    save_new_requested = Signal()
    save_copy_requested = Signal()

    def __init__(self, editor, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowCloseButtonHint)
        self._editor = editor
        self.setWindowTitle(tr("图像编辑"))
        self.setStyleSheet(ThemeManager.get_dialog_style())
        self.setMinimumWidth(380)
        self.resize(380, 720)

        self._checks = {}
        self._param_widgets = {}

        # 参数改动先攒着，停手再写配置：每转一格就 fsync 一次太贵
        self._persist_timer = QTimer(self)
        self._persist_timer.setSingleShot(True)
        self._persist_timer.setInterval(_PERSIST_DELAY_MS)
        self._persist_timer.timeout.connect(self._persist_params)

        # 状态行分两级：常规状态（预览提示）与一次性提示（保存结果）。
        # 一次性提示显示期间，预览提示只记下不显示，到点再顶上。
        self._status_base = ""
        self._status_key = ""
        self._flash_timer = QTimer(self)
        self._flash_timer.setSingleShot(True)
        self._flash_timer.timeout.connect(self._restore_status)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_operations())
        layout.addWidget(scroll, 1)

        layout.addLayout(self._build_nav_bar())
        layout.addLayout(self._build_selection_bar())
        layout.addLayout(self._build_action_bar())
        # 状态行固定占一行高度：文字出现或消失都不能把上面的按钮顶上去。
        # 放不下时按宽度省略，不换行（换行会把整块布局挤高）。
        self._status = QLabel("")
        self._status.setWordWrap(False)
        self._status.setFixedHeight(self._status.fontMetrics().height() + 2)
        layout.addWidget(self._status)

        sync_titlebar(self)
        self._apply_saved_params()
        self._enter_region_mode(self.region_mode())

    # ---------- 界面构建 ----------

    def _build_operations(self):
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)

        # 分组框与操作勾选框都记下来，切语言时按 key 重设文字
        self._group_boxes = {}
        current_group = None
        group_layout = None
        for key in it.OPERATION_ORDER:
            group, label, params = it.OPERATION_META[key]
            if group != current_group:
                current_group = group
                group_box = QGroupBox(tr(group))
                group_layout = QVBoxLayout(group_box)
                box.addWidget(group_box)
                self._group_boxes[group] = group_box
            group_layout.addWidget(self._build_operation_row(key, label, params))
        box.addStretch()
        return holder

    def _check_width(self):
        """勾选框宽度：按当前语言里最长的一条文案算，英文界面自动变宽。"""
        metrics = QFontMetrics(self.font())
        widest = max(
            (metrics.horizontalAdvance(tr(it.OPERATION_META[key][1]))
             for key in it.OPERATION_ORDER), default=0)
        return widest + _CHECK_TEXT_PADDING

    def _fit_mode_combo(self):
        """选区下拉按最长选项定宽，英文的 Rectangle 比「矩形」宽不少。"""
        metrics = QFontMetrics(self._mode_combo.font())
        widest = max((metrics.horizontalAdvance(self._mode_combo.itemText(i))
                      for i in range(self._mode_combo.count())), default=0)
        self._mode_combo.setMinimumWidth(widest + 44)

    def _build_operation_row(self, key, label, params):
        """一个操作占一行；参数多于两个时换到第二行，避免超出面板宽度。"""
        check = QCheckBox(tr(label))
        check.setFixedWidth(self._check_width())
        self._checks[key] = check
        check.stateChanged.connect(
            lambda _state, k=key: self._on_operation_toggled(k))

        row = QWidget()
        if len(params) <= 2:
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(check)
            # 空白放在勾选框之后、参数框之前：参数框因此贴着面板右边，
            # 中英文勾选框宽度不同也不会让输入框左右漂移
            layout.addStretch()
            widgets = [self._make_param_widget(layout, spec) for spec in params]
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
                line.addStretch()
                for spec in params[start:start + 2]:
                    widgets.append(self._make_param_widget(line, spec))
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
        # 数字右对齐：个位与十位对齐，扫一列就能看出量级
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        parent_layout.addWidget(widget)
        return (name, widget, getter)

    def _build_selection_bar(self):
        bar = QHBoxLayout()
        self._selection_lbl = QLabel(tr("选区:"))
        bar.addWidget(self._selection_lbl)

        self._mode_combo = QComboBox()
        for text, key in _SELECTION_MODES:
            self._mode_combo.addItem(tr(text), key)
        # 打开面板就能直接拉选区：默认停在矩形，别让「无选区」变成隐形门槛
        self._mode_combo.setCurrentIndex(1)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # 同一项再点一次也要能重新进入绘制态（画完一处想再画一处）
        self._mode_combo.activated.connect(self._on_mode_activated)
        bar.addWidget(self._mode_combo)
        self._fit_mode_combo()

        self._add_btn = QPushButton(tr("＋选区"))
        self._add_btn.clicked.connect(self._on_add_region)
        bar.addWidget(self._add_btn)

        self._clear_btn = QPushButton(tr("清除选区"))
        self._clear_btn.clicked.connect(self.clear_selection)
        bar.addWidget(self._clear_btn)

        bar.addStretch()
        return bar

    def _build_nav_bar(self):
        """翻图与对比：图标按钮，省出一行宽度给选区。"""
        bar = QHBoxLayout()

        self._prev_btn = self._icon_button(SVG_ARROW_LEFT, tr("上一张"))
        self._prev_btn.clicked.connect(self.previous_requested)
        bar.addWidget(self._prev_btn)

        self._next_btn = self._icon_button(SVG_ARROW_RIGHT, tr("下一张"))
        self._next_btn.clicked.connect(self.next_requested)
        bar.addWidget(self._next_btn)

        self._compare_btn = self._icon_button(SVG_EYE, tr("按住看原图"))
        self._compare_btn.pressed.connect(self.compare_started)
        self._compare_btn.released.connect(self.compare_finished)
        bar.addWidget(self._compare_btn)

        bar.addStretch()
        return bar

    def _icon_button(self, svg, tooltip):
        """图标按钮：外观与主界面工具栏的图标键一致，文字只放在提示里。"""
        btn = QPushButton("")
        btn.setObjectName("iconBtn")
        btn.setIcon(QIcon(_load_svg_icon(svg, 14)))
        btn.setFixedSize(24, 24)
        btn.setToolTip(tooltip)
        return btn

    def _build_action_bar(self):
        """保存与预设：两行各两个按钮，每个都是完整宽度。"""
        bar = QVBoxLayout()
        bar.setSpacing(6)

        row1 = QHBoxLayout()
        self._save_btn = QPushButton(tr("保存图片"))
        self._save_btn.setToolTip(tr("写进与数据集同级的 exports 目录"))
        self._save_btn.clicked.connect(self.save_new_requested)
        row1.addWidget(self._save_btn)

        self._save_copy_btn = QPushButton(tr("另存为…"))
        self._save_copy_btn.setToolTip(tr("自行选择位置与文件名"))
        self._save_copy_btn.clicked.connect(self.save_copy_requested)
        row1.addWidget(self._save_copy_btn)
        bar.addLayout(row1)

        row2 = QHBoxLayout()
        self._reset_btn = QPushButton(tr("复位"))
        self._reset_btn.clicked.connect(self.reset_all)
        row2.addWidget(self._reset_btn)

        self._preset_save_btn = QPushButton(tr("存为预设"))
        self._preset_save_btn.setToolTip(
            tr("存进预设槽 A~E，数据增强面板直接选用"))
        self._preset_save_btn.clicked.connect(self._on_save_preset)
        row2.addWidget(self._preset_save_btn)
        bar.addLayout(row2)
        return bar

    # ---------- 对外接口 ----------

    def operation_keys(self):
        return list(it.OPERATION_ORDER)

    def region_mode(self):
        return self._mode_combo.currentData()

    def mask_enabled(self):
        """有选区就只在选区内生效，不需要额外的开关。"""
        return bool(self.selections())

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

    def preset_steps(self):
        """当前勾选项的参数快照，用于保存预设。

        裁剪依赖选区坐标，不属于预设能携带的东西，跳过。
        """
        steps = []
        for key in it.OPERATION_ORDER:
            if not self._checks[key].isChecked():
                continue
            if key == "crop_region":
                continue
            params = {name: getter()
                      for name, _widget, getter in self._param_widgets[key]}
            steps.append({"name": key, "params": params})
        return steps

    def reset_all(self):
        """清空勾选并把每个参数退回算子类的默认值。"""
        for key in it.OPERATION_ORDER:
            self._checks[key].blockSignals(True)
            self._checks[key].setChecked(False)
            self._checks[key].blockSignals(False)
            for name, widget, _getter in self._param_widgets[key]:
                params = it.OPERATION_META[key][2]
                default = next((spec[-1] for spec in params if spec[0] == name),
                               None)
                if default is None:
                    continue
                widget.blockSignals(True)
                self._set_param_value(widget, default)
                widget.blockSignals(False)
        self._emit_specs_changed()

    def _apply_saved_params(self):
        """把配置里记住的参数填回控件；勾选状态不恢复。"""
        saved = it.load_saved_params()
        for key, params in saved.items():
            for name, widget, _getter in self._param_widgets.get(key, []):
                if name in params:
                    widget.blockSignals(True)
                    self._set_param_value(widget, params[name])
                    widget.blockSignals(False)

    def _persist_params(self):
        """把每个算子的当前参数写进配置（不看有没有勾选）。"""
        entries = {}
        for key, widgets in self._param_widgets.items():
            params = {name: getter() for name, _widget, getter in widgets}
            if params:
                entries[key] = params
        it.save_saved_params(entries)

    def _set_param_value(self, widget, value):
        """按控件类型写入参数值；类型对不上就原样不动。"""
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index >= 0:
                widget.setCurrentIndex(index)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                widget.setValue(value)

    def _on_save_preset(self):
        """存成预设：自动占 A~E 中第一个空槽，五个满了覆盖最旧的。"""
        steps = self.preset_steps()
        if not steps:
            self.set_status(tr("没有勾选任何处理项"))
            return
        slot = it.store_preset(steps)
        if not slot:
            self.set_status(tr("保存失败"))
            return
        self.set_status(f"{tr('已存为')} {it.preset_name(slot)}")

    def retranslate(self):
        """切语言后重设面板上的全部静态文字（运行中切语言时由主窗口调用）。"""
        self.setWindowTitle(tr("图像编辑"))
        for group, box in getattr(self, '_group_boxes', {}).items():
            box.setTitle(tr(group))
        # 英文文案比中文长，宽度要跟着重算，否则勾选框会被截断
        width = self._check_width()
        for key, check in self._checks.items():
            label = it.OPERATION_META[key][1]
            check.setText(tr(label))
            check.setFixedWidth(width)
        self._selection_lbl.setText(tr("选区:"))
        for index, (text, _key) in enumerate(_SELECTION_MODES):
            self._mode_combo.setItemText(index, tr(text))
        self._fit_mode_combo()
        self._add_btn.setText(tr("＋选区"))
        self._clear_btn.setText(tr("清除选区"))
        self._prev_btn.setToolTip(tr("上一张"))
        self._next_btn.setToolTip(tr("下一张"))
        self._compare_btn.setToolTip(tr("按住看原图"))
        self._save_btn.setText(tr("保存图片"))
        self._save_btn.setToolTip(tr("写进与数据集同级的 exports 目录"))
        self._save_copy_btn.setText(tr("另存为…"))
        self._save_copy_btn.setToolTip(tr("自行选择位置与文件名"))
        self._reset_btn.setText(tr("复位"))
        self._preset_save_btn.setText(tr("存为预设"))
        self._preset_save_btn.setToolTip(tr("存进预设槽 A~E，数据增强面板直接选用"))
        # 状态行按记下的键重译；一次性提示（已生成新图等）不再翻译
        if getattr(self, '_status_key', ''):
            self._status_base = tr(self._status_key)
            self._show_status(self._status_base)

    def set_status(self, text):
        """常规状态（预览提示）：一次性提示还在显示时只记下，到点再顶上。"""
        self._status_key = text or ""
        self._status_base = tr(self._status_key) if self._status_key else ""
        if not self._flash_timer.isActive():
            self._show_status(self._status_base)

    def flash_status(self, text):
        """一次性提示（保存结果等）：立刻显示并保持几秒，不被预览提示冲掉。"""
        self._show_status(text or "")
        self._flash_timer.start(_FLASH_MS)

    def set_status_now(self, text):
        """用户直接操作引起的状态（按住看原图）：立刻显示，撤掉一次性提示。"""
        self._cancel_flash()
        self._status_key = text or ""
        self._status_base = tr(self._status_key) if self._status_key else ""
        self._show_status(self._status_base)

    def _cancel_flash(self):
        """撤掉一次性提示，立即回落到常规状态。"""
        if self._flash_timer.isActive():
            self._flash_timer.stop()
            self._show_status(self._status_base)

    def _restore_status(self):
        self._show_status(self._status_base)

    def _show_status(self, text):
        """状态行：一行高度，过长按宽度省略，不挤压上面的按钮。"""
        metrics = self._status.fontMetrics()
        available = max(self._status.width() - 4, 60)
        self._status.setText(metrics.elidedText(text, Qt.ElideRight, available))

    def closeEvent(self, event):
        # 关窗时把没落盘的参数立刻写掉，不然最后那次改动会丢
        self._persist_timer.stop()
        self._persist_params()
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
        self._emit_specs_changed()

    def geometric_active(self):
        """当前勾选里是否含几何类（会改尺寸，没法做选区合成）。"""
        return any(it.is_geometric(k) and self._checks[k].isChecked()
                   for k in it.OPERATION_ORDER)

    def _emit_specs_changed(self, *_args):
        # 参数一改，上一条保存提示就过期了
        self._cancel_flash()
        self._persist_timer.start()
        self.specs_changed.emit()
