"""Panels UI building mixin."""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame, QPushButton
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QTimer, QRect
from ...canvas import Canvas
from ...widgets.canvas_adjustment import CanvasAdjustmentWidget
from ...widgets.spinner import _SpinnerWidget
from ..i18n import t as tr


class PanelsMixin:
    """Panels widgets and behavior."""

    def _create_splitter(self):
        """创建分割器"""
        from PySide6.QtWidgets import QSizePolicy

        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        # 模型行插在这上面（索引 0），所以先把布局记下来
        self._canvas_layout = canvas_layout
        self.canvas = Canvas(self)
        canvas_scroll = QScrollArea()
        canvas_scroll.setObjectName("canvasScroll")
        canvas_scroll.setWidget(self.canvas)
        canvas_scroll.setWidgetResizable(True)
        canvas_layout.addWidget(canvas_scroll)

        self._loading_spinner = _SpinnerWidget(canvas_scroll.viewport())
        self._loading_spinner.hide()

        self.canvas_adjustment = CanvasAdjustmentWidget(canvas_scroll.viewport())
        self.canvas_adjustment.opacity_changed.connect(self._on_opacity_changed)
        self.canvas_adjustment.brightness_contrast_changed.connect(self._on_brightness_contrast_changed)
        self.canvas_adjustment.geometry_changed.connect(
            lambda: self._position_canvas_adjustment(canvas_scroll)
        )
        canvas_scroll.viewport().installEventFilter(self)
        self._canvas_scroll_area = canvas_scroll
        QTimer.singleShot(0, lambda: self._position_canvas_adjustment(canvas_scroll))

        control_widget = self._create_control_panel()
        control_widget.setFixedWidth(350)

        canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        control_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        container = QWidget()
        container_layout = QHBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(canvas_widget, 1)
        container_layout.addWidget(control_widget, 0)

        return container

    def _position_canvas_adjustment(self, scroll_area):
        viewport = scroll_area.viewport()
        self.canvas_adjustment.adjustSize()
        margin = 10
        height = self.canvas_adjustment.height()
        self.canvas_adjustment.move(
            margin, max(0, viewport.height() - height - margin)
        )
        self.canvas_adjustment.raise_()

    def _side_panel_geometry(self):
        """右侧控制栏相对主窗口的矩形；还没建好时返回 None。

        给「推理范围」弹窗定位用：它要摆在这块地方（悬浮在背景图列表上方），
        而不是屏幕正中——居中会挡住画布。
        """
        widget = getattr(self, '_side_control_widget', None)
        if widget is None or self.width() <= 0:
            return None
        top_left = widget.mapTo(self, widget.rect().topLeft())
        return QRect(top_left, widget.size())

    def _on_opacity_changed(self, value):
        if hasattr(self, 'canvas'):
            self.canvas.shape_opacity = value / 100.0
            self.canvas.update()

    def _on_brightness_contrast_changed(self, brightness, contrast):
        if hasattr(self, 'canvas'):
            self.canvas.apply_brightness_contrast(brightness, contrast)

    def eventFilter(self, obj, event):
        # 枚举写成 `QEvent.Type.Resize`：PySide6 只把枚举挂在类上，实例上取不到
        # （`event.Resize` 会 AttributeError），PyQt5 那套实例访问在这里不成立。
        from PySide6.QtCore import QEvent
        if (hasattr(self, '_canvas_scroll_area') and
                obj == self._canvas_scroll_area.viewport() and
                event.type() == QEvent.Type.Resize):
            self._position_canvas_adjustment(self._canvas_scroll_area)
        return super().eventFilter(obj, event)

    def _create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setSpacing(6)
        # Top 0 matches canvas top edge (sides/bottom keep inset).
        control_layout.setContentsMargins(6, 0, 6, 6)
        self._side_control_layout = control_layout
        self._side_control_widget = control_widget
        self._side_sections = []

        self._create_background_list_section(control_layout)
        self._create_label_list_section(control_layout)
        self._create_small_list_section(control_layout)
        self._update_side_panel_stretches()

        return control_widget

    def _update_side_panel_stretches(self):
        """Equal height among expanded sections; collapsed keep header only."""
        from PySide6.QtWidgets import QSizePolicy

        layout = getattr(self, '_side_control_layout', None)
        sections = getattr(self, '_side_sections', None) or []
        if layout is None or not sections:
            return
        for section in sections:
            outer = section['outer']
            content = section['content']
            is_open = getattr(section['header'], '_expanded', True)
            content.setVisible(is_open)
            if is_open:
                outer.setMinimumHeight(0)
                outer.setMaximumHeight(16777215)
                outer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
                layout.setStretchFactor(outer, 1)
            else:
                outer.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                header_h = section['header'].sizeHint().height()
                margins = outer.layout().contentsMargins() if outer.layout() else None
                extra = (margins.top() + margins.bottom()) if margins else 12
                outer.setFixedHeight(max(28, header_h + extra))
                layout.setStretchFactor(outer, 0)

    def _make_side_collapsible(self, title_key):
        """Create a collapsible side section with arrow on the left of the title."""
        outer = QFrame()
        outer.setObjectName("sideCollapsible")
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(6, 6, 6, 6)
        outer_layout.setSpacing(4)

        header = QPushButton(f"▼  {tr(title_key)}")
        header.setFlat(True)
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet(
            "border: none; text-align: left; font-weight: bold; padding: 2px 0;"
        )
        header.setFixedHeight(22)
        header.setProperty("title_key", title_key)
        header._expanded = True
        outer_layout.addWidget(header)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        outer_layout.addWidget(content, 1)

        def _toggle():
            header._expanded = not getattr(header, '_expanded', True)
            key = header.property("title_key") or title_key
            header.setText(f"{'▼' if header._expanded else '▶'}  {tr(key)}")
            self._update_side_panel_stretches()

        header.clicked.connect(_toggle)
        if not hasattr(self, '_side_sections'):
            self._side_sections = []
        self._side_sections.append({
            'outer': outer,
            'content': content,
            'header': header,
        })
        return outer, content_layout, header
