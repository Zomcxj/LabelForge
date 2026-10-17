"""Theme and title-bar behavior for the main window."""

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ..dwm import set_titlebar_dark
from ..theme import ThemeManager
from ..icons import MOON_SVG, SUN_SVG, _load_svg_icon


class ThemeMixin:
    def showEvent(self, event):
        """窗口显示后设置标题栏颜色（winId 必须在 show 之后获取）"""
        super().showEvent(event)
        from PySide6.QtCore import QTimer
        is_dark = ThemeManager.get_mode().value == "dark"
        QTimer.singleShot(30, lambda: self._set_titlebar_dark(is_dark))

    def _set_titlebar_dark(self, dark, force_refresh=False):
        """设置系统标题栏颜色"""
        hwnd = int(self.winId())
        set_titlebar_dark(hwnd, dark, force_refresh=force_refresh)

    def _sync_all_titlebars(self, dark, force_refresh=False):
        """同步所有已创建顶层窗口的系统标题栏颜色。"""
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if not widget.isWindow():
                continue
            try:
                set_titlebar_dark(int(widget.winId()), dark, force_refresh=force_refresh)
            except Exception as e:
                from ...core.exception_hook import _write_log
                _write_log(f"标题栏深色同步失败 {type(widget).__name__}: {e}")

    def _apply_app_palette(self):
        """同步 Qt 调色板，补足 Win10 原生控件/窗口背景刷新。"""
        app = QApplication.instance()
        if app is None:
            return
        try:
            from PySide6.QtGui import QPalette, QColor
        except ImportError:
            return
        t = ThemeManager.get_theme()
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(t['window_bg']))
        palette.setColor(QPalette.WindowText, QColor(t['text_primary']))
        palette.setColor(QPalette.Base, QColor(t['widget_bg']))
        palette.setColor(QPalette.AlternateBase, QColor(t['panel_bg']))
        palette.setColor(QPalette.Text, QColor(t['text_primary']))
        palette.setColor(QPalette.Button, QColor(t['widget_bg']))
        palette.setColor(QPalette.ButtonText, QColor(t['text_primary']))
        palette.setColor(QPalette.Highlight, QColor(t['accent']))
        palette.setColor(QPalette.HighlightedText, QColor(t['widget_bg']))
        palette.setColor(QPalette.ToolTipBase, QColor(t['tooltip_bg']))
        palette.setColor(QPalette.ToolTipText, QColor(t['tooltip_text']))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(t['text_disabled']))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(t['text_disabled']))
        app.setPalette(palette)

    def _apply_theme(self):
        """应用当前主题样式"""
        app = QApplication.instance()
        self._apply_app_palette()
        app.setStyleSheet(ThemeManager.get_stylesheet())
        for widget in app.topLevelWidgets():
            try:
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
            except Exception as e:
                from ...core.exception_hook import _write_log
                _write_log(f"主题刷新控件失败 {type(widget).__name__}: {e}")
        if hasattr(self, 'theme_btn'):
            is_dark = ThemeManager.get_mode().value == "dark"
            svg = MOON_SVG if is_dark else SUN_SVG
            icon = QIcon(_load_svg_icon(svg, 16, "#D4AF37"))
            self.theme_btn.setIcon(icon)
        if hasattr(self, 'prefix_input'):
            has_text = bool(self.prefix_input.text().strip())
            self.prefix_input.setProperty("placeholder", not has_text)
            self.prefix_input.style().unpolish(self.prefix_input)
            self.prefix_input.style().polish(self.prefix_input)
        self._update_mode_seg_style()
        if hasattr(self, 'canvas'):
            self.canvas.update()
        if hasattr(self, 'canvas_adjustment'):
            self.canvas_adjustment.refresh_theme()
        # 推理参数行与进度小窗自带样式表，不会随全局样式刷新
        for name in ('model_row', '_progress_dialog', '_class_filter_panel'):
            widget = getattr(self, name, None)
            if widget is not None and hasattr(widget, 'refresh_theme'):
                widget.refresh_theme()
        app.processEvents()

    def _update_status_info(self):
        """更新状态栏信息"""
        info = self.get_image_info()
        if info:
            stats = self.get_label_stats()
            stats_text = " | ".join([f"{k}:{v}" for k, v in list(stats.items())[:3]])
            self.status_label.setText(
                f"Box: {info['box_count']} Paste: {info['paste_count']}"
                + (f" | {stats_text}" if stats_text else "")
            )

    def toggle_theme(self):
        """切换主题"""
        from ...core import config_manager
        ThemeManager.toggle()
        self._apply_theme()
        self._update_mode_seg_style()
        is_dark = ThemeManager.get_mode().value == "dark"
        self._sync_all_titlebars(is_dark, force_refresh=True)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(80, lambda: self._sync_all_titlebars(is_dark, force_refresh=True))
        config_manager.save_theme('dark' if is_dark else 'light')
        self.status_label.setText(f"Theme: {'Dark' if is_dark else 'Light'}")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))
