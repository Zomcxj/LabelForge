"""
Theme module - manages dark/light theme switching
"""
import os
from enum import Enum


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


# 界面主字体。JetBrains Mono 排在第一位（用户指定的观感），中文由
# Microsoft YaHei 承接——JetBrains Mono 没有中文字形，系统会自动回落。
UI_FONT_FAMILY = "'JetBrains Mono', 'Microsoft YaHei', 'Segoe UI', sans-serif"
# 等宽字体只留给坐标、阈值这类需要对齐的数字。
MONO_FONT_FAMILY = "'JetBrains Mono', 'Consolas', 'Courier New', monospace"

# 字号以「100% 缩放下多少 px」书写，渲染时按屏幕 DPI 换算。
# QSS 里的 px 不随系统缩放走，12px 在 150% 缩放的笔记本上会显得很小。
_BASE_FONT_SIZE = 12
_BASE_FONT_SIZE_SMALL = 11


def _scale_px(base: int) -> int:
    """把 100% 缩放下的字号换算成当前 DPI 下的 px。

    只用逻辑 DPI（logicalDotsPerInch）而不是物理 DPI：Qt 已经开了
    AA_EnableHighDpiScaling，逻辑 DPI 里已经含了系统缩放比例，再乘物理
    DPI 会放大两次。拿不到屏幕时（无 GUI、离屏）原样返回。
    """
    scale = float(os.environ.get("LABELFORGE_UI_SCALE") or 0) or None
    if scale is None:
        try:
            from PySide6.QtGui import QGuiApplication
            screen = QGuiApplication.primaryScreen()
            scale = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        except Exception:
            scale = 1.0
    # 下限 9px：再小就看不清；上限 3 倍够覆盖到 300% 缩放。
    return max(9, min(36, round(base * min(max(scale, 1.0), 3.0))))


LIGHT_THEME = {
    "name": "light",
    "font_family": UI_FONT_FAMILY,
    "font_mono": MONO_FONT_FAMILY,
    "font_size": _BASE_FONT_SIZE,
    "font_size_small": _BASE_FONT_SIZE_SMALL,
    "font_size_px": _scale_px(_BASE_FONT_SIZE),
    "font_size_px_small": _scale_px(_BASE_FONT_SIZE_SMALL),
    "interaction_active": "#2950ff",
    "interaction_active_text": "#FFFFFF",
    "window_bg": "#F5F5F5",
    "widget_bg": "#FFFFFF",
    "panel_bg": "#FAFAFA",
            "canvas_bg": "#FFFFFF",
    "border_color": "#E0E0E0",
    "border_hover": "#BBDEFB",
    "border_input": "#C4C4C4",
    "text_primary": "#1F1F1F",
    "button_text": "#1F1F1F",
    "text_secondary": "#1F1F1F",
    # 禁用文字也要读得清：用户得看懂这一项是什么，才能判断要不要启用它。
    # 用虚线边框表达「不可用」，不用把字压到看不见。
    "text_disabled": "#8A8A8A",
    "accent": "#2196F3",
    "accent_hover": "#1976D2",
    "accent_light": "#E3F2FD",
    "success": "#4CAF50",
    "success_light": "#E8F5E9",
    "warning": "#FF9800",
    "warning_light": "#FFF3E0",
    "danger": "#F44336",
    "danger_light": "#FFEBEE",
    "list_hover": "#F5F5F5",
    "list_selected_bg": "#E3F2FD",
    "list_selected_text": "#1976D2",
    "list_border": "#F0F0F0",
    "toolbar_bg": "#FFFFFF",
    "statusbar_bg": "#FAFAFA",
    "scrollbar_bg": "#F0F0F0",
    "scrollbar_handle": "#BDBDBD",
    "scrollbar_handle_hover": "#9E9E9E",
    "groupbox_bg": "#FFFFFF",
    "groupbox_border": "#E0E0E0",
    "tooltip_bg": "#616161",
    "tooltip_text": "#FFFFFF",
    "surface_secondary": "#F0F0F0",
}

DARK_THEME = {
    "name": "dark",
    "font_family": UI_FONT_FAMILY,
    "font_mono": MONO_FONT_FAMILY,
    "font_size": _BASE_FONT_SIZE,
    "font_size_small": _BASE_FONT_SIZE_SMALL,
    "font_size_px": _scale_px(_BASE_FONT_SIZE),
    "font_size_px_small": _scale_px(_BASE_FONT_SIZE_SMALL),
    "interaction_active": "#2950ff",
    "interaction_active_text": "#FFFFFF",
    "window_bg": "#1E1E1E",
    "widget_bg": "#2D2D2D",
    "panel_bg": "#252525",
    "canvas_bg": "#2D2D2D",
    # 边框对 widget_bg 只有 1.29:1 时基本看不见；WCAG 1.4.11 要求
    # 可交互组件的边界至少 3:1。border_color 用于装饰分隔，
    # border_input 用于输入框/复选框这类必须看得见的边界。
    "border_color": "#505050",
    "border_hover": "#5E5E5E",
    "border_input": "#757575",
    "text_primary": "#F5F5F5",
    "button_text": "#F5F5F5",
    "text_secondary": "#F5F5F5",
    # 原 #616161 对 widget_bg 只有 2.22:1，禁用态的字读不清，
    # 用户没法判断那个选项到底是什么。提到 #969696（对 widget_bg 4.66:1）。
    "text_disabled": "#969696",
    "accent": "#42A5F5",
    "accent_hover": "#1E88E5",
    "accent_light": "#1A2332",
    "success": "#66BB6A",
    "success_light": "#1A2E1A",
    "warning": "#FFA726",
    "warning_light": "#2E2218",
    # 原 #EF5350 在 widget_bg 上只有 3.95:1，当错误提示文字用时不够。
    "danger": "#FF6B6B",
    "danger_light": "#2E1A1A",
    "list_hover": "#333333",
    "list_selected_bg": "#1A2332",
    "list_selected_text": "#42A5F5",
    "list_border": "#3E3E3E",
    "toolbar_bg": "#252525",
    "statusbar_bg": "#1E1E1E",
    "scrollbar_bg": "#2D2D2D",
    "scrollbar_handle": "#555555",
    "scrollbar_handle_hover": "#777777",
    "groupbox_bg": "#2D2D2D",
    "groupbox_border": "#3E3E3E",
    "tooltip_bg": "#424242",
    "tooltip_text": "#E0E0E0",
    "surface_secondary": "#383838",
}


class ThemeManager:
    _instance = None
    _current_theme = LIGHT_THEME
    _mode = ThemeMode.LIGHT

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_theme(cls):
        return cls._current_theme

    @classmethod
    def get_theme_colors(cls):
        return cls._current_theme

    @classmethod
    def get_mode(cls):
        return cls._mode

    @classmethod
    def set_mode(cls, mode):
        cls._mode = mode
        if mode == ThemeMode.DARK:
            cls._current_theme = DARK_THEME
        else:
            cls._current_theme = LIGHT_THEME

    @classmethod
    def toggle(cls):
        if cls._mode == ThemeMode.LIGHT:
            cls.set_mode(ThemeMode.DARK)
        else:
            cls.set_mode(ThemeMode.LIGHT)
        return cls._current_theme

    @classmethod
    def get_stylesheet(cls):
        """整份 QSS，按界面区域拼装；每段见对应的 _qss_* 方法。

        原先是一个 600 多行的 f-string，改一处样式得在整份里翻。
        这里只做切分：各段的顺序与内容都不变，顺序本身有语义
        （同名规则后者胜），重排会改变外观。
        """
        t = cls._current_theme
        parts = (
            cls._qss_base_widgets(t),
            cls._qss_button_base(t),
            cls._qss_containers(t),
            cls._qss_toolbar_buttons(t),
            cls._qss_menus(t),
            cls._qss_list_buttons(t),
            cls._qss_action_buttons(t),
            cls._qss_toolbar(t),
            cls._qss_overrides(t),
        )
        return "\n" + "\n".join(parts) + "\n        "

    @classmethod
    def _qss_base_widgets(cls, t):
        """基础控件：窗口、滚动条、状态栏、标签、列表、输入框、复选框。"""
        return f"""\
            QMainWindow {{
                background-color: {t['window_bg']};
            }}
            QWidget {{
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
                font-family: {t['font_family']};
                font-size: {t['font_size_px']}px;
            }}
            QSplitter {{
                background-color: {t['widget_bg']};
            }}
            QSplitter::handle {{
                background-color: {t['border_color']};
                width: 2px;
            }}
            QScrollBar:vertical {{
                background: transparent;
                width: 5px;
            }}
            QScrollBar::handle:vertical {{
                background: rgba(128, 128, 128, 150);
                border-radius: 2px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: rgba(96, 96, 96, 220);
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
            QScrollBar:horizontal {{
                background: transparent;
                height: 5px;
            }}
            QScrollBar::handle:horizontal {{
                background: rgba(128, 128, 128, 150);
                border-radius: 2px;
                min-width: 20px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: rgba(96, 96, 96, 220);
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
                background: transparent;
            }}
            QStatusBar {{
                background-color: {t['statusbar_bg']};
                color: {t['text_secondary']};
                border-top: 1px solid {t['border_color']};
            }}
            QLabel {{
                background-color: transparent;
                color: {t['text_primary']};
            }}
            QLabel#statusLabel {{
                color: {t['text_secondary']};
            }}
            QLabel#shortcutStatusLabel {{
                padding: 2px 8px;
            }}
            QLabel#versionLabel {{
                color: {t['text_secondary']};
                padding-left: 2px;
            }}
            QListWidget {{
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
                outline: none;
                padding: 2px;
                alternate-background-color: {t['widget_bg']};
            }}
            QListWidget:disabled {{
                background-color: {t['scrollbar_bg']};
                color: {t['text_disabled']};
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {t['list_border']};
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
            }}
            QListWidget::item:hover {{
                background-color: {t['list_hover']};
            }}
            QListWidget::item:selected {{
                background-color: {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QLineEdit {{
                border: 1px solid {t['border_input']};
                border-radius: 4px;
                padding: 3px 6px;
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
                selection-background-color: {t['accent']};
            }}
            QLineEdit:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QLineEdit:focus {{
                border-color: {t['accent']};
            }}
            QSpinBox, QDoubleSpinBox {{
                border: 1px solid {t['border_input']};
                border-radius: 4px;
                padding: 3px 2px;
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border-color: {t['accent']};
            }}
            QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
            QComboBox:disabled {{
                color: {t['text_disabled']};
                border: 1px dashed {t['border_input']};
            }}
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 16px;
                border: none;
                background-color: transparent;
            }}
            QCheckBox {{
                spacing: 6px;
                color: {t['text_primary']};
                background-color: transparent;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 2px solid {t['border_input']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {t['accent']};
                border-color: {t['accent']};
            }}
            QCheckBox::indicator:hover {{
                border-color: {t['accent']};
            }}
            /* 列表项的勾选框由样式自己画，不继承 QCheckBox::indicator。不显式
               给规则的话勾会走平台原生样式（Windows 下是缩在左上角、溢出色块
               的小勾），和旁边的复选框完全不是一个样子，暗色主题下几乎看不出
               勾没勾上。 */
            QListWidget::indicator, QListView::indicator {{
                width: 14px;
                height: 14px;
                border: 2px solid {t['border_color']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
            }}
            QListWidget::indicator:checked, QListView::indicator:checked {{
                background-color: {t['accent']};
                border-color: {t['accent']};
            }}
            QListWidget::indicator:hover, QListView::indicator:hover {{
                border-color: {t['accent']};
            }}"""

    @classmethod
    def _qss_button_base(cls, t):
        """QPushButton 的默认外观，含 :hover / :pressed / :disabled。"""
        return f"""\
            QPushButton {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                color: {t['button_text']};
                font-weight: bold;
            }}
            QPushButton:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton:disabled {{
                background-color: {t['scrollbar_bg']};
                color: {t['text_disabled']};
                border: 1px dashed {t['border_input']};
            }}"""

    @classmethod
    def _qss_containers(cls, t):
        """日志框、分组框、侧栏、提示、滚动区、下拉框。"""
        return f"""\
            QTextEdit#logArea {{
                border: 1px solid {t['groupbox_border']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
                color: {t['text_secondary']};
                font-family: {t['font_mono']};
                font-size: {t['font_size_px']}px;
                padding: 4px;
            }}
            QGroupBox {{
                border: 1px solid {t['groupbox_border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: bold;
                color: {t['text_primary']};
                background-color: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: {t['text_primary']};
            }}
            QFrame#sideCollapsible {{
                border: 1px solid {t['groupbox_border']};
                border-radius: 4px;
                background-color: transparent;
            }}
            QToolTip {{
                background-color: {t['tooltip_bg']};
                color: {t['tooltip_text']};
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
            }}
            QFrame {{
                background-color: transparent;
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollArea#canvasScroll {{
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
            }}
            QComboBox {{
                border: 1px solid {t['border_input']};
                border-radius: 4px;
                padding: 3px 6px;
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
            }}
            QComboBox:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QComboBox::drop-down {{
                border: none;
                background-color: transparent;
            }}
            QComboBox QAbstractItemView {{
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
                border: 1px solid {t['border_color']};
                selection-background-color: {t['list_selected_bg']};
                selection-color: {t['list_selected_text']};
            }}
"""

    @classmethod
    def _qss_toolbar_buttons(cls, t):
        """工具栏上的具名按钮：图标、翻页、背景/贴图、标签、主题、设置、选项。"""
        return f"""\
            /* objectName specific styles */
            QPushButton#iconBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
            }}
            QPushButton#iconBtn:hover {{
                background-color: {t['accent_light']};
            }}
            QPushButton#iconBtn:pressed {{
                background-color: {t['accent']};
            }}
            QPushButton#navBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                font-size: 10px;
                color: {t['text_secondary']};
                /* 必须显式归零：不写 padding 时 Qt 走平台默认内边距
                   （每侧约 5px），再算上菜单箭头，22px 的按钮只剩 4px
                   内容宽度——「T」「G」和 14px 的图标全被裁掉。
                   归零后内容可用宽 = 按钮宽 − 2，22px 也放得下。 */
                padding: 0px;
            }}
            QPushButton#navBtn:hover {{
                background-color: {t['accent_light']};
                color: {t['button_text']};
            }}
            QPushButton#navBtn:pressed {{
                background-color: {t['accent']};
                color: {t['widget_bg']};
            }}
            QPushButton#bgBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
            }}
            QPushButton#bgBtn:hover {{
                background-color: #E3F2FD;
            }}
            QPushButton#bgBtn:pressed {{
                background-color: #BBDEFB;
            }}
            QPushButton#pasteBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
            }}
            QPushButton#pasteBtn:hover {{
                background-color: #E8F5E9;
            }}
            QPushButton#pasteBtn:pressed {{
                background-color: #C8E6C9;
            }}
            QPushButton#labelBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
            }}
            QPushButton#labelBtn:hover {{
                background-color: #FFF3E0;
            }}
            QPushButton#labelBtn:pressed {{
                background-color: #FFE0B2;
            }}
            QPushButton#themeBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
                font-size: 16px;
            }}
            QPushButton#themeBtn:hover {{
                background-color: {t['accent_light']};
            }}

            /* 设置按钮 */
            QPushButton#settingsBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 4px;
                font-size: 16px;
            }}
            QPushButton#settingsBtn:hover {{
                background-color: {t['accent_light']};
            }}

            /* 选项按钮 */
            QPushButton#optionsBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#optionsBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#optionsBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#optionsBtn:checked {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#optionsBtn:checked:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#optionsBtn::menu-indicator {{
                width: 0px;
                height: 0px;
            }}"""

    @classmethod
    def _qss_menus(cls, t):
        """选项菜单本体与菜单项。"""
        return f"""\
            QMenu#optionsMenu {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
                border-radius: 4px;
            }}
            QMenu#optionsMenu::item {{
                padding: 6px 16px;
            }}
            QMenu#optionsMenu::item:selected {{
                background-color: {t['accent_light']};
                color: {t['accent']};
            }}"""

    @classmethod
    def _qss_list_buttons(cls, t):
        """列表页与弹窗里的语言、绘制、强调、警告、贴图列表按钮。"""
        return f"""\
            QPushButton#langBtn {{
                background-color: transparent;
                border: none;
                border-radius: 4px;
                padding: 2px;
                font-size: 11px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#langBtn:hover {{
                background-color: {t['accent_light']};
                color: {t['button_text']};
            }}
            QPushButton#drawBoxBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: 600;
                color: {t['button_text']};
            }}
            QPushButton#drawBoxBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#drawBoxBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#accentBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#accentBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#accentBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#warningBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#warningBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#warningBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#pasteListAccentBtn {{
                background-color: {t['accent_light']};
                border: none;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['accent']};
            }}
            QPushButton#pasteListAccentBtn:hover {{
                background-color: {t['accent']};
                color: {t['widget_bg']};
            }}
            QPushButton#pasteListAccentBtn:pressed {{
                background-color: {t['accent_hover']};
                color: {t['widget_bg']};
            }}
            QPushButton#pasteListWarningBtn {{
                background-color: {t['warning_light']};
                border: none;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['warning']};
            }}
            QPushButton#pasteListWarningBtn:hover, QPushButton#pasteListWarningBtn:pressed {{
                background-color: {t['warning']};
                color: {t['widget_bg']};
            }}"""

    @classmethod
    def _qss_action_buttons(cls, t):
        """视图切换、成功、危险三类操作按钮。"""
        return f"""\
            QPushButton#viewToggleBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#viewToggleBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#successBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#successBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#successBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
            QPushButton#dangerBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
                font-weight: bold;
                color: {t['button_text']};
            }}
            QPushButton#dangerBtn:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton#dangerBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}"""


    @classmethod
    def _qss_toolbar(cls, t):
        """工具栏分隔线、文件计数、前缀输入框、模式分段控件、选项弹窗。"""
        return f"""\
            QFrame#toolbarSep {{
                background-color: transparent;
                border-left: 1px solid {t['border_color']};
                max-width: 1px;
                min-width: 1px;
                margin: 2px 4px;
            }}
            QLabel#fileCountLabel {{
                color: {t['text_secondary']};
                font-size: 11px;
                background-color: transparent;
            }}
            QLineEdit#prefixInput[placeholder="true"] {{
                color: {t['text_disabled']};
            }}
            QLineEdit#prefixInput[placeholder="false"] {{
                color: {t['text_primary']};
            }}
            QFrame#modeSeg {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
            }}
            QPushButton#modeSegBtn {{
                background: transparent;
                border: none;
                color: {t['button_text']};
                font-size: {t['font_size_px_small']}px;
                padding: 3px 8px;
            }}
            QPushButton#modeSegBtn:hover {{
                color: {t['button_text']};
            }}
            QPushButton#modeSegBtn:checked {{
                color: {t['interaction_active_text']};
            }}
            QLineEdit#cacheSlotName {{
                padding: 0 6px;
            }}
            QLineEdit#cacheSlotName[active="true"] {{
                font-weight: bold;
            }}
            QWidget#optionsPopup {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
            }}
            QWidget#optionsPopup QPushButton {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                padding: 6px 16px;
                text-align: left;
                color: {t['button_text']};
            }}
            QWidget#optionsPopup QPushButton:hover {{
                border-color: {t['interaction_active']};
            }}
            QWidget#optionsPopup QPushButton:pressed {{
                background-color: {t['interaction_active']};
                border-color: {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}"""

    @classmethod
    def _qss_overrides(cls, t):
        """末尾对工具栏右上角三个按钮的统一覆盖。"""
        return f"""\
            QPushButton#iconBtn, QPushButton#navBtn, QPushButton#bgBtn,
            QPushButton#pasteBtn, QPushButton#labelBtn, QPushButton#datasetBtn,
            QPushButton#themeBtn, QPushButton#settingsBtn, QPushButton#langBtn {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
            }}
            QPushButton#iconBtn:hover, QPushButton#navBtn:hover, QPushButton#bgBtn:hover,
            QPushButton#pasteBtn:hover, QPushButton#labelBtn:hover, QPushButton#datasetBtn:hover,
            QPushButton#themeBtn:hover, QPushButton#settingsBtn:hover, QPushButton#langBtn:hover {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['interaction_active']};
                color: {t['button_text']};
            }}
            QPushButton#iconBtn:pressed, QPushButton#navBtn:pressed, QPushButton#bgBtn:pressed,
            QPushButton#pasteBtn:pressed, QPushButton#labelBtn:pressed, QPushButton#datasetBtn:pressed,
            QPushButton#themeBtn:pressed, QPushButton#settingsBtn:pressed, QPushButton#langBtn:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}"""


    @classmethod
    def get_dialog_style(cls):
        """小窗共用的底色与正文色。

        这条规则原先在十一个对话框里各写了一遍，换配色得挨个找；而且各处的
        写法还不一致（有的用 `theme`、有的用 `t`），搜起来更容易漏。

        注意别和 `get_dialog_button_style()` 混：那个管按钮外观，这里只管
        对话框本身的底色与文字色。
        """
        t = cls._current_theme
        return (f"QDialog {{ background-color: {t['widget_bg']}; "
                f"color: {t['text_primary']}; }}")

    @classmethod
    def get_label_style(cls):
        """对话框/面板里标签的正文色。

        主样式表已经给 QLabel 设过这一条，但小窗是独立顶层窗口，样式不一定
        继承下来，所以各处又写了一遍 —— 六个地方。收在这里，改配色只改一处。
        """
        t = cls._current_theme
        return f"QLabel {{ color: {t['text_primary']}; }}"

    @classmethod
    def get_list_style(cls):
        t = cls._current_theme
        return f"""
            QListWidget {{
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
            }}
            QListWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {t['list_border']};
            }}
            QListWidget::item:hover {{
                background-color: {t['list_hover']};
            }}
            QListWidget::item:selected {{
                background-color: {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
        """

    @classmethod
    def get_button_style(cls, bg_color=None, text_color=None, variant="default"):
        t = cls._current_theme
        if variant == "icon":
            return f"""
                QPushButton {{
                    background-color: {t['widget_bg']};
                    border: 1px solid {t['border_color']};
                    border-radius: 4px;
                    padding: 2px;
                }}
                QPushButton:hover {{
                    border: 1px solid {t['interaction_active']};
                }}
                QPushButton:pressed {{
                    background-color: {t['interaction_active']};
                    border: 1px solid {t['interaction_active']};
                    color: {t['interaction_active_text']};
                }}
            """
        bg = bg_color or t['widget_bg']
        fg = text_color or t['button_text']
        return f"""
            QPushButton {{
                background-color: {bg};
                border: 1px solid {t['border_color']};
                border-radius: 4px;
                padding: 3px 8px;
                font-size: {t['font_size_px']}px;
                font-weight: bold;
                color: {fg};
            }}
            QPushButton:hover {{
                border: 1px solid {t['interaction_active']};
            }}
            QPushButton:pressed {{
                background-color: {t['interaction_active']};
                border: 1px solid {t['interaction_active']};
                color: {t['interaction_active_text']};
            }}
        """

    @classmethod
    def get_dialog_button_style(cls):
        """小窗按钮复用设置弹窗的普通按钮圆角。"""
        return cls.get_button_style()

    @classmethod
    def get_progress_style(cls, radius=8, height=16, text_visible=True):
        """全项目进度条统一圆角样式。

        圆角要同时写在 QProgressBar 与 ::chunk 上：只写外层会让填充块顶着
        直角边角，看起来像没生效。`radius` 给 0 以外的值时记得 height 至少
        是它的两倍，否则圆角被压扁。
        """
        t = cls._current_theme
        text = (f"text-align: center; color: {t['text_primary']};"
                if text_visible else "")
        return f"""
            QProgressBar {{
                border: none;
                border-radius: {radius}px;
                background-color: {t['scrollbar_bg']};
                {text}
                min-height: {height}px;
                max-height: {height}px;
            }}
            QProgressBar::chunk {{
                background-color: {t['accent']};
                border-radius: {radius}px;
            }}
        """
