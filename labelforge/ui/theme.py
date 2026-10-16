"""
Theme module - manages dark/light theme switching
"""
from enum import Enum


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


LIGHT_THEME = {
    "name": "light",
    "font_family": "'JetBrains Mono', 'Microsoft YaHei', 'Segoe UI', sans-serif",
    "font_size": 12,
    "font_size_small": 11,
    "interaction_active": "#2950ff",
    "interaction_active_text": "#FFFFFF",
    "window_bg": "#F5F5F5",
    "widget_bg": "#FFFFFF",
    "panel_bg": "#FAFAFA",
            "canvas_bg": "#FFFFFF",
    "border_color": "#E0E0E0",
    "border_hover": "#BBDEFB",
    "text_primary": "#1F1F1F",
    "button_text": "#1F1F1F",
    "text_secondary": "#1F1F1F",
    "text_disabled": "#BDBDBD",
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
    "font_family": "'JetBrains Mono', 'Microsoft YaHei', 'Segoe UI', sans-serif",
    "font_size": 12,
    "font_size_small": 11,
    "interaction_active": "#2950ff",
    "interaction_active_text": "#FFFFFF",
    "window_bg": "#1E1E1E",
    "widget_bg": "#2D2D2D",
    "panel_bg": "#252525",
    "canvas_bg": "#2D2D2D",
    "border_color": "#3E3E3E",
    "border_hover": "#505050",
    "text_primary": "#F5F5F5",
    "button_text": "#F5F5F5",
    "text_secondary": "#F5F5F5",
    "text_disabled": "#616161",
    "accent": "#42A5F5",
    "accent_hover": "#1E88E5",
    "accent_light": "#1A2332",
    "success": "#66BB6A",
    "success_light": "#1A2E1A",
    "warning": "#FFA726",
    "warning_light": "#2E2218",
    "danger": "#EF5350",
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
        t = cls._current_theme
        return f"""
            QMainWindow {{
                background-color: {t['window_bg']};
            }}
            QWidget {{
                background-color: {t['widget_bg']};
                color: {t['text_primary']};
                font-family: {t['font_family']};
                font-size: {t['font_size']}px;
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
                border: 1px solid {t['border_color']};
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
                border: 1px solid {t['border_color']};
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
                border: 2px solid {t['border_color']};
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
            }}
            QTextEdit#logArea {{
                border: 1px solid {t['groupbox_border']};
                border-radius: 4px;
                background-color: {t['widget_bg']};
                color: {t['text_secondary']};
                font-family: Consolas, "Courier New", monospace;
                font-size: 12px;
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
                border: 1px solid {t['border_color']};
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
            }}
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
            }}
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
            }}
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
            }}
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
                font-size: {t['font_size_small']}px;
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
            }}
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
            }}
        """

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
                font-size: {t['font_size']}px;
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
