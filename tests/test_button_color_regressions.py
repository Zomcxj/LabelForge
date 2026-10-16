from pathlib import Path

import pytest

from labelforge.ui.theme import ThemeManager, ThemeMode


@pytest.mark.parametrize(
    ("mode", "text_color"),
    [(ThemeMode.LIGHT, "#1F1F1F"), (ThemeMode.DARK, "#F5F5F5")],
)
def test_base_text_and_unselected_button_states_use_the_soft_theme_color(mode, text_color):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()
    theme = ThemeManager.get_theme()

    assert theme["text_primary"] == text_color
    assert theme["text_secondary"] == text_color
    assert theme["button_text"] == text_color
    for selector in ("QPushButton", "QPushButton#langBtn", "QPushButton#langBtn:hover"):
        block = style.split(f"{selector} {{", 1)[1].split("}", 1)[0]
        assert f"color: {text_color};" in block
    mode_button_hover = style.split("QPushButton#modeSegBtn:hover {", 1)[1].split("}", 1)[0]
    assert f"color: {theme['button_text']};" in mode_button_hover

    pressed_block = style.split("QPushButton:pressed {", 1)[1].split("}", 1)[0]
    assert f"color: {theme['interaction_active_text']};" in pressed_block


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_buttons_use_theme_primary_text_except_background_list_controls(mode):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()
    theme = ThemeManager.get_theme()

    assert f"QPushButton {{\n                background-color: {theme['widget_bg']};" in style
    assert f"color: {theme['button_text']};" in style
    for name in ("optionsBtn", "drawBoxBtn", "successBtn", "dangerBtn"):
        block = style.split(f"QPushButton#{name} {{", 1)[1].split("}", 1)[0]
        assert f"color: {theme['button_text']};" in block

    nav_block = style.split("QPushButton#navBtn {", 1)[1].split("}", 1)[0]
    for name in ("accentBtn", "warningBtn", "successBtn", "dangerBtn"):
        block = style.split(f"QPushButton#{name} {{", 1)[1].split("}", 1)[0]
        assert f"color: {theme['button_text']};" in block

    warning_block = style.split("QPushButton#warningBtn {", 1)[1].split("}", 1)[0]
    accent_block = style.split("QPushButton#accentBtn {", 1)[1].split("}", 1)[0]
    assert f"color: {theme['text_secondary']};" in nav_block
    assert f"color: {theme['button_text']};" in warning_block
    assert f"color: {theme['button_text']};" in accent_block

    for name, color in (("pasteListAccentBtn", theme["accent"]), ("pasteListWarningBtn", theme["warning"])):
        block = style.split(f"QPushButton#{name} {{", 1)[1].split("}", 1)[0]
        assert f"color: {color};" in block


def test_color_buttons_render_their_selected_color_without_blue_states():
    import inspect
    from labelforge.ui.main_window import ImageEditor

    root = Path(__file__).resolve().parents[1]
    settings = (root / "labelforge" / "ui" / "settings_dialog.py").read_text(encoding="utf-8")

    label_style = inspect.getsource(ImageEditor._set_label_color_button)
    crosshair_style = settings.split("def _update_crosshair_color_button", 1)[1].split("def _choose_crosshair_color", 1)[0]
    assert "background-color: {color}" in label_style
    assert "#2950ff" not in label_style
    assert "background-color: {self._crosshair_color}" in crosshair_style
    assert "#2950ff" not in crosshair_style


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_language_button_uses_theme_button_text_until_pressed(mode):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()
    color = ThemeManager.get_theme()["button_text"]

    for state in ("", ":hover"):
        block = style.split(f"QPushButton#langBtn{state} {{", 1)[1].split("}", 1)[0]
        assert f"color: {color};" in block

    pressed_block = style.split("QPushButton#langBtn:pressed {", 1)[1].split("}", 1)[0]
    theme = ThemeManager.get_theme()
    assert f"background-color: {theme['interaction_active']};" in pressed_block
    assert f"color: {theme['interaction_active_text']};" in pressed_block
