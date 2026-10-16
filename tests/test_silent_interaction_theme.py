import pytest
from pathlib import Path

from labelforge.ui.theme import ThemeManager, ThemeMode


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_named_action_buttons_use_silent_surface_and_blue_hover(mode):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()
    theme = ThemeManager.get_theme()

    assert f"QPushButton#navBtn {{\n                background-color: transparent;" in style
    assert f"color: {theme['text_secondary']};" in style.split("QPushButton#navBtn {", 1)[1].split("}", 1)[0]
    for name in ("accentBtn", "warningBtn"):
        block = style.split(f"QPushButton#{name} {{", 1)[1].split("}", 1)[0]
        hover_block = style.split(f"QPushButton#{name}:hover {{", 1)[1].split("}", 1)[0]
        assert f"background-color: {theme['widget_bg']};" in block
        assert f"border: 1px solid {theme['border_color']};" in block
        assert f"color: {theme['button_text']};" in block
        assert "border: 1px solid #2950ff;" in hover_block
        assert "background-color" not in hover_block


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_mode_segment_uses_theme_text_until_selected(mode):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()
    theme = ThemeManager.get_theme()

    container = style.split("QFrame#modeSeg {", 1)[1].split("}", 1)[0]
    button = style.split("QPushButton#modeSegBtn {", 1)[1].split("}", 1)[0]
    selected = style.split("QPushButton#modeSegBtn:checked {", 1)[1].split("}", 1)[0]
    assert f"background-color: {theme['widget_bg']};" in container
    assert f"border: 1px solid {theme['border_color']};" in container
    assert f"color: {theme['button_text']};" in button
    assert f"color: {theme['interaction_active_text']};" in selected


@pytest.mark.parametrize("mode", [ThemeMode.LIGHT, ThemeMode.DARK])
def test_options_buttons_use_blue_pressed_state(mode):
    ThemeManager.set_mode(mode)
    style = ThemeManager.get_stylesheet()

    pressed_block = style.split("QPushButton#optionsBtn:pressed {", 1)[1].split("}", 1)[0]
    assert "background-color: #2950ff;" in pressed_block
    assert "border: 1px solid #2950ff;" in pressed_block
    assert "color: #FFFFFF;" in pressed_block


def test_mode_segment_indicator_reads_the_active_interaction_token():
    import inspect
    from labelforge.ui.main_window import ImageEditor

    root = Path(__file__).resolve().parents[1]
    builder = (root / "labelforge" / "ui" / "mixins" / "options_popup.py").read_text(encoding="utf-8")
    window = inspect.getsource(ImageEditor._update_mode_seg_style)
    style = ThemeManager.get_stylesheet()

    token = 'ThemeManager.get_theme()["interaction_active"]'
    assert f"self.mode_seg_ctrl.set_accent({token})" in builder
    assert f"self.mode_seg_ctrl.set_accent({token})" in window
    assert 'set_accent("#2950ff")' not in builder
    assert 'set_accent("#2950ff")' not in window
    button = style.split("QPushButton#modeSegBtn {", 1)[1].split("}", 1)[0]
    assert "color: #2950ff;" not in button


def test_named_controls_use_the_shared_silent_button_styles():
    root = Path(__file__).resolve().parents[1]
    lists = (root / "labelforge" / "ui" / "mixins" / "lists.py").read_text(encoding="utf-8")

    assert lists.count('setObjectName("accentBtn")') == 2
    assert 'self.toggle_view_btn.setObjectName("warningBtn")' in lists


def test_all_inputs_hover_with_blue_border():
    style = ThemeManager.get_stylesheet()

    assert "QLineEdit:hover {\n                border: 1px solid #2950ff;" in style
    assert "QSpinBox:hover, QDoubleSpinBox:hover {\n                border: 1px solid #2950ff;" in style
    assert "QComboBox:hover {\n                border: 1px solid #2950ff;" in style


def test_label_path_list_selection_is_blue_with_white_text():
    style = ThemeManager.get_stylesheet()

    assert "QListWidget::item:selected {\n                background-color: #2950ff;\n                color: #FFFFFF;" in style
