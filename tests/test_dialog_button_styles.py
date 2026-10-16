from labelforge.ui.theme import ThemeManager


def test_dialog_button_style_matches_settings_radius():
    """弹窗按钮与全局按钮用同一个圆角（第 11 轮统一为 4px）。

    用户要求「统一所有按钮输入框圆角与视频窗口中的一致」——视频窗口自己写的是
    4px，所以全局值跟着它走；这条测试盯的是「弹窗按钮别又长回一个自己的值」。
    """
    style = ThemeManager.get_dialog_button_style()
    theme = ThemeManager.get_theme()

    assert "border-radius: 4px" in style
    assert f"font-size: {theme['font_size']}px" in style
