"""加载流程和滚轮优先级回归测试。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_load_background_folder_initializes_first_image_inside_loop():
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")

    assert "if load_first:" in source
    assert "self.current_background = self._get_cached_pixmap(first_path)" in source


def test_load_background_folder_does_not_block_background_list_during_label_scan():
    source = (ROOT / "labelforge" / "engine" / "image_loader" / "mixin.py").read_text(encoding="utf-8")
    method_source = source[source.index("    def load_background_folder("):source.index("    def upload_small_images(")]

    assert "self._show_loading_spinner()" not in method_source
    assert "self._hide_loading_spinner()" not in method_source
    assert "self.background_list.setUpdatesEnabled(False)" not in method_source
    assert "self.background_list.setUpdatesEnabled(True)" not in method_source


def test_ctrl_wheel_scales_background_before_selected_objects():
    source = (ROOT / "labelforge" / "canvas" / "canvas_interaction.py").read_text(encoding="utf-8")

    ctrl_pos = source.index("if event.modifiers() & Qt.ControlModifier:")
    item_pos = source.index("if self._editor.selected_item is not None:")
    box_pos = source.index("elif self.selected_box is not None:")
    assert ctrl_pos < item_pos < box_pos
