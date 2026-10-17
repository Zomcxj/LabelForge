"""Main window layout regressions (canvas vs side panel)."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout

from labelforge.ui.mixins.panels import PanelsMixin


def _inspect_create_splitter_source():
    import inspect
    return inspect.getsource(PanelsMixin._create_splitter)


def test_create_splitter_expands_canvas_and_panel_vertically():
    """Canvas and right panel must both expand vertically (same height)."""
    src = _inspect_create_splitter_source()
    assert "QSizePolicy.Expanding" in src
    assert "setContentsMargins" in src
    # Must not pin widgets with Qt.AlignTop (disables vertical stretch).
    assert "Qt.AlignTop" not in src
    assert "alignment=Qt.AlignTop" not in src


def test_create_control_panel_top_margin_matches_canvas():
    """Control panel top margin must not exceed canvas top margin."""
    import inspect
    splitter_src = inspect.getsource(PanelsMixin._create_splitter)
    control_src = inspect.getsource(PanelsMixin._create_control_panel)
    # Canvas side: canvas_layout.setContentsMargins(0, 0, 0, 0) is the baseline.
    assert "canvas_layout.setContentsMargins(0, 0, 0, 0)" in splitter_src
    # Control panel outer top must be 0 to match canvas top edge.
    assert "setContentsMargins(6, 0," in control_src


def test_label_list_item_padding_matches_bg_list():
    """Background / label / paste list rows share the same item padding."""
    import inspect
    from labelforge.ui.theme import ThemeManager

    src = inspect.getsource(ThemeManager.get_stylesheet)
    # No tighter override for labelList/pasteLabelList item padding.
    assert "QListWidget#labelList::item, QListWidget#pasteLabelList::item" not in src
    assert "QListWidget::item" in src
