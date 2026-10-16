"""Label color assignment regressions."""
from labelforge.core import config_manager
from labelforge.ui.main_window import ImageEditor


def test_default_palette_assignment_is_independent_of_label_order():
    labels = [f"label-{index}" for index in range(30)]
    reversed_labels = list(reversed(labels))

    assigned = {
        label: config_manager.get_label_color(labels, label, label_color_map={})
        for label in labels
    }

    assert {
        label: config_manager.get_label_color(reversed_labels, label, label_color_map={})
        for label in reversed_labels
    } == assigned
    assert len(set(assigned.values())) == 30


def test_image_editor_uses_explicit_label_color_map_and_keeps_first_assignment(monkeypatch):
    monkeypatch.setattr(config_manager, 'save_all', lambda **kwargs: None)
    editor = ImageEditor.__new__(ImageEditor)
    editor.label_colors = ['#111111', '#222222']
    editor.label_color_map = {'person': '#abcdef'}
    editor._get_session_labels = lambda: ['person', 'car']

    assert ImageEditor.get_label_color(editor, 'person') == '#abcdef'
    assert ImageEditor.get_label_color(editor, 'car') == '#222222'
    assert editor.label_color_map['car'] == '#222222'


def test_image_editor_incremental_label_assignment_skips_existing_colors(monkeypatch):
    monkeypatch.setattr(config_manager, 'save_all', lambda **kwargs: None)
    editor = ImageEditor.__new__(ImageEditor)
    editor.label_colors = config_manager.LABEL_COLORS
    editor.label_color_map = {}
    session_labels = ['zebra']
    editor._get_session_labels = lambda: session_labels

    zebra_color = ImageEditor.get_label_color(editor, 'zebra')
    session_labels.append('apple')
    apple_color = ImageEditor.get_label_color(editor, 'apple')

    assert zebra_color != apple_color


def test_image_editor_expanded_legacy_palette_avoids_hash_collisions(tmp_path):
    old_palette = [f'#{index:06x}' for index in range(16)]
    original = config_manager.CONFIG_PATH
    config_manager.CONFIG_PATH = str(tmp_path / 'config.json')
    try:
        editor = ImageEditor.__new__(ImageEditor)
        editor.label_colors = config_manager._normalize_label_colors(old_palette, extend=True)
        editor.label_color_map = {}
        session_labels = ['a']
        editor._get_session_labels = lambda: session_labels

        a_color = ImageEditor.get_label_color(editor, 'a')
        session_labels.append('5a')
        five_a_color = ImageEditor.get_label_color(editor, '5a')

        assert len(editor.label_colors) >= 30
        assert a_color != five_a_color
        assert config_manager.load_all()['label_color_map'] == {
            'a': a_color,
            '5a': five_a_color,
        }
    finally:
        config_manager.CONFIG_PATH = original


def test_image_editor_persists_new_automatic_assignment_across_instances(tmp_path):
    original = config_manager.CONFIG_PATH
    config_manager.CONFIG_PATH = str(tmp_path / "config.json")
    try:
        first = ImageEditor.__new__(ImageEditor)
        first.label_colors = config_manager.load_all()['label_colors']
        first.label_color_map = {}
        first._get_session_labels = lambda: ['zebra']

        color = ImageEditor.get_label_color(first, 'zebra')

        assert config_manager.load_all()['label_color_map']['zebra'] == color

        second = ImageEditor.__new__(ImageEditor)
        settings = config_manager.load_all()
        second.label_colors = settings['label_colors']
        second.label_color_map = settings['label_color_map']
        second._get_session_labels = lambda: ['zebra']
        assert ImageEditor.get_label_color(second, 'zebra') == color
    finally:
        config_manager.CONFIG_PATH = original


def test_image_editor_color_change_updates_only_the_selected_label(monkeypatch):
    class Color:
        def isValid(self):
            return True

        def name(self):
            return '#abcdef'

    class ColorDialog:
        def __init__(self, parent):
            pass

        def setWindowTitle(self, title):
            pass

        def setCurrentColor(self, color):
            pass

        def exec_(self):
            return 1

        def currentColor(self):
            return Color()

    from labelforge.ui import dialog_helpers

    monkeypatch.setattr(dialog_helpers, 'ThemedColorDialog', ColorDialog)
    saved = {}
    monkeypatch.setattr(config_manager, 'save_all', lambda **kwargs: saved.update(kwargs))

    editor = ImageEditor.__new__(ImageEditor)
    editor.label_colors = ['#111111', '#222222']
    editor.label_color_map = {'cat': '#111111', 'dog': '#222222'}
    editor._get_session_labels = lambda: ['dog', 'cat']
    editor.canvas = type('Canvas', (), {'update': lambda self: None})()

    ImageEditor._change_label_color(editor, 'cat', None)

    assert editor.label_color_map == {'cat': '#abcdef', 'dog': '#222222'}
    assert editor.label_colors == ['#111111', '#222222']
    assert saved == {
        'label_colors': editor.label_colors,
        'label_color_map': editor.label_color_map,
    }
