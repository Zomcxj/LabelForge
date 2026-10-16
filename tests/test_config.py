"""
config.py 常量测试 - 验证配置完整性和一致性
"""
import sys
import os
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), '..', 'labelforge', 'core')
config = _load_module('config', os.path.join(_base, 'config.py'))


class TestShortcutConfig:

    REQUIRED_KEYS = [
        'undo', 'redo', 'toggle_grid',
        'toggle_labels', 'toggle_label_names',
        'auto_save_b', 'auto_save_p',
        'toggle_paste_names', 'draw_box', 'draw_polygon', 'draw_point', 'draw_obb', 'quit_draw',
        'next_image', 'prev_image', 'delete_selected', 'fit_view',
        'zoom_in', 'zoom_out',
    ]

    def test_has_all_required_keys(self):
        for key in self.REQUIRED_KEYS:
            assert key in config.SHORTCUT_CONFIG, f"missing key: {key}"

    def test_no_duplicate_single_keys(self):
        single = [v for v in config.SHORTCUT_CONFIG.values() if '+' not in v]
        assert len(single) == len(set(single)), f"dup: {single}"

    def test_all_values_are_strings(self):
        for k, v in config.SHORTCUT_CONFIG.items():
            assert isinstance(v, str), f"{k} is {type(v)}"

    def test_no_clear_or_zoom_reset(self):
        assert 'clear' not in config.SHORTCUT_CONFIG
        assert 'zoom_reset' not in config.SHORTCUT_CONFIG


class TestLabelColors:

    def test_has_at_least_30_distinct_colors(self):
        assert len(config.LABEL_COLORS) >= 30
        assert len(set(config.LABEL_COLORS)) >= 30

    def test_all_hex_format(self):
        for c in config.LABEL_COLORS:
            assert c.startswith('#') and len(c) == 7, f"bad color: {c}"
            int(c[1:], 16)  # should not raise


class TestConfigStructures:

    def test_supported_image_extensions(self):
        assert '.png' in config.SUPPORTED_IMAGE_EXTENSIONS
        assert '.jpg' in config.SUPPORTED_IMAGE_EXTENSIONS
        assert '.bmp' in config.SUPPORTED_IMAGE_EXTENSIONS

    def test_window_config_dimensions(self):
        assert config.WINDOW_CONFIG['default_width'] > 0
        assert config.WINDOW_CONFIG['default_height'] > 0

    def test_paste_params_valid(self):
        p = config.PASTE_PARAMS
        assert p['min_count'] <= p['default_count'] <= p['max_count']
        assert p['min_size_range'][0] < p['min_size_range'][1]

    def test_undo_config_positive(self):
        assert config.UNDO_CONFIG['max_history'] > 0

    def test_max_polygon_points_default(self):
        assert config.DETECTION_BOX_CONFIG['max_polygon_points'] == 32

    def test_first_palette_color_is_not_red(self):
        """新标签按排序位置分配 colors[0]，不能是红色系（用户感知差）。"""
        r = int(config.LABEL_COLORS[0][1:3], 16)
        assert r < 200, config.LABEL_COLORS[0]
