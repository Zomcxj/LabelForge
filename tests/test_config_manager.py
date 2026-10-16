"""
config_manager.py 测试 - 配置读写一致性
"""
import os
import sys
import json
import importlib.util
import tempfile
from pathlib import Path

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path),
        submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_base = os.path.join(os.path.dirname(__file__), '..', 'labelforge', 'core')
config = _load_module('config', os.path.join(_base, 'config.py'))
config_manager = _load_module('config_manager', os.path.join(_base, 'config_manager.py'))


def _with_temp_config(tmp_path):
    original = config_manager.CONFIG_PATH
    config_manager.CONFIG_PATH = str(Path(tmp_path) / 'missing.json')
    return original


class TestLoadAll:

    def test_returns_dict_with_expected_keys(self):
        result = config_manager.load_all()
        assert {'shortcuts', 'theme', 'language', 'max_labels',
                'grid_line_width', 'grid_alpha', 'memory'}.issubset(result.keys())

    def test_shortcuts_has_all_keys(self):
        sc = config_manager.load_all()['shortcuts']
        for key in config.SHORTCUT_CONFIG:
            assert key in sc, f"fallback missing: {key}"

    def test_legacy_delete_shortcut_uses_new_default(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_config({'shortcuts': {'delete_selected': 'Delete'}})
            assert config_manager.load_all()['shortcuts']['delete_selected'] == 'E'
        finally:
            config_manager.CONFIG_PATH = original

    def test_default_theme(self):
        theme = config_manager.load_all()['theme']
        assert theme in ('light', 'dark')

    def test_default_language(self):
        lang = config_manager.load_all()['language']
        assert lang in ('zh', 'en')

    def test_magnifier_disabled_by_default_when_config_missing(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            assert config_manager.load_all()['magnifier_enabled'] is False
        finally:
            config_manager.CONFIG_PATH = original

    def test_magnifier_zoom_defaults_to_config_when_config_missing(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            assert config_manager.load_all()['magnifier_zoom'] == config.MAGNIFIER_CONFIG['zoom']
        finally:
            config_manager.CONFIG_PATH = original

    def test_label_cache_slots_defaults_to_three_numeric_shortcuts(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            result = config_manager.load_all()
            assert 'label_cache_slots' in result
            assert len(result['label_cache_slots']) == 3
            assert [slot['name'] for slot in result['label_cache_slots']] == ['缓存槽1', '缓存槽2', '缓存槽3']
            assert [slot['shortcut'] for slot in result['label_cache_slots']] == ['1', '2', '3']
        finally:
            config_manager.CONFIG_PATH = original

    def test_label_cache_slots_load_preserves_saved_shortcuts(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(label_cache_slots=[
                {'name': '一号槽', 'locked': True, 'items': [], 'shortcut': 'Ctrl+1'},
                {'name': '二号槽', 'locked': False, 'items': [], 'shortcut': 'Ctrl+2'},
                {'name': '三号槽', 'locked': False, 'items': [], 'shortcut': 'Ctrl+3'},
            ])
            result = config_manager.load_all()['label_cache_slots']
            assert [slot['name'] for slot in result] == ['一号槽', '二号槽', '三号槽']
            assert [slot['shortcut'] for slot in result] == ['Ctrl+1', 'Ctrl+2', 'Ctrl+3']
        finally:
            config_manager.CONFIG_PATH = original


class TestSaveLoadRoundtrip:

    def test_shortcuts_roundtrip(self):
        original = config.SHORTCUT_CONFIG.copy()
        test_sc = original.copy()
        test_sc['toggle_labels'] = 'X'
        config_manager.save_shortcuts(test_sc)
        loaded = config_manager.load_shortcuts()
        assert loaded['toggle_labels'] == 'X'
        config_manager.save_shortcuts(original)

    def test_theme_roundtrip(self):
        original = config_manager.load_theme()
        config_manager.save_theme('dark')
        assert config_manager.load_theme() == 'dark'
        config_manager.save_theme(original)

    def test_language_roundtrip(self):
        original = config_manager.load_language()
        config_manager.save_language('en')
        assert config_manager.load_language() == 'en'
        config_manager.save_language(original)

    def test_save_all(self):
        original = config_manager.load_all()
        config_manager.save_all(theme='dark', language='en')
        result = config_manager.load_all()
        assert result['theme'] == 'dark'
        assert result['language'] == 'en'
        config_manager.save_all(
            theme=original['theme'],
            language=original['language'],
        )

    def test_max_polygon_points_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(max_polygon_points=16)
            assert config_manager.load_all()['max_polygon_points'] == 16
            config_manager.save_all(max_polygon_points=3)
            assert config_manager.load_all()['max_polygon_points'] == 3
            config_manager.save_all(max_polygon_points=99)
            assert config_manager.load_all()['max_polygon_points'] == 64
        finally:
            config_manager.CONFIG_PATH = original

    def test_magnifier_enabled_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(magnifier_enabled=True)
            assert config_manager.load_all()['magnifier_enabled'] is True
        finally:
            config_manager.CONFIG_PATH = original

    def test_magnifier_zoom_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(magnifier_zoom=2.5)
            assert config_manager.load_all()['magnifier_zoom'] == 2.5
        finally:
            config_manager.CONFIG_PATH = original

    def test_wheel_scale_settings_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(
                detection_box_scale_step=0.08,
                paste_item_scale_step=0.18,
                detection_box_wheel_edge_step=9,
            )
            result = config_manager.load_all()
            assert result['detection_box_scale_step'] == 0.08
            assert result['paste_item_scale_step'] == 0.18
            assert result['detection_box_wheel_edge_step'] == 9
        finally:
            config_manager.CONFIG_PATH = original

    def test_label_colors_fall_back_to_defaults_for_legacy_or_invalid_config(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_config({'label_colors': ['bad', 42]})
            assert config_manager.load_all()['label_colors'] == config.LABEL_COLORS
        finally:
            config_manager.CONFIG_PATH = original

    def test_wheel_scale_settings_use_legacy_value_only_when_new_keys_are_missing(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_config({'detection_box_wheel_scale_step': 0.12})
            result = config_manager.load_all()

            assert result['detection_box_scale_step'] == 0.12
            assert result['paste_item_scale_step'] == 0.12

            config_manager.save_config({
                'detection_box_wheel_scale_step': 0.12,
                'detection_box_scale_step': 0.08,
                'paste_item_scale_step': 0.18,
            })
            result = config_manager.load_all()

            assert result['detection_box_scale_step'] == 0.08
            assert result['paste_item_scale_step'] == 0.18
        finally:
            config_manager.CONFIG_PATH = original

    def test_wheel_scale_settings_have_distinct_defaults_and_save_only_new_keys(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            assert config_manager.load_all()['detection_box_scale_step'] == 0.05
            assert config_manager.load_all()['paste_item_scale_step'] == 0.15

            config_manager.save_all(detection_box_scale_step=0.06, paste_item_scale_step=0.16)
            saved = config_manager.load_config()

            assert saved['detection_box_scale_step'] == 0.06
            assert saved['paste_item_scale_step'] == 0.16
            assert 'detection_box_wheel_scale_step' not in saved
        finally:
            config_manager.CONFIG_PATH = original

    def test_label_colors_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            palette = ['#123456', '#abcdef']
            config_manager.save_all(label_colors=palette)
            saved_palette = config_manager.load_all()['label_colors']
            assert saved_palette[:2] == palette
            assert len(saved_palette) >= 30
            assert len(set(saved_palette[:30])) == 30
            assert config_manager.load_config()['label_colors'] == saved_palette
        finally:
            config_manager.CONFIG_PATH = original

    def test_legacy_valid_palette_is_extended_to_thirty_distinct_colors(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            custom = ['#123456', '#abcdef']
            config_manager.save_config({'label_colors': custom})

            palette = config_manager.load_all()['label_colors']
            assert palette[:2] == custom
            assert len(palette) >= 30
            assert len(set(palette[:30])) == 30
        finally:
            config_manager.CONFIG_PATH = original

    def test_label_color_map_is_normalized_and_roundtrips_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(label_color_map={
                'person': '#abcdef',
                42: '#123456',
                'invalid': 'blue',
                'empty': '',
            })

            expected = {'person': '#abcdef', '42': '#123456'}
            assert config_manager.load_all()['label_color_map'] == expected
            assert config_manager.load_config()['label_color_map'] == expected
        finally:
            config_manager.CONFIG_PATH = original


def test_label_color_stable_by_label_name():
    palette = ['#111111', '#222222']
    labels = ['zebra', 'apple', 'mango']

    assert config_manager.get_label_color(labels, 'apple', palette, {}) == '#111111'
    assert config_manager.get_label_color(labels, 'mango', palette, {}) == '#111111'
    assert config_manager.get_label_color(labels, 'zebra', palette, {}) == '#222222'

    # adding new label does not shift existing colors
    labels2 = ['zebra', 'apple', 'mango', 'newlabel']
    assert config_manager.get_label_color(labels2, 'apple', palette, {}) == '#111111'
    assert config_manager.get_label_color(labels2, 'zebra', palette, {}) == '#222222'


def test_default_label_palette_has_at_least_thirty_distinct_hex_colors():
    assert len(config.LABEL_COLORS) >= 30
    assert len(set(config.LABEL_COLORS)) >= 30
    assert all(len(color) == 7 and color.startswith('#') for color in config.LABEL_COLORS)


def test_label_not_in_session_still_gets_distinct_palette_color():
    """新标签在被加入会话列表前请求颜色时，不能落入哈希兜底导致撞成红色。"""
    palette = list(config.LABEL_COLORS)
    # 颜色不应因“标签是否已进入会话列表”而改变（否则新标签落哈希兜底撞成红）
    before = config_manager.get_label_color(['SCar', 'Rider', 'MCar'], 'Person', palette, {})
    after = config_manager.get_label_color(['SCar', 'Rider', 'MCar', 'Person'], 'Person', palette, {})
    assert before == after


def test_label_color_map_overrides_default_allocation():
    labels = ['person', 'car']
    palette = ['#111111', '#222222']

    assert config_manager.get_label_color(
        labels, 'person', palette, {'person': '#abcdef'},
    ) == '#abcdef'


def test_default_palette_assigns_first_thirty_labels_distinct_colors():
    labels = [f'label-{index}' for index in range(30)]
    colors = [config_manager.get_label_color(labels, label, label_color_map={}) for label in labels]

    assert len(set(colors)) == 30


def test_load_all_migrates_legacy_red_first_palette_order(tmp_path):
    """旧默认调色板红打头；同集合的旧顺序必须迁移成新默认顺序，否则首标签又变红。"""
    original = _with_temp_config(tmp_path)
    try:
        legacy = list(config.LABEL_COLORS)
        legacy.remove('#E53935')
        legacy.insert(0, '#E53935')
        config_manager.save_config({'label_colors': legacy})

        migrated = config_manager.load_all()['label_colors']
        assert migrated == list(config.LABEL_COLORS)
        assert migrated[0] != '#E53935'
    finally:
        config_manager.CONFIG_PATH = original


def test_load_all_keeps_custom_palette_that_is_not_default_set(tmp_path):
    """真正的自定义调色板（集合不同）不能被迁移覆盖。"""
    original = _with_temp_config(tmp_path)
    try:
        custom = ['#010203', '#040506', '#070809']
        config_manager.save_config({'label_colors': custom})

        palette = config_manager.load_all()['label_colors']
        assert palette[:3] == custom
    finally:
        config_manager.CONFIG_PATH = original


def test_saved_label_color_map_takes_precedence_when_no_explicit_map_is_given(tmp_path):
    original = _with_temp_config(tmp_path)
    try:
        config_manager.save_all(label_color_map={'person': '#abcdef'})

        assert config_manager.get_label_color(['person'], 'person') == '#abcdef'
    finally:
        config_manager.CONFIG_PATH = original


def test_load_all_repairs_duplicate_colors_from_old_algorithm(tmp_path):
    """旧算法写入的撞色缓存必须在加载时重排，否则新标签会永久渲染成脏色。"""
    original = _with_temp_config(tmp_path)
    try:
        config_manager.save_all(label_color_map={
            'BCar': '#E53935',      # 旧算法把新标签写成红色
            'Rider': '#039BE5',
            'Bucket': '#039BE5',    # 与 Rider 撞色 —— 旧哈希兜底特征
            'Person': '#a4c6a1',    # 非调色板色
        })

        repaired = config_manager.load_all()['label_color_map']
        palette = config_manager.load_all()['label_colors']
        assert len(repaired) == 4
        assert len(set(repaired.values())) == 4, repaired
        assert all(color in palette for color in repaired.values()), repaired
    finally:
        config_manager.CONFIG_PATH = original


def test_load_all_keeps_manual_color_map_without_collisions(tmp_path):
    """颜色互不重复的表（含用户手动选色）不能被重排覆盖。"""
    original = _with_temp_config(tmp_path)
    try:
        manual = {'person': '#abcdef', 'car': '#123456'}
        config_manager.save_all(label_color_map=manual)

        assert config_manager.load_all()['label_color_map'] == manual
    finally:
        config_manager.CONFIG_PATH = original

    def test_crosshair_settings_roundtrip_through_save_all(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_all(crosshair_width=2.5, crosshair_color='#123456', crosshair_alpha=96)
            result = config_manager.load_all()
            assert result['crosshair_width'] == 2.5
            assert result['crosshair_color'] == '#123456'
            assert result['crosshair_alpha'] == 96
        finally:
            config_manager.CONFIG_PATH = original

    def test_memory_records_dedupe_limit_and_edit_mode(self):
        original = config_manager.load_memory_records()
        try:
            records = [
                {'note': f'n{i}', 'background_path': f'b{i}', 'paste_path': '', 'label_path': ''}
                for i in range(11)
            ]
            config_manager.save_memory_records(records)
            assert len(config_manager.load_memory_records()) == 10

            config_manager.upsert_memory_record({
                'note': 'updated', 'background_path': 'b10', 'paste_path': '', 'label_path': '',
                'background_index': 3, 'edit_mode': 'annotate',
            })
            loaded = config_manager.load_memory_records()
            assert loaded[0]['note'] == 'updated'
            assert loaded[0]['background_index'] == 3
            assert loaded[0]['edit_mode'] == 'annotate'
            assert sum(1 for r in loaded if r['background_path'] == 'b10') == 1
            assert loaded[0]['paste_path'] == ''
            assert loaded[0]['label_path'] == ''

            config_manager.upsert_memory_record({
                'note': '', 'background_path': 'b10', 'paste_path': '', 'label_path': ''
            })
            assert config_manager.load_memory_records()[0]['note'] == ''
        finally:
            config_manager.save_memory_records(original)

    def test_load_memory_records_ignores_legacy_config_key(self):
        original = config_manager.load_config()
        try:
            config_manager.save_config({
                'handy_records': [{
                    'note': 'legacy', 'background_path': 'legacy_bg',
                    'paste_path': '', 'label_path': '', 'edit_mode': 'annotate',
                }]
            })
            loaded = config_manager.load_memory_records()
            assert loaded == []
        finally:
            config_manager.save_config(original)

    def test_save_memory_records_writes_memory_and_removes_legacy_key(self):
        original = config_manager.load_config()
        try:
            config_manager.save_config({'handy_records': [{'background_path': 'old'}]})
            config_manager.save_memory_records([
                {'note': 'new', 'background_path': 'new_bg', 'paste_path': '', 'label_path': ''}
            ])
            saved = config_manager.load_config()
            assert 'memory' in saved
            assert 'handy_records' not in saved
            assert saved['memory'][0]['background_path'].endswith('new_bg')
        finally:
            config_manager.save_config(original)

    def test_memory_record_preserves_bg_label_stats(self):
        original = config_manager.load_memory_records()
        try:
            config_manager.save_memory_records([{
                'note': 'stats',
                'background_path': 'bg_stats',
                'paste_path': '',
                'label_path': '',
                'bg_label_stats': [
                    {'label': 'cat', 'count': 3, 'color': '#ff0000'},
                    {'label': '', 'count': 1, 'color': '#000'},
                    {'label': 'dog', 'count': 'bad', 'color': '#00ff00'},
                ],
            }])
            loaded = config_manager.load_memory_records()
            assert loaded[0]['bg_label_stats'] == [
                {'label': 'cat', 'count': 3, 'color': '#ff0000'},
                {'label': 'dog', 'count': 0, 'color': '#00ff00'},
            ]
            bare = config_manager._normalize_memory_record({'background_path': 'x'})
            assert bare['bg_label_stats'] == []
        finally:
            config_manager.save_memory_records(original)


class TestDefaults:

    def test_load_shortcuts_fallback(self):
        """load_shortcuts returns SHORTCUT_CONFIG when no saved shortcuts"""
        loaded = config_manager.load_shortcuts()
        for key in config.SHORTCUT_CONFIG:
            assert key in loaded


class TestCropTemplates:
    SPEC_A = {"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5,
              "square": False}
    SPEC_B = {"w": 512, "h": 512, "overlap": 64, "min_visible": 0.75,
              "square": True}

    def test_add_prepends_new_template(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            assert config_manager.add_crop_template(self.SPEC_A) is True
            templates = config_manager.load_crop_templates()
            assert len(templates) == 1
            assert templates[0]["w"] == 800
            assert templates[0]["h"] == 640
            assert templates[0]["overlap"] == 200
            assert templates[0]["min_visible"] == 0.5
            assert templates[0]["square"] is False
        finally:
            config_manager.CONFIG_PATH = original

    def test_add_duplicate_moves_to_top_without_new_entry(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.add_crop_template(self.SPEC_A)
            config_manager.add_crop_template(self.SPEC_B)
            assert config_manager.add_crop_template(self.SPEC_A) is True
            templates = config_manager.load_crop_templates()
            assert len(templates) == 2
            assert templates[0]["w"] == 800
            assert templates[1]["w"] == 512
        finally:
            config_manager.CONFIG_PATH = original

    def test_add_limit_three_evicts_oldest(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            for i in range(4):
                config_manager.add_crop_template(
                    {"w": 100 + i, "h": 100, "overlap": 10, "min_visible": 0.5})
            templates = config_manager.load_crop_templates()
            assert len(templates) == 3
            assert [t["w"] for t in templates] == [103, 102, 101]
        finally:
            config_manager.CONFIG_PATH = original

    def test_add_invalid_spec_returns_false_and_writes_nothing(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            invalid_specs = [
                None, 42, "x", [],
                {"w": 4, "h": 640, "overlap": 200, "min_visible": 0.5},
                {"w": 800, "h": 9000, "overlap": 200, "min_visible": 0.5},
                {"w": 800, "h": 640, "overlap": 5000, "min_visible": 0.5},
                {"w": 800, "h": 640, "overlap": -1, "min_visible": 0.5},
                {"w": 800, "h": 640, "overlap": 200, "min_visible": 0.3},
                {"w": 800, "h": 640, "overlap": 200, "min_visible": 1.5},
                # 显式给了坏值仍要拒（与「没给」区分开）
                {"w": 800, "h": 640, "overlap": 200, "min_visible": "x"},
            ]
            for spec in invalid_specs:
                assert config_manager.add_crop_template(spec) is False, spec
            assert config_manager.load_crop_templates() == []
            assert "crop_templates" not in config_manager.load_config()
        finally:
            config_manager.CONFIG_PATH = original

    def test_add_without_min_visible_fills_the_default(self, tmp_path):
        """没给 `min_visible` 不再算非法，按数据增强侧的默认值补齐。

        推理侧的窗口参数只有 w/h/重叠，它把模板写回共享列表时不该被
        `min_visible` 卡住（那样是静默失败：点确定却什么都没记住）。
        缺字段用默认值补齐，与「显式给了 0.3 这种越界值要拒」是两回事。
        """
        original = _with_temp_config(tmp_path)
        try:
            assert config_manager.add_crop_template(
                {"w": 800, "h": 640, "overlap": 200}) is True
            templates = config_manager.load_crop_templates()
            assert len(templates) == 1
            assert templates[0]["w"] == 800
            assert templates[0]["min_visible"] == (
                config_manager.DEFAULT_CROP_MIN_VISIBLE)
            # 与数据增强侧输入框的默认值一致，不是凭空造的数
            assert config_manager.DEFAULT_CROP_MIN_VISIBLE == 0.7
            # 其余字段也照常补齐
            assert templates[0]["square"] is False
            assert templates[0]["fill"] is False
        finally:
            config_manager.CONFIG_PATH = original

    def test_load_drops_dirty_history(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_config({"crop_templates": [
                {"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5},
                {"w": 4, "h": 640, "overlap": 200, "min_visible": 0.5},
                "garbage",
                {"w": 640, "h": 480, "overlap": 100, "min_visible": 0.6,
                 "square": True},
            ]})
            templates = config_manager.load_crop_templates()
            assert len(templates) == 2
            assert templates[0]["w"] == 800
            assert templates[1]["w"] == 640
            assert templates[1]["square"] is True
        finally:
            config_manager.CONFIG_PATH = original

    def test_load_newest_first_and_dedupes(self, tmp_path):
        original = _with_temp_config(tmp_path)
        try:
            config_manager.save_config({"crop_templates": [
                {"w": 300, "h": 300, "overlap": 30, "min_visible": 0.7},
                {"w": 200, "h": 200, "overlap": 20, "min_visible": 0.6},
                {"w": 200, "h": 200, "overlap": 20, "min_visible": 0.6},
                {"w": 100, "h": 100, "overlap": 10, "min_visible": 0.5},
                {"w": 400, "h": 400, "overlap": 40, "min_visible": 0.8},
            ]})
            templates = config_manager.load_crop_templates()
            assert [t["w"] for t in templates] == [300, 200, 100]
        finally:
            config_manager.CONFIG_PATH = original


class TestLegacyConfigMigration:
    """改名迁移：旧版 ~/.pastelabel.json 自动搬到 ~/.labelforge.json。

    这些用例替换 CONFIG_PATH 与 LEGACY_CONFIG_PATH 到临时目录，不碰用户真实文件。
    """

    def _paths(self, tmp_path, monkeypatch):
        new = Path(tmp_path) / '.labelforge.json'
        old = Path(tmp_path) / '.pastelabel.json'
        monkeypatch.setattr(config_manager, 'CONFIG_PATH', str(new))
        monkeypatch.setattr(config_manager, 'LEGACY_CONFIG_PATH', str(old))
        # 环境变量会短路迁移，这里清掉以保证走文件分支
        monkeypatch.delenv('LABELFORGE_CONFIG_PATH', raising=False)
        monkeypatch.delenv('PASTELABEL_CONFIG_PATH', raising=False)
        return new, old

    def test_moves_legacy_file_and_keeps_the_original(self, tmp_path, monkeypatch):
        new, old = self._paths(tmp_path, monkeypatch)
        old.write_text(json.dumps({'theme': 'dark'}), encoding='utf-8')

        assert config_manager.migrate_legacy_config() is True

        assert json.loads(new.read_text(encoding='utf-8')) == {'theme': 'dark'}
        # 旧文件保留：新版本出问题回退旧版本时设置还在
        assert old.exists()

    def test_does_not_overwrite_an_existing_new_file(self, tmp_path, monkeypatch):
        new, old = self._paths(tmp_path, monkeypatch)
        new.write_text(json.dumps({'theme': 'light'}), encoding='utf-8')
        old.write_text(json.dumps({'theme': 'dark'}), encoding='utf-8')

        assert config_manager.migrate_legacy_config() is False

        assert json.loads(new.read_text(encoding='utf-8')) == {'theme': 'light'}

    def test_no_legacy_file_is_not_an_error(self, tmp_path, monkeypatch):
        new, old = self._paths(tmp_path, monkeypatch)
        assert config_manager.migrate_legacy_config() is False
        assert not new.exists()

    def test_env_var_path_disables_migration(self, tmp_path, monkeypatch):
        new, old = self._paths(tmp_path, monkeypatch)
        old.write_text(json.dumps({'theme': 'dark'}), encoding='utf-8')
        monkeypatch.setenv('LABELFORGE_CONFIG_PATH', str(new))

        assert config_manager.migrate_legacy_config() is False
        assert not new.exists()

    def test_load_config_triggers_the_migration(self, tmp_path, monkeypatch):
        new, old = self._paths(tmp_path, monkeypatch)
        old.write_text(json.dumps({'theme': 'dark'}), encoding='utf-8')

        assert config_manager.load_config().get('theme') == 'dark'
        assert new.exists()

    def test_legacy_env_var_still_selects_the_path(self, tmp_path, monkeypatch):
        """老排查脚本用 PASTELABEL_CONFIG_PATH 时仍要生效。"""
        monkeypatch.delenv('LABELFORGE_CONFIG_PATH', raising=False)
        target = Path(tmp_path) / 'from-env.json'
        monkeypatch.setenv('PASTELABEL_CONFIG_PATH', str(target))
        assert config_manager._resolve_config_path() == str(target)


class TestLegacyTraceEnvVar:
    """json 探针的旧环境变量仍要认，免得老排查脚本失效。"""

    def _fresh(self, monkeypatch, value, legacy=False):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from labelforge.core import json_trace
        monkeypatch.setattr(json_trace, '_ON', None)
        monkeypatch.delenv('LABELFORGE_TRACE_JSON', raising=False)
        monkeypatch.delenv('PASTELABEL_TRACE_JSON', raising=False)
        monkeypatch.setenv('PASTELABEL_TRACE_JSON' if legacy else 'LABELFORGE_TRACE_JSON', value)
        return json_trace._on()

    def test_new_env_var_enables(self, monkeypatch):
        assert self._fresh(monkeypatch, '1') is True

    def test_legacy_env_var_still_enables(self, monkeypatch):
        assert self._fresh(monkeypatch, '1', legacy=True) is True

    def test_absent_means_off(self, monkeypatch):
        assert self._fresh(monkeypatch, '') is False
