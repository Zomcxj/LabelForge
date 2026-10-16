"""
配置管理器 - 统一管理用户配置的加载和保存
"""
import os
import json
from . import config as config_module
from .config import SHORTCUT_CONFIG, STATUSBAR_CONFIG, DETECTION_BOX_CONFIG, MAGNIFIER_CONFIG, LABEL_CACHE_SLOTS, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG, BOX_BORDER_CONFIG, LABEL_COLORS, OBB_CONFIG, normalized_fill_rgb


LEGACY_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), '.pastelabel.json')


def _resolve_config_path():
    """配置文件路径。新环境变量优先，旧变量仍认（老排查脚本不至于失效）。"""
    return (os.environ.get('LABELFORGE_CONFIG_PATH')
            or os.environ.get('PASTELABEL_CONFIG_PATH')
            or os.path.join(os.path.expanduser("~"), '.labelforge.json'))


CONFIG_PATH = _resolve_config_path()
MEMORY_LIMIT = 10
DISABLED_SHORTCUT_ACTIONS = {'save', 'save_all'}


def migrate_legacy_config():
    """把旧版 `~/.pastelabel.json` 搬到 `~/.labelforge.json`，老用户无感升级。

    只在两个前提下搬：新路径不存在、旧路径存在。旧文件保留不删——万一新版本
    有问题，回退旧版本时设置还在。环境变量指定了路径就完全不碰。
    """
    if os.environ.get('LABELFORGE_CONFIG_PATH') or os.environ.get(
            'PASTELABEL_CONFIG_PATH'):
        return False
    if os.path.exists(CONFIG_PATH) or not os.path.exists(LEGACY_CONFIG_PATH):
        return False
    try:
        with open(LEGACY_CONFIG_PATH, 'rb') as src:
            data = src.read()
        with open(CONFIG_PATH, 'wb') as dst:
            dst.write(data)
    except OSError:
        return False
    return True


def _normalize_label_cache_slots(slots):
    defaults = [dict(slot) for slot in LABEL_CACHE_SLOTS]
    if not isinstance(slots, list):
        return defaults

    normalized = []
    for index, default in enumerate(defaults):
        slot = slots[index] if index < len(slots) and isinstance(slots[index], dict) else {}
        items = slot.get('items', default['items'])
        normalized.append({
            'name': str(slot.get('name', default['name']) or default['name']),
            'locked': bool(slot.get('locked', default['locked'])),
            'items': items if isinstance(items, list) else [],
            'shortcut': str(slot.get('shortcut', default['shortcut']) or default['shortcut']),
            'copy_order': int(slot.get('copy_order', 0) or 0),
            'copied_at': str(slot.get('copied_at', '') or ''),
        })
    return normalized


def _normalize_lint_ignored_rules(rules):
    """质检忽略规则：{kind: [key,...]}。key 为 detail 或 '*'（整类忽略）。"""
    if not isinstance(rules, dict):
        return {}
    normalized = {}
    for kind, keys in rules.items():
        if not isinstance(kind, str) or not kind:
            continue
        if isinstance(keys, str):
            keys = [keys]
        if not isinstance(keys, list):
            continue
        cleaned = sorted({str(k) for k in keys if str(k or '').strip()})
        if cleaned:
            normalized[kind] = cleaned
    return normalized


def _normalize_inference_settings(raw):
    """推理配置：只保留已知字段，避免旧配置里的杂项一直留在文件里。"""
    from labelforge.engine.inference_task import normalize_settings
    if not isinstance(raw, dict):
        raw = {}
    normalized = normalize_settings(raw)
    normalized.pop("class_filter", None)
    normalized.pop("replace_auto", None)
    return normalized


def _filter_shortcuts(shortcuts):
    """过滤已禁用的快捷键动作，避免旧配置继续生效。"""
    if not isinstance(shortcuts, dict):
        return {}
    return {k: v for k, v in shortcuts.items() if k not in DISABLED_SHORTCUT_ACTIONS}


def _normalize_label_colors(colors, extend=False):
    if not isinstance(colors, list) or not colors:
        return list(LABEL_COLORS)
    normalized = [str(color) for color in colors]
    try:
        if all(len(color) == 7 and color.startswith('#') and int(color[1:], 16) >= 0 for color in normalized):
            # 旧版默认调色板是同一组颜色但把红色排在首位（#E53935）。若保存的
            # 调色板集合与当前默认相同且仍是红打头，按当前默认顺序迁移，否则
            # 新标签会继续被分到刺眼的红色。
            if normalized[0] == '#E53935' and set(normalized) == set(LABEL_COLORS):
                return list(LABEL_COLORS)
            if not extend:
                return normalized
            # Keep custom colors first, then fill from defaults without duplicates.
            palette = []
            for color in normalized + list(LABEL_COLORS):
                if color not in palette:
                    palette.append(color)
                if len(palette) >= 30:
                    return palette
    except ValueError:
        pass
    return list(LABEL_COLORS)


def _normalize_label_color_map(color_map):
    if not isinstance(color_map, dict):
        return {}
    normalized = {}
    for label, color in color_map.items():
        label = str(label)
        color = str(color)
        try:
            if label and len(color) == 7 and color.startswith('#') and int(color[1:], 16) >= 0:
                normalized[label] = color
        except ValueError:
            continue
    return normalized


def get_label_color(labels, label, palette=None, label_color_map=None):
    """获取标签颜色，优先使用已保存的显式颜色。"""
    colors = _normalize_label_colors(
        palette if palette is not None else LABEL_COLORS, extend=palette is None,
    )
    if not label:
        return colors[0]
    label = str(label)
    color_map = _normalize_label_color_map(
        load_config().get('label_color_map') if palette is None and label_color_map is None
        else label_color_map
    )
    if label in color_map:
        return color_map[label]
    if len(colors) >= 30:
        # 把待分配标签并入有序集合，保证“未进入会话”的标签也走确定索引分配，
        # 不再落哈希兜底（哈希会把不同标签撞到同一红色系）。
        unique_labels = sorted({str(value) for value in labels if value} | {label})
        start = unique_labels.index(label) % len(colors)
        used_colors = set(color_map.values())
        for offset in range(len(colors)):
            color = colors[(start + offset) % len(colors)]
            if color not in used_colors:
                return color
        return colors[start]
    idx = sum(ord(c) * (i + 1) for i, c in enumerate(label)) % len(colors)
    return colors[idx]


def _repair_label_color_map(color_map, palette):
    """修复历史脏数据：旧算法把不同标签写成同一颜色时整表按确定算法重排。

    颜色撞车是旧哈希兜底的确定特征；一旦发现，就整表重排，既清掉撞色，
    也顺带替换掉旧算法写入的其他脏色（如非调色板色、首标签红色）。
    颜色互不重复的表原样返回，避免覆盖用户手动选色。
    """
    values = list(color_map.values())
    if len(values) == len(set(values)):
        return color_map
    colors = _normalize_label_colors(palette, extend=palette is None)
    labels = sorted(color_map.keys())
    rebuilt = {}
    for label in labels:
        rebuilt[label] = get_label_color(labels, label, colors, rebuilt)
    return rebuilt


def get_config_path():
    """获取配置文件路径"""
    return CONFIG_PATH


def load_config():
    """加载完整配置"""
    migrate_legacy_config()
    config = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception:
            pass
    
    # 迁移旧版快捷键：delete_label 从 Ctrl+Shift+Delete 改为 Ctrl+Shift+Z
    shortcuts = config.get('shortcuts', {})
    if shortcuts.get('delete_label') == 'Ctrl+Shift+Delete':
        shortcuts['delete_label'] = 'Ctrl+Shift+Z'
        config['shortcuts'] = shortcuts
        save_config(config)  # 自动保存迁移后的配置

    return config


def save_config(config):
    """保存完整配置"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


def load_shortcuts():
    """加载快捷键配置"""
    config = load_config()
    return _filter_shortcuts(config.get('shortcuts', SHORTCUT_CONFIG))


def save_shortcuts(shortcuts):
    """保存快捷键配置"""
    config = load_config()
    config['shortcuts'] = _filter_shortcuts(shortcuts)
    return save_config(config)


def load_theme():
    """加载主题配置"""
    config = load_config()
    return config.get('theme', 'light')


def save_theme(theme):
    """保存主题配置"""
    config = load_config()
    config['theme'] = theme
    return save_config(config)


def load_language():
    """加载语言配置"""
    config = load_config()
    return config.get('language', 'zh')


def save_language(language):
    """保存语言配置"""
    config = load_config()
    config['language'] = language
    return save_config(config)


def _normalize_memory_path(value):
    value = str(value or '').strip()
    return os.path.normpath(value) if value else ''


def _normalize_bg_label_stats(value):
    """Normalize background label stats snapshot for memory records."""
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get('label', '') or '').strip()
        if not label:
            continue
        try:
            count = max(0, int(item.get('count', 0) or 0))
        except (TypeError, ValueError):
            count = 0
        color = str(item.get('color', '') or '').strip()
        result.append({'label': label, 'count': count, 'color': color})
    return result


def _normalize_memory_record(record):
    try:
        background_index = max(0, int(record.get('background_index', 0) or 0))
    except (TypeError, ValueError):
        background_index = 0
    edit_mode = record.get('edit_mode', 'paste')
    if edit_mode not in ('paste', 'annotate'):
        edit_mode = 'paste'
    return {
        'note': str(record.get('note', '') or ''),
        'background_path': _normalize_memory_path(record.get('background_path', '')),
        'paste_path': _normalize_memory_path(record.get('paste_path', '')),
        'label_path': _normalize_memory_path(record.get('label_path', '')),
        'background_index': background_index,
        'edit_mode': edit_mode,
        'updated_at': str(record.get('updated_at', '') or ''),
        'bg_label_stats': _normalize_bg_label_stats(record.get('bg_label_stats', [])),
    }


def _memory_key(record):
    return (record['background_path'], record['paste_path'], record['label_path'])


def load_memory_records():
    """加载记忆记录，最多返回 10 条。"""
    config = load_config()
    records = config.get('memory', [])
    if not isinstance(records, list):
        return []
    return [_normalize_memory_record(r) for r in records if isinstance(r, dict)][:MEMORY_LIMIT]


def save_memory_records(records):
    """保存记忆记录，按路径组合去重并限制 10 条。"""
    seen = set()
    normalized = []
    for record in records:
        if not isinstance(record, dict):
            continue
        item = _normalize_memory_record(record)
        if not any([item['background_path'], item['paste_path'], item['label_path']]):
            continue
        key = _memory_key(item)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
        if len(normalized) >= MEMORY_LIMIT:
            break
    config = load_config()
    config['memory'] = normalized
    config.pop('handy_records', None)
    return save_config(config)


def upsert_memory_record(record):
    """新增或更新记忆记录，最新记录排在最前。"""
    item = _normalize_memory_record(record)
    if not any([item['background_path'], item['paste_path'], item['label_path']]):
        return False
    key = _memory_key(item)
    remaining = [r for r in load_memory_records() if _memory_key(r) != key]
    return save_memory_records([item] + remaining)


def delete_memory_record(index):
    records = load_memory_records()
    if index < 0 or index >= len(records):
        return False
    records.pop(index)
    return save_memory_records(records)


CROP_TEMPLATES_LIMIT = 3

# 模板里 `min_visible` 的兜底值。与数据增强侧输入框的默认值（0.7）一致——
# 推理侧不传该字段时用它补齐，而不是凭空造一个 0.5 或让整条模板被丢掉。
DEFAULT_CROP_MIN_VISIBLE = 0.7


def _normalize_crop_template(spec):
    """裁剪模板：{w, h, overlap, min_visible, square, fill, fill_rgb}，非法条目返回 None。

    `min_visible` 是**数据增强侧**的概念（跨窗标注的最低可见比例）。推理侧
    只关心 w/h/重叠，调用方不传该字段时按 `DEFAULT_CROP_MIN_VISIBLE` 补齐，
    而不是让整条模板因为缺字段被丢掉（那会让推理侧点确定后「什么都没记住」，
    且失败是静默的）。
    """
    if not isinstance(spec, dict):
        return None
    try:
        w = int(spec.get('w', 0))
        h = int(spec.get('h', 0))
        overlap = int(spec.get('overlap', 0))
    except (TypeError, ValueError):
        return None
    raw_visible = spec.get('min_visible')
    if raw_visible is None:
        min_visible = DEFAULT_CROP_MIN_VISIBLE
    else:
        try:
            min_visible = float(raw_visible)
        except (TypeError, ValueError):
            return None
    if not (8 <= w <= 8192 and 8 <= h <= 8192):
        return None
    if not (0 <= overlap <= 4096):
        return None
    if not (0.5 <= min_visible <= 1.0):
        return None
    return {
        'w': w, 'h': h, 'overlap': overlap,
        'min_visible': round(min_visible, 4),
        'square': bool(spec.get('square')),
        'fill': bool(spec.get('fill')),
        'fill_rgb': list(normalized_fill_rgb(spec)),
    }


def _crop_template_key(t):
    return (t['w'], t['h'], t['overlap'], t['min_visible'], t['square'],
            t['fill'], tuple(t['fill_rgb']))


def load_crop_templates():
    """加载最近裁剪模板，最多 3 条，最新在前。"""
    config = load_config()
    records = config.get('crop_templates', [])
    if not isinstance(records, list):
        return []
    templates = []
    seen = set()
    for record in records:
        item = _normalize_crop_template(record)
        if item is None:
            continue
        key = _crop_template_key(item)
        if key in seen:
            continue
        seen.add(key)
        templates.append(item)
        if len(templates) >= CROP_TEMPLATES_LIMIT:
            break
    return templates


def add_crop_template(spec):
    """记录一条裁剪模板：完全相同的去重置顶，最多保留 3 条。"""
    item = _normalize_crop_template(spec)
    if item is None:
        return False
    key = _crop_template_key(item)
    rest = [t for t in load_crop_templates() if _crop_template_key(t) != key]
    config = load_config()
    config['crop_templates'] = [item] + rest[:CROP_TEMPLATES_LIMIT - 1]
    return save_config(config)


def reset_all():
    """恢复出厂设置：删除配置文件，并把内存里的配置字典还原为默认值。

    返回 True 表示配置文件已删除（或本来就不存在）。
    """
    removed = True
    if os.path.exists(CONFIG_PATH):
        try:
            os.remove(CONFIG_PATH)
        except OSError:
            removed = False
    config_module.reset_defaults()
    return removed


def load_all():
    """加载所有配置"""
    config = load_config()
    legacy_wheel_scale_step = config.get('detection_box_wheel_scale_step')
    saved_sc = _filter_shortcuts(config.get('shortcuts', {}))
    if saved_sc.get('delete_selected') == 'Delete':
        saved_sc['delete_selected'] = SHORTCUT_CONFIG['delete_selected']
    merged_sc = {**SHORTCUT_CONFIG, **saved_sc}
    normalized_palette = _normalize_label_colors(config.get('label_colors'), extend=True)
    return {
        'shortcuts': merged_sc,
        'theme': config.get('theme', 'light'),
        'language': config.get('language', 'zh'),
        'max_labels': config.get('max_labels', STATUSBAR_CONFIG['max_labels']),
        'grid_line_width': config.get('grid_line_width', None),
        'grid_alpha': config.get('grid_alpha', None),
        'resize_handle_size': config.get('resize_handle_size', DETECTION_BOX_CONFIG['resize_handle_size']),
        'max_polygon_points': max(3, min(64, int(config.get(
            'max_polygon_points', DETECTION_BOX_CONFIG['max_polygon_points'])))),
        "rotate_step": max(1, min(6, int(config.get(
            "rotate_step", OBB_CONFIG['rotate_step'])))),
        'label_font_size': config.get('label_font_size', DETECTION_BOX_CONFIG['label_font_size']),
        'label_position': config.get('label_position', DETECTION_BOX_CONFIG['label_position']),
        'canvas_image_copy_enabled': bool(config.get('canvas_image_copy_enabled', False)),
        'relative_path_display': bool(config.get('relative_path_display', False)),
        'magnifier_enabled': bool(config.get('magnifier_enabled', False)),
        # 默认开启：画布空了就删掉 sidecar json
        'delete_empty_json_on_switch': bool(config.get('delete_empty_json_on_switch', True)),
        'magnifier_position': str(config.get('magnifier_position', MAGNIFIER_CONFIG['position'])),
        'magnifier_zoom': float(config.get('magnifier_zoom', MAGNIFIER_CONFIG['zoom'])),
        'magnifier_size': int(config.get('magnifier_size', MAGNIFIER_CONFIG['size'])),
        'label_cache_slots': _normalize_label_cache_slots(config.get('label_cache_slots')),
        'nudge_step': int(config.get('nudge_step', NUDGE_CONFIG['step'])),
        'lint_ignored_rules': _normalize_lint_ignored_rules(config.get('lint_ignored_rules')),
        'inference_settings': _normalize_inference_settings(config.get('inference_settings')),
        'detection_box_scale_step': float(config.get(
            'detection_box_scale_step',
            legacy_wheel_scale_step if legacy_wheel_scale_step is not None else DETECTION_BOX_WHEEL_CONFIG['detection_box_scale_step'],
        )),
        'paste_item_scale_step': float(config.get(
            'paste_item_scale_step',
            legacy_wheel_scale_step if legacy_wheel_scale_step is not None else DETECTION_BOX_WHEEL_CONFIG['paste_item_scale_step'],
        )),
        'detection_box_wheel_edge_step': int(config.get('detection_box_wheel_edge_step', DETECTION_BOX_WHEEL_CONFIG['edge_step'])),
        'crosshair_width': float(config.get('crosshair_width', CROSSHAIR_CONFIG['width'])),
        'crosshair_color': str(config.get('crosshair_color', CROSSHAIR_CONFIG['color'])),
        'crosshair_alpha': int(config.get('crosshair_alpha', CROSSHAIR_CONFIG['alpha'])),
        'box_border_width': float(config.get('box_border_width', BOX_BORDER_CONFIG['width'])),
        'label_colors': normalized_palette,
        'label_color_map': _repair_label_color_map(
            _normalize_label_color_map(config.get('label_color_map')), normalized_palette,
        ),
        'memory': load_memory_records(),
    }


def save_all(shortcuts=None, theme=None, language=None, max_labels=None,
             grid_line_width=None, grid_alpha=None, resize_handle_size=None,
             max_polygon_points=None,
             rotate_step=None,
             label_font_size=None, label_position=None,
             canvas_image_copy_enabled=None, relative_path_display=None,
             magnifier_enabled=None, delete_empty_json_on_switch=None,
             magnifier_position=None, magnifier_zoom=None, magnifier_size=None,
             label_cache_slots=None, nudge_step=None,
               detection_box_scale_step=None, paste_item_scale_step=None,
               detection_box_wheel_edge_step=None,
              crosshair_width=None, crosshair_color=None, crosshair_alpha=None,
               box_border_width=None, label_colors=None, label_color_map=None,
               lint_ignored_rules=None, inference_settings=None):
    """保存所有配置"""
    config = load_config()
    if shortcuts is not None:
        config['shortcuts'] = _filter_shortcuts(shortcuts)
    if theme is not None:
        config['theme'] = theme
    if language is not None:
        config['language'] = language
    if max_labels is not None:
        config['max_labels'] = max_labels
    if grid_line_width is not None:
        config['grid_line_width'] = grid_line_width
    if grid_alpha is not None:
        config['grid_alpha'] = grid_alpha
    if resize_handle_size is not None:
        config['resize_handle_size'] = resize_handle_size
    if max_polygon_points is not None:
        config['max_polygon_points'] = max(3, min(64, int(max_polygon_points)))
    if rotate_step is not None:
        config['rotate_step'] = max(1, min(6, int(rotate_step)))
    if label_font_size is not None:
        config['label_font_size'] = label_font_size
    if label_position is not None:
        config['label_position'] = label_position
    if canvas_image_copy_enabled is not None:
        config['canvas_image_copy_enabled'] = bool(canvas_image_copy_enabled)
    if relative_path_display is not None:
        config['relative_path_display'] = bool(relative_path_display)
    if magnifier_enabled is not None:
        config['magnifier_enabled'] = bool(magnifier_enabled)
    if delete_empty_json_on_switch is not None:
        config['delete_empty_json_on_switch'] = bool(delete_empty_json_on_switch)
    if magnifier_position is not None:
        config['magnifier_position'] = str(magnifier_position)
    if magnifier_zoom is not None:
        config['magnifier_zoom'] = max(0.8, min(3.0, float(magnifier_zoom)))
    if magnifier_size is not None:
        config['magnifier_size'] = max(80, min(400, int(magnifier_size)))
    if label_cache_slots is not None:
        config['label_cache_slots'] = _normalize_label_cache_slots(label_cache_slots)
    if nudge_step is not None:
        config['nudge_step'] = max(1, min(5, int(nudge_step)))
    if lint_ignored_rules is not None:
        config['lint_ignored_rules'] = _normalize_lint_ignored_rules(lint_ignored_rules)
    if detection_box_scale_step is not None:
        config['detection_box_scale_step'] = max(0.01, min(0.30, float(detection_box_scale_step)))
    if paste_item_scale_step is not None:
        config['paste_item_scale_step'] = max(0.01, min(0.30, float(paste_item_scale_step)))
    if detection_box_scale_step is not None or paste_item_scale_step is not None:
        config.pop('detection_box_wheel_scale_step', None)
    if detection_box_wheel_edge_step is not None:
        config['detection_box_wheel_edge_step'] = max(1, min(50, int(detection_box_wheel_edge_step)))
    if crosshair_width is not None:
        config['crosshair_width'] = max(0.5, min(3.0, float(crosshair_width)))
    if crosshair_color is not None:
        color = str(crosshair_color)
        config['crosshair_color'] = color if len(color) == 7 and color.startswith('#') else CROSSHAIR_CONFIG['color']
    if crosshair_alpha is not None:
        config['crosshair_alpha'] = max(0, min(255, int(crosshair_alpha)))
    if box_border_width is not None:
        config['box_border_width'] = max(1, min(4, float(box_border_width)))
    if label_colors is not None:
        config['label_colors'] = _normalize_label_colors(label_colors, extend=True)
    if label_color_map is not None:
        config['label_color_map'] = _normalize_label_color_map(label_color_map)
    if inference_settings is not None:
        config['inference_settings'] = _normalize_inference_settings(inference_settings)
    return save_config(config)
