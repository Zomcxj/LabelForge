"""
配置文件 - 定义常量配置
"""

# 支持的图片扩展名
SUPPORTED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

# 缩略图配置
THUMBNAIL_CONFIG = {
    'grid_width': 56,  # 网格宽度（像素）
    'grid_height': 60,  # 网格高度（像素）
    'spacing': 2,  # 网格间距（像素）
}

# 贴图参数默认值
PASTE_PARAMS = {
    'default_count': 1,  # 默认贴图个数
    'min_count': 1,  # 最小贴图个数
    'max_count': 20,  # 最大贴图个数
    'default_min_size': 30,  # 默认最小尺寸
    'min_size_range': (15, 100),  # 最小尺寸范围
    'default_max_size': 60,  # 默认最大尺寸
    'max_size_range': (30, 200),  # 最大尺寸范围
}

# 背景图缩放配置
BACKGROUND_SCALE_CONFIG = {
    'default_scale': 1.0,  # 默认缩放比例
    'min_scale': 0.5,  # 最小缩放比例
    'max_scale': 3.0,  # 最大缩放比例
    'scale_factor': 1.1,  # 滚轮缩放因子
}

# 检测框配置
DETECTION_BOX_CONFIG = {
    'min_width': 3,  # 最小宽度
    'min_height': 3,  # 最小高度
    'resize_handle_size': 8,  # 调整手柄大小
    'max_polygon_points': 32,  # 多边形最大顶点数
    'label_font_size': 9,  # 标签名字号
    'label_position': 'outside',  # 标签名位置：outside=框外侧，inside=框内侧
    'border_color_selected': (0, 255, 128),  # 选中边框颜色 (RGB)
    'border_color_normal': (0, 255, 128),  # 普通边框颜色 (RGB)
    'fill_color_selected': (0, 255, 128, 50),  # 选中填充颜色 (RGBA)
}

# 语义区域配置（仅运行时存在，不持久化）
#: 区域句柄/边框比检测框大多少像素。两处都引用它，改一处两边一起变。
REGION_HANDLE_EXTRA = 3
REGION_BORDER_EXTRA = 1

REGION_CONFIG = {
    'min_size': 10,            # 区域最小边长（图像像素）
    'border_color': '#00D4FF',  # 区域边框颜色（青色）
    'border_width': 2,         # 参考值；实际绘制走「检测框 + REGION_BORDER_EXTRA」
    'dash_pattern': (6, 4),    # 虚线样式
    'handle_size': 9,          # 参考值；实际绘制走「检测框 + REGION_HANDLE_EXTRA」
    'fill_alpha': 30,          # 选中时填充透明度
    'polygon_max_points': 64,  # 多边形区域最大顶点数
    'polygon_max_retries': 200,  # 多边形区域内采样贴图中心的最大重试次数
}

# 贴图配置
PASTE_ITEM_CONFIG = {
    'border_color': (135, 206, 250),  # 天蓝色边框 (RGB)
    'border_width_selected': 3,  # 选中时边框宽度
    'border_width_normal': 1,  # 正常时边框宽度
    'handle_size': 8,  # 缩放手柄大小
    'min_size': 15,  # 最小尺寸
    'base_scale_factor': 0.5,  # 基础缩放因子
}

# 窗口配置
WINDOW_CONFIG = {
    'default_width': 1600,  # 默认窗口宽度
    'default_height': 1000,  # 默认窗口高度
    'canvas_min_width': 800,  # 画布最小宽度
    'canvas_min_height': 600,  # 画布最小高度
    'control_panel_width': 300,  # 控制面板宽度
}

# Labelme JSON 版本
LABELME_VERSION = "5.0.1"

# 输出目录后缀
OUTPUT_DIR_SUFFIX = "_paste_output"

# 默认前缀
DEFAULT_PREFIX = "paste"

# 随机贴图参数
RANDOM_POSITION_CONFIG = {
    'margin_left': 50,       # 左侧边距
    'margin_top': 200,       # 顶部边距
    'margin_right': 100,     # 右侧安全边距
    'max_retries': 1000,     # 最大重试次数
    'overlap_iou_detection': 0.02,  # 与检测框的 IoU 阈值
    'overlap_iou_pasted': 0.1,      # 与已贴图的 IoU 阈值
    'min_edge_size': 35,     # 手动添加贴图最小边
}

# 自动保存配置
AUTO_SAVE_CONFIG = {
    'enabled': False,  # 默认不启用自动保存
}

# 窗口放大器配置
MAGNIFIER_CONFIG = {
    'enabled': False,
    'size': 160,
    'zoom': 2.0,
    'position': 'side',  # 显示位置: side(侧边) / center(鼠标中心)
}

# 主题配置
THEME_CONFIG = {
    'default_mode': 'light',  # 默认主题: 'light' 或 'dark'
}

# 状态栏配置
STATUSBAR_CONFIG = {
    'max_labels': 3,  # 状态栏最多显示的类别数
}
GRID_CONFIG = {
    'enabled': False,           # 默认关闭
    'spacing': 50,              # 网格间距（像素）
    'color_light': '#E0E0E0',   # 浅色模式网格线颜色
    'color_dark': '#3E3E3E',    # 深色模式网格线颜色
    'color_ink': '#C8BFB0',     # 水墨模式网格线颜色
    'line_width': 1,            # 网格线粗细（像素）
    'alpha': 120,               # 网格线透明度（0-255）
}

# 标签颜色配置 - 高对比度颜色
LABEL_COLORS = [
    '#43A047', '#D81B60', '#8E24AA', '#5E35B1',
    '#3949AB', '#1E88E5', '#039BE5', '#00ACC1',
    '#00897B', '#E53935', '#7CB342', '#C0CA33',
    '#FDD835', '#FFB300', '#FB8C00', '#F4511E',
    '#6D4C41', '#757575', '#546E7A', '#C62828',
    '#AD1457', '#7B1FA2', '#4527A0', '#283593',
    '#1565C0', '#0277BD', '#00838F', '#00695C',
    '#2E7D32', '#558B2F',
]

# 快捷键配置
SHORTCUT_CONFIG = {
    'undo': 'Ctrl+Z',
    'redo': 'Ctrl+Y',
    'toggle_grid': 'Ctrl+G',
    'toggle_labels': 'R',
    'toggle_label_names': 'T',
    'toggle_scores': 'Y',
    'auto_save_b': 'G',
    'auto_save_p': 'H',
    'toggle_paste_names': 'F',
    'draw_box': 'W',
    'draw_polygon': 'P',
    'draw_point': 'K',
    'draw_obb': 'O',
    'rotate_cw': 'Z',
    'rotate_ccw': 'X',
    'quit_draw': 'Q',
    'next_image': 'D',
    'prev_image': 'A',
    'delete_selected': 'E',
    'fit_view': 'Ctrl+F',
    'zoom_in': 'Ctrl++',
    'zoom_out': 'Ctrl+-',
    'remove_image': 'Ctrl+Shift+D',
    'restore_image': 'Ctrl+Shift+R',
    'delete_label': 'Ctrl+Shift+Z',
}

LABEL_CACHE_SLOTS = [
    {'name': '缓存槽1', 'locked': False, 'items': [], 'shortcut': '1', 'copy_order': 0, 'copied_at': ''},
    {'name': '缓存槽2', 'locked': False, 'items': [], 'shortcut': '2', 'copy_order': 0, 'copied_at': ''},
    {'name': '缓存槽3', 'locked': False, 'items': [], 'shortcut': '3', 'copy_order': 0, 'copied_at': ''},
]

# 撤销/重做配置
UNDO_CONFIG = {
    'max_history': 50,  # 最大历史记录数
}

# 检测框锁定配置
LOCK_CONFIG = {
    'enabled': False,  # 默认不锁定
}

# 方向键步进移动配置
NUDGE_CONFIG = {
    'step': 1,
}

# 检测框与贴图框统一边框粗细（0.2-2.0）
BOX_BORDER_CONFIG = {'width': 1.0}

# 检测框滚轮编辑配置
DETECTION_BOX_WHEEL_CONFIG = {
    'detection_box_scale_step': 0.05,  # 检测框内滚轮整体缩放步长
    'paste_item_scale_step': 0.15,  # 贴图内滚轮整体缩放步长
    'region_scale_step': 0.05,  # 区域悬停滚轮缩放步长
    'edge_step': 5,      # 框外滚轮单边位移像素
}

# 标注模式十字线配置
CROSSHAIR_CONFIG = {
    'width': 1.0,
    'color': '#00FF80',
    'alpha': 160,
}

# OBB 旋转配置
OBB_CONFIG = {
    'rotate_step': 1,  # 键盘旋转单次角度（度）
}

# 标注质检 lint 配置
QUALITY_LINT_CONFIG = {
    'bounds_tolerance': 1.0,    # 越界容忍像素
    'dup_iou': 0.85,            # 同类框重复判定 IoU 阈值
    'cross_iou': 0.85,          # 异类框重叠判定 IoU 阈值
    'name_edit_distance': 0,    # 近似名编辑距离阈值（0=仅大小写/空白差异，避免业务命名误报）
    'max_boxes_pairwise': 500,  # 超过此框数跳过两两比对
    # 小目标可检测性：把原图与框**等比缩放**（长边→640、短边补边）到模型输入后，
    # 框短边占 640 的比例 = min(w, h) / max(W, H)。阈值就是这个空间里的
    # 4px / 8px / 16px —— 判据与训练分辨率无关，只有输入尺寸不是 640 时才需要
    # 重新标定（见 quality_lint.TINY_INPUT_SIZE）。
    'tiny_fail_ratio': 0.00625,   # 低于此值：常规配置检不到（🔴）
    'tiny_warn_ratio': 0.0125,    # 低于此值：需 P2 头或滑窗放大（🟡）
    'tiny_stable_ratio': 0.025,   # 达到此值：稳定检出（报告参考线，不产问题）
}

# 滑窗裁剪补边默认色（YOLO letterbox 灰）
FILL_RGB_DEFAULT = (114, 114, 114)


def normalized_fill_rgb(spec):
    """从 spec 取补边色并规范成 (r, g, b) 整数三元组；缺失或损坏一律回落默认灰。

    放在 core 而非 engine：持久化层（config_manager）与引擎层必须共用同一份规则，
    而 core 是能被独立加载的底层，不能反向依赖 engine。
    模板从磁盘读回可能损坏，故非数值/长度不符都回落，不抛异常。
    """
    raw = (spec or {}).get("fill_rgb")
    if raw is None:
        return FILL_RGB_DEFAULT
    try:
        parts = [int(c) for c in raw]
    except (TypeError, ValueError):
        return FILL_RGB_DEFAULT
    if len(parts) != 3:
        return FILL_RGB_DEFAULT
    return tuple(max(0, min(255, v)) for v in parts)


# 出厂默认快照：这些字典在运行时会被就地修改（设置对话框、加载配置），
# 「恢复默认设置」需要一份未被污染的副本来还原。
_DEFAULT_SNAPSHOT = {
    name: dict(value)
    for name, value in list(globals().items())
    if name.endswith('_CONFIG') and isinstance(value, dict)
}


def reset_defaults():
    """把所有可变的配置字典还原为模块导入时的默认值。"""
    for name, snapshot in _DEFAULT_SNAPSHOT.items():
        target = globals().get(name)
        if isinstance(target, dict):
            target.clear()
            target.update(snapshot)
