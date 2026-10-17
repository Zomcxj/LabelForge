"""
工具模块 - 提供跨平台路径处理、排序等工具函数
"""
import os
import sys
import re
from PySide6.QtGui import QPixmap, QIcon, QColor, QPainter
from PySide6.QtCore import Qt, QSize


class PathUtils:
    """跨平台路径处理工具类"""
    
    @staticmethod
    def normalize_path(path):
        """
        规范化路径，使用当前操作系统的路径分隔符
        :param path: 输入路径
        :return: 规范化后的路径
        """
        if not path:
            return path
        return os.path.normpath(path)
    
    @staticmethod
    def to_display_path(path):
        """
        将路径转换为显示格式
        Windows 下使用反斜杠，Linux 下使用正斜杠
        :param path: 输入路径
        :return: 显示格式的路径
        """
        if not path:
            return path
        normalized = os.path.normpath(path)
        if sys.platform == 'win32':
            return normalized.replace('/', '\\')
        else:
            return normalized.replace('\\', '/')
    
    @staticmethod
    def to_relative_display_path(path, base):
        """
        把路径转成相对 base 目录的显示路径。

        base 为空、path 与 base 不同盘符（Windows）时回退为绝对显示路径。
        :param path: 要显示的路径
        :param base: 基准目录
        :return: 显示格式的相对路径，无法相对化时返回绝对显示路径
        """
        if not path:
            return path
        if not base:
            return PathUtils.to_display_path(path)
        try:
            relative = os.path.relpath(path, base)
        except ValueError:
            # Windows 下跨盘符无法求相对路径
            return PathUtils.to_display_path(path)
        return PathUtils.to_display_path(relative)

    @staticmethod
    def to_file_path(path):
        """
        将路径转换为文件系统格式
        使用当前操作系统的路径分隔符
        :param path: 输入路径
        :return: 文件系统格式的路径
        """
        if not path:
            return path
        return os.path.normpath(path)
    
    @staticmethod
    def join_path(*args):
        """
        跨平台路径拼接
        :param args: 路径组件
        :return: 拼接后的路径
        """
        return os.path.join(*args)
    
    @staticmethod
    def get_path_separator():
        """
        获取当前操作系统的路径分隔符
        :return: 路径分隔符
        """
        return os.sep
    
    @staticmethod
    def get_output_dir(original_file_path):
        """
        生成输出目录路径（不创建目录，由实际保存时创建）
        :param original_file_path: 原始文件路径
        :return: 输出目录路径
        """
        from .config import OUTPUT_DIR_SUFFIX
        background_dir = os.path.dirname(original_file_path)
        output_dir = f"{background_dir}{OUTPUT_DIR_SUFFIX}"
        return output_dir
    
    @staticmethod
    def generate_save_path(original_file_path, prefix=None):
        """
        生成保存路径
        :param original_file_path: 原始文件路径
        :param prefix: 文件名前缀
        :return: (完整路径，基础文件名，前缀)
        """
        from .config import DEFAULT_PREFIX
        
        original_file_name = os.path.basename(original_file_path)
        output_dir = PathUtils.get_output_dir(original_file_path)
        
        # 处理前缀
        if not prefix:
            prefix = ""
        
        # 生成文件名
        if prefix:
            base_name = f"{prefix}_{original_file_name}"
        else:
            base_name = original_file_name
        
        file_path = os.path.join(output_dir, base_name)
        return (file_path, base_name, prefix)


def output_sidecar_paths(image_path, output_dir):
    """输出目录中属于该图片的 sidecar 路径（前缀可变，按文件名 stem 匹配）。

    保存名是 `{prefix}_{stem}.json`（前缀由用户自由输入，可为空），不能
    硬编码 `{stem}.json`；按 stem 完全相等或以 `_{stem}` 结尾匹配。
    每图一次 listdir：超大输出目录可改为按 output_dir 缓存目录清单。
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    exact = []
    suffix = []
    try:
        names = os.listdir(output_dir)
    except OSError:
        return exact
    for name in names:
        if not name.lower().endswith('.json'):
            continue
        base = name[:-5]
        if base == stem:
            exact.append(os.path.join(output_dir, name))
        elif base.endswith(f"_{stem}"):
            suffix.append(os.path.join(output_dir, name))
    # 精确匹配优先：`big_cat.json` 不该被 `cat.png` 当成自己的 sidecar
    # （残留：无精确匹配时 `_{stem}` 后缀仍可能撞名，前缀由用户自由输入，无法完全消除）
    return exact or suffix


def natural_sort_key(s):
    """
    自然排序键函数，用于正确处理数字和字母的混合排序
    :param s: 字符串
    :return: 排序键
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', s)]


def format_size_ratio(width, height, image_width, image_height):
    """把框的像素宽高格式化成「像素 + 占图片百分比」。

    占比基准是图片原始宽高，与画布缩放无关。图片尺寸缺失或非正时
    只返回像素部分，避免除零。
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        w, h = 0, 0
    base = f"W:{w} H:{h}"
    try:
        img_w, img_h = float(image_width), float(image_height)
        raw_w, raw_h = float(width), float(height)
    except (TypeError, ValueError):
        return base
    if img_w <= 0 or img_h <= 0:
        return base
    return (f"{base} ({raw_w / img_w * 100:.1f}% × "
            f"{raw_h / img_h * 100:.1f}%)")


def create_thumbnail(pixmap, max_width, max_height):
    """
    创建缩略图，等比缩放到网格大小，边缘填充透明背景
    :param pixmap: 原始图片 QPixmap
    :param max_width: 网格宽度
    :param max_height: 网格高度
    :return: 缩略图 QPixmap（固定大小）
    """
    orig_width = pixmap.width()
    orig_height = pixmap.height()

    if orig_width <= 0 or orig_height <= 0:
        return pixmap

    # 计算等比缩放比例，使图片完整显示在网格内
    scale_x = max_width / orig_width
    scale_y = max_height / orig_height
    scale = min(scale_x, scale_y)  # 选择较小的比例，确保不超过任一边

    # 计算缩放后的尺寸
    scaled_width = int(orig_width * scale)
    scaled_height = int(orig_height * scale)

    # 等比缩放图片
    scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # 创建目标大小的空白图片（网格大小）
    target_pixmap = QPixmap(max_width, max_height)
    target_pixmap.fill(Qt.transparent)  # 填充透明背景

    # 计算居中位置
    x = (max_width - scaled_width) // 2
    y = (max_height - scaled_height) // 2

    # 将缩放后的图片绘制到目标图片的中心
    painter = QPainter(target_pixmap)
    painter.drawPixmap(x, y, scaled_pixmap)
    painter.end()

    return target_pixmap


def create_app_icon(script_dir):
    """
    从本地 icoo.png 文件加载应用图标
    支持 PyInstaller 打包后的环境
    :param script_dir: 脚本所在目录
    :return: QIcon
    """
    # 尝试不同的图标路径
    icon_paths = [
        # 开发环境路径（从 ui/ 目录向上两级到项目根目录）
        os.path.join(script_dir, "../../ico_image", "icoo.png"),
        # 开发环境路径（从 ui/ 目录向上一级到 labelforge 目录）
        os.path.join(script_dir, "../ico_image", "icoo.png"),
        # 打包环境路径
        os.path.join(script_dir, "ico_image", "icoo.png"),
        # 当前目录
        os.path.join(os.getcwd(), "ico_image", "icoo.png"),
        # 绝对路径（备用）
        os.path.abspath(os.path.join(script_dir, "../../ico_image", "icoo.png"))
    ]
    
    # 尝试加载图标
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            return QIcon(icon_path)
    
    # 如果所有路径都失败，使用默认图标
    icon_size = 64
    pixmap = QPixmap(icon_size, icon_size)
    pixmap.fill(QColor(0, 128, 255))  # 使用蓝色作为默认图标
    return QIcon(pixmap)


def extract_label_name(label_text):
    """
    从标签文本中提取纯标签名称，去除"(count)"部分
    :param label_text: 标签文本（可能包含计数）
    :return: 纯标签名称
    """
    if "⚠" in label_text:
        label_text = label_text.split("⚠")[0].rstrip()
    if " (" in label_text:
        label_text = label_text.split(" (")[0]
    if label_text.endswith('°'):
        head, _, tail = label_text.rpartition(' ')
        if head and tail[:-1].strip().replace('.', '', 1).isdigit():
            label_text = head
    if label_text.endswith(']') and ' [' in label_text:
        label_text = label_text.rsplit(' [', 1)[0]
    return label_text


# 标签列表“任务徽标”在 item 上存储任务类型的 role（Qt.UserRole 段，避开 0x0100 的 box 索引）
TASK_DATA_ROLE = 0x0101


def shape_task_type(box):
    """检测框对应的标注任务：det / seg / pose / obb。"""
    if not isinstance(box, dict):
        return "det"
    return {
        "polygon": "seg",
        "point": "pose",
        "rotation": "obb",
    }.get(box.get("shape_type") or "rectangle", "det")


def box_matches_filter(box, task_filter=None, group_filter=None):
    """框是否通过任务/分组筛选。filter 为空集合表示不筛选（全部）。"""
    if task_filter:
        if shape_task_type(box) not in task_filter:
            return False
    if group_filter:
        if box.get("group_id") not in group_filter:
            return False
    return True


def box_visible(editor, box):
    """按编辑器的任务/分组筛选判断框是否可见（属性缺失视为不筛选）。"""
    return box_matches_filter(
        box,
        getattr(editor, '_task_filter', None) or None,
        getattr(editor, '_group_filter', None) or None,
    )


def calculate_iou(box1, box2):
    """
    计算两个矩形的 IoU（交并比）
    :param box1: 矩形 1 (x1, y1, x2, y2)
    :param box2: 矩形 2 (x1, y1, x2, y2)
    :return: IoU 值
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union
