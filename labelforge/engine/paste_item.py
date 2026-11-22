"""贴图项：画布上一个已粘贴的素材。"""

import math


class PasteItem:
    """画布贴图项。

    早期版本的贴图是三元组 `(pixmap, rect, label)`，散落在 20 处解包点。
    形状由角度派生：0° 记 rectangle（det 任务），非 0° 记 rotation（obb 任务）。
    贴图不提供 polygon 形态，需要分割标注的由用户删掉后手画。
    """

    __slots__ = ("pixmap", "rect", "label", "angle", "path")

    def __init__(self, pixmap, rect, label, angle=0.0, path=None):
        self.pixmap = pixmap
        self.rect = rect
        self.label = label
        self.angle = float(angle)
        # 贴图项记着磁盘上的源文件：抠图产物指向落盘的 PNG，从贴图列表
        # 拖进来的项指向列表里那张图。裁剪时按它覆盖写回。
        self.path = path

    @property
    def shape_type(self):
        return "rectangle" if abs(self.angle) < 1e-9 else "rotation"

    def rect_tuple(self):
        """外接框 -> (x, y, w, h)。rect 允许是 QRectF 或四元组。"""
        rect = self.rect
        if hasattr(rect, "x"):
            return (rect.x(), rect.y(), rect.width(), rect.height())
        x, y, w, h = rect
        return (float(x), float(y), float(w), float(h))

    def corners(self):
        """矩形四角（含角度），顺序为左上、右上、右下、左下。"""
        x, y, w, h = self.rect_tuple()
        base = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if abs(self.angle) < 1e-9:
            return base
        cx, cy = x + w / 2.0, y + h / 2.0
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        out = []
        for px, py in base:
            dx, dy = px - cx, py - cy
            # 顺时针旋转：y 轴向下，故用此式
            out.append((cx + dx * cos_a - dy * sin_a,
                        cy + dx * sin_a + dy * cos_a))
        return out

    def __iter__(self):
        """兼容旧写法：仍按旧三元组 `(pixmap, rect, label)` 解包。

        只覆盖外部（第三方脚本、旧测试替身）残留的三元组解包方式，
        仓库内代码一律用属性访问；新代码不得依赖本方法。
        """
        yield self.pixmap
        yield self.rect
        yield self.label

    def __repr__(self):
        x, y, w, h = self.rect_tuple()
        return (f"PasteItem({self.label!r}, x={x:.1f}, y={y:.1f}, "
                f"w={w:.1f}, h={h:.1f}, angle={self.angle:.1f})")


def as_paste_item(value):
    """把三元组或 PasteItem 归一化为 PasteItem。

    撤销栈、`canvas_items_dict`、记忆记录里可能存着旧三元组，
    读取方一律经此函数，避免出现两套解包逻辑。
    """
    if isinstance(value, PasteItem):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        pixmap, rect, label = value
        return PasteItem(pixmap=pixmap, rect=rect, label=label)
    raise TypeError(f"不是合法的贴图项：{type(value).__name__}")
