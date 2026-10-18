"""检测框命中判定：把鼠标位置（画布坐标）落到具体的框上。

判据原先写了三份 —— 画布交互两处（悬停、点选）与右键菜单各一份，坐标换算
和三种形状的分支逐字重复。三份一旦分叉，就会出现「能点中但右键找不到这个框」
这类不一致，而这类问题从代码上看不出来。

坐标换算走 `canvas_point_of` / `canvas_rect_of`，命中判定走 `box_hit_by_cursor`。
"""

from PySide6.QtCore import QRectF

from ..engine.shape_io import point_in_polygon


def canvas_point_of(point, background_rect, scale):
    """图像坐标 -> 画布坐标，返回 (x, y)。"""
    return (point[0] * scale + background_rect.left(),
            point[1] * scale + background_rect.top())


def canvas_rect_of_bounds(left, top, width, height, background_rect, scale):
    """把「图像坐标下的一个矩形」换算到画布坐标。"""
    return QRectF(left * scale + background_rect.left(),
                  top * scale + background_rect.top(),
                  width * scale, height * scale)


def canvas_rect_of(box, background_rect, scale):
    """框的外接矩形，换算到画布坐标。"""
    return canvas_rect_of_bounds(box["x"], box["y"],
                                 box["width"], box["height"],
                                 background_rect, scale)


def box_hit_by_cursor(box, cursor, background_rect, scale, point_radius):
    """鼠标是否落在这个框上。

    三种形状各有口径：关键点按半径 `point_radius` 的圆（点本身没有面积，
    得给容差才点得中）；多边形与旋转框按「点在多边形内」；其余按外接矩形。
    """
    shape_type = box.get("shape_type")
    points = box.get("points")
    if shape_type == "point" and points:
        cx, cy = canvas_point_of(points[0], background_rect, scale)
        return ((cursor.x() - cx) ** 2 + (cursor.y() - cy) ** 2
                <= point_radius ** 2)
    if shape_type in ("polygon", "rotation") and points:
        return point_in_polygon(
            cursor.x(), cursor.y(),
            [canvas_point_of(p, background_rect, scale) for p in points])
    return canvas_rect_of(box, background_rect, scale).contains(cursor)