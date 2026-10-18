# -*- coding: utf-8 -*-
"""检测框命中判定：三处共用同一份判据。

原先画布交互（悬停、点选）和右键菜单各写一份，坐标换算与三种形状的分支
逐字重复。分叉的风险是「能点中但右键找不到这个框」——代码上看不出来。

conftest 在导入期把 PySide6 换成桩，QRectF/QPoint 都不是真的，所以这里按
项目既有做法跑真 Qt 子进程。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_QT_EXIT_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""

_SCRIPT = r"""
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.canvas.hit_test import (box_hit_by_cursor, canvas_point_of,
                                        canvas_rect_of)

RECT = QRectF(10, 20, 100, 50)      # 画布上的背景矩形
SCALE = 2.0
EPS = 8


def box(**kw):
    base = {"x": 5, "y": 5, "width": 10, "height": 10}
    base.update(kw)
    return base


def hit(b, x, y):
    return box_hit_by_cursor(b, QPointF(x, y), RECT, SCALE, EPS)


# 坐标换算
px, py = canvas_point_of((3, 4), RECT, SCALE)
print("POINT", px, py)                       # 期望 16.0 28.0

r = canvas_rect_of({"x": 2, "y": 3, "width": 4, "height": 5}, RECT, SCALE)
print("RECT", r.x(), r.y(), r.width(), r.height())   # 期望 14 26 8 10

# 矩形：画布上 (10,20)-(30,40)
print("RECT_HIT_IN", hit(box(x=0, y=0, width=10, height=10), 20, 30))
print("RECT_HIT_OUT", hit(box(x=0, y=0, width=10, height=10), 100, 100))

# 关键点：画布上 (20, 30)，半径 EPS
p = box(shape_type="point", points=[[5, 5]])
print("PT_CENTER", hit(p, 20, 30))
print("PT_EDGE", hit(p, 20 + EPS - 1, 30))
print("PT_FAR", hit(p, 20 + EPS + 5, 30))

# 多边形：三角形，外接矩形角落不该命中
tri = box(shape_type="polygon", points=[[0, 0], [10, 0], [5, 10]],
          x=0, y=0, width=10, height=10)
ix, iy = canvas_point_of((5, 3), RECT, SCALE)
cx, cy = canvas_point_of((0.5, 9.5), RECT, SCALE)
print("POLY_IN", hit(tri, ix, iy))
print("POLY_CORNER", hit(tri, cx, cy))

# 退化多边形（点不足）：不命中，也不抛
print("POLY_DEGEN", hit(box(shape_type="polygon", points=[[0, 0], [1, 1]]), 20, 30))

# 命中顺序：关键点必须排在检测框之前
from labelforge.canvas.hit_test import iter_boxes_by_hit_priority

nested = [box(x=0, y=0, width=10, height=10),          # 下标 0：矩形，盖住下面
          box(shape_type="point", points=[[5, 5]])]     # 下标 1：关键点，在矩形里
print("ORDER_DEFAULT", ",".join(
    str(i) for i, _b in iter_boxes_by_hit_priority(nested)))

# 「右键找框」必须选到关键点而不是矩形
found = None
for i, b in iter_boxes_by_hit_priority(nested):
    if box_hit_by_cursor(b, QPointF(20, 30), RECT, SCALE, EPS):
        found = i
        break
print("FIND_NESTED", found)

two_points = [box(shape_type="point", points=[[5, 5]]),
              box(shape_type="point", points=[[5, 5]])]
print("ORDER_TWO_POINTS", ",".join(
    str(i) for i, _b in iter_boxes_by_hit_priority(two_points)))
print("ORDER_SELECTED0", ",".join(
    str(i) for i, _b in iter_boxes_by_hit_priority(two_points, selected={0})))
"""


def _run():
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", _SCRIPT + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=180, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    out = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split()
        if parts:
            out[parts[0]] = parts[1:]
    return out


def test_canvas_point_maps_image_to_canvas():
    out = _run()
    assert [float(v) for v in out["POINT"]] == [16.0, 28.0]


def test_canvas_rect_maps_image_box_to_canvas():
    out = _run()
    assert [float(v) for v in out["RECT"]] == [14.0, 26.0, 8.0, 10.0]


def test_rectangle_hit_uses_the_box():
    out = _run()
    assert out["RECT_HIT_IN"] == ["True"]
    assert out["RECT_HIT_OUT"] == ["False"]


def test_point_hit_uses_a_radius_not_an_area():
    """关键点没有面积，得靠半径容差才点得中。"""
    out = _run()
    assert out["PT_CENTER"] == ["True"]
    assert out["PT_EDGE"] == ["True"], "半径内应当命中"
    assert out["PT_FAR"] == ["False"], "半径外不该命中"


def test_polygon_hit_follows_the_outline_not_the_bbox():
    """多边形按点在多边形内判定：外接矩形的角落不该命中。"""
    out = _run()
    assert out["POLY_IN"] == ["True"]
    assert out["POLY_CORNER"] == ["False"], "落在 bbox 内但在多边形外，不该算命中"


def test_degenerate_polygon_is_not_a_hit():
    out = _run()
    assert out["POLY_DEGEN"] == ["False"]


def test_keypoint_is_hit_before_the_box_that_covers_it():
    """关键点落在检测框里时，「找鼠标下的框」要选关键点。

    用户报的：「关键点在检测框内能点击拖动，但右击修改无法选中，选的还是
    检测框」。右键菜单原来正序遍历、返回第一个命中，于是先撞上包住关键点的
    那个框。绘制顺序里关键点在最上层，命中顺序必须与之一致。
    """
    out = _run()
    assert out["FIND_NESTED"] == ["1"], (
        f"右键应当选到关键点（下标 1），实际 {out['FIND_NESTED']}")


def test_hit_priority_puts_keypoints_first_then_later_boxes():
    out = _run()
    assert out["ORDER_DEFAULT"] == ["1,0"], out["ORDER_DEFAULT"]
    # 两个关键点：后画的（下标大的）在上层
    assert out["ORDER_TWO_POINTS"] == ["1,0"], out["ORDER_TWO_POINTS"]


def test_hit_priority_lifts_a_selected_keypoint_above_the_others():
    """已选中的关键点画在最上层，命中也要最先轮到它。"""
    out = _run()
    assert out["ORDER_SELECTED0"] == ["0,1"], out["ORDER_SELECTED0"]