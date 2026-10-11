"""LabelMe shape <-> internal box conversion."""
import math


def box_from_labelme_shape(shape):
    if not isinstance(shape, dict):
        return None
    points = shape.get("points") or []
    shape_type = shape.get("shape_type") or "rectangle"
    flags = shape.get("flags") if isinstance(shape.get("flags"), dict) else {}
    is_paste = bool(flags.get("paste"))
    if shape_type == "point":
        if len(points) != 1:
            return None
        x, y = points[0][0], points[0][1]
        point_box = {
            "x": x, "y": y, "width": 0, "height": 0,
            "label": shape.get("label") or "",
            "shape_type": shape_type,
            "group_id": shape.get("group_id"),
            "points": [list(points[0])],
            "is_paste": is_paste,
        }
        if flags:
            point_box["flags"] = dict(flags)
        return point_box
    if len(points) < 2:
        return None
    label = shape.get("label") or ""
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    x = min(x_coords)
    y = min(y_coords)
    box = {
        "x": x,
        "y": y,
        "width": max(x_coords) - x,
        "height": max(y_coords) - y,
        "label": label,
        "shape_type": shape_type,
        "group_id": shape.get("group_id"),
        "is_paste": is_paste,
    }
    if shape_type != "rectangle":
        box["points"] = [list(p) for p in points]
    if "visible" in shape:
        box["visible"] = shape["visible"]
    # 置信度按 X-AnyLabeling 的约定存顶层 "score"。手工框没有这个键，
    # 只有推理结果带；缺失即 None，界面据此决定要不要显示数字。
    if isinstance(shape.get("score"), (int, float)) and not isinstance(shape["score"], bool):
        box["score"] = float(shape["score"])
    # 自定义 flag（如质检豁免 tiny_keep）要跟着 box 走，否则载入即丢、
    # labelme_shape_from_box 再存回去时也就无从恢复。空 flags 不塞键，
    # 让绝大多数无 flag 的 JSON 载入结果与历史行为完全一致。
    if flags:
        box["flags"] = dict(flags)
    return box


def labelme_shape_from_box(box):
    shape_type = box.get("shape_type") or "rectangle"
    if shape_type == "rectangle":
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    else:
        points = [list(p) for p in (box.get("points") or [])]
    raw_flags = box.get("flags")
    flags = dict(raw_flags) if isinstance(raw_flags, dict) else {}
    if box.get("is_paste"):
        flags["paste"] = True
    shape = {
        "label": box.get("label", ""),
        "points": points,
        "group_id": box.get("group_id"),
        "description": "",
        "shape_type": shape_type,
        "flags": flags,
    }
    # 置信度写顶层 score，跟 X-AnyLabeling 保持一致；没有就写 null 占位，
    # 让 JSON 里这个字段始终存在，外部工具读取时不用判键是否存在。
    score = box.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    shape["score"] = None if score is None else float(score)
    # 键序按 labelme 习惯：score 跟在 label 后面更好读
    ordered = {"label": shape.pop("label"), "score": shape.pop("score")}
    ordered.update(shape)
    shape = ordered
    if "visible" in box:
        shape["visible"] = box["visible"]
    return shape


def format_label_display(label, group_id=None):
    if group_id is not None:
        return f"{label} [{group_id}]"
    return label


def format_score_display(score):
    """置信度显示文本；没有分数返回空串。"""
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return ""
    return f"{float(score):.2f}"


def rotation_angle_degrees(points):
    """OBB 旋转角（度），基于 p0->p1 边相对水平线，归一化到 [0, 360)。"""
    if not points or len(points) < 2:
        return 0.0
    dx = points[1][0] - points[0][0]
    dy = points[1][1] - points[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return angle


def box_display_label(box, show_score=False):
    """标签显示文本：类别[分组] 角度，选带置信度。

    置信度跟着类别名一起显示（参考 X-AnyLabeling 的 show_scores），
    由调用方按「显示置信度」开关传入 show_score。
    """
    label = box.get("label", "") if isinstance(box, dict) else ""
    text = format_label_display(label, box.get("group_id") if isinstance(box, dict) else None)
    if isinstance(box, dict) and box.get("shape_type") == "rotation" and len(box.get("points") or []) == 4:
        angle = rotation_angle_degrees(box['points'])
        if angle >= 359.5:
            angle = 0.0
        text = f"{text} {angle:.0f}°"
    if show_score and isinstance(box, dict):
        shown = format_score_display(box.get("score"))
        if shown:
            text = f"{text} {shown}"
    return text


def point_warning(point_box, all_boxes):
    """关键点校验：优先无同组框 → '无同组框'，否则不在框内 → '不在框内'，正常为 None。"""
    gid = point_box.get("group_id")
    if gid is None:
        return "无同组框"
    candidates = [
        box for box in (all_boxes or [])
        if (box.get("shape_type") or "rectangle") != "point"
        and box.get("group_id") == gid
    ]
    if not candidates:
        return "无同组框"
    points = point_box.get("points") or []
    try:
        if points:
            px, py = float(points[0][0]), float(points[0][1])
        else:
            px, py = float(point_box.get("x", 0)), float(point_box.get("y", 0))
    except (TypeError, ValueError, IndexError, KeyError):
        # 坏数据（非数值坐标）：不能中断 lint/列表渲染，按"不在框内"处理
        return "不在框内"
    for box in candidates:
        if point_in_box(px, py, box):
            return None
    return "不在框内"


def point_in_box(x, y, box):
    """(x, y) 是否落在框内。矩形用 bbox，多边形/旋转框有 points 时用多边形判定。

    坏数据（非数值坐标）返回 False，不能中断整库扫描/列表渲染。
    """
    if not isinstance(box, dict):
        return False
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return False
    points = box.get("points")
    if points and (box.get("shape_type") or "rectangle") != "rectangle":
        try:
            return point_in_polygon(x, y, points)
        except (TypeError, ValueError, IndexError):
            return False
    try:
        bx, by = float(box.get("x", 0)), float(box.get("y", 0))
        bw, bh = float(box.get("width", 0)), float(box.get("height", 0))
    except (TypeError, ValueError):
        return False
    return bx <= x <= bx + bw and by <= y <= by + bh


def point_in_polygon(x, y, points):
    """Ray-casting; True if (x, y) is inside the polygon."""
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def point_line_distance(point, line):
    """Distance from point (x, y) to segment line[(x1,y1),(x2,y2)]."""
    (x1, y1), (x2, y2) = line
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = x1 + t * dx, y1 + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def nearest_polygon_edge(point, points, epsilon):
    """Insertion index of the nearest edge within epsilon, else None.

    Index i means: insert between points[i-1] and points[i].
    """
    best_i = None
    min_dist = float("inf")
    for i in range(len(points)):
        line = (points[i - 1], points[i])
        dist = point_line_distance(point, line)
        if dist <= epsilon and dist < min_dist:
            min_dist = dist
            best_i = i
    return best_i


def yolo_pose_line(class_id, bbox, keypoints, image_width, image_height):
    """`cls cx cy w h x1 y1 v1 ...`; visibility fixed 2 (visible)."""
    if not image_width or not image_height:
        return None
    x, y, w, h = bbox
    parts = [str(class_id),
             f"{(x + w / 2) / image_width:.6f}",
             f"{(y + h / 2) / image_height:.6f}",
             f"{w / image_width:.6f}",
             f"{h / image_height:.6f}"]
    for kx, ky in keypoints:
        parts.append(f"{kx / image_width:.6f}")
        parts.append(f"{ky / image_height:.6f}")
        parts.append("2")
    return " ".join(parts)


def yolo_seg_line(box, class_id, image_width, image_height):
    points = box.get("points") or []
    if len(points) < 3 or not image_width or not image_height:
        return None
    parts = [str(class_id)]
    for x, y in points:
        parts.append(f"{x / image_width:.6f}")
        parts.append(f"{y / image_height:.6f}")
    return " ".join(parts)


def rect_to_rotation_points(x, y, width, height):
    """Axis-aligned rectangle -> 4 CCW corner points (LabelMe rotation order)."""
    return [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height],
    ]


def rotation_center(points):
    """Center of a 4-point rotation shape (diagonal midpoint)."""
    if not points or len(points) < 4:
        return None
    cx = (points[0][0] + points[2][0]) / 2
    cy = (points[0][1] + points[2][1]) / 2
    return cx, cy


def rotate_points(points, center, angle):
    """Rotate points around center by angle (radians), returning new point list."""
    cx, cy = center
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    out = []
    for px, py in points:
        dx = px - cx
        dy = py - cy
        out.append([cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a])
    return out


def rotation_points_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def move_rotation_vertex(points, index, new_pos):
    """Drag one OBB corner while keeping a rectangle with the same orientation.

    The opposite corner stays fixed; the two adjacent corners slide along the
    original edge directions so all angles remain 90°.
    """
    if len(points) != 4:
        return points
    i = index % 4
    opp = (i + 2) % 4
    nx, ny = float(new_pos[0]), float(new_pos[1])
    ux = points[(i + 1) % 4][0] - points[i][0]
    uy = points[(i + 1) % 4][1] - points[i][1]
    vx = points[(i + 3) % 4][0] - points[i][0]
    vy = points[(i + 3) % 4][1] - points[i][1]
    u_len = math.hypot(ux, uy) or 1.0
    v_len = math.hypot(vx, vy) or 1.0
    ux, uy = ux / u_len, uy / u_len
    vx, vy = vx / v_len, vy / v_len
    dx = points[opp][0] - nx
    dy = points[opp][1] - ny
    a = dx * ux + dy * uy
    b = dx * vx + dy * vy
    out = [list(p) for p in points]
    out[i] = [nx, ny]
    out[(i + 1) % 4] = [nx + a * ux, ny + a * uy]
    out[(i + 3) % 4] = [nx + b * vx, ny + b * vy]
    return out


def rotation_handle_point(points, distance):
    """Point offset from the midpoint of edge p0-p1 along its outward normal."""
    if len(points) != 4:
        return None
    p0, p1 = points[0], points[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    edge_len = math.hypot(dx, dy)
    if edge_len < 1e-6:
        return None
    mid_x = (p0[0] + p1[0]) / 2
    mid_y = (p0[1] + p1[1]) / 2
    normal_x = dy / edge_len
    normal_y = -dx / edge_len
    return mid_x + normal_x * distance, mid_y + normal_y * distance


def yolo_obb_line(box, class_id, image_width, image_height):
    """`cls x0 y0 x1 y1 x2 y2 x3 y3` normalized; None if not a valid 4-point OBB."""
    points = box.get("points") or []
    if len(points) != 4 or not image_width or not image_height:
        return None
    parts = [str(class_id)]
    for x, y in points:
        parts.append(f"{x / image_width:.6f}")
        parts.append(f"{y / image_height:.6f}")
    return " ".join(parts)
