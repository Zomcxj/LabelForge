"""
Canvas 交互混入 - 鼠标/键盘事件入口、拖拽、缩放
"""
import os
from PySide6.QtCore import Qt, QRectF, QUrl, QMimeData
from PySide6.QtGui import QDrag

from ..core.config import BACKGROUND_SCALE_CONFIG, PASTE_ITEM_CONFIG, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, DETECTION_BOX_CONFIG
from .canvas_drawing import CanvasDrawingMixin
from .canvas_menu import CanvasMenuMixin

# 画布上「按住拖动 = 把当前图拖出去」的触发阈值（像素）。
# 与背景图列表 `DragOutListWidget` 的 20px 一致：两边是同一个功能，
# 用户不该感觉画布上更难拖出来。留点余量是为了避开手抖与单击。
_DRAG_OUT_THRESHOLD = 20


class CanvasInteractionMixin(CanvasDrawingMixin, CanvasMenuMixin):
    """Canvas 交互混入类 - 事件入口 + 通用操作"""

    # ---------- 区域绘制 ----------

    def _notify_region_changed(self):
        """区域增删/落笔后通知编辑器（状态栏、面板区域列表要跟着刷）。"""
        editor = getattr(self, '_editor', None)
        handler = getattr(editor, '_on_region_changed', None)
        if callable(handler):
            handler()

    def _region_draw_armed(self):
        """当前的「区域绘制态」还算数吗。

        判据是**有没有被入口 arm 过**：右键菜单的「添加区域（矩形/多边形）」、
        或范围面板上的「添加区域」按钮，都会经 `_start_add_region*()` 打个
        `_region_draw_armed_flag`。

        这样判而不是「模式对不对」，因为绘制态的残留和模式切换**无关**：从贴图
        模式切到标注模式、或面板关掉时忘了复位，`is_drawing_region` 都会留在
        True。那时若照旧放行，之后每一个左键都被当成「继续画区域」，用户会看到
        「切到标注模式后画不了标注框，反而冒出一个区域」。

        在 `_reset_region_drawing_state()` / `_cancel_region_drawing()` 里清标记，
        所以画完一个就自动失效（「点一下画一个」）。
        """
        return bool(getattr(self, '_region_draw_armed_flag', False)
                    or self._is_paste_mode())

    def _box_visible(self, box):
        """框是否通过任务/分组筛选（未启用筛选则全部可见）。"""
        from ..core.utils import box_visible
        return box_visible(self._editor, box)


    def enterEvent(self, event):
        self.mouse_inside = True
        self.update_status_label()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.mouse_inside = False
        self.update_status_label()
        self._drag_out_pending = False
        self.update()
        super().leaveEvent(event)


    def mousePressEvent(self, event):
        self.setFocus()
        mouse_pos = event.pos()

        if not self._can_edit_canvas():
            self._drag_out_pending = False
            if event.button() == Qt.RightButton:
                self._handle_right_click(mouse_pos)
                return
            if event.button() == Qt.LeftButton and self._editor.current_background:
                self._handle_background_click(mouse_pos)
            return

        if getattr(self, 'is_drawing_region', False):
            # 区域绘制模式：左键画矩形，右键取消
            self._drag_out_pending = False
            if not self._region_draw_armed():
                self._reset_region_drawing_state()
                return
            if event.button() == Qt.LeftButton:
                self._handle_region_press(mouse_pos)
            elif event.button() == Qt.RightButton:
                self._cancel_region_drawing()
            return

        if getattr(self, 'is_drawing_region_polygon', False):
            # 多边形区域绘制：左键逐点添加，右键完成
            self._drag_out_pending = False
            if not self._region_draw_armed():
                self._reset_region_drawing_state()
                return
            if event.button() == Qt.LeftButton:
                self._handle_region_polygon_press(mouse_pos)
            elif event.button() == Qt.RightButton:
                self._prompt_finish_region_polygon()
            return

        if self.is_drawing_box:
            # 绘制模式下只响应左键画框，禁止贴图/检测框进入编辑态
            if event.button() == Qt.LeftButton:
                self._drag_out_pending = False
                self._handle_drawing_press(mouse_pos)
            return

        if getattr(self, 'is_drawing_polygon', False):
            self._drag_out_pending = False
            if event.button() == Qt.LeftButton:
                self._handle_polygon_press(mouse_pos)
            elif event.button() == Qt.RightButton:
                self._prompt_finish_polygon()
            return

        if getattr(self, 'is_drawing_point', False):
            self._drag_out_pending = False
            if event.button() == Qt.LeftButton:
                self._handle_point_press(mouse_pos)
            return

        if event.button() == Qt.RightButton:
            self._drag_out_pending = False
            if self._handle_right_click(mouse_pos):
                return
        elif event.button() != Qt.LeftButton:
            self._drag_out_pending = False
            return

        # 「准备拖出复制」只在该次按下**没抓到任何东西**时才挂起：
        # 抓到标注框/贴图/区域说明用户是在编辑，不是要往外拖文件。
        # 判据放在点击**之后**：`_handle_background_click` 等要等点击处理完
        # 才知道有没有真的抓起来（按在图上就会进入画布拖动）。
        #
        # **图有没有被缩放**决定这次拖动是「拖出文件」还是「平移画布」：
        # - 图没缩放（自适应）：按在图上拖动 = 把这张图拖出去（用户要的「直接
        #   左键拖出图片」）。这也是原来「开启复制却拖不出去」的修复点——按在
        #   图上会先被判成平移，拖出永远挂不起来。
        # - 图已手动缩放：拖动 = 平移画布，**不**拖出（用户明确「如果缩放了
        #   只能平移」）。放大后要挪动看细节，这时候弹文件是干扰。
        # - 复制关掉：一律平移，与以前逐字一致。
        copy_on = bool(getattr(self._editor, '_canvas_image_copy_enabled', False))
        press_on_image = self._press_is_on_background(mouse_pos)
        zoomed = bool(getattr(self, 'is_manual_scale', False))

        self._drag_out_pending = False
        self._handle_left_click(mouse_pos)
        grabbed = (self.is_dragging_box
                   or self.is_resizing_box or self.is_rotating_box
                   or self.is_dragging_item or self.is_dragging_region
                   or self.is_resizing_region or self.is_dragging_region_vertex)
        if grabbed:
            self._drag_out_pending = False
        elif copy_on and press_on_image and not zoomed:
            # 没缩放 = 用户的意图是「把这张图拖出去」，平移让位。
            self.is_dragging_background = False
            self._drag_out_pending = True
        else:
            self._drag_out_pending = (
                not self.is_dragging_background
                and copy_on
                and not zoomed
                and self.find_item_at_position(mouse_pos) is None
                # 这里问的是「有没有东西挡着」，与能否编辑无关，所以**要**算上
                # 已锁定的区域：锁定的区域也是区域，压在它上面往外拖不该弹文件。
                and self._region_at(mouse_pos, include_locked=True) is None
            )
        # 记下起点：`mouseMoveEvent` 靠与它的距离判断该不该拖出。
        self._drag_out_start = mouse_pos if self._drag_out_pending else None

    def _press_is_on_background(self, mouse_pos):
        """这次按下是否落在**图片本身**上（而不是图外的空白）。

        拖出复制只对「拖这张图」有意义；点在图外空白处（画布的留白）仍按老
        规矩处理，免得在空白处拖动也弹出一个文件。
        """
        if self._editor.current_background is None:
            return False
        rect = self.get_background_rect()
        if rect is None:
            return False
        return rect.contains(mouse_pos)

    def _handle_left_click(self, mouse_pos):
        if (getattr(self, 'is_drawing_box', False)
                or getattr(self, 'is_drawing_polygon', False)
                or getattr(self, 'is_drawing_region', False)
                or getattr(self, 'is_drawing_region_polygon', False)):
            return
        is_annotate = getattr(self._editor, 'edit_mode', 'paste') == 'annotate'

        if is_annotate:
            # **未锁定的区域优先于标签框**（用户要求：「不锁定默认最上层与
            # 检测框一样可修改，锁定后最底层不可修改」）——未锁区域画在框
            # 之上，重叠处点击就该归它；想让下面的框拿到点击，先把区域锁上，
            # 锁定的区域在 `_region_at()` 里已被跳过。
            if self._region_editing_allowed():
                if self._handle_region_click(mouse_pos):
                    return
            if self._editor.show_labels_checkbox.isChecked() and self._editor.current_background:
                if self._handle_detection_box_click(mouse_pos):
                    return
            item_at_pos = self.find_item_at_position(mouse_pos)
            if item_at_pos is not None:
                self._handle_item_click(item_at_pos, mouse_pos)
                return
        else:
            item_at_pos = self.find_item_at_position(mouse_pos)
            if item_at_pos is not None:
                self._handle_item_click(item_at_pos, mouse_pos)
                return
            if self._editor.show_labels_checkbox.isChecked() and self._editor.current_background:
                if self._handle_detection_box_click(mouse_pos):
                    return
            if self._handle_region_click(mouse_pos):
                return

        if self._editor.current_background:
            if self._handle_background_click(mouse_pos):
                return

        self._clear_selection()

    def _handle_region_click(self, mouse_pos):
        """区域编辑入口：顶点/手柄→缩放，边上→插点，内部→拖动。"""
        # 平时只在贴图模式编辑区域；但**范围面板开着**时例外：用户要求
        # 「区域框可以调整大小和位置，只在范围面板开着时允许」，而面板开着时
        # 画布通常停在标注模式（正在审标注）。所以面板开着就允许调区域。
        # 判据与滚轮缩放共用 `_region_editable()`，保证两处不会各说各话。
        if not self._region_editable():
            return False

        # 多边形顶点优先
        vertex = self._region_vertex_at(mouse_pos)
        if vertex is not None:
            self.selected_region = vertex[0]
            self.selected_region_vertex = vertex[1]
            self.region_vertex_drag_index = vertex[1]
            self.is_dragging_region_vertex = True
            self.region_drag_start = mouse_pos
            self._editor.selected_item = None
            self.selected_item_size = None
            self.selected_box = None
            self.selected_boxes = []
            self.update()
            return True

        # 多边形边上点击 → 插入顶点
        edge = self._region_polygon_edge_at(mouse_pos)
        if edge is not None:
            r_idx, v_idx, img_pt = edge
            if self._insert_region_vertex(r_idx, v_idx, img_pt):
                self.selected_region = r_idx
                self.update()
                return True

        index = self._region_at(mouse_pos)
        if index is None:
            self.selected_region = None
            self.selected_region_vertex = None
            return False
        self.selected_region = index
        self.selected_region_vertex = None
        self._editor.selected_item = None
        self.selected_item_size = None
        self.selected_box = None
        self.selected_boxes = []
        handle = self._region_handle_at(mouse_pos, index)
        if handle:
            self.region_resize_handle = handle
            self.is_resizing_region = True
            self.region_resize_start = mouse_pos
        else:
            self.is_dragging_region = True
            self.region_drag_start = mouse_pos
        self.update()
        return True

    def _region_polygon_edge_at(self, mouse_pos):
        """返回鼠标附近的区域多边形边 (region_index, vertex_index, image_point)。"""
        from ..engine.shape_io import point_line_distance
        regions = self._active_regions()
        if not regions:
            return None
        background_rect = self.get_background_rect()
        if background_rect is None:
            return None
        from ..core.config import REGION_CONFIG
        threshold = REGION_CONFIG['handle_size'] * 1.5
        for i in range(len(regions) - 1, -1, -1):
            region = regions[i]
            if (region.get('shape_type') or 'rectangle') != 'polygon':
                continue
            points = region.get('points') or []
            if len(points) < 3:
                continue
            n = len(points)
            for v in range(n):
                p1 = points[v]
                p2 = points[(v + 1) % n]
                c1 = (p1[0] * self.background_scale + background_rect.left(),
                      p1[1] * self.background_scale + background_rect.top())
                c2 = (p2[0] * self.background_scale + background_rect.left(),
                      p2[1] * self.background_scale + background_rect.top())
                if point_line_distance((mouse_pos.x(), mouse_pos.y()), (c1, c2)) <= threshold:
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    return i, v + 1, [mid_x, mid_y]
        return None

    def _insert_region_vertex(self, region_index, vertex_index, point):
        if self._active_region_fixed():
            return False
        regions = self._active_regions()
        if not regions or not (0 <= region_index < len(regions)):
            return False
        region = regions[region_index]
        if (region.get('shape_type') or 'rectangle') != 'polygon':
            return False
        points = region.get('points') or []
        if not (0 <= vertex_index <= len(points)):
            return False
        from ..core.config import REGION_CONFIG
        if len(points) >= REGION_CONFIG['polygon_max_points']:
            return False
        points.insert(vertex_index, [float(point[0]), float(point[1])])
        region['points'] = points
        self._sync_region_bbox(region)
        return True

    def _drag_region_vertex(self):
        if self._active_region_fixed():
            return
        regions = self._active_regions()
        if not regions or self.selected_region is None:
            return
        if not (0 <= self.selected_region < len(regions)):
            return
        region = regions[self.selected_region]
        points = region.get('points') or []
        v = self.region_vertex_drag_index
        if v is None or not (0 <= v < len(points)):
            return
        delta = self.mouse_pos - self.region_drag_start
        dx = delta.x() / self.background_scale
        dy = delta.y() / self.background_scale
        px = points[v][0] + dx
        py = points[v][1] + dy
        bg = self._editor.current_background
        if bg is not None:
            px = max(0, min(px, bg.width()))
            py = max(0, min(py, bg.height()))
        points[v] = [px, py]
        region['points'] = points
        self._sync_region_bbox(region)
        self.region_drag_start = self.mouse_pos
        self.update()

    def _drag_region(self):
        if self._active_region_fixed():
            return
        regions = self._active_regions()
        if not regions or self.selected_region is None:
            return
        if not (0 <= self.selected_region < len(regions)):
            return
        delta = self.mouse_pos - self.region_drag_start
        region = regions[self.selected_region]
        dx = delta.x() / self.background_scale
        dy = delta.y() / self.background_scale
        bg = self._editor.current_background

        if (region.get('shape_type') or 'rectangle') == 'polygon' and region.get('points'):
            # 整体平移：按允许的位移夹紧
            region_w = region['width']
            region_h = region['height']
            nx = region['x'] + dx
            ny = region['y'] + dy
            if bg is not None:
                nx = max(0, min(nx, bg.width() - region_w))
                ny = max(0, min(ny, bg.height() - region_h))
            applied_dx = nx - region['x']
            applied_dy = ny - region['y']
            region['points'] = [
                [p[0] + applied_dx, p[1] + applied_dy] for p in region['points']
            ]
            region['x'] = nx
            region['y'] = ny
        else:
            nx = region['x'] + dx
            ny = region['y'] + dy
            if bg is not None:
                nx = max(0, min(nx, bg.width() - region['width']))
                ny = max(0, min(ny, bg.height() - region['height']))
            region['x'] = nx
            region['y'] = ny
        self.region_drag_start = self.mouse_pos
        self.update()

    def _resize_region(self):
        from ..core.config import REGION_CONFIG
        if self._active_region_fixed():
            return
        regions = self._active_regions()
        if not regions or self.selected_region is None:
            return
        if not (0 <= self.selected_region < len(regions)):
            return
        delta = self.mouse_pos - self.region_resize_start
        dx = delta.x() / self.background_scale
        dy = delta.y() / self.background_scale
        region = regions[self.selected_region]

        # 多边形区域：bbox 缩放同步到所有顶点
        if (region.get('shape_type') or 'rectangle') == 'polygon' and region.get('points'):
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            nx, ny, nw, nh = self._compute_region_resize_box(
                x, y, w, h, dx, dy, REGION_CONFIG['min_size']
            )
            if w > 0 and h > 0 and (nw != w or nh != h or nx != x or ny != y):
                sx = nw / w
                sy = nh / h
                region['points'] = [
                    [nx + (p[0] - x) * sx, ny + (p[1] - y) * sy]
                    for p in region['points']
                ]
                region['x'], region['y'] = nx, ny
                region['width'], region['height'] = nw, nh
            self.region_resize_start = self.mouse_pos
            self.update()
            return

        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        nx, ny, nw, nh = self._compute_region_resize_box(
            x, y, w, h, dx, dy, REGION_CONFIG['min_size']
        )
        region['x'], region['y'], region['width'], region['height'] = nx, ny, nw, nh
        self.region_resize_start = self.mouse_pos
        self.update()

    def _compute_region_resize_box(self, x, y, w, h, dx, dy, min_size):
        """按当前拖拽手柄计算新的 bbox（夹紧背景与最小尺寸）。

        四角 `tl/tr/bl/br` 改两个方向，四边 `t/b/l/r` 只改一个方向——与
        检测框的 `_resize_box` 同构，差别只在最小尺寸取自
        `REGION_CONFIG['min_size']` 而不是写死的 10。
        """
        bg = self._editor.current_background
        max_w = max(min_size, bg.width() - x) if bg else None
        max_h = max(min_size, bg.height() - y) if bg else None

        nx, ny, nw, nh = x, y, w, h
        handle = self.region_resize_handle
        if handle in ("br", "r", "b"):
            # 右下方向：只往右下长，左上角不动
            if handle in ("br", "r"):
                nw = max(min_size, w + dx)
                if max_w is not None:
                    nw = min(nw, max_w)
            if handle in ("br", "b"):
                nh = max(min_size, h + dy)
                if max_h is not None:
                    nh = min(nh, max_h)
        elif handle in ("tl", "l", "t"):
            # 左上方向：改左/上边，右下角不动
            if handle in ("tl", "l"):
                nx = max(0, min(x + dx, x + w - min_size))
                nw = w + x - nx
            if handle in ("tl", "t"):
                ny = max(0, min(y + dy, y + h - min_size))
                nh = h + y - ny
        elif handle in ("tr", "bl"):
            if handle == "tr":
                nw = max(min_size, w + dx)
                if max_w is not None:
                    nw = min(nw, max_w)
                ny = max(0, min(y + dy, y + h - min_size))
                nh = h + y - ny
            else:  # bl
                nx = max(0, min(x + dx, x + w - min_size))
                nw = w + x - nx
                nh = max(min_size, h + dy)
                if max_h is not None:
                    nh = min(nh, max_h)
        return nx, ny, nw, nh

    def _handle_item_click(self, item_index, mouse_pos):
        if self.is_drawing_box:
            return
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.selected_box = None
        self.selected_boxes = []
        self.resize_handle = None
        _, rect, _ = self._editor.canvas_items[item_index]

        if self._editor.selected_item != item_index:
            self._editor.selected_item = item_index
            self.selected_item_size = (rect.width(), rect.height())
            self.update()

        self._check_resize_handle(mouse_pos, rect)

        if not self.resize_handle:
            self._editor.save_undo_state()
            self.drag_start = mouse_pos
            self.is_dragging_item = True
            self.setCursor(Qt.ClosedHandCursor)

    def _check_resize_handle(self, mouse_pos, rect):
        handle_name = self._item_handle_at_pos(mouse_pos, rect)
        if handle_name:
            self.resize_handle = handle_name
            self.resize_start = mouse_pos
            self.hover_resize_target = 'item'
            self.hover_resize_handle = handle_name
            return True

        return False

    def _item_handle_at_pos(self, mouse_pos, rect):
        """返回鼠标所在的贴图圆形缩放手柄名称，不修改编辑状态。"""
        if not self._editor.current_background:
            return None

        background_rect = self.get_background_rect()
        if background_rect is None:
            return None
        item_x = (rect.x() * self.background_scale) + background_rect.left()
        item_y = (rect.y() * self.background_scale) + background_rect.top()
        item_width = rect.width() * self.background_scale
        item_height = rect.height() * self.background_scale
        item_rect = QRectF(item_x, item_y, item_width, item_height)

        handle_size = PASTE_ITEM_CONFIG['handle_size']
        br_handle = item_rect.bottomRight()

        handle_rect = QRectF(
            br_handle.x() - handle_size / 2,
            br_handle.y() - handle_size / 2,
            handle_size,
            handle_size,
        )
        if handle_rect.contains(mouse_pos):
            return 'br'

        return None

    def _collect_nearest_handle(self, mouse_pos):
        """遍历所有框收集手柄命中，返回离鼠标最近的那个"""
        background_rect = self.get_background_rect()
        if not background_rect:
            return None, None
        handle_size = DETECTION_BOX_CONFIG['resize_handle_size']
        best_box = best_handle = None
        min_dist = float('inf')
        for i, box in enumerate(self._editor.detection_boxes):
            if not self._box_visible(box):
                continue
            if box.get("shape_type") == "point":
                continue
            if box.get("shape_type") in ("polygon", "rotation") and box.get("points"):
                corners = [
                    (f"v{n}", (
                        p[0] * self.background_scale + background_rect.left(),
                        p[1] * self.background_scale + background_rect.top(),
                    ))
                    for n, p in enumerate(box["points"])
                ]
            else:
                x = box["x"] * self.background_scale + background_rect.left()
                y = box["y"] * self.background_scale + background_rect.top()
                w = box["width"] * self.background_scale
                h = box["height"] * self.background_scale
                corners = (("tl", (x, y)), ("tr", (x+w, y)),
                           ("bl", (x, y+h)), ("br", (x+w, y+h)))
            for hname, (hx, hy) in corners:
                hr = QRectF(hx - handle_size/2, hy - handle_size/2, handle_size, handle_size)
                if hr.contains(mouse_pos):
                    dx = mouse_pos.x() - hx
                    dy = mouse_pos.y() - hy
                    d2 = dx*dx + dy*dy
                    if d2 < min_dist:
                        min_dist = d2
                        best_box = i
                        best_handle = hname
        return best_box, best_handle

    def _polygon_edge_at(self, mouse_pos):
        """多边形边缘命中：(box_index, 插入索引, 图片坐标)，否则 None。"""
        from ..engine.shape_io import nearest_polygon_edge
        background_rect = self.get_background_rect()
        if background_rect is None:
            return None
        eps = DETECTION_BOX_CONFIG['resize_handle_size'] / self.background_scale
        img_pt = (
            (mouse_pos.x() - background_rect.left()) / self.background_scale,
            (mouse_pos.y() - background_rect.top()) / self.background_scale,
        )
        for i, box in enumerate(self._editor.detection_boxes):
            if not self._box_visible(box):
                continue
            if box.get("shape_type") != "polygon" or not box.get("points"):
                continue
            idx = nearest_polygon_edge(img_pt, box["points"], eps)
            if idx is not None:
                return i, idx, img_pt
        return None

    def _handle_detection_box_click(self, mouse_pos):
        if self.is_drawing_box or getattr(self, 'is_drawing_polygon', False):
            return False
        if not self._can_edit_canvas():
            return False
        background_rect = self.get_background_rect()
        if not background_rect:
            return False

        ctrl_pressed = bool(self._current_modifiers() & Qt.ControlModifier)

        # 旋转框手柄优先：选中 OBB 时其手柄可拖动旋转
        if (self.selected_box is not None and
                self._rotation_handle_at_pos(mouse_pos, self.selected_box)):
            return True

        # 第一遍：检查所有框的手柄，选最近的那个（解决重叠时下层框手柄被遮挡的问题）
        box_idx, handle = self._collect_nearest_handle(mouse_pos)
        if box_idx is not None:
            box = self._editor.detection_boxes[box_idx]
            box_x = box["x"] * self.background_scale + background_rect.left()
            box_y = box["y"] * self.background_scale + background_rect.top()
            box_width = box["width"] * self.background_scale
            box_height = box["height"] * self.background_scale
            if self._check_box_handle(mouse_pos, box_x, box_y, box_width, box_height, box_idx):
                return True

        # 第二遍：多边形边缘点击插入顶点
        edge = self._polygon_edge_at(mouse_pos)
        if edge is not None:
            i, vidx, img_pt = edge
            if self._insert_polygon_vertex(i, vidx, img_pt):
                self.selected_boxes = [i]
                self.selected_box = i
                return True

        # 第二遍：按顺序检查框内命中
        # 命中顺序与绘制顺序相反：关键点在最上层，先于框被命中；
        # 其余框后画的在上层，故倒序遍历。
        from ..engine.shape_io import point_in_polygon
        eps = DETECTION_BOX_CONFIG['resize_handle_size']

        def _hit(i, box):
            if box.get("shape_type") == "point" and box.get("points"):
                cx = box["points"][0][0] * self.background_scale + background_rect.left()
                cy = box["points"][0][1] * self.background_scale + background_rect.top()
                return (mouse_pos.x() - cx) ** 2 + (mouse_pos.y() - cy) ** 2 <= eps * eps
            if box.get("shape_type") in ("polygon", "rotation") and box.get("points"):
                canvas_pts = [
                    [
                        p[0] * self.background_scale + background_rect.left(),
                        p[1] * self.background_scale + background_rect.top(),
                    ]
                    for p in box["points"]
                ]
                return point_in_polygon(mouse_pos.x(), mouse_pos.y(), canvas_pts)
            box_x = box["x"] * self.background_scale + background_rect.left()
            box_y = box["y"] * self.background_scale + background_rect.top()
            box_width = box["width"] * self.background_scale
            box_height = box["height"] * self.background_scale
            return QRectF(box_x, box_y, box_width, box_height).contains(mouse_pos)

        indexed = [
            (i, box) for i, box in enumerate(self._editor.detection_boxes)
            if self._box_visible(box)
        ]
        points_first = [(i, b) for i, b in indexed if b.get("shape_type") == "point"]
        rest = [(i, b) for i, b in indexed if b.get("shape_type") != "point"]
        # 与绘制一致：已选中的点在最上层，优先命中；其余点后画的在上层。
        selected = set(getattr(self, 'selected_boxes', []) or [])
        if self.selected_box is not None:
            selected.add(self.selected_box)
        points_first.sort(key=lambda item: item[0] in selected)
        points_first = list(reversed(points_first))
        for i, box in points_first + list(reversed(rest)):
            hit = _hit(i, box)
            if hit:
                self.hover_resize_target = None
                self.hover_resize_handle = None
                self._editor.selected_item = None
                self.selected_item_size = None

                if ctrl_pressed:
                    self._toggle_box_selection(i)
                    self.update_status_label()
                    self.update()
                    return True

                self._editor.save_undo_state()
                self.selected_boxes = [i]
                self.selected_box = i
                self.box_drag_start = mouse_pos
                self.is_dragging_box = True
                self.setCursor(Qt.ClosedHandCursor)
                self.update_status_label()
                sync = getattr(self._editor, 'sync_label_list_to_box', None)
                if callable(sync):
                    sync(i)
                self.update()
                return True

        return False

    def _handle_background_click(self, mouse_pos):
        background_rect = self.get_background_rect()
        if not background_rect.contains(mouse_pos):
            return False

        # 按在图上就是「拖动画布」。**不再要求图比画布大**：图装得下时
        # `background_offset` 照样参与渲染（`canvas.py` 的 `get_background_rect`
        # 会把它加进去），所以照样该能挪。原来这里只在大图时才进入拖动，
        # 用户在不放大时按住拖就完全没反应（用户报的 bug）。
        self._clear_selection()
        self.drag_start = mouse_pos
        self.is_dragging_background = True
        return True

    def _clear_selection(self):
        self._editor.selected_item = None
        self.selected_item_size = None
        self.selected_box = None
        self.selected_boxes = []
        self.selected_region = None
        self.hover_resize_target = None
        self.hover_resize_handle = None
        self.resize_handle = None
        self.setCursor(Qt.ArrowCursor)
        self.update_status_label()
        clear_list = getattr(self._editor, 'clear_label_list_selection', None)
        if callable(clear_list):
            clear_list()
        self.update()

    def _toggle_box_selection(self, box_index):
        if box_index in self.selected_boxes:
            self.selected_boxes = [idx for idx in self.selected_boxes if idx != box_index]
        else:
            self.selected_boxes = [
                idx for idx in self.selected_boxes
                if 0 <= idx < len(self._editor.detection_boxes)
            ]
            self.selected_boxes.append(box_index)

        self.selected_box = self.selected_boxes[-1] if self.selected_boxes else None
        self.is_dragging_box = False
        self.is_resizing_box = False
        self.resize_handle = None
        sync = getattr(self._editor, 'sync_label_list_to_box', None)
        if callable(sync) and self.selected_box is not None:
            sync(self.selected_box)
        elif self.selected_box is None:
            clear_list = getattr(self._editor, 'clear_label_list_selection', None)
            if callable(clear_list):
                clear_list()

    def _current_modifiers(self):
        app = getattr(self._editor, 'app', None)
        if app is not None and hasattr(app, 'keyboardModifiers'):
            return app.keyboardModifiers()

        from PySide6.QtWidgets import QApplication
        return QApplication.keyboardModifiers()

    def _do_canvas_drag_out(self):
        """把当前图作为文件拖出（Qt 拖放），交给系统/其他程序接收。

        触发时机由 `mousePressEvent`（挂起）与 `mouseMoveEvent`（位移够阈值）
        共同决定，**这里不再自己判窗口内外**：原来那句「光标离开主窗口才算」
        让主窗口最大化时永远拖不出去。
        """
        if not getattr(self._editor, '_canvas_image_copy_enabled', False):
            self._drag_out_pending = False
            return
        idx = self._editor.current_background_index
        if 0 <= idx < len(self._editor.background_images):
            file_path = self._editor.background_images[idx]
            if os.path.isfile(file_path):
                self._editor._canvas_drag_active = True
                drag = QDrag(self)
                mime = QMimeData()
                mime.setUrls([QUrl.fromLocalFile(file_path)])
                drag.setMimeData(mime)
                self._drag_out_pending = False
                self._drag_out_start = None
                drag.exec_(Qt.CopyAction)
                self._editor._canvas_drag_active = False
        self._drag_out_pending = False
        self._drag_out_start = None

    def mouseMoveEvent(self, event):
        self.mouse_pos = event.pos()
        self._wheel_edge_target = None
        self.update_status_label()

        if self._drag_out_pending:
            # 阈值触发，与背景图列表（`DragOutListWidget` 的 20px）保持一致。
            #
            # 原来判的是「光标离开了整个主窗口」：主窗口最大化时光标出不去，
            # 于是按住左键拖多远都没反应（用户报的「画布图片复制打开后按住
            # 左键拖动没反应」）。改成移动够一段距离就触发，两边功能行为一致。
            start = getattr(self, '_drag_out_start', None)
            if start is not None:
                delta = event.pos() - start
                if (abs(delta.x()) > _DRAG_OUT_THRESHOLD
                        or abs(delta.y()) > _DRAG_OUT_THRESHOLD):
                    self._do_canvas_drag_out()
                    return

        if getattr(self, 'is_drawing_polygon', False):
            self.update()
            return

        if getattr(self, 'is_drawing_region_polygon', False):
            self.update()
            return

        if getattr(self, 'is_drawing_region', False) and not (
                self.is_dragging_region or self.is_resizing_region
                or self.is_dragging_region_vertex):
            if self.draw_start_pos is not None:
                bg_rect = self.get_background_rect()
                if bg_rect:
                    c = self._constrain_to_background(self.mouse_pos, bg_rect)
                    x1 = min(self.draw_start_pos.x(), c.x())
                    y1 = min(self.draw_start_pos.y(), c.y())
                    x2 = max(self.draw_start_pos.x(), c.x())
                    y2 = max(self.draw_start_pos.y(), c.y())
                    self.temp_draw_box = QRectF(x1, y1, x2 - x1, y2 - y1)
                else:
                    x1 = min(self.draw_start_pos.x(), self.mouse_pos.x())
                    y1 = min(self.draw_start_pos.y(), self.mouse_pos.y())
                    x2 = max(self.draw_start_pos.x(), self.mouse_pos.x())
                    y2 = max(self.draw_start_pos.y(), self.mouse_pos.y())
                    self.temp_draw_box = QRectF(x1, y1, x2 - x1, y2 - y1)
            self.update()
            return

        if self.is_drawing_box:
            if self.draw_start_pos:
                bg_rect = self.get_background_rect()
                if bg_rect:
                    c = self._constrain_to_background(self.mouse_pos, bg_rect)
                    x1 = min(self.draw_start_pos.x(), c.x())
                    y1 = min(self.draw_start_pos.y(), c.y())
                    x2 = max(self.draw_start_pos.x(), c.x())
                    y2 = max(self.draw_start_pos.y(), c.y())
                    self.temp_draw_box = QRectF(x1, y1, x2 - x1, y2 - y1)
                else:
                    x1 = min(self.draw_start_pos.x(), self.mouse_pos.x())
                    y1 = min(self.draw_start_pos.y(), self.mouse_pos.y())
                    x2 = max(self.draw_start_pos.x(), self.mouse_pos.x())
                    y2 = max(self.draw_start_pos.y(), self.mouse_pos.y())
                    self.temp_draw_box = QRectF(x1, y1, x2 - x1, y2 - y1)
            self.update()
            return

        if self.is_dragging_background:
            self.setCursor(Qt.ClosedHandCursor)
            delta = self.mouse_pos - self.drag_start
            self.background_offset += delta
            self.drag_start = self.mouse_pos
            self.update()
            return

        if self.is_dragging_region:
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_region()
            return

        if self.is_resizing_region:
            self.setCursor(Qt.ClosedHandCursor)
            self._resize_region()
            return

        if self.is_dragging_region_vertex:
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_region_vertex()
            return

        if self.is_dragging_box and self.selected_box is not None:
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_box()
            return

        if self.is_rotating_box and self.selected_box is not None:
            self.setCursor(Qt.ClosedHandCursor)
            self._rotate_selected_box()
            return

        if self.is_resizing_box and self.selected_box is not None:
            self.setCursor(Qt.ClosedHandCursor)
            self._resize_box()
            return

        if self.is_dragging_item and self._editor.selected_item is not None:
            self.setCursor(Qt.ClosedHandCursor)
            self._drag_item()
            return

        if self.resize_handle and self._editor.selected_item is not None:
            self.setCursor(Qt.ClosedHandCursor)
            self._scale_item()
            return

        self._check_hover()
        self.update()

    def _check_hover(self):
        """更新悬停状态。

        保持原有进入编辑状态方式：贴图点击进入；标注框鼠标移上去进入。
        同时只在当前编辑对象的缩放手柄上显示高亮提示。
        """
        old_target = self.hover_resize_target
        old_handle = self.hover_resize_handle
        self.hover_resize_target = None
        self.hover_resize_handle = None

        if self.is_drawing_box or getattr(self, 'is_drawing_polygon', False):
            return
        if self._editor.current_background is None or self._editor.current_background_index < 0:
            if (old_target, old_handle) != (self.hover_resize_target, self.hover_resize_handle):
                self.update()
            return

        bg_rect = self.get_background_rect()
        if bg_rect is None:
            if (old_target, old_handle) != (self.hover_resize_target, self.hover_resize_handle):
                self.update()
            return

        is_annotate = getattr(self._editor, 'edit_mode', 'paste') == 'annotate'
        if (is_annotate and self._editor.show_labels_checkbox.isChecked() and
                self._select_hovered_detection_box(bg_rect)):
            pass
        elif not is_annotate and (item_index := self.find_item_at_position(self.mouse_pos)) is not None:
            _, rect, _ = self._editor.canvas_items[item_index]
            self._editor.selected_item = item_index
            self.selected_item_size = (rect.width(), rect.height())
            self.selected_box = None
            self.selected_boxes = []
            handle = self._item_handle_at_pos(self.mouse_pos, rect)
            if handle:
                self.hover_resize_target = 'item'
                self.hover_resize_handle = handle
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
        elif (self._editor.selected_item is not None and
            0 <= self._editor.selected_item < len(self._editor.canvas_items)):
            _, rect, _ = self._editor.canvas_items[self._editor.selected_item]
            handle = self._item_handle_at_pos(self.mouse_pos, rect)
            if handle:
                self.hover_resize_target = 'item'
                self.hover_resize_handle = handle
                self.setCursor(Qt.PointingHandCursor)
            elif self._item_rect_contains(rect, self.mouse_pos):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif (self.selected_box is not None and
              0 <= self.selected_box < len(self._editor.detection_boxes)):
            handle = self._box_handle_at_pos(self.mouse_pos, self.selected_box)
            if handle:
                self.hover_resize_target = 'box'
                self.hover_resize_handle = handle
                self.setCursor(Qt.PointingHandCursor)
            elif self._box_rect_contains(self.selected_box, self.mouse_pos):
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if (self.hover_resize_target is None
                and not self.is_drawing_box
                and not getattr(self, 'is_drawing_polygon', False)
                and not getattr(self, 'is_drawing_region', False)
                and self.find_item_at_position(self.mouse_pos) is None):
            prev_region_handle = getattr(self, 'hover_region_handle', None)
            self._check_region_hover()
            # 区域的悬停手柄变了要重画，否则「白圆圈」不会变成「白正方形」
            if getattr(self, 'hover_region_handle', None) != prev_region_handle:
                self.update()
        elif getattr(self, 'hover_region_handle', None) is not None:
            self.hover_region_handle = None
            self.update()

        if (self.hover_resize_target is None
                and not self.is_drawing_box
                and not getattr(self, 'is_drawing_polygon', False)
                and getattr(self._editor, 'edit_mode', 'paste') == 'annotate'
                and self._editor.show_labels_checkbox.isChecked()
                and self._polygon_edge_at(self.mouse_pos) is not None):
            self.setCursor(Qt.CrossCursor)

        if (old_target, old_handle) != (self.hover_resize_target, self.hover_resize_handle):
            self.update()

    def _check_region_hover(self):
        """区域悬停：记下**悬停在哪条手柄上**，并把光标切成对应形状。

        用户要求「推理范围内的区域编辑要和检测框的编辑一致」——检测框把悬停的
        那个角画成**白色正方形**（`_draw_box_handles()` 里的 `is_hovered` 分支），
        区域过去只有白圆圈、没有悬停态，看起来「没有这个功能」。这里把检测框
        那套状态搬过来：`hover_region_handle` 存手柄名，渲染端据此画正方形。

        返回值沿用旧语义：**是否落在某个区域上**（供调用方决定后续分流）。
        """
        self.hover_region_handle = None
        if self._active_region_fixed():
            return False
        # 悬停态只在**当前模式可编辑**时给，与点击/滚轮同一把尺子；
        # 面板没开时不该画出「可以拖」的假象。
        if not self._region_editable():
            return False
        # 顶点命中（多边形）优先：与检测框的多边形分支一致，给十字光标
        if self._region_vertex_at(self.mouse_pos) is not None:
            self.setCursor(Qt.PointingHandCursor)
            return True
        index = self._region_at(self.mouse_pos)
        if index is None:
            return False
        region = (self._active_regions() or [])[index]
        if (region.get('shape_type') or 'rectangle') == 'polygon':
            if self._region_polygon_edge_at(self.mouse_pos) is not None:
                self.setCursor(Qt.CrossCursor)
                return True
            self.setCursor(Qt.OpenHandCursor)
            return True
        handle = self._region_handle_at(self.mouse_pos, index)
        if handle:
            # 与检测框一致：悬停在手柄上画白色正方形 + **小手**光标。
            # 检测框那边（`_select_hovered_detection_box`）给的就是
            # `PointingHandCursor`；这里曾错写成十字（CrossCursor），
            # 用户看到「移到区域手柄上变十字」就是那个错。
            self.hover_region_handle = handle
            self.hover_region_index = index
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)
        return True

    def _select_hovered_detection_box(self, background_rect):
        """鼠标移入检测框即进入该框编辑状态。"""
        try:
            modifiers = self._current_modifiers()
        except AttributeError:
            modifiers = 0
        ctrl_pressed = bool(modifiers & Qt.ControlModifier)

        # 第一遍：检查所有框的手柄，选最近的
        box_idx, handle = self._collect_nearest_handle(self.mouse_pos)
        if box_idx is not None:
            i = box_idx
            selection_changed = self.selected_box != i
            if ctrl_pressed:
                valid_indexes = [
                    idx for idx in self.selected_boxes
                    if 0 <= idx < len(self._editor.detection_boxes)
                ]
                if i not in valid_indexes:
                    valid_indexes.append(i)
                    selection_changed = True
                self.selected_boxes = valid_indexes
                self.selected_box = i
            else:
                if self.selected_boxes != [i]:
                    selection_changed = True
                self.selected_box = i
                self.selected_boxes = [i]

            if selection_changed:
                self._editor.selected_item = None
                self.selected_item_size = None
                self.update_status_label()
                sync = getattr(self._editor, 'sync_label_list_to_box', None)
                if callable(sync):
                    sync(i)
            self.hover_resize_target = 'box'
            self.hover_resize_handle = handle
            self.setCursor(Qt.PointingHandCursor)
            return True

        from ..engine.shape_io import point_in_polygon
        eps = DETECTION_BOX_CONFIG['resize_handle_size']
        for i, box in enumerate(self._editor.detection_boxes):
            if not self._box_visible(box):
                continue
            if box.get("shape_type") == "point" and box.get("points"):
                cx = box["points"][0][0] * self.background_scale + background_rect.left()
                cy = box["points"][0][1] * self.background_scale + background_rect.top()
                hit = ((self.mouse_pos.x() - cx) ** 2 +
                       (self.mouse_pos.y() - cy) ** 2) <= eps * eps
            elif box.get("shape_type") in ("polygon", "rotation") and box.get("points"):
                canvas_pts = [
                    [
                        p[0] * self.background_scale + background_rect.left(),
                        p[1] * self.background_scale + background_rect.top(),
                    ]
                    for p in box["points"]
                ]
                hit = point_in_polygon(self.mouse_pos.x(), self.mouse_pos.y(), canvas_pts)
                box_rect = QRectF(
                    box["x"] * self.background_scale + background_rect.left(),
                    box["y"] * self.background_scale + background_rect.top(),
                    box["width"] * self.background_scale,
                    box["height"] * self.background_scale,
                )
            else:
                box_rect = QRectF(
                    box["x"] * self.background_scale + background_rect.left(),
                    box["y"] * self.background_scale + background_rect.top(),
                    box["width"] * self.background_scale,
                    box["height"] * self.background_scale,
                )
                hit = box_rect.contains(self.mouse_pos)

            if hit:
                handle = self._box_handle_at_pos(
                    self.mouse_pos, i,
                    box_rect.x(), box_rect.y(), box_rect.width(), box_rect.height()
                )
                selection_changed = self.selected_box != i
                if ctrl_pressed:
                    valid_indexes = [
                        idx for idx in self.selected_boxes
                        if 0 <= idx < len(self._editor.detection_boxes)
                    ]
                    if i not in valid_indexes:
                        valid_indexes.append(i)
                        selection_changed = True
                    self.selected_boxes = valid_indexes
                    self.selected_box = i
                else:
                    if self.selected_boxes != [i]:
                        selection_changed = True
                    self.selected_box = i
                    self.selected_boxes = [i]

                if selection_changed:
                    self._editor.selected_item = None
                    self.selected_item_size = None
                    self.update_status_label()
                    sync = getattr(self._editor, 'sync_label_list_to_box', None)
                    if callable(sync):
                        sync(i)

                if handle:
                    self.hover_resize_target = 'box'
                    self.hover_resize_handle = handle
                    self.setCursor(Qt.PointingHandCursor)
                else:
                    self.setCursor(Qt.OpenHandCursor)
                return True

        if self.selected_box is None or not (0 <= self.selected_box < len(self._editor.detection_boxes)):
            return False

        handle = self._box_handle_at_pos(self.mouse_pos, self.selected_box)
        if handle:
            self.hover_resize_target = 'box'
            self.hover_resize_handle = handle
            self.setCursor(Qt.PointingHandCursor)
            return True

        return False

    def _item_rect_contains(self, rect, mouse_pos):
        background_rect = self.get_background_rect()
        if background_rect is None:
            return False

        item_rect = QRectF(
            rect.x() * self.background_scale + background_rect.left(),
            rect.y() * self.background_scale + background_rect.top(),
            rect.width() * self.background_scale,
            rect.height() * self.background_scale,
        )
        return item_rect.contains(mouse_pos)

    def _box_rect_contains(self, box_index, mouse_pos):
        if box_index is None or not (0 <= box_index < len(self._editor.detection_boxes)):
            return False

        background_rect = self.get_background_rect()
        if background_rect is None:
            return False

        box = self._editor.detection_boxes[box_index]
        box_rect = QRectF(
            box["x"] * self.background_scale + background_rect.left(),
            box["y"] * self.background_scale + background_rect.top(),
            box["width"] * self.background_scale,
            box["height"] * self.background_scale,
        )
        return box_rect.contains(mouse_pos)

    def _drag_item(self):
        if not self._can_edit_canvas():
            return
        delta = self.mouse_pos - self.drag_start
        bg_rect = self.get_background_rect()
        if bg_rect:
            p, rect, label = self._editor.canvas_items[self._editor.selected_item]
            dx = delta.x() / self.background_scale
            dy = delta.y() / self.background_scale

            nx = rect.x() + dx
            ny = rect.y() + dy

            nx = max(0, nx)
            ny = max(0, ny)

            if self._editor.current_background:
                bw = self._editor.current_background.width()
                bh = self._editor.current_background.height()
                nx = min(nx, bw - rect.width())
                ny = min(ny, bh - rect.height())

            nr = QRectF(nx, ny, rect.width(), rect.height())
            self._editor.canvas_items[self._editor.selected_item] = (p, nr, label)
            self.drag_start = self.mouse_pos
            self.update()

    def _scale_item(self):
        if not self._can_edit_canvas():
            return
        if self._editor.current_background:
            p, rect, label = self._editor.canvas_items[self._editor.selected_item]
            nr = QRectF(rect)

            bg_rect = self.get_background_rect()
            if bg_rect is None:
                return
            ix = (rect.x() * self.background_scale) + bg_rect.left()
            iy = (rect.y() * self.background_scale) + bg_rect.top()
            iw = rect.width() * self.background_scale
            ih = rect.height() * self.background_scale

            if self.resize_handle == 'br':
                nw = self.mouse_pos.x() - ix
                nh = self.mouse_pos.y() - iy

                wr = max(0.01, nw / iw)
                hr = max(0.01, nh / ih)
                sr = min(wr, hr)

                new_w = rect.width() * sr
                new_h = rect.height() * sr

                min_edge = 10
                if new_w < new_h:
                    if new_w < min_edge:
                        new_w = min_edge
                        new_h = min_edge * (rect.height() / rect.width())
                else:
                    if new_h < min_edge:
                        new_h = min_edge
                        new_w = min_edge * (rect.width() / rect.height())

                if self._editor.current_background:
                    bw = self._editor.current_background.width()
                    bh = self._editor.current_background.height()
                    new_w = min(new_w, bw - rect.x())
                    new_h = min(new_h, bh - rect.y())

                nr.setWidth(new_w)
                nr.setHeight(new_h)

                self.selected_item_size = (new_w, new_h)

            self._editor.canvas_items[self._editor.selected_item] = (p, nr, label)
            self.update()

    def mouseDoubleClickEvent(self, event):
        if getattr(self, 'is_drawing_polygon', False) and event.button() == Qt.LeftButton:
            self._prompt_finish_polygon(pop_duplicate=True)
            event.accept()
            return
        if getattr(self, 'is_drawing_region_polygon', False) and event.button() == Qt.LeftButton:
            self._prompt_finish_region_polygon(pop_duplicate=True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if getattr(self, 'is_drawing_region', False) and event.key() == Qt.Key_Escape:
            self._cancel_region_drawing()
            event.accept()
            return
        if getattr(self, 'is_drawing_region_polygon', False):
            if event.key() == Qt.Key_Backspace:
                self._region_polygon_pop_last_point()
                event.accept()
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._prompt_finish_region_polygon()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self._cancel_region_drawing()
                event.accept()
                return
        if getattr(self, 'is_drawing_polygon', False):
            if event.key() == Qt.Key_Backspace:
                self._polygon_pop_last_point()
                event.accept()
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._prompt_finish_polygon()
                event.accept()
                return
            if event.key() == Qt.Key_Escape:
                self._reset_drawing_state()
                event.accept()
                return
        super().keyPressEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_out_pending = False

        if getattr(self, 'is_drawing_region', False) and self.draw_start_pos is not None:
            self._complete_region_drawing(event.pos())
            self._check_hover()
            self.update()
            return

        if self._can_edit_canvas() and (self.is_dragging_box or self.is_resizing_box or self.is_rotating_box):
            if hasattr(self, '_needs_save') and self._needs_save:
                self._save_current_detection_boxes()
        self._needs_save = False

        self.is_dragging_item = False
        self.is_dragging_background = False
        self.is_dragging_box = False
        self.is_resizing_box = False
        # 区域手势收尾要通知宿主：范围面板显示的是 `region_boxes` 的**快照**，
        # 拖动/缩放/改顶点之后坐标已经变了，不通知它就一直停在旧坐标上
        # （用户第 10 轮：「推理时调整区域，窗口内没更新坐标信息」）。
        # 只在**真的动过区域**时通知，避免每次松手都白刷一遍面板。
        region_touched = bool(getattr(self, 'is_dragging_region_vertex', False)
                              or getattr(self, 'is_dragging_region', False)
                              or getattr(self, 'is_resizing_region', False))
        self.is_dragging_region_vertex = False
        self.region_vertex_drag_index = None
        self.is_rotating_box = False
        self.is_dragging_region = False
        self.is_resizing_region = False
        self.region_resize_handle = None
        self.rotation_prev_angle = None
        self.resize_handle = None
        if region_touched:
            self._notify_region_changed()
        self._check_hover()
        self.update()

    def wheelEvent(self, event):
        if not self._editor.current_background:
            return

        region_wheeled = False
        if event.modifiers() & Qt.ControlModifier:
            self._scale_background(event)
        elif (self._region_editable() and
              self._wheel_target_region() is not None and
              self.find_item_at_position(self.mouse_pos) is None and
              not self._selected_box_takes_wheel()):
            # 与检测框同一套分流：鼠标**贴边**时滚轮单边调整，在**中间**时
            # 整体缩放（用户要求「单边调整包括滚轮，复用检测框的」）。
            # 边上不再画手柄，所以这里要主动判一次位置，不能靠「有没有抓手柄」。
            self._wheel_region(event)
            region_wheeled = True
        elif self._editor.selected_item is not None:
            self._scale_selected_item(event)
        elif self.selected_box is not None:
            box = self._editor.detection_boxes[self.selected_box]
            if box.get("shape_type") == "rotation":
                # 旋转框整体缩放，不做 bbox 单边调整（否则与 points 失同步）
                self._scale_selected_box(event)
            else:
                locked_target = getattr(self, '_wheel_edge_target', None)
                if locked_target and locked_target[0] == self.selected_box:
                    self._adjust_selected_box_edge(event, locked_target[1])
                elif self._is_mouse_inside_selected_box():
                    self._scale_selected_box(event)
                else:
                    edge = self._get_selected_box_edge()
                    if edge:
                        self._wheel_edge_target = (self.selected_box, edge)
                        self._adjust_selected_box_edge(event, edge)

        # 滚轮不产生鼠标移动事件，必须主动刷新状态栏宽高
        self.update_status_label()
        # 滚轮改的是区域几何：范围面板的坐标快照也要跟着刷，否则「滚完面板
        # 还是旧数字」（用户第 10 轮：「调整区域，窗口内没更新坐标信息」）。
        if region_wheeled:
            self._notify_region_changed()
        self.update()

    def _selected_box_takes_wheel(self):
        """当前选中的检测框是否应当**优先**吃掉这次滚轮。

        用户第 10 轮把贴边滚轮也放给区域（"检测框处于编辑模式，鼠标放在一侧
        滚轮可以调整对应边"），但不能因此把检测框自己的滚轮抢走——已有回归
        测试钉住「标注模式下选中检测框时，悬停在区域上也必须缩框、不动区域」。

        判据：**标注模式**下、有选中的检测框，且鼠标落在它身上（框内，或贴着
        它的某条边）时，滚轮归框；区域只在没有框在管这件事时接手。

        贴图模式下恒为 False：那时画布改的是素材，标注框不该抢滚轮——已有的
        对照测试正是「贴图模式下选中着框，悬停区域也要缩放区域」。
        """
        if self._is_paste_mode():
            return False
        if self.selected_box is None:
            return False
        boxes = getattr(self._editor, 'detection_boxes', None) or []
        if not (0 <= self.selected_box < len(boxes)):
            return False
        if self._is_mouse_inside_selected_box():
            return True
        return self._get_selected_box_edge() is not None

    def _wheel_target_region(self):
        """滚轮该作用在哪个区域上——**贴边含外侧**，与检测框一致。

        用户第 10 轮：「检测框处于编辑模式，鼠标放在一侧滚轮可以调整对应边平移」，
        即检测框的滚轮命中**不要求鼠标在框内**，贴在外侧正对某条边也算。

        原先这里用的是 `_region_at()`（严格在区域内部），实测反了：
        图内坐标 scale=1.28 时「右边内侧 5px」返回 None（进不来），
        「右边外侧 5px」反而命中——因为内侧那点已经落在 `min_size` 边距
        之外被算作「不在区域内」。于是用户按检测框的手感去贴边滚，没反应。

        现在的判据：鼠标在区域内 **或** 落在某条边的容差带里（含外侧）都算。
        容差用 `_region_handle_hit_size()`（屏上像素）换算回图内坐标，与
        单边调整/手柄命中同一把尺子。

        返回区域下标；没有则 None。
        """
        regions = self._active_regions()
        if not regions:
            return None

        inside = self._region_at(self.mouse_pos)
        if inside is not None:
            return inside

        # 区域内没有：看是否贴在某个**未锁定**区域的边附近（外侧也算）
        background_rect = self.get_background_rect()
        if background_rect is None or not self.background_scale:
            return None
        mx = (self.mouse_pos.x() - background_rect.left()) / self.background_scale
        my = (self.mouse_pos.y() - background_rect.top()) / self.background_scale
        tol = self._region_wheel_edge_tolerance()

        # 倒序：与 `_region_at()` 一致，重叠时取最上面那条
        for i in range(len(regions) - 1, -1, -1):
            r = regions[i]
            if r.get('locked'):
                continue          # 锁定 = 最底层且不可修改，滚轮也不该碰
            if (r.get('shape_type') or 'rectangle') == 'polygon' and r.get('points'):
                # 多边形：用外接矩形判"贴边"，够用且与手柄绘制口径一致
                xs = [p[0] for p in r['points']]
                ys = [p[1] for p in r['points']]
                left, right = min(xs), max(xs)
                top, bottom = min(ys), max(ys)
            else:
                left, top = r.get('x', 0), r.get('y', 0)
                right, bottom = left + r.get('width', 0), top + r.get('height', 0)
            # 与检测框同一判据：**外侧不设距离上限**——鼠标贴在右边多远都算
            # 「在管右边这一条」（检测框实测偏移 60px 仍返回 'right'）；另一轴
            # 只要求落在该边覆盖范围内（用 tol 放宽一点，边缘上仍算）。
            if (min(abs(mx - left), abs(mx - right)) <= tol
                    and top - tol <= my <= bottom + tol):
                return i
            if (min(abs(my - top), abs(my - bottom)) <= tol
                    and left - tol <= mx <= right + tol):
                return i
            # 外侧正对某条边：横向/纵向**任意远**都接管（与 `_wheel_region`
            # 的判定配套，否则这里先返回 None，那边根本没机会判）。
            if mx < left and top - tol <= my <= bottom + tol:
                return i
            if mx > right and top - tol <= my <= bottom + tol:
                return i
            if my < top and left - tol <= mx <= right + tol:
                return i
            if my > bottom and left - tol <= mx <= right + tol:
                return i
        return None

    def _region_wheel_edge_tolerance(self):
        """滚轮判「贴边」的容差（图内坐标），与检测框同一把尺子。

        检测框那边（`_get_selected_box_edge()`）的命中规则是：

        - 鼠标在框**外**、且落在另一轴的范围里 → 直接认正对着的那条边；
          **外侧多远都算**（实测偏移 60px 仍返回 'right'）。
        - 四条都不满足 → 取距离最近的一条。

        区域这边原先只有 `_region_handle_hit_size()`（屏上 8px 换算回图内，
        scale=1.28 时图内仅 6.25px），于是鼠标压在外侧 8px 就没反应——用户报的
        「鼠标放在一侧滚动滚轮会调整一边，区域都没有」正是这个。

        现在**外侧不设限**（与检测框一致），只在「鼠标落在区域内部」时才用
        这个容差去决定「算贴边还是算中间」，避免把整个内部都当边缘。
        """
        base = self._region_handle_hit_size()          # 屏上像素，已夹在 8..24
        return base / max(self.background_scale, 1e-6)

    def _wheel_region(self, event):
        """区域滚轮：贴边 → 单边调整；在中间 → 整体缩放。

        判据是「鼠标是否落在某条边的容差带里」——**含区域外侧**，与检测框
        「鼠标放在一侧滚轮可以调整对应边」的手感对齐（用户要求）。
        容差随缩放换算，见 `_region_wheel_edge_tolerance()`。
        """
        index = self._wheel_target_region()
        if index is None:
            return
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return
        background_rect = self.get_background_rect()
        if background_rect is None or not self.background_scale:
            return

        r = regions[index]
        mx = (self.mouse_pos.x() - background_rect.left()) / self.background_scale
        my = (self.mouse_pos.y() - background_rect.top()) / self.background_scale
        left, top = r['x'], r['y']
        right, bottom = left + r['width'], top + r['height']
        tol = self._region_wheel_edge_tolerance()

        # 与检测框 `_get_selected_box_edge()` **同一套判据**（用户要求「复用检测
        # 框的」）：先在**外侧**找正对着的那条边——外侧不设距离上限，鼠标贴在
        # 右边多远都算右边；再退回「内部且贴边」的容差带；都不满足才是整体缩放。
        edge = None
        if mx < left and top - tol <= my <= bottom + tol:
            edge = 'left'
        elif mx > right and top - tol <= my <= bottom + tol:
            edge = 'right'
        elif my < top and left - tol <= mx <= right + tol:
            edge = 'top'
        elif my > bottom and left - tol <= mx <= right + tol:
            edge = 'bottom'
        elif min(abs(mx - left), abs(mx - right),
                 abs(my - top), abs(my - bottom)) <= tol:
            # 区域**内部**贴近某条边：也算单边调整（用户：滚轮贴边调整）
            edge = self._region_edge_at(self.mouse_pos, index)

        if edge is not None:
            self._adjust_hovered_region_edge(event, index, edge)
        else:
            self._scale_hovered_region(event, index)

    def _scale_hovered_region(self, event, index=None):
        """鼠标悬停在区域上时滚轮缩放该区域（绕中心，夹紧背景边界）。

        `index` 允许调用方传入（`_wheel_region` 已经解析过目标区域，它可能
        是"贴在区域外侧"的那种命中，重新用 `_region_at()` 会解析不出来）。
        """
        if self._active_region_fixed():
            return
        if index is None:
            index = self._region_at(self.mouse_pos)
        if index is None:
            return
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return

        from ..core.config import REGION_CONFIG, DETECTION_BOX_WHEEL_CONFIG
        delta = event.angleDelta().y()
        step = max(0.01, min(0.30, float(
            DETECTION_BOX_WHEEL_CONFIG.get('region_scale_step', 0.05))))
        scale_factor = 1.0 + step if delta > 0 else max(0.1, 1.0 - step)

        region = regions[index]
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        cx = x + w / 2
        cy = y + h / 2
        min_size = REGION_CONFIG['min_size']

        new_w = max(min_size, w * scale_factor)
        new_h = max(min_size, h * scale_factor)

        bg = self._editor.current_background
        if bg is not None:
            new_w = min(new_w, bg.width())
            new_h = min(new_h, bg.height())

        nx = cx - new_w / 2
        ny = cy - new_h / 2
        if bg is not None:
            nx = max(0, min(nx, bg.width() - new_w))
            ny = max(0, min(ny, bg.height() - new_h))

        region['x'], region['y'] = nx, ny
        region['width'], region['height'] = new_w, new_h
        if (region.get('shape_type') or 'rectangle') == 'polygon' and region.get('points'):
            sx = new_w / w if w > 0 else 1.0
            sy = new_h / h if h > 0 else 1.0
            region['points'] = [
                [nx + (p[0] - x) * sx, ny + (p[1] - y) * sy]
                for p in region['points']
            ]
        self.selected_region = index
        self.update()

    def _region_edge_at(self, mouse_pos, index):
        """返回鼠标相对该区域最近的一条边：`left/right/top/bottom`。

        与检测框的 `_get_selected_box_edge()` 同一套判据（用户要求「复用检测框
        的了」）：先在框**外**找正对着的那条边，都在外面时取距离最近的。
        区域内侧也算——用户要的是「滚轮贴边调整」，鼠标压在边上时也该调那条边。
        """
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return None
        background_rect = self.get_background_rect()
        if background_rect is None or not self.background_scale:
            return None
        r = regions[index]
        left = r['x']
        top = r['y']
        right = left + r['width']
        bottom = top + r['height']
        mx = (mouse_pos.x() - background_rect.left()) / self.background_scale
        my = (mouse_pos.y() - background_rect.top()) / self.background_scale

        # 框外且正对着某条边 → 直接是它
        if mx < left and top <= my <= bottom:
            return 'left'
        if mx > right and top <= my <= bottom:
            return 'right'
        if my < top and left <= mx <= right:
            return 'top'
        if my > bottom and left <= mx <= right:
            return 'bottom'
        # 否则取四条边里最近的一条
        distances = {
            'left': abs(mx - left),
            'right': abs(mx - right),
            'top': abs(my - top),
            'bottom': abs(my - bottom),
        }
        return min(distances, key=distances.get)

    def _adjust_hovered_region_edge(self, event, index, edge):
        """滚轮**单边**调整该区域：只动被点名的那条边，其余三边不动。

        这是检测框 `_adjust_selected_box_edge()` 的区域版。与
        `_scale_hovered_region()`（绕中心等比缩放）互补：
        - 鼠标压在某条边附近 → 本条（单边）
        - 鼠标在区域中间 → 那条（整体缩放）
        """
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return
        if edge is None:
            return
        from ..core.config import (
            DETECTION_BOX_WHEEL_CONFIG, REGION_CONFIG,
        )
        step = max(1, min(50, int(
            DETECTION_BOX_WHEEL_CONFIG.get('edge_step', 5))))
        step = step if event.angleDelta().y() > 0 else -step

        r = regions[index]
        min_size = REGION_CONFIG['min_size']
        left = r['x']
        top = r['y']
        right = left + r['width']
        bottom = top + r['height']
        bg = self._editor.current_background
        max_w = bg.width() if bg is not None else None
        max_h = bg.height() if bg is not None else None

        if edge == 'left':
            new_left = max(0, min(left - step, right - min_size))
            r['x'] = new_left
            r['width'] = right - new_left
        elif edge == 'right':
            new_right = right + step
            if max_w is not None:
                new_right = min(max_w, new_right)
            r['width'] = max(min_size, new_right - left)
        elif edge == 'top':
            new_top = max(0, min(top - step, bottom - min_size))
            r['y'] = new_top
            r['height'] = bottom - new_top
        else:                                   # bottom
            new_bottom = bottom + step
            if max_h is not None:
                new_bottom = min(max_h, new_bottom)
            r['height'] = max(min_size, new_bottom - top)

        # 多边形区域：points 必须跟着矩形一起走，否则渲染与命中的是两个形状
        if (r.get('shape_type') or 'rectangle') == 'polygon' and r.get('points'):
            old_left, old_top = left, top
            sx = r['width'] / (right - old_left) if right > old_left else 1.0
            sy = r['height'] / (bottom - old_top) if bottom > old_top else 1.0
            r['points'] = [
                [r['x'] + (p[0] - old_left) * sx,
                 r['y'] + (p[1] - old_top) * sy]
                for p in r['points']
            ]
        self.selected_region = index
        self.update()

    def _is_mouse_inside_selected_box(self):
        return self._box_rect_contains(self.selected_box, self.mouse_pos)

    def _get_mouse_pos_in_image_coords(self):
        background_rect = self.get_background_rect()
        if background_rect is None or not self.background_scale:
            return None
        return (
            (self.mouse_pos.x() - background_rect.left()) / self.background_scale,
            (self.mouse_pos.y() - background_rect.top()) / self.background_scale,
        )

    @staticmethod
    def _clamp_size_with_aspect(new_w, new_h, orig_w, orig_h, min_size=10, max_size=None):
        if max_size is None:
            max_size = min_size * 100
        ratio = orig_h / orig_w if orig_w else 1.0
        if new_w < new_h:
            if new_w < min_size:
                new_w, new_h = min_size, min_size * ratio
            elif new_w > max_size:
                new_w, new_h = max_size, max_size * ratio
        else:
            if new_h < min_size:
                new_h, new_w = min_size, min_size / ratio
            elif new_h > max_size:
                new_h, new_w = max_size, max_size / ratio
        return new_w, new_h

    def _scale_selected_item(self, event):
        if not self._can_edit_canvas():
            return
        if (self._editor.selected_item is None or
            self._editor.selected_item >= len(self._editor.canvas_items)):
            return

        delta = event.angleDelta().y()
        step = max(0.01, min(0.30, float(DETECTION_BOX_WHEEL_CONFIG.get('paste_item_scale_step', 0.15))))
        scale_factor = 1.0 + step if delta > 0 else max(0.1, 1.0 - step)

        pixmap, rect, label = self._editor.canvas_items[self._editor.selected_item]

        new_width = rect.width() * scale_factor
        new_height = rect.height() * scale_factor

        bg_short_side = min(self._editor.current_background.width(),
                          self._editor.current_background.height())
        new_width, new_height = self._clamp_size_with_aspect(
            new_width, new_height,
            rect.width(), rect.height(),
            min_size=10, max_size=bg_short_side * 0.9
        )

        center_x = rect.center().x()
        center_y = rect.center().y()

        new_rect = QRectF(
            center_x - new_width / 2,
            center_y - new_height / 2,
            new_width, new_height
        )

        self._editor.canvas_items[self._editor.selected_item] = (pixmap, new_rect, label)
        self.selected_item_size = (new_width, new_height)

    def _scale_selected_box(self, event):
        if not self._can_edit_canvas():
            return
        if (self.selected_box is None or
            self.selected_box >= len(self._editor.detection_boxes)):
            return

        delta = event.angleDelta().y()
        step = max(0.01, min(0.30, float(DETECTION_BOX_WHEEL_CONFIG.get('detection_box_scale_step', 0.05))))
        scale_factor = 1.0 + step if delta > 0 else max(0.1, 1.0 - step)

        box = self._editor.detection_boxes[self.selected_box]
        if box.get("shape_type") in ("polygon", "rotation") and box.get("points"):
            self._scale_polygon(box, scale_factor)
            self._sync_detection_box_to_dict(self.selected_box)
            return

        x, y, width, height = box["x"], box["y"], box["width"], box["height"]

        new_width = width * scale_factor
        new_height = height * scale_factor

        bg_size = min(self._editor.current_background.width(),
                     self._editor.current_background.height())
        new_width, new_height = self._clamp_size_with_aspect(
            new_width, new_height,
            width, height,
            min_size=10, max_size=bg_size * 0.9
        )

        center_x = x + width / 2
        center_y = y + height / 2

        new_x = center_x - new_width / 2
        new_y = center_y - new_height / 2

        new_x = max(0, min(new_x, self._editor.current_background.width() - new_width))
        new_y = max(0, min(new_y, self._editor.current_background.height() - new_height))

        box["x"] = new_x
        box["y"] = new_y
        box["width"] = new_width
        box["height"] = new_height

        self._sync_detection_box_to_dict(self.selected_box)

    def _get_selected_box_edge(self):
        """返回鼠标相对当前检测框的最近边。"""
        if (self.selected_box is None or
                self.selected_box >= len(self._editor.detection_boxes)):
            return None

        box = self._editor.detection_boxes[self.selected_box]
        mouse_pos = self._get_mouse_pos_in_image_coords()
        if mouse_pos is None:
            return None

        left = box["x"]
        top = box["y"]
        right = left + box["width"]
        bottom = top + box["height"]
        mouse_x, mouse_y = mouse_pos

        if mouse_x < left and top <= mouse_y <= bottom:
            return 'left'
        if mouse_x > right and top <= mouse_y <= bottom:
            return 'right'
        if mouse_y < top and left <= mouse_x <= right:
            return 'top'
        if mouse_y > bottom and left <= mouse_x <= right:
            return 'bottom'

        distances = {
            'left': abs(mouse_x - left),
            'right': abs(mouse_x - right),
            'top': abs(mouse_y - top),
            'bottom': abs(mouse_y - bottom),
        }
        return min(distances, key=distances.get)

    def _adjust_selected_box_edge(self, event, edge=None):
        if not self._can_edit_canvas():
            return
        if (self.selected_box is None or
            self.selected_box >= len(self._editor.detection_boxes)):
            return

        box = self._editor.detection_boxes[self.selected_box]
        bg = self._editor.current_background
        min_width = 10
        min_height = 10
        edge_step = max(1, min(50, int(DETECTION_BOX_WHEEL_CONFIG.get('edge_step', 5))))
        step = edge_step if event.angleDelta().y() > 0 else -edge_step

        left = box["x"]
        top = box["y"]
        right = left + box["width"]
        bottom = top + box["height"]
        nearest_edge = edge or self._get_selected_box_edge()
        if nearest_edge is None:
            return

        if nearest_edge == 'left':
            new_left = max(0, min(left - step, right - min_width))
            box["x"] = new_left
            box["width"] = right - new_left
        elif nearest_edge == 'right':
            new_right = min(bg.width(), max(right + step, left + min_width))
            box["width"] = new_right - left
        elif nearest_edge == 'top':
            new_top = max(0, min(top - step, bottom - min_height))
            box["y"] = new_top
            box["height"] = bottom - new_top
        else:
            new_bottom = min(bg.height(), max(bottom + step, top + min_height))
            box["height"] = new_bottom - top

        self._sync_detection_box_to_dict(self.selected_box)

    def _scale_background(self, event):
        delta = event.angleDelta().y()
        scale_factor = 1.1 if delta > 0 else 0.9

        self.background_scale *= scale_factor
        self.background_scale = max(
            BACKGROUND_SCALE_CONFIG['min_scale'],
            min(self.background_scale, BACKGROUND_SCALE_CONFIG['max_scale'])
        )
        self.is_manual_scale = True

    def _nudge_selected(self, dx, dy):
        if not self._can_edit_canvas():
            return
        if self.selected_box is not None and 0 <= self.selected_box < len(self._editor.detection_boxes):
            box = self._editor.detection_boxes[self.selected_box]
            step = NUDGE_CONFIG['step']
            bg = self._editor.current_background
            bw = bg.width() if bg else 0
            bh = bg.height() if bg else 0
            nx = max(0, min(box["x"] + dx * step, bw - box["width"]))
            ny = max(0, min(box["y"] + dy * step, bh - box["height"]))
            applied_dx = nx - box["x"]
            applied_dy = ny - box["y"]
            box["x"] = nx
            box["y"] = ny
            if box.get("points"):
                box["points"] = [[p[0] + applied_dx, p[1] + applied_dy] for p in box["points"]]
            self._sync_detection_box_to_dict(self.selected_box)
            self.update()
        elif self._editor.selected_item is not None and 0 <= self._editor.selected_item < len(self._editor.canvas_items):
            p, rect, label = self._editor.canvas_items[self._editor.selected_item]
            step = NUDGE_CONFIG['step']
            bg = self._editor.current_background
            bw = bg.width() if bg else 0
            bh = bg.height() if bg else 0
            nx = max(0, min(rect.x() + dx * step, bw - rect.width()))
            ny = max(0, min(rect.y() + dy * step, bh - rect.height()))
            nr = QRectF(nx, ny, rect.width(), rect.height())
            self._editor.canvas_items[self._editor.selected_item] = (p, nr, label)
            self.update()

    def resizeEvent(self, event):
        self.update()
        super().resizeEvent(event)
