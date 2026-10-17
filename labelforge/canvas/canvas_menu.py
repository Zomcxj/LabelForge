"""
Canvas 右键菜单 - 贴图标签管理
"""
import os
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu
from PySide6.QtCore import Qt, QPoint

from ..core.utils import extract_label_name
from ..ui import dialog_helpers
from ..ui.i18n import t as tr


class CanvasMenuMixin:
    """右键菜单逻辑"""

    def _active_regions(self):
        """**当前模式**该操作/显示的那份区域列表。

        用户第 10 轮把区域拆成两份，各自属于一种模式：
        - 贴图模式 → `paste_region_boxes`（贴图落点）
        - 标注模式 → `region_boxes`（推理范围）

        画布上所有区域相关的读写都经这里取列表，**不要**直接摸
        `region_boxes`——否则贴图模式会改到推理的那份（这正是第 10 轮报的
        「推理的区域在贴图下还能操作」）。
        """
        if self._is_paste_mode():
            if not isinstance(getattr(self._editor, 'paste_region_boxes', None), list):
                self._editor.paste_region_boxes = []
            return self._editor.paste_region_boxes
        if not isinstance(getattr(self._editor, 'region_boxes', None), list):
            self._editor.region_boxes = []
        return self._editor.region_boxes

    def _active_region_fixed(self):
        """当前模式的区域是否已固定（两份各记各的）。"""
        if self._is_paste_mode():
            return bool(getattr(self._editor, 'paste_region_fixed', False))
        return bool(getattr(self._editor, 'region_fixed', False))

    @staticmethod
    def _rounded_menu(parent):
        """建一个带圆角的右键悬浮菜单。

        只写 QSS 的 `border-radius` 是不够的：QMenu 是顶层窗口，Qt 默认给它
        一块不透明的方角窗口背景，圆角会被盖住（实测四角 alpha 仍为 255）。
        必须同时打开 `WA_TranslucentBackground` 并去掉边框，角上的像素才会
        真的被切掉。样式本身在 `theme.py` 的全局 `QMenu` 规则里。
        """
        menu = QMenu(parent)
        menu.setAttribute(Qt.WA_TranslucentBackground)
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint
                            | Qt.NoDropShadowWindowHint)
        return menu

    @staticmethod
    def _remove_index_from_mapping(mapping, removed_index):
        """删除背景图后同步重排按图片索引保存的数据。"""
        return {
            (index - 1 if index > removed_index else index): value
            for index, value in mapping.items()
            if index != removed_index
        }

    def _handle_right_click(self, mouse_pos):
        item_index = self.find_item_at_position(mouse_pos)
        if item_index is not None:
            self._show_paste_context_menu(item_index, mouse_pos)
            return True
        if self._is_paste_mode():
            vertex = self._region_vertex_at(mouse_pos)
            if vertex is not None:
                self._show_region_vertex_menu(vertex[0], vertex[1], mouse_pos)
                return True
            if self._region_at(mouse_pos) is not None:
                self._show_region_context_menu(mouse_pos)
                return True
        labels_visible = self._editor.show_labels_checkbox.isChecked()
        vertex = self._polygon_vertex_at(mouse_pos) if labels_visible else None
        if vertex is not None and not self._editor._is_delete_view:
            self._show_polygon_vertex_menu(vertex[0], vertex[1], mouse_pos)
            return True
        box_index = self._find_detection_box_at(mouse_pos) if labels_visible else None
        if box_index is not None and not self._editor._is_delete_view:
            self._show_box_label_menu(box_index, mouse_pos)
            return True
        if self._editor.current_background is not None:
            if self._editor._is_delete_view:
                self._show_restore_context_menu(mouse_pos)
            else:
                self._show_background_context_menu(mouse_pos)
            return True
        return False

    # ---------- 语义区域（仅运行时，贴图模式） ----------

    def _is_paste_mode(self):
        return (getattr(self._editor, 'edit_mode', 'paste') == 'paste'
                and not getattr(self._editor, '_is_delete_view', False))

    def _region_at(self, mouse_pos, include_locked=False):
        """返回鼠标位置下的区域索引（画布坐标转图像坐标判断）。

        `include_locked=False`（默认）跳过**已锁定**的区域：锁定 = 「最底层、
        不可修改」（用户要求），所以它必须从交互里彻底消失，点上去直接落到
        它下面的检测框或画布。只有「这里有没有区域」这类纯几何查询才传
        `True`（比如往外拖文件的守卫，那里关心的是有没有东西挡着，与能否
        编辑无关）。
        """
        regions = self._active_regions()
        if not regions:
            return None
        background_rect = self.get_background_rect()
        if background_rect is None or self.background_scale <= 0:
            return None
        ix = (mouse_pos.x() - background_rect.left()) / self.background_scale
        iy = (mouse_pos.y() - background_rect.top()) / self.background_scale
        from ..engine.shape_io import point_in_polygon
        for i in range(len(regions) - 1, -1, -1):
            r = regions[i]
            if not include_locked and r.get('locked'):
                continue
            if (r.get('shape_type') or 'rectangle') == 'polygon' and r.get('points'):
                if point_in_polygon(ix, iy, r['points']):
                    return i
                continue
            if (r['x'] <= ix <= r['x'] + r['width'] and
                    r['y'] <= iy <= r['y'] + r['height']):
                return i
        return None

    # ---------- 每条区域的锁定（用户要求：区域列表里每条各自锁） ----------

    @staticmethod
    def region_locked(region):
        """这条区域是否被锁定。缺键视为未锁（老数据/新画的区域都是未锁）。"""
        return bool((region or {}).get('locked', False))

    def region_locked_at(self, index):
        """第 index 条区域是否被锁定；越界或没有区域时返回 False。"""
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return False
        return self.region_locked(regions[index])

    def set_region_locked(self, index, locked):
        """锁定/解锁第 index 条区域；返回是否真的改了。

        锁定的语义（用户要求）：
        - **不锁定**：画在检测框**之上**，和检测框一样可拖动/缩放/改顶点；
        - **锁定**：落到检测框**之下**，不可修改，点击也穿透到下面的框；
        - 只是「不许在画布上改」，**区域照常参与推理**——锁不等于禁用。

        锁只写在区域字典自己身上。`region_fixed`（右键菜单里的「固定区域」）
        是**贴图模式**的全局开关，两者互不相干，不去动它。
        """
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return False
        region = regions[index]
        if not isinstance(region, dict):
            return False
        if bool(region.get('locked', False)) == bool(locked):
            return False
        region['locked'] = bool(locked)
        # 锁上时若正抓着这条（拖动/缩放/顶点），必须立刻松手：锁上还继续改
        # 就自相矛盾，而且松手前那次拖动会把刚锁的状态又写一遍。
        if locked and getattr(self, 'selected_region', None) == index:
            self.selected_region = None
            self.selected_region_vertex = None
            self.is_dragging_region = False
            self.is_resizing_region = False
            self.is_dragging_region_vertex = False
        self.update()
        return True

    def _region_vertex_at(self, mouse_pos):
        """返回鼠标位置下的区域多边形顶点 (index, vertex_index)。"""
        regions = self._active_regions()
        if not regions:
            return None
        background_rect = self.get_background_rect()
        if background_rect is None:
            return None
        from ..core.config import REGION_CONFIG
        size = REGION_CONFIG['handle_size']
        half = size / 2
        for i in range(len(regions) - 1, -1, -1):
            r = regions[i]
            if (r.get('shape_type') or 'rectangle') != 'polygon' or not r.get('points'):
                continue
            for v, p in enumerate(r['points']):
                hx = p[0] * self.background_scale + background_rect.left()
                hy = p[1] * self.background_scale + background_rect.top()
                if hx - half <= mouse_pos.x() <= hx + half and hy - half <= mouse_pos.y() <= hy + half:
                    return i, v
        return None

    def _show_region_vertex_menu(self, region_index, vertex_index, mouse_pos):
        regions = self._active_regions()
        if not regions or not (0 <= region_index < len(regions)):
            return
        region = regions[region_index]
        points = region.get('points') or []
        fixed = self._active_region_fixed()

        menu = self._rounded_menu(self)
        delete_action = QAction(tr("删除顶点"), self)
        delete_action.setEnabled(not fixed and len(points) > 3)
        delete_action.triggered.connect(
            lambda checked=False, r=region_index, v=vertex_index: self._delete_region_vertex(r, v)
        )
        menu.addAction(delete_action)
        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _delete_region_vertex(self, region_index, vertex_index):
        if self._active_region_fixed():
            return False
        regions = self._active_regions()
        if not regions or not (0 <= region_index < len(regions)):
            return False
        region = regions[region_index]
        points = region.get('points') or []
        if len(points) <= 3 or not (0 <= vertex_index < len(points)):
            return False
        del points[vertex_index]
        region['points'] = points
        self._sync_region_bbox(region)
        self.selected_region = region_index
        self.update()
        return True

    @staticmethod
    def _sync_region_bbox(region):
        """多边形区域的外接框与 points 同步。"""
        points = region.get('points') or []
        if not points:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        region['x'] = float(min(xs))
        region['y'] = float(min(ys))
        region['width'] = float(max(xs) - region['x'])
        region['height'] = float(max(ys) - region['y'])

    def _region_handle_at(self, mouse_pos, index):
        """返回区域缩放手柄名称：**四角 + 四边**都算。

        四角 `tl/tr/bl/br` 改两个方向，四边 `t/b/l/r` 只改一个方向
        （用户要求「复用检测框的缩放，单边调整和角点拖动」）。

        命中容差**不写死像素**：`handle_size` 是设计尺寸（8），实际比较用
        `_region_handle_hit_size()`，它会随 `background_scale` 放大——图缩小
        后区域框在屏上只有几十像素，固定 8px 的方框基本点不中，那正是用户
        看到的「没办法调整大小」。四角优先于四边：角上同时落在两条边的
        判定里，取角才能一次改两个方向。
        """
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return None
        background_rect = self.get_background_rect()
        if background_rect is None:
            return None
        r = regions[index]
        x = r['x'] * self.background_scale + background_rect.left()
        y = r['y'] * self.background_scale + background_rect.top()
        w = r['width'] * self.background_scale
        h = r['height'] * self.background_scale
        half = self._region_handle_hit_size() / 2
        mx, my = mouse_pos.x(), mouse_pos.y()

        def near(hx, hy):
            return hx - half <= mx <= hx + half and hy - half <= my <= hy + half

        # 四角优先
        for name, (hx, hy) in (
                ("tl", (x, y)), ("tr", (x + w, y)),
                ("bl", (x, y + h)), ("br", (x + w, y + h))):
            if near(hx, hy):
                return name
        # 再四边（中点）
        for name, (hx, hy) in (
                ("t", (x + w / 2, y)), ("b", (x + w / 2, y + h)),
                ("l", (x, y + h / 2)), ("r", (x + w, y + h / 2))):
            if near(hx, hy):
                return name
        return None

    def _region_handle_hit_size(self):
        """区域手柄的命中边长（画布像素）；与绘制共用 `_region_handle_size()`。

        过去这里是「设计尺寸 / 缩放」，图缩小时会把命中区放大到 24px。但
        用户要求手柄只比检测框大一点、且**看着别忽大忽小**，所以现在跟检测框
        一样是恒定像素（`_region_handle_size()`），不再随 `background_scale`
        变化。命中区与绘制必须同源，否则会出现「看着能抓却抓不到」。
        """
        return self._region_handle_size()

    def _region_handle_size(self):
        """区域句柄边长（画布像素）= **检测框句柄 + 3**，**不随缩放变化**。

        用户要求「句柄只比检测框大一点」。关键点是**不能缩放**：检测框
        `_draw_box_handles()` 用的是写死的 `DETECTION_BOX_CONFIG
        ['resize_handle_size']`，无论 `background_scale` 是多少都是 8px。
        区域过去按 `handle_size / scale` 放大（上限 24），于是图一缩小，区域
        句柄就变成检测框的两三倍大——用户原话「我参数拉到 15 都没你的区域框
        句柄大」，说的就是这个，跟基准值是 8 还是 9 完全无关。

        定义放在这里（与命中判定同一个类），绘制端 `_region_handle_draw_size()`
        转调它，保证「看到的」就是「抓得到的」。
        """
        from ..core.config import DETECTION_BOX_CONFIG, REGION_HANDLE_EXTRA
        return DETECTION_BOX_CONFIG['resize_handle_size'] + REGION_HANDLE_EXTRA

    def _show_region_context_menu(self, mouse_pos):
        index = self._region_at(mouse_pos)
        if index is None:
            return
        fixed = self._active_region_fixed()
        menu = self._rounded_menu(self)

        delete_action = QAction(tr("删除区域"), self)
        delete_action.setEnabled(not fixed)
        delete_action.triggered.connect(
            lambda checked, i=index: self._delete_region(i)
        )
        menu.addAction(delete_action)

        menu.addSeparator()
        self._add_region_actions(menu, fixed)

        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _add_region_actions(self, menu, fixed):
        """追加 添加区域（矩形）、固定、清空 菜单项。

        **只有矩形**（用户第 11 轮：「多边形不要了」）：推理范围只做矩形，
        多边形那条路（`_start_add_region_polygon`）已随右键段一起去掉，
        免得留一个画得出来、面板里却看不全的入口。
        """
        add_rect_action = QAction(tr("添加区域（矩形）"), self)
        add_rect_action.setEnabled(not fixed)
        add_rect_action.triggered.connect(self._start_add_region)
        menu.addAction(add_rect_action)

        # 「固定区域」**不用 `setCheckable`**：Qt 会给可勾选项的文字额外缩进
        # 一个 `PM_MenuButtonIndicator` 的宽度（本主题下 12px），于是它比同菜单
        # 里其它项多空出一个字——用户报的「固定区域没有左对齐空了一个字」。
        # 按 Qt 画菜单项的口径复算，可勾选时缩进是 [0, 12, 0]，去掉 checkable
        # 后是 [0, 0, 0]。
        #
        # 勾**仍然要有**（用户明确「我不是固定区域左边有打钩吗，不勾选不就解除了」），
        # 所以把勾画进文案本身：未固定显示「固定区域」，已固定显示「✔ 固定区域」，
        # 点一下照样来回切。这样勾看得见、各项也严格左对齐。
        fix_action = QAction(
            tr("✔ 固定区域") if fixed else tr("固定区域"), self)
        fix_action.triggered.connect(self._toggle_fix_regions)
        menu.addAction(fix_action)

        clear_action = QAction(tr("清空区域"), self)
        clear_action.setEnabled(not fixed and bool(self._active_regions()))
        clear_action.triggered.connect(self._clear_regions)
        menu.addAction(clear_action)

    def _start_add_region(self):
        if self._active_region_fixed():
            return
        self.is_drawing_region = True
        # 记下「这一次绘制是刚被 arm 的」：见 `_region_draw_armed()`。
        self._region_draw_armed_flag = True
        self.is_drawing_region_polygon = False
        self.temp_region_points = []
        self.selected_region = None
        self.draw_start_pos = None
        self.temp_draw_box = None
        self.setCursor(Qt.CrossCursor)
        status = getattr(self._editor, 'status_label', None)
        if status is not None:
            status.setText(tr("添加区域（矩形）"))
        self.update()

    def _toggle_fix_regions(self):
        """固定/取消固定**当前模式**那把锁（两份各记各的）。"""
        new_state = not self._active_region_fixed()
        if self._is_paste_mode():
            self._editor.paste_region_fixed = new_state
        else:
            self._editor.region_fixed = new_state
        if new_state:
            self.is_drawing_region = False
            self.is_drawing_region_polygon = False
            self.temp_region_points = []
            self.selected_region = None
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def _clear_regions(self):
        if self._active_region_fixed():
            return
        if not self._active_regions():
            return
        reply = dialog_helpers.question(
            self._editor,
            tr("确认删除"),
            tr("确定要清空所有区域吗？"),
            dialog_helpers.QMessageBox.Yes | dialog_helpers.QMessageBox.No,
            dialog_helpers.QMessageBox.No,
        )
        if reply != dialog_helpers.QMessageBox.Yes:
            return
        regions = self._active_regions()
        regions.clear()
        self.selected_region = None
        self.is_drawing_region = False
        self.is_drawing_region_polygon = False
        self.temp_region_points = []
        self.update()

    def _delete_region(self, index):
        if self._active_region_fixed():
            return
        regions = self._active_regions()
        if not regions or not (0 <= index < len(regions)):
            return
        regions.pop(index)
        self.selected_region = None
        self.update()

    def _find_detection_box_at(self, mouse_pos):
        """查找鼠标位置下的检测框"""
        background_rect = self.get_background_rect()
        if not background_rect:
            return None
        from PySide6.QtCore import QRectF
        from ..core.config import DETECTION_BOX_CONFIG
        from ..engine.shape_io import point_in_polygon
        mx, my = mouse_pos.x(), mouse_pos.y()
        eps = DETECTION_BOX_CONFIG['resize_handle_size']
        for i, box in enumerate(self._editor.detection_boxes):
            if box.get("shape_type") == "point" and box.get("points"):
                cx = box["points"][0][0] * self.background_scale + background_rect.left()
                cy = box["points"][0][1] * self.background_scale + background_rect.top()
                if (mx - cx) ** 2 + (my - cy) ** 2 <= eps * eps:
                    return i
                continue
            if box.get("shape_type") in ("polygon", "rotation") and box.get("points"):
                canvas_pts = [
                    [
                        p[0] * self.background_scale + background_rect.left(),
                        p[1] * self.background_scale + background_rect.top(),
                    ]
                    for p in box["points"]
                ]
                if point_in_polygon(mx, my, canvas_pts):
                    return i
                continue
            box_x = box["x"] * self.background_scale + background_rect.left()
            box_y = box["y"] * self.background_scale + background_rect.top()
            box_w = box["width"] * self.background_scale
            box_h = box["height"] * self.background_scale
            if QRectF(box_x, box_y, box_w, box_h).contains(mouse_pos):
                return i
        return None

    def _collect_dataset_labels(self):
        """All labels known for the current background dataset (for box context menu)."""
        labels = set()
        bg = getattr(self._editor, 'background_dataset_labels', None) or set()
        if isinstance(bg, (set, list, tuple)):
            labels.update(x for x in bg if isinstance(x, str) and x.strip())
        gl = getattr(self._editor, 'global_labels', None) or set()
        if isinstance(gl, (set, list, tuple)):
            labels.update(x for x in gl if isinstance(x, str) and x.strip())
        for boxes in (getattr(self._editor, 'detection_boxes_dict', {}) or {}).values():
            for box in boxes or []:
                lab = box.get('label') if isinstance(box, dict) else None
                if isinstance(lab, str) and lab.strip():
                    labels.add(lab)
        for box in getattr(self._editor, 'detection_boxes', []) or []:
            lab = box.get('label') if isinstance(box, dict) else None
            if isinstance(lab, str) and lab.strip():
                labels.add(lab)
        label_list = getattr(self._editor, 'label_list', None)
        if label_list is not None:
            for i in range(label_list.count()):
                item = label_list.item(i)
                if item is None:
                    continue
                pure = extract_label_name(item.text())
                if pure:
                    labels.add(pure)
        return sorted(labels, key=lambda x: x.casefold())

    def _polygon_vertex_at(self, mouse_pos):
        finder = getattr(self, "_box_handle_at_pos", None)
        if finder is None:
            return None
        for i, box in enumerate(getattr(self._editor, "detection_boxes", []) or []):
            if box.get("shape_type") != "polygon" or not box.get("points"):
                continue
            handle = finder(mouse_pos, i)
            if isinstance(handle, str) and handle.startswith("v"):
                return i, int(handle[1:])
        return None

    def _show_polygon_vertex_menu(self, box_index, vertex_index, mouse_pos):
        menu = self._rounded_menu(self)
        action = QAction(tr("删除顶点"), self)
        box = self._editor.detection_boxes[box_index]
        action.setEnabled(len(box.get("points") or []) > 3)
        action.triggered.connect(
            lambda checked=False, idx=box_index, v=vertex_index: self._delete_polygon_vertex(idx, v)
        )
        menu.addAction(action)
        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _show_box_label_menu(self, box_index, mouse_pos):
        """检测框右键修改标签菜单"""
        menu = self._rounded_menu(self)

        current_label = self._editor.detection_boxes[box_index].get("label", "")

        modify_action = QAction(tr("修改标签"), self)
        modify_action.triggered.connect(
            lambda checked, idx=box_index: self._modify_box_label(idx)
        )
        menu.addAction(modify_action)

        remove_action = QAction(tr("移除"), self)
        remove_action.triggered.connect(
            lambda checked: self._editor._delete_selected_box()
        )
        menu.addAction(remove_action)

        box = self._editor.detection_boxes[box_index]
        shape_type = box.get("shape_type") or "rectangle"
        if shape_type == "rotation":
            reset_action = QAction(tr("重置旋转"), self)
            reset_action.triggered.connect(
                lambda checked, idx=box_index: self._reset_box_rotation(idx)
            )
            menu.addAction(reset_action)
            to_hbb_action = QAction(tr("转为HBB"), self)
            to_hbb_action.triggered.connect(
                lambda checked, idx=box_index: self._rotation_to_rectangle(idx)
            )
            menu.addAction(to_hbb_action)
        elif shape_type == "rectangle":
            to_obb_action = QAction(tr("转为OBB"), self)
            to_obb_action.triggered.connect(
                lambda checked, idx=box_index: self._rectangle_to_rotation(idx)
            )
            menu.addAction(to_obb_action)

        menu.addSeparator()

        for label in self._collect_dataset_labels():
            if label == current_label:
                continue
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, l=label, idx=box_index: self._change_box_label(idx, l)
            )
            menu.addAction(action)

        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _modify_box_label(self, box_index):
        """修改**这一个**检测框的标签（不动同名的其它框）。

        用户明确要求：「画布上右键修改标签怎么是改全部？我只是改一个啊」。
        这里此前直接调 `rename_detection_label`，那是**按标签名全局改名**——
        右键一个框会把当前图乃至整个数据集里所有同名框一起改掉，用户想只改
        一个也做不到。

        现在只改选中的那一个框，复用 `_change_box_label()`——那条路径（右键
        菜单里直接点已有标签）本来就是「只改一个」，颜色迁移、统计缓存
        delta=1、写盘都齐了，不再另写一套。分组（`group_id`）它不管，这里
        单独处理。

        注意：如果用户想给**整个类别**改名，入口是背景图标签列表的
        「修改标签」（`LabelManager.modify_label`，stats 模式），那里才是
        按名字全改的语义。
        """
        if self._editor._is_delete_view:
            return
        if not (0 <= box_index < len(self._editor.detection_boxes)):
            return
        current_label = self._editor.detection_boxes[box_index].get("label", "")
        anchor = None
        try:
            from PySide6.QtGui import QCursor
            anchor = QCursor.pos()
        except Exception:
            anchor = None
        labels = self._collect_dataset_labels()
        if current_label and current_label not in labels:
            labels = [current_label] + list(labels)
        from ..ui.dialogs import LabelSelectionDialog
        current_group = self._editor.detection_boxes[box_index].get("group_id")
        result = LabelSelectionDialog.select_label(
            self, labels, title="修改标签", initial_text=current_label,
            anchor_pos=anchor, show_group=True, current_group_id=current_group,
        )
        if isinstance(result, tuple):
            new_label, new_group = result
        else:
            new_label, new_group = result, None
        if not new_label or not str(new_label).strip():
            return
        new_label = str(new_label).strip()

        group_changed = self._editor.detection_boxes[box_index].get("group_id") != new_group
        label_changed = new_label != current_label

        if label_changed:
            # 只改这一个框。`_change_box_label` 内部会同步 global_labels、
            # 迁移颜色、按 delta=1 修正统计缓存并写盘。
            self._change_box_label(box_index, new_label)

        if group_changed:
            self._editor.detection_boxes[box_index]["group_id"] = new_group
            if self._editor.current_background_index >= 0:
                self._editor.detection_boxes_dict[self._editor.current_background_index] = \
                    list(self._editor.detection_boxes)
            # 只改了分组也要落盘，否则切图就丢
            lm = getattr(self._editor, 'label_manager', None)
            if lm is not None and hasattr(lm, '_save_detection_json_for_index'):
                lm._save_detection_json_for_index(self._editor.current_background_index)
            self._editor.update_label_list()
            self.update()

    def _reset_box_rotation(self, box_index):
        """将 OBB 旋转角度归零，恢复水平矩形。"""
        if not (0 <= box_index < len(self._editor.detection_boxes)):
            return
        box = self._editor.detection_boxes[box_index]
        if box.get("shape_type") != "rotation":
            return
        from ..engine.shape_io import rect_to_rotation_points
        self._editor.save_undo_state()
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        box["points"] = rect_to_rotation_points(x, y, w, h)
        if self._editor.current_background_index >= 0:
            self._editor.detection_boxes_dict[self._editor.current_background_index] = \
                list(self._editor.detection_boxes)
        self._persist_boxes()
        self.update()

    def _rotation_to_rectangle(self, box_index):
        """将 OBB 转为普通矩形（丢弃旋转，保留外接框）。"""
        if not (0 <= box_index < len(self._editor.detection_boxes)):
            return
        box = self._editor.detection_boxes[box_index]
        if box.get("shape_type") != "rotation":
            return
        self._editor.save_undo_state()
        box["shape_type"] = "rectangle"
        box.pop("points", None)
        if self._editor.current_background_index >= 0:
            self._editor.detection_boxes_dict[self._editor.current_background_index] = \
                list(self._editor.detection_boxes)
        self._persist_boxes()
        self._editor.update_label_list()
        self.update()

    def _rectangle_to_rotation(self, box_index):
        """将普通矩形转为 OBB（角度 0，points 为四角）。"""
        if not (0 <= box_index < len(self._editor.detection_boxes)):
            return
        box = self._editor.detection_boxes[box_index]
        if (box.get("shape_type") or "rectangle") != "rectangle":
            return
        from ..engine.shape_io import rect_to_rotation_points
        self._editor.save_undo_state()
        box["shape_type"] = "rotation"
        box["points"] = rect_to_rotation_points(
            box["x"], box["y"], box["width"], box["height"]
        )
        if self._editor.current_background_index >= 0:
            self._editor.detection_boxes_dict[self._editor.current_background_index] = \
                list(self._editor.detection_boxes)
        self._persist_boxes()
        self._editor.update_label_list()
        self.update()

    def _persist_boxes(self):
        lm = getattr(self._editor, 'label_manager', None)
        if lm is not None and hasattr(lm, '_save_detection_json_for_index'):
            lm._save_detection_json_for_index(self._editor.current_background_index)

    def _change_box_label(self, box_index, new_label):
        """切换单个检测框标签到数据集中已有标签。"""
        if self._editor._is_delete_view:
            return
        if not (0 <= box_index < len(self._editor.detection_boxes)):
            return
        old_label = self._editor.detection_boxes[box_index].get("label", "")
        new_label = (new_label or "").strip()
        if not new_label or new_label == old_label:
            return
        lm = getattr(self._editor, 'label_manager', None)
        target_exists = True
        if lm is not None and hasattr(lm, '_label_already_exists'):
            target_exists = lm._label_already_exists(new_label)
        self._editor.detection_boxes[box_index]["label"] = new_label
        if self._editor.current_background_index >= 0:
            self._editor.detection_boxes_dict[self._editor.current_background_index] = \
                list(self._editor.detection_boxes)
        gl = getattr(self._editor, 'global_labels', None)
        if isinstance(gl, set):
            gl.add(new_label)
        still_used = any(
            box.get("label") == old_label
            for boxes in (getattr(self._editor, 'detection_boxes_dict', {}) or {}).values()
            for box in (boxes or [])
        ) or any(
            box.get("label") == old_label
            for box in (getattr(self._editor, 'detection_boxes', []) or [])
        )
        if isinstance(gl, set) and not still_used and old_label in gl:
            gl.discard(old_label)
        if lm is not None:
            if hasattr(lm, '_apply_rename_color'):
                lm._apply_rename_color(
                    old_label, new_label, target_exists, move_if_unused=not still_used,
                )
            if hasattr(lm, '_sync_cached_stats_label_rename'):
                lm._sync_cached_stats_label_rename(old_label, new_label, delta=1)
            if hasattr(lm, '_save_detection_json_for_index'):
                lm._save_detection_json_for_index(self._editor.current_background_index)
        elif hasattr(self._editor, 'get_label_color'):
            self._editor.get_label_color(new_label)
        self._editor.update_label_list()
        self.update()

    def _show_restore_context_menu(self, mouse_pos):
        menu = self._rounded_menu(self)

        restore_action = QAction(tr("恢复到工作路径"), self)
        restore_action.triggered.connect(self._restore_current_background)
        menu.addAction(restore_action)

        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _restore_current_background(self):
        """从移除路径恢复当前图片到工作路径"""
        import shutil
        if not self._editor.background_images:
            return
        if not hasattr(self._editor, '_delete_files') or not self._editor._delete_files:
            return
        idx = self._editor.background_list.currentRow()
        if idx < 0 or idx >= len(self._editor._delete_files):
            return
        file_path = self._editor._delete_files[idx]
        delete_dir = os.path.dirname(file_path)
        original_dir = os.path.dirname(delete_dir)

        dst = os.path.join(original_dir, os.path.basename(file_path))
        shutil.move(file_path, dst)
        json_path = os.path.splitext(file_path)[0] + '.json'
        if os.path.isfile(json_path):
            from ..core import json_trace
            json_trace.trace('canvas_menu 移动到原目录', json_path)
            shutil.move(json_path, os.path.splitext(dst)[0] + '.json')

        insert_idx = len(self._editor.background_images)
        self._editor.background_images.append(dst)
        self._editor.canvas_items_dict[insert_idx] = []
        self._editor.detection_boxes_dict[insert_idx] = []

        self._editor._show_delete_view()

    def _show_background_context_menu(self, mouse_pos):
        menu = self._rounded_menu(self)

        delete_labels_action = QAction(tr("删除标签文件"), self)
        delete_labels_action.triggered.connect(self._delete_current_label_file)
        menu.addAction(delete_labels_action)

        remove_action = QAction(tr("移除图片"), self)
        remove_action.triggered.connect(self._remove_current_background)
        menu.addAction(remove_action)

        # 区域那一整段**只在贴图模式**给（用户第 10 轮）：
        # 「推理是推理的操作，贴图是贴图的操作」——推理区域的增删改全部走
        # 「推理范围」面板（`ScopePanel`），标注模式的右键不该还留一份入口，
        # 否则同一个推理区域有两个地方能改，用户分不清哪份才算数。
        # 贴图模式下的这段作用于贴图那份（`_active_regions()`）。
        if self._is_paste_mode():
            menu.addSeparator()
            self._add_region_actions(menu, self._active_region_fixed())

        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def _delete_current_label_file(self):
        """Delete the current image's sidecar LabelMe JSON and clear in-memory boxes."""
        if self._editor._is_delete_view:
            return
        idx = self._editor.current_background_index
        if idx < 0 or idx >= len(self._editor.background_images):
            return
        file_path = self._editor.background_images[idx]
        json_path = os.path.splitext(file_path)[0] + ".json"
        has_boxes = bool(self._editor.detection_boxes)
        has_json = os.path.isfile(json_path)
        if not has_boxes and not has_json:
            return

        reply = dialog_helpers.question(
            self._editor,
            tr("确认删除"),
            f"{tr('确定要删除当前图片的标签文件吗？')}",
            dialog_helpers.QMessageBox.Yes | dialog_helpers.QMessageBox.No,
            dialog_helpers.QMessageBox.No,
        )
        if reply != dialog_helpers.QMessageBox.Yes:
            return

        self._editor.detection_boxes = []
        self._editor.detection_boxes_dict[idx] = []
        self.selected_box = None
        self.selected_boxes = []
        if has_json:
            from ..core import json_trace
            json_trace.trace('canvas_menu._delete_current_label_file 即将删',
                             json_path)
            try:
                os.remove(json_path)
            except OSError:
                from ..core.exception_hook import _write_log
                _write_log(f"删除标签文件失败: {json_path}")
            json_trace.trace_delete(
                'canvas_menu._delete_current_label_file', json_path,
                survived=os.path.exists(json_path))

        refresh = getattr(self._editor, "_refresh_background_item_status", None)
        if callable(refresh):
            refresh(idx, file_path)
        self._editor.update_label_list()
        self.update()
        notify = getattr(self._editor, '_notify_lint_boxes_changed', None)
        if callable(notify):
            notify(idx)

        # 删除后自动跳转到下一张
        if self._editor.background_images:
            new_idx = min(idx, len(self._editor.background_images) - 1)
            self._editor.switch_background_to_index(new_idx)
            row = self._editor._find_bg_list_row_for_index(new_idx)
            if row is not None:
                self._editor.background_list.setCurrentRow(row)

    def _remove_current_background(self):
        """从画布移除当前背景图（移到删除文件夹）"""
        import shutil
        if self._editor._is_delete_view:
            return
        idx = self._editor.current_background_index
        if idx < 0 or idx >= len(self._editor.background_images):
            return
        # 移除后索引重排，质检弹窗继续指向旧索引会跳错图
        close_lint = getattr(self._editor, '_close_lint_dialog', None)
        if callable(close_lint):
            close_lint()
        file_path = self._editor.background_images[idx]

        delete_dir = os.path.join(os.path.dirname(file_path), '_delete_')
        os.makedirs(delete_dir, exist_ok=True)
        shutil.move(file_path, os.path.join(delete_dir, os.path.basename(file_path)))
        json_path = os.path.splitext(file_path)[0] + '.json'
        if os.path.isfile(json_path):
            from ..core import json_trace
            json_trace.trace('canvas_menu 移入 _delete_ 目录', json_path)
            shutil.move(json_path, os.path.join(delete_dir, os.path.basename(json_path)))

        self._editor.background_images.pop(idx)
        self._editor.canvas_items_dict = self._remove_index_from_mapping(
            self._editor.canvas_items_dict, idx
        )
        self._editor.detection_boxes_dict = self._remove_index_from_mapping(
            self._editor.detection_boxes_dict, idx
        )

        new_idx = min(idx, len(self._editor.background_images) - 1)
        if self._editor.background_images:
            self._editor.current_background_index = new_idx
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(self._editor.background_images[new_idx])
            if not pixmap.isNull():
                self._editor.current_background = pixmap
                self._editor.detection_boxes = self._editor.detection_boxes_dict.get(new_idx, []).copy()
                self._editor.canvas_items = self._editor.canvas_items_dict.get(new_idx, [])
                self.reset_view()
                self.update()
        else:
            self._editor.current_background = None
            self._editor.current_background_index = -1
            self.update()

        self._editor.update_file_count()
        self._editor.update_label_list()
        self._editor._show_work_view()

    def _show_paste_context_menu(self, item_index, mouse_pos):
        menu = self._rounded_menu(self)

        remove_action = QAction(tr("移除"), self)
        remove_action.triggered.connect(
            lambda checked, idx=item_index: self._remove_paste_item(idx)
        )
        menu.addAction(remove_action)

        menu.addSeparator()

        label_items = []
        for i in range(self._editor.paste_label_list.count()):
            label = self._editor.paste_label_list.item(i).text()
            pure_label = extract_label_name(label)
            label_items.append(pure_label)

        for label in label_items:
            action = QAction(label, self)
            action.triggered.connect(
                lambda checked, l=label, idx=item_index: self.change_item_label(idx, l)
            )
            menu.addAction(action)

        menu.addSeparator()

        new_label_action = QAction(tr("增加标签"), self)
        new_label_action.triggered.connect(
            lambda checked, idx=item_index: self.add_new_label(idx)
        )
        menu.addAction(new_label_action)

        menu.exec_(QPoint(self.mapToGlobal(mouse_pos)))

    def change_item_label(self, item_index, new_label):
        if self._editor._is_delete_view:
            return
        if 0 <= item_index < len(self._editor.canvas_items):
            pixmap, rect, _ = self._editor.canvas_items[item_index]
            self._editor.canvas_items[item_index] = (pixmap, rect, new_label)
            self.update()

    def add_new_label(self, item_index):
        if self._editor._is_delete_view:
            return
        new_label, ok = dialog_helpers.get_text(
            self, "增加标签", "请输入新的标签名称:"
        )
        if ok and new_label.strip():
            new_label = new_label.strip()
            self._editor.paste_label_list.addItem(new_label)

            if 0 <= item_index < len(self._editor.canvas_items):
                pixmap, rect, _ = self._editor.canvas_items[item_index]
                self._editor.canvas_items[item_index] = (pixmap, rect, new_label)
                self.update()

    def _remove_paste_item(self, item_index):
        """移除贴图项"""
        if self._editor._is_delete_view:
            return
        if 0 <= item_index < len(self._editor.canvas_items):
            self._editor.canvas_items.pop(item_index)
            self._editor.selected_item = None
            self.update()
