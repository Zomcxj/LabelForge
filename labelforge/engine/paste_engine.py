"""
贴图引擎混入 - 负责贴图的添加、删除、随机/批量放置
"""
import os
import random
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QRectF

from ..core.config import RANDOM_POSITION_CONFIG, PASTE_ITEM_CONFIG
from ..core.utils import extract_label_name, calculate_iou
from ..ui.i18n import t as tr


OVERLAP_OFFSET_STEP = 3


class PasteEngineMixin:
    """贴图引擎混入类 - 管理贴图的添加、删除、随机和批量放置"""

    def _offset_overlapping_paste_group(self, pasted_group):
        """把整组框整体右移下移，直到不再与已有框完全重合。

        组里每项是 `(rect, label[, box])`：`box` 是原始检测框字典（可选，旧调用
        方只给两元组）。多出来的尾随元素原样带着走，否则粘贴时置信度、形状类型
        这些字段会在这一步被丢掉。
        """
        existing_boxes = {
            (box['x'], box['y'], box['width'], box['height'])
            for box in getattr(self, 'detection_boxes', [])
        }
        if not existing_boxes:
            return pasted_group

        adjusted_group = []
        for item in pasted_group:
            rect, label = item[0], item[1]
            adjusted_group.append(
                (QRectF(rect.x(), rect.y(), rect.width(), rect.height()), label)
                + tuple(item[2:]))
        while True:
            group_boxes = [
                (item[0].x(), item[0].y(), item[0].width(), item[0].height())
                for item in adjusted_group
            ]
            if not any(box in existing_boxes for box in group_boxes):
                return adjusted_group
            adjusted_group = [
                (QRectF(item[0].x() + OVERLAP_OFFSET_STEP,
                        item[0].y() + OVERLAP_OFFSET_STEP,
                        item[0].width(),
                        item[0].height()),)
                + tuple(item[1:])
                for item in adjusted_group
            ]

    def add_small_to_canvas(self, item):
        """添加贴图到画布"""
        if self._is_delete_view or self._busy:
            return
        self.save_undo_state()
        index = item.data(Qt.UserRole)
        pixmap = self.small_images[index][1]

        if self.current_background is None:
            return

        base_scale_factor = PASTE_ITEM_CONFIG['base_scale_factor'] * self.canvas.background_scale
        scale_factor = max(0.1, min(base_scale_factor, 2.0))

        width = pixmap.width() * scale_factor
        height = pixmap.height() * scale_factor

        if width <= 0 or height <= 0:
            return

        min_edge = self.min_size_spin.value()
        aspect_ratio = width / height
        if width < height:
            width = min_edge
            height = width / aspect_ratio
        else:
            height = min_edge
            width = height * aspect_ratio

        bg_width = self.current_background.width()
        bg_height = self.current_background.height()
        x = max(0, (bg_width - width) / 2)
        y = max(0, (bg_height - height) / 2)

        rect = QRectF(x, y, width, height)
        paste_label = self._get_paste_label(index)

        self.canvas_items.append((pixmap, rect, paste_label))
        self.canvas.update()

    def _get_paste_label(self, index):
        """获取贴图标签"""
        paste_label = "paste"

        if hasattr(self, 'auto_label_checkbox') and self.auto_label_checkbox.isChecked():
            image_path = self.small_images[index][0]
            image_name = os.path.basename(image_path)
            label_part = image_name.split('_')[0]
            label_part = os.path.splitext(label_part)[0]

            if label_part:
                paste_label = label_part

                label_exists = False
                for i in range(self.paste_label_list.count()):
                    existing_label = self.paste_label_list.item(i).text()
                    pure_label = extract_label_name(existing_label)
                    if pure_label == paste_label:
                        label_exists = True
                        break

                if not label_exists:
                    self.paste_label_list.addItem(paste_label)

        elif hasattr(self, 'paste_label_list'):
            if self.paste_label_list.count() == 0:
                self.paste_label_list.addItem("paste")
            else:
                selected_items = self.paste_label_list.selectedItems()
                if selected_items:
                    paste_label = selected_items[0].text()
                else:
                    paste_label = self.paste_label_list.item(0).text()

        return paste_label

    def clear_canvas(self):
        """清空画布"""
        if self._is_delete_view or self._busy:
            return
        self.save_undo_state()
        self.canvas_items.clear()
        if self.current_background_index >= 0:
            self.canvas_items_dict[self.current_background_index] = self.canvas_items.copy()
        self.selected_item = None
        self.canvas.update()

    def _sample_center_in_regions(self, regions, new_width, new_height, bg_w, bg_h, rng):
        """在语义区域并集内采样一个贴图中心点，保证贴图完整落在区域内。

        矩形区域：解析求中心区间 [区域左+半宽, 区域右-半宽]（∩ 背景范围）。
        多边形区域：在 bbox 内拒绝采样，要求中心与贴图四角都在多边形内。
        全部失败时返回 None（调用方跳过该次贴图）。
        """
        from ..core.config import REGION_CONFIG
        order = list(range(len(regions)))
        rng.shuffle(order)
        max_retries = REGION_CONFIG.get('polygon_max_retries', 200)

        for idx in order:
            region = regions[idx]
            if (region.get('shape_type') or 'rectangle') == 'polygon' and region.get('points'):
                sample = self._sample_in_polygon_region(
                    region, new_width, new_height, bg_w, bg_h, rng, max_retries
                )
                if sample is not None:
                    return sample
                continue

            half_w = new_width / 2
            half_h = new_height / 2
            rx1 = max(region['x'] + half_w, half_w)
            rx2 = min(region['x'] + region['width'] - half_w, bg_w - half_w)
            ry1 = max(region['y'] + half_h, half_h)
            ry2 = min(region['y'] + region['height'] - half_h, bg_h - half_h)
            if rx1 > rx2 or ry1 > ry2:
                continue
            return rng.uniform(rx1, rx2), rng.uniform(ry1, ry2)

        return None

    def _sample_in_polygon_region(self, region, new_width, new_height, bg_w, bg_h,
                                  rng, max_retries):
        """多边形区域内拒绝采样：中心与贴图四角都必须在多边形内。"""
        from .shape_io import point_in_polygon
        points = region['points']
        half_w = new_width / 2
        half_h = new_height / 2

        px1 = max(region['x'] + half_w, half_w)
        px2 = min(region['x'] + region['width'] - half_w, bg_w - half_w)
        py1 = max(region['y'] + half_h, half_h)
        py2 = min(region['y'] + region['height'] - half_h, bg_h - half_h)
        if px1 > px2 or py1 > py2:
            return None

        for _ in range(max_retries):
            cx = rng.uniform(px1, px2)
            cy = rng.uniform(py1, py2)
            corners = (
                (cx - half_w, cy - half_h),
                (cx + half_w, cy - half_h),
                (cx - half_w, cy + half_h),
                (cx + half_w, cy + half_h),
            )
            if not point_in_polygon(cx, cy, points):
                continue
            if all(point_in_polygon(x, y, points) for x, y in corners):
                return cx, cy
        return None

    def random_paste_images(self, background=None, detection_boxes=None, seed=None):
        """随机贴图 - 中心点避让算法

        seed 给定时结果可复现；不传时沿用全局 random（保持既有行为）。
        """
        if self._is_delete_view or not self.small_images or not self.current_background:
            return

        rng = random.Random(seed)

        self.save_undo_state()
        self._validate_size_range()

        current_background = background if background else self.current_background
        current_detection_boxes = detection_boxes if detection_boxes else self.detection_boxes

        self.canvas_items.clear()

        det_boxes = []
        for box in current_detection_boxes:
            det_boxes.append((box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']))

        bg_w = current_background.width()
        bg_h = current_background.height()

        ml = RANDOM_POSITION_CONFIG['margin_left']
        mt = RANDOM_POSITION_CONFIG['margin_top']
        mr = RANDOM_POSITION_CONFIG['margin_right']

        num_paste = self.paste_count_spin.value()
        selected_indices = rng.choices(range(len(self.small_images)), k=num_paste)

        # 用**贴图**那份区域（用户第 10 轮把区域拆成两份：贴图落点看
        # `paste_region_boxes`，推理范围看 `region_boxes`）。以前这里读的是
        # `region_boxes`，等于贴图的落点约束就是推理范围——两类区域混在一起。
        regions = [
            r for r in (getattr(self, 'paste_region_boxes', None) or [])
            if r.get('width', 0) > 0 and r.get('height', 0) > 0
        ]

        pasted_boxes = []

        for idx in selected_indices:
            file_path, pixmap = self.small_images[idx]

            if pixmap.width() <= 0 or pixmap.height() <= 0:
                continue

            aspect_ratio = pixmap.width() / pixmap.height()
            min_size = self.min_size_spin.value()
            max_size = self.max_size_spin.value()
            target_size = rng.randint(min_size, max_size)

            if pixmap.width() > pixmap.height():
                new_width = target_size
                new_height = new_width / aspect_ratio
            else:
                new_height = target_size
                new_width = new_height * aspect_ratio

            if min(new_width, new_height) < min_size:
                if new_width < new_height:
                    new_width = min_size
                    new_height = new_width / aspect_ratio
                else:
                    new_height = min_size
                    new_width = new_height * aspect_ratio

            valid_position = False
            x, y = ml, mt

            place_w = bg_w - new_width - mr
            place_h = bg_h - new_height

            if not regions and (place_w <= ml or place_h <= mt):
                continue

            for _ in range(RANDOM_POSITION_CONFIG['max_retries']):
                if regions:
                    sample = self._sample_center_in_regions(
                        regions, new_width, new_height, bg_w, bg_h, rng
                    )
                    if sample is None:
                        # 所有区域都装不下该贴图（尺寸固定，重试无意义）
                        break
                    cx, cy = sample
                else:
                    cx = rng.uniform(ml + new_width / 2, place_w + new_width / 2)
                    cy = rng.uniform(mt + new_height / 2, place_h + new_height / 2)

                tx = cx - new_width / 2
                ty = cy - new_height / 2

                point_in_det = False
                for db in det_boxes:
                    if db[0] <= cx <= db[2] and db[1] <= cy <= db[3]:
                        point_in_det = True
                        break

                if point_in_det:
                    continue

                new_box = (tx, ty, tx + new_width, ty + new_height)

                overlaps_pasted = False
                for pb in pasted_boxes:
                    if calculate_iou(new_box, pb) > RANDOM_POSITION_CONFIG['overlap_iou_pasted']:
                        overlaps_pasted = True
                        break

                if overlaps_pasted:
                    continue

                x, y = tx, ty
                valid_position = True
                break

            if valid_position:
                rect = QRectF(x, y, new_width, new_height)
                paste_label = self._get_paste_label(idx)
                self.canvas_items.append((pixmap, rect, paste_label))
                pasted_boxes.append((x, y, x + new_width, y + new_height))

        if not background:
            self.canvas.update()

        if self.current_background_index >= 0:
            self.canvas_items_dict[self.current_background_index] = self.canvas_items.copy()

    def batch_paste_images(self):
        """从当前图片开始依次处理所有图片，应用随机贴图，然后返回当前图片"""
        if self._busy:
            return

        if not self.small_images:
            return

        if not self.background_images:
            return

        self._busy = True

        total_count = len(self.background_images)
        start_index = self.current_background_index if self.current_background_index >= 0 else 0
        process_count = total_count - start_index

        from ..ui.dialogs import ProgressDialogFactory
        progress_dialog = ProgressDialogFactory.create_progress_dialog(
            self, tr("一键贴图进度"), tr("正在一键贴图..."), process_count
        )
        progress_dialog.show()

        original_index = self.current_background_index
        original_background = self.current_background
        original_detection_boxes = self.detection_boxes.copy()
        original_canvas_items = self.canvas_items.copy()

        processed_count = 0

        for i in range(start_index, total_count):
            if progress_dialog.wasCanceled():
                break

            file_path = self.background_images[i]
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                if i in self.detection_boxes_dict and len(self.detection_boxes_dict[i]) > 0:
                    temp_detection_boxes = self.detection_boxes_dict[i].copy()
                else:
                    temp_detection_boxes = self.load_detection_boxes(file_path)
                    self.detection_boxes_dict[i] = temp_detection_boxes.copy()

                self.current_background_index = i
                self.canvas_items = self.canvas_items_dict.get(i, []).copy()

                self.random_paste_images(background=pixmap, detection_boxes=temp_detection_boxes)

                self.canvas_items_dict[i] = self.canvas_items.copy()

                processed_count += 1

                progress_dialog.setValue(processed_count)
                progress_dialog.setLabelText(f"{tr('正在处理第')} {i+1} {tr('张图片，共')} {total_count} {tr('张')}")

                QApplication.processEvents()

        progress_dialog.setValue(process_count)

        if original_index >= 0:
            self.current_background_index = original_index
            self.current_background = original_background
            self.detection_boxes = original_detection_boxes
            self.canvas_items = self.canvas_items_dict.get(original_index, []).copy()
            self.canvas.update()

        self._busy = False
