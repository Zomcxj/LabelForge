"""
保存管理器模块 - 处理贴图保存、JSON 生成等逻辑
"""

import os
import json
from typing import TYPE_CHECKING
from PySide6.QtCore import Signal, QObject, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPainter, QColor, QImage
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication
from ..core.config import LABELME_VERSION, DEFAULT_PREFIX
from ..core.utils import PathUtils, calculate_iou
from .paste_item import as_paste_item
from ..ui.i18n import t as tr

if TYPE_CHECKING:
    from ..core.editor_protocol import EditorProtocol


class _SyncTitleBarDialog(QDialog):
    """标题栏同步的对话框"""
    def __init__(self, msg_type, title, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self.setObjectName("syncDialog")

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 16, 20, 16)

        text_layout = QHBoxLayout()
        text_layout.setSpacing(8)

        icon_label = QLabel("⚠")
        icon_label.setStyleSheet("background: transparent; font-size: 20px; color: #FF9800;")
        icon_label.setFixedWidth(24)

        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setStyleSheet("background: transparent; font-size: 13px;")

        text_layout.addWidget(icon_label)
        text_layout.addWidget(text_label, 1)
        layout.addLayout(text_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        from ..ui import i18n
        tr = i18n.t
        ok_btn = QPushButton(tr("确定"))
        ok_btn.setObjectName("successBtn")
        ok_btn.setFixedWidth(80)
        from ..ui.theme import ThemeManager
        ok_btn.setStyleSheet(ThemeManager.get_dialog_button_style())
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def showEvent(self, event):
        super().showEvent(event)
        from ..ui.dialog_helpers import center_on_parent
        center_on_parent(self)
        self._sync()

    def _sync(self):
        from ..ui.dwm import set_titlebar_dark
        from ..ui.theme import ThemeManager
        is_dark = ThemeManager.get_mode().value == "dark"
        hwnd = int(self.winId())
        set_titlebar_dark(hwnd, is_dark)


def _show_messagebox(msg_type, parent, title, text):
    """显示同步标题栏颜色的对话框"""
    from ..ui import i18n
    tr = i18n.t
    dialog = _SyncTitleBarDialog(msg_type, tr(title), tr(text), parent)
    dialog.exec_()


class SaveManagerMixin:
    """写盘路径共用的贴图转换：贴图项 -> LabelMe shape、贴图项 -> 合成图。

    写盘路径（当前图保存、保存全部、sidecar）都从这里取：形状、融合参数与
    角度只有一套口径，否则同一张图换个入口保存就和画布对不上。
    """

    #: 泊松融合时贴图四周要留的边界像素（cv2.seamlessClone 需要 mask 外的
    #: 一圈参考像素；留不出来就退回普通合成）
    SEAMLESS_MARGIN = 8

    @staticmethod
    def _paste_item_to_shape(item):
        """贴图项 -> LabelMe shape。角度为 0 写矩形，否则写旋转框。"""
        from .shape_io import labelme_shape_from_box
        x, y, w, h = item.rect_tuple()
        box = {"label": item.label, "x": x, "y": y, "width": w, "height": h}
        if item.shape_type == "rotation":
            box["shape_type"] = "rotation"
            box["points"] = [list(p) for p in item.corners()]
        shape = labelme_shape_from_box(box)
        shape["flags"] = {"paste": True}
        return shape

    @staticmethod
    def _item_rgba(item):
        """贴图本体 -> (H, W, 4) uint8，按外接框尺寸取样。"""
        from .paste_fusion import pixmap_to_rgba
        x, y, w, h = item.rect_tuple()
        return pixmap_to_rgba(item.pixmap, max(1, int(round(w))),
                              max(1, int(round(h))))

    @classmethod
    def _draw_paste_item_on(cls, painter, item):
        """把一张贴图画到 painter 上；有角度时绕外接框中心旋转。"""
        x, y, w, h = item.rect_tuple()
        if not item.angle:
            painter.drawPixmap(QRectF(x, y, w, h).toRect(), item.pixmap)
            return
        painter.save()
        painter.translate(QPointF(x + w / 2.0, y + h / 2.0))
        painter.rotate(item.angle)
        painter.drawPixmap(
            QRectF(-w / 2.0, -h / 2.0, w, h).toRect(), item.pixmap)
        painter.restore()

    @classmethod
    def _draw_paste_items_on(cls, painter, items):
        """把贴图按外接框角度画到合成画布上。

        泊松融合要写回像素，只在 `_compose_pastes()` 里做——这里纯粹是绘制。
        """
        for raw in items:
            cls._draw_paste_item_on(painter, as_paste_item(raw))

    @classmethod
    def _compose_pastes(cls, image, items):
        """在 QImage 上合成所有贴图，必要时做泊松融合（A5，仅保存时生效）。

        普通项目走 painter 绘制；泊松要写回像素，所以先把 painter 收尾、
        混合完再重开，保证与列表顺序一致。
        """
        from ..core.config import PASTE_FUSION_CONFIG
        seamless = bool(PASTE_FUSION_CONFIG.get('seamless'))
        painter = QPainter(image)
        try:
            for raw in items:
                item = as_paste_item(raw)
                if seamless:
                    painter.end()
                    if cls._seamless_blend_into(image, item):
                        painter = QPainter(image)
                        continue
                    painter = QPainter(image)
                cls._draw_paste_item_on(painter, item)
        finally:
            painter.end()

    @classmethod
    def _seamless_blend_into(cls, image, item):
        """把一张贴图用泊松融合写进 image；返回是否已经画过这张贴图。

        返回 False 表示「什么都没画」，调用方仍需要走普通绘制；返回 True 表示
        贴图已经落在 image 上（混合成功，或混合失败后按普通合成落笔）。

        只有贴图本体参与混合，透明像素不进 mask。
        旋转过的贴图不做泊松（seamlessClone 只支持轴对齐落笔）。
        """
        import cv2
        import numpy as np
        from .paste_fusion import bgr_to_qimage, qimage_to_bgr
        if item.angle:
            return False
        x, y, w, h = item.rect_tuple()
        width = max(1, int(round(w)))
        height = max(1, int(round(h)))
        core = cls._item_rgba(item)
        if core.shape[:2] != (height, width):
            return False
        mask = np.where(core[:, :, 3] > 0, 255, 0).astype(np.uint8)
        if not mask.any():
            return False
        left, top = int(round(x)), int(round(y))
        margin = cls.SEAMLESS_MARGIN
        rx = max(0, left - margin)
        ry = max(0, top - margin)
        rw = min(image.width() - rx, width + 2 * margin)
        rh = min(image.height() - ry, height + 2 * margin)
        if rw < width + 2 or rh < height + 2:
            # 贴到图边上了，留不出 mask 外的参考像素
            return False

        painter = QPainter(image)
        try:
            cls._draw_paste_item_on(painter, item)
        finally:
            painter.end()

        dst = qimage_to_bgr(image.copy(rx, ry, rw, rh))
        src = np.ascontiguousarray(core[:, :, 2::-1])      # RGBA -> BGR
        center = (left - rx + width // 2, top - ry + height // 2)
        try:
            blended = cv2.seamlessClone(src, dst, mask, center,
                                        cv2.NORMAL_CLONE)
        except cv2.error:
            # 区域退化（mask 贴边、尺寸不合法等）：保留已经画好的普通合成
            return True
        painter = QPainter(image)
        try:
            painter.drawImage(rx, ry, bgr_to_qimage(blended))
        finally:
            painter.end()
        return True


class SaveManager(SaveManagerMixin, QObject):
    """保存管理器 - 管理贴图的保存操作"""

    # 信号：保存完成后通知编辑器刷新 UI
    save_completed = Signal()
    label_list_changed = Signal()

    def __init__(self, editor: "EditorProtocol", parent=None):
        """
        :param editor: 实现 EditorProtocol 的编辑器实例
        """
        super().__init__(parent)
        self.editor = editor
    
    def get_save_info(self):
        """获取保存信息"""
        if self.editor.current_background is None or self.editor.current_background_index < 0:
            return None
        
        original_file_path = self.editor.background_images[self.editor.current_background_index]
        original_file_name = os.path.basename(original_file_path)
        
        output_dir = PathUtils.get_output_dir(original_file_path)
        
        prefix = ""
        if hasattr(self.editor, 'prefix_checkbox') and self.editor.prefix_checkbox.isChecked():
            prefix = self.editor.prefix_input.text().strip()
            if not prefix:
                prefix = DEFAULT_PREFIX
        
        base_name = f"{prefix}_{original_file_name}" if prefix else original_file_name
        file_path = os.path.join(output_dir, base_name)
        
        return (file_path, base_name, prefix)
    
    def _get_save_info_or_none(self):
        if (self.editor._is_delete_view
                or getattr(self.editor, '_image_tools_active', False)
                or self.editor.current_background is None
                or not self.editor.canvas_items):
            return None
        return self.get_save_info()

    def auto_save_background(self):
        """自动保存合成图（B）"""
        save_info = self._get_save_info_or_none()
        if not save_info:
            return
        file_path, _, _ = save_info
        result = self._new_composite(self.editor.current_background)
        painter = QPainter(result)
        try:
            painter.drawPixmap(0, 0, self.editor.current_background)
        finally:
            painter.end()
        self._compose_pastes(result, self.editor.canvas_items)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not result.save(file_path):
            from ..core.exception_hook import _write_log
            _write_log(f"自动保存B失败: {file_path}")

    @staticmethod
    def _new_composite(background):
        """建一张白底 QImage：合成图在这里画，泊松融合才好写回像素。

        用 RGB32（不带 alpha）与旧的 `QPixmap` 合成图保持同一落盘格式：
        成品是白底不透明的图，多出的 alpha 通道只会让 PNG 变大。
        """
        image = QImage(background.size(), QImage.Format_RGB32)
        image.fill(QColor(255, 255, 255))
        return image

    def auto_save_project(self):
        """自动保存 JSON（P）"""
        save_info = self._get_save_info_or_none()
        if not save_info:
            return
        file_path, base_name, prefix = save_info
        self.save_json(file_path, base_name, prefix)
    
    def save_canvas(self):
        """保存当前画布"""
        from ..ui.dialogs import SaveTipDialog

        if self.editor._is_delete_view:
            return
        
        if self.editor.current_background is None:
            _show_messagebox("warning", self.editor, tr("警告"), tr("请先选择背景图片"))
            return
        
        save_info = self.get_save_info()
        if not save_info:
            _show_messagebox("warning", self.editor, tr("警告"), tr("无法获取保存信息"))
            return
        
        file_path, base_name, prefix = save_info
        
        result = self._new_composite(self.editor.current_background)
        painter = QPainter(result)
        try:
            painter.drawPixmap(0, 0, self.editor.current_background)
        finally:
            painter.end()
        
        self._compose_pastes(result, self.editor.canvas_items)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        save_success = result.save(file_path)
        
        if save_success:
            self.save_json(file_path, base_name, prefix)
        
        SaveTipDialog.show_save_tip(self.editor, file_path, save_success and os.path.exists(file_path))

    def save_current_json(self):
        """仅保存当前图的标注 JSON，用于切图和关闭窗口时的保底写入。

        Annotate mode: save when there are boxes, or when all boxes were
        cleared but a sidecar JSON already exists (persist empty / 空标签).
        Never create a brand-new empty JSON for never-annotated images.

        画布被清空时（默认行为，可在「选项」里关掉）**删除**已存在的
        sidecar JSON：留着空 json 会让「这张图没标过」变成「这张图标过但
        一个标签都没有」，下游训练/统计都得额外分辨这两种情况。

        判空以 **json 原文的 `shapes`** 为准，不看内存里的 `detection_boxes`。
        内存状态会漏判两类真有内容的文件，而这是**删用户数据**：
          1. 贴图写在 `canvas_items`，不进 `detection_boxes`；
          2. 工具解析不出的 shape（如 `points` 为空的多边形）载入后不产生框。
        两者都会让「非空 json」看起来像空画布。读原文则一律保留。
        读不出来（损坏）也保留——无法证明为空时不删。
        """
        if self.editor._is_delete_view:
            return
        mode = getattr(self.editor, 'edit_mode', 'annotate')
        if mode == 'paste' and not self.editor.canvas_items:
            return
        if mode == 'annotate':
            idx = self.editor.current_background_index
            if idx < 0 or not self.editor.background_images:
                return
            if idx >= len(self.editor.background_images):
                return
            orig_path = self.editor.background_images[idx]
            json_path = f"{os.path.splitext(orig_path)[0]}.json"
            has_boxes = bool(self.editor.detection_boxes)
            json_exists = os.path.exists(json_path)
            if json_exists and not has_boxes and self._json_shapes_empty(json_path):
                if getattr(self.editor, '_delete_empty_json_on_switch', True):
                    if self._delete_sidecar_json(json_path):
                        # 图标要跟着改，否则列表行仍显示「空标签」的橙色圆环，
                        # 看起来像没有删除。保存路径本来就有这一步。
                        refresh = getattr(self.editor,
                                          '_refresh_background_item_status', None)
                        if callable(refresh):
                            refresh(idx, orig_path)
                    return
                # 开关关掉：维持旧行为，写一份空 json 覆盖
            # Has labels → save; never annotated → skip
            if not has_boxes and not json_exists:
                return
            if self.editor.current_background is None:
                return
            base_name = os.path.basename(orig_path)
            # Pass image path; save_json derives sibling .json itself.
            self.save_json(
                orig_path, base_name, '',
                canvas_items=[],
                image_width=self.editor.current_background.width(),
                image_height=self.editor.current_background.height(),
            )
            return
        save_info = self.get_save_info()
        if not save_info:
            return
        self.save_json(*save_info)

    @staticmethod
    def _json_shapes_empty(json_path):
        """sidecar json 的 `shapes` 是否确实为空（可以安全删除）。

        只有**读成功且 `shapes` 是非空列表以外的东西**才算空。以下情况一律
        返回 False（= 不删），因为无法证明它没有内容：
        - 文件读不出来 / 不是合法 JSON；
        - 顶层不是对象，或没有 `shapes` 键；
        - `shapes` 不是列表。

        注意 `shapes: []`（真正的空标签）返回 True，而 `shapes` 里有任何
        条目（含贴图、含解析不出的 shape）都返回 False。
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        if 'shapes' not in data:
            return False
        shapes = data['shapes']
        if not isinstance(shapes, list):
            return False
        return len(shapes) == 0

    def _delete_sidecar_json(self, json_path):
        """删除空的 sidecar JSON；失败只记日志，不打断切图。

        返回是否真的删掉了（调用方据此决定要不要刷新列表图标）。

        切图是高频动作，删不掉（只读盘、被占用）也不该弹窗拦住用户；
        下次切回来若仍是空画布还会再试一次。
        """
        try:
            os.remove(json_path)
        except OSError as exc:
            from ..core.exception_hook import _write_log
            _write_log(f"删除空标注 JSON 失败：{json_path}：{exc}")
            return False
        from ..core import json_trace
        json_trace.trace_delete('save_manager._delete_sidecar_json',
                                json_path, survived=os.path.exists(json_path))
        return True
    
    def save_all_canvas(self):
        """保存所有画布"""
        from ..ui.dialogs import ProgressDialogFactory

        if self.editor._is_delete_view or self.editor._busy:
            return

        if not self.editor.background_images:
            _show_messagebox("warning", self.editor, tr("警告"), tr("没有背景图片可保存"))
            return
        
        self.editor._busy = True

        progress_dialog = ProgressDialogFactory.create_progress_dialog(
            self.editor, tr("保存进度"), tr("正在保存所有图片..."), len(self.editor.background_images)
        )
        progress_dialog.show()
        QApplication.processEvents()  # 确保进度条立即显示
        
        original_background = self.editor.current_background
        original_index = self.editor.current_background_index
        original_canvas_items = self.editor.canvas_items.copy()
        
        saved_count = 0
        for i, file_path in enumerate(self.editor.background_images):
            if progress_dialog.wasCanceled():
                break
            
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"{tr('正在保存第')} {i+1} {tr('张图片')}...")
            
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                continue
            
            if i == getattr(self.editor, 'current_background_index', -1):
                # 当前图用编辑中的那一份：字典里存的是上次切图/退出时的快照，
                # 贴图改过标签或旋转后还没回写，用它会写出旧状态。
                temp_canvas_items = list(self.editor.canvas_items)
            else:
                temp_canvas_items = self.editor.canvas_items_dict.get(i, [])
            
            original_file_name = os.path.basename(file_path)
            output_dir = PathUtils.get_output_dir(file_path)
            
            prefix = ""
            if hasattr(self.editor, 'prefix_checkbox') and self.editor.prefix_checkbox.isChecked():
                prefix = self.editor.prefix_input.text().strip()
                if not prefix:
                    prefix = DEFAULT_PREFIX
            
            base_name = f"{prefix}_{original_file_name}" if prefix else original_file_name
            save_file_path = os.path.join(output_dir, base_name)
            
            result = self._new_composite(pixmap)
            painter = QPainter(result)
            try:
                painter.drawPixmap(0, 0, pixmap)
            finally:
                painter.end()
            
            self._compose_pastes(result, temp_canvas_items)
            
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if not result.save(save_file_path):
                from ..core.exception_hook import _write_log
                _write_log(f"批量保存失败: {save_file_path}")
            
            self.save_json(save_file_path, base_name, prefix, temp_canvas_items,
                          pixmap.width(), pixmap.height(), i)
            
            saved_count += 1
            
            # 处理事件，确保进度条更新
            QApplication.processEvents()
        
        progress_dialog.setValue(len(self.editor.background_images))
        
        # 关闭进度对话框
        progress_dialog.close()
        
        # 显示保存结果
        if saved_count > 0:
            _show_messagebox(
                "information", self.editor, tr("保存完成"),
                f"{tr('全部保存完成！')}\n{tr('成功保存')} {saved_count} {tr('张图片')}。"
            )
        else:
            _show_messagebox(
                "warning", self.editor, tr("保存结果"),
                tr("没有保存任何图片。")
            )
        
        if original_index >= 0:
            self.editor.current_background = original_background
            self.editor.current_background_index = original_index
            self.editor.canvas_items = original_canvas_items
            self.save_completed.emit()
        
        self.label_list_changed.emit()
        self.editor._busy = False
    
    @staticmethod
    def _deduplicate_rectangles(items, get_rect):
        """按列表顺序保留首个 IoU 达 98% 的对象；带 label 的按类别分别去重。

        类别必须参与去重：推理结果里不同类别的框常常高度重合，纯几何去重会把
        它们静默删掉，保存/重载后标注数量就对不上了。
        """
        kept = []
        for item in items:
            rect = get_rect(item)
            box = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
            label = item.get('label') if isinstance(item, dict) else None
            duplicates = False
            for saved in kept:
                saved_rect = get_rect(saved)
                if label is not None and isinstance(saved, dict):
                    if saved.get('label') != label:
                        continue
                if calculate_iou(box, (
                    saved_rect.x(), saved_rect.y(),
                    saved_rect.x() + saved_rect.width(), saved_rect.y() + saved_rect.height(),
                )) >= 0.98:
                    duplicates = True
                    break
            if duplicates:
                continue
            kept.append(item)
        return kept

    def save_json(self, image_path, image_name, label_prefix, canvas_items=None,
                 image_width=None, image_height=None, current_index=None):
        """生成并保存 JSON 文件"""
        if self.editor._is_delete_view:
            return
        if getattr(self.editor, '_lint_removal_busy', False):
            # 质检删除 worker 正在后台改写同一 sidecar：主线程写入会导致 JSON 交错损坏
            from ..core.exception_hook import _write_log
            _write_log(f"质检删除进行中，跳过保存 JSON: {image_path}")
            return
        if hasattr(self.editor, '_dataset_stats_dirty'):
            self.editor._dataset_stats_dirty = True
        json_path = os.path.splitext(image_path)[0] + '.json'
        
        items_to_use = canvas_items if canvas_items is not None else self.editor.canvas_items
        # 入参可能来自旧三元组（撤销栈、外部调用方），统一归一化后再用
        items_to_use = [as_paste_item(entry) for entry in items_to_use]
        width = image_width if image_width is not None else (
            self.editor.current_background.width() if self.editor.current_background else 0
        )
        height = image_height if image_height is not None else (
            self.editor.current_background.height() if self.editor.current_background else 0
        )
        index_to_use = current_index if current_index is not None else self.editor.current_background_index
        items_to_use = self._deduplicate_rectangles(
            items_to_use,
            # 外接框可以是 QRectF，也可以是四元组；去重只认几何，
            # 统一成 QRectF 免得四元组在 `.x()` 上炸。
            lambda item: QRectF(*item.rect_tuple()),
        )
        if canvas_items is None:
            self.editor.canvas_items = items_to_use
        elif isinstance(canvas_items, list):
            canvas_items[:] = items_to_use

        boxes_to_use = []
        if index_to_use >= 0 and index_to_use in self.editor.detection_boxes_dict:
            raw_boxes = self.editor.detection_boxes_dict[index_to_use]
            rects = [b for b in raw_boxes if (b.get('shape_type') or 'rectangle') == 'rectangle']
            kept_rects = self._deduplicate_rectangles(
                rects,
                lambda box: QRectF(box['x'], box['y'], box['width'], box['height']),
            )
            kept_ids = {id(b) for b in kept_rects}
            boxes_to_use = [
                b for b in raw_boxes
                if (b.get('shape_type') or 'rectangle') != 'rectangle' or id(b) in kept_ids
            ]
            self.editor.detection_boxes_dict[index_to_use] = boxes_to_use
            if index_to_use == self.editor.current_background_index:
                self.editor.detection_boxes = boxes_to_use.copy()
        
        json_data = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }
        
        # 添加贴图（flags.paste 供统计区分贴图与检测框）
        for raw in items_to_use:
            json_data["shapes"].append(
                self._paste_item_to_shape(as_paste_item(raw)))
        
        # 添加检测框
        if index_to_use >= 0:
            from .shape_io import labelme_shape_from_box
            for box in boxes_to_use:
                json_data["shapes"].append(labelme_shape_from_box(box))
        
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        # 先写临时文件再原子替换：中断/异常不会留下被截断的 JSON
        tmp_path = f"{json_path}.tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, json_path)
            refresh = getattr(self.editor, '_refresh_background_item_status', None)
            if callable(refresh):
                refresh(index_to_use, image_path)
        except Exception as e:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            from ..core.exception_hook import _write_log
            _write_log(f"保存 JSON 失败: {json_path}, 错误: {e}")
