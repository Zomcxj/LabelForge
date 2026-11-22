"""图像编辑面板入口：浮窗贴在主窗口右侧，处理结果显示在主画布上。

面板只管收集操作项，原图与处理链都在这里：切换图片、跑链、把关结果写进
`current_background`，关面板时把主界面那张图载回来。
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

from ..i18n import t as tr

# 参数连续变动时只在停手后跑一次处理链
_REFRESH_DELAY_MS = 150


class ImageToolsMixin:
    def _current_image_path(self):
        """当前编辑的图；未加载数据集时返回空串。"""
        images = getattr(self, 'background_images', None) or []
        index = getattr(self, 'current_background_index', -1)
        if 0 <= index < len(images):
            return images[index]
        return ""

    def _open_image_tools(self):
        """打开图像编辑面板（非模态，画布仍可操作）。"""
        from ..image_tools_panel import ImageToolsPanel
        from ..dialog_helpers import place_within

        if getattr(self, '_is_delete_view', False):
            status = getattr(self, 'status_label', None)
            if status is not None:
                status.setText(tr("删除视图下不可用"))
            return

        self._discard_image_tools_panel()

        panel = ImageToolsPanel(self, self)
        self._image_tools_panel = panel
        place_within(panel, self._side_panel_geometry(), align="top-left")
        panel.specs_changed.connect(self._schedule_image_tools_apply)
        panel.closed.connect(self._image_tools_close)
        panel.previous_requested.connect(lambda: self._image_tools_step(-1))
        panel.next_requested.connect(lambda: self._image_tools_step(1))
        panel.compare_started.connect(lambda: self._image_tools_compare(True))
        panel.compare_finished.connect(lambda: self._image_tools_compare(False))
        panel.save_new_requested.connect(self._image_tools_save_new)
        panel.save_copy_requested.connect(self._image_tools_save_copy)

        self._image_tools_active = True
        self._image_tools_path = ""
        self._image_tools_original = None
        self._image_tools_debounce = QTimer(self)
        self._image_tools_debounce.setSingleShot(True)
        self._image_tools_debounce.setInterval(_REFRESH_DELAY_MS)
        self._image_tools_debounce.timeout.connect(self._image_tools_apply)

        self.background_list.itemClicked.connect(self._image_tools_on_item_clicked)
        self.background_list.currentRowChanged.connect(self._image_tools_on_item_clicked)

        panel.show()
        # 状态行常显：图像编辑期间标注一律不画，进来就说清楚
        panel.set_status(tr("预览中, 标注已隐藏"))
        self._image_tools_reload()

    def _discard_image_tools_panel(self):
        """内部重开时丢掉旧面板：不走用户关闭那套恢复流程，也不留残余连接。"""
        panel = getattr(self, '_image_tools_panel', None)
        timer = getattr(self, '_image_tools_debounce', None)
        if timer is not None:
            timer.stop()
            self._image_tools_debounce = None
        if panel is None:
            return
        self._image_tools_panel = None
        panel.blockSignals(True)
        panel.close()
        panel.deleteLater()

    def _image_tools_reload(self):
        """按主界面当前图重读原图，然后重算预览。同一张图不重读。"""
        from ..image_tools_panel import _flatten_alpha

        panel = getattr(self, '_image_tools_panel', None)
        path = self._current_image_path()
        if path and path == getattr(self, '_image_tools_path', ''):
            return
        self._image_tools_path = path

        image = QImage(path) if path else QImage()
        if image.isNull():
            self._image_tools_original = None
            if isinstance(getattr(self, 'image_tools_regions', None), list):
                self.image_tools_regions.clear()
            if panel is not None:
                panel.set_status(tr("未打开图片"))
            return

        self._image_tools_original = _flatten_alpha(image)
        # 换图后旧选区按像素坐标复用会在新图上错位，直接清掉
        if isinstance(getattr(self, 'image_tools_regions', None), list):
            self.image_tools_regions.clear()
        self._image_tools_apply()

    def _image_tools_preview_limit(self):
        """预览处理的上限边长：只查看不保存，按画布尺寸留一点余量即可。"""
        return max(256, int(self.canvas.width() * 1.5))

    def _schedule_image_tools_apply(self, *_args):
        timer = getattr(self, '_image_tools_debounce', None)
        if timer is None:
            self._image_tools_apply()
            return
        timer.start()

    def _image_tools_apply(self):
        """跑一遍处理链，把结果画到主画布上。"""
        from ...engine import image_tools as it

        original = getattr(self, '_image_tools_original', None)
        panel = getattr(self, '_image_tools_panel', None)
        if original is None or panel is None:
            return

        source = it.downsample_for_preview(
            original, self._image_tools_preview_limit())
        specs = panel.checked_specs(source.width(), source.height())
        geometric = any(it.is_geometric(spec["name"]) for spec in specs)
        # 图像编辑期间标注框一律不画（见 canvas_renderer 的
        # `_hide_detection_boxes`），提示常显，与有没有勾处理项无关。
        panel.set_status(tr("预览中, 标注已隐藏"))

        # 没有任何处理项时原样显示原图：降采样再放大回来会把细纹理抹平
        if not specs:
            self._image_tools_swap_background(QPixmap.fromImage(original))
            return

        mask = self._image_tools_mask(panel, it, source)
        if mask is not None:
            result = it.apply_chain_with_mask(source, specs, mask, seed=0)
        else:
            result = it.apply_chain(source, specs, seed=0)
        # 非几何类操作不改尺寸：把降采样的结果缩回原尺寸，标注坐标才仍然对得上
        if not geometric:
            result = result.scaled(original.size(), Qt.IgnoreAspectRatio,
                                   Qt.FastTransformation)
        self._image_tools_swap_background(QPixmap.fromImage(result))

    def _image_tools_swap_background(self, pixmap):
        """换画布上的图并按新尺寸重算视图。

        几何类操作会改尺寸：原图与处理结果宽高不同，沿用旧缩放比会让图
        顶出画布边缘（按住看原图时最明显）。未手动缩放时按新图重新适配，
        手动缩放过则只把偏移夹回可视范围，保持用户的缩放比例。
        """
        self.current_background = pixmap
        canvas = getattr(self, 'canvas', None)
        if canvas is None:
            return
        if getattr(canvas, 'is_manual_scale', False):
            # 手动缩放过：保留缩放比，只把平移量夹回可视范围
            canvas.clamp_background_offset()
        else:
            canvas.reset_view()
        canvas.update()

    def _image_tools_mask(self, panel, it, source):
        """所有选区的并集蒙版；没有选区时返回 None（整图生效）。"""
        if not panel.mask_enabled():
            return None
        selections = panel.selections()
        if not selections:
            return None
        width, height = source.width(), source.height()
        if len(selections) == 1:
            points, mode = selections[0]
            return it.selection_mask_from_points(points, width, height, mode)
        import numpy as np

        masks = [it.qimage_gray_to_array(
            it.selection_mask_from_points(points, width, height, mode))
            for points, mode in selections]
        merged = np.maximum.reduce(masks)
        buffer = np.ascontiguousarray(merged, dtype=np.uint8)
        return QImage(buffer.data, width, height, width,
                      QImage.Format_Grayscale8).copy()

    def _image_tools_on_item_clicked(self, _item=None):
        """主界面换了当前图：重读原图并重算。"""
        self._image_tools_reload()

    def _image_tools_render_full(self):
        """按原图尺寸跑一遍处理链，返回 (结果, 处理项)。

        预览是降采样后再缩回去的，像素已经变过；落盘必须重跑，不能存预览。
        """
        from ...engine import image_tools as it

        original = getattr(self, '_image_tools_original', None)
        panel = getattr(self, '_image_tools_panel', None)
        if original is None or panel is None:
            return None, []
        specs = panel.checked_specs(original.width(), original.height())
        if not specs:
            return None, []
        mask = self._image_tools_mask(panel, it, original)
        if mask is not None:
            return it.apply_chain_with_mask(original, specs, mask, seed=0), specs
        return it.apply_chain(original, specs, seed=0), specs

    def _image_tools_compare(self, show_original):
        """按住看原图，松开回到处理结果。"""
        original = getattr(self, '_image_tools_original', None)
        if original is None:
            return
        panel = getattr(self, '_image_tools_panel', None)
        if show_original:
            # 待跑的处理链要停掉：按住期间它一旦触发，画布会从原图跳回结果
            timer = getattr(self, '_image_tools_debounce', None)
            if timer is not None:
                timer.stop()
            self._image_tools_swap_background(QPixmap.fromImage(original))
            if panel is not None:
                panel.set_status_now(tr("原图"))
            return
        if panel is not None:
            panel.set_status("")
        self._image_tools_apply()

    def _image_tools_output_dir(self):
        """本次运行的输出目录：另存为选过就用那个，否则 <数据集同级>/export/edit。"""
        from ...engine import image_tools as it

        remembered = getattr(self, '_image_tools_out_dir', '')
        if remembered:
            return remembered
        path = self._current_image_path()
        return it.default_output_dir(path) if path else ''

    def _image_tools_save_new(self):
        """按处理项命名，写进本次运行的输出目录（默认与数据集同级的 export/edit）。"""
        import os

        from ...engine import image_tools as it

        path = self._current_image_path()
        if not path:
            self._image_tools_save_flash(tr("未打开图片"))
            return
        result, specs = self._image_tools_render_full()
        if result is None:
            self._image_tools_save_flash(tr("没有勾选任何处理项"))
            return
        target = it.derived_output_path(
            path, it.ops_suffix(specs), self._image_tools_output_dir())
        if self._image_tools_write(result, target):
            self._image_tools_save_flash(
                f"{tr('已生成新图')}: {os.path.basename(target)}")

    def _image_tools_save_copy(self):
        """另存到指定位置，原图与标注都不动。"""
        import os

        from ...engine import image_tools as it

        path = self._current_image_path()
        folder = self._image_tools_output_dir() or os.getcwd()
        result, specs = self._image_tools_render_full()
        if result is None:
            self._image_tools_save_flash(tr("没有勾选任何处理项"))
            return
        default = (it.derived_output_path(path, it.ops_suffix(specs), folder)
                   if path else os.path.join(folder, "image_edit.png"))
        target = self._image_tools_ask_save_path(default)
        if not target:
            return
        if self._image_tools_write(result, target):
            # 只本次运行有效：关掉软件就回到默认位置
            self._image_tools_out_dir = os.path.dirname(
                os.path.abspath(target))
            self._image_tools_save_flash(
                f"{tr('已保存')}: {os.path.basename(target)}")

    def _image_tools_ask_save_path(self, default):
        """另存为的目标路径；用户取消时返回空串。"""
        from PySide6.QtWidgets import QFileDialog

        target, _filter = QFileDialog.getSaveFileName(
            self, tr("另存为"), default,
            tr("图片文件 (*.png *.jpg *.jpeg *.bmp)"))
        return target

    def _image_tools_write(self, image, path):
        """写盘；处理不碰原图，写成功只回报路径。"""
        from ...engine import image_tools as it

        if not it.save_result(image, path):
            self._image_tools_save_flash(tr("保存失败"))
            return False
        return True

    def _image_tools_save_flash(self, text):
        panel = getattr(self, '_image_tools_panel', None)
        if panel is not None:
            panel.flash_status(text)
    def _image_tools_step(self, direction):
        """按主界面的切图规则前后翻一张，翻完重算预览。"""
        self.switch_background(direction)
        self._image_tools_reload()

    def _image_tools_close(self):
        """用户关面板：解锁画布，把主界面原来那张图载回来。"""
        panel = getattr(self, '_image_tools_panel', None)
        timer = getattr(self, '_image_tools_debounce', None)
        self._image_tools_panel = None
        self._image_tools_active = False
        self._image_tools_original = None
        self._image_tools_path = ""
        if timer is not None:
            timer.stop()
            self._image_tools_debounce = None
        self._disconnect_background_list()
        if isinstance(getattr(self, 'image_tools_regions', None), list):
            self.image_tools_regions.clear()
        if panel is not None:
            panel.deleteLater()
        self.load_image_by_index(getattr(self, 'current_background_index', -1))
        self.canvas.update()

    def _disconnect_background_list(self):
        listing = getattr(self, 'background_list', None)
        if listing is None:
            return
        for signal in (listing.itemClicked, listing.currentRowChanged):
            try:
                signal.disconnect(self._image_tools_on_item_clicked)
            except (RuntimeError, TypeError):
                pass
