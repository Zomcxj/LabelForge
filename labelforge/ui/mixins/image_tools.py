"""图像处理面板入口：浮窗贴在主窗口右侧，处理结果显示在主画布上。

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
        """打开图像处理面板（非模态，画布仍可操作）。"""
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

        self._image_tools_active = True
        self._image_tools_geometric = False
        self._image_tools_path = ""
        self._image_tools_original = None
        self._image_tools_debounce = QTimer(self)
        self._image_tools_debounce.setSingleShot(True)
        self._image_tools_debounce.setInterval(_REFRESH_DELAY_MS)
        self._image_tools_debounce.timeout.connect(self._image_tools_apply)

        self.background_list.itemClicked.connect(self._image_tools_on_item_clicked)
        self.background_list.currentRowChanged.connect(self._image_tools_on_item_clicked)

        panel.show()
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
            self._image_tools_geometric = False
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
        self._image_tools_geometric = geometric
        panel.set_status(tr("预览中, 标注已隐藏") if geometric else "")

        # 没有任何处理项时原样显示原图：降采样再放大回来会把细纹理抹平
        if not specs:
            self.current_background = QPixmap.fromImage(original)
            self.canvas.update()
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
        self.current_background = QPixmap.fromImage(result)
        self.canvas.update()

    def _image_tools_mask(self, panel, it, source):
        """所有选区的并集蒙版；没有可用选区或没勾「仅选区内」时返回 None。"""
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
        self._image_tools_geometric = False
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
