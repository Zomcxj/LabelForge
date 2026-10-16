"""Dataset format conversion entry point behavior for the main window."""
import os


class DatasetToolsMixin:
    def _get_dataset_tools_folder(self):
        """默认目录: 取第一张背景图所在目录；页面未加载数据集时留空。"""
        images = getattr(self, 'background_images', None)
        if images:
            return os.path.dirname(images[0])
        return ""

    def _open_dataset_tools(self):
        """打开格式转换对话框。"""
        from ..dataset_tools_dialog import DatasetToolsDialog
        dialog = DatasetToolsDialog(self, default_folder=self._get_dataset_tools_folder())
        dialog.exec_()
