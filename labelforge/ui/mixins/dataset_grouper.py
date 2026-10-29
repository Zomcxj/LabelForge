"""Dataset grouping (label similarity clustering) behavior for the main window."""
import os


class DatasetGrouperMixin:
    def _get_dataset_grouper_folder(self):
        """当前打开的文件夹路径: 优先取第一张背景图片所在目录。"""
        images = getattr(self, 'background_images', None)
        if images:
            return os.path.dirname(images[0])
        return os.getcwd()

    def _open_dataset_grouper(self):
        """打开数据集分组对话框。"""
        from ..dataset_grouper_dialog import DatasetGrouperDialog
        dialog = DatasetGrouperDialog(self, default_folder=self._get_dataset_grouper_folder())
        dialog.exec_()
