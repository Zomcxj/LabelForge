"""Dataset classification (KMeans freq+area) behavior for the main window."""
import os


class DatasetClassifierMixin:
    def _get_dataset_classifier_folder(self):
        """当前打开的文件夹路径: 优先取第一张背景图片所在目录。"""
        images = getattr(self, 'background_images', None)
        if images:
            return os.path.dirname(images[0])
        return os.getcwd()

    def _open_dataset_classifier(self):
        """打开数据集分类对话框。"""
        from ..dataset_classifier_dialog import DatasetClassifierDialog
        dialog = DatasetClassifierDialog(self, default_folder=self._get_dataset_classifier_folder())
        dialog.exec_()
