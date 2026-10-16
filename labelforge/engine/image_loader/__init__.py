"""图片加载包 —— 向后兼容聚合入口。

原 image_loader.py 按职责拆为 status / scan / mixin 三个子模块，
本模块把三者的公开面完整再导出，外部 `from ...engine.image_loader import X`
无需改动。模块级名字 json/os/QFileDialog/QMessageBox 一并暴露，供既有
测试 monkeypatch 使用。
"""
import json
import os

from PyQt5.QtWidgets import QFileDialog, QMessageBox

from .status import *
from .status import _status_icon
from .scan import *
from .scan import _count_labels_in_json, _scan_single_json
from . import mixin
from .mixin import ImageLoaderMixin

__all__ = [
    "BG_ROLE_INDEX", "BG_ROLE_PATH", "BG_ROLE_STATUS",
    "STATUS_UNANNOTATED", "STATUS_ANNOTATED", "STATUS_EMPTY", "STATUS_PENDING",
    "annotation_status_for_image", "annotation_status_for_image_light",
    "decorate_background_list_item", "apply_background_status_to_item",
    "scan_dataset_labels", "scan_dataset_labels_with_counts", "scan_dataset_full",
    "collect_background_label_counts", "collect_background_label_tasks",
    "count_labels_in_json", "DatasetLabelScanWorker",
    "ImageLoaderMixin",
]
