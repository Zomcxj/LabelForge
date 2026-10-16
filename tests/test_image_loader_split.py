"""image_loader 包拆分后的契约测试：导出面完整、私名兼容、无循环 import。"""
import importlib

import pytest


PUBLIC_SYMBOLS = [
    # 常量
    "BG_ROLE_INDEX", "BG_ROLE_PATH", "BG_ROLE_STATUS",
    "STATUS_UNANNOTATED", "STATUS_ANNOTATED", "STATUS_EMPTY", "STATUS_PENDING",
    # status
    "annotation_status_for_image", "annotation_status_for_image_light",
    "decorate_background_list_item", "apply_background_status_to_item",
    # scan
    "scan_dataset_labels", "scan_dataset_labels_with_counts", "scan_dataset_full",
    "collect_background_label_counts", "collect_background_label_tasks",
    "count_labels_in_json", "DatasetLabelScanWorker",
    # mixin
    "ImageLoaderMixin",
]

PRIVATE_SYMBOLS = ["_status_icon", "_count_labels_in_json", "_scan_single_json"]

MODULE_ATTRS = ["json", "os", "QFileDialog", "QMessageBox"]


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS + PRIVATE_SYMBOLS + MODULE_ATTRS)
def test_image_loader_package_exposes_symbol(name):
    from labelforge.engine import image_loader

    assert hasattr(image_loader, name), f"image_loader 缺少 {name}"


def test_count_labels_public_alias_is_same_object():
    from labelforge.engine import image_loader

    assert image_loader.count_labels_in_json is image_loader._count_labels_in_json


@pytest.mark.parametrize("sub", ["status", "scan", "mixin"])
def test_submodule_imports_independently(sub):
    importlib.import_module(f"labelforge.engine.image_loader.{sub}")


def test_status_and_scan_do_not_depend_on_mixin():
    import labelforge.engine.image_loader.status as status
    import labelforge.engine.image_loader.scan as scan

    for mod in (status, scan):
        src = open(mod.__file__, encoding="utf-8").read()
        assert "from .mixin import" not in src
        assert "from . import mixin" not in src
        assert "import mixin" not in src
