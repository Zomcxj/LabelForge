"""背景图状态判定、状态图标与列表项装饰。"""
import os
import json

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt

from ...ui.i18n import t as tr

__all__ = [
    "BG_ROLE_INDEX", "BG_ROLE_PATH", "BG_ROLE_STATUS",
    "STATUS_UNANNOTATED", "STATUS_ANNOTATED", "STATUS_EMPTY", "STATUS_PENDING",
    "annotation_status_for_image", "annotation_status_for_image_light",
    "decorate_background_list_item", "apply_background_status_to_item",
]


# background list item UserRole keys (Qt.ItemDataRole.UserRole == 0x0100)
BG_ROLE_INDEX = 0x0100
BG_ROLE_PATH = 0x0101
BG_ROLE_STATUS = 0x0102

STATUS_UNANNOTATED = "unannotated"
STATUS_ANNOTATED = "annotated"
STATUS_EMPTY = "empty"
# Provisional status used while bulk-populating the list: the sidecar JSON
# exists but has not been scanned yet. Replaced by the background scan worker.
STATUS_PENDING = "pending"

_STATUS_ICON_CACHE = {}


def annotation_status_for_image(image_path):
    """Classify sidecar LabelMe JSON: unannotated / annotated / empty."""
    if not image_path:
        return STATUS_UNANNOTATED
    json_path = os.path.splitext(image_path)[0] + ".json"
    if not os.path.exists(json_path):
        return STATUS_UNANNOTATED
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return STATUS_EMPTY
    shapes = data.get("shapes") if isinstance(data, dict) else None
    if isinstance(shapes, list) and len(shapes) > 0:
        return STATUS_ANNOTATED
    return STATUS_EMPTY


def annotation_status_for_image_light(image_path):
    """Cheap provisional status: no JSON parse, only a file-existence check.

    Used while bulk-populating the background list so the UI thread never
    parses thousands of sidecar JSONs. The real status is filled in later by
    the background scan worker.
    """
    if not image_path:
        return STATUS_UNANNOTATED
    json_path = os.path.splitext(image_path)[0] + ".json"
    if not os.path.exists(json_path):
        return STATUS_UNANNOTATED
    return STATUS_PENDING


def _status_icon(status, size=12):
    """Small circular status icon for background list rows / filter button."""
    cache_key = (status, size)
    if cache_key in _STATUS_ICON_CACHE:
        return _STATUS_ICON_CACHE[cache_key]
    from PyQt5.QtGui import QColor, QPainter, QPen
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    if status == "all":
        # three mini dots: green / gray / orange
        for i, color in enumerate(("#2ecc71", "#95a5a6", "#e67e22")):
            painter.setBrush(QColor(color))
            painter.setPen(Qt.NoPen)
            x = 1 + i * max(3, size // 3)
            painter.drawEllipse(x, size // 2 - 2, 4, 4)
    elif status == STATUS_ANNOTATED:
        painter.setBrush(QColor("#2ecc71"))
        painter.setPen(QPen(QColor("#1e8449"), 1))
        painter.drawEllipse(1, 1, size - 2, size - 2)
    elif status == STATUS_EMPTY:
        painter.setBrush(Qt.transparent)
        painter.setPen(QPen(QColor("#e67e22"), 1.5))
        painter.drawEllipse(1, 1, size - 2, size - 2)
        painter.setBrush(QColor("#e67e22"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(4, 4, size - 8, size - 8)
    elif status == STATUS_PENDING:
        # Dashed gray ring: JSON exists but has not been scanned yet.
        painter.setBrush(Qt.transparent)
        painter.setPen(QPen(QColor("#95a5a6"), 1.5, Qt.DashLine))
        painter.drawEllipse(1, 1, size - 2, size - 2)
    else:
        painter.setBrush(Qt.transparent)
        painter.setPen(QPen(QColor("#95a5a6"), 1.5))
        painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    icon = QIcon(pm)
    _STATUS_ICON_CACHE[cache_key] = icon
    return icon


def decorate_background_list_item(item, image_path, index=None, light=False):
    """Attach path/index/status metadata and status icon to a list item.

    light=True skips parsing the sidecar JSON (only checks existence) so
    bulk-populating a large dataset stays off the JSON parser. The accurate
    status is applied later via apply_background_status_to_item.
    """
    if item is None:
        return STATUS_UNANNOTATED
    if index is not None and hasattr(item, 'setData'):
        item.setData(BG_ROLE_INDEX, index)
    if image_path and hasattr(item, 'setData'):
        item.setData(BG_ROLE_PATH, image_path)
    status = (annotation_status_for_image_light(image_path) if light
              else annotation_status_for_image(image_path))
    if hasattr(item, 'setData'):
        item.setData(BG_ROLE_STATUS, status)
    if hasattr(item, 'setIcon'):
        try:
            item.setIcon(_status_icon(status))
        except Exception:
            pass
    tips = {
        STATUS_ANNOTATED: tr("已标注"),
        STATUS_EMPTY: tr("空标签"),
        STATUS_UNANNOTATED: tr("未标注"),
        STATUS_PENDING: tr("扫描中"),
    }
    if hasattr(item, 'setToolTip'):
        item.setToolTip(tips.get(status, ""))
    return status


def apply_background_status_to_item(item, status):
    """Update an existing list row with a resolved status + icon + tooltip."""
    if item is None:
        return
    if hasattr(item, 'setData'):
        item.setData(BG_ROLE_STATUS, status)
    if hasattr(item, 'setIcon'):
        try:
            item.setIcon(_status_icon(status))
        except Exception:
            pass
    tips = {
        STATUS_ANNOTATED: tr("已标注"),
        STATUS_EMPTY: tr("空标签"),
        STATUS_UNANNOTATED: tr("未标注"),
        STATUS_PENDING: tr("扫描中"),
    }
    if hasattr(item, 'setToolTip'):
        item.setToolTip(tips.get(status, ""))
