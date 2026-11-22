"""背景图状态判定、状态图标与列表项装饰。"""
import os
import json

from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt

from ...ui.i18n import t as tr

__all__ = [
    "BG_ROLE_INDEX", "BG_ROLE_PATH", "BG_ROLE_STATUS",
    "STATUS_UNANNOTATED", "STATUS_ANNOTATED", "STATUS_EMPTY", "STATUS_PENDING",
    "annotation_status_for_image", "annotation_status_for_image_light",
    "decorate_background_list_item", "apply_background_status_to_item",
]


# 背景图列表项的 UserRole 键，UserRole 的取值是 0x0100
BG_ROLE_INDEX = 0x0100
BG_ROLE_PATH = 0x0101
BG_ROLE_STATUS = 0x0102

STATUS_UNANNOTATED = "unannotated"
STATUS_ANNOTATED = "annotated"
STATUS_EMPTY = "empty"
# 批量填充列表时的临时状态：sidecar JSON 已存在但还没扫过，
# 由后台扫描完成后替换。
STATUS_PENDING = "pending"

_STATUS_ICON_CACHE = {}


def annotation_status_for_image(image_path):
    """给 sidecar JSON 归类：未标注 / 已标注 / 空。"""
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
    """先给一个省事的临时状态：只查文件在不在，不解析 JSON。

    批量铺列表时用它，免得 UI 线程去啃几千份 sidecar；
    真正的状态由后台扫描线程随后补上。
    """
    if not image_path:
        return STATUS_UNANNOTATED
    json_path = os.path.splitext(image_path)[0] + ".json"
    if not os.path.exists(json_path):
        return STATUS_UNANNOTATED
    return STATUS_PENDING


def _status_icon(status, size=12):
    """背景图列表行与筛选按钮用的小圆点状态图标。"""
    cache_key = (status, size)
    if cache_key in _STATUS_ICON_CACHE:
        return _STATUS_ICON_CACHE[cache_key]
    from PySide6.QtGui import QColor, QPainter, QPen
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    if status == "all":
        # 三颗小圆点：绿 / 灰 / 橙
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
        # 灰色虚线圆环：JSON 已存在但还没扫过。
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
    """给列表项挂上路径、下标、状态与状态图标。

    light=True 时不解析 sidecar，只查存在性，批量铺大列表因此绕开
    JSON 解析；准确状态稍后由 apply_background_status_to_item 补上。
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
    """用算好的状态刷新已有列表行的图标与提示。"""
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
