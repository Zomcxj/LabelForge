"""Shared SVG icon rendering helpers and icon data."""
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt

try:
    from PyQt5.QtSvg import QSvgRenderer
    _has_svg = True
except ImportError:
    _has_svg = False


def _load_svg_icon(svg_data, size=16, color="#999"):
    if not _has_svg:
        return QPixmap(size, size)
    svg_data = svg_data.replace('currentColor', color)
    renderer = QSvgRenderer(bytearray(svg_data.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap


SVG_FILE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/><polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/></svg>'
SVG_FOLDER = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="2" fill="none" stroke-linejoin="round"/></svg>'
# 文件 + 向下的箭头：表示「从文件导入背景标签」
SVG_IMPORT = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><line x1="12" y1="12" x2="12" y2="18"/><polyline points="9,15 12,18 15,15"/></svg>'
# 文件 + 内部的标签框：表示「导入贴图标签」，与背景标签导入的箭头图标区分
SVG_PASTE_LABEL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14,2 14,8 20,8"/><rect x="8" y="12.5" width="8" height="5.5" rx="1"/></svg>'
SVG_CLASSIFY = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="5" cy="6" r="2.5"/><circle cx="19" cy="7" r="2.5"/><circle cx="12" cy="18" r="2.5"/><line x1="7" y1="7.5" x2="16.5" y2="15.5"/><line x1="7" y1="7.5" x2="17" y2="9"/></svg>'
SUN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>'
MOON_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>'
PROCESS_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 2l2.4 7.2H22l-6.2 4.4 2.4 7.2L12 16.4 5.8 20.8l2.4-7.2L2 9.2h7.6z" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linejoin="round"/></svg>'
SVG_REFRESH = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>'

# 芯片 + 向下箭头：加载模型
SVG_MODEL_LOAD = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/><path d="M12 9.5v5M9.5 12l2.5 2.5 2.5-2.5"/></svg>'
# 单张图 + 扫描角标：只推理当前图
SVG_MODEL_SINGLE = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="9.5" r="1.5"/><path d="M21 16l-5-5-5.5 5.5"/><path d="M14 3v3h3"/></svg>'
# 多层图 + 闪电：从当前位置一键跑完
SVG_MODEL_ALL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="13" height="10" rx="2"/><path d="M17 9.5h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H8"/><path d="M19 2l-3 5h3l-3 5"/></svg>'
