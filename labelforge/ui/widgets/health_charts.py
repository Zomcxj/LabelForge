"""数据集健康自绘图表：QPainter 条形图 / 直方图，无第三方依赖。"""
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QColor, QPen

from ..theme import ThemeManager

_PALETTE = (
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#795548', '#607D8B', '#E91E63', '#3F51B5',
)


def _bar_label(value, total):
    """条形图右侧文字：数量 + 整数百分比。"""
    pct = int(round(value / total * 100)) if total else 0
    return f"{_fmt_num(value)} ({pct}%)"


def _fmt_num(v):
    """面积等大数格式化：1.2k / 2.5M / 整数。"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '0'
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}k"
    return str(int(round(v)))


def _hist_header(ylabel, counts):
    """直方图左上角文字：轴名 + 总数。

    必须显示总和而不是峰值柱高：三个直方图的柱高口径不同
    （IoU 图统计的是同类框配对数），只显示峰值会让人误以为框数不一致。
    """
    return f"{ylabel}: {_fmt_num(sum(counts or []))}"


def _fmt_tick_series(values):
    """一组刻度用同一单位显示，避免 222 与 245.3k 混排看不懂。"""
    try:
        nums = [float(v) for v in values]
    except (TypeError, ValueError):
        return ['0'] * len(list(values))
    peak = max((abs(v) for v in nums), default=0.0)
    if peak >= 1e6:
        return [f"{v / 1e6:.1f}M" for v in nums]
    if peak >= 1e3:
        return [f"{v / 1e3:.1f}k" for v in nums]
    if peak >= 100:
        return [f"{v:.0f}" for v in nums]
    return [f"{v:.1f}".rstrip('0').rstrip('.') or '0' for v in nums]


class HealthBarChart(QWidget):
    """横向条形图（类别分布）或纵向直方图（尺寸/长宽比/IoU）。"""

    def __init__(self, parent=None, height=140):
        super().__init__(parent)
        self._mode = 'empty'
        self._items = []
        self._edges = []
        self._counts = []
        self._horizontal = True
        self._highlight = False
        self._placeholder = ''
        self._xlabel = ''
        self._ylabel = ''
        self.setMinimumHeight(height)

    def set_data(self, items, horizontal=True, highlight_extremes=False):
        self._mode = 'bars'
        self._items = list(items or [])
        self._horizontal = horizontal
        self._highlight = highlight_extremes
        self.update()

    def set_histogram(self, edges, counts, xlabel='', ylabel=''):
        self._mode = 'hist'
        self._edges = list(edges or [])
        self._counts = list(counts or [])
        self._xlabel = str(xlabel or '')
        self._ylabel = str(ylabel or '')
        self.update()

    def set_placeholder(self, text):
        self._mode = 'empty'
        self._placeholder = str(text or '')
        self.update()

    def paintEvent(self, event):
        t = ThemeManager.get_theme()
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            w = self.width()
            h = self.height()
            if self._mode == 'empty':
                painter.setPen(QPen(QColor(t['text_secondary'])))
                painter.drawText(0, 0, w, h, Qt.AlignCenter, self._placeholder)
                return
            if self._mode == 'bars':
                self._paint_bars(painter, w, h, t)
            else:
                self._paint_hist(painter, w, h, t)
        finally:
            painter.end()

    def _paint_bars(self, painter, w, h, t):
        if not self._items:
            return
        values = [float(i.get('value', 0) or 0) for i in self._items]
        peak = max(values) or 1.0
        total = sum(values) or 1.0
        border = QColor(t['border_color'])
        text_pen = QPen(QColor(t['text_primary']))
        if self._horizontal:
            fm = painter.fontMetrics()
            text_h = max(12, fm.height())
            row_h = max(text_h, h // max(1, len(self._items)))
            labels = [str(i.get('label', '')) for i in self._items]
            value_labels = [_bar_label(v, total) for v in values]
            left_w = min(max([fm.width(s) for s in labels] + [0]) + 8,
                         max(20, w // 3))
            right_w = min(max([fm.width(s) for s in value_labels] + [0]) + 8,
                          max(20, w // 3))
            bar_area_w = max(10, w - left_w - right_w)
            for idx, item in enumerate(self._items):
                y = idx * row_h
                value = float(item.get('value', 0) or 0)
                bar_w = int((value / peak) * bar_area_w)
                color = QColor(item.get('color') or _PALETTE[idx % len(_PALETTE)])
                if self._highlight and idx in (0, len(self._items) - 1):
                    pen = QPen(QColor(t['accent_hover']))
                    pen.setWidth(2)
                    painter.setPen(pen)
                else:
                    painter.setPen(QPen(border))
                painter.setBrush(color)
                painter.drawRect(left_w, y + 2, bar_w, row_h - 4)
                painter.setPen(text_pen)
                label = fm.elidedText(labels[idx], Qt.ElideRight,
                                      max(0, left_w - 4))
                painter.drawText(0, y, left_w - 4, row_h,
                                 Qt.AlignVCenter | Qt.AlignRight, label)
                painter.drawText(left_w + bar_w + 4, y,
                                 max(0, w - left_w - bar_w - 4), row_h,
                                 Qt.AlignVCenter | Qt.AlignLeft,
                                 value_labels[idx])
        else:
            col_w = max(4, w // max(1, len(self._items)))
            for idx, item in enumerate(self._items):
                x = idx * col_w
                value = float(item.get('value', 0) or 0)
                bar_h = int((value / peak) * (h - 20))
                color = QColor(item.get('color') or _PALETTE[idx % len(_PALETTE)])
                painter.setPen(QPen(border))
                painter.setBrush(color)
                painter.drawRect(x + 2, h - bar_h, col_w - 4, bar_h)

    def _paint_hist(self, painter, w, h, t):
        if not self._counts:
            return
        peak = max(self._counts) or 1
        accent = QColor(t['accent'])
        border = QColor(t['border_color'])
        text_pen = QPen(QColor(t['text_primary']))
        fm = painter.fontMetrics()
        line_h = max(12, fm.height())
        # 文字分带：ylabel 独占顶部一行，柱顶计数在其下；xlabel 独占底部一行
        top_pad = line_h + (line_h if self._ylabel else 0)
        bottom_pad = line_h + (line_h if self._xlabel else 0)
        plot_top = top_pad
        plot_bottom = max(plot_top + 10, h - bottom_pad)
        plot_h = plot_bottom - plot_top
        n = len(self._counts)
        col_w = max(2, w // max(1, n))
        for idx, count in enumerate(self._counts):
            x = idx * col_w
            bar_h = int((count / peak) * max(1, plot_h - line_h))
            top = plot_bottom - bar_h
            color = accent.lighter(100 + (idx % 5) * 6)
            painter.setPen(QPen(border))
            painter.setBrush(color)
            painter.drawRect(x + 1, top, col_w - 2, bar_h)
            if count > 0:
                painter.setPen(text_pen)
                painter.drawText(x, top - line_h, col_w, line_h,
                                 Qt.AlignHCenter | Qt.AlignBottom,
                                 _fmt_num(count))
        edges = self._edges
        if edges:
            painter.setPen(text_pen)
            ticks = [0, len(edges) // 2, len(edges) - 1]
            tick_labels = _fmt_tick_series([edges[i] for i in ticks])
            for i, label in zip(ticks, tick_labels):
                tx = min(max(0, i * col_w - 30), max(0, w - 60))
                painter.drawText(tx, plot_bottom + 2, 60, line_h - 4,
                                 Qt.AlignHCenter | Qt.AlignTop, label)
        painter.setPen(text_pen)
        if self._ylabel:
            painter.drawText(0, 0, w, line_h, Qt.AlignLeft | Qt.AlignTop,
                             _hist_header(self._ylabel, self._counts))
        if self._xlabel:
            painter.drawText(0, plot_bottom + line_h, w, line_h,
                             Qt.AlignRight | Qt.AlignTop, self._xlabel)
