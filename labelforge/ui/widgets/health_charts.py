"""数据集健康自绘图表：QPainter 条形图 / 直方图，无第三方依赖。"""
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen

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
            left_w = min(max([fm.horizontalAdvance(s) for s in labels] + [0]) + 8,
                         max(20, w // 3))
            right_w = min(max([fm.horizontalAdvance(s) for s in value_labels] + [0]) + 8,
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

    def _edge_label(self, idx):
        """第 idx 个柱子的区间文字，用于 tooltip。"""
        edges = self._edges or []
        try:
            lo = edges[idx]
            hi = edges[idx + 1] if idx + 1 < len(edges) else None
        except IndexError:
            return ''
        lo_s = _fmt_tick_series([lo])[0]
        if hi is None:
            return lo_s
        return f"{lo_s} ~ {_fmt_tick_series([hi])[0]}"

    def mouseMoveEvent(self, event):
        """窄柱不标数字时，用 tooltip 给出该柱的区间与计数。"""
        hits = getattr(self, '_bar_hits', None)
        if self._mode == 'hist' and hits:
            x = event.position().x()
            for x0, x1, count, label in hits:
                if x0 <= x < x1:
                    self.setToolTip(
                        f"{label}: {_fmt_num(count)}" if label else _fmt_num(count))
                    break
            else:
                self.setToolTip('')
        super().mouseMoveEvent(event)

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
        fmc = painter.fontMetrics()
        # 柱顶计数只横向写。柱子窄到放不下时**不标**，而不是竖排——竖排的
        # 数字要一位一位往下读，比不标更难用。不标的柱子靠悬停 tooltip 补足。
        self._bar_hits = []          # (x0, x1, 计数) 供 mouseMoveEvent 查
        last_label_right = {}
        for idx, count in enumerate(self._counts):
            x = idx * col_w
            bar_h = int((count / peak) * max(1, plot_h - line_h))
            top = plot_bottom - bar_h
            color = accent.lighter(100 + (idx % 5) * 6)
            painter.setPen(QPen(border))
            painter.setBrush(color)
            painter.drawRect(x + 1, top, col_w - 2, bar_h)
            self._bar_hits.append((x, x + col_w, count,
                                   self._edge_label(idx)))
            if count <= 0:
                continue
            text = _fmt_num(count)
            tw = fmc.horizontalAdvance(text)
            if tw > col_w - 2:
                continue             # 柱子比数字还窄：留白，别硬塞
            painter.setPen(text_pen)
            tx = x + col_w // 2 - tw // 2
            # 与同一行上相邻标签重叠就跳过，宁可少标也不要糊成一团
            key = int((top - line_h) // max(1, line_h))
            if tx < last_label_right.get(key, -10 ** 9) + 3:
                continue
            painter.drawText(tx, top - line_h, tw, line_h,
                             Qt.AlignLeft | Qt.AlignBottom, text)
            last_label_right[key] = tx + tw
        edges = self._edges
        if edges:
            painter.setPen(text_pen)
            # 原来只标首/中/末三个刻度，且中间那个落在 `len(edges)//2` 上，
            # 与柱子的实际边界对不齐（柱子 x = idx*col_w，刻度却画在 idx*col_w-30
            # 处居中），看起来像错位。改成标在柱边界上，并按可用宽度决定
            # 标几个：太窄就只标首末，避免标签互相压住。
            n_edges = len(edges)
            fm = painter.fontMetrics()
            widest = max((fm.horizontalAdvance(_fmt_tick_series([e])[0])
                          for e in edges), default=0)
            # 每个刻度至少隔 4 根柱子：只按「文字宽度能否放下」算会让 900px 宽
            # 的图上标出十几个刻度（截图里 0.2k 47.4k 94.6k 挤成一排）。
            spacing_slots = max(1, (widest + 18) // max(1, col_w))
            max_ticks = max(2, n_edges // max(4, spacing_slots))
            if n_edges <= max_ticks:
                tick_idx = list(range(n_edges))
            elif max_ticks <= 2:
                tick_idx = [0, n_edges - 1]
            else:
                # 均匀取点，首末必取，中间按步长
                step = (n_edges - 1) / (max_ticks - 1)
                tick_idx = sorted({int(round(k * step)) for k in range(max_ticks)})
            tick_labels = _fmt_tick_series([edges[i] for i in tick_idx])
            last_right = -10 ** 9
            for i, label in zip(tick_idx, tick_labels):
                lw = fm.horizontalAdvance(label)
                # 刻度画在柱边界 i*col_w 处居中；首末两枚改用左右对齐，
                # 否则会被窗口边缘裁掉一半。
                if i == 0:
                    tx = 0
                elif i == n_edges - 1:
                    tx = max(0, w - lw)
                else:
                    tx = int(i * col_w - lw / 2)
                tx = min(max(0, tx), max(0, w - lw))
                if tx < last_right + 6:
                    continue          # 与前一个标签挤在一起就跳过
                painter.drawText(tx, plot_bottom + 2, lw, line_h - 4,
                                 Qt.AlignLeft | Qt.AlignTop, label)
                last_right = tx + lw
        painter.setPen(text_pen)
        if self._ylabel:
            painter.drawText(0, 0, w, line_h, Qt.AlignLeft | Qt.AlignTop,
                             _hist_header(self._ylabel, self._counts))
        if self._xlabel:
            painter.drawText(0, plot_bottom + line_h, w, line_h,
                             Qt.AlignRight | Qt.AlignTop, self._xlabel)
