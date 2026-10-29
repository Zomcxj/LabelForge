"""滑窗裁剪配置小窗：左参数（双向联动），右分离小图预览（绿=保留，红=空窗丢弃）。"""
import os

from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from .dwm import set_titlebar_dark
from .i18n import t as tr
from .theme import ThemeManager
from ..core.config_manager import load_crop_templates
from ..engine.augmenter.crop import (
    MAX_WINDOW, PAD_MAX_PERCENT, crop_boxes, exact_overlap_for_flush,
    nearest_pad_compliant_overlap, normalized_fill_rgb, normalized_window,
    normalized_window_axes,
    overlap_for_count, overlap_range_for_count, pad_compliant_overlap_run,
    pad_compliant_size_run, pad_overflow, size_for_count, size_for_flush,
    size_range_for_count, window_starts,
)


class _CropPreview(QWidget):
    """分离小图预览：各窗口拆成分离小图并排且均显示内容；绿=保留，红=空窗仅红框。"""

    _GAP = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 220)
        self._image = None
        self._rects = []  # ((x0, y0, cw, ch), kept, r, c)
        self._boxes = []
        self._fill_rgb = None  # QColor；None = 不填充（窗口不会越界）

    def set_content(self, image, rects, boxes=(), fill_rgb=None):
        self._image = image
        self._rects = list(rects or [])
        self._boxes = list(boxes or [])
        self._fill_rgb = fill_rgb
        self.update()

    def _draw_window(self, p, dst, x0, y0, cw, ch):
        """把原图 (x0,y0,cw,ch) 画进 dst；开启填充时先铺底色，越界区就露出填充色。"""
        if self._fill_rgb is None:
            p.drawImage(dst, self._image, QRect(x0, y0, cw, ch))
            return
        p.fillRect(dst, self._fill_rgb)
        src = QRect(x0, y0, cw, ch).intersected(self._image.rect())
        if src.isEmpty():
            return
        kx, ky = dst.width() / cw, dst.height() / ch
        p.drawImage(QRect(dst.left() + int(round((src.left() - x0) * kx)),
                          dst.top() + int(round((src.top() - y0) * ky)),
                          max(1, int(round(src.width() * kx))),
                          max(1, int(round(src.height() * ky)))),
                    self._image, src)

    def paintEvent(self, _):
        p = QPainter(self)
        if self._image is None or not self._rects:
            p.drawText(self.rect(), Qt.AlignCenter, tr("未配置"))
            return
        rows = max(item[2] for item in self._rects)
        cols = max(item[3] for item in self._rects)
        cw0, ch0 = self._rects[0][0][2], self._rects[0][0][3]
        if rows <= 0 or cols <= 0 or cw0 <= 0 or ch0 <= 0:
            return
        gap = self._GAP
        tw = (max(1, self.width()) - gap * (cols - 1)) / cols
        th = tw * ch0 / cw0
        max_h = max(1, self.height()) - gap * (rows - 1)
        if th * rows > max_h:  # 高度不够按比例缩
            k = max_h / (th * rows)
            tw *= k
            th *= k
        ox = (self.width() - (tw * cols + gap * (cols - 1))) / 2
        oy = (self.height() - (th * rows + gap * (rows - 1))) / 2
        font = p.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        p.setFont(font)
        ab = QColor("#FFD600")
        ab.setAlpha(128)
        for (x0, y0, cw, ch), kept, r, c in self._rects:
            dst = QRect(int(ox + (c - 1) * (tw + gap)),
                        int(oy + (r - 1) * (th + gap)),
                        max(1, int(tw)), max(1, int(th)))
            self._draw_window(p, dst, x0, y0, cw, ch)
            if self._boxes:
                sx, sy = dst.width() / cw, dst.height() / ch
                p.save()
                p.setClipRect(dst)
                p.setPen(QPen(ab, 2))
                for b in self._boxes:
                    try:
                        bx = float(b.get("x", 0))
                        by = float(b.get("y", 0))
                        bw = float(b.get("width", 0))
                        bh = float(b.get("height", 0))
                    except (TypeError, ValueError):
                        continue
                    if bw <= 0 or bh <= 0:
                        continue
                    p.drawRect(QRect(dst.left() + int((bx - x0) * sx),
                                     dst.top() + int((by - y0) * sy),
                                     max(1, int(bw * sx)), max(1, int(bh * sy))))
                p.restore()
            # 2px 细框在缩小后的预览图上几乎看不见。加粗到 3px，并沿边框
            # 外侧再描一圈同色半透明的光晕（而不是整块盖住窗口内部——那样
            # 会连同窗口内容与填充色一起染色）。
            rect_color = QColor("#4CAF50") if kept else QColor("#ff3b30")
            halo = QColor(rect_color)
            halo.setAlpha(70)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(halo, 7))
            p.drawRect(dst.adjusted(-2, -2, 2, 2))
            p.setPen(QPen(rect_color, 3))
            p.drawRect(dst)
            label_color = QColor("#ffffff")
            if kept and dst.width() >= 44 and dst.height() >= 30:
                p.setPen(label_color)
                p.drawText(dst.adjusted(2, 0, -2, -2),
                           Qt.AlignBottom | Qt.AlignLeft, f"r{r}c{c}")


class CropConfigDialog(QDialog):
    """裁剪参数配置；确定后面板取 spec() 写回 _crop_spec。"""

    _LOCK_MAX = 2                       # 最多同时锁定 2 组，第 3 个挤掉最早锁的
    _LOCK_KEYS = ("size", "ov", "cnt")
    _LOCK_ICON_ON = "\U0001F512"        # 🔒 已锁
    _LOCK_ICON_OFF = "\U0001F513"       # 🔓 未锁

    def __init__(self, images, boxes_by_index, spec, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("滑窗裁剪"))
        self.setMinimumSize(1270, 800)
        self.resize(1270, 800)
        self._images = list(images or [])
        self._boxes = boxes_by_index or {}
        self._linking = False
        self._lock_order = []           # 按点击先后排序的锁定队列
        self._infeasible = False        # 锁定数量不可达时为 True，摘要行给提示
        self._pad_clamped = False       # 填充模式吸附过超限值时摘要行给提示
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._refresh)

        self._w = QSpinBox(); self._w.setRange(8, MAX_WINDOW)
        self._h = QSpinBox(); self._h.setRange(8, MAX_WINDOW)
        self._ov_x = QSpinBox(); self._ov_x.setRange(0, 4096)
        self._ov_y = QSpinBox(); self._ov_y.setRange(0, 4096)
        self._ov = self._ov_x  # compatibility alias; new UI uses both axes
        self._rows = QSpinBox(); self._rows.setRange(1, 128)
        self._cols = QSpinBox(); self._cols.setRange(1, 128)
        self._vis = QSpinBox(); self._vis.setRange(50, 100); self._vis.setSuffix(" %")
        self._square = QCheckBox(tr("正方形"))
        self._fill = QCheckBox(tr("边缘填充"))
        self._fill_r = QSpinBox(); self._fill_r.setRange(0, 255)
        self._fill_g = QSpinBox(); self._fill_g.setRange(0, 255)
        self._fill_b = QSpinBox(); self._fill_b.setRange(0, 255)
        self._fill_swatch = QLabel(); self._fill_swatch.setFixedSize(34, 20)
        self._fill.setToolTip(tr("fill_hint"))
        self._lock_size = self._make_lock_btn()
        self._lock_ov = self._make_lock_btn()
        self._lock_cnt = self._make_lock_btn()
        self._img = QComboBox()
        self._est = QLabel()
        self._est.setWordWrap(True)
        self._preview = _CropPreview()

        self._w.setValue(int(spec["w"]))
        self._h.setValue(int(spec["h"]))
        legacy_ov = int(spec.get("overlap", 0))
        self._ov_x.setValue(int(spec.get("overlap_x", legacy_ov)))
        self._ov_y.setValue(int(spec.get("overlap_y", legacy_ov)))
        self._vis.setValue(int(round(spec.get("min_visible", 0.7) * 100)))
        self._square.setChecked(bool(spec.get("square")))
        self._fill.setChecked(bool(spec.get("fill")))
        self._fill_r.setValue(normalized_fill_rgb(spec)[0])
        self._fill_g.setValue(normalized_fill_rgb(spec)[1])
        self._fill_b.setValue(normalized_fill_rgb(spec)[2])
        self._sync_fill_ui()
        for p in self._images[:100]:
            self._img.addItem(os.path.basename(p))
        self._clamp_ov_max()
        self._sync_counts()
        # 初始 spec 也要过一遍补边上限：面板/模板传来的宽高可能本就超限，
        # 不在这里收口就会带着 33% 补边进界面，且没有任何提示。
        self._enforce_pad_cap("fill")

        self._w.valueChanged.connect(lambda _v: self._on_size("w"))
        self._h.valueChanged.connect(lambda _v: self._on_size("h"))
        self._ov_x.valueChanged.connect(lambda: self._on_overlap("x"))
        self._ov_y.valueChanged.connect(lambda: self._on_overlap("y"))
        self._rows.valueChanged.connect(lambda _v: self._on_count("rows"))
        self._cols.valueChanged.connect(lambda _v: self._on_count("cols"))
        self._square.toggled.connect(self._on_square)
        self._lock_size.toggled.connect(lambda on: self._on_lock("size", on))
        self._lock_ov.toggled.connect(lambda on: self._on_lock("ov", on))
        self._lock_cnt.toggled.connect(lambda on: self._on_lock("cnt", on))
        self._fill.toggled.connect(self._on_fill)
        for _sb in (self._fill_r, self._fill_g, self._fill_b):
            _sb.valueChanged.connect(self._on_fill_rgb)
        self._vis.valueChanged.connect(lambda _v: self._debounce.start())
        self._img.currentIndexChanged.connect(self._on_image)

        # ---- 顶部参数区：三列并排（原来是一整列竖排在左侧） ----
        # 三行的标签文字长短不一（裁剪尺寸 / 重叠 / 裁剪数量），各自 addWidget
        # 会让输入框起点差出 20~60px，整块看着是歪的。这里给标签统一宽度，
        # 输入框与 × 也统一宽度，三行的列就自然对齐了。
        _LABEL_W = 132
        _SPIN_W = 64
        _COL_W = _SPIN_W * 2 + 22       # 两个输入框 + 中间的 ×
        _LOCK_W = 28
        _GAP = 6

        def _row(label_text, left, right, lock, extra=None):
            """构造一行：定宽标签 + 两个定宽输入框 + 锁（列对齐）。"""
            lay = QHBoxLayout()
            lay.setSpacing(_GAP)
            lbl = QLabel(label_text)
            lbl.setFixedWidth(_LABEL_W)
            lay.addWidget(lbl)
            for _sb in (left, right):
                _sb.setFixedWidth(_SPIN_W)
            lay.addWidget(left)
            times = QLabel("×")
            times.setFixedWidth(10)
            times.setAlignment(Qt.AlignCenter)
            lay.addWidget(times)
            lay.addWidget(right)
            if extra is not None:
                lay.addWidget(extra)
            # 用 addStretch 把「参数组 + 锁」这一整块推到列宽边界，锁在三行里
            # 才会落在同一条竖线上；只靠前面的定宽做不到——行列那两个框更窄。
            lay.addStretch()
            lay.addWidget(lock, 0, Qt.AlignVCenter)
            lay.addSpacing(_GAP)
            return lay

        col_a = QVBoxLayout()
        col_a.setSpacing(4)
        self._square.setText(tr("正方形"))
        col_a.addLayout(_row(tr("裁剪尺寸（像素）"), self._w, self._h,
                             self._lock_size, extra=self._square))
        col_a.addLayout(_row(tr("重叠（像素）"), self._ov_x, self._ov_y,
                             self._lock_ov))
        col_a.addLayout(_row(tr("裁剪数量（行 × 列）"), self._rows, self._cols,
                             self._lock_cnt))
        for _sb in (self._rows, self._cols):
            _sb.setFixedWidth(46)       # 行列只到 128，3 位数够用
        col_a.addStretch()

        col_b = QVBoxLayout()
        row = QHBoxLayout()
        self._vis.setFixedWidth(64)     # 百分比最多 3 位 + 后缀
        row.addWidget(QLabel(tr("最小可见比例（%）"))); row.addWidget(self._vis); row.addStretch()
        col_b.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self._fill)
        row.addWidget(QLabel("RGB"))
        for _sb in (self._fill_r, self._fill_g, self._fill_b):
            _sb.setFixedWidth(52)       # 0~255 三位数
        row.addWidget(self._fill_r); row.addWidget(self._fill_g); row.addWidget(self._fill_b)
        row.addWidget(self._fill_swatch)
        row.addStretch()
        col_b.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("预览图")))
        self._img.setFixedWidth(220)
        row.addWidget(self._img)
        row.addStretch()
        col_b.addLayout(row)
        self._dims = QLabel()
        col_b.addWidget(self._dims)
        col_b.addStretch()

        col_c = QVBoxLayout()
        templates = load_crop_templates()
        if templates:
            col_c.addWidget(QLabel(tr("模板")))
            for item in templates:
                txt = tr("template_fmt").format(
                    w=item["w"], h=item["h"], ov=item["overlap"],
                    vis=int(round(item["min_visible"] * 100)))
                if item["square"]:
                    txt += "·" + tr("正方形")
                if item.get("fill"):
                    txt += "·" + tr("补边")
                btn = QPushButton(txt)
                btn.clicked.connect(lambda _c=False, spec=item: self._apply_template(spec))
                col_c.addWidget(btn)
        col_c.addStretch()

        top = QHBoxLayout()
        top.addLayout(col_a, 1)
        top.addSpacing(18)
        top.addLayout(col_b, 1)
        top.addSpacing(18)
        top.addLayout(col_c, 1)

        btns = QHBoxLayout()
        # 图例只用带色块的标签，不再另附一行「绿=保留 · 红=空窗丢弃」：
        # 色块里已经写了保留/空窗丢弃，再加一句是同一件事说两遍。
        btns.addWidget(self._make_legend_chip("#4CAF50", tr("保留")))
        btns.addWidget(self._make_legend_chip("#ff3b30", tr("空窗丢弃")))
        btns.addStretch()
        ok = QPushButton(tr("确定")); ok.setObjectName("successBtn"); ok.clicked.connect(self.accept)
        cancel = QPushButton(tr("取消")); cancel.setObjectName("accentBtn"); cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)

        # 参数在上 → 预览图居中占满剩余空间 → 底部红绿说明与确定/取消不变
        main = QVBoxLayout(self)
        main.addLayout(top)
        main.addWidget(self._est)
        main.addWidget(self._preview, 1)
        main.addLayout(btns)
        self._refresh()

    # ---- 工具 ----

    def _make_lock_btn(self):
        """🔒/🔓 图标按钮，点击上锁/解锁。

        用 checkable QPushButton 而非 QCheckBox：isChecked/setChecked/toggled 语义完全一致，
        既有联动与测试无需改动，同时图标随状态切换、不再占一行文字宽度。
        """
        btn = QPushButton(self._LOCK_ICON_OFF)
        btn.setCheckable(True)
        btn.setFixedSize(28, 22)
        btn.setToolTip(tr("lock_hint"))
        btn.setCursor(Qt.PointingHandCursor)
        btn.toggled.connect(lambda on, b=btn: b.setText(
            self._LOCK_ICON_ON if on else self._LOCK_ICON_OFF))
        return btn

    def _make_legend_chip(self, color, text):
        """图例色块：一个 8×8 圆角色块 + 12px 文字。"""
        chip = QLabel(f'<span style="color:{color};">&#9632;</span>&nbsp;{text}')
        chip.setTextFormat(Qt.RichText)
        font = chip.font()
        font.setPointSizeF(max(9.0, font.pointSizeF()))
        chip.setFont(font)
        return chip

    def _set(self, spin, value):
        spin.blockSignals(True)
        spin.setValue(int(value))
        spin.blockSignals(False)

    def _current_path(self):
        i = self._img.currentIndex()
        return self._images[i] if 0 <= i < len(self._images) else None

    def _ref_size(self):
        path = self._current_path()
        if path:
            img = QImage(path)
            if not img.isNull():
                return img.width(), img.height()
        return 0, 0

    def _update_dims(self):
        w, h = self._ref_size()
        self._dims.setText(f"{w} × {h}" if w and h else "")

    # ---- 双向联动 ----

    def _apply_template(self, spec):
        """套用最近模板：回填全部参数（含填充），其余按当前锁定状态重算。

        回填前必须先放开区间：上一轮的补边硬墙/锁定区间还留在控件上，
        setValue(800) 会被旧范围静默钳成 352（模板值根本没生效）。
        """
        controls = (self._w, self._h, self._ov_x, self._ov_y, self._rows, self._cols,
                    self._vis, self._square, self._fill,
                    self._fill_r, self._fill_g, self._fill_b)
        self._relax_ranges()
        for c in controls:
            c.blockSignals(True)
        self._w.setValue(int(spec["w"]))
        self._h.setValue(int(spec["h"]))
        legacy_ov = int(spec.get("overlap", 0))
        self._ov_x.setValue(int(spec.get("overlap_x", legacy_ov)))
        self._ov_y.setValue(int(spec.get("overlap_y", legacy_ov)))
        self._vis.setValue(int(round(spec["min_visible"] * 100)))
        self._square.setChecked(bool(spec["square"]))
        self._fill.setChecked(bool(spec.get("fill")))
        r, g, b = normalized_fill_rgb(spec)
        self._fill_r.setValue(r)
        self._fill_g.setValue(g)
        self._fill_b.setValue(b)
        if self._square.isChecked():
            v = min(self._w.value(), self._h.value())
            self._w.setValue(v)
            self._h.setValue(v)
        for c in controls:
            c.blockSignals(False)
        self._sync_fill_ui()
        self._linking = True
        try:
            self._sync_counts()
            self._apply_readonly()
        finally:
            self._linking = False
        self._debounce.start()

    def _ov_hi_limit(self):
        """重叠的硬上限（半窗 / 锁数量时 窗口−1）；填充模式的合规段也要叠加它。"""
        base = max(0, min(self._w.value(), self._h.value()))
        return base - 1 if "cnt" in self._lock_order else base // 2

    def _clamp_ov_max(self):
        """重叠上限：数量锁定时放宽到 窗口−1（数量已锁死不会爆炸），否则半窗。"""
        limit = max(0, self._ov_hi_limit())
        self._ov_x.setMaximum(limit)
        self._ov_y.setMaximum(limit)

    def _axis_count(self, length, size, ov):
        return len(window_starts(length, size, max(0, min(ov, size - 1))))

    def _exact_size(self, length, size0, req, ov):
        """窗口数对尺寸单调非增：返回使窗口数最接近 req 的尺寸（可精确则精确）。"""
        if self._axis_count(length, size0, ov) == req:
            return size0
        lo = max(1, min(ov, length - 1) + 1)
        a, b = lo, length
        while a < b:
            mid = (a + b) // 2
            if self._axis_count(length, mid, ov) <= req:
                b = mid
            else:
                a = mid + 1
        if self._axis_count(length, a, ov) == req:
            return a
        if a > lo and abs(self._axis_count(length, a - 1, ov) - req) < \
                abs(self._axis_count(length, a, ov) - req):
            return a - 1
        return a

    def _sync_counts(self):
        rw, rh = self._ref_size()
        if not rw:
            return
        sp = self.spec()
        cw, ch, ov_x, ov_y = normalized_window_axes(rw, rh, sp)
        fill = bool(sp["fill"])
        self._set(self._rows, len(window_starts(rh, ch, ov_y, fill)))
        self._set(self._cols, len(window_starts(rw, cw, ov_x, fill)))

    # ---- 锁定：按点击先后排队，最多 2 个 ----

    def _lock_box(self, key):
        return {"size": self._lock_size, "ov": self._lock_ov,
                "cnt": self._lock_cnt}[key]

    def _on_lock(self, key, on):
        """点第 3 个锁时挤掉最早锁的那个（轮转队列）。"""
        if self._linking:
            return
        self._linking = True
        try:
            if key in self._lock_order:
                self._lock_order.remove(key)
            if on:
                self._lock_order.append(key)
            while len(self._lock_order) > self._LOCK_MAX:
                self._lock_box(self._lock_order.pop(0)).setChecked(False)
            self._apply_readonly()
            self._relayout("lock")
        finally:
            self._linking = False
        self._debounce.start()

    def _apply_readonly(self):
        """锁满 2 个时管理第三项的自由度：区间可编辑，而不是一律置灰只读。

        锁定两项**不等于**唯一确定第三项：
          · 锁「重叠+数量」→ 尺寸是一段区间（如 1920 图上锁 3 列/重叠 88 有 305 个可行值）
          · 锁「尺寸+数量」→ 重叠同理（两轴共用重叠，故取两轴可行区间的交集）
          · 锁「尺寸+重叠」→ 数量是纯结果，没有任何自由度，才该只读
        把输入范围收窄到可行区间，用户可以在区间内自由调，两项锁定永不被破坏。
        """
        full = len(self._lock_order) >= self._LOCK_MAX
        rw, rh = self._ref_size()
        if not full or not rw:
            self._w.setRange(8, MAX_WINDOW)
            self._h.setRange(8, MAX_WINDOW)
            self._ov_x.setRange(0, 4096)
            self._ov_y.setRange(0, 4096)
            for wdg in (self._rows, self._cols):
                wdg.setReadOnly(False)
            self._clamp_ov_max()
            return

        if "size" not in self._lock_order:        # 尺寸是派生量：区间可编辑
            fill = self._fill.isChecked()
            lo_w, hi_w = size_range_for_count(rw, self._cols.value(), self._ov.value(),
                                              fill=fill)
            lo_h, hi_h = size_range_for_count(rh, self._rows.value(), self._ov.value(),
                                              fill=fill)
            self._set_range(self._w, lo_w, hi_w)
            self._set_range(self._h, lo_h, hi_h)
        else:
            self._w.setRange(8, MAX_WINDOW)
            self._h.setRange(8, MAX_WINDOW)

        if "ov" not in self._lock_order:          # 重叠是派生量：区间可编辑（取两轴交集）
            fill = self._fill.isChecked()
            lo_w, hi_w = overlap_range_for_count(rw, self._cols.value(), self._w.value(),
                                                 fill=fill)
            lo_h, hi_h = overlap_range_for_count(rh, self._rows.value(), self._h.value(),
                                                 fill=fill)
            self._set_range(self._ov_x, 0, 4096)
            self._set_range(self._ov_y, 0, 4096)
        else:
            self._ov_x.setRange(0, 4096)
            self._ov_y.setRange(0, 4096)
            self._clamp_ov_max()

        ro_cnt = full and "cnt" not in self._lock_order
        for wdg in (self._rows, self._cols):
            wdg.setReadOnly(ro_cnt)

    def _relax_ranges(self):
        """放开上一轮的区间收窄，回到控件原始范围。

        与 _set_ov 先放开重叠上限同理：不放开，本轮的派生值就会被上一轮的区间钳掉。
        收尾由 _apply_readonly 按当前锁定状态重新收窄。
        """
        for spin in (self._w, self._h):
            spin.setRange(8, MAX_WINDOW)
        self._ov.setMaximum(4096)

    def _set_range(self, spin, lo, hi):
        """把 spin 的范围收窄到 [lo, hi]；无解时退化为空区间并保持当前值可见。"""
        if hi < lo:
            lo = hi = spin.value()
        spin.setRange(int(lo), int(hi))

    # ---- 三量互推 ----

    def _set_ov(self, ov):
        """赋重叠前先放开上限，避免被旧上限静默钳掉；收尾由 _clamp_ov_max 回钳。"""
        self._ov_x.setMaximum(4096)
        self._ov_y.setMaximum(4096)
        self._set(self._ov_x, ov)
        self._set(self._ov_y, ov)

    def _size_from_count(self, length, count, ov):
        return self._exact_size(length, size_for_count(length, count, ov), count, ov)

    def _derive_sizes_from_ov(self, rw, rh, ov):
        """重叠 + 数量 → 两轴尺寸。"""
        w = self._size_from_count(rw, self._cols.value(), ov)
        h = self._size_from_count(rh, self._rows.value(), ov)
        self._set(self._w, w)
        self._set(self._h, w if self._square.isChecked() else h)

    def _derive_size_for_count(self, axis):
        """数量 → 该轴尺寸（重叠保持不动）。"""
        rw, rh = self._ref_size()
        ov = self._ov.value()
        if axis == "cols":
            w = self._size_from_count(rw, self._cols.value(), ov)
            self._set(self._w, w)
            if self._square.isChecked():
                self._set(self._h, w)
        else:
            h = self._size_from_count(rh, self._rows.value(), ov)
            self._set(self._h, h)
            if self._square.isChecked():
                self._set(self._w, h)

    def _derive_ov_from(self, axis, rw, rh):
        """数量 + 尺寸 → 重叠。重叠两轴共用，须择一轴驱动。"""
        if axis == "h":
            ov = overlap_for_count(rh, self._rows.value(), self._h.value())
        else:
            ov = overlap_for_count(rw, self._cols.value(), self._w.value())
        self._set_ov(ov)

    def _derive_flush_from_size(self, source, rw, rh):
        """不开填充：由刚编辑的那一轴反推共用重叠，再让另一轴也恰好铺满。

        约束是 n·c − (n−1)·ov = L，三量联动：重叠两轴共用一个值，故须择一轴驱动
        （改宽用宽轴解、改高用高轴解），另一轴再按该重叠反推，于是两轴同时贴边。
        某轴在目标数量下无整数解时保持不动，由 _sync_counts 显示实际数量。
        """
        cols, rows = self._cols.value(), self._rows.value()
        if source == "h":
            ov = exact_overlap_for_flush(rh, rows, self._h.value())
            other_axis, other_len, other_cnt = "w", rw, cols
        else:                                   # w / square / lock / image / fill
            ov = exact_overlap_for_flush(rw, cols, self._w.value())
            other_axis, other_len, other_cnt = "h", rh, rows
        if ov is None:
            return                              # 当前数量下无精确解，保持用户输入
        self._set_ov(ov)
        c = size_for_flush(other_len, other_cnt, ov)
        if c is not None:
            self._set(self._w if other_axis == "w" else self._h, c)

    def _derive_flush_from_ov(self, rw, rh):
        """不开填充：由重叠反推两轴尺寸，两轴都恰好铺满。

        两轴都要解 n·c − (n−1)·ov = L：同一重叠未必让两轴同时整除，整除的那轴
        精确铺满，不整除的那轴用 ceil 取最小可行窗口（末窗贴边、数量仍达标），
        保证「改重叠」不会覆盖数量。
        """
        for spin, length, count in ((self._w, rw, self._cols.value()),
                                    (self._h, rh, self._rows.value())):
            c = size_for_flush(length, count, self._ov.value())
            if c is None:
                c = self._size_from_count(length, count, self._ov.value())
            if c:
                self._set(spin, c)
        if self._square.isChecked():
            self._apply_square()

    def _apply_square(self):
        if self._square.isChecked():
            v = min(self._w.value(), self._h.value())
            self._set(self._w, v)
            self._set(self._h, v)

    def _relayout(self, source):
        """按锁定状态重算派生量；source 是刚被编辑的字段。

        锁定项永不被自动改，未锁项按锁定项反推；编辑某轴时由该轴驱动共用重叠，
        另一轴再按新重叠反推。调用方需自行持有 _linking。
        """
        rw, rh = self._ref_size()
        if not rw:
            return
        # 先解除上一轮的区间收窄：否则新算出的派生值会被旧范围静默截断
        # （例如锁 cnt+ov 后改重叠，宽高反推值 734 会被旧下界 800 覆盖）。
        self._relax_ranges()
        locked = set(self._lock_order)
        cnt_l, ov_l, size_l = "cnt" in locked, "ov" in locked, "size" in locked
        want = (self._rows.value(), self._cols.value())

        if cnt_l and ov_l:                       # 尺寸是唯一未知量（区间）
            if source not in ("w", "h", "square"):
                # 上下文变了（换图/开关填充/刚上锁）才反推一个具体尺寸；
                # 用户手动调尺寸时必须尊重其选择，否则区间可编辑形同虚设
                self._derive_sizes_from_ov(rw, rh, self._ov.value())
        elif cnt_l and size_l:                   # 重叠是唯一未知量（区间）
            if source != "ov":
                self._derive_ov_from("h" if source == "h" else "w", rw, rh)
        elif size_l and ov_l:                    # 数量是唯一未知量
            pass
        elif cnt_l:                              # 只锁数量：改一轴 → 反推重叠 → 反推另一轴
            if source == "ov":
                self._derive_sizes_from_ov(rw, rh, self._ov.value())
            elif source in ("h", "rows"):
                self._derive_ov_from("h", rw, rh)
                self._set(self._w, self._size_from_count(
                    rw, self._cols.value(), self._ov.value()))
            else:                                # w / cols / square / lock / image
                self._derive_ov_from("w", rw, rh)
                self._set(self._h, self._size_from_count(
                    rh, self._rows.value(), self._ov.value()))
            self._apply_square()
        elif ov_l:                               # 只锁重叠：尺寸 ↔ 数量
            if source in ("cols", "rows"):
                self._derive_size_for_count(source)
        elif size_l:                             # 只锁尺寸：改数量 → 反推重叠
            if source in ("cols", "rows"):
                self._derive_ov_from("h" if source == "rows" else "w", rw, rh)
        else:                                    # 无锁
            if source in ("cols", "rows"):
                self._derive_size_for_count(source)
            elif source == "ov" and not self._fill.isChecked():
                self._derive_flush_from_ov(rw, rh)
            elif source in ("w", "h", "square") and not self._fill.isChecked():
                # 不开填充：宽高与重叠互相反推，两轴都保持恰好铺满（同步）。
                # 开填充则不反推——等步长落位下两轴不可能都精确，差多少补多少，
                # 强行反推会覆盖用户设定的宽高。
                # 仅在宽高确实被手动修改时反推：套模板/换图/开关填充时宽高是给定的，
                # 派生会直接把模板值改掉。
                self._derive_flush_from_size(source, rw, rh)

        # 范围与只读状态统一在这里收口：锁定状态下由 _apply_readonly 按可行区间收窄，
        # 未锁满时它内部会回落到 _clamp_ov_max 的半窗口径
        self._apply_readonly()
        self._enforce_pad_cap(source)
        self._sync_counts()                      # 写回真实数量（不可达时即为提示）
        self._infeasible = cnt_l and want != (self._rows.value(), self._cols.value())

    def _enforce_pad_cap(self, source):
        """填充模式：补边不得超窗口 30%——超限值吸附到最近合规值，并收紧输入硬墙。

        合规集合在固定重叠下是锯齿状多段（窗口数每变一次 pad 就跳回 0），故：
          · 吸附：把违规值回退到最近（等距取较小）的合规值，只动刚编辑的那一项；
          · 硬墙：把宽高/重叠的范围钳到「当前值所在的那一段」，用户拖不出超限值。
        不开填充时不做任何处理（也不留提示）。
        """
        if not self._fill.isChecked():
            self._pad_clamped = False
            return
        rw, rh = self._ref_size()
        if not rw:
            return
        ov = self._ov.value()
        lim = self._ov_hi_limit()
        clamped = False
        # ① 重叠：两轴共用一个值，须同时满足两轴；也须尊重半窗上限（两者叠加）。
        # 改重叠本身、开关填充、套模板、换图、上锁后都要重算。
        # 宽高不在这里吸附：③ 的 setRange 会以同样的合规段钳住越界值（等价，已对拍验证）。
        if source in ("ov", "fill", "template", "image", "lock", None):
            fixed = nearest_pad_compliant_overlap(rw, rh, self._w.value(),
                                                  self._h.value(), ov, lim)
            if fixed != ov:
                self._set_ov(fixed)
                clamped = True
        # ② 硬墙：把三项范围都钳到当前值所在段（放最后，用吸附后的值定位）。
        # 收窄范围本身也会钳住越界的当前值，同样算「吸附过」，必须提示。
        if self._apply_pad_walls(rw, rh):
            clamped = True
        # 每轮都重算：提示要能随「调到合规值」自动消失，不能常驻
        self._pad_clamped = clamped or not self._pad_compliant(rw, rh)

    def _pad_compliant(self, rw, rh) -> bool:
        ov = self._ov.value()
        return (pad_overflow(rw, self._w.value(), ov) * 100
                <= PAD_MAX_PERCENT * self._w.value()
                and pad_overflow(rh, self._h.value(), ov) * 100
                <= PAD_MAX_PERCENT * self._h.value())

    def _apply_pad_walls(self, rw, rh) -> bool:
        """把宽高/重叠的输入范围钳到「当前值所在的合规段」——用户拖不出超限值。

        重叠的段必须同时叠加半窗上限：否则段会越过该上限，随后 _clamp_ov_max 再钳一次
        就得到空区间，重叠值会被覆盖（实测 114 被钳到 100 后合规段失效）。
        返回是否真的钳动了当前值（setRange 会顺带钳值，这属于静默改动，须提示）。
        """
        ov = self._ov.value()
        changed = False
        for spin, length in ((self._w, rw), (self._h, rh)):
            before = spin.value()
            lo, hi = pad_compliant_size_run(length, before, ov)
            if hi >= lo:
                self._set_range(spin, lo, hi)
                if spin.value() != before:
                    changed = True
        before_ov = self._ov.value()
        # 段必须叠加半窗上限，否则 setRange 会把范围放宽到上限之外
        # （实测 (60,62) 的段为 (30,31)，半窗上限 30，不叠加就放行了 31）
        lim = self._ov_hi_limit()
        o_lo, o_hi = pad_compliant_overlap_run(rw, rh, self._w.value(),
                                               self._h.value(), before_ov, lim)
        if o_hi >= o_lo:
            self._set_range(self._ov, o_lo, min(o_hi, lim))
            if self._ov.value() != before_ov:
                changed = True
        else:
            self._clamp_ov_max()
        return changed

    # ---- 控件回调 ----

    def _on_size(self, axis):
        if self._linking:
            return
        self._linking = True
        try:
            if self._apply_square():
                axis = "w"
            rw, rh = self._ref_size()
            if rw and rh:
                if axis == "w":
                    self._set(self._ov_x, overlap_for_count(
                        rw, self._cols.value(), self._w.value()))
                else:
                    self._set(self._ov_y, overlap_for_count(
                        rh, self._rows.value(), self._h.value()))
                self._sync_counts()
        finally:
            self._linking = False
        self._debounce.start()

    def _on_overlap(self, axis="x"):
        if self._linking:
            return
        self._linking = True
        try:
            if axis == "y":
                rw, rh = self._ref_size()
                if rh:
                    self._relayout_axis("y", rh, self._rows.value(), self._ov_y.value())
            else:
                ov = self._ov_x.value()
                if "size" not in self._lock_order and ov >= self._w.value():
                    self._set(self._w, ov + 1)
            rw, rh = self._ref_size()
            if rw and rh:
                if axis == "x":
                    self._relayout_axis("x", rw, self._cols.value(), self._ov_x.value())
                self._sync_counts()
        finally:
            self._linking = False
        self._debounce.start()

    def _relayout_axis(self, axis, length, count, overlap):
        """Recompute one dimension while preserving its row/column count."""
        minimum = max(8, overlap + 1)
        size = max(minimum, (length + (count - 1) * overlap + count - 1) // count)
        starts = window_starts(length, size, overlap, fill=self._fill.isChecked())
        if len(starts) != count:
            size = self._size_from_count(length, count, overlap)
        self._set(self._h if axis == "y" else self._w, size)

    def _on_count(self, axis):
        if self._linking:
            return
        rw, _rh = self._ref_size()
        if not rw:
            return
        self._linking = True
        try:
            rw, rh = self._ref_size()
            if axis == "cols":
                self._relayout_axis("x", rw, self._cols.value(), self._ov_x.value())
            else:
                self._relayout_axis("y", rh, self._rows.value(), self._ov_y.value())
            self._sync_counts()
        finally:
            self._linking = False
        self._debounce.start()

    def _on_square(self, checked):
        if self._linking:
            return
        self._linking = True
        try:
            if checked:
                self._apply_square()
            self._relayout("square")
        finally:
            self._linking = False
        self._debounce.start()

    def _on_image(self, _idx):
        self._update_dims()
        self._linking = True
        try:
            self._relayout("image")
        finally:
            self._linking = False
        self._debounce.start()

    # ---- 边缘填充 ----

    def _fill_rgb(self):
        return (self._fill_r.value(), self._fill_g.value(), self._fill_b.value())

    def _sync_fill_ui(self):
        """未勾选填充时 RGB 输入禁用；色块始终显示当前色值。"""
        on = self._fill.isChecked()
        for sb in (self._fill_r, self._fill_g, self._fill_b):
            sb.setEnabled(on)
        r, g, b = self._fill_rgb()
        self._fill_swatch.setStyleSheet(
            f"background-color: rgb({r},{g},{b});"
            "border:1px solid rgba(128,128,128,180);border-radius:4px;")

    def _on_fill(self, _checked):
        """填充开关改变网格口径（等步长可越界 vs 贴边不越界），须重排。"""
        self._sync_fill_ui()
        self._linking = True
        try:
            self._relayout("fill")
        finally:
            self._linking = False
        self._debounce.start()

    def _on_fill_rgb(self, _v):
        self._sync_fill_ui()
        self._debounce.start()

    # ---- 预览与结果 ----

    def _refresh(self):
        self._update_dims()
        path = self._current_path()
        if not path:
            return
        img = QImage(path)
        if img.isNull():
            return
        iw, ih = img.width(), img.height()
        sp = self.spec()
        cw, ch, ov_x, ov_y = normalized_window_axes(iw, ih, sp)
        fill = bool(sp["fill"])
        boxes = self._boxes.get(self._img.currentIndex(), [])
        min_vis = self._vis.value() / 100.0
        rects = []
        kept = 0
        for r, y0 in enumerate(window_starts(ih, ch, ov_y, fill), 1):
            for c, x0 in enumerate(window_starts(iw, cw, ov_x, fill), 1):
                has = bool(crop_boxes(boxes, x0, y0, cw, ch, min_vis))
                rects.append(((x0, y0, cw, ch), has, r, c))
                kept += 1 if has else 0
        self._preview.set_content(img, rects, boxes,
                                  QColor(*sp["fill_rgb"]) if fill else None)
        total_n = len(self._images)
        est_n = kept * len(self._images)
        txt = tr("crop_estimate").format(total=total_n, est=est_n)
        # 摘要行里这两个数字是结论数据，其余是说明文字。把数字加粗
        # 并套强调色，扫一眼就能拿到，不用读完整句。
        accent = ThemeManager.get_theme()["accent"]
        txt = txt.replace(str(total_n), f'<b style="color:{accent}">{total_n}</b>', 1)
        txt = txt.replace(str(est_n), f'<b style="color:{accent}">{est_n}</b>', 1)
        txt = "<br>".join(txt.split("\n"))
        if self._infeasible:
            txt = tr("crop_infeasible") + "<br>" + txt
        if self._pad_clamped:
            txt = tr("crop_pad_clamped").format(pct=PAD_MAX_PERCENT) + "<br>" + txt
        self._est.setText(txt)

    def spec(self):
        return {"w": self._w.value(), "h": self._h.value(),
                "overlap": self._ov_x.value(),
                "overlap_x": self._ov_x.value(),
                "overlap_y": self._ov_y.value(),
                "min_visible": self._vis.value() / 100.0,
                "square": self._square.isChecked(),
                "fill": self._fill.isChecked(),
                "fill_rgb": self._fill_rgb()}

    def showEvent(self, event):
        super().showEvent(event)
        try:
            set_titlebar_dark(int(self.winId()), ThemeManager.get_mode().name == "DARK")
        except Exception:
            pass
