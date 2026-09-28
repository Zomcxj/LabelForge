"""滑窗裁剪配置小窗：左参数（双向联动），右分离小图预览（绿=保留，红=空窗丢弃）。"""
import os

from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from .dwm import set_titlebar_dark
from .i18n import t as tr
from .theme import ThemeManager
from ..core.config_manager import load_crop_templates
from ..engine.augmenter.crop import (
    crop_boxes, normalized_window, size_for_count, window_starts,
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

    def set_content(self, image, rects, boxes=()):
        self._image = image
        self._rects = list(rects or [])
        self._boxes = list(boxes or [])
        self.update()

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
            p.drawImage(dst, self._image, QRect(x0, y0, cw, ch))
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
            p.setPen(QPen(QColor("#4CAF50") if kept else QColor("#ff3b30"), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(dst)
            if kept and dst.width() >= 44 and dst.height() >= 30:
                p.setPen(QColor("#ffffff"))
                p.drawText(dst.adjusted(2, 0, -2, -2),
                           Qt.AlignBottom | Qt.AlignLeft, f"r{r}c{c}")


class CropConfigDialog(QDialog):
    """裁剪参数配置；确定后面板取 spec() 写回 _crop_spec。"""

    def __init__(self, images, boxes_by_index, spec, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("滑窗裁剪设置"))
        self.setMinimumSize(1170, 600)
        self.resize(1170, 600)
        self._images = list(images or [])
        self._boxes = boxes_by_index or {}
        self._linking = False
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._refresh)

        self._w = QSpinBox(); self._w.setRange(8, 8192)
        self._h = QSpinBox(); self._h.setRange(8, 8192)
        self._ov = QSpinBox(); self._ov.setRange(0, 4096)
        self._rows = QSpinBox(); self._rows.setRange(1, 128)
        self._cols = QSpinBox(); self._cols.setRange(1, 128)
        self._vis = QSpinBox(); self._vis.setRange(50, 100); self._vis.setSuffix(" %")
        self._square = QCheckBox(tr("正方形"))
        self._img = QComboBox()
        self._est = QLabel()
        self._preview = _CropPreview()

        self._w.setValue(int(spec["w"]))
        self._h.setValue(int(spec["h"]))
        self._ov.setValue(int(spec["overlap"]))
        self._vis.setValue(int(round(spec.get("min_visible", 0.7) * 100)))
        self._square.setChecked(bool(spec.get("square")))
        for p in self._images[:100]:
            self._img.addItem(os.path.basename(p))
        self._clamp_ov_max()
        self._sync_counts()

        self._w.valueChanged.connect(self._on_size)
        self._h.valueChanged.connect(self._on_size)
        self._ov.valueChanged.connect(self._on_overlap)
        self._rows.valueChanged.connect(self._on_rows)
        self._cols.valueChanged.connect(self._on_cols)
        self._square.toggled.connect(self._on_square)
        self._vis.valueChanged.connect(lambda _v: self._debounce.start())
        self._img.currentIndexChanged.connect(self._on_image)

        left = QVBoxLayout()
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("裁剪尺寸（像素）")))
        row.addWidget(self._w); row.addWidget(QLabel("×")); row.addWidget(self._h)
        row.addWidget(self._square)
        row.addStretch()
        left.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("重叠长度（像素）"))); row.addWidget(self._ov); row.addStretch()
        left.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("裁剪数量（行 × 列）")))
        row.addWidget(self._rows); row.addWidget(QLabel("×")); row.addWidget(self._cols)
        row.addStretch()
        left.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel(tr("最小可见比例（%）"))); row.addWidget(self._vis); row.addStretch()
        left.addLayout(row)
        templates = load_crop_templates()
        if templates:
            box = QVBoxLayout()
            box.addWidget(QLabel(tr("模板")))
            for item in templates:
                txt = tr("template_fmt").format(
                    w=item["w"], h=item["h"], ov=item["overlap"],
                    vis=int(round(item["min_visible"] * 100)))
                if item["square"]:
                    txt += "·" + tr("正方形")
                btn = QPushButton(txt)
                btn.clicked.connect(lambda _c=False, spec=item: self._apply_template(spec))
                box.addWidget(btn)
            left.addLayout(box)
        lbl = QLabel(tr("预览图"))
        row = QHBoxLayout()
        row.addWidget(lbl)
        self._img.setFixedWidth(220)
        row.addWidget(self._img)
        row.addStretch()
        left.addLayout(row)
        row = QHBoxLayout()
        row.addSpacing(lbl.sizeHint().width() + 6)
        self._dims = QLabel()
        row.addWidget(self._dims)
        row.addStretch()
        left.addLayout(row)
        left.addWidget(self._est)
        left.addStretch()

        right = QVBoxLayout()
        right.addWidget(self._preview, 1)

        body = QHBoxLayout()
        body.addLayout(left, 0)
        body.addLayout(right, 1)

        btns = QHBoxLayout()
        btns.addWidget(QLabel(tr("绿=保留 · 红=空窗丢弃")))
        btns.addStretch()
        ok = QPushButton(tr("确定")); ok.setObjectName("successBtn"); ok.clicked.connect(self.accept)
        cancel = QPushButton(tr("取消")); cancel.setObjectName("accentBtn"); cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)

        main = QVBoxLayout(self)
        main.addLayout(body, 1)
        main.addLayout(btns)
        self._refresh()

    # ---- 工具 ----

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
        """套用最近模板：只回填参数，行列由当前图自动重算。"""
        controls = (self._w, self._h, self._ov, self._rows, self._cols,
                    self._vis, self._square)
        for c in controls:
            c.blockSignals(True)
        self._w.setValue(int(spec["w"]))
        self._h.setValue(int(spec["h"]))
        self._ov.setValue(int(spec["overlap"]))
        self._vis.setValue(int(round(spec["min_visible"] * 100)))
        self._square.setChecked(bool(spec["square"]))
        if self._square.isChecked():
            v = min(self._w.value(), self._h.value())
            self._w.setValue(v)
            self._h.setValue(v)
        self._clamp_ov_max()
        for c in controls:
            c.blockSignals(False)
        self._sync_counts()
        self._debounce.start()

    def _clamp_ov_max(self):
        """重叠上限 = 半窗：保证步长 ≥ 半窗，窗口数不爆炸。"""
        self._ov.setMaximum(max(0, min(self._w.value(), self._h.value()) // 2))

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
        cw, ch, ov = normalized_window(rw, rh, self.spec())
        self._set(self._rows, len(window_starts(rh, ch, ov)))
        self._set(self._cols, len(window_starts(rw, cw, ov)))

    def _on_size(self):
        if self._linking:
            return
        self._linking = True
        if self._square.isChecked():
            v = min(self._w.value(), self._h.value())
            self._set(self._w, v)
            self._set(self._h, v)
        self._clamp_ov_max()
        self._sync_counts()
        self._linking = False
        self._debounce.start()

    def _on_overlap(self):
        if self._linking:
            return
        self._linking = True
        ov = self._ov.value()
        if ov >= self._w.value():
            self._set(self._w, ov + 1)
        if ov >= self._h.value():
            self._set(self._h, ov + 1)
        self._sync_counts()
        self._linking = False
        self._debounce.start()

    def _on_cols(self):
        if self._linking:
            return
        rw, rh = self._ref_size()
        if not rw:
            return
        self._linking = True
        ov = self._ov.value()
        req = self._cols.value()
        w = self._exact_size(rw, size_for_count(rw, req, ov), req, ov)
        self._set(self._w, w)
        if self._square.isChecked():
            self._set(self._h, w)
        self._clamp_ov_max()
        self._sync_counts()  # 写回实际数量（不可达时钳到最近）
        self._linking = False
        self._debounce.start()

    def _on_rows(self):
        if self._linking:
            return
        rw, rh = self._ref_size()
        if not rw:
            return
        self._linking = True
        ov = self._ov.value()
        req = self._rows.value()
        h = self._exact_size(rh, size_for_count(rh, req, ov), req, ov)
        self._set(self._h, h)
        if self._square.isChecked():
            self._set(self._w, h)
        self._clamp_ov_max()
        self._sync_counts()
        self._linking = False
        self._debounce.start()

    def _on_square(self, checked):
        if self._linking:
            return
        self._linking = True
        if checked:
            v = min(self._w.value(), self._h.value())
            self._set(self._w, v)
            self._set(self._h, v)
            self._clamp_ov_max()
            self._sync_counts()
        self._linking = False
        self._debounce.start()

    def _on_image(self, _idx):
        self._update_dims()
        self._sync_counts()
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
        cw, ch, ov = normalized_window(iw, ih, self.spec())
        boxes = self._boxes.get(self._img.currentIndex(), [])
        min_vis = self._vis.value() / 100.0
        rects = []
        kept = 0
        for r, y0 in enumerate(window_starts(ih, ch, ov), 1):
            for c, x0 in enumerate(window_starts(iw, cw, ov), 1):
                has = bool(crop_boxes(boxes, x0, y0, cw, ch, min_vis))
                rects.append(((x0, y0, cw, ch), has, r, c))
                kept += 1 if has else 0
        self._preview.set_content(img, rects, boxes)
        self._est.setText(tr("crop_estimate").format(
            total=len(self._images), est=kept * len(self._images)))

    def spec(self):
        return {"w": self._w.value(), "h": self._h.value(),
                "overlap": self._ov.value(),
                "min_visible": self._vis.value() / 100.0,
                "square": self._square.isChecked()}

    def showEvent(self, event):
        super().showEvent(event)
        try:
            set_titlebar_dark(int(self.winId()), ThemeManager.get_mode().name == "DARK")
        except Exception:
            pass
