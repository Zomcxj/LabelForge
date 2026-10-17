import json
import os
import shutil
from datetime import datetime
from typing import List

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
    QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from .i18n import t as tr
from .theme import ThemeManager
from .dwm import set_titlebar_dark
from ..core.utils import create_app_icon
from ..core.config_manager import add_crop_template
from ..engine.yolo_exporter import YoloExporter
from ..engine.voc_exporter import VocExporter
from ..engine.coco_exporter import CocoExporter
from ..engine.splitter import Splitter
from ..engine.augmenter import Augmenter, get_all_transforms
from .mixins.processing_panel_builder import ProcessingPanelBuilderMixin, SPLIT_KEYS, TRANSFORM_META
from ..widgets.processing import Worker, CollapsibleSection







class ProcessingPanel(ProcessingPanelBuilderMixin, QWidget):
    def __init__(self, editor, parent=None):
        super().__init__(parent, Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self._editor = editor
        self.setObjectName("processingPanel")
        self.setWindowIcon(create_app_icon(os.path.dirname(__file__)))
        self.setMinimumWidth(500)
        self.resize(832, 850)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(0)
        self._build_path_selector(main_layout)
        self._collapsible_sections = []
        self._split_lbls = []
        self._interrupted = False
        self._workers = set()
        self._build_augment_section(main_layout)
        self._build_export_section(main_layout)
        self._build_split_section(main_layout)
        self._build_pipeline_section(main_layout)
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(80)
        self._log_area.setPlaceholderText(tr("操作日志将显示在这里..."))
        self._log_area.setObjectName("logArea")
        main_layout.addWidget(self._log_area, 1)
        self._augment_result = None
        self._crop_spec = None
        self._run_images = []
        self._run_boxes = {}
        # 构造时 _path_edit 已填好当前数据集，先记下基线，避免首次 _refresh_texts 误判为切换
        self._last_session_dir = os.path.normcase(
            os.path.normpath(self._get_session_dir() or ""))
        self._path_edit.textChanged.connect(self._on_path_changed)
        self._pipe_mode = False
        self._pipe_steps = []
        self._pipe_idx = 0
        self._pipe_error = False
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._spinner_tick)
        self._spinner_btn = None
        self._scan_check_timer = QTimer(self)
        self._scan_check_timer.setInterval(800)
        self._scan_check_timer.timeout.connect(self._update_labels_list)
        self._refresh_texts()

    def _build_path_selector(self, layout):
        path_layout = QHBoxLayout()
        path_layout.setContentsMargins(0, 0, 0, 4)
        path_layout.setSpacing(6)
        self._path_lbl = QLabel()
        path_layout.addWidget(self._path_lbl)
        self._path_edit = QLineEdit(self._get_session_dir())
        self._path_edit.setObjectName("outputPath")
        self._path_edit.textChanged.connect(self._update_labels_list)
        path_layout.addWidget(self._path_edit, 1)
        self._browse_btn = QPushButton()
        self._browse_btn.setObjectName("accentBtn")
        self._browse_btn.setFixedWidth(60)
        self._browse_btn.clicked.connect(self._browse_output_dir)
        path_layout.addWidget(self._browse_btn)
        layout.addLayout(path_layout)

    def _follow_session_dir(self):
        """数据集变化时无条件跟随，把「数据集目录」切到新路径。

        导出目录 = 该路径父目录下的 exports，不跟随就会把新数据集导出到旧数据集的
        exports 里。判据是「数据集是否变了」而非「输入框是否为空」——旧实现只在为空时
        回填，记忆切换数据集后框里仍是旧路径，于是导出路径不跟着走。
        注意只在数据集变化时写入：_refresh_texts 也会被切语言等无关场合调用，
        每次都无脑覆盖会把用户手输的路径冲掉。
        """
        session = os.path.normcase(os.path.normpath(self._get_session_dir() or ""))
        if session and session != getattr(self, '_last_session_dir', None):
            self._path_edit.setText(self._get_session_dir())
        self._last_session_dir = session

    def _get_session_dir(self):
        imgs = self._editor.background_images
        if imgs:
            return os.path.dirname(imgs[0])
        return ""

    def _get_output_dir(self):
        working = self._path_edit.text().strip()
        if not working:
            working = self._get_session_dir()
        if not working:
            return os.getcwd()
        exports_dir = os.path.normpath(os.path.join(os.path.dirname(working), "exports"))
        if not os.path.exists(exports_dir):
            os.makedirs(exports_dir, exist_ok=True)
        return exports_dir

    def _browse_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, tr("选择数据集目录"), self._path_edit.text())
        if d:
            self._path_edit.setText(d)

    def _collapsible_section(self, title, parent_layout, color="#4CAF50"):
        self._collapsible_sections.append(title)
        s = CollapsibleSection(title, color=color)
        parent_layout.addWidget(s)
        return s


    def _make_spin(self, typ, vmin, vmax, default):
        s = typ()
        s.setRange(vmin, vmax)
        s.setSingleStep(0.05 if typ == QDoubleSpinBox else 1)
        if typ == QDoubleSpinBox:
            s.setDecimals(1)
        s.setValue(default)
        s.setFixedWidth(60)
        return s




    def _scan_labels_from_json(self):
        """Scan background image sidecar JSONs only (no detection boxes / paste)."""
        labels = set()
        bg_labels = getattr(self._editor, 'background_dataset_labels', None)
        if bg_labels:
            return sorted(bg_labels)
        worker = getattr(self._editor, '_background_label_scan_worker', None)
        if worker is not None and worker.isRunning():
            return []
        images = self._editor.background_images
        if not images:
            work_dir = self._path_edit.text().strip()
            if work_dir and os.path.isdir(work_dir):
                exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
                images = sorted([
                    os.path.join(work_dir, f) for f in os.listdir(work_dir)
                    if os.path.splitext(f)[1].lower() in exts
                ])
        total = len(images)
        for i, img_path in enumerate(images):
            jp = os.path.splitext(img_path)[0] + ".json"
            if os.path.exists(jp):
                try:
                    with open(jp, encoding='utf-8') as f:
                        data = json.load(f)
                    for sh in data.get("shapes", []):
                        lbl = sh.get("label", "")
                        if isinstance(lbl, str) and lbl.strip():
                            labels.add(lbl.strip())
                except Exception as e:
                    from ..core.exception_hook import _write_log
                    _write_log(f"扫描标签 JSON 失败 {jp}: {e}")
            if total > 2 and i % 5 == 0:
                QApplication.processEvents()
        if labels and hasattr(self._editor, 'background_dataset_labels'):
            self._editor._scanned_background_labels = set(labels)
            imported = set(getattr(self._editor, 'imported_background_labels', None) or ())
            self._editor.background_dataset_labels = set(labels) | imported
        return sorted(labels)

    def _fast_labels(self):
        """Return background-dataset labels only, or None if not ready."""
        completed = getattr(self._editor, '_background_label_scan_completed', False)
        bg_labels = getattr(self._editor, 'background_dataset_labels', None)
        if completed:
            return sorted(bg_labels or ())
        if bg_labels:
            return sorted(bg_labels)
        return None

    def _update_labels_list(self):
        self._clear_grid()
        pending = getattr(self._editor, '_background_label_scan_pending', False)
        completed = getattr(self._editor, '_background_label_scan_completed', False)
        labels = self._fast_labels()
        if labels is not None:
            if hasattr(self, '_scan_check_timer'):
                self._scan_check_timer.stop()
            self._exp_scan_label.setVisible(False)
        elif completed:
            if hasattr(self, '_scan_check_timer'):
                self._scan_check_timer.stop()
            self._exp_scan_label.setVisible(False)
        elif pending:
            self._exp_scan_label.setVisible(True)
            self._exp_scan_label.setLabel(tr("识别中..."))
            QApplication.processEvents()
            if hasattr(self, '_scan_check_timer') and not self._scan_check_timer.isActive():
                self._scan_check_timer.start()
            return
        else:
            labels = self._scan_labels_from_json()
            worker = getattr(self._editor, '_background_label_scan_worker', None)
            if worker is not None and worker.isRunning():
                self._exp_scan_label.setVisible(True)
                self._exp_scan_label.setLabel(tr("识别中..."))
                QApplication.processEvents()
                if hasattr(self, '_scan_check_timer') and not self._scan_check_timer.isActive():
                    self._scan_check_timer.start()
                return
            if hasattr(self, '_scan_check_timer'):
                self._scan_check_timer.stop()
            self._exp_scan_label.setVisible(False)
        if not labels:
            msg = QLabel(tr("未检测到标签"))
            msg.setStyleSheet("color: gray; padding: 4px;")
            self._exp_label_grid.addWidget(msg, 0, 0)
            return
        for i, lbl in enumerate(labels):
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            cb.setMinimumWidth(100)
            row, col = divmod(i, 6)
            self._exp_label_grid.addWidget(cb, row, col)
            self._exp_label_checkboxes[lbl] = cb

    def _clear_grid(self):
        for w in list(self._exp_label_grid_container.children()):
            if not isinstance(w, QWidget) or w is self._exp_scan_label:
                continue
            self._exp_label_grid.removeWidget(w)
            w.hide()
            w.deleteLater()
        self._exp_label_checkboxes.clear()

    def _refresh_texts(self):
        self.setWindowTitle(tr("数据处理"))
        self._follow_session_dir()
        self._path_lbl.setText(tr("数据集目录:"))
        self._browse_btn.setText(tr("选择"))
        self._aug_section.set_title(tr("数据增强"))
        for name, (cb, param_spins, cn_name) in self._aug_widgets.items():
            if name == "__orig__":
                cb.setText(tr("原图"))
            else:
                cb.setText(tr(cn_name))
        self._ratio_lbl.setText(tr("图片比例:"))
        self._mode_lbl.setText(tr("模式:"))
        idx = self._aug_mode.currentIndex()
        self._aug_mode.clear()
        self._aug_mode.addItems([tr("全部增强"), tr("随机几项")])
        self._aug_mode.setCurrentIndex(idx)
        self._aug_btn.setText(tr("增强"))
        self._aug_skip_empty.setText(tr("跳过空标签"))
        self._aug_uncheck_all_btn.setText(tr("取消全选"))
        self._aug_clear_btn.setText(tr("清空"))
        self._crop_check.setText(tr("滑窗裁剪"))
        self._crop_config_btn.setText(tr("配置…"))
        self._update_crop_summary()
        self._exp_section.set_title(tr("标签导出"))
        self._exp_fmt_label.setText(tr("格式:"))
        self._exp_label_title.setText(tr("类别:"))
        self._exp_skip_empty.setText(tr("跳过空标签"))
        self._exp_btn.setText(tr("导出"))
        self._exp_clear_btn.setText(tr("清空"))
        self._split_section.set_title(tr("数据集划分"))
        for key, lbl in zip(SPLIT_KEYS, self._split_lbls):
            lbl.setText(tr(key))
        self._split_btn.setText(tr("划分"))
        self._split_clear_btn.setText(tr("清空"))
        self._pipe_section.set_title(tr("流水线"))
        self._pipe_aug.setText(tr("增强"))
        self._pipe_exp.setText(tr("导出"))
        self._pipe_split.setText(tr("划分"))
        self._pipe_btn.setText(tr("▶ 一键流水线"))

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_texts()
        self._update_labels_list()
        try:
            hwnd = int(self.winId())
            is_dark = ThemeManager.get_mode().name == "DARK"
            set_titlebar_dark(hwnd, is_dark)
        except Exception as e:
            from ..core.exception_hook import _write_log
            _write_log(f"面板标题栏深色同步失败: {e}")

    def closeEvent(self, event):
        self._scan_check_timer.stop()
        self._spinner_timer.stop()
        for w in list(self._workers):
            w.requestInterruption()
            w.wait(2000)
        self._log_area.clear()
        self.hide()
        event.accept()

    def _button_busy(self, btn, text=None, clear_btn=None):
        if text is None:
            text = tr("执行中...")
        btn._orig_text = btn.text()
        btn._busy_text = text
        btn.setText(text)
        btn.setEnabled(False)
        self._interrupted = False
        for b in (self._aug_btn, self._exp_btn, self._split_btn, self._pipe_btn):
            if b is not btn:
                b.setEnabled(False)
        if hasattr(btn, '_stop_btn'):
            btn._stop_btn.setVisible(True)
            btn._stop_btn.setEnabled(True)
        if clear_btn:
            clear_btn.setEnabled(False)
        self._spinner_idx = 0
        self._spinner_btn = btn
        self._spinner_timer.start(80)

    def _button_ready(self, btn, clear_btn=None):
        self._spinner_timer.stop()
        self._spinner_btn = None
        btn.setText(getattr(btn, '_orig_text', btn.text()))
        btn.setEnabled(True)
        if hasattr(btn, '_stop_btn'):
            btn._stop_btn.setVisible(False)
        for b in (self._aug_btn, self._exp_btn, self._split_btn, self._pipe_btn):
            if b is not btn:
                b.setEnabled(True)
        if clear_btn:
            clear_btn.setEnabled(True)

    def _spinner_tick(self):
        if self._spinner_btn is None:
            return
        c = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
        self._spinner_idx += 1
        text = getattr(self._spinner_btn, '_busy_text', tr("执行中..."))
        self._spinner_btn.setText(f"{c} {text}")

    def _make_btn_with_stop(self, btn, stop_handler):
        stop_btn = QPushButton("■")
        stop_btn.setObjectName("accentBtn")
        stop_btn.setFixedWidth(28)
        stop_btn.setVisible(False)
        stop_btn.setStyleSheet("QPushButton { color: #F44336; }")
        stop_btn.clicked.connect(stop_handler)
        btn._stop_btn = stop_btn
        return stop_btn

    def _do_interrupt(self):
        self._interrupted = True
        self._log(tr("操作中断"))

    def _ensure_boxes_loaded(self):
        work_dir = self._path_edit.text().strip()
        if not work_dir:
            work_dir = self._get_session_dir()
        if not work_dir or not os.path.isdir(work_dir):
            self._run_images = []
            self._run_boxes = {}
            return
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        img_files = sorted([
            f for f in os.listdir(work_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        self._run_images = [os.path.join(work_dir, f) for f in img_files]
        self._run_boxes = {}
        from ..engine.shape_io import box_from_labelme_shape
        for idx, img_path in enumerate(self._run_images):
            json_path = os.path.splitext(img_path)[0] + ".json"
            if not os.path.exists(json_path):
                self._run_boxes[idx] = []
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                boxes = []
                for sh in data.get("shapes", []):
                    box = box_from_labelme_shape(sh)
                    if box is not None:
                        boxes.append(box)
                self._run_boxes[idx] = boxes
            except Exception:
                self._run_boxes[idx] = []
            if idx % 5 == 0:
                QApplication.processEvents()

    def _get_selected_labels(self) -> List[str]:
        return [lbl for lbl, cb in self._exp_label_checkboxes.items() if cb.isChecked()]

    def _get_image_size(self, img_path):
        try:
            from PySide6.QtGui import QImage
            img = QImage(img_path)
            if not img.isNull():
                return img.width(), img.height()
        except Exception as e:
            from ..core.exception_hook import _write_log
            _write_log(f"读取图片尺寸失败 {img_path}: {e}")
        return 0, 0

    def _detect_augmented_images(self):
        output_dir = self._get_output_dir()
        aug_dir = os.path.join(output_dir, "images")
        if not os.path.isdir(aug_dir):
            return None
        imgs = [f for f in os.listdir(aug_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
        if not imgs:
            return None
        aug_dir = os.path.normpath(aug_dir)
        aug_images = [os.path.join(aug_dir, f) for f in sorted(imgs)]
        aug_boxes = {}
        from ..engine.shape_io import box_from_labelme_shape
        for i, img_path in enumerate(aug_images):
            base = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(aug_dir, base + ".json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    box_list = []
                    for shape in data.get("shapes", []):
                        box = box_from_labelme_shape(shape)
                        if box is not None:
                            box_list.append(box)
                    aug_boxes[i] = box_list
                except Exception:
                    aug_boxes[i] = []
            else:
                aug_boxes[i] = []
        return aug_images, aug_boxes

    def _clear_dir(self, dir_path):
        dir_path = os.path.normpath(dir_path)
        exports_dir = os.path.normpath(self._get_output_dir())
        if not dir_path.startswith(exports_dir):
            return False
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            return True
        return False

    def _has_files(self, *paths):
        for p in paths:
            if os.path.isdir(p) and os.listdir(p):
                return True
        return False

    def _require_path(self):
        if not self._path_edit.text().strip():
            self._log(tr("log_no_files"))
            return False
        return True

    def _confirm_clear(self, path):
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle(tr("清空"))
        box.setText(tr("确认清空") + "\n" + path)
        yes_btn = box.addButton(tr("是"), QMessageBox.YesRole)
        no_btn = box.addButton(tr("否"), QMessageBox.NoRole)
        box.setDefaultButton(no_btn)
        hwnd = int(box.winId())
        set_titlebar_dark(hwnd, True)
        box.exec_()
        return box.clickedButton() == yes_btn

    def _uncheck_all_transforms(self):
        """一键取消所有已勾选的增强方式（含原图与滑窗裁剪），便于从零重新挑选。"""
        for _name, (cb, _params, _disp) in self._aug_widgets.items():
            cb.setChecked(False)
        self._crop_check.setChecked(False)

    def _clear_aug(self):
        if not self._require_path():
            return
        output_dir = self._get_output_dir()
        aug_dir = os.path.join(output_dir, "images")
        if not self._has_files(aug_dir):
            self._log(tr("log_no_files"))
            return
        if not self._confirm_clear(aug_dir):
            return
        self._log(tr("log_clearing_path").format(path=aug_dir))
        if self._clear_dir(aug_dir):
            self._log(tr("log_aug_cleared"))
            self._augment_result = None

    def _clear_exp(self):
        if not self._require_path():
            return
        output_dir = self._get_output_dir()
        imgs = os.path.join(output_dir, "images")
        lbls = os.path.join(output_dir, "labels")
        cls_f = os.path.join(output_dir, "classes.txt")
        if not self._has_files(imgs, lbls) and not os.path.exists(cls_f):
            self._log(tr("log_no_files"))
            return
        if not self._confirm_clear(output_dir):
            return
        self._log(tr("log_clearing_path").format(path=output_dir))
        self._clear_dir(imgs)
        self._clear_dir(lbls)
        if os.path.exists(cls_f):
            os.remove(cls_f)
        self._log(tr("log_exp_cleared"))
        self._augment_result = None

    def _clear_split(self):
        if not self._require_path():
            return
        output_dir = self._get_output_dir()
        divide_dir = os.path.join(output_dir, "divide")
        if not self._has_files(divide_dir):
            self._log(tr("log_no_files"))
            return
        if not self._confirm_clear(divide_dir):
            return
        self._log(tr("log_clearing_path").format(path=divide_dir))
        if self._clear_dir(divide_dir):
            self._log(tr("log_split_cleared"))
        else:
            self._log(tr("log_no_files"))

    def _log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self._log_area.textCursor()
        cursor.movePosition(cursor.End)
        self._log_area.setTextCursor(cursor)
        self._log_area.append(f"[{ts}] {msg}")
        sb = self._log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _get_transform_specs(self):
        all_t = get_all_transforms()
        specs = []
        for name, (cb, param_spins, _) in self._aug_widgets.items():
            if not cb.isChecked() or name not in all_t:
                continue
            cls = all_t[name]
            ranges = {}
            for pname, (smin, smax) in param_spins.items():
                ranges[pname] = (smin.value(), smax.value())
            specs.append((cls, ranges))
        return specs

    def _crop_ref_size(self):
        if self._run_images:
            w, h = self._get_image_size(self._run_images[0])
            if w and h:
                return w, h
        return 0, 0

    def _default_crop_spec(self):
        from ..engine.augmenter.crop import FILL_RGB_DEFAULT, default_window
        W, H = self._crop_ref_size()
        w, h, ov = default_window(W, H)   # 按参考图反推 2行3列、重叠为窗口的 15%
        return {"w": w, "h": h, "overlap": ov,
                "min_visible": 0.7, "square": False,
                "fill": False, "fill_rgb": FILL_RGB_DEFAULT}

    def _on_path_changed(self, _text=""):
        """切换数据集目录：清空本轮裁剪配置与缓存，摘要回到默认口径。"""
        self._crop_spec = None
        self._run_images = []
        self._run_boxes = {}
        self._update_crop_summary()

    def _get_crop_spec(self):
        if not self._crop_check.isChecked():
            return None
        if self._crop_spec is None:
            self._crop_spec = self._default_crop_spec()
        return self._crop_spec

    def _update_crop_summary(self):
        if not self._crop_check.isChecked():
            self._crop_summary_lbl.setText(tr("未配置"))
            return
        spec = self._crop_spec or self._default_crop_spec()
        rows = cols = "?"
        W, H = self._crop_ref_size()
        if W and H:
            from ..engine.augmenter.crop import normalized_window, window_starts
            fill = bool(spec.get("fill"))
            cw, ch, ov = normalized_window(W, H, spec)
            cols = len(window_starts(W, cw, ov, fill))
            rows = len(window_starts(H, ch, ov, fill))
            spec = {**spec, "overlap": ov}
        txt = tr("crop_summary").format(
            w=spec["w"], h=spec["h"], ov=spec["overlap"], rows=rows, cols=cols)
        if spec.get("fill"):
            txt += " · " + tr("补边")   # 开填充时导出名带 _pad，摘要须点明
        self._crop_summary_lbl.setText(txt)

    def _open_crop_config(self):
        self._ensure_boxes_loaded()
        if not self._run_images:
            self._log(tr("log_no_images"))
            return
        from .crop_dialog import CropConfigDialog
        dlg = CropConfigDialog(self._run_images, self._run_boxes,
                               self._get_crop_spec() or self._default_crop_spec(), self)
        if dlg.exec_() == QDialog.Accepted:
            self._crop_spec = dlg.spec()
            add_crop_template(self._crop_spec)
            self._update_crop_summary()

    def _run_augment(self):
        self._ensure_boxes_loaded()
        if not self._run_images:
            self._log(tr("log_no_images"))
            self._pipe_finish(True)
            return
        specs = self._get_transform_specs()
        crop_spec = self._get_crop_spec()
        if not specs and not crop_spec:
            self._pipe_finish(True)
            return
        ratio = self._aug_ratio.value()
        mode = "all" if self._aug_mode.currentIndex() == 0 else "random"
        include_original = self._aug_inc_orig.isChecked()
        skip_empty = self._aug_skip_empty.isChecked()
        output_dir = self._get_output_dir()
        transform_names = [name for name, (cb, _, _) in self._aug_widgets.items()
                          if cb.isChecked() and not name.startswith("__")]
        if crop_spec:
            transform_names.append(tr("滑窗裁剪"))
        self._log(tr("log_aug_start").format(
            count=len(self._run_images),
            transforms=", ".join(transform_names) if transform_names else tr("原图"),
            ratio=ratio
        ))
        self._button_busy(self._aug_btn, clear_btn=self._aug_clear_btn)
        total = len(self._run_images)
        self._interrupted = False

        def _aug_task(log_fn, progress_fn):
            def _on_transform_progress(tname, cur, tot):
                display = TRANSFORM_META.get(tname, (tname,))[0]
                progress_fn(cur, total, tr(display))
            aug = Augmenter(
                output_dir,
                on_progress=lambda c, t: None,
                is_interrupted=lambda: self._interrupted,
                on_transform_progress=_on_transform_progress,
            )
            return aug.run(self._run_images, self._run_boxes, specs, ratio, mode,
                          include_original=include_original, skip_empty=skip_empty,
                          crop_spec=crop_spec)

        def _on_finished(result):
            if self._interrupted:
                self._log(tr("已中断"))
                self._button_ready(self._aug_btn, clear_btn=self._aug_clear_btn)
                self._pipe_finish(True)
                return
            self._augment_result = result
            aug_count = len(result) if result else 0
            self._log(tr("log_aug_done").format(aug_count=aug_count, total=aug_count))
            self._button_ready(self._aug_btn, clear_btn=self._aug_clear_btn)
            self._pipe_finish()
        def _on_error(err):
            self._log(tr("log_aug_err").format(err=err))
            self._augment_result = None
            self._button_ready(self._aug_btn, clear_btn=self._aug_clear_btn)
            self._pipe_finish(True)
        worker = Worker(_aug_task)
        worker.progress.connect(lambda *a: None)
        worker.result_ready.connect(_on_finished)
        worker.error.connect(_on_error)
        self._workers.add(worker)
        worker.destroyed.connect(lambda: self._workers.discard(worker))
        worker.start()

    def _run_export(self):
        self._ensure_boxes_loaded()
        if not self._run_images:
            self._log(tr("log_no_images"))
            self._pipe_finish(True)
            return
        labels = self._get_selected_labels()
        if not labels:
            self._log(tr("log_no_labels"))
            self._pipe_finish(True)
            return
        output_dir = self._get_output_dir()
        self._button_busy(self._exp_btn, clear_btn=self._exp_clear_btn)
        self._interrupted = False
        skip_empty = self._exp_skip_empty.isChecked()

        def _exp_task(log_fn, progress_fn):
            detected = self._detect_augmented_images()
            input_data = []
            if detected:
                aug_images, aug_boxes = detected
                output_dir_inner = os.path.join(output_dir, "images")
                json_count = sum(1 for f in os.listdir(output_dir_inner) if f.endswith('.json'))
                img_count = len(aug_images)
                if json_count != img_count:
                    log_fn(tr("log_split_count_mismatch").format(txt_count=json_count, img_count=img_count))
                    return
                for i, img_path in enumerate(aug_images):
                    w, h = self._get_image_size(img_path)
                    input_data.append({
                        "image": None,
                        "boxes": aug_boxes.get(i, []),
                        "width": w, "height": h,
                        "stem": os.path.splitext(os.path.basename(img_path))[0],
                        "img_path": img_path,
                    })
                log_fn(tr("log_aug_detected").format(count=img_count))
            else:
                for idx in range(len(self._run_images)):
                    w, h = self._get_image_size(self._run_images[idx])
                    input_data.append({
                        "image": None,
                        "boxes": self._run_boxes.get(idx, []),
                        "width": w, "height": h,
                        "stem": os.path.splitext(os.path.basename(self._run_images[idx]))[0],
                        "img_path": self._run_images[idx],
                    })
                log_fn(tr("log_orig_detected").format(count=len(input_data)))
            if not input_data:
                log_fn(tr("log_no_images"))
                return
            log_fn(tr("log_exp_start").format(
                count=len(input_data),
                labels=", ".join(labels),
            ))
            fmt = self._exp_format.currentText()
            if fmt == "VOC Detection":
                exp = VocExporter(
                    output_dir,
                    on_progress=lambda c, t: progress_fn(c, t, ""),
                    is_interrupted=lambda: self._interrupted,
                )
            elif fmt == "COCO Detection":
                exp = CocoExporter(
                    output_dir,
                    on_progress=lambda c, t: progress_fn(c, t, ""),
                    is_interrupted=lambda: self._interrupted,
                )
            else:
                exp = YoloExporter(
                    output_dir,
                    on_progress=lambda c, t: progress_fn(c, t, ""),
                    is_interrupted=lambda: self._interrupted,
                    mode=("seg" if fmt == "YOLO Seg"
                          else "pose" if fmt == "YOLO Pose"
                          else "obb" if fmt == "YOLO OBB" else "hbb"),
                )
            exp.run(
                self._run_images,
                self._run_boxes,
                labels,
                skip_empty=skip_empty,
                input_data=input_data
            )

        def _on_finished(_):
            if self._interrupted:
                self._log(tr("已中断"))
                self._button_ready(self._exp_btn, clear_btn=self._exp_clear_btn)
                self._pipe_finish(True)
                return
            self._log(tr("log_exp_done").format(path=output_dir))
            self._button_ready(self._exp_btn, clear_btn=self._exp_clear_btn)
            self._pipe_finish()
        def _on_error(err):
            self._log(tr("log_exp_err").format(err=err))
            self._button_ready(self._exp_btn, clear_btn=self._exp_clear_btn)
            self._pipe_finish(True)
        worker = Worker(_exp_task)
        worker.log.connect(self._log)
        worker.progress.connect(lambda *a: None)
        worker.result_ready.connect(_on_finished)
        worker.error.connect(_on_error)
        self._workers.add(worker)
        worker.destroyed.connect(lambda: self._workers.discard(worker))
        worker.start()

    def _on_split_changed(self, changed_key):
        others = [k for k in SPLIT_KEYS if k != changed_key]
        other_sum = sum(getattr(self, f"_split_{k}").value() for k in others)
        sb = getattr(self, f"_split_{changed_key}")
        new_val = sb.value()
        max_val = round(1.0 - other_sum, 2)
        if new_val > max_val:
            sb.blockSignals(True)
            sb.setValue(max(0.0, max_val))
            sb.blockSignals(False)

    def _run_split(self):
        self._ensure_boxes_loaded()
        if not self._run_images:
            self._log(tr("log_no_images"))
            self._pipe_finish(True)
            return
        output_dir = self._get_output_dir()
        labels_dir = os.path.join(output_dir, "labels")
        images_dir = os.path.join(output_dir, "images")
        txt_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')] if os.path.isdir(labels_dir) else []
        if not txt_files:
            self._log(tr("log_split_no_labels"))
            self._pipe_finish(True)
            return
        aug_jpgs = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.isdir(images_dir) else []
        txt_count = len(txt_files)
        if aug_jpgs:
            img_count = len(aug_jpgs)
            if img_count > txt_count:
                self._log(tr("log_split_aug_gt_txt").format(img_count=img_count, txt_count=txt_count))
            elif img_count < txt_count:
                self._log(tr("log_split_aug_lt_txt").format(img_count=img_count, txt_count=txt_count))
            split_images, split_boxes = self._detect_augmented_images()
            txt_bases = {os.path.splitext(f)[0] for f in txt_files}
            filtered = []
            filtered_boxes = {}
            for i, img_path in enumerate(split_images):
                base = os.path.splitext(os.path.basename(img_path))[0]
                if base in txt_bases:
                    new_idx = len(filtered)
                    filtered.append(img_path)
                    filtered_boxes[new_idx] = split_boxes.get(i, [])
            split_images = filtered
            split_boxes = filtered_boxes
        else:
            src_bases = {os.path.splitext(os.path.basename(p))[0]: i
                         for i, p in enumerate(self._run_images)}
            txt_bases = {os.path.splitext(f)[0] for f in txt_files}
            matched_bases = [b for b in txt_bases if b in src_bases]
            if not matched_bases:
                self._log(tr("log_split_no_labels"))
                self._pipe_finish(True)
                return
            self._log(tr("log_aug_detected").format(count=len(matched_bases)))
            split_images = []
            split_boxes = {}
            for i, b in enumerate(matched_bases):
                orig_idx = src_bases[b]
                split_images.append(self._run_images[orig_idx])
                split_boxes[i] = self._run_boxes.get(orig_idx, [])
        ratios = {k: getattr(self, f"_split_{k}").value() for k in SPLIT_KEYS}
        total = len(split_images)
        train_n = int(total * ratios["train"])
        val_n = int(total * ratios["val"])
        test_n = int(total * ratios["test"])
        rest = total - train_n - val_n - test_n
        if rest > 0:
            train_n += rest
        self._log(tr("log_split_start").format(total=total, train=train_n, val=val_n, test=test_n))
        self._button_busy(self._split_btn, clear_btn=self._split_clear_btn)
        self._interrupted = False

        def _split_task(log_fn, progress_fn):
            sp = Splitter(
                output_dir,
                on_progress=lambda c, t: progress_fn(c, t, ""),
                is_interrupted=lambda: self._interrupted,
            )
            sp.run(split_images, split_boxes, ratios)

        def _on_finished(_):
            if self._interrupted:
                self._log(tr("已中断"))
                self._button_ready(self._split_btn, clear_btn=self._split_clear_btn)
                self._pipe_finish(True)
                return
            self._log(tr("log_split_done").format(train=train_n, val=val_n, test=test_n))
            self._button_ready(self._split_btn, clear_btn=self._split_clear_btn)
            self._pipe_finish()
        def _on_error(err):
            self._log(tr("log_split_err").format(err=err))
            self._button_ready(self._split_btn, clear_btn=self._split_clear_btn)
            self._pipe_finish(True)
        worker = Worker(_split_task)
        worker.progress.connect(lambda *a: None)
        worker.result_ready.connect(_on_finished)
        worker.error.connect(_on_error)
        self._workers.add(worker)
        worker.destroyed.connect(lambda: self._workers.discard(worker))
        worker.start()

    def _run_pipeline(self):
        steps = [s for s in ["augment", "export", "split"]
                 if getattr(self, {"augment": "_pipe_aug",
                                   "export": "_pipe_exp",
                                   "split": "_pipe_split"}[s]).isChecked()]
        if len(steps) < 2:
            return
        self._augment_result = None
        self._interrupted = False
        self._pipe_mode = True
        self._pipe_steps = list(steps)
        self._pipe_idx = 0
        self._pipe_error = False
        self._pipe_next()

    def _pipe_next(self):
        if self._pipe_error or self._interrupted or self._pipe_idx >= len(self._pipe_steps):
            self._pipe_finish(self._pipe_error)
            return
        step = self._pipe_steps[self._pipe_idx]
        self._pipe_idx += 1
        getattr(self, f"_run_{step}")()

    def _pipe_finish(self, error=False):
        if error:
            self._pipe_error = True
        if self._pipe_idx >= len(self._pipe_steps) or self._pipe_error or self._interrupted:
            if self._interrupted:
                self._log(tr("已中断"))
            self._pipe_mode = False
            self._pipe_steps = []
            self._pipe_idx = 0
            self._pipe_error = False
            self._augment_result = None
            self._interrupted = False
        else:
            QTimer.singleShot(50, self._pipe_next)
