"""推理弹窗：模型配置、类别与阈值、单张/批量执行。

模型加载与推理都在后台线程，UI 只负责配置与结果展示；
真正的标注写入由主窗口在结果返回后一次性完成。
"""
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QGroupBox, QListWidget,
    QListWidgetItem, QFileDialog, QProgressBar, QTextEdit,
    QAbstractItemView,
)

from ..engine import inference_task
from ..engine.inference_image import qimage_to_rgb_array, resize_for_model
from ..engine.onnx_inference import available_providers, runtime_available
from . import i18n
from .dialog_helpers import center_on_parent, warning
from .theme import ThemeManager
from .dwm import set_titlebar_dark

tr = i18n.t


def _load_rgb(path):
    """读图并转成模型输入用的 RGB 数组；读不出来返回 None。"""
    from PyQt5.QtGui import QImage
    image = QImage(path)
    if image.isNull():
        return None
    return qimage_to_rgb_array(image)


class _ModelLoadWorker(QThread):
    """后台加载模型并读取输入尺寸/格式/类别，避免卡住界面。"""
    loaded = pyqtSignal(object, object, object)   # detector, size, format
    failed = pyqtSignal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._settings = settings

    def run(self):
        try:
            detector = inference_task.build_detector(self._settings)
            size = detector.model_input_size()
            fmt = detector.resolve_format()
            self.loaded.emit(detector, size, fmt)
        except Exception as exc:
            self.failed.emit(str(exc))


class _InferenceWorker(QThread):
    """后台顺序推理。progress: (done, total, 文件名)。"""
    progress = pyqtSignal(int, int, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, keys, detector, class_filter, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._keys = list(keys)
        self._detector = detector
        self._class_filter = class_filter
        self._interrupted = False

    def run(self):
        try:
            results = inference_task.run_batch(
                self._paths, self._detector, _load_rgb, resize_for_model,
                keys=self._keys,
                progress_fn=self.progress.emit,
                is_interrupted=lambda: self._interrupted,
                class_filter=self._class_filter,
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._interrupted = True


class InferenceDialog(QDialog):
    """推理配置弹窗；apply_requested 由主窗口接住并应用结果。"""

    apply_requested = pyqtSignal(object, bool)   # (results, replace_auto)

    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self._detector = None
        self._model_size = None
        self._model_format = None
        self._load_worker = None
        self._run_worker = None
        self._images = []
        self._settings = inference_task.normalize_settings(settings or {})

        self.setWindowTitle(tr("模型推理"))
        self.setMinimumSize(620, 560)
        self._apply_style()
        self._build_ui()
        self._restore_settings()

    def _apply_style(self):
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QGroupBox {{ border: 1px solid {t['border_color']}; border-radius: 8px;
                margin-top: 10px; padding-top: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTextEdit {{
                background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px; padding: 3px 6px; }}
            QPushButton {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px; padding: 4px 10px; }}
            QPushButton:hover {{ border-color: {t['accent']}; }}
            QPushButton:disabled {{ color: {t['text_secondary']}; }}
            QProgressBar {{ border: none; border-radius: 8px;
                background-color: {t['scrollbar_bg']}; text-align: center;
                color: {t['text_primary']}; height: 16px; }}
            QProgressBar::chunk {{ background-color: {t['accent']}; border-radius: 8px; }}
        """)
        is_dark = ThemeManager.get_mode().value == "dark"
        QTimer.singleShot(30, lambda: set_titlebar_dark(int(self.winId()), is_dark))

    def _build_ui(self):
        layout = QVBoxLayout(self)

        model_group = QGroupBox(tr("模型"))
        model_layout = QVBoxLayout(model_group)
        row = QHBoxLayout()
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText(tr("选择 .onnx 模型文件"))
        self.model_edit.textChanged.connect(self._on_config_changed)
        row.addWidget(self.model_edit)
        self.model_btn = QPushButton(tr("浏览..."))
        self.model_btn.clicked.connect(self._pick_model)
        row.addWidget(self.model_btn)
        model_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("类别文件:")))
        self.names_edit = QLineEdit()
        self.names_edit.setPlaceholderText(tr("可选，一行一个类别"))
        self.names_edit.textChanged.connect(self._on_config_changed)
        row.addWidget(self.names_edit)
        self.names_btn = QPushButton(tr("浏览..."))
        self.names_btn.clicked.connect(self._pick_names)
        row.addWidget(self.names_btn)
        model_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("输出格式:")))
        self.format_combo = QComboBox()
        self.format_combo.addItem(tr("自动识别"), "auto")
        self.format_combo.addItem("YOLOv5", "v5")
        self.format_combo.addItem("YOLOv8", "v8")
        self.format_combo.currentIndexChanged.connect(self._on_config_changed)
        row.addWidget(self.format_combo)

        row.addWidget(QLabel(tr("输入尺寸:")))
        self.size_spin_w = QSpinBox()
        self.size_spin_w.setRange(0, 8192)
        self.size_spin_w.setSpecialValueText(tr("自动"))
        self.size_spin_h = QSpinBox()
        self.size_spin_h.setRange(0, 8192)
        self.size_spin_h.setSpecialValueText(tr("自动"))
        row.addWidget(self.size_spin_w)
        row.addWidget(QLabel("x"))
        row.addWidget(self.size_spin_h)
        model_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel(tr("置信度:")))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.valueChanged.connect(self._on_config_changed)
        row.addWidget(self.conf_spin)
        row.addWidget(QLabel("IoU:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setDecimals(2)
        row.addWidget(self.iou_spin)
        row.addWidget(QLabel(tr("设备:")))
        self.device_combo = QComboBox()
        self.device_combo.addItem(tr("自动"), None)
        for provider in available_providers():
            self.device_combo.addItem(provider, provider)
        row.addWidget(self.device_combo)
        model_layout.addLayout(row)

        self.load_btn = QPushButton(tr("加载模型"))
        self.load_btn.clicked.connect(self._load_model)
        model_layout.addWidget(self.load_btn)
        self.model_info_label = QLabel(tr("尚未加载模型"))
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)
        layout.addWidget(model_group)

        scope_group = QGroupBox(tr("范围"))
        scope_layout = QVBoxLayout(scope_group)
        self.scope_combo = QComboBox()
        self.scope_combo.addItem(tr("当前图片"), "current")
        self.scope_combo.addItem(tr("全部图片"), "all")
        scope_layout.addWidget(self.scope_combo)
        self.replace_auto_check = QCheckBox(tr("替换上次推理结果"))
        self.replace_auto_check.setToolTip(tr("人工标注始终保留"))
        scope_layout.addWidget(self.replace_auto_check)
        scope_layout.addWidget(QLabel(tr("类别筛选（不勾选表示全部）:")))
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.class_list.setMaximumHeight(110)
        scope_layout.addWidget(self.class_list)
        layout.addWidget(scope_group)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(90)
        layout.addWidget(self.log_view)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.run_btn = QPushButton(tr("开始推理"))
        self.run_btn.clicked.connect(self._start)
        self.run_btn.setEnabled(False)
        buttons.addWidget(self.run_btn)
        self.close_btn = QPushButton(tr("关闭"))
        self.close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.close_btn)
        layout.addLayout(buttons)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())

    # ---------- 配置 ----------

    def set_images(self, entries, current_index=-1):
        """entries: [(数据集索引, 路径)]，current_index 用于「当前图片」范围。"""
        self._images = [(int(i), str(p)) for i, p in (entries or [])]
        self._current_index = int(current_index)
        total = len(self._images)
        self.scope_combo.setItemText(0, f"{tr('当前图片')} (1/{total})")
        self.scope_combo.setItemText(1, f"{tr('全部图片')} ({total})")
        self._update_run_enabled()

    def settings(self):
        """当前界面配置（供主窗口持久化）。"""
        size = None
        if self.size_spin_w.value() > 0 and self.size_spin_h.value() > 0:
            size = (self.size_spin_w.value(), self.size_spin_h.value())
        return {
            "model_path": self.model_edit.text().strip(),
            "class_names_path": self.names_edit.text().strip(),
            "output_format": self.format_combo.currentData(),
            "conf_threshold": self.conf_spin.value(),
            "iou_threshold": self.iou_spin.value(),
            "input_size": size,
            "providers": ([self.device_combo.currentData()]
                          if self.device_combo.currentData() else None),
        }

    def selected_classes(self):
        checked = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.text())
        return checked

    def _restore_settings(self):
        s = self._settings
        self.model_edit.setText(s["model_path"])
        self.names_edit.setText(s["class_names_path"])
        index = self.format_combo.findData(s["output_format"])
        self.format_combo.setCurrentIndex(max(0, index))
        if s["input_size"]:
            self.size_spin_w.setValue(s["input_size"][0])
            self.size_spin_h.setValue(s["input_size"][1])
        self.conf_spin.setValue(s["conf_threshold"])
        self.iou_spin.setValue(s["iou_threshold"])
        providers = s.get("providers")
        if providers:
            index = self.device_combo.findData(providers[0])
            if index >= 0:
                self.device_combo.setCurrentIndex(index)

    def _on_config_changed(self):
        # 配置变了就作废已加载的模型，避免用旧会话跑新阈值/新格式
        if self._detector is not None:
            self._detector = None
            self._model_size = None
            self._model_format = None
            self.model_info_label.setText(tr("尚未加载模型"))
        self._update_run_enabled()

    def _update_run_enabled(self):
        self.run_btn.setEnabled(
            self._detector is not None and self._load_worker is None
            and self._run_worker is None)

    def _pick_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择模型文件"), self.model_edit.text() or "",
            "ONNX (*.onnx)")
        if path:
            self.model_edit.setText(path)

    def _pick_names(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择类别文件"), self.names_edit.text() or "",
            "Text (*.txt *.names);;All (*)")
        if path:
            self.names_edit.setText(path)

    # ---------- 加载模型 ----------

    def _load_model(self):
        if self._load_worker is not None or self._run_worker is not None:
            return
        if not runtime_available():
            warning(self, tr("错误"), tr("未安装 onnxruntime，无法执行推理"))
            return
        settings = self.settings()
        error = inference_task.validate_settings(settings)
        if error:
            warning(self, tr("错误"), error)
            return
        self._settings = inference_task.normalize_settings(settings)
        self._save_settings()
        self.load_btn.setEnabled(False)
        self.model_info_label.setText(tr("正在加载模型..."))
        worker = _ModelLoadWorker(settings, self)
        worker.loaded.connect(self._on_model_loaded)
        worker.failed.connect(self._on_model_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_load_worker(w))
        self._load_worker = worker
        self._update_run_enabled()
        worker.start()

    def _cleanup_load_worker(self, worker):
        if worker is getattr(self, '_load_worker', None):
            self._load_worker = None
        worker.deleteLater()
        self.load_btn.setEnabled(True)
        self._update_run_enabled()

    def _on_model_loaded(self, detector, size, fmt):
        self._detector = detector
        self._model_size = size
        self._model_format = fmt
        names = detector.class_names
        providers = detector.active_providers
        text = (f"{tr('输入尺寸')}: {size[0]}x{size[1]} · {tr('输出格式')}: "
                f"{fmt.upper()} · {tr('类别数')}: {len(names)}")
        if providers:
            text += f" · {tr('设备')}: {', '.join(providers)}"
        self.model_info_label.setText(text)
        self._fill_class_list(names)
        if not names:
            self._log(tr("模型没有类别信息，请选择类别文件"))
        else:
            self._log(tr("模型加载完成"))

    def _on_model_failed(self, message):
        self.model_info_label.setText(tr("模型加载失败"))
        warning(self, tr("错误"), message)

    def _fill_class_list(self, names):
        self.class_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.class_list.addItem(item)

    # ---------- 执行 ----------

    def _start(self):
        if self._detector is None or self._run_worker is not None:
            return
        entries = self._target_entries()
        if not entries:
            warning(self, tr("错误"), tr("请先加载数据集"))
            return
        self._settings = inference_task.normalize_settings(self.settings())
        self._save_settings()
        self.log_view.clear()
        self.progress.setMaximum(len(entries))
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        paths = [path for _, path in entries]
        keys = [index for index, _ in entries]
        worker = _InferenceWorker(paths, keys, self._detector,
                                  self.selected_classes(), self)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_run_worker(w))
        self._run_worker = worker
        worker.start()

    def _target_entries(self):
        if self.scope_combo.currentData() == "current":
            current = [e for e in self._images if e[0] == self._current_index]
            if current:
                return current
            return self._images[:1]
        return list(self._images)

    def _cleanup_run_worker(self, worker):
        if worker is getattr(self, '_run_worker', None):
            self._run_worker = None
        worker.deleteLater()
        self.load_btn.setEnabled(True)
        self._update_run_enabled()

    def _on_progress(self, done, total, name):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"{tr('推理中')} {done}/{total} · {name}")

    def _on_failed(self, message):
        self.status_label.setText(tr("推理失败"))
        warning(self, tr("错误"), message)

    def _on_finished(self, results):
        failed = inference_task.failure_summary(results)
        self.status_label.setText(
            f"{tr('推理完成')}: {len(results)} {tr('张图')}" + (f" · {failed}" if failed else ""))
        if failed:
            self._log(failed)
        self.apply_requested.emit(results, self.replace_auto_check.isChecked())

    def _log(self, message):
        self.log_view.append(message)

    def _save_settings(self):
        from ..core import config_manager
        config_manager.save_all(inference_settings=self._settings)

    def closeEvent(self, event):
        self._cancel_workers()
        super().closeEvent(event)

    def _cancel_workers(self):
        worker = self._run_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait()
        load_worker = self._load_worker
        if load_worker is not None and load_worker.isRunning():
            load_worker.wait()
