"""
数据集分类对话框 - KMeans[频次+面积]聚类的GUI流程

流程:
1. 设置对话框: 输入图片文件夹(train/val 分割比例), 默认填充当前打开的文件夹
2. 点击"开始分析"后弹出进度条, 后台分析数据集并评估KMeans聚类
3. 弹出聚类指数(K-Silhouette/Inertia)对话框, 用户确认或自定义K
4. 点击"开始分类"后弹出进度条, 按K分组并分割 train/val
"""
import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QFileDialog, QMessageBox, QApplication,
)

from ..core.utils import PathUtils
from ..engine.dataset_classifier import (
    analyze_dataset, build_features, cluster_labels, classify_and_split, recommend_k,
)
from . import i18n
from .dialog_helpers import ThemedMessageBox, center_on_parent, warning
from .dialogs import ProgressDialogFactory
from .theme import ThemeManager
from .dwm import set_titlebar_dark

tr = i18n.t


class _ClassifyWorker(QThread):
    """后台执行数据集分析/分割任务的线程。"""
    progress = pyqtSignal(int, int, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self._func = func
        self._interrupted = False

    def run(self):
        try:
            result = self._func(self.progress.emit, lambda: self._interrupted)
            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

    def cancel(self):
        self._interrupted = True


def _run_with_progress(parent, title, label, task):
    """在子线程中执行 task(progress_fn, is_interrupted)，弹出进度条对话框。

    返回 (result, error)；用户取消时 error 为 "canceled"。
    """
    from PyQt5.QtCore import QEventLoop

    progress = ProgressDialogFactory.create_progress_dialog(parent, title, label, 100)
    worker = _ClassifyWorker(task, parent)
    workers = getattr(parent, '_classify_workers', None)
    if workers is None:
        workers = parent._classify_workers = set()

    result_box = {}
    cancelled = [False]
    completed = [False]

    def _on_progress(cur, total, msg):
        if cancelled[0]:
            return
        progress.setMaximum(max(int(total), 1))
        progress.setValue(int(cur))
        if msg:
            progress.setLabelText(str(msg))

    def _on_done(result):
        if not cancelled[0]:
            result_box['result'] = result
        completed[0] = True
        loop.quit()

    def _on_error(err):
        if not cancelled[0]:
            result_box['error'] = err
        completed[0] = True
        loop.quit()

    def _on_canceled():
        if completed[0]:
            return
        cancelled[0] = True
        worker.cancel()
        progress.setLabelText(tr("正在中断..."))

    loop = QEventLoop()
    worker.progress.connect(_on_progress)
    worker.finished_ok.connect(_on_done)
    worker.failed.connect(_on_error)
    progress.canceled.connect(_on_canceled)
    worker.finished.connect(loop.quit)
    progress.show()
    worker.start()
    loop.exec_()
    progress.close()
    if worker.isRunning():
        worker.wait()
    workers.discard(worker)
    worker.deleteLater()
    if cancelled[0]:
        return None, "canceled"
    return result_box.get('result'), result_box.get('error')


class DatasetClassifierDialog(QDialog):
    """设置对话框: 输入文件夹 + train/val 比例，并触发分析。"""

    def __init__(self, editor, parent=None, default_folder=None):
        super().__init__(parent or editor)
        self._editor = editor
        self._classify_workers = set()
        self._analysis = None
        self.setWindowTitle(tr("数据集分类"))
        self.setMinimumWidth(480)

        t = ThemeManager.get_theme()
        self.setStyleSheet(
            f"QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}"
        )

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(tr("图片文件夹:")))
        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit(default_folder or os.getcwd())
        folder_row.addWidget(self._folder_edit, 1)
        browse_btn = QPushButton(tr("选择"))
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel(tr("分割比例:")))
        self._train_spin = QDoubleSpinBox()
        self._train_spin.setRange(0.5, 0.95)
        self._train_spin.setSingleStep(0.05)
        self._train_spin.setDecimals(2)
        self._train_spin.setValue(0.8)
        self._train_spin.valueChanged.connect(self._on_train_changed)
        ratio_row.addWidget(QLabel(tr("train")))
        ratio_row.addWidget(self._train_spin)
        self._val_label = QLabel()
        ratio_row.addWidget(self._val_label)
        ratio_row.addStretch()
        layout.addLayout(ratio_row)
        self._on_train_changed()

        btn_row = QHBoxLayout()
        self._analyze_btn = QPushButton(tr("开始分析"))
        self._analyze_btn.setObjectName("successBtn")
        self._analyze_btn.clicked.connect(self._start_analysis)
        btn_row.addWidget(self._analyze_btn, 1)
        cancel_btn = QPushButton(tr("取消"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn, 1)
        layout.addLayout(btn_row)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self)
        set_titlebar_dark(int(self.winId()), ThemeManager.get_mode().value == "dark")

    def _browse_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("选择图片文件夹"), self._folder_edit.text())
        if d:
            self._folder_edit.setText(PathUtils.to_display_path(d))

    def _on_train_changed(self):
        val = 1.0 - self._train_spin.value()
        self._val_label.setText(f"val: {val:.0%}")

    def _start_analysis(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            warning(self, tr("错误"), tr("路径不存在"))
            return
        folder = os.path.normpath(folder)
        self._folder_edit.setText(PathUtils.to_display_path(folder))
        train_ratio = self._train_spin.value()

        def task(progress_fn, is_interrupted):
            return self._analyze_task(progress_fn, is_interrupted, folder)

        result, error = _run_with_progress(
            self, tr("数据集分类"), tr("正在分析数据集..."), task,
        )
        if error == "canceled":
            return
        if error:
            warning(self, tr("错误"), error)
            return
        if result is None:
            warning(self, tr("错误"), tr("未找到有效的标签"))
            return
        self._analysis = result
        info_dlg = KMeansInfoDialog(self, result, train_ratio)
        if info_dlg.exec_():
            self.accept()

    @staticmethod
    def _analyze_task(progress_fn, is_interrupted, folder):
        analysis = analyze_dataset(folder, progress_fn, is_interrupted)
        if analysis is None:
            return None
        images, label_image_count, mean_areas, image_labels, image_primary_label, classified_images = analysis
        if not images:
            return None
        labels, X_scaled, X = build_features(label_image_count, mean_areas)
        if len(labels) < 2:
            return None
        best_k, scores = recommend_k(X_scaled)
        if best_k is None:
            return None
        return {
            'folder': folder,
            'analysis': analysis,
            'labels': labels,
            'X': X,
            'X_scaled': X_scaled,
            'best_k': best_k,
            'scores': scores,
            'n_labels': len(labels),
        }


class KMeansInfoDialog(QDialog):
    """展示KMeans聚类指数(Silhouette/Inertia)，让用户确认K后执行分类。"""

    def __init__(self, parent, info, train_ratio):
        super().__init__(parent)
        self._info = info
        self._train_ratio = train_ratio
        self._classify_workers = set()
        self.setWindowTitle(tr("KMeans聚类结果"))
        self.setMinimumWidth(560)

        t = ThemeManager.get_theme()
        self.setStyleSheet(
            f"QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}"
        )

        layout = QVBoxLayout(self)

        title = QLabel(tr("KMeans聚类指数 (Silhouette 越高越好, Inertia 越低越好)"))
        title.setWordWrap(True)
        layout.addWidget(title)

        self._k_table = QTextEdit()
        self._k_table.setReadOnly(True)
        self._k_table.setFixedHeight(150)
        self._k_table.setObjectName("logArea")
        layout.addWidget(self._k_table)

        k_row = QHBoxLayout()
        self._rec_label = QLabel()
        k_row.addWidget(self._rec_label, 1)
        k_row.addWidget(QLabel(tr("自定义K")))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, max(2, info['n_labels']))
        self._k_spin.setValue(max(2, min(info['best_k'], info['n_labels'])))
        self._k_spin.valueChanged.connect(self._refresh_groups_preview)
        k_row.addWidget(self._k_spin)
        layout.addLayout(k_row)

        layout.addWidget(QLabel(tr("分组预览")))
        self._group_text = QTextEdit()
        self._group_text.setReadOnly(True)
        self._group_text.setMinimumHeight(150)
        self._group_text.setObjectName("logArea")
        layout.addWidget(self._group_text)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton(tr("开始分类"))
        self._start_btn.setObjectName("successBtn")
        self._start_btn.clicked.connect(self._start_split)
        btn_row.addWidget(self._start_btn, 1)
        cancel_btn = QPushButton(tr("取消"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn, 1)
        layout.addLayout(btn_row)

        self._refresh_k_table()
        self._refresh_groups_preview()

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self)
        set_titlebar_dark(int(self.winId()), ThemeManager.get_mode().value == "dark")

    def _refresh_k_table(self):
        scores = self._info['scores']
        best_k = self._info['best_k']
        n_labels = self._info['n_labels']
        if not scores:
            self._k_table.setPlainText(tr("标签数量不足，无法聚类评估"))
        else:
            lines = [f"{'K':>3}  {'Silhouette':>12}  {'Inertia':>14}  "]
            lines.append("-" * 42)
            for k in sorted(scores):
                sil, inert = scores[k]
                mark = tr("← 推荐") if k == best_k else ""
                lines.append(f"{k:>3}  {sil:>12.4f}  {inert:>14.1f}  {mark}")
            self._k_table.setPlainText("\n".join(lines))
        self._rec_label.setText(
            tr("推荐聚类数 K = {k}，共 {n} 个标签").format(k=best_k, n=n_labels)
        )

    def _refresh_groups_preview(self):
        info = self._info
        k = self._k_spin.value()
        group_labels, _, _ = cluster_labels(info['X_scaled'], k, info['labels'])
        label_image_count = info['analysis'][1]
        mean_areas = info['analysis'][2]
        lines = []
        for gi, glabels in enumerate(group_labels):
            lines.append(f"g{gi} ({len(glabels)} {tr('个标签')}):")
            for l in sorted(glabels, key=lambda x: label_image_count[x], reverse=True):
                lines.append(
                    f"    {l}: {tr('频次')} {label_image_count[l]}, "
                    f"{tr('平均面积')} {mean_areas.get(l, 0):.1f}"
                )
        self._group_text.setPlainText("\n".join(lines))

    def _start_split(self):
        info = self._info
        k = self._k_spin.value()

        def task(progress_fn, is_interrupted):
            return classify_and_split(
                info['folder'], self._train_ratio, k, analysis=info['analysis'],
                progress_callback=progress_fn, is_interrupted=is_interrupted,
            )

        result, error = _run_with_progress(
            self, tr("数据集分类"), tr("正在分割数据集..."), task,
        )
        if error == "canceled":
            return
        if error:
            warning(self, tr("错误"), error)
            return
        if result is None:
            warning(self, tr("错误"), tr("分割失败"))
            return
        self._show_summary(result)
        self.accept()

    def _show_summary(self, result):
        lines = [
            f"{tr('总图片数')}: {len(result['images'])}",
            f"train: {len(result['train_images'])}, val: {len(result['val_images'])}",
        ]
        for gi in range(result['k']):
            lines.append(
                f"g{gi}: train {result['group_train_count'][gi]}, "
                f"val {result['group_val_count'][gi]} "
                f"({len(result['group_labels'][gi])} {tr('个标签')})"
            )
        lines.append(f"{tr('输出目录')}: {result['output_dir']}")
        box = ThemedMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle(tr("分割完成"))
        box.setText("\n".join(lines))
        box.setStandardButtons(QMessageBox.Ok)
        ok_btn = box.button(QMessageBox.Ok)
        if ok_btn:
            ok_btn.setText(tr("确定"))
        box.exec_()
