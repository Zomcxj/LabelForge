"""格式转换对话框（自研实现，无外部依赖）

流程:
1. 选择输入格式 -> 图片目录 + 标注路径（yolo 额外需要 data.yaml）
2. 可选点击"校验"，只读检查图片/标注/类别一致性
3. 点击"开始转换"，后台线程写出目标格式，可中途取消

设计约束见 .superpowers/plans/2026-09-14-supervision-integration.md：
默认不覆盖原始数据，取消或失败不保留半成品。
"""
import os

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QCheckBox, QTextEdit, QFileDialog,
)

from ..core.utils import PathUtils
from ..engine import dataset_converter as st
from . import i18n
from .dataset_classifier_dialog import _run_with_progress
from .dialog_helpers import center_on_parent, warning
from .dwm import set_titlebar_dark
from .theme import ThemeManager

tr = i18n.t

#: 下拉框展示名 -> adapter 格式键
FORMAT_CHOICES = (
    ("YOLO", "yolo"),
    ("COCO", "coco"),
    ("Pascal VOC", "voc"),
    ("LabelMe", "labelme"),
)

#: 默认选中的输入/输出格式（最常见的转换方向：LabelMe -> YOLO）
DEFAULT_INPUT_FORMAT = "labelme"
DEFAULT_OUTPUT_FORMAT = "yolo"

#: 任务类型：决定 YOLO 行格式（检测框/分割/旋转框/关键点）
TASK_CHOICES = (
    ("检测框 (det)", "det"),
    ("分割 (seg)", "seg"),
    ("旋转框 (obb)", "obb"),
    ("关键点 (pose)", "pose"),
)
DEFAULT_TASK = "det"


def _format_index(key):
    """格式键 -> 下拉框索引。"""
    for index, (_name, fmt_key) in enumerate(FORMAT_CHOICES):
        if fmt_key == key:
            return index
    raise ValueError(f"未知格式: {key}")


class DatasetToolsDialog(QDialog):
    """格式转换与校验。"""

    def __init__(self, parent=None, default_folder=None):
        super().__init__(parent)
        self._classify_workers = set()
        self.setWindowTitle(tr("格式转换"))
        self.setMinimumWidth(560)

        t = ThemeManager.get_theme()
        self.setStyleSheet(
            f"QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}"
        )

        layout = QVBoxLayout(self)
        self._text_widgets = []

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        layout.addWidget(self._status_lbl)

        fmt_row = QHBoxLayout()
        self._in_fmt_lbl = QLabel(tr("输入格式:"))
        self._text_widgets.append((self._in_fmt_lbl, "输入格式:"))
        fmt_row.addWidget(self._in_fmt_lbl)
        self._in_combo = QComboBox()
        for name, _key in FORMAT_CHOICES:
            self._in_combo.addItem(name)
        self._in_combo.currentIndexChanged.connect(self._on_input_format_changed)
        fmt_row.addWidget(self._in_combo)
        fmt_row.addSpacing(12)
        self._out_fmt_lbl = QLabel(tr("输出格式:"))
        self._text_widgets.append((self._out_fmt_lbl, "输出格式:"))
        fmt_row.addWidget(self._out_fmt_lbl)
        self._out_combo = QComboBox()
        for name, _key in FORMAT_CHOICES:
            self._out_combo.addItem(name)
        fmt_row.addWidget(self._out_combo)
        fmt_row.addStretch()
        layout.addLayout(fmt_row)

        task_row = QHBoxLayout()
        self._task_lbl = QLabel(tr("任务类型:"))
        self._text_widgets.append((self._task_lbl, "任务类型:"))
        task_row.addWidget(self._task_lbl)
        self._task_combo = QComboBox()
        for name, _key in TASK_CHOICES:
            self._task_combo.addItem(tr(name))
        task_row.addWidget(self._task_combo)
        task_row.addStretch()
        layout.addLayout(task_row)

        default_folder = default_folder or ""
        self._images_edit = self._add_path_row(
            layout, "图片目录:", default_folder, self._browse_images)
        self._ann_lbl = QLabel(tr("标注目录:"))
        self._ann_edit = self._add_path_row(
            layout, self._ann_lbl, default_folder, self._browse_annotations)
        self._yaml_lbl = QLabel("data.yaml:")
        self._yaml_edit = self._add_path_row(
            layout, self._yaml_lbl, "", self._browse_yaml,
            button_attr="_yaml_btn")
        self._out_edit = self._add_path_row(
            layout, "输出目录:", "", self._browse_output)

        self._overwrite_cb = QCheckBox(tr("覆盖输出目录"))
        self._text_widgets.append((self._overwrite_cb, "覆盖输出目录"))
        layout.addWidget(self._overwrite_cb)

        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(160)
        # objectName 与处理面板（增强划分）的日志框一致，
        # 复用 theme.py 里的 QTextEdit#logArea 圆角样式
        self._log_area.setObjectName("logArea")
        self._log_area.setPlaceholderText(tr("操作日志将显示在这里..."))
        layout.addWidget(self._log_area)

        btn_row = QHBoxLayout()
        self._validate_btn = QPushButton(tr("校验"))
        self._validate_btn.clicked.connect(self._run_validate)
        btn_row.addWidget(self._validate_btn, 1)
        self._convert_btn = QPushButton(tr("开始转换"))
        self._convert_btn.setObjectName("successBtn")
        self._convert_btn.clicked.connect(self._run_convert)
        btn_row.addWidget(self._convert_btn, 1)
        self._close_btn = QPushButton(tr("关闭"))
        self._close_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._close_btn, 1)
        self._text_widgets.extend([
            (self._validate_btn, "校验"),
            (self._convert_btn, "开始转换"),
            (self._close_btn, "关闭"),
        ])
        layout.addLayout(btn_row)

        # 默认转换方向：LabelMe -> YOLO。必须等 _on_input_format_changed
        # 依赖的控件都建好之后再设，否则信号会在半初始化状态下触发。
        self._in_combo.setCurrentIndex(_format_index(DEFAULT_INPUT_FORMAT))
        self._out_combo.setCurrentIndex(_format_index(DEFAULT_OUTPUT_FORMAT))
        for idx, (_name, key) in enumerate(TASK_CHOICES):
            if key == DEFAULT_TASK:
                self._task_combo.setCurrentIndex(idx)
                break
        self._on_input_format_changed()
        self._refresh_availability()

    # ---------- 构建辅助 ----------

    def _add_path_row(self, layout, label, value, slot, button_attr=None):
        """label 传 i18n key 时登记为可刷新文案；传 QLabel 时由调用方自行管理。

        必须传 key 而不是 tr() 的结果，否则英文切回中文时 tr() 会原样返回
        英文串，文案就回不去了。

        button_attr 给定时把「选择」按钮挂到该实例属性上，
        便于把整行（输入框 + 按钮）一起禁用。
        """
        row = QHBoxLayout()
        if isinstance(label, QLabel):
            row.addWidget(label)
        else:
            lbl = QLabel(tr(label))
            self._text_widgets.append((lbl, label))
            row.addWidget(lbl)
        edit = QLineEdit(value)
        row.addWidget(edit, 1)
        btn = QPushButton(tr("选择"))
        btn.clicked.connect(slot)
        row.addWidget(btn)
        if button_attr:
            setattr(self, button_attr, btn)
        layout.addLayout(row)
        self._text_widgets.append((btn, "选择"))
        return edit

    def _refresh_ui_texts(self):
        """按当前语言刷新对话框标题与所有静态文案。

        对话框每次打开都会重新构造，正常路径下文案已经正确；
        showEvent 里再刷一次，避免语言在构造之后变化时整体过期。
        """
        self.setWindowTitle(tr("格式转换"))
        for widget, key in self._text_widgets:
            widget.setText(tr(key))
        self._log_area.setPlaceholderText(tr("操作日志将显示在这里..."))
        task_combo = getattr(self, "_task_combo", None)
        if task_combo is not None:
            for idx, (name, _key) in enumerate(TASK_CHOICES):
                task_combo.setItemText(idx, tr(name))
        self._on_input_format_changed()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_ui_texts()
        center_on_parent(self)
        set_titlebar_dark(int(self.winId()), ThemeManager.get_mode().value == "dark")

    # ---------- 状态 ----------

    def _refresh_availability(self):
        """转换实现已内置（无外部依赖），始终放行操作按钮。"""
        self._status_lbl.setText("")
        self._status_lbl.setVisible(False)
        self._validate_btn.setEnabled(True)
        self._convert_btn.setEnabled(True)
        return True

    def _input_format(self):
        return FORMAT_CHOICES[self._in_combo.currentIndex()][1]

    def _output_format(self):
        return FORMAT_CHOICES[self._out_combo.currentIndex()][1]

    def _task(self):
        return TASK_CHOICES[self._task_combo.currentIndex()][1]

    def _on_input_format_changed(self):
        """COCO 标注是单文件，yolo 需要 data.yaml，按格式切换输入行。

        data.yaml 行始终保持可见，非 yolo 时整行禁用，避免布局跳动。
        """
        fmt = self._input_format()
        is_coco = fmt == "coco"
        is_yolo = fmt == "yolo"
        self._ann_lbl.setText(tr("标注文件:") if is_coco else tr("标注目录:"))
        self._yaml_lbl.setEnabled(is_yolo)
        self._yaml_edit.setEnabled(is_yolo)
        self._yaml_btn.setEnabled(is_yolo)

    # ---------- 目录选择 ----------

    def _browse_images(self):
        d = QFileDialog.getExistingDirectory(
            self, tr("选择图片目录"), self._images_edit.text())
        if d:
            self._images_edit.setText(PathUtils.to_display_path(d))

    def _browse_annotations(self):
        if self._input_format() == "coco":
            path, _ = QFileDialog.getOpenFileName(
                self, tr("选择标注文件"), self._ann_edit.text(), "JSON (*.json)")
        else:
            path = QFileDialog.getExistingDirectory(
                self, tr("选择标注目录"), self._ann_edit.text())
        if path:
            self._ann_edit.setText(PathUtils.to_display_path(path))

    def _browse_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "data.yaml", self._yaml_edit.text(), "YAML (*.yaml *.yml)")
        if path:
            self._yaml_edit.setText(PathUtils.to_display_path(path))

    def _browse_output(self):
        d = QFileDialog.getExistingDirectory(
            self, tr("选择输出目录"), self._out_edit.text())
        if d:
            self._out_edit.setText(PathUtils.to_display_path(d))

    # ---------- 日志 ----------

    def _log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor = self._log_area.textCursor()
        cursor.movePosition(cursor.End)
        self._log_area.setTextCursor(cursor)
        self._log_area.append(f"[{ts}] {msg}")
        sb = self._log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _log_report(self, report):
        """report 为 ValidationReport.as_dict() 结果。"""
        if report["ok"]:
            self._log(tr("log_sv_validate_ok").format(
                images=report["image_count"],
                annotations=report["annotation_count"],
                classes=len(report["classes"])))
        else:
            self._log(tr("log_sv_validate_fail").format(
                count=len(report["errors"])))
            for msg in report["errors"]:
                self._log(msg)
        for msg in report["warnings"]:
            self._log(tr("log_sv_warn").format(msg=msg))

    # ---------- 输入收集 ----------

    def _collect_inputs(self, need_output):
        """校验界面输入，返回参数字典；不合法时提示并返回 None。"""
        images_dir = self._images_edit.text().strip()
        if not images_dir:
            warning(self, tr("错误"), tr("sv_need_images_dir"))
            return None
        ann_path = self._ann_edit.text().strip()
        if not ann_path:
            warning(self, tr("错误"), tr("sv_need_annotations"))
            return None
        params = {
            "input_format": self._input_format(),
            "images_dir": os.path.normpath(images_dir),
            "annotations_path": os.path.normpath(ann_path),
            "data_yaml_path": None,
            "task": self._task(),
        }
        if params["input_format"] == "yolo":
            yaml_path = self._yaml_edit.text().strip()
            params["data_yaml_path"] = os.path.normpath(yaml_path) if yaml_path else ""
        if not need_output:
            return params
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            warning(self, tr("错误"), tr("sv_need_output"))
            return None
        out_dir = os.path.normpath(out_dir)
        if os.path.normcase(out_dir) == os.path.normcase(params["images_dir"]):
            warning(self, tr("错误"), tr("sv_same_dir"))
            return None
        params["output_dir"] = out_dir
        params["output_format"] = self._output_format()
        params["overwrite"] = self._overwrite_cb.isChecked()
        params["task"] = self._task()
        return params

    # ---------- 动作 ----------

    def _run_validate(self):
        if not self._refresh_availability():
            return
        params = self._collect_inputs(need_output=False)
        if params is None:
            return

        def task(progress_fn, is_interrupted):
            dataset = st.load_dataset(
                params["input_format"], params["images_dir"],
                params["annotations_path"],
                data_yaml_path=params["data_yaml_path"],
                task=params["task"])
            report_task = params["task"] if self._output_format() == "yolo" else "auto"
            return st.validate_dataset(dataset, task=report_task).as_dict()

        result, error = _run_with_progress(
            self, tr("格式转换"), tr("正在校验数据集..."), task)
        if error == "canceled":
            return
        if error:
            self._log(tr("log_sv_err").format(err=error))
            return
        if result:
            self._log_report(result)

    def _run_convert(self):
        if not self._refresh_availability():
            return
        params = self._collect_inputs(need_output=True)
        if params is None:
            return
        self._log(tr("log_sv_start").format(
            src=params["input_format"], dst=params["output_format"]))

        def task(progress_fn, is_interrupted):
            return st.convert_paths(
                params["input_format"], params["images_dir"],
                params["annotations_path"],
                params["output_format"], params["output_dir"],
                data_yaml_path=params["data_yaml_path"],
                overwrite=params["overwrite"],
                on_progress=lambda cur, total: progress_fn(cur, total, None),
                is_interrupted=is_interrupted,
                task=params["task"])

        result, error = _run_with_progress(
            self, tr("格式转换"), tr("正在转换数据集..."), task)
        if error == "canceled":
            self._log(tr("log_sv_cancelled"))
            return
        if error:
            self._log(tr("log_sv_err").format(err=error))
            return
        if not result:
            return
        if result.get("cancelled"):
            self._log(tr("log_sv_cancelled"))
            return
        if result.get("report"):
            self._log_report(result["report"])
        self._log(tr("log_sv_done").format(
            images=result["image_count"],
            annotations=result["annotation_count"],
            path=result["output_dir"]))
