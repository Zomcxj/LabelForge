"""视频弹窗：抽帧与合成两个标签页。

- **抽帧页**：选视频 → 探测 → 按「每秒抽几帧」导出图片 + 清单。
- **合成页**：选图片目录 → 自动判断有没有 `video_manifest.json`：
  有就按清单的时长合成（不用填帧率），没有才用用户填的「每秒几帧」。

两页各有自己的动作按钮与提示区，`action` 记录用户按的是哪一个，
主窗口据此分发到对应的后台任务。

按「开始」只发 `start_requested`，**不关窗**：任务在跑时弹窗留着显示底部
进度条（参数锁住），跑完由主窗口 `accept()` 关掉。参数非法不发信号，
在当前页的提示区说明原因。
"""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QProgressBar, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget,
)

from .dialog_helpers import center_on_parent, sync_titlebar
from .i18n import t as tr
from .theme import ThemeManager

VIDEO_FILTER = ("视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm *.m4v);;"
                "所有文件 (*)")

ACTION_EXTRACT = "extract"
ACTION_COMPOSE = "compose"
ACTION_RENDER = "render"


class _OverlayGroup:
    """一组「叠加标签框」控件。

    两个标签页各有一组：合成页的类别来自 sidecar 标注，推理成片页的类别
    来自模型的类别表，数据源不同，不能共用同一个列表控件（两页同时存在，
    各自要显示各自的内容）。
    """

    def __init__(self, overlay_check, label_list, all_btn, none_btn,
                 show_label_check, show_score_check):
        self.overlay_check = overlay_check
        self.label_list = label_list
        self.all_btn = all_btn
        self.none_btn = none_btn
        self.show_label_check = show_label_check
        self.show_score_check = show_score_check

    def widgets(self):
        return (self.label_list, self.all_btn, self.none_btn,
                self.show_label_check, self.show_score_check)

    def set_checked(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        for index in range(self.label_list.count()):
            self.label_list.item(index).setCheckState(state)

    def selected(self):
        """勾选的类别名列表（顺序跟列表一致）。"""
        chosen = []
        for index in range(self.label_list.count()):
            item = self.label_list.item(index)
            if item.checkState() == Qt.Checked:
                chosen.append(item.data(Qt.UserRole))
        return chosen

    def fill(self, labels):
        """重建类别列表（有内容时主开关默认打开：用户就是要看框）。"""
        self.label_list.clear()
        for label, count in (labels or {}).items():
            text = f"{label} ({count})" if count else str(label)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.label_list.addItem(item)
        self.overlay_check.setChecked(bool(labels))

    def options(self):
        """当前勾选；没有可选类别或主开关关着时返回 None。"""
        if self.label_list.count() == 0:
            return None
        if not self.overlay_check.isChecked():
            return None
        return {
            "labels": self.selected(),
            "show_labels": self.show_label_check.isChecked(),
            "show_scores": self.show_score_check.isChecked(),
        }


class VideoDialog(QDialog):
    """抽帧 / 合成配置弹窗。

    `action` 是用户最后按下的动作（`extract` 或 `compose`）；
    `settings()` 按当前动作返回对应参数。
    """

    start_requested = Signal(str, dict)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path = ""
        self._info = None
        self._compose_dir = ""
        self._compose_manifest = None      # 探测到的清单，None = 没有
        self._compose_plan = None
        self._compose_images = []          # 合成要用的图片（扫标注用）
        self._render_class_names = []      # 模型类别表（推理成片页列类别）
        self._render_model_name = ""
        self._running = False              # 任务在跑：锁参数、亮进度条
        self.action = ACTION_EXTRACT
        self._build_ui()
        self._apply_style()
        self._sync_extract_enabled()
        self._sync_compose_enabled()
        self._sync_render_enabled()

    # ---------- 界面 ----------

    def _build_ui(self):
        self.setWindowTitle(tr("选择视频抽帧或图片合成"))
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_extract_tab(), tr("抽帧"))
        self.tabs.addTab(self._build_compose_tab(), tr("合成"))
        self.tabs.addTab(self._build_render_tab(), tr("推理成片"))
        layout.addWidget(self.tabs)

        # 底部进度条：任务跑起来才显示，空闲时不占视觉
        self.progress_label = QLabel("")
        self.progress_label.setObjectName("videoHint")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("videoProgress")
        # 与全项目进度条一致：16px 高 + 8px 圆角（原来 12px 直角，看着像另一种控件）
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    # ---------- 任务状态 ----------

    def _verb(self):
        if self.action == ACTION_COMPOSE:
            return tr("合成中")
        if self.action == ACTION_RENDER:
            return tr("推理成片中")
        return tr("抽帧中")

    def set_running(self, running):
        """锁/解锁参数与开始按钮；「取消」按钮的含义随之改变。"""
        self._running = bool(running)
        self._sync_extract_enabled()
        self._sync_compose_enabled()
        self._sync_render_enabled()

    def begin_run(self):
        """按下「开始」：先亮不确定态进度条，收到进度再变确定态。"""
        self.set_running(True)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(self._verb())

    def set_progress(self, done, total):
        """进度回灌：`total` 为 0 时保持不确定态。"""
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        if total > 0:
            self.progress_bar.setRange(0, int(total))
            self.progress_bar.setValue(int(done))
            self.progress_label.setText(f"{self._verb()} {int(done)}/{int(total)}")
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText(self._verb())

    def show_hint(self, message):
        """在当前页的提示区说明原因（不关窗）。"""
        if self.action == ACTION_COMPOSE:
            self.compose_hint_label.setText(message)
        elif self.action == ACTION_RENDER:
            self.render_hint_label.setText(message)
        else:
            self.hint_label.setText(message)

    def _on_cancel_clicked(self):
        """跑任务时「取消」是取消任务（留着看进度），空闲时才是关窗。"""
        if self._running:
            self.progress_label.setText(tr("正在取消..."))
            self.cancel_requested.emit()
            return
        self.reject()

    def _build_extract_tab(self):
        self.extract_tab = QWidget()
        layout = QVBoxLayout(self.extract_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source = QHBoxLayout()
        source.setSpacing(6)
        self.pick_btn = QPushButton(tr("选择视频"))
        self.pick_btn.setObjectName("optionsBtn")
        self.pick_btn.setFixedHeight(24)
        self.pick_btn.setToolTip(tr("选择视频文件"))
        self.pick_btn.clicked.connect(self._pick_video)
        source.addWidget(self.pick_btn)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(tr("尚未选择视频"))
        source.addWidget(self.path_edit, 1)
        layout.addLayout(source)

        self.info_label = QLabel("")
        self.info_label.setObjectName("videoInfo")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        self.fps_lbl = QLabel(tr("每秒抽几帧"))
        grid.addWidget(self.fps_lbl, 0, 0)
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.01, 1000.0)
        self.fps_spin.setDecimals(3)
        self.fps_spin.setSingleStep(1.0)
        self.fps_spin.setValue(1.0)
        self.fps_spin.setToolTip(tr("超过源帧率时按源帧率上限"))
        grid.addWidget(self.fps_spin, 0, 1)

        self.format_lbl = QLabel(tr("图片格式"))
        grid.addWidget(self.format_lbl, 0, 2)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["jpg", "png"])
        self.format_combo.currentTextChanged.connect(self._on_format_changed)
        grid.addWidget(self.format_combo, 0, 3)

        self.start_lbl = QLabel(tr("起始时间(秒)"))
        grid.addWidget(self.start_lbl, 1, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0.0, 100000.0)
        self.start_spin.setDecimals(3)
        grid.addWidget(self.start_spin, 1, 1)

        self.end_lbl = QLabel(tr("结束时间(秒)"))
        grid.addWidget(self.end_lbl, 1, 2)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0.0, 100000.0)
        self.end_spin.setDecimals(3)
        grid.addWidget(self.end_spin, 1, 3)

        self.quality_lbl = QLabel(tr("jpg 质量"))
        grid.addWidget(self.quality_lbl, 2, 0)
        self.quality_spin = QDoubleSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setDecimals(0)
        self.quality_spin.setValue(95)
        grid.addWidget(self.quality_spin, 2, 1)

        self.out_lbl = QLabel(tr("输出目录"))
        grid.addWidget(self.out_lbl, 3, 0)
        # 只读：输出目录固定在视频同级、自动创建，用户不能另选位置
        self.out_edit = QLineEdit()
        self.out_edit.setReadOnly(True)
        grid.addWidget(self.out_edit, 3, 1, 1, 3)

        layout.addLayout(grid)

        self.hint_label = QLabel("")
        self.hint_label.setObjectName("videoHint")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_btn = QPushButton(tr("取消"))
        self.cancel_btn.setObjectName("optionsBtn")
        self.cancel_btn.setFixedHeight(26)
        self.cancel_btn.clicked.connect(self._on_cancel_clicked)
        buttons.addWidget(self.cancel_btn)

        self.start_btn = QPushButton(tr("开始抽帧"))
        self.start_btn.setObjectName("optionsBtn")
        self.start_btn.setFixedHeight(26)
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self._on_start_extract)
        buttons.addWidget(self.start_btn)
        layout.addLayout(buttons)

        self._on_format_changed(self.format_combo.currentText())
        return self.extract_tab

    def _build_compose_tab(self):
        self.compose_tab = QWidget()
        layout = QVBoxLayout(self.compose_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source = QHBoxLayout()
        source.setSpacing(6)
        self.compose_pick_btn = QPushButton(tr("选择图片目录"))
        self.compose_pick_btn.setObjectName("optionsBtn")
        self.compose_pick_btn.setFixedHeight(24)
        self.compose_pick_btn.clicked.connect(self._pick_compose_dir)
        source.addWidget(self.compose_pick_btn)

        self.compose_dir_edit = QLineEdit()
        self.compose_dir_edit.setReadOnly(True)
        self.compose_dir_edit.setPlaceholderText(tr("请先选择图片目录"))
        source.addWidget(self.compose_dir_edit, 1)
        layout.addLayout(source)

        self.compose_info_label = QLabel("")
        self.compose_info_label.setObjectName("videoInfo")
        self.compose_info_label.setWordWrap(True)
        layout.addWidget(self.compose_info_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.compose_fps_lbl = QLabel(tr("每秒几帧"))
        grid.addWidget(self.compose_fps_lbl, 0, 0)
        self.compose_fps_spin = QDoubleSpinBox()
        self.compose_fps_spin.setRange(0.01, 1000.0)
        self.compose_fps_spin.setDecimals(3)
        self.compose_fps_spin.setValue(5.0)
        self.compose_fps_spin.setToolTip(tr("无清单时按每秒几帧合成"))
        grid.addWidget(self.compose_fps_spin, 0, 1)

        self.compose_out_lbl = QLabel(tr("输出视频"))
        grid.addWidget(self.compose_out_lbl, 1, 0)
        # 只读：输出路径由来源推导，避免覆盖别人的文件
        self.compose_out_edit = QLineEdit()
        self.compose_out_edit.setReadOnly(True)
        grid.addWidget(self.compose_out_edit, 1, 1)

        # 合成范围：从第几张到第几张（包含两端），默认 0 到最后一张
        self.compose_range_lbl = QLabel(tr("合成范围"))
        grid.addWidget(self.compose_range_lbl, 2, 0)
        range_row = QHBoxLayout()
        range_row.setSpacing(6)
        self.compose_start_spin = QSpinBox()
        self.compose_start_spin.setRange(0, 0)
        self.compose_start_spin.setValue(0)
        self.compose_start_spin.setToolTip(tr("从第几张开始（含这张，从 0 数）"))
        range_row.addWidget(self.compose_start_spin)
        range_row.addWidget(QLabel(tr("到")))
        self.compose_end_spin = QSpinBox()
        self.compose_end_spin.setRange(0, 0)
        self.compose_end_spin.setValue(0)
        self.compose_end_spin.setToolTip(tr("合成到第几张（含这张）"))
        range_row.addWidget(self.compose_end_spin)
        range_row.addStretch()
        grid.addLayout(range_row, 2, 1)
        # 输入完就纠正越界（起始 > 结束这种）
        self.compose_start_spin.valueChanged.connect(self._on_compose_range_changed)
        self.compose_end_spin.valueChanged.connect(self._on_compose_range_changed)

        layout.addLayout(grid)

        layout.addWidget(self._build_overlay_group("compose"))

        self.compose_hint_label = QLabel("")
        self.compose_hint_label.setObjectName("videoHint")
        self.compose_hint_label.setWordWrap(True)
        layout.addWidget(self.compose_hint_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.compose_cancel_btn = QPushButton(tr("取消"))
        self.compose_cancel_btn.setObjectName("optionsBtn")
        self.compose_cancel_btn.setFixedHeight(26)
        self.compose_cancel_btn.clicked.connect(self._on_cancel_clicked)
        buttons.addWidget(self.compose_cancel_btn)

        self.compose_btn = QPushButton(tr("开始合成"))
        self.compose_btn.setObjectName("optionsBtn")
        self.compose_btn.setFixedHeight(26)
        self.compose_btn.clicked.connect(self._on_start_compose)
        buttons.addWidget(self.compose_btn)
        layout.addLayout(buttons)

        return self.compose_tab

    def _build_render_tab(self):
        """④A 逐帧推理成片：源视频 + 模型 + 叠加选项 + 输出。

        与抽帧页共用同一个「已选视频」（同一份源，选一次就能抽帧或成片），
        但输出路径与叠加选项独立。
        """
        self.render_tab = QWidget()
        layout = QVBoxLayout(self.render_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        source = QHBoxLayout()
        source.setSpacing(6)
        self.render_pick_btn = QPushButton(tr("选择视频"))
        self.render_pick_btn.setObjectName("optionsBtn")
        self.render_pick_btn.setFixedHeight(24)
        self.render_pick_btn.setToolTip(tr("选择视频文件"))
        self.render_pick_btn.clicked.connect(self._pick_video)
        source.addWidget(self.render_pick_btn)

        self.render_path_edit = QLineEdit()
        self.render_path_edit.setReadOnly(True)
        self.render_path_edit.setPlaceholderText(tr("尚未选择视频"))
        source.addWidget(self.render_path_edit, 1)
        layout.addLayout(source)

        self.render_info_label = QLabel("")
        self.render_info_label.setObjectName("videoInfo")
        self.render_info_label.setWordWrap(True)
        layout.addWidget(self.render_info_label)

        self.render_model_label = QLabel("")
        self.render_model_label.setObjectName("videoInfo")
        self.render_model_label.setWordWrap(True)
        layout.addWidget(self.render_model_label)

        # 没加载模型时的显眼提示：只把开始按钮变灰，用户会以为这页坏了。
        # 用强调色（videoHint）并说清去哪加载，加载完自动隐藏。
        self.render_model_hint_label = QLabel(tr(
            "还没有加载模型：点画布上方参数行里的「加载模型」，"
            "本页才能开始逐帧推理。"))
        self.render_model_hint_label.setObjectName("videoHint")
        self.render_model_hint_label.setWordWrap(True)
        self.render_model_hint_label.setVisible(False)
        layout.addWidget(self.render_model_hint_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.render_out_lbl = QLabel(tr("输出视频"))
        grid.addWidget(self.render_out_lbl, 0, 0)
        self.render_out_edit = QLineEdit()
        self.render_out_edit.setReadOnly(True)
        grid.addWidget(self.render_out_edit, 0, 1)
        layout.addLayout(grid)

        layout.addWidget(self._build_overlay_group("render"))

        note = QLabel(tr("逐帧推理：每一帧都推理并画框，速度比抽帧慢；"
                         "框会烧进画面，之后不能再改。"))
        note.setObjectName("videoHint")
        note.setWordWrap(True)
        layout.addWidget(note)

        self.render_hint_label = QLabel("")
        self.render_hint_label.setObjectName("videoHint")
        self.render_hint_label.setWordWrap(True)
        layout.addWidget(self.render_hint_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.render_cancel_btn = QPushButton(tr("取消"))
        self.render_cancel_btn.setObjectName("optionsBtn")
        self.render_cancel_btn.setFixedHeight(26)
        self.render_cancel_btn.clicked.connect(self._on_cancel_clicked)
        buttons.addWidget(self.render_cancel_btn)

        self.render_btn = QPushButton(tr("开始推理成片"))
        self.render_btn.setObjectName("optionsBtn")
        self.render_btn.setFixedHeight(26)
        self.render_btn.clicked.connect(self._on_start_render)
        buttons.addWidget(self.render_btn)
        layout.addLayout(buttons)

        return self.render_tab

    def _build_overlay_group(self, attr):
        """「叠加标签框」一组控件：主开关 + 类别多选 + 类别名/置信度开关。

        默认跟着画布走（显示类别名、显示置信度都开），免得编辑器里看到的
        和导出的视频不一样。

        `attr` 是属性前缀：`compose` 或 `render`，两组控件各自独立（合成页
        的类别来自 sidecar，推理成片页的来自模型类别表）。
        """
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        overlay_check = QCheckBox(tr("叠加标签框"))
        overlay_check.setToolTip(tr("在画面里画出每张图的标注框"))
        overlay_check.stateChanged.connect(self._on_overlay_toggled)
        layout.addWidget(overlay_check)

        label_list = QListWidget()
        label_list.setObjectName(f"{attr}LabelList")
        label_list.setSelectionMode(QAbstractItemView.NoSelection)
        label_list.setFixedHeight(84)
        layout.addWidget(label_list)

        tools = QHBoxLayout()
        tools.setSpacing(6)
        tools.addWidget(QLabel(tr("类别")))
        all_btn = QPushButton(tr("全选"))
        all_btn.setObjectName("optionsBtn")
        all_btn.setFixedHeight(22)
        tools.addWidget(all_btn)
        none_btn = QPushButton(tr("全不选"))
        none_btn.setObjectName("optionsBtn")
        none_btn.setFixedHeight(22)
        tools.addWidget(none_btn)
        tools.addStretch()
        layout.addLayout(tools)

        texts = QHBoxLayout()
        texts.setSpacing(12)
        show_label_check = QCheckBox(tr("类别名"))
        show_label_check.setChecked(True)
        texts.addWidget(show_label_check)
        show_score_check = QCheckBox(tr("置信度"))
        show_score_check.setChecked(True)
        texts.addWidget(show_score_check)
        texts.addStretch()
        layout.addLayout(texts)

        group = _OverlayGroup(overlay_check, label_list, all_btn, none_btn,
                              show_label_check, show_score_check)
        setattr(self, f"_{attr}_overlay", group)
        # 兼容既有属性名：合成页的控件与测试都按 compose_* 访问
        setattr(self, f"{attr}_overlay_check", overlay_check)
        setattr(self, f"{attr}_label_list", label_list)
        setattr(self, f"{attr}_all_btn", all_btn)
        setattr(self, f"{attr}_none_btn", none_btn)
        setattr(self, f"{attr}_show_label_check", show_label_check)
        setattr(self, f"{attr}_show_score_check", show_score_check)
        all_btn.clicked.connect(lambda: group.set_checked(True))
        none_btn.clicked.connect(lambda: group.set_checked(False))
        return box

    def _apply_style(self):
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QTabWidget::pane {{ border: 1px solid {t['border_color']};
                border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background-color: {t['panel_bg']};
                color: {t['text_secondary']}; padding: 5px 16px;
                border: 1px solid {t['border_color']}; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ color: {t['text_primary']};
                background-color: {t['widget_bg']}; }}
            QLabel {{ color: {t['text_secondary']}; }}
            QLabel#videoInfo {{ color: {t['text_primary']}; }}
            QLabel#videoHint {{ color: {t['accent']}; }}
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
                background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px;
                padding: 2px 6px;
            }}
            QPushButton {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px;
                padding: 3px 12px; }}
            QPushButton:hover {{ border-color: {t['accent']}; }}
            QPushButton:disabled {{ color: {t['text_secondary']}; }}
            QCheckBox {{ color: {t['text_secondary']}; padding: 2px; }}
            QListWidget#composeLabelList {{
                background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px;
            }}
            QListWidget#composeLabelList::item {{ padding: 2px 4px; }}
            {ThemeManager.get_progress_style(text_visible=False)}
        """)

    # ---------- 对外接口 ----------

    def video_path(self):
        return self._video_path

    def images_dir(self):
        return self._compose_dir

    def has_manifest(self):
        return self._compose_manifest is not None

    def settings(self):
        """按 `action` 返回对应参数。

        抽帧：起始为 0 且结束为全片时按「全片」传 None——全片抽帧要读至
        EOF 核对实际帧数，传显式范围会丢掉这层检查。
        合成：有清单时只传清单路径（时长以清单为准），没有才传帧率。
        """
        if self.action == ACTION_RENDER:
            settings = {
                "video_path": self._video_path,
                "output_path": self.render_out_edit.text(),
            }
            render = self._render_overlay.options()
            if render is not None:
                settings["render"] = render
            return settings

        if self.action == ACTION_COMPOSE:
            start_index, end_index = self.compose_range()
            settings = {
                "images_dir": self._compose_dir,
                "output_path": self.compose_out_edit.text(),
                "start_index": start_index,
                "end_index": end_index,
            }
            if self._compose_manifest is None:
                settings["output_fps"] = self.compose_fps_spin.value()
            else:
                settings["manifest_path"] = self._compose_manifest
            render = self.compose_overlay_settings()
            if render is not None:
                settings["render"] = render
            return settings

        start = self.start_spin.value()
        end = self.end_spin.value()
        full = start <= 1e-9 and self._info is not None and \
            abs(end - self._info.duration) <= 1e-6
        return {
            "video_path": self._video_path,
            "target_fps": self.fps_spin.value(),
            "start": None if full else start,
            "end": None if full else end,
            "image_format": self.format_combo.currentText(),
            "quality": int(self.quality_spin.value()),
        }

    def validate(self):
        """按当前动作校验；返回错误文案，通过则返回 None。"""
        if self.action == ACTION_RENDER:
            if not self._video_path:
                return tr("请先选择视频文件")
            if self._info is None:
                return self.render_hint_label.text() or tr("视频文件不存在")
            if not getattr(self, "_render_class_names", None):
                return tr("请先加载模型")
            if not self.render_out_edit.text():
                return tr("输出路径为空")
            return None

        if self.action == ACTION_COMPOSE:
            if not self._compose_dir:
                return tr("请先选择图片目录")
            if self._compose_manifest is None and self.compose_fps_spin.value() <= 0:
                return tr("每秒几帧必须为正数")
            return None

        if not self._video_path:
            return tr("请先选择视频文件")
        if self._info is None:
            return self.hint_label.text() or tr("视频文件不存在")
        if self.fps_spin.value() <= 0:
            return tr("帧率必须为正数")
        start, end = self.start_spin.value(), self.end_spin.value()
        if not (0.0 <= start < end <= self._info.duration + 1e-9):
            return tr("时间范围非法")
        return None

    # ---------- 抽帧页 ----------

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择视频"), self._video_path or "", VIDEO_FILTER)
        if not path:
            return
        self.set_video(path)

    def set_video(self, path):
        """选好视频后探测并填默认值；探测失败在提示区说明。"""
        from ..engine import video_io

        self._video_path = path
        self.path_edit.setText(path)
        self.path_edit.setToolTip(path)
        try:
            self._info = video_io.probe(path)
        except video_io.VideoError as exc:
            self._info = None
            self.hint_label.setText(str(exc))
            self.info_label.setText("")
            self.out_edit.setText("")
            self._sync_extract_enabled()
            self._sync_render_enabled()
            return

        info = self._info
        self.hint_label.setText("")
        self.info_label.setText(
            f"{info.width}×{info.height}  {info.fps:.3f} FPS  "
            f"{info.frame_count} 帧  {info.duration:.3f} 秒")
        # 抽帧不重复源帧：上限直接给源帧率，超出会被引擎钳制
        self.fps_spin.setMaximum(max(0.01, info.fps))
        self.fps_spin.setValue(min(max(1.0, 0.01), info.fps))
        self.start_spin.setMaximum(info.duration)
        self.end_spin.setMaximum(info.duration)
        self.end_spin.setValue(info.duration)
        self.out_edit.setText(video_io.output_dir_for(path))
        self._sync_extract_enabled()
        self._sync_render_enabled()

    def _sync_extract_enabled(self):
        ready = self._info is not None and not self._running
        for widget in (self.fps_spin, self.start_spin, self.end_spin,
                       self.format_combo, self.quality_spin):
            widget.setEnabled(ready)
        self.pick_btn.setEnabled(not self._running)
        self.start_btn.setEnabled(ready)
        self._on_format_changed(self.format_combo.currentText())

    def _on_format_changed(self, text):
        # png 无损，质量参数只对 jpg 有意义
        self.quality_spin.setEnabled(
            self._info is not None and not self._running
            and text.lower() in ("jpg", "jpeg"))

    def _on_start_extract(self):
        self.action = ACTION_EXTRACT
        message = self.validate()
        if message:
            self.hint_label.setText(message)
            return
        self.start_requested.emit(ACTION_EXTRACT, self.settings())

    # ---------- 合成页 ----------

    def _pick_compose_dir(self):
        from ..core.config import SUPPORTED_IMAGE_EXTENSIONS  # noqa: F401
        path = QFileDialog.getExistingDirectory(
            self, tr("选择图片目录"), self._compose_dir or "")
        if not path:
            return
        self.set_compose_dir(path)

    def set_compose_dir(self, path):
        """选好目录后自动判断有没有清单，并推导输出路径。"""
        from ..engine import video_io

        self._compose_dir = path
        self.compose_dir_edit.setText(path)
        self.compose_dir_edit.setToolTip(path)
        self.compose_hint_label.setText("")

        manifest_path = os.path.join(path, video_io.MANIFEST_NAME)
        self._compose_manifest = None
        self._compose_plan = None

        if os.path.isfile(manifest_path):
            try:
                manifest = video_io.read_manifest(manifest_path)
            except video_io.VideoError as exc:
                self.compose_hint_label.setText(str(exc))
                self.compose_info_label.setText("")
                self.compose_out_edit.setText("")
                self._fill_compose_labels([])
                self._sync_compose_enabled()
                return
            self._compose_manifest = manifest_path
            count = len(manifest.frames)
            duration = manifest.end_time - manifest.start_time
            fps = count / duration if duration > 0 else 0.0
            self.compose_info_label.setText(
                f"{tr('有清单，按清单合成')}：{count} {tr('张图片')}  "
                f"{duration:.3f} {tr('秒')}  {fps:.3f} FPS")
            # 显示实际会用的帧率：禁用状态下残留的默认值会被当成「写死 5 帧」。
            # 超出 spin 量程就只靠上面的说明文字，别让 setValue 悄悄夹到边界值。
            if self.compose_fps_spin.minimum() <= fps <= self.compose_fps_spin.maximum():
                self.compose_fps_spin.setValue(fps)
            self.compose_fps_spin.setToolTip(
                tr("有清单时按清单时长，实际帧率如上"))
            self._compose_images = [
                os.path.join(os.path.dirname(os.path.abspath(manifest_path)), f.file)
                for f in manifest.frames]
        else:
            try:
                images = video_io.list_sequence_images(path)
            except video_io.VideoError as exc:
                self.compose_hint_label.setText(str(exc))
                self.compose_info_label.setText("")
                self.compose_out_edit.setText("")
                self._fill_compose_labels([])
                self._sync_compose_enabled()
                return
            self.compose_info_label.setText(
                f"{tr('没有清单，按下面的帧率合成')}：{len(images)} {tr('张图片')}")
            self.compose_fps_spin.setToolTip(tr("无清单时按每秒几帧合成"))
            self._compose_images = list(images)

        # 来源定了才知道总共几张：范围重置为「0 到最后一张」
        self._set_compose_frame_count(len(self._compose_images or []))
        self._load_compose_labels()
        self.compose_out_edit.setText(self._compose_output_for(path))
        self._sync_compose_enabled()

    def _load_compose_labels(self):
        """扫 sidecar 列出可选类别；没有标注就禁用叠加，标注坏了只提示不拦合成。"""
        from ..engine import video_io

        try:
            labels = video_io.scan_sidecar_labels(self._compose_images)
        except video_io.VideoError as exc:
            self._fill_compose_labels([])
            self.compose_hint_label.setText(f"{tr('标注读不出来')}：{exc}")
            return
        self._fill_compose_labels(labels)

    def _fill_compose_labels(self, labels):
        """重建合成页的类别列表；有标注时主开关默认打开（用户就是要看框）。"""
        self._compose_overlay.fill(labels)

    def _compose_frame_count(self):
        """当前来源一共有多少张图；没有来源时 0。"""
        return len(self._compose_images or [])

    def _on_compose_range_changed(self, _value=None):
        """范围输入变化：夹在有效区间内，并挡住「起始 > 结束」。"""
        total = self._compose_frame_count()
        if total <= 0:
            return
        start = self.compose_start_spin.value()
        end = self.compose_end_spin.value()
        if start > end:
            # 谁被改成越界的就纠谁：把后改的那个拉回与另一个同值，而不是
            # 一律重置成 0/最后——用户改哪一头就保留哪一头。
            if self.sender() is self.compose_start_spin:
                self.compose_start_spin.setValue(end)
            else:
                self.compose_end_spin.setValue(start)
        self._update_compose_range_hint()

    def _update_compose_range_hint(self):
        total = self._compose_frame_count()
        if total <= 0:
            self.compose_range_lbl.setToolTip("")
            return
        start = self.compose_start_spin.value()
        end = self.compose_end_spin.value()
        self.compose_range_lbl.setToolTip(
            f"{tr('共')} {total} {tr('张，将合成第')} {start}–{end} {tr('张')}")

    def compose_range(self):
        """当前选择的范围 `(start, end)`，**包含两端**；无来源时 `(None, None)`。"""
        if self._compose_frame_count() <= 0:
            return None, None
        return self.compose_start_spin.value(), self.compose_end_spin.value()

    def _set_compose_frame_count(self, total):
        """来源变了：把范围重置为「0 到最后一张」（用户要的默认）。"""
        total = int(total or 0)
        last = max(0, total - 1)
        # 先断开信号：这里只是重建量程，不该触发「越界纠正」把值改走
        for spin in (self.compose_start_spin, self.compose_end_spin):
            spin.blockSignals(True)
        self.compose_start_spin.setRange(0, last)
        self.compose_end_spin.setRange(0, last)
        self.compose_start_spin.setValue(0)
        self.compose_end_spin.setValue(last)
        for spin in (self.compose_start_spin, self.compose_end_spin):
            spin.blockSignals(False)
        enabled = total > 0
        self.compose_start_spin.setEnabled(enabled)
        self.compose_end_spin.setEnabled(enabled)
        self._update_compose_range_hint()

    def _on_overlay_toggled(self, _state=None):
        """两页的叠加主开关都会触发：各自刷新各自的可用状态。"""
        self._sync_compose_enabled()
        self._sync_render_enabled()

    def selected_labels(self):
        """合成页勾选的类别名列表（顺序跟列表一致）。"""
        return self._compose_overlay.selected()

    def _compose_output_for(self, images_dir):
        """输出视频路径：图片目录同级，`<目录名>.mp4`；重名时加序号。"""
        parent = os.path.dirname(os.path.abspath(images_dir))
        stem = os.path.basename(os.path.abspath(images_dir)) or "compose"
        candidate = os.path.join(parent, f"{stem}.mp4")
        index = 2
        while os.path.exists(candidate):
            candidate = os.path.join(parent, f"{stem}_{index}.mp4")
            index += 1
        return candidate

    def _sync_compose_enabled(self):
        ready = bool(self._compose_dir) and (
            self._compose_manifest is not None or bool(self.compose_info_label.text()))
        idle = not self._running
        # 有清单时不用填帧率：时长以清单为准，填了也不采用
        self.compose_fps_spin.setEnabled(ready and self._compose_manifest is None
                                        and idle)
        self.compose_pick_btn.setEnabled(idle)
        self.compose_btn.setEnabled(ready and idle)
        # 有清单时 spin 里显示的是清单算出来的实际帧率，只是不能改
        if self._compose_manifest is not None:
            self.compose_fps_spin.setToolTip(
                tr("有清单时按清单时长，实际帧率如上"))
        else:
            self.compose_fps_spin.setToolTip(tr("无清单时按每秒几帧合成"))

        # 叠加：没标注就没得画；关掉主开关时子选项一并禁用，避免误以为还在画
        has_labels = self.compose_label_list.count() > 0
        self.compose_overlay_check.setEnabled(ready and has_labels and idle)
        if not has_labels:
            self.compose_overlay_check.setChecked(False)
        overlay = (ready and has_labels and idle
                   and self.compose_overlay_check.isChecked())
        for widget in self._compose_overlay.widgets():
            widget.setEnabled(overlay)

    def compose_overlay_settings(self):
        """合成页的叠加参数；不叠加（或没有类别可选）时返回 None。

        名字带 compose 是因为**两页各有一组叠加控件**：合成页的类别来自
        sidecar 标注、推理成片页的来自模型类别表。叫 render_settings 会
        让人以为是推理成片页的参数。

        `labels` 用列表而不是集合：空列表 = 一个都不画，与 None（全画）
        区分开，引擎那边靠这个区分。

        这里只看「有没有得画」和用户的勾选，**不看控件的 enabled 状态**：
        任务在跑时控件是禁用的，而 `settings()` 恰好可能在那之后被调用。
        """
        return self._compose_overlay.options()

    # ---------- 推理成片页 ----------

    def set_detector_info(self, class_names=None, model_name=""):
        """告诉弹窗当前加载的模型与类别表（推理成片页据此列类别）。

        弹窗是每次新建的，所以打开时由主窗口灌一次；模型没加载就传空，
        页面会禁用开始按钮并说明原因。

        类别来自模型（不是 sidecar），所以直接填列表、默认全勾上——
        用户要的就是「把模型认得的东西都画出来」。
        """
        self._render_class_names = [str(n) for n in (class_names or [])]
        self._render_model_name = model_name or ""
        group = self._render_overlay
        group.label_list.clear()
        if self._render_class_names:
            group.fill({name: 0 for name in self._render_class_names})
        else:
            group.overlay_check.setChecked(False)
        self._sync_render_enabled()

    def _sync_render_enabled(self):
        """推理成片页的可用状态：要有源视频、有模型、不在跑任务。"""
        has_model = bool(getattr(self, "_render_class_names", None))
        ready = self._info is not None and has_model and not self._running

        self.render_pick_btn.setEnabled(not self._running)
        self.render_btn.setEnabled(ready)
        self.render_path_edit.setText(self._video_path)
        if self._info is not None:
            info = self._info
            self.render_info_label.setText(
                f"{info.width}×{info.height}  {info.fps:.3f} FPS  "
                f"{info.frame_count} 帧  {info.duration:.3f} 秒")
            self.render_out_edit.setText(self._render_output_for(self._video_path))
        else:
            self.render_info_label.setText("")
            self.render_out_edit.setText("")

        if has_model:
            self.render_model_label.setText(
                f"{tr('模型')}: {getattr(self, '_render_model_name', '') or '-'}"
                f"  {tr('类别')}: {len(self._render_class_names)}")
        else:
            self.render_model_label.setText(tr("尚未加载模型"))
        self.render_model_hint_label.setVisible(not has_model)

        # 类别来自模型，不像合成页那样需要扫 sidecar：有模型就有类别可选
        group = self._render_overlay
        group.overlay_check.setEnabled(ready)
        overlay = ready and group.overlay_check.isChecked()
        for widget in group.widgets():
            widget.setEnabled(overlay)

    def _render_output_for(self, video_path):
        """推理成片的输出：视频同级 `<主名>_render.mp4`，重名加序号。"""
        parent = os.path.dirname(os.path.abspath(video_path))
        stem = os.path.splitext(os.path.basename(video_path))[0] or "render"
        candidate = os.path.join(parent, f"{stem}_render.mp4")
        index = 2
        while os.path.exists(candidate):
            candidate = os.path.join(parent, f"{stem}_render_{index}.mp4")
            index += 1
        return candidate

    def _on_start_render(self):
        self.action = ACTION_RENDER
        message = self.validate()
        if message:
            self.render_hint_label.setText(message)
            return
        self.start_requested.emit(ACTION_RENDER, self.settings())

    def _on_start_compose(self):
        self.action = ACTION_COMPOSE
        message = self.validate()
        if message:
            self.compose_hint_label.setText(message)
            return
        self.start_requested.emit(ACTION_COMPOSE, self.settings())

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())
        sync_titlebar(self)
