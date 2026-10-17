"""标注质检结果弹窗：汇总 + 分组表格 + 双击跳图 + 右键忽略 + 筛选。

非模态（show 而非 exec_），打开后可继续操作主窗口。
"""
import math
import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressBar, QMenu,
    QComboBox,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal, QTimer

from .dialog_helpers import center_on_parent
from .i18n import t as tr
from .theme import ThemeManager
from ..engine.quality_lint import KIND_LABELS, TINY_KINDS


def _progress_qss():
    """质检进度条样式：与全项目进度条一致（走 ThemeManager 的统一实现）。

    单独成函数是因为 stylesheet 拼在 __init__ 的大 f-string 里没法直接测。
    """
    return ThemeManager.get_progress_style()


class QualityLintDialog(QDialog):
    issue_activated = Signal(object)
    ignore_requested = Signal(object, str)   # (issue, scope: 'item' | 'kind')
    ignore_many_requested = Signal(object, str)  # (issues list, scope)
    delete_requested = Signal(object, str)  # (选中的问题列表, 目标类别)
    mark_keep_requested = Signal(object)  # 小目标问题列表：写 flags.tiny_keep

    def __init__(self, parent=None, total=0):
        super().__init__(parent)
        self._issues = []
        self._all_issues = []
        self._kind_filter = None
        self._scanned_images = 0
        self.setWindowTitle(tr("标注质检"))
        self.setMinimumSize(720, 580)
        t = ThemeManager.get_theme()
        self.setStyleSheet(f"""
            QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QTableWidget {{ background-color: {t['widget_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; gridline-color: {t['border_color']}; }}
            QTableWidget::item {{ padding: 4px; }}
            QTableWidget::item:selected {{ background-color: {t['accent_light']}; color: {t['accent']}; }}
            QHeaderView::section {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; padding: 4px; font-weight: bold; }}
            QTableWidget QTableCornerButton::section {{ background-color: {t['panel_bg']};
                border: 1px solid {t['border_color']}; }}
            {_progress_qss()}
            QComboBox {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; padding: 2px 6px; }}
        """)
        from .dwm import set_titlebar_dark
        is_dark = ThemeManager.get_mode().value == "dark"
        QTimer.singleShot(30, lambda: set_titlebar_dark(int(self.winId()), is_dark))
        layout = QVBoxLayout(self)
        top_row = QHBoxLayout()
        self.status_label = QLabel(tr("正在扫描"))
        top_row.addWidget(self.status_label)
        top_row.addStretch()
        top_row.addWidget(QLabel(tr("筛选") + ":"))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(133)
        self.filter_combo.addItem(tr("全部"), None)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        top_row.addWidget(self.filter_combo)
        layout.addLayout(top_row)
        self.progress = QProgressBar()
        self.progress.setMaximum(max(1, total))
        layout.addWidget(self.progress)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.hint_label = QLabel(
            f"{tr('双击跳转到问题图')} · {tr('右键忽略')} · {tr('Ctrl/Shift 多选')}")
        self.hint_label.setStyleSheet("font-size: 11px; opacity: 0.7;")
        layout.addWidget(self.hint_label)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            [tr("框类型"), tr("类别"), tr("背景图:"), tr("尺寸/提示")])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        # 提示列是审核时最常读的内容，给它固定宽度免得被前三列挤没
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.doubleClicked.connect(self._on_double_click)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.table)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())

    def set_progress(self, done, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(f"{tr('扫描中')} {done}/{total}")

    def set_result(self, result):
        if result.get('error'):
            self._all_issues = []
            self._scanned_images = 0
            self._rebuild_filter_combo()
            self._apply_filter()
            self.progress.setVisible(False)
            self.status_label.setText(tr("扫描失败"))
            self.summary_label.setText(tr("扫描失败"))
            return
        summary = result.get('summary') or {}
        self._all_issues = list(result.get('issues') or [])
        self._scanned_images = int(summary.get('scanned_images') or 0)
        self._rebuild_filter_combo()
        self._apply_filter()
        self.status_label.setText("")
        self.progress.setValue(self.progress.maximum())
        # 扫描结束进度条就没用了，钉在 100% 只占地方；重扫会新建弹窗，无需恢复
        self.progress.setVisible(False)
        self._update_summary(summary)

    def _size_hint_text(self, issue):
        """第 4 列：小目标两档给"宽/高 + 占比 + 建议"，其它 kind 沿用 detail。

        引擎给的 min_side/both_below 说明该拉宽、拉高还是整个放大；
        both_below 优先 —— 两侧都低于命中档的线时单侧说法是错的。
        老数据没有这两个键时退回纯占比，不能抛。
        """
        if not isinstance(issue, dict):
            return ""
        kind = issue.get('kind')
        if kind not in TINY_KINDS:
            return str(issue.get('detail') or '')
        ratio = issue.get('ratio')
        pct = None
        if isinstance(ratio, (int, float)) and math.isfinite(ratio):
            pct = f"{ratio * 100.0:.2f}%"
        if issue.get('both_below'):
            side = tr("宽高均不足")
        elif issue.get('min_side') == 'width':
            side = tr("宽不足")
        elif issue.get('min_side') == 'height':
            side = tr("高不足")
        else:
            side = ""
        size = f"{side} {pct}" if side and pct else (side or pct or "")
        advice = (tr("常规配置检不到，建议提高分辨率或滑窗切片")
                  if kind == 'tiny_unreachable'
                  else tr("需 P2 检测头或裁剪放大"))
        return f"{size} · {advice}" if size else advice

    def _update_summary(self, summary):
        parts = []
        if self._scanned_images:
            parts.append(tr("扫描图片") + f": {self._scanned_images}")
        for kind, label in KIND_LABELS.items():
            count = summary.get(kind, 0)
            if count:
                parts.append(f"{tr(label)}: {count}")
        self.summary_label.setText(
            " | ".join(parts) if parts else tr("未发现问题"))

    def _rebuild_filter_combo(self):
        counts = {}
        for issue in self._all_issues:
            counts[issue.get('kind')] = counts.get(issue.get('kind'), 0) + 1
        previous = self._kind_filter
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem(f"{tr('全部')} ({len(self._all_issues)})", None)
        for kind, label in KIND_LABELS.items():
            if counts.get(kind):
                self.filter_combo.addItem(f"{tr(label)} ({counts[kind]})", kind)
        # 保留用户当前筛选；该类型已无问题时回到「全部」
        target = 0
        if previous is not None:
            for i in range(self.filter_combo.count()):
                if self.filter_combo.itemData(i) == previous:
                    target = i
                    break
        self._kind_filter = self.filter_combo.itemData(target)
        self.filter_combo.setCurrentIndex(target)
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self, _index):
        self._kind_filter = self.filter_combo.currentData()
        self._apply_filter()

    def _kind_order(self, kind):
        order = list(KIND_LABELS.keys())
        return order.index(kind) if kind in order else len(order)

    def _image_order(self, issue):
        """图片排序键：有 image_index 时按索引，否则按路径。"""
        index = issue.get('image_index')
        if isinstance(index, int):
            return (0, index, '')
        return (1, 0, str(issue.get('image_path') or '').casefold())

    def _sort_issues(self, issues):
        """问题排序：先按背景图分组，同图内按类型（KIND_LABELS 顺序）再按类别名。"""
        return sorted(
            issues or [],
            key=lambda i: (
                self._image_order(i),
                self._kind_order(i.get('kind')),
                str(i.get('label_a') or i.get('label') or '').casefold(),
                str(i.get('label_b') or '').casefold(),
            ),
        )

    def _apply_filter(self):
        if self._kind_filter is None:
            self._issues = self._sort_issues(self._all_issues)
        else:
            self._issues = self._sort_issues([
                i for i in self._all_issues if i.get('kind') == self._kind_filter])
        self._populate_table()

    def visible_issues(self):
        """当前表格实际显示的问题（已筛选、已排序）。"""
        return list(self._issues or [])

    def focus_image(self, image_index):
        """滚动并选中指定背景图的第一条问题行（跳图后同步表格）。"""
        for row, issue in enumerate(self._issues or []):
            if issue.get('image_index') == image_index:
                item = self.table.item(row, 0)
                if item is not None:
                    self.table.scrollToItem(item, QAbstractItemView.PositionAtTop)
                    self.table.selectRow(row)
                return True
        return False

    def _issue_label_text(self, issue):
        """类别列文本：异类重叠显示两个类名，其它显示自身类别。"""
        if issue.get('kind') == 'cross_label_overlap':
            label_a = str(issue.get('label_a') or '').strip()
            label_b = str(issue.get('label_b') or '').strip()
            if label_a and label_b:
                return f"{label_a} ↔ {label_b}"
        return issue.get('label', '') or ''

    def _populate_table(self):
        self.table.setRowCount(len(self._issues))
        # 一律不着色（用户裁决）：审核靠第 4 列文字，红黄底反而把审核结论
        # 和普通问题（越界/重复…）混在一起。
        for row, issue in enumerate(self._issues):
            items = [
                tr(KIND_LABELS.get(issue.get('kind'), issue.get('kind', ''))),
                self._issue_label_text(issue),
                os.path.basename(issue.get('image_path', '') or ''),
                self._size_hint_text(issue),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setData(Qt.UserRole, issue)
                self.table.setItem(row, col, item)
            # 提示列文字最长且最常被读，列宽受限时悬停可看全文
            hint_item = self.table.item(row, 3)
            if hint_item is not None:
                hint_item.setToolTip(str(items[3]))

    def _selected_issues(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        issues = []
        for row in rows:
            item = self.table.item(row, 0)
            if item is not None:
                issue = item.data(Qt.UserRole)
                if issue:
                    issues.append(issue)
        return issues

    def _remove_issues(self, issues):
        """忽略后从内存与表格移除（当前筛选下重建）。

        注意：表格 UserRole 取回的 issue 可能是副本，不能按 id 匹配，
        必须按内容比较。
        """
        targets = list(issues or [])
        self._all_issues = [
            i for i in self._all_issues
            if not any(i == t for t in targets)
        ]
        self._rebuild_filter_combo()
        self._apply_filter()
        remaining = {}
        for i in self._all_issues:
            remaining[i.get('kind')] = remaining.get(i.get('kind'), 0) + 1
        self._update_summary(remaining)

    def refresh_issues_for_images(self, image_indexes, refreshed_issues):
        """删除框后刷新指定图的问题：先清旧问题再插入最新结果。"""
        changed = set(image_indexes or [])
        if not changed:
            return
        self._all_issues = [
            i for i in self._all_issues
            if i.get('image_index') not in changed
        ]
        self._all_issues.extend(refreshed_issues or [])
        self._rebuild_filter_combo()
        self._apply_filter()
        remaining = {}
        for i in self._all_issues:
            remaining[i.get('kind')] = remaining.get(i.get('kind'), 0) + 1
        self._update_summary(remaining)

    def _on_double_click(self, index):
        item = self.table.item(index.row(), 0)
        if item is None:
            return
        issue = item.data(Qt.UserRole)
        if issue:
            self.issue_activated.emit(issue)

    def _cross_labels_in(self, issues):
        """选中问题里出现的类别（保持出现顺序，去重）。"""
        labels = []
        for issue in issues:
            if issue.get('kind') != 'cross_label_overlap':
                continue
            for key in ('label_a', 'label_b'):
                value = str(issue.get(key) or '').strip()
                if value and value not in labels:
                    labels.append(value)
        return labels

    def _on_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        issue = item.data(Qt.UserRole)
        if not issue:
            return
        selected = self._selected_issues()
        if issue not in selected:
            selected = [issue]
        menu = QMenu(self)
        kind = issue.get('kind')
        kind_label = tr(KIND_LABELS.get(kind, kind or ''))
        if kind == 'cross_label_overlap':
            cross_selected = [
                i for i in selected if i.get('kind') == 'cross_label_overlap']
            labels = self._cross_labels_in(cross_selected)
            count = len(cross_selected)
            for value in labels:
                text = tr("删除 {n} 项中的 '{label}' 框").replace(
                    "{n}", str(count)).replace("{label}", value)
                action = QAction(text, self)
                action.triggered.connect(
                    lambda _, target=value, items=cross_selected:
                    self.delete_requested.emit(items, target))
                menu.addAction(action)
            if labels:
                menu.addSeparator()
        if kind in TINY_KINDS:
            # 小目标两类的 detail 里带实时比例与像素（"0.94% W:6 H:6"），
            # 框改 1px 就失配，按 detail 忽略会静默失效。豁免只能走
            # flags.tiny_keep（跟着框走），整类忽略则与 detail 无关，两者都保留。
            tiny_selected = [i for i in selected if i.get('kind') in TINY_KINDS]
            if len(tiny_selected) > 1:
                text = tr("标记保留选中的 {n} 项").replace(
                    "{n}", str(len(tiny_selected)))
            else:
                text = tr("标记保留")
            keep = QAction(text, self)
            keep.triggered.connect(
                lambda _, items=tiny_selected:
                self.mark_keep_requested.emit(items))
            menu.addAction(keep)
            ignore_kind = QAction(
                tr("忽略此类问题").replace("{kind}", kind_label), self)
            ignore_kind.triggered.connect(
                lambda _, i=issue: self.ignore_requested.emit(i, 'kind'))
            menu.addAction(ignore_kind)
            menu.exec_(self.table.viewport().mapToGlobal(pos))
            return
        if len(selected) > 1:
            # 多选忽略是按 detail 记规则，小目标两类的 detail 会变，
            # 混在里面对它们无效，故先剔除，避免用户以为已豁免。
            detail_selected = [
                i for i in selected if i.get('kind') not in TINY_KINDS]
            if len(detail_selected) > 1:
                ignore_many = QAction(
                    tr("忽略选中的 {n} 项").replace(
                        "{n}", str(len(detail_selected))), self)
                ignore_many.triggered.connect(
                    lambda _, items=detail_selected:
                    self.ignore_many_requested.emit(items, 'item'))
                menu.addAction(ignore_many)
        ignore_kind = QAction(
            tr("忽略此类问题").replace("{kind}", kind_label), self)
        ignore_kind.triggered.connect(
            lambda _, i=issue: self.ignore_requested.emit(i, 'kind'))
        menu.addAction(ignore_kind)
        if issue.get('detail'):
            ignore_item = QAction(
                tr("忽略此项").replace("{detail}", str(issue['detail'])), self)
            ignore_item.triggered.connect(
                lambda _, i=issue: self.ignore_requested.emit(i, 'item'))
            menu.addAction(ignore_item)
        menu.exec_(self.table.viewport().mapToGlobal(pos))
