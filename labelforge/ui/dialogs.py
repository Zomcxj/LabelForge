"""
对话框模块 - 定义各种对话框
"""
import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QLineEdit, QPushButton, QApplication
)
from PySide6.QtCore import Qt

from ..core.utils import extract_label_name
from . import i18n
from .dialog_helpers import center_on_parent
from .theme import ThemeManager
from .dwm import set_titlebar_dark


class LabelSelectionDialog(QDialog):
    """标签选择对话框（标注绘制 / 修改标签共用）。"""

    def __init__(self, parent=None, labels=None, anchor_rect=None,
                 title=None, initial_text="", anchor_pos=None,
                 show_group=False, current_group_id=None):
        super().__init__(parent)
        tr = i18n.t
        self.setWindowTitle(tr(title) if title else tr("选择标签"))
        self.anchor_rect = anchor_rect
        self.anchor_pos = anchor_pos  # global QPoint; used when no anchor_rect
        self.setMinimumWidth(267)
        self.group_input = None

        if labels is None:
            labels = []

        layout = QVBoxLayout()

        top_layout = QHBoxLayout()
        self.ok_btn = QPushButton(tr("确定"))
        self.ok_btn.setObjectName("successBtn")
        top_layout.addWidget(self.ok_btn)

        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText(tr("或输入新标签："))
        if initial_text:
            self.new_label_input.setText(str(initial_text))
            self.new_label_input.selectAll()
        # 输入框里的预填值是**旧名**，不是用户的选择。用户如果去点列表里
        # 另一个标签，那才是他要的结果——但旧代码只认「输入框非空」，于是
        # 点了列表也没用（改名场景里表现为「怎么改都改不动」，因为返回的
        # 还是旧名、判定为无变化）。这里记一下输入框**有没有被用户真正编辑
        # 过**：编辑过就听输入框，没编辑过就听列表选中。
        self._input_edited = False
        self.new_label_input.textEdited.connect(self._on_input_edited)
        top_layout.addWidget(self.new_label_input, 1)
        layout.addLayout(top_layout)

        layout.addWidget(QLabel(tr("现有标签：")))
        self.label_list = QListWidget()
        seen = set()
        current_row = 0
        pure_initial = self._extract_pure_label(initial_text) if initial_text else ""
        for label in labels:
            pure_label = self._extract_pure_label(label)
            if not pure_label or pure_label in seen:
                continue
            seen.add(pure_label)
            self.label_list.addItem(pure_label)
            if pure_initial and pure_label == pure_initial:
                current_row = self.label_list.count() - 1
        layout.addWidget(self.label_list)

        if show_group:
            from PySide6.QtWidgets import QSpinBox
            group_row = QHBoxLayout()
            group_row.addWidget(QLabel(tr("分组")))
            self.group_input = QSpinBox()
            self.group_input.setMinimum(0)
            self.group_input.setMaximum(9999)
            self.group_input.setSpecialValueText(tr("留空"))
            self.group_input.setValue(current_group_id or 0)
            group_row.addWidget(self.group_input)
            layout.addLayout(group_row)

        self.setLayout(layout)

        self.ok_btn.clicked.connect(self.accept)

        self.new_label_input.returnPressed.connect(self.accept)
        self.label_list.itemDoubleClicked.connect(self.accept)

        if self.label_list.count() > 0:
            self.label_list.setCurrentRow(current_row)
        # 点列表 = 用户的明确选择，此后不该再被输入框里的旧名盖过。
        # 连的是 currentRowChanged：点选会走它，setCurrentRow 也会，但
        # `_input_edited` 只在用户真的敲了字之后才为真，所以在放好初始
        # 选中之后再连、且这里只做「清空输入框编辑标记」这一件事。
        self.label_list.itemClicked.connect(self._on_list_item_picked)

    def _on_input_edited(self, _text):
        """用户在输入框里敲过字：从此以输入框为准。"""
        self._input_edited = True

    def _on_list_item_picked(self, _item):
        """用户点了列表里的标签：以列表为准。

        只把「输入框被编辑过」的标记清掉，**不清空输入框内容**——用户可能
        先在输入框里打了半截又改主意去点列表，这时候把他打的字抹掉更烦人。
        取值优先级由 `get_selected_label()` 依据这个标记决定。
        """
        self._input_edited = False

    def showEvent(self, event):
        super().showEvent(event)
        self._position_dialog()
        self._sync_titlebar()

    def _position_dialog(self):
        """优先 anchor_rect（画布框）→ anchor_pos（光标）→ 父窗口居中。"""
        from PySide6.QtCore import QPoint

        if self.anchor_rect is not None and self.parent() is not None:
            anchor_pos = self.anchor_rect.bottomRight().toPoint()
            pos = self.parent().mapToGlobal(anchor_pos)
        elif isinstance(self.anchor_pos, QPoint):
            pos = QPoint(self.anchor_pos.x() + 8, self.anchor_pos.y() + 8)
        else:
            center_on_parent(self)
            return

        screen = QApplication.screenAt(pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            pos.setX(max(available.left(), min(pos.x(), available.right() - self.width())))
            pos.setY(max(available.top(), min(pos.y(), available.bottom() - self.height())))

        self.move(pos)

    def _sync_titlebar(self):
        is_dark = ThemeManager.get_mode().value == "dark"
        set_titlebar_dark(int(self.winId()), is_dark)
    
    @staticmethod
    def _extract_pure_label(label_text):
        """从标签文本中提取纯标签名称"""
        return extract_label_name(label_text)
    
    def get_selected_label(self):
        """获取选中的标签。

        取值优先级按「用户最后动过谁」决定，而不是「输入框非空就用输入框」：

        - 用户在输入框里敲过字（`_input_edited`）→ 用输入框的内容（打新标签）；
        - 否则用户在列表里点了某项 → 用列表项（改名时从已有标签里挑一个，
          这正是「把 drone 改成 Drone」走的路）；
        - 都没有 → 退回输入框的原有内容（可能是预填的旧名），与旧行为一致。
        """
        input_text = self.new_label_input.text().strip()
        selected_items = self.label_list.selectedItems()
        if self._input_edited and input_text:
            return input_text
        if selected_items:
            return selected_items[0].text()
        if input_text:
            return input_text
        return ""

    def get_selected_group_id(self):
        if self.group_input is None:
            return None
        value = self.group_input.value()
        return value if value > 0 else None
    
    @staticmethod
    def select_label(parent, labels, anchor_rect=None, title=None,
                     initial_text="", anchor_pos=None, show_group=False,
                     current_group_id=None):
        """静态方法：显示标签选择对话框并返回选中的标签"""
        dialog = LabelSelectionDialog(
            parent, labels,
            anchor_rect=anchor_rect,
            title=title,
            initial_text=initial_text,
            anchor_pos=anchor_pos,
            show_group=show_group,
            current_group_id=current_group_id,
        )
        if dialog.exec_():
            if show_group:
                return dialog.get_selected_label(), dialog.get_selected_group_id()
            return dialog.get_selected_label()
        return (None, None) if show_group else None


class ProgressDialogFactory:
    """进度条对话框工厂类"""

    @staticmethod
    def create_progress_dialog(parent, title, label_text, maximum):
        from PySide6.QtWidgets import QProgressDialog, QProgressBar
        from . import i18n  # noqa: F401
        t = ThemeManager.get_theme()

        progress_dialog = QProgressDialog(label_text, i18n.t("取消"), 0, maximum, parent)
        progress_dialog.setWindowTitle(title)
        progress_dialog.setMinimumWidth(400)
        progress_dialog.setModal(True)
        progress_dialog.setAutoReset(False)
        progress_dialog.setAutoClose(False)
        progress_dialog.setStyleSheet(f"""
            QProgressDialog {{
                background-color: {t['widget_bg']};
                border: 1px solid {t['border_color']};
                border-radius: 12px;
            }}
        """)

        bar = progress_dialog.findChild(QProgressBar)
        if bar is not None:
            # objectName 留着：只对这条进度条生效，不影响对话框里其它控件
            bar.setObjectName("progressBar")
            qss = ThemeManager.get_progress_style()
            bar.setStyleSheet(qss.replace("QProgressBar", "QProgressBar#progressBar"))

        ProgressDialogFactory._center_dialog(progress_dialog)
        ProgressDialogFactory._sync_titlebar(progress_dialog)
        return progress_dialog
    
    @staticmethod
    def _center_dialog(dialog):
        center_on_parent(dialog)

    @staticmethod
    def _sync_titlebar(dialog):
        from .dwm import set_titlebar_dark, is_available
        if not is_available():
            return
        if not hasattr(dialog, 'winId'):
            return
        is_dark = ThemeManager.get_mode().value == 'dark'
        set_titlebar_dark(int(dialog.winId()), is_dark)


class SaveTipDialog:
    """保存提示对话框"""

    @staticmethod
    def show_save_tip(parent, file_path, success=True):
        from PySide6.QtWidgets import QLabel
        from PySide6.QtCore import QTimer
        from ..core.utils import PathUtils

        t = ThemeManager.get_theme()
        formatted_file_path = PathUtils.to_display_path(file_path)

        if success and os.path.exists(file_path):
            tip_text = f"已保存！路径：{formatted_file_path}"
            label_style = f"""
                QLabel {{
                    color: {t['text_primary']};
                    background: {t['widget_bg']};
                    border: 1px solid {t['success']};
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px 16px;
                    border-radius: 10px;
                }}
            """
            show_time = 1000
        else:
            tip_text = f"未保存成功！\n目标路径：{formatted_file_path}"
            label_style = f"""
                QLabel {{
                    color: {t['widget_bg']};
                    background: {t['danger']};
                    font-size: 14px;
                    font-weight: bold;
                    padding: 12px 16px;
                    border-radius: 10px;
                }}
            """
            show_time = 3000

        save_label = QLabel(tip_text, parent)
        save_label.setStyleSheet(label_style)
        save_label.setWordWrap(True)
        save_label.adjustSize()

        canvas_rect = parent.canvas.rect()
        if parent.current_background:
            img_width = parent.current_background.width()
            img_height = parent.current_background.height()
        else:
            img_width = canvas_rect.width()
            img_height = canvas_rect.height()
        img_x = (canvas_rect.width() - img_width) // 2
        img_y = (canvas_rect.height() - img_height) // 2
        img_center_x = img_x + img_width // 2
        img_center_y = img_y + img_height // 2
        label_x = img_center_x - save_label.width() // 2
        label_y = img_center_y - save_label.height() // 2

        save_label.move(label_x, label_y)
        save_label.setAttribute(Qt.WA_TranslucentBackground, False)
        save_label.raise_()
        save_label.show()

        QTimer.singleShot(show_time, save_label.deleteLater)
