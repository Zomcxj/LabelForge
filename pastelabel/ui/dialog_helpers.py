"""对话框辅助函数：统一标题栏主题和按钮文字。"""
from PyQt5.QtCore import QTimer, QPoint
from PyQt5.QtWidgets import (
    QDialog, QInputDialog, QMessageBox, QColorDialog, QLabel, QPushButton, QApplication,
)

from . import i18n
from .dwm import set_titlebar_dark
from .theme import ThemeManager


def sync_titlebar(widget):
    is_dark = ThemeManager.get_mode().value == "dark"
    set_titlebar_dark(int(widget.winId()), is_dark)


def center_on_parent(dialog, parent=None):
    parent = parent or dialog.parent()
    if not parent:
        return
    parent_geometry = parent.geometry()
    dialog_geometry = dialog.geometry()
    x = parent_geometry.x() + (parent_geometry.width() - dialog_geometry.width()) // 2
    y = parent_geometry.y() + (parent_geometry.height() - dialog_geometry.height()) // 2
    dialog.move(x, y)


def place_within(dialog, rect):
    """把对话框摆到 rect（父窗口坐标）那块，并钳到所在屏幕的可用区。

    rect 是右侧侧栏那块位置。**不要求弹窗装进 rect**：范围弹窗最小宽 440px，
    而侧栏固定 350px，几何上装不下——用户要的是「盖在背景图列表上方、别盖画布」，
    不是「塞进侧栏」。所以这里只把左上角对齐到 rect 内，超出部分由钳制与
    弹窗自身尺寸决定，允许盖住侧栏右边一点。
    """
    if rect is None or rect.isEmpty():
        center_on_parent(dialog)
        return
    # 左上角落在 rect 内即可；水平居中到 rect 会因弹窗更宽而反向溢出到画布上
    x = rect.left() + 8
    y = rect.top() + 8
    screen = QApplication.screenAt(dialog.parent().mapToGlobal(rect.center())) \
        if dialog.parent() is not None else None
    if screen is None:
        screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        top_left = dialog.parent().mapToGlobal(QPoint(0, 0)) \
            if dialog.parent() is not None else QPoint(0, 0)
        gx, gy = top_left.x() + x, top_left.y() + y
        gx = max(available.left(), min(gx, available.right() - dialog.width()))
        gy = max(available.top(), min(gy, available.bottom() - dialog.height()))
        dialog.move(gx - top_left.x(), gy - top_left.y())
        return
    dialog.move(x, y)


class ThemedInputDialog(QInputDialog):
    def __init__(self, parent=None, anchor_pos=None):
        super().__init__(parent)
        self._anchor_pos = anchor_pos  # global QPoint; None = center on parent

    def showEvent(self, event):
        super().showEvent(event)
        if self._anchor_pos is not None:
            self._move_near_anchor(self._anchor_pos)
        else:
            center_on_parent(self)
        sync_titlebar(self)

    def _move_near_anchor(self, pos):
        if not isinstance(pos, QPoint):
            center_on_parent(self)
            return
        screen = QApplication.screenAt(pos)
        if screen is None:
            screen = QApplication.primaryScreen()
        x, y = pos.x() + 8, pos.y() + 8
        if screen is not None:
            available = screen.availableGeometry()
            x = max(available.left(), min(x, available.right() - self.width()))
            y = max(available.top(), min(y, available.bottom() - self.height()))
        self.move(x, y)


class ThemedMessageBox(QMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.buttonClicked.connect(
            lambda _button: QTimer.singleShot(0, self._fix_detail_buttons)
        )

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self)
        sync_titlebar(self)
        QTimer.singleShot(0, self._fix_detail_buttons)
        QTimer.singleShot(100, self._fix_detail_buttons)

    def _fix_detail_buttons(self):
        for btn in self.findChildren(QPushButton):
            raw = btn.text().replace('&', '').strip().lower()
            if raw in ("show details...", "show details", "显示详情"):
                btn.setText(i18n.t("显示详情"))
                if not btn.property("_pastelabel_detail_wired"):
                    btn.setProperty("_pastelabel_detail_wired", True)
                    btn.clicked.connect(
                        lambda: QTimer.singleShot(0, self._fix_detail_buttons)
                    )
            elif raw in ("hide details...", "hide details", "隐藏详情"):
                btn.setText(i18n.t("隐藏详情"))
                if not btn.property("_pastelabel_detail_wired"):
                    btn.setProperty("_pastelabel_detail_wired", True)
                    btn.clicked.connect(
                        lambda: QTimer.singleShot(0, self._fix_detail_buttons)
                    )


class ThemedColorDialog(QColorDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOption(QColorDialog.DontUseNativeDialog, True)

    def showEvent(self, event):
        super().showEvent(event)
        sync_titlebar(self)
        translations = {
            '&Basic colors': '基本颜色：',
            '&Custom colors': '自定义颜色：',
            '&Pick Screen Color': '拾取屏幕颜色',
            '&Add to Custom Colors': '添加到自定义颜色',
            'Hu&e:': '色调：',
            '&Sat:': '饱和度：',
            '&Val:': '亮度：',
            '&Red:': '红：',
            '&Green:': '绿：',
            'Bl&ue:': '蓝：',
            'A&lpha channel:': '透明度：',
            '&HTML:': 'HTML：',
            'OK': '确定',
            'Cancel': '取消',
        }
        for widget in self.findChildren(QLabel) + self.findChildren(QPushButton):
            text = widget.text()
            if text in translations:
                widget.setText(i18n.t(translations[text]))


def get_text(parent, title, label, text="", anchor_pos=None):
    """Show themed text input. If anchor_pos (global QPoint) is set, open near cursor."""
    dialog = ThemedInputDialog(parent, anchor_pos=anchor_pos)
    dialog.setWindowTitle(i18n.t(title))
    dialog.setLabelText(i18n.t(label))
    dialog.setTextValue(text)
    dialog.setOkButtonText(i18n.t("确定"))
    dialog.setCancelButtonText(i18n.t("取消"))
    accepted = dialog.exec_() == QDialog.Accepted
    return dialog.textValue(), accepted


class ThemedMultiLineDialog(QDialog):
    """多行文本输入框。

    QInputDialog 只有单行，一行一个类别名这种场景写不下第二个，所以单独做一个：
    多行编辑区 + 实时行数提示。行数提示让用户能核对「我是不是漏填了」。
    """

    def __init__(self, parent=None, expected_lines=None):
        super().__init__(parent)
        self._expected_lines = expected_lines
        self._build_ui()

    def _build_ui(self):
        from PyQt5.QtWidgets import (
            QDialogButtonBox, QPlainTextEdit, QVBoxLayout,
        )
        self._theme = ThemeManager.get_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)

        self.label = QLabel("")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.edit = QPlainTextEdit()
        self.edit.setObjectName("multiLineInput")
        self.edit.setMinimumSize(420, 200)
        self.edit.textChanged.connect(self._refresh_count)
        layout.addWidget(self.edit, 1)

        self.count_label = QLabel("")
        self.count_label.setObjectName("multiLineCount")
        layout.addWidget(self.count_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        self.ok_btn = buttons.button(QDialogButtonBox.Ok)
        self.cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        self.ok_btn.setText(i18n.t("确定"))
        self.cancel_btn.setText(i18n.t("取消"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._apply_style()
        self._refresh_count()

    def _apply_style(self):
        t = self._theme
        self.setStyleSheet(f"""
            QDialog {{ background-color: {t['widget_bg']}; color: {t['text_primary']}; }}
            QLabel {{ color: {t['text_primary']}; }}
            QLabel#multiLineCount {{ color: {t['text_secondary']}; }}
            QPlainTextEdit {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px; padding: 4px; }}
            QPushButton {{ background-color: {t['panel_bg']}; color: {t['text_primary']};
                border: 1px solid {t['border_color']}; border-radius: 4px; padding: 4px 10px; }}
            QPushButton:hover {{ border-color: {t['accent']}; }}
        """)

    def set_label_text(self, text):
        self.label.setText(text)

    def set_text(self, text):
        self.edit.setPlainText(text or "")

    def lines(self):
        """去掉空行与首尾空格后的有效行。"""
        return [ln.strip() for ln in self.edit.toPlainText().splitlines()
                if ln.strip()]

    def _refresh_count(self):
        n = len(self.lines())
        tip = f"{i18n.t('已输入')} {n} {i18n.t('个类别')}"
        if self._expected_lines:
            tip += f" / {i18n.t('模型需要')} {self._expected_lines}"
            if n != self._expected_lines:
                tip += f"（{i18n.t('数量不一致，请核对')}）"
        self.count_label.setText(tip)

    def showEvent(self, event):
        super().showEvent(event)
        center_on_parent(self, self.parent())
        sync_titlebar(self)


def get_multiline_text(parent, title, label, text="", expected_lines=None):
    """多行输入；返回 (输入内容, 是否确定)。"""
    dialog = ThemedMultiLineDialog(parent, expected_lines=expected_lines)
    dialog.setWindowTitle(i18n.t(title))
    dialog.set_label_text(i18n.t(label))
    dialog.set_text(text)
    accepted = dialog.exec_() == QDialog.Accepted
    return dialog.edit.toPlainText(), accepted


def question(parent, title, text, buttons=QMessageBox.Yes | QMessageBox.No,
             default_button=QMessageBox.No):
    box = ThemedMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(i18n.t(title))
    box.setText(text)
    box.setStandardButtons(buttons)
    box.setDefaultButton(default_button)
    translations = {
        QMessageBox.Yes: "是",
        QMessageBox.No: "否",
        QMessageBox.Cancel: "取消",
        QMessageBox.Ok: "确定",
    }
    for button, label in translations.items():
        btn = box.button(button)
        if btn:
            btn.setText(i18n.t(label))
    return box.exec_()


def warning(parent, title, text):
    box = ThemedMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle(i18n.t(title))
    box.setText(text)
    box.setStandardButtons(QMessageBox.Ok)
    btn = box.button(QMessageBox.Ok)
    if btn:
        btn.setText(i18n.t("确定"))
    return box.exec_()


def information(parent, title, text):
    """信息提示（与 warning 同一套主题与按钮文案）。"""
    box = ThemedMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(i18n.t(title))
    box.setText(text)
    box.setStandardButtons(QMessageBox.Ok)
    btn = box.button(QMessageBox.Ok)
    if btn:
        btn.setText(i18n.t("确定"))
    return box.exec_()
