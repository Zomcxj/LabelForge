"""对话框辅助函数：统一标题栏主题和按钮文字。"""
from PySide6.QtCore import QTimer, QPoint
from PySide6.QtWidgets import (
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


def place_within(dialog, rect, align="center"):
    """把对话框浮到 rect（父窗口坐标）那块，并钳到所在屏幕的可用区。

    rect 是**右侧侧栏那块**（面板要盖在它上面，而不是画布上）。**不要求弹窗
    装进 rect**：范围弹窗比侧栏宽，几何上装不下——用户要的是「浮在侧栏上层、
    别盖画布」。

    竖直方向：
    - `align="center"`（默认）：按 rect 的**中心**居中。范围弹窗原来用这个。
    - `align="top-left"`：弹窗左上角对齐 rect 左上角（范围弹窗现在用这个，
      用户要求「左上角始终对齐标签管理区域的左上角」）。

    水平方向：以 rect 的中心线对齐弹窗中心，但**优先保住左边不越过 rect
    左边界**——用户报过「面板跑到屏幕左边挡住画布」，根因是这里曾经用
    `rect.left() + 8` 当左上角、而 rect.x 又被上游写死成 0。现在先按居中算，
    再钳到「不小于 rect.left()」，让弹窗向右溢出（压住侧栏右侧的空白）而不是
    向左盖画布（画布在主窗左侧）。
    """
    if rect is None or rect.isEmpty():
        center_on_parent(dialog)
        return
    # 统一到**全局**坐标再算：rect 是父窗口坐标（局部），而屏幕可用区是全局的，
    # 混着比会把弹窗放到别的地方（实测主窗不在屏幕原点时，面板会整体左移一个
    # 主窗 x 的量，落到画布上）。算完直接用全局坐标 `move()`。
    parent = dialog.parent()
    if parent is not None:
        origin = parent.mapToGlobal(QPoint(0, 0))
    else:
        origin = QPoint(0, 0)
    left = origin.x() + rect.left()
    center_x = origin.x() + rect.center().x()
    center_y = origin.y() + rect.center().y()

    # 用**整窗**尺寸（含窗口边框）而不是客户区宽度：`frameGeometry` 才是屏幕上
    # 真正占的地方，拿客户区宽度去钳，边框会多探出几像素。
    frame = dialog.frameGeometry()
    span_x = max(dialog.width(), frame.width())
    span_y = max(dialog.height(), frame.height())
    # 边框相对客户区左上角的偏移；`move()` 定的是客户区左上角，钳制要按整窗算。
    inset_x = frame.x() - dialog.x()
    inset_y = frame.y() - dialog.y()

    # 水平：以 rect 中心线对齐弹窗中心，再保证左边不越过 rect 左边界，
    # 让溢出发生在右侧（侧栏右边是空白）而不是左侧（左侧是画布）。
    gx = center_x - span_x // 2
    if gx < left:
        gx = left
    if align == "top-left":
        gy = origin.y() + rect.top()
    else:
        gy = center_y - span_y // 2

    screen = QApplication.screenAt(QPoint(center_x, center_y)) \
        or QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        # 按整窗的左右边界钳，再换算回客户区左上角；直接减 span 会因为
        # 边框偏移而多探出 1~3px。
        gx = max(available.left() - inset_x,
                 min(gx, available.right() + 1 - span_x - inset_x))
        gy = max(available.top() - inset_y,
                 min(gy, available.bottom() + 1 - span_y - inset_y))
    # 直接用**全局**坐标 move()：`ScopePanel` 是顶层窗口，`QWidget.move()` 对
    # 顶层窗口按全局坐标解释（实测 move(1056,323) 后 pos() 就是 1056,323）。
    # 早先这里减掉了 `origin`，等于把面板整体左移一个主窗 x 的量——主窗不在
    # 屏幕原点时，面板就跑到画布上去了（用户报的「跑到屏幕左边、遮挡画布」）。
    dialog.move(gx, gy)


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
                if not btn.property("_labelforge_detail_wired"):
                    btn.setProperty("_labelforge_detail_wired", True)
                    btn.clicked.connect(
                        lambda: QTimer.singleShot(0, self._fix_detail_buttons)
                    )
            elif raw in ("hide details...", "hide details", "隐藏详情"):
                btn.setText(i18n.t("隐藏详情"))
                if not btn.property("_labelforge_detail_wired"):
                    btn.setProperty("_labelforge_detail_wired", True)
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
        from PySide6.QtWidgets import (
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
