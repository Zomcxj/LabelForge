"""
pytest conftest - 在收集测试前 mock PySide6
使用真实类继承，避免 metaclass 冲突
"""
import sys
import types

# 隔离测试：默认把配置指向临时文件，避免污染用户真实的 ~/.labelforge.json
# （部分用例会真实构造 ImageEditor 并触发 save_all）。
import os as _os
import tempfile as _tempfile
_os.environ.setdefault(
    'LABELFORGE_CONFIG_PATH',
    _os.path.join(_tempfile.mkdtemp(prefix='labelforge_test_'), 'config.json'),
)
# 真 Qt 子进程里要 print 中文再回读到父进程断言，编码必须两端一致。
# 不设这个变量时子进程按 Windows 默认代码页（GBK）编码 stdout，中文变成
# `\ufffd` 乱码、断言里那句「期望的中文」永远匹配不上——症状是「真 Qt 用例
# 里凡是比中文字的都在报错」，但看起来像功能坏了。设一次即可，子进程通过
# `os.environ | {...}` 一并继承。
_os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
del _os, _tempfile

# 子进程默认使用当前测试解释器；需要指定其他解释器时显式设置环境变量
import os as _os
_sub = _os.environ.get('LABELFORGE_SUBPROC_PYTHON')
if _sub and _os.path.exists(_sub):
    sys.executable = _sub
del _os, _sub


def _make_mock_module(name):
    """创建一个支持 from X import Y 的 mock 模块"""
    mod = types.ModuleType(name)
    mod.__path__ = []  # 标记为包
    mod.__file__ = f"<mock {name}>"
    # 预注册常用 Qt 名称，确保 from X import Y 可用
    for attr in [
        'QPoint', 'Qt', 'QUrl', 'QSize', 'QTimer', 'QRectF', 'QRect',
        'QPropertyAnimation', 'QEasingCurve',
        'Signal', 'QObject', 'QPixmap', 'QPainter', 'QColor', 'QIcon',
        'QFont', 'QFontDatabase', 'QEvent', 'QKeySequence', 'QImage',
        'QStandardItem', 'QStandardItemModel', 'QSortFilterProxyModel',
        'QValidator', 'QRegExp', 'QDesktopWidget', 'QApplication',
        'QMainWindow', 'QDialog', 'QWidget', 'QVBoxLayout', 'QHBoxLayout',
        'QLabel', 'QPushButton', 'QListWidget', 'QListWidgetItem',
        'QLineEdit', 'QCheckBox', 'QSpinBox', 'QGroupBox', 'QFrame',
        'QSplitter', 'QScrollArea', 'QFileDialog', 'QMessageBox',
        'QInputDialog', 'QMenu', 'QAction', 'QWidgetAction', 'QShortcut', 'QKeySequence',
        'QProgressDialog', 'QProgressBar', 'QSvgRenderer', 'QCursor',
        'QStackedWidget', 'QComboBox', 'QDoubleSpinBox', 'QTextBrowser',
        'QTabWidget', 'QToolButton', 'QSlider', 'QRadioButton', 'QColorDialog', 'QDialogButtonBox',
    ]:
        setattr(mod, attr, type(f'Mock{attr}', (), {
            '__init__': lambda self, *a, **kw: None,
        }))
    return mod


class _MockQWidget:
    """最小化的 QWidget mock，支持继承"""
    def __init__(self, *a, **kw):
        pass
    def setMinimumSize(self, *a): pass
    def setMinimumHeight(self, *a): pass
    def setMouseTracking(self, *a): pass
    def setFocusPolicy(self, *a): pass
    def setFocus(self): pass
    def update(self): pass
    def width(self): return 800
    def height(self): return 600
    def rect(self):
        r = type('R', (), {'left': lambda: 0, 'top': lambda: 0,
                           'width': lambda: 800, 'height': lambda: 600,
                           'x': lambda: 0, 'y': lambda: 0})()
        return r
    def setCursor(self, *a): pass
    def mapToGlobal(self, p): return p
    def installEventFilter(self, *a): pass
    def children(self): return []
    def parent(self): return None
    def repaint(self): pass
    def updateGeometry(self): pass
    def show(self): pass
    def raise_(self): pass
    def setAttribute(self, *a): pass
    def adjustSize(self): pass
    def move(self, *a): pass
    def setWordWrap(self, *a): pass
    def style(self): return type('Style', (), {'unpolish': lambda self, *a: None, 'polish': lambda self, *a: None})()
    def layout(self): return None
    def setVisible(self, *a): pass
    def setHidden(self, *a): pass
    def isVisible(self): return True


class _MockQMainWindow(_MockQWidget):
    """最小化的 QMainWindow mock"""
    def statusBar(self):
        sb = type('StatusBar', (), {'addWidget': lambda self, w: None})()
        return sb
    def setCentralWidget(self, *a): pass
    def setWindowTitle(self, *a): pass
    def resize(self, *a): pass
    def setWindowIcon(self, *a): pass
    def addDockWidget(self, *a): pass
    def splitDockWidget(self, *a): pass
    def resizeDocks(self, *a): pass
    def centralWidget(self): return None
    def setDockNestingEnabled(self, *a): pass


class _MockQDialog(_MockQWidget):
    Accepted = 1
    Rejected = 0
    def setWindowTitle(self, *a): pass
    def setMinimumWidth(self, *a): pass
    def setLayout(self, *a): pass
    def exec_(self): return False
    def accept(self): pass
    def reject(self): pass


class _MockQListWidget(_MockQWidget):
    IconMode = 0
    ListMode = 1
    LeftToRight = 0
    TopToBottom = 1
    Adjust = 1
    ScrollPerPixel = 0
    ScrollPerItem = 1
    CustomContextMenu = 1

    def addItem(self, *a): pass
    def clear(self): pass
    def count(self): return 0
    def item(self, *a): return None
    def selectedItems(self): return []
    def setCurrentRow(self, *a): pass
    def setContextMenuPolicy(self, *a): pass
    def setViewMode(self, *a): pass
    def setIconSize(self, *a): pass
    def setGridSize(self, *a): pass
    def setSpacing(self, *a): pass
    def setWrapping(self, *a): pass
    def setFlow(self, *a): pass
    def setResizeMode(self, *a): pass
    def setVerticalScrollMode(self, *a): pass
    def setHorizontalScrollMode(self, *a): pass
    def setUniformItemSizes(self, *a): pass
    def setUpdatesEnabled(self, *a): pass
    def scrollToTop(self): pass
    def mapToGlobal(self, p): return p


class _MockQSplitter(_MockQWidget):
    def setStretchFactor(self, *a): pass
    def setSizes(self, *a): pass
    def addWidget(self, *a): pass


class _MockQScrollArea(_MockQWidget):
    def setWidget(self, *a): pass
    def setWidgetResizable(self, *a): pass


class _MockQBoxLayout:
    def __init__(self, *a): pass
    def addWidget(self, *a): pass
    def addLayout(self, *a): pass
    def addSpacing(self, *a): pass
    def addStretch(self): pass
    def setStretch(self, *a): pass
    def setContentsMargins(self, *a): pass
    def setSpacing(self, *a): pass


class _MockQLabel(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
    def setText(self, *a): pass
    def text(self): return ''
    def hide(self): pass
    def show(self): pass
    def setStyleSheet(self, *a): pass
    def setMaximumWidth(self, *a): pass
    def setToolTip(self, *a): pass


class _MockQPushButton(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
    def setIcon(self, *a): pass
    def setMaximumWidth(self, *a): pass
    def setToolTip(self, *a): pass
    def setStyleSheet(self, *a): pass
    def setText(self, *a): pass
    def setMinimumWidth(self, *a): pass
    clicked = type('Signal', (), {'connect': lambda self, f: None})()


class _MockQCheckBox(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
    def isChecked(self): return False
    def setChecked(self, *a): pass
    def setStyleSheet(self, *a): pass
    stateChanged = type('Signal', (), {'connect': lambda self, f: None})()


class _MockQRadioButton(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
        self._checked = False
    def isChecked(self): return self._checked
    def setChecked(self, checked):
        self._checked = bool(checked)
    def setStyleSheet(self, *a): pass
    toggled = type('Signal', (), {'connect': lambda self, f: None})()


class _MockQSpinBox(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
    def minimum(self): return 0
    def maximum(self): return 100
    def value(self): return 0
    def setValue(self, *a): pass
    def setMinimum(self, *a): pass
    def setMaximum(self, *a): pass
    def setMinimumWidth(self, *a): pass
    def setStyleSheet(self, *a): pass
    valueChanged = type('Signal', (), {'connect': lambda self, f: None})()
    editingFinished = type('Signal', (), {'connect': lambda self, f: None})()


class _MockQLineEdit(_MockQWidget):
    def __init__(self, *a, **kw):
        super().__init__()
    def text(self): return ''
    def setText(self, *a): pass
    def setStyleSheet(self, *a): pass
    def setMaximumWidth(self, *a): pass
    def focusInEvent(self, *a): pass
    def focusOutEvent(self, *a): pass
    returnPressed = type('Signal', (), {'connect': lambda self, f: None})()


class _MockQListWidgetItem:
    def __init__(self, *a, **kw):
        pass
    def setData(self, *a): pass
    def data(self, *a): return None
    def setIcon(self, *a): pass
    def setSizeHint(self, *a): pass
    def text(self): return ''


class _MockQProgressDialog(_MockQWidget):
    last_args = None
    def __init__(self, *a, **kw):
        super().__init__()
        self.args = a
        _MockQProgressDialog.last_args = a
    def setWindowTitle(self, *a): pass
    def setMinimumWidth(self, *a): pass
    def setModal(self, *a): pass
    def setAutoReset(self, *a): pass
    def setAutoClose(self, *a): pass
    def setStyleSheet(self, *a): pass
    def show(self): pass
    def setValue(self, *a): pass
    def setLabelText(self, *a): pass
    def wasCanceled(self): return False
    def close(self): pass
    def findChild(self, *a): return None
    def geometry(self):
        return type('G', (), {'width': lambda self: 400, 'height': lambda self: 200})()


class _MockQInputDialog:
    def __init__(self, *a, **kw): pass
    def setWindowTitle(self, *a): pass
    def setLabelText(self, *a): pass
    def setTextValue(self, *a): pass
    def setOkButtonText(self, *a): pass
    def setCancelButtonText(self, *a): pass
    def textValue(self): return ''
    def exec_(self): return _MockQDialog.Rejected
    def showEvent(self, *a): pass
    @staticmethod
    def getText(*a, **kw): return ('', False)


class _MockQMessageBox:
    Critical = 2
    Warning = 2
    Question = 4
    Ok = 0x00000400
    Yes = 0x00000400
    No = 0x00000200
    Cancel = 0x00400000
    def __init__(self, *a, **kw): pass
    def setIcon(self, *a): pass
    def setWindowTitle(self, *a): pass
    def setText(self, *a): pass
    def setStandardButtons(self, *a): pass
    def setDefaultButton(self, *a): pass
    def button(self, *a): return type('Button', (), {'setText': lambda self, text: None})()
    def exec_(self): return _MockQMessageBox.No
    def showEvent(self, *a): pass
    @staticmethod
    def warning(*a, **kw): pass
    @staticmethod
    def critical(*a, **kw): pass
    @staticmethod
    def information(*a, **kw): pass
    @staticmethod
    def question(*a, **kw): return _MockQMessageBox.No


class _MockQFileDialog:
    @staticmethod
    def getOpenFileNames(*a, **kw): return ([], '')
    @staticmethod
    def getExistingDirectory(*a, **kw): return ''
    @staticmethod
    def getOpenFileName(*a, **kw): return ('', '')


class _MockQApplication:
    @staticmethod
    def primaryScreen():
        return type('S', (), {'availableGeometry': lambda self: type('R', (), {
            'width': lambda self: 1920, 'height': lambda self: 1080})()})()
    @staticmethod
    def desktop():
        return type('D', (), {'screenGeometry': lambda: type('R', (), {
            'width': lambda: 1920, 'height': lambda: 1080})()})()
    @staticmethod
    def processEvents(): pass
    @staticmethod
    def instance(): return None
    def __init__(self, *a): pass
    def exec_(self): return 0


# ========== 构建 mock 模块 ==========

qtcore = types.ModuleType('PySide6.QtCore')
qtcore.Qt = type('Qt', (), {
    'Key_A': 0x41, 'Key_D': 0x44, 'Key_R': 0x52, 'Key_T': 0x54,
    'Key_W': 0x57, 'Key_Q': 0x51, 'Key_E': 0x45, 'Key_G': 0x47,
    'Key_Delete': 0x01000007,
    'Horizontal': 1, 'CustomContextMenu': 1, 'StrongFocus': 2,
    'LeftButton': 1, 'RightButton': 2,
    'ControlModifier': 0x04000000, 'KeepAspectRatio': 1,
    'SmoothTransformation': 1, 'transparent': 0,
    'PointingHandCursor': 18,
    'OpenHandCursor': 17,
    'ClosedHandCursor': 18,
    'ArrowCursor': 0,
    'WA_StyledBackground': 9,
    'NoFocus': 0,
    'AlignCenter': 0x0084,
    'AlignRight': 0x0002,
    'AlignVCenter': 0x0080,
    'AlignLeft': 0x0001,
    'AlignTop': 0x0020,
    'AlignBottom': 0x0040,
    'AlignHCenter': 0x0004,
    'ElideRight': 2,
    'DotLine': 1,
    'NoPen': 0,
    'SolidLine': 1,
    'Key_Space': 0x20,
    'Vertical': 2,
    'LeftDockWidgetArea': 1, 'RightDockWidgetArea': 2,
    'TopDockWidgetArea': 4, 'BottomDockWidgetArea': 8,
    # item data 角色，取真实 Qt 的值；质检弹窗靠 UserRole 挂 issue
    'DisplayRole': 0, 'EditRole': 2, 'ToolTipRole': 3, 'UserRole': 0x0100,
})()
qtcore.QPoint = type('QPoint', (), {'__init__': lambda self, x=0, y=0: None})
qtcore.QPointF = type('QPointF', (), {'__init__': lambda self, x=0, y=0: None})
qtcore.QRectF = type('QRectF', (), {
    '__init__': lambda self, *a: None,
    'contains': lambda self, p: True,
    'x': lambda self: 0,
    'y': lambda self: 0,
    'width': lambda self: 10,
    'height': lambda self: 10,
    'left': lambda self: 0,
    'top': lambda self: 0,
})
qtcore.QRect = type('QRect', (), {'__init__': lambda self, *a: None})
qtcore.QSize = type('QSize', (), {'__init__': lambda self, w=0, h=0: None})
qtcore.QSizeF = type('QSizeF', (), {'__init__': lambda self, w=0, h=0: None})
qtcore.QThread = type('QThread', (), {'__init__': lambda self, *a, **kw: None})
qtcore.QTimer = type('QTimer', (), {'singleShot': staticmethod(lambda *a: None)})
qtcore.QPropertyAnimation = type('QPropertyAnimation', (), {'__init__': lambda self, *a: None})
qtcore.QEasingCurve = type('QEasingCurve', (), {'OutCubic': 0})
qtcore.QEvent = type('QEvent', (), {'KeyPress': 6})
qtcore.QUrl = type('QUrl', (), {'fromLocalFile': staticmethod(lambda f: f)})
qtcore.QMimeData = type('QMimeData', (), {
    '__init__': lambda self, *a, **kw: None,
    'hasUrls': lambda self: False,
    'setUrls': lambda self, *a: None,
})
qtcore.qInstallMessageHandler = lambda *a: None
qtcore.Signal = lambda *a, **kw: type('Signal', (), {'connect': lambda self, f: None})()

class _MockQObject:
    """QObject mock - 支持继承和 Signal 作为类属性"""
    def __init__(self, *a, **kw):
        pass
    def connect(self, *a): pass
    def emit(self, *a): pass

qtcore.QObject = _MockQObject

qtgui = types.ModuleType('PySide6.QtGui')
qtgui.QPixmap = type('QPixmap', (), {
    '__init__': lambda self, *a: None,
    'isNull': lambda self: False,
    'width': lambda self: 100,
    'height': lambda self: 100,
    'scaled': lambda self, *a: self,
    'save': lambda self, *a: True,
    'fill': lambda self, *a: None,
})
qtgui.QIcon = type('QIcon', (), {'__init__': lambda self, *a: None})
qtgui.QColor = type('QColor', (), {
    '__init__': lambda self, *a: None,
    'setAlpha': lambda self, *a: None,
    'isValid': lambda self: True,
    'name': lambda self: '#00FF80',
    'lighter': lambda self, *a: self,
})
qtgui.QPainter = type('QPainter', (), {
    '__init__': lambda self, *a: None,
    'Antialiasing': 1,
    'setRenderHint': lambda self, *a: None,
    'setPen': lambda self, *a: None,
    'setBrush': lambda self, *a: None,
    'setFont': lambda self, *a: None,
    'drawPixmap': lambda self, *a: None,
    'drawRect': lambda self, *a: None,
    'drawLine': lambda self, *a: None,
    'drawText': lambda self, *a: None,
    'fillRect': lambda self, *a: None,
    'end': lambda self: None,
    'font': lambda self: type('F', (), {'pointSize': lambda: 12})(),
    'fontMetrics': lambda self: type('FM', (), {
        'horizontalAdvance': lambda s, *a: 50,
        'width': lambda s, *a: 50,
        'height': lambda s: 16,
        'elidedText': lambda s, text, *a: text,
    })(),
})
qtgui.QPen = type('QPen', (), {
    '__init__': lambda self, *a: None,
    'setWidth': lambda self, *a: None,
})
qtgui.QFontMetrics = type('QFontMetrics', (), {
    '__init__': lambda self, *a: None,
    'horizontalAdvance': lambda self, *a: 50,
    'width': lambda self, *a: 50,
    'height': lambda self: 20,
})
# `canvas_renderer` 给滑窗格子标行列号时要 `QFont`（`QFont(painter.font())`
# 再 setPointSizeF）；桩里缺了它，凡导入 canvas_renderer 的测试都会在收集期
# 直接 ImportError。
qtgui.QFont = type('QFont', (), {
    '__init__': lambda self, *a: None,
    'pointSizeF': lambda self: 10.0,
    'setPointSizeF': lambda self, *a: None,
})
qtgui.QKeySequence = type('QKeySequence', (), {
    '__init__': lambda self, *a: None,
})
qtgui.QDragEnterEvent = type('QDragEnterEvent', (), {'__init__': lambda self, *a: None})
qtgui.QDropEvent = type('QDropEvent', (), {'__init__': lambda self, *a: None})
qtgui.QDrag = type('QDrag', (), {'__init__': lambda self, *a: None, 'exec_': lambda self, *a: 0})
qtgui.QFontDatabase = type('QFontDatabase', (), {'addApplicationFont': staticmethod(lambda *a: 0)})
qtgui.QImage = type('QImage', (), {'Format_ARGB32': 5})
# Qt6 把 QAction / QShortcut 从 QtWidgets 挪到了 QtGui；源码按新位置导入，
# mock 也得跟着挪，否则 `from PySide6.QtGui import QAction` 在收集期就炸。
qtgui.QAction = type('QAction', (), {'__init__': lambda self, *a: None})
qtgui.QShortcut = type('QShortcut', (_MockQWidget,), {
    '__init__': lambda self, *a, **kw: None,
    'setContext': lambda self, *a: None,
    'activated': type('Signal', (), {'connect': lambda self, f: None})(),
})

qtwidgets = types.ModuleType('PySide6.QtWidgets')
qtwidgets.QWidget = _MockQWidget
qtwidgets.QMainWindow = _MockQMainWindow
qtwidgets.QDialog = _MockQDialog
qtwidgets.QListWidget = _MockQListWidget
qtwidgets.QListWidgetItem = _MockQListWidgetItem
qtwidgets.QSplitter = _MockQSplitter
qtwidgets.QStackedWidget = type('QStackedWidget', (_MockQWidget,), {
    'addWidget': lambda self, *a: None,
    'setCurrentIndex': lambda self, *a: None,
    'currentIndex': lambda self: 0,
    'count': lambda self: 1,
    'widget': lambda self, i: None,
})
qtwidgets.QScrollArea = _MockQScrollArea
qtwidgets.QVBoxLayout = type('QVBoxLayout', (_MockQBoxLayout,), {})
qtwidgets.QHBoxLayout = type('QHBoxLayout', (_MockQBoxLayout,), {})
qtwidgets.QLabel = _MockQLabel
qtwidgets.QPushButton = _MockQPushButton
qtwidgets.QCheckBox = _MockQCheckBox
qtwidgets.QRadioButton = _MockQRadioButton
qtwidgets.QSpinBox = _MockQSpinBox
qtwidgets.QDoubleSpinBox = type('QDoubleSpinBox', (_MockQWidget,), {
    'minimum': lambda self: 0.0,
    'maximum': lambda self: 100.0,
    'value': lambda self: 0.0,
    'setValue': lambda self, *a: None,
    'setMinimum': lambda self, *a: None,
    'setMaximum': lambda self, *a: None,
    'setSingleStep': lambda self, *a: None,
    'setDecimals': lambda self, *a: None,
    'setMinimumWidth': lambda self, *a: None,
    'setStyleSheet': lambda self, *a: None,
    'valueChanged': type('Signal', (), {'connect': lambda self, f: None})(),
    'editingFinished': type('Signal', (), {'connect': lambda self, f: None})(),
})
qtwidgets.QLineEdit = _MockQLineEdit
qtwidgets.QProgressDialog = _MockQProgressDialog
qtwidgets.QProgressBar = type('QProgressBar', (_MockQWidget,), {
    'setObjectName': lambda self, *a: None,
    'setStyleSheet': lambda self, *a: None,
    'setMaximum': lambda self, *a: None,
    'setValue': lambda self, *a: None,
    'maximum': lambda self: 100,
    'value': lambda self: 0,
    'setTextVisible': lambda self, *a: None,
})
qtwidgets.QTableWidget = type('QTableWidget', (_MockQWidget,), {
    '__init__': lambda self, *a, **kw: None,
    'setHorizontalHeaderLabels': lambda self, *a: None,
    'setRowCount': lambda self, *a: None,
    'setColumnCount': lambda self, *a: None,
    'rowCount': lambda self: 0,
    'columnCount': lambda self: 0,
    'setItem': lambda self, *a: None,
    'item': lambda self, *a: None,
    'setCellWidget': lambda self, *a: None,
    'horizontalHeader': lambda self: type('Header', (), {
        'setSectionResizeMode': lambda s, *a: None,
        'setStretchLastSection': lambda s, *a: None,
    })(),
    'verticalHeader': lambda self: type('Header', (), {
        'setVisible': lambda s, *a: None,
    })(),
    'setEditTriggers': lambda self, *a: None,
    'setSelectionBehavior': lambda self, *a: None,
    'setSelectionMode': lambda self, *a: None,
    'blockSignals': lambda self, *a: None,
    'clearContents': lambda self: None,
    'setStyleSheet': lambda self, *a: None,
    'cellClicked': type('Signal', (), {'connect': lambda self, f: None})(),
})
qtwidgets.QTableWidgetItem = type('QTableWidgetItem', (), {
    '__init__': lambda self, *a, **kw: None,
    'setFlags': lambda self, *a: None,
    'flags': lambda self: 0,
    'setData': lambda self, *a: None,
    'data': lambda self, *a: None,
    'text': lambda self: '',
    'setText': lambda self, *a: None,
    'setBackground': lambda self, *a: None,
    'setToolTip': lambda self, *a: None,
})
qtwidgets.QHeaderView = type('QHeaderView', (), {
    'Stretch': 1, 'ResizeToContents': 2, 'Interactive': 0,
    'setSectionResizeMode': lambda self, *a: None,
    'setStretchLastSection': lambda self, *a: None,
    'setVisible': lambda self, *a: None,
})
qtwidgets.QAbstractItemView = type('QAbstractItemView', (), {
    'NoEditTriggers': 0, 'DoubleClicked': 1, 'EditKeyPressed': 2,
    'SelectedClicked': 4, 'SelectRows': 1, 'SingleSelection': 1,
})
qtwidgets.QInputDialog = _MockQInputDialog
qtwidgets.QMessageBox = _MockQMessageBox
qtwidgets.QFileDialog = _MockQFileDialog
qtwidgets.QColorDialog = type('QColorDialog', (), {
    'DontUseNativeDialog': 1,
    '__init__': lambda self, *a, **kw: None,
    'getColor': staticmethod(lambda *a, **kw: qtgui.QColor()),
    'setOption': lambda self, *a: None,
    'setWindowTitle': lambda self, *a: None,
    'setCurrentColor': lambda self, *a: None,
    'setStyleSheet': lambda self, *a: None,
    'showEvent': lambda self, *a: None,
    'findChild': lambda self, *a: None,
    'findChildren': lambda self, *a: [],
    'exec_': lambda self: _MockQDialog.Rejected,
    'currentColor': lambda self: qtgui.QColor(),
})
qtwidgets.QDialogButtonBox = type('QDialogButtonBox', (), {'Ok': 1, 'Cancel': 2})
qtwidgets.QApplication = _MockQApplication
qtwidgets.QMenu = type('QMenu', (_MockQWidget,), {
    'addAction': lambda self, *a: None,
    'addSeparator': lambda self: None,
    'exec_': lambda self, *a: None,
})
qtwidgets.QWidgetAction = type('QWidgetAction', (), {
    '__init__': lambda self, *a: None,
    'setDefaultWidget': lambda self, *a: None,
})
qtwidgets.QGroupBox = type('QGroupBox', (_MockQWidget,), {
    'setStyleSheet': lambda self, *a: None,
})
qtwidgets.QFrame = type('QFrame', (_MockQWidget,), {
    'VLine': 1,
    'Sunken': 3,
    'setFrameShape': lambda self, *a: None,
    'setFrameShadow': lambda self, *a: None,
    'setFixedHeight': lambda self, *a: None,
})
qtwidgets.QGridLayout = type('QGridLayout', (_MockQBoxLayout,), {})
qtwidgets.QTextBrowser = type('QTextBrowser', (_MockQWidget,), {
    'setReadOnly': lambda self, *a: None,
    'setText': lambda self, *a: None,
    'append': lambda self, *a: None,
    'setPlaceholderText': lambda self, *a: None,
    'verticalScrollBar': lambda self: type('SB', (), {'setValue': lambda s, v: None, 'maximum': lambda s: 0})(),
})
qtwidgets.QTextEdit = type('QTextEdit', (_MockQWidget,), {
    'setReadOnly': lambda self, *a: None,
    'setText': lambda self, *a: None,
    'append': lambda self, *a: None,
    'setPlaceholderText': lambda self, *a: None,
    'textCursor': lambda self: type('TC', (), {
        'movePosition': lambda s, *a: None,
        'setPosition': lambda s, *a: None,
        'insertText': lambda s, *a: None,
        'End': 0,
        'Start': 1,
        'EndOfBlock': 2,
        'KeepAnchor': 0,
    })(),
    'setVerticalScrollBar': lambda self, *a: None,
    'setContentsMargins': lambda self, *a: None,
    'setMinimumHeight': lambda self, *a: None,
})
qtwidgets.QComboBox = type('QComboBox', (_MockQWidget,), {
    'addItems': lambda self, *a: None,
    'addItem': lambda self, *a: None,
    'setCurrentIndex': lambda self, *a: None,
    'currentIndex': lambda self: 0,
    'currentData': lambda self: None,
    'currentText': lambda self: '',
    'count': lambda self: 0,
    'clear': lambda self: None,
    'blockSignals': lambda self, *a: None,
    'setMaximumWidth': lambda self, *a: None,
    'setFixedWidth': lambda self, *a: None,
    'currentIndexChanged': type('Signal', (), {'connect': lambda self, f: None})(),
})
class _MockQDockWidget(_MockQWidget):
    DockWidgetClosable = 4
    DockWidgetMovable = 1
    DockWidgetFloatable = 2
    def __init__(self, *a, **kw):
        super().__init__()
    def setWidget(self, *a): pass
    def widget(self): return None
    def setFeatures(self, *a): pass
    def features(self): return 1
    def setAllowedAreas(self, *a): pass
    def setWindowTitle(self, *a): pass
    def setObjectName(self, *a): pass
    def toggleViewAction(self):
        return type('Act', (), {'setVisible': lambda s, *a: None})()


qtwidgets.QDockWidget = _MockQDockWidget
qtwidgets.QTabWidget = type('QTabWidget', (_MockQWidget,), {
    '__init__': lambda self, *a, **kw: None,
    'addTab': lambda self, *a: 0,
    'setCurrentIndex': lambda self, *a: None,
    'currentIndex': lambda self: 0,
    'count': lambda self: 0,
    'widget': lambda self, i: None,
    'setDocumentMode': lambda self, *a: None,
    'currentChanged': type('Signal', (), {'connect': lambda self, f: None})(),
})
qtwidgets.QToolButton = type('QToolButton', (_MockQWidget,), {
    'setIcon': lambda self, *a: None,
    'setIconSize': lambda self, *a: None,
    'setFixedSize': lambda self, *a: None,
    'setToolTip': lambda self, *a: None,
    'setFocusPolicy': lambda self, *a: None,
    'setStyleSheet': lambda self, *a: None,
    'clicked': type('Signal', (), {'connect': lambda self, f: None})(),
})
qtwidgets.QSlider = type('QSlider', (_MockQWidget,), {
    'setRange': lambda self, *a: None,
    'setValue': lambda self, *a: None,
    'value': lambda self: 50,
    'setFixedWidth': lambda self, *a: None,
    'setTracking': lambda self, *a: None,
    'setFocusPolicy': lambda self, *a: None,
    'blockSignals': lambda self, *a: None,
    'valueChanged': type('Signal', (), {'connect': lambda self, f: None})(),
})
# 注册模块
sys.modules['PySide6'] = types.ModuleType('PySide6')
sys.modules['PySide6.QtCore'] = qtcore
sys.modules['PySide6.QtGui'] = qtgui
sys.modules['PySide6.QtWidgets'] = qtwidgets
