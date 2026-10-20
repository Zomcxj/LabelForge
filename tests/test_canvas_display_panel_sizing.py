# -*- coding: utf-8 -*-
"""画布显示面板的尺寸回归：换语言后标题不能被裁。

用户报的「英文模式下亮度和对比度被遮挡」：英文的 Brightness / Contrast
比中文的「画面亮度」「对比度」宽得多（实测需要 100/80px，中文只要 40/30px），
而面板矩形不会自己变大。

这里量的是真实 Qt 控件，不是源码字符串：问题出在布局缓存，只有真的建出
面板、真的换语言才看得出来。
"""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程退出用 sys.exit（走正常 finalize），不用 os._exit：后者仍会经
# ExitProcess 卸载 DLL 触发 Qt 静态析构，实测 150 次崩 12 次；sys.exit 崩 0 次。
# 断言失败时脚本先抛异常，两种写法返回码都是 1，不会掩盖真实失败。
_QT_EXIT_GUARD = '''
import sys
sys.stdout.flush()
sys.exit(0)
'''


def _run_real_qt(script):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", script + _QT_EXIT_GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        timeout=180, encoding="utf-8", errors="replace",
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_panel_widens_when_switching_to_english():
    """切英文后面板必须变宽，且宽度容得下 body 的 sizeHint + 左右边距。"""
    script = '''
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.widgets.canvas_display_panel import CanvasDisplayPanel
import labelforge.ui.i18n as i18n

p = CanvasDisplayPanel()
p._collapsed = False
p._apply_collapsed_state()
p.show()
app.processEvents()

m = p.layout().contentsMargins()
margin = m.left() + m.right()
zh_w = p.width()
zh_need = p._body.sizeHint().width()
assert zh_w >= zh_need + margin, ("zh 面板比内容窄", zh_w, zh_need, margin)

i18n.set_lang("en")
p.retranslate_ui()
app.processEvents()
en_w = p.width()
en_need = p._body.sizeHint().width()
assert en_need > zh_need, ("英文内容并没变宽，用例失去意义", zh_need, en_need)
assert en_w >= en_need + margin, ("英文下面板比内容窄", en_w, en_need, margin)
assert en_w > zh_w, ("切英文后面板没变宽", zh_w, en_w)

i18n.set_lang("zh")
p.retranslate_ui()
app.processEvents()
assert p.width() <= en_w, ("切回中文后没收窄", p.width(), en_w)
print("OK_PANEL_WIDENS")
'''
    assert "OK_PANEL_WIDENS" in _run_real_qt(script)


def test_panel_resizes_when_expanded_after_a_language_switch():
    """收展与切语言的顺序不该把尺寸算坏。

    注意：这条在修复前后都通过，属于**行为顺序的守护**，不是严格回归用例
    ——独立面板的收→展本身就会触发布局重算。真正卡住这个 bug 的是上面
    那条「已展开时切语言」（实测未修复时面板停在 218，英文需要 318）。
    """
    script = '''
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.widgets.canvas_display_panel import CanvasDisplayPanel
import labelforge.ui.i18n as i18n

p = CanvasDisplayPanel()
p.show()
app.processEvents()

# 1) 中文下先展开，让布局按中文算出一版尺寸
p._collapsed = False
p._apply_collapsed_state()
app.processEvents()
zh_w = p.width()
assert zh_w > 0

# 2) 收起
p._collapsed = True
p._apply_collapsed_state()
app.processEvents()

# 3) 收起状态下切英文
i18n.set_lang("en")
p.retranslate_ui()
app.processEvents()
assert p._collapsed is True, "收起状态不该被展开"

# 4) 再展开
p._collapsed = False
p._apply_collapsed_state()
app.processEvents()

m = p.layout().contentsMargins()
margin = m.left() + m.right()
need = p._body.sizeHint().width()
assert p.width() >= need + margin, ("展开后仍按中文旧尺寸", p.width(), need, margin)
assert p.width() > zh_w, ("展开后没按英文变宽", zh_w, p.width())
print("OK_EXPAND_AFTER_SWITCH")
'''
    assert "OK_EXPAND_AFTER_SWITCH" in _run_real_qt(script)


def test_panel_collapsed_size_is_just_the_chevron():
    """收起时只留图标：尺寸与文字无关，切语言不该把它撑大。"""
    script = '''
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.widgets.canvas_display_panel import CanvasDisplayPanel
import labelforge.ui.i18n as i18n

p = CanvasDisplayPanel()
p.show()
app.processEvents()
zh = (p.width(), p.height())

i18n.set_lang("en")
p.retranslate_ui()
app.processEvents()
assert (p.width(), p.height()) == zh, ("收起态被切语言撑大了", zh, (p.width(), p.height()))
assert p._body.isVisible() is False
print("OK_COLLAPSED_STABLE")
'''
    assert "OK_COLLAPSED_STABLE" in _run_real_qt(script)
