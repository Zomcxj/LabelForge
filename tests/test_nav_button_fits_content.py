# -*- coding: utf-8 -*-
"""navBtn 小按钮必须放得下自己的内容。

用户报「标签列表里的 T / G 看不见」。根因不在文字，在 QSS：`navBtn` 没写
`padding`，Qt 走平台默认内边距（每侧约 5px），再算上菜单箭头，22px 宽的
按钮只剩 **4px** 内容宽度，而一个字母就要 10px。Qt 画按钮文字时会裁到
`SE_PushButtonContents` 这个矩形里，于是字母被切掉。

这里量的是真实控件：把 `SE_PushButtonContents` 的宽度与文字/图标的实际
需要宽度对比。用像素比对更直接，但离屏环境没有任何可用字体（`JetBrains
Mono` 渲染出来是空心方框），截图看不出对错，所以走字体度量。
"""
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Qt 子进程退出用 sys.exit（走正常 finalize），不用 os._exit：后者仍会经
# ExitProcess 卸载 DLL 触发 Qt 静态析构，实测 150 次崩 12 次；sys.exit 崩 0 次。
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


def test_every_nav_button_has_room_for_its_content():
    """主界面里每个 navBtn 的内容可用宽都必须 >= 文字/图标需要宽。"""
    script = '''
from PySide6.QtWidgets import QApplication, QStyleOptionButton, QStyle
from PySide6.QtGui import QFontMetrics
app = QApplication.instance() or QApplication([])
from labelforge.ui.main_window import ImageEditor

w = ImageEditor()
w.resize(1500, 950)
w.show()
app.processEvents()

NAMES = ("prev_img_btn", "next_img_btn", "bg_filter_btn", "bg_refresh_btn",
         "bg_label_mode_btn", "task_filter_btn", "group_filter_btn")
report = []
too_tight = []
for name in NAMES:
    b = getattr(w, name, None)
    assert b is not None, ("按钮不存在", name)
    opt = QStyleOptionButton()
    b.initStyleOption(opt)
    avail = b.style().subElementRect(QStyle.SE_PushButtonContents, opt, b).width()
    text = b.text()
    need_text = QFontMetrics(b.font()).horizontalAdvance(text) if text else 0
    need_icon = b.iconSize().width() if not b.icon().isNull() else 0
    need = max(need_text, need_icon)
    report.append((name, b.width(), avail, text, need_text, need_icon))
    if avail < need:
        too_tight.append((name, avail, need))

print("REPORT:", report)
assert not too_tight, ("navBtn 内容放不下会被裁", too_tight, report)
print("OK_NAV_FITS")
'''
    out = _run_real_qt(script)
    assert "OK_NAV_FITS" in out, out


def test_navbtn_qss_zeroes_the_platform_padding():
    """navBtn 的 QSS 必须显式写 padding。

    不写时走平台默认内边距，22px 的按钮只剩 4px 内容宽——这条是上面那条
    的成因锁：谁把 padding 删了，上面那条会红，这条会指出「为什么」。
    """
    source = (ROOT / "labelforge" / "ui" / "theme.py").read_text(encoding="utf-8")
    block = source.split("QPushButton#navBtn {{", 1)[1].split("}}", 1)[0]
    assert "padding: 0px;" in block, (
        "navBtn 必须显式 padding: 0px，否则内容被平台默认内边距裁掉",
        block,
    )