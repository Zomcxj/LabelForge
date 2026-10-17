"""切英文后，推理参数行不能残留中文。

用户反馈「推理的类别筛选没同步英文」。查下来是三个各自漏掉的位置：

- 筛选按钮文字由 `_refresh_filter_button()` 按当前选择拼出（`类别筛选(2/80)`），
  切语言路径没调它；`ModelRow.retranslate` 的注释把这件事推给了主窗口。
- 范围摘要同理，要 `_refresh_scope_button()` 重设。
- `ModelRow.retranslate` 只刷新了范围按钮的 tooltip，漏了按钮文字本身。

前两条在 `_refresh_ui_texts` 里补调用，第三条给按钮补一句 setText。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_real_qt(script, timeout=180):
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD],
                       cwd=ROOT, env=env, text=True, capture_output=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    return r.stdout


_SWITCH_TO_EN = '''
import os, sys
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor

def has_cjk(s):
    return any("\\u4e00" <= ch <= "\\u9fff" for ch in s or "")

ed = ImageEditor()
ed._all_class_names = lambda: ["car", "person", "bus"]
ed._ensure_model_row()
ed._class_filter = {"car"}          # 筛掉两类，按钮才会带上计数
ed._refresh_filter_button()

i18n.set_lang("zh")
i18n.set_lang("en")
ed._refresh_ui_texts()

row = ed.model_row
leftovers = []
for name in ("filter_btn", "scope_btn", "load_btn", "size_lbl",
             "batch_to_label", "scope_summary_label", "conf_lbl", "iou_lbl"):
    w = getattr(row, name, None)
    if w is None:
        continue
    text = w.text() if hasattr(w, "text") else ""
    if has_cjk(text):
        leftovers.append((name, text))
for name in ("filter_btn", "scope_btn", "load_btn"):
    w = getattr(row, name)
    if has_cjk(w.toolTip()):
        leftovers.append((name + ".tooltip", w.toolTip()))

print("FILTER_TEXT", repr(row.filter_btn.text()))
print("SCOPE_TEXT", repr(row.scope_btn.text()))
print("LEFTOVERS", leftovers)
print("DONE")
'''


def test_inference_row_has_no_chinese_left_after_switching_to_english():
    out = _run_real_qt(_SWITCH_TO_EN)
    assert "DONE" in out, out
    line = [l for l in out.splitlines() if l.startswith("LEFTOVERS")][0]
    leftovers = line[len("LEFTOVERS"):].strip()
    assert leftovers == "[]", f"切英文后仍有中文残留：{leftovers}\n{out}"


def test_filter_button_keeps_its_count_when_relabelled():
    """切语言的刷新不能把 `(2/80)` 计数丢掉，也不能把它变成英文以外的格式。"""
    out = _run_real_qt(_SWITCH_TO_EN)
    text = [l for l in out.splitlines() if l.startswith("FILTER_TEXT")][0]
    assert "Class filter" in text, text
    assert "(1/3)" in text, f"计数丢失或串位：{text}"


def test_scope_button_text_is_relabelled_by_model_row():
    """范围按钮文字在 ModelRow.retranslate 里重设。

    此前只重设了 tooltip，按钮上仍写着中文。
    """
    out = _run_real_qt(_SWITCH_TO_EN)
    text = [l for l in out.splitlines() if l.startswith("SCOPE_TEXT")][0]
    assert "Inference scope" in text, text


def test_language_switch_path_calls_both_refreshers():
    """`_refresh_ui_texts` 必须真的调到两个按状态拼文案的刷新方法。"""
    source = (ROOT / "labelforge" / "ui" / "mixins" / "translation.py").read_text(
        encoding="utf-8")
    block = source[source.index("row = getattr(self, 'model_row', None)"):
                   source.index("panel = getattr(self, '_class_filter_panel', None)")]
    assert "_refresh_filter_button" in block, "切语言没重设筛选按钮文案"
    assert "_refresh_scope_button" in block, "切语言没重设范围摘要文案"


def test_open_scope_panel_is_relabelled_on_language_switch():
    """范围面板非模态，可能正开着；切语言要跟着变。"""
    source = (ROOT / "labelforge" / "ui" / "mixins" / "translation.py").read_text(
        encoding="utf-8")
    assert "_scope_panel" in source, "切语言没刷新正开着的范围面板"


def test_open_scope_panel_actually_switches_text():
    """真实 Qt：面板开着时切语言，模式按钮与窗口标题都要变。"""
    script = '''
import os, sys
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])

from labelforge.ui import i18n
from labelforge.ui.main_window import ImageEditor
from labelforge.ui.scope_panel import ScopePanel

def has_cjk(s):
    return any("\\u4e00" <= ch <= "\\u9fff" for ch in s or "")

ed = ImageEditor()
ed._ensure_model_row()
i18n.set_lang("zh")
ed._scope_panel = ScopePanel(parent=ed)   # 模拟「面板正开着」

panel = ed._scope_panel
i18n.set_lang("en")
ed._refresh_ui_texts()

bad = []
if has_cjk(panel.windowTitle()):
    bad.append(("windowTitle", panel.windowTitle()))
from labelforge.engine.inference_scope import MODE_FULL, MODE_WINDOW, MODE_ROI
for mode in (MODE_FULL, MODE_WINDOW, MODE_ROI):
    text = panel._buttons[mode].text()
    if has_cjk(text):
        bad.append((mode, text))
print("BAD", bad)
print("REASON", panel.windowTitle())
print("DONE")
'''
    out = _run_real_qt(script)
    assert "DONE" in out, out
    line = [l for l in out.splitlines() if l.startswith("BAD")][0]
    assert line[len("BAD"):].strip() == "[]", f"范围面板仍有中文：{line}\n{out}"
