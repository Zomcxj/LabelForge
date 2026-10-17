"""统计弹窗 ⑦ 的回归测试：不再阻塞主线程、先弹窗后填充。

用户报的「统计有时候需要 5s 才能打开」。病因是 `_collect_bg_stats_for_dialog()`
在 UI 线程上把每个 sidecar json **解析两遍**（一遍数数量、一遍取框类型，
`collect_background_label_counts` + `collect_background_label_tasks`）。
"""
import json
import os


def _write_dataset(tmp_path, count=30):
    paths = []
    for i in range(count):
        img = tmp_path / f"i{i:02d}.jpg"
        img.write_bytes(b"")
        (tmp_path / f"i{i:02d}.json").write_text(json.dumps({
            "shapes": [{"label": "cat", "points": [[0, 0], [5, 5]],
                        "shape_type": "rectangle"}],
        }), encoding="utf-8")
        paths.append(str(img))
    return paths


def test_combined_scan_parses_each_json_once(tmp_path, monkeypatch):
    """`collect_background_label_stats_and_tasks()` 每张 json 只解析一次。"""
    import builtins
    from labelforge.engine.image_loader import scan as scan_mod

    paths = _write_dataset(tmp_path, 5)
    opens = []
    real_open = builtins.open

    def counting_open(*args, **kwargs):
        target = args[0] if args else kwargs.get("file")
        opens.append(str(target))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    counts, tasks = scan_mod.collect_background_label_stats_and_tasks(paths)

    assert counts == {"cat": 5}, counts
    assert tasks.get("cat") == {"det"}, tasks
    json_opens = [p for p in opens if p.endswith(".json")]
    assert len(json_opens) == len(paths), (
        f"每张 json 应当只解析一次，实际打开 {len(json_opens)} 次"
        f"（{len(paths)} 张图）")


def test_combined_scan_counts_match_the_old_single_purpose_path(tmp_path):
    """合并扫描的计数与旧的只数数量那条路径**完全一致**（不能改口径）。"""
    from labelforge.engine.image_loader import scan as scan_mod

    paths = _write_dataset(tmp_path, 4)
    old = scan_mod.collect_background_label_counts(paths)
    new, _tasks = scan_mod.collect_background_label_stats_and_tasks(paths)
    assert new == old, f"合并扫描改了计数结果：{new} != {old}"


_STATS_HOST = """
import json, os, tempfile, time
from PySide6.QtWidgets import QApplication, QListWidgetItem, QDialog, QWidget
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QEventLoop, QTimer
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
tmp = tempfile.mkdtemp(prefix='pl-stats-')
paths = []
for i in range(20):
    p = os.path.join(tmp, 'i%02d.jpg' % i)
    QImage(60, 40, QImage.Format_RGB888).save(p)
    with open(os.path.join(tmp, 'i%02d.json' % i), 'w', encoding='utf-8') as f:
        json.dump({'shapes': [{'label': 'cat', 'points': [[0, 0], [5, 5]],
                               'shape_type': 'rectangle'}]}, f)
    paths.append(p)

w = ImageEditor()
w.resize(1000, 800)
w.show()
app.processEvents()
w.background_images = list(paths)
w.background_list.clear()
for p in paths:
    w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(paths[0]))
app.processEvents()


def wait_for(pred, timeout_ms=15000):
    loop = QEventLoop()
    tm = QTimer(); tm.timeout.connect(lambda: (pred() and loop.quit()))
    tm.start(10); QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_(); tm.stop()
"""


def test_stats_dialog_opens_without_scanning_on_the_ui_thread():
    """打开统计弹窗时 UI 线程**不再**同步扫盘（把扫描换成炸弹也不会炸）。

    加载中的提示在**总计标签**上（背景表保持 0 行：总计与表是同源的，插假行会
    让求和对不上）。
    """
    from tests.test_inference_scope_ui import _run_real_qt, _lines

    out = _run_real_qt(_STATS_HOST + """
# 把两条同步扫盘入口换成炸弹：打开弹窗若同步碰了它们就会抛
import labelforge.engine.image_loader as il

def boom(*a, **k):
    raise AssertionError('UI 线程同步扫盘了')

il.collect_background_label_counts = boom
il.collect_background_label_tasks = boom

orig = QDialog.exec_
def fake_exec(self, *a):
    print('打开那一刻表行数:', self._bg_table.rowCount())
    print('打开那一刻总计标签:', self._total_label.text())
    print('打开没触发同步扫盘: True')
    return 0
QDialog.exec_ = fake_exec
try:
    w._show_label_stats()
finally:
    QDialog.exec_ = orig
""")
    lines = _lines(out)
    assert lines[0].endswith("0"), (
        f"扫描前不该有数据行（插假行会让总计与表求和不一致）：{lines[0]}")
    assert "正在统计" in lines[1], (
        f"扫描前应当在总计标签上提示正在统计：{lines[1]}")
    assert lines[2].endswith("True"), lines[2]


def test_stats_scan_fills_the_table_after_the_dialog_is_up():
    """先弹窗、后台扫完再填表：表格最终出现真实数据，且期间不阻塞。"""
    from tests.test_inference_scope_ui import _run_real_qt, _lines

    out = _run_real_qt(_STATS_HOST + """
orig = QDialog.exec_
marks = {}

def fake_exec(self, *a):
    marks['loading'] = self._total_label.text()
    marks['rows_before'] = self._bg_table.rowCount()
    wait_for(lambda: getattr(self, '_scan_applied', False))
    marks['after'] = (self._bg_table.item(0, 0).text()
                      if self._bg_table.item(0, 0) else None)
    marks['count'] = (self._bg_table.item(0, 1).text()
                      if self._bg_table.item(0, 1) else None)
    marks['total_after'] = self._total_label.text()
    return 0

QDialog.exec_ = fake_exec
try:
    w._show_label_stats()
finally:
    QDialog.exec_ = orig

print('扫描前表行数:', marks.get('rows_before'))
print('扫描前标签含提示:', '正在统计' in (marks.get('loading') or ''))
print('扫完:', marks.get('after'), marks.get('count'))
print('扫完总计不再说正在统计:', '正在统计' not in (marks.get('total_after') or ''))
""")
    lines = _lines(out)
    assert lines[0].endswith("0"), lines[0]
    assert lines[1].endswith("True"), lines[1]
    assert lines[2] == "扫完: cat 20", (
        f"后台扫描没把结果填进表：{lines[2]}")
    assert lines[3].endswith("True"), lines[3]


def test_stats_second_open_shows_cache_immediately():
    """第二次打开：已有缓存就**立刻**显示，不必再等扫描。"""
    from tests.test_inference_scope_ui import _run_real_qt, _lines

    out = _run_real_qt(_STATS_HOST + """
orig = QDialog.exec_

def wait_exec(self, *a):
    wait_for(lambda: getattr(self, '_scan_applied', False))
    return 0

QDialog.exec_ = wait_exec
try:
    w._show_label_stats()      # 第一次：建立缓存
finally:
    QDialog.exec_ = orig

marks = {}
def instant_exec(self, *a):
    marks['first'] = (self._bg_table.item(0, 0).text()
                      if self._bg_table.item(0, 0) else None)
    marks['count'] = (self._bg_table.item(0, 1).text()
                      if self._bg_table.item(0, 1) else None)
    return 0
QDialog.exec_ = instant_exec
try:
    w._show_label_stats()      # 第二次：应当立刻显示
finally:
    QDialog.exec_ = orig

print('二次打开立刻显示:', marks.get('first'), marks.get('count'))
""")
    lines = _lines(out)
    assert lines[0] == "二次打开立刻显示: cat 20", (
        f"第二次打开没有用缓存立刻显示：{lines[0]}")


def test_stats_closing_during_scan_does_not_crash():
    """扫描还没回来就关掉弹窗：不能崩，结果照样进缓存。"""
    from tests.test_inference_scope_ui import _run_real_qt, _lines

    out = _run_real_qt(_STATS_HOST + """
orig = QDialog.exec_
def close_exec(self, *a):
    self.finished.emit(0)      # 用户立刻关掉
    return 0
QDialog.exec_ = close_exec
try:
    w._show_label_stats()
finally:
    QDialog.exec_ = orig

wait_for(lambda: bool(getattr(w, '_cached_bg_label_stats', None)))
print('关窗后仍落地到缓存:', bool(getattr(w, '_cached_bg_label_stats', None)))
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), (
        f"关窗时结果丢了或崩了：{lines}")


def test_stats_scan_on_an_empty_dataset_does_not_hang():
    """空数据集：打开统计不该卡住，也不该留个半死不活的 worker。"""
    from tests.test_inference_scope_ui import _run_real_qt, _lines

    out = _run_real_qt(_STATS_HOST + """
w.background_images = []
app.processEvents()
orig = QDialog.exec_
marks = {}
def instant_exec(self, *a):
    marks['rows'] = self._bg_table.rowCount()
    return 0
QDialog.exec_ = instant_exec
try:
    w._show_label_stats()
finally:
    QDialog.exec_ = orig
print('空数据集行数:', marks.get('rows'))
print('没留挂着的 worker:', getattr(w, '_stats_scan_worker', None) is None)
""")
    lines = _lines(out)
    assert lines[0].endswith("0"), lines[0]
    assert lines[1].endswith("True"), lines[1]
