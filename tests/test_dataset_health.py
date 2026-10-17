"""数据集健康仪表盘纯逻辑测试。"""
import json
import os


def _write_output_sidecar(tmp_path, image_path, shapes):
    """贴图实际保存在输出目录 sidecar：{background_dir}_paste_output/{stem}.json。"""
    from labelforge.core.utils import PathUtils
    out_dir = PathUtils.get_output_dir(image_path)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"shapes": shapes}, f, ensure_ascii=False)
    return json_path


def _write(tmp_path, stem, shapes, size=(200, 200)):
    img = tmp_path / f"{stem}.png"
    img.write_bytes(b"x")
    (tmp_path / f"{stem}.json").write_text(
        json.dumps({"shapes": shapes, "imageWidth": size[0], "imageHeight": size[1]}),
        encoding="utf-8")
    return str(img)


def _box(label="cat", x=0, y=0, w=100, h=50, shape_type=None):
    b = {"label": label, "x": x, "y": y, "width": w, "height": h}
    if shape_type:
        b["shape_type"] = shape_type
    return b


class _SaveRect:
    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self): return self._x
    def y(self): return self._y
    def width(self): return self._w
    def height(self): return self._h


class _Toggle:
    def __init__(self, checked):
        self._checked = checked

    def isChecked(self):
        return self._checked


class _TextInput:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


def _real_save(img_path, prefix_checked=True, prefix="paste"):
    """走真实 SaveManager：默认前缀 paste 落盘 {prefix}_{stem}.json。"""
    from labelforge.engine.save_manager import SaveManager

    class _Bg:
        def width(self): return 100
        def height(self): return 80

    class Editor:
        _is_delete_view = False
        edit_mode = "paste"
        current_background = _Bg()
        current_background_index = 0
        background_images = [img_path]
        canvas_items = [(None, _SaveRect(5, 5, 10, 10), "logo")]
        detection_boxes_dict = {0: []}
        detection_boxes = []
        prefix_checkbox = _Toggle(prefix_checked)
        prefix_input = _TextInput(prefix)
        _dataset_stats_dirty = False

    mgr = SaveManager(Editor())
    mgr.save_json(*mgr.get_save_info())


def _image_in_subdir(tmp_path, name="cat.png"):
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    img = img_dir / name
    img.write_bytes(b"x")
    return str(img)


def test_collect_shape_geometry_basic(tmp_path):
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "a", [
        _box("cat", w=100, h=50),
        _box("dog", w=40, h=80),
    ])
    out = collect_shape_geometry([img])
    assert out["images_scanned"] == 1
    assert len(out["boxes"]) == 2
    cat = next(b for b in out["boxes"] if b["label"] == "cat")
    assert cat["width"] == 100.0 and cat["height"] == 50.0
    assert cat["area"] == 5000.0
    assert cat["aspect"] == 2.0


def test_collect_shape_geometry_skips_points_and_degenerate(tmp_path):
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "b", [
        _box("kp", w=0, h=0, shape_type="point"),
        _box("zero", w=0, h=0),
        _box("ok", w=10, h=10),
    ])
    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["ok"]


def test_collect_shape_geometry_skips_missing_sidecar(tmp_path):
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = tmp_path / "none.png"
    img.write_bytes(b"x")
    out = collect_shape_geometry([str(img)])
    assert out["boxes"] == []
    assert out["images_scanned"] == 0


def test_collect_shape_geometry_memory_overrides_disk(tmp_path):
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "c", [_box("cat", w=100, h=50)])
    memory = {0: [_box("cat", w=10, h=10), _box("bird", w=20, h=20)]}
    out = collect_shape_geometry([img], memory_boxes=memory)
    assert sorted(b["label"] for b in out["boxes"]) == ["bird", "cat"]
    assert len(out["boxes"]) == 2


def test_collect_paste_geometry():
    from labelforge.engine.dataset_health import collect_paste_geometry

    class Rect:
        def __init__(self, w, h, x=0, y=0):
            self._w, self._h, self._x, self._y = w, h, x, y
        def width(self):
            return self._w
        def height(self):
            return self._h
        def x(self):
            return self._x
        def y(self):
            return self._y

    items = {0: [(None, Rect(100, 50), "car"), (None, Rect(20, 20), "car")]}
    out = collect_paste_geometry(items)
    assert len(out) == 2
    assert all(b["label"] == "car" for b in out)
    assert out[0]["area"] == 5000.0


def _geo(label, w, h, x=0.0, y=0.0):
    return {'label': label, 'x': float(x), 'y': float(y),
            'width': float(w), 'height': float(h),
            'area': float(w * h), 'aspect': w / h}


def test_compute_health_class_dist_descending():
    from labelforge.engine.dataset_health import compute_health
    stats = compute_health([_geo("a", 10, 10), _geo("a", 10, 10),
                            _geo("b", 10, 10)])
    assert stats['class_dist'] == [{'label': 'a', 'count': 2},
                                   {'label': 'b', 'count': 1}]
    assert stats['summary']['total_boxes'] == 3
    assert stats['summary']['class_count'] == 2


def test_compute_health_size_hist_covers_all_boxes():
    from labelforge.engine.dataset_health import compute_health
    boxes = [_geo("a", i * 10, 10) for i in range(1, 21)]
    stats = compute_health(boxes)
    assert sum(stats['size_hist']['counts']) == len(boxes)
    assert len(stats['size_hist']['edges']) == len(stats['size_hist']['counts']) + 1


def test_compute_health_aspect_hist():
    from labelforge.engine.dataset_health import compute_health
    boxes = [_geo("a", 10, 10), _geo("a", 20, 10), _geo("a", 40, 10)]
    stats = compute_health(boxes)
    assert sum(stats['aspect_hist']['counts']) == 3


def test_compute_health_iou_hist_same_label_only():
    from labelforge.engine.dataset_health import compute_health
    # 两个同类框完全重叠 -> IoU 1.0 落最后一桶
    boxes = [_geo("a", 100, 100), _geo("a", 100, 100)]
    for i, b in enumerate(boxes):
        b['image_index'] = 0
    stats = compute_health(boxes)
    assert stats['iou_hist']['counts'][-1] == 1


def test_health_advice_rare_class():
    from labelforge.engine.dataset_health import health_advice
    stats = {
        'class_dist': [{'label': 'a', 'count': 1950}, {'label': 'b', 'count': 50}],
        'size_hist': {'edges': [0, 1, 2], 'counts': [1000, 1000]},
        'aspect_hist': {'edges': [0, 1, 2, 3], 'counts': [800, 600, 600]},
        'iou_hist': {'edges': [0.0, 0.5, 1.0], 'counts': [1800, 200]},
        'summary': {'total_boxes': 2000, 'class_count': 2, 'images_scanned': 1},
    }
    advice = health_advice(stats)
    assert all(isinstance(a, dict) and 'key' in a and 'params' in a
               for a in advice)
    assert any('b' in a['params'].get('label', '') for a in advice)


def test_health_advice_no_issue():
    from labelforge.engine.dataset_health import health_advice
    stats = {
        'class_dist': [{'label': 'a', 'count': 50}, {'label': 'b', 'count': 50}],
        'size_hist': {'edges': [0, 1, 2], 'counts': [50, 50]},
        'aspect_hist': {'edges': [0, 1, 2, 3], 'counts': [40, 30, 30]},
        'iou_hist': {'edges': [i / 10 for i in range(11)],
                     'counts': [90, 0, 0, 0, 0, 0, 0, 0, 0, 10]},
        'summary': {'total_boxes': 100, 'class_count': 2, 'images_scanned': 1},
    }
    assert health_advice(stats) == [{'key': "未发现明显失衡", 'params': {}}]


def test_worker_signals_and_run_source():
    import inspect
    from labelforge.engine import dataset_health
    assert hasattr(dataset_health, 'DatasetHealthWorker')
    src = inspect.getsource(dataset_health.DatasetHealthWorker)
    assert 'health_ready' in src
    assert 'collect_shape_geometry' in src
    assert 'health_advice' in src


def test_health_bar_chart_api_and_paint():
    from labelforge.ui.widgets.health_charts import HealthBarChart
    chart = HealthBarChart()
    chart.set_data([{'label': 'a', 'value': 3}, {'label': 'b', 'value': 1}])
    chart.paintEvent(None)
    chart.set_data([{'label': 'a', 'value': 3}], horizontal=False,
                   highlight_extremes=True)
    chart.paintEvent(None)
    chart.set_histogram([0.0, 0.5, 1.0], [4, 6])
    chart.paintEvent(None)
    chart.set_placeholder("暂无数据")
    # paintEvent 必须可调用（conftest mock QPainter）
    chart.paintEvent(None)


def test_i18n_health_terms():
    from labelforge.ui.i18n import _strings
    for key in ("数据集健康", "正在分析", "暂无贴图数据", "请先加载数据集",
                "类别分布", "尺寸分布", "长宽比分布", "IoU 重叠分布",
                "建议", "未发现明显失衡"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_stats_mixin_exposes_health_section_source():
    import inspect
    from labelforge.ui.mixins.stats import StatsMixin
    src = inspect.getsource(StatsMixin)
    assert "_build_health_section" in src
    assert "DatasetHealthWorker" in src
    assert "HealthBarChart" in src
    assert "_close_health_worker" in src
    assert "worker.finished.connect(worker.deleteLater)" in src
    assert "requestInterruption" in src
    assert "worker.wait(" in src


def test_stats_dialog_themes_tabs_and_dock_titles():
    import inspect
    from labelforge.ui.mixins.stats import StatsMixin
    src = inspect.getsource(StatsMixin._show_label_stats)
    assert "QTabBar::tab" in src
    assert "QDockWidget::title" in src
    assert "QTabWidget::pane" in src


class _FakeSignal:
    def __init__(self):
        self._fns = []
    def connect(self, fn):
        self._fns.append(fn)
    def emit(self, payload):
        for fn in self._fns:
            fn(payload)


def _run_health_worker(image_paths=(), memory_boxes=None, canvas_items=None):
    from labelforge.engine.dataset_health import DatasetHealthWorker
    w = DatasetHealthWorker(tuple(image_paths), memory_boxes or {},
                            canvas_items or {})
    w.health_ready = _FakeSignal()
    w.isInterruptionRequested = lambda: False
    captured = {}
    w.health_ready.connect(lambda p: captured.update(p))
    w.run()
    return captured


def test_collect_paste_geometry_carries_image_index():
    from labelforge.engine.dataset_health import collect_paste_geometry

    class Rect:
        def __init__(self, w, h, x=0, y=0):
            self._w, self._h, self._x, self._y = w, h, x, y
        def width(self): return self._w
        def height(self): return self._h
        def x(self): return self._x
        def y(self): return self._y

    items = {3: [(None, Rect(10, 10), "a")], 7: [(None, Rect(20, 20), "b")]}
    out = collect_paste_geometry(items)
    assert sorted(b['image_index'] for b in out) == [3, 7]


def test_worker_payload_has_annot_and_paste_sections():
    captured = _run_health_worker()
    assert 'annot' in captured
    assert 'paste' in captured
    assert 'stats' in captured['annot']
    assert 'advice' in captured['annot']
    assert 'stats' in captured['paste']
    assert 'advice' in captured['paste']
    assert captured['images_scanned'] == 0


def test_worker_paste_section_empty_without_paste():
    captured = _run_health_worker()
    assert captured['paste']['stats'] == {}
    assert captured['paste']['advice'] == []


def test_fmt_num_formats_thousands_and_millions():
    from labelforge.ui.widgets.health_charts import _fmt_num
    assert _fmt_num(0) == '0'
    assert _fmt_num(999) == '999'
    assert _fmt_num(1200) == '1.2k'
    assert _fmt_num(2500000) == '2.5M'


def test_bar_label_formats_value_and_percent():
    from labelforge.ui.widgets.health_charts import _bar_label
    assert _bar_label(3, 4) == '3 (75%)'
    assert _bar_label(1200, 2400) == '1.2k (50%)'
    assert _bar_label(0, 0) == '0 (0%)'


def test_chart_accepts_colors_and_axis_labels():
    from labelforge.ui.widgets.health_charts import HealthBarChart
    chart = HealthBarChart()
    chart.set_data([{'label': 'a', 'value': 3, 'color': '#FF0000'},
                    {'label': 'b', 'value': 1}])
    chart.paintEvent(None)
    chart.set_histogram([0.0, 0.5, 1.0], [4, 6],
                        xlabel='面积', ylabel='框数')
    chart.paintEvent(None)


def test_hist_header_shows_total_not_peak():
    """顶部标签显示柱高总和：IoU 图口径是配对数，显示峰值会误以为框数不一致。"""
    from labelforge.ui.widgets.health_charts import _hist_header
    assert _hist_header('框数', [5, 20, 10, 2]) == '框数: 37'
    assert _hist_header('框对数', [30, 8, 1]) == '框对数: 39'
    assert _hist_header('框数', []) == '框数: 0'


# Qt 子进程在解释器退出阶段析构顺序不定，偶发挂死或崩溃（已观察到 paint
# 测试在整套件里从 0.7s 变成 >120s 超时）。断言全部通过、标记打印之后
# 立即 flush + os._exit(0)，绕开不可控的析构阶段。os._exit 不冲刷缓冲，
# 必须先 flush，否则 pytest 收不到 OK 标记会误报失败。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def test_chart_paint_offscreen_real_qt(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtWidgets import QApplication
from labelforge.ui.widgets.health_charts import HealthBarChart, _fmt_num
app = QApplication.instance() or QApplication([])
assert _fmt_num(1200) == "1.2k"
c = HealthBarChart()
c.resize(400, 140)
c.set_data([{"label": "cat", "value": 30, "color": "#FF0000"},
            {"label": "dog", "value": 10}])
c.paintEvent(None)
c.set_histogram([0.0, 0.5, 1.0], [4, 6], xlabel="面积", ylabel="框数")
c.paintEvent(None)
c.set_placeholder("暂无数据")
c.paintEvent(None)
from PySide6.QtGui import QImage
from PySide6.QtCore import Qt

img = QImage(200, 60, QImage.Format_ARGB32)
img.fill(Qt.white)
c2 = HealthBarChart()
c2.resize(200, 60)
c2.set_data([{'label': 'a', 'value': 10, 'color': '#FF0000'}])
c2.render(img)
pix = img.pixelColor(90, 20)
assert pix.red() > 200 and pix.green() < 80, (pix.red(), pix.green(), pix.blue())

img2 = QImage(200, 140, QImage.Format_ARGB32)
img2.fill(Qt.white)
c3 = HealthBarChart()
c3.resize(200, 140)
c3.set_histogram([0.0, 1.0], [5, 1])
c3.render(img2)
dark = any(img2.pixelColor(x, y).lightness() < 128
           for x in range(0, 200, 4) for y in range(0, 48, 2))
assert dark, "peak bar count label missing in top band"

img3 = QImage(200, 140, QImage.Format_ARGB32)
img3.fill(Qt.white)
c4 = HealthBarChart()
c4.resize(200, 140)
c4.set_data([{"label": "a", "value": 10, "color": "#FF0000"},
             {"label": "b", "value": 5, "color": "#00FF00"}], horizontal=False)
c4.render(img3)
v1 = img3.pixelColor(50, 130)
v2 = img3.pixelColor(150, 130)
assert v1.red() > 200 and v1.green() < 80, v1.getRgb()
assert v2.green() > 200 and v2.red() < 80, v2.getRgb()
print("OK")
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    font_dir = next((p for p in (r"C:\Windows\Fonts", "/usr/share/fonts",
                                 "/System/Library/Fonts") if Path(p).is_dir()), None)
    if font_dir:
        env["QT_QPA_FONTDIR"] = font_dir
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_collect_shape_geometry_empty_memory_slot_falls_back_to_disk(tmp_path):
    """已加载但内存为空的图（应用里每张图都有空槽）必须回读磁盘。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "mem_empty", [_box("cat", w=100, h=50)])
    out = collect_shape_geometry([img], memory_boxes={0: []})
    assert [b["label"] for b in out["boxes"]] == ["cat"]
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_routes_paste_shapes(tmp_path):
    """磁盘 flags.paste 的 shape 进 paste_boxes，不进 boxes。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "flagged", [
        _box("cat", w=100, h=50),
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
    ])
    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["cat"]
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["paste_boxes"][0]["image_index"] == 0


def test_collect_shape_geometry_memory_paste_boxes_routed(tmp_path):
    """内存框带 is_paste（新会话从磁盘载入的贴图）也要进 paste_boxes。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "mem_paste", [_box("cat", w=100, h=50)])
    memory = {0: [
        {"label": "cat", "x": 0, "y": 0, "width": 100, "height": 50},
        {"label": "logo", "x": 5, "y": 5, "width": 10, "height": 10,
         "is_paste": True},
    ]}
    out = collect_shape_geometry([img], memory_boxes=memory)
    assert [b["label"] for b in out["boxes"]] == ["cat"]
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]


def test_collect_shape_geometry_reads_output_dir_paste_sidecar(tmp_path):
    """贴图只写在输出目录 sidecar：原始 sidecar 无贴图也必须采到。

    输出 sidecar 的非贴图框是保存时的副本，不能作为来源（见 FIX B）。
    """
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "out_paste", [_box("cat", w=100, h=50)])
    _write_output_sidecar(tmp_path, img, [
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
        _box("cat2", x=200, y=200, w=20, h=20),
    ])
    out = collect_shape_geometry([img])
    assert sorted(b["label"] for b in out["boxes"]) == ["cat"]
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_reads_prefixed_output_sidecar_from_real_save(tmp_path):
    """默认前缀开启时真实保存为 paste_cat.json，统计必须看到贴图。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    from labelforge.core.utils import PathUtils
    img = _image_in_subdir(tmp_path)
    _real_save(img)

    sidecar = os.path.join(PathUtils.get_output_dir(img), "paste_cat.json")
    assert os.path.isfile(sidecar), "真实保存路径变了，测试前提失效"

    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["paste_boxes"][0]["image_index"] == 0
    assert out["boxes"] == []
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_reads_custom_prefix_output_sidecar(tmp_path):
    """前缀是用户自由输入的，不能硬编码 paste_。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _image_in_subdir(tmp_path)
    _real_save(img, prefix="sess2")

    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]


def test_collect_shape_geometry_reads_output_sidecar_without_prefix(tmp_path):
    """关闭前缀时保存为 cat.json，同样要能采到。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _image_in_subdir(tmp_path)
    _real_save(img, prefix_checked=False)

    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]


def test_collect_shape_geometry_scans_image_with_only_prefixed_output_sidecar(tmp_path):
    """原图未标注、只有带前缀的输出 sidecar 时也要计入扫描数。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    from labelforge.core.utils import PathUtils
    img = _image_in_subdir(tmp_path)
    _real_save(img)
    assert not os.path.exists(os.path.splitext(img)[0] + ".json")
    assert os.path.isdir(PathUtils.get_output_dir(img))

    out = collect_shape_geometry([img])
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_does_not_resurrect_deleted_output_boxes(tmp_path):
    """已加载图以内存为准：输出 sidecar 的非贴图框不能复活。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "resurrect", [])
    _write_output_sidecar(tmp_path, img, [
        _box("cat", w=100, h=50),
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
    ])
    out = collect_shape_geometry([img], memory_boxes={0: []})
    assert out["boxes"] == []
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_output_non_paste_not_counted_when_original_missing(tmp_path):
    """原目录无 sidecar 时，输出 sidecar 的非贴图框也不能单独算数。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = tmp_path / "only_out2.png"
    img.write_bytes(b"x")
    _write_output_sidecar(tmp_path, str(img), [
        _box("cat2", x=200, y=200, w=20, h=20),
    ])
    out = collect_shape_geometry([str(img)])
    assert out["boxes"] == []
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_counts_memory_box_and_output_paste_once(tmp_path):
    """内存检测框与输出 sidecar 的同几何副本只计一次，贴图另计。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "move_case", [])
    shape = _box("cat", w=100, h=50)
    _write_output_sidecar(tmp_path, img, [
        shape,
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
    ])
    memory = {0: [dict(shape)]}
    out = collect_shape_geometry([img], memory_boxes=memory)
    assert [b["label"] for b in out["boxes"]] == ["cat"]
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]


def test_collect_shape_geometry_dedupes_paste_across_memory_and_output(tmp_path):
    """内存贴图（is_paste）与磁盘贴图（flags.paste）同几何只能计一次。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "paste_dup", [])
    _write_output_sidecar(tmp_path, img, [
        {"label": "logo", "points": [[5, 5], [15, 5], [15, 15], [5, 15]],
         "shape_type": "rectangle", "flags": {"paste": True}},
    ])
    memory = {0: [{"label": "logo", "x": 5, "y": 5, "width": 10, "height": 10,
                   "is_paste": True}]}
    out = collect_shape_geometry([img], memory_boxes=memory)
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["boxes"] == []


def test_close_health_worker_disconnects_signal_before_drop():
    """超时未停的 worker 不能再往已销毁的弹窗发信号：丢引用前必须断开。"""
    from labelforge.ui.mixins.stats import StatsMixin

    class Signal:
        def __init__(self):
            self.disconnected = 0

        def disconnect(self):
            self.disconnected += 1

    class Worker:
        def __init__(self):
            self.health_ready = Signal()
            self.interrupted = 0
            self.waited = []

        def isRunning(self):
            return True

        def requestInterruption(self):
            self.interrupted += 1

        def wait(self, timeout=0):
            self.waited.append(timeout)
            return False

    editor = type("Editor", (StatsMixin,), {})()
    worker = Worker()
    editor._health_worker = worker

    editor._close_health_worker()

    assert worker.interrupted == 1
    assert worker.waited and worker.waited[0] == 3000
    assert worker.health_ready.disconnected == 1
    assert editor._health_worker is None


def test_collect_shape_geometry_abandons_pending_reads_on_interrupt(tmp_path, monkeypatch):
    """健康扫描中断后不能等所有排队读盘 future 完成。"""
    import time
    from labelforge.engine import dataset_health

    def _slow_read(json_path):
        time.sleep(0.5)
        return []

    monkeypatch.setattr(dataset_health, "_read_json_shapes", _slow_read)
    monkeypatch.setattr(dataset_health, "_read_output_sidecar", lambda path: None)
    paths = [str(tmp_path / f"h{i}.png") for i in range(40)]

    t0 = time.perf_counter()
    out = dataset_health.collect_shape_geometry(
        paths, is_interrupted=lambda: True)
    elapsed = time.perf_counter() - t0

    assert out["boxes"] == []
    # 8 workers 跑完 40 个 0.5s 任务要 2.5s；中断后只等已在跑的
    assert elapsed < 1.2, elapsed


def test_collect_shape_geometry_output_sidecar_only(tmp_path):
    """只有输出目录 sidecar（原图未标注）时，贴图要采到，非贴图不算。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = tmp_path / "only_out.png"
    img.write_bytes(b"x")
    _write_output_sidecar(tmp_path, str(img), [
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
        _box("cat2", x=200, y=200, w=20, h=20),
    ])
    out = collect_shape_geometry([str(img)])
    assert out["boxes"] == []
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_dedupes_shapes_in_both_sidecars(tmp_path):
    """save_json 把当前图检测框同时写进两份 sidecar：同一形状只能计一次。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    shape = _box("cat", w=100, h=50)
    img = _write(tmp_path, "dup_side", [shape])
    _write_output_sidecar(tmp_path, img, [
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
        shape,
    ])
    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["cat"]
    assert [b["label"] for b in out["paste_boxes"]] == ["logo"]


def test_collect_shape_geometry_skips_malformed_shape(tmp_path):
    """坏 shape（点数为 1 / 非数值坐标）不能中断整库扫描。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "malformed", [
        {"label": "bad", "points": [[5]]},
        {"label": "bad2", "points": [["x", "y"], ["z", "w"]]},
        _box("ok", w=10, h=10),
    ])
    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["ok"]
    assert out["images_scanned"] == 1


def test_collect_shape_geometry_skips_non_finite_coords(tmp_path):
    """NaN/Inf 坐标（json 允许 NaN/Infinity 字面量）跳过，不能整库分析失败。"""
    from labelforge.engine.dataset_health import collect_shape_geometry
    img = _write(tmp_path, "nonfinite", [
        {"label": "nan", "points": [[float("nan"), 0], [10, 0],
                                    [10, 10], [float("nan"), 10]]},
        {"label": "inf", "x": float("inf"), "y": 0, "width": 10, "height": 10},
        _box("ok", w=10, h=10),
    ])
    out = collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["ok"]
    assert out["images_scanned"] == 1


def test_worker_survives_nan_in_sidecar(tmp_path):
    """含 NaN 的 sidecar 不能让 worker 落到 error 分支。"""
    img = _write(tmp_path, "nan_worker", [
        {"label": "nan", "points": [[float("nan"), 0], [10, 0],
                                    [10, 10], [float("nan"), 10]]},
        _box("ok", w=10, h=10),
    ])
    captured = _run_health_worker(image_paths=[img])
    assert not captured.get("error")
    assert captured["annot"]["stats"]["class_dist"] == [
        {"label": "ok", "count": 1}]


def test_compute_health_survives_non_finite_values():
    """聚合入口的防御：坏 area/aspect/IoU 值跳过，好框照常计数。"""
    from labelforge.engine.dataset_health import compute_health
    boxes = [
        {"label": "a", "x": 0, "y": 0, "width": float("nan"),
         "height": 10, "area": float("nan"), "aspect": float("inf"),
         "image_index": 0},
        {"label": "a", "x": 0, "y": 0, "width": 10, "height": 10,
         "area": 100.0, "aspect": 1.0, "image_index": 0},
    ]
    stats = compute_health(boxes)
    assert stats["summary"]["total_boxes"] == 2
    assert sum(stats["size_hist"]["counts"]) == 1
    assert sum(stats["aspect_hist"]["counts"]) == 1


def test_merge_shapes_keeps_primary_duplicates(tmp_path):
    """primary（原始 sidecar）内部重复框必须保留：IoU 直方图靠它暴露重复。"""
    from labelforge.engine.dataset_health import _merge_shapes
    shape = _box("cat", w=100, h=50)
    assert len(_merge_shapes([shape, dict(shape)], None)) == 2


def test_merge_shapes_dedupes_only_secondary():
    """secondary（输出 sidecar）与 primary 同几何只追加一次，primary 全保留。"""
    from labelforge.engine.dataset_health import _merge_shapes
    shape = _box("cat", w=100, h=50)
    merged = _merge_shapes([shape], [dict(shape), _box("dog", w=1, h=1)])
    assert [s["label"] for s in merged] == ["cat", "dog"]


def test_health_advice_items_are_i18n_keys():
    """建议项为 {key, params}，zh 键为 identity，en 键可 format 渲染。"""
    from labelforge.engine.dataset_health import health_advice
    from labelforge.ui.i18n import _strings
    stats = {
        'class_dist': [{'label': '类别A', 'count': 20},
                       {'label': 'b', 'count': 980}],
        'size_hist': {'edges': [0, 1], 'counts': [1000]},
        'aspect_hist': {'edges': [0, 1], 'counts': [1000]},
        'iou_hist': {'edges': [0.0, 1.0], 'counts': [1000]},
        'summary': {'total_boxes': 1000, 'class_count': 2},
    }
    advice = health_advice(stats)
    rare = next(a for a in advice if a['params'].get('label') == '类别A')
    for lang in ('zh', 'en'):
        assert rare['key'] in _strings[lang], rare['key']
        rendered = _strings[lang][rare['key']].format(**rare['params'])
        assert '类别A' in rendered, rendered
    assert rare['key'] == _strings['zh'][rare['key']]


def test_iou_hist_payload_has_no_dead_skipped_images():
    from labelforge.engine.dataset_health import compute_health
    stats = compute_health([_geo("a", 10, 10)])
    assert 'skipped_images' not in stats['iou_hist']


def test_collect_shape_geometry_survives_geometry_exception(tmp_path, monkeypatch):
    """_geometry_of 抛异常时跳过该 shape，不能中断整库扫描。"""
    from labelforge.engine import dataset_health
    img = _write(tmp_path, "geo_raise", [
        _box("bad", w=10, h=10),
        _box("ok", w=20, h=20),
    ])
    real_geometry = dataset_health._geometry_of
    calls = {"n": 0}

    def _flaky_geometry(label, rect):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("malformed")
        return real_geometry(label, rect)

    monkeypatch.setattr(dataset_health, "_geometry_of", _flaky_geometry)
    out = dataset_health.collect_shape_geometry([img])
    assert [b["label"] for b in out["boxes"]] == ["ok"]


def test_merge_paste_geometry_prefers_memory_and_dedupes():
    from labelforge.engine.dataset_health import merge_paste_geometry
    disk = [{"label": "old", "x": 1.0, "y": 2.0, "width": 10.0, "height": 10.0,
             "area": 100.0, "aspect": 1.0, "image_index": 0}]
    memory = [{"label": "new", "x": 1.0, "y": 2.0, "width": 10.0, "height": 10.0,
               "area": 100.0, "aspect": 1.0, "image_index": 0},
              {"label": "extra", "x": 50.0, "y": 50.0, "width": 5.0, "height": 5.0,
               "area": 25.0, "aspect": 1.0, "image_index": 1}]
    merged = merge_paste_geometry(disk, memory)
    assert sorted(b["label"] for b in merged) == ["extra", "new"]


def test_worker_paste_section_includes_disk_paste(tmp_path):
    img = _write(tmp_path, "disk_paste", [
        _box("cat", w=100, h=50),
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
    ])
    captured = _run_health_worker(image_paths=[img])
    assert captured["paste"]["stats"]["class_dist"] == [
        {"label": "logo", "count": 1}]
    assert captured["annot"]["stats"]["class_dist"] == [
        {"label": "cat", "count": 1}]


def test_worker_paste_section_includes_output_dir_paste(tmp_path):
    """worker 也必须读到输出目录 sidecar 里的贴图。"""
    img = _write(tmp_path, "worker_out", [_box("cat", w=100, h=50)])
    _write_output_sidecar(tmp_path, img, [
        dict(_box("logo", w=10, h=10), flags={"paste": True}),
    ])
    captured = _run_health_worker(image_paths=[img])
    assert captured["paste"]["stats"]["class_dist"] == [
        {"label": "logo", "count": 1}]


def test_worker_paste_section_computed_with_paste():
    class Rect:
        def __init__(self, w, h, x=0, y=0):
            self._w, self._h, self._x, self._y = w, h, x, y
        def width(self): return self._w
        def height(self): return self._h
        def x(self): return self._x
        def y(self): return self._y

    captured = _run_health_worker(
        canvas_items={0: [(None, Rect(30, 20), "a"), (None, Rect(30, 20, 40, 0), "a")]})
    paste_stats = captured['paste']['stats']
    assert paste_stats['class_dist'] == [{'label': 'a', 'count': 2}]
    assert paste_stats['summary']['total_boxes'] == 2
    assert isinstance(captured['paste']['advice'], list)


def test_i18n_stats_redesign_terms():
    from labelforge.ui.i18n import _strings
    for key in ("数据源：背景图标签（切换标签页）",
                "数据源：贴图标签（切换标签页）",
                "各类别占比应接近均衡；长尾类别建议多合成",
                "框尺寸应覆盖多种尺度，避免集中于单一范围",
                "长宽比多样化更贴近真实场景",
                "高 IoU 区间框多说明重复标注偏多",
                "面积", "长宽比", "框数", "面积 (px²)", "框对数",
                "背景图标签", "贴图标签"):
        assert key in _strings["zh"], key
        assert key in _strings["en"], key


def test_stats_health_section_supports_source_switch():
    import inspect
    from labelforge.ui.mixins.stats import StatsMixin
    src = inspect.getsource(StatsMixin)
    assert "_set_health_source" in src
    assert "'annot'" in src and "'paste'" in src
    assert "currentChanged" in src
    assert "_health_source" in src
    assert "_health_payload" in src
    assert "QDockWidget" in src


def test_health_source_switch_routes_payload_real_qt(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QDialog, QVBoxLayout
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeEditor(StatsMixin, QObject):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {}
        self.canvas_items_dict = {}
        self.canvas_items = []
        self.current_background_index = -1
        self._health_worker = None

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor()
dialog = QDialog()
layout = QVBoxLayout(dialog)
editor._build_health_section(dialog, layout)
# 无事件循环，queued signal 不会投递；先停掉 worker，再同步注入 payload
editor._close_health_worker()

payload = {
    "annot": {
        "stats": {
            "class_dist": [{"label": "annot_cls", "count": 9}],
            "size_hist": {"edges": [0, 1], "counts": [9]},
            "aspect_hist": {"edges": [0, 1], "counts": [9]},
            "iou_hist": {"edges": [0.0, 1.0], "counts": [9]},
        },
        "advice": [{"key": "类别 {label} 样本偏少（{count} 个，占比 {share}%），建议多合成",
                    "params": {"label": "annot_cls", "count": 3, "share": "4"}}],
    },
    "paste": {
        "stats": {
            "class_dist": [{"label": "paste_cls", "count": 4}],
            "size_hist": {"edges": [0, 1], "counts": [4]},
            "aspect_hist": {"edges": [0, 1], "counts": [4]},
            "iou_hist": {"edges": [0.0, 1.0], "counts": [4]},
        },
        "advice": ["paste advice"],  # 旧字符串格式兼容
    },
    "images_scanned": 1,
}
charts = dialog._health_charts

dialog._health_payload = payload
dialog._set_health_source("annot")
assert charts["class"]._items[0]["label"] == "annot_cls", charts["class"]._items
assert charts["class"]._items[0]["color"] == "#123456"
assert charts["size"]._xlabel != ""
assert charts["size"]._counts == [9], charts["size"]._counts
assert "annot_cls" in dialog._health_advice_label.text()  # 字典格式经 tr().format() 渲染

dialog._set_health_source("paste")
assert charts["class"]._items[0]["label"] == "paste_cls", charts["class"]._items
assert charts["class"]._items[0]["color"] == "#123456"
assert charts["size"]._counts == [4], charts["size"]._counts
assert dialog._health_source == "paste"
assert "paste advice" in dialog._health_advice_label.text()

# paste 空 → 全部显示占位符
empty = dict(payload)
empty["paste"] = {"stats": {}, "advice": []}
dialog._health_payload = empty
dialog._set_health_source("paste")
assert charts["class"]._mode == "empty", charts["class"]._mode
assert charts["class"]._placeholder == "暂无贴图数据", charts["class"]._placeholder

# error → 全部分析失败
dialog._health_payload = {"error": True}
dialog._set_health_source("annot")
assert charts["class"]._placeholder == "分析失败", charts["class"]._placeholder
assert charts["iou"]._placeholder == "分析失败", charts["iou"]._placeholder

# 健康区不折叠：标题为普通 QLabel，数据源与建议始终可见
dialog.show()
app.processEvents()
from PySide6.QtWidgets import QPushButton, QLabel
assert dialog._health_source_label.text() != ""
assert dialog._health_advice_label.text() != ""
assert not dialog._health_source_label.isHidden()
assert not dialog._health_advice_label.isHidden()
# 标题不再是可点击折叠按钮
assert not any("数据集健康" in b.text()
               for b in dialog.findChildren(QPushButton))
assert any("数据集健康" in lbl.text()
           for lbl in dialog.findChildren(QLabel))

editor._close_health_worker()
print("OK")
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_stats_dialog_min_width_810():
    import inspect
    from labelforge.ui.main_window import ImageEditor
    src = inspect.getsource(ImageEditor._show_label_stats)
    assert "dialog.setMinimumSize(1200, 640)" in src


def test_stats_dialog_table_wiring_and_worker_payload_real_qt(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QDialog, QWidget
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new):
        return False


class FakeEditor(StatsMixin, QWidget):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {
            0: [{"label": "bg_cls", "x": 0, "y": 0, "width": 10, "height": 10}]}
        self.canvas_items_dict = {0: [(None, None, "paste_cls")]}
        self.canvas_items = []
        self.current_background_index = 0
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = "x.png"
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor()
# 无事件循环：拦截 exec_ 以捕获真实 _StatsDialog
captured = {}


def _capture_exec(self):
    captured["dialog"] = self
    return 0


QDialog.exec_ = _capture_exec
editor._show_label_stats()
dialog = captured["dialog"]
editor._close_health_worker()

bg_table = dialog._bg_table
paste_table = dialog._paste_table
charts = dialog._health_charts

# Gap 1: 真实接线（切换标签页 → 切换数据源）
tabs = dialog._tabs
tabs.setCurrentIndex(0)
assert dialog._health_source == "annot", dialog._health_source
tabs.setCurrentIndex(1)
assert dialog._health_source == "paste", dialog._health_source
tabs.setCurrentIndex(0)
assert dialog._health_source == "annot", dialog._health_source

payload = {
    "annot": {
        "stats": {
            "class_dist": [{"label": "annot_cls", "count": 9}],
            "size_hist": {"edges": [0, 1], "counts": [9]},
            "aspect_hist": {"edges": [0, 1], "counts": [9]},
            "iou_hist": {"edges": [0.0, 1.0], "counts": [9]},
        },
        "advice": ["annot advice"],
    },
    "paste": {
        "stats": {
            "class_dist": [{"label": "paste_cls", "count": 4}],
            "size_hist": {"edges": [0, 1], "counts": [4]},
            "aspect_hist": {"edges": [0, 1], "counts": [4]},
            "iou_hist": {"edges": [0.0, 1.0], "counts": [4]},
        },
        "advice": ["paste advice"],
    },
    "images_scanned": 1,
}

# Gap 2: worker payload 投递（存储 + 渲染）
payload2 = {
    "annot": payload["annot"],
    "paste": {"stats": {
        "class_dist": [{"label": "from_worker", "count": 7}],
        "size_hist": {"edges": [0, 1], "counts": [7]},
        "aspect_hist": {"edges": [0, 1], "counts": [7]},
        "iou_hist": {"edges": [0.0, 1.0], "counts": [7]},
    }, "advice": ["w"]},
    "images_scanned": 2,
}
dialog._set_health_source("paste")
dialog._on_health_payload(payload2)
assert dialog._health_payload is payload2
assert charts["class"]._items[0]["label"] == "from_worker", charts["class"]._items
assert dialog._health_source == "paste"

# payload 到达前已选 paste：存储的数据源优先
dialog._health_payload = None
dialog._set_health_source("paste")
dialog._on_health_payload(payload)
assert dialog._health_payload is payload
assert charts["class"]._items[0]["label"] == "paste_cls", charts["class"]._items

editor._close_health_worker()
print("OK")
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_stats_total_label_payload_paste_count_real_qt(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QDialog
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new, rewrite_disk=False):
        return False


class FakeEditor(StatsMixin, QDialog):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {
            0: [{"label": "bg_cls", "x": 0, "y": 0, "width": 10, "height": 10}]}
        self.canvas_items_dict = {}
        self.canvas_items = []
        self.current_background_index = 0
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = "x.png"
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor()
captured = {}


def _capture_exec(self):
    captured["dialog"] = self
    return 0


QDialog.exec_ = _capture_exec
editor._show_label_stats()
dialog = captured["dialog"]
editor._close_health_worker()

# 会话内存无贴图 → 初始总计贴图数为 0
label = dialog._total_label
bg_part, paste_part = label.text().split("|")
assert "0" in paste_part, label.text()

# worker payload 携带整库贴图统计（磁盘上有 7 个）→ 总计贴图数须为 7
payload = {
    "annot": {"stats": {"class_dist": [{"label": "bg_cls", "count": 9}],
                        "summary": {"total_boxes": 9}},
              "advice": ["a"]},
    "paste": {"stats": {"class_dist": [{"label": "disk_logo", "count": 7}],
                        "summary": {"total_boxes": 7}},
              "advice": ["p"]},
    "images_scanned": 1,
}
dialog._on_health_payload(payload)
text = dialog._total_label.text()
bg_part, paste_part = text.split("|")
assert "7" in paste_part, text
# 背景计数必须与背景表同源（此处磁盘无 sidecar → 表为空 → 总计背景为 0），
# payload 的 annot 计数（9）不得覆盖它，否则总计与表行求和不一致。
bg_table = dialog._bg_table
table_sum = sum(int(bg_table.item(r, 1).text())
                for r in range(bg_table.rowCount()))
assert table_sum == 0, table_sum
assert "0" in bg_part and "9" not in bg_part, text
assert dialog._paste_table.item(0, 0).text() == "disk_logo"

# payload 贴图段为空 → 保留会话内存贴图计数，而不是盲写 0
editor.canvas_items = [(None, None, "mem_paste")]
empty_payload = {"annot": {"stats": {"summary": {"total_boxes": 9}}},
                 "paste": {"stats": {}, "advice": []}, "images_scanned": 1}
dialog._on_health_payload(empty_payload)
paste_part = dialog._total_label.text().split("|")[1]
assert " 1 " in paste_part, dialog._total_label.text()

editor._close_health_worker()
print("OK")
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_stats_total_bg_count_matches_bg_table_with_disk_paste_real_qt(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    sidecar = tmp_path / "img.json"
    sidecar.write_text(json.dumps({
        "shapes": [
            {"label": "car", "shape_type": "rectangle",
             "points": [[0, 0], [10, 10]]},
            {"label": "logo", "shape_type": "rectangle",
             "points": [[20, 20], [30, 30]], "flags": {"paste": True}},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    script = '''
import json
from PySide6.QtWidgets import QApplication, QDialog
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new, rewrite_disk=False):
        return False


class FakeEditor(StatsMixin, QDialog):
    def __init__(self, image_path):
        super().__init__()
        self.background_images = [image_path]
        self.detection_boxes_dict = {}
        self.canvas_items_dict = {}
        self.canvas_items = []
        self.current_background_index = -1
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = image_path
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor(IMAGE_PATH)
captured = {}


def _capture_exec(self):
    captured["dialog"] = self
    # ⑦ 起后台扫描之后，等它落地再读表：统计现在是「先弹窗、后填充」。
    from PySide6.QtCore import QEventLoop, QTimer
    loop = QEventLoop()
    tm = QTimer()
    tm.timeout.connect(
        lambda: (getattr(self, "_scan_applied", False) and loop.quit()))
    tm.start(10)
    QTimer.singleShot(20000, loop.quit)
    loop.exec_()
    tm.stop()
    return 0


QDialog.exec_ = _capture_exec
editor._show_label_stats()
dialog = captured["dialog"]
editor._close_health_worker()

bg_table = dialog._bg_table
# 背景表按 sidecar 全量计数：1 矩形 + 1 贴图 = 2
table_sum = sum(int(bg_table.item(r, 1).text())
                for r in range(bg_table.rowCount()))
assert table_sum == 2, table_sum

# payload 的 annot 统计排除贴图 → total_boxes=1；总计背景计数必须仍等于表求和 2
payload = {
    "annot": {"stats": {"class_dist": [{"label": "car", "count": 1}],
                        "summary": {"total_boxes": 1}},
              "advice": ["a"]},
    "paste": {"stats": {"class_dist": [{"label": "logo", "count": 1}],
                        "summary": {"total_boxes": 1}},
              "advice": ["p"]},
    "images_scanned": 1,
}
dialog._on_health_payload(payload)
bg_part = dialog._total_label.text().split("|")[0]
assert " 2 " in bg_part, dialog._total_label.text()

editor._close_health_worker()
print("OK")
'''.replace("IMAGE_PATH", repr(str(img)))
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_stats_dialog_tabs_and_docks_real_qt(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = '''
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget
from labelforge.ui.mixins.stats import StatsMixin

app = QApplication.instance() or QApplication([])


class FakeLabelManager:
    def rename_detection_label(self, old, new):
        return False

    def rename_paste_label(self, old, new, rewrite_disk=False):
        return False


class FakeEditor(StatsMixin, QDialog):
    def __init__(self):
        super().__init__()
        self.background_images = ["x.png"]
        self.detection_boxes_dict = {
            0: [{"label": "bg_cls", "x": 0, "y": 0, "width": 10, "height": 10}]}
        self.canvas_items_dict = {0: [(None, None, "paste_cls")]}
        self.canvas_items = []
        self.current_background_index = 0
        self._health_worker = None
        self._cached_bg_label_stats = None
        self._memory_background_path = "x.png"
        self.label_color_map = {}
        self.global_labels = set()
        self.background_dataset_labels = set()
        self.label_manager = FakeLabelManager()
        self._dataset_stats_dirty = False
        self._background_label_scan_completed = True

    def get_label_color(self, label):
        return "#123456"


editor = FakeEditor()
captured = {}


def _capture_exec(self):
    captured["dialog"] = self
    return 0


QDialog.exec_ = _capture_exec
editor._show_label_stats()
dialog = captured["dialog"]
editor._close_health_worker()

# 单表位置：一个 QTabWidget，两个页签
tabs = dialog._tabs
assert tabs.count() == 2, tabs.count()
assert dialog._bg_table is not None
assert dialog._paste_table is not None

# 切换页签 → 健康数据源联动
tabs.setCurrentIndex(1)
assert dialog._health_source == "paste", dialog._health_source
tabs.setCurrentIndex(0)
assert dialog._health_source == "annot", dialog._health_source

# 4 个统计面板为可拖拽 dock
docks = dialog._health_docks
assert sorted(docks) == ["aspect", "class", "iou", "size"], sorted(docks)
for dock in docks.values():
    assert isinstance(dock, QDockWidget)
    assert dock.features() & QDockWidget.DockWidgetMovable
    assert not (dock.features() & QDockWidget.DockWidgetFloatable)

# 右侧图表区 : 左侧表格区 ≈ 2:1（showEvent 延时到 dock 布局完成后应用）
dialog.show()
from PySide6.QtTest import QTest
QTest.qWait(120)
for _ in range(5):
    app.processEvents()
docks_w = docks["class"].width() + docks["iou"].width()
table_w = dialog._dock_host.centralWidget().width()
assert docks_w > 0 and table_w > 0, (docks_w, table_w)
assert 1.6 <= docks_w / table_w <= 2.4, (docks_w, table_w)

# 表格在左、图表在右（同一坐标系：host）
host = dialog._dock_host
table_x = dialog._tabs.mapTo(host, dialog._tabs.rect().topLeft()).x()
assert table_x < docks["class"].x(), (table_x, docks["class"].x())

# 同一行 dock 等宽 1:1
top_row = sorted([d for d in docks.values() if d.y() == 0], key=lambda d: d.x())
assert len(top_row) == 2, [d.objectName() for d in top_row]
assert abs(top_row[0].width() - top_row[1].width()) <= 2, (
    top_row[0].width(), top_row[1].width())

# 浮动 dock 的独立顶层窗口：原生标题栏需随主题设深色
from labelforge.ui import dwm as _dwm
_dwm_calls = []
_orig_dwm = _dwm.set_titlebar_dark


def _trace_dwm(hwnd, dark, force_refresh=False):
    _dwm_calls.append((hwnd, dark))
    return True


_dwm.set_titlebar_dark = _trace_dwm
try:
    docks["class"].setFloating(True)
    app.processEvents()
    QTest.qWait(150)
    # 只看这个 dock 的调用：弹窗自己有 30ms 的 _sync 定时器也会调
    # set_titlebar_dark(弹窗 hwnd)，负载高时它可能晚于 dock 浮动事件触发，
    # 于是「最后一条」未必是 dock 的。断言该 dock 被同步过即可。
    dock_hwnd = int(docks["class"].winId())
    dock_calls = [c for c in _dwm_calls if c[0] == dock_hwnd]
    assert dock_calls, ("floating dock titlebar not synced", _dwm_calls, dock_hwnd)
    from labelforge.ui.theme import ThemeManager
    expected_dark = ThemeManager.get_mode().value == "dark"
    assert dock_calls[-1][1] is expected_dark, (dock_calls, expected_dark)
finally:
    _dwm.set_titlebar_dark = _orig_dwm
    docks["class"].setFloating(False)
    app.processEvents()

# 单列布局：拖动重排成单列后高度自动 1:1
from PySide6.QtCore import Qt
host.splitDockWidget(docks['class'], docks['size'], Qt.Vertical)
host.splitDockWidget(docks['size'], docks['aspect'], Qt.Vertical)
host.splitDockWidget(docks['aspect'], docks['iou'], Qt.Vertical)
app.processEvents()
QTest.qWait(150)
host.resizeDocks([docks['class'], docks['size']], [300, 80], Qt.Vertical)
app.processEvents()
assert dialog._dock_norm_timer is not None
dialog._dock_norm_timer.start()
QTest.qWait(250)
col = sorted(docks.values(), key=lambda d: d.y())
hs = [d.height() for d in col]
assert max(hs) - min(hs) <= 3, hs

# 整库贴图：payload 到达后贴图表重建（磁盘贴图统计进表）
payload = {
    "annot": {"stats": {"class_dist": [{"label": "bg_cls", "count": 9}]},
              "advice": ["a"]},
    "paste": {"stats": {"class_dist": [{"label": "disk_logo", "count": 7}]},
              "advice": ["p"]},
    "images_scanned": 1,
}
dialog._on_health_payload(payload)
rows = dialog._paste_table.rowCount()
assert rows == 1, rows
assert dialog._paste_table.item(0, 0).text() == "disk_logo"
assert dialog._paste_table.item(0, 1).text() == "7"

editor._close_health_worker()
print("OK")
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}
    result = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=root, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
