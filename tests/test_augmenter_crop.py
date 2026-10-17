"""滑窗裁剪测试：网格公式 + 标注过滤（纯逻辑，conftest mock 下可直接跑）。"""
from labelforge.engine.augmenter.crop import (
    crop_boxes, normalized_fill_rgb, normalized_window, overlap_for_count,
    size_for_count, window_starts,
)


# ---------- 网格公式 ----------

def test_window_starts_exact_fit():
    assert window_starts(100, 50, 0) == [0, 50]


def test_window_starts_covers_axis():
    assert window_starts(100, 100, 0) == [0]


def test_window_starts_last_flush_and_overlap_ge_requested():
    # c=60, ov=10: step=50 → n=2, 均摊后 [0,40]，末窗贴边且实际重叠 20 ≥ 请求 10
    starts = window_starts(100, 60, 10)
    assert starts == [0, 40]
    assert starts[-1] + 60 == 100


def test_window_starts_slack_spread_three():
    # c=35, ov=10: step=25 → n=4, span=65 → [0,22,43,65]
    assert window_starts(100, 35, 10) == [0, 22, 43, 65]


def test_window_starts_step_floor_one_when_overlap_too_big():
    # ov ≥ c 时 step 兜底为 1（调用方另有 ov 钳制，这里保证不产生非法步长）
    assert window_starts(10, 3, 50) == list(range(8))


def test_size_for_count_exact_fit():
    assert size_for_count(100, 3, 20) == 47   # (100+2*20)/3 = 46.67 → 47
    assert size_for_count(100, 1, 20) == 100


def test_size_for_count_clamps():
    assert size_for_count(50, 99, 10) == 11   # 10.4 → 钳到 ov+1


def test_size_for_count_overlap_exceeds_length_returns_length():
    # ov+1 > length 时钳到 length（不再返回 > length 的尺寸）
    assert size_for_count(10, 5, 20) == 10


def test_normalized_window_clamps_to_image():
    assert normalized_window(10, 10, {"w": 4, "h": 4, "overlap": 10}) == (4, 4, 3)
    assert normalized_window(100, 80, {"w": 640, "h": 640, "overlap": 54}) == (100, 80, 54)
    assert normalized_window(20000, 20000, {"w": 640, "h": 640, "overlap": 320}) == (640, 640, 320)
    assert normalized_window(640, 480, {"w": 640, "h": 640, "overlap": 0}) == (640, 480, 0)


# ---------- 重叠反推（锁定「数量+尺寸」时用） ----------

def test_overlap_for_count_exact_coverage():
    # n·size − (n−1)·ov = L → ov = (n·size − L)/(n−1)
    assert overlap_for_count(2000, 3, 800) == 200    # (2400−2000)/2
    assert overlap_for_count(1000, 2, 600) == 200    # (1200−1000)/1


def test_overlap_for_count_floors_to_keep_coverage():
    # (2400−2000)/3 = 133.33 → 向下取整 133；向上取整会盖不满总宽
    assert overlap_for_count(2000, 4, 600) == 133


def test_overlap_for_count_single_window_returns_zero():
    assert overlap_for_count(2000, 1, 800) == 0


def test_overlap_for_count_clamps_infeasible_to_zero():
    # 3×600=1800 < 2000，重叠为 0 也盖不满 → 钳到 0，由调用方按实际数量提示
    assert overlap_for_count(2000, 3, 600) == 0


def test_overlap_for_count_clamps_to_size_minus_one():
    # size ≥ L 时反推值会 ≥ size，钳到 size−1 保证步长 ≥ 1
    assert overlap_for_count(100, 3, 200) == 199


def test_overlap_for_count_roundtrips_to_requested_count():
    # 反推出的重叠代回网格公式，窗口数必须等于请求数
    for L, n, size, want_ov in ((2000, 3, 800, 200), (2000, 4, 600, 133),
                                (1000, 2, 600, 200)):
        assert len(window_starts(L, size, want_ov)) == n, (L, n, size)


# ---------- 边缘填充：等步长网格 + 尺寸不钳 ----------

def test_window_starts_fill_uniform_step_allows_overflow():
    # fill 开启 → 等步长（步长=size−overlap），末窗可越出图像边缘交由引擎补边
    assert window_starts(100, 60, 10, fill=True) == [0, 50]        # 末窗 [50,110] 越界 10
    assert window_starts(100, 35, 10, fill=True) == [0, 25, 50, 75]  # 末窗 [75,110] 越界 10


def test_window_starts_fill_exact_multiple_stays_flush():
    assert window_starts(100, 50, 0, fill=True) == [0, 50]   # 整除时恰好贴边，无需补边


def test_window_starts_fill_window_larger_than_image_single():
    assert window_starts(40, 60, 10, fill=True) == [0]       # 图小于窗 → 单窗，其余全补边


def test_window_starts_without_fill_keeps_edge_flush_grid():
    # 回归：不启用填充时维持旧的「末窗贴边、余量均摊」网格
    assert window_starts(100, 60, 10) == [0, 40]
    assert window_starts(100, 60, 10, fill=False) == [0, 40]


def test_normalized_window_keeps_size_when_fill_enabled():
    # 填充开启：窗口尺寸严格固定，不再钳到图像尺寸内
    assert normalized_window(40, 30, {"w": 60, "h": 60, "overlap": 10,
                                      "fill": True}) == (60, 60, 10)


def test_normalized_window_still_clamps_when_fill_disabled():
    # 填充关闭：完全维持现状（钳到图像内）
    assert normalized_window(40, 30, {"w": 60, "h": 60, "overlap": 10,
                                      "fill": False}) == (40, 30, 10)
    assert normalized_window(40, 30, {"w": 60, "h": 60, "overlap": 10}) == (40, 30, 10)


def test_normalized_window_fill_still_caps_overlap_to_window():
    # 填充开启不放松重叠上限：仍钳到 min(overlap, cw−1, ch−1)
    assert normalized_window(40, 30, {"w": 4, "h": 4, "overlap": 10,
                                      "fill": True}) == (4, 4, 3)


# ---------- 填充色规范化 ----------

def test_normalized_fill_rgb_defaults_to_letterbox_gray():
    # 缺省 / 空值回落到 YOLO letterbox 灰
    assert normalized_fill_rgb(None) == (114, 114, 114)
    assert normalized_fill_rgb({}) == (114, 114, 114)


def test_normalized_fill_rgb_accepts_list_and_tuple():
    assert normalized_fill_rgb({"fill_rgb": [10, 20, 30]}) == (10, 20, 30)
    assert normalized_fill_rgb({"fill_rgb": (200, 100, 50)}) == (200, 100, 50)


def test_normalized_fill_rgb_clamps_out_of_range():
    assert normalized_fill_rgb({"fill_rgb": (-5, 300, 114)}) == (0, 255, 114)


def test_normalized_fill_rgb_falls_back_on_malformed():
    # 模板从磁盘读回可能损坏：长度不对或非数值一律回落默认灰，不得抛异常
    assert normalized_fill_rgb({"fill_rgb": (1, 2)}) == (114, 114, 114)
    assert normalized_fill_rgb({"fill_rgb": "abc"}) == (114, 114, 114)
    assert normalized_fill_rgb({"fill_rgb": (1, 2, "x")}) == (114, 114, 114)


def test_normalized_fill_rgb_keeps_pure_black():
    # (0,0,0) 是合法填充色，不能被当成「未设置」而回落默认灰
    assert normalized_fill_rgb({"fill_rgb": (0, 0, 0)}) == (0, 0, 0)


# ---------- 标注过滤 ----------

def _rect(x, y, w, h, **kw):
    b = {"label": "cat", "shape_type": "rectangle", "x": x, "y": y, "width": w, "height": h}
    b.update(kw)
    return b


def test_crop_rect_clips_and_translates():
    # 窗口 (10,10,20,20)，框 (5,5,20,20)：交集 15×15，比例 0.5625 ≥ 0.3 → 保留
    out = crop_boxes([_rect(5, 5, 20, 20)], 10, 10, 20, 20, 0.3)
    assert len(out) == 1
    assert (out[0]["x"], out[0]["y"], out[0]["width"], out[0]["height"]) == (0, 0, 15, 15)
    assert out[0]["label"] == "cat"


def test_crop_rect_boundary_ratio_kept():
    # 交集 3×100=300，面积 1000，比例恰为 0.3 → 保留（>= 阈值）
    out = crop_boxes([_rect(0, 0, 10, 100)], 0, 0, 3, 100, 0.3)
    assert len(out) == 1


def test_crop_rect_below_threshold_dropped():
    out = crop_boxes([_rect(0, 0, 10, 100)], 0, 0, 2, 100, 0.3)
    assert out == []


def test_crop_polygon_translates_without_point_clipping():
    poly = {"label": "car", "shape_type": "polygon",
            "points": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "x": 0, "y": 0, "width": 30, "height": 30}
    out = crop_boxes([poly], 10, 10, 20, 20, 0.3)  # bbox 交比例 400/900 ≈ 0.44
    assert len(out) == 1
    assert out[0]["points"][0] == [-10, -10]       # 越界顶点不裁剪（与 RandomTranslate 一致）
    assert (out[0]["x"], out[0]["y"]) == (-10, -10)


def test_crop_polygon_low_ratio_dropped():
    poly = {"label": "car", "shape_type": "polygon",
            "points": [[0, 0], [30, 0], [30, 30], [0, 30]],
            "x": 0, "y": 0, "width": 30, "height": 30}
    assert crop_boxes([poly], 25, 25, 20, 20, 0.3) == []  # 25/900 < 0.3


def test_crop_zero_intersection_dropped_even_with_zero_min_visible():
    # min_visible=0 时零交集也必须丢弃，否则整框平移出窗、坐标越界污染导出
    poly = {"label": "car", "shape_type": "polygon",
            "points": [[200, 0], [230, 0], [230, 30], [200, 30]],
            "x": 200, "y": 0, "width": 30, "height": 30}
    rot = {"label": "car", "shape_type": "rotation",
           "points": [[200, 200], [230, 200], [230, 230], [200, 230]],
           "x": 200, "y": 200, "width": 30, "height": 30}
    assert crop_boxes([poly, rot], 0, 0, 100, 100, 0.0) == []


def test_crop_point_inside_kept_outside_dropped():
    pt = lambda x, y: {"label": "nose", "shape_type": "point", "points": [[x, y]], "x": x, "y": y, "width": 0, "height": 0}
    out = crop_boxes([pt(15, 15)], 10, 10, 20, 20, 0.3)
    assert len(out) == 1 and out[0]["points"] == [[5, 5]]
    assert crop_boxes([pt(5, 15)], 10, 10, 20, 20, 0.3) == []


def test_crop_degenerate_box_treated_as_point():
    out = crop_boxes([_rect(15, 15, 0, 0)], 10, 10, 20, 20, 0.3)
    assert len(out) == 1 and (out[0]["x"], out[0]["y"]) == (5, 5)
    assert crop_boxes([_rect(5, 5, 0, 0)], 10, 10, 20, 20, 0.3) == []


def test_crop_preserves_metadata():
    b = _rect(5, 5, 20, 20, group_id=3, flags={"paste": True})
    out = crop_boxes([b], 0, 0, 100, 100, 0.3)
    assert out[0]["group_id"] == 3


# ---------- Augmenter 集成（真实 Qt 子进程） ----------

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Qt 子进程在解释器退出阶段析构顺序不定，偶发崩溃/挂死。断言通过、标记打印后
# 立即 flush + os._exit(0) 绕开支路阶段；os._exit 不冲刷缓冲，必须先 flush。
_QT_EXIT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_real_qt(script, timeout=180):
    """真实 Qt 子进程执行。

    必须带超时，否则子进程卡死会让整个 pytest 无限挂起；
    必须显式 UTF-8，否则中文 Windows 下按 gbk 解码，子进程一旦输出 emoji/生僻字
    就抛 UnicodeDecodeError，把真正的断言错误整个吃掉。
    """
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT),
                        "PYTHONIOENCODING": "utf-8"}
    try:
        r = subprocess.run([sys.executable, "-c", script + _QT_EXIT_GUARD], cwd=ROOT, env=env,
                           text=True, capture_output=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"Qt 子进程 {timeout}s 超时（疑似死循环/窗口数爆炸）\n"
            f"stdout: {e.stdout}\nstderr: {e.stderr}") from None
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_augmenter_crop_all_mode_outputs_and_sidecar(tmp_path):
    script = f'''
import json, os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "img.png")
os.makedirs(out, exist_ok=True)
QImage(100, 100, QImage.Format_RGB888).save(src)
boxes = [{{"label": "cat", "shape_type": "rectangle", "x": 10, "y": 10, "width": 40, "height": 40}}]
res = Augmenter(os.path.join(out, "exports")).run(
    [src], {{0: boxes}}, [], mode="all",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.3}})
stems = sorted(r["stem"] for r in res)
assert stems == ["img_crop_r1c1_ov10"], stems  # 其余三窗可见比例 ≤0.25 < 0.3
d = os.path.join(out, "exports", "images")
assert os.path.exists(os.path.join(d, "img_crop_r1c1_ov10.png"))
data = json.load(open(os.path.join(d, "img_crop_r1c1_ov10.json"), encoding="utf-8"))
assert (data["imageWidth"], data["imageHeight"]) == (60, 60)
assert data["shapes"][0]["points"] == [[10, 10], [50, 10], [50, 50], [10, 50]]
print("OK_ALL")
'''
    assert "OK_ALL" in _run_real_qt(script)


def test_augmenter_crop_fill_pads_overflow_with_rgb_and_names_pad(tmp_path):
    # fill 开启：等步长网格 [0,50]，框(60,60,40,40) 仅 r2c2 全可见。
    # r2c2 窗口 [50,110]×[50,110] 右/下各越界 10px，须以 fill_rgb 补齐，命名加 _pad。
    script = f'''
import json, os
from PySide6.QtGui import QColor, QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "img.png")
os.makedirs(out, exist_ok=True)
img = QImage(100, 100, QImage.Format_RGB888)
img.fill(QColor(255, 0, 0))
img.save(src)
boxes = [{{"label": "cat", "shape_type": "rectangle", "x": 60, "y": 60, "width": 40, "height": 40}}]
res = Augmenter(os.path.join(out, "exports")).run(
    [src], {{0: boxes}}, [], mode="all",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.3,
               "fill": True, "fill_rgb": (114, 114, 114)}})
stems = sorted(r["stem"] for r in res)
assert stems == ["img_crop_r2c2_ov10_pad"], stems
d = os.path.join(out, "exports", "images")
assert os.path.exists(os.path.join(d, "img_crop_r2c2_ov10_pad.png"))
data = json.load(open(os.path.join(d, "img_crop_r2c2_ov10_pad.json"), encoding="utf-8"))
assert (data["imageWidth"], data["imageHeight"]) == (60, 60)
assert data["shapes"][0]["points"] == [[10, 10], [50, 10], [50, 50], [10, 50]]

def rgb(tile, x, y):
    c = tile.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())

tile = QImage(os.path.join(d, "img_crop_r2c2_ov10_pad.png"))
assert (tile.width(), tile.height()) == (60, 60)
assert rgb(tile, 30, 30) == (255, 0, 0)        # 窗内源区仍是红
assert rgb(tile, 55, 30) == (114, 114, 114)    # 右侧越界 → 补边
assert rgb(tile, 30, 55) == (114, 114, 114)    # 下侧越界 → 补边
assert rgb(tile, 55, 55) == (114, 114, 114)    # 右下角越界 → 补边
print("OK_FILL_PAD")
'''
    assert "OK_FILL_PAD" in _run_real_qt(script)


def test_augmenter_crop_fill_keeps_window_size_when_image_smaller(tmp_path):
    # 图(40×40) 小于窗口(60×60)：填充开启时尺寸严格固定，输出 60×60 且大部分为补边；
    # 用非默认 RGB(0,255,0) 证明填充值确实按 spec 生效而非写死。
    script = f'''
import os
from PySide6.QtGui import QColor, QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "s.png")
os.makedirs(out, exist_ok=True)
img = QImage(40, 40, QImage.Format_RGB888)
img.fill(QColor(255, 0, 0))
img.save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 5, "y": 5, "width": 20, "height": 20}}]
res = Augmenter(os.path.join(out, "exports")).run(
    [src], {{0: boxes}}, [], mode="all",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.3,
               "fill": True, "fill_rgb": (0, 255, 0)}})
assert len(res) == 1, res
assert (res[0]["width"], res[0]["height"]) == (60, 60)   # 未被钳到 40×40
assert res[0]["stem"] == "s_crop_r1c1_ov10_pad"

def rgb(tile, x, y):
    c = tile.pixelColor(x, y)
    return (c.red(), c.green(), c.blue())

tile = QImage(os.path.join(out, "exports", "images", "s_crop_r1c1_ov10_pad.png"))
assert rgb(tile, 5, 5) == (255, 0, 0)      # 图内区域
assert rgb(tile, 50, 50) == (0, 255, 0)    # 图外区域 → 补边
print("OK_FILL_SMALL")
'''
    assert "OK_FILL_SMALL" in _run_real_qt(script)


def test_augmenter_crop_image_smaller_than_window(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "s.png")
os.makedirs(out, exist_ok=True)
QImage(40, 40, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 1, "y": 1, "width": 10, "height": 10}}]
res = Augmenter(os.path.join(out, "exports")).run(
    [src], {{0: boxes}}, [], mode="all",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.3}})
assert len(res) == 1 and res[0]["width"] == 40 and res[0]["height"] == 40
assert res[0]["stem"] == "s_crop_r1c1_ov10"
print("OK_SMALL")
'''
    assert "OK_SMALL" in _run_real_qt(script)


def test_augmenter_crop_empty_source_writes_nothing(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "e.png")
os.makedirs(out, exist_ok=True)
QImage(50, 50, QImage.Format_RGB888).save(src)
res = Augmenter(os.path.join(out, "exports")).run(
    [src], {{0: []}}, [], mode="all", skip_empty=False,
    crop_spec={{"w": 30, "h": 30, "overlap": 5, "min_visible": 0.3}})
assert res == []
assert not os.path.exists(os.path.join(out, "exports", "images")) or \
       not os.listdir(os.path.join(out, "exports", "images"))
print("OK_EMPTY")
'''
    assert "OK_EMPTY" in _run_real_qt(script)


def test_augmenter_crop_random_mode_pool(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter
from labelforge.engine.augmenter.color import Brightness

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "r.png")
os.makedirs(out, exist_ok=True)
QImage(60, 60, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 5, "y": 5, "width": 20, "height": 20}}]
res = Augmenter(os.path.join(out, "exports"), seed=3).run(
    [src], {{0: boxes}}, [(Brightness, {{"delta": (10, 20)}})], mode="random",
    crop_spec={{"w": 30, "h": 30, "overlap": 5, "min_visible": 0.3}})
assert res, "random 模式应有产出（亮度或裁剪）"
print("OK_RANDOM")
'''
    assert "OK_RANDOM" in _run_real_qt(script)


def test_panel_crop_row_and_default_spec(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel

QImage(100, 80, QImage.Format_RGB888).save(os.path.join(r"{tmp_path.as_posix()}", "p.png"))
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)

# 回归：未加载数据集就点勾选（_run_images 未经过 _ensure_boxes_loaded）不得崩溃
panel._crop_check.setChecked(True)
spec = panel._get_crop_spec()
# 无参考图 → 退回 640×640 名义尺寸，仍按 2行3列 / 重叠 15% 反推
assert spec == {{"w": 238, "h": 338, "overlap": 36, "min_visible": 0.7, "square": False,
                "fill": False, "fill_rgb": (114, 114, 114)}}, spec

panel._path_edit.setText(r"{tmp_path.as_posix()}")
panel._ensure_boxes_loaded()
panel._crop_spec = None  # 重新按首图派生
spec = panel._get_crop_spec()
# 100×80 首图 → 恰好 2行3列，重叠为较小窗口边的 15%
assert spec == {{"w": 38, "h": 43, "overlap": 6, "min_visible": 0.7, "square": False,
                "fill": False, "fill_rgb": (114, 114, 114)}}, spec
panel._update_crop_summary()
assert "38×43" in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()
assert "2×3" in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()
panel._crop_check.setChecked(False)
assert panel._get_crop_spec() is None
assert panel._crop_summary_lbl.text() != ""  # 未配置 提示
print("OK_PANEL")
'''
    assert "OK_PANEL" in _run_real_qt(script)


def test_crop_dialog_lock_queue_rotates_oldest_out(tmp_path):
    # 锁按点击先后排队、最多 2 个：已锁 2 个再点第 3 个，最早锁的自动解锁
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})

dlg._lock_size.setChecked(True)
dlg._lock_ov.setChecked(True)
assert (dlg._lock_size.isChecked(), dlg._lock_ov.isChecked(),
        dlg._lock_cnt.isChecked()) == (True, True, False)

dlg._lock_cnt.setChecked(True)          # 第 3 个 → 挤掉最早的 size
assert (dlg._lock_size.isChecked(), dlg._lock_ov.isChecked(),
        dlg._lock_cnt.isChecked()) == (False, True, True), "应轮转解锁最早锁定的尺寸"

dlg._lock_size.setChecked(True)         # 再点 size → 挤掉 ov
assert (dlg._lock_size.isChecked(), dlg._lock_ov.isChecked(),
        dlg._lock_cnt.isChecked()) == (True, False, True)
print("OK_LOCK_QUEUE")
'''
    assert "OK_LOCK_QUEUE" in _run_real_qt(script)


def test_crop_dialog_lock_count_and_overlap_derives_size_in_range(tmp_path):
    # 2000×1000 图，锁「数量」(3列2行) + 锁「重叠」→ 尺寸可调但被收窄到可行区间。
    # 区间端点：w∈[734,999]（734 是刚够 3 列的最小值，733 会变 4 列），h∈[550,999]。
    # 改重叠 100 后：尺寸反推到 w=734、h=550（与旧行为一致，锁定项仍驱动派生项）。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.engine.augmenter.crop import size_range_for_count, window_starts

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})
dlg._cols.setValue(3)
dlg._rows.setValue(2)
dlg._lock_cnt.setChecked(True)
dlg._lock_ov.setChecked(True)
assert (dlg._w.value(), dlg._h.value()) == (800, 600), (dlg._w.value(), dlg._h.value())
# 锁「数量+重叠」并不唯一确定尺寸，故尺寸可编辑，只是范围被收窄到可行区间
assert not dlg._w.isReadOnly() and not dlg._h.isReadOnly(), "尺寸应区间可编辑"
assert (dlg._w.minimum(), dlg._w.maximum()) == size_range_for_count(2000, 3, 200), \\
    (dlg._w.minimum(), dlg._w.maximum())
assert (dlg._h.minimum(), dlg._h.maximum()) == size_range_for_count(1000, 2, 200), \\
    (dlg._h.minimum(), dlg._h.maximum())

# 区间内任意尺寸都保持 3×2，锁定数量不漂移
for value in (dlg._w.minimum(), 850, dlg._w.maximum()):
    dlg._w.setValue(value)
    assert len(window_starts(2000, dlg._w.value(), dlg._ov.value())) == 3, value
    assert (dlg._cols.value(), dlg._rows.value()) == (3, 2), (value, dlg._cols.value())

# 改重叠（锁定项，用户主动改）→ 尺寸仍按它反推
dlg._ov.setValue(100)
assert (dlg._w.value(), dlg._h.value()) == (734, 550), (dlg._w.value(), dlg._h.value())
assert (dlg._cols.value(), dlg._rows.value()) == (3, 2), "锁定的数量不得漂移"
print("OK_LOCK_CNT_OV")
'''
    assert "OK_LOCK_CNT_OV" in _run_real_qt(script)


def test_crop_dialog_lock_count_edit_width_syncs_overlap_and_height(tmp_path):
    # 只锁「数量」(3列2行)：把宽 800→900，应反推重叠=(3*900−2000)/2=350，
    # 再按新重叠反推高=(1000+350)/2=675，且行列数保持 3×2 不漂。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})
dlg._cols.setValue(3)
dlg._rows.setValue(2)
dlg._lock_cnt.setChecked(True)
assert not dlg._w.isReadOnly() and not dlg._ov.isReadOnly(), "只锁 1 个时其余应可编辑"

dlg._w.setValue(900)
assert dlg._ov.value() == 350, dlg._ov.value()
assert dlg._h.value() == 675, dlg._h.value()
assert (dlg._cols.value(), dlg._rows.value()) == (3, 2)
print("OK_LOCK_CNT_EDIT_W")
'''
    assert "OK_LOCK_CNT_EDIT_W" in _run_real_qt(script)


def test_crop_dialog_lock_size_and_overlap_derives_readonly_count(tmp_path):
    # 锁「尺寸」(800×600) + 锁「重叠」(200) → 数量只读，且为真实网格数 3列2行
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})
dlg._lock_size.setChecked(True)
dlg._lock_ov.setChecked(True)
assert (dlg._cols.value(), dlg._rows.value()) == (3, 2), (dlg._cols.value(), dlg._rows.value())
assert dlg._cols.isReadOnly() and dlg._rows.isReadOnly(), "锁满 2 个后数量应只读"
print("OK_LOCK_SIZE_OV")
'''
    assert "OK_LOCK_SIZE_OV" in _run_real_qt(script)


def test_crop_dialog_lock_count_infeasible_clamps_and_warns(tmp_path):
    # 2000×1000 图，锁「数量」3 列；把宽压到 600 → 反推重叠 (3*600−2000)/2=−100 不可行，
    # 钳到 0 后实际只能切 4 列：数量写回真实值 4 并给出不可达提示
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})
dlg._cols.setValue(3)
dlg._rows.setValue(2)
dlg._lock_cnt.setChecked(True)
dlg._w.setValue(600)
assert dlg._ov.value() == 0, dlg._ov.value()
assert dlg._cols.value() == 4, dlg._cols.value()
assert dlg._infeasible is True
dlg._refresh()
assert "无法凑出锁定的数量" in dlg._est.text(), dlg._est.text()
print("OK_INFEASIBLE")
'''
    assert "OK_INFEASIBLE" in _run_real_qt(script)


def test_crop_dialog_locked_count_infeasible_shows_hint(tmp_path):
    # 2000×1000 图，锁「数量」3列；把宽压到 600 → 反推重叠 (3*600−2000)/2 < 0 钳到 0，
    # 3 窗只盖 1800 < 2000，实际需 4 列 → 数量按实际写回并给出不可达提示。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.ui.i18n import t as tr

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(2000, 1000, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 800, "h": 600, "overlap": 200, "min_visible": 0.7}})
dlg._cols.setValue(3)
dlg._rows.setValue(2)
dlg._lock_cnt.setChecked(True)
assert dlg._cols.value() == 3 and not dlg._infeasible

dlg._w.setValue(600)
assert dlg._ov.value() == 0, dlg._ov.value()      # 反推为负 → 钳到 0
assert dlg._cols.value() == 4, dlg._cols.value()  # 3 列盖不满 → 写回实际 4 列
assert dlg._infeasible is True
dlg._refresh()
assert tr("crop_infeasible") in dlg._est.text(), dlg._est.text()
print("OK_INFEASIBLE")
'''
    assert "OK_INFEASIBLE" in _run_real_qt(script)


def test_crop_dialog_fill_defaults_and_spec(tmp_path):
    # spec 里没有 fill/fill_rgb 时：默认不填充，色值回落 114 灰，RGB 输入禁用。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "a.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 300, "h": 200, "overlap": 60, "min_visible": 0.7}})
s = dlg.spec()
assert s["fill"] is False, s
assert tuple(s["fill_rgb"]) == (114, 114, 114), s
assert dlg._fill.isChecked() is False
assert (dlg._fill_r.value(), dlg._fill_g.value(), dlg._fill_b.value()) == (114, 114, 114)
assert not dlg._fill_r.isEnabled() and not dlg._fill_b.isEnabled()
print("OK_FILL_DEFAULT")
'''
    assert "OK_FILL_DEFAULT" in _run_real_qt(script)


def test_crop_dialog_fill_toggle_rgb_and_swatch(tmp_path):
    # 勾选填充 → RGB 可编辑；改色值 → spec() 与色块同步。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "a.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 300, "h": 200, "overlap": 60, "min_visible": 0.7}})
dlg._fill.setChecked(True)
assert dlg._fill_r.isEnabled() and dlg._fill_g.isEnabled() and dlg._fill_b.isEnabled()
dlg._fill_r.setValue(255); dlg._fill_g.setValue(0); dlg._fill_b.setValue(255)
s = dlg.spec()
assert s["fill"] is True and tuple(s["fill_rgb"]) == (255, 0, 255), s

img = dlg._fill_swatch.grab().toImage()
c = img.pixelColor(img.width() // 2, img.height() // 2)
assert (c.red(), c.green(), c.blue()) == (255, 0, 255), (c.red(), c.green(), c.blue())

dlg._fill.setChecked(False)
assert not dlg._fill_r.isEnabled()
assert dlg.spec()["fill"] is False
print("OK_FILL_TOGGLE")
'''
    assert "OK_FILL_TOGGLE" in _run_real_qt(script)


def test_crop_dialog_fill_spec_roundtrip(tmp_path):
    # 传入 fill=True + fill_rgb 列表 → UI 回填，spec() 原样吐回。
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "a.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 300, "h": 200, "overlap": 60, "min_visible": 0.7,
                         "fill": True, "fill_rgb": [10, 20, 30]}})
assert dlg._fill.isChecked() is True
assert (dlg._fill_r.value(), dlg._fill_g.value(), dlg._fill_b.value()) == (10, 20, 30)
s = dlg.spec()
assert s["fill"] is True and tuple(s["fill_rgb"]) == (10, 20, 30), s
print("OK_FILL_ROUNDTRIP")
'''
    assert "OK_FILL_ROUNDTRIP" in _run_real_qt(script)


def test_crop_dialog_fill_preview_uses_pad_grid_and_paints_fill(tmp_path):
    # 200×200 白图，窗 102、重叠 2（补边 2px = 1%，在 30% 上限内）：
    #   填充开 → 等步长 [0,100]，末窗越界 2px，预览须画出填充色
    #   填充关 → 末窗贴边 [0,98]，全程不越界，预览无填充色
    # 注：夹具不能用「大补边」（如 150/50 补 33%），那已被 30% 硬墙吸附。
    script = f'''
import os
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "w.png")
img = QImage(200, 200, QImage.Format_RGB888)
img.fill(QColor(255, 255, 255))
img.save(src)
base = {{"w": 102, "h": 102, "overlap": 2, "min_visible": 0.0, "square": False}}


def xs_of(dlg):
    dlg._preview.resize(420, 320)
    dlg._refresh()
    return sorted({{r[0][0] for r in dlg._preview._rects}})


def has_magenta(dlg):
    shot = dlg._preview.grab().toImage()
    mag = QColor(255, 0, 255).rgb()
    return any(shot.pixelColor(x, y).rgb() == mag
               for y in range(0, shot.height(), 3)
               for x in range(0, shot.width(), 3))


on = dict(base, fill=True, fill_rgb=[255, 0, 255])
dlg_on = CropConfigDialog([src], {{0: []}}, on)
assert xs_of(dlg_on) == [0, 100], xs_of(dlg_on)
assert has_magenta(dlg_on) is True

off = dict(base, fill=False)
dlg_off = CropConfigDialog([src], {{0: []}}, off)
assert xs_of(dlg_off) == [0, 98], xs_of(dlg_off)
assert has_magenta(dlg_off) is False
print("OK_FILL_PREVIEW")
'''
    assert "OK_FILL_PREVIEW" in _run_real_qt(script)


def test_crop_dialog_linkage_and_preview(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.engine.augmenter.crop import size_for_flush, window_starts

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "d.png")
QImage(200, 100, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 10, "y": 10, "width": 40, "height": 40}}]
dlg = CropConfigDialog([src], {{0: boxes}},
                       {{"w": 100, "h": 50, "overlap": 10, "min_visible": 0.3}})
assert (dlg._rows.value(), dlg._cols.value()) == (3, 3)  # 初始化即联动（brief 笔误：(3,2)，实际 window_starts(200,100,10)=3）

dlg._w.setValue(150)  # 尺寸 → 数量
assert dlg._cols.value() == len(window_starts(200, 150, 10))

dlg._cols.setValue(5)  # 数量 → 尺寸（沿用当前重叠，保持恰好铺满）
assert dlg._w.value() == size_for_flush(200, 5, dlg._ov.value()), \
    (dlg._w.value(), dlg._ov.value())
assert 5 * dlg._w.value() - 4 * dlg._ov.value() == 200, \
    (dlg._w.value(), dlg._ov.value())

dlg._square.setChecked(True)  # 正方形镜像
assert dlg._w.value() == dlg._h.value()

dlg._refresh()
assert dlg._preview._rects and dlg._est.text()
s = dlg.spec()
assert s["min_visible"] == 0.5 and s["w"] == dlg._w.value() and s["square"] is True  # 0.3 低于下限被钳到 0.5
print("OK_DIALOG")
'''
    assert "OK_DIALOG" in _run_real_qt(script)


def test_panel_crop_clamp_summary_and_path_reset(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.engine.augmenter.crop import window_starts

d = r"{tmp_path.as_posix()}"
d2 = os.path.join(d, "other")
os.makedirs(d2, exist_ok=True)
QImage(10, 10, QImage.Format_RGB888).save(os.path.join(d, "t.png"))
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)

# 钳制后摘要口径与引擎一致（10×10 图、4×4 窗、ov=10 → 7×7）
panel._path_edit.setText(d)
panel._ensure_boxes_loaded()
panel._crop_check.setChecked(True)
panel._crop_spec = {{"w": 4, "h": 4, "overlap": 10, "min_visible": 0.0, "square": True}}
panel._update_crop_summary()
assert "7×7" in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()

# 切换数据集目录清空裁剪配置与缓存
assert panel._crop_spec is not None
panel._path_edit.setText(d2)
assert panel._crop_spec is None
assert panel._run_images == []

# 超大图：默认仍按参考图反推出 2行3列，窗口数不爆炸，重叠不超半窗上限
panel._crop_ref_size = lambda: (20000, 20000)
spec = panel._default_crop_spec()
w, h, ov = spec["w"], spec["h"], spec["overlap"]
assert len(window_starts(20000, w, ov)) == 3, spec
assert len(window_starts(20000, h, ov)) == 2, spec
assert ov <= min(w, h) // 2, spec
print("OK_PANEL2")
'''
    assert "OK_PANEL2" in _run_real_qt(script)


def test_crop_dialog_ov_cap_and_count_roundtrip(tmp_path):
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.engine.augmenter.crop import size_for_count, size_for_flush, window_starts

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "big.png")
QImage(1920, 1080, QImage.Format_RGB888).save(src)
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 640, "h": 640, "overlap": 64, "min_visible": 0.3, "square": True}})

# 重叠上限 = 半窗
assert dlg._ov.maximum() == 320, dlg._ov.maximum()

# 数量→尺寸闭环：设定列数后实际窗口数 == 显示列数
for req in (2, 3, 5, 7):
    dlg._cols.setValue(req)
    assert dlg._cols.value() == req, (req, dlg._cols.value())
    assert len(window_starts(1920, dlg._w.value(), dlg._ov.value())) == req, req
    assert dlg._rows.value() == len(window_starts(1080, dlg._h.value(), dlg._ov.value()))

# 不可达数量钳到最近（50 长、ov=49 → 只能 1 窗）
size = dlg._exact_size(50, size_for_count(50, 3, 49), 3, 49)
assert size == 50 and len(window_starts(50, size, 49)) == 1
print("OK_DIALOG2")
'''
    assert "OK_DIALOG2" in _run_real_qt(script)


def test_augmenter_crop_random_mode_deterministic_crop_path(tmp_path):
    # 池=[("__crop__",spec)] → 必选；ratio 门 1.0 必过 → 裁剪路径确定性执行
    # 窗口起点两轴 [0,40]；框(10,10,40,40) 面积1600：r1c1 全含=1.0，
    # r1c2/r2c1=400/1600=0.25、r2c2=100/1600=0.0625，均 <0.3 丢弃
    script = f'''
import os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "x.png")
os.makedirs(out, exist_ok=True)
QImage(100, 100, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 10, "y": 10, "width": 40, "height": 40}}]
res = Augmenter(os.path.join(out, "exports"), seed=1).run(
    [src], {{0: boxes}}, [], mode="random",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.3}})
stems = sorted(r["stem"] for r in res)
assert stems == ["x_crop_r1c1_ov10"], stems
d = os.path.join(out, "exports", "images")
assert not os.path.exists(os.path.join(d, "x.png"))  # 裁剪已产出 → 原图兜底不得触发
print("OK_RANDOM_CROP")
'''
    assert "OK_RANDOM_CROP" in _run_real_qt(script)


def test_augmenter_crop_random_mode_zero_windows_fallback(tmp_path):
    # 框(30,30,60,60) 面积3600，min_visible=0.9：四窗交比 0.25 / 0.417 / 0.417 / 0.694
    # 全部 <0.9 → 0 窗产出 → _apply_crop 返回 0 → 走 _save_original 兜底
    script = f'''
import os
from PySide6.QtGui import QImage
from labelforge.engine.augmenter import Augmenter

out = r"{(tmp_path / "out").as_posix()}"
src = os.path.join(out, "z.png")
os.makedirs(out, exist_ok=True)
QImage(100, 100, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 30, "y": 30, "width": 60, "height": 60}}]
res = Augmenter(os.path.join(out, "exports"), seed=1).run(
    [src], {{0: boxes}}, [], mode="random",
    crop_spec={{"w": 60, "h": 60, "overlap": 10, "min_visible": 0.9}})
assert len(res) == 1, res
stem = res[0]["stem"]
assert "_crop_" not in stem, stem
d = os.path.join(out, "exports", "images")
assert os.path.exists(os.path.join(d, "z.png"))
assert os.path.exists(os.path.join(d, "z.json"))
print("OK_RANDOM_FALLBACK")
'''
    assert "OK_RANDOM_FALLBACK" in _run_real_qt(script)


def test_crop_dialog_templates_row_apply_and_recompute(tmp_path):
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.core.config_manager import add_crop_template
from labelforge.engine.augmenter.crop import normalized_window, window_starts

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "t.png")
QImage(200, 100, QImage.Format_RGB888).save(src)
boxes = [{{"label": "a", "shape_type": "rectangle", "x": 10, "y": 10, "width": 40, "height": 40}}]
# 初始 spec 足够大，使模板 overlap 不被旧半窗上限钳制
init_spec = {{"w": 1024, "h": 768, "overlap": 128, "min_visible": 0.6}}

dlg0 = CropConfigDialog([src], {{0: boxes}}, init_spec)
row_labels = [w for w in dlg0.findChildren(QLabel) if w.text() == "模板"]
assert not row_labels, "无模板时不应出现「模板」行"

add_crop_template({{"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5}})
add_crop_template({{"w": 512, "h": 512, "overlap": 64, "min_visible": 0.75, "square": True}})
dlg = CropConfigDialog([src], {{0: boxes}}, init_spec)

row_labels = [w for w in dlg.findChildren(QLabel) if w.text() == "模板"]
assert len(row_labels) == 1, row_labels
btns = [b for b in dlg.findChildren(QPushButton) if "×" in b.text()]
assert len(btns) == 2, [b.text() for b in btns]
texts = sorted(b.text() for b in btns)
assert texts == ["512×512·重叠64·75%·正方形", "800×640·重叠200·50%"], texts

target = [b for b in btns if b.text() == "800×640·重叠200·50%"][0]
target.click()
s = dlg.spec()
assert s == {{"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5, "square": False,
             "fill": False, "fill_rgb": (114, 114, 114)}}, s
cw, ch, ov = normalized_window(200, 100, s)
assert dlg._cols.value() == len(window_starts(200, cw, ov)), (dlg._cols.value(), cw, ov)
assert dlg._rows.value() == len(window_starts(100, ch, ov)), (dlg._rows.value(), ch, ov)

target2 = [b for b in btns if b.text() == "512×512·重叠64·75%·正方形"][0]
target2.click()
s2 = dlg.spec()
assert s2["square"] is True and s2["w"] == s2["h"] == 512, s2
cw2, ch2, ov2 = normalized_window(200, 100, s2)
assert dlg._cols.value() == len(window_starts(200, cw2, ov2))
assert dlg._rows.value() == len(window_starts(100, ch2, ov2))
print("OK_TEMPLATES")
'''
    assert "OK_TEMPLATES" in _run_real_qt(script)


def test_panel_open_crop_config_records_template(tmp_path):
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.core.config_manager import load_crop_templates

d = r"{tmp_path.as_posix()}"
QImage(100, 80, QImage.Format_RGB888).save(os.path.join(d, "p.png"))
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)
panel._path_edit.setText(d)
panel._ensure_boxes_loaded()

spec = {{"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5, "square": False}}
from labelforge.ui import crop_dialog as cd
orig = cd.CropConfigDialog
class FakeDlg:
    def __init__(self, *a, **kw): pass
    def exec_(self): return 1
    def spec(self): return spec
cd.CropConfigDialog = FakeDlg
try:
    panel._open_crop_config()
finally:
    cd.CropConfigDialog = orig
templates = load_crop_templates()
assert len(templates) == 1, templates
assert templates[0]["w"] == 800 and templates[0]["h"] == 640
assert templates[0]["overlap"] == 200 and templates[0]["min_visible"] == 0.5
print("OK_PANEL_TEMPLATE")
'''
    assert "OK_PANEL_TEMPLATE" in _run_real_qt(script)


def test_crop_template_persists_fill_and_rgb(tmp_path):
    # 模板落盘/读回必须保住 fill 与 fill_rgb；损坏值回落默认灰并钳到 0..255。
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from labelforge.core.config_manager import add_crop_template, load_crop_templates

base = {{"w": 800, "h": 640, "overlap": 200, "min_visible": 0.5}}

assert add_crop_template({{**base, "fill": True, "fill_rgb": [10, 20, 30]}})
t = load_crop_templates()[0]
assert t["fill"] is True, t
assert tuple(t["fill_rgb"]) == (10, 20, 30), t

# 损坏：非数值 → 回落 114 灰
assert add_crop_template({{**base, "h": 600, "fill": True, "fill_rgb": "junk"}})
t = load_crop_templates()[0]
assert t["h"] == 600 and tuple(t["fill_rgb"]) == (114, 114, 114), t

# 越界：钳到 0..255
assert add_crop_template({{**base, "h": 500, "fill": True, "fill_rgb": [999, -5, 12]}})
t = load_crop_templates()[0]
assert t["h"] == 500 and tuple(t["fill_rgb"]) == (255, 0, 12), t

# 缺 fill 键 → 默认关，色值仍给默认（老配置向后兼容）
assert add_crop_template({{**base, "h": 400}})
t = load_crop_templates()[0]
assert t["h"] == 400 and t["fill"] is False and tuple(t["fill_rgb"]) == (114, 114, 114), t
print("OK_TEMPLATE_FILL")
'''
    assert "OK_TEMPLATE_FILL" in _run_real_qt(script)


def test_crop_template_fill_variant_not_deduped(tmp_path):
    # 去重键须含 fill/fill_rgb：同尺寸但填充不同的两条模板都要留住。
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from labelforge.core.config_manager import add_crop_template, load_crop_templates

base = {{"w": 640, "h": 640, "overlap": 100, "min_visible": 0.7}}
add_crop_template({{**base, "fill": False}})
add_crop_template({{**base, "fill": True, "fill_rgb": [0, 0, 0]}})
add_crop_template({{**base, "fill": True, "fill_rgb": [255, 255, 255]}})
ts = load_crop_templates()
assert len(ts) == 3, ts
assert ts[0]["fill_rgb"] == [255, 255, 255] or tuple(ts[0]["fill_rgb"]) == (255, 255, 255), ts[0]
keys = {{(t["fill"], tuple(t["fill_rgb"])) for t in ts}}
assert keys == {{(False, (114, 114, 114)), (True, (0, 0, 0)), (True, (255, 255, 255))}}, keys
print("OK_TEMPLATE_DEDUP")
'''
    assert "OK_TEMPLATE_DEDUP" in _run_real_qt(script)


def test_panel_default_spec_and_summary_pad_marker(tmp_path):
    # 面板默认 spec 带 fill/fill_rgb；开填充后摘要与模板按钮都给出「补边」标记。
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QPushButton
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.ui.i18n import t as tr
from labelforge.core.config_manager import add_crop_template

d = r"{tmp_path.as_posix()}"
QImage(200, 100, QImage.Format_RGB888).save(os.path.join(d, "p.png"))
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)
panel._path_edit.setText(d)
panel._ensure_boxes_loaded()

spec = panel._default_crop_spec()
assert spec["fill"] is False, spec
assert tuple(spec["fill_rgb"]) == (114, 114, 114), spec

panel._crop_check.setChecked(True)
panel._crop_spec = {{"w": 150, "h": 80, "overlap": 30, "min_visible": 0.7,
                    "square": False, "fill": False, "fill_rgb": (114, 114, 114)}}
panel._update_crop_summary()
assert tr("补边") not in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()

panel._crop_spec = dict(panel._crop_spec, fill=True)
panel._update_crop_summary()
assert tr("补边") in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()

# 模板按钮：带填充的模板要显示「补边」，不带的不显示
add_crop_template({{"w": 640, "h": 640, "overlap": 100, "min_visible": 0.7,
                   "fill": True, "fill_rgb": [114, 114, 114]}})
add_crop_template({{"w": 512, "h": 512, "overlap": 64, "min_visible": 0.75,
                   "fill": False}})
dlg = CropConfigDialog([os.path.join(d, "p.png")], {{0: []}}, panel._default_crop_spec())
btns = [b for b in dlg.findChildren(QPushButton) if "×" in b.text()]
assert len(btns) == 2, [b.text() for b in btns]
marked = [b.text() for b in btns if tr("补边") in b.text()]
assert len(marked) == 1 and "640" in marked[0], [b.text() for b in btns]
print("OK_PANEL_PAD")
'''
    assert "OK_PANEL_PAD" in _run_real_qt(script)


def test_crop_dialog_apply_template_restores_fill(tmp_path):
    # 套用带填充的模板，fill 与 fill_rgb 必须一起回填到 UI 与 spec()。
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QPushButton
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.core.config_manager import add_crop_template

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "t.png")
QImage(200, 100, QImage.Format_RGB888).save(src)
add_crop_template({{"w": 640, "h": 480, "overlap": 100, "min_visible": 0.7,
                   "fill": True, "fill_rgb": [7, 8, 9]}})
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 1024, "h": 768, "overlap": 128, "min_visible": 0.6}})
assert dlg.spec()["fill"] is False
btn = [b for b in dlg.findChildren(QPushButton) if "640" in b.text()][0]
btn.click()
s = dlg.spec()
assert s["fill"] is True, s
assert tuple(s["fill_rgb"]) == (7, 8, 9), s
assert dlg._fill.isChecked() is True
assert (dlg._fill_r.value(), dlg._fill_g.value(), dlg._fill_b.value()) == (7, 8, 9)
print("OK_TEMPLATE_RESTORE_FILL")
'''
    assert "OK_TEMPLATE_RESTORE_FILL" in _run_real_qt(script)


# ---------- UI 改版：锁图标 / 文案 / 尺寸 / 布局 / 面板按钮 ----------

_DIALOG_BOOT = '''
import os
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QPushButton
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.ui.i18n import t as tr

d = r"{d}"
src = os.path.join(d, "a.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
SPEC = {{"w": 300, "h": 200, "overlap": 60, "min_visible": 0.7}}
dlg = CropConfigDialog([src], {{0: []}}, SPEC)
'''


def test_crop_dialog_lock_is_icon_toggle_button(tmp_path):
    # 锁不再是文字 checkbox，而是 🔓/🔒 图标按钮，点击切换；被轮转挤掉时图标须复位。
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
for b in (dlg._lock_size, dlg._lock_ov, dlg._lock_cnt):
    assert isinstance(b, QPushButton), type(b)
    assert not isinstance(b, QCheckBox), "锁应改为图标按钮而非 checkbox"
    assert b.isCheckable()
    assert b.text() == "\\U0001F513", b.text()          # 未锁 = 开锁

dlg._lock_cnt.setChecked(True)
assert dlg._lock_cnt.text() == "\\U0001F512", dlg._lock_cnt.text()   # 已锁 = 闭锁
dlg._lock_cnt.setChecked(False)
assert dlg._lock_cnt.text() == "\\U0001F513"

dlg._lock_size.setChecked(True)
dlg._lock_ov.setChecked(True)
assert dlg._lock_size.text() == "\\U0001F512" and dlg._lock_ov.text() == "\\U0001F512"
dlg._lock_cnt.setChecked(True)                            # 挤掉最早的 size
assert dlg._lock_size.isChecked() is False
assert dlg._lock_size.text() == "\\U0001F513", "被挤掉的锁图标须复位为开锁"
print("OK_LOCK_ICON")
'''
    assert "OK_LOCK_ICON" in _run_real_qt(script)


def test_crop_dialog_title_size_and_fill_label(tmp_path):
    # 标题去掉「设置」，尺寸 1270×800，填充勾选框去掉「启用」二字。
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
assert dlg.windowTitle() == tr("滑窗裁剪"), dlg.windowTitle()
assert "设置" not in dlg.windowTitle(), dlg.windowTitle()
assert (dlg.minimumWidth(), dlg.minimumHeight()) == (1270, 800), \\
    (dlg.minimumWidth(), dlg.minimumHeight())
assert dlg._fill.text() == tr("边缘填充"), dlg._fill.text()
assert "启用" not in dlg._fill.text(), dlg._fill.text()
print("OK_TITLE_SIZE_LABEL")
'''
    assert "OK_TITLE_SIZE_LABEL" in _run_real_qt(script)


def test_crop_dialog_layout_params_on_top_preview_middle(tmp_path):
    # 参数区全部移到上方按列排布，预览图居中占满宽度，底部红绿说明与确定/取消在最下方。
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg.show()
app.processEvents()


def at(w):
    return w.mapTo(dlg, QPoint(0, 0))


# 参数在上、预览图在下
assert at(dlg._w).y() < at(dlg._preview).y(), (at(dlg._w).y(), at(dlg._preview).y())
# 按列排布：第二列的控件在第一列控件的右侧（原来是正下方）
assert at(dlg._vis).x() > at(dlg._w).x(), (at(dlg._vis).x(), at(dlg._w).x())
assert at(dlg._fill).x() > at(dlg._ov).x(), (at(dlg._fill).x(), at(dlg._ov).x())
# 预览图占满整个宽度
assert dlg._preview.width() >= dlg.width() * 0.9, (dlg._preview.width(), dlg.width())
# 底部按钮行仍在预览图下方
ok = [b for b in dlg.findChildren(QPushButton) if b.text() == tr("确定")][0]
assert at(ok).y() > at(dlg._preview).y() + dlg._preview.height(), (at(ok).y(),)
print("OK_LAYOUT")
'''
    assert "OK_LAYOUT" in _run_real_qt(script)


def test_panel_crop_config_btn_not_clipped_in_english(tmp_path):
    # 「配置…」按钮不能是固定窄宽，否则英文 Config… 会被截断。
    script = f'''
import os
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.ui.i18n import set_lang

d = r"{tmp_path.as_posix()}"
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)
for lang in ("zh", "en"):
    set_lang(lang)
    panel._refresh_texts()
    b = panel._crop_config_btn
    assert b.sizeHint().width() <= b.maximumWidth(), \\
        (lang, b.text(), b.sizeHint().width(), b.maximumWidth())
set_lang("zh")
print("OK_CONFIG_BTN_WIDTH")
'''
    assert "OK_CONFIG_BTN_WIDTH" in _run_real_qt(script)


def test_panel_uncheck_all_transforms_button(tmp_path):
    # 「跳过空标签」右侧新增一键取消全部增强方式的按钮。
    script = f'''
import os
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.ui.i18n import t as tr

d = r"{tmp_path.as_posix()}"
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)
btn = panel._aug_uncheck_all_btn
assert btn.text() == tr("取消全选"), btn.text()

# 同一行、且在「跳过空标签」右侧
panel.show()
app.processEvents()
assert btn.parentWidget() is panel._aug_skip_empty.parentWidget()
assert btn.x() > panel._aug_skip_empty.x(), (btn.x(), panel._aug_skip_empty.x())
assert abs(btn.y() - panel._aug_skip_empty.y()) < 20

# 点一下：所有增强方式（含原图与滑窗裁剪）全部取消勾选
for _n, (cb, _p, _d) in panel._aug_widgets.items():
    cb.setChecked(True)
panel._crop_check.setChecked(True)
btn.click()
assert all(not cb.isChecked() for _n, (cb, _p, _d) in panel._aug_widgets.items())
assert not panel._crop_check.isChecked()
print("OK_UNCHECK_ALL")
'''
    assert "OK_UNCHECK_ALL" in _run_real_qt(script)


def test_default_window_gives_two_by_three_grid_with_fifteen_percent_overlap():
    """默认网格 2 行 3 列、重叠为窗口的 15%：逐尺寸验证实际排布严格等于目标。

    用 ceil 而非 round 反推尺寸：round 会把窗口取小，导致余量多凑出一个窗口
    （100×80 上 round 得 w=37 → 4 列），ceil 取到能盖满目标列数的最小窗口。
    """
    from labelforge.engine.augmenter.crop import default_window, window_starts

    for iw, ih in ((1920, 1080), (1280, 720), (800, 600), (640, 480),
                   (4000, 3000), (1000, 1000), (300, 200), (100, 80), (64, 64)):
        w, h, ov = default_window(iw, ih)
        cols = len(window_starts(iw, w, ov))
        rows = len(window_starts(ih, h, ov))
        assert (rows, cols) == (2, 3), (iw, ih, w, h, ov, rows, cols)
        # 重叠两轴共用一个像素值，无法在两轴都恰好 15%，以较小窗口边为基准
        assert abs(ov - round(0.15 * min(w, h))) <= 1, (iw, ih, w, h, ov)
        assert 0 < ov < min(w, h) // 2, (iw, ih, w, h, ov)  # 不触半窗上限
        assert w <= iw and h <= ih, (iw, ih, w, h)


def test_default_window_falls_back_to_square_ratio_reference():
    """无参考图时用 640×640 名义尺寸，仍保证 2×3 且比例口径一致。"""
    from labelforge.engine.augmenter.crop import default_window, window_starts

    w, h, ov = default_window(0, 0)
    assert w > 0 and h > 0 and ov >= 0
    w2, h2, ov2 = default_window(640, 640)
    assert (w, h, ov) == (w2, h2, ov2)
    assert len(window_starts(640, w2, ov2)) == 3
    assert len(window_starts(640, h2, ov2)) == 2


def test_panel_default_crop_spec_derives_two_by_three_from_reference(tmp_path):
    """面板默认 spec 不再是写死的 640×640/200，而是按首图反推出 2 行 3 列。"""
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel
from labelforge.engine.augmenter.crop import window_starts

d = r"{tmp_path.as_posix()}"
QImage(1920, 1080, QImage.Format_RGB888).save(os.path.join(d, "p.png"))
editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)
panel._path_edit.setText(d)
panel._ensure_boxes_loaded()
panel._crop_check.setChecked(True)
panel._crop_spec = None
spec = panel._get_crop_spec()
w, h, ov = spec["w"], spec["h"], spec["overlap"]
assert len(window_starts(1920, w, ov)) == 3, spec
assert len(window_starts(1080, h, ov)) == 2, spec
assert abs(ov - round(0.15 * min(w, h))) <= 1, spec
assert spec["square"] is False and spec["fill"] is False
assert spec["min_visible"] == 0.7 and spec["fill_rgb"] == (114, 114, 114)
# 摘要里的行列数须真的是 2×3，而不是 "?"
assert "2×3" in panel._crop_summary_lbl.text(), panel._crop_summary_lbl.text()
print("OK_DEFAULT_SPEC")
'''
    assert "OK_DEFAULT_SPEC" in _run_real_qt(script)


def test_crop_dialog_min_height_raised_by_100(tmp_path):
    """裁剪小窗高度 700 → 800，宽度不变。"""
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
p = os.path.join(d, "a.png")
QImage(1920, 1080, QImage.Format_RGB888).save(p)
spec = {{"w": 640, "h": 640, "overlap": 200, "min_visible": 0.7,
        "square": False, "fill": False, "fill_rgb": (114, 114, 114)}}
dlg = CropConfigDialog([p], {{0: []}}, spec)
assert dlg.minimumWidth() == 1270, dlg.minimumWidth()
assert dlg.minimumHeight() == 800, dlg.minimumHeight()
dlg.show(); app.processEvents()
assert (dlg.width(), dlg.height()) == (1270, 800), (dlg.width(), dlg.height())
print("OK_HEIGHT")
'''
    assert "OK_HEIGHT" in _run_real_qt(script)


def test_crop_dialog_lock_buttons_aligned_at_any_width(tmp_path):
    """三个 🔒 必须始终右对齐成一条竖线，且与同行输入框垂直居中。

    回归：_lock_size 曾加在 addStretch() 之前，1270 宽下恰好与另两个重合，
    但窗口拉宽到 1900 时错位 103px。
    """
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog

d = r"{tmp_path.as_posix()}"
p = os.path.join(d, "a.png")
QImage(1920, 1080, QImage.Format_RGB888).save(p)
spec = {{"w": 699, "h": 584, "overlap": 88, "min_visible": 0.7,
        "square": False, "fill": False, "fill_rgb": (114, 114, 114)}}
for width in (1270, 1600, 1900):
    dlg = CropConfigDialog([p], {{0: []}}, spec)
    dlg.resize(width, 800); dlg.show(); app.processEvents()
    xs = []
    for b in (dlg._lock_size, dlg._lock_ov, dlg._lock_cnt):
        xs.append(b.mapTo(dlg, b.rect().topLeft()).x())
    if width == 1270:
        # 最小宽度下首行零余量，锁的位置由前序控件累加得出，允许 1px 取整误差
        assert max(xs) - min(xs) <= 1, (width, xs)
    else:
        # 有余量时 stretch 把三个锁都推到 col_a 右边缘，必须严格成一条竖线
        # （回归：修复前 1600 宽错位 3px、1900 宽错位 103px）
        assert len(set(xs)) == 1, (width, xs)
    # 垂直居中：锁图标中心应与同行输入框中心齐平
    for b, ref in ((dlg._lock_size, dlg._w), (dlg._lock_ov, dlg._ov),
                   (dlg._lock_cnt, dlg._cols)):
        gb = b.mapTo(dlg, b.rect().topLeft())
        gr = ref.mapTo(dlg, ref.rect().topLeft())
        cb = gb.y() + b.height() / 2
        cr = gr.y() + ref.height() / 2
        assert abs(cb - cr) <= 1.0, (width, b.height(), ref.height(), cb, cr)
    dlg.close()
print("OK_LOCK_ALIGN")
'''
    assert "OK_LOCK_ALIGN" in _run_real_qt(script)


def test_size_range_for_count_matches_brute_force():
    """尺寸可行区间闭式解必须与暴力枚举完全一致。

    锁定「重叠+数量」时尺寸不是唯一解而是一个区间：任何落在区间内的尺寸都恰好
    排出 count 个窗口。闭式解若与枚举差一个边界，用户就会看到数量莫名跳变。
    """
    from labelforge.engine.augmenter.crop import (
        MAX_WINDOW, size_range_for_count, window_starts,
    )

    for length in (40, 100, 300, 640, 1080, 1920, 4000):
        for count in (1, 2, 3, 4, 7):
            for ov in (0, 1, 9, 40, 88, 200, 500):
                if ov >= length:
                    continue
                want = [c for c in range(ov + 1, MAX_WINDOW + 1)
                        if len(window_starts(length, c, ov)) == count]
                lo, hi = size_range_for_count(length, count, ov)
                if not want:
                    assert hi < lo, (length, count, ov, lo, hi)
                else:
                    assert (lo, hi) == (want[0], want[-1]), \
                        (length, count, ov, lo, hi, want[0], want[-1])


def test_overlap_range_for_count_matches_brute_force():
    """重叠可行区间闭式解必须与暴力枚举完全一致（n=2 与 n>=3 是两套推导）。"""
    from labelforge.engine.augmenter.crop import (
        overlap_range_for_count, window_starts,
    )

    for length in (40, 100, 300, 640, 1080, 1920):
        for count in (1, 2, 3, 4, 7):
            for size in (5, 30, 88, 100, 300, 640, 699, 1003, 1920, 3000):
                want = [ov for ov in range(0, size)
                        if len(window_starts(length, size, ov)) == count]
                lo, hi = overlap_range_for_count(length, count, size)
                if not want:
                    assert hi < lo, (length, count, size, lo, hi)
                else:
                    assert (lo, hi) == (want[0], want[-1]), \
                        (length, count, size, lo, hi, want[0], want[-1])


def test_crop_dialog_locked_ov_and_cnt_keeps_size_editable_in_range(tmp_path):
    """锁「重叠+数量」时尺寸不是唯一解：应可编辑，且范围恰好是可行区间。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import size_range_for_count, window_starts

dlg._lock_ov.setChecked(True)
dlg._lock_cnt.setChecked(True)
app.processEvents()

assert dlg._lock_order == ["ov", "cnt"], dlg._lock_order
assert not dlg._w.isReadOnly(), "锁重叠+数量时尺寸应可编辑"
assert not dlg._h.isReadOnly(), "锁重叠+数量时尺寸应可编辑"

lo_w, hi_w = size_range_for_count(600, dlg._cols.value(), dlg._ov.value())
lo_h, hi_h = size_range_for_count(400, dlg._rows.value(), dlg._ov.value())
assert (dlg._w.minimum(), dlg._w.maximum()) == (lo_w, hi_w), \
    (dlg._w.minimum(), dlg._w.maximum(), lo_w, hi_w)
assert (dlg._h.minimum(), dlg._h.maximum()) == (lo_h, hi_h), \
    (dlg._h.minimum(), dlg._h.maximum(), lo_h, hi_h)

# 区间两端都能调，且数量始终锁在 3×3
for value in (lo_w, (lo_w + hi_w) // 2, hi_w):
    dlg._w.setValue(value)
    assert dlg._cols.value() == 3, (value, dlg._cols.value())
    assert len(window_starts(600, dlg._w.value(), dlg._ov.value())) == 3, value
for value in (lo_h, hi_h):
    dlg._h.setValue(value)
    assert dlg._rows.value() == 3, (value, dlg._rows.value())

# 数量与重叠都没被顶动
assert dlg._ov.value() == 60, dlg._ov.value()
print("OK_SIZE_RANGE")
'''
    assert "OK_SIZE_RANGE" in _run_real_qt(script)


def test_crop_dialog_locked_size_and_cnt_keeps_overlap_editable_in_range(tmp_path):
    """锁「尺寸+数量」时重叠同样有自由度：应可编辑，范围取两轴可行区间的交集。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import overlap_range_for_count, window_starts

dlg._lock_size.setChecked(True)
dlg._lock_cnt.setChecked(True)
app.processEvents()

assert dlg._lock_order == ["size", "cnt"], dlg._lock_order
assert not dlg._ov.isReadOnly(), "锁尺寸+数量时重叠应可编辑"

w_lo, w_hi = overlap_range_for_count(600, dlg._cols.value(), dlg._w.value())
h_lo, h_hi = overlap_range_for_count(400, dlg._rows.value(), dlg._h.value())
want_lo, want_hi = max(w_lo, h_lo), min(w_hi, h_hi)
assert (dlg._ov.minimum(), dlg._ov.maximum()) == (want_lo, want_hi), \
    (dlg._ov.minimum(), dlg._ov.maximum(), want_lo, want_hi)

# 交集内任意重叠都要同时保住 3 行与 3 列
for value in (want_lo, (want_lo + want_hi) // 2, want_hi):
    dlg._ov.setValue(value)
    assert len(window_starts(600, dlg._w.value(), dlg._ov.value())) == 3, value
    assert len(window_starts(400, dlg._h.value(), dlg._ov.value())) == 3, value
    assert dlg._cols.value() == 3 and dlg._rows.value() == 3, (value, dlg._rows.value())
print("OK_OV_RANGE")
'''
    assert "OK_OV_RANGE" in _run_real_qt(script)


def test_crop_dialog_locked_size_and_ov_keeps_count_readonly(tmp_path):
    """锁「尺寸+重叠」时数量是纯结果、没有自由度，必须仍为只读。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg._lock_size.setChecked(True)
dlg._lock_ov.setChecked(True)
app.processEvents()

assert dlg._lock_order == ["size", "ov"], dlg._lock_order
assert dlg._rows.isReadOnly(), "数量是纯结果，应保持只读"
assert dlg._cols.isReadOnly(), "数量是纯结果，应保持只读"
assert not dlg._w.isReadOnly() and not dlg._ov.isReadOnly()
print("OK_CNT_READONLY")
'''
    assert "OK_CNT_READONLY" in _run_real_qt(script)


def test_crop_dialog_unlock_restores_original_ranges(tmp_path):
    """解锁后必须恢复原始输入范围，不能把区间钳制永久留在控件上。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg._lock_ov.setChecked(True)
dlg._lock_cnt.setChecked(True)
app.processEvents()
assert dlg._w.maximum() != 8192, "锁定期间范围应已收窄"

dlg._lock_ov.setChecked(False)
dlg._lock_cnt.setChecked(False)
app.processEvents()
assert (dlg._w.minimum(), dlg._w.maximum()) == (8, 8192), \
    (dlg._w.minimum(), dlg._w.maximum())
assert (dlg._h.minimum(), dlg._h.maximum()) == (8, 8192), \
    (dlg._h.minimum(), dlg._h.maximum())
assert dlg._ov.maximum() == min(dlg._w.value(), dlg._h.value()) // 2, \
    (dlg._ov.maximum(), dlg._w.value(), dlg._h.value())
assert not dlg._w.isReadOnly() and not dlg._ov.isReadOnly()
print("OK_UNLOCK_RESTORE")
'''
    assert "OK_UNLOCK_RESTORE" in _run_real_qt(script)


def test_panel_dataset_dir_follows_background_folder(tmp_path):
    """① 切换数据集后「数据集目录」无条件跟随，导出目录随之切换。"""
    script = f'''
import os
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.processing_panel import ProcessingPanel

d1 = os.path.join(r"{tmp_path.as_posix()}", "set1")
d2 = os.path.join(r"{tmp_path.as_posix()}", "set2")
d3 = os.path.join(r"{tmp_path.as_posix()}", "manual")
for d in (d1, d2, d3):
    os.makedirs(d, exist_ok=True)

editor = type("E", (), {{"background_images": []}})()
panel = ProcessingPanel(editor)

editor.background_images = [os.path.join(d1, "a.png")]
panel._refresh_texts()
assert panel._path_edit.text() == d1, panel._path_edit.text()

# 数据集切换 → 跟随，导出目录随之切到新数据集的同级 exports
editor.background_images = [os.path.join(d2, "b.png")]
panel._refresh_texts()
assert panel._path_edit.text() == d2, panel._path_edit.text()
assert panel._get_output_dir() == os.path.normpath(os.path.join(r"{tmp_path.as_posix()}", "exports")), \
    panel._get_output_dir()

# 手动改过、且数据集没变 → 不能被冲掉
panel._path_edit.setText(d3)
panel._refresh_texts()
assert panel._path_edit.text() == d3, panel._path_edit.text()

# 数据集再变 → 无条件跟随（用户选定的规则），手动值让位
editor.background_images = [os.path.join(d1, "c.png")]
panel._refresh_texts()
assert panel._path_edit.text() == d1, panel._path_edit.text()
print("OK_PATH_FOLLOW")
'''
    assert "OK_PATH_FOLLOW" in _run_real_qt(script)


def test_crop_dialog_lock_buttons_shifted_left_with_gap(tmp_path):
    """③ 三个锁要往左挪一点：与所在行右边缘留出可见间距，且仍严格对齐。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg.resize(1600, 800)
dlg.show(); app.processEvents()

# 布局链：主布局 → 顶部三列 → 第一列（三个锁都在这里）→ 每一行
col_a = dlg.layout().itemAt(0).layout().itemAt(0).layout()
locks = (dlg._lock_size, dlg._lock_ov, dlg._lock_cnt)

xs = [b.mapTo(dlg, b.rect().topLeft()).x() for b in locks]
assert len(set(xs)) == 1, xs          # 仍严格成一条竖线

for idx, b in enumerate(locks):
    row_right = col_a.itemAt(idx).layout().geometry().right()
    lock_right = b.mapTo(dlg, b.rect().topLeft()).x() + b.width()
    assert row_right - lock_right >= 8, (idx, row_right, lock_right)
print("OK_LOCK_SHIFT")
'''
    assert "OK_LOCK_SHIFT" in _run_real_qt(script)


def test_pad_overflow_measures_uniform_step_overflow():
    """pad_overflow 就是「需要补边的像素数」：恰好铺满时为 0，步长口径才可能非 0。"""
    from labelforge.engine.augmenter.crop import pad_overflow, window_starts

    assert pad_overflow(1920, 640, 0) == 0          # 3*640 == 1920 恰好铺满
    assert pad_overflow(1920, 640, 320) == 0        # step=320, 1+ceil(1280/320)=5, 4*320+640=1920
    assert pad_overflow(1920, 1003, 88) == 913      # 上一轮实测的 91% 越界
    # step=900 → starts=[0,900,1800]，末窗右边界 2800，越界 880（占窗口 88%）
    assert pad_overflow(1920, 1000, 100) == 880
    # 与等步长网格的末窗右边界严格一致
    for length, size, ov in ((1920, 699, 88), (1920, 963, 10), (1080, 584, 88), (600, 300, 60)):
        starts = window_starts(length, size, ov, True)
        assert pad_overflow(length, size, ov) == starts[-1] + size - length, (length, size, ov)


def test_size_range_for_count_fill_caps_pad_at_thirty_percent():
    """开启填充时尺寸区间必须排除补边超过窗口 30% 的组合（与暴力枚举对拍）。"""
    from labelforge.engine.augmenter.crop import (
        MAX_WINDOW, pad_overflow, size_range_for_count, window_starts,
    )

    for length in (600, 1080, 1920, 2000):
        for count in (2, 3, 4):
            for ov in (0, 40, 88, 200):
                if ov >= length:
                    continue
                want = [c for c in range(ov + 1, MAX_WINDOW + 1)
                        if len(window_starts(length, c, ov, True)) == count
                        and pad_overflow(length, c, ov) * 100 <= 30 * c]
                lo, hi = size_range_for_count(length, count, ov, fill=True)
                if not want:
                    assert hi < lo, (length, count, ov, lo, hi)
                else:
                    assert (lo, hi) == (want[0], want[-1]), \
                        (length, count, ov, lo, hi, want[0], want[-1])
                    # 区间端点本身必须合规，且区间比不设上限时更窄（或有界）
                    assert pad_overflow(length, hi, ov) * 100 <= 30 * hi, (length, count, ov, hi)


def test_size_range_for_count_without_fill_is_unchanged_by_pad_cap():
    """不开填充时区间不该被 30% 上限动过：上限只能收窄、不能放宽。"""
    from labelforge.engine.augmenter.crop import size_range_for_count

    for length in (600, 1080, 1920, 2000):
        for count in (2, 3, 4):
            for ov in (0, 40, 88, 200):
                if ov >= length:
                    continue
                plain = size_range_for_count(length, count, ov)
                with_cap = size_range_for_count(length, count, ov, fill=True)
                assert plain[0] <= with_cap[0] and with_cap[1] <= plain[1], \
                    (length, count, ov, plain, with_cap)


def test_overlap_range_for_count_fill_caps_pad_at_thirty_percent():
    """锁「尺寸+数量」时重叠区间同样受 30% 补边上限约束。"""
    from labelforge.engine.augmenter.crop import (
        overlap_range_for_count, pad_overflow, window_starts,
    )

    for length in (600, 1080, 1920):
        for count in (2, 3, 4):
            for size in (300, 500, 640, 699, 1003):
                if size >= length:
                    continue
                want = [ov for ov in range(0, size)
                        if len(window_starts(length, size, ov, True)) == count
                        and pad_overflow(length, size, ov) * 100 <= 30 * size]
                lo, hi = overlap_range_for_count(length, count, size, fill=True)
                if not want:
                    assert hi < lo, (length, count, size, lo, hi)
                else:
                    assert (lo, hi) == (want[0], want[-1]), \
                        (length, count, size, lo, hi, want[0], want[-1])


def test_crop_dialog_fill_on_narrows_size_range_to_pad_cap(tmp_path):
    """开启填充后，锁「重叠+数量」的尺寸区间要收紧到补边不超过 30%。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import (
    MAX_WINDOW, pad_overflow, size_range_for_count, window_starts,
)

dlg._lock_ov.setChecked(True)
dlg._lock_cnt.setChecked(True)
app.processEvents()
plain = size_range_for_count(600, dlg._cols.value(), dlg._ov.value())
assert (dlg._w.minimum(), dlg._w.maximum()) == plain, "不开填充时区间不该被 30% 动过"

dlg._fill.setChecked(True)
app.processEvents()
lo, hi = size_range_for_count(600, dlg._cols.value(), dlg._ov.value(), fill=True)
assert (dlg._w.minimum(), dlg._w.maximum()) == (lo, hi), \
    (dlg._w.minimum(), dlg._w.maximum(), lo, hi)
assert hi < plain[1], (hi, plain[1])          # 确实收紧了

# 区间端点（最坏情况）补边必须刚好不超 30%
for c in (dlg._w.minimum(), (dlg._w.minimum() + dlg._w.maximum()) // 2, dlg._w.maximum()):
    dlg._w.setValue(c)
    cw, ov = dlg._w.value(), dlg._ov.value()
    assert len(window_starts(600, cw, ov, True)) == 3, (c, cw)
    assert pad_overflow(600, cw, ov) * 100 <= 30 * cw, (c, cw, pad_overflow(600, cw, ov))
print("OK_FILL_PAD_CAP")
'''
    assert "OK_FILL_PAD_CAP" in _run_real_qt(script)


def test_crop_dialog_toggling_fill_back_restores_plain_range(tmp_path):
    """关掉填充后区间要回到不含 30% 上限的版本，不能把收紧后的范围留在控件上。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import size_range_for_count

dlg._lock_ov.setChecked(True)
dlg._lock_cnt.setChecked(True)
plain = size_range_for_count(600, dlg._cols.value(), dlg._ov.value())

dlg._fill.setChecked(True)
app.processEvents()
assert dlg._w.maximum() < plain[1]

dlg._fill.setChecked(False)
app.processEvents()
assert (dlg._w.minimum(), dlg._w.maximum()) == plain, \
    (dlg._w.minimum(), dlg._w.maximum(), plain)
print("OK_FILL_TOGGLE_BACK")
'''
    assert "OK_FILL_TOGGLE_BACK" in _run_real_qt(script)


def test_exact_overlap_derives_flush_grid():
    """恰好铺满的唯一解：n·c−(n−1)·ov=L 反解重叠，且该重叠真能让网格贴边。"""
    from labelforge.engine.augmenter.crop import exact_overlap_for_flush, window_starts

    assert exact_overlap_for_flush(1920, 3, 800) == 240    # 3*800-2*240=1920
    assert exact_overlap_for_flush(1080, 2, 660) == 240    # 2*660-240=1080
    assert exact_overlap_for_flush(1920, 3, 640) == 0      # 3*640=1920
    assert exact_overlap_for_flush(600, 3, 240) == 60      # 3*240-2*60=600
    # 无解（尺寸盖不满或超界）返回 None，调用方据此判定不可行
    assert exact_overlap_for_flush(1920, 3, 500) is None   # 3*500=1500 < 1920，需负重叠
    assert exact_overlap_for_flush(100, 2, 200) is None    # 窗口比轴还长
    # 反解出的重叠必须真的让末窗贴边
    for L, n, c in ((1920, 3, 800), (1080, 2, 660), (600, 3, 240), (1920, 4, 540)):
        ov = exact_overlap_for_flush(L, n, c)
        assert ov is not None, (L, n, c)
        starts = window_starts(L, c, ov)
        assert len(starts) == n, (L, n, c, ov, starts)
        assert starts[-1] + c == L, (L, n, c, ov, starts)


def test_size_range_for_count_without_fill_stays_flush():
    """不开填充时区间内每个尺寸都必须真能贴边（这是互相反推成立的前提）。"""
    from labelforge.engine.augmenter.crop import (
        size_range_for_count, window_starts,
    )

    for length in (600, 1080, 1920):
        for count in (2, 3, 4):
            for ov in (0, 40, 88, 200):
                if ov >= length:
                    continue
                lo, hi = size_range_for_count(length, count, ov)
                if hi < lo:
                    continue
                for c in (lo, hi):
                    starts = window_starts(length, c, ov)
                    assert len(starts) == count, (length, count, ov, c)
                    assert starts[-1] + c == length, (length, count, ov, c, starts)


def test_crop_dialog_no_lock_derives_overlap_from_size_without_fill(tmp_path):
    """不开填充 + 无锁定：改宽高要反推重叠，两轴保持恰好铺满（同步）。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import window_starts

assert not dlg._fill.isChecked()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

for w, h in ((200, 200), (240, 230), (280, 260), (300, 275)):
    dlg._w.setValue(w)
    dlg._h.setValue(h)
    cw, ch, ov = dlg._w.value(), dlg._h.value(), dlg._ov.value()
    # 宽轴：以重叠为共用值恰好铺满；高轴由该重叠反推
    starts_w = window_starts(600, cw, ov)
    assert len(starts_w) == 3, (w, h, cw, ov, starts_w)
    assert starts_w[-1] + cw == 600, (w, h, cw, ov, starts_w)
    starts_h = window_starts(400, ch, ov)
    assert starts_h[-1] + ch == 400, (w, h, ch, ov, starts_h)
print("OK_DERIVE_OV_FROM_SIZE")
'''
    assert "OK_DERIVE_OV_FROM_SIZE" in _run_real_qt(script)


def test_crop_dialog_no_lock_derives_size_from_overlap_without_fill(tmp_path):
    """不开填充 + 无锁定：改重叠要反推宽高，仍保持恰好铺满。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import window_starts

assert not dlg._fill.isChecked()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

for ov in (0, 2, 5, 30, 60, 90):
    dlg._ov.setValue(ov)
    cw, ch, got = dlg._w.value(), dlg._h.value(), dlg._ov.value()
    starts_w = window_starts(600, cw, got)
    # ov=2/5 等使 600+2ov 不被 3 整除 → 无精确解，走 ceil 回退，
    # 此时 round 会把窗口取小、多凑一列（202 vs 201 恰好跨过 3 窗边界）
    assert len(starts_w) == 3, (ov, cw, got, starts_w)
    assert starts_w[-1] + cw == 600, (ov, cw, got, starts_w)
    starts_h = window_starts(400, ch, got)
    assert len(starts_h) == 2, (ov, ch, got, starts_h)
    assert starts_h[-1] + ch == 400, (ov, ch, got, starts_h)
print("OK_DERIVE_SIZE_FROM_OV")
'''
    assert "OK_DERIVE_SIZE_FROM_OV" in _run_real_qt(script)


def test_crop_dialog_fill_on_does_not_derive_overlap_from_size(tmp_path):
    """开填充：动宽高不反推重叠（不够就填充），重叠保持用户/原值不动。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg._fill.setChecked(True)
app.processEvents()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

before = dlg._ov.value()
for w in (700, 800, 900):
    dlg._w.setValue(w)
    assert dlg._ov.value() == before, (w, dlg._ov.value(), before)
print("OK_FILL_NO_DERIVE_FROM_SIZE")
'''
    assert "OK_FILL_NO_DERIVE_FROM_SIZE" in _run_real_qt(script)


def test_crop_dialog_fill_on_does_not_derive_size_from_overlap(tmp_path):
    """开填充：动重叠也不反推宽高，不够就填充。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg._fill.setChecked(True)
app.processEvents()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

before_w, before_h = dlg._w.value(), dlg._h.value()
for ov in (10, 50, 90):
    dlg._ov.setValue(ov)
    assert (dlg._w.value(), dlg._h.value()) == (before_w, before_h), \
        (ov, dlg._w.value(), dlg._h.value(), before_w, before_h)
print("OK_FILL_NO_DERIVE_FROM_OV")
'''
    assert "OK_FILL_NO_DERIVE_FROM_OV" in _run_real_qt(script)


def test_crop_dialog_pad_cap_blocks_oversized_pad_without_locks(tmp_path):
    """无锁定 + 开填充：30% 补边上限也必须生效，不能调出几乎全补边的窗口。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import pad_overflow

dlg._fill.setChecked(True)
app.processEvents()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

# 600×400 图、3 列：把宽度顶到会超限的值，实际必须被限制在合规范围内
for w in (200, 240, 300, 400, 599):
    dlg._w.setValue(w)
    cw, ov = dlg._w.value(), dlg._ov.value()
    pad = pad_overflow(600, cw, ov)
    assert pad * 100 <= 30 * cw, (w, cw, ov, pad)
print("OK_PAD_CAP_NO_LOCK")
'''
    assert "OK_PAD_CAP_NO_LOCK" in _run_real_qt(script)


# ---------- 填充模式的 30% 补边硬墙 ----------

def test_pad_compliant_size_run_matches_brute_force():
    """合规尺寸段必须与暴力枚举完全一致：段内每个值补边都 ≤30%，段外紧邻值都超标。

    硬墙依赖「段」这个结构：把输入框范围钳到当前值所在的段，用户就撞不到超限值。
    段算错一格，墙就会漏掉一个超限值或多挡一个合规值。
    """
    from labelforge.engine.augmenter.crop import (
        MAX_WINDOW, pad_compliant_size_run, pad_overflow,
    )

    def ok(length, size, ov):
        return pad_overflow(length, size, ov) * 100 <= 30 * size

    for length in (400, 600, 1080, 1920):
        for ov in (0, 5, 20, 60, 88, 120, 200):
            if ov >= length:
                continue
            want = [c for c in range(ov + 1, MAX_WINDOW + 1) if ok(length, c, ov)]
            for probe in (ov + 1, length // 2, length, length + 100):
                lo, hi = pad_compliant_size_run(length, probe, ov)
                seg = [c for c in want if lo <= c <= hi]
                if not want:
                    assert hi < lo, (length, probe, ov)
                    continue
                # probe 违规时返回最近的合规段；合规时返回其所在段
                assert lo <= hi, (length, probe, ov, lo, hi)
                assert lo in want and hi in want, (length, probe, ov, lo, hi)
                # 段内连续且没有空洞
                assert seg == list(range(lo, hi + 1)), (length, probe, ov, lo, hi)
                # 段边界外紧邻值必须超标（否则段被切小了）
                if lo - 1 >= ov + 1:
                    assert not ok(length, lo - 1, ov), (length, probe, ov, lo)
                if hi + 1 <= MAX_WINDOW:
                    assert not ok(length, hi + 1, ov), (length, probe, ov, hi)


def test_nearest_pad_compliant_size_prefers_closest_then_smaller():
    """超限值回退到最近合规值：距离相同取较小值，且绝不改出更远的解。"""
    from labelforge.engine.augmenter.crop import (
        MAX_WINDOW, nearest_pad_compliant_size, pad_overflow,
    )

    def ok(length, size, ov):
        return pad_overflow(length, size, ov) * 100 <= 30 * size

    for length in (400, 600, 1080, 1920):
        for ov in (0, 20, 60, 88, 200):
            if ov >= length:
                continue
            want = [c for c in range(ov + 1, MAX_WINDOW + 1) if ok(length, c, ov)]
            # probe 必须落在合法域 size > ov 内（域外的值无合规可言）
            for probe in (ov + 1, ov + 7, max(ov + 1, length // 3), length, length + 50):
                got = nearest_pad_compliant_size(length, probe, ov)
                assert got in want, (length, probe, ov, got)
                best = min(abs(c - probe) for c in want)
                assert abs(got - probe) == best, (length, probe, ov, got, best)
                # 等距取较小
                assert got == min(c for c in want if abs(c - probe) == best), \
                    (length, probe, ov, got)
                # 已经合规时必须原样返回（不许无谓改动用户输入）
                if ok(length, probe, ov):
                    assert got == probe, (length, probe, ov, got)


def test_pad_compliant_overlap_run_uses_both_axes():
    """重叠硬墙要同时满足两轴：返回的段内每个值两轴补边都合规，段外紧邻值至少一轴超标。

    重叠两轴共用一个值，只按单轴算墙会让另一轴偷偷超限（实测 93%/90%）。
    """
    from labelforge.engine.augmenter.crop import (
        pad_compliant_overlap_run, pad_overflow,
    )

    def ok(length, size, ov):
        return pad_overflow(length, size, ov) * 100 <= 30 * size

    for iw, ih in ((1920, 1080), (1280, 720), (600, 400), (800, 600)):
        for w, h in ((300, 200), (200, 150), (640, 480), (330, 192)):
            if w >= iw or h >= ih:
                continue
            want = [ov for ov in range(0, min(w, h))
                    if ok(iw, w, ov) and ok(ih, h, ov)]
            for probe in (0, min(w, h) // 4, min(w, h) // 2, min(w, h) - 1):
                lo, hi = pad_compliant_overlap_run(iw, ih, w, h, probe)
                if not want:
                    assert hi < lo, (iw, ih, w, h, probe)
                    continue
                assert lo <= hi, (iw, ih, w, h, probe, lo, hi)
                assert lo in want and hi in want, (iw, ih, w, h, probe, lo, hi)
                seg = [ov for ov in want if lo <= ov <= hi]
                assert seg == list(range(lo, hi + 1)), (iw, ih, w, h, probe, lo, hi)
                if lo - 1 >= 0:
                    assert not (ok(iw, w, lo - 1) and ok(ih, h, lo - 1)), \
                        (iw, ih, w, h, lo)
                if hi + 1 < min(w, h):
                    assert not (ok(iw, w, hi + 1) and ok(ih, h, hi + 1)), \
                        (iw, ih, w, h, hi)


def test_nearest_pad_compliant_overlap_prefers_closest_then_smaller():
    """重叠超限回退到最近的两轴同时合规值。"""
    from labelforge.engine.augmenter.crop import (
        nearest_pad_compliant_overlap, pad_overflow,
    )

    def ok(length, size, ov):
        return pad_overflow(length, size, ov) * 100 <= 30 * size

    for iw, ih in ((1920, 1080), (600, 400), (800, 600)):
        for w, h in ((300, 200), (200, 150), (640, 480)):
            if w >= iw or h >= ih:
                continue
            want = [ov for ov in range(0, min(w, h))
                    if ok(iw, w, ov) and ok(ih, h, ov)]
            if not want:
                continue
            for probe in (0, 10, 60, 100, min(w, h) - 1):
                got = nearest_pad_compliant_overlap(iw, ih, w, h, probe)
                assert got in want, (iw, ih, w, h, probe, got)
                best = min(abs(v - probe) for v in want)
                assert abs(got - probe) == best, (iw, ih, w, h, probe, got, best)
                assert got == min(v for v in want if abs(v - probe) == best), \
                    (iw, ih, w, h, probe, got)
                if probe in want:
                    assert got == probe, (iw, ih, w, h, probe, got)


def test_crop_dialog_fill_on_clamps_size_inputs_to_compliant_run(tmp_path):
    """开填充时宽高输入框范围被钳到「当前值所在的合规段」：拖不出超限值。

    合规集合在固定重叠下是锯齿状多段，只把整体上下界钳到最外端会让用户停在
    中间的空洞里（600 图上 ov=60 有 15 段）。硬墙须按「当前所在段」设置范围。
    """
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import pad_overflow, pad_compliant_size_run

# 300x200/60 在 600x400 上本就超限（宽 93%/高 90%）
dlg._fill.setChecked(True)
app.processEvents()

for spin, length in ((dlg._w, 600), (dlg._h, 400)):
    lo, hi = pad_compliant_size_run(length, spin.value(), dlg._ov.value())
    assert (spin.minimum(), spin.maximum()) == (lo, hi), \
        (length, spin.value(), spin.minimum(), spin.maximum(), lo, hi)
    assert lo <= spin.value() <= hi, (length, spin.value(), lo, hi)
    # 段内任意值都能调进去，且都合规
    for v in (lo, (lo + hi) // 2, hi):
        spin.setValue(v)
        assert spin.value() == v, (length, v, spin.value())
        assert pad_overflow(length, spin.value(), dlg._ov.value()) * 100 \
            <= 30 * spin.value(), (length, spin.value())
print("OK_SIZE_WALL")
'''
    assert "OK_SIZE_WALL" in _run_real_qt(script)


def test_crop_dialog_fill_on_clamps_overlap_input_to_joint_compliant_run(tmp_path):
    """开填充时重叠输入框范围钳到「两轴同时合规」的段，拖不出单轴超限值。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import pad_overflow, pad_compliant_overlap_run

dlg._fill.setChecked(True)
app.processEvents()
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()

rw, rh = dlg._ref_size()
w, h = dlg._w.value(), dlg._h.value()
lo, hi = pad_compliant_overlap_run(rw, rh, w, h, dlg._ov.value())
assert (dlg._ov.minimum(), dlg._ov.maximum()) == (lo, hi), \
    (dlg._ov.minimum(), dlg._ov.maximum(), lo, hi, w, h)
for v in (lo, (lo + hi) // 2, hi):
    dlg._ov.setValue(v)
    assert dlg._ov.value() == v, (v, dlg._ov.value())
    w2, h2 = dlg._w.value(), dlg._h.value()
    assert pad_overflow(rw, w2, dlg._ov.value()) * 100 <= 30 * w2, (v, w2)
    assert pad_overflow(rh, h2, dlg._ov.value()) * 100 <= 30 * h2, (v, h2)
print("OK_OV_WALL")
'''
    assert "OK_OV_WALL" in _run_real_qt(script)


def test_crop_dialog_fill_on_snaps_oversized_values_on_toggle(tmp_path):
    """开填充那一刻若当前宽高已超限，回退到最近合规值，而不是留着违规值。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import (
    nearest_pad_compliant_size, pad_overflow,
)

assert not dlg._fill.isChecked()
before = (dlg._w.value(), dlg._h.value())
assert pad_overflow(600, before[0], dlg._ov.value()) * 100 > 30 * before[0], \
    "夹具前提：600 宽下 300 宽窗口应超限"

dlg._fill.setChecked(True)
app.processEvents()
rw, rh, ov = *dlg._ref_size(), dlg._ov.value()
want_w = nearest_pad_compliant_size(rw, before[0], ov)
want_h = nearest_pad_compliant_size(rh, before[1], ov)
assert (dlg._w.value(), dlg._h.value()) == (want_w, want_h), \
    ((dlg._w.value(), dlg._h.value()), (want_w, want_h))
assert pad_overflow(rw, dlg._w.value(), ov) * 100 <= 30 * dlg._w.value()
assert pad_overflow(rh, dlg._h.value(), ov) * 100 <= 30 * dlg._h.value()
# 回退发生了 → 摘要行必须给出提示，不能静默改值
dlg._refresh()          # 摘要是防抖渲染的，测试里显式刷新
assert "补边" in dlg._est.text() and "30" in dlg._est.text(), dlg._est.text()
print("OK_SNAP_ON_TOGGLE")
'''
    assert "OK_SNAP_ON_TOGGLE" in _run_real_qt(script)


def test_crop_dialog_fill_on_notifies_when_pad_clamped(tmp_path):
    """回退发生时摘要行必须提示（回归：_pad_clamped 曾是只写不读的死代码）。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import pad_overflow

dlg._fill.setChecked(True)
app.processEvents()
dlg._refresh()          # 摘要是防抖渲染的，测试里显式刷新
assert "补边" in dlg._est.text(), dlg._est.text()   # 开填充时已发生回退

# 手动调到一个本来合规的值 → 提示应消失（不能常驻）
w_lo, w_hi = dlg._w.minimum(), dlg._w.maximum()
assert w_hi > w_lo, (w_lo, w_hi)          # 段内要有可调空间，否则这次编辑是空操作
dlg._w.setValue(w_hi)
app.processEvents()
dlg._refresh()
w, h, ov = dlg._w.value(), dlg._h.value(), dlg._ov.value()
assert pad_overflow(600, w, ov) * 100 <= 30 * w, (w, ov)
assert pad_overflow(400, h, ov) * 100 <= 30 * h, (h, ov)
assert not dlg._pad_clamped, "调到合规值后不该还挂着「已吸附」标记"
assert "补边" not in dlg._est.text(), dlg._est.text()
print("OK_PAD_NOTICE")
'''
    assert "OK_PAD_NOTICE" in _run_real_qt(script)


def test_crop_dialog_fill_on_count_edit_keeps_count_and_cap(tmp_path):
    """开填充改数量：派生的尺寸若超限，回退必须保住数量且满足 30% 上限。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
from labelforge.engine.augmenter.crop import (
    pad_overflow, size_range_for_count, window_starts,
)

dlg._fill.setChecked(True)
app.processEvents()
dlg._ov.setValue(0)
app.processEvents()

for n in (2, 3, 4, 5, 6):
    dlg._cols.setValue(n)
    app.processEvents()
    w, ov = dlg._w.value(), dlg._ov.value()
    assert len(window_starts(600, w, ov, True)) == n, (n, w, ov)
    assert pad_overflow(600, w, ov) * 100 <= 30 * w, (n, w, ov, pad_overflow(600, w, ov))
    lo, hi = size_range_for_count(600, n, ov, fill=True)
    assert lo <= w <= hi, (n, w, lo, hi)
print("OK_COUNT_EDIT")
'''
    assert "OK_COUNT_EDIT" in _run_real_qt(script)


def test_crop_dialog_fill_on_template_over_cap_snaps_and_notifies(tmp_path):
    """套用模板时模板宽高在填充模式下超限：回退到最近合规值并提示。"""
    script = f'''
import os
os.environ["LABELFORGE_CONFIG_PATH"] = r"{tmp_path.as_posix()}/config.json"
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QPushButton
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.core.config_manager import add_crop_template
from labelforge.engine.augmenter.crop import (
    nearest_pad_compliant_size, pad_overflow,
)

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "t.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
add_crop_template({{"w": 800, "h": 640, "overlap": 200, "min_visible": 0.7,
                   "fill": True, "fill_rgb": [114, 114, 114]}})
dlg = CropConfigDialog([src], {{0: []}},
                       {{"w": 300, "h": 200, "overlap": 60, "min_visible": 0.7,
                         "fill": True}})
btn = [b for b in dlg.findChildren(QPushButton) if "640" in b.text()][0]
btn.click()
app.processEvents()

s = dlg.spec()
assert s["fill"] is True, s
assert pad_overflow(600, s["w"], s["overlap"]) * 100 <= 30 * s["w"], s
assert pad_overflow(400, s["h"], s["overlap"]) * 100 <= 30 * s["h"], s
want_w = nearest_pad_compliant_size(600, 800, s["overlap"])
assert s["w"] == want_w, (s["w"], want_w)
dlg._refresh()          # 摘要是防抖渲染的，测试里显式刷新
assert "补边" in dlg._est.text() and "30" in dlg._est.text(), dlg._est.text()
print("OK_TEMPLATE_SNAP")
'''
    assert "OK_TEMPLATE_SNAP" in _run_real_qt(script)


def test_crop_dialog_no_fill_keeps_full_ranges(tmp_path):
    """不开填充时不该出现 30% 相关的收窄或提示（回归保护）。"""
    script = _DIALOG_BOOT.format(d=tmp_path.as_posix()) + '''
dlg._cols.setValue(3)
dlg._rows.setValue(2)
app.processEvents()
assert (dlg._w.minimum(), dlg._w.maximum()) == (8, 8192), \
    (dlg._w.minimum(), dlg._w.maximum())
assert "补边" not in dlg._est.text(), dlg._est.text()

# 关掉填充后要能回到完整范围（不能把硬墙留在控件上）
dlg._fill.setChecked(True)
app.processEvents()
dlg._fill.setChecked(False)
app.processEvents()
assert (dlg._w.minimum(), dlg._w.maximum()) == (8, 8192), \
    (dlg._w.minimum(), dlg._w.maximum())
assert (dlg._h.minimum(), dlg._h.maximum()) == (8, 8192), \
    (dlg._h.minimum(), dlg._h.maximum())
print("OK_NO_FILL_FULL_RANGE")
'''
    assert "OK_NO_FILL_FULL_RANGE" in _run_real_qt(script)


def test_crop_dialog_fill_on_prefers_moving_overlap_over_resizing(tmp_path):
    """开填充时超限应改重叠、保住用户设的宽高（宽高是自变量）。

    回归：若只靠「硬墙钳宽高」，用户设的 400×300 会被改成 400×294——
    宽高本该由用户说了算，该动的是重叠。
    """
    script = f'''
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
from labelforge.ui.crop_dialog import CropConfigDialog
from labelforge.engine.augmenter.crop import pad_overflow

d = r"{tmp_path.as_posix()}"
src = os.path.join(d, "a.png")
QImage(600, 400, QImage.Format_RGB888).save(src)
# 构造时就带一个超限的组合（400×300 / 重叠100 在高轴补边 33%）
spec = {{"w": 400, "h": 300, "overlap": 100, "min_visible": 0.7, "fill": True}}
dlg = CropConfigDialog([src], {{0: []}}, spec)
app.processEvents()

w, h, ov = dlg._w.value(), dlg._h.value(), dlg._ov.value()
# 宽高是自变量：用户设的 400×300 必须原样保住，该动的是重叠
assert (w, h) == (400, 300), (w, h)
assert ov != 100, "重叠本应被吸附到合规值"
assert pad_overflow(600, w, ov) * 100 <= 30 * w, (ov, pad_overflow(600, w, ov))
assert pad_overflow(400, h, ov) * 100 <= 30 * h, (ov, pad_overflow(400, h, ov))
assert dlg._pad_clamped is True
print("OK_PREFER_OV")
'''
    assert "OK_PREFER_OV" in _run_real_qt(script)


def test_pad_compliant_overlap_run_respects_half_window_limit():
    """联合合规段必须叠加半窗上限：否则 setRange 会把范围放宽到上限之外。

    回归：(60,62) 两轴的合规段是 (30,31)，而半窗上限为 30，不叠加就放行了 31。
    """
    from labelforge.engine.augmenter.crop import (
        pad_compliant_overlap_run, pad_overflow,
    )

    def ok(L, c, ov):
        return pad_overflow(L, c, ov) * 100 <= 30 * c

    for iw, ih, w, h in ((600, 400, 60, 62), (600, 400, 65, 62), (600, 400, 70, 62),
                         (1920, 1080, 200, 120)):
        half = min(w, h) // 2
        lo, hi = pad_compliant_overlap_run(iw, ih, w, h, half, half)
        if hi < lo:
            continue
        assert hi <= half, (iw, ih, w, h, half, lo, hi)
        assert lo <= half, (iw, ih, w, h, half, lo, hi)
        # 段内每个值都必须两轴合规（且在上限内）
        for ov in range(lo, hi + 1):
            assert ok(iw, w, ov) and ok(ih, h, ov), (iw, ih, w, h, ov)


def test_nearest_pad_compliant_size_breaks_segment_ties_toward_smaller():
    """两段等距时固定取较小段（结果必须确定，不能随遍历顺序漂移）。

    L=400/ov=0 时 size=3 到段 (1,2) 与 (4,5) 等距，必须稳定回 2 而非 4。
    """
    from labelforge.engine.augmenter.crop import (
        nearest_pad_compliant_size, pad_overflow,
    )

    def ok(L, c, ov):
        return pad_overflow(L, c, ov) * 100 <= 30 * c

    # 等距夹具：两侧各一段，距离相同
    for L, ov, v, small, big in ((400, 0, 3, 2, 4), (400, 0, 9, 8, 10),
                                 (400, 0, 18, 16, 20), (400, 0, 26, 25, 27)):
        assert not ok(L, v, ov), (L, ov, v)
        got = nearest_pad_compliant_size(L, v, ov)
        assert got == small, (L, ov, v, got, small, big)
        # 两侧都确实合规，且距离真的相同（否则这个夹具没意义）
        assert ok(L, small, ov) and ok(L, big, ov), (L, ov, small, big)
        assert abs(small - v) == abs(big - v), (L, ov, v, small, big)
        # 多次调用结果一致
        assert all(nearest_pad_compliant_size(L, v, ov) == got for _ in range(5))
