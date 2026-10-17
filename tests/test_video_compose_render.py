"""④B 合成 + 标签框叠加：可选的绘制、类别筛选、无标注计数、坏 JSON 处理。

测试分两层（同项目其它 Qt 相关测试的约定）：

- **进程内**：conftest 把 PySide6 换成了假模块，所以这里把
  `video_render.draw_boxes` 换成一个纯 numpy 的替身，专门验证**接线**——
  是否按需调用、传参对不对、计数是否准确、坏标注是否报错。
- **子进程 + 真实 Qt**：验证真绘制（框的位置、类别色、筛选、中文标签）
  确实进了编码后的视频。

绘制是可选功能：不传 render 时合成必须与原来的纯合成走同一条路径，
一个 sidecar 都不该读（否则坏 JSON 会在用户没开叠加时报错）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelforge.engine.video_io import (
    Manifest, ManifestFrame, VideoError, compose_video, output_dir_for,
    probe, read_image, write_image, write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 64, 48


# ---------- 素材 ----------

def _make_images(tmp_path, count=4, duration=2.0):
    """造一个「像抽帧产物」的目录：图片 + 合法清单。"""
    path = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (FRAME_W, FRAME_H))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((FRAME_H, FRAME_W, 3), np.uint8))
    writer.release()

    info = probe(str(path))
    out_dir = output_dir_for(str(path))
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(count):
        name = f"src_{i + 1:06d}.png"
        write_image(os.path.join(out_dir, name),
                    np.zeros((FRAME_H, FRAME_W, 3), np.uint8), ext="png")
        # source_frame 与 timestamp 必须自洽（源 10 FPS：第 n 帧在 n/10 秒）
        source_frame = i * 5
        frames.append(ManifestFrame(file=name, source_frame=source_frame,
                                    timestamp=source_frame / info.fps,
                                    sample_time=source_frame / info.fps))
    manifest = Manifest(
        source=info.path, source_name="src.mp4", width=info.width,
        height=info.height, source_fps=info.fps, frame_count=info.frame_count,
        duration=info.duration, target_fps=2.0, start_time=0.0,
        end_time=duration, mode="sample", complete=True, frames=frames)
    manifest_path = os.path.join(out_dir, "video_manifest.json")
    write_manifest(manifest_path, manifest)
    return out_dir, manifest_path


def _image_names(out_dir):
    return sorted(p for p in os.listdir(out_dir) if p.endswith(".png"))


def _sidecar_for(image_path, shapes):
    json_path = os.path.splitext(str(image_path))[0] + ".json"
    payload = {"version": "5.5.0", "flags": {}, "shapes": list(shapes),
               "imagePath": os.path.basename(str(image_path)), "imageData": None}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return json_path


def _rect_shape(label, x, y, w, h, score=None):
    shape = {"label": label,
             "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
             "group_id": None, "shape_type": "rectangle", "flags": {}}
    if score is not None:
        shape["score"] = score
    return shape


def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"读不开合成视频：{path}"
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _hex_to_rgb(value):
    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def _count_color(frame_bgr, rgb, tol=40):
    """数一帧里接近某个 RGB 的像素个数（有损编码，不能要求颜色完全相等）。"""
    rgb_frame = frame_bgr[:, :, ::-1].astype(int)
    diff = np.abs(rgb_frame - np.array(rgb, dtype=int))
    return int((diff.max(axis=2) <= tol).sum())


# ---------- 假绘制器：只验证接线 ----------

class _FakeDraw:
    """纯 numpy 版 draw_boxes 替身：把框填成类别色，并记录调用参数。"""

    def __init__(self):
        self.calls = []

    def __call__(self, frame_rgb, boxes, spec=None):
        labels = None if spec is None else spec.labels
        colors = {} if spec is None else (spec.label_colors or {})
        self.calls.append({
            "boxes": [dict(b) for b in boxes],
            "labels": None if labels is None else set(labels),
            "show_labels": bool(spec.show_labels) if spec else False,
            "show_scores": bool(spec.show_scores) if spec else False,
        })
        out = frame_rgb.copy()
        for box in boxes:
            if labels is not None and box.get("label") not in labels:
                continue
            color = _hex_to_rgb(colors.get(box.get("label"), "#00FF80"))
            x, y = int(box["x"]), int(box["y"])
            w, h = int(box["width"]), int(box["height"])
            out[max(0, y):y + h, max(0, x):x + w] = color
        return out


@pytest.fixture
def fake_draw(monkeypatch):
    import labelforge.engine.video_render as video_render
    fake = _FakeDraw()
    monkeypatch.setattr(video_render, "draw_boxes", fake)
    return fake


def _spec(**kw):
    from labelforge.engine.video_render import RenderSpec
    return RenderSpec(**kw)


# ---------- 不传 render：与纯合成同一路径 ----------

def test_compose_without_render_does_not_read_sidecars(tmp_path, monkeypatch):
    """不给 render 时合成不该读任何 sidecar。"""
    import labelforge.engine.video_io as video_io

    out_dir, manifest_path = _make_images(tmp_path, count=4)
    called = []

    def _boom(image_path):
        called.append(image_path)
        raise AssertionError(f"没开叠加却读了标注：{image_path}")

    monkeypatch.setattr(video_io, "_read_sidecar_boxes_for_compose", _boom)
    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "plain.mp4"))

    assert result.image_count == 4
    assert called == [], "没开叠加却读了标注"
    assert result.unlabeled_count == 0


# ---------- 接线：调用次数、传参、计数、错误 ----------

def test_compose_draws_once_per_image_not_per_frame(tmp_path, fake_draw):
    """标准帧率下一张图会写多个输出帧，绘制只该做一次（整帧缓存复用）。"""
    out_dir, manifest_path = _make_images(tmp_path, count=2, duration=2.0)
    for name in _image_names(out_dir):
        _sidecar_for(os.path.join(out_dir, name),
                     [_rect_shape("person", 6, 6, 24, 16)])

    spec = _spec(label_colors={"person": "#FF0000"})
    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "cache.mp4"),
                           render=spec, standard_fps=30)

    assert result.total_frames > result.image_count, "样例应展开成多帧"
    assert len(fake_draw.calls) == result.image_count, \
        f"每张图只该画一次，实际画了 {len(fake_draw.calls)} 次"


def test_compose_draws_and_counts_the_first_image_too(tmp_path, fake_draw):
    """首图不能漏画也不能漏计数。

    回归：首图在开编码器之前就预读好了，曾直接进缓存、绕过绘制那一步，
    于是第一张的框永远不画、无标注计数也少一张。
    """
    out_dir, manifest_path = _make_images(tmp_path, count=3, duration=1.5)
    names = _image_names(out_dir)
    _sidecar_for(os.path.join(out_dir, names[0]), [_rect_shape("person", 6, 6, 20, 12)])

    spec = _spec(label_colors={"person": "#FF0000"})
    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "first.mp4"), render=spec)

    # 只有第一张有标注，所以只该画一次；另外两张没有 sidecar
    assert len(fake_draw.calls) == 1, f"只该画有标注的那一张：{len(fake_draw.calls)}"
    assert result.unlabeled_count == 2, result.unlabeled_count
    drawn_frames = _read_frames(tmp_path / "first.mp4")
    assert _count_color(drawn_frames[0][:20, :30], (255, 0, 0)) > 10, \
        "第一张的框没画进视频"


def test_compose_passes_render_options_through(tmp_path, fake_draw):
    """渲染规格（类别筛选/类别名/置信度开关）要原样传到绘制器。"""
    out_dir, manifest_path = _make_images(tmp_path, count=2, duration=1.0)
    for name in _image_names(out_dir):
        _sidecar_for(os.path.join(out_dir, name),
                     [_rect_shape("person", 6, 6, 20, 12, score=0.9)])

    spec = _spec(label_colors={"person": "#FF0000"}, labels={"person"},
                 show_labels=True, show_scores=True)
    compose_video(manifest_path=manifest_path,
                  output_path=str(tmp_path / "opts.mp4"), render=spec)

    assert fake_draw.calls, "绘制器没被调用"
    for call in fake_draw.calls:
        assert call["labels"] == {"person"}
        assert call["show_labels"] is True
        assert call["show_scores"] is True
        assert call["boxes"], "应把解析出的框交给绘制器"


def test_compose_reports_unlabeled_frames(tmp_path, fake_draw):
    """缺 sidecar 的图片按无标签画面写入，并报出无标注张数。"""
    out_dir, manifest_path = _make_images(tmp_path, count=4)
    for name in _image_names(out_dir)[:2]:
        _sidecar_for(os.path.join(out_dir, name),
                     [_rect_shape("person", 6, 6, 24, 16)])

    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "partial.mp4"),
                           render=_spec(label_colors={"person": "#FF0000"}))

    assert result.image_count == 4
    assert result.unlabeled_count == 2, result.unlabeled_count


def test_compose_empty_shapes_counts_as_unlabeled(tmp_path, fake_draw):
    """sidecar 在但 shapes 为空：画面上确实没框，计入无标注。"""
    out_dir, manifest_path = _make_images(tmp_path, count=2, duration=1.0)
    for name in _image_names(out_dir):
        _sidecar_for(os.path.join(out_dir, name), [])

    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "empty.mp4"),
                           render=_spec(label_colors={"person": "#FF0000"}))

    assert result.unlabeled_count == 2, result.unlabeled_count


def test_compose_no_boxes_when_nothing_selected(tmp_path, fake_draw):
    """一个类别都没勾 → 画面干净，但标注文件在，不算无标注。"""
    out_dir, manifest_path = _make_images(tmp_path, count=2, duration=1.0)
    for name in _image_names(out_dir):
        _sidecar_for(os.path.join(out_dir, name),
                     [_rect_shape("person", 6, 6, 24, 16)])

    result = compose_video(manifest_path=manifest_path,
                           output_path=str(tmp_path / "none.mp4"),
                           render=_spec(label_colors={"person": "#FF0000"},
                                        labels=set()))

    frames = _read_frames(tmp_path / "none.mp4")
    assert _count_color(frames[0], (255, 0, 0)) == 0, "没勾任何类别却画了框"
    assert result.unlabeled_count == 0


def test_compose_rejects_corrupt_sidecar(tmp_path, fake_draw):
    """sidecar 存在但损坏 → 报错，不能静默当成无标注。"""
    out_dir, manifest_path = _make_images(tmp_path, count=3)
    bad = os.path.splitext(os.path.join(out_dir, _image_names(out_dir)[1]))[0] + ".json"
    with open(bad, "w", encoding="utf-8") as f:
        f.write("{ 这不是合法 JSON")

    out_path = tmp_path / "corrupt.mp4"
    with pytest.raises(VideoError, match="标注"):
        compose_video(manifest_path=manifest_path, output_path=str(out_path),
                      render=_spec(label_colors={"person": "#FF0000"}))
    assert not os.path.exists(out_path), "失败不能留下正式输出"


def test_compose_rejects_sidecar_missing_shapes_key(tmp_path, fake_draw):
    """结构不对的 JSON（没有 shapes 列表）同样要报错。"""
    out_dir, manifest_path = _make_images(tmp_path, count=2, duration=1.0)
    bad = os.path.splitext(os.path.join(out_dir, _image_names(out_dir)[0]))[0] + ".json"
    with open(bad, "w", encoding="utf-8") as f:
        json.dump({"version": "5.5.0", "shapes": "不是列表"}, f)

    with pytest.raises(VideoError, match="结构"):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "badschema.mp4"),
                      render=_spec(label_colors={"person": "#FF0000"}))


def test_compose_keeps_original_colors(tmp_path, fake_draw):
    """BGR→RGB→BGR 往返：画面原有颜色不能被改掉。

    合成链路读到的帧是 BGR，绘制模块吃 RGB，画完要转回 BGR 再编码。
    顺序错一次整个视频色偏全反，而框的颜色看着还「对」，所以要单独钉。

    必须给一个真框：没有标注时 `_prepare` 会在绘制前就返回，颜色往返
    根本不执行，测了等于没测。
    """
    out_dir, manifest_path = _make_images(tmp_path, count=1, duration=1.0)
    name = _image_names(out_dir)[0]
    frame = read_image(os.path.join(out_dir, name))
    frame[6:20, 6:30] = (255, 0, 0)          # BGR 纯蓝，别被绘制链路改掉
    write_image(os.path.join(out_dir, name), frame, ext="png")
    # 框放在右下角，不盖住待检查的蓝色块
    _sidecar_for(os.path.join(out_dir, name), [_rect_shape("x", 40, 30, 18, 12)])

    compose_video(manifest_path=manifest_path,
                  output_path=str(tmp_path / "color.mp4"),
                  render=_spec(label_colors={"x": "#FFFF00"}))

    video_frame = _read_frames(tmp_path / "color.mp4")[0]
    patch = video_frame[10:16, 10:26].reshape(-1, 3).mean(axis=0)   # BGR
    assert patch[0] > 150 and patch[1] < 90 and patch[2] < 90, \
        f"原本的蓝色被改成了 {tuple(round(float(v)) for v in patch)}"
    # 框也得画上，证明绘制路径确实走过
    assert _count_color(video_frame[28:, 36:], (255, 255, 0)) > 10, \
        "框没画出来，这轮没真正走绘制路径"


# ---------- 子进程 + 真实 Qt：真绘制确实进了视频 ----------

def _run_real_qt(script):
    proc = subprocess.run([sys.executable, "-c", script % str(ROOT)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


_COMMON = '''
import json, os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import numpy as np
import cv2
from labelforge.engine import video_io as vio
from labelforge.engine.video_render import RenderSpec

TMP = tempfile.mkdtemp()
# 画面不能太小、线不能太细：mp4v 有损，小图 + 1px 线会把类别色抹掉，
# 那样测出来的「颜色不对」是编码噪声，不是绘制问题。
W, H = 160, 120
PEN = 3

def make(count, duration):
    src = os.path.join(TMP, "src.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    for _ in range(20):
        vw.write(np.zeros((H, W, 3), np.uint8))
    vw.release()
    info = vio.probe(src)
    out_dir = os.path.join(TMP, "frames_%%d" %% count)
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for i in range(count):
        name = "img_%%06d.png" %% (i + 1)
        vio.write_image(os.path.join(out_dir, name), np.zeros((H, W, 3), np.uint8), ext="png")
        sf = i * 5
        frames.append(vio.ManifestFrame(file=name, source_frame=sf,
                                        timestamp=sf / info.fps, sample_time=sf / info.fps))
    man = vio.Manifest(source=src, source_name="src.mp4", width=info.width,
                       height=info.height, source_fps=info.fps,
                       frame_count=info.frame_count, duration=info.duration,
                       target_fps=2.0, start_time=0.0, end_time=duration,
                       mode="sample", complete=True, frames=frames)
    path = os.path.join(out_dir, "video_manifest.json")
    vio.write_manifest(path, man)
    return out_dir, path

def sidecar(image_path, shapes):
    with open(os.path.splitext(image_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes,
                   "imagePath": os.path.basename(image_path), "imageData": None},
                  f, ensure_ascii=False)

def rect(label, x, y, w, h, score=None):
    s = {"label": label, "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
         "group_id": None, "shape_type": "rectangle", "flags": {}}
    if score is not None:
        s["score"] = score
    return s

def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return out

def count_rgb(frame_bgr, rgb, tol=40):
    d = np.abs(frame_bgr[:, :, ::-1].astype(int) - np.array(rgb, dtype=int))
    return int((d.max(axis=2) <= tol).sum())
'''


def test_real_qt_draws_boxes_into_video(tmp_path):
    """真实 Qt：框按 sidecar 位置与类别色画进合成视频，且位置不跑偏。"""
    result = _run_real_qt(_COMMON + '''
out_dir, manifest = make(4, 2.0)
for name in sorted(os.listdir(out_dir)):
    if name.endswith(".png"):
        sidecar(os.path.join(out_dir, name), [rect("person", 20, 20, 70, 50)])
spec = RenderSpec(label_colors={"person": "#FF0000"}, show_labels=False, pen_width=PEN)
out = os.path.join(TMP, "drawn.mp4")
res = vio.compose_video(manifest_path=manifest, output_path=out, render=spec)
frames = read_frames(out)
tl = count_rgb(frames[0][10:80, 10:100], (255, 0, 0))
br = count_rgb(frames[0][95:, 110:], (255, 0, 0))
print("RESULT", repr((res.image_count, len(frames), tl, br, res.unlabeled_count)))
''')
    image_count, frame_count, top_left, bottom_right, unlabeled = result
    assert image_count == 4 and frame_count >= 4
    assert top_left > 100, f"框没画到左上角：{top_left}"
    assert bottom_right == 0, f"框画到了不该出现的位置：{bottom_right}"
    assert unlabeled == 0, unlabeled


def test_real_qt_label_filter_and_scores(tmp_path):
    """真实 Qt：类别筛选生效，置信度开关控制分数是否上画面。"""
    result = _run_real_qt(_COMMON + '''
out_dir, manifest = make(2, 1.0)
names = [n for n in sorted(os.listdir(out_dir)) if n.endswith(".png")]
for name in names:
    sidecar(os.path.join(out_dir, name),
            [rect("keep", 10, 10, 50, 40, score=0.87), rect("drop", 100, 70, 50, 40)])

spec = RenderSpec(label_colors={"keep": "#FF0000", "drop": "#00FF00"},
                  labels={"keep"}, show_labels=False, pen_width=PEN)
out = os.path.join(TMP, "filtered.mp4")
vio.compose_video(manifest_path=manifest, output_path=out, render=spec)
f = read_frames(out)[0]
kept = count_rgb(f[:60, :70], (255, 0, 0))
dropped = count_rgb(f[60:, 90:], (0, 255, 0))

# 开类别名 + 置信度：绘制链路要能跑通，框还在
out2 = os.path.join(TMP, "labelled.mp4")
spec2 = RenderSpec(label_colors={"keep": "#FF0000"}, labels={"keep"},
                   show_labels=True, show_scores=True, pen_width=PEN)
vio.compose_video(manifest_path=manifest, output_path=out2, render=spec2)
f2 = read_frames(out2)[0]
print("RESULT", repr((kept, dropped, int(count_rgb(f2, (255, 0, 0))), f2.shape)))
''')
    kept, dropped, with_labels, shape = result
    assert kept > 100, f"勾选的类别没画：{kept}"
    assert dropped == 0, f"没勾的类别画出来了：{dropped}"
    assert with_labels > 0, "开类别名后框与标签条应还在"
    assert shape == (120, 160, 3), shape      # 子进程里的画面尺寸
