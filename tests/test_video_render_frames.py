"""④A 原视频逐帧推理编码：探测校验、逐帧推理绘制、编码与回读验收。

与 ④B 的区别：画面来自原视频的**全部**帧（不是抽帧图片），标注是现场推理
得到的，输出帧数/分辨率/帧率/时长都要与源一致。

测试分两层（同项目约定）：

- **进程内**：假 detector + 纯 numpy 假绘制，验证接线——每帧都推理、框来自
  推理、进度/取消、失败不留正式文件。
- **子进程 + 真实 Qt**：验证真绘制（类别色、中文标签）确实进了编码后的视频。
"""
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from labelforge.engine.video_io import VideoError, probe

ROOT = Path(__file__).resolve().parents[1]
FRAME_W, FRAME_H = 64, 48

# 真实 Qt 绘制用例要求帧够大、线够粗，否则 mp4v 有损会把类别色抹掉
QT_W, QT_H = 160, 120


# ---------- 素材 ----------

def _encode_marker(frame, index):
    """帧号写成 8 个高对比块，挺得过高斯有损压缩（同 test_video_io 的做法）。"""
    for bit in range(8):
        if (index >> bit) & 1:
            frame[8:24, 8 + bit * 6: 8 + bit * 6 + 5] = (255, 255, 255)


def _decode_marker(frame):
    value = 0
    for bit in range(8):
        if int(frame[16, 10 + bit * 6, 0]) > 127:
            value |= (1 << bit)
    return value


def _make_video(path, count=20, fps=10.0, size=(FRAME_W, FRAME_H)):
    """合成源视频：每帧画上帧号，便于断言输出帧与源帧一一对应。"""
    width, height = size
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    assert writer.isOpened(), f"VideoWriter 打不开：{path}"
    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        _encode_marker(frame, index)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return str(path)


class _FakeDetector:
    """假 detector：返回固定框，记录每帧收到的尺寸，便于断言逐帧都推理了。"""

    def __init__(self, boxes=None, iou_threshold=0.45):
        self.boxes = boxes if boxes is not None else [
            {"label": "person", "x": 10.0, "y": 10.0, "width": 20.0,
             "height": 20.0, "score": 0.9, "class_id": 0},
        ]
        self.iou_threshold = iou_threshold
        self.calls = []

    def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
        self.calls.append((int(rgb.shape[1]), int(rgb.shape[0])))
        return [dict(b) for b in self.boxes]


def _noop_resize(rgb, geometry, fill):
    """resize_fn 替身：原样返回（假 detector 不看内容，只记尺寸）。"""
    return rgb


def _fake_draw(boxes, spec, color=(255, 0, 0)):
    """纯 numpy 的绘制替身：在框的位置填一块纯色，验证「画了、画在哪」。"""
    def _draw(image_rgb, box_list, render_spec=None):
        out = np.asarray(image_rgb).copy()
        for box in box_list or []:
            x = int(box.get("x", 0))
            y = int(box.get("y", 0))
            w = int(box.get("width", 0))
            h = int(box.get("height", 0))
            out[max(0, y):y + h, max(0, x):x + w] = color
        return out
    return _draw


def _make_odd_size_video(path):
    """用 ffmpeg 造一个真·奇数尺寸源（cv2 自己写不出，会被截成偶数）。

    本机没有 ffmpeg 时返回 None，由调用方 skip——这是外部素材缺失，
    计划允许注明原因 skip；cv2 缺失不允许 skip。
    """
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=s=65x49:d=1:r=10",
         "-c:v", "ffv1", "-pix_fmt", "yuv444p", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not path.exists():
        return None
    return str(path)


def _bump_frame_count(info, delta):
    """把探测到的帧数改大，模拟「探测说 15 帧、实际只有 10 帧」。"""
    from labelforge.engine.video_io import VideoInfo

    return VideoInfo(path=info.path, fps=info.fps,
                     frame_count=info.frame_count + delta,
                     width=info.width, height=info.height,
                     duration=info.duration)


# ---------- 探测与参数校验 ----------

def test_render_video_rejects_missing_source(tmp_path):
    """源视频不存在 → 报错，不留空输出。"""
    from labelforge.engine.video_io import render_video

    with pytest.raises(VideoError, match="不存在"):
        render_video(str(tmp_path / "nope.mp4"), _FakeDetector(),
                     output_path=str(tmp_path / "out.mp4"),
                     resize_fn=_noop_resize)


def test_render_video_requires_output_path(tmp_path):
    """没给输出路径 → 报错，不能自己猜一个位置。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=5)
    with pytest.raises(VideoError, match="输出路径"):
        render_video(video, _FakeDetector(), resize_fn=_noop_resize)


def test_render_video_rejects_output_equal_to_source(tmp_path):
    """输出与源同一文件 → 拒绝（规范化后比较，含大小写），且不动源文件。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=5)
    before = os.path.getsize(video)
    with pytest.raises(VideoError, match="相同"):
        render_video(video, _FakeDetector(), output_path=video,
                     resize_fn=_noop_resize)
    assert os.path.getsize(video) == before, "拒绝时不能动源文件"


# ---------- 逐帧推理与绘制 ----------

def test_render_video_detects_every_frame(tmp_path):
    """每一帧都要推理一次：源 20 帧 → detect 调用 20 次（不是只跑抽帧那几张）。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=20, fps=10.0)
    detector = _FakeDetector()
    out = tmp_path / "out.mp4"

    result = render_video(video, detector, output_path=str(out),
                          resize_fn=_noop_resize)

    assert len(detector.calls) == 20, f"只推理了 {len(detector.calls)} 帧"
    assert result.frame_count == 20
    assert result.source_frame_count == 20


def test_render_video_output_matches_source_frame_count_and_size(tmp_path):
    """输出帧数/分辨率/帧率与源一致；逐帧回读核对，不信 CAP_PROP_FRAME_COUNT。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=30, fps=10.0)
    source = probe(video)
    out = tmp_path / "out.mp4"

    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize)
    assert result.output_fps == pytest.approx(source.fps, abs=0.05)

    capture = cv2.VideoCapture(str(out))
    assert capture.isOpened()
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    assert count == source.frame_count, f"实际回读 {count} 帧 != 源 {source.frame_count}"
    assert (width, height) == (source.width, source.height)


def test_render_video_frames_stay_in_source_order(tmp_path):
    """输出第 n 帧就是源第 n 帧：逐帧比对帧号标记，证明没有跳帧或重排。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=24, fps=12.0)
    out = tmp_path / "out.mp4"
    render_video(video, _FakeDetector(), output_path=str(out),
                 resize_fn=_noop_resize)

    capture = cv2.VideoCapture(str(out))
    seen = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        seen.append(_decode_marker(frame))
    capture.release()

    assert seen == list(range(24)), f"帧序不对：{seen}"


def test_render_video_draws_detected_boxes(tmp_path):
    """推理出的框要真画进画面：假绘制填色块，回读后该处是类别色。"""
    from labelforge.engine.video_io import render_video
    from labelforge.engine.video_render import RenderSpec

    video = _make_video(tmp_path / "src.mp4", count=6)
    out = tmp_path / "out.mp4"
    boxes = [{"label": "person", "x": 30, "y": 20, "width": 20, "height": 10,
              "score": 0.9, "class_id": 0}]

    import labelforge.engine.video_render as video_render
    original = video_render.draw_boxes
    video_render.draw_boxes = _fake_draw(boxes, None, color=(255, 0, 0))
    try:
        render_video(video, _FakeDetector(boxes), output_path=str(out),
                     resize_fn=_noop_resize,
                     render=RenderSpec(labels=None, show_labels=False))
    finally:
        video_render.draw_boxes = original

    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()
    capture.release()
    assert ok
    patch = frame[22:28, 35:45]        # 框内取样（避开边缘压缩）
    assert patch.size > 0
    assert (patch[:, :, 2] > 150).all(), f"框没画出来：{patch.reshape(-1, 3)[:3]}"


def test_render_video_without_render_spec_writes_clean_frames(tmp_path):
    """不给 render 时只推理不绘制：画面与源一致（没有框的颜色块）。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=4)
    out = tmp_path / "out.mp4"
    render_video(video, _FakeDetector(), output_path=str(out),
                 resize_fn=_noop_resize)

    capture = cv2.VideoCapture(str(out))
    ok, frame = capture.read()
    capture.release()
    assert ok
    # 假 detector 的框在 (10,10)-(30,30)；不给 render 就不该出现红色块
    assert not (frame[15:25, 15:25, 2] > 150).all(), "没给 render 却画了框"


# ---------- 失败路径：不产生「看起来成功」的输出 ----------

def test_render_video_odd_size_source_fails_clearly(tmp_path):
    """奇数尺寸源：编码器会静默截边（65x49 → 64x48），必须明确失败。

    实测 mp4v 的 `isOpened()` 对 65x49 返回 True，但回读是 64x48——只靠
    isOpened 判成功会产出一条尺寸不对的片子，用户拿去用才发现。
    """
    from labelforge.engine.video_io import render_video

    odd = _make_odd_size_video(tmp_path / "odd.avi")
    if odd is None:
        pytest.skip("本机无法合成奇数尺寸源视频（需要 ffmpeg）")
    info = probe(odd)
    if (info.width % 2 == 0) and (info.height % 2 == 0):
        pytest.skip(f"合成的源不是奇数尺寸：{info.width}x{info.height}")

    out = tmp_path / "out.mp4"
    with pytest.raises(VideoError, match="尺寸|截边"):
        render_video(odd, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)
    assert not out.exists(), "失败不能留下正式输出"


def test_render_video_removes_temp_file_on_failure(tmp_path, monkeypatch):
    """中途异常要清掉 .partial 临时文件，不能留半个视频冒充产物。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=10)
    out = tmp_path / "out.mp4"
    real_probe = video_io._verify_rendered_output
    monkeypatch.setattr(video_io, "_verify_rendered_output",
                        lambda path, info, cancel=None: (
                            real_probe(path, info)[0] + 1,
                            real_probe(path, info)[1]))

    with pytest.raises(VideoError, match="帧数不符"):
        render_video(video, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)

    assert not out.exists(), "校验失败不能发布正式文件"
    leftovers = [p for p in os.listdir(tmp_path) if "partial" in p]
    assert leftovers == [], f"临时文件没清掉：{leftovers}"


def test_render_video_verifies_readback_fps(tmp_path, monkeypatch):
    """回读帧率与源不符 → 报错（编码器可能表达不了源帧率）。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=10, fps=10.0)
    out = tmp_path / "out.mp4"
    real_probe = video_io._verify_rendered_output
    monkeypatch.setattr(video_io, "_verify_rendered_output",
                        lambda path, info, cancel=None: (real_probe(path, info)[0], 999.0))

    with pytest.raises(VideoError, match="帧率不符"):
        render_video(video, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)
    assert not out.exists()


def test_render_video_early_eof_is_a_failure(tmp_path, monkeypatch):
    """实际解码帧数少于探测值 → 失败，不能把短片当成功。

    断言**具体诊断**而不是「只要抛错就行」：源被截断（读到 EOF）与编码器
    丢帧（回读帧数不符）是两种不同的故障，用户要采取的动作也不同。
    宽松匹配会让「读到 EOF 就 break」这种实现悄悄通过——它最后也会因为
    回读帧数不符而报错，但报的是错误的病因。
    """
    from labelforge.engine import video_io
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=10)
    out = tmp_path / "out.mp4"
    real_probe = video_io.probe
    monkeypatch.setattr(video_io, "probe",
                        lambda path: _bump_frame_count(real_probe(path), 5))

    with pytest.raises(VideoError) as excinfo:
        render_video(video, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)
    message = str(excinfo.value)
    assert "提前结束" in message, f"报错了但不是「源提前结束」：{message}"
    assert "10" in message, f"要报出实际解出多少帧：{message}"
    assert not out.exists()


def test_render_video_failure_keeps_previous_output(tmp_path, monkeypatch):
    """输出已存在时：失败/取消不改动旧成品。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=10)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"OLD-OUTPUT")
    real_probe = video_io._verify_rendered_output
    monkeypatch.setattr(video_io, "_verify_rendered_output",
                        lambda path, info, cancel=None: (
                            real_probe(path, info)[0] + 3,
                            real_probe(path, info)[1]))

    with pytest.raises(VideoError):
        render_video(video, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)

    assert out.read_bytes() == b"OLD-OUTPUT", "失败不能动旧成品"
    assert [p for p in os.listdir(tmp_path) if "partial" in p] == [], \
        "失败时半成品也要清掉"


def test_render_video_reports_detector_failure_per_frame(tmp_path):
    """某帧推理抛异常 → 明确失败，不静默跳过该帧当没事。"""
    from labelforge.engine.video_io import render_video

    class _Boom(_FakeDetector):
        def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
            raise RuntimeError("模型炸了")

    video = _make_video(tmp_path / "src.mp4", count=5)
    out = tmp_path / "out.mp4"
    with pytest.raises((VideoError, RuntimeError)):
        render_video(video, _Boom(), output_path=str(out),
                     resize_fn=_noop_resize)
    assert not out.exists()


# ---------- 取消 ----------

def test_render_video_cancel_stops_and_releases(tmp_path):
    """取消：在指定帧数后停，不留正式文件，报告已完成帧数。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=40, fps=10.0)
    out = tmp_path / "out.mp4"
    detector = _FakeDetector()
    calls = {"n": 0}

    def cancel():
        return calls["n"] >= 8

    result = render_video(video, detector, output_path=str(out),
                          resize_fn=_noop_resize, cancel=cancel,
                          on_frame=lambda index: calls.__setitem__("n", calls["n"] + 1))

    assert result.cancelled is True
    assert not out.exists(), "取消不能留正式输出"
    assert result.frames_written <= 9, f"取消后还写了 {result.frames_written} 帧"


def test_render_video_cancel_leaves_no_partial_when_nothing_written(tmp_path):
    """一帧都没写就取消 → 不留垃圾文件（mp4v 会留下解不出帧的文件头）。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=20)
    out = tmp_path / "out.mp4"
    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, cancel=lambda: True)

    assert result.cancelled is True
    assert result.frames_written == 0
    assert result.partial_path == "", "零帧取消没有可用的半成品"
    assert not out.exists()
    assert [p for p in os.listdir(tmp_path) if "partial" in p] == []


def test_render_video_cancel_keeps_previous_output(tmp_path):
    """取消时旧的正式输出原样保留。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=20)
    out = tmp_path / "out.mp4"
    out.write_bytes(b"OLD-OUTPUT")

    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, cancel=lambda: True)

    assert result.cancelled is True
    assert out.read_bytes() == b"OLD-OUTPUT"


# ---------- 进度 ----------

def test_render_video_reports_progress_per_frame(tmp_path):
    """进度按帧上报：总数为源帧数，最后一条是全部完成。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=12)
    seen = []
    render_video(video, _FakeDetector(), output_path=str(tmp_path / "out.mp4"),
                 resize_fn=_noop_resize,
                 progress=lambda done, total: seen.append((done, total)))

    assert seen, "没有上报进度"
    assert all(total == 12 for _, total in seen), seen
    assert seen[-1][0] == 12, seen[-1]
    assert [d for d, _ in seen] == sorted(d for d, _ in seen), "进度必须单调"


# ---------- 范围（③ 复用） ----------

def test_render_video_uses_scope_for_every_frame(tmp_path):
    """给了滑窗范围 → 每帧都按滑窗推理（detect 被多次调用），框坐标回到原图。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=3, fps=10.0)
    detector = _FakeDetector([{"label": "person", "x": 1.0, "y": 1.0,
                               "width": 4.0, "height": 4.0, "score": 0.9,
                               "class_id": 0}])
    scope = {"mode": "window", "windows": {"w": 32, "h": 32, "overlap": 0}}

    result = render_video(video, detector, output_path=str(tmp_path / "out.mp4"),
                          resize_fn=_noop_resize, scope=scope)

    assert result.frame_count == 3
    # 64x48 图按 32x32 窗、无重叠 → 每帧 2x2=4 窗（高只排得下 2 个 32 窗）
    assert len(detector.calls) == 3 * 4, f"每帧窗数不对：{detector.calls}"


def test_render_video_rejects_invalid_scope(tmp_path):
    """非法范围（如没画 ROI）→ 启动时报错，不进后台跑出 0 框结果。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=3)
    with pytest.raises(Exception):
        render_video(video, _FakeDetector(), output_path=str(tmp_path / "out.mp4"),
                     resize_fn=_noop_resize, scope={"mode": "roi", "regions": []})


# ---------- 中文路径 ----------

def test_render_video_accepts_chinese_paths(tmp_path):
    """中文源路径、中文输出名都要能用（Windows 真路径，不只 ASCII）。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "中文视频.mp4", count=8, fps=8.0)
    out = tmp_path / "输出视频.mp4"

    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize)

    assert out.exists()
    assert result.frame_count == 8
    info = probe(str(out))
    assert (info.width, info.height) == (FRAME_W, FRAME_H)


# ---------- 真实 Qt：绘制真的进了视频 ----------

_QT_GUARD = '''
import os, sys
sys.stdout.flush()
os._exit(0)
'''


def _run_qt(script, **values):
    """子进程跑真实 Qt，返回 RESULT 行。

    用命名占位符替换而不是 %-格式化：脚本里可能自带 %（例如 QFont 的
    字体串），与 %-格式化的转义规则混在一起容易出错。
    """
    script = script.replace("@@ROOT@@", str(ROOT))
    for key, value in values.items():
        script = script.replace(f"@@{key.upper()}@@", str(value))
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=300)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


def test_render_video_draws_category_color_with_real_qt(tmp_path):
    """真实 Qt 绘制：推理框的类别色要出现在编码后的视频里。

    画面用 160x120、线宽 3：mp4v 是有损的，小画面配 1px 线会把类别色
    抹成背景色，数不出像素。
    """
    video = _make_video(tmp_path / "src.mp4", count=4, fps=4.0,
                        size=(QT_W, QT_H))
    out = tmp_path / "out.mp4"
    result = _run_qt('''
import sys
sys.path.insert(0, r"@@ROOT@@")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import numpy as np
import cv2
from labelforge.engine.video_io import render_video
from labelforge.engine.inference_image import resize_for_model
from labelforge.engine.video_render import RenderSpec

class Det:
    iou_threshold = 0.45
    def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
        return [{"label": "缺陷", "x": 20.0, "y": 20.0, "width": 60.0,
                 "height": 40.0, "score": 0.87, "class_id": 0}]

spec = RenderSpec(label_colors={"缺陷": "#FF0000"}, show_labels=True,
                  show_scores=True, pen_width=3)
res = render_video(r"@@VIDEO@@", Det(), output_path=r"@@OUT@@",
                   resize_fn=resize_for_model, render=spec)
cap = cv2.VideoCapture(r"@@OUT@@")
ok, frame = cap.read()
cap.release()
rgb = frame[:, :, ::-1].astype(int)
red = (np.abs(rgb - np.array([255, 0, 0])).max(axis=2) <= 60)
# 只看框线附近，别把整幅统计进去
band = red[18:66, 16:84]
print("RESULT", repr((res.frame_count, int(band.sum()))))
''', video=video, out=str(out))
    frame_count, red_pixels = result
    assert frame_count == 4
    assert red_pixels > 50, f"类别色没画进视频：{red_pixels} 个红像素"


def test_render_video_chinese_label_is_drawn_with_real_qt(tmp_path):
    """中文类别名要真画出字形：标签条上出现黑色笔画像素。

    显式挑一个 `inFont("缺")` 为真的字体，把「有没有画字」与「系统装了
    什么字体」解耦（offscreen 平台默认 Helvetica 没有汉字字形）。
    """
    video = _make_video(tmp_path / "src.mp4", count=2, fps=2.0,
                        size=(QT_W, QT_H))
    out = tmp_path / "out.mp4"
    result = _run_qt('''
import sys
sys.path.insert(0, r"@@ROOT@@")
from PySide6.QtWidgets import QApplication
app = QApplication([])
import numpy as np
import cv2
from PySide6.QtGui import QFont, QFontMetrics
from labelforge.engine.video_io import render_video
from labelforge.engine.inference_image import resize_for_model
from labelforge.engine.video_render import RenderSpec

CANDIDATES = ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
              "Source Han Sans SC", "WenQuanYi Zen Hei", "PingFang SC")
chosen = next((f for f in CANDIDATES if QFontMetrics(QFont(f)).inFont("缺")), None)
if chosen is None:
    print("RESULT", repr(("NOFONT",)))
else:
    app.setFont(QFont(chosen, 10))

    class Det:
        iou_threshold = 0.45
        def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
            return [{"label": "缺陷", "x": 20.0, "y": 60.0, "width": 60.0,
                     "height": 40.0, "score": 0.87, "class_id": 0}]

    spec = RenderSpec(label_colors={"缺陷": "#FF0000"}, show_labels=True,
                      show_scores=False, pen_width=3)
    render_video(r"@@VIDEO@@", Det(), output_path=r"@@OUT@@",
                 resize_fn=resize_for_model, render=spec)
    cap = cv2.VideoCapture(r"@@OUT@@")
    ok, frame = cap.read()
    cap.release()
    # 标签条画在框上方（y=60 往上），找黑字形笔画
    above = frame[30:62].reshape(-1, 3)
    black = int((above == (0, 0, 0)).all(axis=1).sum())
    print("RESULT", repr((chosen, black)))
''', video=video, out=str(out))
    # 先判「没字体」再解包：脚本那时打印的是 1 元组 `("NOFONT",)`，
    # 直接解包会 ValueError，skip 分支反而永远走不到。
    if result == ("NOFONT",):
        pytest.skip("系统没有可画汉字的字体")
    chosen, black = result
    assert black > 5, f"标签条上没有字形笔画，中文可能没渲染：{black}"


# ---------- 类别筛选 ----------

def test_render_video_filters_boxes_by_selected_classes(tmp_path):
    """勾了类别就只算/只画勾上的；没勾的类别不进 detected_total。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=4)
    out = tmp_path / "out.mp4"
    detector = _FakeDetector(boxes=[
        {"label": "person", "x": 5.0, "y": 5.0, "width": 10.0, "height": 10.0,
         "score": 0.9, "class_id": 0},
        {"label": "car", "x": 30.0, "y": 5.0, "width": 10.0, "height": 10.0,
         "score": 0.8, "class_id": 1},
    ])

    result = render_video(video, detector, output_path=str(out),
                          resize_fn=_noop_resize, class_filter=["person"])

    assert result.detected_total == 4, "每帧只该留下 person 一个框"
    assert result.frame_count == 4


def test_render_video_empty_class_filter_drops_everything(tmp_path):
    """「全不选」传空集合 = 一个框都不留（与引擎既有语义一致）。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=3)
    out = tmp_path / "out.mp4"
    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, class_filter=[])
    assert result.detected_total == 0
    assert result.frame_count == 3, "不画框也要出完整时长的视频"


def test_render_video_none_class_filter_keeps_everything(tmp_path):
    """不过滤（None）时所有框都算，不能顺手清空。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=3)
    out = tmp_path / "out.mp4"
    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, class_filter=None)
    assert result.detected_total == 3


def test_render_video_cancel_keeps_playable_partial(tmp_path):
    """取消时已经写了好几帧 → 保留可播放的半成品，但绝不占正式输出名。

    ④A 是全 app 最慢的一条路（18 秒视频要推理 541 次），跑完的帧不白费。
    计划 §4.5/§7.4 要求：可播放的部分产物以 `.partial.mp4` 名保留并报告
    已完成帧数；零帧或不可播放的才清理。

    既有用例只覆盖「一帧没写就取消」：那种情况下 mp4v 连文件头都落盘了
    （几十字节），但一帧都解不出来——所以「文件存在」不能当判据。
    """
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=40, fps=10.0)
    out = tmp_path / "out.mp4"
    seen = {"n": 0}

    def cancel():
        return seen["n"] >= 6

    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, cancel=cancel,
                          on_frame=lambda i: seen.__setitem__("n", seen["n"] + 1))

    assert result.cancelled is True
    assert result.frames_written >= 6, "要真写过帧，这条才有效"
    assert not out.exists(), "取消不能留正式输出（正式名永远不占用）"

    partial = result.partial_path
    assert partial, "写过帧的取消要留一个能播的半成品"
    assert "partial" in os.path.basename(partial), partial
    assert os.path.exists(partial)
    assert os.path.normcase(partial) != os.path.normcase(str(out))

    # 「留着」必须是真能播的：解出来的帧数要等于报告给用户的已完成帧数
    cap = cv2.VideoCapture(partial)
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    assert decoded == result.frames_written, (decoded, result.frames_written)


def test_render_video_cancel_result_reports_partial_not_final(tmp_path):
    """取消的结果里 output_path 仍是正式名（供 UI 显示），半成品另给字段。"""
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=40, fps=10.0)
    out = tmp_path / "out.mp4"
    seen = {"n": 0}
    result = render_video(
        video, _FakeDetector(), output_path=str(out), resize_fn=_noop_resize,
        cancel=lambda: seen["n"] >= 5,
        on_frame=lambda i: seen.__setitem__("n", seen["n"] + 1))

    assert result.cancelled is True
    assert result.output_path == str(out), "正式路径要原样回报，别改成半成品名"
    assert result.partial_path != result.output_path


def test_render_video_detector_failure_after_frames_writes_no_leftover(tmp_path):
    """跑了几帧后推理炸了 → 临时文件同样要清掉。"""
    from labelforge.engine.video_io import render_video

    class _BoomAfter(_FakeDetector):
        def detect(self, rgb, resize_fn, conf_threshold=None, iou_threshold=None):
            if len(self.calls) >= 4:
                raise RuntimeError("模型炸了")
            return super().detect(rgb, resize_fn, conf_threshold, iou_threshold)

    video = _make_video(tmp_path / "src.mp4", count=20)
    out = tmp_path / "out.mp4"
    with pytest.raises((VideoError, RuntimeError)):
        render_video(video, _BoomAfter(), output_path=str(out),
                     resize_fn=_noop_resize)

    assert not out.exists()
    leftovers = [p for p in os.listdir(tmp_path) if "partial" in p]
    assert leftovers == [], f"失败时不能拿半成品当挡箭牌，要清干净：{leftovers}"


def test_render_video_early_eof_leaves_no_leftover(tmp_path, monkeypatch):
    """探测帧数比实际多 → 报错且不留临时文件（已经写了若干帧）。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=10)
    out = tmp_path / "out.mp4"
    real_probe = video_io.probe
    monkeypatch.setattr(video_io, "probe",
                        lambda path: _bump_frame_count(real_probe(path), 5))

    with pytest.raises(VideoError, match="提前结束"):
        render_video(video, _FakeDetector(), output_path=str(out),
                     resize_fn=_noop_resize)

    assert not out.exists()
    leftovers = [p for p in os.listdir(tmp_path) if "partial" in p]
    assert leftovers == [], f"提前结束时留下半个视频：{leftovers}"


def test_render_video_cancel_during_readback_stops_and_keeps_partial(tmp_path):
    """取消也要能打断回读：长视频的回读本身要花时间。

    回读被打断不是失败——半成品照样留着，结果标记 cancelled。原实现里
    取消只在写帧循环里查，回读阶段按了取消要等整条视频解完才有反应。
    """
    from labelforge.engine.video_io import render_video

    video = _make_video(tmp_path / "src.mp4", count=30, fps=10.0)
    out = tmp_path / "out.mp4"
    state = {"written": 0, "verifying": False}

    def cancel():
        # 写帧阶段不取消；等所有帧写完、进入回读阶段再取消
        return state["verifying"]

    def on_frame(index):
        state["written"] = index
        if index >= 30:
            state["verifying"] = True

    result = render_video(video, _FakeDetector(), output_path=str(out),
                          resize_fn=_noop_resize, cancel=cancel,
                          on_frame=on_frame)

    assert result.cancelled is True, "回读阶段的取消没被响应"
    assert not out.exists(), "回读被取消不能发布正式文件"
    assert result.frames_written == 30
    partial = result.partial_path
    assert partial and os.path.exists(partial), "半成品要留着"
