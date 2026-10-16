"""④B 抽帧标注合成：帧率推导、两种输入源、图片序列读回校验。

核心规则（用户 2026-10-12 确认）：
- 有 video_manifest.json：按清单顺序与清单时长 D 合成，输出帧率 = N / D。
- 没有清单（外部序列图片）：用户自己指定每秒几帧，时长 = N / 该帧率。

合成要写成标准帧率的视频（每张图撑满自己的时间槽），否则播放器按低帧率
播放、拖进度条会跳；但「一张图写一帧」那种低帧率输出也要支持，用于逐帧
检查标注（不重复帧）。
"""
import json
import os
import pathlib

import cv2
import numpy as np
import pytest

from labelforge.engine.video_io import (
    VideoError, compose_video, compose_plan, list_sequence_images,
    read_manifest, write_manifest,
)


# ---------- 测试素材 ----------

FRAME_W, FRAME_H = 64, 48
BIT_COUNT = 8


def _encode_marker(frame, index):
    """帧号写成 8 个高对比黑/白块（有损编码下仍可辨认）。"""
    for bit in range(BIT_COUNT):
        if (index >> bit) & 1:
            frame[8:24, 8 + bit * 6: 8 + bit * 6 + 5] = (255, 255, 255)


def _decode_marker(frame):
    value = 0
    for bit in range(BIT_COUNT):
        if int(frame[16, 10 + bit * 6, 0]) > 127:
            value |= (1 << bit)
    return value


def _write_frame_image(path, index, size=(FRAME_W, FRAME_H)):
    """写一张带帧号的图片（走 video_io.write_image，中文路径安全）。"""
    from labelforge.engine.video_io import write_image
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    if (w, h) == (FRAME_W, FRAME_H):
        _encode_marker(frame, index)
    else:
        frame[:, :] = (index * 10 % 256, 30, 200)
    write_image(str(path), frame, ext="png")
    return str(path)


def _make_manifest_dir(tmp_path, count=10, target_fps=2.0, duration=5.0):
    """造一个「像抽帧产物」的目录：图片 + 合法清单。"""
    from labelforge.engine.video_io import (
        Manifest, ManifestFrame, output_dir_for, probe,
    )
    video = _make_source_video(tmp_path / "src.mp4", 50, fps=10.0)
    info = probe(str(video))
    out_dir = output_dir_for(str(video))
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for i in range(count):
        name = f"src_{i + 1:06d}.png"
        _write_frame_image(os.path.join(out_dir, name), i)
        frames.append(ManifestFrame(
            file=name,
            source_frame=i * 5,
            timestamp=i * 0.5,
            sample_time=i * 0.5,
        ))
    manifest = Manifest(
        source=info.path, source_name="src.mp4",
        width=info.width, height=info.height,
        source_fps=info.fps, frame_count=info.frame_count,
        duration=info.duration, target_fps=target_fps,
        start_time=0.0, end_time=duration,
        mode="sample", complete=True, frames=frames,
    )
    manifest_path = os.path.join(out_dir, "video_manifest.json")
    write_manifest(manifest_path, manifest)
    return out_dir, manifest_path


def _make_source_video(path, frame_count, fps=10.0):
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (FRAME_W, FRAME_H))
    assert writer.isOpened(), f"VideoWriter 打不开：{path}"
    for index in range(frame_count):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        _encode_marker(frame, index)
        writer.write(frame)
    writer.release()
    return path


def _read_image_sequence(path):
    """读回合成视频，返回 (fps, 出现过的图片序列)。

    标准帧率下每张图会重复写多帧，按「相邻变化」去重才是图片序列。
    """
    fps, markers = _read_video_frames(path)
    sequence = []
    for m in markers:
        if not sequence or sequence[-1] != m:
            sequence.append(m)
    return fps, sequence


def _read_video_frames(path, limit=None):
    """顺序读回合成视频，返回 (fps, 帧号列表)。"""
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"读不开合成视频：{path}"
    fps = cap.get(cv2.CAP_PROP_FPS)
    markers = []
    while limit is None or len(markers) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        markers.append(_decode_marker(frame))
    cap.release()
    return fps, markers


# ---------- 合成计划（纯逻辑，不碰 cv2） ----------

def test_plan_from_manifest_uses_manifest_duration():
    """有清单：时长取清单的 end-start，输出帧率 = N/D。"""
    plan = compose_plan(image_count=21, duration=10.1)

    assert plan.image_count == 21
    assert plan.duration == pytest.approx(10.1)
    assert plan.output_fps == pytest.approx(21 / 10.1)


def test_plan_without_manifest_uses_user_fps():
    """没有清单：用户自己指定每秒几帧，时长 = N/帧率。"""
    plan = compose_plan(image_count=30, output_fps=5.0)

    assert plan.image_count == 30
    assert plan.output_fps == pytest.approx(5.0)
    assert plan.duration == pytest.approx(6.0)


def test_plan_rejects_both_sources_given():
    """时长与输出帧率同时给出是矛盾的：必须只给一个。"""
    with pytest.raises(VideoError):
        compose_plan(image_count=10, duration=5.0, output_fps=2.0)


def test_plan_rejects_neither_source_given():
    """两个都不给 → 无法确定时长，报错而不是猜默认值。"""
    with pytest.raises(VideoError):
        compose_plan(image_count=10)


def test_plan_rejects_empty_sequence():
    """零张图不能合成，报错而不是产出一个空视频。"""
    with pytest.raises(VideoError):
        compose_plan(image_count=0, output_fps=5.0)


@pytest.mark.parametrize("count,fps,duration", [
    (-1, 5.0, None),
    (10, 0.0, None),
    (10, -5.0, None),
    (10, None, 0.0),
    (10, None, -1.0),
])
def test_plan_rejects_invalid_numbers(count, fps, duration):
    """非法数量/帧率/时长一律报错。"""
    with pytest.raises(VideoError):
        compose_plan(image_count=count, output_fps=fps, duration=duration)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_plan_rejects_non_finite(value):
    with pytest.raises(VideoError):
        compose_plan(image_count=10, output_fps=value)
    with pytest.raises(VideoError):
        compose_plan(image_count=10, duration=value)


def test_plan_frame_slots_cover_every_image_exactly_once():
    """标准帧率下每张图撑满自己的时间槽，且每张恰好用一次。

    这是「每张图等时展示」的核心：不能用整除（21 张 10.1 秒 @30fps
    整除成每张 14 帧 → 294 帧 = 9.8 秒，少 300ms）。
    """
    plan = compose_plan(image_count=21, duration=10.1, standard_fps=30.0)

    assert plan.duration == pytest.approx(10.1)
    total = plan.total_frames
    assert total == round(10.1 * 30)          # 303

    indices = [plan.image_index_for_frame(j) for j in range(total)]
    assert set(indices) == set(range(21)), "每张图都要出现"
    assert indices == sorted(indices), "顺序不能乱"
    # 每张图出现的帧数要么是 k 要么是 k+1（等时展示）
    counts = [indices.count(i) for i in range(21)]
    assert max(counts) - min(counts) <= 1


def test_plan_one_frame_per_image_mode():
    """低帧率模式：一张图写一帧（逐帧检查标注用，不重复帧）。"""
    plan = compose_plan(image_count=21, duration=10.1, one_frame_per_image=True)

    assert plan.total_frames == 21
    assert plan.output_fps == pytest.approx(21 / 10.1)
    assert [plan.image_index_for_frame(j) for j in range(21)] == list(range(21))


def test_plan_single_image_does_not_divide_by_zero():
    """只有一张图也要能算出合法帧率。"""
    plan = compose_plan(image_count=1, output_fps=5.0)

    assert plan.total_frames == 1
    assert plan.output_fps == pytest.approx(5.0)


# ---------- 图片序列发现（无清单输入） ----------

def test_list_sequence_images_natural_order(tmp_path):
    """外部序列按自然排序：img2 在 img10 之前，不能按字典序排。"""
    for name in ("img10.png", "img2.png", "img1.png"):
        _write_frame_image(tmp_path / name, 1)

    paths = list_sequence_images(str(tmp_path))

    assert [os.path.basename(p) for p in paths] == ["img1.png", "img2.png", "img10.png"]


def test_list_sequence_images_ignores_non_images(tmp_path):
    """非图片文件（含 sidecar json）不进入序列。"""
    _write_frame_image(tmp_path / "a.png", 1)
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")

    paths = list_sequence_images(str(tmp_path))

    assert [os.path.basename(p) for p in paths] == ["a.png"]


def test_list_sequence_images_empty_dir_raises(tmp_path):
    with pytest.raises(VideoError):
        list_sequence_images(str(tmp_path))


def test_list_sequence_images_missing_dir_raises(tmp_path):
    with pytest.raises(VideoError):
        list_sequence_images(str(tmp_path / "没有这个目录"))


# ---------- 真正合成 ----------

def test_compose_from_manifest_keeps_duration(tmp_path):
    """有清单：合成时长等于清单时长，帧号顺序与清单一致。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=10, duration=5.0)
    manifest = read_manifest(manifest_path)
    video = tmp_path / "out.mp4"

    result = compose_video(manifest_path=manifest_path, output_path=str(video))

    assert result.image_count == 10
    assert result.duration == pytest.approx(5.0)
    fps, sequence = _read_image_sequence(video)
    # 每张图带自己的序号，顺序必须与清单一致
    assert sequence == list(range(10))
    assert fps == pytest.approx(result.output_fps, rel=1e-2)
    # 有清单时的默认语义（用户指定）：一张图一帧，帧率 = N/D
    assert result.total_frames == result.image_count
    assert result.output_fps == pytest.approx(10 / 5.0, rel=1e-2)


def test_compose_without_manifest_uses_user_fps(tmp_path):
    """没有清单：按用户给的每秒几帧合成，时长 = N/帧率。"""
    seq = tmp_path / "序列"
    for i in range(12):
        _write_frame_image(seq / f"f{i:03d}.png", i)
    video = tmp_path / "out.mp4"

    result = compose_video(images_dir=str(seq), output_fps=4.0,
                           output_path=str(video))

    assert result.image_count == 12
    assert result.duration == pytest.approx(3.0)
    fps, sequence = _read_image_sequence(video)
    assert sequence == list(range(12))
    assert fps == pytest.approx(4.0, rel=1e-2)


def test_compose_accepts_chinese_paths(tmp_path):
    """中文视频/图片/输出路径都要能合成（Windows 真路径）。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5, duration=2.5)
    video = tmp_path / "中文目录" / "我的 合成.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)

    result = compose_video(manifest_path=manifest_path, output_path=str(video))

    assert os.path.isfile(str(video))
    assert result.image_count == 5
    _, sequence = _read_image_sequence(video)
    assert sequence == list(range(5))


def test_compose_rejects_incomplete_manifest(tmp_path):
    """complete: false 的清单不能进正式合成。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    data = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    data["complete"] = False
    pathlib.Path(manifest_path).write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(VideoError):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))


def test_compose_rejects_missing_image(tmp_path):
    """清单在但图片缺一张 → 报错，不能删掉缺帧冒充完整合成。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    manifest = read_manifest(manifest_path)
    os.remove(os.path.join(out_dir, manifest.frames[2].file))

    with pytest.raises(VideoError):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))


def test_compose_rejects_mismatched_sizes(tmp_path):
    """图片尺寸不一致 → 报错（清单声明了尺寸，不能静默缩放）。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    manifest = read_manifest(manifest_path)
    _write_frame_image(os.path.join(out_dir, manifest.frames[1].file), 1,
                       size=(FRAME_W * 2, FRAME_H * 2))

    with pytest.raises(VideoError, match="尺寸"):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))


def test_compose_rejects_corrupt_image(tmp_path):
    """图片损坏 → 报错，不当作无标注画面静默跳过。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    manifest = read_manifest(manifest_path)
    pathlib.Path(os.path.join(out_dir, manifest.frames[1].file)).write_bytes(b"not an image")

    with pytest.raises(VideoError):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))


def test_compose_failure_leaves_no_output(tmp_path):
    """失败时不能留下看似成功的输出文件。"""
    seq = tmp_path / "seq"
    _write_frame_image(seq / "a.png", 1)
    (seq / "b.png").write_bytes(b"broken")
    video = tmp_path / "out.mp4"

    with pytest.raises(VideoError):
        compose_video(images_dir=str(seq), output_fps=2.0,
                      output_path=str(video))

    assert not os.path.exists(str(video)), "失败后不能留下输出文件"
    leftovers = [n for n in os.listdir(str(tmp_path)) if "partial" in n]
    assert not leftovers, f"失败不能留下临时文件：{leftovers}"


def test_compose_reports_progress_and_cancels(tmp_path):
    """进度按已写图片数上报；取消后不产正式输出。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=8, duration=4.0)
    seen = []
    video = tmp_path / "out.mp4"

    result = compose_video(
        manifest_path=manifest_path, output_path=str(video),
        progress=lambda done, total: seen.append((done, total)),
        cancel=lambda: len(seen) >= 3,
    )

    assert result.cancelled is True
    assert seen and all(total == 8 for _, total in seen)
    assert not os.path.exists(str(video)), "取消不能留下正式输出"
    leftovers = [n for n in os.listdir(str(tmp_path)) if "partial" in n]
    assert not leftovers, f"取消不能留下临时文件：{leftovers}"


def test_compose_rejects_output_equal_to_source(tmp_path):
    """输出路径与源视频相同 → 拒绝，避免写输出毁掉输入。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    manifest = read_manifest(manifest_path)

    with pytest.raises(VideoError):
        compose_video(manifest_path=manifest_path,
                      output_path=manifest.source)


def test_compose_standard_fps_option_expands_frames(tmp_path):
    """显式给标准帧率时，每张图撑满时间槽（播放流畅、时长不变）。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=10, duration=5.0)
    video = tmp_path / "out.mp4"

    result = compose_video(manifest_path=manifest_path, output_path=str(video),
                           standard_fps=30.0)

    assert result.duration == pytest.approx(5.0)
    assert result.total_frames == 150, result.total_frames
    assert result.total_frames > result.image_count
    fps, sequence = _read_image_sequence(video)
    assert sequence == list(range(10)), "顺序与清单一致且每张都出现"
    assert fps == pytest.approx(30.0, rel=1e-2)


def test_compose_verifies_readback_frame_count(tmp_path, monkeypatch):
    """编码后回读帧数不符必须报错。

    这条守的是编码器静默丢帧（cv2 对尺寸不符的帧就是直接丢弃、write()
    不抛异常），所以要单独钉住，不能指望尺寸检查替它兜底。
    """
    import labelforge.engine.video_io as video_io

    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5, duration=2.5)
    real_probe = video_io._probe_output
    monkeypatch.setattr(video_io, "_probe_output",
                        lambda path: (4, real_probe(path)[1]))   # 谎报帧数

    with pytest.raises(VideoError, match="帧数不符"):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))

    assert not os.path.exists(str(tmp_path / "out.mp4")), "验收失败不能发布输出"


def test_compose_verifies_readback_fps(tmp_path, monkeypatch):
    """回读帧率与计划不符也要报错（编码器可能不认非标准帧率）。"""
    import labelforge.engine.video_io as video_io

    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5, duration=2.5)
    real_probe = video_io._probe_output
    monkeypatch.setattr(video_io, "_probe_output",
                        lambda path: (real_probe(path)[0], 99.0))

    with pytest.raises(VideoError, match="帧率不符"):
        compose_video(manifest_path=manifest_path,
                      output_path=str(tmp_path / "out.mp4"))


def test_compose_rejects_output_overwriting_source(tmp_path):
    """输出路径与源视频相同 → 拒绝（规范化后比较，防大小写/相对路径绕过）。"""
    out_dir, manifest_path = _make_manifest_dir(tmp_path, count=5)
    manifest = read_manifest(manifest_path)
    source = manifest.source

    with pytest.raises(VideoError, match="相同"):
        compose_video(manifest_path=manifest_path, output_path=source)

    assert os.path.exists(source), "拒绝时不能动到源文件"


def test_compose_requires_some_input_source(tmp_path):
    """既没清单也没图片目录 → 报错，不能拿默认帧率凑一个视频出来。"""
    with pytest.raises(VideoError, match="清单或图片目录"):
        compose_video(output_path=str(tmp_path / "out.mp4"))
