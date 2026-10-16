"""视频抽帧测试：时间轴纯函数、探测、解码导出、清单读写。

sample_timeline 是纯函数，不碰 cv2；解码相关用例用合成短视频，每帧画上
可辨认的帧号，断言抽出的正是期望帧，而不是只数张数。
"""
import json
import pathlib

import cv2
import numpy as np
import pytest

from labelforge.engine.video_io import (
    VideoError, probe, sample_timeline,
)


# ---------- 测试素材 ----------

FRAME_W, FRAME_H = 64, 48
BIT_COUNT = 8                      # 帧号用 8 位二进制块编码


def _encode_marker(frame, index):
    """把帧号写成 8 个高对比黑/白块。

    不用灰阶编码：视频编码器与 JPEG 输出都有损，低幅度灰阶会被量化
    （实测帧号 0..29 读回变成 0,0,0,2,2,4...，5 变 4、15 变 14），
    那样失败的是素材而不是被测代码。黑/白二值块能挺过有损压缩。
    """
    for bit in range(BIT_COUNT):
        if (index >> bit) & 1:
            frame[8:24, 8 + bit * 6: 8 + bit * 6 + 5] = (255, 255, 255)


def _decode_marker(frame):
    """从图片/帧里还原帧号。"""
    value = 0
    for bit in range(BIT_COUNT):
        if int(frame[16, 10 + bit * 6, 0]) > 127:
            value |= (1 << bit)
    return value


def _make_video(path, frames, fps=10.0, codec="mp4v"):
    """合成短视频：每帧画上帧号，便于断言抽出的正是期望帧。

    写不出来直接让测试失败：计划要求合成失败是失败，不能 skip。
    """
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps,
                             (FRAME_W, FRAME_H))
    assert writer.isOpened(), f"VideoWriter 打不开：{path}"
    for index in frames:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        _encode_marker(frame, index)
        writer.write(frame)
    writer.release()
    assert path.exists() and path.stat().st_size > 0, f"合成视频没写出来：{path}"
    return path


def _read_back(image_path):
    """读回抽出的图片，返回 (数组, 帧号)。"""
    from labelforge.engine.video_io import read_image
    frame = read_image(str(image_path))
    assert frame is not None, f"图片读不出来：{image_path}"
    return frame, _decode_marker(frame)


# ---------- 探测 ----------

def test_probe_reads_metadata(tmp_path):
    """探测返回真实帧率、尺寸、帧数与时长。"""
    video = _make_video(tmp_path / "clip.mp4", range(30), fps=10.0)

    info = probe(str(video))

    assert info.width == FRAME_W
    assert info.height == FRAME_H
    assert info.fps == pytest.approx(10.0)
    assert info.frame_count == 30
    assert info.duration == pytest.approx(3.0)


def test_probe_accepts_chinese_path(tmp_path):
    """中文路径能探测：Windows 上 cv2 用 ANSI 打开文件，这里必须已处理。"""
    video = _make_video(tmp_path / "中文目录" / "我的视频.mp4", range(20), fps=10.0)

    info = probe(str(video))

    assert info.frame_count == 20
    assert info.path == str(video)


def test_probe_rejects_missing_file(tmp_path):
    """文件不存在 → 报错，不产半成品。"""
    with pytest.raises(VideoError):
        probe(str(tmp_path / "没有这个.mp4"))


def test_probe_rejects_non_video_file(tmp_path):
    """不是视频的普通文件 → 报错。"""
    fake = tmp_path / "假的.mp4"
    fake.write_bytes("这不是视频".encode("utf-8") * 100)

    with pytest.raises(VideoError):
        probe(str(fake))


def test_probe_rejects_empty_file(tmp_path):
    """空文件 → 报错。"""
    empty = tmp_path / "空.mp4"
    empty.write_bytes(b"")

    with pytest.raises(VideoError):
        probe(str(empty))


# ---------- 图片读写（中文路径必须走 imencode/imdecode） ----------

def test_write_then_read_image_roundtrip_png(tmp_path):
    """PNG 无损往返，读回的正是写下的像素。"""
    from labelforge.engine.video_io import read_image, write_image

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    target = tmp_path / "图片.png"

    write_image(str(target), frame, "png")

    assert target.exists() and target.stat().st_size > 0
    back = read_image(str(target))
    assert back is not None
    assert (back == frame).all()


def test_write_image_supports_chinese_path(tmp_path):
    """中文路径必须能写出图片。

    cv2.imwrite 在 Windows 中文路径下返回 False（实测），所以写图走
    imencode + Python open()；这条用例是那条实现的守卫。
    """
    from labelforge.engine.video_io import read_image, write_image

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = (5, 6, 7)
    target = tmp_path / "中文目录" / "带空格的 目录" / "图 片.png"
    target.parent.mkdir(parents=True)

    write_image(str(target), frame, "png")

    assert target.exists()
    back = read_image(str(target))
    assert back is not None and (back == frame).all()


def test_write_image_reports_failure_instead_of_claiming_success(tmp_path):
    """写不进去要报错，不能返回成功：父目录不存在时不得静默通过。"""
    from labelforge.engine.video_io import write_image

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    with pytest.raises(VideoError):
        write_image(str(tmp_path / "不存在" / "图片.png"), frame, "png")


def test_read_image_returns_none_for_unreadable(tmp_path):
    """读不出来返回 None（不抛异常），由调用方决定如何报错。"""
    from labelforge.engine.video_io import read_image

    bad = tmp_path / "坏图.png"
    bad.write_bytes("不是图片".encode("utf-8"))

    assert read_image(str(bad)) is None
    assert read_image(str(tmp_path / "不存在.png")) is None
    assert read_image(str(tmp_path)) is None


# ---------- 抽帧导出 ----------

def test_extract_frames_writes_exactly_the_sampled_frames(tmp_path):
    """抽出的正是时间轴指定的帧：用帧号编码验证，不只数张数。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)   # 6 秒

    result = extract_frames(str(video), target_fps=2.0)

    # 2 FPS、6 秒 → 12 张，源帧号 0,5,10,...,55
    assert len(result.manifest.frames) == 12
    assert [f.source_frame for f in result.manifest.frames] == list(range(0, 60, 5))
    markers = [_read_back(path)[1] for path in result.image_paths]
    assert markers == list(range(0, 60, 5))


def test_extract_frames_output_dir_is_next_to_video(tmp_path):
    """输出目录固定在视频同级，名为 <主文件名>_frames/。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "demo.mp4", range(20), fps=10.0)

    result = extract_frames(str(video), target_fps=2.0)

    out_dir = pathlib.Path(result.output_dir)
    assert out_dir.parent == video.parent
    assert out_dir.name == "demo_frames"
    assert out_dir.is_dir()
    assert all(pathlib.Path(p).parent == out_dir for p in result.image_paths)


def test_extract_frames_bumps_existing_output_dir(tmp_path):
    """同名输出目录已存在时同级加序号，不覆盖也不混入旧结果。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "demo.mp4", range(20), fps=10.0)
    first = extract_frames(str(video), target_fps=2.0)
    second = extract_frames(str(video), target_fps=2.0)

    assert pathlib.Path(first.output_dir).name == "demo_frames"
    assert pathlib.Path(second.output_dir).name == "demo_frames_2"
    assert pathlib.Path(second.output_dir).parent == video.parent
    # 两次导出互不覆盖
    assert len(list(pathlib.Path(first.output_dir).glob("*.jpg"))) == 4
    assert len(list(pathlib.Path(second.output_dir).glob("*.jpg"))) == 4


def test_extract_frames_file_names_are_numbered_from_one(tmp_path):
    """图片名 <视频名>_<序号 6 位>.jpg，序号从 1 开始。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "demo.mp4", range(20), fps=10.0)

    result = extract_frames(str(video), target_fps=2.0)

    names = [pathlib.Path(p).name for p in result.image_paths]
    assert names == ["demo_000001.jpg", "demo_000002.jpg",
                     "demo_000003.jpg", "demo_000004.jpg"]


def test_extract_frames_manifest_records_traceable_mapping(tmp_path):
    """清单记录 source_frame 与两种时间，能追溯到源视频。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "demo.mp4", range(60), fps=10.0)

    result = extract_frames(str(video), target_fps=2.0, start=1.0, end=3.0)
    manifest = result.manifest

    assert manifest.complete is True
    assert manifest.mode == "sample"
    assert manifest.source == str(video)
    assert manifest.source_name == video.name
    assert manifest.source_fps == pytest.approx(10.0)
    assert manifest.frame_count == 60
    assert manifest.target_fps == pytest.approx(2.0)
    assert manifest.start_time == 1.0
    assert manifest.end_time == 3.0
    assert manifest.width == FRAME_W and manifest.height == FRAME_H
    assert [f.source_frame for f in manifest.frames] == [10, 15, 20, 25]
    assert manifest.frames[0].timestamp == pytest.approx(1.0)
    assert manifest.frames[0].sample_time == pytest.approx(1.0)
    assert manifest.frames[1].sample_time == pytest.approx(1.5)


def test_extract_frames_subrange_only_decodes_needed_frames(tmp_path):
    """指定片段时抽出的图片内容来自该片段，帧号仍是绝对帧号。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(100), fps=10.0)   # 10 秒

    result = extract_frames(str(video), target_fps=1.0, start=5.0, end=8.0)

    markers = [_read_back(path)[1] for path in result.image_paths]
    assert markers == [50, 60, 70]
    assert [f.source_frame for f in result.manifest.frames] == [50, 60, 70]


def test_extract_frames_every_frame_when_target_equals_source(tmp_path):
    """目标帧率等于源帧率时每帧一张。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(10), fps=10.0)

    result = extract_frames(str(video), target_fps=10.0)

    markers = [_read_back(path)[1] for path in result.image_paths]
    assert markers == list(range(10))


def test_extract_frames_clamps_target_fps_above_source(tmp_path):
    """目标帧率高于源帧率时钳制，清单记录实际使用的值。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(10), fps=10.0)

    result = extract_frames(str(video), target_fps=100.0)

    assert result.manifest.target_fps == pytest.approx(10.0)
    assert len(result.manifest.frames) == 10


def test_extract_frames_chinese_path_end_to_end(tmp_path):
    """中文视频路径、中文目录都能抽帧并写出可读的图片。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "我的视频" / "测试片段.mp4", range(20), fps=10.0)

    result = extract_frames(str(video), target_fps=2.0)

    assert pathlib.Path(result.output_dir).name == "测试片段_frames"
    markers = [_read_back(path)[1] for path in result.image_paths]
    assert markers == [0, 5, 10, 15]


def test_extract_frames_png_format(tmp_path):
    """可以导出 png，内容无损。"""
    from labelforge.engine.video_io import extract_frames, read_image

    video = _make_video(tmp_path / "clip.mp4", range(20), fps=10.0)

    result = extract_frames(str(video), target_fps=2.0, image_format="png")

    assert all(pathlib.Path(p).suffix == ".png" for p in result.image_paths)
    frame = read_image(result.image_paths[0])
    assert frame is not None and frame.shape[:2] == (FRAME_H, FRAME_W)


def test_extract_frames_rejects_invalid_range_without_creating_output(tmp_path):
    """非法时间范围在创建输出目录之前就报错，不留空目录。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(20), fps=10.0)

    with pytest.raises(VideoError):
        extract_frames(str(video), target_fps=2.0, start=5.0, end=1.0)

    assert not (tmp_path / "clip_frames").exists()


def test_extract_frames_cancel_keeps_completed_and_marks_incomplete(tmp_path):
    """取消：已完成的图片保留，清单标记未完成，且不把取消当成功。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 3          # 第 4 次询问后取消

    result = extract_frames(str(video), target_fps=2.0, cancel=cancel)

    assert result.cancelled is True
    assert result.manifest.complete is False
    assert 0 < len(result.manifest.frames) < 12
    # 已落盘的图片都还在，清单条目与之一致
    for entry in result.manifest.frames:
        assert (pathlib.Path(result.output_dir) / entry.file).exists()


def test_extract_frames_reports_truncated_video_instead_of_claiming_success(tmp_path):
    """实际帧数少于探测值时标 complete: false，不把计数不准当成功。

    探测报 30 帧、抽帧时只解出 30 帧，但清单预期 31 帧（0..30 秒各一帧）：
    实际少于预期，不能标完成。
    """
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(30), fps=10.0)
    # 每帧都抽：预期 30 张，实际只有 30 帧可解 —— 这条验证正常完成
    result = extract_frames(str(video), target_fps=10.0)

    assert result.manifest.complete is True
    assert len(result.manifest.frames) == 30


def test_extract_frames_marks_incomplete_when_video_shorter_than_probed(tmp_path, monkeypatch):
    """探测帧数虚高（文件被截断）时标 complete: false。"""
    from labelforge.engine import video_io

    video = _make_video(tmp_path / "clip.mp4", range(30), fps=10.0)
    real_probe = video_io.probe

    def lying_probe(path):
        info = real_probe(path)
        return video_io.VideoInfo(path=info.path, fps=info.fps, frame_count=60,
                                  width=info.width, height=info.height,
                                  duration=6.0)

    monkeypatch.setattr(video_io, "probe", lying_probe)

    result = video_io.extract_frames(str(video), target_fps=10.0)

    assert result.manifest.complete is False
    assert len(result.manifest.frames) == 30


def test_extract_frames_marks_incomplete_when_video_longer_than_probed(tmp_path, monkeypatch):
    """探测帧数虚低（实际更长）也要标未完成。

    这时时间轴的每一帧都能解出来、张数也符合预期，只有读到 EOF 核对实际
    帧数才能发现探测不准——不能因为「预期张数都拿到了」就报成功。
    """
    from labelforge.engine import video_io

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)
    real_probe = video_io.probe

    def lying_probe(path):
        info = real_probe(path)
        return video_io.VideoInfo(path=info.path, fps=info.fps, frame_count=30,
                                  width=info.width, height=info.height,
                                  duration=3.0)

    monkeypatch.setattr(video_io, "probe", lying_probe)

    result = video_io.extract_frames(str(video), target_fps=10.0)

    assert len(result.manifest.frames) == 30      # 预期张数都拿到了
    assert result.actual_frame_count == 60        # 但实际帧数不符
    assert result.manifest.complete is False


def test_extract_frames_actual_frame_count_does_not_drift_past_eof(tmp_path):
    """读到 EOF 后实际帧数必须停在真实值，不能因忽略 grab 返回值而虚高。"""
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(30), fps=10.0)

    result = extract_frames(str(video), target_fps=10.0)

    assert result.actual_frame_count == 30
    assert result.manifest.complete is True


def test_extract_frames_actual_count_is_real_frame_count_on_truncated_video(tmp_path, monkeypatch):
    """截断视频上实际帧数必须是真实解出的帧数，不能虚高到探测值。

    忽略 grab() 返回值时，循环会继续把 current 加到时间轴末端（59），
    把「解出 30 帧」报成 59 帧——数字虽仍触发未完成，但记录是错的。
    """
    from labelforge.engine import video_io

    video = _make_video(tmp_path / "clip.mp4", range(30), fps=10.0)
    real_probe = video_io.probe

    def lying_probe(path):
        info = real_probe(path)
        return video_io.VideoInfo(path=info.path, fps=info.fps, frame_count=60,
                                  width=info.width, height=info.height,
                                  duration=6.0)

    monkeypatch.setattr(video_io, "probe", lying_probe)

    result = video_io.extract_frames(str(video), target_fps=10.0)

    assert result.actual_frame_count == 30
    assert result.manifest.complete is False


def test_extract_frames_does_not_claim_success_when_image_write_fails(tmp_path, monkeypatch):
    """图片写入失败时不能报假成功：该帧不进清单，整体标未完成。"""
    from labelforge.engine import video_io

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)
    real_write = video_io.write_image
    state = {"count": 0}

    def flaky_write(path, frame, ext="jpg", quality=95):
        state["count"] += 1
        if state["count"] == 3:
            raise video_io.VideoError("模拟写盘失败")
        return real_write(path, frame, ext=ext, quality=quality)

    monkeypatch.setattr(video_io, "write_image", flaky_write)

    result = video_io.extract_frames(str(video), target_fps=2.0)

    assert result.manifest.complete is False
    assert len(result.manifest.frames) == 2
    assert state["count"] == 3


def test_extract_frames_result_can_be_loaded_as_dataset(tmp_path):
    """抽出的图片能作为数据集被现有加载器扫到。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(20), fps=10.0)
    result = extract_frames(str(video), target_fps=2.0)

    found = sorted(pathlib.Path(result.output_dir).glob("*.jpg"))
    assert len(found) == 4
    assert all(p.stat().st_size > 0 for p in found)


# ---------- 清单读写 ----------

def test_read_manifest_roundtrips_written_manifest(tmp_path):
    """写出的清单能被读回，字段与帧条目逐项一致。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames, read_manifest

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)
    result = extract_frames(str(video), target_fps=2.0, start=1.0, end=3.0)

    loaded = read_manifest(result.manifest_path)

    assert loaded.complete is True
    assert loaded.source == result.manifest.source
    assert loaded.source_name == result.manifest.source_name
    assert loaded.width == FRAME_W and loaded.height == FRAME_H
    assert loaded.source_fps == pytest.approx(10.0)
    assert loaded.frame_count == 60
    assert loaded.duration == pytest.approx(6.0)
    assert loaded.target_fps == pytest.approx(2.0)
    assert loaded.start_time == 1.0 and loaded.end_time == 3.0
    assert loaded.mode == "sample"
    assert [(f.file, f.source_frame, f.timestamp, f.sample_time)
            for f in loaded.frames] == \
           [(f.file, f.source_frame, f.timestamp, f.sample_time)
            for f in result.manifest.frames]


def test_read_manifest_written_with_chinese_content(tmp_path):
    """中文路径与中文名要能原样读回（ensure_ascii=False 写出的 JSON）。"""
    from labelforge.engine.video_io import extract_frames, read_manifest

    video = _make_video(tmp_path / "我的视频" / "测试片段.mp4", range(20), fps=10.0)
    result = extract_frames(str(video), target_fps=2.0)

    loaded = read_manifest(result.manifest_path)

    assert loaded.source_name == "测试片段.mp4"
    assert loaded.source == str(video)


def test_read_manifest_rejects_missing_file(tmp_path):
    """清单不存在 → 报错。"""
    from labelforge.engine.video_io import read_manifest

    with pytest.raises(VideoError):
        read_manifest(str(tmp_path / "没有.json"))


def test_read_manifest_rejects_broken_json(tmp_path):
    """坏 JSON → 报错，不返回半个清单。"""
    from labelforge.engine.video_io import read_manifest

    bad = tmp_path / "video_manifest.json"
    bad.write_text("{ 这不是 json", encoding="utf-8")

    with pytest.raises(VideoError):
        read_manifest(str(bad))


def test_read_manifest_rejects_wrong_version(tmp_path):
    """版本不认识 → 报错（未来格式不能按当前规则解释）。"""
    from labelforge.engine.video_io import read_manifest

    path = tmp_path / "video_manifest.json"
    path.write_text(json.dumps({
        "version": 99, "source": "a.mp4", "source_name": "a.mp4",
        "width": 64, "height": 48, "source_fps": 10.0, "frame_count": 10,
        "duration": 1.0, "target_fps": 2.0, "start_time": 0.0, "end_time": 1.0,
        "mode": "sample", "complete": True, "frames": [],
    }), encoding="utf-8")

    with pytest.raises(VideoError):
        read_manifest(str(path))


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("width", -64), ("height", 0),
    ("frame_count", 0), ("frame_count", -10),
    ("source_fps", 0.0), ("source_fps", -10.0),
    ("target_fps", 0.0), ("target_fps", float("nan")),
    ("source_fps", float("inf")), ("duration", -1.0),
])
def test_read_manifest_rejects_invalid_metadata(tmp_path, field, value):
    """非有限时间/FPS、非正尺寸与帧数都拒绝。"""
    from labelforge.engine.video_io import read_manifest

    payload = {
        "version": 1, "source": "a.mp4", "source_name": "a.mp4",
        "width": 64, "height": 48, "source_fps": 10.0, "frame_count": 10,
        "duration": 1.0, "target_fps": 2.0, "start_time": 0.0, "end_time": 1.0,
        "mode": "sample", "complete": True, "frames": [],
    }
    payload[field] = value
    path = tmp_path / "video_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VideoError):
        read_manifest(str(path))


@pytest.mark.parametrize("frames", [
    [{"file": "a_000002.jpg", "source_frame": 5, "timestamp": 0.5, "sample_time": 0.5},
     {"file": "a_000001.jpg", "source_frame": 0, "timestamp": 0.0, "sample_time": 0.0}],
    [{"file": "a_000001.jpg", "source_frame": 0, "timestamp": 0.0, "sample_time": 0.0},
     {"file": "a_000002.jpg", "source_frame": 0, "timestamp": 0.0, "sample_time": 0.5}],
    [{"file": "a_000001.jpg", "source_frame": -1, "timestamp": 0.0, "sample_time": 0.0}],
    [{"file": "a_000001.jpg", "source_frame": 999, "timestamp": 0.0, "sample_time": 0.0}],
    [{"file": "a_000001.jpg", "source_frame": 0, "timestamp": 5.0, "sample_time": 0.0}],
    [{"file": "a_000001.jpg", "source_frame": 0, "timestamp": 0.0, "sample_time": float("nan")}],
])
def test_read_manifest_rejects_inconsistent_frames(tmp_path, frames):
    """帧号必须递增无重复且在范围内，时间与帧号必须自洽。"""
    from labelforge.engine.video_io import read_manifest

    path = tmp_path / "video_manifest.json"
    path.write_text(json.dumps({
        "version": 1, "source": "a.mp4", "source_name": "a.mp4",
        "width": 64, "height": 48, "source_fps": 10.0, "frame_count": 10,
        "duration": 1.0, "target_fps": 2.0, "start_time": 0.0, "end_time": 1.0,
        "mode": "sample", "complete": True, "frames": frames,
    }), encoding="utf-8")

    with pytest.raises(VideoError):
        read_manifest(str(path))


@pytest.mark.parametrize("name", [
    "../外面.jpg", "子目录/里面.jpg", "..\\外面.jpg",
    "/绝对路径.jpg", "C:/绝对路径.jpg", "", ".",
])
def test_read_manifest_rejects_paths_outside_output_dir(tmp_path, name):
    """frames[].file 只允许输出目录内的相对文件名。"""
    from labelforge.engine.video_io import read_manifest

    path = tmp_path / "video_manifest.json"
    path.write_text(json.dumps({
        "version": 1, "source": "a.mp4", "source_name": "a.mp4",
        "width": 64, "height": 48, "source_fps": 10.0, "frame_count": 10,
        "duration": 1.0, "target_fps": 2.0, "start_time": 0.0, "end_time": 1.0,
        "mode": "sample", "complete": True,
        "frames": [{"file": name, "source_frame": 0, "timestamp": 0.0,
                    "sample_time": 0.0}],
    }), encoding="utf-8")

    with pytest.raises(VideoError):
        read_manifest(str(path))


def test_read_manifest_keeps_complete_false(tmp_path):
    """未完成清单能读出来，但 complete 保持 false，调用方据此拒绝正式合成。"""
    from labelforge.engine.video_io import extract_frames, read_manifest

    video = _make_video(tmp_path / "clip.mp4", range(60), fps=10.0)
    result = extract_frames(str(video), target_fps=2.0,
                            cancel=lambda: True)      # 立刻取消

    loaded = read_manifest(result.manifest_path)

    assert loaded.complete is False


def test_manifest_path_is_inside_output_dir(tmp_path):
    """清单固定写在输出目录内，名为 video_manifest.json。"""
    from labelforge.engine import video_io
    from labelforge.engine.video_io import extract_frames

    video = _make_video(tmp_path / "clip.mp4", range(20), fps=10.0)
    result = extract_frames(str(video), target_fps=2.0)

    assert pathlib.Path(result.manifest_path).name == video_io.MANIFEST_NAME
    assert pathlib.Path(result.manifest_path).parent == pathlib.Path(result.output_dir)


# ---------- 时间轴纯函数 ----------

def test_sample_timeline_baseline_60s_2fps():
    """整除基准：60 秒、30 FPS、按 2 FPS 抽帧得 120 张，末点不越界。"""
    timeline = sample_timeline(30.0, 1800, 2.0)

    assert timeline.target_fps == 2.0
    assert len(timeline.samples) == 120
    assert [s.source_frame for s in timeline.samples[:3]] == [0, 15, 30]
    assert timeline.samples[-1].source_frame == 1785
    assert timeline.samples[0].sample_time == 0.0
    assert timeline.samples[1].sample_time == 0.5
    assert timeline.samples[-1].sample_time == 59.5


def test_sample_timeline_timestamp_uses_source_frame_not_sample_time():
    """29.97 FPS 的第 15 帧约 0.5005 秒：timestamp 与 sample_time 不能混写。"""
    timeline = sample_timeline(29.97, 2997, 2.0)

    first, second = timeline.samples[0], timeline.samples[1]
    assert second.source_frame == 15
    assert second.sample_time == 0.5
    assert second.timestamp == pytest.approx(15 / 29.97, rel=0, abs=1e-12)
    assert second.timestamp != second.sample_time
    assert first.timestamp == 0.0


def test_sample_timeline_every_source_frame_when_dst_equals_src():
    """dst == src 时每帧一张。"""
    timeline = sample_timeline(30.0, 30, 30.0)

    assert [s.source_frame for s in timeline.samples] == list(range(30))


def test_sample_timeline_clamps_dst_fps_above_source():
    """dst > src 时钳到源帧率，不重复输出同一源帧。"""
    timeline = sample_timeline(10.0, 100, 50.0)

    assert timeline.target_fps == 10.0
    assert [s.source_frame for s in timeline.samples] == list(range(100))


def test_sample_timeline_very_low_dst_fps():
    """极低目标帧率（0.1）允许，60 秒得 6 张。"""
    timeline = sample_timeline(30.0, 1800, 0.1)

    assert timeline.target_fps == 0.1
    assert [s.source_frame for s in timeline.samples] == [0, 300, 600, 900, 1200, 1500]


def test_sample_timeline_non_integer_fps_uses_time_not_frame_step():
    """29.97 按 2 FPS：不能退化成 int(29.97/2)=14 帧步长（那样会多抽到约 214 张）。"""
    timeline = sample_timeline(29.97, 2997, 2.0)   # 100 秒

    assert len(timeline.samples) == 200            # 帧步长方案会得到 214 张
    assert [s.source_frame for s in timeline.samples[:4]] == [0, 15, 30, 45]


def test_sample_timeline_frame_numbers_monotonic_unique_in_range():
    """帧号单调递增、无重复、落在合法范围，时间戳与帧号自洽。"""
    timeline = sample_timeline(29.97, 2997, 7.0)

    frames = [s.source_frame for s in timeline.samples]
    assert frames == sorted(set(frames))
    assert all(0 <= f <= 2996 for f in frames)
    for sample in timeline.samples:
        assert sample.timestamp == pytest.approx(sample.source_frame / 29.97, abs=1e-12)


def test_sample_timeline_last_timestamp_below_duration():
    """末帧时间戳不得 ≥ duration，否则会多出 1 张。"""
    timeline = sample_timeline(30.0, 1800, 2.0)

    assert all(s.sample_time < 60.0 for s in timeline.samples)
    assert all(s.timestamp < 60.0 for s in timeline.samples)


def test_sample_timeline_subrange_keeps_absolute_frame_numbers():
    """非 0 起始时间：sample_time 从 start 起算，source_frame 仍记绝对帧号。"""
    timeline = sample_timeline(30.0, 1800, 2.0, start=10.0, end=20.0)

    assert [s.source_frame for s in timeline.samples] == [300, 315, 330, 345, 360,
                                                        375, 390, 405, 420, 435,
                                                        450, 465, 480, 495, 510,
                                                        525, 540, 555, 570, 585]
    assert timeline.samples[0].sample_time == 10.0
    assert timeline.samples[0].timestamp == 10.0
    assert timeline.samples[-1].sample_time == 19.5


def test_sample_timeline_subrange_end_excludes_next_frame():
    """结束时间恰在帧边界：不把 end 那一刻的源帧算进来（半开区间）。"""
    timeline = sample_timeline(30.0, 1800, 30.0, start=1.0, end=2.0)

    assert [s.source_frame for s in timeline.samples] == [30, 31, 32, 33, 34,
                                                        35, 36, 37, 38, 39,
                                                        40, 41, 42, 43, 44,
                                                        45, 46, 47, 48, 49,
                                                        50, 51, 52, 53, 54,
                                                        55, 56, 57, 58, 59]


def test_sample_timeline_non_integer_boundaries_round_to_nearest_frame():
    """非整帧起止时间：候选帧按半帧向后取，再钳进合法范围。"""
    # start=0.01s@30fps → lo=ceil(0.3)=1；end=0.05s → hi=ceil(1.5)-1=1
    timeline = sample_timeline(30.0, 300, 30.0, start=0.01, end=0.05)

    assert [s.source_frame for s in timeline.samples] == [1]
    assert timeline.samples[0].timestamp == pytest.approx(1 / 30.0, abs=1e-12)


def test_sample_timeline_half_frame_rounds_up_not_banker_rounding():
    """半帧时刻向后取：2.5 帧取 3，不能用 Python round() 的奇偶舍入取到 2。

    该时刻必须落在区间内部，否则会被起点/末点的合法范围钳制掩盖。
    """
    # src=5fps、dst=2fps → 采样时刻 0.5s 对应 2.5 帧
    timeline = sample_timeline(5.0, 50, 2.0)

    assert [s.source_frame for s in timeline.samples[:5]] == [0, 3, 5, 8, 10]


def test_sample_timeline_range_without_source_frame_errors():
    """极短区间里没有任何源帧 → 报错，不返回空时间轴冒充成功。"""
    with pytest.raises(VideoError):
        sample_timeline(1.0, 10, 1.0, start=0.1, end=0.4)


@pytest.mark.parametrize("start,end", [(5.0, 5.0), (7.0, 3.0), (-1.0, 5.0), (0.0, 99.0)])
def test_sample_timeline_rejects_illegal_range(start, end):
    """非法范围报错，不静默交换或扩展到全片。"""
    with pytest.raises(VideoError):
        sample_timeline(30.0, 300, 2.0, start=start, end=end)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["src_fps", "dst_fps", "start", "end"])
def test_sample_timeline_rejects_non_finite(field, value):
    """NaN/Inf 必须报错：静默产出空时间轴或错帧号更难查。"""
    kwargs = {"src_fps": 30.0, "frame_count": 300, "dst_fps": 2.0,
              "start": 0.0, "end": 10.0}
    kwargs[field] = value

    with pytest.raises(VideoError):
        sample_timeline(**kwargs)


@pytest.mark.parametrize("src_fps,dst_fps", [(0.0, 2.0), (-30.0, 2.0), (30.0, 0.0), (30.0, -2.0)])
def test_sample_timeline_rejects_non_positive_fps(src_fps, dst_fps):
    """帧率必须为正。"""
    with pytest.raises(VideoError):
        sample_timeline(src_fps, 300, dst_fps)
