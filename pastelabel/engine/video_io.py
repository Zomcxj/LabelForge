"""视频抽帧：探测、时间轴、解码导出与清单读写。

cv2 只在真正解码的函数里导入（与 onnxruntime 同样处理），缺 cv2 时图片
功能照常可用，只有视频入口不可用。

`sample_timeline()` 是纯函数，只算时刻与帧号，不碰 cv2；文件是否损坏、
实际帧数与探测是否相符属于 `extract_frames()` 的职责。
"""
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

MANIFEST_NAME = "video_manifest.json"
MANIFEST_VERSION = 1


class VideoError(RuntimeError):
    """视频探测、抽帧或清单读写失败。"""


@dataclass(frozen=True)
class VideoInfo:
    """探测到的视频元数据。"""
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class Sample:
    """一个采样点：计划时刻、实际源帧号与该帧的真实时间。"""
    source_frame: int
    timestamp: float
    sample_time: float


@dataclass
class Timeline:
    """采样时间轴；start_time/end_time 是解析后的实际范围（缺省补全）。"""
    target_fps: float
    start_time: float
    end_time: float
    samples: List[Sample] = field(default_factory=list)


@dataclass(frozen=True)
class ManifestFrame:
    """清单里的一条：文件名、绝对源帧号、两种时间。"""
    file: str
    source_frame: int
    timestamp: float       # 实际源帧时间 = source_frame / source_fps
    sample_time: float     # 计划采样时刻 t_k


@dataclass
class Manifest:
    """抽帧映射清单；complete 只有全部预期帧成功落盘才为真。"""
    source: str
    source_name: str
    width: int
    height: int
    source_fps: float
    frame_count: int
    duration: float
    target_fps: float
    start_time: float
    end_time: float
    mode: str = "sample"
    complete: bool = False
    frames: List[ManifestFrame] = field(default_factory=list)
    version: int = MANIFEST_VERSION


@dataclass
class ExtractionResult:
    """抽帧结果：输出位置、清单与本次成功落盘的图片。"""
    output_dir: str
    manifest_path: str
    manifest: Manifest
    image_paths: List[str] = field(default_factory=list)
    cancelled: bool = False
    actual_frame_count: int = 0


def write_manifest(path, manifest):
    """原子写清单：先写同目录临时文件再 os.replace，中断不留半截 JSON。

    与 inference_task.write_sidecar_boxes 同一写法。
    """
    payload = {
        "version": manifest.version,
        "source": manifest.source,
        "source_name": manifest.source_name,
        "width": manifest.width,
        "height": manifest.height,
        "source_fps": manifest.source_fps,
        "frame_count": manifest.frame_count,
        "duration": manifest.duration,
        "target_fps": manifest.target_fps,
        "start_time": manifest.start_time,
        "end_time": manifest.end_time,
        "mode": manifest.mode,
        "complete": manifest.complete,
        "frames": [
            {"file": f.file, "source_frame": f.source_frame,
             "timestamp": f.timestamp, "sample_time": f.sample_time}
            for f in manifest.frames
        ],
    }
    tmp_path = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise VideoError(f"清单写入失败：{path}（{exc}）") from exc


def _is_safe_relative_name(name):
    """只接受输出目录内的相对文件名（无目录分隔、非绝对、非空、非 . 与 ..）。"""
    if not isinstance(name, str) or not name:
        return False
    if name in (".", ".."):
        return False
    if os.path.isabs(name) or os.path.splitdrive(name)[0]:
        return False
    if "/" in name or "\\" in name:
        return False
    return name == os.path.basename(name)


def read_manifest(path):
    """读并校验清单；任何不一致都报错，不返回半可信的清单。

    ④B 正式合成只依赖清单与图片，因此这里必须挡住：版本不认识、时间/FPS
    非有限、尺寸/帧数非正、帧号不递增或重复、时间与帧号不自洽、文件名越出
    输出目录。complete 原样带出，由调用方决定是否可用于正式合成。
    """
    if not os.path.isfile(path):
        raise VideoError(f"清单不存在：{path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise VideoError(f"清单无法解析：{path}（{exc}）") from exc
    if not isinstance(data, dict):
        raise VideoError(f"清单格式不对（应为对象）：{path}")

    version = data.get("version")
    if version != MANIFEST_VERSION:
        raise VideoError(f"清单版本不支持：{version!r}（期望 {MANIFEST_VERSION}）")

    def number(key):
        value = _check_finite(data.get(key), f"清单字段 {key}")
        if value <= 0:
            raise VideoError(f"清单字段 {key} 必须为正：{value}")
        return value

    def positive_int(key):
        value = data.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise VideoError(f"清单字段 {key} 必须是正整数：{value!r}")
        return value

    source = data.get("source")
    source_name = data.get("source_name")
    if not isinstance(source, str) or not isinstance(source_name, str):
        raise VideoError("清单缺少 source/source_name")

    source_fps = number("source_fps")
    frame_count = positive_int("frame_count")
    duration = number("duration")
    target_fps = number("target_fps")
    start_time = _check_finite(data.get("start_time"), "清单字段 start_time")
    end_time = _check_finite(data.get("end_time"), "清单字段 end_time")
    if start_time < 0 or start_time >= end_time:
        raise VideoError(f"清单时间范围非法：{start_time} → {end_time}")

    raw_frames = data.get("frames")
    if not isinstance(raw_frames, list):
        raise VideoError("清单 frames 必须是列表")

    frames = []
    previous = None
    for index, entry in enumerate(raw_frames):
        if not isinstance(entry, dict):
            raise VideoError(f"清单第 {index + 1} 条不是对象")
        name = entry.get("file")
        if not _is_safe_relative_name(name):
            raise VideoError(f"清单第 {index + 1} 条文件名越出输出目录：{name!r}")
        frame_no = entry.get("source_frame")
        if not isinstance(frame_no, int) or isinstance(frame_no, bool):
            raise VideoError(f"清单第 {index + 1} 条 source_frame 不是整数")
        if frame_no < 0 or frame_no >= frame_count:
            raise VideoError(f"清单第 {index + 1} 条 source_frame 越界：{frame_no}")
        if previous is not None and frame_no <= previous:
            raise VideoError(
                f"清单第 {index + 1} 条 source_frame 未递增：{frame_no} ≤ {previous}")
        previous = frame_no
        timestamp = _check_finite(entry.get("timestamp"), "清单字段 timestamp")
        sample_time = _check_finite(entry.get("sample_time"), "清单字段 sample_time")
        # timestamp 是实际源帧时间，必须与帧号自洽（一个源帧间隔的容差）
        if abs(timestamp - frame_no / source_fps) > 1.0 / source_fps:
            raise VideoError(
                f"清单第 {index + 1} 条 timestamp 与 source_frame 不符："
                f"{timestamp} ≠ {frame_no}/{source_fps}")
        if sample_time < start_time - 1.0 / target_fps or sample_time >= end_time:
            raise VideoError(
                f"清单第 {index + 1} 条 sample_time 越出范围：{sample_time}")
        frames.append(ManifestFrame(file=name, source_frame=frame_no,
                                    timestamp=timestamp, sample_time=sample_time))

    mode = data.get("mode", "sample")
    if mode != "sample":
        raise VideoError(f"清单 mode 不支持：{mode!r}")

    return Manifest(
        source=source, source_name=source_name,
        width=positive_int("width"), height=positive_int("height"),
        source_fps=source_fps, frame_count=frame_count, duration=duration,
        target_fps=target_fps, start_time=start_time, end_time=end_time,
        mode=mode, complete=bool(data.get("complete")), frames=frames,
        version=version,
    )


def _check_finite(value, name):
    """有限性校验：NaN/Inf 会静默产出空时间轴或错帧号，必须挡在入口。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise VideoError(f"{name} 不是有效数字：{value!r}")
    if not math.isfinite(number):
        raise VideoError(f"{name} 必须是有限数字：{value!r}")
    return number


def sample_timeline(src_fps, frame_count, dst_fps, start=None, end=None):
    """按时间决定采样帧：t_k = start + k/dst_fps，映射到最近的合法源帧。

    采样按时间算，不按固定帧步长整除——29.97 FPS 按 2 FPS 抽帧时
    int(29.97/2)=14 只能得到约 2.14 FPS，不能代表目标时间间隔。
    候选帧取 floor(t*fps + 0.5)（半帧向后取，避免 round() 的奇偶舍入），
    再钳入 [ceil(start*fps), min(frame_count-1, ceil(end*fps)-1)]。
    """
    src_fps = _check_finite(src_fps, "源帧率")
    dst_fps = _check_finite(dst_fps, "目标帧率")
    if src_fps <= 0:
        raise VideoError(f"源帧率必须为正：{src_fps}")
    if dst_fps <= 0:
        raise VideoError(f"目标帧率必须为正：{dst_fps}")

    duration = frame_count / src_fps
    start = 0.0 if start is None else _check_finite(start, "起始时间")
    end = duration if end is None else _check_finite(end, "结束时间")
    if start < 0 or end > duration or start >= end:
        raise VideoError(
            f"时间范围非法：start={start}, end={end}, 时长={duration}")

    # 抽帧不重复源帧：目标帧率高于源帧率没有意义，钳到源帧率
    target_fps = min(dst_fps, src_fps)

    lo = math.ceil(start * src_fps - 1e-9)
    hi = min(frame_count - 1, math.ceil(end * src_fps - 1e-9) - 1)
    if lo > hi:
        raise VideoError(
            f"时间范围内没有源帧：start={start}, end={end}, 源帧率={src_fps}")

    samples = []
    last = None
    k = 0
    while True:
        sample_time = start + k / target_fps
        if sample_time >= end:
            break
        frame = int(math.floor(sample_time * src_fps + 0.5))
        frame = min(max(frame, lo), hi)
        if frame != last:                      # 区间过窄时相邻采样点会撞同一源帧
            samples.append(Sample(
                source_frame=frame,
                timestamp=frame / src_fps,     # 实际源帧时间，不是计划时刻
                sample_time=sample_time,
            ))
            last = frame
        k += 1

    return Timeline(target_fps=target_fps, start_time=start, end_time=end,
                    samples=samples)


def _load_cv2():
    """延迟导入 cv2：缺它时图片功能照常可用，只有视频入口不可用。"""
    try:
        import cv2
    except ImportError as exc:
        raise VideoError("未安装 opencv（cv2），视频功能不可用") from exc
    return cv2


def probe(path):
    """探测视频元数据；读不出有效元数据直接报错，不产半成品。

    帧率/尺寸/帧数任一无效（0/负/非有限）都报错：后续时间轴按这些值
    计算，放任无效值会算出空时间轴或错帧号。
    """
    cv2 = _load_cv2()

    if not os.path.isfile(path):
        raise VideoError(f"视频文件不存在：{path}")

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise VideoError(f"无法打开视频（格式不支持或文件损坏）：{path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise VideoError(f"视频帧率无效：{fps}（{path}）")
    if frame_count <= 0:
        raise VideoError(f"视频帧数无效：{frame_count}（{path}）")
    if width <= 0 or height <= 0:
        raise VideoError(f"视频尺寸无效：{width}x{height}（{path}）")

    return VideoInfo(
        path=path, fps=fps, frame_count=frame_count,
        width=width, height=height, duration=frame_count / fps,
    )


def read_image(path):
    """读图片为 BGR 数组；读不出来返回 None（由调用方决定如何报错）。

    不能用 cv2.imread：Windows 上它按 ANSI 打开文件，中文路径直接失败
    （实测返回 None，连 imwrite 自己写出的中文路径文件也读不回来）。
    走 Python 读字节 + imdecode，两者都支持任意 Unicode 路径。
    """
    cv2 = _load_cv2()
    if not os.path.isfile(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    try:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def write_image(path, frame, ext="jpg", quality=95):
    """把 BGR 数组写成图片；失败抛 VideoError，不静默返回。

    与 read_image 同理不能用 cv2.imwrite：中文路径下它返回 False 且不写
    任何文件（实测），静默失败会让清单记下并不存在的图片。编码失败、目录
    不存在、写盘异常都报错。
    """
    cv2 = _load_cv2()
    suffix = "." + ext.lstrip(".").lower()
    params = []
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    try:
        ok, buffer = cv2.imencode(suffix, frame, params)
    except cv2.error as exc:
        raise VideoError(f"图片编码失败（{suffix}）：{exc}") from exc
    if not ok:
        raise VideoError(f"图片编码失败（{suffix}）：{path}")

    try:
        with open(path, "wb") as handle:
            handle.write(buffer.tobytes())
    except OSError as exc:
        raise VideoError(f"图片写入失败：{path}（{exc}）") from exc


def output_dir_for(video_path):
    """抽帧输出目录：视频同级 <主文件名>_frames/；同名时同级加序号。

    只改变文件夹名，不跳到程序目录、工作目录或用户另选位置。父目录不可写
    由调用方在写入时发现（这里不预检权限，避免 TOCTOU 与平台差异）。
    """
    parent = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    candidate = os.path.join(parent, f"{stem}_frames")
    index = 2
    while os.path.exists(candidate):
        candidate = os.path.join(parent, f"{stem}_frames_{index}")
        index += 1
    return candidate


def _decode_at(capture, target_frame, current):
    """顺序解码到 target_frame：grab() 快进、retrieve() 取帧。

    不用 cap.set(CAP_PROP_POS_FRAMES)：部分后端 seek 不准，且逐帧 grab 更快。
    grab() 每次把内部指针推进一帧，因此取第 n 帧需要已 grab 过 n+1 次；
    返回 (frame 或 None, 新的 current)，None 表示提前 EOF 或解码失败。
    """
    while current <= target_frame:
        if not capture.grab():
            return None, current
        current += 1
    ok, frame = capture.retrieve()
    if not ok or frame is None:
        return None, current
    return frame, current


def extract_frames(video_path, target_fps, start=None, end=None,
                   image_format="jpg", quality=95, output_dir=None,
                   cancel=None, progress=None):
    """按目标帧率抽帧，写图片与清单，返回 ExtractionResult。

    图片只有成功落盘才计入清单。取消、提前 EOF、写盘失败、实际帧数与探测
    不符都不会被当作成功：清单标记 complete: false，已完成的部分保留。
    """
    info = probe(video_path)
    timeline = sample_timeline(info.fps, info.frame_count, target_fps,
                               start=start, end=end)

    # 参数校验通过后才创建输出目录，非法输入不留空目录
    out_dir = output_dir or output_dir_for(video_path)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise VideoError(f"输出目录不可用：{out_dir}（{exc}）") from exc

    stem = os.path.splitext(os.path.basename(video_path))[0]
    suffix = "." + image_format.lstrip(".").lower()
    manifest = Manifest(
        source=os.path.abspath(video_path),
        source_name=os.path.basename(video_path),
        width=info.width, height=info.height,
        source_fps=info.fps, frame_count=info.frame_count,
        duration=info.duration,
        target_fps=timeline.target_fps,
        start_time=timeline.start_time,
        end_time=timeline.end_time,
        complete=False,
    )

    cv2 = _load_cv2()
    capture = cv2.VideoCapture(video_path)
    frames = []
    cancelled = False
    current = 0
    actual_count = 0
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    # 开始就写 complete: false，中断/强制结束都留下未完成标记，不冒充完整
    write_manifest(manifest_path, manifest)

    try:
        if not capture.isOpened():
            raise VideoError(f"无法打开视频：{video_path}")

        for index, sample in enumerate(timeline.samples):
            if cancel and cancel():
                cancelled = True
                break
            frame, current = _decode_at(capture, sample.source_frame, current)
            if frame is None:
                break                      # 提前 EOF 或解码失败
            if progress:
                progress(index + 1, len(timeline.samples))

            name = f"{stem}_{index + 1:06d}{suffix}"
            target = os.path.join(out_dir, name)
            try:
                write_image(target, frame, image_format, quality)
            except VideoError:
                break                      # 写盘失败：该帧不进清单
            frames.append(ManifestFrame(
                file=name, source_frame=sample.source_frame,
                timestamp=sample.timestamp, sample_time=sample.sample_time,
            ))
            # 每写一张就更新清单：进程被强杀时也能看出抽到哪
            manifest.frames = frames
            write_manifest(manifest_path, manifest)

        # 全片抽帧需读至 EOF 核对计数；指定片段只验证该片段所需帧
        if not cancelled and start is None and end is None:
            while capture.grab():
                actual_count += 1
            actual_count += current
        else:
            actual_count = current
    finally:
        capture.release()

    expected = len(timeline.samples)
    truncated = (start is None and end is None
                 and actual_count != info.frame_count)
    manifest.frames = frames
    manifest.complete = bool(frames) and not cancelled and not truncated \
        and len(frames) == expected
    write_manifest(manifest_path, manifest)

    return ExtractionResult(
        output_dir=out_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        image_paths=[os.path.join(out_dir, f.file) for f in frames],
        cancelled=cancelled,
        actual_frame_count=actual_count,
    )


# ---------- ④B 抽帧标注合成 ----------

COMPOSE_SUFFIX = ".mp4"


@dataclass
class ComposePlan:
    """合成计划：多少张图、多长、输出帧率，以及每帧用哪张图。

    两种输入源：
    - 有清单：时长 = 清单 end_time - start_time，输出帧率 = N / 时长。
    - 无清单：用户给每秒几帧，时长 = N / 该帧率。

    `one_frame_per_image` 是低帧率模式：一张图写一帧，输出帧率 = N/D。
    默认走标准帧率（每张图撑满自己的时间槽），否则播放器按低帧率播放、
    拖进度条会跳。
    """
    image_count: int
    duration: float
    output_fps: float
    total_frames: int
    one_frame_per_image: bool = False

    def image_index_for_frame(self, frame_index):
        """第 j 个输出帧用第几张图（0 起）。

        用 `int(j * N / total)` 均匀映射：不能用整除（21 张 10.1 秒按
        30 fps 整除成每张 14 帧 → 294 帧 = 9.8 秒，少 300ms）。
        """
        if frame_index < 0 or frame_index >= self.total_frames:
            raise VideoError(f"帧号越界：{frame_index}")
        if self.one_frame_per_image:
            return frame_index
        return min(self.image_count - 1,
                   int(frame_index * self.image_count / self.total_frames))


def compose_plan(image_count, duration=None, output_fps=None,
                 one_frame_per_image=False, standard_fps=None):
    """算合成计划。

    必须且只能给一个时间基准：
    - `duration`（有清单时用清单时长）→ 输出帧率 = N / duration
    - `output_fps`（无清单时用户指定）→ 时长 = N / output_fps

    `standard_fps` 给定时走标准帧率输出（每张图撑满时间槽，总帧数 =
    round(duration * standard_fps)）；不给则一张图一帧。
    """
    if not isinstance(image_count, int) or isinstance(image_count, bool):
        raise VideoError(f"图片数量必须是整数：{image_count!r}")
    if image_count <= 0:
        raise VideoError(f"图片数量必须为正：{image_count}")
    if (duration is None) == (output_fps is None):
        raise VideoError("时长与输出帧率必须只给一个")

    if duration is not None:
        duration = _check_finite(duration, "合成时长")
        if duration <= 0:
            raise VideoError(f"合成时长必须为正：{duration}")
        fps = image_count / duration
    else:
        fps = _check_finite(output_fps, "输出帧率")
        if fps <= 0:
            raise VideoError(f"输出帧率必须为正：{fps}")
        duration = image_count / fps

    if one_frame_per_image:
        return ComposePlan(image_count=image_count, duration=duration,
                           output_fps=fps, total_frames=image_count,
                           one_frame_per_image=True)

    target = _check_finite(standard_fps, "标准帧率") if standard_fps else fps
    if target <= 0:
        raise VideoError(f"标准帧率必须为正：{target}")
    total = max(image_count, round(duration * target))
    return ComposePlan(image_count=image_count, duration=duration,
                       output_fps=total / duration, total_frames=total)


def list_sequence_images(images_dir):
    """外部序列的图片列表，按自然排序（img2 在 img10 之前）。

    只收支持的图片扩展名；sidecar json 与其它文件不进入序列。
    """
    if not os.path.isdir(images_dir):
        raise VideoError(f"目录不存在：{images_dir}")

    from ..core.config import SUPPORTED_IMAGE_EXTENSIONS
    from ..core.utils import natural_sort_key

    try:
        names = os.listdir(images_dir)
    except OSError as exc:
        raise VideoError(f"无法读取目录：{images_dir}（{exc}）") from exc

    exts = {e.lower() for e in SUPPORTED_IMAGE_EXTENSIONS}
    paths = [
        os.path.join(images_dir, name) for name in names
        if os.path.splitext(name)[1].lower() in exts
    ]
    paths.sort(key=lambda p: natural_sort_key(os.path.basename(p)))
    if not paths:
        raise VideoError(f"目录里没有图片：{images_dir}")
    return paths


@dataclass
class ComposeResult:
    """合成结果：输出文件、张数、时长与实际帧率。

    `unlabeled_count` 只在开启了标签框叠加时有意义：统计没有可用标注
    （sidecar 缺失或 shapes 为空）的图片张数。
    """
    output_path: str
    image_count: int
    duration: float
    output_fps: float
    total_frames: int
    cancelled: bool = False
    frames_written: int = 0
    unlabeled_count: int = 0


def _compose_images_and_duration(manifest_path, images_dir, output_fps,
                                 duration):
    """决定图片序列与时间基准：清单优先，没有清单才用用户给的帧率。"""
    if manifest_path is not None:
        manifest = read_manifest(manifest_path)
        if not manifest.complete:
            raise VideoError("清单未完成（complete: false），不能合成")
        if not manifest.frames:
            raise VideoError("清单里没有帧，不能合成")
        base = os.path.dirname(os.path.abspath(manifest_path))
        paths = [os.path.join(base, f.file) for f in manifest.frames]
        # 清单时长优先：用户同时给了帧率也不采用，避免与清单矛盾
        return paths, manifest, manifest.end_time - manifest.start_time

    if images_dir is None:
        raise VideoError("必须给清单或图片目录")
    return list_sequence_images(images_dir), None, None


def _read_sidecar_boxes_for_compose(image_path):
    """读一张图的标注，返回 `(boxes, 是否有可用标注)`。

    - sidecar 不存在 → `([], False)`：按无标签画面写入，计入「无标注」。
    - 存在但读不出来/结构不对 → 抛 `VideoError`。**不当作无标注**：
      损坏的标注静默变成干净画面，用户会以为这段本来就没标。
    - 正常解析出框 → `(boxes, True)`；shapes 为空（或没解析出有效框）
      也是 `([], False)`——画面上确实没有框。
    """
    json_path = f"{os.path.splitext(image_path)[0]}.json"
    if not os.path.exists(json_path):
        return [], False
    try:
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise VideoError(f"标注读不出来：{json_path}（{exc}）") from exc
    if not isinstance(data, dict) or not isinstance(data.get("shapes"), list):
        raise VideoError(f"标注结构不对（缺少 shapes 列表）：{json_path}")

    from .shape_io import box_from_labelme_shape
    boxes = []
    for shape in data["shapes"]:
        box = box_from_labelme_shape(shape)
        if box is not None:
            boxes.append(box)
    return boxes, bool(boxes)


def _render_bgr_frame(frame_bgr, boxes, render):
    """在 BGR 帧上叠加标注，返回 BGR 帧。

    绘制模块吃 RGB，合成链路给的是 BGR；转两次，别把画面颜色搞反。
    """
    cv2 = _load_cv2()
    from .video_render import draw_boxes

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    drawn = draw_boxes(rgb, boxes, render)
    return cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)


def scan_sidecar_labels(image_paths):
    """扫描一组图片的 sidecar，统计每个类别出现的次数。

    返回 `{类别: 次数}`，按次数从多到少、同次数按名称排序（界面直接照着
    列表展示，顺序稳定）。缺 sidecar 的图跳过；**标注损坏则报错**，不能
    当成「那张图没标注」——否则界面会少列类别，用户以为自己没标这类。
    """
    counts = {}
    for image_path in image_paths:
        boxes, has_labels = _read_sidecar_boxes_for_compose(image_path)
        if not has_labels:
            continue
        for box in boxes:
            label = str(box.get("label") or "")
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _slice_paths(paths, start_index=None, end_index=None):
    """按「从第几张到第几张」裁出要合成的图片（**都是包含端点的下标**）。

    用户要求「视频合成支持选择从哪张到哪张，默认 0 到最后」。裁在**建
    plan 之前**：这样帧号、进度、清单时长全都基于裁过的序列，不需要在
    循环里到处判范围（漏一处就会写出越界的帧）。

    两端都可以为 None（= 不裁那一端）。越界不报错而是夹紧——界面上的
    上限就是图片张数，用户手输超了不该让整次合成失败。`start > end`
    是真的没得合成，返回空列表由调用方报错。
    """
    total = len(paths)
    start = 0 if start_index is None else int(start_index)
    end = total - 1 if end_index is None else int(end_index)
    if total == 0:
        return []
    # 「空范围」要在夹紧**之前**判：夹紧会把 end=-1 抬成 0，那样 0..-1 就会
    # 悄悄变成「只合成第一张」，用户选了空却拿到一帧，比报错更糟。
    # 只把「真的没有重叠」当空：start 已经超过末张，或 end 还没到首张。
    if start > total - 1 or end < 0 or start > end:
        return []
    start = max(0, min(total - 1, start))
    end = max(0, min(total - 1, end))
    return list(paths[start:end + 1])


def compose_video(manifest_path=None, images_dir=None, output_path=None,
                  output_fps=None, one_frame_per_image=False,
                  standard_fps=None, cancel=None, progress=None, render=None,
                  start_index=None, end_index=None):
    """把抽帧图片合成为视频（④B）。

    有清单按清单顺序与时长；没有清单则用 `output_fps`（每秒几帧）。
    `start_index`/`end_index` 是**包含端点**的下标，用来只合成其中一段
    （默认从第一张到最后一张）。时长按裁后的帧数等比折算，否则「选 10 张
    却按整段时长出片」会变成极慢的播放。
    编码走同目录 `.partial.mp4` 临时文件，全部校验通过才 `os.replace`
    到正式名——失败或取消不留看似成功的输出。

    `render` 给一个 `video_render.RenderSpec` 时叠加标签框（可选类别、
    可选类别名与置信度）；不给就是纯合成，一个 sidecar 都不读。
    """
    cv2 = _load_cv2()
    paths, manifest, duration = _compose_images_and_duration(
        manifest_path, images_dir, output_fps, None)

    total_before = len(paths)
    paths = _slice_paths(paths, start_index, end_index)
    if not paths:
        raise VideoError("选择的图片范围是空的，没有可合成的帧")

    # 裁过的序列：时长按比例折，否则 10 张图会摊到整段视频的时长
    if duration is not None and total_before:
        duration = duration * (len(paths) / total_before)

    plan = compose_plan(image_count=len(paths), duration=duration,
                        output_fps=output_fps if duration is None else None,
                        one_frame_per_image=one_frame_per_image,
                        standard_fps=standard_fps)

    if not output_path:
        raise VideoError("必须指定输出路径")
    output_path = os.path.abspath(output_path)
    if manifest is not None and os.path.normcase(output_path) == \
            os.path.normcase(os.path.abspath(manifest.source)):
        raise VideoError(f"输出路径与源视频相同：{output_path}")

    parent = os.path.dirname(output_path)
    try:
        os.makedirs(parent or ".", exist_ok=True)
    except OSError as exc:
        raise VideoError(f"输出目录不可用：{parent}（{exc}）") from exc

    # 先读第一张定尺寸，尺寸不一致直接报错（不静默缩放）
    first = read_image(paths[0])
    if first is None:
        raise VideoError(f"图片读不出来：{paths[0]}")
    height, width = first.shape[:2]
    if manifest is not None and (width, height) != (manifest.width, manifest.height):
        raise VideoError(
            f"图片尺寸与清单不符：{width}x{height} != "
            f"{manifest.width}x{manifest.height}")

    tmp_path = os.path.join(parent, f".{os.path.basename(output_path)}.partial.mp4")
    writer = None
    frames_written = 0
    cancelled = False
    failed = False
    unlabeled = 0

    def _prepare(index, frame):
        """校验尺寸并按需叠加标注；返回 (可用帧, 该图是否无标注)。

        第一张和后续帧都走这里。首图若绕过这一步，它的框就永远画不出来，
        无标注计数也会少一张。
        """
        if frame.shape[:2] != (height, width):
            raise VideoError(
                f"图片尺寸不一致：{paths[index]} "
                f"{frame.shape[1]}x{frame.shape[0]} != {width}x{height}")
        if render is None:
            return frame, False
        boxes, has_labels = _read_sidecar_boxes_for_compose(paths[index])
        if not has_labels:
            # sidecar 缺失，或 shapes 为空/没解析出框：画面照写，但确实没框
            return frame, True
        return _render_bgr_frame(frame, boxes, render), False

    try:
        if render is not None:
            first, first_unlabeled = _prepare(0, first)
            unlabeled += int(first_unlabeled)

        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 plan.output_fps, (width, height))
        if not writer.isOpened():
            raise VideoError(f"编码器打不开：{tmp_path}")

        cache_index, cache_frame = 0, first
        last_written = -1
        for j in range(plan.total_frames):
            if cancel is not None and cancel():
                cancelled = True
                break
            index = plan.image_index_for_frame(j)
            if index != cache_index or cache_frame is None:
                frame = read_image(paths[index])
                if frame is None:
                    raise VideoError(f"图片读不出来：{paths[index]}")
                frame, missing = _prepare(index, frame)
                unlabeled += int(missing)
                cache_index, cache_frame = index, frame
            # write() 的 Python 返回值是 None，不是成功布尔值，不能用来判失败
            writer.write(cache_frame)
            frames_written += 1
            if index != last_written:
                last_written = index
                if progress is not None:
                    progress(index + 1, len(paths))
    except BaseException:
        # 异常也要清掉临时文件：半个视频留在盘上会被当成可用的产物
        failed = True
        raise
    finally:
        if writer is not None:
            writer.release()
        if failed:
            _remove_quietly(tmp_path)

    if cancelled:
        _remove_quietly(tmp_path)
        return ComposeResult(output_path=output_path, image_count=len(paths),
                             duration=plan.duration, output_fps=plan.output_fps,
                             total_frames=plan.total_frames, cancelled=True,
                             frames_written=frames_written,
                             unlabeled_count=unlabeled)

    # 回读验收：帧数、尺寸、时长都要对得上才发布正式文件
    actual_frames, actual_fps = _probe_output(tmp_path)
    if actual_frames != plan.total_frames:
        _remove_quietly(tmp_path)
        raise VideoError(
            f"合成结果帧数不符：{actual_frames} != {plan.total_frames}")
    if actual_fps <= 0 or abs(actual_fps - plan.output_fps) > 0.05:
        _remove_quietly(tmp_path)
        raise VideoError(
            f"合成结果帧率不符：{actual_fps} != {plan.output_fps}")

    try:
        os.replace(tmp_path, output_path)
    except OSError as exc:
        _remove_quietly(tmp_path)
        raise VideoError(f"输出文件写入失败：{output_path}（{exc}）") from exc

    return ComposeResult(output_path=output_path, image_count=len(paths),
                         duration=plan.duration, output_fps=plan.output_fps,
                         total_frames=plan.total_frames,
                         frames_written=frames_written,
                         unlabeled_count=unlabeled)


def _probe_output(path):
    """回读合成产物：返回 (帧数, 帧率)。读不开则报错。"""
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise VideoError(f"合成产物读不回来：{path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count, fps


@dataclass
class RenderVideoResult:
    """④A 逐帧推理编码的结果。

    `frames_written` 在取消时表示已写进去的帧数；`source_frame_count` 是
    探测到的源帧数，用来对照 `frame_count`（实际编码帧数）。
    `partial_path` 只在取消时非空：那是保留下来的可播放半成品（`.partial.mp4`），
    正式输出名始终不会被占用。
    """

    output_path: str
    frame_count: int
    source_frame_count: int
    duration: float
    output_fps: float
    cancelled: bool = False
    frames_written: int = 0
    detected_total: int = 0
    partial_path: str = ""


class _VerifyCancelled(Exception):
    """内部信号：回读校验途中被取消（不是失败，不该按失败清理）。"""


def _partial_is_playable(tmp_path, info):
    """取消时判断半成品能不能播：能解出第一帧且尺寸正确才算。

    只解一帧就够——writer 正常 `release()` 过，容器是完整的。零帧取消时
    mp4v 仍会留下几十字节的文件头，按「文件存在」判断会留下一个打不开的
    假产物。尺寸也要查：编码器截过边的半成品同样是废的。
    """
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(tmp_path)
    try:
        if not capture.isOpened():
            return False
        ok, frame = capture.read()
        if not ok or frame is None:
            return False
        height, width = frame.shape[:2]
        return (width, height) == (info.width, info.height)
    finally:
        capture.release()


def _verify_rendered_output(tmp_path, info, cancel=None):
    """顺序解码临时产物，返回实际帧数；同时核对尺寸。

    不用 `CAP_PROP_FRAME_COUNT`：它可能报的是容器声明值，与真能解出来的
    帧数不符（尤其被截断的文件）。逐帧 read() 才是实际能播放的长度。

    尺寸必须逐帧核对：实测 mp4v 对 65x49 的 `isOpened()` 返回 True，
    但回读是 64x48——只看 isOpened 会产出一条尺寸不对的片子。

    `cancel` 每帧查一次：长视频的回读本身也要花时间，用户按了取消就该停
    （抛 `_VerifyCancelled`，由调用方按「取消」而不是「失败」处理）。
    """
    cv2 = _load_cv2()
    capture = cv2.VideoCapture(tmp_path)
    if not capture.isOpened():
        raise VideoError(f"产物读不回来：{tmp_path}")
    count = 0
    try:
        while True:
            if cancel is not None and cancel():
                raise _VerifyCancelled()
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if (width, height) != (info.width, info.height):
                raise VideoError(
                    f"产物尺寸被截边：{width}x{height} != "
                    f"{info.width}x{info.height}（编码器不支持该尺寸）")
            count += 1
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return count, fps

def _cancelled_render_result(tmp_path, output_path, info, frames_written,
                             detected_total):
    """取消收尾：能播的半成品留在 `.partial.mp4`，正式输出名绝不占用。

    ④A 是全app最慢的一条路（18 秒视频要推理 541 次），用户中途取消时已经
    跑完的那几分钟不该白费——留一个能播的半成品，比删掉更有用。零帧取消
    或尺寸被截边的半成品是废的，直接清掉，不留打不开的假产物。
    """
    if frames_written > 0 and _partial_is_playable(tmp_path, info):
        partial = tmp_path
    else:
        _remove_quietly(tmp_path)
        partial = ""
    return RenderVideoResult(
        output_path=output_path, frame_count=frames_written,
        source_frame_count=info.frame_count, duration=info.duration,
        output_fps=info.fps, cancelled=True, frames_written=frames_written,
        detected_total=detected_total, partial_path=partial)


def render_video(video_path, detector, output_path=None, resize_fn=None,
                 scope=None, render=None, class_filter=None,
                 cancel=None, progress=None, on_frame=None):
    """④A：原视频逐帧推理，把框画在每一帧上，再编码成新视频。

    与 ④B（`compose_video`）的区别：画面来自原视频的**全部**帧而不是抽帧
    图片，标注是现场推理得到的，所以不需要 ② 的清单。输出帧数、分辨率、
    帧率、时长都与源一致。

    `scope` 是 ③ 的范围配置（None = 全图），复用 `detect_in_scope()`。
    `render` 是 `video_render.RenderSpec`（None = 只推理不绘制）；
    类别色必须由调用方在主线程预先取好放进 spec，后台不能读 editor 取色。

    编码走同目录 `.partial.mp4` 临时文件，逐帧回读核对帧数/尺寸/帧率通过
    才 `os.replace` 到正式名——失败不留看似成功的输出，也不动旧成品。

    取消是例外：跑完的帧不白费，能播的半成品以 `.partial.mp4` 名保留并在
    结果里给出 `partial_path`（正式输出名始终不占用）。零帧取消或半成品
    尺寸不对则清掉。这是本条与 ④B（取消即删）的唯一分歧，见计划 §4.5。
    """
    cv2 = _load_cv2()
    if not output_path:
        raise VideoError("必须指定输出路径")
    if resize_fn is None:
        raise VideoError("必须提供 resize_fn（模型输入缩放函数）")

    info = probe(video_path)
    output_path = os.path.abspath(output_path)
    if os.path.normcase(output_path) == os.path.normcase(
            os.path.abspath(info.path)):
        raise VideoError(f"输出路径与源视频相同：{output_path}")

    parent = os.path.dirname(output_path)
    try:
        os.makedirs(parent or ".", exist_ok=True)
    except OSError as exc:
        raise VideoError(f"输出目录不可用：{parent}（{exc}）") from exc

    # 范围在启动时定稿：非法配置在这里就报错，不进后台跑出「成功但 0 框」
    from .inference_scope import check_reference_size, normalize_scope
    active_scope = normalize_scope(scope, (info.width, info.height))
    if active_scope is not None:
        check_reference_size(active_scope, (info.width, info.height))

    tmp_path = os.path.join(
        parent, f".{os.path.basename(output_path)}.partial.mp4")
    capture = None
    writer = None
    frames_written = 0
    detected_total = 0
    cancelled = False
    failed = False

    try:
        capture = cv2.VideoCapture(info.path)
        if not capture.isOpened():
            raise VideoError(f"无法打开视频：{info.path}")

        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))
        if not writer.isOpened():
            raise VideoError(f"编码器打不开：{tmp_path}")

        while frames_written < info.frame_count:
            if cancel is not None and cancel():
                cancelled = True
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                raise VideoError(
                    f"视频提前结束：只解出 {frames_written} 帧，"
                    f"探测为 {info.frame_count} 帧")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if active_scope is not None:
                from .inference_task import detect_in_scope
                boxes = detect_in_scope(rgb, detector, resize_fn, active_scope)
            else:
                boxes = detector.detect(rgb, resize_fn)
            if class_filter is not None:
                allowed = {str(c) for c in class_filter}
                boxes = [b for b in boxes if b.get("label") in allowed]
            detected_total += len(boxes)

            if render is not None and boxes:
                from .video_render import draw_boxes
                rgb = draw_boxes(rgb, boxes, render)

            # write() 的 Python 返回值是 None，不是成功布尔值，不能判失败；
            # 尺寸不符的帧会被静默丢弃，所以尺寸由回读验收兜底
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            frames_written += 1
            if on_frame is not None:
                on_frame(frames_written)
            if progress is not None:
                progress(frames_written, info.frame_count)
    except BaseException:
        failed = True
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if failed:
            _remove_quietly(tmp_path)

    if cancelled:
        return _cancelled_render_result(
            tmp_path, output_path, info, frames_written, detected_total)

    # 回读验收也要能取消：长视频的逐帧回读本身就要花时间，用户按了取消就该停
    try:
        actual_frames, actual_fps = _verify_rendered_output(tmp_path, info,
                                                            cancel=cancel)
    except _VerifyCancelled:
        return _cancelled_render_result(
            tmp_path, output_path, info, frames_written, detected_total)
    if actual_frames != info.frame_count:
        _remove_quietly(tmp_path)
        raise VideoError(
            f"产物帧数不符：{actual_frames} != {info.frame_count}")
    if actual_fps <= 0 or abs(actual_fps - info.fps) > 0.05:
        _remove_quietly(tmp_path)
        raise VideoError(
            f"产物帧率不符：{actual_fps} != {info.fps}")

    try:
        os.replace(tmp_path, output_path)
    except OSError as exc:
        _remove_quietly(tmp_path)
        raise VideoError(f"输出文件写入失败：{output_path}（{exc}）") from exc

    return RenderVideoResult(
        output_path=output_path, frame_count=info.frame_count,
        source_frame_count=info.frame_count, duration=info.duration,
        output_fps=info.fps, frames_written=frames_written,
        detected_total=detected_total)


def _remove_quietly(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
