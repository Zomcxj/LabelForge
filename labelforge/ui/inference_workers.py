"""推理用的后台线程：模型加载与批量执行。

加载与推理都放到线程里，否则大模型会冻住界面。模型加载除了建会话，
还要解析类别名：同级同名 txt → 模型自带 names → 提示手动输入并写 txt。
"""
import os

from PySide6.QtCore import QThread, Signal

from ..engine import inference_task
from ..engine.inference_image import qimage_to_rgb_array, resize_for_model
from ..engine.onnx_inference import read_class_names
from .i18n import t as tr


def load_rgb(path):
    """读图并转成模型输入用的 RGB 数组；读不出来返回 None。"""
    from PySide6.QtGui import QImage
    image = QImage(path)
    if image.isNull():
        return None
    return qimage_to_rgb_array(image)


class ModelLoadWorker(QThread):
    """加载模型并解析类别名。

    need_class_names 表示模型同级没有同名 txt、模型自身也没有类别表，
    必须让用户提供；此时不加载模型，由界面收集后带着类别名重新发起。
    """
    loaded = Signal(object, object, object)     # detector, size, class_source
    # model_path, 建议的 txt 路径, 模型需要的类别数（推不出为 None）
    need_class_names = Signal(str, str, object)
    failed = Signal(str)

    def __init__(self, model_path, class_names=None, settings=None, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._class_names = list(class_names or [])
        settings = settings or {}
        self._conf = settings.get("conf_threshold")
        self._iou = settings.get("iou_threshold")
        self._input_size = settings.get("input_size")

    def run(self):
        try:
            from ..engine.onnx_inference import OnnxDetector

            detector = OnnxDetector(
                self._model_path,
                class_names=self._class_names,
                conf_threshold=self._conf,
                iou_threshold=self._iou,
                input_size=self._input_size,
            )
            session = detector.session        # 触发加载，类别缺失要在这里发现
            if not self._class_names:
                names, source = inference_task.resolve_class_names(
                    self._model_path, session)
                if source == "missing":
                    self.need_class_names.emit(
                        self._model_path,
                        inference_task.sidecar_class_names_path(self._model_path),
                        self._guess_class_count(session))
                    return
                detector.class_names = names
            else:
                source = "user"
            size = detector.model_input_size()
            # 在这里就把「不是检测模型」挡下来（例如误选了抠图权重）：
            # 主线程随后会解析任务与输出格式，失败时只能打印异常栈
            detector.resolve_task()
            if detector.resolve_task() == "det":
                detector.resolve_format()
            self.loaded.emit(detector, size, source)
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _guess_class_count(session):
        """从检测头的 shape 推类别数，供手输框提示该填几个；推不出返回 None。

        带辅助输出的训练导出（多输出）也要能给出数：先挑出唯一的检测头，
        再交给 guess_class_count，与真正推理时用的是同一个头。
        """
        from ..engine.onnx_inference import guess_class_count, pick_detection_output
        try:
            _, shape = pick_detection_output(session.get_outputs())
        except Exception:
            return None
        return guess_class_count(shape)


class InferenceWorker(QThread):
    """后台顺序推理。progress: (done, total, 文件名, 框数, 耗时毫秒)。

    每张跑完就通过 `result_ready` 把结果发回主线程，由主线程立刻写盘并刷新
    列表圆点；worker 自己不动磁盘（Qt 对象与 editor 状态都归主线程）。
    """
    progress = Signal(int, int, object, int, int)
    result_ready = Signal(int, object)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, paths, keys, detector, class_filter=None, scope=None,
                 parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._keys = list(keys)
        self._detector = detector
        self._class_filter = class_filter
        self._scope = scope
        self._interrupted = False

    def run(self):
        try:
            results = inference_task.run_batch(
                self._paths, self._detector, load_rgb, resize_for_model,
                keys=self._keys,
                progress_fn=self.progress.emit,
                is_interrupted=lambda: self._interrupted,
                class_filter=self._class_filter,
                scope=self._scope,
                on_result=lambda key, res: self.result_ready.emit(key, res),
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._interrupted = True


class WeightDownloadWorker(QThread):
    """后台下载抠图权重：先写 .part，校验大小与 sha256 通过后才改名。

    下载与校验都在线程里做，界面只收进度；取消时删掉半截文件，目标路径上
    永远不会留下不完整的权重。
    """
    progress = Signal(int, int)      # (已下载字节, 总字节；未知时为 0)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, url, target, expected_size=0, expected_sha256="",
                 parent=None):
        super().__init__(parent)
        self._url = url
        self._target = target
        self._expected_size = int(expected_size or 0)
        self._expected_sha256 = (expected_sha256 or "").lower()
        self._cancelled = False

    def run(self):
        import hashlib
        import urllib.request

        part = self._target + ".part"
        try:
            folder = os.path.dirname(os.path.abspath(self._target))
            if folder:
                os.makedirs(folder, exist_ok=True)
            request = urllib.request.Request(
                self._url, headers={"User-Agent": "LabelForge"})
            with urllib.request.urlopen(request, timeout=30) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with open(part, "wb") as handle:
                    while True:
                        if self._cancelled:
                            raise RuntimeError(tr("已取消下载"))
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        self.progress.emit(done, total)

            size = os.path.getsize(part)
            if self._expected_size and size != self._expected_size:
                raise RuntimeError(
                    tr("文件大小不符：{size} 字节，应为 {want}").format(
                        size=size, want=self._expected_size))
            if self._expected_sha256:
                digest = hashlib.sha256()
                with open(part, "rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != self._expected_sha256:
                    raise RuntimeError(tr("校验值不符，下载可能被截断或损坏"))
            os.replace(part, self._target)
            self.finished_ok.emit(self._target)
        except Exception as exc:
            if os.path.exists(part):
                try:
                    os.remove(part)
                except OSError:
                    pass
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True


class MatteWorker(QThread):
    """后台单次抠图：裁块、推理、裁边、存盘。

    `done` 带回 `matting_service.MatteResult`，主线程据此建 Pixmap 追加到贴图
    列表（Qt 对象与 editor 状态只归主线程）。取消只在步骤之间生效：取消时
    既不写临时文件也不改名，所以不会留下半截 PNG。
    """
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, model, rgb, plan, threshold=0.0, margin_ratio=0.05,
                 parent=None):
        super().__init__(parent)
        self._model = model
        self._rgb = rgb
        self._plan = dict(plan)
        self._threshold = threshold
        self._margin_ratio = margin_ratio
        self._cancelled = False

    def run(self):
        from ..engine import matting_service
        try:
            result = matting_service.matte_selection(
                self._model, self._rgb,
                threshold=self._threshold,
                margin_ratio=self._margin_ratio,
                cancel=lambda: self._cancelled,
                **self._plan)
            if result is not None:
                self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True


def class_names_prompt(parent, model_path, suggested_path, expected_count=None):
    """类别缺失时收集类别名，并在模型同级目录写同名 txt。

    expected_count 是模型输出能推出来的类别数（拿不到就 None）。有它就能
    提示用户「需要几个」，填错数量时也能当场看到。
    返回写好的路径；用户取消则返回 None（调用方据此判定不可推理）。
    """
    from .dialog_helpers import get_multiline_text

    text, accepted = get_multiline_text(
        parent, tr("模型缺少类别信息"),
        tr("一行一个类别名；将保存到模型同级的同名 txt"),
        "", expected_lines=expected_count)
    if not accepted:
        return None
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if not names:
        return None
    if not inference_task.write_class_names(suggested_path, names):
        return None
    return suggested_path


def read_saved_class_names(path):
    """读同级同名 txt；没有或读不出来返回空表。"""
    if not path or not os.path.isfile(path):
        return []
    return read_class_names(path)


class VideoExtractWorker(QThread):
    """后台抽帧。progress: (done, total)；完成后带回 ExtractionResult。"""
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path, target_fps, start=None, end=None,
                 image_format="jpg", quality=95, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._target_fps = target_fps
        self._start = start
        self._end = end
        self._image_format = image_format
        self._quality = quality
        self._cancelled = False

    def run(self):
        from ..engine import video_io
        try:
            result = video_io.extract_frames(
                self._video_path, self._target_fps,
                start=self._start, end=self._end,
                image_format=self._image_format, quality=self._quality,
                cancel=lambda: self._cancelled,
                progress=self.progress.emit,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True


class VideoComposeWorker(QThread):
    """后台合成抽帧视频。progress: (done, total)；完成后带回 ComposeResult。

    有清单时按清单时长合成（不用帧率），没有清单才用 `output_fps`。
    `render` 给一个 `video_render.RenderSpec` 时在画面上叠加标签框；
    类别色必须由主线程预先取好放进 spec，后台不能去读 editor 的取色。
    """
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, output_path, manifest_path=None, images_dir=None,
                 output_fps=None, render=None, parent=None,
                 start_index=None, end_index=None):
        super().__init__(parent)
        self._output_path = output_path
        self._manifest_path = manifest_path
        self._images_dir = images_dir
        self._output_fps = output_fps
        self._render = render
        self._start_index = start_index
        self._end_index = end_index
        self._cancelled = False

    def run(self):
        from ..engine import video_io
        try:
            result = video_io.compose_video(
                manifest_path=self._manifest_path,
                images_dir=self._images_dir,
                output_path=self._output_path,
                output_fps=self._output_fps,
                render=self._render,
                cancel=lambda: self._cancelled,
                progress=self.progress.emit,
                start_index=self._start_index,
                end_index=self._end_index,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True


class VideoRenderWorker(QThread):
    """后台逐帧推理编码。progress: (done, total) 按帧。

    源视频的每一帧都跑一次推理并绘制，所以比 抽帧合成 慢得多：进度按帧上报、
    必须能取消。`render`/`scope`/`class_filter` 都由主线程定稿后传入——
    `get_label_color()` 会写缓存与配置，后台线程不能调用它。
    `tracking`/`dataset` 同样由主线程构造（写盘目录、会话状态都在主线程定稿）。
    """
    progress = Signal(int, int)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, video_path, detector, output_path=None, resize_fn=None,
                 scope=None, render=None, class_filter=None, parent=None,
                 tracking=None, dataset=None, min_hits=2,
                 class_names=None, snapshot=None):
        super().__init__(parent)
        self._video_path = video_path
        self._detector = detector
        self._output_path = output_path
        self._resize_fn = resize_fn
        self._scope = scope
        self._render = render
        self._class_filter = class_filter
        self._tracking = tracking
        self._dataset = dataset
        self._min_hits = min_hits
        self._class_names = class_names
        self._snapshot = snapshot
        self._cancelled = False

    def run(self):
        from ..engine import video_io
        from ..engine.inference_image import resize_for_model
        try:
            result = video_io.render_video(
                self._video_path, self._detector,
                output_path=self._output_path,
                resize_fn=self._resize_fn or resize_for_model,
                scope=self._scope,
                render=self._render,
                class_filter=self._class_filter,
                tracking=self._tracking,
                dataset=self._dataset,
                min_hits=self._min_hits,
                class_names=self._class_names,
                snapshot=self._snapshot,
                cancel=lambda: self._cancelled,
                progress=self.progress.emit,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancelled = True
