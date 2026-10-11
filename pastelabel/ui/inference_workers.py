"""推理用的后台线程：模型加载与批量执行。

加载与推理都放到线程里，否则大模型会冻住界面。模型加载除了建会话，
还要解析类别名：同级同名 txt → 模型自带 names → 提示用户手输并写 txt。
"""
import os

from PyQt5.QtCore import QThread, pyqtSignal

from ..engine import inference_task
from ..engine.inference_image import qimage_to_rgb_array, resize_for_model
from ..engine.onnx_inference import read_class_names
from .i18n import t as tr


def load_rgb(path):
    """读图并转成模型输入用的 RGB 数组；读不出来返回 None。"""
    from PyQt5.QtGui import QImage
    image = QImage(path)
    if image.isNull():
        return None
    return qimage_to_rgb_array(image)


class ModelLoadWorker(QThread):
    """加载模型并解析类别名。

    need_class_names 表示模型同级没有同名 txt、模型自身也没有类别表，
    必须让用户提供；此时不加载模型，由界面收集后带着类别名重新发起。
    """
    loaded = pyqtSignal(object, object, object)     # detector, size, class_source
    need_class_names = pyqtSignal(str, str)         # model_path, 建议的 txt 路径
    failed = pyqtSignal(str)

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
                        inference_task.sidecar_class_names_path(self._model_path))
                    return
                detector.class_names = names
            else:
                source = "user"
            size = detector.model_input_size()
            self.loaded.emit(detector, size, source)
        except Exception as exc:
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """后台顺序推理。progress: (done, total, 文件名)。"""
    progress = pyqtSignal(int, int, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, keys, detector, parent=None):
        super().__init__(parent)
        self._paths = list(paths)
        self._keys = list(keys)
        self._detector = detector
        self._interrupted = False

    def run(self):
        try:
            results = inference_task.run_batch(
                self._paths, self._detector, load_rgb, resize_for_model,
                keys=self._keys,
                progress_fn=self.progress.emit,
                is_interrupted=lambda: self._interrupted,
            )
            self.finished_ok.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._interrupted = True


def class_names_prompt(parent, model_path, suggested_path):
    """类别缺失时收集类别名，并在模型同级目录写同名 txt。

    返回写好的路径；用户取消则返回 None（调用方据此判定不可推理）。
    """
    from .dialog_helpers import get_text

    text, accepted = get_text(
        parent, tr("模型缺少类别信息"),
        tr("一行一个类别名；将保存到模型同级的同名 txt"),
        "")
    if not accepted or not text.strip():
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
