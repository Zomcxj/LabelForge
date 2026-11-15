"""抠图模型适配：预设规格、ONNX 会话、基准测试。

onnxruntime 与 opencv 都在用到时才导入：本模块本身不依赖它们即可导入与测试。
"""

import os
import time

import numpy as np

from ..core.config import MATTING_CONFIG


#: 预设规格。mean/std 为 0~255 像素归一化后的均值与标准差。
#: 只支持 RMBG-1.4 一套：多预设会让「权重路径 + 预设」两张表不一致就报错，
#: 而用户实际只需要一个能用的模型。
MATTING_PRESETS = {
    # BRIA RMBG-1.4：U2Net 系架构，CPU 上单张约 2 秒，权重为非商用许可
    "rmbg14": {
        "input_size": 1024,
        "mean": 0.5,
        "std": 1.0,
    },
}

#: 权重的官方下载地址与校验值（设置页「下载权重」用）。
#: 镜像地址，直连 huggingface.co 在部分网络下不通；文件与官方 onnx/model.onnx 一致。
WEIGHT_DOWNLOAD = {
    "rmbg14": {
        "url": ("https://hf-mirror.com/briaai/RMBG-1.4/resolve/main/"
                "onnx/model.onnx"),
        "size": 176153355,
        "sha256": ("8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f0153"
                   "5ba742e0f"),
        "license": "bria-rmbg-1.4（仅限非商业用途）",
    },
}

#: 权重的约定目录：仓库根的 model/（已在 .gitignore 内，不随源码分发）
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model")


def preset_weight_path(preset, model_dir=None):
    """预设权重在约定目录里的路径；只拼路径，不检查文件是否存在。"""
    return os.path.join(model_dir or MODEL_DIR, f"{preset}.onnx")


class MattingModel:
    """单张裁块的抠图推理。

    实例化只做参数校验，不读权重、不建 session；第一次推理才加载。
    `session` 可注入替身，用来在没有权重文件的环境里验证前后处理链路。
    """

    def __init__(self, model_path=None, preset=None, input_size=None,
                 threads=None, session=None):
        preset = MATTING_CONFIG.get('preset', 'rmbg14') if preset is None else preset
        if preset not in MATTING_PRESETS:
            raise ValueError(f"未知预设：{preset}")
        spec = dict(MATTING_PRESETS[preset])
        if input_size is None:
            input_size = MATTING_CONFIG.get('input_size', 0)
        size = int(input_size or spec["input_size"])
        if not size:
            raise ValueError(f"预设 {preset} 缺少输入尺寸")
        spec["input_size"] = size

        if model_path is None:
            model_path = MATTING_CONFIG.get('model_path', '')
        if session is None and (not model_path or not os.path.isfile(model_path)):
            raise FileNotFoundError(
                f"抠图模型不存在：{model_path or '（未配置）'}。"
                f"请在「设置 - 抠图」里点「下载权重」下到 "
                f"{preset_weight_path(preset)}，或自己指定已有的 {preset}.onnx。")

        self.spec = spec
        self.preset = preset
        self.model_path = model_path
        self._session = session
        if threads is None:
            threads = MATTING_CONFIG.get('threads', 0)
        self._threads = int(threads or 0)
        self.last_seconds = 0.0

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "未安装 onnxruntime，无法执行抠图推理。请安装 onnxruntime（CPU）后重试。") from exc
        options = ort.SessionOptions()
        options.log_severity_level = 3
        if self._threads > 0:
            options.intra_op_num_threads = self._threads
        try:
            self._session = ort.InferenceSession(
                self.model_path, sess_options=options,
                providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise RuntimeError(f"加载抠图模型失败：{self.model_path}\n{exc}") from exc
        return self._session

    def preprocess(self, rgb):
        """RGB uint8 -> NCHW float32。"""
        import cv2
        size = self.spec["input_size"]
        image = np.asarray(rgb)[..., :3]
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        data = image.astype(np.float32) / 255.0
        data = (data - self.spec["mean"]) / self.spec["std"]
        return np.expand_dims(np.transpose(data, (2, 0, 1)), axis=0)

    def predict(self, rgb):
        """RGB uint8 -> 原尺寸的 alpha（0~1 float32）。"""
        import cv2
        from .matting import alpha_from_output
        session = self._ensure_session()
        blob = self.preprocess(rgb)
        started = time.perf_counter()
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        self.last_seconds = time.perf_counter() - started
        raw = np.squeeze(outputs[0])
        alpha = alpha_from_output(raw)
        h, w = np.asarray(rgb).shape[:2]
        if alpha.shape != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return alpha.astype(np.float32)

    def benchmark(self, rgb, runs=3):
        """跑 runs 次取中位数耗时（秒）。"""
        timings = []
        for _ in range(max(1, int(runs))):
            started = time.perf_counter()
            self.predict(rgb)
            timings.append(time.perf_counter() - started)
        timings.sort()
        return timings[len(timings) // 2]
