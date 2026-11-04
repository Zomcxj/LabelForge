# -*- coding: utf-8 -*-
"""回归用例：真实 yolov5s 导出（[1, N, 5+C]，行内 cx,cy,w,h,obj,cls...）。

与 `test_inference_e2e.py` 一个路子：找不到模型就跳过。真正防的是「通道数
刚好等于 4+C+1」时把检测头当成 OBB 解——那会让每张图吐出一堆角度乱码的
旋转框。

模型与张量缓存路径可用环境变量覆盖：
  LF_YOLOV5_ONNX   .onnx 路径
  LF_YOLOV5_RAW    .npy 原始输出缓存（省一次推理）
"""
import os

import numpy as np
import pytest

MODEL = os.environ.get("LF_YOLOV5_ONNX",
                       r"F:\Study\AutoLabel\X-AnyLabeling-yaml\yolov5s.onnx")
RAW = os.environ.get(
    "LF_YOLOV5_RAW",
    r"F:\VibeCoding_workspace\LabelForge\.superpowers\inference-tasks\samples\v5raw.npy")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(MODEL) or os.path.isfile(RAW)),
    reason="没有 yolov5s 导出样本，跳过真实模型回归")


def test_real_yolov5_export_is_detection_not_obb():
    from labelforge.engine.onnx_inference import OnnxDetector, metadata_class_names

    if os.path.isfile(MODEL):
        detector = OnnxDetector(MODEL)
        detector.session
    else:
        pytest.skip("缺少模型文件，只验证已缓存的输出形状")

    names = metadata_class_names(detector.session)
    assert len(names) == 80, f"yolov5s 应为 80 类，实际 {len(names)}"
    # 85 列 == 4+80+1，与「80 类 OBB」同形；必须按检测解
    assert detector.resolve_task() == "det"
    assert detector.resolve_format() == "v5"


def test_real_yolov5_output_decodes_in_v5_layout():
    if not os.path.isfile(RAW):
        pytest.skip("缺少原始输出缓存")
    from labelforge.engine.onnx_inference import decode_yolov5

    raw = np.load(RAW)
    assert raw.shape == (25200, 85)
    decoded = decode_yolov5(raw[None, ...], 0.25, num_classes=80)
    assert decoded["boxes"].shape[1] == 4
    assert decoded["scores"].max() > 0.5, "这张图至少应有一个可信目标"
    assert decoded["class_ids"].max() < 80
    # 中心宽高必须在模型输入范围内；解错布局时坐标会是几十万的量级
    assert decoded["boxes"].max() < 640 * 4
