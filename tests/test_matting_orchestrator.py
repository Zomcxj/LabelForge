"""抠图编排：裁块、推理、裁边、命名、存盘、取消。

本工作树没有抠图权重，所有用例都往 `MattingModel(session=...)` 注入会话替身，
因此不需要下载模型，也不会跳过编排逻辑本身。
"""
import os
import types

import cv2
import numpy as np
import pytest

from labelforge.engine import matting_service
from labelforge.engine.matting_model import MattingModel
from labelforge.engine.matting_service import (
    MatteResult, default_output_dir, expand_box, matte_crop, matte_selection,
    next_matte_path, save_matte_png,
)


class _SolidSession:
    """会话替身：整块都是前景（+10）或整块都不是（-10），输出 logits。"""

    def __init__(self, foreground=True):
        self.foreground = foreground
        self.calls = 0

    def get_inputs(self):
        return [types.SimpleNamespace(name="image")]

    def run(self, output_names, feeds):
        blob = feeds["image"]
        _, _, height, width = blob.shape
        self.calls += 1
        value = 10.0 if self.foreground else -10.0
        return [np.full((1, 1, height, width), value, dtype=np.float32)]


class _CentreBlockSession:
    """会话替身：中间四分之一是前景，其余是背景（输出 logits）。"""

    def get_inputs(self):
        return [types.SimpleNamespace(name="image")]

    def run(self, output_names, feeds):
        blob = feeds["image"]
        _, _, height, width = blob.shape
        out = np.full((1, 1, height, width), -10.0, dtype=np.float32)
        out[:, :, height // 4: 3 * height // 4,
            width // 4: 3 * width // 4] = 10.0
        return [out]


def _model(session, input_size=64):
    """注入替身的模型；不读权重、不建 onnxruntime 会话。"""
    return MattingModel(preset="rmbg14", input_size=input_size, session=session)


def _decode(path):
    """按 Unicode 路径读回 PNG（IMREAD_UNCHANGED，通道顺序为 BGRA）。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def test_matte_selection_saves_a_png_matching_the_result(tmp_path):
    rgb = np.full((80, 80, 3), 120, dtype=np.uint8)
    model = _model(_SolidSession())

    result = matte_selection(model, rgb, (20, 20, 40, 40), "car",
                             "frame_001.png", str(tmp_path))

    expanded = expand_box((20, 20, 40, 40), (80, 80), 0.05)
    assert isinstance(result, MatteResult)
    # 外扩只作用于模型输入：落盘的外接框仍是被抠的那个框
    assert result.rect == (20, 20, 40, 40)
    assert expanded == (18, 18, 44, 44)   # 模型实际看到的是外扩后的裁块
    assert result.rgba.shape == (40, 40, 4)
    assert os.path.basename(result.path) == "frame_001_car_001.png"
    assert os.path.isfile(result.path)

    decoded = _decode(result.path)
    assert decoded.shape == result.rgba.shape
    assert decoded[:, :, 3].min() >= 250            # 全前景：alpha 接近不透明


def test_matte_selection_never_overwrites_existing_files(tmp_path):
    rgb = np.full((80, 80, 3), 90, dtype=np.uint8)
    model = _model(_SolidSession())

    first = matte_selection(model, rgb, (10, 10, 30, 30), "car",
                            "frame_001.png", str(tmp_path))
    first_bytes = open(first.path, "rb").read()
    second = matte_selection(model, rgb, (10, 10, 30, 30), "car",
                             "frame_001.png", str(tmp_path))

    assert os.path.basename(first.path) == "frame_001_car_001.png"
    assert os.path.basename(second.path) == "frame_001_car_002.png"
    assert open(first.path, "rb").read() == first_bytes


def test_matte_selection_trims_to_the_alpha_bbox(tmp_path):
    rgb = np.full((200, 300, 3), 100, dtype=np.uint8)
    model = _model(_CentreBlockSession())
    box = (60, 40, 80, 80)

    result = matte_selection(model, rgb, box, "car", "frame_001.png",
                             str(tmp_path), threshold=0.5)

    expanded = expand_box(box, (200, 300), 0.05)
    assert expanded == (56, 36, 88, 88)
    height, width = result.rgba.shape[:2]
    # 只留中间方块（+1px 边），比外扩后的裁块小
    assert 42 <= width <= 48 and 42 <= height <= 48
    assert width < expanded[2] and height < expanded[3]
    # rect 是这块在整图里的位置
    assert abs(result.rect[0] - (expanded[0] + 22)) <= 3
    assert abs(result.rect[1] - (expanded[1] + 22)) <= 3
    assert result.rect[2] == width and result.rect[3] == height
    assert _decode(result.path).shape == result.rgba.shape


def test_matte_selection_returns_none_when_nothing_is_found(tmp_path):
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    model = _model(_SolidSession(foreground=False))

    result = matte_selection(model, rgb, (10, 10, 20, 20), "car",
                             "frame_001.png", str(tmp_path), threshold=0.5)

    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_cancel_before_the_first_step_writes_nothing(tmp_path):
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    model = _model(_SolidSession())

    result = matte_selection(model, rgb, (10, 10, 20, 20), "car",
                             "frame_001.png", str(tmp_path),
                             cancel=lambda: True)

    assert result is None
    assert model._session.calls == 0
    assert list(tmp_path.iterdir()) == []


def test_cancel_during_inference_writes_nothing(tmp_path):
    state = {"cancel": False}

    class CancelDuringRun(_SolidSession):
        def run(self, output_names, feeds):
            state["cancel"] = True          # 模拟推理途中用户点了取消
            return super().run(output_names, feeds)

    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    model = _model(CancelDuringRun())

    result = matte_selection(model, rgb, (10, 10, 20, 20), "car",
                             "frame_001.png", str(tmp_path),
                             cancel=lambda: state["cancel"])

    assert model._session.calls == 1
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_polygon_region_intersects_the_model_output(tmp_path):
    rgb = np.full((400, 400, 3), 100, dtype=np.uint8)
    model = _model(_SolidSession())
    # 覆盖扩边后裁块左半边的多边形：区域之外必须透明
    polygon = [(40, 0), (150, 0), (150, 400), (40, 400)]

    result = matte_selection(model, rgb, (100, 100, 100, 100), "car",
                             "frame_001.png", str(tmp_path),
                             polygon=polygon)

    assert result is not None
    height, width = result.rgba.shape[:2]
    assert width < 70                          # 只剩左半边
    assert result.rgba[:, -1, 3].max() == 0     # 最右一列已透明
    assert result.rgba[:, 0, 3].max() >= 250    # 最左一列仍是前景


def test_empty_patch_is_reported_as_nothing(tmp_path):
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    rgba, rect = matte_crop(_model(_SolidSession()), rgb, (500, 500, 10, 10))
    assert (rgba, rect) == (None, None)


def test_default_output_dir_of_an_empty_base_is_the_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = default_output_dir("")
    assert os.path.dirname(str(out)) == str(tmp_path.parent)
    assert out.name == tmp_path.name + "_matting_output"


def test_next_matte_path_creates_the_default_output_dir(tmp_path):
    directory = default_output_dir(str(tmp_path / "images"))

    path = next_matte_path(directory, "car", "frame_001.png")

    assert os.path.isdir(str(directory))
    assert path.name == "frame_001_car_001.png"


def test_save_matte_png_writes_through_a_temp_file(tmp_path, monkeypatch):
    calls = []
    real_replace = os.replace

    def recording_replace(src, dst):
        calls.append((src, dst, os.path.exists(src)))
        real_replace(src, dst)

    monkeypatch.setattr(matting_service.os, "replace", recording_replace)
    path = next_matte_path(tmp_path, "car", "frame_001.png")
    rgba = np.full((4, 4, 4), 200, dtype=np.uint8)

    save_matte_png(rgba, path)

    assert len(calls) == 1
    temp_path, final_path, existed = calls[0]
    assert temp_path.endswith(matting_service.TEMP_SUFFIX)
    assert final_path == str(path)
    assert existed is True                      # 临时文件先落好再改名
    assert not os.path.exists(temp_path)
    assert os.path.isfile(str(path))


def test_save_matte_png_leaves_nothing_when_replace_fails(tmp_path, monkeypatch):
    path = next_matte_path(tmp_path, "car", "frame_001.png")

    def failing_replace(src, dst):
        raise OSError("磁盘已满")

    monkeypatch.setattr(matting_service.os, "replace", failing_replace)
    with pytest.raises(OSError):
        save_matte_png(np.zeros((4, 4, 4), dtype=np.uint8), path)

    assert not os.path.exists(str(path))
    assert not [name for name in os.listdir(tmp_path)
                if name.endswith(matting_service.TEMP_SUFFIX)]


def test_save_matte_png_keeps_rgba_pixels(tmp_path):
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = (200, 100, 50, 255)           # RGB + A
    path = next_matte_path(tmp_path, "car", "frame_001.png")

    save_matte_png(rgba, path)

    decoded = _decode(path)                    # 解出来是 BGRA
    assert list(decoded[0, 0]) == [50, 100, 200, 255]
    assert decoded[1, 1, 3] == 0


def test_save_matte_png_rejects_arrays_without_alpha(tmp_path):
    path = next_matte_path(tmp_path, "car", "frame_001.png")
    with pytest.raises(ValueError):
        save_matte_png(np.zeros((4, 4, 3), dtype=np.uint8), path)


def test_matte_never_exceeds_the_requested_box(tmp_path):
    """外扩只给模型看上下文：落盘像素必须留在调用方给的框内。"""
    rgb = np.full((400, 400, 3), 100, dtype=np.uint8)
    box = (100, 100, 60, 60)

    result = matte_selection(_model(_SolidSession()), rgb, box, "car",
                             "frame_001.png", str(tmp_path))

    x, y, w, h = result.rect
    assert x >= box[0] and y >= box[1]
    assert x + w <= box[0] + box[2]
    assert y + h <= box[1] + box[3]
    # 整块都是前景：结果正好等于原框，不会多出外扩的那一圈
    assert (w, h) == (box[2], box[3])
    assert result.rgba.shape[:2] == (box[3], box[2])


def test_box_mask_zeroes_the_margin_ring():
    """扩边那一圈必须清零：它只是模型输入的上下文。"""
    from labelforge.engine.matting_service import box_mask, expand_box

    shape = (40, 40)
    box = (10, 10, 20, 20)
    patch = expand_box(box, shape, 0.05)
    mask = box_mask(shape, box, (patch[0], patch[1]))

    assert mask[0].max() == 0            # 扩边的最外一行已清零
    assert mask[-1].max() == 0
    assert mask[:, 0].max() == 0
    assert mask[:, -1].max() == 0
    # 原框范围内的像素仍是前景
    assert mask[5, 5] == 1.0
