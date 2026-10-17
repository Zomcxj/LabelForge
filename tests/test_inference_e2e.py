"""端到端推理测试：用真实 onnxruntime 跑现场生成的 YOLOv5/v8 模型。

模型输出是常量张量，期望框可以精确算出来；覆盖 mock 测不到的部分：
真实 Session、张量喂入、元数据读取，以及「模糊形状必须拒绝而不是猜」。

进程内跑（conftest 把 PySide6 换成了 mock），所以注入 numpy 版 letterbox；
生产用的 QImage 版绘制由 test_inference_image.py 在真实 Qt 子进程里验证。
"""
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")   # 真实权重仍需人工对照，这里只保证链路是通的

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from labelforge.engine.inference_task import model_class_names  # noqa: E402
from labelforge.engine.onnx_inference import (  # noqa: E402
    InferenceError, OnnxDetector,
)

SIZE = 64
NAMES = ["cat", "dog", "bird"]


def _letterbox(rgb, geometry, fill):
    """numpy 版 letterbox，契约与 inference_image.letterbox_to_array 一致。"""
    canvas = np.full((geometry.dst_h, geometry.dst_w, 3), int(fill), dtype=np.uint8)
    src_h, src_w = rgb.shape[0], rgb.shape[1]
    if (src_w, src_h) == (geometry.new_w, geometry.new_h):
        scaled = rgb
    else:
        ys = np.minimum((np.arange(geometry.new_h) / geometry.new_h * src_h).astype(int),
                        src_h - 1)
        xs = np.minimum((np.arange(geometry.new_w) / geometry.new_w * src_w).astype(int),
                        src_w - 1)
        scaled = rgb[ys][:, xs]
    canvas[geometry.pad_y:geometry.pad_y + geometry.new_h,
           geometry.pad_x:geometry.pad_x + geometry.new_w] = scaled
    return canvas


def _save(model, path):
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return str(path)


def _v5_model(path, rows, num_classes=3):
    """[1, N, 5+C]，行内 cx, cy, w, h, obj, cls..."""
    flat = [list(row[:5]) + list(row[5]) for row in rows]
    data = np.array([flat], dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"],
                          value=numpy_helper.from_array(data, "pred"))],
        "v5",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, len(rows), 5 + num_classes])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    return _save(model, path)


def _v8_model(path, cols, num_classes=3):
    """[1, 4+C, N]，列为 cx, cy, w, h, cls..."""
    flat = [list(col[:4]) + list(col[4]) for col in cols]
    data = np.array([flat], dtype=np.float32).transpose(0, 2, 1)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"],
                          value=numpy_helper.from_array(data, "pred"))],
        "v8",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, 4 + num_classes, len(cols)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    return _save(model, path)


def _with_names(path, names):
    """给模型写入 ultralytics 风格的 names 元数据。"""
    model = onnx.load(path)
    entry = model.metadata_props.add()
    entry.key = "names"
    entry.value = "{" + ", ".join(f"{i}: '{n}'" for i, n in enumerate(names)) + "}"
    onnx.save(model, path)
    return path


def _blank(w=SIZE, h=SIZE):
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------- YOLOv5 ----------

def test_v5_real_session_decodes_obj_times_class(tmp_path):
    """v5 置信度必须是 objectness × 类别分数，且 xywh 转成 xyxy。"""
    path = _v5_model(tmp_path / "v5.onnx", [
        # obj=0.9, cls0=0.9 → 0.81；中心 (32,32) 尺寸 20x10
        (32.0, 32.0, 20.0, 10.0, 0.9, [0.9, 0.05, 0.05]),
        # obj=0.1 → 0.009，低于默认阈值
        (10.0, 10.0, 6.0, 6.0, 0.1, [0.05, 0.1, 0.05]),
        # 与第一条同类高度重合 → 被 NMS 去掉
        (33.0, 32.0, 20.0, 10.0, 0.8, [0.8, 0.05, 0.05]),
    ])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v5",
                            conf_threshold=0.25)

    boxes = detector.detect(_blank(), _letterbox)

    assert len(boxes) == 1, boxes
    box = boxes[0]
    assert box["label"] == "cat"
    assert (box["x"], box["y"]) == (22.0, 27.0)
    assert (box["width"], box["height"]) == (20.0, 10.0)
    assert box["score"] == pytest.approx(0.81, abs=1e-5)


def test_v5_real_session_resolves_size_and_format(tmp_path):
    path = _v5_model(tmp_path / "v5.onnx", [
        (32.0, 32.0, 20.0, 10.0, 0.9, [0.9, 0.05, 0.05])])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v5")

    assert detector.model_input_size() == (SIZE, SIZE)
    assert detector.resolve_format() == "v5"
    assert detector.active_providers, "真实 Session 必须报出实际 provider"


# ---------- YOLOv8 ----------

def test_v8_real_session_keeps_overlapping_different_classes(tmp_path):
    """不同类别的框完全重合也要都保留（纯几何去重会静默删掉一个）。"""
    path = _v8_model(tmp_path / "v8.onnx", [
        (20.0, 20.0, 10.0, 10.0, [0.85, 0.05, 0.05]),
        (20.0, 20.0, 10.0, 10.0, [0.05, 0.75, 0.05]),
    ])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v8",
                            conf_threshold=0.25)

    boxes = detector.detect(_blank(), _letterbox)

    assert {b["label"] for b in boxes} == {"cat", "dog"}
    assert all((b["x"], b["y"]) == (15.0, 15.0) for b in boxes)


def test_v8_real_session_high_threshold_yields_empty_not_error(tmp_path):
    path = _v8_model(tmp_path / "v8.onnx", [
        (20.0, 20.0, 10.0, 10.0, [0.85, 0.05, 0.05])])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v8",
                            conf_threshold=0.99)

    assert detector.detect(_blank(), _letterbox) == []


def test_v8_real_session_reads_names_from_metadata(tmp_path):
    path = _with_names(_v8_model(tmp_path / "v8.onnx", [
        (20.0, 20.0, 10.0, 10.0, [0.85, 0.05, 0.05])]), ["c0", "c1", "c2"])
    detector = OnnxDetector(path, output_format="v8")

    assert model_class_names(detector.session) == ["c0", "c1", "c2"]


# ---------- 格式识别必须拒绝而不是猜 ----------

def test_ambiguous_output_shape_is_rejected_not_guessed(tmp_path):
    """真实回归：没有类别表时 v5 模型 [1, 3, 8] 曾被猜成 v8，静默产出错误标注。

    猜成 v8 会把 objectness 当成类别分数（0.9 而非 obj×cls=0.81），
    并把列当行读，类别也跟着错位。有类别表时 5+3=8 可判定为 v5，
    所以只在缺类别表时要求拒绝。
    """
    path = _v5_model(tmp_path / "v5.onnx", [
        (32.0, 32.0, 20.0, 10.0, 0.9, [0.9, 0.05, 0.05]),
        (20.0, 20.0, 8.0, 8.0, 0.7, [0.05, 0.9, 0.05]),
        (12.0, 12.0, 6.0, 6.0, 0.6, [0.05, 0.05, 0.9]),
    ])
    detector = OnnxDetector(path)   # 无类别文件、无元数据

    with pytest.raises(InferenceError) as exc:
        detector.detect(_blank(), _letterbox)

    assert "请显式选择" in str(exc.value)


def test_class_table_disambiguates_small_v5_output(tmp_path):
    """有类别表时 5+C 与形状吻合即可判定为 v5，不该再要求显式指定。"""
    path = _v5_model(tmp_path / "v5.onnx", [
        (32.0, 32.0, 20.0, 10.0, 0.9, [0.9, 0.05, 0.05]),
        (20.0, 20.0, 8.0, 8.0, 0.7, [0.05, 0.9, 0.05]),
        (12.0, 12.0, 6.0, 6.0, 0.6, [0.05, 0.05, 0.9]),
    ])
    detector = OnnxDetector(path, class_names=NAMES, conf_threshold=0.25)

    boxes = detector.detect(_blank(), _letterbox)

    assert {b["label"] for b in boxes} == {"cat", "dog", "bird"}
    assert [b["score"] for b in boxes] == pytest.approx([0.81, 0.63, 0.54], abs=1e-5)


def test_ambiguous_multiple_detection_heads_are_rejected(tmp_path):
    """两个都是 3 维输出时无法确定哪个是检测头，必须拒绝而不是瞎挑。"""
    const = numpy_helper.from_array(np.zeros((1, 3, 8), dtype=np.float32), "pred")
    other = numpy_helper.from_array(np.zeros((1, 25200, 9), dtype=np.float32), "pred2")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=const),
         helper.make_node("Constant", [], ["output2"], value=other)],
        "two",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8]),
         helper.make_tensor_value_info("output2", TensorProto.FLOAT, [1, 25200, 9])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    path = _save(model, tmp_path / "two.onnx")

    with pytest.raises(InferenceError) as exc:
        OnnxDetector(path, class_names=NAMES).detect(_blank(), _letterbox)

    assert "无法确定哪个是检测头" in str(exc.value)


def test_model_without_detection_head_is_rejected(tmp_path):
    """一个 3 维输出都没有（如分割/姿态导出），拒绝并说明原因。"""
    const = numpy_helper.from_array(np.zeros((1, 3, 8, 8), dtype=np.float32), "pred")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=const)],
        "seg",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    path = _save(model, tmp_path / "seg.onnx")

    with pytest.raises(InferenceError) as exc:
        OnnxDetector(path, class_names=NAMES).detect(_blank(), _letterbox)

    assert "没有 3 维检测输出" in str(exc.value)



def test_missing_model_file_raises_inference_error(tmp_path):
    detector = OnnxDetector(str(tmp_path / "nope.onnx"), class_names=NAMES,
                            output_format="v8")
    with pytest.raises(InferenceError) as exc:
        detector.detect(_blank(), _letterbox)
    assert "加载模型失败" in str(exc.value)


# ---------- letterbox 逆变换 ----------

def test_wide_image_boxes_map_back_to_original_coordinates(tmp_path):
    """4:1 宽图：scale=0.25、pad_y=24，模型框必须还原并钳到原图内。"""
    path = _v8_model(tmp_path / "v8.onnx", [
        (20.0, 20.0, 10.0, 10.0, [0.85, 0.05, 0.05])])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v8",
                            conf_threshold=0.25)

    boxes = detector.detect(_blank(w=256, h=64), _letterbox)

    assert len(boxes) == 1, boxes
    box = boxes[0]
    # 模型 x 15→25 除以 0.25 得 60→100
    assert box["x"] == pytest.approx(60.0, abs=0.02)
    assert box["width"] == pytest.approx(40.0, abs=0.02)
    # 模型 y 15→25 落在补边区内，钳到图内后只剩顶部 4px
    assert box["y"] == 0.0
    assert box["height"] == pytest.approx(4.0, abs=0.02)


def test_tall_image_boxes_stay_inside_bounds(tmp_path):
    path = _v8_model(tmp_path / "v8.onnx", [
        (20.0, 20.0, 10.0, 10.0, [0.85, 0.05, 0.05])])
    detector = OnnxDetector(path, class_names=NAMES, output_format="v8",
                            conf_threshold=0.25)

    boxes = detector.detect(_blank(w=64, h=256), _letterbox)

    for box in boxes:
        assert 0 <= box["x"] and box["x"] + box["width"] <= 64
        assert 0 <= box["y"] and box["y"] + box["height"] <= 256


def test_box_fully_inside_padding_is_dropped(tmp_path):
    """整体落在补边区里的框必须丢弃，而不是变成贴边的零面积框。"""
    path = _v8_model(tmp_path / "v8.onnx", [
        (32.0, 4.0, 6.0, 6.0, [0.85, 0.05, 0.05])])   # 模型顶部 = 补边区
    detector = OnnxDetector(path, class_names=NAMES, output_format="v8",
                            conf_threshold=0.25)

    boxes = detector.detect(_blank(w=256, h=64), _letterbox)

    assert boxes == [], boxes


def test_dynamic_input_size_defaults_and_is_overridable(tmp_path):
    """动态轴模型给默认 640×640，并允许调用方覆盖（模型行里可改）。"""
    const = numpy_helper.from_array(np.zeros((1, 3, 8), dtype=np.float32), "pred")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=const)],
        "dyn",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    path = _save(model, tmp_path / "dyn.onnx")

    default = OnnxDetector(path, class_names=NAMES, output_format="v5")
    assert default.is_dynamic_input() is True
    assert default.model_input_size() == (640, 640)

    overridden = OnnxDetector(path, class_names=NAMES, output_format="v5",
                              input_size=(128, 96))
    assert overridden.model_input_size() == (128, 96)
    # 改尺寸不必重建会话，直接写回即可
    overridden.set_input_size((320, 320))
    assert overridden.model_input_size() == (320, 320)


def test_static_input_size_is_not_dynamic(tmp_path):
    const = numpy_helper.from_array(np.zeros((1, 3, 8), dtype=np.float32), "pred")
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"], value=const)],
        "static",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 48])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    path = _save(model, tmp_path / "static.onnx")

    detector = OnnxDetector(path, class_names=NAMES, output_format="v5")
    assert detector.is_dynamic_input() is False
    # 静态轴以模型自身为准，用户填的尺寸不生效
    assert detector.model_input_size() == (48, 64)


# ---------- 真实模型（可选，仅在 model/ 存在时运行）----------

REPO_ROOT = Path(__file__).resolve().parents[1]
#: 每个格式给多个候选：n/s 只是权重规模不同，输出布局与类别表一致，
#: 测试要的是「这一代导出能被自动识别并跑通」，不绑定具体某个权重文件。
REAL_MODELS = {
    "v5": ("yolov5n.onnx", "yolov5s.onnx"),
    "v8": ("yolov8n.onnx", "yolov8s.onnx", "yolo11s.onnx"),
}
REAL_IMAGES = sorted((REPO_ROOT / "model").glob("*.jpg"))


def _real_model_path(fmt):
    """返回该格式实际可用的模型路径；一个都没有时返回 None。"""
    for name in REAL_MODELS[fmt]:
        path = REPO_ROOT / "model" / name
        if path.exists():
            return path
    return None


def _require_real_model(fmt):
    path = _real_model_path(fmt)
    if path is None:
        candidates = " / ".join(f"model/{n}" for n in REAL_MODELS[fmt])
        pytest.skip(f"缺少真实模型 {candidates}（不入库，需本地提供）")
    return str(path)


@pytest.mark.parametrize("fmt", ["v5", "v8"])
def test_real_model_matches_expected_output_shape(fmt):
    """真实权重必须是已知布局：v5 [1, N, 5+C]、v8 [1, 4+C, N]，且带 COCO 类别表。

    这条测试的价值在于锁住「自动识别在这个模型上真的能判定」——
    如果换了导出方式（例如内置 NMS 的双输出），这里会先失败。
    """
    detector = OnnxDetector(_require_real_model(fmt))
    shape = list(detector.session.get_outputs()[0].shape)

    if fmt == "v5":
        assert len(shape) == 3 and shape[2] == 85, shape
        assert shape[1] > 1000, shape
    else:
        assert len(shape) == 3 and shape[1] == 84, shape
        assert shape[2] > 1000, shape

    names = model_class_names(detector.session)
    assert len(names) == 80, len(names)
    assert names[0] == "person" and names[2] == "car"
    assert detector.model_input_size() == (640, 640)
    # 自动识别必须给出正确结论，而不是要求用户显式指定
    assert detector.resolve_format() == fmt


@pytest.mark.parametrize("fmt", ["v5", "v8"])
def test_real_model_detects_on_real_image(fmt):
    """真实图片跑通全链路：框在图内、非退化、分数与标签有效。

    走真实 Qt 子进程：本文件进程内 PySide6 是 mock，QImage 读不了图。

    模型与图片都在测试体内检查：两者都是本地可选素材，缺任意一个都该跳过，
    只检查图片会让缺模型时变成失败（守卫与断言说的不是一回事）。
    """
    model_path = _require_real_model(fmt)
    if not REAL_IMAGES:
        pytest.skip("model/ 下没有测试图片")
    from tests.test_inference_image import _run_real_qt

    out = _run_real_qt(f'''
import sys
sys.path.insert(0, r"{REPO_ROOT}")
from PySide6.QtGui import QImage
from labelforge.engine.inference_image import qimage_to_rgb_array, resize_for_model
from labelforge.engine.inference_task import build_detector

# 走生产装配路径：类别表来自模型元数据，标签才不会是空串
det = build_detector({{"model_path": r"{model_path}", "conf_threshold": 0.35}})
img = QImage(r"{REAL_IMAGES[0]}")
assert not img.isNull(), "测试图片读不出来"
rgb = qimage_to_rgb_array(img)
h, w = rgb.shape[:2]
boxes = det.detect(rgb, resize_for_model)
assert boxes, "真实图片上至少应检出若干目标"
for b in boxes:
    assert 0 <= b["x"] and b["x"] + b["width"] <= w, b
    assert 0 <= b["y"] and b["y"] + b["height"] <= h, b
    assert b["width"] > 0 and b["height"] > 0, b
    assert 0.35 <= b["score"] <= 1.0, b
    assert b["label"], b
print("REAL_OK", len(boxes), sorted({{b["label"] for b in boxes}})[:5])
''')
    assert "REAL_OK" in out


def _v5_model_with_aux_heads(path, rows, num_classes=3):
    """v5 主检测头 + 每 stride 的原始辅助头（第三方训练导出的常见形态）。

    辅助头是 5 维 [1, 每格锚数, H, W, 5+C]，值还没解码，不参与推理。
    """
    flat = [list(row[:5]) + list(row[5]) for row in rows]
    data = np.array([flat], dtype=np.float32)
    aux = np.zeros((1, 3, 8, 8, 5 + num_classes), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"],
                          value=numpy_helper.from_array(data, "pred")),
         helper.make_node("Constant", [], ["346"],
                          value=numpy_helper.from_array(aux, "aux"))],
        "v5aux",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT,
                                       [1, len(rows), 5 + num_classes]),
         helper.make_tensor_value_info("346", TensorProto.FLOAT,
                                       [1, 3, 8, 8, 5 + num_classes])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    return _save(model, path)


def test_auxiliary_training_heads_are_ignored(tmp_path):
    """多挂训练辅助头的第三方模型要照常推理，不能被「仅支持单输出」拒掉。

    这是 nonmotor.onnx 那个崩溃的回归护栏：主检测头是唯一的 3 维输出，
    5 维的辅助头只是同一次导出的副产品。
    """
    path = _v5_model_with_aux_heads(tmp_path / "v5aux.onnx", [
        (32.0, 32.0, 20.0, 10.0, 0.9, [0.9, 0.05, 0.05]),
        (20.0, 20.0, 8.0, 8.0, 0.7, [0.05, 0.9, 0.05]),
    ])
    detector = OnnxDetector(path, class_names=NAMES, conf_threshold=0.25)

    boxes = detector.detect(_blank(), _letterbox)

    assert {b["label"] for b in boxes} == {"cat", "dog"}
    assert [b["score"] for b in boxes] == pytest.approx([0.81, 0.63], abs=1e-5)


def test_detection_head_index_is_not_zero_when_aux_comes_first(tmp_path):
    """检测头在输出列表里排第二时也要挑对，不能死认下标 0。"""
    # 行数要够多，否则 [1, N, 8] 太小、格式本身判不出来（那是对的）
    rows = [list((32.0, 32.0, 20.0, 10.0, 0.9)) + [0.9, 0.05, 0.05]]
    rows += [list((0.0, 0.0, 1.0, 1.0, 0.0)) + [0.0, 0.0, 0.0]] * 255
    data = np.array([rows], dtype=np.float32)
    aux = np.zeros((1, 3, 8, 8, 8), dtype=np.float32)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["346"],
                          value=numpy_helper.from_array(aux, "aux")),
         helper.make_node("Constant", [], ["output"],
                          value=numpy_helper.from_array(data, "pred"))],
        "auxfirst",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("346", TensorProto.FLOAT, [1, 3, 8, 8, 8]),
         helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256, 8])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    path = _save(model, tmp_path / "auxfirst.onnx")

    detector = OnnxDetector(path, class_names=NAMES, conf_threshold=0.25)
    boxes = detector.detect(_blank(), _letterbox)

    assert [b["label"] for b in boxes] == ["cat"]
    assert detector._output_index == 1


def test_pick_detection_output_reports_shapes_in_error():
    """挑不出来时错误里要带上实际形状，方便用户自己看模型导出了什么。"""
    from labelforge.engine.onnx_inference import pick_detection_output

    class Out:
        def __init__(self, shape):
            self.shape = shape

    # 一个 3 维都没有
    with pytest.raises(InferenceError) as exc:
        pick_detection_output([Out([1, 3, 8, 8]), Out([1, 2, 3, 3])])
    text = str(exc.value)
    assert "没有 3 维检测输出" in text
    assert "(1, 3, 8, 8)" in text

    # 有两个 3 维，说不清哪个是检测头
    with pytest.raises(InferenceError) as exc2:
        pick_detection_output([Out([1, 300, 6]), Out([1, 25200, 9])])
    text2 = str(exc2.value)
    assert "2 个 3 维输出" in text2
    assert "(1, 300, 6)" in text2 and "(1, 25200, 9)" in text2
