"""动态轴导出的 ONNX（维度是符号名）不能把加载流程打崩。

用户把抠图权重 `rmbg14.onnx` 误选进「加载模型」时，输出维度写的是
`batch_size` 一类的名字，`int()` 直接抛 ValueError，异常一路冒到主线程。
这里钉住：非整型维度按「不是检测头」跳过，不是错误。
"""
import onnxruntime as ort
import pytest


class _Output:
    """模拟 onnxruntime 的 NodeArg：只用到 shape。"""

    def __init__(self, shape):
        self.shape = shape


def test_static_shape_returns_none_for_symbolic_dims():
    from labelforge.engine.onnx_tasks import static_shape

    assert static_shape(_Output([1, 1, 1024, 1024])) == (1, 1, 1024, 1024)
    assert static_shape(_Output(["batch_size", 1, 1024, 1024])) is None
    assert static_shape(_Output([1, 1, None, 1024])) is None


def test_proto_index_skips_symbolic_outputs():
    from labelforge.engine.onnx_tasks import proto_index

    outputs = [_Output(["batch_size", 1, 1024, 1024])]

    assert proto_index(outputs) is None, "符号维度不能当成掩膜原型"


def test_detect_task_from_outputs_survives_symbolic_dims():
    from labelforge.engine.onnx_tasks import detect_task_from_outputs

    outputs = [_Output(["batch_size", 1, 1024, 1024])]

    assert detect_task_from_outputs(outputs) is None


def test_pick_detection_output_reports_a_non_detection_model():
    """4 维输出的模型（抠图 / 分割原型）要给出可读的错误，而不是 ValueError。"""
    from labelforge.engine.onnx_inference import InferenceError, pick_detection_output

    outputs = [_Output(["batch_size", 1, 1024, 1024])]

    with pytest.raises(InferenceError) as excinfo:
        pick_detection_output(outputs)
    assert "没有 3 维检测输出" in str(excinfo.value)


def test_pick_detection_output_skips_symbolic_three_dim_outputs():
    """3 维但维度是符号名：跳过它，不抛 ValueError。"""
    from labelforge.engine.onnx_inference import pick_detection_output

    outputs = [_Output(["batch_size", "anchors", 85])]

    # 跳过之后没有可用检测头，报的是「没有 3 维检测输出」
    with pytest.raises(Exception) as excinfo:
        pick_detection_output(outputs)
    assert "没有 3 维检测输出" in str(excinfo.value)
