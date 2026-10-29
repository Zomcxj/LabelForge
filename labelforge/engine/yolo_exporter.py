"""YOLO 格式导出：标签写成每图一个 txt，附 data.yaml 与 classes.txt。

行格式由 ``dataset_converter._yolo_lines_for()`` 统一产出（det/seg/obb/pose
四种任务各有一套规则），本模块只决定用哪种任务。
"""
from .base_exporter import BaseExporter

#: 界面下拉框里「YOLO」系列的取值 -> 任务类型
MODE_TO_TASK = {
    "hbb": "det",
    "det": "det",
    "seg": "seg",
    "obb": "obb",
    "pose": "pose",
}


class YoloExporter(BaseExporter):

    def __init__(self, output_dir, on_progress=None, is_interrupted=None,
                 mode="hbb"):
        super().__init__(output_dir, on_progress, is_interrupted)
        self.mode = mode

    def _output_format(self) -> str:
        return "yolo"

    def _task(self) -> str:
        return MODE_TO_TASK.get(str(self.mode or "hbb").lower(), "det")
