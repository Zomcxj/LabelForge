"""YOLO 格式导出：标签写成每图一个 txt，附 data.yaml 与 classes.txt。

行格式由 ``dataset_converter._yolo_lines_for()`` 统一产出（det/seg/obb/pose
四种任务各有一套规则），本模块只决定用哪种任务。

任务类型不需要调用方指定：JSON 里每个 shape 都带 ``shape_type``，扫描一遍
就能判定该用哪种任务。调用方仍可显式传 ``mode`` 覆盖（测试与特殊场景用）。
"""
from typing import Iterable, Optional

from .base_exporter import BaseExporter

#: 界面下拉框里「YOLO」系列的取值 -> 任务类型
MODE_TO_TASK = {
    "hbb": "det",
    "det": "det",
    "seg": "seg",
    "obb": "obb",
    "pose": "pose",
    "auto": "det",      # 仅占位；实际由 detect_task() 判定
}

#: shape_type -> 对应的 YOLO 任务。多边形的目标检测导出取外接框，所以
#: polygon 同时意味着 seg；这里单列是为了判定"是否需要分割任务"。
SHAPE_TO_TASK = {
    "rectangle": "det",
    "polygon": "seg",
    "rotation": "obb",
    "point": "pose",
}


def detect_task(boxes: Iterable[dict]) -> str:
    """按框的 shape_type 判定该用哪种 YOLO 任务。

    判定顺序按"信息量"从多到少：
    1. 有 ``rotation`` → obb（旋转框必须用 OBB 写，否则丢角度）
    2. 有点且带 ``group_id`` → pose（关键点靠 group_id 与框配对）
    3. 有 ``polygon`` → seg（多边形必须保留顶点，降级成框会丢轮廓）
    4. 其余 → det

    pose 要先于 seg：人体关键点标注常常"一个矩形 + 若干点"，其中笔画框可能
    被标成矩形而非多边形，先判 pose 才不会把关键点丢掉。
    """
    has_rotation = False
    has_polygon = False
    has_keypoint = False

    for box in boxes:
        shape_type = str(box.get("shape_type") or "rectangle").lower()
        if shape_type == "rotation":
            has_rotation = True
        elif shape_type == "polygon":
            has_polygon = True
        elif shape_type == "point" and box.get("group_id") is not None:
            has_keypoint = True

    if has_rotation:
        return "obb"
    if has_keypoint:
        return "pose"
    if has_polygon:
        return "seg"
    return "det"


def detect_task_for_items(entries: Iterable[dict]) -> str:
    """对整批导出条目判定任务类型（``entries`` 每项含 ``boxes``）。"""
    for entry in entries:
        task = detect_task(entry.get("boxes") or ())
        if task != "det":
            return task
    return "det"


class YoloExporter(BaseExporter):

    def __init__(self, output_dir, on_progress=None, is_interrupted=None,
                 mode: Optional[str] = None):
        super().__init__(output_dir, on_progress, is_interrupted)
        #: None 或 "auto" 表示按数据自动判定；其余按 MODE_TO_TASK 映射。
        self.mode = mode
        self._detected_task = "det"

    def _output_format(self) -> str:
        return "yolo"

    def _task(self) -> str:
        return self._detected_task

    def _write(self, items, selected_labels=None):
        """先判定任务类型，再交给基类写出。

        判定要在 ``_collect`` 之后、``_write`` 之前完成——基类 ``_write``
        里会按 ``_task()`` 决定行格式，所以这里先算好再调 super()。
        """
        mode = str(self.mode or "auto").lower()
        if mode in ("", "auto"):
            self._detected_task = detect_task_for_items(items)
        else:
            self._detected_task = MODE_TO_TASK.get(mode, "det")
        super()._write(items, selected_labels)
