"""COCO Detection 格式导出：所有图片与标注汇总进一个 coco_detection.json。

序列化由 ``dataset_converter._write_coco()`` 完成，本模块只声明格式。
"""
from .base_exporter import BaseExporter


class CocoExporter(BaseExporter):

    def _output_format(self) -> str:
        return "coco"
