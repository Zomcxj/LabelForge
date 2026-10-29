"""Pascal VOC Detection 格式导出：每图一个 XML，放在 labels/ 下。

序列化由 ``dataset_converter._write_voc()`` 完成。那边用 ``ET.SubElement``
建树，标签里的 ``&``、``<``、``>`` 由标准库自动转义——手拼字符串会在这些
字符上产出坏 XML。
"""
from .base_exporter import BaseExporter


class VocExporter(BaseExporter):

    def _output_format(self) -> str:
        return "voc"
