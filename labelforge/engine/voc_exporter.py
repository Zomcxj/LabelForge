import os
from typing import Callable, Dict, List, Optional

from .base_exporter import BaseExporter


class VocExporter(BaseExporter):

    def _write_files(self, items: List[dict], selected_labels: List[str] = None):
        total = len(items)
        for idx, item in enumerate(items):
            self._write_one(item)
            if self.on_progress:
                self.on_progress(idx + 1, total)

    def _write_one(self, item: dict):
        stem = item["stem"]
        boxes = item["boxes"]
        iw = item.get("width", 0)
        ih = item.get("height", 0)
        if iw == 0 or ih == 0:
            return
        xml_path = os.path.join(self.output_dir, "labels", f"{stem}.xml")
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" ?>\n')
            f.write('<annotation>\n')
            f.write('  <folder>LabelForge</folder>\n')
            f.write(f'  <filename>{stem}</filename>\n')
            f.write('  <size>\n')
            f.write(f'    <width>{iw}</width>\n')
            f.write(f'    <height>{ih}</height>\n')
            f.write('    <depth>3</depth>\n')
            f.write('  </size>\n')
            f.write('  <source>\n')
            f.write('    <database>LabelForge</database>\n')
            f.write('  </source>\n')
            for b in boxes:
                x1 = max(0, b["x"])
                y1 = max(0, b["y"])
                x2 = min(iw, b["x"] + b["width"])
                y2 = min(ih, b["y"] + b["height"])
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                f.write('  <object>\n')
                f.write(f'    <name>{b["label"]}</name>\n')
                f.write('    <pose>Unspecified</pose>\n')
                f.write('    <truncated>0</truncated>\n')
                f.write('    <difficult>0</difficult>\n')
                f.write('    <bndbox>\n')
                f.write(f'      <xmin>{int(x1)}</xmin>\n')
                f.write(f'      <ymin>{int(y1)}</ymin>\n')
                f.write(f'      <xmax>{int(x2)}</xmax>\n')
                f.write(f'      <ymax>{int(y2)}</ymax>\n')
                f.write('    </bndbox>\n')
                f.write('  </object>\n')
            f.write('</annotation>\n')
        self._copy_image(item)
