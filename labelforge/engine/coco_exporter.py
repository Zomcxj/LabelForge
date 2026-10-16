import json
import os
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .base_exporter import BaseExporter


class CocoExporter(BaseExporter):

    def _write_files(self, items: List[dict], selected_labels: List[str] = None):
        classes = self._get_classes(items, selected_labels)
        categories = [{"id": i + 1, "name": c, "supercategory": ""}
                      for i, c in enumerate(classes)]
        images = []
        annotations = []
        annotation_id = 1
        total = len(items)
        for idx, item in enumerate(items):
            self._process_one(item, classes, images, annotations, annotation_id)
            if annotations:
                annotation_id += len(item["boxes"])
            if self.on_progress:
                self.on_progress(idx + 1, total)
        coco_data = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": "LabelForge COCO Export",
                "contributor": "LabelForge",
                "url": "",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        json_path = os.path.join(self.output_dir, "coco_detection.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        for item in items:
            self._copy_image(item)

    def _process_one(self, item: dict, classes: List[str],
                     images: list, annotations: list, annotation_id: int):
        stem = item["stem"]
        boxes = item["boxes"]
        iw = item.get("width", 0)
        ih = item.get("height", 0)
        if iw == 0 or ih == 0:
            return
        image_id = len(images) + 1
        ext = ".jpg"
        if "img_path" in item:
            ext = os.path.splitext(item["img_path"])[1]
        images.append({
            "id": image_id,
            "file_name": f"{stem}{ext}",
            "width": iw,
            "height": ih,
        })
        for b in boxes:
            x1 = max(0, b["x"])
            y1 = max(0, b["y"])
            x2 = min(iw, b["x"] + b["width"])
            y2 = min(ih, b["y"] + b["height"])
            w = x2 - x1
            h = y2 - y1
            if w < 1 or h < 1:
                continue
            cat_id = classes.index(b["label"]) + 1
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [int(x1), int(y1), int(w), int(h)],
                "area": int(w * h),
                "iscrowd": 0,
                "segmentation": [],
            })
            annotation_id += 1
