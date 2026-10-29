"""导出器公共基类：把画布数据收集成条目，交给 dataset_converter 写出。

职责划分：
- **本模块**只负责「从编辑器侧取数」——挑选类别、跳过空图、按需复制/解码
  源图，产出统一结构的条目列表；
- **序列化与落盘**全部交给 ``dataset_converter`` 的写出层（``_write_yolo`` /
  ``_write_coco`` / ``_write_voc``）。那边已有分块、进度与中断处理，且 VOC
  走 ``ET.SubElement``（自动转义标签里的 ``&``/``<``），不再手拼 XML。

这样做的原因：同一份数据集曾经有三套写出实现（导出器、格式转换、数据增强
落盘），VOC 的转义与 YOLO 的钳制规则各写各的，改一处漏两处。现在只有一份。
"""
import os
from typing import Callable, Dict, List, Optional

#: 导出条目：{'stem', 'boxes', 'width', 'height', 'img_path', 'image'}
#: ``image`` 为 QImage 或 None；为 None 时按 ``img_path`` 复制源图。
ExportItem = Dict[str, object]


class BaseExporter:
    """导出器基类。

    子类实现 ``_output_format()`` 返回目标格式（``yolo`` / ``coco`` / ``voc``），
    必要时覆盖 ``_task()`` 指定 YOLO 的任务类型。
    """

    def __init__(self, output_dir: str, on_progress: Callable = None,
                 is_interrupted: Callable = None):
        self.output_dir = output_dir
        self.on_progress = on_progress
        self.is_interrupted = is_interrupted or (lambda: False)

    # ---------- 子类接口 ----------

    def _output_format(self) -> str:
        """目标格式键，供 ``dataset_converter.output_layout()`` 使用。"""
        raise NotImplementedError

    def _task(self) -> str:
        """YOLO 任务类型（det/seg/obb/pose）；非 YOLO 格式忽略。"""
        return "det"

    # ---------- 收集 ----------

    def run(
        self,
        background_images: List[str],
        detection_boxes_dict: Dict[int, List[dict]],
        selected_labels: List[str],
        skip_empty: bool = False,
        input_data: Optional[List[dict]] = None,
    ):
        """执行导出。

        ``input_data`` 给定时按它导出（每条自带 stem/boxes/尺寸），否则用
        ``background_images`` + ``detection_boxes_dict`` 组合。
        """
        items = self._collect(input_data, background_images,
                              detection_boxes_dict, selected_labels, skip_empty)
        self._write(items, selected_labels)

    def _collect(self, input_data, background_images, detection_boxes_dict,
                 selected_labels, skip_empty) -> List[ExportItem]:
        """把两种输入形态统一成条目列表，并按勾选的类别过滤。"""
        allowed = set(selected_labels or ())
        items: List[ExportItem] = []

        if input_data is not None:
            for entry in input_data:
                if self.is_interrupted():
                    break
                boxes = self._filter_boxes(entry.get("boxes") or [], allowed)
                if skip_empty and not boxes:
                    continue
                items.append({
                    "stem": str(entry.get("stem") or ""),
                    "boxes": boxes,
                    "width": int(entry.get("width") or 0),
                    "height": int(entry.get("height") or 0),
                    "image": entry.get("image"),
                    "img_path": entry.get("img_path") or "",
                })
            return items

        for index, img_path in enumerate(background_images or []):
            if self.is_interrupted():
                break
            boxes = self._filter_boxes(detection_boxes_dict.get(index) or [], allowed)
            if skip_empty and not boxes:
                continue
            width, height = self._image_size(img_path)
            items.append({
                "stem": os.path.splitext(os.path.basename(img_path))[0],
                "boxes": boxes,
                "width": width,
                "height": height,
                "image": None,
                "img_path": img_path,
            })
        return items

    @staticmethod
    def _filter_boxes(boxes, allowed) -> List[dict]:
        """只保留勾选类别的框；空集合表示一个都不留（与「全选」区分在调用方）。"""
        if not allowed:
            return []
        return [box for box in boxes if box.get("label") in allowed]

    @staticmethod
    def _image_size(img_path: str):
        """读图片宽高；只读文件头，不整图解码（万图导出时快一个数量级）。"""
        from .dataset_converter import read_image_size
        size = read_image_size(img_path)
        return size if size else (0, 0)

    # ---------- 写出 ----------

    def _write(self, items: List[ExportItem], selected_labels: List[str] = None):
        """把条目交给 dataset_converter 写出。"""
        from . import dataset_converter as dc

        class_names = self._class_names(items, selected_labels)
        if not class_names or not items:
            return

        dataset = self._build_dataset(items, class_names)
        layout = self._layout()
        os.makedirs(self.output_dir, exist_ok=True)
        # 导出器的输出结构是对外契约（用户按它在训练脚本里找文件），
        # 与格式转换窗口的目录名不同——这里显式给布局，不用 layout() 的默认值。
        os.makedirs(layout["images"], exist_ok=True)
        if self._uses_dir_annotations():
            os.makedirs(layout["annotations"], exist_ok=True)

        fmt = self._output_format()
        task = self._task()
        total = len(items)

        if fmt == "coco":
            # COCO 的标注是单个 json，只能整体写出：分块写出会让后一块
            # 覆盖前一块，最后只剩最后一批图。
            dc._write_chunk(dataset, fmt, layout, first=True, task=task,
                            is_interrupted=self.is_interrupted)
            if self.on_progress:
                self.on_progress(total, total)
            return

        done = 0
        for start in range(0, total, dc._CHUNK_SIZE):
            if self.is_interrupted():
                return
            chunk = self._slice(dataset, items, start, start + dc._CHUNK_SIZE)
            dc._write_chunk(chunk, fmt, layout, first=(start == 0), task=task)
            done += len(chunk.image_paths)
            if self.on_progress:
                self.on_progress(done, total)

    def _uses_dir_annotations(self) -> bool:
        """标注是按图各写一个文件（放目录里）还是汇总成一个文件。"""
        return self._output_format() in ("yolo", "voc")

    def _layout(self) -> dict:
        """输出目录结构。

        - yolo：`images/` + `labels/` + `data.yaml`
        - voc：`images/` + `labels/`（每图一个 xml）
        - coco：`images/` + 根目录下 `coco_detection.json`

        注意与 ``dataset_converter.output_layout()`` 不同：那边是「格式转换」
        窗口的目录约定（voc 用 `annotations/`），这里沿用导出器一直以来的
        `labels/`——用户的训练脚本按这个名字找标注。
        """
        fmt = self._output_format()
        images = os.path.join(self.output_dir, "images")
        if fmt == "coco":
            return {
                "images": images,
                "annotations": os.path.join(self.output_dir, "coco_detection.json"),
            }
        layout = {
            "images": images,
            "annotations": os.path.join(self.output_dir, "labels"),
        }
        if fmt == "yolo":
            layout["data_yaml"] = os.path.join(self.output_dir, "data.yaml")
        return layout

    @staticmethod
    def _class_names(items: List[ExportItem], selected_labels: List[str] = None):
        """类别表：优先用勾选的（保序），否则按条目里出现的类别排序。"""
        if selected_labels:
            names = sorted(set(str(name) for name in selected_labels if str(name)))
            if names:
                return names
        found = set()
        for item in items:
            for box in item["boxes"]:
                label = box.get("label")
                if label:
                    found.add(str(label))
        return sorted(found)

    def _build_dataset(self, items: List[ExportItem], class_names: List[str]):
        """把条目转成 dataset_converter 的 Dataset/Detections 结构。"""
        from . import dataset_converter as dc

        index_of = {name: i for i, name in enumerate(class_names)}
        dataset = dc.Dataset(classes=list(class_names), image_paths=[],
                             annotations={}, image_sizes={})
        for position, item in enumerate(items):
            stem = str(item.get("stem") or f"item_{position}")
            source = item.get("img_path") or ""
            ext = os.path.splitext(source)[1] if source else ".jpg"
            file_name = f"{stem}{ext or '.jpg'}"

            # 用「条目序号」当键：img_path 可能为空（画布内合成图），
            # 空字符串做键会让所有条目落到同一条记录上。
            key = self._key_for(item, position)
            dataset.image_paths.append(key)
            dataset.output_names[key] = file_name
            dataset.source_of[key] = source or None
            if item["width"] and item["height"]:
                dataset.image_sizes[key] = (item["width"], item["height"])
            dataset.annotations[key] = self._detections_for(item, index_of)

            # 内存图没有源文件：先落盘到输出目录，再让写出层按路径复制。
            image = item.get("image")
            if image is not None and not source:
                written = self._persist_in_memory_image(image, file_name)
                if written:
                    dataset.source_of[key] = written
        return dataset

    def _persist_in_memory_image(self, image, file_name: str) -> str:
        """把内存 QImage 写进输出目录的 images/ 下，返回落盘路径。

        写出层随后会把它「复制」到最终位置；同目录时复制是空操作。
        """
        images_dir = os.path.join(self.output_dir, "images")
        try:
            os.makedirs(images_dir, exist_ok=True)
            target = os.path.join(images_dir, file_name)
            if hasattr(image, "isNull") and image.isNull():
                return ""
            if hasattr(image, "save") and image.save(target):
                return target
        except Exception:
            return ""
        return ""

    @staticmethod
    def _key_for(item: ExportItem, position: int) -> str:
        """条目标识：优先用源图路径；没有源图时用「stem + 全局唯一后缀」。

        键会经 ``dataset.name_of()`` 决定输出文件名，所以不能用 ``__item_0__``
        这类占位符——那会变成磁盘上的文件名。
        """
        source = item.get("img_path") or ""
        if source:
            return source
        stem = str(item.get("stem") or f"item_{position}")
        return f"{stem}#{position}"

    def _detections_for(self, item: ExportItem, index_of: Dict[str, int]):
        """把一条目的框转成 Detections。"""
        from . import dataset_converter as dc

        det = dc._empty_detections()
        for box in item["boxes"]:
            label = box.get("label")
            if label not in index_of:
                continue
            class_id = index_of[label]
            shape_type = box.get("shape_type") or "rectangle"
            points = box.get("points") or None

            if shape_type == "point":
                # 关键点的外接框是零尺寸，写进 box_xyxy 会被「退化框」判据
                # 连整条记录一起丢掉（pose 就只剩关键点、没有 bbox）。
                # 这里给它一个名义上的 1px 框，实际坐标由 points 决定。
                anchor = points[0] if points else (box.get("x", 0), box.get("y", 0))
                ax, ay = float(anchor[0]), float(anchor[1])
                dc._append_detection(
                    det, class_id, (ax, ay, ax, ay),
                    shape_type="point",
                    points=[(ax, ay)],
                    group_id=box.get("group_id"),
                    flags=box.get("flags") if isinstance(box.get("flags"), dict) else None,
                )
                continue

            if shape_type == "rectangle" and not points:
                xyxy = (float(box["x"]), float(box["y"]),
                        float(box["x"]) + float(box["width"]),
                        float(box["y"]) + float(box["height"]))
                dc._append_detection(
                    det, class_id, xyxy,
                    shape_type="rectangle",
                    group_id=box.get("group_id"),
                    flags=box.get("flags") if isinstance(box.get("flags"), dict) else None,
                )
                continue
            if not points:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            xyxy = (min(xs), min(ys), max(xs), max(ys))
            mask = points if shape_type == "polygon" else None
            dc._append_detection(
                det, class_id, xyxy,
                shape_type=shape_type,
                points=[(float(p[0]), float(p[1])) for p in points],
                group_id=box.get("group_id"),
                mask=mask,
                flags=box.get("flags") if isinstance(box.get("flags"), dict) else None,
            )
        return det

    def _slice(self, dataset, items: List[ExportItem], start: int, stop: int):
        """取 [start, stop) 区间的子数据集，供分块写出。"""
        from . import dataset_converter as dc

        keys = [self._key_for(item, i) for i, item in enumerate(items)][start:stop]
        return dc._slice_dataset(dataset, keys)
