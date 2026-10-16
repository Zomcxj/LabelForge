import os
from typing import Callable, Dict, List, Optional


class BaseExporter:

    def __init__(self, output_dir: str, on_progress: Callable = None,
                 is_interrupted: Callable = None):
        self.output_dir = output_dir
        self.on_progress = on_progress
        self.is_interrupted = is_interrupted or (lambda: False)

    def run(
        self,
        background_images: List[str],
        detection_boxes_dict: Dict[int, List[dict]],
        selected_labels: List[str],
        skip_empty: bool = False,
        input_data: Optional[List[dict]] = None,
    ):
        if input_data is not None:
            items = []
            self._ensure_dirs()
            for idx, item in enumerate(input_data):
                if self.is_interrupted():
                    break
                boxes = [b for b in item["boxes"] if b["label"] in selected_labels]
                if skip_empty and not boxes:
                    continue
                items.append({
                    "stem": item["stem"],
                    "boxes": boxes,
                    "width": item["width"],
                    "height": item["height"],
                    "image": item["image"],
                    "img_path": item.get("img_path", ""),
                })
            self._write_files(items, selected_labels)
            return
        total = len(background_images)
        items = []
        self._ensure_dirs()
        for idx in range(total):
            if self.is_interrupted():
                break
            img_path = background_images[idx]
            base = os.path.splitext(os.path.basename(img_path))[0]
            boxes = detection_boxes_dict.get(idx, [])
            boxes = [b for b in boxes if b["label"] in selected_labels]
            if skip_empty and not boxes:
                continue
            try:
                from PyQt5.QtGui import QImage as _QImage
                img = _QImage(img_path)
                w, h = img.width(), img.height()
            except Exception:
                w, h = 0, 0
                img = None
            items.append({
                "stem": base,
                "boxes": boxes,
                "width": w,
                "height": h,
                "img_path": img_path,
                "image": img
            })
        self._write_files(items, selected_labels)

    def _ensure_dirs(self):
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels"), exist_ok=True)

    def _write_files(self, items: List[dict], selected_labels: List[str] = None):
        raise NotImplementedError

    def _copy_image(self, item: dict):
        img = item.get("image")
        stem = item["stem"]
        if img is not None and not img.isNull():
            ext = ".jpg"
            img_path = os.path.join(self.output_dir, "images", f"{stem}{ext}")
            tmp = img_path + ".tmp"
            img.save(tmp)
            img = None
            if os.path.exists(img_path):
                os.remove(img_path)
            os.rename(tmp, img_path)
        elif item.get("img_path"):
            # 空 img_path 必须视为「没有源图」：normpath("") 会得到 "."，
            # 继续走会把当前工作目录当成文件去复制并抛 PermissionError。
            ext = os.path.splitext(item["img_path"])[1]
            import shutil
            dst = os.path.join(self.output_dir, "images", f"{stem}{ext}")
            src = os.path.normpath(item["img_path"])
            dst_n = os.path.normpath(dst)
            if src != dst_n:
                if os.path.exists(dst_n):
                    os.remove(dst_n)
                shutil.copy2(src, dst_n)

    def _get_classes(self, items: List[dict], selected_labels: List[str] = None):
        if selected_labels:
            classes = sorted(selected_labels)
        else:
            classes_set = set()
            for item in items:
                for b in item["boxes"]:
                    classes_set.add(b["label"])
            classes = sorted(classes_set)
        classes_txt = os.path.join(self.output_dir, "classes.txt")
        if os.path.exists(classes_txt):
            with open(classes_txt, 'r') as f:
                existing = [line.strip() for line in f if line.strip()]
            if existing:
                existing_set = set(existing)
                classes = existing + [c for c in classes if c not in existing_set]
        return classes
