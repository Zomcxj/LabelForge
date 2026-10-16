import os
import random
import shutil
from typing import Callable, Dict, List


class Splitter:

    def __init__(self, output_dir: str, on_progress: Callable = None,
                 is_interrupted: Callable = None):
        self.output_dir = os.path.join(output_dir, "divide")
        self.on_progress = on_progress
        self.is_interrupted = is_interrupted or (lambda: False)

    def run(
        self,
        background_images: List[str],
        detection_boxes_dict: Dict[int, List[dict]],
        ratios: Dict[str, float],
        classes: List[str] = None,
    ):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        total = float(sum(ratios.values()))
        if total <= 0:
            return
        ratios = {k: v / total for k, v in ratios.items()}
        files = list(range(len(background_images)))
        random.Random(42).shuffle(files)
        n = len(files)
        train_n = int(n * ratios.get("train", 0.8))
        val_n = int(n * ratios.get("val", 0.1))
        test_n = int(n * ratios.get("test", 0.1))
        rest = n - train_n - val_n - test_n
        train_n += rest
        splits = {
            "train": files[:train_n],
            "val": files[train_n:train_n + val_n],
            "test": files[train_n + val_n:]
        }
        count = 0
        total_files = len(files)
        yolo_labels_dir = os.path.join(os.path.dirname(self.output_dir), "labels")
        for split_name, indices in splits.items():
            split_img_dir = os.path.join(self.output_dir, "images", split_name)
            split_lbl_dir = os.path.join(self.output_dir, "labels", split_name)
            os.makedirs(split_img_dir, exist_ok=True)
            os.makedirs(split_lbl_dir, exist_ok=True)
            for idx in indices:
                if self.is_interrupted():
                    return
                src = background_images[idx]
                dst = os.path.join(split_img_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                base = os.path.splitext(os.path.basename(src))[0]
                src_txt = os.path.join(yolo_labels_dir, f"{base}.txt")
                if os.path.exists(src_txt):
                    dst_txt = os.path.join(split_lbl_dir, f"{base}.txt")
                    if os.path.exists(dst_txt):
                        os.remove(dst_txt)
                    shutil.copy2(src_txt, dst_txt)
                count += 1
                if self.on_progress:
                    self.on_progress(count, total_files)
        self._write_dataset_yaml(splits, background_images, classes)

    def _write_dataset_yaml(self, splits: Dict[str, list], images: List[str],
                             classes: List[str] = None):
        if classes is None:
            classes_txt = os.path.join(os.path.dirname(self.output_dir), "classes.txt")
            if os.path.exists(classes_txt):
                with open(classes_txt, 'r') as f:
                    classes = [line.strip() for line in f if line.strip()]
            else:
                classes = []
        lines = [f"path: {os.path.abspath(self.output_dir)}"]
        for split_name in ("train", "val", "test"):
            if split_name in splits and splits[split_name]:
                rel = f"./images/{split_name}"
                lines.append(f"{split_name}: {rel}")
        lines.append(f"nc: {len(classes)}")
        lines.append(f"names: {classes}")
        path = os.path.join(self.output_dir, "dataset.yaml")
        with open(path, 'w') as f:
            f.write("\n".join(lines) + "\n")
        if not splits.get("val"):
            import logging
            logging.warning("Validation set is empty")
