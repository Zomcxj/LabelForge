import copy
import json
import os
import random
from typing import Callable, Dict, List, Tuple, Type

from PySide6.QtGui import QColor, QImage, QPainter

from .base import BaseTransform, get_all_transforms, register_transform
from .crop import (
    crop_boxes, normalized_fill_rgb, normalized_window, window_starts,
)

from . import flipt, color, noise, translate, rotate, scale


class Augmenter:

    def __init__(self, output_dir: str, on_progress: Callable = None,
                 is_interrupted: Callable = None, on_transform_progress: Callable = None,
                 seed: int = None):
        self.output_dir = os.path.join(output_dir, "images")
        self.on_progress = on_progress
        self.is_interrupted = is_interrupted or (lambda: False)
        self.on_transform_progress = on_transform_progress
        # 私有 RNG：seed 给定时结果可复现，且不污染全局 random
        self._rng = random.Random(seed)

    def run(
        self,
        background_images: List[str],
        detection_boxes_dict: Dict[int, List[dict]],
        transform_specs: List[Tuple[Type[BaseTransform], Dict[str, Tuple[float, float]]]],
        image_ratio: float = 1.0,
        mode: str = "all",
        include_original: bool = False,
        skip_empty: bool = True,
        crop_spec: dict = None,
    ) -> List[dict]:
        os.makedirs(self.output_dir, exist_ok=True)
        total = len(background_images)
        results = []

        if include_original:
            for idx in range(total):
                if self.is_interrupted():
                    break
                img_path = background_images[idx]
                base = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
                    ext = '.jpg'
                boxes = detection_boxes_dict.get(idx, [])
                if skip_empty and not boxes:
                    continue
                self._save_original(img_path, base, ext, boxes, results)

        if mode == "random":
            results = self._run_random(background_images, detection_boxes_dict,
                                       transform_specs, image_ratio, total, results, skip_empty,
                                       crop_spec)
        else:
            results = self._run_all(background_images, detection_boxes_dict,
                                    transform_specs, image_ratio, total, results, skip_empty,
                                    crop_spec)
        return results

    def _run_all(self, background_images, detection_boxes_dict,
                 transform_specs, image_ratio, total, results=None, skip_empty=True,
                 crop_spec=None):
        if results is None:
            results = []
        for cls, ranges in transform_specs:
            if self.is_interrupted():
                break
            t_name = cls.__name__
            if self.on_transform_progress:
                self.on_transform_progress(t_name, 0, total)
            for idx in range(total):
                if self.is_interrupted():
                    break
                if self._rng.random() >= image_ratio:
                    if self.on_transform_progress:
                        self.on_transform_progress(t_name, idx + 1, total)
                    continue
                img_path = background_images[idx]
                base = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
                    ext = '.jpg'
                boxes = detection_boxes_dict.get(idx, [])
                if skip_empty and not boxes:
                    if self.on_transform_progress:
                        self.on_transform_progress(t_name, idx + 1, total)
                    continue
                original_image = QImage(img_path)
                iw = original_image.width()
                ih = original_image.height()
                kwargs = self._build_kwargs(cls, ranges, "all")
                t = cls(**kwargs)
                img_copy = original_image.copy()
                boxes_copy = copy.deepcopy(boxes)
                img_copy, boxes_copy = t.apply(img_copy, boxes_copy, iw, ih)
                suffix = "_" + t.name
                out_name = base + suffix + ext
                img_copy.save(os.path.join(self.output_dir, out_name))
                self._save_labelme_json(
                    os.path.join(self.output_dir, base + suffix + ".json"),
                    out_name, boxes_copy, img_copy.width(), img_copy.height()
                )
                results.append({
                    "image": img_copy, "boxes": boxes_copy,
                    "width": img_copy.width(), "height": img_copy.height(),
                    "stem": base + suffix
                })
                if self.on_transform_progress:
                    self.on_transform_progress(t_name, idx + 1, total)
        if crop_spec:
            t_name = "crop"
            if self.on_transform_progress:
                self.on_transform_progress(t_name, 0, total)
            for idx in range(total):
                if self.is_interrupted():
                    break
                img_path = background_images[idx]
                base = os.path.splitext(os.path.basename(img_path))[0]
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
                    ext = '.jpg'
                boxes = detection_boxes_dict.get(idx, [])
                if skip_empty and not boxes:
                    if self.on_transform_progress:
                        self.on_transform_progress(t_name, idx + 1, total)
                    continue
                if self._rng.random() >= image_ratio:
                    if self.on_transform_progress:
                        self.on_transform_progress(t_name, idx + 1, total)
                    continue
                original_image = QImage(img_path)
                self._apply_crop(original_image, boxes, crop_spec, base, ext, results)
                if self.on_transform_progress:
                    self.on_transform_progress(t_name, idx + 1, total)
        return results

    def _run_random(self, background_images, detection_boxes_dict,
                    transform_specs, image_ratio, total, results=None, skip_empty=True,
                    crop_spec=None):
        if results is None:
            results = []
        for idx in range(total):
            if self.is_interrupted():
                break
            img_path = background_images[idx]
            base = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
                ext = '.jpg'
            boxes = detection_boxes_dict.get(idx, [])
            if skip_empty and not boxes:
                if self.on_progress:
                    self.on_progress(idx + 1, total)
                continue
            original_image = QImage(img_path)
            original_boxes = copy.deepcopy(boxes)
            iw = original_image.width()
            ih = original_image.height()
            pool = list(transform_specs)
            if crop_spec:
                pool.append(("__crop__", crop_spec))
            n = self._rng.randint(1, len(pool))
            active = self._rng.sample(pool, n)
            any_applied = False
            for cls, ranges in active:
                if cls == "__crop__":
                    if self._rng.random() >= image_ratio:
                        continue
                    if self._apply_crop(original_image, boxes, ranges, base, ext, results):
                        any_applied = True
                    continue
                if self._rng.random() >= image_ratio:
                    continue
                any_applied = True
                kwargs = self._build_kwargs(cls, ranges, "random")
                t = cls(**kwargs)
                img_copy = original_image.copy()
                boxes_copy = copy.deepcopy(original_boxes)
                img_copy, boxes_copy = t.apply(img_copy, boxes_copy, iw, ih)
                suffix = "_" + t.name
                out_name = base + suffix + ext
                img_copy.save(os.path.join(self.output_dir, out_name))
                self._save_labelme_json(
                    os.path.join(self.output_dir, base + suffix + ".json"),
                    out_name, boxes_copy, img_copy.width(), img_copy.height()
                )
                results.append({
                    "image": img_copy, "boxes": boxes_copy,
                    "width": img_copy.width(), "height": img_copy.height(),
                    "stem": base + suffix
                })
            if not any_applied:
                self._save_original(img_path, base, ext, boxes, results)
            if self.on_progress:
                self.on_progress(idx + 1, total)
        return results

    def _apply_crop(self, original_image, boxes, crop_spec, base, ext, results):
        """滑窗裁剪 1→N：产出全部非空窗口（空窗无条件丢弃）。返回产出数。

        crop_spec['fill'] 为真时：窗口尺寸不钳到图像内、按等步长网格排布，
        末行/末列越出图像的部分以 fill_rgb 补齐，命名追加 _pad。
        """
        iw, ih = original_image.width(), original_image.height()
        cw, ch, ov = normalized_window(iw, ih, crop_spec)
        min_vis = float(crop_spec.get("min_visible", 0.7))
        pad = bool(crop_spec.get("fill"))
        rgb = normalized_fill_rgb(crop_spec) if pad else None
        made = 0
        for r, y0 in enumerate(window_starts(ih, ch, ov, fill=pad)):
            for c, x0 in enumerate(window_starts(iw, cw, ov, fill=pad)):
                if self.is_interrupted():
                    return made
                win_boxes = crop_boxes(boxes, x0, y0, cw, ch, min_vis)
                if not win_boxes:
                    continue
                win_img = self._crop_window(original_image, x0, y0, cw, ch, rgb)
                suffix = f"_crop_r{r + 1}c{c + 1}_ov{ov}"
                if pad:
                    suffix += "_pad"
                out_name = base + suffix + ext
                win_img.save(os.path.join(self.output_dir, out_name))
                self._save_labelme_json(
                    os.path.join(self.output_dir, base + suffix + ".json"),
                    out_name, win_boxes, win_img.width(), win_img.height()
                )
                results.append({
                    "image": win_img, "boxes": win_boxes,
                    "width": win_img.width(), "height": win_img.height(),
                    "stem": base + suffix
                })
                made += 1
        return made

    @staticmethod
    def _crop_window(original_image, x0, y0, cw, ch, rgb):
        """切取窗口；rgb 给定时，窗口越出图像边缘的部分以该 RGB 补齐。"""
        if rgb is None or (x0 + cw <= original_image.width()
                           and y0 + ch <= original_image.height()):
            return original_image.copy(x0, y0, cw, ch)
        win = QImage(cw, ch, original_image.format())
        win.fill(QColor(*rgb))
        painter = QPainter(win)
        painter.drawImage(-x0, -y0, original_image)
        painter.end()
        return win

    def _build_kwargs(self, cls, ranges, mode):
        kwargs = {}
        for pname, (vmin, vmax) in ranges.items():
            kwargs[pname] = self._rng.uniform(vmin, vmax)
        if cls.__name__ in ("RandomRotation", "RandomScale"):
            if "angle" in kwargs:
                kwargs["max_angle"] = abs(kwargs.pop("angle"))
            if "scale" in kwargs:
                v = kwargs.pop("scale")
                kwargs["min"] = min(v, 1.0)
                kwargs["max"] = max(v, 1.0)
        if cls.__name__ == "RandomTranslate":
            if "offset" in kwargs:
                v = int(abs(kwargs.pop("offset")))
                kwargs["max_dx"] = v
                kwargs["max_dy"] = v
        if cls.__name__ in ("Brightness", "Hue", "Value") and "delta" in kwargs:
            kwargs["delta"] = int(kwargs["delta"])
        if cls.__name__ == "GaussianNoise" and "sigma" in kwargs:
            kwargs["sigma"] = int(kwargs["sigma"])
        kwargs["rng"] = self._rng
        return kwargs

    def _save_original(self, img_path, base, ext, boxes, results):
        data = QImage(img_path)
        out_img = os.path.join(self.output_dir, base + ext)
        if not os.path.exists(out_img):
            data.save(out_img)
        out_json = os.path.join(self.output_dir, base + ".json")
        if not os.path.exists(out_json):
            self._save_labelme_json(out_json, base + ext, boxes, data.width(), data.height())
        results.append({
            "image": data, "boxes": boxes,
            "width": data.width(), "height": data.height(),
            "stem": base
        })

    def _save_labelme_json(self, json_path, image_name, boxes, width, height):
        from ..shape_io import labelme_shape_from_box
        shapes = [labelme_shape_from_box(b) for b in boxes]
        data = {
            "version": "5.2.1", "flags": {},
            "shapes": shapes, "imagePath": image_name,
            "imageData": None,
            "imageHeight": height, "imageWidth": width
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
