"""逐帧推理结果落盘：图片边跑边写，标注在收尾时统一写。"""

import json
import os
import re

import cv2

from ..core.config import LABELME_VERSION

INVALID_NAME_CHARS = re.compile(r'[\\/:*?"<>|]+')
IMAGE_DIR = "images"          # 图片与同名 JSON 同放这一层
MANIFEST_NAME = "video_inference_manifest.json"
CLASSES_NAME = "classes.txt"


MARKER_NAME = ".incomplete"


def _safe(name):
    """文件名里不能出现的字符换成下划线。"""
    return INVALID_NAME_CHARS.sub("_", name or "clip")


def _unique_root(root):
    """目录已存在且非空时改用 `<root>_2`、`<root>_3`，不覆盖上次的导出。"""
    candidate = root
    index = 2
    while os.path.isdir(candidate) and os.listdir(candidate):
        candidate = f"{root}_{index}"
        index += 1
    return candidate


def _shape_from_box(box):
    """内部框 → LabelMe shape。跟踪编号写进 flags，不占用类别名。"""
    x, y = float(box["x"]), float(box["y"])
    w, h = float(box["width"]), float(box["height"])
    flags = {"inference": True}
    if box.get("track_id") is not None:
        flags["track_id"] = int(box["track_id"])
    return {
        "label": box.get("label", ""),
        "score": None if box.get("score") is None else float(box["score"]),
        "points": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        "group_id": box.get("group_id"),
        "description": "",
        "shape_type": "rectangle",
        "flags": flags,
    }


class FrameDatasetWriter:
    """写 `<root>/images` 下的逐帧图片与同名 JSON，根下 `classes.txt` 与清单。

    `images/` 里 `<stem>.jpg` 与 `<stem>.json` 成对摆放，可直接当 LabelMe
    数据集加载；`classes.txt` 与清单留在根目录，位置与旧版一致。
    """

    def __init__(self, root, video_name, image_format="jpg", quality=95):
        self.root = _unique_root(os.path.abspath(root))
        self.name = _safe(os.path.splitext(os.path.basename(video_name))[0])
        self.image_format = image_format.lower().lstrip(".")
        self.quality = int(quality)
        os.makedirs(self.root, exist_ok=True)
        self._image_dir = os.path.join(self.root, IMAGE_DIR)
        os.makedirs(self._image_dir, exist_ok=True)
        # 收尾前留个标记：进程被强杀时能看出这套数据没写完
        with open(os.path.join(self.root, MARKER_NAME), "w",
                  encoding="utf-8") as handle:
            handle.write("incomplete\n")
        self.image_count = 0
        # 画面尺寸取自第一张写入的图；标注文件里的 imageWidth/Height
        # 必须与图片一致，转 YOLO 时按它归一化坐标
        self.width = 0
        self.height = 0

    def _stem(self, index):
        return f"{self.name}_{index + 1:06d}"

    def write_image(self, index, rgb):
        """写第 index 帧（0 起）的干净画面，返回路径。"""
        if not self.width or not self.height:
            self.height, self.width = int(rgb.shape[0]), int(rgb.shape[1])
        path = os.path.join(self._image_dir,
                            f"{self._stem(index)}.{self.image_format}")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if self.image_format in ("jpg", "jpeg"):
            written = cv2.imwrite(
                path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        else:
            written = cv2.imwrite(path, bgr)
        # imwrite 失败只返回 False（磁盘满、路径被占），不检查会写出
        # 「有标注、没图片」的半套数据集
        if not written:
            raise OSError(f"逐帧图片写入失败：{path}")
        self.image_count += 1
        return path

    def finalize(self, frames, class_names, snapshot, completed):
        """写每帧标注、classes.txt 与清单。frames 为 [(帧号, 框列表)]。

        `classes.txt` 以帧里实际出现的类别为准；`class_names` 只为调用方
        兼容保留，不再整表写盘。
        """
        for index, boxes in frames:
            data = {
                "version": LABELME_VERSION,
                "flags": {"inference": True},
                "shapes": [_shape_from_box(box) for box in boxes],
                "imagePath": f"{self._stem(index)}.{self.image_format}",
                "imageData": None,
                "imageHeight": self.height,
                "imageWidth": self.width,
            }
            path = os.path.join(self._image_dir,
                                f"{self._stem(index)}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)

        # 只写这批帧里真正出现的类别：把模型全部类别抄进来会让训练集的
        # data.yaml 多出一堆没有任何样本的类
        names = []
        for _, rows in frames:
            for box in rows:
                label = str(box.get("label", ""))
                if label and label not in names:
                    names.append(label)
        with open(os.path.join(self.root, CLASSES_NAME), "w",
                  encoding="utf-8") as handle:
            if names:
                handle.write("\n".join(names) + "\n")

        manifest = {
            "video": self.name,
            "completed": bool(completed),
            "frame_count": len(frames),
            "image_count": self.image_count,
            "image_width": self.width,
            "image_height": self.height,
            "image_dir": IMAGE_DIR,
            "snapshot": snapshot or {},
        }
        with open(os.path.join(self.root, MANIFEST_NAME), "w",
                  encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        try:
            os.remove(os.path.join(self.root, MARKER_NAME))
        except OSError:
            pass
        return manifest
