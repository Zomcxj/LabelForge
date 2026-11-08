"""逐帧数据集：目录结构、命名、JSON 字段、清单。"""
import json
import os

import numpy as np

from labelforge.engine.video_dataset import FrameDatasetWriter


def _rgb(w=16, h=12):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_write_image_uses_six_digit_index(tmp_path):
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    path = writer.write_image(0, _rgb())
    assert os.path.basename(path) == "clip_000001.jpg"


def test_write_image_records_the_frame_size(tmp_path):
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    writer.write_image(0, _rgb(40, 30))
    assert (writer.width, writer.height) == (40, 30)


def test_finalize_writes_labels_classes_and_manifest(tmp_path):
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    writer.write_image(0, _rgb())
    frames = [(0, [{"x": 1, "y": 2, "width": 3, "height": 4,
                    "label": "car", "track_id": 5, "score": 0.8}])]
    writer.finalize(frames, class_names=["car"], snapshot={"conf": 0.25},
                    completed=True)

    label_path = tmp_path / "images" / "clip_000001.json"
    assert label_path.exists()
    data = json.loads(label_path.read_text(encoding="utf-8"))
    shape = data["shapes"][0]
    assert shape["label"] == "car"
    assert shape["flags"]["track_id"] == 5
    assert shape["flags"]["inference"] is True
    assert shape["score"] == 0.8
    assert shape["shape_type"] == "rectangle"
    assert data["imageWidth"] == 16
    assert data["imageHeight"] == 12
    assert data["imagePath"] == "clip_000001.jpg"

    classes = (tmp_path / "classes.txt").read_text(encoding="utf-8")
    # 图片与标注同层，classes.txt 与清单留在根目录
    assert (tmp_path / "images" / "clip_000001.jpg").exists()
    assert (tmp_path / "images" / "clip_000001.json").exists()
    assert classes.split() == ["car"]
    manifest = json.loads(
        (tmp_path / "video_inference_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is True
    assert manifest["frame_count"] == 1
    assert manifest["image_count"] == 1
    assert manifest["snapshot"]["conf"] == 0.25


def test_frames_without_boxes_still_get_a_label_file(tmp_path):
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    writer.write_image(0, _rgb())
    writer.finalize([(0, [])], class_names=[], snapshot={}, completed=True)
    data = json.loads(
        (tmp_path / "images" / "clip_000001.json").read_text(encoding="utf-8"))
    assert data["shapes"] == []


def test_incomplete_run_is_marked_in_the_manifest(tmp_path):
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    writer.write_image(0, _rgb())
    writer.finalize([(0, [])], class_names=[], snapshot={}, completed=False)
    manifest = json.loads(
        (tmp_path / "video_inference_manifest.json").read_text(encoding="utf-8"))
    assert manifest["completed"] is False


def test_image_name_is_sanitised(tmp_path):
    # 目录部分先被 basename 去掉（调用方传的是视频路径），剩下的非法字符再净化
    writer = FrameDatasetWriter(str(tmp_path), "a/b:c")
    path = writer.write_image(0, _rgb())
    name = os.path.basename(path)
    assert "/" not in name and ":" not in name
    assert name == "b_c_000001.jpg"


def test_video_path_becomes_a_plain_stem(tmp_path):
    video = f"{tmp_path}{os.sep}clips{os.sep}video:clip.mp4"
    writer = FrameDatasetWriter(str(tmp_path / "out"), video)
    path = writer.write_image(0, _rgb())
    assert os.path.basename(path) == "video_clip_000001.jpg"


def test_write_image_reports_disk_failure(tmp_path, monkeypatch):
    """imwrite 失败必须报错：否则会留下有标注、没图片的半套数据集。"""
    from labelforge.engine import video_dataset

    monkeypatch.setattr(video_dataset.cv2, "imwrite", lambda *a, **kw: False)
    writer = FrameDatasetWriter(str(tmp_path), "clip")

    try:
        writer.write_image(0, _rgb())
    except OSError as exc:
        assert "逐帧图片写入失败" in str(exc)
    else:
        raise AssertionError("imwrite 返回 False 时必须报错")


def test_existing_dataset_directory_gets_a_suffix(tmp_path):
    """目录非空时换一个名字，绝不覆盖上次的导出。"""
    first = FrameDatasetWriter(str(tmp_path / "ds"), "clip")
    first.write_image(0, _rgb())
    second = FrameDatasetWriter(str(tmp_path / "ds"), "clip")

    assert second.root != first.root
    assert second.root.endswith("ds_2")


def test_classes_txt_only_lists_used_labels(tmp_path):
    """只写这批帧里出现的类别，不把模型全部类别抄进来。"""
    writer = FrameDatasetWriter(str(tmp_path), "clip")
    writer.write_image(0, _rgb())
    writer.finalize([(0, [{"x": 0, "y": 0, "width": 2, "height": 2,
                           "label": "car"}])],
                    class_names=["car", "bus", "person"], snapshot={},
                    completed=True)

    classes = (tmp_path / "classes.txt").read_text(encoding="utf-8")
    assert classes.split() == ["car"]


def test_incomplete_marker_is_removed_after_finalize(tmp_path):
    """收尾前留标记，写完删掉：进程被强杀时能看出数据没写完。"""
    from labelforge.engine.video_dataset import MARKER_NAME

    writer = FrameDatasetWriter(str(tmp_path), "clip")
    marker = tmp_path / MARKER_NAME
    assert marker.exists(), "开始写之前就要有标记"

    writer.write_image(0, _rgb())
    writer.finalize([(0, [])], class_names=[], snapshot={}, completed=True)
    assert not marker.exists()
