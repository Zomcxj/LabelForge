"""数据集分组引擎的回归测试。

覆盖：类别画像统计、层次凝聚分组、推荐组数、导出目录结构与 sidecar 裁剪。
"""
import json
import os

import pytest

from labelforge.engine.dataset_grouper import (
    GroupAnalysis, LabelProfile, analyze_dataset, export_groups,
    group_labels, recommend_k, write_classes_txt, write_dataset_yaml,
)


def _write_sample(folder, name, shapes):
    """造一张图 + 它的 sidecar。"""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "wb") as fh:
        fh.write(b"\xff\xd8\xff\xe0")  # 只要文件名像图片即可
    with open(os.path.join(folder, os.path.splitext(name)[0] + ".json"),
              "w", encoding="utf-8") as fh:
        json.dump({"version": "5.3.1", "flags": {}, "shapes": shapes,
                   "imagePath": name, "imageData": None,
                   "imageHeight": 100, "imageWidth": 100}, fh)


def _rect(label, x, y, w, h):
    return {
        "label": label,
        "shape_type": "rectangle",
        "points": [[x, y], [x + w, y + h]],
        "group_id": None,
        "flags": {},
    }


# ---------- 画像统计 ----------

def test_analyze_dataset_counts_coverage_and_median_area(tmp_path):
    folder = str(tmp_path)
    _write_sample(folder, "a.jpg", [_rect("cat", 0, 0, 10, 10),
                                    _rect("cat", 0, 0, 20, 20)])
    _write_sample(folder, "b.jpg", [_rect("cat", 0, 0, 30, 30)])
    _write_sample(folder, "c.jpg", [_rect("dog", 0, 0, 5, 5)])

    analysis = analyze_dataset(folder)
    assert analysis is not None
    assert analysis.images == ["a.jpg", "b.jpg", "c.jpg"]

    cat = analysis.profiles["cat"]
    assert cat.coverage == 2          # 出现在两张图上
    assert cat.instances == 3         # 一共三个框
    assert cat.median_area == 400.0   # 面积 100/400/900 的中位数
    assert analysis.profiles["dog"].coverage == 1

    # 主类别 = 出现图片数最少者
    assert analysis.primary_label["c.jpg"] == "dog"


def test_analyze_dataset_ignores_broken_sidecar(tmp_path):
    folder = str(tmp_path)
    _write_sample(folder, "good.jpg", [_rect("cat", 0, 0, 10, 10)])
    with open(os.path.join(folder, "bad.jpg"), "wb") as fh:
        fh.write(b"\xff\xd8")
    with open(os.path.join(folder, "bad.json"), "w", encoding="utf-8") as fh:
        fh.write("{ this is not json")

    analysis = analyze_dataset(folder)
    assert analysis is not None
    # 坏文件只是被跳过，不影响其余统计
    assert analysis.profiles["cat"].coverage == 1


def test_analyze_dataset_returns_none_for_missing_folder(tmp_path):
    assert analyze_dataset(str(tmp_path / "nope")) is None


def test_polygon_shapes_do_not_pollute_area_statistics(tmp_path):
    folder = str(tmp_path)
    _write_sample(folder, "a.jpg", [
        _rect("cat", 0, 0, 10, 10),
        {"label": "cat", "shape_type": "polygon",
         "points": [[0, 0], [100, 0], [100, 100]], "flags": {}},
    ])
    analysis = analyze_dataset(folder)
    profile = analysis.profiles["cat"]
    # 多边形不计入面积统计（否则同一类别的画像随标注方式飘）
    assert profile.instances == 1
    assert profile.median_area == 100.0


# ---------- 分组 ----------

def _synthetic_analysis(labels, coverages, areas):
    analysis = GroupAnalysis(folder=".", images=["x.jpg"])
    for label, coverage, area in zip(labels, coverages, areas):
        analysis.profiles[label] = LabelProfile(
            label=label, coverage=coverage, instances=coverage,
            median_area=area, area_iqr=0.0,
        )
        analysis.image_labels["x.jpg"] = analysis.image_labels.get("x.jpg", set())
        analysis.image_labels["x.jpg"].add(label)
    return analysis


def test_group_labels_separates_distant_labels():
    analysis = _synthetic_analysis(
        ["big", "small", "tiny"],
        [100, 100, 100],
        [100000.0, 100.0, 1.0],
    )
    groups = group_labels(analysis, 2)
    assert len(groups) == 2
    # 每个类别只能出现在一组里
    seen = [label for group in groups for label in group]
    assert sorted(seen) == ["big", "small", "tiny"]
    assert len(seen) == len(set(seen))


def test_group_labels_k_one_keeps_everything_together():
    analysis = _synthetic_analysis(["a", "b", "c"], [10, 10, 10], [1.0, 2.0, 3.0])
    groups = group_labels(analysis, 1)
    assert len(groups) == 1
    assert sorted(groups[0]) == ["a", "b", "c"]


def test_group_labels_is_deterministic():
    analysis = _synthetic_analysis(
        ["a", "b", "c", "d"], [50, 50, 500, 500], [10.0, 12.0, 900.0, 950.0],
    )
    first = group_labels(analysis, 2)
    second = group_labels(analysis, 2)
    assert first == second


def test_recommend_k_returns_score_pairs():
    analysis = _synthetic_analysis(
        ["a", "b", "c", "d"], [50, 50, 500, 500], [10.0, 12.0, 900.0, 950.0],
    )
    best_k, scores = recommend_k(analysis)
    assert best_k is not None
    assert 1 <= best_k <= 3
    assert scores
    for k, (sil, inertia) in scores.items():
        assert isinstance(k, int)
        assert -1.0 <= sil <= 1.0
        assert inertia >= 0.0


def test_recommend_k_needs_at_least_two_labels():
    analysis = _synthetic_analysis(["only"], [10], [1.0])
    best_k, scores = recommend_k(analysis)
    assert best_k is None
    assert scores == {}


def test_recommend_k_does_not_always_pick_the_maximum():
    """推荐组数不能恒等于标签数（那说明评分退化成了「越大越好」）。"""
    analysis = _synthetic_analysis(
        [f"c{i}" for i in range(8)],
        [100] * 4 + [5] * 4,
        [10.0] * 4 + [5000.0] * 4,
    )
    best_k, _scores = recommend_k(analysis, max_k=7)
    assert best_k is not None
    assert best_k < 7


# ---------- 导出 ----------

def test_export_groups_writes_train_val_classes_and_yaml(tmp_path):
    folder = str(tmp_path)
    for i in range(4):
        _write_sample(folder, f"img{i}.jpg", [_rect("cat", 0, 0, 10, 10),
                                              _rect("dog", 0, 0, 20, 20)])
    analysis = analyze_dataset(folder)
    groups = [["cat"], ["dog"]]

    result = export_groups(folder, groups, analysis, 0.75)
    assert result is not None
    assert result["k"] == 2

    for gi in range(2):
        assert os.path.isfile(os.path.join(folder, f"g{gi}", "classes.txt"))
        assert os.path.isfile(os.path.join(folder, f"{os.path.basename(folder)}_g{gi}.yaml"))
        assert os.path.isdir(os.path.join(folder, f"g{gi}", "images", "train"))

    with open(os.path.join(folder, "g0", "classes.txt"), encoding="utf-8") as fh:
        assert fh.read().strip() == "cat"

    with open(os.path.join(folder, f"{os.path.basename(folder)}_g0.yaml"),
              encoding="utf-8") as fh:
        text = fh.read()
    assert "names:" in text
    assert "0: cat" in text


def test_export_groups_trims_sidecar_to_group_labels(tmp_path):
    folder = str(tmp_path)
    _write_sample(folder, "img0.jpg", [_rect("cat", 0, 0, 10, 10),
                                        _rect("dog", 0, 0, 20, 20)])
    analysis = analyze_dataset(folder)
    export_groups(folder, [["cat"], ["dog"]], analysis, 0.8)

    with open(os.path.join(folder, "g0", "labels", "train", "img0.json"),
              encoding="utf-8") as fh:
        payload = json.load(fh)
    labels = {s["label"] for s in payload["shapes"]}
    assert labels == {"cat"}


def test_generated_yaml_carries_no_foreign_license(tmp_path):
    """生成的 yaml 描述用户自己的数据，不该盖上别人的许可证帽子。"""
    out = tmp_path / "dataset.yaml"
    write_dataset_yaml(str(out), str(tmp_path), "images/train", "images/val",
                       ["cat", "dog"])
    text = out.read_text(encoding="utf-8")
    assert "AGPL" not in text
    assert "Ultralytics" not in text
    assert "path:" in text and "0: cat" in text


def test_write_classes_txt_is_sorted(tmp_path):
    out = tmp_path / "classes.txt"
    write_classes_txt(str(out), ["dog", "cat"])
    assert out.read_text(encoding="utf-8").split() == ["cat", "dog"]
