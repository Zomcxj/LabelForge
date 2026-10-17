"""数据集分类引擎 + GUI 接线测试。"""
import json
import os

from labelforge.engine.dataset_classifier import (
    analyze_dataset, build_features, classify_and_split, cluster_labels,
    get_image_extensions, recommend_k, split_by_class, write_yaml,
)


def _make_dataset(root, images):
    """images: {name: {label: [(x1,y1,x2,y2), ...]}}；纯文本假图片即可。"""
    for name, shapes in images.items():
        if shapes is None:
            with open(os.path.join(root, name), 'w', encoding='utf-8') as f:
                f.write('dummy')
            continue
        with open(os.path.join(root, name), 'w', encoding='utf-8') as f:
            f.write('dummy')
        data = {'shapes': []}
        for label, boxes in shapes.items():
            for (x1, y1, x2, y2) in boxes:
                data['shapes'].append({
                    'label': label,
                    'shape_type': 'rectangle',
                    'points': [[x1, y1], [x2, y2]],
                })
        stem = os.path.splitext(name)[0]
        with open(os.path.join(root, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f)


def test_image_extensions():
    assert '.jpg' in get_image_extensions()
    assert '.webp' in get_image_extensions()


def test_analyze_dataset_counts(tmp_path):
    _make_dataset(tmp_path, {
        'a.jpg': {'cat': [(0, 0, 10, 10)]},
        'b.jpg': {'dog': [(0, 0, 20, 20)]},
        'c.jpg': {'cat': [(0, 0, 5, 5)], 'bird': [(1, 1, 11, 11)]},
        'noise.txt': None,
    })
    images, label_image_count, mean_areas, image_labels, image_primary_label, classified_images = analyze_dataset(str(tmp_path))
    assert len(images) == 3
    assert label_image_count['cat'] == 2
    assert label_image_count['dog'] == 1
    assert label_image_count['bird'] == 1
    assert mean_areas['cat'] == (100 + 25) / 2
    # 主标签取全局出现最少的标签
    assert image_primary_label['a.jpg'] == 'cat'
    assert image_primary_label['b.jpg'] == 'dog'
    assert image_primary_label['c.jpg'] == 'bird'
    assert sorted(classified_images['cat']) == ['a.jpg']


def test_recommend_and_cluster(tmp_path):
    shapes = {}
    for i in range(10):
        label = 'tiny' if i < 4 else ('small' if i < 6 else ('medium' if i < 8 else 'big'))
        if label == 'tiny':
            box = (0, 0, 8, 8)
        elif label == 'small':
            box = (0, 0, 20, 20)
        elif label == 'medium':
            box = (0, 0, 40, 40)
        else:
            box = (0, 0, 100, 100)
        shapes[f'{i:02d}.jpg'] = {label: [box]}
    _make_dataset(tmp_path, shapes)
    analysis = analyze_dataset(str(tmp_path))
    images, label_image_count, mean_areas, image_labels, image_primary_label, classified_images = analysis
    labels, X_scaled, X = build_features(label_image_count, mean_areas)
    assert len(labels) == 4
    best_k, scores = recommend_k(X_scaled)
    assert best_k >= 2
    assert 2 in scores
    group_labels, kmeans, cluster_of_label = cluster_labels(X_scaled, best_k, labels)
    assert len(group_labels) == best_k
    assert sum(len(g) for g in group_labels) == len(labels)


def test_split_by_class_min_threshold(tmp_path):
    _make_dataset(tmp_path, {
        f'cat{i}.jpg': {'cat': [(0, 0, 8, 8)]} for i in range(5)
    })
    analysis = analyze_dataset(str(tmp_path))
    classified_images = analysis[5]
    train, val = split_by_class(classified_images, 0.8)
    assert len(train) == 4
    assert len(val) == 1


def test_classify_and_split_output(tmp_path):
    _make_dataset(tmp_path, {
        f'{i:02d}.jpg': {('big' if i % 2 == 0 else 'small'): [(0, 0, 40, 40) if i % 2 == 0 else (0, 0, 8, 8)]}
        for i in range(10)
    })
    analysis = analyze_dataset(str(tmp_path))
    result = classify_and_split(str(tmp_path), 0.8, 2, analysis=analysis)
    assert result is not None
    assert result['k'] == 2
    assert len(result['train_images']) + len(result['val_images']) == 10
    for gi in range(2):
        train_img_dir = os.path.join(str(tmp_path), f'g{gi}', 'images', 'train')
        val_img_dir = os.path.join(str(tmp_path), f'g{gi}', 'images', 'val')
        train_lbl_dir = os.path.join(str(tmp_path), f'g{gi}', 'labels', 'train')
        val_lbl_dir = os.path.join(str(tmp_path), f'g{gi}', 'labels', 'val')
        assert os.path.isdir(train_img_dir)
        assert os.path.isdir(val_img_dir)
        assert os.path.isdir(train_lbl_dir)
        assert os.path.isdir(val_lbl_dir)
        assert os.path.isfile(os.path.join(str(tmp_path), f'g{gi}', 'classes.txt'))
    yaml_files = [f for f in os.listdir(str(tmp_path)) if f.endswith('.yaml')]
    assert len(yaml_files) == 2


def test_classify_and_split_progress_callback(tmp_path):
    _make_dataset(tmp_path, {
        f'{i:02d}.jpg': {('big' if i % 2 == 0 else 'small'): [(0, 0, 40, 40) if i % 2 == 0 else (0, 0, 8, 8)]}
        for i in range(6)
    })
    analysis = analyze_dataset(str(tmp_path))
    steps = []
    classify_and_split(
        str(tmp_path), 0.8, 2, analysis=analysis,
        progress_callback=lambda cur, total, msg: steps.append((cur, total)),
    )
    assert steps


def test_dataset_classifier_folder_defaults_to_open_folder():
    from labelforge.ui.main_window import ImageEditor
    editor = ImageEditor.__new__(ImageEditor)
    editor.background_images = [os.path.join('D:', 'data', 'img1.png')]
    assert editor._get_dataset_classifier_folder() == os.path.join('D:', 'data')
    editor.background_images = []
    assert os.path.isdir(editor._get_dataset_classifier_folder())


def test_dataset_classifier_dialog_defaults(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = f'''
import os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from labelforge.ui.dataset_classifier_dialog import DatasetClassifierDialog

app = QApplication.instance() or QApplication([])
folder = {str(tmp_path)!r}
dlg = DatasetClassifierDialog(None, default_folder=folder)
assert os.path.normpath(dlg._folder_edit.text()) == os.path.normpath(folder)
assert abs(dlg._train_spin.value() - 0.8) < 1e-9
assert "val: 20%" in dlg._val_label.text()
assert dlg._analyze_btn.text() == "开始分析"
dlg2 = DatasetClassifierDialog(None, default_folder="")
assert dlg2._folder_edit.text()
print("OK")

# Qt 子进程退出阶段析构顺序不定，偶发崩溃/挂死；断言通过后 flush + os._exit(0)
# 绕开析构（os._exit 不冲刷缓冲，必须先 flush，否则标记丢失会误报失败）。
import os, sys
sys.stdout.flush()
os._exit(0)
'''
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(root)}

    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr


def test_generated_dataset_yaml_claims_no_foreign_license(tmp_path):
    """生成的 dataset.yaml 是用户自己的数据，不应盖上别人的许可证帽子。"""
    out = tmp_path / 'dataset.yaml'
    write_yaml(str(out), str(tmp_path), 'images/train', 'images/val', ['cat', 'dog'])
    text = out.read_text(encoding='utf-8')
    assert 'AGPL' not in text
    assert 'Ultralytics' not in text
    assert "path:" in text and "0: cat" in text
