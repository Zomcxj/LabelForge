"""
数据集分类引擎 - 按 [出现频次 + HBB框大小] 二维KMeans聚类分割

流程:
1. 遍历所有标注，统计每个标签的出现图片数(频次) 和 平均HBB面积
2. 每张图取全局出现次数最少的标签为主标签
3. 对每个标签构建二维特征 [log1p(频次), log1p(平均面积)]，标准化后做KMeans
4. 用轮廓系数(Silhouette)评估并推荐最优K
5. 按输入比例随机拆分train/val（主标签为准，<20张全进train）
6. train/val中，根据图片所有标签的聚类归属分发到对应组
7. 输出到输入文件夹内，生成各组的classes.txt和yaml配置文件
"""
import json
import os
import random
import shutil

import numpy as np


def _whiten(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-10] = 1.0
    return (X - mean) / std


def _kmeans(X, k, n_iter=20, tol=1e-4, rng=None):
    rng = rng or np.random.default_rng()
    n, d = X.shape
    centroids = np.empty((k, d), dtype=X.dtype)
    idx0 = int(rng.integers(0, n))
    centroids[0] = X[idx0]
    for i in range(1, k):
        dists = np.min(np.sum((X[:, None, :] - centroids[:i][None, :, :]) ** 2, axis=2), axis=1)
        probs = dists / (dists.sum() + 1e-12)
        cum = np.cumsum(probs)
        r = rng.random()
        pick = int(np.searchsorted(cum, r))
        if pick >= n:
            pick = int(rng.integers(0, n))
        centroids[i] = X[pick]
    for _ in range(n_iter):
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        codes = np.argmin(dists, axis=1)
        new_cent = np.empty_like(centroids)
        for i in range(k):
            mask = codes == i
            if np.any(mask):
                new_cent[i] = X[mask].mean(axis=0)
            else:
                new_cent[i] = centroids[i]
        shift = np.linalg.norm(new_cent - centroids)
        centroids = new_cent
        if shift < tol:
            break
    return centroids, float(np.mean(np.min(dists, axis=1)))


def _vq(X, centroids):
    dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    codes = np.argmin(dists, axis=1)
    min_dists = np.sqrt(np.min(dists, axis=1))
    return codes, min_dists


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def load_json_labels_and_boxes(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = set()
        boxes = {}
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if not label:
                continue
            labels.add(label)
            if shape.get('shape_type') == 'rectangle':
                pts = shape.get('points', [])
                if len(pts) == 2:
                    w = abs(pts[1][0] - pts[0][0])
                    h = abs(pts[1][1] - pts[0][1])
                    area = w * h
                    boxes.setdefault(label, []).append(area)
        return labels, boxes
    except Exception:
        return set(), {}


def analyze_dataset(image_dir, progress_callback=None, is_interrupted=None):
    image_extensions = get_image_extensions()
    all_files = os.listdir(image_dir)

    images = []
    for file in all_files:
        full_path = os.path.join(image_dir, file)
        if os.path.isdir(full_path):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            images.append(file)

    label_image_count = {}
    label_total_area = {}
    label_instance_count = {}
    image_labels = {}
    image_primary_label = {}

    total = len(images)
    for idx, image_name in enumerate(images):
        if is_interrupted and is_interrupted():
            return None
        if progress_callback:
            progress_callback(idx + 1, total, None)
        json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(json_path):
            continue
        labels, boxes = load_json_labels_and_boxes(json_path)
        if not labels:
            continue

        image_labels[image_name] = labels

        for label in labels:
            label_image_count[label] = label_image_count.get(label, 0) + 1
            if label in boxes:
                label_total_area[label] = label_total_area.get(label, 0) + sum(boxes[label])
                label_instance_count[label] = label_instance_count.get(label, 0) + len(boxes[label])

    for image_name, labels in image_labels.items():
        rarest = min(labels, key=lambda l: label_image_count.get(l, 0))
        image_primary_label[image_name] = rarest

    label_mean_area = {}
    for label in label_total_area:
        if label_instance_count.get(label, 0) > 0:
            label_mean_area[label] = label_total_area[label] / label_instance_count[label]

    if label_mean_area:
        fallback_area = float(np.median(list(label_mean_area.values())))
    else:
        fallback_area = 1.0

    mean_areas = {}
    for label in label_image_count:
        if label in label_mean_area:
            mean_areas[label] = label_mean_area[label]
        else:
            mean_areas[label] = fallback_area

    classified_images = {}
    for image_name, label in image_primary_label.items():
        classified_images.setdefault(label, []).append(image_name)

    return (images, label_image_count, mean_areas, image_labels,
            image_primary_label, classified_images)


def build_features(label_image_count, mean_areas):
    labels = sorted(label_image_count.keys())
    freq = np.array([label_image_count[l] for l in labels]).reshape(-1, 1)
    area = np.array([mean_areas[l] for l in labels]).reshape(-1, 1)
    X = np.hstack([np.log1p(freq), np.log1p(area)])
    X_scaled = _whiten(X)
    return labels, X_scaled, X


def recommend_k(X, max_k=10, progress_callback=None, is_interrupted=None):
    n = X.shape[0]
    if n < 2:
        return 1, {}
    k_max = min(max_k, n - 1)
    k_range = list(range(2, k_max + 1))
    scores = {}
    total = len(k_range)
    for i, k in enumerate(k_range):
        if is_interrupted and is_interrupted():
            return None, {}
        centroids, distortion = _kmeans(X, k)
        # distortion 越小聚类越好，用负值模拟得分（越大越好）
        scores[k] = (-distortion, distortion)
        if progress_callback:
            progress_callback(i + 1, total, k)
    if not scores:
        return 1, {}
    best_k = max(scores, key=lambda k: scores[k][0])
    return best_k, scores


def cluster_labels(X, k, labels):
    centroids, _ = _kmeans(X, k)
    codes, _ = _vq(X, centroids)
    cluster_of_label = {label: int(cid) for label, cid in zip(labels, codes)}

    # 过滤空 centroid（某些质心附近没有被分配的数据点）
    used = sorted(set(codes))
    n_groups = len(used)
    if n_groups < centroids.shape[0]:
        old_to_new = {old: new for new, old in enumerate(used)}
        centroids = centroids[used]
        cluster_of_label = {label: old_to_new[cid] for label, cid in cluster_of_label.items() if cid in old_to_new}
        codes = np.array([old_to_new[c] for c in codes])

    order = sorted(range(n_groups), key=lambda c: centroids[c][1], reverse=True)
    new_id = {old: new for new, old in enumerate(order)}

    group_labels = [set() for _ in range(n_groups)]
    for label, old_cid in cluster_of_label.items():
        group_labels[new_id[old_cid]].add(label)

    return group_labels, centroids, cluster_of_label


def split_by_class(classified_images, train_ratio, min_threshold=20):
    train_images = []
    val_images = []

    for label, image_list in classified_images.items():
        random.shuffle(image_list)
        count = len(image_list)
        if count < min_threshold:
            if count == 1:
                train_images.extend(image_list)
            else:
                train_images.extend(image_list[:count - 1])
                val_images.append(image_list[count - 1])
        else:
            split_idx = int(count * train_ratio)
            train_images.extend(image_list[:split_idx])
            val_images.extend(image_list[split_idx:])

    return train_images, val_images


def copy_with_trimmed_annotation(src_dir, image_name, img_dest_dir, lbl_dest_dir, allowed_labels):
    """复制图片和标签到对应目录，只保留 allowed_labels 中的标签。"""
    image_src = os.path.join(src_dir, image_name)
    if os.path.exists(image_src):
        shutil.copy2(image_src, img_dest_dir)

    json_src = os.path.join(src_dir, os.path.splitext(image_name)[0] + '.json')
    if not os.path.exists(json_src):
        return

    with open(json_src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['shapes'] = [
        shape for shape in data.get('shapes', [])
        if shape.get('label', '') in allowed_labels
    ]

    json_dst = os.path.join(lbl_dest_dir, os.path.splitext(image_name)[0] + '.json')
    with open(json_dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_classes_txt(path, labels):
    with open(path, 'w', encoding='utf-8') as f:
        for label in sorted(labels):
            f.write(label + '\n')


def write_yaml(path, dataset_root, train_rel, val_rel, labels):
    sorted_labels = sorted(labels)
    lines = [
        f"# Ultralytics AGPL-3.0 License",
        f"#",
        f"# Train/val sets",
        f"path: {dataset_root}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"test: ",
        f"",
        f"# Classes",
        f"names:",
    ]
    for i, label in enumerate(sorted_labels):
        lines.append(f"  {i}: {label}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def classify_and_split(image_dir, train_ratio, k, analysis=None,
                       progress_callback=None, is_interrupted=None, log_fn=None):
    """完整分类 + 分割流水线。

    analysis 可传入已分析结果（避免重复分析），格式同 analyze_dataset 返回值。
    """
    if analysis is None:
        analysis = analyze_dataset(image_dir, progress_callback, is_interrupted)
        if analysis is None:
            return None
    (images, label_image_count, mean_areas, image_labels,
     image_primary_label, classified_images) = analysis

    if not images:
        return None

    labels, X_scaled, X = build_features(label_image_count, mean_areas)
    if len(labels) < 2:
        return None

    group_labels, _centroids, cluster_of_label = cluster_labels(X_scaled, k, labels)
    n_groups = len(group_labels)

    train_images, val_images = split_by_class(classified_images, train_ratio)

    group_dirs = []
    for gi in range(n_groups):
        train_img_dir = os.path.join(image_dir, f'g{gi}', 'images', 'train')
        val_img_dir = os.path.join(image_dir, f'g{gi}', 'images', 'val')
        train_lbl_dir = os.path.join(image_dir, f'g{gi}', 'labels', 'train')
        val_lbl_dir = os.path.join(image_dir, f'g{gi}', 'labels', 'val')
        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)
        group_dirs.append((train_img_dir, val_img_dir))

    group_train_count = [0] * n_groups
    group_val_count = [0] * n_groups

    copy_tasks = []
    for gi in range(n_groups):
        for image_name in train_images:
            if image_labels.get(image_name, set()) & group_labels[gi]:
                copy_tasks.append((image_name, gi, 'train'))
        for image_name in val_images:
            if image_labels.get(image_name, set()) & group_labels[gi]:
                copy_tasks.append((image_name, gi, 'val'))
    total = len(copy_tasks)
    for idx, (image_name, gi, split_kind) in enumerate(copy_tasks):
        if is_interrupted and is_interrupted():
            return None
        img_dest = group_dirs[gi][0] if split_kind == 'train' else group_dirs[gi][1]
        lbl_dest = os.path.join(image_dir, f'g{gi}', 'labels', split_kind)
        if split_kind == 'train':
            group_train_count[gi] += 1
        else:
            group_val_count[gi] += 1
        copy_with_trimmed_annotation(image_dir, image_name, img_dest, lbl_dest, group_labels[gi])
        if progress_callback:
            progress_callback(idx + 1, total, None)

    folder_name = os.path.basename(os.path.normpath(image_dir))
    for gi in range(n_groups):
        write_classes_txt(os.path.join(image_dir, f'g{gi}', 'classes.txt'), group_labels[gi])
        write_yaml(os.path.join(image_dir, f'{folder_name}_g{gi}.yaml'),
                   image_dir, f'g{gi}/train', f'g{gi}/val', group_labels[gi])

    if log_fn:
        log_fn(f"train: {len(train_images)}, val: {len(val_images)}")

    return {
        'images': images,
        'label_image_count': label_image_count,
        'mean_areas': mean_areas,
        'image_labels': image_labels,
        'image_primary_label': image_primary_label,
        'classified_images': classified_images,
        'group_labels': group_labels,
        'k': n_groups,
        'train_images': train_images,
        'val_images': val_images,
        'group_train_count': group_train_count,
        'group_val_count': group_val_count,
        'output_dir': image_dir,
    }
