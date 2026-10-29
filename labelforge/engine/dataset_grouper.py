"""数据集标签分组：按类别统计特征做层次凝聚聚类（自研实现）。

用途：把一个标注数据集按「类别之间的相似度」自动分成若干组，每组导出成
独立可训练的数据集（train/val + classes.txt + yaml）。

与其它模块的关系：
- 逐框几何的采集复用 ``dataset_health.collect_shape_geometry``（同一份
  解析口径，不另写一套 sidecar 解析）。
- 特征与分组逻辑全在本模块内，纯函数、无 Qt 依赖，可在无 GUI 环境下测。

特征设计（每个类别三项）：
1. ``coverage``  —— 出现在多少张图上（去重后的图片数），取对数压缩量纲。
2. ``log_area`` —— 该类别所有框面积的中位数（对数）。中位数比均值抗离群。
3. ``spread``   —— 该类别框面积的对数离散度（四分位距），刻画「大小是否稳定」。

三项都做 log1p/标准化后再算距离：类别数多时原始频次与面积的量级差几个
数量级，不压一下欧氏距离会被频次单独主导。

聚类用**层次凝聚（Agglomerative）**，单链距离：
- 不用 k-means++：它要随机初值，同一份数据每次聚类结果可能不同（对「把
  数据集切成几份」这种要可复现的操作是硬伤）。
- 单链的链式效应在这里正合适：我们要的是「把相近的类别串成一串」，
  而不是把空间均匀切块。
- K 不由用户事先给定：用**树状图剪枝**按距离跳变自动选，得到的 K 与
  数据本身的层次结构一致；用户仍可在界面上手动改。

相似度评估用**轮廓系数**的标准定义（点到点的距离，不用到簇心的距离）：
- 到簇心的写法在簇退化成单点时 a(i)→0、s(i)→1，会系统性偏向最大的 K。
- 点到点的写法里，簇一旦过分裂，最近异簇距离 b(i) 跟着下降、s(i) 跟着
  下跌，过分裂会被惩罚掉。
"""
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: 支持的图片扩展名（小写，含点）
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".tif", ".webp")

#: 每一组至少要有这么多张图才单列一组；不足的并入最相近的一组
MIN_GROUP_IMAGES = 20

#: 剪枝选 K 时，距离跳变的相对阈值：相邻合并距离超过中位跳变的这个倍数
#: 才认为「这里该切一刀」。
PRUNE_JUMP_RATIO = 1.6


class DatasetGroupError(RuntimeError):
    """分组过程中的应用层错误，消息可直接展示给用户。"""


@dataclass
class LabelProfile:
    """一个类别的统计画像。"""

    label: str
    #: 出现在多少张图上
    coverage: int = 0
    #: 该类别的框总数
    instances: int = 0
    #: 框面积中位数（像素²）
    median_area: float = 0.0
    #: 框面积的四分位距（像素²），0 表示所有框一样大
    area_iqr: float = 0.0
    #: 该类别出现在哪些图片上（图片文件名）
    images: set = field(default_factory=set)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "coverage": self.coverage,
            "instances": self.instances,
            "median_area": self.median_area,
            "area_iqr": self.area_iqr,
        }


@dataclass
class GroupAnalysis:
    """一次数据集分组分析的完整结果。"""

    #: 数据集目录
    folder: str = ""
    #: 数据集里的图片文件名（排序后）
    images: List[str] = field(default_factory=list)
    #: 类别 -> 画像
    profiles: Dict[str, LabelProfile] = field(default_factory=dict)
    #: 每张图上出现的类别集合
    image_labels: Dict[str, set] = field(default_factory=dict)
    #: 每张图的主类别（出现图片数最少者；并列时取字典序最小）
    primary_label: Dict[str, str] = field(default_factory=dict)

    @property
    def labels(self) -> List[str]:
        """按名称排序的类别列表。"""
        return sorted(self.profiles)

    def feature_matrix(self) -> List[List[float]]:
        """按 ``labels`` 顺序给出的标准化特征向量。"""
        return [_feature_vector(self.profiles[name]) for name in self.labels]


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: Sequence[float], q: float) -> float:
    """线性插值分位数（q ∈ [0,1]）。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(ordered[low])
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _feature_vector(profile: LabelProfile) -> List[float]:
    """一个类别的三维特征（未标准化）。"""
    return [
        math.log1p(max(0, profile.coverage)),
        math.log1p(max(0.0, profile.median_area)),
        math.log1p(max(0.0, profile.area_iqr)),
    ]


def _standardize(rows: List[List[float]]) -> List[List[float]]:
    """按列做 z-score 标准化；某列全同则该列置 0（不放大噪声）。"""
    if not rows:
        return []
    dim = len(rows[0])
    out = []
    for col in range(dim):
        values = [row[col] for row in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if std < 1e-9:
            out.append([0.0] * len(rows))
        else:
            out.append([(v - mean) / std for v in values])
    return [[out[col][i] for col in range(dim)] for i in range(len(rows))]


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    """欧氏距离。"""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _profile_labels_from_json(json_path: str) -> Tuple[set, Dict[str, List[float]]]:
    """读一份 sidecar，返回 (类别集合, 类别 -> 框面积列表)。

    读不出来就当这张图没有标注（返回空），不抛异常——扫描整个数据集时，
    单张图损坏不该中断整件事。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return set(), {}
    if not isinstance(payload, dict):
        return set(), {}
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        return set(), {}

    labels = set()
    areas: Dict[str, List[float]] = {}
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if not isinstance(label, str) or not label.strip():
            continue
        label = label.strip()
        labels.add(label)
        # 只用矩形框算面积：多边形/旋转框的「面积」口径与训练侧不一致，
        # 混进来会让同一类别的画像在两种标注方式之间飘。
        if (shape.get("shape_type") or "rectangle") != "rectangle":
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue
        try:
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
        except (TypeError, ValueError):
            continue
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > 0:
            areas.setdefault(label, []).append(area)
    return labels, areas


def analyze_dataset(folder: str,
                    progress_callback: Optional[Callable] = None,
                    is_interrupted: Optional[Callable] = None) -> Optional[GroupAnalysis]:
    """扫描数据集目录，统计每个类别的画像。

    ``progress_callback(current, total, message)`` 与
    ``is_interrupted() -> bool`` 与本项目其它后台任务同一套约定。

    目录不可读时返回 None（调用方给用户一句「路径不存在」即可）。
    """
    if not folder or not os.path.isdir(folder):
        return None
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return None

    images = []
    for name in names:
        if os.path.splitext(name)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        if os.path.isfile(os.path.join(folder, name)):
            images.append(name)

    analysis = GroupAnalysis(folder=os.path.abspath(folder), images=images)
    if not images:
        return analysis

    total = len(images)
    for index, name in enumerate(images):
        if is_interrupted and is_interrupted():
            return None
        if progress_callback:
            progress_callback(index + 1, total, None)

        json_path = os.path.join(folder, os.path.splitext(name)[0] + ".json")
        labels, areas = _profile_labels_from_json(json_path)
        if not labels:
            continue
        analysis.image_labels[name] = labels
        for label in labels:
            profile = analysis.profiles.get(label)
            if profile is None:
                profile = analysis.profiles[label] = LabelProfile(label=label)
            profile.coverage += 1
            profile.images.add(name)
            for area in areas.get(label, ()):
                profile.instances += 1
                profile._areas = getattr(profile, "_areas", [])
                profile._areas.append(area)

    for profile in analysis.profiles.values():
        areas = getattr(profile, "_areas", [])
        profile.median_area = _median(areas)
        profile.area_iqr = _quantile(areas, 0.75) - _quantile(areas, 0.25)
        if hasattr(profile, "_areas"):
            del profile._areas

    # 主类别：出现图片数最少者（同频次取字典序最小），保证「稀有类别」
    # 决定这张图属于哪一组——稀有类别正是划分数据集时最该单独成组的。
    for name, labels in analysis.image_labels.items():
        analysis.primary_label[name] = min(
            labels, key=lambda lab: (analysis.profiles[lab].coverage, lab)
        )

    return analysis


def _pairwise_distances(points: List[List[float]]) -> List[List[float]]:
    n = len(points)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _distance(points[i], points[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def _agglomerate(matrix: List[List[float]]) -> Tuple[List[List[int]], List[float]]:
    """单链层次凝聚。

    返回 ``(合并过程, 每次合并的距离)``：合并过程第 i 项是
    ``[簇号...]``，表示第 i 步之后各簇包含的原始下标。

    用「最近邻链」实现（每次找全局最近的一对合并），类别数是几十量级，
    O(n³) 的朴素实现也够快，但实现更短、更好读。
    """
    n = len(matrix)
    if n == 0:
        return [], []

    # clusters[i] = 该簇包含的原始下标；active 里的簇号是当前有效簇
    clusters: List[List[int]] = [[i] for i in range(n)]
    # 簇间距离矩阵，簇号即下标
    size = n
    dist = [row[:] for row in matrix]
    history: List[List[int]] = []
    merge_distances: List[float] = []

    while size > 1:
        best = None
        best_dist = float("inf")
        for i in range(size):
            for j in range(i + 1, size):
                if dist[i][j] < best_dist:
                    best_dist = dist[i][j]
                    best = (i, j)
        if best is None:
            break
        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
        # 新簇 i 到其余簇的距离取两者较小者（单链）
        for k in range(size):
            if k == i or k == j:
                continue
            value = min(dist[i][k], dist[j][k])
            dist[i][k] = value
            dist[k][i] = value
        del dist[j]
        for row in dist:
            del row[j]
        size -= 1
        history.append([list(c) for c in clusters])
        merge_distances.append(best_dist)

    return history, merge_distances


def _choose_k(merge_distances: List[float], max_k: int) -> int:
    """按树状图的距离跳变自动选 K。

    相邻两次合并的距离差得越明显，说明那一刀切得越「该切」。取跳变最大
    的那一处作为切点；跳变都不明显时保守地给 2 组。
    """
    n = len(merge_distances)
    if n == 0:
        return 1
    if n == 1:
        return 1
    jumps = []
    for i in range(1, n):
        prev = merge_distances[i - 1]
        jumps.append(merge_distances[i] - prev)
    if not jumps:
        return 1
    positive = [j for j in jumps if j > 0]
    if not positive:
        return 1
    threshold = (sum(positive) / len(positive)) * PRUNE_JUMP_RATIO
    candidates = [i for i, j in enumerate(jumps, start=1) if j >= threshold]
    # 跳变最大处就是切点。第 i 步合并（i 从 1 起）之后剩 n-i 个簇，
    # 所以该处的簇数 = n - i。
    best_index = max(candidates, key=lambda i: jumps[i - 1]) if candidates else \
        max(range(1, n), key=lambda i: jumps[i - 1])
    k = n - best_index
    return max(1, min(k, max_k))


def _silhouette(points: List[List[float]], groups: List[List[int]]) -> float:
    """平均轮廓系数（标准定义：a/b 都用点到**点**的距离）。

    单点簇按惯例记 0；某点所在簇之外没有别的簇时记 0。
    """
    n = len(points)
    if n < 2 or len(groups) < 2:
        return 0.0
    matrix = _pairwise_distances(points)
    assignment = {}
    for gid, members in enumerate(groups):
        for idx in members:
            assignment[idx] = gid

    total = 0.0
    for i in range(n):
        own = assignment[i]
        same = [j for j in groups[own] if j != i]
        if not same:
            continue
        a = sum(matrix[i][j] for j in same) / len(same)
        b = float("inf")
        for gid, members in enumerate(groups):
            if gid == own or not members:
                continue
            mean = sum(matrix[i][j] for j in members) / len(members)
            b = min(b, mean)
        if b == float("inf"):
            continue
        denom = max(a, b)
        if denom > 1e-12:
            total += (b - a) / denom
    return total / n


def _partition_indices(points: List[List[float]], k: int) -> List[List[int]]:
    """把特征点分成 k 组，返回每组包含的**原始下标**。

    先做层次凝聚得到合并序列，再在「还剩 k 个簇」的那一步切开。
    """
    n = len(points)
    if n == 0:
        return []
    if k <= 1 or n == 1:
        return [list(range(n))]

    history, _merge_distances = _agglomerate(_pairwise_distances(points))
    if not history:
        return [list(range(n))]

    # history[i] 是**第 i+1 次合并之后**的簇（此时剩 n-i-1 个簇）。
    # 要剩 k 个簇，就取 i = n - k - 1。
    index = n - k - 1
    if index < 0:
        index = 0
    clusters = history[index]
    return [sorted(members) for members in clusters]


def _partition(labels: List[str], points: List[List[float]],
               k: int) -> List[List[str]]:
    """把类别分成 k 组，返回每组的类别名列表。"""
    groups = _partition_indices(points, k)
    result = [[labels[i] for i in members] for members in groups]
    # 按「组内类别名的字典序」给组排序，保证输出稳定可复现
    result.sort(key=lambda g: g[0] if g else "")
    return result


def group_labels(analysis: GroupAnalysis, k: int) -> List[List[str]]:
    """按 k 把类别分组，返回每组的类别名列表。"""
    labels = analysis.labels
    if len(labels) < 2:
        return [list(labels)] if labels else []
    rows = _standardize(analysis.feature_matrix())
    return _partition(labels, rows, max(1, int(k)))


def recommend_k(analysis: GroupAnalysis, max_k: int = 10) -> Tuple[Optional[int], Dict[int, Tuple[float, float]]]:
    """给出推荐的分组数与每个候选 K 的评估指数。

    返回 ``(best_k, scores)``；``scores[k] = (silhouette, inertia)``，
    silhouette 越高越好，inertia 越低越好（与界面表格的两列语义一致）。
    """
    labels = analysis.labels
    if len(labels) < 2:
        return None, {}

    rows = _standardize(analysis.feature_matrix())
    if not rows:
        return None, {}

    limit = max(1, min(int(max_k), len(labels) - 1))
    scores: Dict[int, Tuple[float, float]] = {}
    for k in range(1, limit + 1):
        index_groups = _partition_indices(rows, k)
        inertia = _inertia(rows, index_groups)
        scores[k] = (_silhouette(rows, index_groups), inertia)

    # 推荐 K：层次凝聚的树状图距离跳变处（数据的层次结构）+ 轮廓系数
    # 综合。跳变给出「结构上该在这里切」，轮廓给出「切了之后分离度好不好」。
    history, merge_distances = _agglomerate(_pairwise_distances(rows))
    structural = _choose_k(merge_distances, limit) if merge_distances else 1
    best_sil = max(scores, key=lambda k: scores[k][0])
    # 两者接近时取结构建议（它更抗「偶然高分离度」的噪声）；差得远时取轮廓。
    if abs(scores.get(structural, (0.0, 0.0))[0] - scores[best_sil][0]) < 0.05:
        best_k = structural
    else:
        best_k = best_sil
    return best_k, scores


def _inertia(points: List[List[float]], groups: List[List[int]]) -> float:
    """组内平方和：每个点到本组重心的距离平方之和。"""
    total = 0.0
    for members in groups:
        if not members:
            continue
        dim = len(points[members[0]])
        center = [sum(points[i][d] for i in members) / len(members) for d in range(dim)]
        for i in members:
            total += sum((points[i][d] - center[d]) ** 2 for d in range(dim))
    return total


def _split_images(analysis: GroupAnalysis, train_ratio: float,
                  groups: List[List[str]]) -> Tuple[List[str], List[str]]:
    """按主类别把图分成 train / val。

    某组太小时不强行按比例切：先全部进 train，只有当该组有足够多的图
    （``MIN_GROUP_IMAGES``）才留出 val，否则 val 里会出现只有一两张的组，
    训练时方差极大。
    """
    by_primary: Dict[str, List[str]] = {}
    for name in analysis.image_labels:
        primary = analysis.primary_label.get(name)
        if not primary:
            continue
        by_primary.setdefault(primary, []).append(name)

    train: List[str] = []
    val: List[str] = []
    for label, names in by_primary.items():
        names = sorted(names)
        if len(names) < MIN_GROUP_IMAGES:
            train.extend(names)
            continue
        cut = int(len(names) * train_ratio)
        cut = max(1, min(cut, len(names) - 1))
        train.extend(names[:cut])
        val.extend(names[cut:])
    return sorted(train), sorted(val)


def _write_json_subset(src_json: str, dst_json: str, allowed: Sequence[str]) -> bool:
    """把 sidecar 复制到目标目录，只保留指定类别的 shape。

    返回是否真的写出了文件（没有可保留的 shape 时不写空文件）。
    """
    try:
        with open(src_json, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    keep = set(allowed)
    shapes = [s for s in (payload.get("shapes") or [])
              if isinstance(s, dict) and s.get("label") in keep]
    if not shapes:
        return False
    payload["shapes"] = shapes
    os.makedirs(os.path.dirname(dst_json), exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return True


def write_classes_txt(path: str, labels: Sequence[str]) -> None:
    """写类别清单（一行一个类别，排序）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for label in sorted(labels):
            fh.write(label + "\n")


def write_dataset_yaml(path: str, dataset_root: str, train_rel: str, val_rel: str,
                       labels: Sequence[str]) -> None:
    """写 YOLO 训练用的 data.yaml。

    生成的 yaml 描述的是**用户自己的数据集**，不带任何第三方许可证标记
    ——YAML 注释虽不影响解析，但把别人的许可证盖在用户的数据上是错的。
    """
    lines = [
        f"# LabelForge dataset group",
        f"# Train/val sets",
        f"path: {dataset_root}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"test: ",
        f"",
        f"# Classes",
        f"names:",
    ]
    for index, label in enumerate(sorted(labels)):
        lines.append(f"  {index}: {label}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def export_groups(folder: str, groups: List[List[str]], analysis: GroupAnalysis,
                  train_ratio: float,
                  progress_callback: Optional[Callable] = None,
                  is_interrupted: Optional[Callable] = None,
                  log_fn: Optional[Callable] = None) -> Optional[dict]:
    """把每组导出成独立的数据集目录（train/val + classes.txt + yaml）。

    目录结构::

        <folder>/g0/images/train/*.jpg
        <folder>/g0/labels/train/*.json
        <folder>/g0/classes.txt
        <folder>/<folder名>_g0.yaml

    一张图若含多个组的类别，会被复制进它涉及的**每个**组——分组的是
    类别而不是图片，一个画面里的不同类别本来就可能属于不同组。
    """
    if not groups:
        return None

    train_images, val_images = _split_images(analysis, train_ratio, groups)
    split_of: Dict[str, str] = {}
    for name in train_images:
        split_of[name] = "train"
    for name in val_images:
        split_of[name] = "val"

    # 每张图涉及哪些组：按主类别定的组，再补上它身上其它类别所属的组
    group_of_label: Dict[str, int] = {}
    for gi, labels in enumerate(groups):
        for label in labels:
            group_of_label[label] = gi

    tasks: List[Tuple[str, int, str]] = []
    for name in sorted(set(train_images) | set(val_images)):
        split = split_of.get(name)
        if not split:
            continue
        labels = analysis.image_labels.get(name) or set()
        targets = sorted({group_of_label[label] for label in labels
                          if label in group_of_label})
        for gi in targets:
            tasks.append((name, gi, split))

    if not tasks:
        return None

    total = len(tasks)
    group_train_count = [0] * len(groups)
    group_val_count = [0] * len(groups)
    group_labels = [set(g) for g in groups]

    for index, (name, gi, split) in enumerate(tasks):
        if is_interrupted and is_interrupted():
            return None
        if progress_callback:
            progress_callback(index + 1, total, None)

        group_root = os.path.join(folder, f"g{gi}")
        img_dir = os.path.join(group_root, "images", split)
        lbl_dir = os.path.join(group_root, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        src_img = os.path.join(folder, name)
        dst_img = os.path.join(img_dir, name)
        if os.path.isfile(src_img) and not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        stem = os.path.splitext(name)[0]
        src_json = os.path.join(folder, stem + ".json")
        dst_json = os.path.join(lbl_dir, stem + ".json")
        _write_json_subset(src_json, dst_json, sorted(group_labels[gi]))

        if split == "train":
            group_train_count[gi] += 1
        else:
            group_val_count[gi] += 1

    folder_name = os.path.basename(os.path.normpath(folder))
    for gi, labels in enumerate(groups):
        write_classes_txt(os.path.join(folder, f"g{gi}", "classes.txt"), labels)
        write_dataset_yaml(
            os.path.join(folder, f"{folder_name}_g{gi}.yaml"),
            folder, f"g{gi}/images/train", f"g{gi}/images/val", labels,
        )

    if log_fn:
        log_fn(f"train: {len(train_images)}, val: {len(val_images)}")

    return {
        "folder": folder,
        "groups": [sorted(g) for g in groups],
        "k": len(groups),
        "train_images": train_images,
        "val_images": val_images,
        "group_train_count": group_train_count,
        "group_val_count": group_val_count,
        "images": list(analysis.images),
        "output_dir": folder,
    }
