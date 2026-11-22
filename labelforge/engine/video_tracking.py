"""视频跨帧跟踪：ByteTrack 式两阶段关联 + 卡尔曼滤波。纯计算，不依赖 Qt。"""

from dataclasses import dataclass, field

import numpy as np

from ..core.utils import calculate_iou

# 单条轨迹保留的历史帧数。平滑只回看窗口长度，长视频没必要留全片，
# 否则一小时 30fps、10 个目标会把几百 MB 的框字典攥在内存里。
HISTORY_LIMIT = 64

# 同一帧里同类别框 IoU 达到这个值就视为重复检测，只留分数最高的
DUPLICATE_IOU = 0.9


@dataclass
class Detection:
    """一帧里的一个检测框，关联使用的最小数据结构。

    `raw` 保留检测器给出的其余字段（`class_id` 等），回写结果时不丢信息。
    """

    x: float
    y: float
    width: float
    height: float
    score: float = 1.0
    label: str = ""
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_box(cls, box):
        """内部框字典 -> Detection；没给分数视为满分。"""
        score = box.get("score")
        return cls(
            x=float(box["x"]), y=float(box["y"]),
            width=max(1.0, float(box["width"])),
            height=max(1.0, float(box["height"])),
            score=1.0 if score is None else float(score),
            label=str(box.get("label") or ""),
            raw=dict(box),
        )

    def to_box(self, track_id=None):
        """回到内部框字典；带上 track_id 与检测器的原始字段。"""
        box = dict(self.raw)
        box.update(x=self.x, y=self.y, width=self.width,
                   height=self.height, score=self.score, label=self.label)
        if track_id is not None:
            box["track_id"] = int(track_id)
        return box

    @property
    def center(self):
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)

    @property
    def aspect(self):
        return self.width / max(1e-6, self.height)

    def as_tuple(self):
        """(x1, y1, x2, y2)，供 IoU 计算。"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class KalmanBoxFilter:
    """匀速模型的卡尔曼滤波。

    状态为 `[中心x, 中心y, 宽高比, 高, 对应速度]`，观测为前四项。
    只用于预测轨迹在下一帧的位置，输出框仍取检测框本身。
    """

    POSITION_WEIGHT = 1.0 / 20.0
    VELOCITY_WEIGHT = 1.0 / 160.0

    def __init__(self, detection):
        height = max(1.0, detection.height)
        cx, cy = detection.center
        self.mean = np.zeros(8, dtype=np.float64)
        self.mean[:4] = [cx, cy, detection.aspect, height]
        std = [
            2 * self.POSITION_WEIGHT * height,
            2 * self.POSITION_WEIGHT * height,
            1e-2,
            2 * self.POSITION_WEIGHT * height,
            10 * self.VELOCITY_WEIGHT * height,
            10 * self.VELOCITY_WEIGHT * height,
            1e-5,
            10 * self.VELOCITY_WEIGHT * height,
        ]
        self.covariance = np.diag(np.square(std))

    @staticmethod
    def _motion():
        motion = np.eye(8)
        for index in range(4):
            motion[index, index + 4] = 1.0
        return motion

    def predict(self):
        """推进一帧，返回预测框。"""
        height = max(1.0, self.mean[3])
        process = np.diag(np.square([
            self.POSITION_WEIGHT * height,
            self.POSITION_WEIGHT * height,
            1e-2,
            self.POSITION_WEIGHT * height,
            self.VELOCITY_WEIGHT * height,
            self.VELOCITY_WEIGHT * height,
            1e-5,
            self.VELOCITY_WEIGHT * height,
        ]))
        motion = self._motion()
        self.mean = motion @ self.mean
        self.covariance = motion @ self.covariance @ motion.T + process
        return self.box()

    def update(self, detection):
        """用观测修正状态。"""
        height = max(1.0, self.mean[3])
        measurement = np.array([
            detection.center[0], detection.center[1], detection.aspect,
            max(1.0, detection.height),
        ])
        project = np.eye(4, 8)
        noise = np.diag(np.square([
            self.POSITION_WEIGHT * height,
            self.POSITION_WEIGHT * height,
            1e-1,
            self.POSITION_WEIGHT * height,
        ]))
        gain = (self.covariance @ project.T
                @ np.linalg.inv(project @ self.covariance @ project.T + noise))
        self.mean = self.mean + gain @ (measurement - project @ self.mean)
        self.covariance = (np.eye(8) - gain @ project) @ self.covariance

    def box(self):
        """当前状态对应的框 (x, y, w, h)。"""
        cx, cy, aspect, height = self.mean[:4]
        height = max(1.0, float(height))
        width = max(1.0, float(aspect) * height)
        return {"x": float(cx - width / 2.0), "y": float(cy - height / 2.0),
                "width": width, "height": height}


class Track:
    """一条轨迹：自己的检测结构 + 卡尔曼滤波状态。"""

    __slots__ = ("id", "label", "box", "hits", "misses", "history", "filter")

    def __init__(self, track_id, detection, frame_index):
        self.id = track_id
        self.label = detection.label
        self.box = detection.to_box()
        self.hits = 1
        self.misses = 0
        self.history = [(frame_index, self.box)]
        self.filter = KalmanBoxFilter(detection)

    def predict(self):
        """推进滤波，返回下一帧的预测框；只用于匹配。"""
        return self.filter.predict()

    def update(self, detection, frame_index):
        self.filter.update(detection)
        self.box = detection.to_box()
        self.label = detection.label
        self.hits += 1
        self.misses = 0
        self.history.append((frame_index, self.box))
        if len(self.history) > HISTORY_LIMIT:
            del self.history[0]

    def miss(self):
        self.misses += 1


def _box_tuple(box):
    """框字典 → `calculate_iou` 需要的 (x1, y1, x2, y2) 元组。"""
    return (box["x"], box["y"],
            box["x"] + box["width"], box["y"] + box["height"])


class Tracker:
    """ByteTrack 式两阶段关联 + 卡尔曼预测。

    - 每帧先用卡尔曼滤波把活动轨迹外推到当前位置（只影响匹配）
    - 高分框先与轨迹匹配（IoU 阈值 `iou_threshold`）
    - 剩下的轨迹再用低分框匹配（阈值更严 `iou_threshold_low`），
      把被置信度滤掉但位置对得上的目标接回原轨迹
    - 只匹配同类别；id 全片递增不复用；连续 `max_misses` 帧未命中则结束
    - 低分框接不上轨迹就丢弃，不进入结果
    """

    def __init__(self, iou_threshold=0.3, max_misses=3,
                 det_high=0.5, det_low=0.3, iou_threshold_low=0.5):
        self.iou_threshold = float(iou_threshold)
        self.iou_threshold_low = float(iou_threshold_low)
        self.max_misses = int(max_misses)
        self.det_high = float(det_high)
        self.det_low = float(det_low)
        self.tracks = []
        self._next_id = 1

    def active_tracks(self):
        return [t for t in self.tracks if t.misses < self.max_misses]

    def _split(self, detections):
        """按置信度把检测分成高分与低分两组，返回两组的下标。"""
        high, low = [], []
        for index, detection in enumerate(detections):
            if detection.score >= self.det_high:
                high.append(index)
            elif detection.score >= self.det_low:
                low.append(index)
        return high, low

    @staticmethod
    def _associate(tracks, predicted, detections, indices, threshold):
        """一对一贪心匹配，返回 [(track, detection_index)]。"""
        candidates = []
        for track in tracks:
            box = predicted.get(track.id)
            if box is None:
                continue
            box_tuple = _box_tuple(box)
            for index in indices:
                detection = detections[index]
                if detection.label != track.label:
                    continue
                iou = calculate_iou(box_tuple, detection.as_tuple())
                if iou >= threshold:
                    candidates.append((iou, track, index))
        candidates.sort(key=lambda item: item[0], reverse=True)

        used_tracks, used_boxes, pairs = set(), set(), []
        for _, track, index in candidates:
            if track.id in used_tracks or index in used_boxes:
                continue
            used_tracks.add(track.id)
            used_boxes.add(index)
            pairs.append((track, index))
        return pairs

    def update(self, boxes, frame_index):
        """推进一帧，返回带 `track_id` 的框列表（保持输入顺序）。

        低分且没能接上轨迹的框不进入返回值，避免它们被画进成片。
        """
        detections = [Detection.from_box(box) for box in (boxes or [])]
        high, low = self._split(detections)
        result = [None] * len(detections)

        active = self.active_tracks()
        # 每帧只推进一步卡尔曼预测，两阶段匹配共用同一份预测框
        predicted = {track.id: track.predict() for track in active}

        pairs = self._associate(active, predicted, detections, high,
                                self.iou_threshold)
        taken = {track.id for track, _ in pairs}
        remaining = [t for t in active if t.id not in taken]
        pairs += self._associate(remaining, predicted, detections, low,
                                 self.iou_threshold_low)

        for track, index in pairs:
            track.update(detections[index], frame_index)
            taken.add(track.id)
            result[index] = detections[index].to_box(track.id)

        for track in active:
            if track.id not in taken:
                track.miss()

        for index in high:
            if result[index] is not None:
                continue
            detection = detections[index]
            track = Track(self._next_id, detection, frame_index)
            self._next_id += 1
            self.tracks.append(track)
            result[index] = detection.to_box(track.id)

        kept = [item for item in result if item is not None]
        # 已结束的轨迹仍留在 self.tracks 里供统计
        return _drop_near_duplicates(kept)


def _drop_near_duplicates(boxes):
    """同一帧里几乎重合的同类别框只留分数最高的一个。

    检测器偶尔会把同一目标检两遍（框几乎重合），进入跟踪前先去掉，
    免得两条轨迹抢同一个目标。
    """
    ordered = sorted(boxes,
                     key=lambda b: float(b.get("score") or 0.0), reverse=True)
    kept = []
    for box in ordered:
        duplicate = False
        for existing in kept:
            if existing.get("label") != box.get("label"):
                continue
            if calculate_iou(_box_tuple(existing), _box_tuple(box)) >= DUPLICATE_IOU:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def smooth_history(history, method="median", window=5, alpha=0.6):
    """对一条轨迹的历史框做因果平滑（只用当前及过去的帧）。

    因果平滑是为了保持单遍解码：若用对称窗口就必须先读完整个视频，
    再解一次码才能绘制。
    """
    if not history:
        raise ValueError("history 不能为空")
    if len(history) == 1:
        return history[-1][1]
    current = history[-1][1]
    if method == "none":
        return current
    recent = [box for _, box in history[-max(1, int(window)):]]
    if method == "median":
        out = dict(current)
        for key in ("x", "y", "width", "height"):
            out[key] = _median([b[key] for b in recent])
        return out
    if method == "ema":
        weight = float(min(1.0, max(0.0, alpha)))
        out = dict(current)
        # 窗口只有一帧时没有上一帧可比，维持当前值
        previous_box = recent[-2] if len(recent) > 1 else current
        for key in ("x", "y", "width", "height"):
            out[key] = previous_box[key] * (1.0 - weight) + current[key] * weight
        return out
    raise ValueError(f"未知平滑方式：{method}")


def drop_short_tracks(tracks, min_hits=2):
    """返回 (保留的轨迹 id 集合, 被丢弃的轨迹数)。"""
    kept, dropped = set(), 0
    for track in tracks:
        if track.hits >= int(min_hits):
            kept.add(track.id)
        else:
            dropped += 1
    return kept, dropped


class TrackingSession:
    """把跟踪、平滑与逐帧缓冲合成一个对象，供 render_video 顺序调用。

    `step()` 逐帧喂入检测框，返回已平滑、已带 track_id 的框；
    帧数据被缓存，收尾时用 `finalize()` / `finalize_frames()` 做短轨迹过滤。
    缓存只是框字典，内存占用远小于帧图像。
    """

    def __init__(self, iou_threshold=0.3, max_misses=3,
                 smooth_method="median", smooth_window=5, smooth_alpha=0.6,
                 det_high=0.5, det_low=0.3):
        # 用户把 conf 调到 0.1 以下时，低分阈值不能高于它，否则推理取框
        # 阈值反而比用户的设置更严
        det_low = min(float(det_low), float(det_high))
        self.tracker = Tracker(iou_threshold=iou_threshold,
                              max_misses=max_misses,
                              det_high=det_high, det_low=det_low)
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window
        self.smooth_alpha = smooth_alpha
        # 推理时按 det_low 取框，交给跟踪器再分高/低两档
        self.det_high = float(det_high)
        self.det_low = float(det_low)
        self._frames = []
        # 本片里靠低分框续回轨迹的次数，结果区会显示，便于判断阈值是否合适
        self.rescued = 0
        # id → Track 缓存：每帧每框只做一次字典取值
        self._tracks_by_id = {}

    def _track(self, track_id):
        """按 id 取轨迹；新出现的 id 只回填一次缓存。"""
        track = self._tracks_by_id.get(track_id)
        if track is not None:
            return track
        for candidate in reversed(self.tracker.tracks):
            if candidate.id == track_id:
                self._tracks_by_id[track_id] = candidate
                return candidate
        return None

    def step(self, boxes, frame_index):
        tracked = self.tracker.update(boxes, frame_index)
        self.rescued += sum(
            1 for box in tracked
            if float(box.get("score") or 1.0) < self.det_high)
        out = []
        for box in tracked:
            track_id = box.get("track_id")
            if track_id is None:
                out.append(box)
                continue
            track = self._track(track_id)
            smoothed = smooth_history(track.history,
                                     method=self.smooth_method,
                                     window=self.smooth_window,
                                     alpha=self.smooth_alpha)
            merged = dict(smoothed, label=track.label, track_id=track_id)
            if "score" in box:
                merged["score"] = box["score"]
            out.append(merged)
        # 逐帧缓存只存元组：长视频要按帧攒几千条记录，字典会多占一倍内存
        self._frames.append((frame_index, [
            (box.get("track_id"), box.get("label", ""), box["x"], box["y"],
             box["width"], box["height"], box.get("score"),
             box.get("class_id")) for box in out]))
        return out

    def finalize(self, min_hits=2):
        """返回 (保留的轨迹 id 集合, 丢弃的轨迹数)。"""
        return drop_short_tracks(self.tracker.tracks, min_hits=min_hits)

    def finalize_frames(self, min_hits=2):
        """按短轨迹过滤后返回 [(帧号, 框列表)]，供落盘使用。"""
        kept_ids, _ = self.finalize(min_hits=min_hits)
        result = []
        for frame_index, rows in self._frames:
            boxes = []
            for (track_id, label, x, y, width, height,
                 score, class_id) in rows:
                if track_id not in kept_ids:
                    continue
                box = {"label": label, "x": x, "y": y,
                       "width": width, "height": height,
                       "track_id": track_id}
                if score is not None:
                    box["score"] = score
                if class_id is not None:
                    box["class_id"] = class_id
                boxes.append(box)
            result.append((frame_index, boxes))
        return result
