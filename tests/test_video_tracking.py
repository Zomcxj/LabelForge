"""跨帧跟踪：关联稳定性、新建与结束时机、同类别约束。"""
from labelforge.engine.video_tracking import Tracker


def _box(x, y, w=10, h=10, label="car"):
    return {"x": x, "y": y, "width": w, "height": h, "label": label}


def test_consecutive_boxes_keep_the_same_track_id():
    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    first = tracker.update([_box(0, 0)], 0)
    second = tracker.update([_box(2, 0)], 1)
    assert first[0]["track_id"] == second[0]["track_id"]


def test_distant_box_creates_a_new_track():
    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    a = tracker.update([_box(0, 0)], 0)
    b = tracker.update([_box(500, 500)], 1)
    assert a[0]["track_id"] != b[0]["track_id"]


def test_different_labels_never_match():
    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    a = tracker.update([_box(0, 0, label="car")], 0)
    b = tracker.update([_box(0, 0, label="bus")], 1)
    assert a[0]["track_id"] != b[0]["track_id"]


def test_track_ends_after_max_misses():
    tracker = Tracker(iou_threshold=0.3, max_misses=2)
    tracker.update([_box(0, 0)], 0)
    tracker.update([], 1)
    tracker.update([], 2)
    assert tracker.active_tracks() == []


def test_hits_accumulate_per_track():
    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    tracker.update([_box(0, 0)], 0)
    tracker.update([_box(1, 0)], 1)
    tracker.update([_box(2, 0)], 2)
    assert tracker.tracks[0].hits == 3


def test_track_ids_are_never_reused():
    tracker = Tracker(iou_threshold=0.3, max_misses=1)
    tracker.update([_box(0, 0)], 0)
    tracker.update([], 1)
    tracker.update([], 2)
    again = tracker.update([_box(0, 0)], 3)
    assert again[0]["track_id"] == 2


def test_median_smoothing_suppresses_a_single_outlier():
    from labelforge.engine.video_tracking import smooth_history

    history = [(0, _box(0, 0)), (1, _box(1, 0)), (2, _box(60, 0)),
               (3, _box(3, 0)), (4, _box(4, 0))]
    box = smooth_history(history, method="median", window=5, alpha=0.6)
    assert box["x"] < 10, "单帧跳变应被中位数压掉"


def test_exponential_smoothing_moves_gradually():
    from labelforge.engine.video_tracking import smooth_history

    history = [(0, _box(0, 0)), (1, _box(100, 0))]
    box = smooth_history(history, method="ema", window=5, alpha=0.5)
    assert 0 < box["x"] < 100


def test_smoothing_without_track_id_returns_input():
    from labelforge.engine.video_tracking import smooth_history

    box = _box(7, 8)
    assert smooth_history([(0, box)], method="median", window=5) == box


def test_short_tracks_are_dropped():
    from labelforge.engine.video_tracking import TrackingSession

    session = TrackingSession(iou_threshold=0.3, max_misses=1)
    session.step([_box(0, 0)], 0)          # 轨迹 1 只命中 1 帧
    session.step([_box(300, 300)], 1)      # 轨迹 2 也只命中 1 帧
    kept_ids, dropped = session.finalize(min_hits=2)
    assert kept_ids == set()
    assert dropped == 2


def test_two_hit_track_survives_the_default_threshold():
    from labelforge.engine.video_tracking import TrackingSession

    session = TrackingSession(iou_threshold=0.3, max_misses=1)
    session.step([_box(0, 0)], 0)
    session.step([_box(1, 0)], 1)
    kept_ids, dropped = session.finalize(min_hits=2)
    assert kept_ids == {1}
    assert dropped == 0


def test_finalize_returns_frames_without_dropped_tracks():
    from labelforge.engine.video_tracking import TrackingSession

    session = TrackingSession(iou_threshold=0.3, max_misses=1)
    session.step([_box(0, 0)], 0)
    session.step([_box(1, 0), _box(400, 400)], 1)
    frames = session.finalize_frames(min_hits=2)
    assert len(frames) == 2
    assert all(b.get("track_id") == 1 for _, boxes in frames for b in boxes)


# ---------- ByteTrack 式两阶段关联 ----------

def _scored(x, score, w=20, h=20, label="car"):
    return {"x": x, "y": 0, "width": w, "height": h, "label": label,
            "score": score}


def test_low_score_box_keeps_the_track_alive():
    """高分框建轨后，低分框位置对得上就续上同一条轨迹。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3,
                      det_high=0.5, det_low=0.1)
    first = tracker.update([_scored(0, 0.9)], 0)
    second = tracker.update([_scored(1, 0.2)], 1)

    assert first[0]["track_id"] == second[0]["track_id"] == 1
    assert second[0]["score"] == 0.2, "救回的框保留自己的分数"


def test_unmatched_low_score_box_is_dropped():
    """低分框接不上任何轨迹时不进入结果，避免被画进成片。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3,
                      det_high=0.5, det_low=0.1)
    assert tracker.update([_scored(0, 0.2)], 0) == []
    assert tracker.tracks == []


def test_score_below_det_low_is_ignored_entirely():
    """低于低分阈值的框既不建轨也不续轨。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3,
                      det_high=0.5, det_low=0.1)
    tracker.update([_scored(0, 0.9)], 0)
    assert tracker.update([_scored(0, 0.05)], 1) == []
    assert tracker.active_tracks()[0].misses == 1


def test_prediction_keeps_a_fast_moving_track():
    """按速度外推后，跳得较远的下一帧仍能接上同一条轨迹。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    tracker.update([_scored(0, 0.9)], 0)
    tracker.update([_scored(5, 0.9)], 1)

    # 上一帧在 x=5，本帧跳到 15：直接算 IoU 只有 0.14，
    # 按速度外推到 10 后 IoU ≈ 0.47，能接上
    out = tracker.update([_scored(15, 0.9)], 2)
    assert out[0]["track_id"] == 1


def test_two_stage_matching_does_not_steal_a_high_score_match():
    """高分框优先匹配，剩下的低分框只能接未被占用的轨迹。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3,
                      det_high=0.5, det_low=0.1)
    tracker.update([_scored(0, 0.9), _scored(100, 0.9)], 0)
    out = tracker.update([_scored(0, 0.95), _scored(100, 0.2)], 1)

    ids = sorted(box["track_id"] for box in out)
    assert ids == [1, 2]


# ---------- 检测结构与卡尔曼滤波 ----------

def test_detection_round_trip_keeps_extra_fields():
    from labelforge.engine.video_tracking import Detection

    detection = Detection.from_box(
        {"x": 1, "y": 2, "width": 3, "height": 4, "label": "car",
         "score": 0.8, "class_id": 7})
    box = detection.to_box(5)

    assert (box["x"], box["y"], box["width"], box["height"]) == (1, 2, 3, 4)
    assert box["class_id"] == 7, "检测器的附加字段不能丢"
    assert box["track_id"] == 5
    assert detection.center == (2.5, 4.0)
    assert abs(detection.aspect - 0.75) < 1e-9


def test_detection_defaults_score_to_one():
    from labelforge.engine.video_tracking import Detection

    assert Detection.from_box(
        {"x": 0, "y": 0, "width": 2, "height": 2}).score == 1.0


def test_kalman_predicts_constant_velocity():
    from labelforge.engine.video_tracking import Detection, KalmanBoxFilter

    def box(x):
        return Detection.from_box(
            {"x": x, "y": 0, "width": 20, "height": 20})

    kalman = KalmanBoxFilter(box(0))
    for step in range(1, 6):
        kalman.predict()
        kalman.update(box(10 * step))

    predicted = kalman.predict()["x"]
    assert 52.0 <= predicted <= 66.0, predicted


def test_track_survives_a_two_frame_gap_with_prediction():
    """遮挡两帧后目标出现在预测位置附近，仍是同一条轨迹。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3)

    def box(x):
        # 用接近实拍的尺寸：20px 小目标在同样位移下外推误差占比过大
        return {"x": x, "y": 0, "width": 100, "height": 100,
                "label": "car", "score": 0.9}

    track_id = None
    for frame in range(4):
        track_id = tracker.update([box(10 * frame)], frame)[0]["track_id"]

    assert tracker.update([], 4) == []
    assert tracker.update([], 5) == []
    assert tracker.active_tracks(), "两帧丢失还没到 max_misses"

    out = tracker.update([box(60)], 6)
    assert out[0]["track_id"] == track_id


def test_track_history_is_bounded():
    """历史只留最近若干帧：长视频不能把每帧的框都攒在内存里。"""
    from labelforge.engine.video_tracking import HISTORY_LIMIT

    tracker = Tracker(iou_threshold=0.3, max_misses=3)
    for frame in range(HISTORY_LIMIT * 3):
        tracker.update([{"x": float(frame), "y": 0, "width": 20,
                         "height": 20, "label": "car", "score": 0.9}], frame)

    track = tracker.active_tracks()[0]
    assert len(track.history) <= HISTORY_LIMIT
    assert track.hits == HISTORY_LIMIT * 3, "命中计数不能被截断影响"


def test_low_score_rescues_are_counted():
    """救回来的低分框要计数，界面上据此判断阈值是否合适。"""
    from labelforge.engine.video_tracking import TrackingSession

    session = TrackingSession(det_high=0.6, det_low=0.2)
    session.step([{"x": 0, "y": 0, "width": 60, "height": 60,
                   "label": "car", "score": 0.9}], 0)
    session.step([{"x": 2, "y": 0, "width": 60, "height": 60,
                   "label": "car", "score": 0.25}], 1)

    assert session.rescued == 1


def test_identical_boxes_are_collapsed():
    """同一帧里几乎重合的两个同类别框只留一个，避免抢同一条轨迹。"""
    tracker = Tracker(iou_threshold=0.3, max_misses=3)

    out = tracker.update([
        {"x": 10, "y": 10, "width": 40, "height": 40, "label": "car",
         "score": 0.9},
        {"x": 11, "y": 10, "width": 40, "height": 40, "label": "car",
         "score": 0.6},
    ], 0)

    assert len(out) == 1
    assert out[0]["score"] == 0.9
