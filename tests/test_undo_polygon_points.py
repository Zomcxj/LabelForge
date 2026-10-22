"""B1 回归：撤销/重做快照必须按值复制 points。

审计 B1：undo_manager 的快照是浅拷贝——[dict(b) for b in detection_boxes]
里 points 仍指向同一个列表对象。canvas_drawing 的三条顶点路径
（加顶点 / 删顶点 / 拖顶点）都是**原地**改 points，所以这三类操作
撤销不生效；其它编辑路径都是重绑定 box["points"] = 新列表，不受影响。

UndoManager 不依赖 Qt，这里直接用纯 pytest（比真实 Qt 子进程快且稳）。
测试镜像 main_window.undo/redo 的真实用法：编辑器把返回的列表对象
重绑为活列表，后续原地编辑落在返回的 dict 上。
"""
from labelforge.engine.undo_manager import UndoManager


def _box(points):
    """构造一个带 points 的多边形框（与 canvas 的 box dict 同构）。"""
    return {"label": "cat", "x": 0, "y": 0, "width": 10, "height": 10,
            "points": [list(p) for p in points]}


def test_insert_vertex_undo_restores_points():
    m = UndoManager()
    box = _box([[0, 0], [10, 0], [10, 10]])
    original = [list(p) for p in box["points"]]
    m.save_state([], [box])
    box["points"].insert(1, [5, 0])  # _insert_polygon_vertex 的原地插入
    _items, boxes = m.undo([], [box])
    assert boxes[0]["points"] == original
    # 恢复出来的是独立副本，不是被污染的同一个列表对象
    assert boxes[0]["points"] is not box["points"]


def test_delete_vertex_undo_restores_points():
    m = UndoManager()
    box = _box([[0, 0], [10, 0], [10, 10]])
    original = [list(p) for p in box["points"]]
    m.save_state([], [box])
    del box["points"][1]  # _delete_polygon_vertex 的原地删除
    _items, boxes = m.undo([], [box])
    assert boxes[0]["points"] == original
    assert len(boxes[0]["points"]) == 3


def test_drag_vertex_undo_restores_points():
    m = UndoManager()
    box = _box([[0, 0], [10, 10]])
    original = [list(p) for p in box["points"]]
    m.save_state([], [box])
    box["points"][1] = [99, 99]  # _resize_box 顶点分支的原地赋值
    _items, boxes = m.undo([], [box])
    assert boxes[0]["points"] == original


def test_snapshot_is_isolated_from_later_inplace_mutation():
    """快照之后继续原地改源，历史不得跟着变（浅拷贝的病根本身）。"""
    m = UndoManager()
    box = _box([[0, 0], [10, 10]])
    m.save_state([], [box])
    box["points"].append([999, 999])
    snap = m._undo_stack[-1]["detection_boxes"][0]
    assert snap["points"] == [[0, 0], [10, 10]]


def test_boxes_without_points_key_keep_their_shape():
    """只有多边形/OBB 带 points；矩形 dict 不该被快照强行加上空 points 键
    （下游用 'points' in box 区分形状）。"""
    m = UndoManager()
    rect = {"label": "dog", "x": 1, "y": 2, "width": 3, "height": 4}
    m.save_state([], [rect])
    snap = m._undo_stack[-1]["detection_boxes"][0]
    assert "points" not in snap
    assert snap == {"label": "dog", "x": 1, "y": 2, "width": 3, "height": 4}


def test_vertex_edit_survives_undo_redo_interleaving():
    """完整用户流回归网：save → 编辑 → undo → redo → 再编辑 → undo。

    编辑器把返回对象重绑为活列表后继续原地编辑，历史栈不得被污染。
    （当前实现里 undo/redo 压栈的快照与调用时的活 dict 共享 points，
    但该 dict 压栈后即脱离活状态、无人再原地改它——此用例把这组别名
    推理钉住：谁要是改了返回语义让它继续共享，这里就会红。）
    """
    m = UndoManager()
    box = _box([[0, 0], [10, 10]])
    baseline = [list(p) for p in box["points"]]
    m.save_state([], [box])

    box["points"].insert(0, [1, 1])  # edit1
    _items, boxes = m.undo([], [box])
    box = boxes[0]
    assert box["points"] == baseline

    _items, boxes = m.redo([], [box])
    box = boxes[0]
    assert box["points"] == [[1, 1]] + baseline

    box["points"].insert(0, [2, 2])  # edit2（原地）
    _items, boxes = m.undo([], [box])
    assert boxes[0]["points"] == baseline
