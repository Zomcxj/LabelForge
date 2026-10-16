"""撤销/重做状态机测试。

`UndoManager` 此前只在 main_window 里被间接使用，测试里仅有 stub 的
`save_undo_state`，真实栈行为从未被覆盖。
"""
from labelforge.core.config import UNDO_CONFIG
from labelforge.engine.undo_manager import UndoManager


def _boxes(*labels):
    return [{"label": label, "x": i, "y": i, "width": 10, "height": 10}
            for i, label in enumerate(labels)]


def test_undo_returns_the_previously_saved_state():
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))

    items, boxes = mgr.undo(["b"], _boxes("dog"))

    assert items == ["a"]
    assert [b["label"] for b in boxes] == ["cat"]


def test_redo_returns_the_state_that_undo_replaced():
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))
    mgr.undo(["b"], _boxes("dog"))

    items, boxes = mgr.redo(["a"], _boxes("cat"))

    assert items == ["b"]
    assert [b["label"] for b in boxes] == ["dog"]


def test_undo_on_empty_stack_returns_inputs_unchanged():
    mgr = UndoManager()
    items, boxes = ["x"], _boxes("cat")

    assert mgr.undo(items, boxes) == (items, boxes)


def test_redo_on_empty_stack_returns_inputs_unchanged():
    mgr = UndoManager()
    items, boxes = ["x"], _boxes("cat")

    assert mgr.redo(items, boxes) == (items, boxes)


def test_new_save_clears_the_redo_stack():
    """撤销后再做新操作，重做分支必须失效，否则会重做到旧分支。"""
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))
    mgr.undo(["b"], _boxes("dog"))
    assert mgr.can_redo()

    mgr.save_state(["c"], _boxes("bird"))

    assert not mgr.can_redo()


def test_can_undo_and_can_redo_track_stack_state():
    mgr = UndoManager()
    assert not mgr.can_undo()
    assert not mgr.can_redo()

    mgr.save_state(["a"], _boxes("cat"))
    assert mgr.can_undo()
    assert not mgr.can_redo()

    mgr.undo(["b"], _boxes("dog"))
    assert not mgr.can_undo()
    assert mgr.can_redo()


def test_history_is_capped_at_max_history_dropping_oldest():
    mgr = UndoManager()
    limit = UNDO_CONFIG["max_history"]
    for i in range(limit + 10):
        mgr.save_state([f"item{i}"], _boxes(f"label{i}"))

    assert len(mgr._undo_stack) == limit
    # 最旧的被丢弃，最新一次保存仍可撤销
    items, _ = mgr.undo([], [])
    assert items == [f"item{limit + 9}"]
    # 最早的已被挤出：栈里第一条应是 item10
    assert mgr._undo_stack[0]["canvas_items"] == ["item10"]


def test_save_state_copies_box_dicts_so_later_mutation_does_not_leak():
    """保存后原地改框的 label，撤销时不应看到被改后的值。"""
    mgr = UndoManager()
    boxes = _boxes("cat")
    mgr.save_state([], boxes)

    boxes[0]["label"] = "dog"

    _, restored = mgr.undo([], [])
    assert restored[0]["label"] == "cat"


def test_undo_copies_current_boxes_so_later_mutation_does_not_leak():
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))

    live = _boxes("dog")
    mgr.undo(["b"], live)
    live[0]["label"] = "bird"

    _, redone = mgr.redo(["a"], _boxes("cat"))
    assert redone[0]["label"] == "dog"


def test_multiple_undos_walk_back_through_history_in_order():
    mgr = UndoManager()
    for step in range(4):
        mgr.save_state([f"step{step}"], [])

    seen = []
    for _ in range(4):
        items, _ = mgr.undo([], [])
        seen.append(items[0])

    assert seen == ["step3", "step2", "step1", "step0"]


def test_undo_redo_round_trip_returns_to_original_state():
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))

    mgr.undo(["b"], _boxes("dog"))
    items, boxes = mgr.redo(["a"], _boxes("cat"))

    assert items == ["b"]
    assert [b["label"] for b in boxes] == ["dog"]


def test_clear_empties_both_stacks():
    mgr = UndoManager()
    mgr.save_state(["a"], _boxes("cat"))
    mgr.undo(["b"], _boxes("dog"))

    mgr.clear()

    assert not mgr.can_undo()
    assert not mgr.can_redo()


def test_canvas_items_list_is_copied_not_aliased():
    """保存后往调用方的列表里追加，撤销结果不应跟着变。"""
    mgr = UndoManager()
    live = ["a"]
    mgr.save_state(live, [])

    live.append("b")

    items, _ = mgr.undo([], [])
    assert items == ["a"]
