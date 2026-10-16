"""
撤销/重做模块 - 操作历史管理
"""
from ..core.config import UNDO_CONFIG


class UndoManager:
    """撤销/重做管理器"""

    def __init__(self):
        self._undo_stack = []
        self._redo_stack = []
        self._max_history = UNDO_CONFIG['max_history']

    def save_state(self, canvas_items, detection_boxes):
        """保存当前状态（浅拷贝，不复制 QPixmap）"""
        state = {
            'canvas_items': list(canvas_items),
            'detection_boxes': [dict(b) for b in detection_boxes],
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self, canvas_items, detection_boxes):
        """撤销"""
        if not self._undo_stack:
            return canvas_items, detection_boxes

        current_state = {
            'canvas_items': list(canvas_items),
            'detection_boxes': [dict(b) for b in detection_boxes],
        }
        self._redo_stack.append(current_state)

        prev_state = self._undo_stack.pop()
        return prev_state['canvas_items'], prev_state['detection_boxes']

    def redo(self, canvas_items, detection_boxes):
        """重做"""
        if not self._redo_stack:
            return canvas_items, detection_boxes

        current_state = {
            'canvas_items': list(canvas_items),
            'detection_boxes': [dict(b) for b in detection_boxes],
        }
        self._undo_stack.append(current_state)

        next_state = self._redo_stack.pop()
        return next_state['canvas_items'], next_state['detection_boxes']

    def can_undo(self):
        return len(self._undo_stack) > 0

    def can_redo(self):
        return len(self._redo_stack) > 0

    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
