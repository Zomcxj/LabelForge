"""推理 UI 接线：入口、数据集代数校验、结果应用（单次撤销/刷新/保存）。

结果应用必须一次性完成：`_create_detection_box()` 每框都会刷新列表并写盘，
逐框调用会让整批推理产生成百上千次保存。这里只保存一次撤销快照、
刷新一次列表、写一次当前图 sidecar。
"""
from ..i18n import t as tr

# 数据集代数：切图/换数据集后，迟到的结果不能写到别的图上
INFERENCE_SOURCE = "inference"


class InferenceMixin:
    def _open_inference(self):
        """打开推理弹窗（非模态，打开后可继续操作主窗口）。"""
        if getattr(self, '_busy', False):
            return
        if not getattr(self, 'background_images', None):
            self.status_label.setText(tr("请先加载数据集"))
            return
        from ..inference_dialog import InferenceDialog
        from ...core import config_manager

        self._close_inference_dialog()
        settings = config_manager.load_all().get('inference_settings') or {}
        dialog = InferenceDialog(self, settings=settings)
        entries = list(enumerate(self.background_images))
        dialog.set_images(entries, self.current_background_index)
        dialog.apply_requested.connect(self._apply_inference_results)
        dialog.rejected.connect(lambda d=dialog: self._cleanup_inference_dialog(d))
        self._inference_dialog = dialog
        self._inference_dataset = tuple(self.background_images)
        dialog.show()

    def _cleanup_inference_dialog(self, dialog):
        if dialog is getattr(self, '_inference_dialog', None):
            self._inference_dialog = None

    def _close_inference_dialog(self):
        dialog = getattr(self, '_inference_dialog', None)
        if dialog is None:
            return
        self._inference_dialog = None
        try:
            dialog.apply_requested.disconnect()
        except TypeError:
            pass
        dialog.close()

    def _inference_dataset_changed(self):
        """数据集是否已换：换了就丢弃迟到结果，不写到别的图上。"""
        current = tuple(getattr(self, 'background_images', None) or ())
        return current != getattr(self, '_inference_dataset', None)

    def _apply_inference_results(self, results, replace_auto):
        """把一批推理结果写进内存与磁盘。

        当前显示图走正常保存流程（保留贴图与画布状态），其余图直接写 sidecar。
        """
        if self._inference_dataset_changed():
            self.status_label.setText(tr("数据集已切换，推理结果已丢弃"))
            return
        if getattr(self, '_is_delete_view', False):
            return
        from ...engine import inference_task

        current = self.current_background_index
        memory = {
            idx: list(boxes) for idx, boxes in
            (getattr(self, 'detection_boxes_dict', None) or {}).items()
        }
        written, failed = inference_task.write_results(
            results, memory_boxes=memory, replace_auto=replace_auto,
            skip_index=current)

        current_result = results.get(current)
        if current_result is not None and not current_result.error:
            self._apply_inference_to_current(current_result, replace_auto)

        total_boxes = sum(len(r.boxes) for r in results.values() if not r.error)
        text = (f"{tr('推理完成')}: {len(results)} {tr('张图')} · "
                f"{total_boxes} {tr('个框')} · {written} {tr('张已写入')}")
        if failed:
            text += f" · {len(failed)} {tr('张写入失败')}"
        self.status_label.setText(text)
        self._dataset_stats_dirty = True

    def _apply_inference_to_current(self, result, replace_auto):
        """当前图的框一次性替换：单次撤销快照 + 单次列表刷新 + 单次保存。"""
        import os
        from ...engine.inference_task import merge_boxes, remove_empty_sidecar

        index = self.current_background_index
        merged = merge_boxes(self.detection_boxes, result.boxes, replace_auto=replace_auto)
        self.save_undo_state()
        self.detection_boxes = merged
        if index >= 0:
            self.detection_boxes_dict[index] = list(merged)
        canvas = getattr(self, 'canvas', None)
        if canvas is not None:
            canvas.selected_box = None
            canvas.selected_boxes = []
            if hasattr(canvas, 'update_status_label'):
                canvas.update_status_label()
            canvas.update()
        self.update_label_list()
        if index < 0:
            return
        path = self.background_images[index]
        # 与批量一致：没框也没贴图就不写盘，避免留下 shapes 为 [] 的空 json
        if not merged and not (getattr(self, 'canvas_items', None) or []):
            remove_empty_sidecar(path)
            return
        # 传入画布贴图，避免重写 sidecar 时把贴图 shape 丢掉
        self.save_json(path, os.path.basename(path), "",
                       canvas_items=self.canvas_items, current_index=index)
