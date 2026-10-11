"""推理接线：工具栏图标、模型参数行、单图与批量推理。

模型只加载一次，阈值与输入尺寸改动即时写回 detector，不重建会话。
参数行是唯一的模型入口：关掉它就不能推理（加载按钮也在行内），再点一次图标恢复。
结果应用必须一次性完成：`_create_detection_box()` 每框都会刷新列表并写盘，
逐框调用会让整批推理产生成百上千次保存。这里只保存一次撤销快照、
刷新一次列表、写一次当前图 sidecar。
"""
import os

from ..i18n import t as tr

INFERENCE_SOURCE = "inference"


class InferenceMixin:
    # ---------- 模型参数行 ----------

    def _toggle_model_row(self):
        """第一个图标：显示/隐藏参数行；关掉后不能推理。"""
        row = self._ensure_model_row()
        if row.isVisible():
            row.hide()
        else:
            row.show()
        self._update_inference_buttons()

    # ---------- 加载模型 ----------

    def _load_inference_model(self):
        """选 .onnx 并后台加载；成功后填入参数行、启用两个推理图标。"""
        if getattr(self, '_busy', False):
            return
        if getattr(self, '_model_worker', None) is not None:
            return
        from PyQt5.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, tr("选择 ONNX 模型"), "", "ONNX (*.onnx)")
        if not path:
            return
        from ...engine.onnx_inference import runtime_available
        if not runtime_available():
            self._inference_warn(tr("未安装 onnxruntime，无法执行推理"))
            return
        self.status_label.setText(tr("正在加载模型..."))
        saved = self._saved_inference_settings()
        self._load_model_with_names(path, self._saved_class_names(path), saved)

    def _saved_inference_settings(self):
        """上次用过的阈值与输入尺寸；没有则给空表用引擎默认值。"""
        from ...core import config_manager
        settings = config_manager.load_all().get('inference_settings') or {}
        return {
            "conf_threshold": settings.get("conf_threshold"),
            "iou_threshold": settings.get("iou_threshold"),
            "input_size": settings.get("input_size"),
        }

    def _saved_class_names(self, model_path):
        """优先用同级同名 txt；没有就交给 worker 读元数据。"""
        from ...engine import inference_task
        from ..inference_workers import read_saved_class_names
        path = inference_task.sidecar_class_names_path(model_path)
        return read_saved_class_names(path)

    def _load_model_with_names(self, model_path, class_names, settings=None):
        from ..inference_workers import ModelLoadWorker

        worker = ModelLoadWorker(model_path, class_names, settings, self)
        worker.loaded.connect(self._on_model_loaded)
        worker.need_class_names.connect(self._on_class_names_needed)
        worker.failed.connect(self._on_model_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_model_worker(w))
        self._model_worker = worker
        worker.start()

    def _on_class_names_needed(self, model_path, suggested_path):
        """同级没有 txt 且模型也没有类别表：让用户手输并写入同级同名 txt。

        取消即视为不支持推理：不加载模型、两个推理图标保持禁用。
        """
        from ..inference_workers import class_names_prompt

        self.status_label.setText(tr("模型缺少类别信息"))
        written = class_names_prompt(self, model_path, suggested_path)
        if not written:
            self._detector = None
            self._update_inference_buttons()
            self.status_label.setText(tr("未提供类别信息，无法推理"))
            return
        self._load_model_with_names(model_path, self._saved_class_names(model_path))

    def _cleanup_model_worker(self, worker):
        if worker is getattr(self, '_model_worker', None):
            self._model_worker = None
        worker.deleteLater()

    def _on_model_loaded(self, detector, size, source):
        from ...engine.onnx_inference import FORMAT_V26

        self._detector = detector
        self._model_size = size
        self._model_format = detector.resolve_format()
        self._inference_dataset = tuple(self.background_images)
        row = self._ensure_model_row()
        row.set_model(
            os.path.basename(detector.model_path), detector.conf_threshold,
            detector.iou_threshold, size, detector.active_providers,
            dynamic=detector.is_dynamic_input(),
            end2end=self._model_format == FORMAT_V26)
        row.show()
        # 换模型后类别表可能完全不同，之前的勾选作废
        self._class_filter = set()
        self._refresh_filter_button()
        self._update_inference_buttons()
        self.status_label.setText(
            f"{tr('模型加载完成')}: {os.path.basename(detector.model_path)}")
        self._save_inference_settings()

    def _on_model_failed(self, message):
        self._detector = None
        self._update_inference_buttons()
        self.status_label.setText(tr("模型加载失败"))
        self._inference_warn(message)

    def _ensure_model_row(self):
        """把参数行插到画布上方（只建一次）。"""
        row = getattr(self, 'model_row', None)
        if row is not None:
            return row
        from ..model_row import ModelRow
        row = ModelRow(self)
        row.thresholds_changed.connect(self._on_model_thresholds_changed)
        row.input_size_changed.connect(self._on_model_size_changed)
        row.load_requested.connect(self._load_inference_model)
        row.filter_requested.connect(self._open_class_filter)
        row.closed.connect(self._update_inference_buttons)
        self.model_row = row
        layout = getattr(self, '_canvas_layout', None)
        if layout is not None:
            layout.insertWidget(0, row)
        return row

    # ---------- 类别筛选 ----------

    def _open_class_filter(self):
        """打开类别勾选面板；确定后更新筛选集合。"""
        detector = getattr(self, '_detector', None)
        if detector is None:
            self.status_label.setText(tr("请先加载模型"))
            return
        from ..class_filter_panel import ClassFilterPanel

        panel = getattr(self, '_class_filter_panel', None)
        if panel is None:
            panel = ClassFilterPanel(self)
            panel.selection_changed.connect(self._on_class_filter_changed)
            self._class_filter_panel = panel
        panel.retranslate()
        panel.set_class_names(detector.class_names)
        panel.set_selection(getattr(self, '_class_filter', None) or set())
        panel.exec_()

    def _on_class_filter_changed(self, selected):
        self._class_filter = {str(name) for name in (selected or [])}
        self._refresh_filter_button()
        self._save_inference_settings()

    def _refresh_filter_button(self):
        row = getattr(self, 'model_row', None)
        if row is None:
            return
        detector = getattr(self, '_detector', None)
        total = len(getattr(detector, 'class_names', None) or [])
        row.set_filter_summary(getattr(self, '_class_filter', None) or set(), total)

    def _filter_for_run(self):
        """传给引擎的筛选集合；空集表示不过滤。

        面板里「全不选」等于不过滤（引擎的约定），与面板提示保持一致。
        """
        selected = getattr(self, '_class_filter', None) or set()
        return selected or None

    def _on_model_thresholds_changed(self, conf, iou):
        """阈值即时生效：写回 detector 即可，不必重新加载模型。"""
        detector = getattr(self, '_detector', None)
        if detector is None:
            return
        detector.set_thresholds(conf, iou)
        self._save_inference_settings()

    def _on_model_size_changed(self, width, height):
        detector = getattr(self, '_detector', None)
        if detector is None:
            return
        detector.set_input_size((width, height))
        self._save_inference_settings()

    # ---------- 推理 ----------

    def _run_inference_current(self):
        """只推理当前显示图。"""
        index = self.current_background_index
        if index < 0 or not getattr(self, 'background_images', None):
            self.status_label.setText(tr("请先加载数据集"))
            return
        self._start_inference([index], current_only=True)

    def _run_inference_from_here(self):
        """从当前位置推理到列表末尾，弹出进度小窗。"""
        images = getattr(self, 'background_images', None) or []
        start = max(0, self.current_background_index)
        if not images or start >= len(images):
            self.status_label.setText(tr("请先加载数据集"))
            return
        self._start_inference(list(range(start, len(images))), current_only=False)

    def _start_inference(self, indices, current_only):
        if getattr(self, '_busy', False):
            return
        if getattr(self, '_run_worker', None) is not None:
            return
        detector = getattr(self, '_detector', None)
        if detector is None:
            self._inference_warn(tr("请先加载模型"))
            return
        # 参数行关着就不能推理；按钮虽已禁用，这里再挡一次，避免快捷键/代码路径绕过
        row = getattr(self, 'model_row', None)
        if row is not None and not row.isVisible():
            self.status_label.setText(tr("模型参数行已关闭，无法推理"))
            return
        images = getattr(self, 'background_images', None) or []
        paths = [images[i] for i in indices if 0 <= i < len(images)]
        if not paths:
            return
        from ..inference_workers import InferenceWorker

        self._save_inference_settings()
        worker = InferenceWorker(
            paths, indices, detector, self._filter_for_run(), self)
        self._run_worker = worker
        self._update_inference_buttons()
        if current_only:
            worker.finished_ok.connect(self._on_single_finished)
        else:
            progress = self._ensure_progress_dialog()
            progress.set_total(len(paths))
            progress.cancelled.connect(worker.cancel)
            worker.progress.connect(progress.set_progress)
            worker.finished_ok.connect(self._on_batch_finished)
            progress.show()
        worker.failed.connect(self._on_inference_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_run_worker(w))
        worker.start()

    def _cleanup_run_worker(self, worker):
        if worker is getattr(self, '_run_worker', None):
            self._run_worker = None
        worker.deleteLater()
        self._update_inference_buttons()

    def _on_inference_failed(self, message):
        self._close_progress_dialog()
        self.status_label.setText(tr("推理失败"))
        self._inference_warn(message)

    def _on_single_finished(self, results):
        self._apply_inference_results(results, self._replace_mode())
        self._update_inference_buttons()

    def _on_batch_finished(self, results):
        self._close_progress_dialog()
        self._apply_inference_results(results, self._replace_mode())
        self._update_inference_buttons()

    def _replace_mode(self):
        row = getattr(self, 'model_row', None)
        return bool(row is not None and row.replace_mode())

    def _update_inference_buttons(self):
        """参数行开着且有模型才让推理；关掉行等于停用推理。"""
        detector = getattr(self, '_detector', None)
        busy = getattr(self, '_run_worker', None) is not None
        row = getattr(self, 'model_row', None)
        row_open = bool(row is not None and row.isVisible())
        ready = detector is not None and not busy and row_open
        for name in ('model_single_btn', 'model_all_btn'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(ready)

    # ---------- 进度小窗 ----------

    def _ensure_progress_dialog(self):
        dialog = getattr(self, '_progress_dialog', None)
        if dialog is None:
            from ..inference_progress import InferenceProgressDialog
            dialog = InferenceProgressDialog(self)
            self._progress_dialog = dialog
        return dialog

    def _close_progress_dialog(self):
        dialog = getattr(self, '_progress_dialog', None)
        if dialog is not None:
            dialog.reset()
            dialog.hide()

    # ---------- 结果应用 ----------

    def _inference_warn(self, message):
        from ..dialog_helpers import warning
        warning(self, tr("错误"), message)

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

    def _inference_dataset_changed(self):
        """数据集是否已换：换了就丢弃迟到结果，不写到别的图上。"""
        current = tuple(getattr(self, 'background_images', None) or ())
        return current != getattr(self, '_inference_dataset', None)

    # ---------- 配置持久化 ----------

    def _save_inference_settings(self):
        from ...core import config_manager
        detector = getattr(self, '_detector', None)
        row = getattr(self, 'model_row', None)
        if detector is None or row is None:
            return
        conf, iou = row.thresholds()
        width, height = row.input_size()
        config_manager.save_all(inference_settings={
            "model_path": detector.model_path,
            "conf_threshold": conf,
            "iou_threshold": iou,
            "input_size": (width, height),
        })
