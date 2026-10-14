"""推理接线：工具栏图标、模型参数行、单图与批量推理。

模型只加载一次，阈值与输入尺寸改动即时写回 detector，不重建会话。
参数行是唯一的模型入口：关掉它就不能推理（加载按钮也在行内），再点一次图标恢复。
结果应用必须一次性完成：`_create_detection_box()` 每框都会刷新列表并写盘，
逐框调用会让整批推理产生成百上千次保存。这里只保存一次撤销快照、
刷新一次列表、写一次当前图 sidecar。
"""
import copy
import os

from ..i18n import t as tr
from ...engine.inference_scope import MODE_FULL

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
        """上次用过的输入尺寸；阈值与窗口参数另有来源，这里不给。

        阈值（置信度/NMS）**刻意不读回**：它们是「这次要怎么看」的临时
        旋钮，跨启动沿用会让用户对着自己忘了调过的值犯迷糊——框多框少
        说不清。不给键即可，worker 拿到 None 就走引擎默认值。
        """
        from ...core import config_manager
        settings = config_manager.load_all().get('inference_settings') or {}
        return {
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

    def _on_class_names_needed(self, model_path, suggested_path,
                               expected_count=None):
        """同级没有 txt 且模型也没有类别表：让用户手输并写入同级同名 txt。

        expected_count 是能从输出 shape 推出来的类别数，用来提示「该填几个」。
        取消即视为不支持推理：不加载模型、两个推理图标保持禁用。
        """
        from ..inference_workers import class_names_prompt

        self.status_label.setText(tr("模型缺少类别信息"))
        written = class_names_prompt(self, model_path, suggested_path,
                                     expected_count=expected_count)
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
        # 换模型后类别表可能完全不同，之前的勾选作废，回到默认全选
        self._class_filter = None
        self._refresh_filter_button()
        # 范围回到全图：ROI 是画布上的临时区域，换模型后不该沿用上一个范围
        self._scope_mode = MODE_FULL
        self._window_spec = self._effective_window_spec()
        self._refresh_scope_button()
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
        row.scope_requested.connect(self._open_scope_panel)
        row.closed.connect(self._update_inference_buttons)
        self.model_row = row
        layout = getattr(self, '_canvas_layout', None)
        if layout is not None:
            layout.insertWidget(0, row)
        return row

    # ---------- 类别筛选 ----------

    def _all_class_names(self):
        detector = getattr(self, '_detector', None)
        return [str(n) for n in (getattr(detector, 'class_names', None) or [])]

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
        # 没设过筛选就按全选显示：默认全部保留
        current = getattr(self, '_class_filter', None)
        panel.set_selection(current if current is not None
                            else detector.class_names)
        panel.exec_()

    def _on_class_filter_changed(self, selected):
        self._class_filter = {str(name) for name in (selected or [])}
        self._refresh_filter_button()
        self._save_inference_settings()

    def _refresh_filter_button(self):
        row = getattr(self, 'model_row', None)
        if row is None:
            return
        row.set_filter_summary(self._checked_classes(), len(self._all_class_names()))

    def _checked_classes(self):
        """当前勾选的类别；没设过就视为全选。"""
        current = getattr(self, '_class_filter', None)
        if current is None:
            return set(self._all_class_names())
        return set(current)

    def _filter_for_run(self):
        """传给引擎的筛选集合；None 表示不过滤。

        默认全选等于没有限制，这时回 None 让引擎走原来的路径；只有真的
        取消了某些类别才传具体集合去过滤。「全不选」传空集合 = 全部丢弃。

        类别表为空（模型没加载好 / 没有类别信息）时同样回 None：这时「勾了
        什么」无从谈起，若按空集合处理会把已有推理框全清掉。
        """
        names = self._all_class_names()
        if not names:
            return None
        selected = self._checked_classes()
        if selected >= set(names):
            return None
        return selected

    def _on_model_thresholds_changed(self, conf, iou):
        """阈值即时生效：写回 detector 即可，不必重新加载模型。"""
        detector = getattr(self, '_detector', None)
        if detector is None:
            return
        detector.set_thresholds(conf, iou)
        self._save_inference_settings()

    def _on_model_size_changed(self, width, height):
        """输入尺寸即时生效；静态轴模型的尺寸由模型自身决定。"""
        detector = getattr(self, '_detector', None)
        if detector is None:
            return
        detector.set_input_size((width, height))
        self._save_inference_settings()

    # ---------- 推理范围 ----------

    def _open_scope_panel(self):
        """打开范围面板；确定后更新范围与窗口参数。"""
        if getattr(self, '_detector', None) is None:
            self.status_label.setText(tr("请先加载模型"))
            return
        from ..scope_panel import ScopePanel

        panel = getattr(self, '_scope_panel', None)
        if panel is None:
            panel = ScopePanel(parent=self)
            panel.edit_regions_requested.connect(self._edit_regions_for_scope)
            self._scope_panel = panel
        panel.set_windows(self._effective_window_spec())
        panel.set_region_count(len(self._region_boxes()))
        panel.set_reference(self._reference_image_size())
        # 防御式取侧栏几何：`_side_panel_geometry` 来自 PanelsMixin，本 mixin
        # 不假定宿主一定混入了它（拿不到就退回居中，与改动前一致）。
        geometry_fn = getattr(self, '_side_panel_geometry', None)
        panel.set_anchor_rect(geometry_fn() if callable(geometry_fn) else None)
        panel.retranslate()
        panel.set_mode(getattr(self, '_scope_mode', MODE_FULL))
        panel.exec_()
        if panel.result() != panel.Accepted:
            return
        mode, windows = panel.selection()
        self._scope_mode = mode
        if windows:
            self._window_spec = windows
        self._save_inference_settings()
        self._refresh_scope_button()

    def _edit_regions_for_scope(self):
        """跳到贴图模式编辑区域；区域只在贴图模式可编辑。"""
        self.status_label.setText(tr("区域在贴图模式里绘制，切过去后不会丢失当前标注"))
        switch = getattr(self, '_set_edit_mode', None)
        if callable(switch):
            switch('paste', animated=True)

    def _region_boxes(self):
        """画布上的临时区域；没有则为空表。"""
        boxes = getattr(self, 'region_boxes', None)
        return list(boxes) if isinstance(boxes, list) else []

    def _reference_image_size(self):
        """ROI 是相对当前显示图量的；拿不到尺寸时返回 None。"""
        image = getattr(self, 'current_background', None)
        if image is None:
            return None
        try:
            width, height = int(image.width()), int(image.height())
        except (AttributeError, TypeError, ValueError):
            return None
        return (width, height) if width > 0 and height > 0 else None

    def _scope_for_run(self):
        """启动任务时的范围快照；全图返回 None。

        区域在这里深拷贝：任务跑起来后用户还会继续编辑画布上的区域，后台不能
        读到正在被改的列表。参考图尺寸一并记下，尺寸不符的图会单独报失败。
        """
        from ...engine.inference_scope import (
            MODE_ROI, MODE_WINDOW, ScopeError, normalize_scope,
        )

        mode = getattr(self, '_scope_mode', MODE_FULL)
        if mode == MODE_FULL:
            return None
        scope = {"mode": mode}
        if mode == MODE_WINDOW:
            windows = self._effective_window_spec()
            # 目前不可达：`_effective_window_spec()` 三条分支都回非空表。
            # 保留是为了以后改默认值（比如允许「不设参数」）时有个兜底，
            # 不然会带着空窗口表进后台。
            if not windows:
                raise ScopeError("滑窗范围缺少窗口参数")
            scope["windows"] = dict(windows)
        if mode == MODE_ROI:
            regions = self._region_boxes()
            if not regions:
                raise ScopeError(tr("没有可用区域，请先画好区域"))
            # 这里深拷贝是为了语义明确（scope 必须是快照）；即使去掉，
            # `normalize_scope()` 内部也会 deepcopy 两次，结果一样。
            scope["regions"] = copy.deepcopy(regions)
            reference = self._reference_image_size()
            if reference:
                scope["reference_size"] = reference
        # 用参考图尺寸先验一次，非法参数在启动时就报错，不留到后台
        return normalize_scope(scope, self._reference_image_size() or (1, 1))

    def _refresh_scope_button(self):
        row = getattr(self, 'model_row', None)
        self._sync_scope_region_display()
        if row is None:
            return
        from ...engine.inference_scope import (
            MODE_ROI, MODE_WINDOW,
        )

        mode = getattr(self, '_scope_mode', MODE_FULL)
        if mode == MODE_FULL:
            row.set_scope_summary(tr("全图"))
            return
        names = {MODE_WINDOW: "滑窗", MODE_ROI: "区域"}
        detail = ""
        if mode == MODE_WINDOW:
            spec = getattr(self, '_window_spec', None) or {}
            if spec:
                detail = f" {spec.get('w')}×{spec.get('h')}/{spec.get('overlap')}"
        row.set_scope_summary(tr(names[mode]), detail)

    def _sync_scope_region_display(self):
        """范围含区域时让画布在标注模式也画区域轮廓。

        只影响显示，不改 `edit_mode`：区域编辑仍然只在贴图模式，否则会
        影响贴图交互。
        """
        from ...engine.inference_scope import MODE_ROI

        mode = getattr(self, '_scope_mode', MODE_FULL)
        self._show_scope_regions = mode == MODE_ROI
        canvas = getattr(self, 'canvas', None)
        if canvas is not None and hasattr(canvas, 'update'):
            canvas.update()

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
        from ...engine.inference_scope import MODE_ROI, ScopeError
        from ..inference_workers import InferenceWorker

        # 范围在启动时定稿：非法配置（没画区域、窗口参数坏）在这里就拦下，
        # 不让它进后台变成「成功但 0 框」，那样覆盖模式会删掉原有自动标注。
        try:
            scope = self._scope_for_run()
        except ScopeError as exc:
            self.status_label.setText(tr("推理范围无效"))
            self._inference_warn(str(exc))
            return
        if scope is not None and scope.get("mode") == MODE_ROI:
            self._note_scope_replace(scope)

        self._save_inference_settings()
        worker = InferenceWorker(
            paths, indices, detector, self._filter_for_run(), scope=scope,
            parent=self)
        self._run_worker = worker
        self._update_inference_buttons()
        if current_only:
            worker.finished_ok.connect(self._on_single_finished)
        else:
            progress = self._ensure_progress_dialog()
            progress.set_total(len(paths))
            progress.cancelled.connect(worker.cancel)
            worker.progress.connect(progress.set_progress)
            # 逐张写盘：跑完一张就落地一张，json 一张张出来、圆点逐个变绿，
            # 取消或崩溃时已完成的图不会白跑。
            worker.result_ready.connect(self._on_result_ready)
            worker.finished_ok.connect(self._on_batch_finished)
            progress.show()
        worker.failed.connect(self._on_inference_failed)
        worker.finished.connect(lambda w=worker: self._cleanup_run_worker(w))
        worker.start()

    def _on_result_ready(self, key, result):
        """单张结果落地：写盘 + 刷新该行的状态圆点。

        逐张写盘是硬要求：json 一张张出来、圆点逐个变绿，用户随时能看到
        进度，取消或崩溃时已完成的图也不会白跑。**包括当前显示图**——批量
        推理的起点就是当前图，跳过它会出现「第一张一直不变绿、结束才绿」。

        当前图走 `_apply_inference_to_current`（正常保存流程：内存、画布、
        撤销快照、写盘、圆点一次到位），其余图直接写 sidecar。进度窗不是
        模态的，用户跑批期间可能切图，所以判断的是「此刻显示的是不是这张」，
        切到哪张就更新哪张。
        """
        if self._inference_dataset_changed():
            return
        if getattr(self, '_is_delete_view', False):
            return
        if result.error:
            return
        if key == self.current_background_index:
            # 当前图走正常保存流程：内存、画布、撤销快照、写盘、圆点一次到位
            self._batch_handled_keys = set(
                getattr(self, '_batch_handled_keys', ())) | {key}
            if self._apply_inference_to_current(result, self._replace_mode()):
                self._batch_written_keys = set(
                    getattr(self, '_batch_written_keys', ())) | {key}
            return
        from ...engine.inference_task import InferenceError, write_one_result

        memory = getattr(self, 'detection_boxes_dict', None) or {}
        self._batch_handled_keys = set(
            getattr(self, '_batch_handled_keys', ())) | {key}
        try:
            wrote = write_one_result(result, existing=memory.get(key),
                                     replace_auto=self._replace_mode(),
                                     class_filter=self._filter_for_run())
        except InferenceError:
            # 单张写失败不影响其他图；整批结束的汇总里会算进「写入失败」
            self._batch_write_failed = set(
                getattr(self, '_batch_write_failed', ())) | {key}
        else:
            if wrote:
                self._batch_written_keys = set(
                    getattr(self, '_batch_written_keys', ())) | {key}
        self._refresh_background_item_status(key, result.path)

    def _note_scope_replace(self, scope):
        """区域范围 + 覆盖模式时在状态栏提醒：区域外的旧自动框也会被删掉。

        只提醒、不弹确认框：每次跑区域都弹一次没人会读。范围面板里也写了
        同样的规则。
        """
        if not self._replace_mode():
            return
        self.status_label.setText(
            f"{tr('区域')}: {tr('覆盖会替换本图全部旧自动框，区域之外的也会被删除')}")

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
        # 逐张处理过的键：收尾时不能再处理一遍（补充模式会写重复框）
        handled = set(getattr(self, '_batch_handled_keys', ()))
        written = set(getattr(self, '_batch_written_keys', ()))
        failed = set(getattr(self, '_batch_write_failed', ()))
        self._batch_handled_keys = set()
        self._batch_written_keys = set()
        self._batch_write_failed = set()
        self._apply_inference_results(results, self._replace_mode(),
                                      skip_indices=handled,
                                      written_indices=written,
                                      failed_indices=failed)
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

    def _apply_inference_results(self, results, replace_auto, skip_indices=(),
                                 written_indices=(), failed_indices=()):
        """整批结束后的收尾：汇总提示。

        绝大多数图在 `_on_result_ready` 里已经逐张处理过了（skip_indices
        就是它们），**这里不能再处理一遍**：补充模式下重复合并会把同一批框
        写两遍，覆盖模式下则会白白多压一层撤销快照。

        单张推理（`current_only=True`）没接 `result_ready`，所以当前图在这里
        处理，skip_indices 为空。
        """
        if self._inference_dataset_changed():
            self.status_label.setText(tr("数据集已切换，推理结果已丢弃"))
            return
        if getattr(self, '_is_delete_view', False):
            return

        skip = set(skip_indices)
        current = self.current_background_index
        current_result = results.get(current)
        wrote_current = False
        if (current_result is not None and not current_result.error
                and current not in skip):
            wrote_current = self._apply_inference_to_current(
                current_result, replace_auto)

        written = len(set(written_indices)) + (1 if wrote_current else 0)
        failed = len(set(failed_indices))
        total_boxes = sum(len(r.boxes) for r in results.values() if not r.error)
        text = (f"{tr('推理完成')}: {len(results)} {tr('张图')} · "
                f"{total_boxes} {tr('个框')} · {written} {tr('张已写入')}")
        if failed:
            text += f" · {failed} {tr('张写入失败')}"
        self.status_label.setText(text)
        self._dataset_stats_dirty = True

    def _apply_inference_to_current(self, result, replace_auto):
        """当前图的框一次性替换：单次撤销快照 + 单次列表刷新 + 单次保存。

        返回是否真的写了盘（没框也没贴图时不写，与批量路径口径一致）。
        圆点由 `save_json` 负责刷新。
        """
        from ...engine.inference_task import merge_boxes, remove_empty_sidecar

        index = self.current_background_index
        merged = merge_boxes(self.detection_boxes, result.boxes,
                             replace_auto=replace_auto,
                             class_filter=self._filter_for_run())
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
            return False
        path = self.background_images[index]
        # 与批量一致：没框也没贴图就不写盘，避免留下 shapes 为 [] 的空 json
        if not merged and not (getattr(self, 'canvas_items', None) or []):
            remove_empty_sidecar(path)
            return False
        # 传入画布贴图，避免重写 sidecar 时把贴图 shape 丢掉
        self.save_json(path, os.path.basename(path), "",
                       canvas_items=self.canvas_items, current_index=index)
        return True

    def _inference_dataset_changed(self):
        """数据集是否已换：换了就丢弃迟到结果，不写到别的图上。"""
        current = tuple(getattr(self, 'background_images', None) or ())
        return current != getattr(self, '_inference_dataset', None)

    # ---------- 配置持久化 ----------

    def _save_inference_settings(self):
        """只持久化「下次打开还想用」的东西。

        明确**不存**置信度/NMS：它们每次启动都该回到默认值（用户要求），
        存了反而会在下次加载模型时被 `_saved_inference_settings` 之外的
        路径读回去。输入尺寸与窗口参数保留。
        """
        from ...core import config_manager
        detector = getattr(self, '_detector', None)
        row = getattr(self, 'model_row', None)
        if detector is None or row is None:
            return
        width, height = row.input_size()
        config_manager.save_all(inference_settings={
            "model_path": detector.model_path,
            "input_size": (width, height),
            # 只存窗口参数；范围模式不存，每次启动都回到全图，避免恢复出一个
            # 没有 ROI 的区域模式。
            "window_spec": dict(getattr(self, '_window_spec', None) or {}),
        })

    def _saved_window_spec(self):
        """上次用过的窗口参数；没存过则返回空表。"""
        from ...core import config_manager
        settings = config_manager.load_all().get('inference_settings') or {}
        spec = settings.get("window_spec")
        return dict(spec) if isinstance(spec, dict) else {}

    def _effective_window_spec(self):
        """实际要用的窗口参数：本次会话设过的优先，其次配置里存的，最后默认值。

        首次使用（没设过也没存过）不能回空表：面板会因此带着「没有窗口参数」
        进滑窗模式，确定后启动任务才报错。
        """
        from ..sliding_window_dialog import (
            DEFAULT_OVERLAP, DEFAULT_WINDOW,
        )

        current = getattr(self, '_window_spec', None)
        if current:
            return dict(current)
        saved = self._saved_window_spec()
        if saved:
            return saved
        return {"w": DEFAULT_WINDOW, "h": DEFAULT_WINDOW,
                "overlap": DEFAULT_OVERLAP}
