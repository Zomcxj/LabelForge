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

    def _inference_allowed_by_mode(self):
        """当前模式允许推理吗——**贴图模式一律不准**。

        用户第 10 轮反馈：「切到贴图模式能够关闭，但是还能手动打开推理，我要
        直接禁用」。原先只做了 `row.hide()`，而参数行的第一个图标在贴图模式下
        照样能把行 `show()` 回来，行一显示两个推理按钮就被重新 enable——那是
        「收起」，不是「禁用」。

        判据收拢到这里，按钮状态（`_update_inference_buttons`）与实际开跑
        （`_start_inference`）都走它，保证「看着灰」和「真跑不了」永远一致。
        """
        return getattr(self, 'edit_mode', 'annotate') != 'paste'

    def _toggle_model_row(self):
        """第一个图标：显示/隐藏参数行；关掉后不能推理。

        贴图模式下**直接拒绝**：推理与贴图是两件事（用户第 10 轮要求），贴图
        模式下这个图标不该还有把人带回推理的能力。
        """
        if not self._inference_allowed_by_mode():
            self.status_label.setText(tr("贴图模式下不可推理，请切回标注模式"))
            return
        row = self._ensure_model_row()
        if row.isVisible():
            row.hide()
        else:
            row.show()
        self._update_inference_buttons()

    def _sync_model_row_to_edit_mode(self):
        """贴图模式自动收起参数行并停用推理，切回标注模式再恢复。

        用户要求「切到贴图模式时自动收起模型参数行/禁用推理」。贴图模式下
        `_update_inference_buttons()` 会因为模式条件把两个推理按钮钉死在
        disable，所以**即使有人把参数行弄出来了也推不了**（第 10 轮：原先靠
        "收起"当禁用，参数行图标一按就能恢复推理）。

        特意**记住用户原本的状态**：他若在标注模式下自己收了参数行，切回时也
        不该替他展开——模式切换只管「贴图模式下必须是收起的」这一条。
        """
        row = getattr(self, 'model_row', None)
        if row is None:
            return
        if getattr(self, 'edit_mode', 'annotate') == 'paste':
            if row.isVisible():
                # 只在「本来是开着的」时记一笔，避免把贴图模式下的隐藏状态记成
                # 用户的偏好，切回标注模式后参数行就再也回不来了。
                self._model_row_open_before_paste = True
            row.hide()
        elif getattr(self, '_model_row_open_before_paste', False):
            row.show()
            self._model_row_open_before_paste = False
        self._update_inference_buttons()

    # ---------- 加载模型 ----------

    def _load_inference_model(self):
        """选 .onnx 并后台加载；成功后填入参数行、启用两个推理图标。"""
        if getattr(self, '_busy', False):
            return
        if getattr(self, '_model_worker', None) is not None:
            return
        from PySide6.QtWidgets import QFileDialog

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
        # 不要把这里当成「数据集快照」定稿：加载模型与「跑哪份数据集」是两回事，
        # 在这里记下来会让「先加载模型、再切数据集」之后的推理结果全被丢弃。
        # 快照在 `_start_inference` 里按当次任务记。
        row = self._ensure_model_row()
        row.set_model(
            os.path.basename(detector.model_path), detector.conf_threshold,
            detector.iou_threshold, size, detector.active_providers,
            dynamic=detector.is_dynamic_input(),
            end2end=self._model_format == FORMAT_V26)
        # 贴图模式下不该把参数行又弹出来（用户要求「切到贴图模式自动收起」）；
        # 走 mode-aware 的同步，标注模式才会真的 show。
        if getattr(self, 'edit_mode', 'annotate') == 'paste':
            row.hide()
        else:
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
        # 张数范围只在「从当前图推理」那条路径上有意义；值本身不用再回传宿主，
        # 运行时直接读 `row.batch_to_target()`，避免多存一份状态而与界面不同步。
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
        """打开范围面板（非模态）；确定后更新范围与窗口参数。

        非模态是**必需**的：弹窗开着时用户要点画布画区域，而应用级模态会
        让 Qt 丢弃发往画布的鼠标事件，「点编辑区域直接在图上画」会失效。
        因此这里用 `show()`，结果由 `finished_ok` 信号回传，不再有 `exec_()`
        的返回值可看。
        """
        if getattr(self, '_detector', None) is None:
            self.status_label.setText(tr("请先加载模型"))
            return
        from ..scope_panel import ScopePanel

        panel = getattr(self, '_scope_panel', None)
        if panel is None:
            panel = ScopePanel(parent=self)
            panel.region_delete_requested.connect(self._delete_scope_region)
            panel.add_region_requested.connect(self._on_add_region_requested)
            panel.region_lock_toggled.connect(self._on_region_lock_toggled)
            panel.finished_ok.connect(self._on_scope_panel_closed)
            panel.overlay_changed.connect(self._on_scope_overlay_changed)
            self._scope_panel = panel
        # 先把宿主当前的范围模式抓下来。下面 set_windows/set_region_count 等都会
        # 经 `overlay_changed` 回调 `_on_scope_overlay_changed`，而那里会读
        # **面板**当时的 mode 写回 `_scope_mode`——面板此刻还停在构造默认值
        # （全图），于是「上次选了区域 → 再开面板」会被静默重置成全图。
        # 抓成局部量、最后再落回去，面板一开就跟宿主状态一致。
        wanted_mode = getattr(self, '_scope_mode', MODE_FULL)
        # 装配期间 `overlay_changed` 会连发多次（`set_windows`/`set_regions`/
        # `set_mode` 各一次），而那时面板还停在构造默认值、宿主的
        # `_scope_mode` 也可能还没落定。这个标记让 `_on_scope_overlay_changed`
        # 在装配期间只重画画布、不写宿主状态也不开区域会话；装配结束再手动
        # 同步一次（见方法末尾）。
        self._scope_panel_configuring = True
        self._scope_panel_open = True
        try:
            panel.set_windows(self._effective_window_spec())
            panel.set_regions(self._region_boxes())
            panel.set_reference(self._reference_image_size())
            panel.set_anchor_rect(self._scope_anchor_rect())
            panel.retranslate()
            panel.set_mode(wanted_mode)
        finally:
            self._scope_panel_configuring = False
        # 装配完再按最终状态同步一次：模式、窗口参数、区域编辑态、画布提示。
        self._scope_mode = panel.mode()
        self._scope_window_spec = panel.window_spec()
        self._window_spec = panel.window_spec()
        self._scope_reference_size = self._reference_image_size()
        self._sync_scope_region_display()
        panel.show()
        panel.raise_()
        panel.activateWindow()

    def _scope_anchor_rect(self):
        """面板左上角该对齐的那块：**「标签管理」区**（用户明确要求）。

        「范围推理窗口左上角始终对齐标签管理区域的左上角」——所以锚点就是
        `label_group` 自己那块（含竖直方向：面板顶边 = 标签管理区顶边）。

        早先这里返回的是「背景图列表 ∪ 标签管理」合并出来的整条侧栏带、由
        `place_within` 竖直居中；现在改成左上角对齐，竖直居中那套不适用了。

        x 用侧栏的左边界（`bg.x()`），不能用 0：画布在主窗左侧，x=0 会把面板
        推到画布上（用户报过）。拿不到 `label_group` 时退回整条侧栏带；
        再拿不到返回 None，由调用方居中。
        """
        from PySide6.QtCore import QRect

        label = getattr(self, 'label_group', None)
        bg = getattr(self, 'bg_list_group', None)
        if label is not None:
            try:
                top_left = label.mapTo(self, label.rect().topLeft())
            except (AttributeError, TypeError):
                top_left = None
            if top_left is not None:
                # 宽度取侧栏的（面板比标签区宽，反正要溢出到右边空白）；
                # 其实就是给 place_within 一个「左上角在哪」的锚，宽度只用于
                # 算水平中心线，取侧栏宽度更稳。
                width = bg.width() if bg is not None else label.width()
                height = label.height()
                return QRect(top_left.x(), top_left.y(),
                             max(1, width), max(1, height))

        # 拿不到标签管理区：退回整条侧栏，至少别盖画布
        side = getattr(self, '_side_panel_geometry', None)
        return side() if callable(side) else None

    def _on_scope_overlay_changed(self):
        """面板状态变了 → 重画主画布上的滑窗网格与区域轮廓。

        网格不再有面板内的小预览，只能靠这条路径刷新。同时把当前窗口参数
        快照到宿主上，画布绘制时不依赖面板还活着（面板关掉后画布可能仍要
        重绘一帧）。

        **这条也是「面板改动即时生效」的唯一通道**：面板没有确定按钮了，
        所以「模式」「窗口参数」「区域编辑态」都在这儿落地。
        """
        panel = getattr(self, '_scope_panel', None)
        if panel is not None:
            # 装配期间面板停在构造默认值，写回宿主会把「上次选的模式」冲掉；
            # 那时只重画画布，状态由 `_open_scope_panel` 装配完统一落。
            if not getattr(self, '_scope_panel_configuring', False):
                from ...engine.inference_scope import MODE_WINDOW

                self._scope_window_spec = panel.window_spec()
                self._scope_mode = panel.mode()
                # 宿主的窗口参数也跟着走：关掉面板后 `_effective_window_spec()`
                # 读的就是它，不更新会让「面板里改的参数」在关窗后失效。
                self._window_spec = panel.window_spec()
        self._sync_scope_region_display()

    def _on_add_region_requested(self):
        """面板「画一个区域」按钮：点一下，下一次在图上拖就画出一个区域。

        用户要求「画区域也加一个按钮啊 点一下画一个」——取代原先「选中区域
        就自动进入绘制会话」的模型。会话每个左键都被区域吃掉，标签框就再也
        点不到；换成按需的一次性按钮之后，画布其余时间和平时完全一样。

        画布没有 `_start_add_region`（测试替身/其它宿主）时安全退化。
        """
        canvas = getattr(self, 'canvas', None)
        if canvas is None or not hasattr(canvas, '_start_add_region'):
            return
        if getattr(self, 'region_fixed', False):
            self.status_label.setText(tr("区域已固定，先解锁再添加"))
            return
        canvas._start_add_region()
        self.status_label.setText(tr("在图上拖动即可添加区域"))

    def _on_region_lock_toggled(self, index, locked):
        """面板某条区域的锁按钮被切换：写到画布那条区域上并刷新。

        锁定的含义见 `canvas_menu.set_region_locked()`：不锁 → 画在检测框之上
        且可改；锁上 → 落到检测框之下且不可改。**锁只是不许在画布上改，区域
        照常参与推理**，所以这里不碰 `_scope_for_run()` 取的那份区域列表。
        """
        canvas = getattr(self, 'canvas', None)
        if canvas is None or not hasattr(canvas, 'set_region_locked'):
            return
        if not canvas.set_region_locked(index, locked):
            return
        panel = getattr(self, '_scope_panel', None)
        if panel is not None:
            panel.set_regions(self._region_boxes())
        self._refresh_scope_button()

    def _on_scope_panel_closed(self, accepted):
        """面板关闭（只剩右上角 X 这一条路径）：撤掉画布上的范围提示。

        **没有「取消」了，所以关窗不回滚**：面板上所有的改动在改的那一刻就已
        经经 `overlay_changed` 写回宿主，这里不再做「确定才生效」那套搬运。
        """
        self._scope_panel_open = False
        canvas = getattr(self, 'canvas', None)
        if canvas is not None and hasattr(canvas, '_cancel_region_drawing'):
            # 面板关掉时若正等着画区域，把它收掉：绘制态是面板按钮开的，
            # 面板没了就不该还挂着「下一次拖动会画区域」。
            canvas._cancel_region_drawing()
        self._sync_scope_region_display()
        # 面板里可能刚改过滑窗参数，落一次盘记住它。
        self._save_inference_settings()
        self._refresh_scope_button()

    def _on_region_changed(self):
        """区域增删/落笔后回调：刷新状态栏、区域明细与范围摘要。"""
        self._refresh_scope_button()
        panel = getattr(self, '_scope_panel', None)
        if panel is not None:
            # 用 set_regions（而不是只报个数）：面板要显示每条区域的坐标，
            # 画完一个就该在列表里看到新的一行。
            #
            # 只有**推理区域**有面板反馈（第 11 轮去掉贴图区域面板后更纯粹）：
            # 贴图区域在画布上直接增删，没有面板要同步，这里不必按模式分流。
            panel.set_regions(self._region_boxes())
        self.status_label.setText(tr("区域绘制结束"))

    def _delete_scope_region(self, index):
        """删除面板里第 index 条区域，并同步画布与列表。

        真正的删除交给画布（`_delete_region`）——它同时清掉选中态、正在画的
        临时状态并重绘。面板拿到的只是显示快照，自己 pop 会和画布对不上。
        画布没有这个方法时（测试替身/其它宿主）退化成直接改 `region_boxes`，
        保证这个功能不因为宿主差异整体失效。
        """
        regions = getattr(self, 'region_boxes', None)
        if not isinstance(regions, list) or not (0 <= index < len(regions)):
            return
        canvas = getattr(self, 'canvas', None)
        delete = getattr(canvas, '_delete_region', None)
        if callable(delete):
            delete(index)
        else:
            regions.pop(index)
        panel = getattr(self, '_scope_panel', None)
        if panel is not None:
            panel.set_regions(self._region_boxes())
        if canvas is not None and hasattr(canvas, 'update'):
            canvas.update()
        self._refresh_scope_button()

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
            windows = self._window_spec_for_run()
            # 目前不可达：`_window_spec_for_run()` 三条分支都回非空表。
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
        """区域轮廓只在「面板开着 **且** 选了区域」时画。

        两条要求合起来：
        - 用户要「滑窗和区域各自的内容显示在画布上」：选了滑窗就只该看到滑窗
          网格，不能还挂着区域框。
        - 用户要「点确定关掉面板后区域框不该还留在画布上」：所以关掉面板就
          收起轮廓（面板只是**调范围**时的参照，不是常驻叠加层）。

        编辑能力不受影响：贴图模式下区域本来就看得到、画得动（见渲染侧的
        `_show_regions()`），这里的开关只管「标注模式下的范围提示」。
        """
        from ...engine.inference_scope import MODE_ROI

        mode = getattr(self, '_scope_mode', MODE_FULL)
        self._show_scope_regions = (bool(getattr(self, '_scope_panel_open', False))
                                    and mode == MODE_ROI)
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
        """从当前位置推理到目标张号（张数范围），弹出进度小窗。

        「目标张号」由参数行「推理范围」右侧的输入框给出，语义是**第几张**
        （1 基，含），不是「跑多少张」：当前是第 200 张时填 300 → 跑 200~300，
        填 100 → 跑 100~200。填 0 表示不限制，跑到列表末尾。
        """
        images = getattr(self, 'background_images', None) or []
        start = max(0, self.current_background_index)
        if not images or start >= len(images):
            self.status_label.setText(tr("请先加载数据集"))
            return
        target = self._batch_to_index()
        if target is None:
            indices = list(range(start, len(images)))
        else:
            # 目标是 1 基张号 → 0 基下标；与起点构成闭区间，方向自动决定，
            # 所以「往回跑」（填比当前小的号）也能用。
            end = min(max(0, target - 1), len(images) - 1)
            low, high = (end, start) if end < start else (start, end)
            indices = list(range(low, high + 1))
        if not indices:
            return
        self._start_inference(indices, current_only=False)

    def _batch_to_index(self):
        """参数行里的目标张号（1 基）；未填/为 0 时返回 None（表示到末尾）。"""
        row = getattr(self, 'model_row', None)
        getter = getattr(row, 'batch_to_target', None)
        if not callable(getter):
            return None
        try:
            value = int(getter())
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _start_inference(self, indices, current_only):
        if getattr(self, '_busy', False):
            return
        if getattr(self, '_run_worker', None) is not None:
            return
        detector = getattr(self, '_detector', None)
        if detector is None:
            self._inference_warn(tr("请先加载模型"))
            return
        # 贴图模式的提示要**先于**「参数行已关闭」：贴图模式下参数行正是被自动
        # 收起的，报「参数行已关闭」会让用户以为自己关掉了什么、去翻按钮。
        if not self._inference_allowed_by_mode():
            self.status_label.setText(tr("贴图模式下不可推理，请切回标注模式"))
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
        # 数据集快照在这里定稿：它表达的是「这轮跑的是哪份数据集」，用来丢弃
        # **跑到一半**换了数据集时迟到的结果。
        #
        # 原来这份快照只在加载模型时写（`_on_model_loaded`），于是「先加载模型、
        # 再切数据集」之后快照永远不等于当前数据集，`_inference_dataset_changed()`
        # 恒为 True，所有结果一进门就被当「过期的」丢掉——现象是日志/进度窗
        # 明明显示检测到框，却一张都没保存（用户报的 bug）。
        self._inference_dataset = tuple(images)
        # 逐张收尾用的三个集合必须**每轮重置**：上一轮若异常结束（比如加载模型
        # 失败、或另一个 worker 抢先），收尾没跑到就留下了旧键，新的一轮会把
        # 这些键当「已处理」跳过，表现成「这几张明明跑了却没应用/没刷新」。
        self._batch_handled_keys = set()
        self._batch_written_keys = set()
        self._batch_write_failed = set()
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
        from ...engine.inference_task import (InferenceError, merge_boxes,
                                             write_one_result)

        memory = getattr(self, 'detection_boxes_dict', None) or {}
        self._batch_handled_keys = set(
            getattr(self, '_batch_handled_keys', ())) | {key}
        replace_auto = self._replace_mode()
        class_filter = self._filter_for_run()
        # 合并一次，**磁盘与内存用同一份结果**：`write_one_result` 内部也按
        # 同样的参数合并，所以两者不会分叉。
        merged = merge_boxes(memory.get(key), result.boxes,
                             replace_auto=replace_auto,
                             class_filter=class_filter)
        try:
            wrote = write_one_result(result, existing=memory.get(key),
                                     replace_auto=replace_auto,
                                     class_filter=class_filter)
        except InferenceError:
            # 单张写失败不影响其他图；整批结束的汇总里会算进「写入失败」
            self._batch_write_failed = set(
                getattr(self, '_batch_write_failed', ())) | {key}
        else:
            if wrote:
                self._batch_written_keys = set(
                    getattr(self, '_batch_written_keys', ())) | {key}
            # 内存缓存同步：切图时的 `_load_detection_boxes_for_index` 只要
            # 发现缓存非空就直接用缓存、不读盘，所以这里不更新的话，用户切回
            # 这张图看到的还是跑批之前的旧框（圆点却是绿的）——用户报的
            # 「取消后切到那张图又恢复原来的状态」。
            if self.detection_boxes_dict is not None:
                self.detection_boxes_dict[key] = list(merged)
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
        # 收尾统一刷新：取消时**已跑完**的图也必须当场变绿，不能等用户切到那张
        # 才更新（用户报的「取消后没有刷新新的标签，切到那张图才更新圆点」）。
        #
        # 逐张路径虽然各自刷过一次，但那几次可能落在盘点之前/被扫描结果覆盖，
        # 这里按最终磁盘状态再刷一遍是幂等的，成本只是一次 json 解析。
        self._refresh_batch_rows(results)
        self._update_inference_buttons()

    def _refresh_batch_rows(self, results):
        """按磁盘最终状态刷新本轮涉及到的行的圆点，并同步当前图画布。

        两件事：
        - 圆点：对本轮每个 key 重新读盘判定状态。取消时后面那些没跑的图不在
          `results` 里，自然不会被刷成「已标注」。
        - 当前图画布：批量跑的是「当前图之后的若干张」，取消后用户往往还停在
          当前图。`_apply_inference_to_current` 只在当前图是刚跑完那张时才被调到，
          取消后缓存里有框、画布却还是旧的（用户报的「显示旧的框」）。这里
          从缓存回灌一次，保证画布与内存一致。
        """
        refreshed = False
        for key in list(results.keys()):
            path = getattr(results[key], 'path', None)
            if not path and 0 <= key < len(getattr(self, 'background_images', ()) or ()):
                path = self.background_images[key]
            if not path:
                continue
            self._refresh_background_item_status(key, path)
            refreshed = True
        if refreshed:
            self._sync_current_boxes_from_cache()

    def _sync_current_boxes_from_cache(self):
        """把当前图的框从逐图缓存回灌到画布（数据集没换时）。

        只在内存与缓存不一致时动，避免白白压一层撤销快照、也避免把用户
        正在编辑的框冲掉。
        """
        if self._inference_dataset_changed():
            return
        if getattr(self, '_is_delete_view', False):
            return
        index = getattr(self, 'current_background_index', -1)
        cache = getattr(self, 'detection_boxes_dict', None) or {}
        if not (0 <= index < len(getattr(self, 'background_images', ()) or ())):
            return
        if index not in cache:
            return
        boxes = list(cache.get(index) or [])
        if len(boxes) == len(self.detection_boxes or []):
            return
        self.detection_boxes = boxes
        canvas = getattr(self, 'canvas', None)
        if canvas is not None:
            canvas.selected_box = None
            canvas.selected_boxes = []
            if hasattr(canvas, 'update_status_label'):
                canvas.update_status_label()
            canvas.update()
        self.update_label_list()

    def _replace_mode(self):
        row = getattr(self, 'model_row', None)
        return bool(row is not None and row.replace_mode())

    def _update_inference_buttons(self):
        """参数行开着、有模型、且**当前模式允许推理**才让推理。

        模式条件是第 10 轮加的硬条件：只要在贴图模式，无论参数行是开是关，
        两个推理按钮都钉在 disable。这样「参数行图标」即便被以任何方式弄出来
        （快捷键、代码路径、以后新增的入口），也推不了。
        """
        detector = getattr(self, '_detector', None)
        busy = getattr(self, '_run_worker', None) is not None
        row = getattr(self, 'model_row', None)
        row_open = bool(row is not None and row.isVisible())
        ready = (detector is not None and not busy and row_open
                 and self._inference_allowed_by_mode())
        for name in ('model_single_btn', 'model_all_btn'):
            button = getattr(self, name, None)
            if button is not None:
                button.setEnabled(ready)
        # 张数范围的上限跟着数据集张数走，免得填出根本不存在的张号
        sync = getattr(row, 'set_batch_to_range', None)
        if callable(sync):
            sync(len(getattr(self, 'background_images', None) or []))

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
        if not merged and not (getattr(self, 'canvas_items', None) or []):
            # 无框也无贴图：既定的「不写盘」口径（`200ace1`：否则数据集里会多出
            # 一堆 shapes 为 [] 的空 json）。
            #
            # 但磁盘上可能还留着**这一轮刚清掉的旧内容**（覆盖模式下的上一轮
            # 自动框）。`remove_empty_sidecar` 只删「原文 shapes 为空」的文件，
            # 看到旧框会拒绝删，于是磁盘停在旧状态、图标也就一直显示「已标注」，
            # 要切图才变（用户报的现象）。这种情况先把清空后的状态写出去，让它
            # 真的变空，再删。
            #
            # 只在「盘上确实有内容」时才多这一次写：本来就空的文件直接删，
            # 不写（否则会给每张无检出的图都补一次无谓的写盘）。
            if self._sidecar_has_content(path):
                self.save_json(path, os.path.basename(path), "",
                               canvas_items=self.canvas_items, current_index=index)
            remove_empty_sidecar(path)
            self._refresh_background_item_status(index, path)
            return False
        # 传入画布贴图，避免重写 sidecar 时把贴图 shape 丢掉
        self.save_json(path, os.path.basename(path), "",
                       canvas_items=self.canvas_items, current_index=index)
        return True

    @staticmethod
    def _sidecar_has_content(image_path):
        """sidecar 里是否有 shape（读不出来/损坏一律当「有」，保守不删）。"""
        import json
        import os

        json_path = os.path.splitext(image_path)[0] + ".json"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return True
        shapes = data.get("shapes")
        if not isinstance(shapes, list):
            return True
        return bool(shapes)

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

    def _window_spec_for_run(self):
        """跑任务时该用的窗口参数：**面板当下的输入**优先。

        面板里的宽/高/重叠/最低可见比例是用户按这一眼要用的值，跑之前最后一刻
        改的那些必须生效。原先这里读 `_effective_window_spec()`（本次会话设过的
        /配置里存的/默认值），跟面板当下的输入是两回事——于是「改了参数，
        跑起来还是老样子」，界面显示新值、实际用旧值。

        面板没开着（比如程序启动后直接用存下来的参数跑）才回退到
        `_effective_window_spec()`。
        """
        panel = getattr(self, '_scope_panel', None)
        if panel is not None and getattr(self, '_scope_panel_open', False):
            try:
                spec = panel.window_spec()
            except Exception:
                spec = None
            if spec:
                return dict(spec)
        return self._effective_window_spec()

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
        from ..widgets.window_params import DEFAULT_MIN_VISIBLE
        return {"w": DEFAULT_WINDOW, "h": DEFAULT_WINDOW,
                "overlap": DEFAULT_OVERLAP,
                "min_visible": DEFAULT_MIN_VISIBLE}
