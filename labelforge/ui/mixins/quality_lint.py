"""标注质检 UI：按钮入口、后台扫描生命周期、双击跳图、忽略误报。"""
from ..i18n import t as tr

_LINT_REFRESH_MAX_IMAGES = 64


class QualityLintMixin:
    def _get_lint_ignored_rules(self):
        rules = getattr(self, '_lint_ignored_rules', None)
        if not isinstance(rules, dict):
            from ...core import config_manager
            rules = dict(config_manager.load_all().get('lint_ignored_rules') or {})
            self._lint_ignored_rules = rules
        return rules

    def _save_lint_ignored_rules(self, rules):
        from ...core import config_manager
        self._lint_ignored_rules = dict(rules or {})
        config_manager.save_all(lint_ignored_rules=self._lint_ignored_rules)

    def _filtered_lint_result(self, result, rules):
        """按忽略规则过滤结果：汇总计数必须重建（否则整类忽略后残留旧计数）。"""
        from ...engine.quality_lint import filter_ignored_issues, summarize_issues
        filtered = filter_ignored_issues(result.get('issues') or [], rules)
        summary = {k: 0 for k in (result.get('summary') or {})}
        summary.update(summarize_issues(filtered))
        summary['scanned_images'] = (result.get('summary') or {}).get('scanned_images', 0)
        out = {'issues': filtered, 'summary': summary}
        if result.get('error'):
            out['error'] = True
        return out

    def _open_quality_lint(self):
        if getattr(self, '_busy', False):
            return
        if not getattr(self, 'background_images', None):
            self.status_label.setText(tr("请先加载数据集"))
            return
        # 已开过质检弹窗先关掉，避免重复弹窗（旧弹窗结果过期仍可跳图/删除）
        self._close_lint_dialog()
        self._lint_refresh_pending = set()
        self._lint_refresh_full = False
        from ...engine.quality_lint import QualityLintWorker
        from ..quality_lint_dialog import QualityLintDialog

        memory_boxes = {
            idx: list(boxes) for idx, boxes in
            (getattr(self, 'detection_boxes_dict', None) or {}).items()
            if boxes
        }
        ignored_rules = self._get_lint_ignored_rules()
        total = len(self.background_images)
        worker = QualityLintWorker(tuple(self.background_images), memory_boxes, self)
        dialog = QualityLintDialog(self, total=total)
        dialog.set_progress(0, total)
        worker.lint_progress.connect(dialog.set_progress)

        def _finish(result, d=dialog, rules=ignored_rules):
            if d is not getattr(self, '_lint_dialog', None):
                return
            d.set_result(self._filtered_lint_result(result, rules))
            self._cleanup_lint_worker(getattr(self, '_lint_worker', None))

        worker.lint_finished.connect(_finish)
        dialog.issue_activated.connect(self._jump_to_lint_issue)
        dialog.ignore_requested.connect(
            lambda issue, scope, d=dialog: self._ignore_lint_issue(d, issue, scope))
        dialog.ignore_many_requested.connect(
            lambda issues, scope, d=dialog: self._ignore_lint_issues(d, issues))
        dialog.delete_requested.connect(
            lambda items, target, d=dialog:
            self._delete_cross_label_overlaps(d, items, target))
        dialog.mark_keep_requested.connect(
            lambda items, d=dialog: self._mark_lint_tiny_keep(d, items))
        dialog.rejected.connect(lambda w=worker: self._cleanup_lint_worker(w))
        self._lint_worker = worker
        self._lint_dialog = dialog
        worker.start()
        dialog.show()

    def _mark_lint_tiny_keep(self, dialog, issues):
        """标记保留：写 flags.tiny_keep 到 sidecar 与内存，再把问题从弹窗移走。

        与「忽略」不同，豁免跟着框走（存在 JSON 里），框改尺寸仍生效；
        磁盘与内存必须一起改，否则已加载图重扫时以内存为准，豁免不生效。
        """
        issues = [i for i in (issues or []) if i.get('kind')]
        if not issues:
            return
        from ...engine.quality_lint import mark_tiny_keep
        memory_boxes = {
            idx: list(boxes) for idx, boxes in
            (getattr(self, 'detection_boxes_dict', None) or {}).items()
            if boxes
        }
        result = mark_tiny_keep(
            issues, list(getattr(self, 'background_images', None) or []),
            memory_boxes=memory_boxes)
        failed = result.get('failed') or []
        if failed:
            self.status_label.setText(tr("标记失败") + f": {len(failed)}")
        # 内存框引用是 detection_boxes_dict 里的同一批 dict，mark_tiny_keep
        # 已就地打上 flags；当前显示图需要重取，画布选中态不变。
        current = getattr(self, 'current_background_index', -1)
        if isinstance(current, int) and current in memory_boxes:
            self.detection_boxes = list(memory_boxes.get(current) or [])
        dialog._remove_issues(issues)
        marked = result.get('marked') or 0
        text = (tr("已标记保留") + f" · {marked}") if not failed else tr("标记失败")
        if not failed and self._maybe_advance_lint_issue(dialog, set_status=False):
            text += f" · {tr('已跳到下一张问题图')}"
        self.status_label.setText(text)

    def _ignore_lint_issue(self, dialog, issue, scope):
        """记录忽略规则：scope='kind' 整类忽略，'item' 仅此 detail。"""
        kind = issue.get('kind')
        if not kind:
            return
        rules = {k: list(v) for k, v in self._get_lint_ignored_rules().items()}
        keys = rules.setdefault(kind, [])
        key = '*' if scope == 'kind' else str(issue.get('detail', ''))
        if not key:
            return
        if key not in keys:
            keys.append(key)
            keys.sort()
        self._save_lint_ignored_rules(rules)
        if key == '*':
            removed = [i for i in dialog._all_issues if i.get('kind') == kind]
        else:
            removed = [issue]
        dialog._remove_issues(removed)
        text = tr("已忽略")
        if self._maybe_advance_lint_issue(dialog, set_status=False):
            text += f" · {tr('已跳到下一张问题图')}"
        self.status_label.setText(text)

    def _ignore_lint_issues(self, dialog, issues):
        """多选忽略：每条记录 detail 规则。"""
        issues = [i for i in (issues or []) if i.get('kind') and i.get('detail')]
        if not issues:
            return
        rules = {k: list(v) for k, v in self._get_lint_ignored_rules().items()}
        for issue in issues:
            keys = rules.setdefault(issue['kind'], [])
            key = str(issue['detail'])
            if key not in keys:
                keys.append(key)
        for keys in rules.values():
            keys.sort()
        self._save_lint_ignored_rules(rules)
        dialog._remove_issues(issues)
        text = tr("已忽略")
        if self._maybe_advance_lint_issue(dialog, set_status=False):
            text += f" · {tr('已跳到下一张问题图')}"
        self.status_label.setText(text)

    def _cleanup_lint_worker(self, worker=None):
        worker = worker or getattr(self, '_lint_worker', None)
        if worker is not None:
            try:
                if worker.isRunning():
                    worker.requestInterruption()
                    # 等 worker 退出，避免销毁运行中的线程/向已关闭弹窗发信号
                    worker.wait(5000)
            except Exception:
                pass
        if worker is getattr(self, '_lint_worker', None):
            self._lint_worker = None

    def _close_lint_dialog(self):
        """关闭质检弹窗并中断扫描（切数据集/关窗时避免旧结果跳错图）。"""
        self._cleanup_lint_worker()
        dialog = getattr(self, '_lint_dialog', None)
        self._lint_dialog = None
        if dialog is not None:
            try:
                dialog.close()
            except Exception:
                pass

    def _notify_lint_boxes_changed(self, index=None):
        """标注变化后实时刷新质检弹窗中该图的问题（去抖 250ms）。"""
        dialog = getattr(self, '_lint_dialog', None)
        if dialog is None:
            return
        check = getattr(dialog, 'isVisible', None)
        if callable(check) and not check():
            return
        if index is None:
            index = getattr(self, 'current_background_index', -1)
        try:
            index = int(index)
        except (TypeError, ValueError):
            return
        if index < 0:
            return
        pending = getattr(self, '_lint_refresh_pending', None)
        if not isinstance(pending, set):
            pending = set()
            self._lint_refresh_pending = pending
        if len(pending) > _LINT_REFRESH_MAX_IMAGES:
            # 积压过多（如整库标签重命名）：不丢事件，改为整库重扫一次
            pending.clear()
            self._lint_refresh_full = True
        else:
            pending.add(index)
        timer = getattr(self, '_lint_refresh_timer', None)
        if timer is None:
            try:
                from PySide6.QtCore import QTimer
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.setInterval(250)
                timer.timeout.connect(self._flush_lint_refresh)
            except Exception:
                timer = False
            self._lint_refresh_timer = timer
        if timer:
            timer.start()
        else:
            self._flush_lint_refresh()

    def _flush_lint_refresh(self):
        """对去抖收集到的图做单图重扫并刷新弹窗；积压过多时整库重扫一次。"""
        pending = getattr(self, '_lint_refresh_pending', None)
        full = getattr(self, '_lint_refresh_full', False)
        if not pending and not full:
            return
        self._lint_refresh_pending = set()
        self._lint_refresh_full = False
        dialog = getattr(self, '_lint_dialog', None)
        if dialog is None:
            return
        check = getattr(dialog, 'isVisible', None)
        if callable(check) and not check():
            return
        images = list(getattr(self, 'background_images', None) or [])
        boxes_dict = getattr(self, 'detection_boxes_dict', None) or {}
        current = getattr(self, 'current_background_index', -1)
        rules = self._get_lint_ignored_rules()
        if full:
            from ...engine.quality_lint import lint_dataset, filter_ignored_issues
            memory = {idx: list(boxes) for idx, boxes in boxes_dict.items()} \
                if isinstance(boxes_dict, dict) else {}
            if isinstance(current, int) and 0 <= current < len(images):
                memory[current] = list(getattr(self, 'detection_boxes', None) or [])
            result = lint_dataset(images, memory_boxes=memory)
            issues = filter_ignored_issues(result.get('issues') or [], rules)
            indexes = set(range(len(images)))
            dialog.refresh_issues_for_images(indexes, issues)
            self._maybe_advance_lint_issue(dialog, indexes)
            return
        from ...engine.quality_lint import lint_single_image, filter_ignored_issues
        refreshed = []
        for index in sorted(pending):
            if not (0 <= index < len(images)):
                continue
            memory = {}
            if isinstance(boxes_dict, dict) and index in boxes_dict:
                memory[index] = list(boxes_dict.get(index) or [])
            if index == current:
                memory[index] = list(getattr(self, 'detection_boxes', None) or [])
            issues = filter_ignored_issues(
                lint_single_image(index, images[index], memory), rules)
            refreshed.extend(issues)
        dialog.refresh_issues_for_images(pending, refreshed)
        self._maybe_advance_lint_issue(dialog, pending)

    def _maybe_advance_lint_issue(self, dialog=None, refreshed_indexes=None,
                                  set_status=True):
        """当前图问题清空后，自动切到下一张仍有问题的图（按索引顺序，末尾回绕）。

        判断依据是表格当前可见问题（已筛选），这样筛选某类问题时，
        当前图该类问题清完即跳，不会因其它类问题卡住。
        """
        if getattr(self, '_busy', False):
            return False
        dialog = dialog or getattr(self, '_lint_dialog', None)
        if dialog is None:
            return False
        check = getattr(dialog, 'isVisible', None)
        if callable(check) and not check():
            return False
        current = getattr(self, 'current_background_index', -1)
        if not isinstance(current, int) or current < 0:
            return False
        if refreshed_indexes is not None:
            try:
                if current not in set(refreshed_indexes):
                    return False
            except TypeError:
                return False
        visible = getattr(dialog, 'visible_issues', None)
        issues = visible() if callable(visible) else None
        if issues is None:
            issues = [i for i in (getattr(dialog, '_all_issues', None) or [])
                      if isinstance(i, dict)]
        issues = [i for i in issues if isinstance(i, dict)]
        if any(i.get('image_index') == current for i in issues):
            return False
        images = list(getattr(self, 'background_images', None) or [])
        candidates = sorted({
            i.get('image_index') for i in issues
            if isinstance(i.get('image_index'), int)
            and i.get('image_index') != current
            and 0 <= i.get('image_index') < len(images)
        })
        if not candidates:
            return False
        next_index = next((i for i in candidates if i > current), candidates[0])
        if next_index == current:
            return False
        switch = getattr(self, 'switch_background_to_index', None)
        if not callable(switch):
            return False
        switch(next_index)
        focus = getattr(dialog, 'focus_image', None)
        if callable(focus):
            focus(next_index)
        if set_status and hasattr(self, 'status_label'):
            self.status_label.setText(tr("已跳到下一张问题图"))
        return True

    def _cleanup_delete_worker(self):
        """关闭窗口前中断并等待删除 worker，避免销毁运行中的线程。

        仍在运行（超时）时返回 False，调用方应推迟关闭。
        """
        worker = getattr(self, '_delete_worker', None)
        if worker is None:
            return True
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(5000)
            still_running = worker.isRunning()
        except Exception:
            still_running = False
        if still_running:
            return False
        self._delete_worker = None
        self._busy = False
        self._lint_removal_busy = False
        return True

    def _delete_cross_label_overlaps(self, dialog, selected, target_label):
        """右键删除：只删选中问题里目标类的重叠框（仅涉及图）。"""
        target_label = str(target_label or '').strip()
        selected = [i for i in (selected or [])
                    if i.get('kind') == 'cross_label_overlap']
        if not target_label or not selected:
            return
        if getattr(self, '_busy', False):
            return
        images = list(getattr(self, 'background_images', None) or [])
        if not images:
            self.status_label.setText(tr("请先加载数据集"))
            return
        from .. import dialog_helpers
        count = len(selected)
        message = tr(
            "将删除选中 {n} 项中 '{label}' 类与异类重叠的检测框。\n"
            "此操作不可撤销，是否继续？"
        ).replace("{n}", str(count)).replace("{label}", target_label)
        reply = dialog_helpers.question(
            self, tr("确认删除"), message,
            dialog_helpers.QMessageBox.Yes | dialog_helpers.QMessageBox.No,
            dialog_helpers.QMessageBox.No)
        if reply != dialog_helpers.QMessageBox.Yes:
            return
        self._start_cross_label_delete(dialog, selected, target_label, images)

    def _start_cross_label_delete(self, dialog, selected, target_label, images):
        from ...engine.quality_lint import CrossLabelDeleteWorker
        from ..dialogs import ProgressDialogFactory
        existing = getattr(self, '_delete_worker', None)
        if existing is not None and existing.isRunning():
            return
        current = getattr(self, 'current_background_index', -1)
        if current >= 0:
            self.detection_boxes_dict[current] = list(self.detection_boxes)
        memory_boxes = {
            idx: list(boxes) for idx, boxes in
            (getattr(self, 'detection_boxes_dict', None) or {}).items()
            if idx in {i.get('image_index') for i in selected}
        }
        # worker 按涉及图片数报进度（非问题条数），进度条上限必须一致
        image_count = len({i.get('image_index') for i in selected
                           if i.get('image_index') is not None})
        progress = ProgressDialogFactory.create_progress_dialog(
            self, tr("删除进度"),
            f"{tr('正在删除')} '{target_label}' {tr('重叠框...')}",
            max(1, image_count))
        progress.show()
        worker = CrossLabelDeleteWorker(
            selected, target_label, images, memory_boxes, self)
        worker.delete_progress.connect(
            lambda done, total, p=progress: self._update_delete_progress(p, done, total))
        worker.delete_finished.connect(
            lambda result, d=dialog, p=progress, w=worker, t=target_label:
            self._on_cross_label_delete_finished(d, p, w, t, result))
        progress.canceled.connect(worker.requestInterruption)
        self._delete_worker = worker
        self._delete_progress = progress
        self._busy = True
        # 删除 worker 后台改写 sidecar：期间主线程保存/切图一律拒绝，避免并发写坏 JSON
        self._lint_removal_busy = True
        worker.start()

    def _update_delete_progress(self, progress, done, total):
        progress.setMaximum(max(1, total))
        progress.setValue(done)
        progress.setLabelText(f"{tr('正在删除')} {done}/{total}")

    def _on_cross_label_delete_finished(self, dialog, progress, worker,
                                        target_label, result):
        try:
            progress.close()
        except Exception:
            pass
        if worker is getattr(self, '_delete_worker', None):
            self._delete_worker = None
        self._delete_progress = None
        self._busy = False
        self._lint_removal_busy = False
        removed_boxes = dict(result.get('removed_boxes') or {})
        if removed_boxes:
            self._sync_boxes_after_delete(removed_boxes)
        refreshed = dict(result.get('refreshed_issues') or {})
        jumped = False
        if dialog is getattr(self, '_lint_dialog', None):
            from ...engine.quality_lint import filter_ignored_issues
            rules = self._get_lint_ignored_rules()
            dialog.refresh_issues_for_images(refreshed.keys(), [
                issue for issues in refreshed.values()
                for issue in filter_ignored_issues(issues, rules)])
            jumped = self._maybe_advance_lint_issue(
                dialog, set_status=False)
        removed = int(result.get('removed', 0) or 0)
        changed = int(result.get('images_changed', 0) or 0)
        failed = len(result.get('failed') or [])
        if result.get('interrupted'):
            text = (f"{tr('删除已中断')}: {removed} {tr('个框')} / "
                    f"{changed} {tr('张图')}")
        else:
            text = (f"{tr('已删除')} '{target_label}' {removed} {tr('个重叠框')} / "
                    f"{changed} {tr('张图')}")
        if failed:
            text += f" · {failed} {tr('张写入失败')}"
        if jumped:
            text += f" · {tr('已跳到下一张问题图')}"
        self.status_label.setText(text)

    def _sync_boxes_after_delete(self, removed_boxes):
        """把删除结果同步到内存：已加载图移除对应框并刷新画布/统计。"""
        from ...engine.quality_lint import box_signature
        boxes_dict = getattr(self, 'detection_boxes_dict', None)
        if not isinstance(boxes_dict, dict):
            return
        changed_indexes = set()
        for index, boxes in removed_boxes.items():
            if index not in boxes_dict:
                continue
            targets = {}
            for box in boxes or []:
                sig = box_signature(box)
                if sig is not None:
                    targets[sig] = targets.get(sig, 0) + 1
            if not targets:
                continue
            kept = []
            for box in boxes_dict.get(index) or []:
                sig = box_signature(box)
                if sig is not None and targets.get(sig, 0) > 0:
                    targets[sig] -= 1
                    continue
                kept.append(box)
            if len(kept) != len(boxes_dict.get(index) or []):
                boxes_dict[index] = kept
                changed_indexes.add(index)
        if not changed_indexes:
            return
        # 跨类删除是已确认的不可逆操作（见 _delete_cross_label_overlaps 提示）。
        # 撤销栈里还留着删除前的快照，Ctrl+Z 会把删掉的框复活并在下次保存写回盘。
        undo_manager = getattr(self, 'undo_manager', None)
        if undo_manager is not None and hasattr(undo_manager, 'clear'):
            try:
                undo_manager.clear()
            except Exception:
                pass
        current = getattr(self, 'current_background_index', -1)
        if current in changed_indexes:
            self.detection_boxes = list(boxes_dict.get(current) or [])
            canvas = getattr(self, 'canvas', None)
            if canvas is not None:
                canvas.selected_box = None
                canvas.selected_boxes = []
                if hasattr(canvas, 'update_status_label'):
                    canvas.update_status_label()
                canvas.update()
        self._dataset_stats_dirty = True
        if hasattr(self, 'update_label_list'):
            self.update_label_list()
        if hasattr(self, 'update_file_count'):
            self.update_file_count()

    def _jump_to_lint_issue(self, issue):
        """双击问题行：切图并选中问题框（删除 worker 改写 sidecar 期间禁止切图）。"""
        if getattr(self, '_lint_removal_busy', False):
            return
        index = issue.get('image_index')
        if index is None or not (0 <= index < len(self.background_images)):
            return
        if index != self.current_background_index:
            self.switch_background_to_index(index)
        box_index = issue.get('box_index')
        if (box_index is not None and
                0 <= box_index < len(self.detection_boxes)):
            self.canvas.selected_box = box_index
            self.canvas.selected_boxes = [box_index]
            self.canvas.update_status_label()
            self.canvas.update()
