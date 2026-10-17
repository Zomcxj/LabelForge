"""Language switching and translated UI text refresh behavior."""


class TranslationMixin:
    def toggle_language(self):
        """切换中英文"""
        from .. import i18n
        from ...core import config_manager
        i18n.toggle_lang()
        config_manager.save_language(i18n.get_lang())
        self._refresh_ui_texts()
        lang_name = "Chinese" if i18n.get_lang() == "zh" else "English"
        self.status_label.setText(f"Language: {lang_name}")
        from PySide6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _refresh_ui_texts(self):
        """刷新所有界面文字"""
        from .. import i18n
        tr = i18n.t
        if hasattr(self, 'draw_box_btn'):
            sc = self._get_shortcut('draw_box')
            self.draw_box_btn.setText(f"{tr('绘制BOX')}({sc})")
            self.draw_box_btn.setToolTip(tr("绘制检测框"))
        self.auto_save_b_checkbox.setText(tr("自动保存B"))
        self.auto_save_p_checkbox.setText(tr("自动保存P"))
        if hasattr(self, 'edit_mode'):
            self.auto_save_b_checkbox.setText(f"{tr('自动保存B')}({tr('标注')})" if self.edit_mode == 'annotate' else tr("自动保存B"))
            self.auto_save_p_checkbox.setText(f"{tr('自动保存P')}({tr('贴图')})" if self.edit_mode == 'paste' else tr("自动保存P"))
        self.show_labels_checkbox.setText(tr("显示BOX"))
        self.show_label_names_checkbox.setText(tr("显示Label"))
        if hasattr(self, 'show_scores_checkbox'):
            self.show_scores_checkbox.setText(tr("显示置信度"))
            self.show_scores_checkbox.setToolTip(tr("在类别名后显示推理置信度"))
        self.auto_label_checkbox.setText(tr("贴图标签"))
        self.prefix_checkbox.setText(tr("添加文件名前缀"))
        self.show_paste_names_checkbox.setText(tr("显示贴图名"))
        self.show_grid_checkbox.setText(tr("显示网格线"))
        self.random_paste_btn.setText(tr("随机贴图"))
        self.batch_paste_btn.setText(tr("一键贴图"))
        is_thumb = self.is_thumbnail_mode
        self.toggle_view_btn.setText(tr("列表视图") if is_thumb else tr("缩略视图"))
        self.clear_btn.setText(tr("清空画布"))
        self.save_btn.setText(tr("保存图片"))
        self.save_all_btn.setText(tr("全部保存"))
        if hasattr(self, 'view_stats_btn'):
            self.view_stats_btn.setText(tr("统计"))
            self.view_stats_btn.setToolTip(tr("标签统计"))
        if hasattr(self, 'lint_btn'):
            self.lint_btn.setText(tr("质检"))
            self.lint_btn.setToolTip(tr("标注质检"))
        if hasattr(self, 'video_lbl'):
            self.video_lbl.setText(tr("视频:"))
        if hasattr(self, 'video_load_btn'):
            self.video_load_btn.setToolTip(tr("视频抽帧:"))
        if hasattr(self, '_refresh_bg_label_mode_button'):
            self._refresh_bg_label_mode_button()
        if hasattr(self, 'view_toggle_btn'):
            if self._is_delete_view:
                self.view_toggle_btn.setText(tr("移除路径"))
            else:
                self.view_toggle_btn.setText(tr("工作路径"))
        if hasattr(self, 'btn_paste_mode'):
            from .. import i18n
            is_en = i18n.get_lang() == "en"
            if is_en:
                self.btn_paste_mode.setText("Paste")
                self.btn_annotate_mode.setText("Annotate")
            else:
                self.btn_paste_mode.setText(tr("贴图"))
                self.btn_annotate_mode.setText(tr("标注"))
            self._update_mode_seg_style()
        if hasattr(self, 'step_label'):
            self.step_label.setText(tr("步长："))
        self.lang_btn.setToolTip(tr("切换中英文"))
        self.theme_btn.setToolTip(tr("切换深色/浅色主题"))
        if hasattr(self, 'format_btn'):
            self.format_btn.setText(tr("格式"))
            self.format_btn.setToolTip(tr("格式转换"))
        if hasattr(self, 'model_lbl'):
            self.model_lbl.setText(tr("模型推理:"))
        for name, tip in (('model_row_btn', "显示模型参数行"),
                          ('model_single_btn', "推理当前图片"),
                          ('model_all_btn', "从当前图片推理到末尾")):
            button = getattr(self, name, None)
            if button is not None:
                button.setToolTip(tr(tip))
        row = getattr(self, 'model_row', None)
        if row is not None and hasattr(row, 'retranslate'):
            row.retranslate()
            # 筛选计数与范围摘要是主窗口按当前状态拼出来的（`类别筛选(2/80)`），
            # `ModelRow.retranslate` 只刷静态文字，这两处得主窗口自己重设，
            # 否则切语言后按钮还挂着中文。
            if hasattr(self, '_refresh_filter_button'):
                self._refresh_filter_button()
            if hasattr(self, '_refresh_scope_button'):
                self._refresh_scope_button()
        panel = getattr(self, '_class_filter_panel', None)
        if panel is not None and hasattr(panel, 'retranslate'):
            panel.retranslate()
        # 范围面板是非模态的，可以一边开着一边切语言；不在刷新列表里就会停在
        # 上一门语言。懒建的面板打开时自己会调 retranslate，这里管的是「正开
        # 着的那个」。
        scope_panel = getattr(self, '_scope_panel', None)
        if scope_panel is not None and hasattr(scope_panel, 'retranslate'):
            scope_panel.retranslate()
        for header_name in ('bg_list_header', 'label_group_header', 'paste_group_header'):
            header = getattr(self, header_name, None)
            if header is None:
                continue
            key = header.property('title_key') or ''
            expanded = getattr(header, '_expanded', True)
            if key:
                header.setText(f"{'▼' if expanded else '▶'}  {tr(key)}")
        if hasattr(self, 'bg_list_group') and hasattr(self.bg_list_group, 'setTitle'):
            self.bg_list_group.setTitle(tr("背景图列表"))
        if hasattr(self, 'label_group') and hasattr(self.label_group, 'setTitle'):
            self.label_group.setTitle(tr("标签管理"))
        if hasattr(self, 'paste_group') and hasattr(self.paste_group, 'setTitle'):
            self.paste_group.setTitle(tr("贴图列表"))
        if hasattr(self, 'bg_label_header_lbl'):
            self.bg_label_header_lbl.setText(tr("背景图标签"))
        if hasattr(self, 'paste_label_header_lbl'):
            self.paste_label_header_lbl.setText(tr("贴图标签_list"))
        if hasattr(self, 'bg_lbl'):
            self.bg_lbl.setText(tr("背景图:"))
        if hasattr(self, 'paste_lbl'):
            self.paste_lbl.setText(tr("贴图:"))
        if hasattr(self, 'label_lbl'):
            self.label_lbl.setText(tr("标签:"))
        if hasattr(self, 'paste_count_lbl'):
            self.paste_count_lbl.setText(tr("贴图个数:"))
        if hasattr(self, 'size_lbl'):
            self.size_lbl.setText(tr("短边尺寸:"))
        if hasattr(self, 'options_btn'):
            self.options_btn.setText(tr("选项"))
        if hasattr(self, 'cache_btn'):
            self.cache_btn.setText(tr("缓存"))
            self.cache_btn.setToolTip(tr("复制缓存管理"))
        if hasattr(self, 'memory_btn'):
            self.memory_btn.setText(tr("记忆"))
            self.memory_btn.setToolTip(tr("记忆记录"))
        if hasattr(self, 'process_btn'):
            self.process_btn.setText(tr("导出"))
            self.process_btn.setToolTip(tr("数据处理"))
        if hasattr(self, '_rebuild_label_cache_menu'):
            self._rebuild_label_cache_menu()
        if hasattr(self, '_refresh_options_menu_texts'):
            self._refresh_options_menu_texts()
        if hasattr(self, 'upload_a_btn'):
            self.upload_a_btn.setToolTip(tr("选择背景图片"))
        if hasattr(self, 'load_folder_btn'):
            self.load_folder_btn.setToolTip(tr("加载文件夹图片"))
        if hasattr(self, 'upload_b_btn'):
            self.upload_b_btn.setToolTip(tr("选择贴图"))
        if hasattr(self, 'load_small_folder_btn'):
            self.load_small_folder_btn.setToolTip(tr("加载贴图文件夹"))
        if hasattr(self, 'upload_bg_label_btn'):
            self.upload_bg_label_btn.setToolTip(tr("导入背景标签文件"))
        if hasattr(self, 'upload_paste_label_btn'):
            self.upload_paste_label_btn.setToolTip(tr("导入贴图标签文件"))
        if hasattr(self, 'random_paste_btn'):
            self.random_paste_btn.setToolTip(tr("随机贴图"))
        if hasattr(self, 'batch_paste_btn'):
            self.batch_paste_btn.setToolTip(tr("一键贴图"))
        if hasattr(self, 'clear_btn'):
            self.clear_btn.setToolTip(tr("清空画布"))
        if hasattr(self, 'save_btn'):
            self.save_btn.setToolTip(tr("保存图片"))
        if hasattr(self, 'save_all_btn'):
            self.save_all_btn.setToolTip(tr("全部保存"))
        if hasattr(self, 'canvas_adjustment'):
            self.canvas_adjustment.retranslate_ui()
        if hasattr(self, '_update_shortcut_status_label'):
            self._update_shortcut_status_label()

    def _refresh_menu_shortcuts(self):
        """刷新选项菜单中的快捷键显示"""
        from .. import i18n
        tr = i18n.t
        if hasattr(self, '_refresh_options_menu_texts'):
            self._refresh_options_menu_texts()
        if hasattr(self, 'draw_box_btn'):
            sc = self._get_shortcut('draw_box')
            self.draw_box_btn.setText(f"{tr('绘制BOX')}({sc})")
        if hasattr(self, '_update_shortcut_status_label'):
            self._update_shortcut_status_label()
