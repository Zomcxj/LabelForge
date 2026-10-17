"""
主窗口模块 - ImageEditor 主窗口逻辑（协调器）
"""
import os
import sys
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QPoint, Qt, QUrl, QTimer, QRectF
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDrag

from ..core.config import WINDOW_CONFIG, THUMBNAIL_CONFIG, MAGNIFIER_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG, BOX_BORDER_CONFIG, DEFAULT_PREFIX
from ..core.utils import create_app_icon
from ..engine.save_manager import SaveManager
from ..engine.label_manager import LabelManager
from ..engine.image_loader import ImageLoaderMixin
from ..engine.paste_engine import PasteEngineMixin
from ..engine.event_handler import EventHandlerMixin
from .i18n import t as tr
from .theme import ThemeManager, ThemeMode
from .dwm import set_titlebar_dark
from .settings_dialog import SettingsDialog
from .processing_panel import ProcessingPanel
from .mixins.label_cache_slot import LabelCacheSlotMixin
from .mixins.memory_record import MemoryRecordMixin
from .mixins.stats import StatsMixin
from .mixins.quality_lint import QualityLintMixin
from .mixins.background_list import BackgroundListMixin
from .mixins.theme import ThemeMixin
from .mixins.translation import TranslationMixin
from .mixins.dataset_classifier import DatasetClassifierMixin
from .mixins.dataset_tools import DatasetToolsMixin
from .mixins.inference import InferenceMixin
from .mixins.video import VideoMixin
from .mixins.toolbar import ToolbarMixin
from .mixins.options_popup import OptionsPopupMixin
from .mixins.cache_menu import CacheMenuMixin
from .mixins.lists import ListsMixin
from .mixins.panels import PanelsMixin


class ImageEditor(TranslationMixin, ThemeMixin, BackgroundListMixin, StatsMixin, QualityLintMixin, MemoryRecordMixin, LabelCacheSlotMixin,
                   ToolbarMixin, OptionsPopupMixin, CacheMenuMixin, ListsMixin, PanelsMixin, ImageLoaderMixin, PasteEngineMixin,
                   DatasetClassifierMixin, DatasetToolsMixin, InferenceMixin, VideoMixin, EventHandlerMixin, QMainWindow):
    """贴图标注工具主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabelForge")
        self.resize(WINDOW_CONFIG['default_width'], WINDOW_CONFIG['default_height'])

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.setWindowIcon(create_app_icon(script_dir))

        self._load_settings()
        self._init_data()
        self._is_delete_view = False
        self._nav_step = 1
        self.edit_mode = 'annotate'
        self.init_ui()
        self._apply_theme()
        self._refresh_ui_texts()
        self._connect_manager_signals()
        self.update_label_list()
        self.installEventFilterRecursive(self)
        self.setup_shortcuts()
        self.setAcceptDrops(True)
        self._apply_paste_label_visibility()
        if hasattr(self, 'auto_save_b_checkbox'):
            self.auto_save_b_checkbox.setChecked(self.edit_mode == 'annotate')
            self.auto_save_p_checkbox.setChecked(self.edit_mode == 'paste')
        self._apply_paste_label_visibility()

    def _load_settings(self):
        """从配置文件加载主题和语言设置"""
        from ..core import config_manager
        from . import i18n
        from .theme import ThemeManager, ThemeMode

        settings = config_manager.load_all()

        theme = settings.get('theme', 'light')
        ThemeManager.set_mode(ThemeMode.DARK if theme == 'dark' else ThemeMode.LIGHT)

        language = settings.get('language', 'zh')
        i18n.set_lang(language)

        self.shortcut_config = settings.get('shortcuts', {})
        self.label_colors = settings.get('label_colors', [])
        self.label_color_map = settings.get('label_color_map', {})
        self._label_color_map_palette = tuple(self.label_colors)
        self._max_labels = settings.get('max_labels', 3)
        self.label_cache_slots = settings.get('label_cache_slots', [])
        self.active_label_cache_slot = 0
        self._label_cache_copy_counter = max(
            [int(slot.get('copy_order', 0) or 0) for slot in self.label_cache_slots] or [0]
        )

        from ..core.config import GRID_CONFIG, DETECTION_BOX_CONFIG, PASTE_ITEM_CONFIG, NUDGE_CONFIG, DETECTION_BOX_WHEEL_CONFIG, CROSSHAIR_CONFIG
        if settings.get('grid_line_width') is not None:
            GRID_CONFIG['line_width'] = settings['grid_line_width']
        if settings.get('grid_alpha') is not None:
            GRID_CONFIG['alpha'] = settings['grid_alpha']
        if settings.get('resize_handle_size') is not None:
            handle_size = max(3, min(15, int(settings['resize_handle_size'])))
            DETECTION_BOX_CONFIG['resize_handle_size'] = handle_size
            PASTE_ITEM_CONFIG['handle_size'] = handle_size
        if settings.get('label_font_size') is not None:
            DETECTION_BOX_CONFIG['label_font_size'] = max(5, min(15, int(settings['label_font_size'])))
        if settings.get('label_position') in ('outside', 'inside'):
            DETECTION_BOX_CONFIG['label_position'] = settings['label_position']
        DETECTION_BOX_CONFIG['max_polygon_points'] = max(
            3, min(64, int(settings.get('max_polygon_points', DETECTION_BOX_CONFIG.get('max_polygon_points', 32))))
        )
        from ..core.config import OBB_CONFIG
        OBB_CONFIG['rotate_step'] = max(
            1, min(6, int(settings.get("rotate_step", OBB_CONFIG.get("rotate_step", 1))))
        )
        self._canvas_image_copy_enabled = bool(settings.get('canvas_image_copy_enabled', False))
        self._relative_path_display = bool(settings.get('relative_path_display', False))
        self._magnifier_enabled = bool(settings.get('magnifier_enabled', False))
        # 默认开启：画布空了就删掉 sidecar json
        self._delete_empty_json_on_switch = bool(
            settings.get('delete_empty_json_on_switch', True))
        MAGNIFIER_CONFIG['zoom'] = max(0.8, min(3.0, float(settings.get('magnifier_zoom', MAGNIFIER_CONFIG['zoom']))))
        MAGNIFIER_CONFIG['size'] = max(80, min(400, int(settings.get('magnifier_size', MAGNIFIER_CONFIG['size']))))
        pos = settings.get('magnifier_position', MAGNIFIER_CONFIG['position'])
        MAGNIFIER_CONFIG['position'] = pos if pos in ('side', 'center') else 'side'
        NUDGE_CONFIG['step'] = max(1, min(5, int(settings.get('nudge_step', NUDGE_CONFIG['step']))))
        DETECTION_BOX_WHEEL_CONFIG['detection_box_scale_step'] = max(0.01, min(0.30, float(settings.get('detection_box_scale_step', DETECTION_BOX_WHEEL_CONFIG['detection_box_scale_step']))))
        DETECTION_BOX_WHEEL_CONFIG['paste_item_scale_step'] = max(0.01, min(0.30, float(settings.get('paste_item_scale_step', DETECTION_BOX_WHEEL_CONFIG['paste_item_scale_step']))))
        DETECTION_BOX_WHEEL_CONFIG['edge_step'] = max(1, min(50, int(settings.get('detection_box_wheel_edge_step', DETECTION_BOX_WHEEL_CONFIG['edge_step']))))
        CROSSHAIR_CONFIG['width'] = max(0.5, min(3.0, float(settings.get('crosshair_width', CROSSHAIR_CONFIG['width']))))
        color = str(settings.get('crosshair_color', CROSSHAIR_CONFIG['color']))
        CROSSHAIR_CONFIG['color'] = color if len(color) == 7 and color.startswith('#') else CROSSHAIR_CONFIG['color']
        CROSSHAIR_CONFIG['alpha'] = max(0, min(255, int(settings.get('crosshair_alpha', CROSSHAIR_CONFIG['alpha']))))
        from ..core.config import BOX_BORDER_CONFIG
        BOX_BORDER_CONFIG['width'] = max(1, min(4, float(settings.get('box_border_width', BOX_BORDER_CONFIG['width']))))

    def _init_data(self):
        """初始化数据结构"""
        from PySide6.QtWidgets import QLineEdit
        from ..core.config import DEFAULT_PREFIX

        self.background_images = []
        self.current_background = None
        self.current_background_index = -1
        self.small_images = []
        self._memory_background_path = ""
        self._memory_paste_path = ""
        self._memory_label_path = ""
        self.canvas_items_dict = {}
        self.canvas_items = []
        self.selected_item = None
        self.is_dragging = False
        self.is_resizing = False
        self._canvas_drag_active = False
        if not hasattr(self, '_canvas_image_copy_enabled'):
            self._canvas_image_copy_enabled = False
        if not hasattr(self, '_relative_path_display'):
            self._relative_path_display = False
        if not hasattr(self, '_magnifier_enabled'):
            self._magnifier_enabled = False
        if not hasattr(self, '_delete_empty_json_on_switch'):
            self._delete_empty_json_on_switch = True
        self.drag_offset = QPoint(0, 0)
        self._busy = False
        # 推理状态：模型只加载一次，阈值改动即时写回 detector
        self._detector = None
        self._model_size = None
        self._model_format = None
        self._model_worker = None
        self._run_worker = None
        self._progress_dialog = None
        self.model_row = None
        # 视频抽帧行与模型行各自独立：只点一个是 1 行，都点开是 2 行
        # None = 还没设过筛选，按全选处理；换成具体集合才是真的过滤
        self._class_filter = None
        self._class_filter_panel = None
        self._inference_dataset = None

        self.detection_boxes_dict = {}
        self.detection_boxes = []
        self._last_paste_slot = None
        self._last_paste_start = -1

        # 语义区域（仅运行时存在，不持久化/不导出）
        #
        # **两份独立区域**（用户第 10 轮：「推理是推理的操作，贴图是贴图的操作；
        # 推理的区域贴图不能显示和操作，贴图的区域推理不能显示和操作」）：
        # - `region_boxes`   ：**推理**范围（滑窗/区域推理用），画在标注模式
        # - `paste_region_boxes`：**贴图**落点约束，画在贴图模式，从空开始
        # 两份各自有面板、各自可编辑，互不可见、互不影响。
        self.region_boxes = []
        self.paste_region_boxes = []
        self.region_fixed = False
        self.paste_region_fixed = False
        # 范围面板是否开着。**在这里就初始化**：画布在鼠标事件里会读它
        # （`_region_editing_allowed`），只靠「第一次开面板时赋值」会让开面板
        # 之前的状态是「属性不存在」，`getattr(..., False)` 虽然兜住了，但
        # 直接读就会 AttributeError。
        self._scope_panel_open = False
        # 面板正在装配（`_open_scope_panel` 里那串 set_*）时置真：期间
        # `overlay_changed` 只重画画布，不写宿主状态。
        self._scope_panel_configuring = False
        self._last_paste_count = 0
        self.global_labels = set()
        self.background_dataset_labels = set()
        # 扫描结果（每次扫描整体替换）与手动导入的类别分开存，
        # 对外暴露的 background_dataset_labels 是两者并集。
        self._scanned_background_labels = set()
        self.imported_background_labels = set()
        self._bg_label_list_mode = 'stats'
        self.pressed_box_index = None
        self._bg_annotation_filter = 'all'  # all | annotated | unannotated | empty
        self._task_filter = set()  # 任务筛选（det/seg/pose/obb），空=全部
        self._group_filter = set()  # 分组筛选（group_id），空=全部
        self._bg_filter_saved_index = 0
        self._cached_bg_label_stats = []
        self._cached_bg_label_stats_path = ""
        self._background_label_scan_generation = 0
        self._background_label_scan_worker = None
        self._background_label_scan_workers = set()
        self._background_label_scan_pending = False
        self._background_label_scan_completed = False
        self._dataset_stats_dirty = False

        self.prefix_input = QLineEdit()
        self.prefix_input.setText(DEFAULT_PREFIX)
        self.prefix_checkbox_state = True
        self.default_prefix = DEFAULT_PREFIX

        self.is_thumbnail_mode = True
        self.thumbnail_grid_width = THUMBNAIL_CONFIG['grid_width']
        self.thumbnail_grid_height = THUMBNAIL_CONFIG['grid_height']
        self.thumbnail_spacing = THUMBNAIL_CONFIG['spacing']

        self.save_manager = SaveManager(self, self)
        self.label_manager = LabelManager(self, self)

        from ..engine.undo_manager import UndoManager
        self.undo_manager = UndoManager()

    def get_label_color(self, label):
        """获取标签颜色并缓存首次分配，避免标签顺序影响已有颜色。"""
        from ..core import config_manager
        color = config_manager.get_label_color(
            self._get_session_labels(), label, self.label_colors,
            self.label_color_map if (
                not hasattr(self, '_label_color_map_palette')
                or tuple(self.label_colors) == self._label_color_map_palette
            ) else {},
        )
        if label and label not in self.label_color_map:
            self.label_color_map[label] = color
            config_manager.save_all(
                label_colors=self.label_colors,
                label_color_map=self.label_color_map,
            )
            self._label_color_map_palette = tuple(self.label_colors)
        return color

    # ===== 委托方法 - 保持对外接口不变 =====

    def save_json(self, image_path, image_name, label_prefix, canvas_items=None,
                  image_width=None, image_height=None, current_index=None):
        """生成并保存 JSON 文件"""
        self.save_manager.save_json(
            image_path, image_name, label_prefix,
            canvas_items, image_width, image_height, current_index
        )
        if hasattr(self, '_notify_lint_boxes_changed'):
            index = (current_index if current_index is not None
                     else self.current_background_index)
            self._notify_lint_boxes_changed(index)

    def auto_save_background(self):
        self.save_manager.auto_save_background()
    def auto_save_project(self):
        self.save_manager.auto_save_project()

    def save_current_json(self):
        """保存当前图的标注 JSON。"""
        self.save_manager.save_current_json()

    def save_canvas(self):
        """保存当前画布"""
        self.save_manager.save_canvas()

    def save_all_canvas(self):
        """保存所有画布"""
        self.save_manager.save_all_canvas()

    def add_label(self, label_name=None):
        """增加标签"""
        self.label_manager.add_label(label_name)

    def delete_label(self):
        """删除标签"""
        self.label_manager.delete_label()

    def delete_selected_label(self):
        """快捷键删除选中标签（不弹确认框）"""
        self.label_manager.delete_selected_label()

    def update_global_labels(self):
        """更新全局标签列表"""
        self.label_manager.update_global_labels()

    def update_label_list(self):
        """更新标签列表显示"""
        self.label_manager.update_label_list()

    def _on_save_completed(self):
        """保存完成后刷新 UI"""
        if self.current_background_index >= 0:
            self.background_list.setCurrentRow(self.current_background_index)
        self.update_file_count()
        self._update_status_info()
        self.canvas.update()

    def _update_mode_seg_style(self, animated=False):
        """同步模式分段控件的选中状态和指示器位置。"""
        if not hasattr(self, 'btn_paste_mode'):
            return
        is_paste = self.edit_mode == 'paste'
        self.btn_paste_mode.blockSignals(True)
        self.btn_annotate_mode.blockSignals(True)
        self.btn_paste_mode.setChecked(is_paste)
        self.btn_annotate_mode.setChecked(not is_paste)
        self.btn_paste_mode.blockSignals(False)
        self.btn_annotate_mode.blockSignals(False)
        if hasattr(self, 'mode_seg_ctrl'):
            self.mode_seg_ctrl.set_accent(ThemeManager.get_theme()["interaction_active"])
            self.mode_seg_ctrl.update_position(animated=animated)

    def _apply_mode_visibility_defaults(self):
        """模式切换时重置显示项，避免上个模式的显示状态串到当前模式。"""
        if not hasattr(self, 'show_label_names_checkbox'):
            return
        is_annotate = self.edit_mode == 'annotate'
        if hasattr(self, 'show_labels_checkbox'):
            self.show_labels_checkbox.setChecked(is_annotate)
        self.show_label_names_checkbox.setChecked(is_annotate)
        if hasattr(self, 'show_scores_checkbox'):
            self.show_scores_checkbox.setChecked(is_annotate)
        if hasattr(self, 'show_paste_names_checkbox'):
            self.show_paste_names_checkbox.setChecked(not is_annotate)

    def _toggle_edit_mode(self):
        """切换标注/贴图模式"""
        sender = self.sender()
        if sender == self.btn_paste_mode:
            self._set_edit_mode('paste', animated=True)
        else:
            self._set_edit_mode('annotate', animated=True)

    def _set_edit_mode(self, mode, animated=False):
        self.edit_mode = 'annotate' if mode == 'annotate' else 'paste'
        self.selected_item = None
        if hasattr(self, 'canvas'):
            self.canvas.selected_item_size = None
            self.canvas.selected_box = None
            self.canvas.selected_boxes = []
            self.canvas.hover_resize_target = None
            self.canvas.hover_resize_handle = None
            self.canvas.selected_region = None
            self.canvas.is_dragging_region = False
            self.canvas.is_resizing_region = False
            self.canvas.region_resize_handle = None
            if (getattr(self.canvas, 'is_drawing_region', False)
                    or getattr(self.canvas, 'is_drawing_region_polygon', False)):
                self.canvas._reset_region_drawing_state()
            self.canvas.update()
        self._apply_mode_visibility_defaults()
        self._update_mode_seg_style(animated=animated)
        self._apply_paste_label_visibility()
        # 贴图模式自动收起模型参数行（推理按钮随之禁用），切回标注模式恢复。
        sync_row = getattr(self, '_sync_model_row_to_edit_mode', None)
        if callable(sync_row):
            sync_row()
        if hasattr(self, 'auto_save_b_checkbox'):
            is_annotate = self.edit_mode == 'annotate'
            self.auto_save_b_checkbox.setChecked(is_annotate)
            self.auto_save_p_checkbox.setChecked(not is_annotate)
            self.auto_save_b_checkbox.setText(f"{tr('自动保存B')}({tr('标注')})" if is_annotate else tr("自动保存B"))
            self.auto_save_p_checkbox.setText(f"{tr('自动保存P')}({tr('贴图')})" if not is_annotate else tr("自动保存P"))
        from PySide6.QtCore import QTimer
        mode_text = "Annotate" if self.edit_mode == 'annotate' else "Paste"
        self.status_label.setText(f"Mode: {mode_text}")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _apply_paste_label_visibility(self):
        """Annotate: hide paste label column + paste list; paste: show both."""
        show = getattr(self, 'edit_mode', 'paste') != 'annotate'
        if hasattr(self, 'paste_label_column'):
            self.paste_label_column.setVisible(show)
        else:
            if hasattr(self, 'paste_label_list'):
                self.paste_label_list.setVisible(show)
            if hasattr(self, 'paste_label_header_lbl'):
                self.paste_label_header_lbl.setVisible(show)

        paste_group = getattr(self, 'paste_group', None)
        paste_header = getattr(self, 'paste_group_header', None)
        if paste_group is not None:
            paste_group.setVisible(show)
            if show and paste_header is not None:
                paste_header._expanded = True
                key = paste_header.property('title_key') or '贴图列表'
                paste_header.setText(f"▼  {tr(key)}")
                for section in getattr(self, '_side_sections', []) or []:
                    if section.get('header') is paste_header:
                        section['content'].setVisible(True)
                        break
            if hasattr(self, '_update_side_panel_stretches'):
                self._update_side_panel_stretches()

    def save_undo_state(self):
        """保存撤销状态"""
        if self._is_delete_view:
            return
        self.undo_manager.save_state(self.canvas_items, self.detection_boxes)

    def undo(self):
        """撤销"""
        if self._is_delete_view:
            return
        self.canvas_items, self.detection_boxes = self.undo_manager.undo(
            self.canvas_items, self.detection_boxes
        )
        self.canvas.update()
        self.update_label_list()
        if hasattr(self, '_notify_lint_boxes_changed'):
            self._notify_lint_boxes_changed()

    def redo(self):
        """重做"""
        if self._is_delete_view:
            return
        self.canvas_items, self.detection_boxes = self.undo_manager.redo(
            self.canvas_items, self.detection_boxes
        )
        self.canvas.update()
        self.update_label_list()
        if hasattr(self, '_notify_lint_boxes_changed'):
            self._notify_lint_boxes_changed()

    def toggle_grid(self):
        """切换网格显示"""
        if hasattr(self, 'show_grid_checkbox'):
            self.show_grid_checkbox.setChecked(not self.show_grid_checkbox.isChecked())
            self.canvas.update()

    def open_settings(self):
        """打开设置对话框"""
        dialog = SettingsDialog(self)
        dialog.exec_()

    def reset_settings_to_default(self):
        """恢复出厂设置：清空配置文件并刷新界面，返回是否成功。"""
        from ..core import config_manager
        from . import i18n

        if not config_manager.reset_all():
            return False

        self._load_settings()
        i18n.set_lang(config_manager.load_language())
        self._apply_theme()
        self._refresh_ui_texts()

        self.prefix_input.setText(DEFAULT_PREFIX)
        self.prefix_checkbox.setChecked(True)
        self.show_labels_checkbox.setChecked(True)
        self.show_label_names_checkbox.setChecked(True)
        self.show_paste_names_checkbox.setChecked(True)
        self.show_grid_checkbox.setChecked(False)
        self._canvas_image_copy_enabled = False
        self._magnifier_enabled = False
        self._relative_path_display = False
        self._delete_empty_json_on_switch = True
        if hasattr(self, 'canvas_copy_action'):
            self.canvas_copy_action.setChecked(False)
        if hasattr(self, 'magnifier_action'):
            self.magnifier_action.setChecked(False)
        if hasattr(self, 'relative_path_action'):
            self.relative_path_action.setChecked(False)
        if hasattr(self, 'delete_empty_json_action'):
            self.delete_empty_json_action.setChecked(True)
        if hasattr(self, '_refresh_background_path_texts'):
            self._refresh_background_path_texts()
        if hasattr(self, 'update_shortcuts'):
            self.update_shortcuts()
        if hasattr(self, '_refresh_menu_shortcuts'):
            self._refresh_menu_shortcuts()
        if hasattr(self, '_rebuild_label_cache_menu'):
            self._rebuild_label_cache_menu()
        if hasattr(self, 'canvas'):
            self.canvas.update()
        return True

    def _toggle_processing_panel(self):
        if not hasattr(self, '_processing_panel') or self._processing_panel is None:
            self._processing_panel = ProcessingPanel(self)
        if self._processing_panel.isVisible():
            self._processing_panel.hide()
        else:
            self._update_processing_panel_labels()
            self._processing_panel.show()
            self._center_processing_panel()

    def _show_export_menu(self):
        """顶部'导出'按钮弹出菜单: 增强划分 / Kmeans划分"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint

        menu = QMenu(self)
        menu.setObjectName("optionsMenu")
        action_aug = menu.addAction(tr("增强划分"))
        action_kmeans = menu.addAction(tr("Kmeans划分"))

        btn_pos = self.process_btn.mapToGlobal(QPoint(0, self.process_btn.height()))
        action = menu.exec_(btn_pos)

        if action == action_aug:
            self._toggle_processing_panel()
        elif action == action_kmeans:
            self._open_dataset_classifier()

    def _update_processing_panel_labels(self):
        if hasattr(self, '_processing_panel') and self._processing_panel:
            self._processing_panel._refresh_texts()
            self._processing_panel._update_labels_list()

    def _center_processing_panel(self):
        if not hasattr(self, '_processing_panel') or not self._processing_panel:
            return
        parent_geo = self.geometry()
        child_geo = self._processing_panel.geometry()
        x = parent_geo.x() + (parent_geo.width() - child_geo.width()) // 2
        y = parent_geo.y() + (parent_geo.height() - child_geo.height()) // 2
        self._processing_panel.move(x, y)

    def closeEvent(self, event):
        if hasattr(self, '_processing_panel') and self._processing_panel and any(w.isRunning() for w in self._processing_panel._workers):
            from PySide6.QtWidgets import QMessageBox
            from .dwm import set_titlebar_dark
            from .i18n import t as _t
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle(tr("提示"))
            box.setText(tr("数据处理正在进行中，无法关闭主界面"))
            ok_btn = box.addButton(_t("确定"), QMessageBox.AcceptRole)
            box.setDefaultButton(ok_btn)
            hwnd = int(box.winId())
            set_titlebar_dark(hwnd, True)
            box.exec_()
            event.ignore()
            return
        if not self._cleanup_background_label_scan_worker():
            event.ignore()
            return
        if hasattr(self, '_cleanup_delete_worker') and not self._cleanup_delete_worker():
            event.ignore()
            return
        if hasattr(self, '_close_lint_dialog'):
            self._close_lint_dialog()
        # 抽帧线程要停掉再放行：活线程会在窗口销毁后继续写状态栏
        video_worker = getattr(self, '_video_worker', None)
        if video_worker is not None and video_worker.isRunning():
            video_worker.cancel()
            video_worker.wait(3000)
        compose_worker = getattr(self, '_compose_worker', None)
        if compose_worker is not None and compose_worker.isRunning():
            compose_worker.cancel()
            compose_worker.wait(3000)
        if hasattr(self, '_close_health_worker'):
            self._close_health_worker()
        if hasattr(self, '_processing_panel') and self._processing_panel:
            self._processing_panel.close()
        if self.current_background_index >= 0:
            self.canvas_items_dict[self.current_background_index] = self.canvas_items.copy()
            self.detection_boxes_dict[self.current_background_index] = self.detection_boxes.copy()
            self.save_current_json()
        if hasattr(self, '_save_memory_record_on_close'):
            self._save_memory_record_on_close()
        event.accept()

    def get_image_info(self):
        """获取当前图片信息"""
        if self.current_background is None:
            return None
        info = {
            'width': self.current_background.width(),
            'height': self.current_background.height(),
            'path': self.background_images[self.current_background_index] if self.current_background_index >= 0 else '',
            'paste_count': len(self.canvas_items),
            'box_count': len(self.detection_boxes),
        }
        return info

    def get_label_stats(self):
        """获取标签统计"""
        stats = {}
        for _, _, label in self.canvas_items:
            stats[label] = stats.get(label, 0) + 1
        for box in self.detection_boxes:
            label = box.get('label', 'unknown')
            stats[label] = stats.get(label, 0) + 1
        return stats


# 程序入口
    # ========== 拖拽：拖入图片文件上传 ==========

    def dragEnterEvent(self, event: QDragEnterEvent):
        """接受图片和JSON文件拖入；文件夹整体拖入按数据集加载"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                # 文件夹没有扩展名，必须先于扩展名判断
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
                ext = os.path.splitext(path)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.json'):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent):
        """处理拖入的图片和JSON文件；拖入文件夹则整包按数据集加载"""
        from ..core.config import SUPPORTED_IMAGE_EXTENSIONS
        # 文件夹优先：拖目录 = 打开该数据集（替换当前，与文件夹按钮同路径）。
        # 拖入多结构（文件夹+散图）时以文件夹为准，不做混合追加。
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self.load_background_folder(path)
                event.acceptProposedAction()
                return
        existing = {os.path.normpath(p) for p in self.background_images}
        images = []
        jsons = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            norm_path = os.path.normpath(path)
            ext = os.path.splitext(path)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                if norm_path not in existing:
                    images.append(path)
            elif ext == '.json':
                jsons.append(path)
        if images:
            self._append_background_images(images)
        if jsons:
            self._apply_dropped_json(jsons)
        event.acceptProposedAction()

    def _append_background_images(self, files):
        """追加背景图片（不替换已有，自动去重）"""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QApplication
        first_new = len(self.background_images)
        for file in files:
            if file in self.background_images:
                continue
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                new_index = len(self.background_images)
                self.background_images.append(file)
                display_path = self._display_path_for(file)
                from PySide6.QtWidgets import QListWidgetItem
                from ..engine.image_loader import decorate_background_list_item
                item = QListWidgetItem(display_path)
                decorate_background_list_item(item, file, new_index)
                self.background_list.addItem(item)
                self.canvas_items_dict[new_index] = []
                self.detection_boxes_dict[new_index] = self.load_detection_boxes(file)

                if self.current_background is None:
                    self.current_background = pixmap
                    self.current_background_index = new_index
                    self.canvas_items = []
                    self.detection_boxes = self.detection_boxes_dict[new_index].copy()
                    self.update_label_list()
                    self.canvas.background_scale = 1.0
                    self.canvas.is_manual_scale = False
                    self.canvas.update()

        self.update_file_count()
        if self.background_images:
            self.background_list.setCurrentRow(first_new)

    def _apply_dropped_json(self, json_files):
        """将拖入的JSON标签文件按文件名匹配应用到对应背景图"""
        import json as _json
        if not json_files:
            return
        for jf in json_files:
            if not os.path.isfile(jf):
                continue
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                if not isinstance(data, dict) or 'shapes' not in data:
                    continue
                boxes = []
                for shape in data['shapes']:
                    if not isinstance(shape, dict):
                        continue
                    if not all(k in shape for k in ('label', 'points')):
                        continue
                    points = shape['points']
                    if len(points) < 2:
                        continue
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    boxes.append({
                        'x': min(xs), 'y': min(ys),
                        'width': max(xs) - min(xs),
                        'height': max(ys) - min(ys),
                        'label': shape['label'],
                    })
                if not boxes:
                    continue
                json_stem = os.path.splitext(os.path.basename(jf))[0]
                target_index = -1
                for idx, img_path in enumerate(self.background_images):
                    img_stem = os.path.splitext(os.path.basename(img_path))[0]
                    if img_stem == json_stem:
                        target_index = idx
                        break
                if target_index < 0:
                    target_index = self.current_background_index
                if target_index < 0:
                    continue
                existing = self.detection_boxes_dict.get(target_index, [])
                existing.extend(boxes)
                self.detection_boxes_dict[target_index] = existing
                if target_index == self.current_background_index:
                    self.detection_boxes = self.detection_boxes_dict[target_index].copy()
            except Exception as e:
                from ..core.exception_hook import _write_log
                _write_log(f"拖入JSON加载失败: {jf}, {e}")

        self.update_label_list()
        self.canvas.update()


    def apply_startup_args(self, background_dir=None, labels_file=None):
        """应用命令行启动参数：打开背景目录、导入标签文件。"""
        if background_dir:
            if os.path.isdir(background_dir):
                self.load_background_folder(background_dir)
            else:
                self.status_label.setText(f"{tr('路径不存在')}: {background_dir}")
        if labels_file:
            if os.path.isfile(labels_file):
                self.load_background_label_file(labels_file)
            else:
                self.status_label.setText(f"{tr('路径不存在')}: {labels_file}")


def _parse_args(argv):
    """解析命令行参数。"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="LabelForge",
        description="贴图标注工具",
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="启动后直接打开的背景图目录",
    )
    parser.add_argument(
        "--labels", default=None, metavar="文件",
        help="启动后导入的背景标签文件（每行一个类别）",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="显示版本号后退出",
    )
    return parser.parse_args(argv)


def main():
    """程序入口函数"""
    import sys
    import warnings

    warnings.simplefilter("ignore", DeprecationWarning)

    args = _parse_args(sys.argv[1:])
    if args.version:
        from .. import __version__
        print(f"LabelForge {__version__}")
        return

    # 这两个属性在 Qt6 已弃用且是空操作（Qt6 默认就开启 high-DPI 缩放与
    # 高分辨率 pixmap），设了不报错、只是发 DeprecationWarning——本项目启动时
    # 已全局忽略 DeprecationWarning。保留是为了显式表达意图，删掉也不会变行为。
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv[:1])

    from PySide6.QtGui import QFontDatabase
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    font_dir = os.path.join(base, "ico_image", "fonts")
    for name in ["JetBrainsMono-Regular.ttf", "JetBrainsMono-Medium.ttf", "JetBrainsMono-Bold.ttf"]:
        fpath = os.path.join(font_dir, name)
        if os.path.exists(fpath):
            QFontDatabase.addApplicationFont(fpath)

    editor = ImageEditor()
    editor.show()
    editor.apply_startup_args(args.path, args.labels)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
