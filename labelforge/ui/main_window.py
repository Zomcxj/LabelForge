"""
主窗口模块 - ImageEditor 主窗口逻辑（协调器）
"""
import os
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from ..core.config import WINDOW_CONFIG, THUMBNAIL_CONFIG, MAGNIFIER_CONFIG, DEFAULT_PREFIX
from ..core.utils import create_app_icon
from ..engine.save_manager import SaveManager
from ..engine.label_manager import LabelManager
from ..engine.image_loader import ImageLoaderMixin
from ..engine.paste_engine import PasteEngineMixin
from ..engine.paste_item import as_paste_item
from ..engine.event_handler import EventHandlerMixin
from .i18n import t as tr
from .theme import ThemeManager
from .settings_dialog import SettingsDialog
from .processing_panel import ProcessingPanel
from .mixins.label_cache_slot import LabelCacheSlotMixin
from .mixins.memory_record import MemoryRecordMixin
from .mixins.stats import StatsMixin
from .mixins.quality_lint import QualityLintMixin
from .mixins.background_list import BackgroundListMixin
from .mixins.theme import ThemeMixin
from .mixins.translation import TranslationMixin
from .mixins.dataset_grouper import DatasetGrouperMixin
from .mixins.dataset_tools import DatasetToolsMixin
from .mixins.image_tools import ImageToolsMixin
from .mixins.inference import InferenceMixin
from .mixins.video import VideoMixin
from .mixins.toolbar import ToolbarMixin
from .mixins.options_popup import OptionsPopupMixin
from .mixins.cache_menu import CacheMenuMixin
from .mixins.lists import ListsMixin
from .mixins.panels import PanelsMixin


class ImageEditor(TranslationMixin, ThemeMixin, BackgroundListMixin, StatsMixin, QualityLintMixin, MemoryRecordMixin, LabelCacheSlotMixin,
                   ToolbarMixin, OptionsPopupMixin, CacheMenuMixin, ListsMixin, PanelsMixin, ImageLoaderMixin, PasteEngineMixin,
                   DatasetGrouperMixin, DatasetToolsMixin, ImageToolsMixin, InferenceMixin, VideoMixin, EventHandlerMixin, QMainWindow):
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
        self._recent_labels = []
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

        # 抠图与融合：设置页写的就是这两份字典，启动时按配置文件回填，
        # 保证「界面上看到的」就是「推理/保存时用的」。
        from ..core.config import MATTING_CONFIG, PASTE_FUSION_CONFIG
        MATTING_CONFIG['model_path'] = str(
            settings.get('matte_model_path', MATTING_CONFIG['model_path']) or '').strip()
        MATTING_CONFIG['preset'] = settings.get('matte_preset', MATTING_CONFIG['preset'])
        MATTING_CONFIG['input_size'] = int(
            settings.get('matte_input_size', MATTING_CONFIG['input_size']) or 0)
        MATTING_CONFIG['threshold'] = float(
            settings.get('matte_threshold', MATTING_CONFIG['threshold']) or 0.0)
        MATTING_CONFIG['output_dir'] = str(
            settings.get('matte_output_dir', MATTING_CONFIG['output_dir']) or '').strip()
        PASTE_FUSION_CONFIG['seamless'] = bool(
            settings.get('seamless_paste', PASTE_FUSION_CONFIG['seamless']))

    # `current_background` 是画布当前要画的图：赋值时就同步画布显示面板的
    # 显隐（不能在 paintEvent 里改控件可见性，那会偶发 0xC0000005 崩溃）。
    @property
    def current_background(self):
        return self._current_background

    @current_background.setter
    def current_background(self, pixmap):
        self._current_background = pixmap
        sync = getattr(self, '_sync_canvas_adjustment_visibility', None)
        if sync is not None:
            sync()

    def _init_data(self):
        """初始化数据结构"""
        from PySide6.QtWidgets import QLineEdit
        from ..core.config import DEFAULT_PREFIX

        self.background_images = []
        self._current_background = None
        self.current_background_index = -1
        # 图像编辑面板状态：面板开着时画布进预览态，标注隐藏、编辑锁定
        self._image_tools_active = False
        self._image_tools_original = None
        self.image_tools_regions = []
        self.image_tools_region_mode = "none"
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
        # 抠图模型惰性构造：只有配置了权重才建实例，未配置时入口置灰
        self._matting_model = None
        self._matte_worker = None
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
        # 两份独立区域：推理的区域贴图不能显示和操作，
        # 贴图的区域推理不能显示和操作。
        # - `region_boxes`   ：推理范围（滑窗/区域推理用），画在标注模式
        # - `paste_region_boxes`：贴图落点约束，画在贴图模式，从空开始
        # 两份各自有面板、各自可编辑，互不可见、互不影响。
        self.region_boxes = []
        self.paste_region_boxes = []
        self.region_fixed = False
        self.paste_region_fixed = False
        # 范围面板是否开着。在这里就初始化：画布在鼠标事件里会读它
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

    def _toggle_edit_mode(self, mode):
        """切换标注/贴图模式。

        模式由调用方直接传入，不读 `sender()`：两个按钮都是用 lambda 连的，
        lambda 是普通 Python 可调用对象，Qt 无法从中还原发送者，`sender()`
        恒返回 None —— 之前正是因此每次都落到 else 分支，贴图模式点不进去。
        """
        self._set_edit_mode('paste' if mode == 'paste' else 'annotate',
                            animated=True)

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
        # 导出任务标签随模式显隐；测试替身没有这个方法，取不到就跳过
        refresh_task = getattr(self, '_refresh_export_task_label', None)
        if callable(refresh_task):
            refresh_task()
        from PySide6.QtCore import QTimer
        mode_text = "Annotate" if self.edit_mode == 'annotate' else "Paste"
        self.status_label.setText(f"Mode: {mode_text}")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))

    def _apply_paste_label_visibility(self):
        """标注模式隐藏贴图标签列与贴图列表，贴图模式两者都显示。"""
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
        self._refresh_export_task_label()
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
        self._refresh_export_task_label()
        if hasattr(self, '_notify_lint_boxes_changed'):
            self._notify_lint_boxes_changed()

    def _refresh_export_task_label(self):
        """右下角显示当前图片的导出任务（det / seg / obb / pose）；只在贴图模式显示。

        导出任务由贴图决定（贴图旋转会把它变成 obb），标注模式下这块信息用不上，
        隐藏它给状态栏留出宽度。

        任务类型不单独存一份：形状一改（贴图旋转、框转 OBB、加多边形/关键点）
        任务就会变，所以每次刷新都按当前的贴图项与检测框重算。
        """
        label = getattr(self, 'export_task_label', None)
        if label is None:
            return
        set_visible = getattr(label, 'setVisible', None)
        if getattr(self, 'edit_mode', 'paste') != 'paste':
            if set_visible is not None:
                set_visible(False)
            return
        if set_visible is not None:
            set_visible(True)
        from ..engine.yolo_exporter import detect_task
        entries = []
        for raw in (getattr(self, 'canvas_items', None) or []):
            try:
                entries.append({"shape_type": as_paste_item(raw).shape_type})
            except TypeError:
                continue
        for box in (getattr(self, 'detection_boxes', None) or []):
            if not isinstance(box, dict):
                continue
            entries.append({"shape_type": box.get("shape_type") or "rectangle",
                            "group_id": box.get("group_id")})
        label.setText(f"{tr('导出任务')}: {detect_task(entries)}")

    # ===== 抠图素材（区域/检测框右键） =====

    @property
    def matting_model(self):
        """抠图模型实例；未配置权重时返回 None（菜单项据此置灰）。

        惰性构造：只在第一次读这个属性（建菜单、点击入口）时检查一次权重，
        配置里没填路径或文件不存在都算未配置，不去碰 onnxruntime。
        """
        if self._matting_model is not None:
            return self._matting_model
        from ..core.config import MATTING_CONFIG
        from ..engine.matting_model import MattingModel

        model_path = (MATTING_CONFIG.get('model_path') or '').strip()
        if not model_path:
            return None
        try:
            self._matting_model = MattingModel(
                model_path=model_path,
                preset=MATTING_CONFIG.get('preset') or 'rmbg14',
                input_size=MATTING_CONFIG.get('input_size') or None,
                threads=MATTING_CONFIG.get('threads') or 0,
            )
        except (FileNotFoundError, ValueError):
            return None
        return self._matting_model

    @matting_model.setter
    def matting_model(self, value):
        """注入/清空模型（设置页改完路径或测试替身用）。"""
        self._matting_model = value

    def _rebuild_matting_model(self):
        """设置保存后按新配置重建模型；路径为空或权重不可用时清成 None。

        构造只做参数与文件存在性校验，不建 onnxruntime 会话，所以这里可以
        同步调；真正加载留到第一次推理。失败时把原因写到状态栏，入口置灰。
        """
        from ..core.config import MATTING_CONFIG
        from ..engine.matting_model import MattingModel

        path = (MATTING_CONFIG.get('model_path') or '').strip()
        self._matting_model = None
        if not path:
            return
        try:
            self._matting_model = MattingModel(
                model_path=path,
                preset=MATTING_CONFIG.get('preset') or 'rmbg14',
                input_size=MATTING_CONFIG.get('input_size') or None,
                threads=MATTING_CONFIG.get('threads') or 0,
            )
        except (FileNotFoundError, ValueError) as exc:
            self.status_label.setText(f"{tr('抠图模型不可用')}: {exc}")


    def _matting_ready(self):
        """抠图模型是否可用。"""
        return self.matting_model is not None

    def _matte_output_dir(self):
        """抠图输出目录：配置值优先，否则 <背景图目录>_matting_output/。"""
        from ..core.config import MATTING_CONFIG
        from ..engine.matting_service import default_output_dir

        configured = (MATTING_CONFIG.get('output_dir') or '').strip()
        if configured:
            return configured
        base = getattr(self, '_memory_background_path', '') or ''
        if not base:
            base = os.path.dirname(self._matte_source_path())
        return str(default_output_dir(base or os.getcwd()))

    def _matte_source_path(self):
        """当前背景图的完整路径；没有打开图片时返回空串。"""
        images = getattr(self, 'background_images', None) or []
        index = getattr(self, 'current_background_index', -1)
        if 0 <= index < len(images):
            return images[index]
        return ''

    def _current_background_rgb(self):
        """当前背景图的 RGB 数组；没有背景图时返回 None。"""
        pixmap = self.current_background
        if pixmap is None:
            return None
        from ..engine.inference_image import qimage_to_rgb_array
        return qimage_to_rgb_array(pixmap.toImage())

    def _matte_plan(self, target):
        """区域/标注 -> 抠图参数；取不到有效尺寸时返回 None。

        只做取值，不碰像素也不碰模型，便于单独验证入口接线。
        """
        from ..engine.matting_service import target_box, target_polygon

        if not isinstance(target, dict):
            return None
        try:
            raw_box = target_box(target)
        except (KeyError, TypeError, ValueError):
            return None
        if raw_box[2] <= 0 or raw_box[3] <= 0:
            return None
        return {
            'box': raw_box,
            'label': target.get('label') or 'object',
            'source_name': os.path.basename(self._matte_source_path())
                           or 'image.png',
            'polygon': target_polygon(target),
            'output_dir': self._matte_output_dir(),
        }

    def matte_selection_to_material(self, box=None, region=None):
        """把 box 或 region 指代的区域抠成贴图素材。

        产物恒为矩形位图；贴图默认 det，不继承来源形状。推理与存盘在后台
        线程里做，结果经 `_apply_matte_result` 回到主线程，因此本方法返回
        None（路径在结果的 `path` 上）。
        """
        from ..core.config import MATTING_CONFIG
        from .inference_workers import MatteWorker

        model = self.matting_model
        if model is None:
            self.status_label.setText(tr("未配置抠图模型"))
            return None

        worker = getattr(self, '_matte_worker', None)
        if worker is not None and worker.isRunning():
            self.status_label.setText(tr("正在抠图，请稍候"))
            return None

        plan = self._matte_plan(box if box is not None else region)
        if plan is None:
            self.status_label.setText(tr("请先选中一个区域或检测框"))
            return None

        rgb = self._current_background_rgb()
        if rgb is None:
            self.status_label.setText(tr("当前没有可抠图的背景图"))
            return None

        worker = MatteWorker(
            model, rgb, plan,
            threshold=MATTING_CONFIG.get('threshold', 0.0),
            margin_ratio=MATTING_CONFIG.get('margin_ratio', 0.05))
        worker.done.connect(self._apply_matte_result)
        worker.failed.connect(self._on_matte_failed)
        worker.finished.connect(self._on_matte_worker_finished)
        self._matte_worker = worker
        # 抠图期间不许再触发贴图/保存一类的画布改动
        self._busy = True
        self.status_label.setText(tr("正在抠图…"))
        worker.start()
        return None

    def is_matte_running(self):
        """是否有抠图线程在跑。"""
        worker = getattr(self, '_matte_worker', None)
        return worker is not None and worker.isRunning()

    def cancel_matte_selection(self):
        """取消进行中的抠图：线程在步骤之间退出，已落盘的文件保留。"""
        worker = getattr(self, '_matte_worker', None)
        if worker is not None and worker.isRunning():
            worker.cancel()
            self.status_label.setText(tr("已取消抠图"))

    def _apply_matte_result(self, result):
        """抠图成功：贴图模式接到画布，标注模式只把 PNG 落到输出目录。

        标注模式下抠图是「导出一张素材」的动作：不往画布加贴图项、不写
        JSON、不动标注，只在状态栏报文件名；贴图模式则把产物接到画布上，
        由用户自己决定什么时候保存。
        """
        from ..engine.matting_service import paste_item_from_matte
        from ..engine.paste_fusion import rgba_to_pixmap

        name = os.path.basename(result.path)
        if getattr(self, 'edit_mode', 'paste') == 'annotate':
            self.status_label.setText(f"{tr('已抠出')}: {name}")
            return

        self.save_undo_state()
        item = paste_item_from_matte(
            pixmap=rgba_to_pixmap(result.rgba),
            rect=result.rect, label=result.label, path=result.path)
        self.canvas_items.append(item)
        # 抠完直接选中这一项：紧接着旋转/裁剪都作用在它身上
        self.selected_item = len(self.canvas_items) - 1
        _x, _y, _w, _h = item.rect_tuple()
        self.selected_item_size = (_w, _h)
        if self.current_background_index >= 0:
            self.canvas_items_dict[self.current_background_index] = \
                self.canvas_items.copy()
        self._refresh_export_task_label()
        # 素材已经写进文件夹：刷新贴图列表，让它在列表里立刻能选到
        # （从文件夹加载的列表会重扫目录，抠图输出目录里的新文件就会出现）
        refresh_list = getattr(self, 'refresh_paste_files', None)
        if callable(refresh_list):
            refresh_list()
        self.canvas.update()
        self.status_label.setText(f"{tr('已抠出')}: {name}")

    def crop_matte_material(self, item_index, rect):
        """把抠图素材裁到 rect（图像坐标），结果覆盖写回原 PNG。

        画布上的贴图项同步改成裁剪后的位置与尺寸；旋转过的贴图不允许裁剪，
        否则写回的像素与画布上看到的不一致。
        """
        from ..engine.matting_service import save_matte_png
        from ..engine.paste_fusion import pixmap_to_rgba, rgba_to_pixmap
        from ..engine.paste_item import PasteItem, as_paste_item

        items = getattr(self, 'canvas_items', None) or []
        if not (0 <= item_index < len(items)):
            return False

        item = as_paste_item(items[item_index])
        path = getattr(item, 'path', None)
        if not path:
            self.status_label.setText(tr("该贴图不是抠图素材，无法写回文件"))
            return False
        if abs(item.angle) > 1e-9:
            self.status_label.setText(tr("请先重置旋转再裁剪"))
            return False

        x, y, w, h = item.rect_tuple()
        left = max(x, float(rect[0]))
        top = max(y, float(rect[1]))
        right = min(x + w, float(rect[0]) + float(rect[2]))
        bottom = min(y + h, float(rect[1]) + float(rect[3]))
        if right - left < 1 or bottom - top < 1:
            self.status_label.setText(tr("裁剪范围太小"))
            return False

        source = pixmap_to_rgba(item.pixmap, max(1, int(round(w))),
                                max(1, int(round(h))))
        scale_x = source.shape[1] / float(w)
        scale_y = source.shape[0] / float(h)
        x0 = max(0, int(round((left - x) * scale_x)))
        y0 = max(0, int(round((top - y) * scale_y)))
        x1 = min(source.shape[1], int(round((right - x) * scale_x)))
        y1 = min(source.shape[0], int(round((bottom - y) * scale_y)))
        cropped = source[y0:y1, x0:x1]
        if cropped.size == 0 or not cropped[:, :, 3].any():
            self.status_label.setText(tr("裁剪后没有内容"))
            return False

        self.save_undo_state()
        try:
            save_matte_png(cropped, path)
        except OSError as exc:
            self.status_label.setText(
                f"{tr('裁剪失败')}: {exc}")
            return False

        new_pixmap = rgba_to_pixmap(cropped)
        items[item_index] = PasteItem(
            pixmap=new_pixmap,
            rect=(left, top, right - left, bottom - top),
            label=item.label, angle=0.0, path=path)
        if self.current_background_index >= 0:
            self.canvas_items_dict[self.current_background_index] = (
                self.canvas_items.copy())
        self.selected_item = item_index
        _cx, _cy, _cw, _ch = items[item_index].rect_tuple()
        self.selected_item_size = (_cw, _ch)
        self._refresh_export_task_label()
        # 文件变了，列表里那份缓存也得换：否则缩略图还是旧图，
        # 再从列表拖到画布上拿到的也是裁剪前的像素。
        sync = getattr(self, 'sync_paste_pixmap', None)
        if callable(sync):
            sync(path, new_pixmap)
        self.canvas.update()
        self.status_label.setText(
            f"{tr('已裁剪素材')}: {os.path.basename(path)}")
        return True

    def _on_matte_failed(self, message):
        """抠图失败：只报原因，画布与标注都不动。"""
        self.status_label.setText(f"{tr('抠图失败')}: {message}")

    def _on_matte_worker_finished(self):
        """线程收尾：无论成功、失败还是取消都复位忙碌态。"""
        self._busy = False
        self._matte_worker = None

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
        """顶部'导出'按钮弹出菜单: 增强划分 / 数据集分组 / 图像编辑"""
        from PySide6.QtWidgets import QMenu
        from PySide6.QtCore import QPoint

        menu = QMenu(self)
        menu.setObjectName("optionsMenu")
        action_aug = menu.addAction(tr("增强划分"))
        action_image_tools = menu.addAction(tr("图像编辑"))
        action_group = menu.addAction(tr("数据集分组"))

        btn_pos = self.process_btn.mapToGlobal(QPoint(0, self.process_btn.height()))
        action = menu.exec_(btn_pos)

        if action == action_aug:
            self._toggle_processing_panel()
        elif action == action_group:
            self._open_dataset_grouper()
        elif action == action_image_tools:
            self._open_image_tools()

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
        # 抠图线程同理：活线程会在窗口销毁后继续写盘
        matte_worker = getattr(self, '_matte_worker', None)
        if matte_worker is not None and matte_worker.isRunning():
            matte_worker.cancel()
            matte_worker.wait(3000)
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
        for entry in self.canvas_items:
            label = as_paste_item(entry).label
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
