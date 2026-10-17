# -*- mode: python ; coding: utf-8 -*-
import os
import sys

_root = os.getcwd()

sys.path.insert(0, SPECPATH)
from build_support import unify_msvc_runtime

_ffi_dll = os.path.join(sys.prefix, 'Library', 'bin', 'ffi.dll')
_binaries = []
if os.path.exists(_ffi_dll):
    _binaries.append((_ffi_dll, '.'))

_icon = [os.path.join(_root, 'ico_image', 'icoo.png')] if sys.platform == 'win32' else []

# 格式转换（自研实现，零外部依赖，仅用标准库 + numpy）
_extra_hiddenimports = [
    'labelforge.engine.dataset_converter',
    'labelforge.engine.shape_io',
    'labelforge.ui.dataset_tools_dialog',
    'labelforge.ui.mixins.dataset_tools',
    'labelforge.engine.onnx_inference',
    'labelforge.engine.inference_image',
    'labelforge.engine.inference_task',
    'labelforge.ui.model_row',
    'labelforge.ui.inference_workers',
    'labelforge.ui.inference_progress',
    'labelforge.ui.class_filter_panel',
    'labelforge.ui.scope_panel',
    'labelforge.ui.sliding_window_dialog',
    'labelforge.ui.mixins.inference',
    'labelforge.engine.inference_scope',
    # onnxruntime 在函数内导入，显式声明避免打包时漏收原生库
    'onnxruntime',
    'cv2',
]

a = Analysis(
    [os.path.join(_root, 'labelforge', 'main.py')],
    pathex=[_root],
    binaries=_binaries,
    datas=[(os.path.join(_root, 'ico_image'), 'ico_image'),
           (os.path.join(_root, 'LICENSE'), '.'),
           (os.path.join(_root, 'THIRD-PARTY.md'), '.')],
    hiddenimports=_extra_hiddenimports + ['labelforge', 'labelforge.main', 'labelforge.ui', 'labelforge.ui.main_window', 'labelforge.ui.icons', 'labelforge.ui.settings_dialog', 'labelforge.ui.theme', 'labelforge.ui.dwm', 'labelforge.ui.dialogs', 'labelforge.ui.i18n', 'labelforge.ui.dialog_helpers', 'labelforge.ui.memory_dialog', 'labelforge.ui.segmented_control', 'labelforge.ui.processing_panel', 'labelforge.ui.dataset_classifier_dialog',
    'labelforge.ui.video_dialog', 'labelforge.ui.mixins', 'labelforge.ui.mixins.dataset_classifier', 'labelforge.ui.mixins.background_list', 'labelforge.ui.mixins.label_cache_slot', 'labelforge.ui.mixins.memory_record', 'labelforge.ui.mixins.stats', 'labelforge.ui.mixins.theme', 'labelforge.ui.mixins.translation', 'labelforge.ui.mixins.toolbar', 'labelforge.ui.mixins.options_popup', 'labelforge.ui.mixins.cache_menu',
    'labelforge.ui.mixins.video', 'labelforge.ui.mixins.lists', 'labelforge.ui.mixins.panels', 'labelforge.ui.mixins.processing_panel_builder', 'labelforge.widgets', 'labelforge.widgets.canvas_adjustment', 'labelforge.widgets.spinner', 'labelforge.widgets.drag_out_list', 'labelforge.widgets.hover_popup', 'labelforge.widgets.hover_menu', 'labelforge.widgets.processing', 'labelforge.engine', 'labelforge.engine.save_manager',
    'labelforge.engine.video_io', 'labelforge.engine.video_render', 'labelforge.engine.undo_manager', 'labelforge.engine.label_manager', 'labelforge.engine.image_loader', 'labelforge.engine.image_loader.status', 'labelforge.engine.image_loader.scan', 'labelforge.engine.image_loader.mixin', 'labelforge.engine.paste_engine', 'labelforge.engine.event_handler', 'labelforge.engine.augmenter', 'labelforge.engine.augmenter.base', 'labelforge.engine.augmenter.color', 'labelforge.engine.augmenter.flipt', 'labelforge.engine.augmenter.noise', 'labelforge.engine.augmenter.translate', 'labelforge.engine.augmenter.rotate', 'labelforge.engine.augmenter.scale', 'labelforge.engine.yolo_exporter', 'labelforge.engine.base_exporter', 'labelforge.engine.voc_exporter', 'labelforge.engine.coco_exporter', 'labelforge.engine.splitter', 'labelforge.engine.dataset_classifier', 'labelforge.canvas', 'labelforge.canvas.canvas', 'labelforge.canvas.canvas_renderer', 'labelforge.canvas.canvas_interaction', 'labelforge.canvas.canvas_drawing', 'labelforge.canvas.canvas_menu', 'labelforge.core', 'labelforge.core.config', 'labelforge.core.config_manager', 'labelforge.core.utils', 'labelforge.core.editor_protocol', 'labelforge.core.exception_hook', 'pkgutil'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 通用体积护栏：保留 PySide6 / numpy / onnxruntime / cv2，排除未用的大型依赖。
    # **不能**再排除 PySide6：迁移前排除它是为瘦身，如今 Qt 就是 PySide6，排除掉
    # 构建照样成功，打出的 exe 却缺 Qt、启动即崩（PyInstaller 不会报错）。
    # 反过来把 PyQt5 排掉是对的：PyInstaller 明令不允许同时收集两套 Qt bindings
    # （utils/hooks/qt.py: ensure_single_qt_bindings_package），环境里残留
    # PyQt5 会把构建直接打断。
    excludes=['tkinter', 'matplotlib', 'pandas', 'pytest',
              'PyQt5', 'PyQt6', 'PySide2',
              'av', 'IPython', 'ipykernel', 'jedi', 'prompt_toolkit', 'pygments',
              'backcall', 'stack_data', 'matplotlib_inline', 'notebook', 'nbformat',
              'nbclient', 'traitlets', 'zmq', 'pyreadline3', 'pickleshare',
              'nest_asyncio', 'appnope', 'gevent', 'html5lib', 'bs4',
              'Cython', 'cython', 'lxml', 'cryptography', 'yaml', 'scipy', 'supervision'],
    noarchive=False,
    optimize=2,
)
# Qt 与 onnxruntime 各带一份 MSVC 运行时，版本不一致会让后加载的库绑到旧版上，
# 导入即 0xC0000005；这里统一成一套，避免点「推理」闪退。
a.binaries = unify_msvc_runtime(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    ([('-O2', None, 'OPTION')] if sys.platform == 'win32' else []),
    name='LabelForge',
    onefile=(sys.platform != 'win32'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_icon,
)
