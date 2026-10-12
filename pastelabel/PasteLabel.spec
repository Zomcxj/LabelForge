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
    'pastelabel.engine.dataset_converter',
    'pastelabel.engine.shape_io',
    'pastelabel.ui.dataset_tools_dialog',
    'pastelabel.ui.mixins.dataset_tools',
    'pastelabel.engine.onnx_inference',
    'pastelabel.engine.inference_image',
    'pastelabel.engine.inference_task',
    'pastelabel.ui.model_row',
    'pastelabel.ui.inference_workers',
    'pastelabel.ui.inference_progress',
    'pastelabel.ui.class_filter_panel',
    'pastelabel.ui.mixins.inference',
    # onnxruntime 在函数内导入，显式声明避免打包时漏收原生库
    'onnxruntime',
    'cv2',
]

a = Analysis(
    [os.path.join(_root, 'pastelabel', 'main.py')],
    pathex=[_root],
    binaries=_binaries,
    datas=[(os.path.join(_root, 'ico_image'), 'ico_image'), (os.path.join(_root, 'LICENSE'), '.')],
    hiddenimports=_extra_hiddenimports + ['pastelabel', 'pastelabel.main', 'pastelabel.sip', 'pastelabel.ui', 'pastelabel.ui.main_window', 'pastelabel.ui.icons', 'pastelabel.ui.settings_dialog', 'pastelabel.ui.theme', 'pastelabel.ui.dwm', 'pastelabel.ui.dialogs', 'pastelabel.ui.i18n', 'pastelabel.ui.dialog_helpers', 'pastelabel.ui.memory_dialog', 'pastelabel.ui.segmented_control', 'pastelabel.ui.processing_panel', 'pastelabel.ui.dataset_classifier_dialog',
    'pastelabel.ui.video_dialog', 'pastelabel.ui.mixins', 'pastelabel.ui.mixins.dataset_classifier', 'pastelabel.ui.mixins.background_list', 'pastelabel.ui.mixins.label_cache_slot', 'pastelabel.ui.mixins.memory_record', 'pastelabel.ui.mixins.stats', 'pastelabel.ui.mixins.theme', 'pastelabel.ui.mixins.translation', 'pastelabel.ui.mixins.toolbar', 'pastelabel.ui.mixins.options_popup', 'pastelabel.ui.mixins.cache_menu',
    'pastelabel.ui.mixins.video', 'pastelabel.ui.mixins.lists', 'pastelabel.ui.mixins.panels', 'pastelabel.ui.mixins.processing_panel_builder', 'pastelabel.widgets', 'pastelabel.widgets.canvas_adjustment', 'pastelabel.widgets.spinner', 'pastelabel.widgets.drag_out_list', 'pastelabel.widgets.hover_popup', 'pastelabel.widgets.hover_menu', 'pastelabel.widgets.processing', 'pastelabel.engine', 'pastelabel.engine.save_manager',
    'pastelabel.engine.video_io', 'pastelabel.engine.video_render', 'pastelabel.engine.undo_manager', 'pastelabel.engine.label_manager', 'pastelabel.engine.image_loader', 'pastelabel.engine.image_loader.status', 'pastelabel.engine.image_loader.scan', 'pastelabel.engine.image_loader.mixin', 'pastelabel.engine.paste_engine', 'pastelabel.engine.event_handler', 'pastelabel.engine.augmenter', 'pastelabel.engine.augmenter.base', 'pastelabel.engine.augmenter.color', 'pastelabel.engine.augmenter.flipt', 'pastelabel.engine.augmenter.noise', 'pastelabel.engine.augmenter.translate', 'pastelabel.engine.augmenter.rotate', 'pastelabel.engine.augmenter.scale', 'pastelabel.engine.yolo_exporter', 'pastelabel.engine.base_exporter', 'pastelabel.engine.voc_exporter', 'pastelabel.engine.coco_exporter', 'pastelabel.engine.splitter', 'pastelabel.engine.dataset_classifier', 'pastelabel.canvas', 'pastelabel.canvas.canvas', 'pastelabel.canvas.canvas_renderer', 'pastelabel.canvas.canvas_interaction', 'pastelabel.canvas.canvas_drawing', 'pastelabel.canvas.canvas_menu', 'pastelabel.core', 'pastelabel.core.config', 'pastelabel.core.config_manager', 'pastelabel.core.utils', 'pastelabel.core.editor_protocol', 'pastelabel.core.exception_hook', 'pkgutil'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 通用体积护栏：保留 PyQt5 / numpy / onnxruntime / cv2，排除未用的大型依赖
    excludes=['tkinter', 'matplotlib', 'pandas', 'pytest', 'PySide6', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'PySide6.QtSvg', 'PySide6.QtNetwork',
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
    name='PasteLabel',
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
