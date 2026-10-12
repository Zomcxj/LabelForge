#!/bin/bash

# ==============================================================================
#  PasteLabel 构建脚本（pastelabel）
#  贴图标注工具主应用
#  支持: Linux / Windows (MSYS2 / Git Bash / Cygwin) / macOS
#  用法: ./build.sh [options]
#  选项:
#    --clean       : 强制清理缓存，从头构建（默认）
#    --no-clean    : 使用缓存，加快重复构建速度（推荐日常开发）
#
#  快捷方式:
#    ./build.sh           # 完整构建（发布版本）
#    ./build.sh fast      # 快速构建（日常开发）= --no-clean
# ==============================================================================

# 记录项目根目录（build.sh 所在目录）
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 解析命令行参数
CLEAN_BUILD=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-clean)
            CLEAN_BUILD=false
            shift
            ;;
        fast)
            CLEAN_BUILD=false
            shift
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--clean|--no-clean] | fast"
            echo ""
            echo "示例:"
            echo "  $0           # 完整构建（发布版本）"
            echo "  $0 fast      # 快速构建（日常开发）"
            exit 1
            ;;
    esac
done

set -e  # 任何命令失败时立即退出

# ==================== 颜色输出定义 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_info()  { printf "%b[INFO]%b  %s %s\n" "$BLUE" "$NC" "$(date '+%H:%M:%S')" "$*"; }
log_ok()    { printf "%b[OK]%b    %s %s\n" "$GREEN" "$NC" "$(date '+%H:%M:%S')" "$*"; }
log_warn()  { printf "%b[WARN]%b  %s %s\n" "$YELLOW" "$NC" "$(date '+%H:%M:%S')" "$*"; }
log_error() { printf "%b[ERROR]%b %s %s\n" "$RED" "$NC" "$(date '+%H:%M:%S')" "$*"; }

# ==================== 检测操作系统 ====================
log_info "检测操作系统..."
OS="$(uname -s)"
case "$OS" in
    Linux*)     PLATFORM="linux" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT) PLATFORM="windows" ;;
    Darwin*)    PLATFORM="macos" ;;
    *)
        log_error "不支持的操作系统: $OS"
        exit 1
        ;;
esac
log_ok "当前平台: $PLATFORM (uname: $OS)"



# ==================== 平台相关配置 ====================
if [ -n "${PYTHON:-}" ]; then
    PYTHON_CMD="$PYTHON"
elif [ "$PLATFORM" = "windows" ] && command -v python &> /dev/null; then
    # Windows 下 python3 可能是 Windows Store stub，优先 python
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    log_error "未找到 Python，请安装 Python 并确保 python/python3 在 PATH 中"
    exit 1
fi

PYINSTALLER_CMD="$PYTHON_CMD -m PyInstaller"

# ponytail: conda 的 _ctypes.pyd 依赖 ffi.dll（在 Library/bin），
# 不加到 PATH 的话 PyInstaller 二进制分析器找不到它
if [ "$PLATFORM" = "windows" ]; then
    PYTHON_DIR="$(dirname "$PYTHON_CMD")"
    CONDA_LIB_BIN="$(cygpath -u "$PYTHON_DIR/Library/bin" 2>/dev/null || echo "$PYTHON_DIR/Library/bin")"
    if [ -d "$CONDA_LIB_BIN" ]; then
        PATH="$CONDA_LIB_BIN:$PATH"
        log_info "已添加 conda Library/bin 到 PATH: $CONDA_LIB_BIN"
    fi
fi

if [ "$PLATFORM" = "windows" ]; then
    PYTHON_OPTIMIZE=2
    SEP="\\\\"
    # Git Bash/MSYS2 下需将 Unix 路径转为 Windows 路径（否则 Python 不认识）
    PROJECT_ROOT="$(cygpath -w "$PROJECT_ROOT" 2>/dev/null || echo "$PROJECT_ROOT")"
    OUTPUT_FILE="${PROJECT_ROOT}\\dist\\PasteLabel.exe"
else
    PYTHON_OPTIMIZE=0
    SEP="/"
    OUTPUT_FILE="${PROJECT_ROOT}/dist/PasteLabel"
fi

log_info "PyInstaller 命令: $PYINSTALLER_CMD"
log_info "Python optimize:  $PYTHON_OPTIMIZE"

# ==================== 构建辅助函数 ====================
ensure_output_not_locked() {
    # Windows 下如果 dist/PasteLabel.exe 正在运行，PyInstaller 在覆盖输出文件时会报：
    # PermissionError: [WinError 5] Access is denied: '...\\dist\\PasteLabel.exe'
    # 因此在打包前先主动删除旧产物；删除失败时尝试结束同名进程，并给出明确提示。
    if [ "$PLATFORM" != "windows" ]; then
        return 0
    fi

    log_info "检查输出文件是否可覆盖: $OUTPUT_FILE"

    if ! "$PYTHON_CMD" -c "import os, sys; raise SystemExit(0 if os.path.exists(sys.argv[1]) else 1)" "$OUTPUT_FILE" 2>/dev/null; then
        log_ok "未发现旧输出文件"
        return 0
    fi

    if "$PYTHON_CMD" -c "import os, sys; p=sys.argv[1]; os.path.exists(p) and os.remove(p)" "$OUTPUT_FILE" 2>/dev/null; then
        log_ok "已删除旧输出文件"
        return 0
    fi

    log_warn "无法删除旧输出文件，可能 PasteLabel.exe 正在运行或被安全软件占用"

    if command -v taskkill.exe &> /dev/null; then
        log_info "尝试结束正在运行的 PasteLabel.exe ..."
        taskkill.exe //F //IM PasteLabel.exe &> /dev/null || true
        sleep 1

        if "$PYTHON_CMD" -c "import os, sys; p=sys.argv[1]; os.path.exists(p) and os.remove(p)" "$OUTPUT_FILE" 2>/dev/null; then
            log_ok "已结束旧进程并删除旧输出文件"
            return 0
        fi
    fi

    log_error "无法覆盖输出文件: $OUTPUT_FILE"
    log_error "请关闭 PasteLabel.exe（或在任务管理器中结束 PasteLabel.exe），然后重新运行构建脚本。"
    log_error "如果仍失败，请检查杀毒/安全软件是否正在扫描或锁定该文件。"
    exit 1
}

# ==================== 前置检查 ====================
log_info "执行前置检查..."

# 检查 Python、pip、PyInstaller、PyQt5 是否来自同一个环境
if ! PYTHON_PATH="$($PYTHON_CMD -c 'import sys; print(sys.executable)' 2>/dev/null)"; then
    log_error "当前 Python 不可用: $PYTHON_CMD"
    exit 1
fi
log_info "Python: $PYTHON_PATH"
log_info "Python version: $($PYTHON_CMD --version)"

if ! "$PYTHON_CMD" -m pip --version &> /dev/null; then
    log_error "当前 Python 不可用 pip，请先安装 pip: $PYTHON_CMD"
    exit 1
fi

if ! "$PYTHON_CMD" -m PyInstaller --version &> /dev/null; then
    log_error "未找到 PyInstaller，请执行: $PYTHON_CMD -m pip install -r requirements-build.txt"
    exit 1
fi

if ! "$PYTHON_CMD" -c "import PyQt5" &> /dev/null; then
    log_error "未找到 PyQt5，请执行: $PYTHON_CMD -m pip install -r requirements.txt"
    exit 1
fi
log_ok "PyInstaller 可用"
log_ok "PyQt5 可用"

# 检查主脚本是否存在
MAIN_SCRIPT_PATH="${PROJECT_ROOT}${SEP}pastelabel${SEP}main.py"
if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
    log_error "主脚本 '$MAIN_SCRIPT_PATH' 不存在"
    exit 1
fi
log_ok "主脚本 '$MAIN_SCRIPT_PATH' 存在"

# 检查图标文件
ICON_FILE="${PROJECT_ROOT}${SEP}ico_image${SEP}icoo.png"
if [ ! -f "$ICON_FILE" ]; then
    log_warn "图标文件 '$ICON_FILE' 不存在，构建将继续但可能缺少图标"
else
    log_ok "图标文件 '$ICON_FILE' 存在"
fi

# 检查资源目录
RESOURCE_DIR="${PROJECT_ROOT}${SEP}ico_image"
if [ ! -d "$RESOURCE_DIR" ]; then
    log_warn "资源目录 '$RESOURCE_DIR' 不存在，构建将继续但可能缺少资源"
else
    log_ok "资源目录 '$RESOURCE_DIR' 存在"
fi

# ==================== 生成 spec 文件 ====================
# 策略：动态生成 spec 文件，与 Windows 已验证的方案一致
# 确保 pathex 指向项目根目录，让 PyInstaller 能正确发现 pastelabel 包结构

SPEC_FILE="${PROJECT_ROOT}${SEP}pastelabel${SEP}PasteLabel_build.spec"
MAIN_SCRIPT_SPEC="$($PYTHON_CMD -c 'import sys; print(repr(sys.argv[1]))' "$MAIN_SCRIPT_PATH")"
PROJECT_ROOT_SPEC="$($PYTHON_CMD -c 'import sys; print(repr(sys.argv[1]))' "$PROJECT_ROOT")"
RESOURCE_DIR_SPEC="$($PYTHON_CMD -c 'import sys; print(repr(sys.argv[1]))' "$RESOURCE_DIR")"
ICON_FILE_SPEC="$($PYTHON_CMD -c 'import sys; print(repr(sys.argv[1]))' "$ICON_FILE")"

log_info "生成 spec 文件..."

cat > "$SPEC_FILE" << 'SPECEOF'
# -*- mode: python ; coding: utf-8 -*-
# 自动生成 - 请勿手动修改
import os as _os
import sys as _sys

_sys.path.insert(0, SPECPATH)
from build_support import unify_msvc_runtime
SPECEOF

# 用追加方式写入 Python 列表和 Analysis 块（需要变量展开的部分）
cat >> "$SPEC_FILE" << SPECEOF

a = Analysis(
    [${MAIN_SCRIPT_SPEC}],
    pathex=[${PROJECT_ROOT_SPEC}],
    binaries=[],
    datas=[(${RESOURCE_DIR_SPEC}, 'ico_image'), (_os.path.join(${PROJECT_ROOT_SPEC}, 'LICENSE'), '.')],
    hiddenimports=[
        'pastelabel',         'pastelabel.ui', 'pastelabel.ui.main_window', 'pastelabel.ui.icons',
        'pastelabel.ui.settings_dialog', 'pastelabel.ui.theme', 'pastelabel.ui.dwm',
        'pastelabel.ui.dialogs', 'pastelabel.ui.i18n',
        'pastelabel.ui.processing_panel', 'pastelabel.ui.dataset_classifier_dialog',
        'pastelabel.engine', 'pastelabel.engine.save_manager',
        'pastelabel.engine.undo_manager', 'pastelabel.engine.label_manager',
        'pastelabel.engine.image_loader', 'pastelabel.engine.image_loader.status',
        'pastelabel.engine.image_loader.scan', 'pastelabel.engine.image_loader.mixin',
        'pastelabel.engine.paste_engine',
        'pastelabel.engine.event_handler',
        'pastelabel.engine.yolo_exporter', 'pastelabel.engine.splitter',
        'pastelabel.engine.dataset_classifier', 'pastelabel.ui.mixins.dataset_classifier',
        'pastelabel.engine.dataset_converter',
        'pastelabel.engine.shape_io',
        'pastelabel.ui.dataset_tools_dialog', 'pastelabel.ui.mixins.dataset_tools',
        'pastelabel.ui.model_row', 'pastelabel.ui.inference_workers',
        'pastelabel.ui.inference_progress', 'pastelabel.ui.class_filter_panel',
        'pastelabel.ui.mixins.inference',
        'pastelabel.engine.onnx_inference', 'pastelabel.engine.inference_image',
        'pastelabel.engine.inference_task',
        'pastelabel.ui.video_dialog', 'pastelabel.ui.mixins.video',
        'pastelabel.engine.video_io', 'pastelabel.engine.video_render',
        'numpy',
        # onnxruntime 在函数内导入，显式声明避免打包时漏收原生库
        'onnxruntime',
        'cv2',
        'pastelabel.canvas', 'pastelabel.canvas.canvas', 'pastelabel.canvas.canvas_renderer',
        'pastelabel.canvas.canvas_interaction', 'pastelabel.canvas.canvas_drawing',
        'pastelabel.canvas.canvas_menu',
        'pastelabel.core', 'pastelabel.core.config', 'pastelabel.core.config_manager',
        'pastelabel.core.utils', 'pastelabel.core.editor_protocol',
        'pastelabel.core.exception_hook',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 通用体积护栏：保留 PyQt5 / numpy / onnxruntime / cv2，排除未用的大型依赖
    excludes=['tkinter', 'matplotlib', 'pandas', 'pytest', 'PySide6', 'sklearn', 'torch', 'torchvision', 'tensorflow', 'keras', 'av',
              'IPython', 'ipykernel', 'jedi', 'prompt_toolkit', 'pygments', 'backcall', 'stack_data',
              'matplotlib_inline', 'notebook', 'nbformat', 'nbclient', 'traitlets', 'zmq',
              'pyreadline3', 'pickleshare', 'nest_asyncio', 'appnope', 'gevent', 'html5lib', 'bs4',
              'Cython', 'cython', 'lxml', 'cryptography', 'yaml', 'scipy', 'supervision'],
    noarchive=False,
    optimize=${PYTHON_OPTIMIZE},
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
$(if [ "$PLATFORM" = "windows" ]; then echo "    [('-O2', None, 'OPTION')],"; else echo "    [],"; fi)
    name='PasteLabel',
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
    icon=[${ICON_FILE_SPEC}],
)
SPECEOF

log_ok "spec 文件已生成: $SPEC_FILE"

# ==================== 执行打包 ====================
log_info "开始打包 ${PLATFORM} 应用程序..."

log_info "执行命令:"
echo "─────────────────────────────────────────"
echo "$PYINSTALLER_CMD \\"
echo "    $SPEC_FILE \\"
echo "    --distpath \"${PROJECT_ROOT}/dist\" \\"
echo "    --workpath \"${PROJECT_ROOT}/build\" \\"
[ "$CLEAN_BUILD" = true ] && echo "    --clean \\"
echo "    --noconfirm"
echo "─────────────────────────────────────────"

ensure_output_not_locked

if [ "$CLEAN_BUILD" = true ]; then
    log_info "清理模式：重新分析所有依赖（适合发布版本）"
    set +e
    "$PYTHON_CMD" -m PyInstaller \
        "$SPEC_FILE" \
        --distpath "${PROJECT_ROOT}/dist" \
        --workpath "${PROJECT_ROOT}/build" \
        --clean \
        --noconfirm
    BUILD_EXIT_CODE=$?
    set -e
else
    log_info "快速模式：使用缓存构建（适合日常开发）"
    set +e
    "$PYTHON_CMD" -m PyInstaller \
        "$SPEC_FILE" \
        --distpath "${PROJECT_ROOT}/dist" \
        --workpath "${PROJECT_ROOT}/build" \
        --noconfirm
    BUILD_EXIT_CODE=$?
    set -e
fi

# 清理临时 spec 文件
rm -f "$SPEC_FILE"

# ==================== 结果处理 ====================
echo ""
if [ $BUILD_EXIT_CODE -eq 0 ]; then
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        log_ok "打包成功！"
        log_ok "可执行文件: ${OUTPUT_FILE} (${FILE_SIZE})"
    else
        log_warn "打包命令执行成功，但未找到输出文件: ${OUTPUT_FILE}"
        log_warn "请检查 ${PROJECT_ROOT}/dist/ 目录"
    fi
    echo ""
    log_info "应用程序特性："
    log_info "  - 无控制台窗口 (-w)"
    log_info "  - 单文件打包 (-F)"
    log_info "  - 应用图标: ${ICON_FILE}"
    log_info "  - 资源目录: ${RESOURCE_DIR}/"
    log_info "  - 包含所有必要依赖"
    exit 0
else
    log_error "打包失败 (退出码: $BUILD_EXIT_CODE)，请检查上方错误信息"
    exit 1
fi
