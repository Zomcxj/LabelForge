"""诊断启动器：与 `main.py` 相同，但**强制开启 json 追踪探针**。

⚠️ 临时排查用，不参与正式发布。产物是 `LabelForge-debug.exe`。

用法：直接运行，然后复现「json 不见了」的流程，再把
`C:\\Users\\<你>\\.labelforge.log` 发我看。

日志里会带 `[JSONTRACE]` 前缀，能回答：
- 切到某张图时，它的 json 在不在、有几条 shape
- 切走前那张图的 json 状态（save_current_json 之前）
- 任何一次删除/覆盖：谁调的、调用栈、删完文件还在不在
"""
import os
import sys

# 必须在任何 labelforge 模块读取它之前设置
os.environ.setdefault('LABELFORGE_TRACE_JSON', '1')

if getattr(sys, 'frozen', False):
    base_dir = sys._MEIPASS
else:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from labelforge.core.exception_hook import install_exception_hook, _write_log
from labelforge.core import exception_hook
install_exception_hook()

_write_log("=" * 60)
_write_log("[JSONTRACE] debug 启动器：json 追踪已开启")
_write_log(f"[JSONTRACE] 日志文件：{exception_hook.LOG_FILE}")

from labelforge.ui.main_window import main

if __name__ == "__main__":
    main()
