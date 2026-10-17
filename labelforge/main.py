"""
贴图标注工具 - 主程序入口
"""
import sys
import os

# PyInstaller 打包后，模块在 sys._MEIPASS 临时目录中
# 需要将临时目录加入 sys.path，让 Python 能找到 labelforge 包
if getattr(sys, 'frozen', False):
    # 打包后：临时解压目录
    base_dir = sys._MEIPASS
else:
    # 开发时：labelforge 的父目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# 安装全局异常钩子（必须在导入 Qt 之前）
from labelforge.core.exception_hook import install_exception_hook
install_exception_hook()

from labelforge.ui.main_window import main

if __name__ == "__main__":
    main()
