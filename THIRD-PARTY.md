# 第三方组件与 Qt 源码获取方式

本程序通过 **PySide6** 使用 **Qt**。Qt 由 The Qt Company 按 **LGPL-3.0** 授权，
其使用受 GNU Lesser General Public License v3.0 约束（正文见 [LICENSE](./LICENSE)）。

## 使用的 Qt

本项目声明的 PySide6 版本范围见 `requirements.txt`（当前 `>=6.5,<6.10`）。
实际打入发行包的版本以构建环境为准。

对应源码：

- PySide6 / shiboken6 官方发行：<https://download.qt.io/official_releases/QtForPython/>
- PySide6 在 PyPI 的源码包：<https://pypi.org/project/PySide6/#files>
- Qt 本体源码：<https://code.qt.io/>

## 如何替换 Qt

LGPL-3.0 §4(d)(0) 要求用户能够用改造过的 Qt 与本程序重新组合。本程序是 Python
应用且源码公开，因此：

1. 自行取得并改造 Qt / PySide6；
2. `pip install <改造过的 PySide6 包>`；
3. 从源码运行：`python labelforge/main.py`。

无需重新编译本程序，即可得到使用改造版 Qt 的组合作品。

## 其余第三方组件

| 组件 | 许可证 | 主页 |
|---|---|---|
| PySide6 / shiboken6 | LGPL-3.0（或 GPL-2.0 / GPL-3.0） | <https://www.qt.io/> |
| numpy | BSD-3-Clause | <https://numpy.org/> |
| onnxruntime | MIT | <https://onnxruntime.ai/> |
| opencv-python-headless | Apache-2.0（内含 FFmpeg，LGPL-2.1+） | <https://opencv.org/> |
| PyInstaller（启动器随产物分发） | GPL-2.0-or-later，附带允许打包任意许可软件的例外 | <https://pyinstaller.org/> |

以上组件的许可证正文可在其主页或发行包内取得，本仓库不重复收录。