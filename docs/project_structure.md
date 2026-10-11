# 项目结构

[返回 README](../README.md)

```
PasteLabel/
├── .gitattributes             # 文本属性（Shell 脚本使用 LF）
├── .gitignore                 # Git 忽略规则
├── .python-version            # 项目 Python 版本锚点
├── build.sh                   # 跨平台 PyInstaller 构建脚本
├── paste_label.txt            # 默认贴图标签文件
├── requirements.txt           # 运行期依赖
├── requirements-dev.txt       # 开发与测试依赖
├── requirements-build.txt     # PyInstaller 构建依赖
├── README.md                  # 项目说明
├── LICENSE                    # MIT 许可证
├── ico_image/                 # 图标、演示图和字体资源
├── .github/
│   └── workflows/
│       ├── build.yml          # 发布构建工作流
│       └── test.yml           # 测试工作流
├── pastelabel/                # 主代码包
│   ├── __init__.py
│   ├── main.py                # 程序入口
│   ├── PasteLabel.spec        # PyInstaller spec
│   ├── build_support.py       # 打包辅助：统一 MSVC 运行库版本（Windows 专用，Linux 上为空操作）
│   ├── sip.py                 # PyInstaller 的 PyQt5 SIP 兼容模块
│   ├── canvas/                # 画布绘制与交互
│   │   ├── __init__.py
│   │   ├── canvas.py
│   │   ├── canvas_drawing.py
│   │   ├── canvas_interaction.py
│   │   ├── canvas_menu.py
│   │   └── canvas_renderer.py
│   ├── core/                  # 配置、工具与异常处理
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── config_manager.py
│   │   ├── editor_protocol.py
│   │   ├── exception_hook.py
│   │   └── utils.py
│   ├── engine/                # 图片、标签、保存、事件与数据处理
│   │   ├── __init__.py
│   │   ├── base_exporter.py
│   │   ├── coco_exporter.py
│   │   ├── dataset_classifier.py
│   │   ├── dataset_converter.py # 数据集格式转换（LabelMe/YOLO/COCO/VOC + 任务类型）
│   │   ├── dataset_health.py   # 数据集健康统计（分布数据与建议，纯逻辑）
│   │   ├── event_handler.py
│   │   ├── inference_image.py # QImage ↔ 模型输入（取图/letterbox 绘制）
│   │   ├── inference_task.py  # 推理配置校验、结果合并与 sidecar 写回（纯逻辑）
│   │   ├── onnx_inference.py  # ONNX 检测推理核心（v5/v8 解码、NMS、letterbox，无 Qt 依赖）
│   │   ├── image_loader/
│   │   │   ├── __init__.py
│   │   │   ├── mixin.py
│   │   │   ├── scan.py
│   │   │   └── status.py
│   │   ├── label_manager.py
│   │   ├── paste_engine.py
│   │   ├── quality_lint.py     # 标注质检扫描（越界/超小/重复/重叠/近似名/分组，纯逻辑）
│   │   ├── save_manager.py
│   │   ├── shape_io.py         # 形状编解码与 YOLO 行生成（矩形/多边形/关键点/旋转框）
│   │   ├── splitter.py
│   │   ├── undo_manager.py
│   │   ├── voc_exporter.py
│   │   ├── yolo_exporter.py
│   │   └── augmenter/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── color.py
│   │       ├── crop.py         # 滑窗裁剪纯函数（窗口起点/数量反推/跨窗标注过滤）
│   │       ├── flipt.py
│   │       ├── noise.py
│   │       ├── rotate.py
│   │       ├── scale.py
│   │       └── translate.py
│   ├── ui/                    # 主窗口、对话框、主题与国际化
│   │   ├── __init__.py
│   │   ├── dataset_classifier_dialog.py # KMeans 数据集分类窗口
│   │   ├── dataset_tools_dialog.py      # 数据集格式转换窗口
│   │   ├── crop_dialog.py     # 滑窗裁剪参数配置窗口（尺寸/重叠/预览）
│   │   ├── dialog_helpers.py
│   │   ├── dialogs.py
│   │   ├── dwm.py
│   │   ├── i18n.py
│   │   ├── icons.py           # SVG 图标常量与渲染辅助函数
│   │   ├── class_filter_panel.py # 类别筛选面板（可滚动勾选列表、全选/全不选）
│   │   ├── inference_progress.py # 推理进度小窗（非模态、可取消、跟随时区/主题、5 行日志）
│   │   ├── inference_workers.py  # 模型加载/批量推理后台线程与类别名输入
│   │   ├── main_window.py     # ImageEditor 协调器
│   │   ├── memory_dialog.py
│   │   ├── model_row.py       # 画布上方的模型参数行（加载/阈值/尺寸/覆盖模式/类别筛选/关闭）
│   │   ├── processing_panel.py # ProcessingPanel 执行与日志逻辑
│   │   ├── quality_lint_dialog.py # 标注质检结果窗口
│   │   ├── segmented_control.py
│   │   ├── settings_dialog.py
│   │   ├── theme.py
│   │   └── mixins/
│   │       ├── __init__.py
│   │       ├── background_list.py
│   │       ├── cache_menu.py
│   │       ├── dataset_classifier.py
│   │       ├── dataset_tools.py
│   │       ├── inference.py
│   │       ├── label_cache_slot.py
│   │       ├── lists.py
│   │       ├── memory_record.py
│   │       ├── options_popup.py
│   │       ├── panels.py
│   │       ├── processing_panel_builder.py
│   │       ├── quality_lint.py
│   │       ├── stats.py
│   │       ├── theme.py
│   │       ├── toolbar.py
│   │       └── translation.py
│   └── widgets/               # 可复用 Qt 控件
│       ├── canvas_adjustment.py
│       ├── drag_out_list.py
│       ├── hover_menu.py
│       ├── hover_popup.py
│       ├── processing.py      # 扫描提示、Worker 与折叠区控件
│       ├── spinner.py
│       └── task_badge.py      # 标签行任务徽标（det / seg / pose / obb）
├── tests/                     # pytest 回归测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_canvas_interaction_regressions.py
│   ├── test_canvas_menu_regressions.py
│   ├── test_canvas_renderer_regressions.py
│   ├── test_config.py
│   ├── test_config_manager.py
│   ├── test_dataset_classifier.py
│   ├── test_dialogs_regressions.py
│   ├── test_documentation_regressions.py
│   ├── test_i18n_regressions.py
│   ├── test_label_cache_multi_select_red.py
│   ├── test_background_label_scan.py # ProcessingPanel/background scan regression
│   ├── test_shape_io.py              # 形状编解码与 YOLO 行生成
│   ├── test_polygon_drawing.py       # 分割多边形绘制/编辑
│   ├── test_pose_drawing.py          # 关键点标注
│   ├── test_obb_drawing.py           # 旋转框绘制/旋转/导出
│   ├── test_dataset_converter.py     # 格式转换（含各任务往返）
│   ├── test_annotation_shortcuts.py  # 标注快捷键与分组对话框
│   ├── test_exporters.py             # YOLO/VOC/COCO 导出
│   ├── test_augmenter.py             # 数据增强（含形状保真）
│   ├── test_augmenter_crop.py        # 滑窗裁剪（网格公式/标注过滤/面板与小窗联动）
│   ├── test_dataset_health.py        # 数据集健康统计
│   ├── test_quality_lint.py          # 标注质检扫描与结果窗口
│   ├── test_label_manager_rename.py  # 标签改名级联（含子串误匹配回归）
│   ├── test_theme_style_consolidation.py
│   ├── test_silent_interaction_theme.py
│   ├── test_ui_layout_regressions.py
│   └── ...                     # 其他领域回归测试
└── docs/
    ├── project_structure.md   # 本文档
    ├── settings_guide.md      # 设置指南
    └── user_guide.md          # 用户使用指南
```
```
运行时文件（不属于仓库源码树）：
└── ~/pastelabel.log          # 未处理异常和 Qt 消息日志
```

运行时日志写入用户主目录 `~/pastelabel.log`，不属于仓库源码树中的持久文件。
