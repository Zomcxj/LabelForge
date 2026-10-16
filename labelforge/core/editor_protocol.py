"""
EditorProtocol - 定义 Manager 可以访问的编辑器接口
Manager 依赖此协议而非直接依赖 ImageEditor，实现解耦和可测试性
"""
from typing import Protocol, runtime_checkable, List, Set, Dict, Optional, Any


@runtime_checkable
class EditorProtocol(Protocol):
    """Manager 可访问的编辑器数据接口（结构化子类型，无需显式继承）"""

    # ===== 只读数据属性 =====

    @property
    def current_background(self) -> Optional[Any]:
        """当前背景图 QPixmap 或 None"""
        ...

    @property
    def current_background_index(self) -> int:
        """当前背景图索引"""
        ...

    @property
    def background_images(self) -> List[str]:
        """背景图路径列表"""
        ...

    @property
    def canvas_items(self) -> List[tuple]:
        """当前画布贴图 list[(QPixmap, QRectF, str)]"""
        ...

    @property
    def canvas_items_dict(self) -> Dict[int, list]:
        """所有画布贴图字典"""
        ...

    @property
    def detection_boxes(self) -> List[dict]:
        """当前检测框"""
        ...

    @property
    def detection_boxes_dict(self) -> Dict[int, list]:
        """所有检测框字典"""
        ...

    @property
    def global_labels(self) -> Set[str]:
        """全局标签集合"""
        ...

    # ===== UI 控件只读访问 =====

    @property
    def prefix_checkbox(self) -> Any:
        """前缀复选框"""
        ...

    @property
    def prefix_input(self) -> Any:
        """前缀输入框"""
        ...

    @property
    def background_list(self) -> Any:
        """背景图列表控件"""
        ...

    @property
    def label_list(self) -> Any:
        """检测框标签列表控件"""
        ...

    @property
    def paste_label_list(self) -> Any:
        """贴图标签列表控件"""
        ...

    @property
    def canvas(self) -> Any:
        """画布控件"""
        ...

    @property
    def status_label(self) -> Any:
        """状态栏标签"""
        ...

    # ===== 可写数据 =====

    @canvas_items.setter
    def canvas_items(self, value: List[tuple]) -> None: ...

    @detection_boxes.setter
    def detection_boxes(self, value: List[dict]) -> None: ...

    @global_labels.setter
    def global_labels(self, value: Set[str]) -> None: ...
