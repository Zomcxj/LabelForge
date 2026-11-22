"""算子登记表：分组、显示名与参数范围。

参数默认值不在这里重复写一遍，一律从算子类的构造签名取，这样「类默认」与
「界面默认」只有一处，两边不会各写一个数再慢慢漂开。本模块只补界面需要的
上下限与可选项。
"""
import inspect

from .base import get_all_transforms

GEOMETRY = "几何"
COLOR = "色彩"
FILTER = "滤镜"
NOISE = "噪声"

#: 界面按这个顺序排分组
GROUP_ORDER = (GEOMETRY, COLOR, FILTER, NOISE)

#: 卷积核一类的取值档位
KSIZE_CHOICES = [3, 5, 7, 9, 15, 21, 31]

# key -> (分组, 显示名, 参数规格)
# 参数规格四种写法：
#   ("名字", "int",   下限, 上限)
#   ("名字", "float", 下限, 上限)
#   ("名字", "choice", 可选项)
#   ("名字", "bool")
# 默认值取算子构造签名的默认；签名里没有则取第一个可选项或 False。
BASE_SPECS = {
    # ---- 几何 ----
    "fliph": (GEOMETRY, "水平翻转", ()),
    "flipv": (GEOMETRY, "垂直翻转", ()),
    "trans": (GEOMETRY, "平移", (
        ("max_dx", "int", 0, 100, (10, 30)),
        ("max_dy", "int", 0, 100, (10, 30)),
    )),
    "rotate": (GEOMETRY, "旋转", (
        ("max_angle", "float", 0.0, 45.0, (5, 25)),
    )),
    "scale": (GEOMETRY, "缩放", (
        ("min", "float", 0.1, 1.0, (0.8, 1.0)),
        ("max", "float", 1.0, 3.0, (1.0, 1.2)),
    )),
    "rotate90": (GEOMETRY, "旋转 90 度", (
        ("angle", "choice", [90, 180, 270]),
    )),
    "skew": (GEOMETRY, "错切", (
        ("x_angle", "float", -45.0, 45.0),
        ("y_angle", "float", -45.0, 45.0),
    )),
    "perspective": (GEOMETRY, "透视", (
        ("offset", "int", 0, 200),
    )),
    "crop_region": (GEOMETRY, "按选区裁剪", ()),
    "pad": (GEOMETRY, "边框填充", (
        ("top", "int", 0, 500),
        ("bottom", "int", 0, 500),
        ("left", "int", 0, 500),
        ("right", "int", 0, 500),
    )),
    "resize": (GEOMETRY, "按长边缩放", (
        ("long_side", "int", 16, 8192),
    )),

    # ---- 色彩 ----
    "bright": (COLOR, "亮度", (
        ("delta", "int", -100, 100, (-30, 30)),
    )),
    "contrast": (COLOR, "对比度", (
        ("factor", "float", 0.5, 2.0, (1.2, 1.8)),
    )),
    "hue": (COLOR, "色相", (
        ("delta", "int", -180, 180, (-30, 30)),
    )),
    "saturation": (COLOR, "饱和度", (
        ("factor", "float", 0.0, 2.0, (1.2, 1.8)),
    )),
    "value": (COLOR, "明度", (
        ("delta", "int", -100, 100, (-30, 30)),
    )),
    "invert": (COLOR, "反色", ()),
    "gamma": (COLOR, "Gamma", (
        ("gamma", "float", 0.1, 3.0),
    )),
    "white_balance": (COLOR, "白平衡", (
        ("temperature", "int", -100, 100),
    )),
    "hist_equal": (COLOR, "直方图均衡", ()),
    "grayscale": (COLOR, "灰度", ()),
    "levels": (COLOR, "色阶", (
        ("black", "int", 0, 254),
        ("white", "int", 1, 255),
    )),
    "channel": (COLOR, "通道提取", (
        ("channel", "choice", ["r", "g", "b"]),
    )),
    "clahe": (COLOR, "自适应均衡", (
        ("clip", "float", 0.5, 8.0),
        ("tile", "int", 2, 32),
    )),
    "auto_levels": (COLOR, "自动色阶", (
        ("clip_percent", "float", 0.0, 5.0),
    )),
    "posterize": (COLOR, "色调分离", (
        ("levels", "int", 2, 32),
    )),

    # ---- 滤镜 ----
    "gauss_blur": (FILTER, "高斯模糊", (
        ("ksize", "choice", KSIZE_CHOICES),
    )),
    "box_blur": (FILTER, "均值模糊", (
        ("ksize", "choice", KSIZE_CHOICES),
    )),
    "median": (FILTER, "中值滤波", (
        ("ksize", "choice", KSIZE_CHOICES),
    )),
    "sharpen": (FILTER, "锐化", (
        ("strength", "float", 0.0, 3.0),
    )),
    "sobel": (FILTER, "边缘检测", (
        ("strength", "float", 0.0, 3.0),
    )),
    "emboss": (FILTER, "浮雕", (
        ("strength", "float", 0.0, 1.0),
    )),
    "dilate": (FILTER, "膨胀", (
        ("ksize", "choice", KSIZE_CHOICES),
    )),
    "erode": (FILTER, "腐蚀", (
        ("ksize", "choice", KSIZE_CHOICES),
    )),
    "threshold": (FILTER, "二值化", (
        ("thresh", "int", 0, 255),
        ("invert", "bool"),
    )),
    "adaptive_thresh": (FILTER, "自适应二值化", (
        ("block_size", "choice", [3, 5, 7, 11, 15, 21, 31]),
        ("c", "float", 0.0, 20.0),
    )),
    "morph_open": (FILTER, "开运算", (("ksize", "choice", KSIZE_CHOICES),)),
    "morph_close": (FILTER, "闭运算", (("ksize", "choice", KSIZE_CHOICES),)),
    "morph_gradient": (FILTER, "形态学梯度",
                       (("ksize", "choice", KSIZE_CHOICES),)),
    "tophat": (FILTER, "顶帽", (("ksize", "choice", KSIZE_CHOICES),)),
    "blackhat": (FILTER, "黑帽", (("ksize", "choice", KSIZE_CHOICES),)),
    "usm": (FILTER, "USM 锐化", (
        ("radius", "float", 0.5, 15.0),
        ("amount", "float", 0.0, 3.0),
    )),
    "laplacian": (FILTER, "拉普拉斯锐化",
                  (("ksize", "choice", KSIZE_CHOICES),)),
    "bilateral": (FILTER, "双边滤波", (
        ("d", "int", 1, 31),
        ("sigma_color", "float", 1.0, 200.0),
        ("sigma_space", "float", 1.0, 200.0),
    )),
    "nlmeans": (FILTER, "非局部均值降噪", (
        ("h", "float", 1.0, 40.0),
    )),
    "motion_blur": (FILTER, "运动模糊", (
        ("length", "int", 1, 99),
        ("angle", "float", 0.0, 180.0),
    )),
    "mosaic": (FILTER, "马赛克", (
        ("block", "int", 2, 64),
    )),

    # ---- 噪声 ----
    "gauss": (NOISE, "高斯噪声", (
        ("mean", "int", 0, 60),
        ("sigma", "int", 1, 100, (10, 40)),
    )),
    "saltpepper": (NOISE, "椒盐噪声", (
        ("prob", "float", 0.0, 0.5, (0.02, 0.08)),
    )),
}


class ParamSpec:
    """一个界面参数：取值范围与默认值。"""

    __slots__ = ("name", "kind", "lo", "hi", "choices", "default", "sample")

    def __init__(self, name, kind, lo=None, hi=None, choices=None,
                 default=None, sample=None):
        self.name = name
        self.kind = kind
        self.lo = lo
        self.hi = hi
        self.choices = choices
        self.default = default
        # 批量增强时这个参数的默认采样区间；不给就取 (默认, 默认)
        self.sample = sample or ((default, default)
                                 if kind in ("int", "float") else None)

    def as_tuple(self):
        """界面要的规格元组，按类型给出不同的形状。

        数值参数：(名字, 类型, 下限, 上限, 默认)
        可选项：  (名字, "choice", 可选项列表, 默认)
        开关：    (名字, "bool", 默认)
        """
        if self.kind == "choice":
            return (self.name, self.kind, list(self.choices), self.default)
        if self.kind == "bool":
            return (self.name, self.kind, self.default)
        return (self.name, self.kind, self.lo, self.hi, self.default)

    def __repr__(self):
        return (f"ParamSpec({self.name!r}, {self.kind!r}, "
                f"default={self.default!r}, sample={self.sample!r})")


class OpSpec:
    """一个算子：分组、显示名与它的界面参数。"""

    __slots__ = ("key", "group", "label", "params", "cls")

    def __init__(self, key, group, label, params, cls):
        self.key = key
        self.group = group
        self.label = label
        self.params = tuple(params)
        self.cls = cls

    @property
    def numeric(self):
        return tuple(p for p in self.params if p.kind in ("int", "float"))

    def defaults(self):
        return {p.name: p.default for p in self.params}

    def __repr__(self):
        return f"OpSpec({self.key!r}, group={self.group!r})"


def _signature_default(cls, name):
    """算子构造签名里该参数的默认值；没有这个参数返回未给出标记。"""
    if cls is None:
        return _MISSING
    param = inspect.signature(cls.__init__).parameters.get(name)
    if param is None:
        return _MISSING
    return param.default


class _Missing:
    def __repr__(self):
        return "<未给出>"


_MISSING = _Missing()


def _make_param(cls, raw):
    name, kind = raw[0], raw[1]
    default = _signature_default(cls, name)
    if kind == "choice":
        choices = list(raw[2])
        if default is _MISSING or default is inspect.Parameter.empty:
            default = choices[0]
        return ParamSpec(name, kind, choices=choices, default=default)
    if kind == "bool":
        if default is _MISSING or default is inspect.Parameter.empty:
            default = False
        return ParamSpec(name, kind, default=bool(default))
    lo, hi = raw[2], raw[3]
    if default is _MISSING or default is inspect.Parameter.empty:
        default = lo
    default = max(lo, min(hi, default))
    sample = raw[4] if len(raw) > 4 else None
    return ParamSpec(name, kind, lo=lo, hi=hi, default=default, sample=sample)


def _build():
    registry = get_all_transforms()
    specs = {}
    for key, (group, label, params) in BASE_SPECS.items():
        cls = registry.get(key)
        specs[key] = OpSpec(key, group, label,
                            [_make_param(cls, raw) for raw in params], cls)
    return specs


#: key -> OpSpec
OP_SPECS = _build()

#: 全部 key，按分组顺序展开
OPERATION_ORDER = [key for group in GROUP_ORDER
                   for key in OP_SPECS if OP_SPECS[key].group == group]

#: 会改变画布尺寸的算子：不能做选区内逐像素合成
GEOMETRIC_KEYS = frozenset(key for key, spec in OP_SPECS.items()
                           if spec.group == GEOMETRY)


#: 算子类名 -> 显示名。增强进度回调拿到的是类名，用它换成中文再显示。
_CLASS_LABELS = {item.cls.__name__: item.label
                 for item in OP_SPECS.values() if item.cls is not None}


def label_of_class(class_name, fallback=None):
    return _CLASS_LABELS.get(class_name, fallback or class_name)


def group_of(key):
    return OP_SPECS[key].group


def label_of(key):
    return OP_SPECS[key].label


def params_of(key):
    return list(OP_SPECS[key].params)


def default_params(key):
    """一个算子的默认参数（界面打开时的初值）。"""
    return OP_SPECS[key].defaults()


def is_geometric(key):
    return key in GEOMETRIC_KEYS


def keys_in(group):
    return [key for key in OPERATION_ORDER if OP_SPECS[key].group == group]
