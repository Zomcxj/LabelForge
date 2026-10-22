"""导出/增强路径取宽高必须只读文件头（Task 15 回归钉）。

`ProcessingPanel._get_image_size()` 原先无条件 `QImage(path)` **整图解码**
只为拿宽高；`engine.dataset_converter.read_image_size()` 早已只读文件头
（PNG/JPEG/GIF/BMP/WebP/TIFF…），导出几千张图时省掉的解码时间相当可观。

钉三件事：
1. 对可识别格式：不构造 QImage（0 次解码），返回值与 read_image_size 一致；
2. read_image_size 返回 None（不识别/损坏）的格式：仍回退到现有 QImage
   逻辑，解不出时保持 (0, 0) 语义；
3. 文件头读取意外抛异常：不炸、照旧走 QImage 兜底。

conftest 把 PySide6 换成了桩，QImage 是构造即可替的类属性，正好用来
统计「构造次数」——这里替换的是函数内 `from PySide6.QtGui import QImage`
解析到的那个模块属性，对真实 Qt 语义没有影响。
"""
import sys

import pytest

from labelforge.engine.dataset_converter import read_image_size
from tests.test_dataset_converter import make_png

_GARBAGE = b"this is not an image at all" * 64


def _make_panel():
    from labelforge.ui.processing_panel import ProcessingPanel
    return ProcessingPanel.__new__(ProcessingPanel)


class _QImageSpy:
    """替身 QImage：记录构造次数，可伪装成「能解码出 (30, 40)」或空图。"""

    instances = []
    decodes = True

    def __init__(self, *args, **kwargs):
        type(self).instances.append(self)

    def isNull(self):
        return not type(self).decodes

    def width(self):
        return 30

    def height(self):
        return 40


@pytest.fixture
def qimage_spy(monkeypatch):
    """把 QtGui 模块里的 QImage 换成计数替身（函数内 import 拿到的就是它）。"""
    qtgui = sys.modules["PySide6.QtGui"]
    _QImageSpy.instances = []
    _QImageSpy.decodes = True
    monkeypatch.setattr(qtgui, "QImage", _QImageSpy)
    return _QImageSpy


def test_get_image_size_reads_header_without_decoding(tmp_path, qimage_spy):
    """大图也只读文件头：0 次 QImage 构造，尺寸与 read_image_size 一致。"""
    big_png = tmp_path / "big.png"
    big_png.write_bytes(make_png(1024, 768))

    result = _make_panel()._get_image_size(str(big_png))

    assert qimage_spy.instances == [], (
        f"拿宽高不该整图解码（QImage 构造了 {len(qimage_spy.instances)} 次）")
    assert result == read_image_size(str(big_png))
    assert result == (1024, 768)


def test_get_image_size_falls_back_to_qimage_when_header_unknown(tmp_path, qimage_spy):
    """read_image_size 认不出的格式必须回退到现有 QImage 逻辑。"""
    weird = tmp_path / "weird.bin"
    weird.write_bytes(_GARBAGE)
    assert read_image_size(str(weird)) is None

    result = _make_panel()._get_image_size(str(weird))

    assert len(qimage_spy.instances) == 1
    assert result == (30, 40)


def test_get_image_size_still_returns_zero_zero_when_fallback_fails(tmp_path, qimage_spy):
    """兜底也解不出时保持 (0, 0) 语义不变。"""
    weird = tmp_path / "weird.bin"
    weird.write_bytes(_GARBAGE)
    qimage_spy.decodes = False

    result = _make_panel()._get_image_size(str(weird))

    assert result == (0, 0)


def test_get_image_size_survives_header_reader_crash(tmp_path, qimage_spy, monkeypatch):
    """文件头读取意外抛异常：照旧记日志并走 QImage 兜底，不向上抛。"""
    import labelforge.engine.dataset_converter as dc

    def _boom(path):
        raise RuntimeError("header reader exploded")

    monkeypatch.setattr(dc, "read_image_size", _boom)
    weird = tmp_path / "weird.bin"
    weird.write_bytes(_GARBAGE)

    result = _make_panel()._get_image_size(str(weird))

    assert result == (30, 40)


def test_get_image_size_matches_qimage_on_a_headerless_only_format(tmp_path, qimage_spy):
    """两路结果一致才允许换源：可识别格式的值必须与 QImage 一致。"""
    png = tmp_path / "small.png"
    png.write_bytes(make_png(150, 200))

    result = _make_panel()._get_image_size(str(png))

    assert result == (150, 200)
    assert qimage_spy.instances == []
