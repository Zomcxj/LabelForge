"""把 ico_image/labelforge.svg 渲染成应用图标的多尺寸位图。

用法：python tools/render_icon.py [输出目录]

为什么走 SVG：形状和渐变都在矢量源里改，位图只是产物。
每个尺寸单独渲染一次，小尺寸才不会糊。
ICO 由 PIL 合成 —— Qt 的 QImage.save 不接受 sizes 参数，写不出多档位图标。
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "ico_image" / "labelforge.svg"

# 图标内容占画布的比例。留一点边，任务栏里不被裁切
CONTENT = 0.92

ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def render(size: int, background: QColor | None) -> QImage:
    """按目标尺寸渲染，内容居中并撑到 CONTENT 比例。"""
    renderer = QSvgRenderer(str(SVG))
    if not renderer.isValid():
        raise SystemExit(f"SVG 无效: {SVG}")

    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(background if background is not None else Qt.transparent)

    box = renderer.defaultSize()
    scale = (size * CONTENT) / max(box.width(), box.height())
    w, h = box.width() * scale, box.height() * scale
    target = QRectF((size - w) / 2, (size - h) / 2, w, h)

    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(painter, target)
    painter.end()
    return img


def to_pil(img: QImage) -> Image.Image:
    img = img.convertToFormat(QImage.Format_RGBA8888)
    w, h = img.width(), img.height()
    data = img.constBits().tobytes()
    return Image.frombytes("RGBA", (w, h), data, "raw", "RGBA", img.bytesPerLine())


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "ico_image"
    out.mkdir(parents=True, exist_ok=True)

    to_pil(render(512, None)).save(out / "icoo.png")

    # 每档单独渲染，再由 PIL 打包成多尺寸 ICO
    frames = [to_pil(render(s, None)) for s in ICO_SIZES]
    frames[-1].save(out / "labelforge.ico", format="ICO",
                    sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1])

    print("icoo.png        512x512")
    print(f"labelforge.ico  {', '.join(str(s) for s in ICO_SIZES)}")
    return 0


if __name__ == "__main__":
    app = QApplication.instance() or QApplication([])
    raise SystemExit(main())
