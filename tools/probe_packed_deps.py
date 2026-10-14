"""把 exe 归档解包到临时目录，然后用**解出来的** cv2 / onnxruntime 做真实调用。

为什么要这么绕：exe 是 `console=False`，拿不到它的 stdout；而只在本机
Python 里 `import cv2` 什么也证明不了（本机 site-packages 本来就装着）。
把归档里的原生库解出来、从解出的目录导入、并真的解一帧视频，才能证明
「这份 exe 里带着能用的 cv2」，而不是「本机装着 cv2」。
"""
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from PyInstaller.archive.readers import CArchiveReader  # noqa: E402


def unpack(exe, dest):
    arch = CArchiveReader(str(exe))
    count = 0
    for name in arch.toc:
        if name.lower().endswith(".pyz"):
            continue
        try:
            data = arch.extract(name)
        except Exception:
            continue
        if data is None:
            continue
        rel = name.replace("\\", "/")
        target = pathlib.Path(dest) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        count += 1
    return count


def main():
    exe = pathlib.Path(sys.argv[1])
    video = pathlib.Path(sys.argv[2])
    dest = pathlib.Path(tempfile.mkdtemp(prefix="pl-unpack-"))
    n = unpack(exe, dest)
    print(f"解包 {n} 个文件 -> {dest}")

    # 只把「解出来的目录」放在最前：若导入成功，说明用的是归档里的库
    sys.path.insert(0, str(dest))
    import cv2                                   # noqa: E402
    print("cv2 版本:", cv2.__version__)
    print("cv2 位置:", cv2.__file__)
    print("来自归档:", str(dest) in str(cv2.__file__))

    cap = cv2.VideoCapture(str(video))
    ok = cap.isOpened()
    print("视频打开:", ok)
    if ok:
        frames = 0
        while frames < 5:
            got, frame = cap.read()
            if not got:
                break
            frames += 1
        print("读到帧数:", frames, "首帧尺寸:", frame.shape if frames else None)
        # 真的做一次图像运算，证明不是只有模块外壳
        import numpy as np                       # noqa: E402
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("转灰度 ok:", gray.shape, "均值>0:", float(gray.mean()) > 0)
    cap.release()

    import onnxruntime                        # noqa: E402
    print("onnxruntime:", onnxruntime.__version__)
    print("providers:", onnxruntime.get_available_providers())
    print("来自归档:", str(dest) in str(onnxruntime.__file__))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"耗时 {time.time() - t0:.1f}s")
