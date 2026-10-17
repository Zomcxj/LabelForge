"""构建期辅助：统一打包产物里的 MSVC 运行时版本。

Qt 与 onnxruntime 各带一份 MSVC 运行时。版本不一致时，先加载的那份会留在进程里，
之后加载的库只能绑到它身上；旧运行时没有新符号，调用时直接 0xC0000005 崩溃。
Qt 附带的 MSVCP140.dll 往往比 onnxruntime 需要的旧（PyQt5 时代那份是 2020 年的
14.26），于是「点推理」一导入 onnxruntime 就闪退。

PyQt5 把运行时放在 `Qt5\\bin\\`，PySide6 直接放在包根 `PySide6\\`——路径不同，
但走到这里的都只是「某个 msvcp140.dll 条目」，逻辑与放在哪儿无关。

处理办法：每个运行时文件名都取所有候选里版本最高的那份，覆盖到包内每个出现位置。
只替换不删除：某个名字挑不出更高版本时原样保留，不会因此丢文件。
"""

MSVC_RUNTIME_NAMES = frozenset({
    'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll', 'concrt140.dll',
    'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
    'vcruntime140.dll', 'vcruntime140_1.dll', 'vcruntime140_threads.dll',
})


def _file_version(path):
    """DLL 的 (fileVersionMS, fileVersionLS)；读不出来返回 (0, 0)。"""
    try:
        from PyInstaller.utils.win32.versioninfo import (
            read_version_info_from_executable,
        )
        ffi = read_version_info_from_executable(path).ffi
        return int(ffi.fileVersionMS), int(ffi.fileVersionLS)
    except Exception:
        return 0, 0


def candidate_dirs():
    """解释器自带的运行时目录。

    conda 的 Library\\bin 是一套完整且通常最新的运行时，而它不一定被 TOC 收录，
    所以要主动作为候选，否则只能退而求其次用别的目录里较旧的版本。
    """
    import os
    import sys

    return [
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(os.path.dirname(sys.executable), 'Library', 'bin'),
        sys.prefix,
        os.path.dirname(sys.executable),
    ]


def unify_msvc_runtime(binaries, extra_dirs=None):
    """把 TOC 里同名的 MSVC 运行时统一成版本最高的那一份。

    只改这些运行时条目的源路径，包内位置与条目数量都不变。
    """
    import os

    entries = list(binaries)
    pool = list(candidate_dirs() if extra_dirs is None else extra_dirs)
    for dest, src, _code in entries:
        if os.path.basename(dest).lower() in MSVC_RUNTIME_NAMES:
            pool.append(os.path.dirname(src))

    best = {}
    for directory in pool:
        for name in MSVC_RUNTIME_NAMES:
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            version = _file_version(path)
            if version == (0, 0):
                continue
            if name not in best or version > best[name][0]:
                best[name] = (version, path)
    if not best:
        return entries

    unified = []
    for dest, src, code in entries:
        name = os.path.basename(dest).lower()
        if name in best:
            src = best[name][1]
        unified.append((dest, src, code))
    return unified
