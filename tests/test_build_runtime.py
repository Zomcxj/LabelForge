"""打包期 MSVC 运行时统一逻辑的回归测试。

背景：Qt 与 onnxruntime 各带一份 MSVC 运行时。版本不一致时先加载的那份会留在
进程里，之后加载的库只能绑到它身上；旧运行时缺新符号，调用即 0xC0000005。
PyQt5 的 Qt5\\bin\\MSVCP140.dll 是 2020 年的 14.26，正好踩中这一点，
表现为打包后点「推理」直接闪退、日志里没有 Python traceback。
"""
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pastelabel"))
from build_support import (  # noqa: E402
    MSVC_RUNTIME_NAMES, _file_version, unify_msvc_runtime,
)

ROOT = Path(__file__).resolve().parents[1]


def _make_fake_runtime(tmp_path, version, names=None):
    """造一个假的运行时目录；版本写成 fileVersionMS/LS，不依赖真实 DLL。"""
    directory = tmp_path / f"crt-{version[0]:x}-{version[1]:x}"
    directory.mkdir(parents=True, exist_ok=True)
    for name in (names or ("msvcp140.dll", "vcruntime140.dll")):
        (directory / name).write_bytes(b"fake")
    return str(directory)


@pytest.fixture
def fake_versions(monkeypatch):
    """让 _file_version 按目录名里的版本号返回，便于构造多版本场景。"""
    def fake(path):
        parent = os.path.basename(os.path.dirname(path))
        if not parent.startswith("crt-"):
            return (0, 0)
        ms, ls = parent.split("-")[1:]
        return (int(ms, 16), int(ls, 16))
    monkeypatch.setattr("build_support._file_version", fake)
    return fake


def test_older_runtime_entries_are_replaced_by_newest(fake_versions, tmp_path):
    """旧版本（如 PyQt5 自带的 14.26）必须被换成包内最高版本。"""
    old = _make_fake_runtime(tmp_path, (0xE001A, 0x70000000))
    new = _make_fake_runtime(tmp_path, (0xE0033, 0x8D000000))
    toc = [
        (r"PyQt5\Qt5\bin\MSVCP140.dll", os.path.join(old, "msvcp140.dll"), "BINARY"),
        ("msvcp140.dll", os.path.join(new, "msvcp140.dll"), "BINARY"),
    ]

    out = unify_msvc_runtime(toc, extra_dirs=[])

    assert [src for _d, src, _c in out] == [
        os.path.join(new, "msvcp140.dll"), os.path.join(new, "msvcp140.dll")]


def test_destination_paths_and_count_are_unchanged(fake_versions, tmp_path):
    """只换来源，包内位置和条目数不能变，否则运行时会缺文件。"""
    old = _make_fake_runtime(tmp_path, (0xE001A, 0x1))
    new = _make_fake_runtime(tmp_path, (0xE0033, 0x2))
    toc = [
        (r"PyQt5\Qt5\bin\MSVCP140.dll", os.path.join(old, "msvcp140.dll"), "BINARY"),
        ("msvcp140.dll", os.path.join(new, "msvcp140.dll"), "BINARY"),
        ("pastelabel/main.py", r"D:\x\main.py", "PYSOURCE"),
    ]

    out = unify_msvc_runtime(toc, extra_dirs=[])

    assert len(out) == len(toc)
    assert [d for d, _s, _c in out] == [d for d, _s, _c in toc]


def test_non_runtime_entries_are_untouched(fake_versions, tmp_path):
    new = _make_fake_runtime(tmp_path, (0xE0033, 0x2))
    toc = [
        ("msvcp140.dll", os.path.join(new, "msvcp140.dll"), "BINARY"),
        ("numpy.libs/msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll",
         r"D:\x\msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll", "BINARY"),
        ("keepme.dll", r"D:\x\keepme.dll", "BINARY"),
    ]

    out = unify_msvc_runtime(toc, extra_dirs=[])

    # mangled 名是 numpy 自己的私有副本，改了名就对不上导入表，必须原样保留
    assert out[1][1] == toc[1][1]
    assert out[2][1] == toc[2][1]


def test_entries_without_version_info_keep_their_source(fake_versions, tmp_path):
    """读不出版本的文件不能被换成别处同名文件，否则可能引入更旧的版本。"""
    new = _make_fake_runtime(tmp_path, (0xE0033, 0x2))
    toc = [
        ("msvcp140.dll", os.path.join(new, "msvcp140.dll"), "BINARY"),
        ("msvcp140_2.dll", r"D:\nowhere\msvcp140_2.dll", "BINARY"),
    ]

    out = unify_msvc_runtime(toc, extra_dirs=[])

    assert out[1][1] == r"D:\nowhere\msvcp140_2.dll"


def test_no_runtime_entries_is_a_noop(tmp_path):
    toc = [("pastelabel/main.py", r"D:\x\main.py", "PYSOURCE")]
    assert unify_msvc_runtime(toc, extra_dirs=[]) == toc


def test_candidate_dirs_include_conda_library_bin():
    """conda 的 Library\\bin 不在 TOC 里，必须主动作为候选，否则挑不到最新版。"""
    from build_support import candidate_dirs
    dirs = [os.path.normcase(d) for d in candidate_dirs()]
    assert os.path.normcase(os.path.join(sys.prefix, "Library", "bin")) in dirs


def test_real_dll_versions_are_readable():
    """真实 DLL 的版本号必须能读出来，否则统一逻辑会静默失效。"""
    conda_bin = os.path.join(sys.prefix, "Library", "bin", "msvcp140.dll")
    if not os.path.isfile(conda_bin):
        pytest.skip("当前环境没有 conda 运行时")
    version = _file_version(conda_bin)
    assert version != (0, 0), conda_bin
    assert version >= (0xE0028, 0), f"版本过低: {version!r}"


# ---------- spec 接线 ----------

@pytest.mark.parametrize("rel", ["build.sh", "pastelabel/PasteLabel.spec"])
def test_spec_files_unify_msvc_runtime(rel):
    """两个 spec 入口都要调用统一逻辑，否则总有一个构建路径仍会闪退。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    assert "unify_msvc_runtime" in source, rel
    assert "a.binaries = unify_msvc_runtime(a.binaries)" in source, rel


@pytest.mark.parametrize("rel", ["build.sh", "pastelabel/PasteLabel.spec"])
def test_spec_files_import_helper_from_pastelabel_dir(rel):
    """辅助模块在 pastelabel/ 下，spec 必须把该目录加进 sys.path 再导入。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    assert "from build_support import" in source, rel
    assert "SPECPATH" in source, rel


def test_runtime_name_set_covers_both_conflicting_libraries():
    """集合要覆盖 Qt 与 onnxruntime 实际导入的那几个名字。"""
    for name in ("msvcp140.dll", "msvcp140_1.dll",
                 "vcruntime140.dll", "vcruntime140_1.dll"):
        assert name in MSVC_RUNTIME_NAMES, name


# ---------- hiddenimports 必须真的存在 ----------

def _hidden_imports(rel):
    """取出 spec/脚本里 hiddenimports 列表里的模块名（忽略注释与其它字符串）。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    block = re.search(r"hiddenimports\s*=\s*\[(.*?)\]", source, re.S)
    assert block, rel
    names = []
    for line in block.group(1).splitlines():
        # 一行里可能写多个模块名，全部取出来
        names.extend(re.findall(r"""['"]([\w.]+)['"]""", line.split("#", 1)[0]))
    return names


@pytest.mark.parametrize("rel", ["build.sh", "pastelabel/PasteLabel.spec"])
def test_hidden_imports_point_to_existing_modules(rel):
    """改名/删模块后 spec 没跟着改，打包出来会在运行时才炸（推理窗口就这么炸过）。"""
    missing = []
    for name in _hidden_imports(rel):
        if not name.startswith("pastelabel."):
            continue
        path = ROOT / (name.replace(".", os.sep) + ".py")
        package = ROOT / name.replace(".", os.sep) / "__init__.py"
        if not path.exists() and not package.exists():
            missing.append(name)
    assert not missing, f"{rel} 里这些模块已不存在: {missing}"
