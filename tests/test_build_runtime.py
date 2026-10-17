"""打包期 MSVC 运行时统一逻辑的回归测试。

背景：Qt 与 onnxruntime 各带一份 MSVC 运行时。版本不一致时先加载的那份会留在
进程里，之后加载的库只能绑到它身上；旧运行时缺新符号，调用即 0xC0000005。
Qt 附带的那份曾经停在 2020 年的 14.26（PyQt5 时代的 `Qt5\\bin\\MSVCP140.dll`；
PySide6 放在包根 `PySide6\\msvcp140.dll`）正好踩中这一点，
表现为打包后点「推理」直接闪退、日志里没有 Python traceback。
"""
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labelforge"))
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
    """旧版本（如 Qt 自带的 14.26）必须被换成包内最高版本。"""
    old = _make_fake_runtime(tmp_path, (0xE001A, 0x70000000))
    new = _make_fake_runtime(tmp_path, (0xE0033, 0x8D000000))
    toc = [
        # 包内路径要用 os.path.join：写死反斜杠的话 Linux 上 basename 取不到
        # `msvcp140.dll`，去重逻辑认不出这两条是同一个文件。
        (os.path.join("PySide6", "msvcp140.dll"),
         os.path.join(old, "msvcp140.dll"), "BINARY"),
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
        (os.path.join("PySide6", "msvcp140.dll"),
         os.path.join(old, "msvcp140.dll"), "BINARY"),
        ("msvcp140.dll", os.path.join(new, "msvcp140.dll"), "BINARY"),
        ("labelforge/main.py", r"D:\x\main.py", "PYSOURCE"),
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
    toc = [("labelforge/main.py", r"D:\x\main.py", "PYSOURCE")]
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

@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_spec_files_unify_msvc_runtime(rel):
    """两个 spec 入口都要调用统一逻辑，否则总有一个构建路径仍会闪退。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    assert "unify_msvc_runtime" in source, rel
    assert "a.binaries = unify_msvc_runtime(a.binaries)" in source, rel


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_spec_files_import_helper_from_labelforge_dir(rel):
    """辅助模块在 labelforge/ 下，spec 必须把该目录加进 sys.path 再导入。"""
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
    """取出 spec/脚本里所有 hiddenimports 列表里的模块名（忽略注释与其它字符串）。

    spec 里有两个列表：`_extra_hiddenimports`（第 18 行）与主列表
    `hiddenimports=_extra_hiddenimports + [...]`。只读第一个会把加在主列表
    里的模块当成缺失，也会漏检主列表里的失效模块名（widgets 等只写在主列表）。
    """
    source = (ROOT / rel).read_text(encoding="utf-8")
    # 逐个 hiddenimports 赋值点，各自吃它后面的第一个方括号块
    names = []
    for match in re.finditer(r"hiddenimports\s*=", source):
        block = re.search(r"\[(.*?)\]", source[match.end():], re.S)
        if not block:
            continue
        for line in block.group(1).splitlines():
            # 一行里可能写多个模块名，全部取出来
            names.extend(re.findall(r"""['"]([\w.]+)['"]""", line.split("#", 1)[0]))
    assert names, rel
    return names


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_hidden_imports_point_to_existing_modules(rel):
    """改名/删模块后 spec 没跟着改，打包出来会在运行时才炸（推理窗口就这么炸过）。"""
    missing = []
    for name in _hidden_imports(rel):
        if not name.startswith("labelforge."):
            continue
        path = ROOT / (name.replace(".", os.sep) + ".py")
        package = ROOT / name.replace(".", os.sep) / "__init__.py"
        # 命名空间包没有 __init__.py（labelforge/widgets 就是），但目录存在即可导入
        namespace = (ROOT / name.replace(".", os.sep)).is_dir()
        if not path.exists() and not package.exists() and not namespace:
            missing.append(name)
    assert not missing, f"{rel} 里这些模块已不存在: {missing}"


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_hidden_imports_cover_inference_modules(rel):
    """推理相关的懒加载模块必须显式声明，否则打包后会缺模块。"""
    names = set(_hidden_imports(rel))
    for name in ("labelforge.engine.onnx_inference",
                 "labelforge.engine.inference_image",
                 "labelforge.engine.inference_task",
                 "labelforge.ui.mixins.inference",
                 "labelforge.ui.model_row",
                 "labelforge.ui.inference_workers",
                 "labelforge.ui.inference_progress"):
        assert name in names, f"{rel} 缺少 {name}"


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_hidden_imports_cover_video_modules(rel):
    """视频抽帧的懒加载模块必须显式声明，否则打包后点「视频抽帧」才炸。

    这三个模块都不在 main_window 的模块级导入链上（cv2 与对话框都是函数内
    导入），漏了不会在启动时报错。
    """
    names = set(_hidden_imports(rel))
    for name in ("labelforge.engine.video_io",
                 "labelforge.engine.video_render",
                 "labelforge.ui.video_dialog",
                 "labelforge.ui.mixins.video"):
        assert name in names, f"{rel} 缺少 {name}"


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_hidden_imports_cover_scope_modules(rel):
    """推理范围的懒加载模块必须显式声明，否则打包后点「推理范围」才炸。

    `scope_panel`、`sliding_window_dialog` 与 `inference_scope` 都在函数内导入
    （避免启动时就拉 Qt 对话框），漏了不会在启动时报错。
    """
    names = set(_hidden_imports(rel))
    for name in ("labelforge.engine.inference_scope",
                 "labelforge.ui.scope_panel",
                 "labelforge.ui.sliding_window_dialog"):
        assert name in names, f"{rel} 缺少 {name}"


# ---------- 视频依赖（cv2）接线 ----------

def _excludes(rel):
    """取出 spec/脚本里 excludes 列表里的模块名。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    block = re.search(r"excludes\s*=\s*\[(.*?)\]", source, re.S)
    assert block, rel
    return set(re.findall(r"""['"]([\w.]+)['"]""", block.group(1)))


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_build_entries_collect_cv2(rel):
    """cv2 是函数内延迟导入，两个构建入口都要显式收集且不能留在 excludes。

    留在 excludes 时打包不报错，但 exe 里点视频功能才炸；只改一处则另一条
    构建路径仍然缺模块。
    """
    assert "cv2" in set(_hidden_imports(rel)), f"{rel} 的 hiddenimports 缺 cv2"
    assert "cv2" not in _excludes(rel), f"{rel} 的 excludes 仍在排除 cv2"


def test_build_script_checks_runtime_dependencies_before_building():
    """build.sh 必须在打包前自检依赖，缺 cv2 就直接失败。

    踩过的坑：Windows 上脚本优先用 PATH 里的 `python`，本机那是没装 cv2 的
    base conda。PyInstaller 遇到缺模块**不报错**，只是静默不收——打出的 exe
    能启动、图片推理正常，点视频才报缺 cv2。产物大小也会明显偏小。
    """
    source = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert "check_python_deps" in source
    # 四个运行时依赖都要查：cv2 漏收最隐蔽，其余缺了会当场炸但也一并挡掉
    for name in ("cv2", "PySide6", "numpy", "onnxruntime"):
        assert name in source, name
    # 自检必须在真正调用 PyInstaller 之前（第 350 行那处 echo "$PYINSTALLER_CMD"）
    check_at = source.index("check_python_deps cv2")
    build_at = source.index('echo "$PYINSTALLER_CMD')
    assert check_at < build_at, "依赖自检必须在打包之前"


@pytest.mark.parametrize("rel", ["build.sh", "labelforge/LabelForge.spec"])
def test_build_entries_bundle_license_file(rel):
    """LGPL 要求随产物分发许可证正文与声明，两个入口都要打进包。"""
    source = (ROOT / rel).read_text(encoding="utf-8")
    assert "'LICENSE'" in source, f"{rel} 没有把 LICENSE 加进 datas"
    assert "THIRD-PARTY.md" in source, f"{rel} 没有把 THIRD-PARTY.md 加进 datas"


def test_requirements_install_headless_opencv_only():
    """requirements 只声明 headless 发行包：四种 OpenCV 共用 cv2，不能混装。

    版本上限钉在 4.11 之前：本机开发用的是 opencv-python 4.10.0.84，只有钉到
    4.10 这个次版本，CI 装的底层 OpenCV 才与本机完全相同（放开到 <5 会装到
    4.14，本地验证过的用例在 CI 就没有同等含义了）。
    """
    source = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [line.strip() for line in source.splitlines()]
    declared = [line for line in lines if line and not line.startswith("#")]
    assert "opencv-python-headless>=4.10,<4.11" in declared
    for line in declared:
        assert not re.match(r"^opencv-python(?!-headless)", line), line
        assert not re.match(r"^opencv-contrib", line), line


def test_ci_pins_same_opencv_major_as_local():
    """CI 装的 OpenCV 必须与本机同版本，否则本地绿、CI 红时无从判断是谁的问题。

    版本只在 requirements.txt 里声明一处，工作流按它安装（`-r requirements.txt`），
    所以这里查「声明存在」+「CI 确实读它」，而不是要求工作流再抄一遍。
    """
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    test_yml = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    build_yml = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    assert "opencv-python-headless>=4.10,<4.11" in requirements
    assert "-r requirements.txt" in test_yml
    # 构建任务同样从 requirements.txt 取版本（Linux 在 Docker 里，路径带前缀）
    assert "requirements.txt" in build_yml


def test_ci_installs_video_dependency_and_caches_pip():
    """CI 必须真装 cv2（不许靠 skip 拿绿），并配置依赖缓存。"""
    test_yml = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    build_yml = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    # cv2 随 requirements.txt 一起装进来（唯一声明处），工作流不得另起清单
    assert "-r requirements.txt" in test_yml
    assert "cache: pip" in test_yml
    assert "requirements.txt" in build_yml
    # Linux 构建在 Docker 里跑，缓存目录必须挂进容器才有效
    assert "/root/.cache/pip" in build_yml


def test_license_is_lgpl3_and_docs_agree():
    """许可证正文与文档口径一致，避免仓库里同时出现两种说法。"""
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert license_text.lstrip().startswith("GNU LESSER GENERAL PUBLIC LICENSE")
    assert "Version 3, 29 June 2007" in license_text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    structure = (ROOT / "docs/project_structure.md").read_text(encoding="utf-8")
    assert "LGPL-3.0-only" in readme
    assert "MIT License" not in readme
    assert "LGPL-3.0-only" in structure


def test_third_party_notice_documents_qt_source():
    """LGPL-3.0 §4a/§4d(0)：要声明用到 Qt，并给出 Qt 源码出口与替换途径。"""
    notice = (ROOT / "THIRD-PARTY.md").read_text(encoding="utf-8")
    assert "Qt" in notice and "LGPL-3.0" in notice
    assert "code.qt.io" in notice or "download.qt.io" in notice
    assert "pip install" in notice


def test_no_stale_gpl_only_mentions():
    """全仓库不得再出现 GPL-3.0-only 的说法，防漏改。

    只扫仓库非测试文件：本文件自己被排除，否则断言里的字面量会命中自己。
    `LGPL-3.0-only` 含 `GPL-3.0-only` 子串，故用否定后顾排除前面是 L 的情形。
    """
    stale = re.compile(r"(?<!L)GPL-3\.0-only")
    targets = [ROOT / "README.md", ROOT / "docs/project_structure.md",
               ROOT / "build.sh", ROOT / "labelforge/LabelForge.spec"]
    targets += sorted((ROOT / "labelforge").rglob("*.py"))
    offenders = [str(p.relative_to(ROOT)) for p in targets
                 if stale.search(p.read_text(encoding="utf-8"))]
    assert not offenders, f"仍写着 GPL-3.0-only: {offenders}"
