"""阶段 2 守卫：迁移到 PySide6 后不许回潮。

这里盯的是**功能性**残留（导入、模块访问、PyQt5 专有 API、信号名），
不盯注释里出现的「PyQt5」字样——有几处是有意对比两代 API 差异的说明，
禁掉它们只会逼着以后写注释时含糊其辞。

真正会静默要命的是打包配置那两条：spec 的 excludes 里若还留着 PySide6，
构建照样成功，打出的 exe 却缺 Qt、启动即崩，而且构建期没有任何提示。
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "labelforge"
TESTS = ROOT / "tests"

# 本文件自己会写出这些名字（断言、API 清单、说明文字），扫自己只会假红。
_GUARD_FILES = {"tests/test_pyside6_migration.py"}


def _py_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel in _GUARD_FILES:
            continue
        yield path, rel


# ---------- 1. 不许再导入 PyQt5 ----------

def test_no_pyqt5_imports_anywhere():
    """源码与测试都不许 import PyQt5 / from PyQt5 import。"""
    offenders = []
    for path, rel in _py_files(SRC):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if re.match(r"\s*(from|import)\s+PyQt5\b", line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "仍有 PyQt5 导入：\n" + "\n".join(offenders)


def test_no_pyqt5_module_attribute_access():
    """`PyQt5.QtCore` 这类模块访问也要清掉（含字符串里的）。"""
    offenders = []
    for root in (SRC, TESTS):
        for path, rel in _py_files(root):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                # 排除注释；`PyQt5.` 出现在说明性文字里是允许的
                code = line.split("#", 1)[0]
                if re.search(r"PyQt5\.[A-Za-z]", code):
                    offenders.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not offenders, "仍有 PyQt5 模块访问：\n" + "\n".join(offenders)


# ---------- 2. 信号名 ----------

def test_no_pyqt_signal_or_slot_names():
    """`pyqtSignal` / `pyqtSlot` 在 PySide6 里不存在，必须叫 Signal / Slot。"""
    offenders = []
    for root in (SRC, TESTS):
        for path, rel in _py_files(root):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                for name in ("pyqtSignal", "pyqtSlot", "pyqtProperty"):
                    if name in code:
                        offenders.append(f"{rel}:{i}: {name}")
    assert not offenders, "仍有 pyqt* 名字：\n" + "\n".join(offenders)


# ---------- 3. PyQt5 专有 API ----------

# 带点号前缀：真调用必然是 `xxx.setsize(...)`，说明文字里裸写名字不该误报。
PYQT5_ONLY_APIS = (".setsize(", ".asstring(", ".byteCount(")


def test_no_pyqt5_only_pixel_buffer_apis():
    """`setsize` / `asstring` / `byteCount` 是 PyQt5 的 sip 专有 API。

    PySide6 下 `bits()` 返回带长度的 memoryview、字节数叫 `sizeInBytes()`。
    """
    offenders = []
    for root in (SRC, TESTS):
        for path, rel in _py_files(root):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                for api in PYQT5_ONLY_APIS:
                    if api in code:
                        offenders.append(f"{rel}:{i}: {api}")
    assert not offenders, "仍有 PyQt5 专有缓冲 API：\n" + "\n".join(offenders)


def test_no_qfontmetrics_width_call():
    """Qt6 移除了 QFontMetrics.width()，要用 horizontalAdvance()。"""
    offenders = []
    for root in (SRC, TESTS):
        for path, rel in _py_files(root):
            text = path.read_text(encoding="utf-8")
            for i, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                # 只盯 fm./metrics. 这类字面量调用；假对象里留个兼容 width 不算
                if re.search(r"\b(fm|metrics|fontMetrics\(\))\.width\(", code):
                    offenders.append(f"{rel}:{i}: {line.strip()[:80]}")
    assert not offenders, "仍调用 QFontMetrics.width()：\n" + "\n".join(offenders)


# ---------- 4. 打包配置（静默要命的那两条） ----------

def _excludes_block(text):
    """取出 `excludes=[...]` 的原文片段。"""
    start = text.index("excludes=[")
    depth = 0
    for i in range(start + len("excludes="), len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise AssertionError("找不到 excludes 列表的结束括号")


def test_spec_does_not_exclude_pyside6():
    """spec 的 excludes 绝不能含 PySide6。

    迁移前排除它是为瘦身；现在 Qt 就是 PySide6，排除掉构建照样成功，
    exe 却缺 Qt、启动即崩，且构建期毫无提示。
    """
    spec = SRC / "LabelForge.spec"
    block = _excludes_block(spec.read_text(encoding="utf-8"))
    assert "PySide6" not in block, f"spec 仍在排除 PySide6：\n{block}"


def test_build_sh_does_not_exclude_pyside6():
    """build.sh 会自己生成一份 spec，是 excludes 的第二处，同样不能排 PySide6。"""
    text = (ROOT / "build.sh").read_text(encoding="utf-8")
    # build.sh 里可能有模板占位，逐段找 excludes
    blocks = []
    cursor = 0
    while True:
        try:
            idx = text.index("excludes=[", cursor)
        except ValueError:
            break
        end = text.index("]", idx)
        blocks.append(text[idx:end + 1])
        cursor = end
    assert blocks, "build.sh 里没找到 excludes，生成 spec 的逻辑可能已改"
    for block in blocks:
        assert "PySide6" not in block, f"build.sh 仍在排除 PySide6：\n{block}"


def test_spec_and_build_exclude_other_qt_bindings():
    """PyInstaller 不允许同时收集两套 Qt bindings，所以反向要排掉 PyQt5。"""
    for rel in ("labelforge/LabelForge.spec", "build.sh"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "PyQt5" in text, f"{rel} 没有排除 PyQt5，环境里残留会打断构建"


# ---------- 5. 依赖声明 ----------

def test_requirements_declares_pyside6_not_pyqt5():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    code = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert any(l.lower().startswith("pyside6") for l in code), code
    assert not any(l.lower().startswith("pyqt") for l in code), code


def test_ci_installs_pyside6():
    """CI 里不能还装 PyQt5。

    大小写都要查：Linux 构建任务用的是 `conda install pyqt`（全小写），
    只匹配 `PyQt5` 会漏掉它——实测就漏了一次，PySide6 压根没被装上。
    """
    for rel in (".github/workflows/test.yml", ".github/workflows/build.yml"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        lowered = text.lower()
        assert "pyqt5" not in lowered, f"{rel} 仍装着 PyQt5"
        assert "pyqt6" not in lowered, f"{rel} 仍装着 PyQt6"
        assert "pyside2" not in lowered, f"{rel} 仍装着 PySide2"
        # 单独的 `pyqt`（conda 包名）也要挡掉
        assert not re.search(r"\bpyqt\b", lowered), (
            f"{rel} 里出现 `pyqt`（conda 的 PyQt5 包名）")


def test_ci_mentions_pyside6():
    """反向：至少要有一处真的把 PySide6 装上。"""
    text = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    assert "PySide6" in text, "构建工作流里没有安装 PySide6"


def test_linux_build_image_is_pinned_to_ubuntu_20_04():
    """Linux 构建镜像固定在 ubuntu:20.04 —— 镜像的 glibc 就是产物的最低要求。

    README 承诺 Linux 支持到 Ubuntu 20.04，所以三个边界都要对：

    - 18.04（glibc 2.27）装不上 PySide6：wheel 是 manylinux_2_28，`pip` 直接
      拒绝（`libxcb-cursor0` 倒是 bionic 就没有，但那是次要原因）；
    - 22.04（glibc 2.35）编出来的产物会引用 GLIBC_2.3x 符号，在 20.04 上起不来；
    - 20.04（glibc 2.31）既装得上 wheel，产物的下限也正好落在 20.04。

    这个 job 要装的 `libxcb-cursor0` / `libegl1-mesa` / `libxkbcommon-x11-0` 在
    focal 都有（已核对 focal 的 `Packages.gz` 索引，而不是凭印象）。
    """
    text = (ROOT / ".github/workflows/build.yml").read_text(encoding="utf-8")
    assert "ubuntu:18.04" not in text, (
        "构建镜像退回 18.04：glibc 2.27 装不上 PySide6（manylinux_2_28）")
    assert "ubuntu:22.04" not in text, (
        "构建镜像抬到 22.04：产物的 glibc 下限会跟着抬到 2.35，20.04 用户跑不起来")
    assert "ubuntu:20.04" in text, "Linux 构建镜像未固定在 20.04"


def test_pyside6_upper_bound_stays_below_the_glibc_bump():
    """PySide6 上限必须在 6.10 以下。

    6.10 起 Linux wheel 从 `manylinux_2_28` 变成 `manylinux_2_34`：在 20.04 上
    pip 会直接拒绝安装（glibc 2.31 < 2.34）；就算在更新的机器上装成功，产物也会
    要求 glibc 2.34，同样跑不了 20.04。所以上限是兼容性约束，不是洁癖。
    """
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [l for l in text.splitlines()
             if l.startswith("PySide6") and not l.lstrip().startswith("#")]
    assert lines, "requirements.txt 里找不到 PySide6"
    spec = lines[0].replace(" ", "")
    assert "<6.10" in spec, (
        f"PySide6 没有卡在 6.10 以下，6.10+ 的 wheel 要求 glibc 2.34：{lines[0]}")


# ---------- 6. 兼容层已删除 ----------

def test_sip_shim_is_gone():
    """PyQt5 专用的 sip shim 必须删掉，并从打包配置里移除。"""
    assert not (SRC / "sip.py").exists(), "labelforge/sip.py 还在"
    for rel in ("labelforge/LabelForge.spec", "build.sh"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "labelforge.sip" not in text
        assert "labelforge/sip" not in text


# ---------- 7. `exec_()` 必须保留（反直觉，但实测有硬约束） ----------

def test_dialogs_keep_the_exec_underscore_suffix():
    """源码继续用 `exec_()`，不要「现代化」成 `exec()`。

    PySide6 下两者都在，但**可补丁性不同**：

      QDialog.exec      -> method_descriptor           可 monkeypatch
      QMessageBox.exec  -> method_descriptor           可 monkeypatch
      QMenu.exec        -> builtin_function_or_method  **不可 monkeypatch**

    `QMenu.exec` 在类上赋值会「成功」，实例却仍解析到原生方法（实测
    `QMenu.exec is fake` 为 True、`menu.exec is fake` 为 False）。本项目的
    右键菜单测试全靠 `QMenu.exec_ = fake` 拦截；源码一旦改调 `exec()`，
    补丁失效 → 真的弹出菜单 → 测试**永久阻塞**（实测卡死两轮，只能强杀）。

    `exec_()` 在 PySide6 下是可用别名**且**可拦截，因此保持现状。
    """
    offenders = []
    for path, rel in _py_files(SRC):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\.exec\(", code):
                offenders.append(f"{rel}:{i}: {line.strip()[:74]}")
    assert not offenders, (
        "源码里出现了 `.exec(` —— QMenu.exec 在 PySide6 下无法被测试补丁拦截，"
        "改过去会让右键菜单测试挂死：\n" + "\n".join(offenders))


def test_qmenu_exec_is_indeed_unpatchable():
    """把上面那条约束的依据钉死：QMenu.exec 补不上，exec_ 能补上。

    版本升级后若 PySide6 改成可补丁，这条会失败，那时才该考虑迁移到 `exec()`。
    """
    import subprocess
    import sys

    script = """
from PySide6.QtWidgets import QApplication, QMenu
app = QApplication([])

orig = QMenu.exec
QMenu.exec = lambda self, *a, **k: None
exec_patchable = QMenu().exec is not orig and QMenu().exec.__name__ == '<lambda>'
QMenu.exec = orig

orig2 = QMenu.exec_
QMenu.exec_ = lambda self, *a, **k: None
underscore_patchable = QMenu().exec_.__name__ == '<lambda>'
QMenu.exec_ = orig2

print('EXEC_PATCHABLE', exec_patchable)
print('UNDERSCORE_PATCHABLE', underscore_patchable)
"""
    import os
    env = os.environ | {"QT_QPA_PLATFORM": "offscreen", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                       text=True, capture_output=True, timeout=120,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    assert "EXEC_PATCHABLE False" in r.stdout, r.stdout
    assert "UNDERSCORE_PATCHABLE True" in r.stdout, r.stdout