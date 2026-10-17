"""项目文档与构建入口回归测试。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_script_is_at_repository_root_and_uses_its_directory_as_root():
    source = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert 'PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"' in source
    assert not (ROOT / "labelforge" / "build.sh").exists()


def test_documentation_uses_labelforge_log_without_a_leading_dot():
    for path in (ROOT / "README.md", ROOT / "docs" / "user_guide.md", ROOT / "docs" / "project_structure.md"):
        source = path.read_text(encoding="utf-8")
        assert "labelforge.log" in source
        assert ".labelforge.log" not in source
        # 旧名不得残留，否则用户按文档找不到日志
        assert "pastelabel.log" not in source


def test_build_script_does_not_list_removed_hidden_imports():
    source = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert "labelforge.ui.widgets" not in source
    assert "labelforge.ui.styles" not in source
    assert "labelforge.core.models" not in source


def test_documentation_describes_stats_color_replacement_and_separate_wheel_steps():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")

    for source in (readme, guide):
        assert "颜色替换" in source
        assert "HTML色值" in source
        assert "检测框缩放步长" in source
        assert "贴图缩放步长" in source
        assert "检测框滚轮缩放" in source
        assert "贴图滚轮缩放" in source


def test_documentation_covers_semantic_region_and_quality_lint():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    settings = (ROOT / "docs" / "settings_guide.md").read_text(encoding="utf-8")

    assert "语义区域" in readme
    assert "完整落在" in readme
    assert "语义区域" in guide
    assert "完整落在" in guide

    assert "标注质检" in readme
    assert "标注质检" in guide
    for term in ("越界框", "小目标不可检出", "重复框", "异类重叠", "近似名", "分组不一致"):
        assert term in guide, term
    assert "双击" in guide and "忽略此类问题" in guide

    assert "质检忽略" in settings
    assert "lint_ignored_rules" in settings


def test_documentation_covers_inference_scope():
    """滑窗/区域推理的用法与限制要写进文档，别只留在计划稿里。"""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    structure = (ROOT / "docs" / "project_structure.md").read_text(encoding="utf-8")

    for source in (readme, guide):
        assert "推理范围" in source
        assert "滑窗" in source
        assert "框中心" in source

    # 用户最容易被坑的两点：范围不持久化、低 IoU 重复可能残留
    assert "不持久化" in guide or "不记忆" in guide
    assert "低 IoU 重复" in guide
    assert "window_spec" in guide

    for module in ("scope_panel.py", "sliding_window_dialog.py",
                   "inference_scope.py"):
        assert module in structure, module


def test_documentation_covers_dataset_health():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "数据集健康" in readme
    assert "数据集健康" in guide
    for term in ("类别分布", "尺寸分布", "长宽比分布", "IoU 重叠分布"):
        assert term in guide, term


def test_documentation_covers_health_source_switch():
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert "切换标签页" in guide
    assert "数据源" in guide
    assert "配方" not in guide


def test_documentation_covers_stats_v2():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "user_guide.md").read_text(encoding="utf-8")
    for text in (readme, guide):
        assert "拖" in text
        assert "整个数据集" in text


def test_readme_no_recipe_bullet():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "配方" not in readme


def test_pipeline_recipe_feature_fully_removed():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "grep", "-n", "-i", "-E",
         "recipe|配方|至少选择两步|划分比例之和", "--", "labelforge"],
        cwd=root, text=True, capture_output=True)
    assert result.returncode in (0, 1), result.stderr
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    assert hits == [], "残留配方引用:\n" + "\n".join(hits)


# ---------- 改名守卫（阶段1：PasteLabel → LabelForge） ----------

# 这些位置允许出现旧名：它们是「旧名兼容」的实现或断言，不是遗漏。
_LEGACY_ALLOWED = {
    "labelforge/core/config_manager.py",      # LEGACY_CONFIG_PATH 常量
    "labelforge/core/json_trace.py",          # 旧探针环境变量的回退
    "tests/test_documentation_regressions.py",  # 本文件的断言
    "tests/test_config_manager.py",           # 迁移用例
}

# 旧名的三种大小写形态。只查前两种会漏掉 `PASTELABEL_SUBPROC_PYTHON` 这类
# 常量——它一个 `.py` 文件就能藏住整个改名遗漏（真的漏过一个）。
_LEGACY_FORMS = ("pastelabel", "PasteLabel", "PASTELABEL")

# 除 .py 外还要查的文本文件：spec / shell / CI / 依赖清单里也可能挂着旧名。
_LEGACY_SCAN_EXTRA = ("*.spec", "*.sh", "*.yml", "*.yaml", "*.txt", "*.cfg",
                      "*.toml", "*.json")


def _legacy_scan_files():
    for path in sorted(ROOT.rglob("*.py")):
        yield path
    for pattern in _LEGACY_SCAN_EXTRA:
        for path in sorted(ROOT.rglob(pattern)):
            yield path


def test_no_legacy_package_name_outside_allowlist():
    """包名与显示名不得在允许清单之外残留，否则改名不彻底。"""
    offenders = []
    seen = set()
    for path in _legacy_scan_files():
        rel = path.relative_to(ROOT).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        if (rel.startswith((".superpowers/", "build/", "dist/"))
                or "__pycache__" in rel):
            continue
        if rel in _LEGACY_ALLOWED:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for old in _LEGACY_FORMS:
            if old in text:
                offenders.append(f"{rel}: {old}")
    assert not offenders, offenders


def test_legacy_guard_covers_the_all_caps_form():
    """守卫必须认全大写旧名，否则 `PASTELABEL_*` 常量会整片溜过去。

    这条是被真实遗漏逼出来的：`tests/conftest.py` 里长期留着
    `PASTELABEL_SUBPROC_PYTHON`，而旧守卫只查小写与首字母大写。

    直接按行为验：把三种形态分别塞进一段文本，都要被判为残留。
    """
    assert "PASTELABEL" in _LEGACY_FORMS, "全大写形态不在检查列表里"

    def _flagged(text):
        return [old for old in _LEGACY_FORMS if old in text]

    assert _flagged("os.environ['PASTELABEL_SUBPROC_PYTHON']") == ["PASTELABEL"]
    assert _flagged("from pastelabel.core import x") == ["pastelabel"]
    assert _flagged("PasteLabel.exe") == ["PasteLabel"]


def test_guard_scans_more_than_python_files():
    """spec / shell / CI / 依赖清单里挂着旧名同样算改名不彻底。"""
    exts = {p.suffix for p in _legacy_scan_files()}
    assert ".spec" in exts or any(
        p.suffix == ".spec" for p in (ROOT / "labelforge").iterdir()
        if p.is_file()), f"未扫到 spec 文件，实际扫到 {exts}"
    assert ".sh" in exts or (ROOT / "build.sh").exists()


def test_spec_file_is_renamed_and_points_at_the_new_package():
    spec = ROOT / "labelforge" / "LabelForge.spec"
    assert spec.exists()
    assert not (ROOT / "labelforge" / "PasteLabel.spec").exists()
    text = spec.read_text(encoding="utf-8")
    assert "'labelforge'" in text
    assert "name='LabelForge'" in text


def test_package_directory_is_renamed():
    assert (ROOT / "labelforge" / "__init__.py").exists()
    assert not (ROOT / "pastelabel").exists()


def test_old_package_name_does_not_import():
    """旧包名必须真的不可导入，否则改名只是表面功夫。"""
    import importlib
    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pastelabel")
