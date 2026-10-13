"""项目文档与构建入口回归测试。"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_build_script_is_at_repository_root_and_uses_its_directory_as_root():
    source = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert 'PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"' in source
    assert not (ROOT / "pastelabel" / "build.sh").exists()


def test_documentation_uses_pastelabel_log_without_a_leading_dot():
    for path in (ROOT / "README.md", ROOT / "docs" / "user_guide.md", ROOT / "docs" / "project_structure.md"):
        source = path.read_text(encoding="utf-8")
        assert "pastelabel.log" in source
        assert ".pastelabel.log" not in source


def test_build_script_does_not_list_removed_hidden_imports():
    source = (ROOT / "build.sh").read_text(encoding="utf-8")

    assert "pastelabel.ui.widgets" not in source
    assert "pastelabel.ui.styles" not in source
    assert "pastelabel.core.models" not in source


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
         "recipe|配方|至少选择两步|划分比例之和", "--", "pastelabel"],
        cwd=root, text=True, capture_output=True)
    assert result.returncode in (0, 1), result.stderr
    hits = [line for line in result.stdout.splitlines() if line.strip()]
    assert hits == [], "残留配方引用:\n" + "\n".join(hits)
