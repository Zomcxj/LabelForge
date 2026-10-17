"""合成页「叠加标签框」控件：类别扫描、开关联动、settings() 输出。

分两层（同项目约定）：进程内做源码/AST 断言，真实控件走子进程 + 真实 Qt。
"""
import inspect
import re
import subprocess
import sys
from pathlib import Path

import pytest

from labelforge.ui import i18n

ROOT = Path(__file__).resolve().parents[1]

OVERLAY_KEYS = (
    "叠加标签框", "类别", "类别名", "置信度", "全选", "全不选",
    "没有标注文件，不能叠加标签框", "标注读不出来",
)


# ---------- i18n ----------

@pytest.mark.parametrize("key", OVERLAY_KEYS)
def test_overlay_texts_are_translated(key):
    """新增文案必须在两个语言表里都有。"""
    for lang in ("zh", "en"):
        assert key in i18n._strings[lang], (lang, key)


def test_overlay_tr_keys_are_defined():
    """视频弹窗与 mixin 用到的 tr() 键都必须在语言表里（防止只写 tr 不加词条）。"""
    missing = []
    for rel in ("labelforge/ui/video_dialog.py",
                "labelforge/ui/mixins/video.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for key in sorted(set(re.findall(r"tr\(['\"]([^'\"]+)['\"]\)", src))):
            for lang in ("zh", "en"):
                if key not in i18n._strings[lang]:
                    missing.append((lang, key, rel))
    assert not missing, missing


# ---------- 扫描类别（纯逻辑） ----------

def test_scan_labels_counts_and_ignores_missing(tmp_path):
    """扫描目录：统计各类别出现次数，缺 sidecar 的图不报错。"""
    import json

    from labelforge.engine import video_io

    def rect(label):
        return {"label": label, "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                "group_id": None, "shape_type": "rectangle", "flags": {}}

    d = tmp_path / "imgs"
    d.mkdir()
    for i, shapes in enumerate([
        [rect("person"), rect("car")],
        [rect("person")],
        [],
    ]):
        (d / f"a{i}.png").write_bytes(b"x")
        if i < 2:
            (d / f"a{i}.json").write_text(
                json.dumps({"shapes": shapes}), encoding="utf-8")

    labels = video_io.scan_sidecar_labels(
        [str(d / f"a{i}.png") for i in range(3)])
    assert labels == {"person": 2, "car": 1}, labels


def test_scan_labels_orders_by_count_then_name(tmp_path):
    """类别顺序要稳定：先按出现次数多的，次数相同再按名称。"""
    import json

    from labelforge.engine import video_io

    def rect(label):
        return {"label": label, "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                "group_id": None, "shape_type": "rectangle", "flags": {}}

    d = tmp_path / "imgs"
    d.mkdir()
    # b 出现 3 次、a 出现 2 次；c 与 a 同为 2 次用来验同分按名称排
    plan = [["a", "b"], ["b"], ["b", "a"], ["c"]]
    for i, labels in enumerate(plan):
        (d / f"a{i}.png").write_bytes(b"x")
        (d / f"a{i}.json").write_text(
            json.dumps({"shapes": [rect(n) for n in labels]}), encoding="utf-8")

    found = video_io.scan_sidecar_labels(
        [str(d / f"a{i}.png") for i in range(len(plan))])
    assert found == {"b": 3, "a": 2, "c": 1}, found
    assert list(found) == ["b", "a", "c"], list(found)


def test_scan_labels_empty_when_no_sidecars(tmp_path):
    """一张标注都没有 → 空字典（界面据此禁用叠加）。"""
    from labelforge.engine import video_io

    d = tmp_path / "imgs"
    d.mkdir()
    (d / "a.png").write_bytes(b"x")
    assert video_io.scan_sidecar_labels([str(d / "a.png")]) == {}


def test_scan_labels_raises_on_corrupt(tmp_path):
    """损坏的标注要报错，不能假装那张图没标注。"""
    from labelforge.engine import video_io

    d = tmp_path / "imgs"
    d.mkdir()
    (d / "a.png").write_bytes(b"x")
    (d / "a.json").write_text("{ 坏 JSON", encoding="utf-8")

    with pytest.raises(video_io.VideoError, match="标注"):
        video_io.scan_sidecar_labels([str(d / "a.png")])


def test_scan_labels_ignores_shapes_without_points(tmp_path):
    """shapes 里解析不出有效框（如缺 points）时不算有标注，也不该报错。"""
    import json

    from labelforge.engine import video_io

    d = tmp_path / "imgs"
    d.mkdir()
    (d / "a.png").write_bytes(b"x")
    (d / "a.json").write_text(json.dumps({"shapes": [{"label": "person"}]}),
                              encoding="utf-8")

    assert video_io.scan_sidecar_labels([str(d / "a.png")]) == {}


# ---------- 弹窗控件 ----------

def test_overlay_group_options_are_behavioral():
    """`_OverlayGroup` 的勾选语义：默认全勾、两个文字开关独立、全不选=不画。

    进程内的假 QListWidgetItem 没有 setFlags/checkState（conftest 只补了
    最薄的替身），所以这条走真实 Qt 子进程。不看源码字符串：那是接线，
    接线错了字符串断言照样通过。
    """
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person", "car"], "b.png": ["person"]}))
g = d._compose_overlay
default = g.options()
g.show_score_check.setChecked(False)
independent = g.options()
g.set_checked(False)
none_checked = (g.options()["labels"], g.overlay_check.isChecked())
g.overlay_check.setChecked(False)
master_off = g.options()
print("RESULT", repr((default, independent, none_checked, master_off)))
''')
    default, independent, none_checked, master_off = result
    assert default["labels"] == ["person", "car"], "默认全勾上"
    assert default["show_labels"] is True and default["show_scores"] is True
    assert independent["show_scores"] is False
    assert independent["show_labels"] is True, "两个开关必须独立"
    assert none_checked[0] == [], "全不选 = 一个都不画（不是全画）"
    assert none_checked[1] is True, "全不选不改主开关"
    assert master_off is None, "主开关关着就不带 render"


def test_overlay_group_fill_keeps_real_label_names():
    """列表项显示「类别 (次数)」，但交回引擎的是原类别名，不是显示文本。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person"], "b.png": ["person", "person"]}))
item = d.compose_label_list.item(0)
print("RESULT", repr((item.text(), d._compose_overlay.options()["labels"])))
''')
    shown, labels = result
    assert shown == "person (3)", shown
    assert labels == ["person"], "不能把显示文本当类别名交回引擎"


def test_settings_omits_render_when_overlay_off():
    """不叠加时不能往参数里塞 render（引擎据此走纯合成路径）。

    真 Qt 子进程里按主开关，再看 settings 里有没有 render。
    """
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person"], "b.png": ["person"]}))
on = "render" in compose_settings(d)
d.compose_overlay_check.setChecked(False)
off = "render" in compose_settings(d)
d.compose_overlay_check.setChecked(True)
again = "render" in compose_settings(d)
print("RESULT", repr((on, off, again)))
''')
    on, off, again = result
    assert on is True, "默认有标注时要带 render"
    assert off is False, "主开关关掉就不能带 render"
    assert again is True, "重新打开要恢复"


# ---------- 子进程 + 真实 Qt ----------

def _run_qt(script):
    proc = subprocess.run([sys.executable, "-c", script % str(ROOT)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    line = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert line, f"子进程没有输出结果：{out[-2000:]}"
    return eval(line[0][len("RESULT"):].strip())


_PRELUDE = '''
import json, os, sys, tempfile
sys.path.insert(0, r"%s")
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
app = QApplication([])
import numpy as np
import cv2
from labelforge.ui.video_dialog import VideoDialog
from labelforge.engine import video_io as vio

TMP = tempfile.mkdtemp()

# 造一个真源视频，才能写出能被 read_manifest 接受的清单
SRC = os.path.join(TMP, "src.mp4")
_vw = cv2.VideoWriter(SRC, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
for _ in range(40):
    _vw.write(np.zeros((32, 32, 3), np.uint8))
_vw.release()
INFO = vio.probe(SRC)

def make_dir(label_map, with_manifest=True):
    """label_map: {图片名: [类别, ...] 或 None 表示不写 sidecar}"""
    d = tempfile.mkdtemp(dir=TMP)
    frames = []
    for i, (name, labels) in enumerate(label_map.items()):
        vio.write_image(os.path.join(d, name), np.zeros((32, 32, 3), np.uint8), ext="png")
        if labels is not None:
            shapes = [{"label": lb, "points": [[1, 1], [9, 1], [9, 9], [1, 9]],
                       "group_id": None, "shape_type": "rectangle", "flags": {}}
                      for lb in labels]
            with open(os.path.splitext(os.path.join(d, name))[0] + ".json", "w",
                      encoding="utf-8") as f:
                json.dump({"version": "5.5.0", "shapes": shapes}, f)
        sf = i * 5
        frames.append(vio.ManifestFrame(file=name, source_frame=sf,
                                        timestamp=sf / INFO.fps,
                                        sample_time=sf / INFO.fps))
    if with_manifest:
        man = vio.Manifest(source=INFO.path, source_name="src.mp4",
                           width=INFO.width, height=INFO.height,
                           source_fps=INFO.fps, frame_count=INFO.frame_count,
                           duration=INFO.duration, target_fps=2.0,
                           start_time=0.0, end_time=2.0, mode="sample",
                           complete=True, frames=frames)
        vio.write_manifest(os.path.join(d, "video_manifest.json"), man)
    return d

def compose_settings(dialog):
    """settings() 默认按「抽帧」动作返回；先按合成按钮切到合成动作。"""
    dialog._on_start_compose()
    return dialog.settings()
'''


def test_real_qt_scans_labels_and_defaults_to_all_checked():
    """真实 Qt：选目录后列出各类别与数量，默认全勾，主开关默认可叠加。

    类别名/置信度默认都开——与画布的「显示Label」「显示置信度」默认一致，
    免得编辑器里看到的和导出的视频不一样。
    """
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
dir1 = make_dir({"a.png": ["person", "car"], "b.png": ["person"]})
d.set_compose_dir(dir1)
names = [d.compose_label_list.item(i).text()
         for i in range(d.compose_label_list.count())]
states = [d.compose_label_list.item(i).checkState().value
          for i in range(d.compose_label_list.count())]
overlay_on = d.compose_overlay_check.isChecked()
settings = compose_settings(d)
print("RESULT", repr((names, states, overlay_on, settings.get("render"),
                      "manifest_path" in settings)))
''')
    names, states, overlay_on, render, has_manifest = result
    assert names == ["person (2)", "car (1)"], names
    assert states == [2, 2], states                      # 2 = Qt.Checked
    assert overlay_on is True, "有标注时主开关应默认可用且勾上"
    assert render is not None
    assert render["labels"] == ["person", "car"], render
    assert render["show_labels"] is True, "默认应与画布一致：显示类别名"
    assert render["show_scores"] is True, "默认应与画布一致：显示置信度"
    assert has_manifest is True, "有清单时要按清单合成"


def test_real_qt_overlay_off_omits_render_and_disables_sub_options():
    """真实 Qt：关掉主开关 → 子选项禁用，settings() 不带 render。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person"]}))
d.compose_overlay_check.setChecked(False)
enabled = (d.compose_label_list.isEnabled(), d.compose_show_label_check.isEnabled(),
           d.compose_show_score_check.isEnabled(), d.compose_all_btn.isEnabled())
settings = compose_settings(d)
print("RESULT", repr((enabled, "render" in settings)))
''')
    enabled, has_render = result
    assert enabled == (False, False, False, False), enabled
    assert has_render is False, "不叠加时不该带 render"


def test_real_qt_unchecking_a_label_removes_it_from_settings():
    """真实 Qt：取消勾选某个类别后，它不该出现在传给引擎的 labels 里。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person", "car"], "b.png": ["person"]}))
# 列表按出现次数排序：person(2) 在前，car(1) 在后；取消 car（第 2 项）
item = d.compose_label_list.item(1)
item.setCheckState(Qt.Unchecked)
render = compose_settings(d)["render"]
print("RESULT", repr((render["labels"], d.compose_label_list.item(0).text())))
''')
    labels, first = result
    assert labels == ["person"], labels
    assert first.startswith("person")


def test_real_qt_all_and_none_buttons():
    """真实 Qt：全选/全不选按钮要真的改变勾选状态。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person", "car"], "b.png": ["person"]}))
d.compose_none_btn.click()
none_labels = compose_settings(d)["render"]["labels"]
none_enabled = d.compose_btn.isEnabled()
d.compose_all_btn.click()
all_labels = compose_settings(d)["render"]["labels"]
print("RESULT", repr((none_labels, all_labels, none_enabled)))
''')
    none_labels, all_labels, still_enabled = result
    assert none_labels == [], "全不选后不该还有类别"
    assert all_labels == ["person", "car"]
    assert still_enabled is True, "一个类别都不勾也该能合成（就是干净画面）"


def test_real_qt_no_sidecars_disables_overlay():
    """真实 Qt：目录里没有标注 → 主开关禁用并给出提示，但仍能合成。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": None, "b.png": None}))
print("RESULT", repr((d.compose_overlay_check.isEnabled(),
                      d.compose_overlay_check.isChecked(),
                      d.compose_label_list.count(),
                      d.compose_btn.isEnabled(),
                      "render" in compose_settings(d))))
''')
    overlay_enabled, overlay_checked, count, can_compose, has_render = result
    assert overlay_enabled is False, "没标注时不该能开叠加"
    assert overlay_checked is False
    assert count == 0
    assert can_compose is True, "没标注也能纯合成"
    assert has_render is False


def test_real_qt_corrupt_sidecar_warns_and_disables_overlay():
    """真实 Qt：标注损坏 → 提示区说明，叠加禁用，纯合成仍可用。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
dir1 = make_dir({"a.png": ["person"]})
with open(os.path.join(dir1, "a.json"), "w", encoding="utf-8") as f:
    f.write("{ 坏 JSON")
d.set_compose_dir(dir1)
print("RESULT", repr((d.compose_hint_label.text(),
                      d.compose_overlay_check.isEnabled(),
                      d.compose_btn.isEnabled())))
''')
    hint, overlay_enabled, can_compose = result
    assert "标注" in hint, hint
    assert overlay_enabled is False
    assert can_compose is True, "坏标注不该挡住纯合成"


def test_real_qt_show_label_and_score_switches_are_independent():
    """真实 Qt：类别名与置信度是两个独立开关，各自进 settings()。"""
    result = _run_qt(_PRELUDE + '''
d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person"]}))
d.compose_show_label_check.setChecked(True)
d.compose_show_score_check.setChecked(False)
a = compose_settings(d)["render"]
d.compose_show_label_check.setChecked(False)
d.compose_show_score_check.setChecked(True)
b = compose_settings(d)["render"]
print("RESULT", repr((a["show_labels"], a["show_scores"],
                      b["show_labels"], b["show_scores"])))
''')
    a_labels, a_scores, b_labels, b_scores = result
    assert (a_labels, a_scores) == (True, False)
    assert (b_labels, b_scores) == (False, True)


# ---------- worker / mixin 接线 ----------

def test_worker_passes_render_to_engine():
    """后台线程要把绘制规格传给引擎，否则界面勾的选项一点用没有。"""
    from labelforge.ui.inference_workers import VideoComposeWorker

    src = inspect.getsource(VideoComposeWorker)
    assert "render" in src
    assert "render=render" in src or "render=self._render" in src


def test_worker_accepts_render_argument():
    """构造签名要收 render（默认 None = 纯合成）。"""
    from labelforge.ui.inference_workers import VideoComposeWorker

    sig = inspect.signature(VideoComposeWorker.__init__)
    assert "render" in sig.parameters
    assert sig.parameters["render"].default is None


def test_mixin_builds_render_spec_with_colors_from_main_thread():
    """类别色必须在主线程取好再交给 worker（后台不能碰 editor 的取色）。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin)
    assert "_compose_render_spec" in src
    spec_src = inspect.getsource(VideoMixin._compose_render_spec)
    assert "get_label_color" in spec_src, "颜色要在主线程取"
    assert "RenderSpec" in spec_src
    assert "labels" in spec_src


def test_mixin_reports_unlabeled_images():
    """完成报告要提到无标注张数，否则用户不知道哪些帧是干净画面。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._on_compose_finished)
    assert "unlabeled_count" in src


def test_mixin_omits_render_when_options_absent():
    """没有叠加选项时不要凭空造一个 RenderSpec（保持纯合成路径）。"""
    from labelforge.ui.mixins.video import VideoMixin

    src = inspect.getsource(VideoMixin._compose_render_spec)
    assert "return None" in src


def test_compose_render_spec_takes_colors_from_the_editor():
    """_compose_render_spec 的行为：类别名 → 颜色，开关原样带过去。"""
    from labelforge.ui.mixins.video import VideoMixin

    class Stub(VideoMixin):
        def get_label_color(self, label):
            return {"person": "#FF0000"}.get(label, "#00FF80")

    spec = Stub()._compose_render_spec(
        {"labels": ["person", "car"], "show_labels": True, "show_scores": False})
    assert spec.labels == {"person", "car"}
    assert spec.show_labels is True and spec.show_scores is False
    assert spec.label_colors["person"] == "#FF0000"
    assert "car" in spec.label_colors, "取不到的类别也要给兜底色"


def test_compose_render_spec_none_when_no_options():
    """没有叠加选项 → None，引擎走纯合成（不读标注）。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = object.__new__(VideoMixin)
    assert stub._compose_render_spec(None) is None
    assert stub._compose_render_spec({}) is None


def test_compose_spec_leaves_font_size_to_the_engine():
    """字号不经过弹窗：`_compose_render_spec` 一律留 0，由引擎按设置算。

    第 12 轮曾加过滑块，用户随后要求删掉，改回「跟设置里的类别名字号走」。
    所以这里钉住「弹窗不再决定字号」——如果哪天有人又把 UI 的字号塞进来，
    设置窗口里那个「类别名字号」就会被架空。
    """
    from labelforge.ui.mixins.video import VideoMixin

    class Stub(VideoMixin):
        def get_label_color(self, label):
            return "#FF0000"

    base = {"labels": ["person"], "show_labels": True, "show_scores": True}

    spec = Stub()._compose_render_spec(dict(base))
    assert spec.font_pixels == 0, (
        "0 = 交给引擎按设置里的「类别名字号」算，弹窗不该自己定字号")

    # 即使调用方带了历史遗留的 font_pixels，也照样忽略
    legacy = Stub()._compose_render_spec(dict(base, font_pixels=36))
    assert legacy.font_pixels == 0, "UI 传的字号不该再生效"



def test_compose_render_spec_survives_editor_without_color_api():
    """editor 没有取色接口时也不能炸（假 editor / 精简场景）。"""
    from labelforge.ui.mixins.video import VideoMixin

    spec = object.__new__(VideoMixin)._compose_render_spec(
        {"labels": ["person"], "show_labels": False, "show_scores": False})
    assert spec.labels == {"person"}
    assert spec.label_colors == {}


def test_full_chain_dialog_to_video_with_real_qt():
    """真实 Qt 全链路：弹窗选项 → mixin 组 spec（含颜色） → 合成 → 画面有框。"""
    result = _run_qt(_PRELUDE + '''
import cv2
from labelforge.ui.mixins.video import VideoMixin
from labelforge.engine import video_io as vio

class Stub(VideoMixin):
    def get_label_color(self, label):
        return {"person": "#FF0000"}.get(label, "#00FF80")

d = VideoDialog()
d.set_compose_dir(make_dir({"a.png": ["person"], "b.png": ["person", "car"]}))
settings = compose_settings(d)
render = Stub()._compose_render_spec(settings["render"])
out = os.path.join(TMP, "chain.mp4")
res = vio.compose_video(manifest_path=settings["manifest_path"],
                        output_path=out, render=render,
                        output_fps=settings.get("output_fps"))
cap = cv2.VideoCapture(out)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
f0 = frames[0]
red = np.abs(f0[:, :, ::-1].astype(int) - np.array([255, 0, 0])).max(axis=2)
print("RESULT", repr((sorted(render.labels), render.label_colors,
                      res.image_count, res.unlabeled_count,
                      int((red <= 40).sum()))))
''')
    labels, colors, image_count, unlabeled, red_pixels = result
    assert labels == ["car", "person"], labels
    assert colors["person"] == "#FF0000", colors
    assert image_count == 2 and unlabeled == 0
    assert red_pixels > 0, "弹窗勾的类别没画到画面上"


class _StatusStub:
    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = text


class _ComposeStub:
    """只喂 _on_compose_finished 要用的东西，不建真实窗口。"""

    def __init__(self):
        self.status_label = _StatusStub()
        self._busy = True
        self.infos = []
        self.warns = []
        self.dialog_texts = []

    def _close_video_dialog(self, text=None):
        self.dialog_texts.append(text)

    def _video_info(self, message):
        self.infos.append(message)

    def _video_warn(self, message):
        self.warns.append(message)


def _compose_result(**overrides):
    from types import SimpleNamespace

    base = dict(cancelled=False, image_count=3, duration=1.0, output_fps=5.0,
                output_path="out.mp4", unlabeled_count=0)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_compose_finished_reports_unlabeled_image_count():
    """有帧没标注时要报出来：用户得知道哪些画面是干净的。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _ComposeStub()
    VideoMixin._on_compose_finished(
        stub, _compose_result(unlabeled_count=2, image_count=5))
    assert stub._busy is False, "跑完要清掉忙标志"
    # 合并进完成提示里，不另开一个模态弹窗打断
    assert stub.warns == [], stub.warns
    assert stub.infos, "没报完成结果"
    assert "2" in stub.infos[0], stub.infos
    assert "5" in stub.status_label.text, stub.status_label.text


def test_compose_finished_closes_the_dialog():
    """跑完要关掉弹窗（进度条留在弹窗里，不关就永远开着）。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _ComposeStub()
    VideoMixin._on_compose_finished(stub, _compose_result())
    assert stub.dialog_texts == ["合成完成"], stub.dialog_texts


def test_compose_finished_quiet_when_every_frame_has_annotations():
    """全都画了框就别多嘴。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _ComposeStub()
    VideoMixin._on_compose_finished(stub, _compose_result(unlabeled_count=0))
    assert stub.warns == [], stub.warns
    assert "没有标注" not in stub.infos[0], stub.infos


def test_compose_finished_stops_early_when_cancelled():
    """取消时不报张数、不当成正常完成，但弹窗要关掉。"""
    from labelforge.ui.mixins.video import VideoMixin

    stub = _ComposeStub()
    VideoMixin._on_compose_finished(stub, _compose_result(cancelled=True, unlabeled_count=9))
    assert stub.warns == [], stub.warns
    assert stub.infos == [], stub.infos
    assert stub.dialog_texts == ["已取消合成"], stub.dialog_texts
    assert stub._busy is False


# ---------- 第 12 轮：合成范围（从第几张到第几张） ----------

def test_compose_range_defaults_to_the_whole_series():
    """范围默认 0 到最后一张（用户要求「默认 0 到最后」）。"""
    from labelforge.engine.video_io import _slice_paths

    paths = [f"{i}.png" for i in range(8)]
    assert _slice_paths(paths) == paths
    assert _slice_paths(paths, None, None) == paths
    assert _slice_paths(paths, 0, 7) == paths


def test_slice_paths_is_inclusive_on_both_ends():
    """两端都是**包含**的：3..7 是 5 张，不是 4 张。"""
    from labelforge.engine.video_io import _slice_paths

    paths = [f"{i}.png" for i in range(10)]
    assert _slice_paths(paths, 3, 7) == [f"{i}.png" for i in range(3, 8)]
    assert len(_slice_paths(paths, 3, 7)) == 5
    assert _slice_paths(paths, 0, 0) == ["0.png"]
    assert _slice_paths(paths, 9, 9) == ["9.png"]


def test_slice_paths_clamps_out_of_range():
    """越界夹紧而不是报错：界面量程就是张数，手输超了不该让合成失败。"""
    from labelforge.engine.video_io import _slice_paths

    paths = [f"{i}.png" for i in range(5)]
    assert _slice_paths(paths, -3, 99) == paths
    assert _slice_paths(paths, 3, 99) == ["3.png", "4.png"]
    assert _slice_paths(paths, None, 1) == ["0.png", "1.png"]


def test_slice_paths_rejects_an_empty_range():
    """起始 > 结束 是真的没得合成，返回空让调用方报错（不静默出空片）。"""
    from labelforge.engine.video_io import _slice_paths

    paths = [f"{i}.png" for i in range(5)]
    assert _slice_paths(paths, 3, 1) == []
    assert _slice_paths(paths, 0, -1) == []
    assert _slice_paths([], 0, 3) == []


def test_compose_video_writes_only_the_selected_range(tmp_path):
    """真跑合成：选 3..7 就只出 5 帧，且是那 5 张的画面。"""
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    from labelforge.engine import video_io

    work = tmp_path / "range"
    work.mkdir()
    for i in range(10):
        arr = np.full((120, 160, 3), i * 22, dtype=np.uint8)
        cv2.imwrite(str(work / f"{i:03d}.png"), arr)

    out = str(work / "mid.mp4")
    video_io.compose_video(images_dir=str(work), output_path=out,
                           output_fps=5, start_index=3, end_index=7)

    cap = cv2.VideoCapture(out)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    assert len(frames) == 5, f"选 3..7 应出 5 帧，实际 {len(frames)}"
    # 编码有压缩，只核对「确实取到了不同的那几张」
    grays = sorted({int(f[60, 80, 0]) for f in frames})
    assert len(grays) == 5, grays
    assert grays[0] < 70 and grays[-1] > 140, grays


def test_compose_video_errors_on_an_empty_range(tmp_path):
    """空范围必须报错，不能生成一个 0 帧的假成品。"""
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from labelforge.engine import video_io

    work = tmp_path / "empty"
    work.mkdir()
    cv2.imwrite(str(work / "000.png"),
                np.zeros((60, 60, 3), dtype=np.uint8))
    out = str(work / "bad.mp4")
    with pytest.raises(video_io.VideoError):
        video_io.compose_video(images_dir=str(work), output_path=out,
                               output_fps=5, start_index=2, end_index=0)
    assert not Path(out).exists(), "报错时不该留下成品"
