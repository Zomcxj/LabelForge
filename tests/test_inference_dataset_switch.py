"""数据集切换后，推理结果必须能落盘（用户报：切了数据集就存不上）。

用户报的现象：切换数据集后跑推理，**日志显示检测到了框**，但结果没保存。

根因：`_inference_dataset_changed()` 把「当前 background_images」与
`_inference_dataset` 快照比较，而那份快照**只在加载模型时**写过一次
（`_on_model_loaded`）。所以「先加载模型 → 再切数据集」之后，快照永远不等于
当前数据集，`_on_result_ready` / `_apply_inference_results` 一进门就
`return`，把结果全丢了；而检测本身是成功的，日志（以及进度窗的计数）照样显示
有框——于是表现为「检测到了但没保存」。

这里用真 ImageEditor 复现这条路径。
"""
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

_GUARD = """
import os, sys
sys.stdout.flush()
os._exit(0)
"""


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    result = subprocess.run(
        [sys.executable, "-c", script + _GUARD],
        cwd=ROOT, env=env, text=True, capture_output=True,
        encoding="utf-8", errors="replace")
    assert result.returncode == 0, result.stderr
    return result.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_HOST = """
import os, json, tempfile
from PySide6.QtWidgets import QApplication, QListWidgetItem
from PySide6.QtGui import QPixmap, QImage
from labelforge.ui.main_window import ImageEditor
from labelforge.engine.inference_task import ImageResult, sidecar_path

app = QApplication([])

def make_dataset(name, n=1):
    d = tempfile.mkdtemp(prefix="pl-ds-" + name + "-")
    paths = []
    for i in range(n):
        p = os.path.join(d, f"{name}{i}.jpg")
        QImage(80, 60, QImage.Format_RGB888).save(p)
        paths.append(p)
    return d, paths

def host_with(paths):
    w = ImageEditor()
    w.resize(1200, 800)
    w.show()
    app.processEvents()
    w.background_images = list(paths)
    w.background_list.clear()
    for p in paths:
        w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
    w.current_background_index = 0
    w.current_background = QPixmap.fromImage(QImage(paths[0]))
    w.detection_boxes = []
    w.detection_boxes_dict = {}
    w.canvas_items = []
    return w

class _Detector:
    model_path = "m.onnx"
    conf_threshold = 0.25
    iou_threshold = 0.45
    class_names = ["person"]
    def resolve_format(self):
        return "v8"
    def is_dynamic_input(self):
        return False
    active_providers = ["CPUExecutionProvider"]

def detect_boxes():
    return [{"label": "person", "x": 10, "y": 10, "width": 30, "height": 30,
             "score": 0.9, "shape_type": "rectangle"}]
"""


def test_inference_still_saves_after_switching_datasets():
    """先加载模型、再切数据集 → 推理结果仍要写进**当前数据集**的 sidecar。

    这是用户报的原始场景。若快照不刷新，这里会看到「检测到 1 个框」但
    json 完全没生成。
    """
    out = _run_real_qt(_HOST + """
d1, paths1 = make_dataset("a")
d2, paths2 = make_dataset("b", n=2)

w = host_with(paths1)
# 1) 加载模型：快照记的是数据集 1
w._on_model_loaded(_Detector(), (640, 640), "model")
app.processEvents()
# 加载模型**不**定数据集快照（那会让「先加载模型再切数据集」之后的
# 结果被全部丢弃）。快照在开跑时按当次任务记。
print('加载模型后没记快照', w._inference_dataset in (None, ()))

# 2) 切到数据集 2（模拟用户换文件夹）
w.background_images = list(paths2)
w.background_list.clear()
for p in paths2:
    w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(paths2[0]))
app.processEvents()

print('判为已切换', w._inference_dataset_changed())

# 3) 对当前图跑推理（有检出）
res = ImageResult(index=0, path=paths2[0], boxes=detect_boxes())
print('检测到框数', len(res.boxes))
w._apply_inference_to_current(res, False)
app.processEvents()

# 4) 结果必须落盘
print('json 已生成', os.path.exists(sidecar_path(paths2[0])))
if os.path.exists(sidecar_path(paths2[0])):
    data = json.load(open(sidecar_path(paths2[0]), encoding="utf-8"))
    print('json 里的框数', len(data.get('shapes') or []))
""")
    lines = _lines(out)
    assert lines[0] == "加载模型后没记快照 True", lines
    # 单图路径没有数据集守卫，且这次没开跑（没定快照），直接保存即可
    assert lines[1] == "判为已切换 True", lines
    assert lines[2] == "检测到框数 1", lines
    assert lines[3] == "json 已生成 True", (
        f"切数据集后推理结果没落盘（{lines}）")
    assert lines[4] == "json 里的框数 1", lines


def test_batch_inference_saves_after_switching_datasets():
    """批量路径（逐张 result_ready）同样不能在切数据集后被丢掉。"""
    out = _run_real_qt(_HOST + """
d1, paths1 = make_dataset("a")
d2, paths2 = make_dataset("b", n=2)

w = host_with(paths1)
w._on_model_loaded(_Detector(), (640, 640), "model")
app.processEvents()

w.background_images = list(paths2)
w.background_list.clear()
for p in paths2:
    w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
w.current_background_index = 0
w.current_background = QPixmap.fromImage(QImage(paths2[0]))
app.processEvents()
# 开跑：快照在此时按当次任务定稿（真实路径由 _start_inference 做）
w._inference_dataset = tuple(w.background_images)
print('开跑时判为已切换', w._inference_dataset_changed())

# 第 1 张（非当前图）走批量路径
other = paths2[1]
w._on_result_ready(1, ImageResult(index=1, path=other, boxes=detect_boxes()))
app.processEvents()
print('非当前图 json 已生成', os.path.exists(sidecar_path(other)))

# 当前图也走一次
w._on_result_ready(0, ImageResult(index=0, path=paths2[0],
                                  boxes=detect_boxes()))
app.processEvents()
print('当前图 json 已生成', os.path.exists(sidecar_path(paths2[0])))
""")
    lines = _lines(out)
    # 开跑时快照 == 当前数据集 → 不是「跑到一半换了数据集」
    assert lines[0] == "开跑时判为已切换 False", lines
    assert lines[1] == "非当前图 json 已生成 True", (
        f"切数据集后批量结果被丢弃（{lines}）")
    assert lines[2] == "当前图 json 已生成 True", lines


def test_results_are_still_dropped_when_the_dataset_changes_mid_run():
    """真正该丢的情况要保住：跑到一半数据集被换掉，迟到结果不能写错地方。

    修「切完再跑存不上」不能把「跑一半换数据集」的保护一起拆掉——后者会把
    A 数据集的结果写进 B 数据集的图里。

    这条直接盯保护本身（`_inference_dataset_changed()` + 迟到结果不落盘），
    不用「先切数据集再跑」的场景去反推，避免又变成一条杀不死变异的空转断言
    （先前那版就是这样：即使把守卫整段删掉它也照样绿）。
    """
    out = _run_real_qt(_HOST + """
d1, paths1 = make_dataset("a")
d2, paths2 = make_dataset("b", n=2)

w = host_with(paths1)
w._on_model_loaded(_Detector(), (640, 640), "model")
app.processEvents()
# 开跑时定稿：这一轮针对数据集 1
w._inference_dataset = tuple(w.background_images)
# 迟到的结果：必须用**非当前图**的 key。key == current_background_index
# 会拐进「当前图」那条分支（它先改内存再走保存流程），测不到这道守卫。
d1b, paths1b = make_dataset("a2", n=2)
w._on_model_loaded(_Detector(), (640, 640), "model")
app.processEvents()
w.background_images = list(paths1b)
w.background_list.clear()
for p in paths1b:
    w.background_list.addItem(QListWidgetItem(os.path.basename(p)))
w.current_background_index = 0
w._inference_dataset = tuple(paths1b)
app.processEvents()

# 跑到一半换数据集
w.background_images = list(paths2)
print('中途换后判为已变', w._inference_dataset_changed())

late = paths1b[1]          # 非当前图（当前 index 还是 0）
print('迟到的是非当前图', w.current_background_index != 1)
w._on_result_ready(1, ImageResult(index=1, path=late, boxes=detect_boxes()))
app.processEvents()
print('迟到结果没落盘', not os.path.exists(sidecar_path(late)))
""")
    lines = _lines(out)
    assert lines[0] == "中途换后判为已变 True", lines
    assert lines[1] == "迟到的是非当前图 True", lines
    assert lines[2] == "迟到结果没落盘 True", (
        f"跑到一半换数据集时，迟到结果被错误写入（{lines}）")
