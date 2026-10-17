"""第 10 轮：贴图区域与推理区域彻底分家。

用户的核心要求（原话）：

    推理是推理的操作，贴图是贴图的操作。
    推理的区域贴图不能显示和操作，贴图的区域推理不能显示和操作。

拆成了两份独立数据：
- `region_boxes`        推理范围（标注模式）
- `paste_region_boxes`  贴图落点（贴图模式，从空开始）

外加两条相关修复：
- 贴图模式**硬禁用**推理（原先只"收起"参数行，一按图标就回来了）
- 拖动/滚轮改区域后刷新面板坐标（`_notify_region_changed()` 原先几乎没被调用）
"""
from tests.test_inference_scope_ui import _run_real_qt, _lines


def test_paste_mode_hard_blocks_inference_even_if_the_row_is_opened():
    """① 贴图模式下**不能**靠参数行图标把推理弄回来（要"禁用"不是"收起"）。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap
from labelforge.ui.main_window import ImageEditor


class Det:
    conf_threshold = 0.25
    iou_threshold = 0.45
    class_names = ['a']
    active_providers = ['CPU']

    def resolve_format(self):
        return 'yolov8'

    def is_dynamic_input(self):
        return False


app = QApplication([])
w = ImageEditor()
w.resize(1200, 900)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(400, 300, QImage.Format_RGB888))
w.background_images = ['a.png']
w._detector = Det()

row = w._ensure_model_row()
row.show()
app.processEvents()
w._update_inference_buttons()
print('标注模式 按钮可用:', w.model_single_btn.isEnabled())

w.edit_mode = 'paste'
w._sync_model_row_to_edit_mode()
app.processEvents()
print('贴图模式 行可见:', row.isVisible())
print('贴图模式 按钮可用:', w.model_single_btn.isEnabled())

# 关键：手动按参数行图标，想把推理弄回来
w._toggle_model_row()
app.processEvents()
print('按图标后 行可见:', row.isVisible())
print('按图标后 按钮可用:', w.model_single_btn.isEnabled())

# 再试一次直接强开行 + 刷新按钮状态，按钮也必须仍是灰的
row.show()
w._update_inference_buttons()
print('强开行后 按钮可用:', w.model_single_btn.isEnabled())

w.edit_mode = 'annotate'
w._sync_model_row_to_edit_mode()
app.processEvents()
print('切回标注 按钮可用:', w.model_single_btn.isEnabled())
""")
    lines = _lines(out)
    assert lines[0].endswith("True"), lines[0]
    assert lines[1].endswith("False"), f"贴图模式参数行没收起：{lines[1]}"
    assert lines[2].endswith("False"), f"贴图模式推理按钮没禁用：{lines[2]}"
    assert lines[3].endswith("False"), (
        f"贴图模式下按图标又把参数行打开了（应当是『禁用』而非『收起』）：{lines[3]}")
    assert lines[4].endswith("False"), (
        f"贴图模式下推理按钮可用了：{lines[4]}")
    assert lines[5].endswith("False"), (
        f"强开参数行后按钮竟变可用，模式条件没生效：{lines[5]}")
    assert lines[6].endswith("True"), f"切回标注模式没恢复：{lines[6]}"


def test_inference_and_paste_regions_are_separate_lists():
    """② 两份区域互不可见：标注模式只认推理那份，贴图模式只认贴图那份。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1200, 900)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(800, 600, QImage.Format_RGB888))
w.background_images = ['a.png']
w.current_background_index = 0
w.region_boxes = [{'x': 100, 'y': 100, 'width': 200, 'height': 150,
                   'shape_type': 'rectangle', 'locked': False}]
w.paste_region_boxes = []
app.processEvents()
cv = w.canvas
br = cv.get_background_rect()
s = cv.background_scale


def hit(ix, iy):
    return cv._region_at(QPoint(int(br.left() + ix * s),
                                int(br.top() + iy * s)))


w.edit_mode = 'annotate'
app.processEvents()
print('标注 推理区域命中:', hit(150, 150))

w.edit_mode = 'paste'
app.processEvents()
print('贴图 推理区域命中:', hit(150, 150))
print('贴图 贴图区域数:', len(cv._active_regions()))

# 贴图模式下画一个区域，只能进贴图那份
cv._active_regions().append({'x': 10, 'y': 10, 'width': 50, 'height': 40,
                             'shape_type': 'rectangle', 'locked': False})
print('画后 推理区域数:', len(w.region_boxes))
print('画后 贴图区域数:', len(w.paste_region_boxes))
print('画后 贴图区域命中:', hit(30, 30))

# 贴图模式固定，不该影响推理那份
cv._toggle_fix_regions()
print('贴图固定:', w.paste_region_fixed, '推理固定:', w.region_fixed)
""")
    lines = _lines(out)
    assert lines[0].endswith("0"), f"标注模式看不到推理区域：{lines[0]}"
    assert lines[1].endswith("None"), (
        f"贴图模式下推理区域仍然可见/可操作：{lines[1]}")
    assert lines[2].endswith("0"), (
        f"贴图区域应当从空开始（用户：贴图区域不留）：{lines[2]}")
    assert lines[3].endswith("1"), f"画区域污染了推理那份：{lines[3]}"
    assert lines[4].endswith("1"), f"画的区域没进贴图那份：{lines[4]}"
    assert lines[5].endswith("0"), f"贴图区域画完命中不到：{lines[5]}"
    assert lines[6] == "贴图固定: True 推理固定: False", (
        f"固定状态两份没分开：{lines[6]}")


def test_editing_a_region_refreshes_the_panel_coordinates():
    """③ 拖动/滚轮改区域后，面板拿到的坐标要跟着变。

    第 11 轮去掉贴图区域面板后，区域面板只剩**推理范围**面板一个；所以这里
    在标注模式下改推理区域（范围面板开着），断言它拿到的是**刷新后的坐标**。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtCore import QPoint, QPointF, Qt
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1200, 900)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(800, 600, QImage.Format_RGB888))
w.background_images = ['a.png']
w.current_background_index = 0
w.edit_mode = 'annotate'
w.region_boxes = [{'x': 200, 'y': 150, 'width': 300, 'height': 200,
                   'shape_type': 'rectangle', 'locked': False}]
w.paste_region_boxes = []
app.processEvents()
cv = w.canvas
br = cv.get_background_rect()
s = cv.background_scale

snapshots = []


class Panel:
    def set_regions(self, regions):
        snapshots.append(list(regions))


# 范围面板开着才允许编辑推理区域（用户：只在面板开着时允许）
w._scope_panel = Panel()
w._scope_panel_open = True

pos = QPoint(int(br.left() + 505 * s), int(br.top() + 250 * s))
cv.mouse_pos = pos
cv.wheelEvent(QWheelEvent(QPointF(pos), QPointF(pos), QPoint(0, 0),
                          QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                          Qt.ScrollUpdate, False))
print('滚轮后 区域宽:', w.region_boxes[0]['width'])
print('滚轮后 面板刷新次数:', len(snapshots))

w.region_boxes[0]['x'] = 210
cv.is_dragging_region = True
cv.mouseReleaseEvent(type('E', (), {'pos': lambda self: pos})())
print('拖动后 面板刷新次数:', len(snapshots))

before = len(snapshots)
cv.mouseReleaseEvent(type('E', (), {'pos': lambda self: pos})())
print('空松手增量:', len(snapshots) - before)
""")
    lines = _lines(out)
    assert lines[0].endswith("305"), f"滚轮没改宽度：{lines[0]}"
    assert lines[1].endswith("1"), f"滚轮后没刷新面板：{lines[1]}"
    assert lines[2].endswith("2"), f"拖动松手后没刷新面板：{lines[2]}"
    assert lines[3].endswith("0"), f"空松手不该白刷面板：{lines[3]}"


def test_wheel_resizes_a_region_edge_from_outside_too():
    """④ 滚轮贴边（含**外侧**）单边调整，与检测框手感一致。"""
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtCore import QPoint, QPointF, Qt
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1200, 900)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(800, 600, QImage.Format_RGB888))
w.background_images = ['a.png']
w.current_background_index = 0
w.edit_mode = 'paste'
app.processEvents()
cv = w.canvas
br = cv.get_background_rect()
s = cv.background_scale
BASE = {'x': 200, 'y': 150, 'width': 300, 'height': 200,
        'shape_type': 'rectangle', 'locked': False}


def run(ix, iy):
    w.paste_region_boxes = [dict(BASE)]
    pos = QPoint(int(br.left() + ix * s), int(br.top() + iy * s))
    cv.mouse_pos = pos
    cv.wheelEvent(QWheelEvent(QPointF(pos), QPointF(pos), QPoint(0, 0),
                              QPoint(0, 120), Qt.NoButton, Qt.NoModifier,
                              Qt.ScrollUpdate, False))
    r = w.paste_region_boxes[0]
    return (r['x'], r['y'], r['width'], r['height'])


print('右边外侧:', run(500 + 5, 250))
print('左边外侧:', run(200 - 5, 250))
print('上边外侧:', run(350, 150 - 5))
print('正中   :', run(350, 250))
""")
    lines = _lines(out)
    # 右边外侧：只加宽（x/y 不动）
    assert lines[0] == "右边外侧: (200, 150, 305, 200)", (
        f"贴右边外侧滚轮没做单边调整（原先这里完全不可达）：{lines[0]}")
    # 左边外侧：只动 x 和宽度，右边界不变（200+300=500）
    assert lines[1] == "左边外侧: (195, 150, 305, 200)", lines[1]
    # 上边外侧：只动 y 和高度，下边界不变（150+200=350）
    assert lines[2] == "上边外侧: (200, 145, 300, 205)", lines[2]
    # 正中：整体等比缩放（四项都变）
    assert lines[3] == "正中   : (192.5, 145.0, 315.0, 210.0)", (
        f"区域中间滚轮应当整体缩放：{lines[3]}")


def test_region_section_only_appears_in_paste_mode():
    """区域那段菜单**只在贴图模式**给（用户第 11 轮）。

    用户口径：「推理是推理的操作，贴图是贴图的操作」「标注模式不要，只走
    推理范围的区域」——推理区域的增删改全部走「推理范围」面板，标注模式的
    右键不该还留一份入口，否则同一份数据两个地方能改。

    贴图模式那份还留着（贴图区域没有面板，右键就是它唯一的编辑入口），
    但**不再有**「贴图区域…」面板入口（用户：「贴图模式右键不需要贴图区域
    管理」）。
    """
    out = _run_real_qt("""
from PySide6.QtWidgets import QApplication, QMenu
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QPoint
from labelforge.ui.main_window import ImageEditor

app = QApplication([])
w = ImageEditor()
w.resize(1200, 900)
w.show()
app.processEvents()
w.current_background = QPixmap.fromImage(
    QImage(800, 600, QImage.Format_RGB888))
w.background_images = ['a.png']
w.current_background_index = 0
w.region_boxes = []
w.paste_region_boxes = []
app.processEvents()
cv = w.canvas

seen = {}
real = QMenu.exec_


def fake_exec(self, *a, **k):
    seen['items'] = [act.text() for act in self.actions()]
    return None


QMenu.exec_ = fake_exec
try:
    w.edit_mode = 'annotate'
    app.processEvents()
    cv._show_background_context_menu(QPoint(10, 10))
    annotate_items = list(seen.get('items', []))

    w.edit_mode = 'paste'
    app.processEvents()
    cv._show_background_context_menu(QPoint(10, 10))
    paste_items = list(seen.get('items', []))
finally:
    QMenu.exec_ = real

REGION_ITEMS = ('添加区域（矩形）', '固定区域', '清空区域')
print('标注 区域项数:', sum(1 for t in annotate_items if t in REGION_ITEMS))
print('贴图 区域项数:', sum(1 for t in paste_items if t in REGION_ITEMS))
print('贴图 有贴图区域面板入口:', any('贴图区域' in t for t in paste_items))
print('贴图 有添加矩形:', '添加区域（矩形）' in paste_items)
print('贴图 有固定:', '固定区域' in paste_items)
print('贴图 有清空:', '清空区域' in paste_items)
print('标注 有删除标签文件:', '删除标签文件' in annotate_items)
print('贴图 有删除标签文件:', '删除标签文件' in paste_items)
""")
    lines = _lines(out)
    assert lines[0].endswith("0"), (
        f"标注模式不该再出现区域段（用户：只走推理范围的区域）：{lines[0]}")
    assert lines[1].endswith("3"), (
        f"贴图模式的区域段应保留三项（添加矩形/固定/清空）：{lines[1]}")
    assert lines[2].endswith("False"), (
        f"贴图模式不该再有「贴图区域…」面板入口：{lines[2]}")
    assert lines[3].endswith("True"), f"贴图模式少了「添加区域（矩形）」：{lines[3]}"
    assert lines[4].endswith("True"), f"贴图模式少了「固定区域」：{lines[4]}"
    assert lines[5].endswith("True"), f"贴图模式少了「清空区域」：{lines[5]}"
    assert lines[6].endswith("True"), "标注模式丢了「删除标签文件」"
    assert lines[7].endswith("True"), "贴图模式丢了「删除标签文件」"
