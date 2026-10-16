"""滑窗裁剪：网格计算与标注过滤（纯函数）。

约定：可见比例 = 框 bbox 与窗口交集面积 / 框自身面积，低于 min_visible 丢弃；
rectangle 保留交集矩形；polygon/rotation 整框平移不逐点裁剪（与 RandomTranslate 一致）；
point 与退化框（宽或高为 0）按点处理，窗口内（含边界）才保留；空窗由调用方丢弃。

网格两种口径：默认末窗贴边不越界；spec['fill'] 为真时等步长排布且尺寸不钳，
末行/末列窗口可越出图像，越界区由调用方按填充色补齐。
"""
from typing import List

# 补边色的默认值与规范规则由 core 持有（持久化层也要用），此处再导出保持原有导入路径
from ...core.config import FILL_RGB_DEFAULT, normalized_fill_rgb  # noqa: F401

# 默认网格口径：首次配置（无已保存 spec）时按参考图反推出 2 行 3 列、重叠 15%
DEFAULT_CROP_ROWS = 2
DEFAULT_CROP_COLS = 3
DEFAULT_CROP_OVERLAP_RATIO = 0.15
DEFAULT_CROP_REF = 640          # 参考图不可用时的名义尺寸
MAX_WINDOW = 8192               # 窗口尺寸上限（小窗输入框与区间解共用同一口径）
PAD_MAX_PERCENT = 30            # 填充模式下补边不得超过窗口该轴尺寸的 30%


def _ceil_div(a: int, b: int) -> int:
    """整数向上取整除法（b > 0）；避免浮点边界误差。"""
    return -(-a // b)


def window_starts(length: int, size: int, overlap: int, fill: bool = False) -> List[int]:
    """单轴窗口起点。

    fill=False：首窗贴 0、末窗贴边（start+size==length）、余量均摊，窗口不越界；
    fill=True：等步长网格（步长=size−overlap），末窗可越出边缘，越界区由调用方补边。
    """
    if size >= length:
        return [0]
    step = max(1, size - overlap)
    n = 1 + -(-(length - size) // step)  # ceil((length-size)/step)
    if fill:
        return [i * step for i in range(n)]
    span = length - size
    return [round(i * span / (n - 1)) for i in range(n)]


def size_for_count(length: int, count: int, overlap: int) -> int:
    """数量反推尺寸：n*c - (n-1)*ov = A → c=(A+(n-1)*ov)/n，钳制到 [ov+1, A]。"""
    if count <= 1:
        return length
    c = round((length + (count - 1) * overlap) / count)
    return min(max(overlap + 1, c), length)


def ceil_size_for_count(length: int, count: int, overlap: int) -> int:
    """数量反推尺寸（向上取整）：返回恰好排出 count 个窗口的最小尺寸。

    size_for_count 的 round 会把窗口取小，余量就可能多凑出一个窗口
    （length=100、count=3、ov=6 时 round 得 37 → 实际排出 4 列）。
    ceil 取到满足 length-c <= (count-1)*(c-overlap) 的最小 c，故不会超发。
    """
    if count <= 1:
        return length
    c = -(-(length + (count - 1) * overlap) // count)
    return min(max(overlap + 1, c), length)


def default_window(iw: int, ih: int, rows: int = DEFAULT_CROP_ROWS,
                   cols: int = DEFAULT_CROP_COLS,
                   ratio: float = DEFAULT_CROP_OVERLAP_RATIO) -> tuple:
    """默认窗口：按参考图反推出恰好 rows×cols 的网格，重叠取窗口的 ratio。

    返回 (w, h, overlap)。重叠两轴共用一个像素值，无法在两轴上都恰好等于 ratio，
    故以较小窗口边为基准——比例更保守，也不会撞上「未锁数量时重叠不超过半窗」上限。
    定点迭代自 ov=0 起单调递增且有上界，必收敛；iw/ih 非正时退回名义尺寸，口径一致。
    """
    W = iw if iw and iw > 0 else DEFAULT_CROP_REF
    H = ih if ih and ih > 0 else DEFAULT_CROP_REF
    ov, w, h = 0, 0, 0
    for _ in range(32):
        w = ceil_size_for_count(W, cols, ov)
        h = ceil_size_for_count(H, rows, ov)
        nxt = int(round(ratio * min(w, h)))
        if nxt == ov:
            break
        ov = nxt
    return w, h, ov


def exact_overlap_for_flush(length: int, count: int, size: int):
    """恰好铺满时的重叠：由 n*c - (n-1)*ov = L 反解。无解返回 None。

    无解两种情形：窗口总数已盖满该轴（n*c <= L，需要负重叠），或窗口比轴还长。
    整除性也要求，否则该重叠只能近似铺满。
    """
    if count <= 1:
        return 0 if size >= length else None
    if size >= length:
        return None
    num = count * size - length
    if num < 0 or num % (count - 1):
        return None
    ov = num // (count - 1)
    return ov if 0 <= ov < size else None


def size_for_flush(length: int, count: int, overlap: int):
    """恰好铺满时的窗口尺寸：由 n*c - (n-1)*ov = L 反解。无解返回 None。"""
    if count <= 1:
        return length
    num = length + (count - 1) * overlap
    if num % count:
        return None
    c = num // count
    return c if overlap < c <= length else None


def pad_overflow(length: int, size: int, overlap: int) -> int:
    """等步长（填充）网格下，末窗越出图像边界、需要补边的像素数。

    = 末窗起点 + size - length = n*size - (n-1)*overlap - length。
    这是**填充口径专属**的量：不开填充时窗口起点在整段余量上均摊，末窗必贴边
    （末窗起点恒为 length-size），不存在越界，该式对那种落位方式没有意义。
    """
    if size >= length:
        return max(0, size - length)
    return max(0, size - overlap) * (len(window_starts(length, size, overlap, True)) - 1) \
        + size - length


def _pad_ok(length: int, count: int, size: int, overlap: int) -> bool:
    """该 (size, overlap) 组合是否满足「补边不超窗口 30%」。"""
    return pad_overflow(length, size, overlap) * 100 <= PAD_MAX_PERCENT * size


def _pad_capped_size_hi(length: int, count: int, overlap: int, lo: int, hi: int) -> int:
    """在固定窗口数 count 下，二分出满足 30% 补边上限的最大尺寸。

    固定 n 时 pad(c) = n*c-(n-1)*ov-L 对 c 严格单调递增，而右端 30%*c 也递增但更慢，
    故「合规」在 [lo, hi] 上是前缀性质，可二分。
    闭式近似会给偏大的上界（600/ov=60/3 列时闭式给 329，实际 266），必须二分校准。
    """
    if lo > hi or not _pad_ok(length, count, lo, overlap):
        return lo - 1                      # 连下界都不合规：本组合无解
    a, b = lo, hi
    while a < b:
        mid = (a + b + 1) // 2
        if _pad_ok(length, count, mid, overlap):
            a = mid
        else:
            b = mid - 1
    return a


def _pad_capped_overlap_lo(length: int, count: int, size: int, lo: int, hi: int) -> int:
    """在固定窗口数 count 的可行重叠区间 [lo, hi] 内，二分出满足 30% 补边上限的最小重叠。

    搜索必须限定在 [lo, hi] 内：区间之外（如 ov 太小）会排出别的窗口数，
    那些点上的 pad 值与本组合无关，却可能恰好「合规」，把二分引到错误答案。
    区间内 n 固定，pad(ov) = n*c-(n-1)*ov-L 对 ov 单调递减，合规性是后缀性质。
    """
    if lo > hi or not _pad_ok(length, count, size, hi):
        return hi + 1                      # 连上界都不合规：本组合无解
    a, b = lo, hi
    while a < b:
        mid = (a + b) // 2
        if _pad_ok(length, count, size, mid):
            b = mid
        else:
            a = mid + 1
    return a


def pad_compliant_size_run(length: int, size: int, overlap: int,
                           max_size: int = MAX_WINDOW) -> tuple:
    """返回包含 size（size 违规时取最近段）的那一段「补边合规」尺寸段 (lo, hi)。

    固定重叠时合规尺寸集是**锯齿状多段**（窗口数每变一次 pad 就跳回 0），不是连续区间，
    所以硬墙必须按「当前值所在的段」来设，不能只钳最外端上下界——否则用户会停在
    两段之间的空洞里（600 图上 ov=60 有 15 段）。
    无任何合规值时返回 hi < lo。

    扫描上界：size >= length 时 pad = size − length，30% 上限即 size ≤ 10·length/7，
    再大的尺寸必然违规，无需继续扫。
    """
    if length <= 0 or overlap >= length or max_size < overlap + 1:
        return (0, -1)
    lo = overlap + 1
    hi = min(max_size, (10 * length) // 7)
    if hi < lo:
        return (0, -1)
    runs = []
    cur_lo = None
    for c in range(lo, hi + 1):
        if _pad_ok(length, len(window_starts(length, c, overlap, True)), c, overlap):
            if cur_lo is None:
                cur_lo = c
        elif cur_lo is not None:
            runs.append((cur_lo, c - 1))
            cur_lo = None
    if cur_lo is not None:
        runs.append((cur_lo, hi))
    if not runs:
        return (0, -1)
    for lo_r, hi_r in runs:
        if lo_r <= size <= hi_r:
            return (lo_r, hi_r)
    # size 违规：取最近的一段（等距取较小值所在段）
    # 段选择层的平局是真实可达的（如 L=400/ov=0 时 size=3 到段 (1,2) 与 (4,5) 等距），
    # 固定取较小段以保证结果确定
    return min(runs, key=lambda r: (min(abs(r[0] - size), abs(r[1] - size)), r[0]))


def nearest_pad_compliant_size(length: int, size: int, overlap: int) -> int:
    """尺寸超限时回退到最近（等距取较小）的合规尺寸；已合规则原样返回。

    尺寸必须 > overlap（否则窗口退化），落在域外的输入按「最近的合规段」吸附。
    """
    if size > overlap and _pad_ok(length, len(window_starts(length, size, overlap, True)),
                                  size, overlap):
        return size
    lo, hi = pad_compliant_size_run(length, size, overlap)
    if hi < lo:
        return size                             # 无合规值：保持原值，由调用方提示
    # 段内连续，故 size 在段外时到两端必不等距：取较近的端点即可
    return min((lo, hi), key=lambda v: abs(v - size))


def pad_compliant_overlap_run(iw: int, ih: int, w: int, h: int,
                              overlap: int, hi_limit: int = None) -> tuple:
    """返回包含 overlap（或最近）的那一段「两轴同时合规」的重叠段 (lo, hi)。

    重叠两轴共用一个值，故硬墙须取两轴的联合合规集；只按单轴算会让另一轴偷偷超限
    （实测宽 93%/高 90%）。该集合同样是锯齿状多段。无解返回 hi < lo。

    hi_limit 用于叠加别的硬上限（未锁数量时的「半窗」规则）——必须在扫描时就限住，
    否则返回的段会越过该上限，调用方再钳一次就会得到空区间、把重叠值吃掉。
    """
    if w <= 0 or h <= 0:
        return (0, -1)
    hi_max = min(w, h) - 1
    if hi_limit is not None:
        hi_max = min(hi_max, int(hi_limit))
    if hi_max < 0:
        return (0, -1)
    runs = []
    cur_lo = None
    for ov in range(0, hi_max + 1):
        if (_pad_ok(iw, len(window_starts(iw, w, ov, True)), w, ov)
                and _pad_ok(ih, len(window_starts(ih, h, ov, True)), h, ov)):
            if cur_lo is None:
                cur_lo = ov
        elif cur_lo is not None:
            runs.append((cur_lo, ov - 1))
            cur_lo = None
    if cur_lo is not None:
        runs.append((cur_lo, hi_max))
    if not runs:
        return (0, -1)
    for lo_r, hi_r in runs:
        if lo_r <= overlap <= hi_r:
            return (lo_r, hi_r)
    return min(runs, key=lambda r: (min(abs(r[0] - overlap), abs(r[1] - overlap)),
                                    r[0]))


def nearest_pad_compliant_overlap(iw: int, ih: int, w: int, h: int,
                                  overlap: int, hi_limit: int = None) -> int:
    """重叠超限时回退到最近（等距取较小）的两轴同时合规值；已合规原样返回。"""
    if (_pad_ok(iw, len(window_starts(iw, w, overlap, True)), w, overlap)
            and _pad_ok(ih, len(window_starts(ih, h, overlap, True)), h, overlap)
            and (hi_limit is None or overlap <= hi_limit)):
        return overlap
    lo, hi = pad_compliant_overlap_run(iw, ih, w, h, overlap, hi_limit)
    if hi < lo:
        return overlap
    return min((lo, hi), key=lambda v: abs(v - overlap))


def size_range_for_count(length: int, count: int, overlap: int,
                         max_size: int = MAX_WINDOW, fill: bool = False) -> tuple:
    """锁定「重叠+数量」时尺寸的可行区间 (lo, hi)；无解返回 hi < lo。

    由 n = 1 + ceil((L-c)/(c-ov)) 反解出 n-2 < (L-c)/(c-ov) <= n-1：
    下界来自右半式（c 太小会多凑出窗口），上界来自左半式（c 太大余量不足会少一个）。
    count=1 时任何 c >= L 都只出一窗，故区间为 [L, max_size]。
    区间内每个尺寸都恰好排出 count 个窗口——这就是「锁定两项并非唯一确定第三项」。

    fill=True 时网格按等步长落位、末窗可越界，需再叠加「补边不超窗口 30%」的硬约束，
    否则会挑出末窗几乎全是补边色的垃圾窗口（实测可越界到窗口宽的 91%）。
    """
    if count <= 1:
        return (min(length, max_size), max_size)
    if overlap >= length:
        return (0, -1)
    lo = max(overlap + 1, _ceil_div(length + (count - 1) * overlap, count))
    hi = min(length - 1, _ceil_div(length + (count - 2) * overlap, count - 1) - 1)
    if fill:
        hi = _pad_capped_size_hi(length, count, overlap, lo, hi)
    lo = min(lo, max_size)
    hi = min(hi, max_size)
    return (lo, hi)


def overlap_range_for_count(length: int, count: int, size: int,
                            fill: bool = False) -> tuple:
    """锁定「尺寸+数量」时重叠的可行区间 (lo, hi)；无解返回 hi < lo。

    同样由 n-2 < (L-c)/(c-ov) <= n-1 反解，但这次未知量是 ov：
      n >= 3：hi = c - ceil(D/(n-1))，lo = c - ceil(D/(n-2)) + 1
      n == 2：n-2=0 使左半式失去下界，退化为 0 <= D/(c-ov) <= 1 → ov <= 2c-L
    再统一钳到 [0, c-1]。fill=True 时叠加补边不超 30% 的下界。
    """
    if count <= 1:
        # 只有窗口本身盖满该轴才会恰好一窗；size < length 时无论重叠多大都至少两窗
        return (0, size - 1) if size >= length else (0, -1)
    if size >= length:
        # window_starts 对 size >= length 直接返回 [0]：恒为一窗，多于一个窗不可达
        return (0, -1)
    d = length - size
    hi = size - _ceil_div(d, count - 1)
    lo = 0 if count == 2 else size - _ceil_div(d, count - 2) + 1
    lo = max(0, lo)
    hi = min(hi, size - 1)
    if fill:
        # 必须把「保持 count 个窗口」的原始区间交给二分，否则区间外的重叠会误导搜索
        lo = max(lo, _pad_capped_overlap_lo(length, count, size, lo, hi))
    return (lo, hi)


def overlap_for_count(length: int, count: int, size: int) -> int:
    """数量+尺寸反推重叠：n*c - (n-1)*ov = A → ov=(n*c-A)/(n-1)，钳制到 [0, c-1]。

    向下取整以保证 n 个窗口仍盖满 length；count*size < length 时（重叠为 0 也盖不满）
    返回 0，由调用方按实际窗口数提示不可行。
    """
    if count <= 1:
        return 0
    ov = (count * size - length) // (count - 1)
    return max(0, min(ov, size - 1))


def normalized_window(iw: int, ih: int, spec: dict) -> tuple:
    """把 spec 钳制到图像尺寸：返回 (cw, ch, ov)，面板/预览/引擎三处共用。

    spec['fill'] 为真时窗口尺寸严格固定（不钳到图像内），越界部分由引擎补边。
    """
    if spec.get("fill"):
        cw, ch = int(spec["w"]), int(spec["h"])
    else:
        cw = min(int(spec["w"]), iw)
        ch = min(int(spec["h"]), ih)
    ov = max(0, min(int(spec["overlap"]), cw - 1, ch - 1))
    return cw, ch, ov


def visible_ratio(box, window):
    """框在窗口内的可见比例 = 交集面积 / 框自身面积。

    `window` 是 `(x, y, w, h)`，`box` 是 `{x, y, width, height}`（原图坐标）。
    零交集返回 0.0；退化框（宽或高为 0）按点处理：落在窗内（含边界）算 1.0，
    否则 0.0——与 `crop_boxes` 对退化框的处理保持一致。

    抽出来是为了让**推理侧**也能用同一条判据：滑窗推理按窗口裁图后，模型的框
    会被钳进裁剪图，跨窗去重前先按「它在原图窗口里露了多少」过滤掉只露一角的
    框。原来这段判定内联在 `crop_boxes` 里，推理侧无从复用。
    """
    x0, y0, cw, ch = window
    wx1, wy1 = x0 + cw, y0 + ch
    bx1 = float(box.get("x", 0.0))
    by1 = float(box.get("y", 0.0))
    bw = float(box.get("width", 0.0))
    bh = float(box.get("height", 0.0))
    if bw <= 0 or bh <= 0:
        cx, cy = bx1 + bw / 2.0, by1 + bh / 2.0
        inside = (x0 <= cx <= wx1 and y0 <= cy <= wy1)
        return 1.0 if inside else 0.0
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(bx1, float(x0)), max(by1, float(y0))
    ix2, iy2 = min(bx2, float(wx1)), min(by2, float(wy1))
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1) / (bw * bh)


def keep_by_min_visible(box, window, min_visible):
    """该框是否通过最低可见比例。零交集无条件丢弃（min_visible=0 也不例外）。"""
    ratio = visible_ratio(box, window)
    if ratio <= 0.0:
        return False
    return ratio >= float(min_visible)


def crop_boxes(boxes: List[dict], x0: int, y0: int, cw: int, ch: int,
               min_visible: float) -> List[dict]:
    """把标注过滤/平移到窗口 (x0,y0,cw,ch) 坐标系；不修改入参 box。"""
    from .base import map_box_points

    out = []
    wx1, wy1 = x0 + cw, y0 + ch
    for b in boxes:
        st = b.get("shape_type") or "rectangle"
        if st == "point":
            pts = b.get("points")
            px, py = (pts[0][0], pts[0][1]) if pts else (b["x"], b["y"])
            if x0 <= px <= wx1 and y0 <= py <= wy1:
                out.append(map_box_points(b, lambda x, y: (x - x0, y - y0)))
            continue
        pts = b.get("points")
        if pts and st != "rectangle":
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
        else:
            xs = [b["x"], b["x"] + b["width"]]
            ys = [b["y"], b["y"] + b["height"]]
        bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
        bw, bh = bx2 - bx1, by2 - by1
        if bw <= 0 or bh <= 0:  # 退化框按点处理
            cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
            if x0 <= cx <= wx1 and y0 <= cy <= wy1:
                out.append(map_box_points(b, lambda x, y: (x - x0, y - y0)))
            continue
        # 用共享判据：推理侧的跨窗去重走同一套公式，两边口径不会分叉
        if not keep_by_min_visible(b, (x0, y0, cw, ch), min_visible):
            continue
        if st == "rectangle" and not pts:
            nx1, ny1 = max(b["x"], x0), max(b["y"], y0)
            nb = dict(b)
            nb["x"], nb["y"] = nx1 - x0, ny1 - y0
            nb["width"] = min(b["x"] + b["width"], wx1) - nx1
            nb["height"] = min(b["y"] + b["height"], wy1) - ny1
            if nb["width"] >= 1 and nb["height"] >= 1:
                out.append(nb)
        else:
            out.append(map_box_points(b, lambda x, y: (x - x0, y - y0)))
    return out
