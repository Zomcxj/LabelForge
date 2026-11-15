"""抠图后处理：纯 numpy，不依赖 Qt，也不依赖具体模型。"""

import numpy as np


def alpha_from_output(raw):
    """模型输出 -> [0, 1] 的 alpha。

    输出越出 [0, 1] 说明是 logits，施加 sigmoid；否则视为概率直接用。
    之所以不用逐图 min-max 拉伸：那会把软边拉硬，也会把低响应图的噪声拉满。
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("模型输出为空")
    if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
        return 1.0 / (1.0 + np.exp(-arr))
    return np.clip(arr, 0.0, 1.0)


def apply_threshold(alpha, threshold):
    """threshold <= 0 保留软 alpha；否则二值化。"""
    arr = np.asarray(alpha, dtype=np.float32)
    if threshold <= 0:
        return arr
    return (arr >= threshold).astype(np.float32)


def quantise_alpha(alpha):
    """float alpha -> 落盘用的 8 位 alpha（与 `compose_rgba` 同一量纲）。"""
    arr = np.asarray(alpha, dtype=np.float32)
    return np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)


def alpha_bbox(alpha, pad=1):
    """alpha > 0 的包围盒 (top, left, bottom, right)；全透明返回 None。

    「> 0」按 **8 位量化后** 的 alpha 判：PNG 里存的就是 8 位 alpha，
    sigmoid 输出的极小概率在 float 下永远不为 0，按 float 判会让包围盒
    恒等于整块裁窗，det 框比物体大一圈。
    """
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"期望二维 alpha，实际 {arr.shape}")
    quantised = quantise_alpha(arr)
    rows = np.flatnonzero(quantised.max(axis=1) > 0)
    cols = np.flatnonzero(quantised.max(axis=0) > 0)
    if rows.size == 0 or cols.size == 0:
        return None
    return (max(0, rows[0] - pad),
            max(0, cols[0] - pad),
            min(arr.shape[0], rows[-1] + 1 + pad),
            min(arr.shape[1], cols[-1] + 1 + pad))


def crop_to_alpha(alpha, pad=1):
    """裁到 alpha 包围盒并外留 pad 像素；全透明返回 None。

    包围盒范围与 `alpha_bbox` 同一口径（8 位量化后 > 0）；裁出来的 alpha
    保持原样，不二值化，软边必须留住。
    """
    box = alpha_bbox(alpha, pad)
    if box is None:
        return None
    top, left, bottom, right = box
    return np.asarray(alpha, dtype=np.float32)[top:bottom, left:right]


def crop_to_alpha_or_raise(alpha, pad=1):
    """同 crop_to_alpha，但全透明时抛 ValueError 而不是返回 None。

    编排层拿不到裁块就没法继续（会写出 0×0 数组），所以在这里给出可读错误。
    """
    cropped = crop_to_alpha(alpha, pad)
    if cropped is None:
        raise ValueError("抠图结果全透明：模型未检出前景，请换一个目标或调整阈值")
    return cropped


def compose_rgba(rgb, alpha):
    """RGB + alpha -> RGBA uint8。"""
    rgb_arr = np.asarray(rgb)
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if rgb_arr.shape[:2] != alpha_arr.shape[:2]:
        raise ValueError(
            f"RGB {rgb_arr.shape[:2]} 与 alpha {alpha_arr.shape[:2]} 尺寸不一致")
    out = np.zeros((*rgb_arr.shape[:2], 4), dtype=np.uint8)
    out[:, :, :3] = rgb_arr[:, :, :3]
    # 与 alpha_bbox 同一套量化规则，避免框比实际内容多出一圈全透明像素
    out[:, :, 3] = quantise_alpha(alpha_arr)
    return out
