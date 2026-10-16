"""诊断探针：追踪「json 为什么不见了」。

⚠️ 这是**临时排查工具**，不是产品功能。用来回答一个问题：

    某个 sidecar json 到底是**谁**删的？还是**从来就没写过**？

做法是在所有会改动/删除 sidecar 的位置埋点，把「谁、对哪个文件、当时
文件在不在、shapes 有几条」写进日志；同时在切图时记录当前图的 json 状态。
用户跑一遍出问题的流程，日志就能直接指认凶手，而不是靠猜。

开法：环境变量 `LABELFORGE_TRACE_JSON=1`（debug exe 里默认开）。
关法：不设那个变量。**正常版本完全不受影响**——所有埋点先问 `_on()`。
"""
import os
import traceback

from .exception_hook import _write_log

_ON = None


def _on():
    """是否开启探针。默认关；`LABELFORGE_TRACE_JSON=1` 时开。

    旧名 `PASTELABEL_TRACE_JSON` 仍然认，免得老排查脚本失效。
    """
    global _ON
    if _ON is None:
        value = os.environ.get('LABELFORGE_TRACE_JSON',
                               os.environ.get('PASTELABEL_TRACE_JSON', ''))
        _ON = value == '1'
    return _ON


def _shapes_count(json_path):
    """"文件里 shapes 有几条"，用于日志；读不出来返回 None。"""
    try:
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        shapes = data.get('shapes')
        return len(shapes) if isinstance(shapes, list) else None
    except Exception:
        return None


def trace(where, json_path, **extra):
    """记一条 sidecar 相关的事件（只有开启时才写）。"""
    if not _on():
        return
    exists = os.path.exists(json_path) if json_path else None
    n = _shapes_count(json_path) if exists else None
    detail = ' '.join(f'{k}={v}' for k, v in extra.items())
    _write_log(
        f"[JSONTRACE] {where} path={json_path} exists={exists} shapes={n} {detail}")


def trace_delete(where, json_path, survived):
    """记一次「删除 sidecar 的尝试」，**带调用栈**——这是定位凶手的关键。"""
    if not _on():
        return
    stack = ' <- '.join(
        f"{os.path.basename(fr.filename)}:{fr.lineno}"
        for fr in traceback.extract_stack()[:-1][-6:]
    )
    _write_log(
        f"[JSONTRACE] {where} path={json_path} "
        f"实际是否删掉={survived} 调用栈={stack}")


def trace_switch(index, image_path):
    """切图时记录：这张图、以及**这张图的 json** 当前状态。"""
    if not _on():
        return
    json_path = f"{os.path.splitext(image_path)[0]}.json" if image_path else None
    trace('切图', json_path, index=index)


def trace_load_result(index, image_path, memory_boxes, from_cache):
    """记一次「切到这张图后到底读到几个框」。

    这是验证缓存修复的关键一行：磁盘上有框而内存是 0，就是老 bug 复发。
    `from_cache=True` 表示命中缓存、没读盘。
    """
    if not _on():
        return
    json_path = f"{os.path.splitext(image_path)[0]}.json" if image_path else None
    disk = _shapes_count(json_path) if json_path and os.path.exists(json_path) else None
    flag = ''
    if isinstance(disk, int) and disk > 0 and memory_boxes == 0:
        flag = '  ⚠️磁盘有框但内存为 0（缓存被污染！）'
    _write_log(
        f"[JSONTRACE] 读框 index={index} 磁盘shapes={disk} "
        f"内存框数={memory_boxes} 来自缓存={from_cache}{flag}")
