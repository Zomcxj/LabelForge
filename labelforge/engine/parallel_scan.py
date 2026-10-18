"""按图并行跑一件读盘工作，并支持中途放弃。

质检和数据集健康两边都需要同一种骨架：把「每张图算一遍」丢进线程池、按完成
顺序消费结果、用户可以中途打断（关窗口或再点一次统计）。这份骨架原先各写了
一份，中断处理（`shutdown(wait=...)` 与 `cancel_futures` 必须跟着中断状态走）
被复制成两处，改动时极易只改一边。

调用方只提供两件事：`work(index, path)` 干什么，`on_result(结果)` 拿到后做什么。
"""

import concurrent.futures

DEFAULT_WORKERS = 8


def scan_images(image_paths, work, on_result, is_interrupted=None,
                max_workers=DEFAULT_WORKERS):
    """对每张图跑 `work`，按完成顺序调用 `on_result`。

    `work(index, path)` 在线程池里执行；返回值原样交给 `on_result`。
    `is_interrupted()` 返回真时立刻停止消费，**并放弃仍在排队、还没开始的
    任务**：`wait=True` 会让 requestInterruption 之后仍被未启动的读盘任务
    拖住，所以中断时两者必须都是「不等」。

    返回 True 表示被中断、结果不完整。
    """
    if not image_paths:
        return False

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    interrupted = False
    try:
        futures = [pool.submit(work, i, p) for i, p in enumerate(image_paths)]
        for future in concurrent.futures.as_completed(futures):
            if is_interrupted is not None and is_interrupted():
                interrupted = True
                break
            on_result(future.result())
    finally:
        pool.shutdown(wait=not interrupted, cancel_futures=interrupted)
    return interrupted