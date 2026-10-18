"""并行扫描骨架：完成顺序、中断语义、结果不全的标记。

这份骨架原先在质检和数据集健康里各写一遍，中断处理最容易只改一边：
`shutdown(wait=...)` 与 `cancel_futures` 必须跟着中断状态走，否则
requestInterruption 之后仍会被排队中的读盘任务拖住。
"""
import concurrent.futures
import threading

from labelforge.engine.parallel_scan import scan_images


def test_results_arrive_for_every_image():
    seen = []
    scan_images([f"img{i}.json" for i in range(5)],
                work=lambda i, p: (i, p.upper()),
                on_result=seen.append)
    assert sorted(seen) == [(i, f"IMG{i}.JSON") for i in range(5)]


def test_work_receives_the_index_and_the_path():
    args = []
    scan_images(["a.json", "b.json"],
                work=lambda i, p: args.append((i, p)) or None,
                on_result=lambda _r: None)
    assert sorted(args) == [(0, "a.json"), (1, "b.json")]


def test_no_images_is_a_no_op():
    calls = []
    interrupted = scan_images([], work=lambda i, p: None,
                              on_result=calls.append)
    assert interrupted is False
    assert calls == []


def test_interruption_stops_consuming_and_reports_incomplete():
    """中断后立刻停手，并把「结果不全」这件事报给调用方。"""
    started = threading.Event()
    release = threading.Event()
    consumed = []

    def work(i, p):
        started.set()
        release.wait(5)
        return i

    def interrupted():
        return started.is_set()

    result = {}
    t = threading.Thread(
        target=lambda: result.update(
            interrupted=scan_images([f"i{i}" for i in range(20)],
                                    work=work, on_result=consumed.append,
                                    is_interrupted=interrupted)))
    t.start()
    started.wait(5)
    release.set()
    t.join(10)

    assert result.get("interrupted") is True, "中断没有被报出来"
    # 20 张里第一张就开始等，中断后消费者不该把 20 张全收完
    assert len(consumed) < 20, f"中断后仍在继续消费: {len(consumed)}"


def test_interruption_is_not_reported_when_never_requested():
    result = scan_images(["a", "b"], work=lambda i, p: i,
                         on_result=lambda _r: None,
                         is_interrupted=lambda: False)
    assert result is False


def test_shutdown_drops_queued_work_when_interrupted(monkeypatch):
    """中断时 shutdown 必须是 (wait=False, cancel_futures=True)。

    这条不靠计时：卡时间会随机器负载抖动，而且真正要守的是调用的**参数**
    契约 —— `wait=True` 会让 requestInterruption 之后仍等到排队任务跑完。
    """
    calls = []
    real = concurrent.futures.ThreadPoolExecutor

    class Spy(real):
        def shutdown(self, wait=True, *, cancel_futures=False):
            calls.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Spy)

    seen = []
    flag = scan_images(["a", "b", "c"], work=lambda i, p: i,
                       on_result=seen.append,
                       is_interrupted=lambda: bool(seen))

    assert flag is True, "中断没有被报出来"
    assert calls and calls[0] == (False, True), (
        f"中断后 shutdown 参数应为 (wait=False, cancel_futures=True)，"
        f"实际 {calls}")


def test_shutdown_waits_when_not_interrupted(monkeypatch):
    """没被中断时要等任务收尾，不能把结果丢掉。"""
    calls = []
    real = concurrent.futures.ThreadPoolExecutor

    class Spy(real):
        def shutdown(self, wait=True, *, cancel_futures=False):
            calls.append((wait, cancel_futures))
            super().shutdown(wait=wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", Spy)

    seen = []
    flag = scan_images(["a", "b"], work=lambda i, p: i, on_result=seen.append)

    assert flag is False
    assert sorted(seen) == [0, 1]
    assert calls == [(True, False)], calls