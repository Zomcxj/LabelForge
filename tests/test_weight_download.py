"""权重下载线程：写 .part → 校验大小与 sha256 → 改名。

真 Qt 的信号要真货，所以整段在子进程里跑；网络用假响应替掉，不联外网。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GUARD = "\nimport os, sys\nsys.stdout.flush()\nos._exit(0)\n"
_QT_EXIT_CRASH = {3221225477, 3221226505}


def _run_real_qt(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT))
    last = None
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", script + _GUARD],
            cwd=ROOT, env=env, text=True, capture_output=True,
            timeout=180, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout
        last = result
        if result.returncode not in _QT_EXIT_CRASH:
            break
    assert last is not None and last.returncode == 0, (
        f"exit={None if last is None else last.returncode} "
        + (last.stderr if last is not None else "子进程没有启动"))
    return last.stdout


def _lines(out):
    return [ln.strip() for ln in out.strip().splitlines() if ln.strip()]


_SETUP = '''
import hashlib
import io
import os
import tempfile
import urllib.request

from PySide6.QtWidgets import QApplication

app = QApplication([])

from labelforge.ui.inference_workers import WeightDownloadWorker

PAYLOAD = b"onnx-bytes" * 512
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Response(io.BytesIO):
    """urlopen 替身：可作上下文管理器，headers 带 Content-Length。"""

    def __init__(self):
        super().__init__(PAYLOAD)
        self.headers = {"Content-Length": str(len(PAYLOAD))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


urllib.request.urlopen = lambda *args, **kwargs: Response()

WORK = tempfile.mkdtemp()


def run_case(name, size, sha):
    target = os.path.join(WORK, name)
    events = []
    worker = WeightDownloadWorker("http://example/x.onnx", target, size, sha)
    worker.progress.connect(lambda done, total: events.append("progress"))
    worker.finished_ok.connect(lambda path: events.append("ok"))
    worker.failed.connect(lambda message: events.append("fail:" + message))
    worker.run()
    return target, events
'''


def test_download_writes_the_file_after_verification():
    out = _run_real_qt(_SETUP + '''
target, events = run_case("ok.onnx", len(PAYLOAD), SHA)
print(os.path.isfile(target))
print(os.path.getsize(target))
print([e for e in events if not e.startswith("fail")])
print(os.path.exists(target + ".part"))
''')
    lines = _lines(out)
    assert lines[0] == "True", lines[0]
    assert lines[1] == str(len(b"onnx-bytes" * 512)), lines[1]
    assert lines[2] == "['progress', 'ok']", lines[2]
    assert lines[3] == "False", "临时文件应当已经改名"


def test_download_rejects_a_size_mismatch():
    out = _run_real_qt(_SETUP + '''
target, events = run_case("size.onnx", len(PAYLOAD) + 1, "")
print(os.path.exists(target))
print(os.path.exists(target + ".part"))
print(events[-1].startswith("fail:"))
print("文件大小不符" in events[-1])
''')
    lines = _lines(out)
    assert lines[0] == "False", "校验失败不能留下目标文件"
    assert lines[1] == "False", "半截文件也要删掉"
    assert lines[2] == "True", lines[2]
    assert lines[3] == "True", "失败原因要说清是大小不符"


def test_download_rejects_a_checksum_mismatch():
    out = _run_real_qt(_SETUP + '''
target, events = run_case("sha.onnx", len(PAYLOAD), "0" * 64)
print(os.path.exists(target))
print(os.path.exists(target + ".part"))
print("校验值不符" in events[-1])
''')
    lines = _lines(out)
    assert lines[0] == "False", lines
    assert lines[1] == "False", lines
    assert lines[2] == "True", lines[2]
