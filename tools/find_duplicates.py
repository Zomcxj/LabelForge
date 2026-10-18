"""定位跨文件的重复代码块，用于决定哪些值得提取。

用法：python tools/find_duplicates.py [窗口行数] [最少重复次数]
"""
import collections
import hashlib
import pathlib
import re
import sys

W = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MIN_HITS = int(sys.argv[2]) if len(sys.argv) > 2 else 2

files = {}
for p in pathlib.Path("labelforge").rglob("*.py"):
    try:
        files[p] = p.read_text(encoding="utf-8").split("\n")
    except Exception:
        continue

windows = collections.defaultdict(list)
for path, lines in files.items():
    for i in range(len(lines) - W + 1):
        stripped = [l.strip() for l in lines[i:i + W]]
        code = [l for l in stripped if l and not l.startswith("#")]
        if len(code) < W - 1:
            continue
        digest = hashlib.md5("\n".join(code).encode()).hexdigest()
        windows[digest].append((path, i + 1, tuple(code)))

# 同一处的连续窗口是同一个重复区域，合并掉：按 (文件, 起始行 - 序号) 分组
regions = collections.defaultdict(list)
for digest, hits in windows.items():
    for path, start, code in hits:
        regions[(digest, str(path), start - len(regions.get((digest, str(path)), [])))].append(
            (path, start, code))

# 收集跨文件重复
cross = []
seen_ranges = set()
for digest, hits in windows.items():
    files_here = {str(p) for p, _, _ in hits}
    if len(files_here) < MIN_HITS:
        continue
    # 合并同文件内的相邻行，得到一个起始位置
    by_file = collections.defaultdict(list)
    for path, start, code in hits:
        by_file[str(path)].append(start)
    starts = {f: min(v) for f, v in by_file.items()}
    key = (digest, tuple(sorted(starts.items())))
    if key in seen_ranges:
        continue
    seen_ranges.add(key)
    cross.append((digest, starts, hits[0][2]))

print(f"跨文件重复块（{W} 行窗口）: {len(cross)}\n")
cross.sort(key=lambda c: -len(c[1]))

for digest, starts, code in cross[:20]:
    names = sorted(starts)
    print("  \u2500\u2500 " + " / ".join(names))
    print("     起始行: " + ", ".join(f"{f}:{starts[f]}" for f in names))
    for line in code[:6]:
        print("       " + line[:96])
    print()