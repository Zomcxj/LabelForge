#!/usr/bin/env bash
# 临时探针（验证完就删）：ubuntu:20.04 容器里，PyInstaller 到底能不能 import Qt。
#
# PyInstaller 收集 Qt 平台插件靠的是在构建环境里真的 import Qt。import 不成功时
# 它只打一条 warning 就继续，产物缺插件、启动即崩。这个脚本用三种 apt 组合看
# 哪种能让 Qt 导入成功，并打出 Qt 自己认为的插件目录。
set -u

MODE="${1:-baselibs}"
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl bzip2 fontconfig binutils >/dev/null 2>&1

XCB="libxcb-cursor0 libxkbcommon-x11-0 libegl1-mesa"
BASE="libglib2.0-0 libdbus-1-3 libx11-6 libxext6 libxrender1 libxi6 libxtst6 libsm6 libice6"

case "$MODE" in
  original) apt-get install -y -qq $XCB >/dev/null 2>&1 ;;
  baselibs) apt-get install -y -qq $XCB $BASE >/dev/null 2>&1 ;;
  sysfirst)
    apt-get install -y -qq $XCB $BASE >/dev/null 2>&1
    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
    ;;
esac

curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p /opt/conda >/dev/null 2>&1
/opt/conda/bin/conda install -y python=3.12 >/dev/null 2>&1
/opt/conda/bin/python -m pip install -q "PySide6==6.9.3" pyinstaller >/dev/null 2>&1

echo "===== MODE=$MODE ====="
for m in QtCore QtGui QtWidgets QtSvg QtNetwork; do
  if /opt/conda/bin/python -c "import PySide6.$m" >/tmp/e.txt 2>&1; then
    echo "  import PySide6.$m  OK"
  else
    echo "  import PySide6.$m  失败: $(tail -1 /tmp/e.txt)"
  fi
done

/opt/conda/bin/python - <<'PY' 2>&1 | tail -4
import os
from PySide6.QtCore import QLibraryInfo
p = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
d = os.path.join(p, "platforms")
print("  PluginsPath =", p)
print("  platforms 目录 =", sorted(os.listdir(d)) if os.path.isdir(d) else "<不存在>")
PY