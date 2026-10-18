#!/usr/bin/env bash
# 临时探针（验证完就删）：在 ubuntu:20.04 + conda 里真跑一次 PyInstaller，
# 看产物里到底有没有 Qt 平台插件（libqxcb 等）。
#
# 为什么必须真跑：PyInstaller 收集 Qt 插件靠的是在构建环境里 import Qt；
# 失败时只打一条 warning 就继续，构建「成功」但产物零插件（rc1 就是这样）。
set -u

export DEBIAN_FRONTEND=noninteractive

XCB="libxcb-cursor0 libxkbcommon-x11-0 libegl1-mesa"
BASE="libglib2.0-0 libdbus-1-3 libx11-6 libxext6 libxrender1 libxi6 libxtst6 libsm6 libice6 libgl1"

apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq curl bzip2 fontconfig binutils $XCB $BASE >/dev/null 2>&1

curl -fsSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/mf.sh -b -p /opt/conda >/dev/null 2>&1
/opt/conda/bin/conda install -y python=3.12 >/dev/null 2>&1
/opt/conda/bin/python -m pip install -q "PySide6==6.9.3" pyinstaller >/dev/null 2>&1

echo "===== 1) 各模块能不能导入 ====="
for m in QtCore QtGui QtWidgets QtSvg QtNetwork; do
  if /opt/conda/bin/python -c "import PySide6.$m" >/tmp/e.txt 2>&1; then
    echo "  import PySide6.$m  OK"
  else
    echo "  import PySide6.$m  失败: $(tail -1 /tmp/e.txt)"
  fi
done

echo "===== 2) 真跑 PyInstaller（排除 QtNetwork）====="
cat > /tmp/app.py <<'PY'
from PySide6.QtWidgets import QApplication, QLabel
app = QApplication([])
print("probe ok", QLabel("x").text())
PY
cd /tmp
rm -rf /tmp/bdist /tmp/dist
/opt/conda/bin/python -m PyInstaller --onefile --name probe --noconfirm \
  --exclude-module PySide6.QtNetwork \
  --exclude-module PySide6.QtQml --exclude-module PySide6.QtQuick \
  --exclude-module PySide6.QtPdf --exclude-module PySide6.Qt3DCore \
  --log-level WARN /tmp/app.py > /tmp/build.log 2>&1
echo "  pyinstaller 退出码: $?"
grep -c "failed to obtain Qt library info" /tmp/build.log | sed 's/^/  Qt 探测失败 warning 次数: /'
tail -3 /tmp/build.log | sed 's/^/  build.log 尾: /'

echo "===== 3) 产物里有没有插件 ====="
for k in libqxcb platforms libglib-2.0; do
  printf "  %-14s %s 处命中\n" "$k" "$(grep -ac "$k" /tmp/dist/probe 2>/dev/null)"
done

echo "===== 4) 产物能不能跑（offscreen）====="
QT_QPA_PLATFORM=offscreen timeout 30 /tmp/dist/probe 2>&1 | tail -3 | sed 's/^/  /'
echo "  退出码: $?"