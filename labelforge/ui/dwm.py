"""
Windows DWM API 模块 - 设置标题栏深色/浅色模式
兼容 Win10 1809+ (属性19) 和 Win10 1903+ / Win11 (属性20)
优先使用 ctypes，不可用时使用 PowerShell
"""
import sys
import subprocess

_dwmapi = None
_initialized = False
_dwm_available = False
_titlebar_states = {}


def _refresh_window_frame(hwnd):
    """强制 Windows 重新绘制非客户区（标题栏/边框）。"""
    if sys.platform != 'win32' or not hwnd:
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32

        swp_nomove = 0x0002
        swp_nosize = 0x0001
        swp_nozorder = 0x0004
        swp_noactivate = 0x0010
        swp_framechanged = 0x0020
        user32.SetWindowPos(
            ctypes.c_void_p(hwnd), None, 0, 0, 0, 0,
            swp_nomove | swp_nosize | swp_nozorder | swp_noactivate | swp_framechanged
        )

        rdw_invalidate = 0x0001
        rdw_frame = 0x0400
        rdw_updatenow = 0x0100
        user32.RedrawWindow(
            ctypes.c_void_p(hwnd), None, None,
            rdw_invalidate | rdw_frame | rdw_updatenow
        )
    except Exception:
        pass


def _init():
    global _dwmapi, _initialized, _dwm_available
    if _initialized:
        return
    _initialized = True
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        _dwmapi = ctypes.windll.dwmapi
        _dwm_available = True
    except Exception:
        _dwm_available = False


def set_titlebar_dark(hwnd, dark, force_refresh=False):
    """
    设置 Windows 标题栏深色/浅色模式
    :param hwnd: 窗口句柄 (int)
    :param dark: True=深色标题栏, False=浅色标题栏
    :param force_refresh: True 时强制刷新窗口框架，用于主题切换；
                          普通弹窗打开时保持 False，避免 Win11 闪烁。
    """
    _init()
    if hwnd and _titlebar_states.get(hwnd) == dark and not force_refresh:
        return True

    if _dwm_available and hwnd:
        try:
            import ctypes
            value = ctypes.c_int(1 if dark else 0)
            # 20: Win10 1903+ / Win11；19: Win10 1809 兼容属性。
            # DwmSetWindowAttribute 返回 HRESULT，不一定抛异常，所以两个都尝试。
            _dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
            _dwmapi.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), ctypes.sizeof(value))
            _titlebar_states[hwnd] = dark
            if force_refresh:
                _refresh_window_frame(hwnd)
            return True
        except Exception:
            pass

    if sys.platform == 'win32' and hwnd:
        try:
            v = 1 if dark else 0
            ps = (
                'Add-Type @"'
                'using System.Runtime.InteropServices;'
                'public class Dwm{{'
                '[DllImport("dwmapi.dll")]'
                'public static extern int DwmSetWindowAttribute(System.IntPtr h,int a,ref int v,int s);}}'
                '"@;'
                '[Dwm]::DwmSetWindowAttribute([IntPtr]{h},20,[ref]$v,4);'
                '[Dwm]::DwmSetWindowAttribute([IntPtr]{h},19,[ref]$v,4)'
            ).format(v=v, h=hwnd)
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps],
                creationflags=0x08000000
            )
            _titlebar_states[hwnd] = dark
            if force_refresh:
                _refresh_window_frame(hwnd)
            return True
        except Exception:
            pass

    return False


def is_available():
    _init()
    return _dwm_available
