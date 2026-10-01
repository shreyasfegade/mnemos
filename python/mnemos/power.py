"""Lock stores when Windows announces suspend or hibernate.

Page locking keeps the arena out of the pagefile but not out of a hibernation image, which
captures physical memory, the unwrapped master key included (paper, Section 3). Key custody
therefore depends on locking -- zeroizing the arena and dropping the key -- before the machine
sleeps. Windows delivers that warning as ``WM_POWERBROADCAST`` with ``PBT_APMSUSPEND`` to
top-level windows, so the guard owns a hidden top-level window (a message-only window does not
receive broadcasts) on its own thread and locks every registered store when the message arrives.

The guarantee is conditional: it holds if the operating system delivers the notification before
memory is captured. A forced power-off, or a hibernate that skips notifications, is not covered.

    guard = PowerGuard([engine]); guard.start()
    ...
    guard.stop()
"""

from __future__ import annotations

import ctypes
import sys
import threading
from collections.abc import Iterable
from ctypes import wintypes
from typing import Any

WM_POWERBROADCAST = 0x0218
WM_CLOSE = 0x0010
PBT_APMSUSPEND = 0x0004


class PowerGuard:
    """Lock every registered store on ``PBT_APMSUSPEND``."""

    def __init__(self, stores: Iterable[Any]) -> None:
        if sys.platform != "win32":
            raise OSError("PowerGuard uses the Windows power broadcast")
        self.stores = list(stores)
        self.suspends = 0
        self.hwnd: int | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def _lock_all(self) -> None:
        for s in self.stores:
            s.lock()
        self.suspends += 1

    def _run(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        lresult = ctypes.c_ssize_t
        wndproc_t = ctypes.WINFUNCTYPE(
            lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )
        user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.DefWindowProcW.restype = lresult

        def proc(hwnd: int | None, msg: int, wparam: int, lparam: int) -> int:
            if msg == WM_POWERBROADCAST and wparam == PBT_APMSUSPEND:
                self._lock_all()
                return 1
            if msg == WM_CLOSE:
                user32.DestroyWindow(hwnd)
                user32.PostQuitMessage(0)
                return 0
            return int(user32.DefWindowProcW(hwnd, msg, wparam, lparam))

        self._proc = wndproc_t(proc)  # keep a reference for the window's lifetime

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", wndproc_t),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        # Pointer-sized return types throughout: ctypes defaults to a 32-bit int, which truncates
        # 64-bit handles.
        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        hinst = kernel32.GetModuleHandleW(None)
        name = f"MnemosPowerGuard{id(self)}"
        wc = WNDCLASSW()
        wc.lpfnWndProc = self._proc
        wc.hInstance = hinst
        wc.lpszClassName = name
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASSW)]
        if not user32.RegisterClassW(ctypes.byref(wc)):
            self._ready.set()
            raise ctypes.WinError(ctypes.get_last_error())
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        # A top-level window that is never shown: it receives broadcasts, and nobody sees it.
        hwnd = user32.CreateWindowExW(0, name, name, 0, 0, 0, 0, 0, None, None, hinst, None)
        self.hwnd = hwnd
        self._ready.set()
        msg = wintypes.MSG()
        pmsg = ctypes.POINTER(wintypes.MSG)
        user32.GetMessageW.argtypes = [pmsg, wintypes.HWND, wintypes.UINT, wintypes.UINT]
        user32.TranslateMessage.argtypes = [pmsg]
        user32.DispatchMessageW.argtypes = [pmsg]
        user32.DispatchMessageW.restype = lresult
        user32.DestroyWindow.argtypes = [wintypes.HWND]
        user32.UnregisterClassW.argtypes = [wintypes.LPCWSTR, wintypes.HINSTANCE]
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.UnregisterClassW(name, hinst)

    def start(self) -> PowerGuard:
        self._thread = threading.Thread(target=self._run, name="mnemos-power-guard", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if not self.hwnd:
            raise OSError("PowerGuard could not create its window")
        return self

    def stop(self) -> None:
        if self.hwnd:
            post = ctypes.WinDLL("user32").PostMessageW
            post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            post(self.hwnd, WM_CLOSE, 0, 0)
        if self._thread:
            self._thread.join(5)
        self.hwnd = None
