"""The power guard locks a store when Windows announces suspend (paper, Section 3, Hibernation).

A real sleep cannot run in a test, so the test sends the guard's window the exact message Windows
broadcasts before suspend, WM_POWERBROADCAST with PBT_APMSUSPEND, and checks the store is locked:
it refuses to search until unlocked again. ``scripts/suspend_test.ps1`` is the manual check with a
real sleep.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows power broadcast")


def _store(tmp_path):
    from eval.exp2 import lab

    rng = np.random.default_rng(0)
    cents = rng.standard_normal((16, 64)).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    over = lab.base_overrides(dim=64, buckets=16, capacity=16, block=256, epoch_writes=4)
    eng, _, _ = lab.create("m", tmp_path / "s", cents, over, {})
    v = rng.standard_normal(64).astype(np.float32)
    eng.put("r0", (v / np.linalg.norm(v)).tolist(), "{}")
    return eng, v


def test_suspend_broadcast_locks_the_store(tmp_path):
    from mnemos.power import PBT_APMSUSPEND, WM_POWERBROADCAST, PowerGuard

    eng, v = _store(tmp_path)
    eng.search(v.tolist())  # unlocked: searching works

    guard = PowerGuard([eng]).start()
    try:
        send = ctypes.windll.user32.SendMessageW
        send.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        send.restype = ctypes.c_ssize_t
        assert send(guard.hwnd, WM_POWERBROADCAST, PBT_APMSUSPEND, 0) == 1
        assert guard.suspends == 1
        with pytest.raises(RuntimeError, match="locked"):
            eng.search(v.tolist())
    finally:
        guard.stop()


def test_other_power_events_do_not_lock(tmp_path):
    from mnemos.power import WM_POWERBROADCAST, PowerGuard

    eng, v = _store(tmp_path)
    guard = PowerGuard([eng]).start()
    try:
        send = ctypes.windll.user32.SendMessageW
        send.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        send.restype = ctypes.c_ssize_t
        send(guard.hwnd, WM_POWERBROADCAST, 0x0012, 0)  # PBT_APMRESUMEAUTOMATIC
        assert guard.suspends == 0
        eng.search(v.tolist())  # still unlocked
    finally:
        guard.stop()
