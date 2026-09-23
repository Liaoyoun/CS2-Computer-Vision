"""Deadline-based Windows high-resolution waitable timer, with explicit fallback."""
import ctypes
from ctypes import wintypes
import time

class ControlTimer:
    def __init__(self, stop):
        self.stop = stop
        self._period_set = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        self.handle = None
        self.deadline = time.perf_counter()
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateWaitableTimerExW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
        self.api.CreateWaitableTimerExW.restype = wintypes.HANDLE
        self.api.SetWaitableTimer.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong), wintypes.LONG, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL]
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateWaitableTimerExW(None, None, 2, 0x1F0003)

    def wait(self, hz):
        interval = 1 / hz
        now = time.perf_counter()
        # Never send catch-up bursts after scheduler stalls.
        self.deadline += interval
        if now >= self.deadline:
            self.deadline = now + interval
        delay = max(0, self.deadline - now)
        if self.handle:
            due = ctypes.c_longlong(-max(1, round(delay * 10_000_000)))
            if not self.api.SetWaitableTimer(self.handle, ctypes.byref(due), 0, None, None, False):
                raise ctypes.WinError(ctypes.get_last_error())
            if self.api.WaitForSingleObject(self.handle, 100) != 0:
                raise RuntimeError('Control timer wait failed')
        else:
            self.stop.wait(delay)
        return not self.stop.is_set()

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
        if self._period_set:
            ctypes.windll.winmm.timeEndPeriod(1)
            self._period_set = False
