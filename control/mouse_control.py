import ctypes
import threading
from ctypes import wintypes

user32 = ctypes.windll.user32

MOUSEEVENTF_MOVE       = 0x0001
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004


class MouseController:
    def __init__(self):
        self.direct_position = False
        self.enabled = False
        self.is_admin = self._check_admin()
        self.fire_mode = "burst"
        self.burst_shots = 3
        self.burst_interval = 0.09
        self.single_hold = 0.04
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread = None
        self._down = False
        self.move_events = 0
        self._saved_acceleration = None

    def set_acceleration_disabled(self, disabled):
        # SPI changes are session-only (no registry broadcast/write).
        with self._lock:
            if disabled and self._saved_acceleration is None:
                previous = (ctypes.c_int * 3)()
                if not user32.SystemParametersInfoW(0x0003, 0, previous, 0):
                    raise RuntimeError("無法讀取 Windows 滑鼠加速設定")
                plain = (ctypes.c_int * 3)(0, 0, 0)
                if not user32.SystemParametersInfoW(0x0004, 0, plain, 0):
                    raise RuntimeError("無法停用 Windows 滑鼠加速")
                self._saved_acceleration = tuple(previous)
            elif not disabled and self._saved_acceleration is not None:
                previous = (ctypes.c_int * 3)(*self._saved_acceleration)
                if not user32.SystemParametersInfoW(0x0004, 0, previous, 0):
                    raise RuntimeError("無法還原 Windows 滑鼠加速")
                self._saved_acceleration = None

    @staticmethod
    def _check_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

    def enable(self):
        self.disable()
        self.wait_stopped()
        with self._lock:
            self._cancel = threading.Event()
            self.enabled = True

    def disable(self):
        with self._lock:
            self.enabled = False
            self._cancel.set()
            self._left_up()

    def wait_stopped(self):
        if self._thread is not None:
            self._thread.join()

    @property
    def busy(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def send_move(self, dx, dy):
        with self._lock:
            if not self.enabled:
                return
            dx, dy = int(round(dx)), int(round(dy))
            if dx or dy:
                if self.direct_position:
                    point = wintypes.POINT()
                    if not user32.GetCursorPos(ctypes.byref(point)):
                        raise RuntimeError("無法取得游標座標")
                    if not user32.SetCursorPos(point.x + dx, point.y + dy):
                        raise RuntimeError("無法設定游標座標")
                else:
                    user32.mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0)
                self.move_events += 1

    def _left_down(self):
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self._down = True

    def _left_up(self):
        if self._down:
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self._down = False

    def set_fire_button(self, down):
        with self._lock:
            if down:
                if self.enabled and not self._down: self._left_down()
            else:
                self._left_up()

    def send_click(self):
        with self._lock:
            if not self.enabled or self.busy:
                return False
            shots = 1 if self.fire_mode == "single" else max(1, min(10, self.burst_shots))
            hold = self.single_hold if self.fire_mode == "single" else 0.03
            self._thread = threading.Thread(
                target=self._click_loop,
                args=(shots, hold, self.burst_interval, self._cancel), daemon=True,
            )
            self._thread.start()
            return True

    def _click_loop(self, shots, hold, interval, cancel):
        try:
            for index in range(shots):
                with self._lock:
                    if not self.enabled or cancel.is_set():
                        return
                    self._left_down()
                cancelled = cancel.wait(hold)
                with self._lock:
                    self._left_up()
                if cancelled or (index < shots - 1 and cancel.wait(interval)):
                    return
        finally:
            with self._lock:
                self._left_up()
