"""Latest-frame capture and bounded mailboxes; no Tk calls from workers."""
import threading
import time
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class TargetSettings:
    auto_aim: bool = False
    auto_fire: bool = False
    shooting_mode: str = "連按"
    trigger_mode: str = "自動"
    trigger_key: str = "F8"
    fire_cps: float = 20.0
    burst_count: int = 3
    burst_gap_ms: float = 100.0
    fire_cooldown_ms: float = 100.0
    fire_jitter_ms: float = 0.0
    fire_weapon: str = "自訂"
    target_part: str = "鎖頭"
    combined_priority: str = "頭部優先"
    head_fallback_distance: float = 90.0
    require_entity_state: bool = False
    require_crosshair_in_box: bool = False
    my_team: str = "CT"
    fov_limit: int = 180  # radius in the original 640 x 640 crop
    conf_threshold: float = 0.35
    aim_tolerance: int = 10
    sensitivity: float = 1.35  # correction gain, not in-game sensitivity
    speed_multiplier: float = 20.0
    max_speed: float = 1000000.0
    max_acceleration: float = 10000000000.0
    bezier_start: float = 1.0
    bezier_end: float = 1.0
    smoothness: float = 0.0
    curve_strength: float = 2.0
    control_hz: float = 2000.0
    disable_acceleration: bool = True
    direct_position: bool = False
    move_deadzone: float = 1.0
    fire_prob: float = 0.7
    reaction_min: float = 0.12
    reaction_max: float = 0.28


class LatestValue:
    def __init__(self):
        self._lock = threading.Condition()
        self._value = None

    def put(self, value):
        with self._lock:
            self._value = value
            self._lock.notify()

    def take(self):
        with self._lock:
            value, self._value = self._value, None
            return value

    def wait_take(self, stop_event):
        with self._lock:
            self._lock.wait_for(lambda: self._value is not None or stop_event.is_set(), timeout=0.05)
            value, self._value = self._value, None
            return value


class LatestCapture:
    """One capture owner continually drains the backend; consumers skip old frames."""
    def __init__(self, index, requested_fps=None, factory=None):
        self.index = index
        self.requested_fps = requested_fps
        self.factory = factory or cv2.VideoCapture
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.sequence = 0
        self.frame = None
        self.timestamp = 0.0
        self.fps = 0.0
        self.reported_fps = 0.0
        self.buffer_supported = False
        self.error = None

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        cap = None
        try:
            cap = self.factory(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError("OBS 畫面開啟失敗，請檢查索引號")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if self.requested_fps is not None:
                cap.set(cv2.CAP_PROP_FPS, self.requested_fps)
            self.buffer_supported = bool(cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            self.reported_fps = cap.get(cv2.CAP_PROP_FPS)
            count, started = 0, time.perf_counter()
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                now = time.perf_counter()
                if not ok:
                    raise RuntimeError("讀不到 OBS 畫面")
                count += 1
                if now - started >= 1.0:
                    self.fps = count / (now - started)
                    count, started = 0, now
                with self.condition:
                    self.sequence += 1
                    self.frame, self.timestamp = frame, now
                    self.condition.notify_all()
        except Exception as error:
            self.error = str(error)
        finally:
            if cap is not None:
                cap.release()
            self.stop_event.set()
            with self.condition:
                self.condition.notify_all()

    def read_latest(self, previous_sequence, timeout=0.05):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence > previous_sequence or self.stop_event.is_set(),
                timeout=timeout,
            )
            if self.stop_event.is_set() or self.sequence <= previous_sequence:
                return None
            return self.sequence, self.timestamp, self.frame

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        # Only the capture thread releases its VideoCapture.
        self.thread.join(timeout=1.0)
        return not self.thread.is_alive()
