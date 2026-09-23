"""Independent, latest-target control worker; never blocks model inference."""
import math
import threading
import time

from control.mouse_control import MouseController
from runtime.high_timer import ControlTimer
from control.fire_scheduler import FireScheduler
from control.physical_trigger import PhysicalTrigger


def bezier_point(t, dx, dy, start_control=0.55, end_control=0.95, strength=1.0):
    """Cubic Bezier P0=0, P1=start, P2=end, P3=1; fast start, soft landing."""
    t = max(0.0, min(1.0, t))
    t = 1 - (1 - t) ** strength
    factor = (3 * (1 - t) ** 2 * t * start_control
              + 3 * (1 - t) * t ** 2 * end_control + t ** 3)
    return round(dx * factor), round(dy * factor)


class TargetControl:
    def __init__(self, get_settings, log, mouse=None, trigger_factory=None):
        self.get_settings = get_settings
        self.log = log
        self.mouse = mouse or MouseController()
        self.scheduler = FireScheduler()
        self.trigger = (trigger_factory or PhysicalTrigger)(get_settings, self.trigger_edge)
        self.trigger_held = False
        self.trigger_latched = False
        initial = get_settings()
        self._trigger_signature = (initial.auto_fire, initial.trigger_mode, initial.trigger_key)
        self.fire_status = "等待目標"
        self.shot_count = 0
        self.trigger_started = False
        self._lock = threading.Lock()
        self._packet = None
        self.status = "等待畫面"
        self._version = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.get_settings().trigger_mode != "自動":
            self.trigger.start()
            self.trigger_started = True
        self.thread.start()
        return self

    def submit(self, target, captured_at, settings):
        with self._lock:
            self._version += 1
            self._packet = (self._version, target, captured_at, settings)

    def stop(self):
        self._stop.set()
        self.trigger.stop()
        self.mouse.disable()
        if self.thread.is_alive():
            self.thread.join(timeout=1)

    def trigger_edge(self, down):
        settings = self.get_settings()
        with self._lock:
            if settings.auto_fire and down and not self.trigger_held and settings.trigger_mode == '切換':
                self.trigger_latched = not self.trigger_latched
            self.trigger_held = down

    def cancel_fire(self):
        with self._lock:
            self.trigger_held = self.trigger_latched = False
        self.mouse.set_fire_button(False)

    def _run(self):
        seen = -1
        plan = None

        timer = None
        measured_start, measured_count = time.perf_counter(), 0
        self.actual_hz = 0.0
        try:
            timer = ControlTimer(self._stop)
            if not timer.handle:
                self.log("高精度計時器不可用，使用一般等待；更新率可能降低")
            while timer.wait(self.get_settings().control_hz):
                now = time.perf_counter()
                measured_count += 1
                if now - measured_start >= 1:
                    self.actual_hz = measured_count / (now - measured_start)
                    measured_start, measured_count = now, 0
                with self._lock:
                    packet = self._packet
                settings = self.get_settings()
                missing_state = False
                if settings.require_entity_state and packet is not None and packet[1] is not None:
                    target_state = packet[1]
                    if not all(target_state.get(key) is True for key in ('alive','visible','enemy')):
                        packet = None
                        missing_state = True
                if settings.trigger_mode != "自動" and not self.trigger_started:
                    self.trigger.start()
                    self.trigger_started = True
                if hasattr(self.mouse, 'set_acceleration_disabled'):
                    self.mouse.set_acceleration_disabled(settings.auto_aim and settings.disable_acceleration)
                self.mouse.direct_position = settings.direct_position
                signature = (settings.auto_fire, settings.trigger_mode, settings.trigger_key)
                if signature != self._trigger_signature:
                    self.cancel_fire()
                    self._trigger_signature = signature
                with self._lock:
                    if settings.trigger_mode == '自動':
                        active_fire = (settings.auto_fire and packet is not None and packet[1] is not None
                            and packet[3] == settings and now - packet[2] <= .1
                            and packet[1]['dist'] <= settings.aim_tolerance
                            and (not settings.require_crosshair_in_box or packet[1].get('crosshair_inside') is True))
                    else:
                        active_fire = settings.auto_fire and (self.trigger_held if settings.trigger_mode == '按住' else self.trigger_latched)
                self.fire_status = "射擊中" if active_fire else "射擊關閉" if not settings.auto_fire else "等待目標／觸發"
                if not self.mouse.enabled:
                    self.mouse.enable()
                edge = self.scheduler.step(now, active_fire, settings)
                if edge and settings == self.get_settings() and not self._stop.is_set():
                    self.mouse.set_fire_button(edge == 'down')
                    if edge == 'down': self.shot_count += 1
                valid = (packet is not None and packet[1] is not None and
                         packet[3] == settings and now - packet[2] <= 0.1 and
                         (settings.auto_aim or settings.auto_fire))
                if not valid:
                    if not active_fire:
                        self.mouse.disable()
                    if missing_state:
                        self.status = "缺少遊戲存活／可視資料，暫停控制"
                    elif not (settings.auto_aim or settings.auto_fire):
                        self.status = "控制開關關閉"
                    elif packet is None:
                        self.status = "等待畫面"
                    elif packet[1] is None:
                        self.status = "沒有符合陣營／FOV／信心度的目標"
                    elif packet[3] != settings:
                        self.status = "設定已變更，等待新結果"
                    else:
                        self.status = "畫面超過 100 ms，等待新結果"
                    plan = None
                    # Discard the current frame; activation needs a fresh detection.
                    seen = packet[0] if packet else -1
                    continue
                if not self.mouse.enabled:
                    self.mouse.enable()
                    self.mouse.fire_mode = 'single'
                version, target, captured_at, _ = packet
                self.status = ("移動開關關閉" if not settings.auto_aim else
                               "目標已在穩定區" if target['dist'] <= settings.move_deadzone else
                               "貝茲移動／相對輸入" if not settings.direct_position else "貝茲移動／桌面游標")
                if version != seen:
                    seen = version
                    plan = None
                    if settings.auto_aim and target['dist'] > settings.move_deadzone:
                        dx = target['dx'] * settings.sensitivity
                        dy = target['dy'] * settings.sensitivity
                        distance = math.hypot(dx, dy)
                        # Bound endpoint in image-space to prevent gain-driven overshoot.
                        limit = max(0, target['dist'] - settings.move_deadzone) * settings.sensitivity
                        if distance > limit and distance:
                            dx, dy = dx * limit / distance, dy * limit / distance
                            distance = limit
                        # Derivative bounds of cubic Bezier constrain peak speed/acceleration.
                        p1, p2 = settings.bezier_start, settings.bezier_end
                        slope = 3 * max(p1, p2 - p1, 1 - p2)
                        curvature = 6 * max(abs(p2 - 2*p1), abs(1 - 2*p2 + p1))
                        strength = settings.curve_strength
                        curvature = curvature * strength**2 + slope * strength * (strength - 1)
                        slope *= strength
                        # Keep at least four nominal samples; speed does not collapse to a teleport.
                        duration = max((4 + 12 * settings.smoothness) / settings.control_hz,
                            0.060 * (1 + 3 * settings.smoothness) / settings.speed_multiplier,
                            slope * distance / settings.max_speed,
                            math.sqrt(curvature * distance / settings.max_acceleration))
                        plan = [now, duration, dx, dy, 0, 0]
                if plan is not None:
                    started, duration, dx, dy, sent_x, sent_y = plan
                    t = min(1, (now - started) / duration)
                    x, y = bezier_point(t, dx, dy, settings.bezier_start, settings.bezier_end, settings.curve_strength)
                    self.mouse.send_move(x - sent_x, y - sent_y)
                    plan[4:6] = [x, y]
                    if t >= 1:
                        plan = None
        except Exception as error:
            self.status = f"控制錯誤：{error}"
            self.log(f'控制已停止：{error}')
        finally:
            self.trigger.stop()
            self.mouse.disable()
            self.mouse.wait_stopped()
            try:
                if hasattr(self.mouse, "set_acceleration_disabled"):
                    self.mouse.set_acceleration_disabled(False)
            finally:
                if timer is not None:
                    timer.close()
