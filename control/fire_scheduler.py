"""Frame-independent firing state machine. step() returns at most one button edge."""
import random

class FireScheduler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.down = False
        self.active = False
        self.due = 0.0
        self.release_at = 0.0
        self.shots = 0
        self.completed = False
        self.signature = None

    def step(self, now, active, settings):
        signature = (settings.shooting_mode, settings.fire_cps, settings.burst_count,
                     settings.burst_gap_ms, settings.fire_cooldown_ms, settings.fire_jitter_ms)
        if not active or signature != self.signature:
            release = self.down
            self.reset()
            self.signature = signature
            if release: return 'up'
            if not active: return None
        if not self.active:
            self.active = True
            self.due = now
        if settings.shooting_mode == '連發':
            if not self.down:
                self.down = True
                return 'down'
            return None
        if self.down:
            if now >= self.release_at:
                self.down = False
                return 'up'
            return None
        if self.completed or now < self.due: return None
        period = 1 / settings.fire_cps
        jitter = random.uniform(-settings.fire_jitter_ms, settings.fire_jitter_ms) / 1000
        self.down = True
        self.shots += 1
        self.release_at = now + min(.004, period * .4)
        # Anchor to the previous deadline, but never catch up with a burst after a stall.
        self.due = max(self.due + max(.006, period + jitter), self.release_at + .001)
        if self.due < now: self.due = now + period
        if settings.shooting_mode == '單發':
            self.completed = True
        elif settings.shooting_mode == '爆發' and self.shots >= settings.burst_count:
            self.shots = 0
            self.due = max(self.due, self.release_at +
                           (settings.burst_gap_ms + settings.fire_cooldown_ms) / 1000)
        return 'down'
