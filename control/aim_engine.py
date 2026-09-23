import math
import random
import time


# 貝茲曲線

def cubic_bezier_point(t, p0, p1, p2, p3):

    x = (
        (1 - t) ** 3 * p0[0]
        + 3 * (1 - t) ** 2 * t * p1[0]
        + 3 * (1 - t) * t ** 2 * p2[0]
        + t ** 3 * p3[0]
    )

    y = (
        (1 - t) ** 3 * p0[1]
        + 3 * (1 - t) ** 2 * t * p1[1]
        + 3 * (1 - t) * t ** 2 * p2[1]
        + t ** 3 * p3[1]
    )

    return int(x), int(y)


# 費茲速度曲線

def fitts_velocity_curve(steps):

    if steps <= 1:
        return [1.0]

    time_fractions = [
        math.sin(
            (i / (steps - 1))
            * math.pi
            / 2
        )
        for i in range(steps)
    ]

    intervals = [
        time_fractions[i + 1]
        - time_fractions[i]
        for i in range(
            len(time_fractions) - 1
        )
    ]

    intervals = [
        value * random.uniform(
            0.92,
            1.08
        )
        for value in intervals
    ]

    total = sum(intervals) or 1.0

    return [
        value / total
        for value in intervals
    ]


# 相對位移貝茲曲線

def fps_bezier_move(
    dx,
    dy,
    mouse,
    min_duration=0.025,
    max_duration=0.085
):

    dx = float(dx)
    dy = float(dy)

    distance = math.hypot(
        dx,
        dy
    )

    if distance < 1:
        return

    if distance < 4:

        mouse.send_move(
            dx,
            dy
        )

        return

    steps = int(
        max(
            4,
            min(
                distance / 18,
                9
            )
        )
    )

    start = (
        0.0,
        0.0
    )

    end = (
        dx,
        dy
    )

    if distance < 30:

        deviation = (
            distance * 0.02
        )

    elif distance < 100:

        deviation = (
            distance * 0.04
        )

    else:

        deviation = (
            distance * 0.06
        )

    p1 = (
        dx / 3
        + random.uniform(
            -deviation,
            deviation
        ),
        dy / 3
        + random.uniform(
            -deviation,
            deviation
        )
    )

    p2 = (
        dx * 2 / 3
        + random.uniform(
            -deviation,
            deviation
        ),
        dy * 2 / 3
        + random.uniform(
            -deviation,
            deviation
        )
    )

    duration = (
        min_duration
        + (distance / 600)
        * (
            max_duration
            - min_duration
        )
    )

    duration = max(
        min_duration,
        min(
            duration,
            max_duration
        )
    )

    sleep_weights = (
        fitts_velocity_curve(
            steps
        )
    )

    sent_x = 0
    sent_y = 0

    for i in range(steps):

        t = i / (
            steps - 1
        )

        cx, cy = (
            cubic_bezier_point(
                t,
                start,
                p1,
                p2,
                end
            )
        )

        target_x = int(
            round(cx)
        )

        target_y = int(
            round(cy)
        )

        step_dx = (
            target_x - sent_x
        )

        step_dy = (
            target_y - sent_y
        )

        if (
            step_dx != 0
            or step_dy != 0
        ):

            mouse.send_move(
                step_dx,
                step_dy
            )

            sent_x = target_x
            sent_y = target_y

        if i < len(
            sleep_weights
        ):

            time.sleep(
                sleep_weights[i]
                * duration
            )


# 取得中央 640x640

def get_area(frame):

    h, w = frame.shape[:2]

    if (
        h < 640
        or w < 640
    ):

        raise ValueError(
            f"畫面尺寸不足 640x640：{w}x{h}"
        )

    cx = w // 2
    cy = h // 2

    return frame[
        cy - 320:cy + 320,
        cx - 320:cx + 320
    ]


# 檢查是否對準

def is_on_target(
    dx,
    dy,
    tolerance=15
):

    return (
        math.hypot(
            dx,
            dy
        )
        <= tolerance
    )


# 目標選擇

class TargetSelector:
    """Head first, then nearest; a fired track ID is retired for this session."""
    ENEMIES = {
        "CT": {"THead", "TBody"},
        "T": {"CTHead", "CTBody"},
        "死鬥": {"CTHead", "CTBody", "THead", "TBody"},
    }

    def __init__(self):
        self.clicked = set()
        self.locked_id = None
        self.team = None

    def pick(self, results, my_team, fov_limit, current_time=None):
        # Team changes invalidate the lock, but must not forget fired IDs.
        if my_team != self.team:
            self.locked_id = None
            self.team = my_team
        enemies = self.ENEMIES.get(my_team, set())
        if not results or not enemies:
            self.locked_id = None
            return None
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            self.locked_id = None
            return None
        # One CPU transfer for the entire result; no per-box GPU .item() calls.
        rows = boxes.cpu().numpy().data
        height, width = result.orig_shape
        cx, cy = width / 2, height / 2
        candidates = {}
        for x1, y1, x2, y2, track_id, confidence, class_id in rows:
            name = result.names[int(class_id)]
            obj_id = int(track_id)
            if name not in enemies or obj_id in self.clicked:
                continue
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            dx, dy = center[0] - cx, center[1] - cy
            distance = math.hypot(dx, dy)
            if distance > fov_limit:
                continue
            target = {
                "id": obj_id, "name": name,
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "center": center, "dx": float(dx), "dy": float(dy),
                "dist": distance, "conf": float(confidence),
            }
            old = candidates.get(obj_id)
            if old is None or self._rank(target) < self._rank(old):
                candidates[obj_id] = target
        if not candidates:
            self.locked_id = None
            return None
        best = min(candidates.values(), key=self._rank)
        locked = candidates.get(self.locked_id)
        # Keep a comparable target to avoid oscillating between nearby boxes.
        if (locked is not None
                and locked["name"].endswith("Head") == best["name"].endswith("Head")
                and locked["dist"] <= best["dist"] + 20):
            best = locked
        self.locked_id = best["id"]
        return best

    @staticmethod
    def _rank(target):
        return (not target["name"].endswith("Head"), target["dist"], -target["conf"])

    def mark_clicked(self, obj_id):
        if obj_id is not None:
            self.clicked.add(obj_id)
            if self.locked_id == obj_id:
                self.locked_id = None


class ReactionGate:
    """Wait across fresh detections, without sleeping or retaining stale coordinates."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.deadline = 0.0

    def ready(self, target, settings, now):
        if target is None or target["dist"] > settings.aim_tolerance:
            self.reset()
            return False
        key = (target["id"], settings)
        if key != self.key:
            self.key = key
            self.deadline = now + random.uniform(settings.reaction_min, settings.reaction_max)
        if now < self.deadline:
            return False
        # One probability decision per reaction interval, independent of FPS.
        self.reset()
        return random.random() < settings.fire_prob

