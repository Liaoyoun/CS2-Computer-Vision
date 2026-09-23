"""Detection-only geometry. No tracking or input-control dependencies."""
import math


def get_area(frame):
    h, w = frame.shape[:2]
    if h < 640 or w < 640:
        raise ValueError(f"畫面尺寸不足 640x640：{w}x{h}")
    return frame[h // 2 - 320:h // 2 + 320, w // 2 - 320:w // 2 + 320]


class TargetSelector:
    ENEMIES = {
        "CT": {"THead", "TBody"},
        "T": {"CTHead", "CTBody"},
        "死鬥": {"CTHead", "CTBody", "THead", "TBody"},
    }

    def pick(self, results, my_team, fov_limit, target_part=None, combined_priority="頭部優先", head_fallback_distance=90.0):
        if not results or results[0].boxes is None:
            return None
        result = results[0]
        height, width = result.orig_shape
        candidates = []
        for index, row in enumerate(result.boxes.cpu().numpy().data):
            x1, y1, x2, y2 = row[:4]
            confidence, class_id = row[-2:]
            name = result.names[int(class_id)]
            if name not in self.ENEMIES.get(my_team, set()):
                continue
            if target_part in ("鎖頭", "鎖身體") and not name.endswith("Head" if target_part == "鎖頭" else "Body"):
                continue
            x, y = float((x1 + x2) / 2), float((y1 + y2) / 2)
            dx, dy = x - width / 2, y - height / 2
            distance = math.hypot(dx, dy)
            if target_part == "鎖頭+身體" and name.endswith("Head") and distance > head_fallback_distance:
                continue
            if distance <= fov_limit:
                candidates.append(dict(index=index, name=name, center=(x, y),
                    box=tuple(float(v) for v in row[:4]), dx=dx, dy=dy,
                    crosshair_inside=bool(x1 <= width/2 <= x2 and y1 <= height/2 <= y2),
                    dist=distance, conf=float(confidence)))
        if target_part == "鎖頭+身體":
            if combined_priority == "綜合評分":
                # Detection confidence and screen distance are proxies, not hit probability.
                return min(candidates, key=lambda t: (
                    t['dist'] / max(float(fov_limit),1) + 1 - t['conf'],
                    t['dist'], t['index']), default=None)
            preferred = 'Body' if combined_priority == '身體優先' else 'Head'
            return min(candidates, key=lambda t: (not t['name'].endswith(preferred),
                       t['dist'], -t['conf']), default=None)
        return min(candidates, key=lambda t: (not t['name'].endswith('Head'),
                   t['dist'], -t['conf']), default=None)
