"""Validated, atomic UI settings persistence."""
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from runtime.realtime import TargetSettings

SETTINGS_PATH = Path(__file__).resolve().parents[1] / 'control_settings.json'
RANGES = {
 'head_fallback_distance': (0,320),
 'fov_limit': (30,320), 'conf_threshold': (.1,.9), 'aim_tolerance': (2,30),
 'sensitivity': (.1,10), 'speed_multiplier': (.5,100), 'max_speed': (1000,10000000),
 'max_acceleration': (100000,1000000000000), 'bezier_start': (.1,1),
 'bezier_end': (.85,1), 'move_deadzone': (0,5), 'fire_prob': (0,1),
 'smoothness': (0,1), 'curve_strength': (1,4), 'control_hz': (1000,4000),
 'fire_cps': (1,100), 'burst_count': (1,30), 'burst_gap_ms': (0,2000),
 'fire_cooldown_ms': (0,5000), 'fire_jitter_ms': (0,20),
 'reaction_min': (0,2), 'reaction_max': (0,2),
}

def load_settings(path=None):
    path = Path(path) if path is not None else SETTINGS_PATH
    if not path.exists(): return TargetSettings()
    raw = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict): raise ValueError('設定必須為 JSON 物件')
    defaults = asdict(TargetSettings())
    for name in defaults:
        if name not in raw: continue
        value = raw[name]
        if name in ('auto_aim','auto_fire','require_entity_state','require_crosshair_in_box','disable_acceleration','direct_position'):
            if type(value) is not bool: raise ValueError(f'{name} 必須為布林值')
        elif name in ('shooting_mode','trigger_mode','trigger_key'):
            allowed = {'shooting_mode': ('單發','連按','連發','爆發'),
                       'trigger_mode': ('自動','按住','切換'),
                       'trigger_key': ('左鍵','F6','F7','F8','F9','F10','F11','F12')}
            if value not in allowed[name]: raise ValueError(f'{name} 無效')
        elif name == 'fire_weapon':
            if not isinstance(value,str) or not value.strip(): raise ValueError('武器名稱無效')
        elif name == 'target_part':
            if value not in ('鎖頭','鎖身體','鎖頭+身體'): raise ValueError('目標部位無效')
        elif name == 'combined_priority':
            if value not in ('頭部優先','身體優先','綜合評分'): raise ValueError('部位優先級無效')
        elif name == 'my_team':
            if value not in ('CT','T','死鬥'): raise ValueError('陣營設定無效')
        else:
            lo, hi = RANGES[name]
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{name} 超出允許範圍')
        defaults[name] = value
    return TargetSettings(**defaults)

def save_settings(settings, path=None):
    path = Path(path) if path is not None else SETTINGS_PATH
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def esports_preset(settings):
    return replace(settings, speed_multiplier=20.0, sensitivity=1.35,
        max_speed=1000000.0, max_acceleration=10000000000.0,
        bezier_start=1.0, bezier_end=1.0, smoothness=0.0,
        curve_strength=2.0, control_hz=2000.0, disable_acceleration=True, direct_position=False,
        move_deadzone=1.0, fire_cps=20.0, fire_jitter_ms=0.0)
