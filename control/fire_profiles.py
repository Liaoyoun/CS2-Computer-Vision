"""Per-weapon input timing profiles; these are user settings, not weapon limits."""
import json
import math
from pathlib import Path
from dataclasses import asdict

PATH=Path(__file__).resolve().parents[1]/'fire_profiles.json'
FIELDS=('shooting_mode','fire_cps','burst_count','burst_gap_ms','fire_cooldown_ms','fire_jitter_ms')

def load_profiles():
    if not PATH.exists(): return {}
    raw=json.loads(PATH.read_text(encoding='utf-8'))
    if not isinstance(raw,dict):raise ValueError('射速設定檔格式無效')
    from ui.control_settings import RANGES
    for name, values in raw.items():
        if not isinstance(name,str) or not name.strip() or not isinstance(values,dict):
            raise ValueError('武器射速設定無效')
        for key,value in values.items():
            if key not in FIELDS:raise ValueError('未知射速參數')
            if key=='shooting_mode':
                if value not in ('單發','連按','連發','爆發'):raise ValueError('射擊模式無效')
            else:
                low,high=RANGES[key]
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
                    raise ValueError('射速參數超出範圍')
    return raw

def save_profile(name,settings):
    if not name.strip():raise ValueError('請輸入武器名稱')
    profiles=load_profiles()
    values=asdict(settings)
    profiles[name]={key:values[key] for key in FIELDS}
    temporary=PATH.with_suffix('.tmp')
    temporary.write_text(json.dumps(profiles,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(PATH)
    return profiles
