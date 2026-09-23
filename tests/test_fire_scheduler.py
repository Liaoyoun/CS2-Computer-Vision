"""Deterministic scheduler and mock-input integration tests; never sends OS input."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import unittest
from runtime.realtime import TargetSettings
from control.fire_scheduler import FireScheduler

class FireTests(unittest.TestCase):
    def simulate(self,mode,cps=20,seconds=1,hz=2000,**kw):
        settings=replace(TargetSettings(),shooting_mode=mode,fire_cps=cps,**kw)
        scheduler=FireScheduler();events=[]
        for i in range(int(seconds*hz)):
            edge=scheduler.step(i/hz,True,settings)
            if edge:events.append((i/hz,edge))
        return scheduler,events,settings
    def test_repeat_cadence(self):
        for cps in (20,60,100):
            for hz in (500,2000):
                _,events,_=self.simulate('連按',cps,hz=hz)
                downs=[t for t,e in events if e=='down']
                self.assertEqual(len(downs),cps)
                self.assertLess(max(abs((b-a)-1/cps) for a,b in zip(downs,downs[1:])),1/hz+.00001)
    def test_single_and_hold_release(self):
        for mode in ('單發','連發'):
            scheduler,events,settings=self.simulate(mode)
            self.assertEqual(sum(e=='down' for t,e in events),1)
            edge=scheduler.step(1,False,settings)
            self.assertEqual(edge,'up' if mode=='連發' else None)
    def test_burst_and_cooldown(self):
        _,events,_=self.simulate('爆發',seconds=.5,burst_count=3,burst_gap_ms=100,fire_cooldown_ms=100)
        times=[t for t,e in events if e=='down']
        self.assertEqual(len(times),6)
        self.assertGreaterEqual(times[3]-times[2],.2)
    def test_no_catchup_after_stall(self):
        scheduler=FireScheduler();s=TargetSettings()
        self.assertEqual(scheduler.step(0,True,s),'down')
        self.assertEqual(scheduler.step(1,True,s),'up')
        self.assertEqual(scheduler.step(1.001,True,s),'down')
        self.assertIsNone(scheduler.step(1.001,True,s))
    def test_off_releases_immediately(self):
        scheduler=FireScheduler();s=TargetSettings()
        scheduler.step(0,True,s)
        self.assertEqual(scheduler.step(.001,False,s),'up')
        self.assertIsNone(scheduler.step(.002,False,s))
    def test_saved_modes_and_profiles(self):
        import tempfile
        from unittest.mock import patch
        from ui.control_settings import save_settings,load_settings
        from control import fire_profiles
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json'
            s=replace(TargetSettings(),shooting_mode='爆發',trigger_mode='自動',fire_cps=60)
            save_settings(s,path);self.assertEqual(load_settings(path),s)
            with patch.object(fire_profiles,'PATH',Path(directory)/'profiles.json'):
                fire_profiles.save_profile('test',s)
                self.assertEqual(fire_profiles.load_profiles()['test']['fire_cps'],60)

if __name__=='__main__': unittest.main(verbosity=2)
