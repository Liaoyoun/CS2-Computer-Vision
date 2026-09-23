"""Control tests use a fake mouse; no OS input is generated."""
from pathlib import Path
import sys
import time
import unittest
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.realtime import TargetSettings
from control.target_control import TargetControl, bezier_point


class Mouse:
    enabled = False
    busy = False
    def __init__(self): self.moves, self.clicks = [], 0
    def enable(self): self.enabled = True
    def disable(self): self.enabled = False
    def wait_stopped(self): pass
    def send_move(self, x, y):
        if self.enabled: self.moves.append((x,y))
    def set_fire_button(self, down):
        if down: self.clicks += 1
    def send_click(self):
        self.clicks += 1
        return True


class FakeTrigger:
    def __init__(self,*args): pass
    def start(self): return self
    def stop(self): pass

class ControlTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.trigger_patch=patch('control.target_control.PhysicalTrigger',FakeTrigger)
        self.trigger_patch.start()
        self.addCleanup(self.trigger_patch.stop)

    def test_curve_endpoints_and_monotonicity(self):
        pts = [bezier_point(i/100, 100, -50) for i in range(101)]
        self.assertEqual(pts[0], (0,0))
        self.assertEqual(pts[-1], (100,-50))
        self.assertEqual(sorted(p[0] for p in pts), [p[0] for p in pts])

    def test_motion_fire_stale_and_release(self):
        settings = replace(TargetSettings(), auto_aim=True, auto_fire=True, sensitivity=1)
        mouse = Mouse()
        errors = []
        control = TargetControl(lambda: settings, errors.append, mouse).start()
        try:
            control.submit(dict(dx=60,dy=0,dist=60), time.perf_counter(), settings)
            time.sleep(.065)
            self.assertEqual(sum(x for x,y in mouse.moves), 59)
            self.assertEqual(mouse.clicks, 0)
            control.submit(dict(dx=0,dy=0,dist=0), time.perf_counter(), settings)
            time.sleep(.025)
            self.assertEqual(mouse.clicks, 1)
            control.submit(dict(dx=80,dy=0,dist=80), time.perf_counter()-1, settings)
            before = len(mouse.moves)
            time.sleep(.025)
            self.assertEqual(len(mouse.moves), before)
            self.assertFalse(mouse.enabled)
            settings = replace(settings, auto_aim=False, auto_fire=False)
            control.submit(dict(dx=80,dy=0,dist=80), time.perf_counter(), settings)
            time.sleep(.025)
            self.assertEqual(len(mouse.moves), before)
        finally:
            control.stop()
        self.assertFalse(control.thread.is_alive())
        self.assertFalse(mouse.enabled)
        self.assertFalse(errors)

    def test_new_target_cancels_previous_path(self):
        settings = replace(TargetSettings(), auto_aim=True, sensitivity=1, max_speed=3000, max_acceleration=100000)
        mouse = Mouse()
        control = TargetControl(lambda: settings, lambda e: None, mouse).start()
        try:
            control.submit(dict(dx=240,dy=0,dist=240), time.perf_counter(), settings)
            time.sleep(.012)
            control.submit(None, time.perf_counter(), settings)
            time.sleep(.01)
            count = len(mouse.moves)
            time.sleep(.05)
            self.assertEqual(len(mouse.moves), count)
            self.assertLess(sum(x for x,y in mouse.moves), 240)
        finally:
            control.stop()

    def test_persistent_toggle_and_reactivation(self):
        settings = replace(TargetSettings(), auto_aim=True, sensitivity=1)
        mouse = Mouse()
        control = TargetControl(lambda: settings, lambda e: None, mouse).start()
        try:
            for _ in range(2):
                control.submit(dict(dx=40,dy=0,dist=40), time.perf_counter(), settings)
                time.sleep(.06)
            self.assertEqual(sum(x for x,y in mouse.moves), 78)
            settings = replace(settings, auto_aim=False)
            time.sleep(.01)
            before = len(mouse.moves)
            control.submit(dict(dx=40,dy=0,dist=40), time.perf_counter(), settings)
            time.sleep(.03)
            self.assertEqual(len(mouse.moves), before)
            settings = replace(settings, auto_aim=True)
            control.submit(dict(dx=40,dy=0,dist=40), time.perf_counter(), settings)
            time.sleep(.06)
            self.assertEqual(sum(x for x,y in mouse.moves), 117)
        finally:
            control.stop()

    def test_persistence_and_invalid_settings(self):
        from tempfile import TemporaryDirectory
        from ui.control_settings import save_settings, load_settings
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            settings = replace(TargetSettings(), auto_aim=True, speed_multiplier=3)
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)
            path.write_text('{"speed_multiplier": -1}')
            with self.assertRaises(ValueError): load_settings(path)

    def test_fast_start_and_no_endpoint_overshoot(self):
        self.assertGreater(bezier_point(.1,1000,0)[0], 28)
        for i in range(101):
            self.assertTrue(0 <= bezier_point(i/100,1000,0)[0] <= 1000)

    def test_esports_curve_and_preset(self):
        from ui.control_settings import esports_preset
        settings = esports_preset(replace(TargetSettings(), auto_aim=True))
        self.assertTrue(settings.auto_aim)
        self.assertEqual(settings.speed_multiplier, 20)
        self.assertGreater(bezier_point(.2, 1000, 0, settings.bezier_start,
                           settings.bezier_end, settings.curve_strength)[0], 700)
        points = [bezier_point(i/1000,1000,0,1,1,2)[0] for i in range(1001)]
        self.assertEqual(points, sorted(points))
        self.assertEqual(points[-1], 1000)

    def test_acceleration_restore_and_direct_position(self):
        from unittest.mock import patch
        from control.mouse_control import MouseController
        with patch('control.mouse_control.user32') as api:
            def spi(action, unused, values, flags):
                if action == 3:
                    values[0], values[1], values[2] = 6,10,1
                return 1
            api.SystemParametersInfoW.side_effect = spi
            mouse = MouseController()
            mouse.set_acceleration_disabled(True)
            mouse.set_acceleration_disabled(False)
            restored = api.SystemParametersInfoW.call_args.args[2]
            self.assertEqual(tuple(restored), (6,10,1))
            mouse.enable()
            mouse.direct_position = True
            mouse.send_move(5,-3)
            api.SetCursorPos.assert_called_once_with(5,-3)
            api.mouse_event.assert_not_called()
            mouse.disable()

    def test_relative_move_is_not_blocked_by_click_worker(self):
        from unittest.mock import patch
        from control.mouse_control import MouseController
        with patch('control.mouse_control.user32') as api:
            mouse = MouseController()
            mouse.enable()
            with patch.object(MouseController, 'busy', property(lambda _: True)):
                mouse.send_move(7,-4)
            api.mouse_event.assert_called_with(1,7,-4,0,0)
            api.SetCursorPos.assert_not_called()
            self.assertEqual(mouse.move_events,1)
            mouse.disable()

    def test_controller_moves_while_click_worker_busy(self):
        settings = replace(TargetSettings(), auto_aim=True, sensitivity=1)
        mouse = Mouse(); mouse.busy = True
        errors = []
        controller = TargetControl(lambda: settings, errors.append, mouse).start()
        try:
            controller.submit(dict(dx=60,dy=0,dist=60),time.perf_counter(),settings)
            time.sleep(.065)
            self.assertEqual(sum(x for x,y in mouse.moves),59)
            self.assertFalse(mouse.direct_position)
            self.assertFalse(errors)
        finally:
            controller.stop()

    def test_automatic_fire_without_any_trigger_and_stale_release(self):
        class RecordingMouse(Mouse):
            def __init__(self):
                super().__init__();self.down=False
            def set_fire_button(self,down):
                self.down=down
                if down:self.clicks+=1
            def disable(self):
                self.enabled=False;self.down=False
        settings=replace(TargetSettings(),auto_fire=True,auto_aim=False,shooting_mode='連發',trigger_mode='自動')
        mouse=RecordingMouse();errors=[]
        control=TargetControl(lambda:settings,errors.append,mouse).start()
        try:
            control.submit(dict(dx=0,dy=0,dist=0),time.perf_counter(),settings)
            time.sleep(.025)
            self.assertTrue(mouse.down)
            self.assertEqual(mouse.clicks,1)
            self.assertFalse(control.trigger_started)
            control.submit(None,time.perf_counter(),settings)
            time.sleep(.025)
            self.assertFalse(mouse.down)
            self.assertFalse(errors)
        finally:control.stop()

    def test_crosshair_must_be_inside_selected_part_to_fire(self):
        settings=replace(TargetSettings(),auto_fire=True,require_crosshair_in_box=True)
        mouse=Mouse();control=TargetControl(lambda:settings,lambda e:None,mouse).start()
        try:
            control.submit(dict(dx=0,dy=0,dist=0,crosshair_inside=False),time.perf_counter(),settings)
            time.sleep(.025)
            self.assertEqual(mouse.clicks,0)
            control.submit(dict(dx=0,dy=0,dist=0,crosshair_inside=True),time.perf_counter(),settings)
            time.sleep(.025)
            self.assertEqual(mouse.clicks,1)
        finally:control.stop()

if __name__ == '__main__': unittest.main(verbosity=2)
