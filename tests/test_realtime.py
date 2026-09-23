"""Regression checks with synthetic detections and mocked mouse/camera IO."""
from dataclasses import replace
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ultralytics.engine.results import Results
from control.aim_engine import TargetSelector, ReactionGate
from runtime.realtime import LatestCapture, LatestValue, TargetSettings
from control.mouse_control import MouseController

NAMES = {0: 'CTBody', 1: 'CTHead', 2: 'TBody', 3: 'THead'}


def result(*rows):
    return [Results(np.zeros((640, 640, 3), dtype=np.uint8), '', NAMES,
                    boxes=np.array(rows, dtype=np.float32).reshape(-1, 7))]


def box(track_id, cls, x=320, y=320):
    return [x-5, y-5, x+5, y+5, track_id, 0.9, cls]


class RealtimeTests(unittest.TestCase):
    def test_team_filters_and_deathmatch(self):
        detections = result(box(1, 1, 325), box(2, 3, 340))
        for team, expected in [('CT', 2), ('T', 1), ('死鬥', 1)]:
            self.assertEqual(TargetSelector().pick(detections, team, 180)['id'], expected)
        self.assertIsNone(TargetSelector().pick(detections, 'invalid', 180))

    def test_clicked_id_never_repeats_across_team_or_time(self):
        selector = TargetSelector()
        selector.mark_clicked(1)
        detections = result(box(1, 1), box(2, 3))
        self.assertEqual(selector.pick(detections, '死鬥', 180, 1e12)['id'], 2)
        self.assertIsNone(selector.pick(detections, 'T', 180, 1e15))

    def test_fov_and_head_priority_and_deduplication(self):
        selector = TargetSelector()
        self.assertEqual(selector.pick(result(box(1, 2), box(2, 3, 400)), 'CT', 180)['id'], 2)
        self.assertIsNone(selector.pick(result(box(3, 3, 600)), 'CT', 180))
        target = selector.pick(result(box(4, 2), box(4, 3, 330)), 'CT', 180)
        self.assertEqual(target['name'], 'THead')
        self.assertIsNone(selector.pick(result(), 'CT', 180))

    def test_lock_hysteresis_and_no_id(self):
        selector = TargetSelector()
        selector.pick(result(box(1, 3, 350)), 'CT', 180)
        self.assertEqual(selector.pick(result(box(1, 3, 350), box(2, 3, 345)), 'CT', 180)['id'], 1)
        self.assertEqual(selector.pick(result(box(1, 3, 380), box(2, 3, 330)), 'CT', 180)['id'], 2)
        untracked = Results(np.zeros((640, 640, 3), dtype=np.uint8), '', NAMES,
                            boxes=np.array([[310, 310, 330, 330, .9, 3]], dtype=np.float32))
        self.assertIsNone(selector.pick([untracked], 'CT', 180))

    def test_reaction_freshness_and_parameter_change(self):
        settings = replace(TargetSettings(), reaction_min=.1, reaction_max=.1, fire_prob=1)
        gate = ReactionGate()
        target = {'id': 1, 'dist': 5}
        self.assertFalse(gate.ready(target, settings, 0))
        self.assertFalse(gate.ready(target, settings, .05))
        self.assertTrue(gate.ready(target, settings, .11))
        gate.ready(target, settings, 1)
        self.assertFalse(gate.ready(None, settings, 1.1))
        self.assertFalse(gate.ready(target, settings, 1.2))
        self.assertFalse(gate.ready(target, replace(settings, my_team='死鬥'), 1.4))
        self.assertFalse(gate.ready({'id': 2, 'dist': 11}, settings, 2))

    def test_raw_prediction_team_selection(self):
        from yolo.detection import TargetSelector as RawSelector
        raw = Results(np.zeros((640, 640, 3), dtype=np.uint8), '', NAMES,
                      boxes=np.array([[315,315,325,325,.9,1],
                                      [335,315,345,325,.8,3]], dtype=np.float32))
        for team, expected in [('CT', 'THead'), ('T', 'CTHead'), ('死鬥', 'CTHead')]:
            self.assertEqual(RawSelector().pick([raw], team, 180)['name'], expected)
        self.assertIsNone(RawSelector().pick([raw], 'CT', 10))
        self.assertIsNone(RawSelector().pick([], '死鬥', 180))

    def test_mailbox_overwrites(self):
        mailbox = LatestValue()
        for value in range(100):
            mailbox.put(value)
        self.assertEqual(mailbox.take(), 99)
        self.assertIsNone(mailbox.take())

    def test_capture_latest_sequence_and_shutdown(self):
        class Camera:
            def __init__(self, *args):
                self.released = False
            def isOpened(self): return True
            def set(self, *args): return False
            def get(self, *args): return 120
            def read(self):
                time.sleep(.002)
                return True, np.zeros((2, 2, 3), dtype=np.uint8)
            def release(self): self.released = True
        camera = Camera()
        capture = LatestCapture(0, factory=lambda *args: camera).start()
        first = capture.read_latest(0, timeout=1)
        time.sleep(.03)
        second = capture.read_latest(first[0])
        self.assertGreater(second[0], first[0] + 1)
        self.assertTrue(capture.stop())
        self.assertTrue(camera.released)

    def test_capture_failure(self):
        class FailedCamera:
            def isOpened(self): return False
            def release(self): pass
        capture = LatestCapture(0, factory=lambda *args: FailedCamera()).start()
        self.assertIsNone(capture.read_latest(0, timeout=1))
        self.assertIsNotNone(capture.error)
        self.assertTrue(capture.stop())

    def test_click_is_nonblocking_and_stop_releases(self):
        events, pressed = [], threading.Event()
        def mouse_event(event, *args):
            events.append(event)
            if event == 2:
                pressed.set()
        with patch('control.mouse_control.user32') as api:
            api.mouse_event.side_effect = mouse_event
            mouse = MouseController()
            mouse.fire_mode = 'single'
            mouse.single_hold = 1
            mouse.enable()
            started = time.perf_counter()
            self.assertTrue(mouse.send_click())
            self.assertLess(time.perf_counter() - started, .1)
            self.assertTrue(pressed.wait(1))
            self.assertFalse(mouse.send_click())
            mouse.disable()
            mouse.wait_stopped()
            self.assertEqual(events, [2, 4])
            self.assertFalse(mouse.send_click())
            mouse.enable()
            mouse.disable()


# Minimal GUI lifecycle coverage now lives in test_minimal_ui.py.

if __name__ == '__main__':
    unittest.main(verbosity=2)
