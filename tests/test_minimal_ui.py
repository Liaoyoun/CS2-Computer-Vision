"""Minimal UI and shutdown validation with mocked capture/model/input/hotkey."""
import sys,time,threading
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch
import unittest
import numpy as np
from ultralytics.engine.results import Results
from runtime.realtime import TargetSettings,LatestCapture
from yolo.detection import TargetSelector
from ui import main

class Hotkey:
    def __init__(self,callback): self.callback=callback
    def start(self):return self
    def stop(self):pass
class Model:
    def __init__(self,*a,**k):pass
    def predict(self,*a,**k):
        return [Results(np.zeros((640,640,3),dtype=np.uint8),'',
            {0:'THead'},boxes=np.array([[310,310,330,330,.9,0]],dtype=np.float32))]
class Camera:
    def __init__(self,*a):pass
    def isOpened(self):return True
    def set(self,*a):return False
    def get(self,*a):return 60
    def read(self):time.sleep(.005);return True,np.zeros((720,1280,3),dtype=np.uint8)
    def release(self):pass

class MinimalUITests(unittest.TestCase):
    def pump(self,app,seconds):
        end=time.perf_counter()+seconds
        while time.perf_counter()<end:app.update();time.sleep(.005)
    def test_start_toggle_L_and_settings(self):
        with patch.object(main,'LStopHotkey',Hotkey),patch.object(main,'load_settings',return_value=TargetSettings()),\
             patch.object(main,'save_settings') as saved,patch.object(main,'YOLO',Model),\
             patch.object(main,'LatestCapture',lambda index:LatestCapture(index,factory=Camera)),\
             patch('control.mouse_control.user32'):
            app=main.App();app.withdraw()
            try:
                self.assertEqual(len(app.winfo_children()),4) # three controls plus status
                app._set_setting('target_part','鎖頭+身體')
                self.assertEqual(app.settings.target_part,'鎖頭+身體')
                for close in ('button','L'):
                    app._toggle();self.pump(app,.15)
                    self.assertTrue(app.running)
                    if close=='L':
                        thread=threading.Thread(target=app.hotkey.callback);thread.start();thread.join()
                    else:app._toggle()
                    self.pump(app,.2)
                    self.assertFalse(app.running)
                    self.assertFalse(app.settings.auto_fire)
                    self.assertEqual(app.start_btn.cget('text'),'開啟')
                self.assertTrue(saved.called)
            finally:app._on_close()
    def test_part_filter_and_crosshair(self):
        r=Results(np.zeros((640,640,3),dtype=np.uint8),'',{0:'THead',1:'TBody'},
          boxes=np.array([[325,315,335,325,.9,0],[300,300,340,340,.9,1]],dtype=np.float32))
        selector=TargetSelector()
        head=selector.pick([r],'CT',180,'鎖頭')
        body=selector.pick([r],'CT',180,'鎖身體')
        self.assertEqual(head['name'],'THead');self.assertFalse(head['crosshair_inside'])
        self.assertEqual(body['name'],'TBody');self.assertTrue(body['crosshair_inside'])
        self.assertIsNone(selector.pick([r],'T',180,'鎖頭'))

    def test_combined_fallback_and_priority(self):
        def result(rows):
            return [Results(np.zeros((640,640,3),dtype=np.uint8),'',{0:'THead',1:'TBody'},
                boxes=np.array(rows,dtype=np.float32).reshape(-1,6))]
        selector=TargetSelector()
        body=[310,310,330,350,.95,1]
        close_head=[325,300,335,320,.8,0]
        far_head=[440,300,450,320,.9,0]
        both=result([body,close_head])
        self.assertEqual(selector.pick(both,'CT',180,'鎖頭+身體')['name'],'THead')
        self.assertEqual(selector.pick(result([body]),'CT',180,'鎖頭+身體')['name'],'TBody')
        self.assertEqual(selector.pick(result([body,far_head]),'CT',180,'鎖頭+身體')['name'],'TBody')
        self.assertEqual(selector.pick(both,'CT',180,'鎖頭+身體','身體優先')['name'],'TBody')
        self.assertEqual(selector.pick(both,'CT',180,'鎖頭+身體','綜合評分')['name'],'TBody')
        self.assertIsNone(selector.pick(result([far_head]),'CT',180,'鎖頭+身體'))
        self.assertIsNone(selector.pick(both,'T',180,'鎖頭+身體'))
        self.assertIsNone(selector.pick(result([]),'CT',180,'鎖頭+身體'))

    def test_combined_settings_roundtrip(self):
        from ui.control_settings import load_settings,save_settings
        from dataclasses import replace
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            path=Path(directory)/'settings.json'
            settings=replace(TargetSettings(),target_part='鎖頭+身體',combined_priority='身體優先')
            save_settings(settings,path)
            self.assertEqual(load_settings(path),settings)

if __name__=='__main__':unittest.main(verbosity=2)
