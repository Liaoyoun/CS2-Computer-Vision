import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading
import queue
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import customtkinter as ctk
from ultralytics import YOLO
from control.target_control import TargetControl
from ui.control_settings import load_settings, save_settings, esports_preset
from yolo.detection import get_area, TargetSelector
from runtime.realtime import LatestCapture, LatestValue, TargetSettings
from runtime.shutdown_hotkey import LStopHotkey

ctk.set_appearance_mode("dark")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CS2 控制")
        self.geometry("430x330")
        self.resizable(False,False)
        try:
            loaded=load_settings()
        except (OSError,ValueError):
            loaded=TargetSettings()
        self.settings=replace(esports_preset(loaded),fire_cps=loaded.fire_cps,fire_jitter_ms=loaded.fire_jitter_ms,auto_aim=False,auto_fire=False,
            trigger_mode="自動",require_entity_state=False,require_crosshair_in_box=True)
        self.model_path=str(Path(__file__).resolve().parents[1]/'best'/'best.engine')
        self.camera_index=2
        self.running=False
        self._closing=False
        self._last_error=''
        self.worker=None
        self.controller=None
        self.stop_event=threading.Event()
        self.worker_done=threading.Event()
        self.shutdown_requested=threading.Event()
        self.stats_queue=LatestValue()
        self.render_queue=LatestValue()
        self.preview_enabled=False
        self.log_queue=queue.Queue(maxsize=200)
        self.start_btn=ctk.CTkButton(self,text="開啟",height=64,command=self._toggle,
            font=ctk.CTkFont(size=24,weight="bold"))
        self.start_btn.pack(fill="x",padx=24,pady=(24,16))
        self.team_seg=ctk.CTkSegmentedButton(self,values=["CT","T","死鬥"],height=42,
            command=lambda v:self._set_setting('my_team',v))
        self.team_seg.set(self.settings.my_team)
        self.team_seg.pack(fill="x",padx=24,pady=8)
        self.part_seg=ctk.CTkSegmentedButton(self,values=["鎖頭","鎖身體","鎖頭+身體"],height=42,
            command=lambda v:self._set_setting('target_part',v))
        self.part_seg.set(self.settings.target_part)
        self.part_seg.pack(fill="x",padx=24,pady=8)
        self.status_label=ctk.CTkLabel(self,text="已關閉 · L 關閉",wraplength=380)
        self.status_label.pack(padx=20,pady=12)
        self.hotkey=LStopHotkey(self._request_shutdown)
        try:self.hotkey.start()
        except RuntimeError as error:
            self.status_label.configure(text=str(error));self.start_btn.configure(state="disabled")
        self._status_after_id=self.after(16,self._tick_status)
        self.protocol("WM_DELETE_WINDOW",self._on_close)

    def _save(self):
        try:save_settings(self.settings)
        except OSError as error:self._log(f"設定保存失敗：{error}")

    def _set_setting(self,name,value):
        self.settings=replace(self.settings,**{name:value})
        if self.controller is not None:self.controller.cancel_fire()
        self._save()

    def _toggle(self):
        if self.running:self._stop()
        else:self._start()

    def _start(self):
        if self.worker and self.worker.is_alive():return
        self._last_error=''
        if not Path(self.model_path).exists():
            self.status_label.configure(text=f"找不到 {self.model_path}");return
        self.settings=replace(esports_preset(self.settings),fire_cps=self.settings.fire_cps,fire_jitter_ms=self.settings.fire_jitter_ms,auto_aim=True,auto_fire=True,
            trigger_mode="自動",require_entity_state=False,require_crosshair_in_box=True)
        self.stop_event=threading.Event()
        self.worker_done.clear();self.shutdown_requested.clear()
        self.stats_queue.take()
        self.running=True
        self.start_btn.configure(text="開啟 ✓",fg_color="#268a5e")
        self.status_label.configure(text="已開啟 · L 關閉")
        self.worker=threading.Thread(target=self._worker_loop,daemon=True)
        self.worker.start();self._save()

    def _request_shutdown(self):
        # No Tk calls from hotkey thread; release output before GUI catches up.
        self.stop_event.set()
        if self.controller is not None:self.controller.mouse.disable()
        self.shutdown_requested.set()

    def _stop(self):
        self.running=False
        self.stop_event.set()
        self.settings=replace(self.settings,auto_aim=False,auto_fire=False)
        if self.controller is not None:self.controller.stop()
        self.start_btn.configure(text="開啟",fg_color="#1f6aa5")
        self.status_label.configure(text=self._last_error or "已關閉 · L 關閉")
        self._save()

    def _tick_status(self):
        if self._closing:return
        self._status_after_id=None
        if self.shutdown_requested.is_set():
            self.shutdown_requested.clear();self._stop()
        if self.worker_done.is_set() and self.worker and not self.worker.is_alive():
            self.worker_done.clear();self._stop()
        if self.running and self.controller is not None and not self._last_error:
            self.status_label.configure(text=self.controller.status)
        while not self.log_queue.empty():
            try:message=self.log_queue.get_nowait()
            except queue.Empty:break
            if "錯誤" in message or "失敗" in message:
                self._last_error=message
                self.status_label.configure(text=message)
        self._status_after_id=self.after(16,self._tick_status)

    def _log(self,message):
        try:self.log_queue.put_nowait(message)
        except queue.Full:pass

    def _on_close(self):
        if self._closing:return
        self._closing=True
        if self._status_after_id is not None:
            self.after_cancel(self._status_after_id)
            self._status_after_id=None
        self._stop();self.hotkey.stop()
        if self.worker and self.worker.is_alive():self.worker.join(timeout=1)
        self.destroy()

    def _worker_loop(self):
        capture = None
        renderer = None
        try:
            self._log(f"載入模型：{self.model_path}")
            model = YOLO(self.model_path, task="detect")
            # Warm the existing engine before opening the live capture.
            model.predict(np.zeros((640, 640, 3), dtype=np.uint8),
                          imgsz=640, device=0, verbose=False)
            if self.stop_event.is_set():
                return
            self._log("✅ 模型暖機完成；推論無軟體 FPS 上限，擷取依來源輸出")
            capture = LatestCapture(self.camera_index).start()
            self.controller = TargetControl(lambda: self.settings, self._log).start()
            selector = TargetSelector()
            sequence = 0
            fps_start = time.perf_counter()
            count, skipped, fps_value = 0, 0, 0.0
            next_status = 0.0
            announced = False
            source_warning = False
            last_frame = time.perf_counter()

            while not self.stop_event.is_set():
                packet = capture.read_latest(sequence)
                if packet is None:
                    if capture.error:
                        raise RuntimeError(capture.error)
                    timeout = 10 if sequence == 0 else 2
                    if time.perf_counter() - last_frame > timeout:
                        raise RuntimeError(f"OBS 超過 {timeout} 秒未提供畫面，已停止")
                    continue
                new_sequence, captured_at, frame = packet
                last_frame = time.perf_counter()
                skipped += max(0, new_sequence - sequence - 1)
                sequence = new_sequence
                if not announced:
                    self._log(f"影像源回報 {capture.reported_fps:.1f} FPS；"
                              f"緩衝設為 1：{'支援' if capture.buffer_supported else '不支援，使用持續擷取覆寫'}")
                    announced = True
                settings = self.settings
                area = get_area(frame)
                started = time.perf_counter()
                results = model.predict(
                    area, imgsz=640, device=0, conf=settings.conf_threshold,
                    verbose=False)
                # Copy boxes once. The preview receives an independent CPU result.
                results = [results[0].cpu()]
                finished = time.perf_counter()
                inference_ms = (finished - started) * 1000
                age_ms = (finished - captured_at) * 1000
                if self.stop_event.is_set():
                    break
                target = selector.pick(results, settings.my_team, settings.fov_limit, settings.target_part,
                                       settings.combined_priority, settings.head_fallback_distance)
                self.controller.submit(target, captured_at, settings)
                if target is None:
                    target_text = "--"
                else:
                    x, y = target["center"]
                    source_x = x + frame.shape[1] // 2 - 320
                    source_y = y + frame.shape[0] // 2 - 320
                    target_text = (f"{target['name']} 原圖 ({source_x:.0f}, {source_y:.0f}) "
                                   f"偏移 ({target['dx']:.0f}, {target['dy']:.0f})")

                count += 1
                if finished - fps_start >= 1.0:
                    fps_value = count / (finished - fps_start)
                    count, fps_start = 0, finished
                    if 0 < capture.fps < 40 and not source_warning:
                        self._log("⚠️ 影像源實測低於 40 FPS；請將 OBS「設定 → 影像 → FPS」"
                                  "改為 60 或以上，再重新啟用虛擬攝影機")
                        source_warning = True
                if finished >= next_status:
                    self.stats_queue.put(
                        f"擷取 {capture.fps:.1f} / 推論 {fps_value:.1f} FPS\n"
                        f"目標 {target_text}\n"
                        f"推論 {inference_ms:.1f} ms | 取幀後 {age_ms:.1f} ms | 略過 {skipped}")
                    next_status = finished + 0.25
                if self.preview_enabled:
                    self.render_queue.put(results)
        except Exception as error:
            self._log(f"❌ Worker 錯誤：{error}")
        finally:
            self.running = False
            self.stop_event.set()
            if self.controller is not None:
                self.controller.stop()
            if capture is not None and not capture.stop():
                self._log("等待影像驅動釋放；釋放完成前不會重複啟動")
                capture.thread.join()
            if renderer is not None:
                renderer.join()
            self.worker_done.set()


if __name__ == "__main__":
    App().mainloop()
