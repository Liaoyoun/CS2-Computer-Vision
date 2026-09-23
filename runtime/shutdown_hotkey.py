"""Only global shutdown shortcut: L. No mouse/focus/foreground gates."""
import ctypes
from ctypes import wintypes
import threading

class LStopHotkey:
    def __init__(self,callback):
        self.callback=callback
        self.ready=threading.Event()
        self.error=None
        self.thread_id=0
        self.thread=threading.Thread(target=self._run,daemon=True)
    def start(self):
        self.thread.start()
        if not self.ready.wait(3):raise RuntimeError('L 快捷鍵啟動逾時')
        if self.error:raise RuntimeError(self.error)
        return self
    def stop(self):
        if self.thread_id:ctypes.windll.user32.PostThreadMessageW(self.thread_id,0x12,0,0)
        if self.thread.is_alive():self.thread.join(timeout=1)
    def _run(self):
        api=ctypes.WinDLL('user32',use_last_error=True)
        self.thread_id=ctypes.windll.kernel32.GetCurrentThreadId()
        message=wintypes.MSG()
        api.PeekMessageW(ctypes.byref(message),None,0,0,0)
        registered=False
        try:
            if not api.RegisterHotKey(None,1,0x4000,ord('L')):
                raise RuntimeError('無法註冊 L 關閉鍵，可能被其他程式占用')
            registered=True;self.ready.set()
            while api.GetMessageW(ctypes.byref(message),None,0,0)>0:
                if message.message==0x312 and message.wParam==1:self.callback()
        except Exception as error:
            self.error=str(error);self.ready.set()
        finally:
            if registered:api.UnregisterHotKey(None,1)
