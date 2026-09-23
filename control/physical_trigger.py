"""Physical left-button / F-key triggers. Injected events never retrigger output."""
import ctypes
from ctypes import wintypes as w
import threading

LPARAM = ctypes.c_ssize_t
WPARAM = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, WPARAM, LPARAM)
class MouseData(ctypes.Structure):
    _fields_ = [('pt',w.POINT),('mouseData',w.DWORD),('flags',w.DWORD),('time',w.DWORD),('extra',WPARAM)]
class KeyData(ctypes.Structure):
    _fields_ = [('vkCode',w.DWORD),('scanCode',w.DWORD),('flags',w.DWORD),('time',w.DWORD),('extra',WPARAM)]

class PhysicalTrigger:
    def __init__(self, settings, edge):
        self.settings, self.edge = settings, edge
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run,daemon=True)
        self.error = None
        self.thread_id = 0
        self.held = False
        self.captured_button = False
        self.ui_rect = None

    def start(self):
        self.thread.start()
        if not self.ready.wait(3): raise RuntimeError('輸入監聽啟動逾時')
        if self.error: raise RuntimeError(self.error)
        return self

    def stop(self):
        if self.thread_id:
            ctypes.windll.user32.PostThreadMessageW(self.thread_id,0x12,0,0)
        if self.thread.is_alive(): self.thread.join(timeout=2)

    def _run(self):
        api = ctypes.WinDLL('user32',use_last_error=True)
        api.SetWindowsHookExW.argtypes=[ctypes.c_int,HOOKPROC,w.HINSTANCE,w.DWORD]
        api.SetWindowsHookExW.restype=w.HANDLE
        api.CallNextHookEx.argtypes=[w.HANDLE,ctypes.c_int,WPARAM,LPARAM]
        api.CallNextHookEx.restype=LRESULT
        api.UnhookWindowsHookEx.argtypes=[w.HANDLE]
        hooks=[]
        def mouse(code,msg,ptr):
            if code >= 0:
                data=ctypes.cast(ptr,ctypes.POINTER(MouseData)).contents
                if not data.flags & 3 and msg in (0x201,0x202):
                    enabled=self.settings().auto_fire and self.settings().trigger_mode!='自動' and self.settings().trigger_key=='左鍵'
                    down=msg==0x201
                    rect = self.ui_rect
                    over_ui = rect and rect[0] <= data.pt.x <= rect[2] and rect[1] <= data.pt.y <= rect[3]
                    if down and enabled and not over_ui:
                        self.captured_button=True
                        self.edge(True)
                        return 1
                    if not down and self.captured_button:
                        self.captured_button=False
                        self.edge(False)
                        return 1
            return api.CallNextHookEx(None,code,msg,ptr)
        def keyboard(code,msg,ptr):
            if code >= 0:
                data=ctypes.cast(ptr,ctypes.POINTER(KeyData)).contents
                setting=self.settings()
                key=setting.trigger_key
                wanted=0x6F+int(key[1:]) if key.startswith('F') else -1
                if not data.flags & 0x10 and data.vkCode==wanted:
                    down=msg in (0x100,0x104)
                    up=msg in (0x101,0x105)
                    if down and setting.auto_fire and setting.trigger_mode != '自動':
                        if not self.held: self.edge(True)
                        self.held=True
                        return 1
                    if up and self.held:
                        self.held=False
                        self.edge(False)
                        return 1
            return api.CallNextHookEx(None,code,msg,ptr)
        self.mouse_callback=HOOKPROC(mouse)
        self.key_callback=HOOKPROC(keyboard)
        try:
            self.thread_id=ctypes.windll.kernel32.GetCurrentThreadId()
            message=w.MSG()
            api.PeekMessageW(ctypes.byref(message),None,0,0,0)
            for kind, callback in [(14,self.mouse_callback),(13,self.key_callback)]:
                hook=api.SetWindowsHookExW(kind,callback,None,0)
                if not hook: raise ctypes.WinError(ctypes.get_last_error())
                hooks.append(hook)
            self.ready.set()
            while api.GetMessageW(ctypes.byref(message),None,0,0)>0:
                api.TranslateMessage(ctypes.byref(message))
                api.DispatchMessageW(ctypes.byref(message))
        except Exception as error:
            self.error=str(error)
            self.ready.set()
        finally:
            for hook in hooks: api.UnhookWindowsHookEx(hook)
            self.edge(False)
