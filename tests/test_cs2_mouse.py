import ctypes
import time

user32 = ctypes.windll.user32

print("3 秒後開始移動滑鼠，請切回 CS2 死鬥模式...")
time.sleep(3)

# 用舊版 mouse_event（不是 SendInput）
for i in range(30):
    user32.mouse_event(0x0001, 10, 10, 0, 0)  # MOUSEEVENTF_MOVE，往右下
    time.sleep(0.02)

print("測試結束！")