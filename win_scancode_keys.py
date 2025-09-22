# win_scancode_keys.py
import ctypes
import time
from ctypes import wintypes

# --- Win32 constants
INPUT_KEYBOARD          = 1
KEYEVENTF_SCANCODE      = 0x0008
KEYEVENTF_KEYUP         = 0x0002

# PC/AT set 1 scancode for SPACE
SC_SPACE = 0x39

# ULONG_PTR type (32/64-bit safe)
ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

# --- Structures for SendInput ---

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk",       wintypes.WORD),
        ("wScan",     wintypes.WORD),
        ("dwFlags",   wintypes.DWORD),
        ("time",      wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx",          wintypes.LONG),
        ("dy",          wintypes.LONG),
        ("mouseData",   wintypes.DWORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg",    wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]

class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u",    INPUT_UNION),
    ]

SendInput = ctypes.windll.user32.SendInput

def _send_key_scancode(scan_code: int, keyup: bool = False):
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if keyup else 0)
    ki = KEYBDINPUT(
        wVk=0,
        wScan=scan_code,
        dwFlags=flags,
        time=0,
        dwExtraInfo=ULONG_PTR(0),
    )
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.ki = ki  # thanks to _anonymous_, we can set .ki directly
    n_sent = SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if n_sent != 1:
        raise ctypes.WinError()

def tap_space_scancode(down_ms: int = 18):
    """Tap SPACE using raw scancode. Adjust down_ms (12–25) if needed."""
    _send_key_scancode(SC_SPACE, keyup=False)
    time.sleep(max(0.001, down_ms / 1000.0))
    _send_key_scancode(SC_SPACE, keyup=True)
