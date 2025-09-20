import random
import time

import cv2
import mss
import numpy as np
import pydirectinput
import pygetwindow as gw


import time
from collections import deque

class SmartFPS:
    def __init__(self, buffer_size=30):
        """
        buffer_size: how many most recent frames to average over.
                     Higher = smoother but slower to respond.
        """
        self.buffer_size = buffer_size
        self._timestamps = deque(maxlen=buffer_size)
        self._start = None

    def start(self):
        """Mark the start time."""
        self._start = time.time()
        self._timestamps.clear()
        return self

    def update(self):
        """Call this once per frame."""
        now = time.time()
        self._timestamps.append(now)

    def fps(self):
        """Return the rolling average FPS over the last buffer_size frames."""
        if len(self._timestamps) < 2:
            return 0.0  # not enough data yet
        # time between first and last timestamp in the buffer
        elapsed = self._timestamps[-1] - self._timestamps[0]
        frame_count = len(self._timestamps) - 1
        return frame_count / elapsed if elapsed > 0 else 0.0

    def elapsed(self):
        """Total time since start."""
        if not self._start:
            return 0.0
        return time.time() - self._start

def start_new_game():
    pydirectinput.press('space')
    print("Game started.")

def activate_window(window_title = "Chicken Invaders"):
    windows = gw.getWindowsWithTitle(window_title)

    if len(windows) < 1 or not windows:
        print(f"Window {window_title} not found!")
        raise SystemExit

    window = windows[0]
    window.activate()
    time.sleep(0.25)
    print(f"{window_title} window activated.")

    return window

def get_bgr_frame(x1, y1, x2, y2):
    """Return current BGR frame (HxWx3) of the game window."""
    CROP = (x1, y1, x2, y2)
    raw = screen_grabber.grab(CROP)              # ScreenShot object
    frame = np.array(raw)                        # BGRA uint8
    frame_bgr = frame[:, :, :3]                  # Drop alpha channel, still BGR
    frage_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    return frame_bgr, frage_gray

def init_templates():
    ship_templates = []
    for p in ["resources/ship1.png", "resources/ship2.png", "resources/ship3.png"]:
        t = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # supports alpha
        if t is None:
            continue
        if t.shape[2] == 4:  # BGRA
            # gray template + mask from alpha
            tpl_gray = cv2.cvtColor(t[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = t[:, :, 3]
            # binarize the mask (in case of semi-transparent edges)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        else:
            tpl_gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            mask = None
        ship_templates.append((tpl_gray, mask))
    if not ship_templates:
        raise RuntimeError("No ship templates could be loaded from resources/ship*.png")

    # --- egg templates (alpha-aware) ---
    egg_templates = []
    for p in ["resources/egg1.png"]:
        t = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if t is None:
            continue
        if t.shape[2] == 4:  # BGRA with alpha
            tpl_gray = cv2.cvtColor(t[:, :, :3], cv2.COLOR_BGR2GRAY)
            mask = t[:, :, 3]
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        else:
            tpl_gray = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            mask = None
        egg_templates.append((tpl_gray, mask))
    if not egg_templates:
        raise RuntimeError("No egg templates found at resource/egg*.png")

    return ship_templates, egg_templates

def get_ship_position(img):

    return 0, 0

def _sanitize_for_cv(img):
    if img is None:
        return None

    # Ensure dtype
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Ensure 3 channels BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Ensure contiguous + writable
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    if not img.flags['WRITEABLE']:
        img = img.copy()

    return img

def draw_overlay():
    fps.update()
    image = _sanitize_for_cv(bgr_frame)
    live_fps = fps.fps()
    cv2.putText(image, f"{live_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(image, (ship_position_x, ship_position_y), 5, (0, 0, 255), 2)
    cv2.imshow("Chicken Invaders", image)


if __name__ == '__main__':
    # ------ CONFIG VARIABLES --------------
    debug = True
    screen_grabber = mss.mss()
    ship_templates, egg_templates = init_templates()
    fps = SmartFPS(buffer_size=30).start()  # rolling average over last 30 frames
    game_window = activate_window()
    ship_position_x = int(game_window.left + game_window.width / 2)
    ship_position_y = game_window.top + game_window.height - 80
    # ------ CONFIG VARIABLES --------------

    start_new_game()

    while True:
        bgr_frame, gray_frame = get_bgr_frame(game_window.top, game_window.left, game_window.width, game_window.height)

        ship_position_x, ship_position_y = get_ship_position(gray_frame)

        if debug:
            draw_overlay()
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        fps.update()


