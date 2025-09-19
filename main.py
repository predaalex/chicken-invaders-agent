import random
import time
import pydirectinput
import pygetwindow as gw


def start_new_game():
    pydirectinput.press('space')
    print("Game started.")

def press_some_keys():
    keys = ['left', 'right']  # possible keys
    for _ in range(60):
        key = random.choice(keys)  # pick a random key
        pydirectinput.keyDown(key)   # press it
        time.sleep(0.05)           # small delay to mimic human input (optional)
        pydirectinput.keyUp(key)   # press it

def activate_window(window_title = "Chicken Invaders"):
    windows = gw.getWindowsWithTitle(window_title)

    if len(windows) < 1:
        print(f"Window {window_title} not found!")
        raise SystemExit

    if not windows:
        print(f"{window_title} window not found.")
        return

    window = windows[0]
    window.activate()
    time.sleep(0.5)
    print(f"{window_title} window activated.")


if __name__ == '__main__':

    activate_window()
    start_new_game()
    press_some_keys()


