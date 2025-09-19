from collections import deque

import cv2
import mss
import numpy as np


class ScreenGrabber:
    def __init__(self, monitor_bbox):
        self.sct = mss.mss()
        self.mon = monitor_bbox  # {'top':y,'left':x,'width':w,'height':h}
        self.stack = deque(maxlen=4)

    def _preproc(self, img):
        # Keep original size, just convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return gray.astype(np.float32) / 255.0

    def grab(self):
        raw = np.array(self.sct.grab(self.mon))
        obs = self._preproc(raw)
        if len(self.stack) < 4:
            for _ in range(4):
                self.stack.append(obs)
        else:
            self.stack.append(obs)
        return np.stack(self.stack, axis=0)  # shape: [4, H, W] (here [4,480,640])
