from collections import deque

import cv2
import mss
import numpy as np


class ScreenGrabber:
    def __init__(self, monitor_bbox, stack_zize=40):
        self.sct = mss.mss()
        self.mon = monitor_bbox
        self.stack_size = stack_zize
        self.stack = deque(maxlen=stack_zize)

    def _preproc(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    def grab(self):
        raw = np.array(self.sct.grab(self.mon))
        obs = self._preproc(raw)
        if len(self.stack) < self.stack_size:
            for _ in range(self.stack_size):
                self.stack.append(obs)
        else:
            self.stack.popleft()
            self.stack.append(obs)
        return self.stack[-1]
