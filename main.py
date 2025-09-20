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
        buffer_size = buffer_size
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

def get_ship_position(gray_image, ship_position_x):
    """
            g: grayscale cropped game frame (HxW, uint8)
            returns smoothed player x (float, in [0,W))
            """
    band_h = int(0.25 * game_window.height)
    band = gray_image[game_window.height - band_h:game_window.height, :]

    # ensure uint8 grayscale
    if band.ndim != 2:
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    if band.dtype != np.uint8:
        band = band.astype(np.uint8)

    # mild blur helps both intensity & edges
    band_blur = cv2.GaussianBlur(band, (3, 3), 0)
    band_edges = cv2.Canny(band_blur, 60, 120)

    prev_px = ship_position_x
    prior_sigma = 60.0  # how strongly to prefer near previous x
    mix_alpha = 0.7  # 0..1: weight for intensity vs edges

    best = {"score": -1.0, "x": None, "y": None, "w": None, "h": None}

    for tpl_gray, mask in ship_templates:
        # templates must be uint8 grayscale
        t = tpl_gray
        if t.ndim != 2:
            t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
        if t.dtype != np.uint8:
            t = t.astype(np.uint8)

        for s in [0.95, 1.0, 1.05]:
            tw = max(1, int(t.shape[1] * s))
            th = max(1, int(t.shape[0] * s))
            tpl_s = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
            if band_blur.shape[0] < th or band_blur.shape[1] < tw:
                continue

            # mask handling
            msk_s = None
            use_mask = False
            if mask is not None:
                msk_s = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                _, msk_s = cv2.threshold(msk_s, 10, 255, cv2.THRESH_BINARY)
                if msk_s.dtype != np.uint8:
                    msk_s = msk_s.astype(np.uint8)
                # skip degenerate masks
                if int(np.count_nonzero(msk_s)) < 12:
                    msk_s = None
                else:
                    use_mask = True

            # -------- intensity match --------
            # with mask: use SQDIFF_NORMED (lower is better), then invert to [0..1] "higher better"
            if use_mask:
                res_i = cv2.matchTemplate(band_blur, tpl_s, cv2.TM_SQDIFF_NORMED, mask=msk_s)
                conf_i = 1.0 - np.clip(np.nan_to_num(res_i, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
            else:
                # no mask: use CCOEFF_NORMED
                res_i = cv2.matchTemplate(band_blur, tpl_s, cv2.TM_CCOEFF_NORMED)
                conf_i = np.clip(np.nan_to_num(res_i, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

            # -------- edge match (unmasked) --------
            tpl_edges = cv2.Canny(tpl_s, 60, 120)
            res_e = cv2.matchTemplate(band_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
            conf_e = np.clip(np.nan_to_num(res_e, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

            # -------- blend intensity + edges --------
            conf = mix_alpha * conf_i + (1.0 - mix_alpha) * conf_e

            # -------- apply soft prior around previous x --------
            if prev_px is not None and prior_sigma > 1.0:
                # build a per-column prior centered at prev_px; add 0.5 floor so it's soft
                xs = (np.arange(conf.shape[1], dtype=np.float32) + tw / 2.0)
                prior = 0.5 + 0.5 * np.exp(-0.5 * ((xs - float(prev_px)) / prior_sigma) ** 2)
                conf *= prior[None, :]

            # sanitize (should already be 0..1)
            conf = np.clip(conf, 0.0, 1.0)

            # pick best location for this template+scale
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(conf)
            score = float(max_val)
            if score > best["score"]:
                x, y = max_loc
                best["score"] = score
                best["x"] = x + tw / 2.0
                best["y"] = y + th / 2.0
                best["w"] = tw
                best["h"] = th

    # threshold + smoothing + safety
    min_conf = 0.45
    if (best["x"] is not None) and (best["score"] >= min_conf):
        px = float(np.clip(best["x"], 0.0, game_window.width - 1.0))
    else:
        # fallback to previous or center
        px = ship_position_x if ship_position_x is not None else int(game_window.width / 2.0)

    # EMA smoothing (prevents jitter even if best jumps a bit)
    smooth = 0.3
    ship_position_x = px if ship_position_x is None else (smooth * ship_position_x + (1.0 - smooth) * px)
    ship_position_score = float(best["score"] if np.isfinite(best["score"]) else 0.0)

    return int(ship_position_x), ship_position_score

def nms_boxes(boxes, scores, iou_thresh):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def get_eggs(g: np.ndarray):
    """
    g: grayscale cropped game frame (H x W)
    returns: list of (cx, cy, r, score)
    """
    egg_min_conf = 0.95
    max_candidates_per_scale = 200
    nms_iou = 0.35

    search = cv2.GaussianBlur(g, (3, 3), 0)  # g must be grayscale uint8
    # Enforce uint8 grayscale
    if search.ndim != 2:
        search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    if search.dtype != np.uint8:
        search = search.astype(np.uint8)

    cand_boxes, cand_scores = [], []

    for tpl_gray, mask in egg_templates:
        # tpl_gray should already be grayscale uint8 from loading stage
        if tpl_gray.ndim != 2:
            tpl_gray = cv2.cvtColor(tpl_gray, cv2.COLOR_BGR2GRAY)
        if tpl_gray.dtype != np.uint8:
            tpl_gray = tpl_gray.astype(np.uint8)

        for s in [1.0]:
            tw = max(1, int(tpl_gray.shape[1] * s))
            th = max(1, int(tpl_gray.shape[0] * s))

            tpl_s = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
            # ensure uint8 grayscale
            if tpl_s.ndim != 2:
                tpl_s = cv2.cvtColor(tpl_s, cv2.COLOR_BGR2GRAY)
            if tpl_s.dtype != np.uint8:
                tpl_s = tpl_s.astype(np.uint8)

            if search.shape[0] < th or search.shape[1] < tw:
                continue

            # ---- mask handling ----
            msk_s = None
            use_mask = False
            if mask is not None:
                msk_s = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                # binarize & ensure uint8 single-channel
                _, msk_s = cv2.threshold(msk_s, 10, 255, cv2.THRESH_BINARY)
                if msk_s.dtype != np.uint8:
                    msk_s = msk_s.astype(np.uint8)
                if msk_s.ndim != 2:
                    msk_s = cv2.cvtColor(msk_s, cv2.COLOR_BGR2GRAY)
                if int(np.count_nonzero(msk_s)) < 12:
                    continue
                use_mask = True

            # ---- choose method ----
            # With mask: TM_CCORR_NORMED is supported; without: TM_CCOEFF_NORMED (if std>0), else CCORR_NORMED
            if use_mask:
                method = cv2.TM_CCORR_NORMED
            else:
                std_tpl = float(np.std(tpl_s))
                method = cv2.TM_CCOEFF_NORMED if std_tpl > 1e-3 else cv2.TM_CCORR_NORMED

            # ---- match (types now guaranteed equal: both uint8) ----
            res = cv2.matchTemplate(search, tpl_s, method, mask=msk_s)

            # sanitize map: remove NaN/Inf and clamp to [0,1]
            conf_map = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
            if method in (cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED):
                conf_map = np.clip(conf_map, 0.0, 1.0)
            else:
                conf_map = 1.0 - np.clip(conf_map, 0.0, 1.0)

            ys, xs = np.where(conf_map >= egg_min_conf)
            coords = list(zip(xs.tolist(), ys.tolist()))
            if len(coords) > max_candidates_per_scale:
                flat = [(conf_map[y, x], x, y) for (x, y) in coords]
                flat.sort(reverse=True)
                coords = [(x, y) for (sc, x, y) in flat[:max_candidates_per_scale]]

            for (x, y) in coords:
                sc = float(conf_map[y, x])
                if not np.isfinite(sc) or sc < 0:
                    sc = 0.0
                cand_boxes.append((float(x), float(y), float(tw), float(th)))
                cand_scores.append(sc)

    # Before NMS, final sanitize
    cand_scores = [0.0 if (not np.isfinite(s) or s < 0) else float(s) for s in cand_scores]

    keep = nms_boxes(cand_boxes, cand_scores, nms_iou)
    boxes = [cand_boxes[i] for i in keep]
    scores = [cand_scores[i] for i in keep]

    eggs = []
    for (x, y, w, h), sc in zip(boxes, scores):
        cx, cy = x + w / 2.0, y + h / 2.0
        r = 0.5 * (w + h) / 2.0
        eggs.append((cx, cy, r, sc))
    return eggs

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
    cv2.putText(image, f"{ship_position_score:.2f}", (ship_position_x + 5, ship_position_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    for (cx, cy, r, sc) in eggs:
        cv2.circle(image, (int(cx), int(cy)), max(2, int(r)), (255, 255, 255), 1)
        cv2.putText(image, f"{sc:.2f}", (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
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
    ship_position_score = 0.0
    eggs = []
    # ------ CONFIG VARIABLES --------------

    start_new_game()

    while True:
        bgr_frame, gray_frame = get_bgr_frame(game_window.top, game_window.left, game_window.width, game_window.height)

        ship_position_x, ship_position_score = get_ship_position(gray_frame, ship_position_x)
        eggs = get_eggs(gray_frame)
        if debug:
            draw_overlay()
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        fps.update()
