import glob
import os

import cv2
import mss
import numpy as np
import pydirectinput
import pygetwindow as gw
import math
import itertools
import time
from collections import deque
from screen_grabber import ScreenGrabber
from win_scancode_keys import tap_space_scancode

class SmartFPS:
    def __init__(self, buffer_size=30):
        """
        buffer_size: how many most recent frames to average over.
                     Higher = smoother but slower to respond.
        """
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

class EggTrack:
    _id_iter = itertools.count(1)
    def __init__(self, x, y, r, conf):
        self.id = next(EggTrack._id_iter)
        self.x, self.y = float(x), float(y)
        self.r = float(r)
        self.conf = float(conf)
        self.vx, self.vy = 0.0, 0.0
        self.age = 1
        self.missed = 0

    def update(self, x, y, r, conf, alpha=0.6):
        # EMA position, then velocity from delta
        new_x, new_y = float(x), float(y)
        vx_new = new_x - self.x
        vy_new = new_y - self.y
        self.x = alpha * new_x + (1 - alpha) * self.x
        self.y = alpha * new_y + (1 - alpha) * self.y
        self.vx = 0.5 * vx_new + 0.5 * self.vx
        self.vy = 0.5 * vy_new + 0.5 * self.vy
        self.r = 0.5 * float(r) + 0.5 * self.r
        self.conf = 0.5 * float(conf) + 0.5 * self.conf
        self.age += 1
        self.missed = 0

class EggTracker:
    def __init__(self, max_dist=40, max_missed=4):
        self.tracks = {}
        self.max_dist = max_dist
        self.max_missed = max_missed

    def step(self, detections):
        # detections: list of (cx, cy, r, conf)
        # 1) match by nearest-neighbor with simple gating
        det_used = [False]*len(detections)
        # try to match existing tracks
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = -1, 1e9
            for j,(cx,cy, r, conf) in enumerate(detections):
                if det_used[j]: continue
                d = math.hypot(cx - tr.x, cy - tr.y)
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= self.max_dist:
                cx,cy,r,conf = detections[best_j]
                tr.update(cx, cy, r, conf)
                det_used[best_j] = True
            else:
                tr.missed += 1

        # 2) spawn new tracks for unmatched detections
        for j,(cx,cy,r,conf) in enumerate(detections):
            if not det_used[j]:
                nt = EggTrack(cx, cy, r, conf)
                self.tracks[nt.id] = nt

        # 3) prune stale tracks
        for tid, tr in list(self.tracks.items()):
            if tr.missed > self.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())

class Dodger:
    def __init__(self,
                 width,
                 ship_speed_px_per_s=300,
                 control_rate_hz=30,
                 deadzone_px=35,
                 target_ema=0.6  ,
                 lookahead_frames=10,
                 basin_rel_delta=0.2):
        """
        Very simple dodger:
          - picks a smoothed target with basin center
          - moves continuously left/right until within deadzone
        """
        self.width = int(width)
        self.vmax = float(ship_speed_px_per_s) / float(control_rate_hz)
        self.deadzone = int(deadzone_px)
        self.target_ema = float(target_ema)
        self.lookahead_frames = int(lookahead_frames)
        self.basin_rel_delta = float(basin_rel_delta)
        self.target_smooth = None
        self.current_dir = 0  # -1 left, 0 stop, +1 right

    def choose_target(self, danger, ship_x):
        """Same basin-centered logic as before, but simpler smoothing."""
        reach = int(self.vmax * self.lookahead_frames)
        lo = max(0, int(ship_x) - reach)
        hi = min(self.width - 1, int(ship_x) + reach)
        local = danger[lo:hi + 1]
        if local.size == 0:
            idx = int(ship_x)
        else:
            jmin = int(np.argmin(local))
            vmin = float(local[jmin])
            span = float(max(1e-6, local.max() - vmin))
            thr = vmin + self.basin_rel_delta * span

            # expand basin left/right
            L = jmin
            while L > 0 and local[L - 1] <= thr:
                L -= 1
            R = jmin
            while R < local.size - 1 and local[R + 1] <= thr:
                R += 1

            xs = np.arange(L, R + 1, dtype=np.float32)
            weights = (thr - local[L:R + 1] + 1e-6)
            cm = float((xs * weights).sum() / max(1e-6, weights.sum()))
            idx = lo + int(round(cm))

        # low-pass filter the target
        if self.target_smooth is None:
            self.target_smooth = float(idx)
        else:
            self.target_smooth = self.target_ema * float(idx) + \
                                 (1.0 - self.target_ema) * self.target_smooth
        return int(np.clip(self.target_smooth, 0, self.width - 1))

    def control(self, ship_x, target_x, shoot=True):
        """Simple continuous left/right control toward the target_x."""
        dx = target_x - ship_x

        if abs(dx) <= self.deadzone:
            # stop
            pydirectinput.keyUp('left')
            pydirectinput.keyUp('right')
            self.current_dir = 0
            return 0

        if dx < 0:
            pydirectinput.keyDown('left')
            pydirectinput.keyUp('right')
            self.current_dir = -1
            return -1
        else:
            pydirectinput.keyDown('right')
            pydirectinput.keyUp('left')
            self.current_dir = +1
            return +1


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

def _load_templates(resources_dir: str, pattern: str, kind: str):
    """
    Load all PNG templates matching `pattern` under `resources_dir`.
    Returns a list of (tpl_gray, mask) tuples.
    Raises RuntimeError if no usable templates are found.
    """
    search_glob = os.path.join(resources_dir, pattern)
    paths = sorted(glob.glob(search_glob))

    if not paths:
        raise RuntimeError(f"No {kind} templates found at {search_glob}")

    templates = []
    for p in paths:
        t = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # keep alpha if present
        if t is None:
            continue

        # Handle grayscale, BGR, or BGRA
        if t.ndim == 3 and t.shape[2] == 4:  # BGRA
            tpl_gray = cv2.cvtColor(t[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = t[:, :, 3]
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
        else:
            # If already single channel, keep as-is; otherwise convert BGR→gray
            tpl_gray = t if t.ndim == 2 else cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            mask = None

        templates.append((tpl_gray, mask))

    if not templates:
        raise RuntimeError(f"No usable {kind} templates could be loaded from {search_glob}")

    return templates


def init_templates(resources_dir: str = "resources"):
    """
    Loads ship, egg, and chicken templates from a resources directory.
    By default, uses '<this_file_dir>/resources'.
    Returns: (ship_templates, egg_templates, chicken_templates)
             where each item is a list of (tpl_gray, mask) tuples.
    """
    if resources_dir is None:
        # Resolve ./resources relative to this file when possible; fallback to CWD.
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        resources_dir = os.path.join(base_dir, "resources")

    ship_templates    = _load_templates(resources_dir, "ship*.png",    "ship")
    egg_templates     = _load_templates(resources_dir, "egg*.png",     "egg")
    chicken_templates = _load_templates(resources_dir, "chicken*.png", "chicken")

    return ship_templates, egg_templates, chicken_templates

def get_ship_position(gray_image, ship_position_x, game_window_height):
    """
    g: grayscale cropped game frame (HxW, uint8)
    returns smoothed player x (float, in [0,W))
    """
    band_h = int(0.25 * game_window_height)
    band = gray_image[game_window_height - band_h:game_window_height, :]

    # ensure uint8 grayscale
    if band.ndim != 2:
        band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    if band.dtype != np.uint8:
        band = band.astype(np.uint8)

    # mild blur helps both intensity & edges
    band_blur = cv2.GaussianBlur(band, (3, 3), 0)
    band_edges = cv2.Canny(band_blur, 60, 120)

    prev_px = ship_position_x
    # prior_sigma = 400.0  # how strongly to prefer near previous x
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
            # if prev_px is not None and prior_sigma > 1.0:
            #     # build a per-column prior centered at prev_px; add 0.5 floor so it's soft
            #     xs = (np.arange(conf.shape[1], dtype=np.float32) + tw / 2.0)
            #     prior = 0.5 + 0.5 * np.exp(-0.5 * ((xs - float(prev_px)) / prior_sigma) ** 2)
            #     conf *= prior[None, :]

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

def find_objects(gray_image, templates, confidence_score=0.95):
    """
    g: grayscale cropped game frame (H x W)
    returns: list of (cx, cy, r, score)
    """
    confidence_score = 0.95
    max_candidates_per_scale = 200
    nms_iou = 0.35

    search = cv2.GaussianBlur(gray_image, (3, 3), 0)  # g must be grayscale uint8
    # Enforce uint8 grayscale
    if search.ndim != 2:
        search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    if search.dtype != np.uint8:
        search = search.astype(np.uint8)

    cand_boxes, cand_scores = [], []

    for tpl_gray, mask in templates:
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

            ys, xs = np.where(conf_map >= confidence_score)
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

def compute_danger_map(
    tracks,
    ship_y,
    width,
    min_vy=0.7,
    horizon_frames=150,
    sigma_px=26.0,
    edge_fraction=0.06,
    edge_penalty_high=1.0,
    edge_penalty_low=0.2
):
    """
    Build a 1D danger (risk) array along the bottom line (length = width).
    Includes edge penalties to discourage camping in corners.
    """
    danger = np.zeros((width,), dtype=np.float32)

    for tr in tracks:
        vy = max(tr.vy, 0.0)
        if vy < min_vy:
            continue
        dy = ship_y - tr.y
        if dy <= 0:
            continue

        t = dy / max(vy, 1e-6)          # frames to reach ship line
        if t > horizon_frames:
            continue

        px = tr.x + tr.vx * t           # predicted x when crossing ship_y
        if px < -50 or px > width + 50:
            continue

        # weight imminent threats higher; include confidence and size
        w_t = 1.0 / max(0.5, t)
        w_c = max(0.1, float(getattr(tr, "conf", 1.0)))
        w_r = 1.0 + (float(getattr(tr, "r", 6.0)) / 12.0)
        weight = float(w_t * w_c * w_r)

        # gaussian splash around predicted x
        sigma = sigma_px + float(getattr(tr, "r", 6.0)) * 0.5
        xs = np.arange(width, dtype=np.float32)
        danger += weight * np.exp(-0.5 * ((xs - px) / sigma) ** 2)

    # ---- edge penalties to avoid corners ----
    edge_width = int(width * edge_fraction)
    if edge_width > 0:
        edge = np.linspace(edge_penalty_high, edge_penalty_low, edge_width, dtype=np.float32)
        danger[:edge_width] += edge
        danger[-edge_width:] += edge[::-1]

    # normalize to [0,1] for display/control (keep zeros if empty)
    m = float(danger.max())
    if m > 0:
        danger /= m
    return danger

def draw_overlay():
    """
    Uses globals:
      gray_frame, fps, ship_position_x, ship_position_y, ship_position_score,
      eggs (list of (cx, cy, r, sc)),
      tracks (list of EggTrack), danger (np.array width-long), target_x (int)
    """
    image = gray_frame.copy()

    # --- FPS ---
    live_fps = fps.fps()
    cv2.putText(image, f"{live_fps:.2f} FPS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Ship marker ---
    cv2.circle(image, (int(ship_position_x), int(ship_position_y)), 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{ship_position_score:.2f}",
                (int(ship_position_x) + 8, int(ship_position_y) + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Chickens ---
    for (cx, cy, r, sc) in chickens:
        cv2.rectangle(image, (int(cx - r), int(cy - r)), (int(cx + r), int(cy + r)), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"{sc:.2f}",
                    (int(cx + r), int(cy -  r)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Helper to get velocity for an egg by nearest track ---
    def nearest_track_vel(ex, ey, max_dist=28.0):
        best, bd = None, 1e9
        for tr in tracks:
            d = ((tr.x - ex) ** 2 + (tr.y - ey) ** 2) ** 0.5
            if d < bd:
                bd, best = d, tr
        if best is not None and bd <= max_dist:
            return best.vx, best.vy
        return 0.0, 0.0

    # --- Eggs with (score + velocity) ---
    for (cx, cy, r, sc) in eggs:
        cx_i, cy_i = int(cx), int(cy)
        cv2.circle(image, (cx_i, cy_i), max(2, int(r)), (255, 0, 255), 1, cv2.LINE_AA)
        vx, vy = nearest_track_vel(cx, cy, max_dist=max(24.0, 1.5 * r))
        cv2.putText(image, f"{sc:.2f},v=({vy:.1f})",
                    (cx_i + 6, cy_i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Danger heat "sparkline" and target marker ---
    if isinstance(danger, np.ndarray) and danger.size > 0:
        # draw sparkline one row above the ship
        ybar = max(0, int(ship_position_y) - 12)
        # scale to 0..255
        line = (np.clip(danger, 0.0, 1.0) * 255.0).astype(np.uint8)
        # ensure width fit
        w = min(len(line), image.shape[1])
        image[ybar, :w] = np.maximum(image[ybar, :w], line[:w])

        # vertical marker at target_x
        tx = int(np.clip(target_x, 0, image.shape[1] - 1))
        cv2.line(image, (tx, max(0, ybar - 6)), (tx, min(image.shape[0] - 1, ybar + 6)),
                 (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Chicken Invaders", image)



if __name__ == '__main__':
    # ------ CONFIG VARIABLES --------------
    debug = True
    pydirectinput.PAUSE = 0  # speed up
    pydirectinput.FAILSAFE = False
    game_window = activate_window()
    ship_templates, egg_templates, chicken_templates = init_templates()
    fps = SmartFPS().start()  # rolling average over last 30 frames
    egg_tracker = EggTracker()
    dodger = Dodger(game_window.width)
    screen_grabber = ScreenGrabber((game_window.top, game_window.left, game_window.width, game_window.height), stack_zize=400)
    ship_position_x = int(game_window.left + game_window.width / 2)  # start in the middle
    ship_position_y = game_window.top + game_window.height - 80
    ship_position_score = 0.0
    eggs = []
    # ------ CONFIG VARIABLES --------------

    start_new_game()
    while True:
        # if fps.fps() % 4  == 0:
        # tap_space_scancode(down_ms=15)
        gray_frame = screen_grabber.grab()
        ship_position_x, ship_position_score = get_ship_position(gray_frame, ship_position_x, game_window.height)
        chickens = find_objects(gray_frame, chicken_templates, confidence_score=0.3 )
        eggs = [(cx, cy, r, sc) for (cx, cy, r, sc) in find_objects(gray_frame, egg_templates) if cy < ship_position_y]
        tracks = egg_tracker.step(eggs)

        danger = compute_danger_map(tracks, ship_position_y, game_window.width,
                                    min_vy=0.7, horizon_frames=200, sigma_px=26.0)
        target_x = dodger.choose_target(danger, ship_position_x)
        _dir = dodger.control(ship_position_x, target_x)

        if debug:
            draw_overlay()

        if cv2.waitKey(1) & 0xFF == 27:
            break

        fps.update()
