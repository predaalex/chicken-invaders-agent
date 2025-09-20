import time, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import mss
import numpy as np
import pydirectinput
import pygetwindow as gw
import cv2


# ---- plug your own implementations here ----
def get_frame(debug=False) -> np.ndarray:
    """Return current BGR frame (HxWx3) of the game window."""
    # Grab raw screen from mss
    raw = screen_grabber.grab(CROP)          # ScreenShot object
    frame = np.array(raw)                        # BGRA uint8
    frame_bgr = frame[:, :, :3]                  # Drop alpha channel, still BGR

    # Show DEBUG
    if debug:
        cv2.imshow("Chicken Invaders", frame_bgr)
        cv2.waitKey(0)  # non-blocking wait

    return frame_bgr

def hold(keys):
    for k in keys:
        pydirectinput.keyDown(k)

def release(keys):
    for k in keys:
        pydirectinput.keyUp(k)

# --------------------------------------------

# ----- config you should set once -----
window_title = "Chicken Invaders"
windows = gw.getWindowsWithTitle(window_title)

if len(windows) < 1:
    print(f"Window {window_title} not found!")
    raise SystemExit

if not windows:
    print(f"{window_title} window not found.")
    raise SystemExit

window = windows[0]
print(str(window) + f" FOUND {window_title}")
window.activate()

screen_grabber = mss.mss()
CROP = (window.top, window.left,window.width, window.height)   # (x0,y0,x1,y1) -> set to your game’s inner playfield
PLAYER_Y_REL = 0.95       # player row as a fraction of cropped height (≈ bottom)
CAP_FPS = 21              # tune to your capture rate

pydirectinput.PAUSE = 0      # avoid the 0.1s default delay
pydirectinput.FAILSAFE = False
keys_down = tuple()
# --------------------------------------
class PulseMover:
    """
    Sends short key presses ("pulses") for movement.
    - hold_ms: how long to hold each key press
    - min_gap_ms: minimum gap between pulses of the SAME direction
    """
    def __init__(self, hold_ms=22, min_gap_ms=10):
        self.hold_ms = hold_ms
        self.min_gap = min_gap_ms / 1000.0
        self._last_pulse = {"left": 0.0, "right": 0.0}

    def _pulse(self, key: str):
        pydirectinput.keyDown(key)
        time.sleep(self.hold_ms / 1000.0)
        pydirectinput.keyUp(key)
        self._last_pulse[key] = time.time()

    def maybe_pulse(self, direction: str):
        now = time.time()
        if direction not in ("left", "right"):
            return  # neutral: do nothing (no keys are held anyway)
        if now - self._last_pulse[direction] >= self.min_gap:
            self._pulse(direction)


class SimpleDodger:
    def __init__(self, width: int, height: int):
        self.player_x_score = None
        self.W, self.H = width, height
        self.prev_eggs: List[Tuple[float,float]] = []  # previous centroids for velocity
        self.player_x: Optional[float] = None
        self.player_y = int(PLAYER_Y_REL * self.H)
        self.prev_frame_gray: Optional[np.ndarray] = None

        # tunables for player matching
        self.band_height_frac = 0.25  # search bottom 25%
        self.scales = [0.95, 1.00, 1.05]  # multi-scale search
        self.min_conf = 0.45  # confidence threshold for a valid match
        self.smooth = 0.3  # EMA smoothing factor
        self.ship_templates = []
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
            self.ship_templates.append((tpl_gray, mask))
        if not self.ship_templates:
            raise RuntimeError("No ship templates could be loaded from resources/ship*.png")

        # --- egg templates (alpha-aware) ---
        self.egg_templates = []
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
            self.egg_templates.append((tpl_gray, mask))
        if not self.egg_templates:
            raise RuntimeError("No egg templates found at resource/egg*.png")

        # --- matching knobs ---
        self.egg_scales = [1.0]  # adjust if needed
        self.egg_min_conf = 0.95  # confidence threshold
        self.max_candidates_per_scale = 200
        self.nms_iou = 0.35

        self.max_vert_horizon = int(0.6 * self.H)  # ignore eggs very far above player
        self.dist_decay_px = 50.0  # larger -> slower decay with dy
        self.base_sigma_px = 1.0  # base horizontal risk width
        self.size_sigma_k = 1.12  # how much radius inflates sigma
        self.dist_sigma_k = 0.02  # widen sigma with vertical distance
        self.min_score_weight = 0.3  # lower bound for weak detections

    def nms_boxes(self, boxes, scores, iou_thresh):
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

    # --- preprocessing ---
    def preproc(self, frame_bgr: np.ndarray) -> np.ndarray:
        x0,y0,x1,y1 = CROP
        roi = frame_bgr[y0:y1, x0:x1]
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # light normalize to reduce flashes
        g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        return g

    # --- player detection (bottom band, bright blob fallback) ---
    def detect_player_x(self, g: np.ndarray) -> float:
        """
        g: grayscale cropped game frame (HxW, uint8)
        returns smoothed player x (float, in [0,W))
        """
        band_h = int(getattr(self, "band_height_frac", 0.25) * self.H)
        band = g[self.H - band_h:self.H, :]

        # ensure uint8 grayscale
        if band.ndim != 2:
            band = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        if band.dtype != np.uint8:
            band = band.astype(np.uint8)

        # mild blur helps both intensity & edges
        band_blur = cv2.GaussianBlur(band, (3, 3), 0)
        band_edges = cv2.Canny(band_blur, 60, 120)

        prev_px = self.player_x
        prior_sigma = float(getattr(self, "player_prior_sigma_px", 60.0))  # how strongly to prefer near previous x
        mix_alpha = float(getattr(self, "ship_match_mix_alpha", 0.7))  # 0..1: weight for intensity vs edges

        best = {"score": -1.0, "x": None, "y": None, "w": None, "h": None}

        for tpl_gray, mask in self.ship_templates:
            # templates must be uint8 grayscale
            t = tpl_gray
            if t.ndim != 2:
                t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            if t.dtype != np.uint8:
                t = t.astype(np.uint8)

            for s in getattr(self, "scales", [0.9, 1.0, 1.1]):
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
        min_conf = float(getattr(self, "min_conf", 0.55))
        if (best["x"] is not None) and (best["score"] >= min_conf):
            px = float(np.clip(best["x"], 0.0, self.W - 1.0))
        else:
            # fallback to previous or center
            px = self.player_x if self.player_x is not None else self.W / 2.0

        # EMA smoothing (prevents jitter even if best jumps a bit)
        smooth = float(getattr(self, "smooth", 0.8))
        self.player_x = px if self.player_x is None else (smooth * self.player_x + (1.0 - smooth) * px)
        self.player_x_score = float(best["score"] if np.isfinite(best["score"]) else 0.0)

        return float(self.player_x)

    def debug_player_overlay(self, g: np.ndarray, win_name="Player Match"):
        band_h = int(self.band_height_frac * self.H)
        band = g[self.H - band_h:self.H, :].copy()
        px = self.player_x if self.player_x is not None else self.W / 2
        vis = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(px), int(0.5 * band.shape[0])), 6, (0, 255, 0), 2)
        cv2.putText(vis, f"{self.player_x_score:.2f}", (int(px) + 6, int(0.5 * band.shape[0]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win_name, vis)

    # --- egg detection (white ovals moving down) ---
    def detect_eggs(self, g: np.ndarray):
        """
        g: grayscale cropped game frame (H x W)
        returns: list of (cx, cy, r, score)
        """
        search = cv2.GaussianBlur(g, (3, 3), 0)  # g must be grayscale uint8
        # Enforce uint8 grayscale
        if search.ndim != 2:
            search = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
        if search.dtype != np.uint8:
            search = search.astype(np.uint8)

        cand_boxes, cand_scores = [], []

        for tpl_gray, mask in self.egg_templates:
            # tpl_gray should already be grayscale uint8 from loading stage
            if tpl_gray.ndim != 2:
                tpl_gray = cv2.cvtColor(tpl_gray, cv2.COLOR_BGR2GRAY)
            if tpl_gray.dtype != np.uint8:
                tpl_gray = tpl_gray.astype(np.uint8)

            for s in self.egg_scales:
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

                ys, xs = np.where(conf_map >= self.egg_min_conf)
                coords = list(zip(xs.tolist(), ys.tolist()))
                if len(coords) > self.max_candidates_per_scale:
                    flat = [(conf_map[y, x], x, y) for (x, y) in coords]
                    flat.sort(reverse=True)
                    coords = [(x, y) for (sc, x, y) in flat[:self.max_candidates_per_scale]]

                for (x, y) in coords:
                    sc = float(conf_map[y, x])
                    if not np.isfinite(sc) or sc < 0:
                        sc = 0.0
                    cand_boxes.append((float(x), float(y), float(tw), float(th)))
                    cand_scores.append(sc)

        # Before NMS, final sanitize
        cand_scores = [0.0 if (not np.isfinite(s) or s < 0) else float(s) for s in cand_scores]

        keep = self.nms_boxes(cand_boxes, cand_scores, self.nms_iou)
        boxes = [cand_boxes[i] for i in keep]
        scores = [cand_scores[i] for i in keep]

        eggs = []
        for (x, y, w, h), sc in zip(boxes, scores):
            cx, cy = x + w / 2.0, y + h / 2.0
            r = 0.5 * (w + h) / 2.0
            eggs.append((cx, cy, r, sc))
        return eggs

    def debug_eggs_overlay(self, g: np.ndarray, eggs, win_name="Eggs"):
        """
        g: grayscale crop
        eggs: list of (cx, cy, r, score)
        returns BGR image for imshow
        """
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for (cx, cy, r, sc) in eggs:
            cv2.circle(vis, (int(cx), int(cy)), max(2, int(r)), (255, 255, 255), 1)
            cv2.putText(vis, f"{sc:.2f}", (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, vis)

    # --- build 1-D risk map over x by projecting to player y ---

    def risk_map(
            self,
            eggs: List[Tuple[float, float, float, float]],  # (x, y, r, sc)
            max_distance: float = 300.0,
            base_sigma: float = 10.0,
            sigma_scale: float = 1.2,
            min_weight: float = 0.2,
            edge_penalty_high: float = 3.0,
            edge_penalty_low: float = 2.0,
            edge_fraction: float = 1 / 5.0,
            # NEW params
            dist_mode: str = "linear",  # "linear" | "inv2" | "exp"
            d_max: float = 350.0,  # for "linear"
            d0: float = 120.0,  # for "inv2"
            lam: float = 140.0  # for "exp"
    ) -> np.ndarray:
        x_axis = np.arange(self.W, dtype=np.float32)
        risk = np.zeros(self.W, dtype=np.float32)
        py = self.player_y
        px = float(self.player_x if self.player_x is not None else self.W / 2)

        for (x, y, r, sc) in eggs:
            # your existing vertical gate
            t = py - y
            if t > max_distance:
                continue

            # Gaussian column under the egg (same as before)
            sigma = base_sigma + sigma_scale * r
            gauss = np.exp(-0.5 * ((x_axis - x) / sigma) ** 2)

            # ---- NEW: distance-based weight to the ship ----
            d_vert = t
            d_horiz = abs(x - px)
            d = (d_vert ** 2 + d_horiz ** 2) ** 0.5

            if dist_mode == "linear":
                w_dist = max(min_weight, 1.0 - d / max(1.0, d_max))
            elif dist_mode == "inv2":
                w_dist = max(min_weight, 1.0 / (1.0 + (d / max(1.0, d0)) ** 2))
            else:  # "exp"
                w_dist = max(min_weight, float(np.exp(-d / max(1.0, lam))))

            w_dist *= 4

            # (optional) also keep your original vertical weighting (closer vertically → more risk)
            w_vert = max(min_weight, 1.0 - t / max_distance)

            # final weight = distance modulator × your original vertical weight
            w = w_dist * w_vert

            risk += (w * gauss).astype(np.float32)

        # edge penalties (unchanged)
        edge_width = int(self.W * edge_fraction)
        if edge_width > 0:
            edge = np.linspace(edge_penalty_high, edge_penalty_low, edge_width)
            risk[:edge_width] += edge
            risk[-edge_width:] += edge[::-1]
        return risk

    # --- choose target x and action ---
    def choose_action(self, px: float, risk: np.ndarray, dead_zone: int = 6) -> str:
        """
        Path-aware chooser:
        - Discretize px
        - Compute 'blocked' columns from a relative threshold
        - Find local minima as candidate targets
        - Keep only candidates reachable from current x without crossing blocked columns
        - Pick the candidate with minimum path cost (integrated risk along the way)
        - Move one step toward it (with small hysteresis)
        """
        W = self.W
        if W <= 1:
            return "noop"

        x0 = int(np.clip(round(px), 0, W - 1))
        r = risk.astype(np.float32)

        # --- 1) blocked columns: relative to distribution, not absolute ---
        # Use median + k*MAD for robustness; also cap by a fraction of max
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) + 1e-6
        thr_rel = med + getattr(self, "block_k_mad", 2.5) * mad
        thr_max = getattr(self, "block_frac_of_max", 0.75) * float(r.max() if r.size else 0.0)
        block_thr = max(thr_rel, thr_max)
        blocked = r >= block_thr

        # If current position is blocked, first objective is to escape to nearest unblocked x
        if blocked[x0]:
            # search left and right for nearest gap
            left_dist = np.argmax(~blocked[:x0 + 1][::-1]) if np.any(~blocked[:x0 + 1]) else W
            right_dist = np.argmax(~blocked[x0:]) if np.any(~blocked[x0:]) else W
            # If both sides blocked (rare), stay put
            if left_dist == 0 or right_dist == 0:
                # we're already unblocked (shouldn't happen here), fall through
                pass
            elif left_dist == W and right_dist == W:
                return "noop"
            else:
                # go to nearer gap; tie-break by lower neighborhood risk
                if left_dist < right_dist:
                    return "left"
                elif right_dist < left_dist:
                    return "right"
                else:
                    # tie: compare local window risk on both sides
                    w = getattr(self, "look_window", 5)
                    lx = max(0, x0 - left_dist)
                    rx = min(W - 1, x0 + right_dist)
                    left_cost = r[max(0, lx - w):min(W, lx + w + 1)].mean() if left_dist < W else 1e9
                    right_cost = r[max(0, rx - w):min(W, rx + w + 1)].mean() if right_dist < W else 1e9
                    return "left" if left_cost < right_cost else "right"

        # --- 2) candidate targets = local minima among unblocked columns ---
        # smooth a tiny bit to avoid trivial micro-minima
        from scipy.ndimage import uniform_filter1d  # if you don't have scipy, replace with simple rolling mean
        smooth_w = getattr(self, "minima_smooth_w", 3)
        rs = uniform_filter1d(r, size=max(1, smooth_w), mode="nearest") if smooth_w > 1 else r

        # local minima mask (strict)
        left_higher = np.r_[True, rs[1:] > rs[:-1]]
        right_higher = np.r_[rs[:-1] < rs[1:], True]
        local_min = left_higher & right_higher & (~blocked)

        # Always include current x neighborhood's best point so we don't get stuck
        local_min[x0] = True

        cand_idxs = np.flatnonzero(local_min)
        if cand_idxs.size == 0:
            # No candidates (all blocked?): fallback to moving toward lower risk side
            left_mean = r[:x0].mean() if x0 > 0 else 1e9
            right_mean = r[x0 + 1:].mean() if x0 < W - 1 else 1e9
            if abs(left_mean - right_mean) <= 1e-6:
                return "noop"
            return "left" if left_mean < right_mean else "right"

        # --- 3) reachability: can we go from x0 to xt without crossing blocked columns? ---
        reachable = []
        for xt in cand_idxs:
            if xt == x0:
                reachable.append(xt)
                continue
            lo = min(x0, xt)
            hi = max(x0, xt)
            # we allow endpoints even if exactly at threshold; require a clear corridor
            if np.any(blocked[lo:hi + 1]):
                continue
            reachable.append(xt)

        if not reachable:
            # corridor blocked both ways; pick the side with lower *path* cost up to the first block
            # compute cumulative sums to get path integrals cheaply
            c = np.cumsum(r)
            # left segment until block
            lb = np.where(blocked[:x0])[0]
            l_lo = 0 if lb.size == 0 else (lb.max() + 1)  # first unblocked index after last block on the left
            left_cost = (c[x0] - (c[l_lo - 1] if l_lo > 0 else 0.0)) / max(1, x0 - l_lo + 1)

            # right segment until block
            rb = np.where(blocked[x0 + 1:])[0]
            if rb.size > 0:
                r_hi = x0 + 1 + rb.min() - 1
            else:
                r_hi = W - 1
            right_cost = (c[r_hi] - c[x0]) / max(1, r_hi - x0)

            return "left" if left_cost < right_cost else "right"

        # --- 4) choose target by minimum path cost (integral of risk) ---
        c = np.cumsum(r)

        def path_cost(a, b):
            lo, hi = (a, b) if a <= b else (b, a)
            seg_sum = c[hi] - (c[lo - 1] if lo > 0 else 0.0)
            # average cost per pixel encourages nearer minima if similar risk
            return seg_sum / max(1, hi - lo + 1)

        # small search radius to avoid chasing far minima if a near one is almost as good
        max_jump = getattr(self, "max_target_jump_px", W // 4)
        scored = []
        for xt in reachable:
            if abs(xt - x0) > max_jump:
                continue
            scored.append((path_cost(x0, xt), xt))
        if not scored:
            # if all reachable are far, keep nearest by distance
            nearest = min(reachable, key=lambda xt: abs(xt - x0))
            target_x = float(nearest)
        else:
            scored.sort()
            best_cost, target_idx = scored[0]
            target_x = float(target_idx)

        # --- 5) dead-zone + hysteresis to avoid jitter ---
        if abs(target_x - px) <= dead_zone:
            return "noop"

        # hysteresis: if we were moving a direction, keep it unless clearly worse
        last_act = getattr(self, "_last_move", "noop")
        dir_now = "left" if target_x < px else "right"
        if last_act in ("left", "right") and last_act != dir_now:
            # only flip if the opposite side is better by a margin
            margin = getattr(self, "flip_margin", 0.10)  # 10% better path needed to flip
            # Compare local average risks to each side
            w = getattr(self, "look_window", 7)
            lx = max(0, int(px) - w)
            rx = min(W - 1, int(px) + w)
            left_avg = r[max(0, int(px) - w):int(px)].mean() if int(px) - w >= 1 else 1e9
            right_avg = r[int(px) + 1:rx + 1].mean() if int(px) + 1 <= rx else 1e9
            if last_act == "left" and (left_avg <= (1.0 + margin) * right_avg):
                dir_now = "left"
            elif last_act == "right" and (right_avg <= (1.0 + margin) * left_avg):
                dir_now = "right"

        self._last_move = dir_now
        return dir_now

    # --- overlay for debugging (optional) ---
    def draw_overlay(self, g, eggs, px, risk):
        """
        g: grayscale crop
        eggs: list of (cx, cy, r, score)
        px: player x (float)
        risk: 1D array length W
        """
        vis = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # # player row
        # py = self.player_y
        # cv2.line(vis, (0, py), (self.W - 1, py), (50, 50, 50), 1)
        # cv2.circle(vis, (int(px), py), 6, (0, 255, 0), 2)
        #
        # # eggs
        # for (cx, cy, r, sc) in eggs:
        #     cv2.circle(vis, (int(cx), int(cy)), max(2, int(r)), (255, 255, 255), 1)
        #     cv2.putText(vis, f"{sc:.2f}", (int(cx) + 6, int(cy) - 6),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # risk strip
        strip = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0)
        mx = float(strip.max()) if strip.size else 0.0
        if mx > 0:
            strip = (255 * (strip / mx)).astype(np.uint8)
        else:
            strip = np.zeros_like(strip, dtype=np.uint8)
        strip = cv2.resize(strip[None, :], (self.W, 40), interpolation=cv2.INTER_NEAREST)
        strip_bgr = cv2.applyColorMap(strip, cv2.COLORMAP_JET)

        return np.vstack([vis, strip_bgr])


# ------------- main control loop -------------
def run_dodger(show_debug=True):
    # probe a frame to get dimensions
    bgr = get_frame()
    x0,y0,x1,y1 = CROP
    W,H = x1-x0, y1-y0
    bot = SimpleDodger(W, H)

    last_release = time.time()
    keys_down: Tuple[str,...] = tuple()
    mover = PulseMover(hold_ms=45, min_gap_ms=5)  # tune these two

    try:
        while True:
            bgr = get_frame()
            roi_g = bot.preproc(bgr)

            px = bot.detect_player_x(roi_g)
            bot.debug_player_overlay(roi_g)

            eggs = bot.detect_eggs(roi_g)
            bot.debug_eggs_overlay(roi_g, eggs)

            risk = bot.risk_map(eggs)
            act = bot.choose_action(px, risk)

            if act == "left":
                mover.maybe_pulse("left")
            elif act == "right":
                mover.maybe_pulse("right")
            else:
                # neutral -> no pulses this frame
                pass

            if show_debug:
                vis = bot.draw_overlay(roi_g, eggs, px, risk)
                cv2.imshow("Dodger Debug (crop + risk)", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            # simple pacing to ~CAP_FPS
            time.sleep(max(0.0, 1.0/CAP_FPS - 0.001))
    finally:
        if keys_down:
            release(keys_down)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dodger()
