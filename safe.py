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

def press(keys: Tuple[str, ...], hold_ms: int = 16):
    """Hold these keys."""
    for key in keys:
        pydirectinput.keyDown(key)

def release(keys: Tuple[str, ...]):
    """Release these keys."""
    for key in keys:
        pydirectinput.keyUp(key)
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

screen_grabber = mss.mss()
CROP = (window.top, window.left,window.width, window.height)   # (x0,y0,x1,y1) -> set to your game’s inner playfield
PLAYER_Y_REL = 0.88       # player row as a fraction of cropped height (≈ bottom)
CAP_FPS = 21              # tune to your capture rate
MOVE_HOLD_MS = 12         # how long to hold a direction per decision
# --------------------------------------

@dataclass
class Egg:
    x: float
    y: float
    vx: float
    vy: float
    r: float  # approx radius (for risk width)

class SimpleDodger:
    def __init__(self, width: int, height: int):
        self.W, self.H = width, height
        self.prev_eggs: List[Tuple[float,float]] = []  # previous centroids for velocity
        self.player_x: Optional[float] = None
        self.player_y = int(PLAYER_Y_REL * self.H)
        self.prev_frame_gray: Optional[np.ndarray] = None

        # tunables for player matching
        self.band_height_frac = 0.25  # search bottom 25%
        self.scales = [1.0, 1.05, 0.95]  # multi-scale search
        self.min_conf = 0.9  # confidence threshold for a valid match
        self.smooth = 0.1  # EMA smoothing factor
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
        self.dist_decay_px = 90.0  # larger -> slower decay with dy
        self.base_sigma_px = 10.0  # base horizontal risk width
        self.size_sigma_k = 1.2  # how much radius inflates sigma
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
        g: grayscale cropped game frame (HxW)
        returns smoothed player x (float, in [0,W))
        """
        band_h = int(self.band_height_frac * self.H)
        band = g[self.H - band_h:self.H, :]  # bottom band
        # Optional: small blur to reduce noise
        band_blur = cv2.GaussianBlur(band, (3, 3), 0)

        best = {
            "score": -1.0,
            "x": None,
            "y": None,
            "w": None,
            "h": None
        }

        for tpl_gray, mask in self.ship_templates:
            for s in self.scales:
                if s != 1.0:
                    tw = max(1, int(tpl_gray.shape[1] * s))
                    th = max(1, int(tpl_gray.shape[0] * s))
                    tpl_s = cv2.resize(tpl_gray, (tw, th), interpolation=cv2.INTER_AREA)
                    msk_s = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST) if mask is not None else None
                else:
                    tpl_s, msk_s = tpl_gray, mask

                # template must fit inside band
                if band_blur.shape[0] < tpl_s.shape[0] or band_blur.shape[1] < tpl_s.shape[1]:
                    continue

                # Use TM_CCORR_NORMED when using a mask (OpenCV limitation)
                method = cv2.TM_CCORR_NORMED if msk_s is not None else cv2.TM_CCOEFF_NORMED
                res = cv2.matchTemplate(band_blur, tpl_s, method, mask=msk_s)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                score = max_val if method in (cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED) else 1.0 - min_val
                if score > best["score"]:
                    best["score"] = score
                    top_left = max_loc if method in (cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED) else min_loc
                    x, y = top_left
                    best["x"] = x + tpl_s.shape[1] / 2.0
                    best["y"] = y + tpl_s.shape[0] / 2.0
                    best["w"] = tpl_s.shape[1]
                    best["h"] = tpl_s.shape[0]

        if best["score"] >= self.min_conf and best["x"] is not None:
            # convert band coords to full image coords (x is the same; y offset if you need it)
            px = float(best["x"])
        else:
            # fallback to previous or center
            px = self.player_x if self.player_x is not None else self.W / 2.0

        # exponential moving average smoothing
        if self.player_x is None:
            self.player_x = px
        else:
            self.player_x = self.smooth * self.player_x + (1.0 - self.smooth) * px

        return float(self.player_x)

    def debug_player_overlay(self, g: np.ndarray, win_name="Player Match"):
        band_h = int(self.band_height_frac * self.H)
        band = g[self.H - band_h:self.H, :].copy()
        px = self.player_x if self.player_x is not None else self.W / 2
        vis = cv2.cvtColor(band, cv2.COLOR_GRAY2BGR)
        cv2.circle(vis, (int(px), int(0.5 * band.shape[0])), 6, (0, 255, 0), 2)
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

    def debug_eggs_overlay(self, g: np.ndarray, eggs):
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
        return vis

    # --- build 1-D risk map over x by projecting to player y ---
    def risk_map(self, eggs):
        """
        eggs: list of (cx, cy, r, score) in crop coords
        returns: 1D risk array length W (float32)
        """
        W = self.W
        x_axis = np.arange(W, dtype=np.float32)
        risk = np.zeros(W, dtype=np.float32)
        py = self.player_y

        # tunables
        max_vert_horizon = getattr(self, "max_vert_horizon", int(0.6 * self.H))
        dist_decay_px = getattr(self, "dist_decay_px", 90.0)
        base_sigma_px = getattr(self, "base_sigma_px", 10.0)
        size_sigma_k = getattr(self, "size_sigma_k", 1.2)
        dist_sigma_k = getattr(self, "dist_sigma_k", 0.02)
        min_score_weight = getattr(self, "min_score_weight", 0.3)
        max_weight_cap = getattr(self, "max_weight_cap", 1.5)  # limits a single egg’s influence

        for (x, y, r, sc) in eggs:
            # guards
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(r) and np.isfinite(sc)):
                continue

            dy = float(py - y)
            if dy <= 0 or dy > max_vert_horizon:
                continue

            # clamp x to image bounds (prevents off-screen centers)
            x = float(np.clip(x, 0.0, W - 1.0))

            # weight = confidence * distance decay
            sc = float(np.clip(sc, 0.0, 1.0))
            w = max(min_score_weight, sc) * np.exp(-dy / max(1e-3, dist_decay_px))
            w = float(np.clip(w, 0.0, max_weight_cap))

            # gaussian width
            sigma = base_sigma_px + size_sigma_k * float(r) + dist_sigma_k * dy
            sigma = float(np.clip(np.nan_to_num(sigma, nan=10.0), 3.0, 1000.0))

            # add contribution
            z = (x_axis - x) / sigma
            gauss = np.exp(-0.5 * (z * z)).astype(np.float32)
            risk += (w * gauss)

        # scale-aware edge penalty (small, relative)
        if W >= 6:
            edge_w = W // 6
            edge = np.linspace(1.0, 0.6, edge_w, dtype=np.float32)  # gentle
            # scale by current median to avoid overpowering
            scale = np.median(risk) if np.any(risk > 0) else 1.0
            risk[:edge_w] += scale * edge[::-1] * 0.25
            risk[-edge_w:] += scale * edge * 0.25

        # sanitize
        risk = np.nan_to_num(risk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # optional: normalize for the debug heatbar (comment out if you want raw)
        denom = float(risk.max())
        if denom > 0:
            risk = risk / denom

        # optional: temporal EMA to reduce flicker (store on self)
        alpha = getattr(self, "risk_ema_alpha", 0.6)  # 0.6 = fairly smooth
        prev = getattr(self, "_risk_prev", None)
        if prev is not None and prev.shape[0] == risk.shape[0]:
            risk = alpha * prev + (1.0 - alpha) * risk
        self._risk_prev = risk.copy()

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

        # player row
        py = self.player_y
        cv2.line(vis, (0, py), (self.W - 1, py), (50, 50, 50), 1)
        cv2.circle(vis, (int(px), py), 6, (0, 255, 0), 2)

        # eggs
        for (cx, cy, r, sc) in eggs:
            cv2.circle(vis, (int(cx), int(cy)), max(2, int(r)), (255, 255, 255), 1)
            cv2.putText(vis, f"{sc:.2f}", (int(cx) + 6, int(cy) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

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

    try:
        while True:
            bgr = get_frame()
            roi_g = bot.preproc(bgr)

            px = bot.detect_player_x(roi_g)
            bot.debug_player_overlay(roi_g)

            eggs = bot.detect_eggs(roi_g)
            # bot.debug_eggs_overlay(roi_g, eggs)

            risk = bot.risk_map(eggs)
            act = bot.choose_action(px, risk)

            # input policy: brief taps with small hold; auto-release
            if act == "left":
                if keys_down != ("left",):
                    if keys_down: release(keys_down)
                    press(("left",), hold_ms=MOVE_HOLD_MS)
                    keys_down = ("left",)
                    last_release = time.time()
            elif act == "right":
                if keys_down != ("right",):
                    if keys_down: release(keys_down)
                    press(("right",), hold_ms=MOVE_HOLD_MS)
                    keys_down = ("right",)
                    last_release = time.time()
            else:
                # neutral
                if keys_down and (time.time() - last_release) > (MOVE_HOLD_MS/1000.0):
                    release(keys_down)
                    keys_down = tuple()

            if show_debug:
                vis = bot.draw_overlay(roi_g, eggs, px, risk)
                cv2.imshow("Dodger Debug (crop + risk)", vis)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                    break

            # simple pacing to ~CAP_FPS
            time.sleep(max(0, 1.0/CAP_FPS - 0.001))
    finally:
        if keys_down:
            release(keys_down)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_dodger()
