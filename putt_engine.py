"""Putt engine for CaddyBrain Green Reader.

Pipeline: heat-map colors -> height field -> slope field -> roll simulation -> aim solver.
Coordinates are feet. x = left to right across the green, y = front to back (up the image).
"""
import numpy as np

G = 32.17          # ft/s^2
ROLL = 5.0 / 7.0   # solid sphere rolling: slope accel = (5/7) * g * slope
CUP_IN = 4.25      # cup diameter, inches


# ---------- 0. Geometry: feet <-> image pixels ----------
def make_geom(img_w, img_h, width_ft, depth_ft):
    """Where the green sits inside the heat-map image (same box the original app used)."""
    return dict(box_x_min=img_w * 0.15, box_x_max=img_w * 0.85, box_y_bottom=img_h * 0.92,
                box_height=img_h * 0.80, width_ft=width_ft, depth_ft=depth_ft)


def ft_to_px(g, x_ft, y_ft):
    px = g["box_x_min"] + x_ft / g["width_ft"] * (g["box_x_max"] - g["box_x_min"])
    py = g["box_y_bottom"] - y_ft / g["depth_ft"] * g["box_height"]
    return px, py


def px_to_ft(g, px, py):
    x = (px - g["box_x_min"]) / (g["box_x_max"] - g["box_x_min"]) * g["width_ft"]
    y = (g["box_y_bottom"] - py) / g["box_height"] * g["depth_ft"]
    return float(np.clip(x, 0, g["width_ft"])), float(np.clip(y, 0, g["depth_ft"]))


# ---------- 1. Image -> slope field ----------
def _blur(a, passes=2):
    """Cheap separable smoothing (3-tap) so the slope field isn't noisy."""
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, mode="edge"), k, "valid"), 1, a)
        a = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 0, a)
    return a


def outline_mask(img_np, grow_px=3, tol=60):
    """True on the solid green boundary line (plus a margin for anti-aliased edge pixels)."""
    a = img_np.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (g > 110) & (g - r > tol) & (g - b > tol)
    for _ in range(grow_px):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    return m


def _outside_region(line):
    """Cells reachable from the image border without crossing the outline = outside the green."""
    out = np.zeros_like(line)
    out[0, :], out[-1, :], out[:, 0], out[:, -1] = True, True, True, True
    out &= ~line
    while True:
        n = out.copy()
        n[1:] |= out[:-1]; n[:-1] |= out[1:]; n[:, 1:] |= out[:, :-1]; n[:, :-1] |= out[:, 1:]
        n &= ~line
        if (n == out).all():
            return out
        out = n


def _shift(a, dy, dx):
    h, w = a.shape
    return np.pad(a, 1)[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def _fill_masked(h, bad):
    """Fill ignored cells from their valid neighbours so no step edge is left behind."""
    h, valid = h.copy(), ~bad
    for _ in range(400):
        if valid.all():
            break
        hv, w = np.where(valid, h, 0.0), valid.astype(float)
        sh = sum(_shift(hv, dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        sw = sum(_shift(w, dy, dx) for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        new = (~valid) & (sw > 0)
        if not new.any():
            break
        h[new] = sh[new] / sw[new]
        valid |= new
    return h


def build_slope_field(img_np, geom, red_is_high=True, relief_ft=1.0, step=4, ignore_outline=True):
    """Return (sx, sy, meta): slope (ft rise per ft) on a coarse grid.

    geom: dict(box_x_min, box_x_max, box_y_bottom, box_height, width_ft, depth_ft).
    relief_ft: assumed elevation difference between the coolest and warmest color on the map.
    Treat this as a calibration knob: bigger = more break.
    ignore_outline: skip the solid green boundary line (and anything outside it) so the edge
    of the map is not mistaken for a slope.
    """
    ignored = np.zeros(img_np[::step, ::step].shape[:2], bool)
    if ignore_outline:
        line = outline_mask(img_np)[::step, ::step]
        if line.mean() > 0.002:                       # an outline is actually present
            outside = _outside_region(line)
            if (~(line | outside)).mean() > 0.2:      # outline is closed; otherwise don't trust the flood fill
                ignored = line | outside
            else:
                ignored = line

    a = img_np.astype(float)[::step, ::step]
    h = (a[..., 0] - a[..., 2]) / 255.0              # warm minus cool, -1..1
    if not red_is_high:
        h = -h
    h = _blur(_fill_masked(h, ignored))
    ok = ~ignored if (~ignored).any() else np.ones_like(ignored)
    h = (h - h[ok].min()) / max(1e-6, h[ok].max() - h[ok].min())   # 0..1 over real map pixels only
    h = np.clip(h, 0, 1) * relief_ft

    px_per_ft_x = (geom["box_x_max"] - geom["box_x_min"]) / geom["width_ft"]
    px_per_ft_y = geom["box_height"] / geom["depth_ft"]
    dh_dpy, dh_dpx = np.gradient(h, step, step)          # per original pixel
    sx = dh_dpx * px_per_ft_x
    sy = -dh_dpy * px_per_ft_y                           # image rows grow downward, y_ft grows upward
    return sx, sy, {"step": step, "field": np.stack([sx, sy], axis=-1), "ignored": ignored, **geom}


def _sample(meta, x, y):
    """Bilinear lookup of the (sx, sy) slope pair at a position in feet."""
    field = meta["field"]
    px = meta["box_x_min"] + x / meta["width_ft"] * (meta["box_x_max"] - meta["box_x_min"])
    py = meta["box_y_bottom"] - y / meta["depth_ft"] * meta["box_height"]
    gx = np.clip(px / meta["step"], 0, field.shape[1] - 1.001)
    gy = np.clip(py / meta["step"], 0, field.shape[0] - 1.001)
    x0, y0 = int(gx), int(gy)
    fx, fy = gx - x0, gy - y0
    return (field[y0, x0] * (1 - fx) * (1 - fy) + field[y0, x0 + 1] * fx * (1 - fy)
            + field[y0 + 1, x0] * (1 - fx) * fy + field[y0 + 1, x0 + 1] * fx * fy)


# ---------- 2. Roll simulation ----------
def simulate(ball, v0, heading, sx, sy, meta, stimp, dt=0.02, max_t=20.0):
    """Roll a ball; return Nx2 array of positions. heading is a unit vector."""
    decel = 18.0 / stimp                       # from 6 ft/s stimpmeter exit speed
    pos = np.array(ball, float)
    vel = np.array(heading, float) * v0
    path = [pos.copy()]
    for _ in range(int(max_t / dt)):
        speed = np.hypot(*vel)
        if speed < 0.05:
            break
        slope = _sample(meta, pos[0], pos[1])
        vel = vel - ROLL * G * slope * dt          # slope pulls the ball downhill
        sp = np.hypot(*vel)
        sp_new = sp - decel * dt                   # friction slows it but never reverses it
        if sp_new <= 0:
            break
        vel = vel * (sp_new / sp)
        pos = pos + vel * dt
        path.append(pos.copy())
    return np.array(path)


def _frame(ball, hole):
    d = np.array(hole, float) - np.array(ball, float)
    dist = np.hypot(*d)
    fwd = d / dist
    left = np.array([-fwd[1], fwd[0]])
    return dist, fwd, left


def _to_frame(path, ball, fwd, left):
    rel = path - np.array(ball)
    return np.stack([rel @ left, rel @ fwd], axis=1)   # (lateral, forward)


def _lateral_at(path_f, dist):
    """Lateral position where the path crosses the hole's forward distance (None if short)."""
    f = path_f[:, 1]
    idx = np.where(f >= dist)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i == 0:
        return path_f[0, 0]
    t = (dist - f[i - 1]) / max(1e-9, f[i] - f[i - 1])
    return path_f[i - 1, 0] + t * (path_f[i, 0] - path_f[i - 1, 0])


def _cut_at_hole(path_f, dist):
    """Trim a (lateral, forward) path at the point where it reaches the hole's distance."""
    idx = np.where(path_f[:, 1] >= dist)[0]
    if len(idx) == 0:
        return path_f
    i = idx[0]
    lat = _lateral_at(path_f, dist)
    return np.vstack([path_f[:i], [[lat, dist]]])


def _speed_for_pace(ball, heading, dist, fwd, left, sx, sy, meta, stimp, past_ft):
    """Bisect launch speed so the ball would stop `past_ft` beyond the hole (along the line)."""
    lo, hi = 0.5, 14.0
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        p = _to_frame(simulate(ball, mid, heading, sx, sy, meta, stimp, dt=0.05), ball, fwd, left)
        if p[:, 1].max() < dist + past_ft:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_putt(ball, hole, sx, sy, meta, stimp, past_ft=1.5):
    """Find the aim angle + speed that brings the ball to the hole, stopping ~past_ft beyond.

    Returns dict with aim_offset_ft (+ = left of hole), aim_side, path_frame (lateral, forward),
    aim_frame, max_break_ft, launch_speed, hit_error_ft.
    """
    dist, fwd, left = _frame(ball, hole)

    def run(theta):
        c, s = np.cos(theta), np.sin(theta)
        heading = fwd * c + left * s            # theta > 0 aims left
        v = _speed_for_pace(ball, heading, dist, fwd, left, sx, sy, meta, stimp, past_ft)
        path = simulate(ball, v, heading, sx, sy, meta, stimp)
        pf = _to_frame(path, ball, fwd, left)
        lat = _lateral_at(pf, dist)
        return (0.0 if lat is None else lat), pf, v

    lo, hi = -0.30, 0.30                       # +/- ~17 degrees
    f_lo, _, _ = run(lo)
    f_hi, _, _ = run(hi)
    if f_lo * f_hi > 0:                        # no bracket: very strong slope or bad calibration
        theta = lo if abs(f_lo) < abs(f_hi) else hi
    else:
        for _ in range(11):
            mid = 0.5 * (lo + hi)
            f_mid, _, _ = run(mid)
            if f_mid * f_lo > 0:
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        theta = 0.5 * (lo + hi)
    err, pf, v = run(theta)
    aim_off = dist * np.tan(theta)
    pf = _cut_at_hole(pf, dist)               # show the roll only up to the cup
    straight_f = _cut_at_hole(_to_frame(
        simulate(ball, v, fwd, sx, sy, meta, stimp), ball, fwd, left), dist)
    ball_a = np.array(ball, float)
    to_world = lambda f: ball_a + f[:, 0:1] * left + f[:, 1:2] * fwd
    return {
        "aim_point_ft": tuple(ball_a + fwd * dist + left * aim_off),
        "path_world": to_world(pf),
        "flat_equiv_ft": v ** 2 / (2 * 18.0 / stimp) - past_ft,   # distance the same stroke rolls on a flat green
        "dist_ft": dist, "theta": theta, "aim_offset_ft": abs(aim_off),
        "aim_side": "Left" if aim_off > 0 else "Right",
        "path_frame": pf, "straight_frame": straight_f,
        "max_break_ft": float(np.abs(pf[:, 0]).max()),
        "launch_speed": v, "hit_error_ft": err,
    }


def classify_break(aim_ft, dist_ft):
    """Effective break factor: aim offset per 10 ft of putt, normalised so 1.0x ~ 1 ft per 10 ft."""
    factor = aim_ft / max(0.1, dist_ft / 10.0)
    if factor < 0.25:
        label = "Mild"
    elif factor < 0.75:
        label = "Moderate"
    else:
        label = "Severe"
    return factor, label
