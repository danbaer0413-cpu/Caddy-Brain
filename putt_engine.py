"""Putt engine for CaddyBrain Green Reader.

Pipeline: heat-map colors -> height field -> slope field -> roll simulation -> aim solver.
Coordinates are feet. x = left to right across the green, y = front to back (up the image).
"""
import numpy as np

G = 32.17          # ft/s^2
ROLL = 5.0 / 7.0   # solid sphere rolling: slope accel = (5/7) * g * slope
CUP_IN = 4.25      # cup diameter, inches


# ---------- 0. Geometry: feet <-> image pixels ----------
# geom: x0, y0 = pixel of (0 ft, 0 ft); ppx, ppy = pixels per foot; xmin/xmax/ymin/ymax_ft = bounds of the green.
def make_geom(img_w, img_h, width_ft, depth_ft):
    """Manual geometry (old method): green assumed to fill a fixed box inside the image."""
    bx0, bx1, by, bh = img_w * 0.15, img_w * 0.85, img_h * 0.92, img_h * 0.80
    return dict(x0=bx0, y0=by, ppx=(bx1 - bx0) / width_ft, ppy=bh / depth_ft, xmin_ft=0.0, xmax_ft=width_ft,
                ymin_ft=0.0, ymax_ft=depth_ft, width_ft=width_ft, depth_ft=depth_ft, source="manual")


def _label_rows(img_np):
    """Vertical centres of the yard labels (29, 22, 15, 7, 0, -5 ...) in the left/right margins."""
    a = img_np.astype(int)
    H, W, _ = a.shape
    dark = (a.mean(-1) < 110) & ((a.max(-1) - a.min(-1)) < 60)
    best = []
    for x0, x1 in ((0, int(0.14 * W)), (int(0.86 * W), W)):
        rows = [r for r in np.nonzero(dark[:, x0:x1].sum(1))[0] if r > 0.15 * H]    # skip the header bar
        groups = []
        for r in rows:
            if groups and r - groups[-1][-1] <= 3:
                groups[-1].append(r)
            else:
                groups.append([r])
        c = [(g[0] + g[-1]) / 2 for g in groups if len(g) >= 6]
        if len(c) > len(best):
            best = c
    return best


def auto_geom(img_np):
    """Work out the scale from the map itself, so no per-hole width or depth is needed.

    The yard labels down the side are evenly spaced and the last one is always -5, i.e. 5 yards below
    the 0 line, which fixes pixels per yard (refined by rounding the top label to whole yards).
    x = 0 is the left edge of the green, y = 0 is the map's 0-yard line. Pixels are assumed square.
    """
    inside, _ = inside_mask(img_np)
    ys, xs = np.nonzero(inside)
    if len(xs) == 0 or inside.all():
        return None
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    labels = _label_rows(img_np)
    source = "yard labels"
    if len(labels) >= 5:
        top, y0, ym5 = labels[0], labels[-2], labels[-1]
        ppy0 = (ym5 - y0) / 5.0                                  # px per yard from the -5 label
        n_yd = max(1, round((y0 - top) / ppy0))                  # top label is a whole number of yards
        ppy_yd = (y0 - top) / n_yd
        if abs(ppy_yd / ppy0 - 1) > 0.15:
            ppy_yd = ppy0
    else:
        source = "estimate"
        ppy_yd, y0 = (gy1 - gy0) / 28.0, float(gy1)              # assume a 28 yard deep green
    ppf = ppy_yd / 3.0
    return dict(x0=float(gx0), y0=float(y0), ppx=ppf, ppy=ppf, xmin_ft=0.0, xmax_ft=(gx1 - gx0) / ppf,
                ymin_ft=(y0 - gy1) / ppf, ymax_ft=(y0 - gy0) / ppf, width_ft=(gx1 - gx0) / ppf,
                depth_ft=(gy1 - gy0) / ppf, source=source, px_per_yd=float(ppy_yd))


def default_markers(g):
    """Ball low and centred on the green, hole about two thirds of the way up."""
    xc = 0.5 * (g["xmin_ft"] + g["xmax_ft"])
    span = g["ymax_ft"] - g["ymin_ft"]
    return (xc, g["ymin_ft"] + 0.2 * span), (xc, g["ymin_ft"] + 0.6 * span)


def ft_to_px(g, x_ft, y_ft):
    return g["x0"] + x_ft * g["ppx"], g["y0"] - y_ft * g["ppy"]


def px_to_ft(g, px, py):
    x, y = (px - g["x0"]) / g["ppx"], (g["y0"] - py) / g["ppy"]
    return float(np.clip(x, g["xmin_ft"], g["xmax_ft"])), float(np.clip(y, g["ymin_ft"], g["ymax_ft"]))


# ---------- 1. Image -> slope field ----------
def _blur(a, passes=2):
    """Cheap separable smoothing (3-tap) so the slope field isn't noisy."""
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        a = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, mode="edge"), k, "valid"), 1, a)
        a = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 1, mode="edge"), k, "valid"), 0, a)
    return a


def _grow(m, n):
    for _ in range(n):
        k = m.copy()
        k[1:] |= m[:-1]; k[:-1] |= m[1:]; k[:, 1:] |= m[:, :-1]; k[:, :-1] |= m[:, 1:]
        m = k
    return m


def outline_mask(img_np, grow_px=2):
    """True on the solid green boundary line (muted green, e.g. ~(100,145,100)) plus a small margin.
    The pale-green middle of the heat colors is much lighter, so a luminance cap keeps it out."""
    a = img_np.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a.mean(-1)
    return _grow((g - r > 30) & (g - b > 30) & (lum < 175), grow_px)


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


def inside_mask(img_np):
    """Full-res bool mask of the putting surface (inside the solid outline, outline excluded)."""
    line = outline_mask(img_np)
    h, w = line.shape
    if line.mean() < 0.002:
        return np.ones((h, w), bool), line
    k = 3
    lc = line[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).max((1, 3))   # max-pool so the flood can't leak
    outside = np.kron(_outside_region(lc), np.ones((k, k), bool))
    outside = np.pad(outside, ((0, h - outside.shape[0]), (0, w - outside.shape[1])), mode="edge")
    inside = ~outside & ~line
    if inside.mean() < 0.08:                      # outline isn't closed; don't trust the flood fill
        return ~line, line
    ic = inside[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).min((1, 3))
    best = max(_label(ic), key=len)               # the putting surface is the largest enclosed region
    keep = np.zeros(ic.shape, bool)
    keep[best[:, 1].astype(int), best[:, 0].astype(int)] = True
    keep = np.pad(np.kron(keep, np.ones((k, k), bool)), ((0, h - keep.shape[0] * k), (0, w - keep.shape[1] * k)), mode="edge")
    return inside & keep, line


def _label(mask):
    """8-connected components as lists of (x, y) points. Masks are small, so plain python is fine."""
    H, W = mask.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, pts = [(y, x)], []
        while stack:
            cy, cx = stack.pop()
            pts.append((cx, cy))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        comps.append(np.array(pts, float))
    return comps


def _box_mean(a, r):
    """Mean over a (2r+1) square window, via an integral image."""
    p = np.pad(a, r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    h, w = a.shape
    tot = c[k:k + h, k:k + w] - c[:h, k:k + w] - c[k:k + h, :w] + c[:h, :w]
    return tot / (k * k)


def detect_arrows(img_np, inside):
    """Find the slate-gray slope arrows drawn on the map.

    Returns (arrows, arrow_pixels). Each arrow: c (x, y centre in px), u (unit direction of the
    arrowhead in px, y down), strength (1 = single head, 2 = double head), length (px).
    Direction comes from the shaft axis (PCA); the head end is the heavier end of the stroke.
    """
    lum = img_np.astype(float).mean(-1)
    # strokes are the slate-gray pixels that are clearly darker than the heat color around them,
    # which works on red, pale-green and blue areas alike
    core = (lum < 0.72 * _box_mean(lum, 9)) & (lum < 135) & inside
    arrows = []
    for P in _label(core):
        n = len(P)
        if n < 10:
            continue
        c = P.mean(0)
        w, v = np.linalg.eigh(np.cov((P - c).T))
        u = v[:, 1]
        t = (P - c) @ u
        length = t.max() - t.min()
        if not (6 <= length <= 40) or np.sqrt(w[1] / max(w[0], 1e-6)) < 1.3:
            continue                                  # text, specks, or merged clutter
        sign = 1.0 if (t.max() + t.min()) < 0 else -1.0   # mass sits toward the head
        arrows.append({"c": c, "u": u * sign, "n": n, "length": length})
    if arrows:
        med = np.median([x["n"] for x in arrows])
        for x in arrows:
            x["strength"] = 2.0 if x["n"] > 1.2 * med else 1.0
    return arrows, _grow(core, 1)


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


def _block_mean(a, valid, step):
    H, W = a.shape
    h2, w2 = H // step, W // step
    a = a[:h2 * step, :w2 * step].reshape(h2, step, w2, step)
    v = valid[:h2 * step, :w2 * step].reshape(h2, step, w2, step)
    cnt = v.sum((1, 3))
    return (a * v).sum((1, 3)) / np.maximum(cnt, 1), cnt


def _arrow_field(arrows, shape, step, sigma):
    """Smooth vector field from sparse arrows: (vx, vy, confidence, mean strength) per coarse cell."""
    H, W = shape
    X, Y = np.meshgrid((np.arange(W) + 0.5) * step, (np.arange(H) + 0.5) * step)
    vx, vy, conf, st = (np.zeros((H, W)) for _ in range(4))
    for a in arrows:
        w = np.exp(-((X - a["c"][0]) ** 2 + (Y - a["c"][1]) ** 2) / (2 * sigma ** 2))
        vx += w * a["u"][0]; vy += w * a["u"][1]; conf += w; st += w * a["strength"]
    return vx, vy, conf, st / np.maximum(conf, 1e-9)


def build_slope_field(img_np, geom, red_is_high=None, relief_ft=1.0, step=4, ignore_outline=True,
                      use_arrows=True, arrow_trust=0.7, double_boost=0.3):
    """Return (sx, sy, meta): slope (ft rise per ft) on a coarse grid.

    Colors give the slope size; the printed arrows give the direction (and double heads add steepness).
    red_is_high=None means "decide from the arrows" (falls back to True if there are none).
    ignore_outline skips the solid green boundary (and everything outside it); arrow pixels are
    always skipped for the color reading so the dark strokes aren't mistaken for color.
    """
    img_np = np.asarray(img_np)
    inside, _ = inside_mask(img_np) if ignore_outline else (np.ones(img_np.shape[:2], bool), None)
    arrows, arrow_px = detect_arrows(img_np, inside)
    valid = inside & ~arrow_px

    a = img_np.astype(float)
    h_full = (a[..., 0] - a[..., 2]) / 255.0              # warm minus cool, -1..1
    h, cnt = _block_mean(h_full, valid, step)
    ignored = cnt < 0.3 * step * step
    h = _blur(_fill_masked(h, ignored))
    ok = ~ignored if (~ignored).any() else np.ones_like(ignored)
    h = np.clip((h - h[ok].min()) / max(1e-6, h[ok].max() - h[ok].min()), 0, 1) * relief_ft   # red = high

    ppx, ppy = geom["ppx"], geom["ppy"]                                  # px per ft
    dh_dpy, dh_dpx = np.gradient(h, step, step)
    cx_, cy_ = dh_dpx * ppx, -dh_dpy * ppy                # uphill gradient, feet space (red = high)

    meta = {"step": step, "ignored": ignored, "arrows": arrows, "n_double": 0, "agreement": None,
            "red_is_high": True if red_is_high is None else bool(red_is_high), "used_arrows": False}

    if arrows:
        # arrow downhill direction in feet space, one value per arrow, vs the color-based downhill there
        cos = []
        for ar in arrows:
            gx, gy = int(ar["c"][0] / step), int(ar["c"][1] / step)
            gx, gy = min(gx, h.shape[1] - 1), min(gy, h.shape[0] - 1)
            dc = np.array([-cx_[gy, gx], -cy_[gy, gx]])
            da = np.array([ar["u"][0] / ppx, -ar["u"][1] / ppy])
            if np.hypot(*dc) > 1e-6:
                cos.append((dc @ da) / (np.hypot(*dc) * np.hypot(*da)))
        raw = float(np.mean(np.array(cos) > 0)) if cos else None
        if red_is_high is None and raw is not None:
            meta["red_is_high"] = raw >= 0.5
        meta["agreement"] = None if raw is None else (raw if meta["red_is_high"] else 1 - raw)
        meta["n_double"] = int(sum(x["strength"] > 1 for x in arrows))

    if not meta["red_is_high"]:
        cx_, cy_ = -cx_, -cy_                             # flipped color scale
    dcx, dcy = -cx_, -cy_                                 # downhill from colors
    mag = np.hypot(dcx, dcy)

    if use_arrows and arrows:
        vx, vy, conf, mean_st = _arrow_field(arrows, h.shape, step, sigma=25.0)
        dax, day = vx / ppx, -vy / ppy                    # arrow downhill, feet space
        dn = np.maximum(np.hypot(dax, day), 1e-9)
        dax, day = dax / dn, day / dn
        c = np.clip(conf / 1.0, 0, 1) * arrow_trust
        cdx, cdy = dcx / np.maximum(mag, 1e-9), dcy / np.maximum(mag, 1e-9)
        bx, by = c * dax + (1 - c) * cdx, c * day + (1 - c) * cdy
        bn = np.hypot(bx, by)
        bx, by = np.where(bn > 1e-6, bx / np.maximum(bn, 1e-9), cdx), np.where(bn > 1e-6, by / np.maximum(bn, 1e-9), cdy)
        boost = 1 + double_boost * (mean_st - 1) * np.clip(conf, 0, 1)
        sx, sy = -mag * boost * bx, -mag * boost * by
        meta["used_arrows"] = True
    else:
        sx, sy = cx_, cy_

    meta.update({"field": np.stack([sx, sy], axis=-1), **geom})
    return sx, sy, meta


def _sample(meta, x, y):
    """Bilinear lookup of the (sx, sy) slope pair at a position in feet."""
    field = meta["field"]
    px = meta["x0"] + x * meta["ppx"]
    py = meta["y0"] - y * meta["ppy"]
    gx = np.clip(px / meta["step"] - 0.5, 0, field.shape[1] - 1.001)
    gy = np.clip(py / meta["step"] - 0.5, 0, field.shape[0] - 1.001)
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
    lo, hi = 0.5, 22.0
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
        if lat is None:                      # ball stalls short of the hole: use where it ended up for the sign
            return pf[-1, 0], pf, v
        return lat, pf, v

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
    reached = bool(pf[:, 1].max() >= dist)
    if not reached:
        err = float("inf")                   # never got to the hole: flag it instead of reporting a fake 0 miss
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
        "launch_speed": v, "hit_error_ft": err, "reached": reached,
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
