"""Intensity-map mode for CaddyBrain Green Reader.

For maps where the COLORS show how steep the slope is (light blue/white = flattest, then green, yellow, orange, red =
steepest) instead of high/low ground, and the black arrows show the downhill direction. Black contour lines, a skewed black
grid and bunkers (beige) are also drawn on these maps and are ignored.

Reading = arrows for direction x color band for size. Output is the same (sx, sy, meta) the normal engine produces, so
putt_engine.solve_putt, the aim chart and the app work unchanged.

The color-to-slope table is a PROVISIONAL, relative scale (the source app prints no legend). A per-course `steep_scale`
(scales.json) and the app's Break amount slider tune it against real putts.
"""
import numpy as np
import putt_engine as pe

INTENSITY_VERSION = "2026-10-05-a"

# provisional % slope per color band, by rank: 0 white, 1 light blue, 2 green, 3 yellow, 4 orange, 5 red, 6 dark red
RANK_PCT = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
MAX_PCT = 8.0


def _rgb01(img_np):
    return np.asarray(img_np)[..., :3].astype(float) / 255.0


def _sat_val(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    return (mx - mn) / np.maximum(mx, 1e-6), mx


def _hue(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4))
    return 60.0 * h


def _erode(m, n=1):
    for _ in range(n):
        k = m.copy()
        k[1:] &= m[:-1]; k[:-1] &= m[1:]; k[:, 1:] &= m[:, :-1]; k[:, :-1] &= m[:, 1:]
        k[1:, 1:] &= m[:-1, :-1]; k[:-1, :-1] &= m[1:, 1:]; k[1:, :-1] &= m[:-1, 1:]; k[:-1, 1:] &= m[1:, :-1]
        m = k
    return m


def _dilate(m, n=1):
    for _ in range(n):
        k = m.copy()
        k[1:] |= m[:-1]; k[:-1] |= m[1:]; k[:, 1:] |= m[:, :-1]; k[:, :-1] |= m[:, 1:]
        k[1:, 1:] |= m[:-1, :-1]; k[:-1, :-1] |= m[1:, 1:]; k[1:, :-1] |= m[:-1, 1:]; k[:-1, 1:] |= m[1:, :-1]
        m = k
    return m


def _pool(mask, k, how="mean"):
    h, w = mask.shape
    a = mask[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k)
    return a.mean((1, 3)) if how == "mean" else a.max((1, 3))


# ---------- 1. the putting surface ----------
def find_green(img_np, k=4):
    """Bool mask (full res) of the green: the biggest blob of heat color plus the thick black outline, with everything it
    encloses (white patches, arrows) filled in, minus beige bunkers. Thin grid lines and the pale arrows outside are not part of it."""
    rgb = _rgb01(img_np)
    h, w, _ = rgb.shape
    sat, mx = _sat_val(rgb)
    hue = _hue(rgb)
    beige = (hue > 20) & (hue < 45) & (sat < 0.40) & (mx > 0.6)           # bunker sand, about (200,175,150)
    heat = (sat > 0.22) & (mx > 0.35) & ~beige
    ink = mx < 0.30
    c = (_pool(heat, k) > 0.5) | (_pool(ink, k) > 0.45)                    # thick outline counts, 2 px grid lines do not
    c = _erode(_dilate(c, 2), 2)
    comps = pe._label(c)
    if not comps:
        return np.ones((h, w), bool)
    best = max(comps, key=len)
    blob = np.zeros(c.shape, bool)
    blob[best[:, 1].astype(int), best[:, 0].astype(int)] = True
    blob = _dilate(_erode(blob, 3), 3)                                    # cut thin attachments
    region = ~pe._outside_region(blob)                                    # fill what the blob encloses
    region = _erode(region, 1)                                            # drop the outline ring itself
    region = np.kron(region, np.ones((k, k), bool))
    out = np.zeros((h, w), bool)
    out[:region.shape[0], :region.shape[1]] = region[:h, :w]
    # take bunkers back out (a bunker touching the green's outline can be joined to it by the black rings)
    bunk = _pool(beige, k) > 0.5
    big = [P for P in pe._label(bunk) if len(P) > 60]
    if big:
        bm = np.zeros(bunk.shape, bool)
        for P in big:
            bm[P[:, 1].astype(int), P[:, 0].astype(int)] = True
        bm = np.kron(_dilate(bm, 2), np.ones((k, k), bool))
        out[:bm.shape[0], :bm.shape[1]] &= ~bm[:h, :w]
        # keep only the biggest piece
        cs = pe._label(_pool(out, k) > 0.5)
        if len(cs) > 1:
            keep = max(cs, key=len)
            km = np.zeros(_pool(out, k).shape, bool)
            km[keep[:, 1].astype(int), keep[:, 0].astype(int)] = True
            km = np.kron(km, np.ones((k, k), bool))
            out[:km.shape[0], :km.shape[1]] &= km[:h, :w]
    return out


# ---------- 2. scale ----------
def _principal_extents(mask):
    ys, xs = np.nonzero(mask)
    p = np.stack([xs, ys], 1).astype(float)
    p -= p.mean(0)
    w, v = np.linalg.eigh(np.cov(p.T))
    t = p @ v
    ext = t.max(0) - t.min(0)
    return float(max(ext)), float(min(ext)), v


def intensity_geom(img_np, depth_yd=None, width_yd=None, k=4):
    """Same dict as putt_engine.auto_geom, in the IMAGE frame (x right, y up the picture), so the app's taps, charts,
    rulers and GPS code work unchanged. Scale (px per yard) comes from the green's long and short sides vs the entered
    depth and width. If the two sides disagree by more than 12% the drawn shape and the numbers don't match: the mean is
    used and `scale_warning` says so."""
    green = find_green(img_np, k)
    ys, xs = np.nonzero(green)
    if len(xs) == 0 or green.all():
        return None
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    source, warn, ppyd = "estimate", None, None
    if depth_yd and width_yd:
        L, S, _ = _principal_extents(green)
        a, b = L / max(depth_yd, width_yd), S / min(depth_yd, width_yd)
        ppyd = float(np.sqrt(a * b))
        source = "entered"
        if abs(a / b - 1) > 0.12:
            warn = (f"the drawn green is {100 * (a / b - 1):+.0f}% off the entered depth/width shape "
                    f"(long side {a:.1f} px/yd, short side {b:.1f} px/yd); the average was used")
    elif depth_yd or width_yd:
        L, S, _ = _principal_extents(green)
        ppyd = L / max(depth_yd or 0, width_yd or 0) if (depth_yd or width_yd) else None
        source = "entered"
    if ppyd is None:
        L, S, _ = _principal_extents(green)
        ppyd = L / 30.0                                                    # assume a 30 yard long green
    ppf = ppyd / 3.0
    front_x = xs[ys >= gy1 - 2].mean()
    back_x = xs[ys <= gy0 + 2].mean()
    y0 = float(gy1)
    return dict(x0=float(gx0), y0=y0, ppx=ppf, ppy=ppf, xmin_ft=0.0, xmax_ft=(gx1 - gx0) / ppf,
                ymin_ft=0.0, ymax_ft=(y0 - gy0) / ppf, width_ft=(gx1 - gx0) / ppf, depth_ft=(y0 - gy0) / ppf,
                source=source, px_per_yd=float(ppyd), labels_read=False, bbox_px=(int(gx0), int(gy0), int(gx1), int(gy1)),
                front_ft=((front_x - gx0) / ppf, 0.0), back_ft=((back_x - gx0) / ppf, (y0 - gy0) / ppf),
                scale_warning=warn, green_mask=green)


# ---------- 3. arrows (direction) ----------
def detect_black_arrows(img_np, green, half=True):
    """Near-black arrows: a thick near-black stroke inside the green. Thin black contour lines and grid lines are removed by
    a morphological opening. Returns arrows in FULL-RES pixel coordinates (same dict as putt_engine.detect_arrows)."""
    a = np.asarray(img_np)[..., :3]
    if half:
        a = a[:a.shape[0] // 2 * 2, :a.shape[1] // 2 * 2]
        a = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2, 3).mean((1, 3))
        g = _pool(green[:a.shape[0] * 2, :a.shape[1] * 2], 2) > 0.5
        f = 2.0
    else:
        g, f = green, 1.0
    dark = a.max(-1) < 75                                                  # near-black, NOT a luminance test: the dark green fill is lum-dark too
    dark &= _erode(g, 3)                                                   # stay off the thick outline
    core = _dilate(_erode(dark, 1), 1)                                     # opening: drops 1-2 px lines
    arrows = []
    for P in pe._label(core):
        n = len(P)
        if n < 18:
            continue
        c = P.mean(0)
        w, v = np.linalg.eigh(np.cov((P - c).T))
        u = v[:, 1]
        t = (P - c) @ u
        length = float(t.max() - t.min())
        if length < 7 or length > 40 or np.sqrt(w[1] / max(w[0], 1e-6)) < 1.25:
            continue                                                        # specks, merged clutter
        asym = abs(t.max() + t.min()) / length
        if asym < 0.05:
            continue
        sign = 1.0 if (t.max() + t.min()) < 0 else -1.0                    # the head is the heavy end
        arrows.append({"c": c * f, "u": u * sign, "n": n * f * f, "length": length * f, "weight": min(1.0, asym / 0.15),
                       "strength": 1.0, "bold": 1.0})
    return arrows


def _spacing(arrows):
    if len(arrows) < 3:
        return 40.0
    c = np.array([a["c"] for a in arrows])
    d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
    np.fill_diagonal(d, 1e9)
    return float(np.median(d.min(1)))


# ---------- 4. colors (steepness) ----------
def color_rank(img_np):
    """Per-pixel steepness rank 0..6 from the color band: white 0, light blue 1, green 2, yellow 3, orange 4, red 5, dark red 6."""
    rgb = _rgb01(img_np)
    sat, mx = _sat_val(rgb)
    hue = _hue(rgb)
    anchors_h = np.array([0.0, 8.0, 25.0, 48.0, 120.0, 190.0, 360.0])      # hue (deg) -> rank, red -> blue
    anchors_r = np.array([5.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0])
    rank = np.interp(hue, anchors_h, anchors_r)
    rank = np.where(hue > 330, 5.0, rank)
    dark_red = ((hue < 12) | (hue > 340)) & (mx < 0.72)
    rank = np.where(dark_red, 6.0, rank)
    rank = np.where((sat < 0.18) & (mx > 0.8), 0.0, rank)                   # white patches
    pale = (sat >= 0.18) & (sat < 0.3) & (hue > 170) & (hue < 230) & (mx > 0.75)   # very pale blue: between white and light blue
    rank = np.where(pale, 0.5, rank)
    return rank


def rank_to_pct(rank):
    return np.interp(rank, np.arange(len(RANK_PCT)), RANK_PCT)


# ---------- 5. the slope field ----------
def build_intensity_field(img_np, geom, steep_scale=1.0, step=4, arrow_sigma_mult=0.7, max_pct=MAX_PCT):
    """Return (sx, sy, meta) like putt_engine.build_slope_field. sx, sy = uphill slope (ft rise per ft) on a coarse grid in
    feet space (x right, y up the picture)."""
    img_np = np.asarray(img_np)
    green = geom.get("green_mask")
    if green is None:
        green = find_green(img_np)
    arrows = detect_black_arrows(img_np, green)
    H, W = img_np.shape[:2]
    gh, gw = H // step, W // step
    # size: block-mean steepness over color pixels (not arrows, lines, outline)
    rgb = _rgb01(img_np)
    sat, mx = _sat_val(rgb)
    ok = green & (mx > 0.35)
    ok = ok & ~_dilate(~(mx > 0.35), 1)
    rank = color_rank(img_np)
    pct = rank_to_pct(rank)
    pct_m, cnt = pe._block_mean(pct, ok, step)
    inside_cells = _pool(green, step) > 0.5
    ignored = ~inside_cells
    pct_m = np.where(cnt < 0.2 * step * step, np.nan, pct_m)
    filled = np.where(np.isnan(pct_m), 0.0, pct_m)
    bad = np.isnan(pct_m) & inside_cells
    pct_f = pe._fill_masked(filled, np.isnan(pct_m)) if np.isnan(pct_m).any() else filled
    pct_f = pe._blur(pct_f)
    pct_f = np.where(inside_cells | (cnt > 0), pct_f, np.nanmedian(pct_m) if np.isfinite(np.nanmedian(pct_m)) else 2.0)
    mag = np.clip(pct_f * steep_scale, 0, max_pct * max(steep_scale, 1.0)) / 100.0

    meta = {"step": step, "ignored": ignored, "arrows": arrows, "n_double": 0, "agreement": None, "color_model": "intensity",
            "red_is_high": True, "used_arrows": bool(arrows), "relief_scale": float(steep_scale)}
    if arrows:
        sp = _spacing(arrows)
        sig = max(arrow_sigma_mult * sp, 8.0)
        vx, vy, conf, _, _ = pe._arrow_field(arrows, (gh, gw), step, sigma=sig)
        # sparse corners: widen the net so every cell has a direction
        vx2, vy2, conf2, _, _ = pe._arrow_field(arrows, (gh, gw), step, sigma=sig * 3.0)
        weak = conf < 0.15
        vx, vy, conf = np.where(weak, vx2, vx), np.where(weak, vy2, vy), np.where(weak, conf2, conf)
        norm = np.hypot(vx, vy)
        coh = np.clip(norm / np.maximum(conf, 1e-9), 0, 1)                 # opposing arrows cancel: near a ridge/valley the roll is gentle
        dx, dy = vx / np.maximum(norm, 1e-9), -vy / np.maximum(norm, 1e-9) # downhill, feet space (y up)
        mag = mag * (0.35 + 0.65 * coh)
        sx, sy = -mag * dx, -mag * dy
        meta["coherence"] = coh
    else:
        sx = sy = np.zeros((gh, gw))
    meta.update({"field": np.stack([sx, sy], axis=-1), **{k: v for k, v in geom.items() if k != "green_mask"}})
    return sx, sy, meta
