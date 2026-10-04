"""Putt engine for CaddyBrain Green Reader.

Pipeline: heat-map colors -> height field -> slope field -> roll simulation -> aim solver.
Coordinates are feet. x = left to right across the green, y = front to back (up the image).
"""
import numpy as np

ENGINE_VERSION = "2026-10-04-h"   # app.py checks this so a stale copy of this file is caught
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


# ---- reading the yard-label numbers (tiny template reader; the labels all use one font) ----
DIGITS = {'0': 'AAAAAAAAAAAAAAAAAAAAT6CnaREAAAAAAAB8+f75/tMbAAAAABbm/5JM0P6kAAAAAG3+1QAAMfP0AAAAAKD/pAAAA+D/KwAAALz/jwAAANT+TQAAAMX/hQAAAM/+ZAAAALD/kAAAANb+WQAAAIz9rAAACd//NQAAAEL34wAASvf5AgAAAAPZ/qJd2f63AAAAAABH5/7++scUAAAAAAAAHl51UQkAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '1': 'AAAAAAAAAAAAAAAAAAAAAAAADxUBAAAAAAAAAABX1uVNAAAAAAAABH7y/v5YAAAAAAAAofXP6v9ZAAAAAAAAt5Qj0/5YAAAAAAAADgIA1v5YAAAAAAAAAAAA1/9YAAAAAAAAAAAA1v5YAAAAAAAAAAAA1v5YAAAAAAAAAAAA1v5YAAAAAAAAAAAA1v5YAAAAAAAAAAAA1/5YAAAAAAAAAAAAoME7AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '2': 'AAAAAAAAAAAAAAAAAAAAChETDgUAAAAAAANTt87UxZgXAAAAAFPu8tXI8/ypAAAAABmjYBYOhPr5AwAAAAAEAAAAKvP+BgAAAAAAAAAAaPzYAAAAAAAAAAAe3vNSAAAAAAAAABzB+I8AAAAAAAAAG8X3hQcAAAAAAAAWwvSJAAAAAAAAABS4/MU0ISEiDAAAAJv8/urW1dbXagAAAKPd3d3c3d3dbQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '3': 'AAAAAAAAAAAAAAAAAAAAFUx4gmAXAAAAAABg6f/9+v/pWAAAAAAAwY9RWc794gAAAAAAAAAAAEz98gAAAAAAAAAAAHv/qgAAAAAAAHeQu+eaAAAAAAAAANL4/OBlAAAAAAAAACtIYsf/1wAAAAAAAAAAACjt/wAAAAAAAAAAACXx/wAAAACofzYzYt/+8QAAAAC6/P////roZAAAAAAAUHaFd0sAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '4': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABE//xQAAAAAAAAAAzc/vxPAAAAAAAAAKno8/1PAAAAAAAAXfJV9P5QAAAAAAAG5LAA/v5PAAAAAACs6wgA//xQAAAAAGjzawAA//xcAAAAAO/5v6qt/v7QfQAAAP7////+/v7/wwAAAAMEBAQJ//13AAAAAAAAAAAA//1OAAAAAAAAAAAApqQnAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '5': 'AAAAAAAAAAAAAAAAAAAAPlJTVFVTEwAAAAAA1v//////dgAAAAAA3P6uk5WUNwAAAAAQ5f8CAAAAAAAAAAAe7PgAAAAAAAAAAAAx9/7w8OSjAAAAAAAm6e7u8Pz+tAAAAAAAAAAAAJz8/wAAAAAAAAAAAB3s/wAAAAAAAAAAACfw/gAAAAByl1otWNj9twAAAACJ/v////vaGgAAAAAAXoqQeD8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '6': 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAM8jw8vWfAAAAAAA67vzv07iCAAAAAADU/4YAAAAAAAAAAEb7vgAAAAAAAAAAAJP/elWepWoAAAAAAL/+1evY8/7IAAAAAND+7kIAWvD+MwAAAL//jwAAAMT/cgAAAI//pAAAAMf/ZQAAADv28B8AQu3/HwAAAACf+/Hm9vmfAAAAAAAAbuLx3HoAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '7': 'AAAAAAAAAAAAAAAAADAwMTAvLzAvEAAAAN/f3t7e2t7fbwAAALCysrGwxvb/XAAAAAAAAAAAVPjpCgAAAAAAAAAAtv6TAAAAAAAAAAAk/esdAAAAAAAAAACT/rEAAAAAAAAAABjn+jsAAAAAAAAAAIz+wAAAAAAAAAAACeH8XQAAAAAAAAAAbvvnAAAAAAAAAAAAz/+HAAAAAAAAAAAbsqYWAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '8': 'AAAAAAAAAAAAAAAAAAAAI0FMORUAAAAAAABXx9zZ3K86AAAAACHq+JNtu/vLAAAAAEr82AAANfH3AAAAACTr5zcHb/fLAAAAAAB88dy19ck3AAAAAAAd3Pv3/JQAAAAAAAqi9LWO5+huAAAAAHL3thUASOLuIwAAALn/bgAAAML9XAAAAJT9uC4aRuH8OgAAACvT9NC94vClAAAAAAA6jq63sXIOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', '9': 'AAAAAAAAAAAAAAAAAAAAR4eZaRAAAAAAAACv+Pr0/eAoAAAAAFn67VMwvf7DAAAAAK3/hQAAD+X/GwAAANP/cAAAAM7/SgAAALH9ugAAROz/bgAAAFf4+9K83/X+ZwAAAAB8+//9dMv+RQAAAAAAAAAAANr7AAAAAAAAAAAAfPm2AAAAAABIWHm//+4ZAAAAAADd/v/80kMAAAAAAABSb2xDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'}
_GLYPH_W, _GLYPH_H = 12, 16


def _label_glyphs(a, rows):
    """For each label row, the list of digit images (darkness 0..1, centred on a small canvas)."""
    H, W, _ = a.shape
    lum = a.astype(float).mean(-1)
    sat = a.max(-1) - a.min(-1)
    out = []
    for row in rows:
        y0, y1 = max(0, int(round(row)) - 9), int(round(row)) + 10
        lw = int(0.14 * W)
        ink = (lum[y0:y1, :lw] < 150) & (sat[y0:y1, :lw] < 60)
        cols = np.where(ink.any(0))[0]
        groups = [[cols[0]]] if len(cols) else []
        for c in cols[1:]:
            (groups[-1].append(c) if c - groups[-1][-1] == 1 else groups.append([c]))
        glyphs = []
        for g in groups:
            r = np.where(ink[:, g[0]:g[-1] + 1].any(1))[0]
            dark = 1.0 - lum[y0:y1, :lw][r.min():r.max() + 1, g[0]:g[-1] + 1] / 255.0
            dark = np.clip((dark - 0.2) / 0.7, 0, 1)
            h, w = dark.shape
            if h > _GLYPH_H or w > _GLYPH_W:
                glyphs.append(None)
                continue
            canvas = np.zeros((_GLYPH_H, _GLYPH_W))
            canvas[(_GLYPH_H - h) // 2:(_GLYPH_H - h) // 2 + h, (_GLYPH_W - w) // 2:(_GLYPH_W - w) // 2 + w] = dark
            glyphs.append(canvas)
        out.append(glyphs)
    return out


def _read_label_values(img_np, rows):
    """Whole-number values of the yard labels at these rows, or None if any glyph can't be read."""
    import base64
    tmpl = {c: np.frombuffer(base64.b64decode(b), np.uint8).reshape(_GLYPH_H, _GLYPH_W) / 255.0 for c, b in DIGITS.items()}
    vals = []
    for glyphs in _label_glyphs(img_np, rows):
        txt = ""
        for g in glyphs:
            if g is None:
                return None
            best = min(tmpl, key=lambda c: min(np.sum((np.roll(np.roll(g, dy, 0), dx, 1) - tmpl[c]) ** 2)
                                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)))
            txt += best
        if not txt:
            return None
        vals.append(int(txt))
    return vals


def _read_scale_labels(img_np):
    """Find the yard labels (28, 21, 14, 7, 0, -5 ...) down the side of the map and read them as numbers.

    Works at a standard width, so resized screenshots read the same. Returns (label rows in the image's own pixels, values)
    only if the rows are evenly spaced and the numbers form an even scale ending at 0; otherwise None. Strict on purpose:
    bunker hatching and other texture in the margins can look like text, and a wrong scale is worse than asking.
    """
    try:
        from PIL import Image
        h, w = img_np.shape[:2]
        f = 472.0 / w
        if abs(f - 1) < 0.03:
            f, a = 1.0, img_np
        else:
            a = np.array(Image.fromarray(img_np).resize((472, max(1, int(round(h * f)))), Image.LANCZOS))
        rows = _label_rows(a)
        if len(rows) < 5:
            return None
        vals = _read_label_values(a, rows[:-1])              # top .. 0 (the -5 label uses a smaller font)
        if not vals:
            return None
        m = len(vals)
        if not (vals[-1] == 0 and vals[0] > 0 and all(abs(v - vals[0] * (m - 1 - k) / (m - 1)) <= 0.6 for k, v in enumerate(vals))):
            return None
        sp = np.diff(np.array(rows[:-1], float))
        if sp.min() <= 0 or sp.max() > 1.08 * sp.min():
            return None
        return [r / f for r in rows], vals
    except Exception:
        return None


def auto_geom(img_np, depth_yd=None, width_yd=None):
    """Work out the scale from the map itself, so no per-hole width or depth is needed.

    The yard labels down the side are evenly spaced and the last one is always -5, i.e. 5 yards below
    the 0 line, which fixes pixels per yard (refined by rounding the top label to whole yards).
    x = 0 is the left edge of the green, y = 0 is the map's 0-yard line.

    Maps with no readable yard labels (some print "D 29 yd / W 19 yd" instead) can be given depth_yd and/or width_yd by
    hand (either one is enough); the green's outline then spans exactly that, and y = 0 is the front (bottom) of the green.
    """
    inside, _ = inside_mask(img_np)
    ys, xs = np.nonzero(inside)
    if len(xs) == 0 or inside.all():
        return None
    gx0, gx1, gy0, gy1 = xs.min(), xs.max(), ys.min(), ys.max()
    lab = _read_scale_labels(img_np)
    labels_read = False
    source = "yard labels"
    if lab:
        rows, vals = lab
        top, y0 = rows[0], rows[-2]
        ppx = ppy = (y0 - top) / vals[0] / 3.0
        labels_read = True
    elif depth_yd or width_yd:                                   # typed in: depth, width, or both (one is enough)
        source = "entered"
        y0 = float(gy1)
        ppy = (gy1 - gy0) / (depth_yd * 3.0) if depth_yd else None
        ppx = (gx1 - gx0) / (width_yd * 3.0) if width_yd else None
        ppx, ppy = ppx or ppy, ppy or ppx                        # with only one, assume square pixels
    else:
        source = "estimate"
        y0 = float(gy1)
        ppx = ppy = (gy1 - gy0) / 28.0 / 3.0                     # assume a 28 yard deep green
    front_x = xs[ys >= gy1 - 2].mean()                          # front-most and back-most points of the green
    back_x = xs[ys <= gy0 + 2].mean()
    return dict(x0=float(gx0), y0=float(y0), ppx=ppx, ppy=ppy, xmin_ft=0.0, xmax_ft=(gx1 - gx0) / ppx,
                ymin_ft=(y0 - gy1) / ppy, ymax_ft=(y0 - gy0) / ppy, width_ft=(gx1 - gx0) / ppx,
                depth_ft=(gy1 - gy0) / ppy, source=source, px_per_yd=float(ppy * 3.0), labels_read=labels_read,
                bbox_px=(int(gx0), int(gy0), int(gx1), int(gy1)),
                front_ft=((front_x - gx0) / ppx, (y0 - gy1) / ppy),
                back_ft=((back_x - gx0) / ppx, (y0 - gy0) / ppy))


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
    """True on the solid green boundary line (a dark, muted green such as (100,145,100) or (76,102,65)) plus a small
    margin. The greens inside the heat colors are much lighter, so the luminance cap keeps them out."""
    a = img_np.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    lum = a.mean(-1)
    return _grow((g - r > 8) & (g - b > 22) & (lum < 125), grow_px)


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


def _color_region(img_np, k=3, heat_sat=0.22, close=1):
    """The putting surface found by color: the biggest connected blob of heat color plus the dark ink around and inside it
    (outline, arrows, dashed lines), closed over small gaps, with everything it encloses filled in. Including the dark
    outline keeps a pale middle band in the heat colors from splitting the green in two. Used when the outline can't be
    traced on its own."""
    h, w, _ = img_np.shape
    rgb = img_np[:h // k * k, :w // k * k].astype(float) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    heat = (sat > heat_sat).reshape(h // k, k, w // k, k).mean((1, 3)) > 0.5
    ink = (mx < 0.55).reshape(h // k, k, w // k, k).mean((1, 3)) > 0.15   # thin dark lines only half-fill a cell
    colorful = heat | ink
    comps = _label(colorful)
    if not comps:
        return None
    blob = np.zeros(colorful.shape, bool)
    best = max(comps, key=len)
    blob[best[:, 1].astype(int), best[:, 0].astype(int)] = True
    blob = ~_grow(~_grow(blob, close), close)            # close small gaps in the outline
    region = ~_outside_region(blob)                      # everything the blob encloses, holes included
    region = np.kron(region, np.ones((k, k), bool))
    return np.pad(region, ((0, h - region.shape[0]), (0, w - region.shape[1])), mode="edge")


def inside_mask(img_np):
    """Full-res bool mask of the putting surface (inside the solid outline, outline excluded)."""
    line = outline_mask(img_np)
    h, w = line.shape
    if line.mean() < 0.002:
        reg = _color_region(img_np)
        return (reg if reg is not None else np.ones((h, w), bool)), line
    k = 3
    lc = line[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).max((1, 3))   # max-pool so the flood can't leak
    outside_c = _outside_region(_grow(lc, 1))                # thicken the barrier so a small gap in the outline can't leak
    outside = np.kron(_grow(outside_c, 1), np.ones((k, k), bool))   # then take back the extra ring we added
    outside = np.pad(outside, ((0, h - outside.shape[0]), (0, w - outside.shape[1])), mode="edge")
    inside = ~outside & ~line
    if inside.mean() < 0.08:                      # outline isn't closed (the flood fill leaked): trace the green by color instead
        reg = _color_region(img_np, heat_sat=0.12, close=2)      # tolerant settings: a pale middle band must not split the green
        if reg is not None and 0.05 < reg.mean() < 0.9:
            return reg, line
        return ~line, line
    ic = inside[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).min((1, 3))
    rgb = img_np[:h // k * k, :w // k * k].astype(float) / 255.0
    sat = ((rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-6)).reshape(h // k, k, w // k, k).mean((1, 3))
    # the putting surface is the enclosed region full of heat color; a picture frame or the white margin around the
    # green can be a bigger enclosed region but has almost no color
    best = max(_label(ic), key=lambda P: (int((sat[P[:, 1].astype(int), P[:, 0].astype(int)] > 0.25).sum()), len(P)))
    keep = np.zeros(ic.shape, bool)
    keep[best[:, 1].astype(int), best[:, 0].astype(int)] = True
    keep = np.pad(np.kron(keep, np.ones((k, k), bool)), ((0, h - keep.shape[0] * k), (0, w - keep.shape[1] * k)), mode="edge")
    inside = inside & keep
    inside = inside | (~inside & ~_outside_region(inside))    # fill pinholes (arrows over yellow-green can look like outline)
    # sanity check: the putting surface is mostly heat color. If it isn't (a frame or white margin leaked in), trace by color
    rgb_all = img_np.astype(float) / 255.0
    sat_all = (rgb_all.max(-1) - rgb_all.min(-1)) / np.maximum(rgb_all.max(-1), 1e-6)
    if (sat_all[inside] > 0.25).mean() < 0.5:
        reg = _color_region(img_np)
        if reg is not None and 0.05 < reg.mean() < 0.9:
            return reg, line
    return inside, line


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
    f = max(1.0, max(lum.shape) / 480.0)                   # size factor: maps ~480 px tall are the baseline
    # strokes are the slate-gray pixels that are clearly darker than the heat color around them,
    # which works on red, pale-green and blue areas alike
    core_all = (lum < 0.72 * _box_mean(lum, 9)) & (lum < 135) & inside
    # dashed guide lines (some maps draw a black crosshair through the green): columns or rows that are dark over a big
    # share of the green. They are masked out of the colors like arrows, but are not mistaken for arrows themselves.
    core = core_all.copy()
    band = int(round(5 * f))
    col_frac = core_all.sum(0) / np.maximum(inside.sum(0), 1)
    row_frac = core_all.sum(1) / np.maximum(inside.sum(1), 1)
    for x in np.nonzero((col_frac > 0.35) & (inside.sum(0) > 0.3 * inside.sum(0).max()))[0]:
        core[:, max(0, x - band):x + band + 1] = False
    for y in np.nonzero((row_frac > 0.35) & (inside.sum(1) > 0.3 * inside.sum(1).max()))[0]:
        core[max(0, y - band):y + band + 1, :] = False
    near_edge = _grow(~inside, 1)
    arrows = []
    for P in _label(core):
        n = len(P)
        if near_edge[P[:, 1].astype(int), P[:, 0].astype(int)].any():
            continue
        if n < 10 * f:
            continue
        c = P.mean(0)
        w, v = np.linalg.eigh(np.cov((P - c).T))
        u = v[:, 1]
        t = (P - c) @ u
        length = t.max() - t.min()
        if not (6 * f <= length <= 40 * f) or np.sqrt(w[1] / max(w[0], 1e-6)) < 1.3:
            continue                                  # text, specks, or merged clutter
        asym = abs(t.max() + t.min()) / length               # an arrowhead puts extra ink at one end; dashes and specks don't
        if asym < 0.06:
            continue
        sign = 1.0 if (t.max() + t.min()) < 0 else -1.0   # mass sits toward the head
        arrows.append({"c": c, "u": u * sign, "n": n, "length": length, "weight": min(1.0, asym / 0.2)})
    if arrows:
        med = np.median([x["n"] for x in arrows])
        for x in arrows:
            x["strength"] = 2.0 if x["n"] > 1.2 * med else 1.0
    return arrows, _grow(core_all, 1)


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
        w = np.exp(-((X - a["c"][0]) ** 2 + (Y - a["c"][1]) ** 2) / (2 * sigma ** 2)) * a.get("weight", 1.0)
        vx += w * a["u"][0]; vy += w * a["u"][1]; conf += w; st += w * a["strength"]
    return vx, vy, conf, st / np.maximum(conf, 1e-9)


def build_slope_field(img_np, geom, red_is_high=None, relief_ft=1.0, step=4, ignore_outline=True,
                      use_arrows=True, arrow_trust=0.7, double_boost=0.3, max_grade=None, p90_target=None):
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
    h_full = (a[..., 0] - a[..., 2]) / 255.0              # warm minus cool, -1..1 (red-to-pale-green-to-blue maps)
    rgb = a / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    d_ = np.maximum(mx - mn, 1e-6)
    hue = 60.0 * np.where(mx == rgb[..., 0], ((rgb[..., 1] - rgb[..., 2]) / d_) % 6,
                          np.where(mx == rgb[..., 1], (rgb[..., 2] - rgb[..., 0]) / d_ + 2, (rgb[..., 0] - rgb[..., 1]) / d_ + 4))
    yellow = valid & (sat > 0.5) & (mx > 0.6) & (hue > 35) & (hue < 75)
    color_model = "r-b"
    if yellow.sum() > 0.02 * max(valid.sum(), 1):         # a full rainbow scale: red and yellow look alike in r-b, so use hue
        h_full = -np.where(hue > 300, hue - 360, hue) / 240.0
        valid = valid & (sat > 0.08)
        color_model = "hue"
    h, cnt = _block_mean(h_full, valid, step)
    ignored = cnt < 0.3 * step * step
    h = _blur(_fill_masked(h, ignored))
    ok = ~ignored if (~ignored).any() else np.ones_like(ignored)
    h = np.clip((h - h[ok].min()) / max(1e-6, h[ok].max() - h[ok].min()), 0, 1) * relief_ft   # red = high

    ppx, ppy = geom["ppx"], geom["ppy"]                                  # px per ft
    dh_dpy, dh_dpx = np.gradient(h, step, step)
    cx_, cy_ = dh_dpx * ppx, -dh_dpy * ppy                # uphill gradient, feet space (red = high)

    meta = {"step": step, "ignored": ignored, "arrows": arrows, "n_double": 0, "agreement": None, "color_model": color_model,
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
        if color_model == "hue":
            for x in arrows:
                x["strength"] = 1.0
        meta["n_double"] = int(sum(x["strength"] > 1 for x in arrows))

    if not meta["red_is_high"]:
        cx_, cy_ = -cx_, -cy_                             # flipped color scale
    dcx, dcy = -cx_, -cy_                                 # downhill from colors
    mag = np.hypot(dcx, dcy)

    if use_arrows and arrows:
        vx, vy, conf, mean_st = _arrow_field(arrows, h.shape, step, sigma=25.0 * max(1.0, max(img_np.shape[:2]) / 480.0))
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

    meta["relief_scale"] = 1.0
    if p90_target:
        # Some maps spread the full red-to-blue range across a narrow band, which reads as slopes far steeper than any real
        # green (a steep 90th percentile). Scale such maps down so that 90% of the green is no steeper than p90_target.
        # Maps that are already gentle are left alone (this only ever reduces).
        mag0 = np.hypot(sx, sy)[~ignored]
        if mag0.size:
            p90 = float(np.percentile(mag0, 90))
            if p90 > p90_target:
                meta["relief_scale"] = p90_target / p90
                sx, sy = sx * meta["relief_scale"], sy * meta["relief_scale"]
    if max_grade:
        # soft limit on steepness (rise per run, so 0.04 = 4%): smooth below the limit, never above it. Fast-changing colors
        # (a narrow band between red and blue) would otherwise read as slopes no real green has.
        mag_ = np.hypot(sx, sy)
        k_ = 1.0 / (1.0 + (mag_ / max_grade) ** 4) ** 0.25
        sx, sy = sx * k_, sy * k_
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


def _smooth(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)


def short_putt_weight(dist_ft, short_ft=3.0, full_at_ft=6.0):
    """1.0 for putts of `short_ft` or less, easing to 0.0 at `full_at_ft` and beyond (long putts are never touched)."""
    return 1.0 - _smooth((dist_ft - short_ft) / max(full_at_ft - short_ft, 1e-6))


def short_putt_factor(dist_ft, at_short=1.0, short_ft=3.0, full_at_ft=6.0):
    """How much of the sideways slope to keep: `at_short` for putts of `short_ft` or less, easing smoothly up to 1.0
    (no change) at `full_at_ft` and beyond."""
    w = short_putt_weight(dist_ft, short_ft, full_at_ft)
    return at_short * w + (1.0 - w)


def solve_putt(ball, hole, sx, sy, meta, stimp, past_ft=1.5, short_break=1.0, short_cap=None):
    """Find the aim angle + speed that brings the ball to the hole, stopping ~past_ft beyond.

    Returns dict with aim_offset_ft (+ = left of hole), aim_side, path_frame (lateral, forward),
    aim_frame, max_break_ft, launch_speed, hit_error_ft.
    """
    dist, fwd, left = _frame(ball, hole)
    w_ = short_putt_weight(dist, 6.0, 8.0)           # slope ceiling: full strength through 6 ft, gone by 8 ft
    if short_cap and w_ > 0.001:                     # only ever bites on slopes steeper than a real green has
        fld = meta["field"]
        k_ = 1.0 / (1.0 + (np.hypot(fld[..., 0], fld[..., 1]) / short_cap) ** 4) ** 0.25
        meta = dict(meta, field=(1.0 - w_) * fld + w_ * fld * k_[..., None])
    g_ = short_putt_factor(dist, short_break)
    if g_ < 0.999:                                   # tighten the break on short putts: shrink only the cross-slope
        fld = meta["field"]
        cross = fld @ left
        meta = dict(meta, field=fld - (1.0 - g_) * cross[..., None] * left)

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
        "launch_speed": v, "hit_error_ft": err, "reached": reached, "short_factor": g_,
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
