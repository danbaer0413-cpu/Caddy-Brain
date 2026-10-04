import os
import re
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

try:
    import putt_engine as pe
    import gps_mode as gm
except ImportError as _e:                              # a code file wasn't uploaded next to app.py
    st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="centered")
    st.error(f"A code file is missing from your repository ({_e}). Upload putt_engine.py and gps_mode.py into the "
             "same folder as app.py, then reboot the app.")
    st.stop()

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="centered")

# the three code files must be the same release; a stale copy of putt_engine.py or gps_mode.py gives confusing errors
import inspect
import importlib
REQUIRED_ENGINE = "2026-10-04-i"


def _engine_ok():
    return (hasattr(pe, "auto_geom") and "depth_yd" in inspect.signature(pe.auto_geom).parameters
            and getattr(pe, "ENGINE_VERSION", "") == REQUIRED_ENGINE)


if not _engine_ok() or not hasattr(gm, "calibrate"):
    # Streamlit can keep an older copy of a module in memory after the file on disk has been replaced. Reload from disk first.
    try:
        pe = importlib.reload(pe)
        gm = importlib.reload(gm)
    except Exception:
        pass
_needs = []
if not _engine_ok():
    _needs.append("putt_engine.py")
if not hasattr(gm, "calibrate"):
    _needs.append("gps_mode.py")
if _needs:
    _here = os.path.dirname(os.path.abspath(globals().get("__file__", ".")))
    _where = getattr(pe, "__file__", "unknown")
    try:
        _sig = str(inspect.signature(pe.auto_geom))
    except Exception:
        _sig = " (not found)"
    try:
        _lines = sum(1 for _ in open(_where))
    except Exception:
        _lines = "?"
    _near = sorted(f for f in os.listdir(_here) if f.lower().startswith(("putt", "gps", "app", "requirements")))
    st.error("This app.py needs the latest " + " and ".join(_needs) + ". The copy in your repository is older. "
             "Upload the latest version of each file into the same folder as app.py, then reboot the app.")
    st.code(f"putt_engine.py loaded from: {_where} ({_lines} lines)\n"
            f"auto_geom{_sig}\n"
            f"putt_engine version: {getattr(pe, 'ENGINE_VERSION', 'old (no version tag)')}\n"
            f"files next to app.py: {_near}\n\n"
            f"expected: putt_engine.py with version {REQUIRED_ENGINE}")
    st.stop()

def discover_courses(root="assets"):
    """Every folder under assets/ that holds files named <hole>_Heat.png (or .jpg/.jpeg/.webp, any capitalization) is a
    course, named after the folder. Folders nested inside other folders are found too, so an extra level from
    unzipping or uploading doesn't hide a course. Returns {display name: {hole number: image path}}."""
    found = {}
    for dirpath, _dirs, files in sorted(os.walk(root)):
        holes = {}
        for fn in files:
            m = re.match(r"^(\d+)_heat\.(png|jpe?g|webp)$", fn, re.I)
            if m:
                holes[int(m.group(1))] = os.path.join(dirpath, fn)
        if holes:
            name = os.path.basename(dirpath).replace("_", " ").title()
            for hole, path in holes.items():
                found.setdefault(name, {}).setdefault(hole, path)    # same name twice: keep the shallower copy
    return found


COURSES = discover_courses()


def load_scales(course_name, hole):
    """Depth/width in yards for maps without yard labels, from an optional scales.json in the course's folder:
    {"1": {"depth_yd": 29, "width_yd": 19}, "2": {"depth_yd": 29, "width_yd": 24}}"""
    import json
    folders = sorted({os.path.dirname(p) for p in COURSES.get(course_name, {}).values()}, key=len)
    for folder in folders:
        try:
            with open(os.path.join(folder, "scales.json")) as f:
                e = json.load(f).get(str(hole))
            if e:
                return float(e.get("depth_yd", 0) or 0), float(e.get("width_yd", 0) or 0)
        except Exception:
            continue
    return 0.0, 0.0

DISPLAY_WIDTH = 640     # heat map width on a computer
PHONE_WIDTH = 340       # heat map width in phone layout (fits an iPhone in portrait)


def looks_like_phone():
    """True when the browser says it's a phone (needs a recent Streamlit; otherwise False)."""
    try:
        ua = (st.context.headers.get("User-Agent") or "").lower()
    except Exception:
        return False
    return any(k in ua for k in ("iphone", "ipod", "android", "mobile"))


def _font(px):
    try:
        from PIL import ImageFont
        return ImageFont.load_default(size=max(8, int(round(px))))
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _yd_text(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def add_pace_rulers(shown, geom, box, ds, pad_l_disp=46, pad_t_disp=24):
    """Put the picture on a larger canvas with a ruler down the left side (yards from the front edge of the green) and one
    across the top (yards from its left edge), marked at each quarter. Returns (canvas, pad_left_px, pad_top_px)."""
    if not geom.get("bbox_px"):
        return shown, 0, 0
    pl, pt = int(round(pad_l_disp * ds)), int(round(pad_t_disp * ds))
    canvas = Image.new("RGB", (shown.width + pl, shown.height + pt), "white")
    canvas.paste(shown, (pl, pt))
    d = ImageDraw.Draw(canvas)
    f = _font(11 * ds)
    ink = (27, 54, 93)
    gx0, gy0, gx1, gy1 = geom["bbox_px"]
    depth_yd = (geom["ymax_ft"] - geom["ymin_ft"]) / 3.0
    width_yd = (geom["xmax_ft"] - geom["xmin_ft"]) / 3.0
    tick = 5 * ds
    for k in range(5):
        yy = gy1 - k * (gy1 - gy0) / 4.0 - box[1] + pt
        xx = gx0 + k * (gx1 - gx0) / 4.0 - box[0] + pl
        ly = f"{_yd_text(k * depth_yd / 4)}" + (" yd" if k == 4 else "")
        lx = f"{_yd_text(k * width_yd / 4)}" + (" yd" if k == 4 else "")
        xl = gx0 - box[0] + pl
        yt = gy0 - box[1] + pt
        d.line([(xl - tick, yy), (xl, yy)], fill=ink, width=max(1, int(round(1.3 * ds))))
        d.line([(xx, yt - tick), (xx, yt)], fill=ink, width=max(1, int(round(1.3 * ds))))
        tw = d.textlength(ly, font=f)
        d.text((max(1, xl - tick - 3 * ds - tw), yy - 6 * ds), ly, fill=ink, font=f)
        tw = d.textlength(lx, font=f)
        d.text((min(canvas.width - tw - 1, max(1, xx - tw / 2)), max(1, yt - tick - 15 * ds)), lx, fill=ink, font=f)
    return canvas, pl, pt


def crop_box(geom, w, h):
    """Crop rectangle (pixels) that frames just the green, with room for the markers."""
    b = geom.get("bbox_px")
    if not b:
        return (0, 0, w, h)
    pad = 22                                  # room for a marker at the edge; small enough to leave out text printed around the green
    return (max(0, b[0] - pad), max(0, b[1] - pad), min(w, b[2] + pad), min(h, b[3] + pad))


# --- 2. HELPERS ---
def format_feet_inches(total_feet):
    ft_total = abs(total_feet)
    ft = int(ft_total)
    inches = round((ft_total - ft) * 12)
    if inches == 12:
        ft, inches = ft + 1, 0
    if ft == 0 and inches == 0:
        return "0 in"
    s = f"{inches} in" if ft == 0 else f"{ft} ft {inches} in"
    return f"-{s}" if total_feet < 0 else s


@st.cache_data
def load_heat_map(path):
    return Image.open(path).convert("RGB")


@st.cache_data
def get_geom(path, depth_yd=0.0, width_yd=0.0):
    """Scale and origin read from the map itself (yard labels + green outline); depth/width typed in only when
    the map has no yard labels."""
    arr = np.array(load_heat_map(path))
    return pe.auto_geom(arr, depth_yd or None, width_yd or None) or pe.make_geom(arr.shape[1], arr.shape[0], 42.0, 84.0)


@st.cache_data
def get_slope(path, red_is_high, relief_ft, ignore_edge, use_arrows, arrow_trust, double_boost, max_grade, p90_target, bold_weight, depth_yd=0.0, width_yd=0.0):
    img = load_heat_map(path)
    geom = get_geom(path, depth_yd, width_yd)
    sx, sy, meta = pe.build_slope_field(np.array(img), geom, red_is_high, relief_ft, ignore_outline=ignore_edge,
                                        use_arrows=use_arrows, arrow_trust=arrow_trust, double_boost=double_boost,
                                        max_grade=max_grade, p90_target=p90_target, bold_weight=bold_weight)
    return sx, sy, meta


@st.cache_data
def get_solution(path, slope_args, ball, hole, stimp, past_ft, short_break=1.0, short_cap=None):
    sx, sy, meta = get_slope(path, *slope_args)
    return pe.solve_putt(ball, hole, sx, sy, meta, stimp, past_ft, short_break, short_cap)


def classic_read(img, geom, ball, hole, stimp):
    """The original sampling method, kept so you can compare it against the physics read."""
    arr = np.array(img)
    bx, by = pe.ft_to_px(geom, *ball)
    hx, hy = pe.ft_to_px(geom, *hole)
    xs = np.linspace(bx, hx, 10).astype(int).clip(0, arr.shape[1] - 1)
    ys = np.linspace(by, hy, 10).astype(int).clip(0, arr.shape[0] - 1)
    px = arr[ys, xs].astype(int)
    red, blue = px[:, 0].sum(), px[:, 2].sum()
    neutral = px.mean(axis=1)
    sat = (abs(px[:, 0] - neutral) + abs(px[:, 2] - neutral)).mean()
    grad = max(0.5, min(2.5, sat / 40.0))
    dist = float(np.hypot(hole[0] - ball[0], hole[1] - ball[1]))
    off = abs(ball[1] - hole[1]) * 0.04 * grad * (stimp / 8.0) * (dist / 12.0)
    side = ("Right" if red > blue else "Left") if abs(red - blue) > 1000 else ("Right" if ball[0] > hole[0] else "Left")
    return off, side


def draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows, show_ignored=False, show_detected=False,
                      extra_marks=None, px_scale=1.0, dot_px=7, pace_grid=False):
    ds = float(px_scale)                   # image pixels per on-screen pixel
    s = max(1.0, ds)
    lw = lambda base: max(base, int(round(base * s)))
    out = img.copy()
    if show_ignored and meta["ignored"].any():     # tint what the slope reader is skipping
        m = np.kron(meta["ignored"], np.ones((meta["step"], meta["step"]), bool))
        m = np.pad(m, ((0, max(0, img.height - m.shape[0])), (0, max(0, img.width - m.shape[1]))))[:img.height, :img.width]
        arr = np.array(out)
        arr[m] = (0.45 * arr[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        out = Image.fromarray(arr)
    d = ImageDraw.Draw(out)
    if show_arrows:   # computed downhill arrows, to compare against the arrows printed on the map
        step = 40
        for py in range(step // 2, img.height, step):
            for px in range(step // 2, img.width, step):
                gx, gy = int(px / meta["step"]), int(py / meta["step"])
                if gy >= sx.shape[0] or gx >= sx.shape[1] or meta["ignored"][gy, gx]:
                    continue
                vx, vy = -sx[gy, gx], sy[gy, gx]       # downhill in pixel space (rows grow downward)
                mag = np.hypot(vx, vy)
                if mag < 0.004:
                    continue
                L = min(step * 0.45, mag * 900)
                ux, uy = vx / mag, vy / mag
                ex, ey = px + ux * L, py + uy * L
                d.line([(px, py), (ex, ey)], fill="black", width=lw(2))
                d.polygon([(ex, ey), (ex - ux * 7 * s - uy * 4 * s, ey - uy * 7 * s + ux * 4 * s),
                           (ex - ux * 7 * s + uy * 4 * s, ey - uy * 7 * s - ux * 4 * s)], fill="black")
    if show_detected:   # arrows the app found on the map: orange = single head, magenta = double head
        for ar in meta["arrows"]:
            c, u = np.array(ar["c"]), np.array(ar["u"])
            col = (255, 140, 0) if ar["strength"] < 2 else (200, 0, 200)
            tail, tip = c - u * 7 * s, c + u * 7 * s
            d.line([tuple(tail), tuple(tip)], fill=col, width=lw(2))
            d.ellipse([tip[0] - 3 * s, tip[1] - 3 * s, tip[0] + 3 * s, tip[1] + 3 * s], fill=col)
    if pace_grid and geom.get("bbox_px"):
        gx0, gy0, gx1, gy1 = geom["bbox_px"]
        grid = Image.new("RGBA", out.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(grid)
        col = (27, 54, 93, 150)
        dash, gap = 7 * ds, 5 * ds
        for k in range(1, 4):                                   # quarter lines (the green's edges need no line)
            yy = gy1 - k * (gy1 - gy0) / 4.0
            xx = gx0 + k * (gx1 - gx0) / 4.0
            t = gx0
            while t < gx1:
                gd.line([(t, yy), (min(t + dash, gx1), yy)], fill=col, width=max(1, int(round(1.2 * ds))))
                t += dash + gap
            t = gy0
            while t < gy1:
                gd.line([(xx, t), (xx, min(t + dash, gy1))], fill=col, width=max(1, int(round(1.2 * ds))))
                t += dash + gap
        out = Image.alpha_composite(out.convert("RGBA"), grid).convert("RGB")
        d = ImageDraw.Draw(out)
    b, h = pe.ft_to_px(geom, *ball), pe.ft_to_px(geom, *hole)
    r_dot = max(2.5, dot_px * ds)          # marker radius as it appears on screen
    marks = [(b, "blue", r_dot), (h, "red", r_dot)]
    if sol is not None:
        a = pe.ft_to_px(geom, *sol["aim_point_ft"])
        curve = [pe.ft_to_px(geom, *p) for p in sol["path_world"][::3]]
        d.line([b, h], fill="gray", width=lw(3))
        if len(curve) > 1:
            d.line(curve, fill="#1f6fd1", width=lw(4))
        d.line([b, a], fill="#1e8e3e", width=lw(3))
        marks.append((a, "cyan", 0.7 * r_dot))
    for pt, col, r in marks:
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=col, outline="white", width=max(1, int(round(1.5 * ds))))
        c_ = max(0.8, 0.16 * r)           # tiny center dot: the exact spot that was tapped
        d.ellipse([pt[0] - c_, pt[1] - c_, pt[0] + c_, pt[1] + c_], fill="black" if col == "cyan" else "white")
    for label, (mx, my), col in (extra_marks or []):     # GPS beta: reference spots A/B and your position
        px_, py_ = pe.ft_to_px(geom, mx, my)
        q = 8 * s
        d.polygon([(px_, py_ - q), (px_ + q, py_), (px_, py_ + q), (px_ - q, py_)], fill=col, outline="white")
        d.text((px_ + q + 3, py_ - 6), label, fill=col)
    return out


def trajectory_chart(sol, compact=False):
    """Top-down view: ball at the bottom, hole straight ahead, aim line vs. the true curved path."""
    dist = sol["dist_ft"]
    sign = 1 if sol["aim_side"] == "Left" else -1
    aim_x = -sign * sol["aim_offset_ft"]                    # plot right = player's right
    px, py = -sol["path_frame"][:, 0], sol["path_frame"][:, 1]
    lim = max(1.0, 1.5 * max(sol["aim_offset_ft"], sol["max_break_ft"]))

    fig, ax = plt.subplots(figsize=(5.0, 6.4) if compact else (7.5, 6.2))
    ax.set_facecolor("#fbfcfb")
    ax.grid(True, color="#d9ded9", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#9aa59a", linewidth=1, linestyle="--", zorder=1)
    ax.plot([0, 0], [0, dist], color="#9aa59a", linewidth=2, zorder=2, label="Straight line")
    ax.plot([0, aim_x], [0, dist], color="#1e8e3e", linewidth=3, zorder=3, label="Aim line")
    ax.plot(px, py, color="#1f6fd1", linewidth=3, zorder=4, label="Expected roll")
    ax.plot([aim_x, aim_x], [0, dist], color="black", linewidth=1, linestyle=":", zorder=2)
    ax.plot([aim_x], [dist], "o", color="black", markersize=11, zorder=6)
    ax.annotate(f"Aim ({format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']})", (aim_x, dist),
                textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10 if compact else 12,
                fontweight="bold", color="#1b365d")
    ax.plot([0], [dist], "o", markerfacecolor="white", markeredgecolor="#c0392b",
            markeredgewidth=3, markersize=14, zorder=5, label="Hole")
    ax.plot([0], [0], "o", color="#1f3a8a", markersize=10, zorder=5, label="Ball")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-dist * 0.04, dist * 1.12)
    ax.set_xlabel("Feet left / right of the straight line" if compact else
                  "Feet left (-) / right (+) of the straight line   (sideways scale exaggerated)")
    ax.set_ylabel("Feet toward the hole")
    ax.legend(loc="lower right", frameon=True, fontsize=8 if compact else 9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --- 3. TUTORIAL TAB ---
@st.cache_data
@st.cache_data(show_spinner=False)
def tutorial_example():
    """A sample read on a hole whose map has yard labels, so the tutorial can show the real thing.
    Cached, and only the first two holes of each course are looked at (a course either has yard labels or it doesn't):
    scanning every map on every tap used to cost many seconds."""
    try:
        for course_name, holes in COURSES.items():
            for hole, p in sorted(holes.items())[:2]:
                if p and get_geom(p)["source"] == "yard labels":
                    img_, g_ = load_heat_map(p), get_geom(p)
                    b_, h_ = pe.default_markers(g_)
                    args = (None, 1.0, True, True, 0.7, 0.3, None, None, 0.5, 0.0, 0.0)
                    sx_, sy_, m_ = get_slope(p, *args)
                    sol_ = get_solution(p, args, b_, h_, 10.0, 1.5, 0.5, 0.04)
                    return draw_heat_overlay(img_, g_, b_, h_, sol_, sx_, sy_, m_, False), hole
    except Exception:
        pass
    return None


def render_tutorial():
    st.header("Welcome to Green Reader")
    st.write("Green Reader tells you **where to aim** and **how hard to hit** a putt. It reads each hole's heat map: "
             "**red is high ground, blue is low ground**, and the small arrows point **downhill** "
             "(double-head arrows mean a steeper slope). When you're ready, switch to the **⛳ Green Reader** tab.")

    st.subheader("Reading the map")
    ex = tutorial_example()
    if ex:
        st.image(ex[0], width=420, caption=f"Example read on hole {ex[1]}")
    st.markdown(
        "- 🔵 **Blue dot:** the ball\n"
        "- 🔴 **Red dot:** the hole\n"
        "- **Cyan dot:** your aim point. Aim here, not at the hole.\n"
        "- **Green line:** the aim line from the ball to the cyan dot\n"
        "- **Blue curve:** where the ball is expected to roll\n"
        "- **Gray line:** the straight line from ball to hole\n"
        "- **Black arrows** (optional overlay): the downhill direction the app computed")

    st.subheader("Where everything is on the Green Reader tab")
    st.markdown(
        "**① Set up the putt:** pick the course and hole, set the **Stimp** (green speed), and choose whether your "
        "next tap places the 🔴 hole or the 🔵 ball.\n\n"
        "**② Heat map:** tap it to place the ball or hole. Tap again to move it. "
        "**Reset markers** puts both back on the green. The dashed lines mark quarter-way across and back, with the yards from the "
        "front edge down the left side and from the left edge across the top, so you can pace it off. Under the map, **Place by "
        "pacing** lets you type the yards instead of tapping.\n\n"
        "**③ Fine-tune the read:** optional settings under the map. You can leave these alone.\n\n"
        "**④ Your numbers:** aim point, putt length, and how hard to hit it.\n\n"
        "**⑤ Trajectory chart:** the aim line next to the curved roll, with the hole at the top and the ball at the bottom.")

    st.subheader("Quick start")
    st.markdown("1. Choose the hole and set the Stimp.\n"
                "2. Tap the map to place the 🔵 ball where it is.\n"
                "3. Switch to 🔴 Hole and tap the cup location.\n"
                "4. Read the numbers and the chart, aim at the cyan dot, and hit the putt.")

    st.subheader("What the numbers mean")
    st.markdown(
        "- 🎯 **Aim point:** how far left or right of the hole to aim, and about how many cup widths that is.\n"
        "- 📏 **Putt length:** the real distance from ball to hole.\n"
        "- ⚡ **Hit it like a:** the stroke to use, shown as the length of a putt on a flat green at your reference speed. "
        "Uphill putts need a longer stroke than the putt length, downhill putts a shorter one.")

    st.subheader("About Stimp (green speed)")
    st.markdown(
        "**Lower numbers are slower greens and higher numbers are faster greens.** On a faster green the ball rolls "
        "farther for the same stroke, so you hit **softer**, and it also **breaks more**. On a slower green you hit "
        "**harder** and it breaks less. The *Hit it like* number is measured against a reference speed "
        "(Stimp 10 unless you change it under ③ Fine-tune).")

    st.subheader("GPS mode (beta)")
    st.markdown(
        "Off by default. Flip **📍 GPS mode (beta)** on under ① to place the ball and hole from your phone's location.\n\n"
        "1. Stand at a spot you can find on the map, like the front of the green, press the location button, mark that spot "
        "with **📍 Ref A** (tap the map, or use the quick button), and press **Lock Ref A**.\n"
        "2. Walk to a second spot far from the first, like the back of the green, and do the same with **Ref B**.\n"
        "3. Stand over the ball, press the location button, then **Set ball here**. Repeat at the hole.\n\n"
        "Phone GPS is only good to a few yards, so treat the placement as a starting point: tap the map to fine-tune, "
        "or type the paced putt length. The reader asks for the phone's best GPS fix, listens for up to 10 seconds, and shows "
        "its accuracy (green is 5 m or better). It also checks that the readings agree with each other and throws out "
        "outliers. **Keep GPS warm** keeps it locked during the round. Switch GPS mode off if "
        "anything acts up.")

    with st.expander("Other settings and questions"):
        st.markdown(
            "- **Green relief:** how much the ground rises between the blue and red areas. Raise it if reads look too "
            "straight, lower it if they look too curvy.\n"
            "- **Arrow trust and double-arrow boost:** how much the app follows the printed arrows versus the colors.\n"
            "- **Arrow boldness weight:** on maps where some arrows are drawn bolder than others, bolder arrows count for more "
            "(they set the local direction and make the slope steeper). 0 turns it off.\n"
            "- **Short-putt break and slope ceiling:** these calm the break on putts of 3 ft or less, fading out by 6 ft. "
            "Short putts are hit firmer, and real greens are rarely steeper than a few percent. Long putts are never "
            "affected. Raise either if short reads look too straight.\n"
            "- **Auto-calibrate steepness:** off by default. It scales down maps that read steeper than any real green "
            "(the Hopewell maps do), and it changes long putts too.\n"
            "- **Heat map colors:** *Auto* works out whether red is high ground from the arrows. You rarely need to change it.\n"
            "- **Phone layout:** on a phone the heat map is cropped to the green and sized to fit the screen. It switches on "
            "by itself on phones, and you can turn it on or off under ③ Fine-tune.\n"
            "- **Scale:** the green's size is read from the yard labels on each map. Maps without yard labels ask you for the "
            "depth (D) and width (W) printed on them.\n"
            "- **Adding a course:** make a folder for it inside `assets/` (for example `assets/pine_valley/`) and put each hole's "
            "heat map in it named `1_Heat.png`, `2_Heat.png`, and so on (`.jpg` works too). The course and its holes appear "
            "automatically. No code changes.\n"
            "- **Maps with D and W printed instead of yard labels:** add a file called `scales.json` to the course folder, "
            "like `{\"1\": {\"depth_yd\": 29, \"width_yd\": 19}, \"2\": {\"depth_yd\": 29, \"width_yd\": 24}}`, and the app "
            "uses those numbers automatically.\n"
            "- **Details & comparison** (bottom of the page) shows the green size, coordinates, and how many arrows were found.")


# --- GPS MODE (beta) ---
def render_gps_panel(geom):
    """Calibrate with two reference spots, then set ball/hole from the phone's GPS. Everything here is optional."""
    ss = st.session_state
    ss.setdefault("ref_tap", {})
    ss.setdefault("gps_refs", {})
    st.markdown("##### 📍 GPS mode (beta)")
    st.caption("Phone GPS is only good to a few yards, so use it to place the ball and hole roughly, then tap the map to "
               "fine-tune or type the paced putt length. Turn the GPS switch off to go back to normal.")

    # current location: the precise reader (asks the phone for its best GPS fix and averages the good readings),
    # or the basic button from the streamlit-geolocation add-on, or typed test coordinates
    raw, precise = None, False
    use_basic = st.checkbox("Use the basic location button instead", value=False, key="gps_basic",
                            help="Fallback in case the precise reader doesn't work on your phone.")
    if not use_basic:
        try:
            import gps_reader
            if gps_reader.available():
                raw = gps_reader.precise_location(seconds=10, key="precise_gps")
                precise = True
        except Exception as e:
            st.warning(f"The precise reader couldn't start ({e}). Using the basic button instead.")
    if not precise:
        try:
            from streamlit_geolocation import streamlit_geolocation
            raw = streamlit_geolocation()
        except ImportError:
            st.info("The phone-location add-on isn't installed. Add `streamlit-geolocation` to requirements.txt, "
                    "or test with typed coordinates below.")
        except Exception as e:
            st.warning(f"Couldn't read the phone's location ({e}). You can still test with typed coordinates below.")
    with st.expander("Getting better GPS accuracy"):
        st.markdown(
            "- **Allow precise location.** iPhone: Settings > Privacy & Security > Location Services > your browser "
            "(for example Safari Websites) > While Using the App, with Precise Location on.\n"
            "- **Hold still with the browser in front** while it listens. The precise reader listens for up to 10 seconds "
            "and averages the best readings.\n"
            "- **Tick Keep GPS warm** at the start of the round so the GPS stays locked and each reading is instant "
            "(uses more battery).\n"
            "- **Open a GPS app first.** A GPS app that has been running (a maps or flight app) keeps the phone's GPS locked "
            "on satellites, and the browser then gets that better fix. Open it, wait a few seconds, then come back here.\n"
            "- **Sky view matters.** Trees, buildings and a phone in a pocket all make GPS worse.")
    if isinstance(raw, dict):
        coords = raw["coords"] if isinstance(raw.get("coords"), dict) else raw
        lat, lon, acc = coords.get("latitude"), coords.get("longitude"), coords.get("accuracy")
        if lat is not None and lon is not None and raw != ss.get("gps_last_raw"):
            ss.gps_last_raw = raw
            ss.gps_fix_n = ss.get("gps_fix_n", 0) + 1
            ss.gps_fix = {"lat": float(lat), "lon": float(lon), "acc": acc, "src": "phone", "id": ss.gps_fix_n,
                          "readings": coords.get("readings"), "total": coords.get("total"), "spread": coords.get("spread"),
                          "quality": coords.get("quality")}
    with st.expander("Test without a phone: type coordinates"):
        la = st.number_input("Latitude", value=0.0, format="%.7f", key="typed_lat")
        lo = st.number_input("Longitude", value=0.0, format="%.7f", key="typed_lon")
        if st.button("Use these as my location"):
            ss.gps_fix_n = ss.get("gps_fix_n", 0) + 1
            ss.gps_fix = {"lat": la, "lon": lo, "acc": None, "src": "typed", "id": ss.gps_fix_n}
    fix = ss.get("gps_fix")
    if fix:
        acc = fix.get("acc")
        acc_txt = f", accurate to about ±{acc:.0f} m ({acc * 3.28:.0f} ft)" if acc else ""
        n_txt = ""
        if fix.get("readings"):
            n_txt = f", {fix['readings']} of {fix['total']} readings agree" if fix.get("total") else f", from {fix['readings']} readings"
            if fix.get("spread") is not None:
                n_txt += f" (within {fix['spread']:.1f} m)"
        q = fix.get("quality")
        q_txt = {"good": "Integrity good. ", "fair": "Integrity fair. ", "poor": "Integrity poor, take another reading. "}.get(q, "")
        msg = f"{q_txt}Your location: {fix['lat']:.6f}, {fix['lon']:.6f}{acc_txt}{n_txt}"
        bad = q == "poor" or (acc and acc > 12)
        ok_ = (q in (None, "good")) and (not acc or acc <= 5)
        (st.error if bad else st.success if ok_ else st.warning)(msg)
    else:
        st.info("No location yet. Press the location button above (allow location access in your browser).")

    def take_fix():
        f = ss.get("gps_fix")
        if f is None:
            st.warning("No location yet. Press the location button first.")
            return None
        if f["id"] == ss.get("gps_used_id"):
            st.warning("That's the same location reading as last time. If you've moved, press the location button again.")
        if f.get("quality") == "poor":
            st.warning("The readings in this fix didn't agree with each other (integrity poor). Take another reading first.")
        elif f.get("acc") and f["acc"] > 8:
            st.warning(f"This reading is only accurate to about ±{f['acc']:.0f} m ({f['acc'] * 3.28:.0f} ft). "
                       "Take another reading for a better position, or tap the map to fine-tune afterwards.")
        ss.gps_used_id = f["id"]
        return f

    def recalibrate():
        refs = ss.gps_refs
        if "A" in refs and "B" in refs:
            try:
                ss.gps_cal = gm.calibrate(refs["A"], refs["B"])
            except ValueError as e:
                ss.pop("gps_cal", None)
                st.error(str(e))

    def lock(name):
        tap = ss.ref_tap.get(name)
        if tap is None:
            st.warning(f"Choose the Ref {name} spot on the map first: pick '📍 Ref {name}' above the map and tap it, "
                       "or use the quick button.")
            return
        f = take_fix()
        if f:
            ss.gps_refs[name] = {"green": tap, "lat": f["lat"], "lon": f["lon"], "acc": f.get("acc")}
            recalibrate()

    st.markdown("**Step 1: calibrate with two spots you can find on the map**")
    st.caption("Stand at a spot, press the location button, mark that spot on the map (pick 📍 Ref A or Ref B above the "
               "map and tap it), then press Lock. Choose two spots far apart, like the front and back of the green.")
    q1, q2 = st.columns(2)
    if q1.button("Ref A = front tip of the green", use_container_width=True):
        ss.ref_tap["A"] = geom["front_ft"]
    if q2.button("Ref B = back tip of the green", use_container_width=True):
        ss.ref_tap["B"] = geom["back_ft"]
    l1, l2, l3 = st.columns(3)
    if l1.button("Lock Ref A here", use_container_width=True):
        lock("A")
    if l2.button("Lock Ref B here", use_container_width=True):
        lock("B")
    if l3.button("Clear references", use_container_width=True):
        ss.gps_refs, ss.ref_tap = {}, {}
        ss.pop("gps_cal", None)
    status = lambda n: "locked ✓" if n in ss.gps_refs else ("spot chosen, not locked" if n in ss.ref_tap else "not set")
    st.write(f"**Ref A:** {status('A')}   **Ref B:** {status('B')}")
    cal = ss.get("gps_cal")
    if cal:
        st.success(f"Calibrated. The two reference spots are {cal['baseline_ft'] / 3:.0f} yd apart.")
        for note in gm.quality_notes(cal):
            st.warning(note)

    st.markdown("**Step 2: set the ball and the hole**")
    st.caption("Stand over the ball, press the location button, then press Set ball here. Do the same at the hole.")
    p1, p2 = st.columns(2)
    for label, key, col in (("Set ball here", "ball_coords", p1), ("Set hole here", "hole_coords", p2)):
        if col.button(label, use_container_width=True):
            if cal is None:
                st.warning("Calibrate first (Step 1).")
                continue
            f = take_fix()
            if f:
                x, y = gm.to_green(cal, f["lat"], f["lon"])
                cx = float(np.clip(x, geom["xmin_ft"], geom["xmax_ft"]))
                cy = float(np.clip(y, geom["ymin_ft"], geom["ymax_ft"]))
                ss[key] = {"x_ft": cx, "y_ft": cy}
                ss.setdefault("marker_src", {})[key] = {"src": "gps", "acc": f.get("acc")}
                off = float(np.hypot(x - cx, y - cy))
                what = "Ball" if "ball" in key else "Hole"
                if off > 3:
                    st.warning(f"{what} placed at the nearest edge: GPS put you about {off:.0f} ft off the green.")
                if f.get("acc"):
                    st.caption(f"{what} set (GPS accuracy about ±{f['acc'] * 3.28:.0f} ft). Tap the map to fine-tune it.")

    st.markdown("**Optional: paced putt length**")
    st.number_input("Putt length in feet (0 = use the markers)", min_value=0.0, max_value=150.0, value=0.0, step=1.0,
                    key="paced_len", help="Keeps the ball-to-hole direction from the markers but uses exactly this distance.")


# --- 4. PAGE LAYOUT ---
st.title("⛳ Green Reader")
tab_help, tab_app = st.tabs(["📖 Tutorial", "⛳ Green Reader"])
with tab_help:
    render_tutorial()

with tab_app:                                          # one page, top to bottom
    top = st.container()                               # course, hole, stimp, marker mode
    scale_box = st.container()                         # only used for maps with no yard labels
    map_box = st.container()                           # heat map
    pace_box = st.expander("📏 Place by pacing (yards)")   # type paced yards instead of tapping
    gps_box = st.container()                           # GPS mode (beta), only filled when the switch is on
    tune_box = st.expander("③ ⚙️ Fine-tune the read")  # advanced controls, right under the map
    out_box = st.container()                           # aim point, putt length, how hard to hit it
    chart_box = st.container()                         # trajectory chart
    details_box = st.expander("📍 Details & comparison")

with top:
    st.markdown("##### ① Set up the putt")
    c1, c2, c3 = st.columns([2, 1, 2])
    if not COURSES:
        st.error("No heat maps found. Put each course in its own folder under `assets/` (for example "
                 "`assets/pine_valley/`) with files named `1_Heat.png`, `2_Heat.png`, and so on. See the Tutorial tab.")
        st.stop()
    course = c1.selectbox("Course", list(COURSES.keys()))
    hole_no = c2.selectbox("Hole", sorted(COURSES[course]))
    stimp = c3.slider("Stimp (green speed)", 6.0, 13.0, 8.0, 0.5,
                      help="Lower = slower greens, higher = faster greens. Faster greens break more and need a softer hit.")
    g1, g2 = st.columns([3, 1])
    gps_on = (st.toggle if hasattr(st, "toggle") else st.checkbox)(
        "📍 GPS mode (beta)", value=False, key="gps_on", help="Off by default. Turn it off any time to go back to normal.")
    reset_markers = g2.button("Reset markers", use_container_width=True)
    placement_mode = st.radio("Tap the map to set the", ["🔴 Hole", "🔵 Ball"] + (["📍 Ref A", "📍 Ref B"] if gps_on else []),
                              horizontal=True)

with tune_box:
    t1, t2 = st.columns(2)
    color_scale = t1.selectbox("Heat map colors", ["Auto (from arrows)", "Red = high ground", "Red = low ground (flip)"])
    relief_ft = t2.slider("Green relief (ft)", 0.3, 4.0, 1.0, 0.1,
                          help="Elevation difference between the coolest and warmest color. "
                               "Raise it if reads look too straight, lower it if they look too curvy.")
    arrow_trust = t1.slider("Arrow trust", 0.0, 1.0, 0.7, 0.05,
                            help="0 = colors only for direction, 1 = follow the arrows wherever they are.")
    double_boost = t2.slider("Double-arrow boost", 0.0, 1.0, 0.3, 0.05,
                             help="Extra steepness where double-head arrows are.")
    phone = t2.checkbox("📱 Phone layout", value=looks_like_phone(), key="phone_layout",
                        help="Crops the heat map to the green and sizes it to fit a phone screen. Switched on automatically "
                             "on phones; turn it off or on here.")
    max_grade_pct = t2.slider("Short-putt slope ceiling (%)", 2.0, 8.0, 4.0, 0.5,
                              help="On putts of 3 ft or less (fading out by 6 ft) the app won't believe a slope steeper than this. "
                                   "Real greens rarely get steeper than a few percent. Long putts are not affected.")
    auto_steep = t2.checkbox("Auto-calibrate steepness (all putts)", value=False,
                             help="Off by default. Some maps use the full red-to-blue range in a narrow band, which reads as slopes "
                                  "steeper than any real green. Turning this on scales those maps down so 90% of the green is no "
                                  "steeper than about 3.5%. It changes long putts too. Gentle maps (like Mercer Oaks) are not changed.")
    bold_pct = t2.slider("Arrow boldness weight (%)", 0, 100, 50, 10,
                         help="Bolder arrows count for more: they pull the local direction toward themselves and make the slope "
                              "steeper where they sit. 0 = every arrow counts the same. Only maps whose arrows vary in weight are affected.")
    show_pace = t1.checkbox("Pace marks (yards)", value=True,
                            help="Quarter-way marks along the left side and across the top, in yards from the front and from the "
                                 "left edge of the green, so you can pace off the ball and hole and match them to the map.")
    dot_size = t1.slider("Ball / hole dot size", 3, 12, 6, 1,
                         help="Radius of the ball and hole dots in screen pixels. Smaller dots make it easier to see exactly "
                              "where you tapped. The tiny center dot marks the exact spot.")
    short_pct = t1.slider("Short-putt break (inside 6 ft)", 0, 100, 50, 5,
                          help="How much of the sideways break to keep on putts of 3 ft or less, easing back to 100% at 6 ft. "
                               "Short putts are hit firmer than the pace the app assumes, so they break less. 100% = no change.")
    ref_stimp = t1.slider("My stroke is calibrated for Stimp", 6.0, 13.0, 10.0, 0.5,
                          help="'Hit it like a ... ft putt' is measured on a flat green at this speed. Pick the green speed "
                               "you practice on or feel most comfortable with.")
    past_ft = t1.slider("Miss-past pace (ft)", 0.5, 3.0, 1.5, 0.25,
                        help="How far past the hole the ball would stop. Slower pace = more break.")
    use_arrows = t2.checkbox("Use the arrows printed on the map", value=True,
                             help="Arrows set the break direction; double-head arrows make the slope steeper.")
    ignore_edge = t2.checkbox("Ignore green boundary line", value=True,
                              help="Skips the solid green outline (and everything outside it) so the edge of the map "
                                   "isn't read as a slope.")
    st.caption("Map overlays")
    o1, o2, o3 = st.columns(3)
    show_arrows = o1.checkbox("Computed slope arrows", value=True)
    show_detected = o2.checkbox("Detected arrows", value=False, help="Orange = single head, purple = double head.")
    show_ignored = o3.checkbox("Ignored edge pixels", value=False, help="Tinted magenta.")

heat_path = COURSES[course].get(hole_no)
if not heat_path:
    st.warning(f"Heat map image not found for Hole #{hole_no}.")
    st.stop()

img = load_heat_map(heat_path)
depth_yd = width_yd = 0.0
try:
    geom = get_geom(heat_path)
except Exception as e:
    st.error(f"Couldn't analyze `{heat_path}`: {type(e).__name__}: {e}. If the other holes work, this image may be "
             "unusual. Send it along with this message and it can be fixed.")
    st.stop()
if geom["source"] == "estimate":                       # no yard labels on this map: use scales.json, else ask for D and W
    depth_yd, width_yd = load_scales(course, hole_no)
    if depth_yd > 0 or width_yd > 0:
        geom = get_geom(heat_path, depth_yd, width_yd)
if geom["source"] == "estimate":
    with scale_box:
        st.info("This map has no readable yard labels, so Green Reader can't tell how big the green is, and distances are only "
                "a guess until you tell it. Enter the green's width or depth in yards (either one is enough). "
                "To save the numbers for good, put them in a scales.json file in the course folder "
                "(see the Tutorial tab).")
        s1, s2 = st.columns(2)
        depth_yd = float(s1.number_input("Depth, D (yards)", 0.0, 120.0, 0.0, 0.5, key=f"depth_{course}_{hole_no}") or 0.0)
        width_yd = float(s2.number_input("Width, W (yards)", 0.0, 120.0, 0.0, 0.5, key=f"width_{course}_{hole_no}") or 0.0)
    if depth_yd > 0 or width_yd > 0:
        geom = get_geom(heat_path, depth_yd, width_yd)
if reset_markers or st.session_state.get("marker_key") != (course, hole_no, depth_yd, width_yd):   # new hole or scale: markers restart
    b0, h0 = pe.default_markers(geom)
    st.session_state.ball_coords = {"x_ft": b0[0], "y_ft": b0[1]}
    st.session_state.hole_coords = {"x_ft": h0[0], "y_ft": h0[1]}
    st.session_state.marker_key = (course, hole_no, depth_yd, width_yd)
    st.session_state.marker_src = {}
    st.session_state.pop("last_click", None)
if gps_on:
    with gps_box:
        render_gps_panel(geom)
ball = (st.session_state.ball_coords["x_ft"], st.session_state.ball_coords["y_ft"])
hole = (st.session_state.hole_coords["x_ft"], st.session_state.hole_coords["y_ft"])
red_high = None if color_scale.startswith("Auto") else color_scale.startswith("Red = high")
slope_args = (red_high, relief_ft, ignore_edge, use_arrows, arrow_trust, double_boost, None, 0.035 if auto_steep else None,
              bold_pct / 100.0, depth_yd, width_yd)

hole_raw = hole
paced = float(st.session_state.get("paced_len") or 0.0) if gps_on else 0.0
if paced >= 1.0:                                       # paced length: keep the direction, use the exact distance
    v = np.array(hole) - np.array(ball)
    n = float(np.hypot(*v))
    u = v / n if n > 0.3 else np.array([0.0, 1.0])
    hole = tuple(float(c) for c in np.array(ball) + u * paced)

marks = []
if gps_on:
    for nm, col in (("A", (255, 140, 0)), ("B", (150, 60, 200))):
        if nm in st.session_state.get("ref_tap", {}):
            marks.append((nm, st.session_state.ref_tap[nm], col))
    if st.session_state.get("gps_cal") and st.session_state.get("gps_fix"):
        f_ = st.session_state.gps_fix
        marks.append(("you", gm.to_green(st.session_state.gps_cal, f_["lat"], f_["lon"]), (0, 160, 90)))

too_close = np.hypot(hole[0] - ball[0], hole[1] - ball[1]) < 1.0
sx, sy, meta = get_slope(heat_path, *slope_args)
sol = None if too_close else get_solution(heat_path, slope_args, ball, hole, stimp, past_ft, short_pct / 100.0, max_grade_pct / 100.0)

# heat map (tapping it moves the marker chosen above)
with map_box:
    st.subheader(f"② Heat map: Hole #{hole_no}")
    disp_w = PHONE_WIDTH if phone else DISPLAY_WIDTH
    box = crop_box(geom, img.width, img.height) if phone else (0, 0, img.width, img.height)
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    rulers = bool(show_pace and geom.get("bbox_px"))
    pad_l_disp = 46
    ds = box_w / (disp_w - pad_l_disp) if rulers else box_w / disp_w            # image pixels per on-screen pixel
    shown = draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows, show_ignored, show_detected, marks,
                              px_scale=ds, dot_px=dot_size, pace_grid=rulers)
    shown = shown.crop(box)
    pad_l = pad_t = 0
    if rulers:
        shown, pad_l, pad_t = add_pace_rulers(shown, geom, box, ds, pad_l_disp)
    comp_w, comp_h = shown.size
    if max(comp_w, comp_h) > 2 * max(disp_w, 400):                               # keep what is sent to the phone light
        k_ = 2 * max(disp_w, 400) / max(comp_w, comp_h)
        shown = shown.resize((max(1, int(comp_w * k_)), max(1, int(comp_h * k_))), Image.LANCZOS)
    clicked = streamlit_image_coordinates(shown, key="map_click_phone" if phone else "map_click", width=disp_w)
    if clicked and clicked != st.session_state.get("last_click"):
        st.session_state.last_click = clicked
        sw = clicked.get("width") or disp_w                       # size the browser actually showed
        sh = clicked.get("height") or sw * comp_h / comp_w
        cx, cy = pe.px_to_ft(geom, box[0] + clicked["x"] * comp_w / sw - pad_l, box[1] + clicked["y"] * comp_h / sh - pad_t)
        if "Ref" in placement_mode:
            st.session_state.setdefault("ref_tap", {})[placement_mode[-1]] = (cx, cy)
        else:
            key = "hole_coords" if "Hole" in placement_mode else "ball_coords"
            st.session_state[key] = {"x_ft": cx, "y_ft": cy}
            st.session_state.setdefault("marker_src", {})[key] = {"src": "tap"}
        st.rerun()
    if rulers:
        st.caption("Pace marks: the dashed lines are quarter-way across and quarter-way back. Numbers down the left side are yards "
                   "from the front edge of the green; numbers across the top are yards from its left edge. One big step is about a yard.")
    if geom["source"] == "estimate":
        st.warning("The scale is only an estimate until you enter the width or depth above, so putt lengths here may be well off. "
                   "Don't trust distances on this hole until you do.")

# where the markers sit, in yards, and a way to type paced distances
with pace_box:
    d_yd = (geom["ymax_ft"] - geom["ymin_ft"]) / 3.0
    w_yd = (geom["xmax_ft"] - geom["xmin_ft"]) / 3.0
    from_left = lambda p: (p[0] - geom["xmin_ft"]) / 3.0
    from_front = lambda p: (p[1] - geom["ymin_ft"]) / 3.0
    bl, bf, hl, hf = from_left(ball), from_front(ball), from_left(hole_raw), from_front(hole_raw)
    st.write(f"**Now:** ball {bf:.1f} yd from the front and {bl:.1f} yd from the left edge. "
             f"Hole {hf:.1f} yd from the front and {hl:.1f} yd from the left edge. "
             f"(The green is about {w_yd:.0f} yd wide and {d_yd:.0f} yd deep.)")
    st.caption("Walk it off: about one big step per yard. Count from the front edge of the green and from its left edge, "
               "type the numbers, and press Place. This matches the numbered marks on the map.")
    kk = f"{course}_{hole_no}_{bf:.1f}_{bl:.1f}_{hf:.1f}_{hl:.1f}"
    p1, p2 = st.columns(2)
    nb_f = p1.number_input("Ball: yards from front", 0.0, float(d_yd), float(np.clip(bf, 0, d_yd)), 0.5, key="pb_f_" + kk)
    nb_l = p2.number_input("Ball: yards from left edge", 0.0, float(w_yd), float(np.clip(bl, 0, w_yd)), 0.5, key="pb_l_" + kk)
    nh_f = p1.number_input("Hole: yards from front", 0.0, float(d_yd), float(np.clip(hf, 0, d_yd)), 0.5, key="ph_f_" + kk)
    nh_l = p2.number_input("Hole: yards from left edge", 0.0, float(w_yd), float(np.clip(hl, 0, w_yd)), 0.5, key="ph_l_" + kk)
    if st.button("Place ball and hole", key="pace_place_" + kk):
        st.session_state.ball_coords = {"x_ft": geom["xmin_ft"] + nb_l * 3.0, "y_ft": geom["ymin_ft"] + nb_f * 3.0}
        st.session_state.hole_coords = {"x_ft": geom["xmin_ft"] + nh_l * 3.0, "y_ft": geom["ymin_ft"] + nh_f * 3.0}
        st.session_state.setdefault("marker_src", {}).update({"ball_coords": {"src": "paced"}, "hole_coords": {"src": "paced"}})
        st.session_state.pop("last_click", None)
        st.rerun()

# the numbers
with out_box:
    st.markdown("##### ④ Your numbers")
    if sol is None:
        st.info("Ball and hole are less than a foot apart. Tap the map to move one of them.")
    else:
        slope_diff = sol["flat_equiv_ft"] - sol["dist_ft"]          # uphill (+) or downhill (-)
        stroke_ft = sol["flat_equiv_ft"] * ref_stimp / stimp         # faster green = shorter stroke, slower = longer
        diff = stroke_ft - sol["dist_ft"]
        factor, severity = pe.classify_break(sol["aim_offset_ft"], sol["dist_ft"])
        cups = sol["aim_offset_ft"] * 12 / pe.CUP_IN
        aim_txt = f"{format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']}" if sol["aim_offset_ft"] > 0.04 else "Straight"
        k1, k2, k3 = st.columns(3)
        k1.metric("🎯 Aim point", aim_txt, delta=f"≈ {cups:.1f} cups {sol['aim_side'].lower()} of the hole", delta_color="off")
        k2.metric("📏 Putt length", f"{sol['dist_ft']:.1f} ft", delta=f"{sol['dist_ft'] / 3:.1f} yd", delta_color="off")
        k3.metric("⚡ Hit it like a", f"{stroke_ft:.1f} ft putt", delta=f"{diff:+.1f} ft vs. putt length", delta_color="off")
        play = "uphill" if slope_diff > 0.3 else "downhill" if slope_diff < -0.3 else "about flat"
        speed_note = "" if abs(stimp - ref_stimp) < 0.25 else (
            f" Stimp {stimp:g} is {'slower' if stimp < ref_stimp else 'faster'} than your Stimp {ref_stimp:g} reference, "
            f"so the stroke is {'longer' if stimp < ref_stimp else 'shorter'}.")
        st.caption(f"{severity} break ({factor:.2f}x), "
                   f"{'right-to-left' if sol['aim_side'] == 'Right' else 'left-to-right'}, playing {play}. "
                   f"'Hit it like' is how far this stroke would roll a ball on a flat Stimp {ref_stimp:g} green.{speed_note}")
        msrc = st.session_state.get("marker_src", {})
        if gps_on and paced < 1.0 and all(msrc.get(k, {}).get("src") == "gps" for k in ("ball_coords", "hole_coords")):
            accs = [msrc[k].get("acc") for k in ("ball_coords", "hole_coords")]
            if all(accs):
                unc_ft = float(np.hypot(*accs)) * 3.28
                if sol["dist_ft"] < 2.5 * unc_ft:
                    st.warning(f"The ball and hole were both placed by GPS, which is only good to about ±{unc_ft:.0f} ft between "
                               f"the two, so this {sol['dist_ft']:.0f} ft putt length (and the direction) may be off by that much. "
                               "For short putts, tap the hole on the map or type the paced length under the GPS controls.")
        if not sol["reached"] or abs(sol["hit_error_ft"]) > 0.15:
            st.warning("The solver could not make this putt drop with the current settings. "
                       "Try lowering Green relief or checking the color setting.")

# the graph
with chart_box:
    if sol is not None:
        st.subheader("⑤ Where to aim and how it breaks")
        st.caption("Green line: aim line. Blue curve: expected roll. Hole at the top, ball at the bottom.")
        fig = trajectory_chart(sol, compact=phone)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with details_box:
    st.write("**Courses found in assets/:** " + ("; ".join(f"{c} ({len(h)} holes)" for c, h in COURSES.items()) or "none"))
    how = {"yard labels": "read from the map's yard labels", "entered": "from the depth and width you entered",
           "estimate": "estimated (no yard labels)"}.get(geom["source"], "")
    st.write(f"**Green:** {geom['width_ft'] / 3:.0f} yd wide x {geom['depth_ft'] / 3:.0f} yd deep, scale {how}. "
             f"Positions are measured from the green's left edge and the map's "
             f"{'0-yard line' if geom['source'] == 'yard labels' else 'front (bottom) edge'}.")
    st.write(f"**Ball:** {ball[0]:.1f} ft, {ball[1]:.1f} ft   **Hole:** {hole[0]:.1f} ft, {hole[1]:.1f} ft")
    if gps_on:
        cal_ = st.session_state.get("gps_cal")
        st.write("**GPS mode:** on. " + (f"Calibrated (references {cal_['baseline_ft'] / 3:.0f} yd apart, "
                 f"distance check {cal_['scale_ratio'] * 100:.0f}%)." if cal_ else "Not calibrated yet."))
        if paced >= 1.0:
            st.write(f"**Paced length in use:** {paced:.0f} ft. The hole is drawn {paced:.0f} ft from the ball along the "
                     f"line to your hole marker (the marker itself is {np.hypot(hole_raw[0] - ball[0], hole_raw[1] - ball[1]):.0f} ft away).")
    if sol is not None:
        old_off, old_side = classic_read(img, geom, ball, hole, stimp)
        st.write(f"**New read:** {format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']}   "
                 f"(max break along the path {format_feet_inches(sol['max_break_ft'])})")
        st.write(f"**Old read:** {format_feet_inches(old_off)} {old_side}")
    if meta["arrows"]:
        agree = "n/a" if meta["agreement"] is None else f"{meta['agreement'] * 100:.0f}%"
        st.write(f"**Arrows found:** {len(meta['arrows'])} ({meta['n_double']} double-head)   "
                 f"**Agree with colors:** {agree}   "
                 f"**Colors read as:** {'red = high' if meta['red_is_high'] else 'red = low'}")
    else:
        st.write("**Arrows found:** none, so this read uses colors only.")
