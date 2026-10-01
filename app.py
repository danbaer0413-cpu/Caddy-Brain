import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

import putt_engine as pe

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="centered")

COURSES = {"Mercer Oaks East": list(range(1, 19))}   # green size and scale are read from each map automatically

DISPLAY_WIDTH = 640


# --- 2. ONBOARDING TUTORIAL MODAL ---
@st.dialog("⛳ Welcome to CaddyBrain Green Reader")
def show_tutorial():
    st.markdown("""
    Welcome! Here is a quick guide to reading putts like a pro:

    1. **Select Course & Hole**  
       Choose your course and hole number at the top of the page.
    2. **Tap to Place Markers**  
       Select whether your next tap sets the **🔴 Hole** or **🔵 Ball**, then tap directly on the heat map.
    3. **Adjust Stimp & Fine-Tune**  
       Set your green speed (**Stimp**) and tweak advanced settings if needed in the expander below the map.
    4. **Read Aim & Trajectory**  
       View your precise **Aim Point**, **Putt Length**, **Hit Power**, and the simulated ball roll curve!
    """)
    if st.button("Let's Read Some Putts! 🏌️‍♂️", use_container_width=True):
        st.session_state.onboarded = True
        st.rerun()

if "onboarded" not in st.session_state:
    st.session_state.onboarded = False
    show_tutorial()


# --- 3. HELPERS ---
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
def get_geom(path):
    """Scale and origin read from the map itself (yard labels + green outline)."""
    arr = np.array(load_heat_map(path))
    return pe.auto_geom(arr) or pe.make_geom(arr.shape[1], arr.shape[0], 42.0, 84.0)


@st.cache_data
def get_slope(path, red_is_high, relief_ft, ignore_edge, use_arrows, arrow_trust, double_boost):
    img = load_heat_map(path)
    geom = get_geom(path)
    sx, sy, meta = pe.build_slope_field(np.array(img), geom, red_is_high, relief_ft, ignore_outline=ignore_edge,
                                        use_arrows=use_arrows, arrow_trust=arrow_trust, double_boost=double_boost)
    return sx, sy, meta


@st.cache_data
def get_solution(path, slope_args, ball, hole, stimp, past_ft):
    sx, sy, meta = get_slope(path, *slope_args)
    return pe.solve_putt(ball, hole, sx, sy, meta, stimp, past_ft)


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


def draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows, show_ignored=False, show_detected=False):
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
                d.line([(px, py), (ex, ey)], fill="black", width=2)
                d.polygon([(ex, ey), (ex - ux * 7 - uy * 4, ey - uy * 7 + ux * 4),
                           (ex - ux * 7 + uy * 4, ey - uy * 7 - ux * 4)], fill="black")
    if show_detected:   # arrows the app found on the map: orange = single head, magenta = double head
        for ar in meta["arrows"]:
            c, u = np.array(ar["c"]), np.array(ar["u"])
            col = (255, 140, 0) if ar["strength"] < 2 else (200, 0, 200)
            tail, tip = c - u * 7, c + u * 7
            d.line([tuple(tail), tuple(tip)], fill=col, width=2)
            d.ellipse([tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3], fill=col)
    b, h = pe.ft_to_px(geom, *ball), pe.ft_to_px(geom, *hole)
    marks = [(b, "blue", 12), (h, "red", 12)]
    if sol is not None:
        a = pe.ft_to_px(geom, *sol["aim_point_ft"])
        curve = [pe.ft_to_px(geom, *p) for p in sol["path_world"][::3]]
        d.line([b, h], fill="gray", width=3)
        if len(curve) > 1:
            d.line(curve, fill="#1f6fd1", width=4)
        d.line([b, a], fill="#1e8e3e", width=3)
        marks.append((a, "cyan", 8))
    for pt, col, r in marks:
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=col, outline="white", width=2)
    return out


def trajectory_chart(sol):
    """Top-down view: ball at the bottom, hole straight ahead, aim line vs. the true curved path."""
    dist = sol["dist_ft"]
    sign = 1 if sol["aim_side"] == "Left" else -1
    aim_x = -sign * sol["aim_offset_ft"]                    # plot right = player's right
    px, py = -sol["path_frame"][:, 0], sol["path_frame"][:, 1]
    lim = max(1.0, 1.5 * max(sol["aim_offset_ft"], sol["max_break_ft"]))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.set_facecolor("#fbfcfb")
    ax.grid(True, color="#d9ded9", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#9aa59a", linewidth=1, linestyle="--", zorder=1)
    ax.plot([0, 0], [0, dist], color="#9aa59a", linewidth=2, zorder=2, label="Straight line")
    ax.plot([0, aim_x], [0, dist], color="#1e8e3e", linewidth=3, zorder=3, label="Aim line")
    ax.plot(px, py, color="#1f6fd1", linewidth=3, zorder=4, label="Expected roll")
    ax.plot([aim_x, aim_x], [0, dist], color="black", linewidth=1, linestyle=":", zorder=2)
    ax.plot([aim_x], [dist], "o", color="black", markersize=11, zorder=6)
    ax.annotate(f"Aim ({format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']})", (aim_x, dist),
                textcoords="offset points", xytext=(0, 14), ha="center", fontsize=12,
                fontweight="bold", color="#1b365d")
    ax.plot([0], [dist], "o", markerfacecolor="white", markeredgecolor="#c0392b",
            markeredgewidth=3, markersize=14, zorder=5, label="Hole")
    ax.plot([0], [0], "o", color="#1f3a8a", markersize=10, zorder=5, label="Ball")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-dist * 0.04, dist * 1.12)
    ax.set_xlabel("Feet left (-) / right (+) of the straight line   (sideways scale exaggerated)")
    ax.set_ylabel("Feet toward the hole")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


# --- 4. PAGE LAYOUT: one page, top to bottom ---
c_title, c_btn = st.columns([4, 1])
c_title.title("⛳ CaddyBrain Green Reader")
if c_btn.button("📖 Tutorial", use_container_width=True):
    show_tutorial()

top = st.container()                                   # course, hole, stimp, marker mode
map_box = st.container()                               # heat map
tune_box = st.expander("⚙️ Fine-tune the read")        # advanced controls, right under the map
out_box = st.container()                               # aim point, putt length, how hard to hit it
chart_box = st.container()                             # trajectory chart
details_box = st.expander("📍 Details & comparison")

with top:
    c1, c2, c3 = st.columns([2, 1, 2])
    course = c1.selectbox("Course", list(COURSES.keys()))
    hole_no = c2.selectbox("Hole", COURSES[course])
    stimp = c3.slider("Stimp", 6.0, 13.0, 8.0, 0.5)
    m1, m2 = st.columns([3, 1])
    placement_mode = m1.radio("Tap the map to set the", ["🔴 Hole", "🔵 Ball"], horizontal=True)
    m2.write("")
    reset_markers = m2.button("Reset markers", use_container_width=True)

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

folder = course.lower().replace(" ", "_")
heat_path = next((f"assets/{folder}/{n}" for n in (f"{hole_no}_Heat.png", f"{hole_no}_heat.png")
                  if os.path.exists(f"assets/{folder}/{n}")), None)
if not heat_path:
    st.warning(f"Heat map image not found for Hole #{hole_no} in `assets/{folder}/`.")
    st.stop()

img = load_heat_map(heat_path)
geom = get_geom(heat_path)
if reset_markers or st.session_state.get("marker_key") != (course, hole_no):   # new hole: markers start on this green
    b0, h0 = pe.default_markers(geom)
    st.session_state.ball_coords = {"x_ft": b0[0], "y_ft": b0[1]}
    st.session_state.hole_coords = {"x_ft": h0[0], "y_ft": h0[1]}
    st.session_state.marker_key = (course, hole_no)
    st.session_state.pop("last_click", None)
ball = (st.session_state.ball_coords["x_ft"], st.session_state.ball_coords["y_ft"])
hole = (st.session_state.hole_coords["x_ft"], st.session_state.hole_coords["y_ft"])
red_high = None if color_scale.startswith("Auto") else color_scale.startswith("Red = high")
slope_args = (red_high, relief_ft, ignore_edge, use_arrows, arrow_trust, double_boost)

too_close = np.hypot(hole[0] - ball[0], hole[1] - ball[1]) < 1.0
sx, sy, meta = get_slope(heat_path, *slope_args)
sol = None if too_close else get_solution(heat_path, slope_args, ball, hole, stimp, past_ft)

# heat map (tapping it moves the marker chosen above)
with map_box:
    st.subheader(f"Hole #{hole_no}")
    shown = draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows, show_ignored, show_detected)
    clicked = streamlit_image_coordinates(shown, key="map_click", width=DISPLAY_WIDTH)
    if clicked and clicked != st.session_state.get("last_click"):
        st.session_state.last_click = clicked
        scale = img.width / float(DISPLAY_WIDTH)
        cx, cy = pe.px_to_ft(geom, clicked["x"] * scale, clicked["y"] * scale)
        key = "hole_coords" if "Hole" in placement_mode else "ball_coords"
        st.session_state[key] = {"x_ft": cx, "y_ft": cy}
        st.rerun()
    if geom["source"] != "yard labels":
        st.warning("Couldn't find the yard labels on this map, so the scale is an estimate and distances may be off.")

# the numbers
with out_box:
    if sol is None:
        st.info("Ball and hole are less than a foot apart. Tap the map to move one of them.")
    else:
        stroke_ft = sol["flat_equiv_ft"]
        diff = stroke_ft - sol["dist_ft"]
        factor, severity = pe.classify_break(sol["aim_offset_ft"], sol["dist_ft"])
        cups = sol["aim_offset_ft"] * 12 / pe.CUP_IN
        aim_txt = f"{format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']}" if sol["aim_offset_ft"] > 0.04 else "Straight"
        k1, k2, k3 = st.columns(3)
        k1.metric("🎯 Aim point", aim_txt, delta=f"≈ {cups:.1f} cups {sol['aim_side'].lower()} of the hole", delta_color="off")
        k2.metric("📏 Putt length", f"{sol['dist_ft']:.1f} ft", delta=f"{sol['dist_ft'] / 3:.1f} yd", delta_color="off")
        k3.metric("⚡ Hit it like a", f"{stroke_ft:.1f} ft putt", delta=f"{diff:+.1f} ft vs. putt length", delta_color="off")
        play = "uphill" if diff > 0.3 else "downhill" if diff < -0.3 else "about flat"
        st.caption(f"{severity} break ({factor:.2f}x), "
                   f"{'right-to-left' if sol['aim_side'] == 'Right' else 'left-to-right'}, "
                   f"playing {play}. 'Hit it like' is the flat-green distance that gives the same roll speed.")
        if not sol["reached"] or abs(sol["hit_error_ft"]) > 0.15:
            st.warning("The solver could not make this putt drop with the current settings. "
                       "Try lowering Green relief or checking the color setting.")

# the graph
with chart_box:
    if sol is not None:
        st.subheader("Where to aim and how it breaks")
        st.caption("Green line: aim line. Blue curve: expected roll. Hole at the top, ball at the bottom.")
        fig = trajectory_chart(sol)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with details_box:
    st.write(f"**Green:** {geom['width_ft'] / 3:.0f} yd wide x {geom['depth_ft'] / 3:.0f} yd deep, scale read from the "
             f"map's yard labels. Positions are measured from the green's left edge and the map's 0-yard line.")
    st.write(f"**Ball:** {ball[0]:.1f} ft, {ball[1]:.1f} ft   **Hole:** {hole[0]:.1f} ft, {hole[1]:.1f} ft")
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