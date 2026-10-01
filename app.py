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
st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="wide",
                   initial_sidebar_state="collapsed")

if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)}
    }
if "ball_coords" not in st.session_state or not isinstance(st.session_state.ball_coords, dict):
    st.session_state.ball_coords = {"x_ft": 29.5, "y_ft": 52.1}
if "hole_coords" not in st.session_state or not isinstance(st.session_state.hole_coords, dict):
    st.session_state.hole_coords = {"x_ft": 15.1, "y_ft": 23.2}

DISPLAY_WIDTH = 420


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
def get_slope(path, width_ft, depth_ft, red_is_high, relief_ft):
    img = load_heat_map(path)
    geom = pe.make_geom(img.width, img.height, width_ft, depth_ft)
    sx, sy, meta = pe.build_slope_field(np.array(img), geom, red_is_high, relief_ft)
    return sx, sy, meta


@st.cache_data
def get_solution(path, width_ft, depth_ft, red_is_high, relief_ft, ball, hole, stimp, past_ft):
    sx, sy, meta = get_slope(path, width_ft, depth_ft, red_is_high, relief_ft)
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


def draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows):
    out = img.copy()
    d = ImageDraw.Draw(out)
    if show_arrows:   # computed downhill arrows, to compare against the arrows printed on the map
        step = 40
        for py in range(int(geom["box_y_bottom"] - geom["box_height"]), int(geom["box_y_bottom"]), step):
            for px in range(int(geom["box_x_min"]), int(geom["box_x_max"]), step):
                gx, gy = int(px / meta["step"]), int(py / meta["step"])
                if gy >= sx.shape[0] or gx >= sx.shape[1]:
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
    b, h = pe.ft_to_px(geom, *ball), pe.ft_to_px(geom, *hole)
    a = pe.ft_to_px(geom, *sol["aim_point_ft"])
    curve = [pe.ft_to_px(geom, *p) for p in sol["path_world"][::3]]
    d.line([b, h], fill="gray", width=3)
    if len(curve) > 1:
        d.line(curve, fill="#1f6fd1", width=4)
    d.line([b, a], fill="#1e8e3e", width=3)
    for pt, col, r in ((b, "blue", 12), (h, "red", 12), (a, "cyan", 8)):
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


# --- 3. SIDEBAR ---
sb = st.sidebar
sb.header("1. Course & Hole")
course = sb.selectbox("Course", list(st.session_state.courses_db.keys()))
hole_no = sb.selectbox("Hole #", list(st.session_state.courses_db[course].keys()))
depth_yds = st.session_state.courses_db[course][hole_no]["max_depth_yds"]
width_yds = st.session_state.courses_db[course][hole_no]["width_yds"]

sb.header("2. Green Speed & Slope")
stimp = sb.slider("Stimp", 6.0, 13.0, 8.0, 0.5)
color_scale = sb.selectbox("Heat map colors", ["Red = high ground", "Red = low ground (flip)"])
relief_ft = sb.slider("Green relief (ft)", 0.3, 4.0, 1.0, 0.1,
                      help="Elevation difference between the coolest and warmest color. "
                           "Raise it if reads look too straight, lower it if they look too curvy.")
past_ft = sb.slider("Miss-past pace (ft)", 0.5, 3.0, 1.5, 0.25,
                    help="How far past the hole the ball would stop. Slower pace = more break.")

sb.header("3. Marker Mode")
placement_mode = sb.radio("Click sets:", ["🔴 Hole Position", "🔵 Ball Position"])
show_arrows = sb.checkbox("Show computed slope arrows", value=True)
if sb.button("Reset Markers"):
    st.session_state.ball_coords = {"x_ft": width_yds * 1.5, "y_ft": 4.0}
    st.session_state.hole_coords = {"x_ft": width_yds * 1.5, "y_ft": 20.0}
    st.session_state.pop("last_click", None)
    st.rerun()

folder = course.lower().replace(" ", "_")
heat_path = next((f"assets/{folder}/{n}" for n in (f"{hole_no}_Heat.png", f"{hole_no}_heat.png")
                  if os.path.exists(f"assets/{folder}/{n}")), None)

# --- 4. MAIN ---
st.title(f"⛳ Hole #{hole_no}")

if not heat_path:
    st.warning(f"Heat map image not found for Hole #{hole_no} in `assets/{folder}/`.")
    st.stop()

img = load_heat_map(heat_path)
width_ft, depth_ft = width_yds * 3.0, depth_yds * 3.0
geom = pe.make_geom(img.width, img.height, width_ft, depth_ft)
ball = (st.session_state.ball_coords["x_ft"], st.session_state.ball_coords["y_ft"])
hole = (st.session_state.hole_coords["x_ft"], st.session_state.hole_coords["y_ft"])
red_high = color_scale.startswith("Red = high")

if np.hypot(hole[0] - ball[0], hole[1] - ball[1]) < 1.0:
    st.info("Ball and hole are less than a foot apart. Move one of them to get a read.")
    st.stop()

sx, sy, meta = get_slope(heat_path, width_ft, depth_ft, red_high, relief_ft)
sol = get_solution(heat_path, width_ft, depth_ft, red_high, relief_ft, ball, hole, stimp, past_ft)
stroke_ft = round(sol["flat_equiv_ft"], 1)
factor, severity = pe.classify_break(sol["aim_offset_ft"], sol["dist_ft"])
cups = sol["aim_offset_ft"] * 12 / pe.CUP_IN

tab_traj, tab_map = st.tabs(["Trajectory & Metrics", "Heat Map Inspector"])

with tab_traj:
    c1, c2, c3 = st.columns(3)
    aim_txt = f"{format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']}" if sol["aim_offset_ft"] > 0.04 else "Straight"
    c1.metric("🎯 Aim Offset", aim_txt, delta=f"~{cups:.1f} cup widths {sol['aim_side'].lower()}", delta_color="off")
    c2.metric("⚡ Stroke Feel Distance", f"{stroke_ft} ft", delta=f"~{stroke_ft / 3:.1f} paces power", delta_color="off")
    c3.metric("📈 Effective Break Factor", f"{factor:.2f}x", delta=f"{severity} break severity", delta_color="off")
    st.markdown("---")
    st.subheader("Top-Down Trajectory")
    st.caption("Green line: where to aim. Blue curve: where the ball is expected to roll. "
               "Stroke feel is the flat-green distance that gives the same roll speed.")
    st.pyplot(trajectory_chart(sol), use_container_width=True)
    if abs(sol["hit_error_ft"]) > 0.15:
        st.warning("The solver could not make this putt drop with the current relief setting. "
                   "Try lowering Green relief or checking the color direction.")

with tab_map:
    st.caption("Tap the map to place the marker chosen in the sidebar. Black arrows show downhill as the app "
               "reads it; if they point opposite to the arrows on the map, flip the color setting.")
    shown = draw_heat_overlay(img, geom, ball, hole, sol, sx, sy, meta, show_arrows)
    clicked = streamlit_image_coordinates(shown, key="map_click", width=DISPLAY_WIDTH)
    if clicked and clicked != st.session_state.get("last_click"):
        st.session_state.last_click = clicked
        scale = img.width / float(DISPLAY_WIDTH)
        cx, cy = pe.px_to_ft(geom, clicked["x"] * scale, clicked["y"] * scale)
        key = "hole_coords" if "Hole" in placement_mode else "ball_coords"
        st.session_state[key] = {"x_ft": cx, "y_ft": cy}
        st.rerun()

    with st.expander("📍 Coordinates & comparison", expanded=True):
        old_off, old_side = classic_read(img, geom, ball, hole, stimp)
        st.write(f"**Ball:** {ball[0]:.1f} ft, {ball[1]:.1f} ft   **Hole:** {hole[0]:.1f} ft, {hole[1]:.1f} ft   "
                 f"**Distance:** {sol['dist_ft']:.1f} ft")
        st.write(f"**New read:** {format_feet_inches(sol['aim_offset_ft'])} {sol['aim_side']}   "
                 f"(max break along the path {format_feet_inches(sol['max_break_ft'])})")
        st.write(f"**Old read:** {format_feet_inches(old_off)} {old_side}")
        st.write("**Break:** " + ("Right-to-Left" if sol["aim_side"] == "Right" else "Left-to-Right"))
