import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw
from io import BytesIO
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt

# --- 1. CONFIG & BULLETPROOF SESSION STATE ---
st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="wide")

if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {
            i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)
        }
    }

if "ball_coords" not in st.session_state or not isinstance(st.session_state.ball_coords, dict) or "x_ft" not in st.session_state.ball_coords or "y_ft" not in st.session_state.ball_coords:
    st.session_state.ball_coords = {"x_ft": 7.0, "y_ft": 4.0}

if "hole_coords" not in st.session_state or not isinstance(st.session_state.hole_coords, dict) or "x_ft" not in st.session_state.hole_coords or "y_ft" not in st.session_state.hole_coords:
    st.session_state.hole_coords = {"x_ft": 7.0, "y_ft": 30.0}

# --- 2. FORMATTER & ENGINE ---
def format_feet_inches(total_feet):
    negative = total_feet < 0
    total_feet = abs(total_feet)
    ft = int(total_feet)
    inches = round((total_feet - ft) * 12)
    if inches == 12:
        ft += 1
        inches = 0
    if ft == 0 and inches == 0:
        return "0 in"
    elif ft == 0:
        return f"{inches} in"
    result_str = f"{ft} ft {inches} in"
    return f"-{result_str}" if negative else result_str

def calculate_putt_solution(x_ball, y_ball, x_hole, y_hole, max_depth_yds, green_width_ft, calibrated_stimp, raw_img, contour_img, slope_steepness, display_width=650):
    orig_w, orig_h = raw_img.size
    
    # Scale factor from displayed component width back to original high-res image dimensions
    scale = orig_w / float(display_width)

    box_x_min, box_x_max = orig_w * 0.15, orig_w * 0.85
    box_y_bottom = orig_h * 0.92
    box_height = orig_h * 0.80
    
    def ft_to_pixels(x_ft, y_ft):
        max_depth_ft = max_depth_yds * 3.0
        px = box_x_min + (x_ft / green_width_ft) * (box_x_max - box_x_min)
        py = box_y_bottom - (y_ft / max_depth_ft) * box_height
        return int(max(0, min(px, orig_w - 1))), int(max(0, min(py, orig_h - 1)))

    def pixels_to_ft(display_px, display_py):
        px = display_px * scale
        py = display_py * scale
        
        max_depth_ft = max_depth_yds * 3.0
        x_ft = ((px - box_x_min) / (box_x_max - box_x_min)) * green_width_ft
        y_ft = ((box_y_bottom - py) / box_height) * max_depth_ft
        return max(0.0, min(x_ft, green_width_ft)), max(0.0, min(y_ft, max_depth_ft))

    straight_dist_ft = np.sqrt((x_hole - x_ball)**2 + (y_hole - y_ball)**2)
    straight_paces = straight_dist_ft / 3.0

    bx_px, by_px = ft_to_pixels(x_ball, y_ball)
    hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
    
    sample_breaks = []
    sample_offset = int(max(4, (box_x_max - box_x_min) * 0.04))
    gradient_multiplier = 1.3 if "Double" in slope_steepness else 0.9

    for t_val in [0.2, 0.4, 0.6, 0.8]:
        chk_x = bx_px + t_val * (hx_px - bx_px)
        chk_y = by_px + t_val * (hy_px - by_px)
        
        left_p = (int(max(0, chk_x - sample_offset)), int(chk_y))
        right_p = (int(min(orig_w - 1, chk_x + sample_offset)), int(chk_y))
        
        try:
            h_left = raw_img.getpixel(left_p)
            h_right = raw_img.getpixel(right_p)
            heat_lateral = (int(h_right[0]) - int(h_right[2]) - (int(h_left[0]) - int(h_left[2]))) / 60.0
        except:
            heat_lateral = 0.0
            
        if contour_img:
            try:
                c_left = contour_img.getpixel(left_p)
                c_right = contour_img.getpixel(right_p)
                contour_lateral = (int(c_right[0]) - int(c_left[0])) / 80.0
                combined_lateral = (heat_lateral * 0.6) + (contour_lateral * 0.4)
            except:
                combined_lateral = heat_lateral
        else:
            combined_lateral = heat_lateral
            
        sample_breaks.append(combined_lateral)

    slope_drop = (sum(sample_breaks) / len(sample_breaks)) * gradient_multiplier if sample_breaks else 0.0

    aim_offset_ft = slope_drop * (calibrated_stimp / 8.0) * 0.15 * straight_dist_ft
    aim_side = "Left" if slope_drop < 0 else "Right"
    aim_ft_val = abs(aim_offset_ft)

    elevation_speed_adj = abs(slope_drop) * 0.2
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj + 0.2
    recommended_speed_paces = max(1.0, round(recommended_speed_paces, 1))

    draw_img = raw_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    target_x_ft = x_hole - aim_offset_ft if aim_side == "Left" else x_hole + aim_offset_ft
    target_y_ft = y_hole
    tx_px, ty_px = ft_to_pixels(target_x_ft, target_y_ft)

    dx_line = hx_px - bx_px
    dy_line = hy_px - by_px
    line_len = np.sqrt(dx_line**2 + dy_line**2)
    
    mid_x, mid_y = (bx_px + hx_px) / 2, (by_px + hy_px) / 2
    if line_len > 0:
        nx, ny = -dy_line / line_len, dx_line / line_len
        break_shift = slope_drop * calibrated_stimp * 1.2
        control_x, control_y = mid_x + nx * break_shift, mid_y + ny * break_shift
    else:
        control_x, control_y = mid_x, mid_y
        
    curve_points = [( (1 - t)**2 * bx_px + 2 * (1 - t) * t * control_x + t**2 * hx_px, 
                      (1 - t)**2 * by_px + 2 * (1 - t) * t * control_y + t**2 * hy_px ) 
                    for t in np.linspace(0, 1, 50)]
        
    for i in range(len(curve_points) - 1):
        if i % 2 == 0:
            draw.line([curve_points[i], curve_points[i+1]], fill="yellow", width=4)
    
    dot_r = 7
    draw.ellipse([bx_px - dot_r, by_px - dot_r, bx_px + dot_r, by_px + dot_r], fill="blue", outline="white", width=2)
    draw.ellipse([hx_px - dot_r, hy_px - dot_r, hx_px + dot_r, hy_px + dot_r], fill="red", outline="white", width=2)
    draw.ellipse([tx_px - (dot_r-1), ty_px - (dot_r-1), tx_px + (dot_r-1), ty_px + (dot_r-1)], fill="cyan", outline="black", width=2)
    
    return draw_img, straight_dist_ft, aim_ft_val, aim_side, recommended_speed_paces, pixels_to_ft, slope_drop

# --- 3. SIDEBAR SETUP ---
st.sidebar.header("1. Course & Hole Setup")
selected_course = st.sidebar.selectbox("Select Course", list(st.session_state.courses_db.keys()))
selected_hole = st.sidebar.selectbox("Select Hole", list(st.session_state.courses_db[selected_course].keys()))

saved_depth = st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"]
saved_width = st.session_state.courses_db[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Stimp Calibration")
base_stimp = st.sidebar.slider("Base Stimp", 6.0, 12.0, 8.0)
actual_test_paces = st.sidebar.number_input("3-Pace Test Roll (paces)", min_value=0.5, max_value=10.0, value=3.2, step=0.5)
calibrated_stimp = base_stimp * (actual_test_paces / 3.0)
st.sidebar.info(f"Calibrated Stimp: **{calibrated_stimp:.1f}**")

# Asset Path Resolver
course_folder = selected_course.lower().replace(" ", "_").replace("(", "").replace(")", "")
heat_path, contour_path = None, None

for filename in [f"{selected_hole}_Heat.png", f"{selected_hole}_heat.png", f"{selected_hole}_Heat.PNG", f"{selected_hole}_heat.PNG"]:
    path = f"assets/{course_folder}/{filename}"
    if os.path.exists(path): heat_path = path; break

for filename in [f"{selected_hole}_Contour.JPG", f"{selected_hole}_contour.JPG", f"{selected_hole}_Contour.jpg", f"{selected_hole}_contour.jpg", f"{selected_hole}_Contour.jpeg", f"{selected_hole}_Contour.JPEG"]:
    path = f"assets/{course_folder}/{filename}"
    if os.path.exists(path): contour_path = path; break

base_img_path = heat_path if heat_path else contour_path

# Execute Engine
straight_dist_ft = aim_ft_val = speed_paces = slope_drop = 0.0
aim_side = "Right"
interactive_display_img = None
contour_display_img = None
pixels_to_ft_func = None
max_depth_yds = saved_depth
green_width_yds = saved_width
DISPLAY_WIDTH = 650  # Enlarged width for precision clicking

if base_img_path:
    try:
        raw_img = Image.open(base_img_path).convert("RGB")
        cont_img = Image.open(contour_path).convert("RGB") if contour_path and os.path.exists(contour_path) else None
        
        annotated_img, straight_dist_ft, aim_ft_val, aim_side, speed_paces, pixels_to_ft_func, slope_drop = calculate_putt_solution(
            x_ball=st.session_state.ball_coords["x_ft"],
            y_ball=st.session_state.ball_coords["y_ft"],
            x_hole=st.session_state.hole_coords["x_ft"],
            y_hole=st.session_state.hole_coords["y_ft"],
            max_depth_yds=saved_depth,
            green_width_ft=saved_width * 3.0,
            calibrated_stimp=calibrated_stimp,
            raw_img=raw_img,
            contour_img=cont_img,
            slope_steepness="Standard Slope (Single Arrow)",
            display_width=DISPLAY_WIDTH
        )
        interactive_display_img = annotated_img
        
        if cont_img:
            cont_annotated, _, _, _, _, _, _ = calculate_putt_solution(
                x_ball=st.session_state.ball_coords["x_ft"],
                y_ball=st.session_state.ball_coords["y_ft"],
                x_hole=st.session_state.hole_coords["x_ft"],
                y_hole=st.session_state.hole_coords["y_ft"],
                max_depth_yds=saved_depth,
                green_width_ft=saved_width * 3.0,
                calibrated_stimp=calibrated_stimp,
                raw_img=cont_img,
                contour_img=None,
                slope_steepness="Standard Slope (Single Arrow)",
                display_width=DISPLAY_WIDTH
            )
            contour_display_img = cont_annotated
    except Exception as e:
        st.warning(f"Error loading maps: {e}")

# --- 4. MAIN INTERFACE TABS ---
st.title(f"⛳ Hole #{selected_hole} ({selected_course})")

tab_dashboard, tab_heat, tab_contour = st.tabs([
    "📊 Trajectory & Metrics Dashboard", 
    "🔥 Interactive Heat Map", 
    "🗺️ Contour"
])

with tab_dashboard:
    c1, c2, c3 = st.columns(3)
    with c1:
        aim_str = f"{format_feet_inches(aim_ft_val)} {aim_side}" if aim_ft_val > 0 else "Straight (0 in)"
        cup_widths = round(aim_ft_val / 0.25, 1)
        st.metric(label="🎯 Aim Offset", value=aim_str, delta=f"~{cup_widths} cup widths {aim_side.lower()}")
    with c2:
        stroke_feel_ft = round(straight_dist_ft * (calibrated_stimp / 8.0), 1)
        st.metric(label="⚡ Stroke Feel Distance", value=f"{stroke_feel_ft} ft", delta=f"~{speed_paces} paces power")
    with c3:
        break_severity = "Severe break" if abs(slope_drop) > 1.0 else "Moderate break severity"
        st.metric(label="📈 Effective Break Factor", value=f"{1.0 + abs(slope_drop):.2f}x", delta=break_severity)

    st.markdown("---")
    st.subheader("Top-Down Trajectory Visualizer *(Aim Line vs True Curve)*")
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xticks(np.arange(0, green_width_yds * 3.0 + 3, 2))
    ax.set_yticks(np.arange(0, 60, 5))
    ax.grid(True, linestyle="--", alpha=0.5, color="#dcdcdc")
    
    bx, by = st.session_state.ball_coords["x_ft"], st.session_state.ball_coords["y_ft"]
    hx, hy = st.session_state.hole_coords["x_ft"], st.session_state.hole_coords["y_ft"]
    
    aim_x = hx - aim_ft_val if aim_side == "Left" else hx + aim_ft_val
    
    ax.plot([bx, aim_x], [by, hy], color="#2b5c8f", linewidth=2, label="Aim Line")
    ax.scatter([aim_x], [hy], color="#2b5c8f", s=60, zorder=5)
    ax.text(aim_x, hy + 1.5, f"Aim ({format_feet_inches(aim_ft_val)})", color="#2b5c8f", fontweight="bold", ha="center")
    
    t_vals = np.linspace(0, 1, 50)
    curve_direction_factor = -1.0 if aim_side == "Left" else 1.0
    curve_x = (1 - t_vals)**2 * bx + 2 * (1 - t_vals) * t_vals * ((bx + hx)/2 + curve_direction_factor * abs(slope_drop)*2.5) + t_vals**2 * hx
    curve_y = (1 - t_vals)**2 * by + 2 * (1 - t_vals) * t_vals * (by + hy)/2 + t_vals**2 * hy
    ax.plot(curve_x, curve_y, color="#2e7d32", linewidth=3.5, label="True Curve")
    
    ax.plot([hx, hx], [by, hy], color="gray", linestyle=":", linewidth=1.5)
    ax.scatter([bx], [by], color="black", s=80, zorder=6, label="Ball")
    ax.scatter([hx], [hy], color="black", s=80, zorder=6, label="Hole")
    
    ax.set_xlim(-1, green_width_yds * 3.0 + 1)
    ax.set_ylim(min(by, hy) - 4, max(by, hy) + 6)
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    
    st.pyplot(fig)
    st.caption(f"Calculated for a {round(straight_dist_ft, 1)} ft putt with Stimp {calibrated_stimp:.1f}.")

with tab_heat:
    col_ctrl, col_heat_img = st.columns([1.2, 2.2])
    
    with col_ctrl:
        st.subheader("Marker Controls")
        placement_mode = st.radio("Click Mode:", ["🔴 Hole Position", "🔵 Ball Position"], key="map_click_mode")
        slope_steepness = st.selectbox("Contour Gradient", ["Standard Slope (Single Arrow)", "Steep Slope (Double Arrows ⚡)"])
        
        if st.button("Reset Markers to Default", type="secondary"):
            st.session_state.ball_coords = {"x_ft": green_width_yds * 1.5, "y_ft": 4.0}
            st.session_state.hole_coords = {"x_ft": green_width_yds * 1.5, "y_ft": 30.0}
            st.rerun()
            
        st.markdown("---")
        st.markdown(f"**Current Ball:** `X: {st.session_state.ball_coords['x_ft']:.1f}ft, Y: {st.session_state.ball_coords['y_ft']:.1f}ft`")
        st.markdown(f"**Current Hole:** `X: {st.session_state.hole_coords['x_ft']:.1f}ft, Y: {st.session_state.hole_coords['y_ft']:.1f}ft`")

    with col_heat_img:
        st.subheader("🔥 Interactive Heat Map")
        if interactive_display_img and pixels_to_ft_func:
            clicked_heat = streamlit_image_coordinates(interactive_display_img, key="heat_map_click", width=DISPLAY_WIDTH)
            if clicked_heat is not None:
                cx, cy = pixels_to_ft_func(clicked_heat["x"], clicked_heat["y"])
                if "Hole" in placement_mode:
                    st.session_state.hole_coords = {"x_ft": cx, "y_ft": cy}
                else:
                    st.session_state.ball_coords = {"x_ft": cx, "y_ft": cy}
                st.rerun()
        else:
            st.info("Heat map asset not found.")

with tab_contour:
    st.subheader(f"🗺️ Contour Map — Hole #{selected_hole}")
    if contour_display_img:
        st.image(contour_display_img, width=DISPLAY_WIDTH)
    else:
        st.info("Contour map asset not found for this hole.")