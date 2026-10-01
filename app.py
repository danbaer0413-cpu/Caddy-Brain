import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# --- INITIALIZE SESSION STATE ---
if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {
            i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)
        }
    }

if "ball_coords" not in st.session_state:
    st.session_state.ball_coords = {"x_ft": 12.0, "y_ft": 4.0}
if "hole_coords" not in st.session_state:
    st.session_state.hole_coords = {"x_ft": 7.0, "y_ft": 36.0}

# --- HELPER: CONVERT FEET TO FEET & INCHES ---
def format_feet_inches(total_feet):
    negative = total_feet < 0
    total_feet = abs(total_feet)
    ft = int(total_feet)
    inches = round((total_feet - ft) * 12)
    if inches == 12:
        ft += 1
        inches = 0
    if ft == 0 and inches == 0:
        return "Straight (0 in)"
    elif ft == 0:
        return f"{inches} in"
    result_str = f"{ft} ft {inches} in"
    return f"-{result_str}" if negative else result_str

# --- APP LAYOUT ---
st.title("⛳ CaddyBrain: Refined Green Reading Assistant")

# Sidebar for Course Selection & Management
st.sidebar.header("1. Course & Hole Setup")
selected_course = st.sidebar.selectbox("Select Course", list(st.session_state.courses_db.keys()))
selected_hole = st.sidebar.selectbox("Select Hole", list(st.session_state.courses_db[selected_course].keys()))

with st.sidebar.expander("➕ Add New Course"):
    new_course_input = st.text_input("Home Course Name")
    if st.button("Create Course", type="primary"):
        if new_course_input and new_course_input not in st.session_state.courses_db:
            st.session_state.courses_db[new_course_input] = {
                i: {"max_depth_yds": 25.0, "width_yds": 15.0} for i in range(1, 19)
            }
            st.success(f"Added {new_course_input}!")
            st.rerun()

saved_depth = st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"]
saved_width = st.session_state.courses_db[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Stimp Calibration")
base_stimp = st.sidebar.slider("Base Stimp", 6.0, 12.0, 8.0)
actual_test_paces = st.sidebar.number_input("3-Pace Test Roll (paces)", min_value=0.5, max_value=10.0, value=3.2, step=0.5)
calibrated_stimp = base_stimp * (actual_test_paces / 3.0)
st.sidebar.info(f"Calibrated Stimp: **{calibrated_stimp:.1f}**")

# --- MAIN SCREEN LAYOUT ---
st.header(f"Hole #{selected_hole} Specifications ({selected_course})")
col_inputs, col_map = st.columns([1.0, 1.3])

with col_inputs:
    st.subheader("Green Dimensions")
    max_depth_yds = st.number_input("Depth (Yds)", min_value=0.0, max_value=100.0, value=float(saved_depth), step=1.0)
    green_width_yds = st.number_input("Width (Yds)", min_value=0.0, max_value=100.0, value=float(saved_width), step=1.0)

    st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"] = max_depth_yds
    st.session_state.courses_db[selected_course][selected_hole]["width_yds"] = green_width_yds

    placement_mode = st.radio("Click Mode:", ["🔴 Hole Position", "🔵 Ball Position"], horizontal=True)
    slope_steepness = st.selectbox("Contour Gradient", ["Standard Slope (Single Arrow)", "Steep Slope (Double Arrows ⚡)"])
    
    if st.button("Reset Markers", type="secondary"):
        st.session_state.ball_coords = {"x_ft": green_width_yds * 1.5, "y_ft": 10.0}
        st.session_state.hole_coords = {"x_ft": green_width_yds * 1.5, "y_ft": 30.0}
        st.rerun()

# Locate assets
course_folder = selected_course.lower().replace(" ", "_").replace("(", "").replace(")", "")
heat_path, contour_path = None, None

for filename in [f"{selected_hole}_Heat.png", f"{selected_hole}_heat.png", f"{selected_hole}_Heat.PNG", f"{selected_hole}_heat.PNG"]:
    path = f"assets/{course_folder}/{filename}"
    if os.path.exists(path): heat_path = path; break

for filename in [f"{selected_hole}_Contour.JPG", f"{selected_hole}_contour.JPG", f"{selected_hole}_Contour.jpg", f"{selected_hole}_contour.jpg", f"{selected_hole}_Contour.jpeg", f"{selected_hole}_Contour.JPEG"]:
    path = f"assets/{course_folder}/{filename}"
    if os.path.exists(path): contour_path = path; break

base_img_path = heat_path if heat_path else contour_path

# Calculations & Multi-Point Sampling Engine
ft_per_pace = 3.0
green_width_ft = green_width_yds * ft_per_pace

x_hole = st.session_state.hole_coords["x_ft"]
y_hole = st.session_state.hole_coords["y_ft"]
x_ball = st.session_state.ball_coords["x_ft"]
y_ball = st.session_state.ball_coords["y_ft"]

straight_dist_ft = np.sqrt((x_hole - x_ball)**2 + (y_hole - y_ball)**2)
straight_paces = straight_dist_ft / ft_per_pace

slope_drop = 0.0
interactive_display_img = None

if base_img_path:
    try:
        raw_img = Image.open(base_img_path).convert("RGB")
        contour_img = Image.open(contour_path).convert("RGB") if contour_path else None
        
        img_w, img_h = raw_img.size
        box_x_min, box_x_max = img_w * 0.22, img_w * 0.78
        box_y_bottom = img_h * 0.88
        box_height = max_depth_yds * ((img_h * 0.76) / 28.0)
        
        def ft_to_pixels(x_ft, y_ft):
            max_depth_ft = max_depth_yds * ft_per_pace
            px = box_x_min + (x_ft / green_width_ft) * (box_x_max - box_x_min)
            py = box_y_bottom - (y_ft / max_depth_ft) * box_height
            return int(max(0, min(px, img_w - 1))), int(max(0, min(py, img_h - 1)))

        def pixels_to_ft(px, py):
            max_depth_ft = max_depth_yds * ft_per_pace
            x_ft = ((px - box_x_min) / (box_x_max - box_x_min)) * green_width_ft
            y_ft = ((box_y_bottom - py) / box_height) * max_depth_ft
            return max(0.0, min(x_ft, green_width_ft)), max(0.0, min(y_ft, max_depth_ft))

        # --- MULTI-POINT PATH SAMPLING (Heat + Contour Cross-Reference) ---
        bx_px, by_px = ft_to_pixels(x_ball, y_ball)
        hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
        
        sample_breaks = []
        sample_offset = int(max(4, (box_x_max - box_x_min) * 0.04))
        gradient_multiplier = 1.3 if "Double" in slope_steepness else 0.9

        # Sample across 5 checkpoints along the putt line
        for t_val in [0.2, 0.4, 0.6, 0.8]:
            chk_x = bx_px + t_val * (hx_px - bx_px)
            chk_y = by_px + t_val * (hy_px - by_px)
            
            left_p = (int(max(0, chk_x - sample_offset)), int(chk_y))
            right_p = (int(min(img_w - 1, chk_x + sample_offset)), int(chk_y))
            
            # Heat map sampling
            h_left = raw_img.getpixel(left_p)
            h_right = raw_img.getpixel(right_p)
            heat_lateral = (h_right[0] - h_right[2] - (h_left[0] - h_left[2])) / 80.0
            
            # Contour cross-reference sampling if available
            if contour_img:
                c_left = contour_img.getpixel(left_p)
                c_right = contour_img.getpixel(right_p)
                contour_lateral = (int(c_right[0]) - int(c_left[0])) / 100.0
                combined_lateral = (heat_lateral * 0.6) + (contour_lateral * 0.4)
            else:
                combined_lateral = heat_lateral
                
            sample_breaks.append(combined_lateral)

        if sample_breaks:
            avg_lateral = sum(sample_breaks) / len(sample_breaks)
            slope_drop = avg_lateral * gradient_multiplier
        
        # Fallback if pixel data yields zero
        if abs(slope_drop) < 0.001:
            slope_drop = ((0.02 * x_hole - 0.015 * y_hole) - (0.02 * x_ball - 0.015 * y_hole)) * gradient_multiplier

        # Refined Aim Calculation (Damped for short putts to prevent over-breaking)
        aim_offset_ft = slope_drop * (calibrated_stimp / 8.0) * 0.18 * straight_dist_ft
        aim_side = "Left" if slope_drop < 0 else "Right"
        aim_ft_val = abs(aim_offset_ft)

        elevation_speed_adj = abs(slope_drop) * 0.2
        recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj + 0.2
        recommended_speed_paces = max(1.0, round(recommended_speed_paces, 1))

        # Draw overlays
        draw_img = raw_img.copy()
        draw = ImageDraw.Draw(draw_img)
        
        target_x_ft = x_hole + aim_offset_ft
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
        
        dot_r = 6
        draw.ellipse([bx_px - dot_r, by_px - dot_r, bx_px + dot_r, by_px + dot_r], fill="blue", outline="white", width=2)
        draw.ellipse([hx_px - dot_r, hy_px - dot_r, hx_px + dot_r, hy_px + dot_r], fill="red", outline="white", width=2)
        draw.ellipse([tx_px - (dot_r-1), ty_px - (dot_r-1), tx_px + (dot_r-1), ty_px + (dot_r-1)], fill="cyan", outline="black", width=2)
        
        interactive_display_img = draw_img

    except Exception as e:
        st.warning(f"Error processing maps: {e}")

with col_map:
    st.subheader("🗺️ Click Map to Position Markers")
    if interactive_display_img:
        clicked = streamlit_image_coordinates(interactive_display_img, key="map_click")
        if clicked is not None:
            clicked_x_ft, clicked_y_ft = pixels_to_ft(clicked["x"], clicked["y"])
            if "Hole" in placement_mode:
                st.session_state.hole_coords = {"x_ft": clicked_x_ft, "y_ft": clicked_y_ft}
            else:
                st.session_state.ball_coords = {"x_ft": clicked_x_ft, "y_ft": clicked_y_ft}
            st.rerun()
    else:
        st.info("Map assets not found.")

# --- SOLUTION READOUT ---
st.markdown("---")
st.success("Refined Target Solution Readout:")
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(f"### 🎯 **Distance:** {format_feet_inches(straight_dist_ft)}")
with r2:
    st.markdown(f"### ➡ **Aim Point:** {format_feet_inches(aim_ft_val)} {aim_side}")
with r3:
    st.markdown(f"### ⚡ **Stroke Speed:** {recommended_speed_paces}-pace power")

st.markdown(
    """
    | Marker / Line | Description |
    | :--- | :--- |
    | 🔵 **Blue Dot** | Ball Position |
    | 🔴 **Red Dot** | Hole / Cup Position |
    | 🩵 **Cyan Dot** | Target Aim Point Abeam with Hole (in Feet & Inches) |
    | 🟡 **Yellow Dashed Line** | Anticipated Break Path |
    """
)