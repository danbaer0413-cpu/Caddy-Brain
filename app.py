import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw

# --- INITIALIZE SESSION STATE FOR MULTI-COURSE DATABASE ---
if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {
            i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)
        }
    }

# --- HELPER: CONVERT FEET TO FEET & INCHES ---
def format_feet_inches(total_feet):
    negative = total_feet < 0
    total_feet = abs(total_feet)
    ft = int(total_feet)
    inches = round((total_feet - ft) * 12)
    if inches == 12:
        ft += 1
        inches = 0
    result_str = f"{ft} ft {inches} in"
    return f"-{result_str}" if negative else result_str

# --- APP LAYOUT ---
st.title("⛳ CaddyBrain: Green Reading Assistant")

# Sidebar for Course Selection & Management
st.sidebar.header("1. Course & Hole Setup")
selected_course = st.sidebar.selectbox("Select Course", list(st.session_state.courses_db.keys()))
selected_hole = st.sidebar.selectbox("Select Hole", list(st.session_state.courses_db[selected_course].keys()))

# --- INTERACTIVE ADD COURSE BUTTON ---
with st.sidebar.expander("➕ Add New Course"):
    new_course_input = st.text_input("Home Course Name")
    if st.button("Create Course", type="primary"):
        if new_course_input and new_course_input not in st.session_state.courses_db:
            st.session_state.courses_db[new_course_input] = {
                i: {"max_depth_yds": 25.0, "width_yds": 15.0} for i in range(1, 19)
            }
            st.success(f"Added {new_course_input}!")
            st.rerun()
        elif new_course_input in st.session_state.courses_db:
            st.warning("Course already exists!")

# Retrieve saved defaults for this specific hole
saved_depth = st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"]
saved_width = st.session_state.courses_db[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Stimp Calibration")
base_stimp = st.sidebar.slider("Base Stimp", 6.0, 12.0, 8.0)
actual_test_paces = st.sidebar.number_input("3-Pace Test Roll (paces)", min_value=0.5, max_value=10.0, value=3.2, step=0.5)

stimp_ratio = actual_test_paces / 3.0
calibrated_stimp = base_stimp * stimp_ratio
st.sidebar.info(f"Calibrated Stimp: **{calibrated_stimp:.1f}**")

# --- MAIN SCREEN LAYOUT: INPUTS & DUAL MAP VIEWER ---
st.header(f"Hole #{selected_hole} Specifications ({selected_course})")

col_inputs, col_maps = st.columns([1.1, 1.2])

with col_inputs:
    st.subheader("Pacing & Setup")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        max_depth_yds = st.number_input("Depth (Yds)", min_value=0.0, max_value=100.0, value=float(saved_depth), step=1.0)
    with col_d2:
        green_width_yds = st.number_input("Width (Yds)", min_value=0.0, max_value=100.0, value=float(saved_width), step=1.0)

    st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"] = max_depth_yds
    st.session_state.courses_db[selected_course][selected_hole]["width_yds"] = green_width_yds

    col3, col4 = st.columns(2)
    with col3:
        hole_from_front = st.number_input("Hole from Front (paces)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
        hole_from_side = st.number_input("Hole from Side (paces)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
        side_ref = st.selectbox("Side Ref", ["Left", "Right"])

    with col4:
        ball_offset_paces = st.number_input("Ball to Hole (paces)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
        ball_direction = st.selectbox("Ball Position", ["Right", "Left", "Front", "Back"])
        slope_steepness = st.selectbox("Contour Gradient", ["Standard Slope (Single Arrow)", "Steep Slope (Double Arrows ⚡)"])

with col_maps:
    st.subheader("📖 Green Book Maps")
    m_col1, m_col2 = st.columns(2)
    
    course_folder = selected_course.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    heat_path = None
    for filename in [f"{selected_hole}_Heat.png", f"{selected_hole}_heat.png", f"{selected_hole}_Heat.PNG", f"{selected_hole}_heat.PNG"]:
        path = f"assets/{course_folder}/{filename}"
        if os.path.exists(path):
            heat_path = path
            break

    contour_path = None
    for filename in [f"{selected_hole}_Contour.JPG", f"{selected_hole}_contour.JPG", f"{selected_hole}_Contour.jpg", f"{selected_hole}_contour.jpg", f"{selected_hole}_Contour.jpeg", f"{selected_hole}_Contour.JPEG"]:
        path = f"assets/{course_folder}/{filename}"
        if os.path.exists(path):
            contour_path = path
            break
    
    with m_col1:
        if heat_path:
            st.image(heat_path, caption=f"Hole {selected_hole} Heat Map", use_container_width=True)
        else:
            st.info(f"Missing Heat Map for Hole {selected_hole}")
            
    with m_col2:
        if contour_path:
            st.image(contour_path, caption=f"Hole {selected_hole} Contour", use_container_width=True)
        else:
            st.info(f"Missing Contour Map for Hole {selected_hole}")

# --- DUAL-MAP VISION ENGINE WITH LOCAL LATERAL GRADIENT SAMPLING ---
if st.button("Calculate Putt Solution", type="primary"):
    ft_per_pace = 3.0
    green_width_ft = green_width_yds * ft_per_pace
    
    # Hole Coordinates
    y_hole = hole_from_front * ft_per_pace
    x_hole = hole_from_side * ft_per_pace if side_ref.lower() == 'left' else green_width_ft - (hole_from_side * ft_per_pace)
        
    # Ball Coordinates (Orientation Relative to Hole)
    dist_ft = ball_offset_paces * ft_per_pace
    if ball_direction.lower() == 'right':
        x_ball, y_ball = x_hole + dist_ft, y_hole
    elif ball_direction.lower() == 'left':
        x_ball, y_ball = x_hole - dist_ft, y_hole
    elif ball_direction.lower() == 'front':
        x_ball, y_ball = x_hole, y_hole - dist_ft
    elif ball_direction.lower() == 'back':
        x_ball, y_ball = x_hole, y_hole + dist_ft
    else:
        x_ball, y_ball = x_hole + dist_ft, y_hole

    straight_dist_ft = np.sqrt((x_hole - x_ball)**2 + (y_hole - y_ball)**2)
    straight_paces = straight_dist_ft / ft_per_pace

    slope_drop = 0.0
    annotated_img = None
    
    if heat_path:
        try:
            heat_img = Image.open(heat_path).convert("RGB")
            img_w, img_h = heat_img.size
            
            box_x_min, box_x_max = img_w * 0.22, img_w * 0.78
            box_y_bottom = img_h * 0.88
            box_height = max_depth_yds * ((img_h * 0.76) / 28.0)
            
            def ft_to_pixels(x_ft, y_ft):
                max_depth_ft = max_depth_yds * ft_per_pace
                px = box_x_min + (x_ft / green_width_ft) * (box_x_max - box_x_min)
                py = box_y_bottom - (y_ft / max_depth_ft) * box_height
                return int(max(0, min(px, img_w - 1))), int(max(0, min(py, img_h - 1)))
                
            bx_px, by_px = ft_to_pixels(x_ball, y_ball)
            hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
            
            mid_x_px = int((bx_px + hx_px) / 2)
            mid_y_px = int((by_px + hy_px) / 2)
            sample_offset = int(max(5, (box_x_max - box_x_min) * 0.05))
            
            left_px = (max(0, mid_x_px - sample_offset), mid_y_px)
            right_px = (min(img_w - 1, mid_x_px + sample_offset), mid_y_px)
            
            left_rgb = heat_img.getpixel(left_px)
            right_rgb = heat_img.getpixel(right_px)
            
            left_elev = left_rgb[0] - left_rgb[2]
            right_elev = right_rgb[0] - right_rgb[2]
            
            lateral_break = (right_elev - left_elev) / 40.0
            gradient_multiplier = 2.0 if "Double" in slope_steepness else 1.0
            slope_drop = lateral_break * gradient_multiplier
            
        except Exception as e:
            st.warning(f"Vision engine parsing error: {e}")
            slope_drop = 0.05 * (x_hole - x_ball)

    if slope_drop == 0.0:
        gradient_multiplier = 2.0 if "Double" in slope_steepness else 1.0
        slope_drop = ((0.04 * x_hole - 0.03 * y_hole) - (0.04 * x_ball - 0.03 * y_ball)) * gradient_multiplier

    # Aim & Target Calculations
    aim_offset_ft = slope_drop * calibrated_stimp * 0.40 * (straight_dist_ft / 10.0)
    aim_side = "Left" if slope_drop < 0 else "Right"
    aim_ft_val = abs(aim_offset_ft)
    aim_paces_val = aim_ft_val / ft_per_pace
    
    target_x_ft = x_hole + aim_offset_ft
    target_y_ft = y_hole
    
    # Speed Recommendation
    elevation_speed_adj = abs(slope_drop) * 0.4
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj + 0.3
    recommended_speed_paces = max(1.0, round(recommended_speed_paces, 1))

    # --- DRAW OVERLAY ON HEAT MAP (MATCHING PIXEL BOUNDS) ---
    base_img_path = heat_path if heat_path else contour_path
    if base_img_path:
        try:
            annotated_img = Image.open(base_img_path).convert("RGB")
            draw = ImageDraw.Draw(annotated_img)
            
            img_w, img_h = annotated_img.size
            box_x_min, box_x_max = img_w * 0.22, img_w * 0.78
            box_y_bottom = img_h * 0.88
            box_height = max_depth_yds * ((img_h * 0.76) / 28.0)
            
            def ft_to_pixels(x_ft, y_ft):
                max_depth_ft = max_depth_yds * ft_per_pace
                px = box_x_min + (x_ft / green_width_ft) * (box_x_max - box_x_min)
                py = box_y_bottom - (y_ft / max_depth_ft) * box_height
                return int(max(0, min(px, img_w - 1))), int(max(0, min(py, img_h - 1)))
                
            bx_px, by_px = ft_to_pixels(x_ball, y_ball)
            hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
            tx_px, ty_px = ft_to_pixels(target_x_ft, target_y_ft)
            
            mid_x = (bx_px + hx_px) / 2
            mid_y = (by_px + hy_px) / 2
            dx = hx_px - bx_px
            dy = hy_px - by_px
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                nx = -dy / length
                ny = dx / length
                break_shift = slope_drop * calibrated_stimp * 3.0
                control_x = mid_x + nx * break_shift
                control_y = mid_y + ny * break_shift
            else:
                control_x, control_y = mid_x, mid_y
                
            curve_points = []
            for t in np.linspace(0, 1, 50):
                px = (1 - t)**2 * bx_px + 2 * (1 - t) * t * control_x + t**2 * hx_px
                py = (1 - t)**2 * by_px + 2 * (1 - t) * t * control_y + t**2 * hy_px
                curve_points.append((px, py))
                
            for i in range(len(curve_points) - 1):
                if i % 2 == 0:
                    draw.line([curve_points[i], curve_points[i+1]], fill="yellow", width=4)
            
            dot_r = 5
            draw.ellipse([bx_px - dot_r, by_px - dot_r, bx_px + dot_r, by_px + dot_r], fill="blue", outline="white", width=1)
            draw.ellipse([hx_px - dot_r, hy_px - dot_r, hx_px + dot_r, hy_px + dot_r], fill="red", outline="white", width=1)
            draw.ellipse([tx_px - (dot_r-1), ty_px - (dot_r-1), tx_px + (dot_r-1), ty_px + (dot_r-1)], fill="cyan", outline="black", width=1)
            
        except Exception as e:
            st.warning(f"Overlay drawing error: {e}")

    # --- OUTPUTS & VISUAL PROOF WITH LEGEND ---
    st.success("Target Solution Readout:")
    st.markdown(f"### 🎯 **Putt Distance:** {format_feet_inches(straight_dist_ft)} ({straight_paces:.1f} paces)")
    st.markdown(f"### ➡ **Aim Point:** {format_feet_inches(aim_ft_val)} ({aim_paces_val:.1f} paces) {aim_side}")
    st.markdown(f"### ⚡ **Stroke Speed:** Putt with **{recommended_speed_paces}-pace** power stroke")
    
    if annotated_img:
        st.subheader("🔍 Visual Putt Solution Overlay")
        st.image(annotated_img, use_container_width=True)
        
        st.markdown(
            """
            | Marker / Line | Description |
            | :--- | :--- |
            | 🔵 **Blue Dot** | Ball Position |
            | 🔴 **Red Dot** | Hole (Cup) |
            | 🩵 **Cyan Dot** | Target Aim Point (Putter Start Line) |
            | 🟡 **Yellow Dashed Line** | Anticipated Break Path |
            """
        )