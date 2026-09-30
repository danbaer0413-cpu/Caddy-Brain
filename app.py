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

# --- DUAL-MAP VISION ENGINE (CONTOUR BOUNDARY + HEAT ELEVATION) ---
if st.button("Calculate Putt Solution", type="primary"):
    ft_per_pace = 3.0
    green_width_ft = green_width_yds * ft_per_pace
    
    # Hole Coordinates
    y_hole = hole_from_front * ft_per_pace
    x_hole = hole_from_side * ft_per_pace if side_ref.lower() == 'left' else green_width_ft - (hole_from_side * ft_per_pace)
        
    # Ball Coordinates
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
    
    # DUAL IMAGE PROCESSING
    if heat_path and contour_path:
        try:
            heat_img = Image.open(heat_path).convert("RGB")
            contour_img = Image.open(contour_path).convert("RGB")
            img_w, img_h = heat_img.size
            max_depth_ft = max_depth_yds * ft_per_pace
            
            # --- AUTOMATED GREEN BOUNDARY DETECTION FROM CONTOUR MAP ---
            # We convert the contour map to grayscale to locate the dark outline boundaries of the green
            gray_contour = contour_img.convert("L")
            arr_contour = np.array(gray_contour)
            
            # Find pixel rows and columns where the green boundary lines/content exist (non-white areas)
            # Threshold to find non-background pixels (assuming white background > 240)
            green_mask = arr_contour < 245
            y_indices, x_indices = np.where(green_mask)
            
            if len(x_indices) > 0 and len(y_indices) > 0:
                box_x_min, box_x_max = float(x_indices.min()), float(x_indices.max())
                box_y_min, box_y_max = float(y_indices.min()), float(y_indices.max())
            else:
                # Fallback box if empty
                box_x_min, box_x_max = img_w * 0.22, img_w * 0.78
                box_y_min, box_y_max = img_h * 0.12, img_h * 0.88
                
            box_width = box_x_max - box_x_min
            box_height = box_y_max - box_y_min
            
            def ft_to_pixels(x_ft, y_ft):
                px = box_x_min + (x_ft / green_width_ft) * box_width
                py = box_y_max - (y_ft / max_depth_ft) * box_height
                return int(max(0, min(px, img_w - 1))), int(max(0, min(py, img_h - 1)))
                
            bx_px, by_px = ft_to_pixels(x_ball, y_ball)
            hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
            
            # Sample heat map colors for elevation/slope
            b_rgb = heat_img.getpixel((bx_px, by_px))
            h_rgb = heat_img.getpixel((hx_px, hy_px))
            
            ball_elevation_score = b_rgb[0] - b_rgb[2]  
            hole_elevation_score = h_rgb[0] - h_rgb[2]  
            
            elevation_diff = (hole_elevation_score - ball_elevation_score) / 50.0
            side_color_shift = (b_rgb[0] - b_rgb[1]) - (h_rgb[0] - h_rgb[1])
            
            gradient_multiplier = 2.0 if "Double" in slope_steepness else 1.0
            slope_drop = (elevation_diff + (side_color_shift / 100.0) * (x_hole - x_ball) / green_width_ft) * gradient_multiplier
            
            # --- DRAW OVERLAY ON HEAT MAP FOR PROOF ---
            annotated_img = heat_img.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            draw.line([(bx_px, by_px), (hx_px, hy_px)], fill="yellow", width=6)
            draw.ellipse([bx_px - 12, by_px - 12, bx_px + 12, by_px + 12], fill="blue", outline="white", width=3)
            draw.ellipse([hx_px - 12, hy_px - 12, hx_px + 12, hy_px + 12], fill="red", outline="white", width=3)
            
        except Exception as e:
            st.warning(f"Dual-map vision parsing error: {e}")
            slope_drop = 0.05 * (x_hole - x_ball)
    elif heat_path:
        # Fallback if contour is missing
        try:
            img = Image.open(heat_path).convert("RGB")
            img_w, img_h = img.size
            max_depth_ft = max_depth_yds * ft_per_pace
            box_x_min, box_x_max = img_w * 0.22, img_w * 0.78
            box_y_min, box_y_max = img_h * 0.12, img_h * 0.88
            box_width = box_x_max - box_x_min
            box_height = box_y_max - box_y_min
            
            def ft_to_pixels(x_ft, y_ft):
                px = box_x_min + (x_ft / green_width_ft) * box_width
                py = box_y_max - (y_ft / max_depth_ft) * box_height
                return int(max(0, min(px, img_w - 1))), int(max(0, min(py, img_h - 1)))
                
            bx_px, by_px = ft_to_pixels(x_ball, y_ball)
            hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
            b_rgb = img.getpixel((bx_px, by_px))
            h_rgb = img.getpixel((hx_px, hy_px))
            elevation_diff = ((h_rgb[0] - h_rgb[2]) - (b_rgb[0] - b_rgb[2])) / 50.0
            slope_drop = elevation_diff
            
            annotated_img = img.copy()
            draw = ImageDraw.Draw(annotated_img)
            draw.line([(bx_px, by_px), (hx_px, hy_px)], fill="yellow", width=6)
            draw.ellipse([bx_px - 12, by_px - 12, bx_px + 12, by_px + 12], fill="blue", outline="white", width=3)
            draw.ellipse([hx_px - 12, by_px - 12, hx_px + 12, hy_px + 12], fill="red", outline="white", width=3)
        except Exception:
            slope_drop = 0.05 * (x_hole - x_ball)

    if slope_drop == 0.0:
        gradient_multiplier = 2.0 if "Double" in slope_steepness else 1.0
        slope_drop = ((0.04 * x_hole - 0.03 * y_hole) - (0.04 * x_ball - 0.03 * y_ball)) * gradient_multiplier

    # Aim Calculation
    aim_offset_ft = slope_drop * calibrated_stimp * 0.40 * (straight_dist_ft / 10.0)
    aim_side = "Left" if slope_drop < 0 else "Right"
    aim_ft_val = abs(aim_offset_ft)
    aim_paces_val = aim_ft_val / ft_per_pace
    
    # Speed Recommendation
    elevation_speed_adj = abs(slope_drop) * 0.4
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj + 0.3
    recommended_speed_paces = max(1.0, round(recommended_speed_paces, 1))

    # --- OUTPUTS & VISUAL PROOF ---
    st.success("Target Solution Readout (Dual-Map Parsed):")
    st.markdown(f"### 🎯 **Putt Distance:** {format_feet_inches(straight_dist_ft)} ({straight_paces:.1f} paces)")
    st.markdown(f"### ➡️️ **Aim Point:** {format_feet_inches(aim_ft_val)} ({aim_paces_val:.1f} paces) {aim_side}")
    st.markdown(f"### ⚡ **Stroke Speed:** Putt with **{recommended_speed_paces}-pace** power stroke")
    
    if annotated_img:
        st.subheader("🔍 Dual-Map Vision Engine Overlay Proof")
        st.image(annotated_img, caption="Contour Map used for Green Boundaries | Heat Map sampled for Slope Elevation", use_container_width=True)