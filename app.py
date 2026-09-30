import streamlit as st
import numpy as np
import os

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

    # Auto-save changes to session database
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
    
    # Automatically formats selected course name to match subfolder structure (e.g., "Mercer Oaks East" -> "mercer_oaks_east")
    course_folder = selected_course.lower().replace(" ", "_").replace("(", "").replace(")", "")
    
    # Routes image path directly to the course subfolder inside assets/
    heat_path = f"assets/{course_folder}/{selected_hole}_Heat.png"
    contour_path = f"assets/{course_folder}/{selected_hole}_Contour.png"
    
    with m_col1:
        if os.path.exists(heat_path):
            st.image(heat_path, caption=f"Hole {selected_hole} Heat Map", use_container_width=True)
        else:
            st.info(f"Missing: `assets/{course_folder}/{selected_hole}_Heat.png`")
            
    with m_col2:
        if os.path.exists(contour_path):
            st.image(contour_path, caption=f"Hole {selected_hole} Contour", use_container_width=True)
        else:
            st.info(f"Missing: `assets/{course_folder}/{selected_hole}_Contour.png`")

# --- CALCULATION ENGINE ---
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

    # Gradient multiplier based on arrow count
    gradient_multiplier = 2.0 if "Double" in slope_steepness else 1.0

    # Slope calculation
    slope_drop = ((0.03 * x_hole - 0.02 * y_hole) - (0.03 * x_ball - 0.02 * y_ball)) * gradient_multiplier
    
    # Aim Calculation
    aim_offset_ft = slope_drop * calibrated_stimp * 0.35 * (straight_dist_ft / 10.0)
    aim_side = "Left" if slope_drop < 0 else "Right"
    aim_ft_val = abs(aim_offset_ft)
    aim_paces_val = aim_ft_val / ft_per_pace
    
    # Speed Recommendation
    elevation_speed_adj = slope_drop * 0.5
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj + 0.3
    recommended_speed_paces = max(1.0, round(recommended_speed_paces, 1))

    # --- CLEAN, PUNCHY OUTPUTS ---
    st.success("Target Solution Readout:")
    st.markdown(f"### 🎯 **Putt Distance:** {format_feet_inches(straight_dist_ft)} ({straight_paces:.1f} paces)")
    st.markdown(f"### ➡️ **Aim Point:** {format_feet_inches(aim_ft_val)} ({aim_paces_val:.1f} paces) {aim_side}")
    st.markdown(f"### ⚡ **Stroke Speed:** Putt with **{recommended_speed_paces}-pace** power stroke")