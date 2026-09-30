import streamlit as st
import numpy as np

# --- COURSE DATABASE ---
COURSES = {
    "Mercer Oaks (Public)": {
        1: {"max_depth_yds": 28.0, "width_yds": 14.0},
        2: {"max_depth_yds": 21.0, "width_yds": 15.0},
        3: {"max_depth_yds": 28.0, "width_yds": 13.0},
        4: {"max_depth_yds": 27.0, "width_yds": 14.0},
        5: {"max_depth_yds": 22.0, "width_yds": 12.0},
        6: {"max_depth_yds": 32.0, "width_yds": 15.0},
        7: {"max_depth_yds": 24.0, "width_yds": 14.0},
        8: {"max_depth_yds": 37.0, "width_yds": 16.0},
        9: {"max_depth_yds": 30.0, "width_yds": 14.0},
        10: {"max_depth_yds": 24.0, "width_yds": 13.0},
        11: {"max_depth_yds": 32.0, "width_yds": 15.0},
        12: {"max_depth_yds": 23.0, "width_yds": 13.0},
        13: {"max_depth_yds": 29.0, "width_yds": 14.0},
        14: {"max_depth_yds": 21.0, "width_yds": 12.0},
        15: {"max_depth_yds": 29.0, "width_yds": 14.0},
        16: {"max_depth_yds": 23.0, "width_yds": 13.0},
        17: {"max_depth_yds": 27.0, "width_yds": 14.0},
        18: {"max_depth_yds": 26.0, "width_yds": 15.0},
    },
    "Sample Country Club": {
        1: {"max_depth_yds": 30.0, "width_yds": 16.0},
        2: {"max_depth_yds": 25.0, "width_yds": 14.0},
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
st.title("⛳ CaddyBrain")

# Sidebar for Course Selection & Calibration
st.sidebar.header("1. Course & Setup")
selected_course = st.sidebar.selectbox("Course", list(COURSES.keys()))
selected_hole = st.sidebar.selectbox("Hole", list(COURSES[selected_course].keys()))

default_depth = COURSES[selected_course][selected_hole]["max_depth_yds"]
default_width = COURSES[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Stimp Calibration")
base_stimp = st.sidebar.slider("Base Stimp", 6.0, 12.0, 8.0)
actual_test_paces = st.sidebar.number_input("3-Pace Test Roll (paces)", min_value=0.5, max_value=10.0, value=3.2, step=0.5)

stimp_ratio = actual_test_paces / 3.0
calibrated_stimp = base_stimp * stimp_ratio
st.sidebar.info(f"Calibrated Stimp: **{calibrated_stimp:.1f}**")

# Main Inputs
st.header(f"Hole #{selected_hole} Setup")
col_d1, col_d2 = st.columns(2)
with col_d1:
    max_depth_yds = st.number_input("Depth (Yds)", min_value=0.0, max_value=100.0, value=float(default_depth), step=1.0)
with col_d2:
    green_width_yds = st.number_input("Width (Yds)", min_value=0.0, max_value=100.0, value=float(default_width), step=1.0)

col3, col4 = st.columns(2)
with col3:
    hole_from_front = st.number_input("Hole from Front (paces)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
    hole_from_side = st.number_input("Hole from Side (paces)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
    side_ref = st.selectbox("Side Ref", ["Left", "Right"])

with col4:
    ball_offset_paces = st.number_input("Ball to Hole (paces)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
    ball_direction = st.selectbox("Ball Position", ["Right", "Left", "Front", "Back"])
    slope_steepness = st.selectbox("Green Contour Gradient", ["Standard Slope (Single Arrow)", "Steep Slope (Double Arrows ⚡)"])

# --- CALCULATION ENGINE ---
if st.button("Calculate", type="primary"):
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

    # Slope calculation (Base to point vector interpretation)
    slope_drop = ((0.03 * x_hole - 0.02 * y_hole) - (0.03 * x_ball - 0.02 * y_ball)) * gradient_multiplier
    
    # Aim Calculation
    aim_offset_ft = slope_drop * calibrated_stimp * 0.35 * (straight_dist_ft / 10.0)
    aim_side = "Left" if slope_drop < 0 else "Right"
    aim_ft_val = abs(aim_offset_ft)
    aim_paces_val = aim_ft_val / ft_per_pace
    
    # Speed Recommendation
    elevation_speed_adj = slope_drop * 0.4
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj
    recommended_speed_paces = max(0.5, round(recommended_speed_paces, 1))

    # --- SHORT, PUNCHY OUTPUTS ---
    st.success("Result:")
    st.markdown(f"### 🎯 **Putt Distance:** {format_feet_inches(straight_dist_ft)} ({straight_paces:.1f} paces)")
    st.markdown(f"### ➡️ **Aim:** {format_feet_inches(aim_ft_val)} ({aim_paces_val:.1f} paces) {aim_side}")
    st.markdown(f"### ⚡ **Stroke Speed:** Putt with **{recommended_speed_paces}-pace** power stroke")