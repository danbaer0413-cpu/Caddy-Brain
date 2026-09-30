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
st.title("⛳ CaddyBrain: Green Reading & Aim Assistant")
st.write("Select your course, hole, and enter your pacing measurements to calculate target aim and stroke speed.")

# Sidebar for Course Selection & Calibration
st.sidebar.header("1. Course & Setup")
selected_course = st.sidebar.selectbox("Select Course", list(COURSES.keys()))
selected_hole = st.sidebar.selectbox("Select Hole Number", list(COURSES[selected_course].keys()))

default_depth = COURSES[selected_course][selected_hole]["max_depth_yds"]
default_width = COURSES[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Practice Green Calibration")
base_stimp = st.sidebar.slider("Base Green Speed (Stimp)", 6.0, 12.0, 8.0)

# 3-pace test (equivalent to 9 feet)
target_test_paces = 3.0
actual_test_paces = st.sidebar.number_input("3-Pace Test: Actual Roll Distance (paces)", min_value=1.0, max_value=6.0, value=3.2, step=0.5, help="Pace out how far a standard 3-pace practice stroke rolled.")

# Corrected ratio: If actual > target (rolled past), green is faster -> higher Stimp
stimp_ratio = actual_test_paces / target_test_paces
calibrated_stimp = base_stimp * stimp_ratio
st.sidebar.info(f"**Calibrated Stimp:** {calibrated_stimp:.1f}")

# Main Inputs (Expanded to 0-100 yards)
st.header(f"Hole #{selected_hole} Specifications")
col_d1, col_d2 = st.columns(2)
with col_d1:
    max_depth_yds = st.number_input("Green Depth (Yards)", min_value=0.0, max_value=100.0, value=float(default_depth), step=1.0)
with col_d2:
    green_width_yds = st.number_input("Green Width (Yards)", min_value=0.0, max_value=100.0, value=float(default_width), step=1.0)

st.header("On-Course Measurements (in Paces)")
col3, col4 = st.columns(2)
with col3:
    hole_from_front = st.number_input("Hole from Front of Green (paces)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
    hole_from_side = st.number_input("Hole from Side Edge (paces)", min_value=0.0, max_value=100.0, value=4.0, step=0.5)
    side_ref = st.selectbox("Side Reference", ["Left", "Right"])

with col4:
    ball_offset_paces = st.number_input("Distance from Hole to Ball (paces)", min_value=0.5, max_value=100.0, value=8.0, step=0.5)
    ball_direction = st.selectbox("Ball Position Relative to Hole", ["Right", "Left", "Front", "Back"])

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

    # Slope gradient estimation
    slope_drop = (0.03 * x_hole - 0.02 * y_hole) - (0.03 * x_ball - 0.02 * y_ball)
    
    # Aim Calculation
    aim_offset_ft = slope_drop * calibrated_stimp * 0.35 * (straight_dist_ft / 10.0)
    aim_side = "Left" if slope_drop < 0 else "Right"
    
    # Speed Recommendation
    elevation_speed_adj = slope_drop * 0.4
    recommended_speed_paces = straight_paces * (8.0 / calibrated_stimp) + elevation_speed_adj
    recommended_speed_paces = max(0.5, round(recommended_speed_paces, 1))

    # --- DISPLAY RESULTS ---
    st.success("Putt Calculation Complete!")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.metric("Straight Distance", f"{straight_paces:.1f} paces", f"{format_feet_inches(straight_dist_ft)}")
    with res_col2:
        st.metric("Target Aim Point", f"Aim {abs(aim_offset_ft/ft_per_pace):.1f} paces {aim_side}", f"{format_feet_inches(abs(aim_offset_ft))}")
    with res_col3:
        st.metric("Stroke Speed", f"{recommended_speed_paces} paces power", f"Putt like a {recommended_speed_paces}-pace putt")