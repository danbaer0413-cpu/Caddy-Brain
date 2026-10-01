import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="CaddyBrain Green Reader", page_icon="⛳", layout="wide")

if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {
            i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)
        }
    }

if "ball_coords" not in st.session_state or not isinstance(st.session_state.ball_coords, dict):
    st.session_state.ball_coords = {"x_ft": 29.5, "y_ft": 52.1}

if "hole_coords" not in st.session_state or not isinstance(st.session_state.hole_coords, dict):
    st.session_state.hole_coords = {"x_ft": 15.1, "y_ft": 23.2}

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

def calculate_putt_solution(x_ball, y_ball, x_hole, y_hole, max_depth_yds, green_width_ft, calibrated_stimp, break_mode, raw_img, display_width=600):
    orig_w, orig_h = raw_img.size
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

    bx_px, by_px = ft_to_pixels(x_ball, y_ball)
    hx_px, hy_px = ft_to_pixels(x_hole, y_hole)
    
    # --- ADVANCED IMAGE SAMPLING FOR GRADIENT & BREAK ---
    img_np = np.array(raw_img)
    num_samples = 15
    sample_x = np.linspace(bx_px, hx_px, num_samples).astype(int)
    sample_y = np.linspace(by_px, hy_px, num_samples).astype(int)
    
    red_score = 0
    blue_score = 0
    gradient_intensity_sum = 0
    
    for sx, sy in zip(sample_x, sample_y):
        if 0 <= sy < img_np.shape[0] and 0 <= sx < img_np.shape[1]:
            r, g, b = img_np[sy, sx][:3]
            red_score += int(r)
            blue_score += int(b)
            neutral = (int(r) + int(g) + int(b)) / 3.0
            saturation = abs(int(r) - neutral) + abs(int(b) - neutral)
            gradient_intensity_sum += saturation

    avg_gradient_factor = max(0.5, min(2.5, (gradient_intensity_sum / num_samples) / 40.0))

    # Determine Break Side
    natural_side = "Right" if red_score > blue_score else "Left"
    geo_side = "Right" if x_ball > x_hole else "Left"
        
    detected_side = natural_side if abs(red_score - blue_score) > 1000 else geo_side
    
    if break_mode == "Inverted (Flip L/R)":
        aim_side = "Left" if detected_side == "Right" else "Right"
    else:
        aim_side = detected_side

    # --- ELEVATION & "PLAY AS" DISTANCE CALCULATION ---
    # Corrected sign: Downhill putts reduce play-as distance, uphill putts increase it
    elevation_delta_ft = (y_hole - y_ball) * 0.15  
    play_as_dist_ft = straight_dist_ft + elevation_delta_ft 
    play_as_dist_ft = max(1.0, play_as_dist_ft)

    stroke_feel_ft = round(play_as_dist_ft * (calibrated_stimp / 8.0), 1)

    # Aim offset calculation driven by gradient intensity and distance
    base_drop = abs(y_ball - y_hole) * 0.04
    aim_offset_ft = base_drop * avg_gradient_factor * (calibrated_stimp / 8.0) * (straight_dist_ft / 12.0)
    aim_ft_val = round(aim_offset_ft, 2)

    # --- SHIFT THE AIM DOT PERPENDICULAR TO THE PUTT LINE ---
    dx = hx_px - bx_px
    dy = hy_px - by_px
    line_len = np.sqrt(dx**2 + dy**2)
    
    if line_len > 0:
        nx, ny = -dy / line_len, dx / line_len
        shift_sign = -1.0 if aim_side == "Left" else 1.0
        pixels_per_ft = (box_x_max - box_x_min) / green_width_ft
        total_pixel_shift = aim_ft_val * pixels_per_ft * shift_sign
        
        tx_px = hx_px + int(nx * total_pixel_shift)
        ty_px = hy_px + int(ny * total_pixel_shift)
    else:
        tx_px, ty_px = hx_px, hy_px

    # Draw overlays directly on the Heat Map
    draw_img = raw_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # 1. Direct line (Gray)
    draw.line([(bx_px, by_px), (hx_px, hy_px)], fill="gray", width=3)
    
    # 2. Aim line (Blue)
    draw.line([(bx_px, by_px), (tx_px, ty_px)], fill="#2b5c8f", width=4)

    # 3. Markers: Ball (Blue), Hole (Red), Target/Aim (Cyan)
    dot_r = 10
    draw.ellipse([bx_px - dot_r, by_px - dot_r, bx_px + dot_r, by_px + dot_r], fill="blue", outline="white", width=2)
    draw.ellipse([hx_px - dot_r, hy_px - dot_r, hx_px + dot_r, hy_px + dot_r], fill="red", outline="white", width=2)
    draw.ellipse([tx_px - 7, ty_px - 7, tx_px + 7, ty_px + 7], fill="cyan", outline="black", width=2)
    
    # 4. Aim Text Label next to Aim Dot
    aim_label_text = f"Aim: {format_feet_inches(aim_ft_val)} {aim_side}"
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text_x = tx_px + 12
    text_y = ty_px - 8
    draw.text((text_x, text_y), aim_label_text, fill="#1b365d", font=font)
    
    return draw_img, straight_dist_ft, play_as_dist_ft, aim_ft_val, aim_side, stroke_feel_ft, pixels_to_ft

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("1. Course & Hole")
selected_course = st.sidebar.selectbox("Course", list(st.session_state.courses_db.keys()))
selected_hole = st.sidebar.selectbox("Hole #", list(st.session_state.courses_db[selected_course].keys()))

saved_depth = st.session_state.courses_db[selected_course][selected_hole]["max_depth_yds"]
saved_width = st.session_state.courses_db[selected_course][selected_hole]["width_yds"]

st.sidebar.header("2. Stimp & Break Settings")
base_stimp = st.sidebar.slider("Stimp", 6.0, 12.0, 8.0)
break_mode = st.sidebar.selectbox("Break Direction Override", ["Auto (Sampled)", "Inverted (Flip L/R)"])

st.sidebar.header("3. Marker Mode")
placement_mode = st.sidebar.radio("Click sets:", ["🔴 Hole Position", "🔵 Ball Position"])

if st.sidebar.button("Reset Markers"):
    st.session_state.ball_coords = {"x_ft": saved_width * 1.5, "y_ft": 4.0}
    st.session_state.hole_coords = {"x_ft": saved_width * 1.5, "y_ft": 20.0}
    st.rerun()

# Asset Path Resolver
course_folder = selected_course.lower().replace(" ", "_")
heat_path = None
for filename in [f"{selected_hole}_Heat.png", f"{selected_hole}_heat.png"]:
    path = f"assets/{course_folder}/{filename}"
    if os.path.exists(path):
        heat_path = path
        break

DISPLAY_WIDTH = 550

# --- 4. MAIN INTERFACE ---
st.title(f"⛳ Hole #{selected_hole} ({selected_course})")

if heat_path:
    raw_img = Image.open(heat_path).convert("RGB")
    
    annotated_img, straight_dist, play_as_dist, aim_val, aim_dir, stroke_dist, pixels_to_ft_func = calculate_putt_solution(
        x_ball=st.session_state.ball_coords["x_ft"],
        y_ball=st.session_state.ball_coords["y_ft"],
        x_hole=st.session_state.hole_coords["x_ft"],
        y_hole=st.session_state.hole_coords["y_ft"],
        max_depth_yds=saved_depth,
        green_width_ft=saved_width * 3.0,
        calibrated_stimp=base_stimp,
        break_mode=break_mode,
        raw_img=raw_img,
        display_width=DISPLAY_WIDTH
    )

    # --- TOP METRICS READOUT ---
    c1, c2, c3 = st.columns(3)
    with c1:
        aim_str = f"{format_feet_inches(aim_val)} {aim_dir}" if aim_val > 0.05 else "Straight"
        st.metric(label="🎯 Aim Offset", value=aim_str)
    with c2:
        st.metric(label="⚡ Stroke Feel Distance", value=f"{stroke_dist} ft", delta=f"Play As: {round(play_as_dist, 1)} ft (Actual: {round(straight_dist, 1)} ft)")
    with c3:
        break_desc = "Right-to-Left Break" if aim_dir == "Right" else "Left-to-Right Break"
        st.metric(label="📈 Expected Break", value=break_desc, delta=f"Gradient-weighted aim")

    st.markdown("---")

    # --- SINGLE MAP & CONTROLS LAYOUT ---
    col_map, col_info = st.columns([1.5, 1])
    
    with col_map:
        st.subheader("🔥 Heat Map Readout")
        clicked = streamlit_image_coordinates(annotated_img, key="single_map_click", width=DISPLAY_WIDTH)
        if clicked is not None:
            cx, cy = pixels_to_ft_func(clicked["x"], clicked["y"])
            if "Hole" in placement_mode:
                st.session_state.hole_coords = {"x_ft": cx, "y_ft": cy}
            else:
                st.session_state.ball_coords = {"x_ft": cx, "y_ft": cy}
            st.rerun()

    with col_info:
        st.subheader("Coordinates")
        st.info(f"**Ball:** X: {st.session_state.ball_coords['x_ft']:.1f}ft, Y: {st.session_state.ball_coords['y_ft']:.1f}ft")
        st.info(f"**Hole:** X: {st.session_state.hole_coords['x_ft']:.1f}ft, Y: {st.session_state.hole_coords['y_ft']:.1f}ft")
        st.markdown("**Instructions:** Select whether your next click sets the Ball or Hole in the sidebar, then click directly on the heat map to update your position and instantly see your updated read.")
else:
    st.warning(f"Heat map image not found for Hole #{selected_hole} in `assets/{course_folder}/`. Please check your file naming.")