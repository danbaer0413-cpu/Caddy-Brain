import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates

# --- ROBUST SESSION STATE INITIALIZATION ---
if "courses_db" not in st.session_state:
    st.session_state.courses_db = {
        "Mercer Oaks East": {
            i: {"max_depth_yds": 28.0, "width_yds": 14.0} for i in range(1, 19)
        }
    }

if "ball_coords" not in st.session_state or not isinstance(st.session_state.ball_coords, dict) or "x_ft" not in st.session_state.ball_coords:
    st.session_state.ball_coords = {"x_ft": 12.0, "y_ft": 4.0}

if "hole_coords" not in st.session_state or not isinstance(st.session_state.hole_coords, dict) or "x_ft" not in st.session_state.hole_coords:
    st.session_state.hole_coords = {"x_ft": 7.0, "y_ft": 36.0}