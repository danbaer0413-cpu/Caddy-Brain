"""High-accuracy GPS reader for Green Reader (beta).

A small browser component that asks the phone for its best GPS fix (enableHighAccuracy, continuous watch),
listens for several seconds, and returns an accuracy-weighted average of the best readings.
"""
import os

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gps_component")
_component = None


def available():
    """True when the component's web page is next to this file (gps_component/index.html)."""
    return os.path.isfile(os.path.join(_DIR, "index.html"))


def precise_location(seconds=10, key="precise_gps"):
    """Show the reader. Returns None until the user taps it, then a dict with latitude, longitude, accuracy (meters),
    readings (how many were averaged) and id (changes with every new reading)."""
    global _component
    import streamlit.components.v1 as components
    if _component is None:
        _component = components.declare_component("caddy_precise_gps", path=_DIR)
    return _component(seconds=seconds, key=key, default=None)
