"""GPS helpers for Green Reader (beta).

Two reference spots, each known both on the map (green feet) and by GPS (lat/lon), fix where the green
sits in the real world: the first gives the origin, the line between them gives the rotation. After that
any GPS fix can be turned into green coordinates. Pure math, no Streamlit.
"""
import numpy as np

R_EARTH_M = 6371000.0
FT_PER_M = 3.28084
MIN_BASELINE_FT = 15.0     # references closer than this give no usable rotation


def enu_ft(lat, lon, lat0, lon0):
    """Small-area flat-earth conversion: (east, north) offset in feet of (lat, lon) from (lat0, lon0)."""
    north = np.radians(lat - lat0) * R_EARTH_M
    east = np.radians(lon - lon0) * R_EARTH_M * np.cos(np.radians(lat0))
    return np.array([east, north]) * FT_PER_M


def calibrate(ref_a, ref_b):
    """ref_x = dict(green=(x_ft, y_ft), lat=..., lon=..., acc=meters or None). Returns a calibration dict.

    Distances use real feet (scale is not fitted); the measured/mapped distance ratio is reported as a
    sanity check on the two fixes.
    """
    ga, gb = np.array(ref_a["green"], float), np.array(ref_b["green"], float)
    qb = enu_ft(ref_b["lat"], ref_b["lon"], ref_a["lat"], ref_a["lon"])
    vg = gb - ga
    if np.hypot(*vg) < MIN_BASELINE_FT or np.hypot(*qb) < MIN_BASELINE_FT:
        raise ValueError("The two reference spots are too close together. Pick spots as far apart as you can, "
                         "like the front and back of the green.")
    theta = np.arctan2(vg[1], vg[0]) - np.arctan2(qb[1], qb[0])
    accs = [a for a in (ref_a.get("acc"), ref_b.get("acc")) if a]
    return {"lat0": ref_a["lat"], "lon0": ref_a["lon"], "origin": tuple(ga), "theta": float(theta),
            "baseline_ft": float(np.hypot(*vg)), "scale_ratio": float(np.hypot(*qb) / np.hypot(*vg)),
            "acc_m": max(accs) if accs else None}


def to_green(cal, lat, lon):
    """GPS fix -> (x_ft, y_ft) on the green's grid."""
    q = enu_ft(lat, lon, cal["lat0"], cal["lon0"])
    c, s = np.cos(cal["theta"]), np.sin(cal["theta"])
    p = np.array(cal["origin"]) + np.array([[c, -s], [s, c]]) @ q
    return float(p[0]), float(p[1])


def from_green(cal, x_ft, y_ft):
    """Inverse of to_green (used by tests and for drawing a simulated position)."""
    c, s = np.cos(-cal["theta"]), np.sin(-cal["theta"])
    q = np.array([[c, -s], [s, c]]) @ (np.array([x_ft, y_ft]) - np.array(cal["origin"]))
    m = q / FT_PER_M
    lat = cal["lat0"] + np.degrees(m[1] / R_EARTH_M)
    lon = cal["lon0"] + np.degrees(m[0] / (R_EARTH_M * np.cos(np.radians(cal["lat0"]))))
    return float(lat), float(lon)


def quality_notes(cal):
    """Plain-English warnings about a calibration (empty list = looks fine)."""
    notes = []
    if cal["baseline_ft"] < 40:
        notes.append("The reference spots are only %.0f ft apart, so the rotation is shaky. Farther apart is better." % cal["baseline_ft"])
    if not 0.75 <= cal["scale_ratio"] <= 1.33:
        notes.append("GPS says the two spots are %.0f%% of the distance the map says. One of the taps or fixes is probably off; "
                     "re-lock the references." % (cal["scale_ratio"] * 100))
    if cal["acc_m"] and cal["acc_m"] > 8:
        notes.append("GPS accuracy was about %.0f m when you locked the references. Wait for a better signal if you can." % cal["acc_m"])
    return notes
