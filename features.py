import numpy as np
import pandas as pd

RAW_FEATURES = [
    "vendor_id", "pickup_month", "pickup_dayofweek", "pickup_hour", "pickup_minute_bucket",
    "passenger_count", "pickup_lon_approx", "pickup_lat_approx",
    "dropoff_lon_approx", "dropoff_lat_approx", "straight_line_km_approx",
]

JFK     = (40.6413, -73.7781)
LGA     = (40.7769, -73.8740)
EWR     = (40.6895, -74.1745)
MIDTOWN = (40.7549, -73.9840)


def haversine(lat1, lon1, lat2, lon2):
    """https://en.wikipedia.org/wiki/Haversine_formula"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """https://en.wikipedia.org/wiki/Bearing_(navigation)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing: {missing}")
    out = df[RAW_FEATURES].copy().astype(float)

    plat, plon = out["pickup_lat_approx"], out["pickup_lon_approx"]
    dlat, dlon = out["dropoff_lat_approx"], out["dropoff_lon_approx"]

    # khoảng cách & hướng
    out["haversine_km"] = haversine(plat, plon, dlat, dlon)
    out["manhattan_km"] = haversine(plat, plon, plat, dlon) + haversine(plat, dlon, dlat, dlon)
    out["delta_lat"]    = dlat - plat
    out["delta_lon"]    = dlon - plon
    out["bearing"]      = bearing(plat, plon, dlat, dlon)
    out["center_lat"]   = (plat + dlat) / 2
    out["center_lon"]   = (plon + dlon) / 2

    # sân bay & trung tâm
    for name, (alat, alon) in {"jfk": JFK, "lga": LGA, "ewr": EWR, "midtown": MIDTOWN}.items():
        out[f"pickup_dist_{name}"]  = haversine(plat, plon, alat, alon)
        out[f"dropoff_dist_{name}"] = haversine(dlat, dlon, alat, alon)
    out["is_airport_trip"] = (
        (out[["pickup_dist_jfk", "pickup_dist_lga", "pickup_dist_ewr",
              "dropoff_dist_jfk", "dropoff_dist_lga", "dropoff_dist_ewr"]] < 2).any(axis=1)
    ).astype(int)

    # thời gian
    out["time_of_day"]  = out["pickup_hour"] + out["pickup_minute_bucket"] / 60
    out["hour_of_week"] = out["pickup_dayofweek"] * 24 + out["pickup_hour"]
    out["is_weekend"]   = (out["pickup_dayofweek"] >= 5).astype(int)
    out["is_rush_hour"] = ((out["is_weekend"] == 0) &
                           (out["pickup_hour"].between(7, 9) | out["pickup_hour"].between(16, 19))).astype(int)
    out["is_night"]     = ((out["pickup_hour"] <= 5) | (out["pickup_hour"] >= 22)).astype(int)
    return out
