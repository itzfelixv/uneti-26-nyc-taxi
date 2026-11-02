import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import haversine, make_features

MODEL_PATH = "model.joblib"

# Một số địa điểm phổ biến ở New York (vĩ độ, kinh độ)
PLACES = {
    "Quảng trường Thời đại (Times Square)": (40.76, -73.99),
    "Công viên Trung tâm (Central Park)": (40.78, -73.97),
    "Phố Wall (Wall Street)": (40.71, -74.01),
    "Ga Grand Central": (40.75, -73.98),
    "Sân bay JFK": (40.64, -73.78),
    "Sân bay LaGuardia": (40.77, -73.87),
    "Brooklyn (Downtown)": (40.69, -73.99),
    "Tùy chỉnh": None,
}
DAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def predict_seconds(bundle, raw_df):
    X = make_features(raw_df)[bundle["features"]]
    pred_log = np.mean([m.predict(X) for m in bundle["models"]], axis=0)
    lo, hi = bundle["clip_seconds"]
    return np.clip(np.expm1(pred_log), lo, hi)


def location_input(label, default):
    name = st.selectbox(label, list(PLACES), index=list(PLACES).index(default))
    lat, lon = PLACES[name] or (40.75, -73.98)
    if PLACES[name] is None:
        c1, c2 = st.columns(2)
        lat = c1.number_input(f"Vĩ độ ({label.lower()})", 40.50, 41.00, lat, 0.01, format="%.2f")
        lon = c2.number_input(f"Kinh độ ({label.lower()})", -74.30, -73.60, lon, 0.01, format="%.2f")
    return round(lat, 2), round(lon, 2)


st.set_page_config(page_title="Dự đoán thời gian taxi NYC", layout="centered")
st.title("Dự đoán thời gian chuyến taxi tại New York")
st.write("Nhập thông tin chuyến đi, mô hình LightGBM đã huấn luyện sẽ dự đoán thời gian di chuyển.")

bundle = load_bundle()

with st.form("trip"):
    st.subheader("Địa điểm")
    p_lat, p_lon = location_input("Điểm đón", "Quảng trường Thời đại (Times Square)")
    d_lat, d_lon = location_input("Điểm đến", "Phố Wall (Wall Street)")

    st.subheader("Thời gian đón")
    c1, c2 = st.columns(2)
    month = c1.selectbox("Tháng", list(range(1, 7)), format_func=lambda m: f"Tháng {m}")
    day = c2.selectbox("Thứ trong tuần", list(range(7)), format_func=lambda d: DAYS[d])
    c3, c4 = st.columns(2)
    hour = c3.slider("Giờ đón", 0, 23, 8)
    minute = c4.selectbox("Phút", [0, 15, 30, 45])

    st.subheader("Thông tin khác")
    c5, c6 = st.columns(2)
    passengers = c5.number_input("Số hành khách", 1, 6, 1)
    vendor = c6.selectbox("Nhà vận tải (vendor_id)", [1, 2])

    submitted = st.form_submit_button("Dự đoán", type="primary")

if submitted:
    dist_km = float(haversine(p_lat, p_lon, d_lat, d_lon))
    row = pd.DataFrame([{
        "vendor_id": vendor,
        "pickup_month": month,
        "pickup_dayofweek": day,
        "pickup_hour": hour,
        "pickup_minute_bucket": minute,
        "passenger_count": passengers,
        "pickup_lon_approx": p_lon,
        "pickup_lat_approx": p_lat,
        "dropoff_lon_approx": d_lon,
        "dropoff_lat_approx": d_lat,
        "straight_line_km_approx": round(dist_km * 2) / 2,  # làm tròn 0,5 km như dữ liệu gốc
        "store_and_fwd_flag": "N",
    }])
    seconds = float(predict_seconds(bundle, row)[0])
    mins, secs = divmod(int(round(seconds)), 60)

    st.subheader("Kết quả")
    c1, c2 = st.columns(2)
    c1.metric("Thời gian dự đoán", f"{mins} phút {secs} giây")
    c2.metric("Khoảng cách đường chim bay", f"{dist_km:.2f} km")
    st.write(f"Tương đương **{seconds:,.0f} giây**.")

    st.map(pd.DataFrame({"lat": [p_lat, d_lat], "lon": [p_lon, d_lon]}), size=150)
    st.caption(f"RMSLE trên tập kiểm định: {bundle['val_rmsle']:.4f}")
