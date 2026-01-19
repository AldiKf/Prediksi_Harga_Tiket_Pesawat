import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Flight Pricing Intelligence",
    layout="centered"
)

# CSS (LIGHT MODE)
st.markdown("""
<style>
html, body, [data-testid="stApp"] {
    background-color: #F5F7FA !important;
    color: #111827 !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #1A73E8;
    font-weight: 800;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-top: 28px;
    margin-bottom: 6px;
}

.caption {
    color: #6B7280;
    font-size: 14px;
}

.card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    border: 1px solid #E5E7EB;
}

.price {
    font-size: 36px;
    font-weight: 800;
    color: #16A34A;
}

.stButton>button {
    background: linear-gradient(135deg, #1A73E8, #2563EB);
    color: white;
    border-radius: 14px;
    padding: 14px 22px;
    font-size: 16px;
    font-weight: 700;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# LIBRARIES
import os
import pandas as pd
import numpy as np
import joblib
import math
import pydeck as pdk
from datetime import date, time, datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin

# FEATURE ENG
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        dt = pd.to_datetime(X["Tanggal_Perjalanan"], dayfirst=True, errors="coerce")
        X["travel_day"] = dt.dt.day
        X["travel_month"] = dt.dt.month
        X["travel_weekday"] = dt.dt.weekday

        dep = X["Waktu_Berangkat"].astype(str).str.extract(r'(\d{1,2}):(\d{2})')
        X["dep_hour"] = pd.to_numeric(dep[0], errors="coerce")
        X["dep_min"] = pd.to_numeric(dep[1], errors="coerce")

        arr = X["Waktu_Tiba"].astype(str).str.extract(r'(\d{1,2}):(\d{2})')
        X["arr_hour"] = pd.to_numeric(arr[0], errors="coerce")
        X["arr_min"] = pd.to_numeric(arr[1], errors="coerce")

        dur = X["Durasi_Penerbangan"].astype(str).str.lower()
        h = dur.str.extract(r'(\d+)\s*h')[0]
        m = dur.str.extract(r'(\d+)\s*m')[0]
        X["dur_min"] = (
            pd.to_numeric(h, errors="coerce").fillna(0) * 60
            + pd.to_numeric(m, errors="coerce").fillna(0)
        )

        jt = X["Jumlah_Transit"].astype(str).str.lower()
        X["transit_count"] = np.where(
            jt.str.contains("non", na=False),
            0,
            pd.to_numeric(jt.str.extract(r'(\d+)')[0], errors="coerce")
        )

        dep_total = X["dep_hour"] * 60 + X["dep_min"]
        X["day_change"] = np.floor_divide(dep_total + X["dur_min"], 1440)

        keep = [
            "Maskapai", "Informasi_Tambahan", "Jarak_Km",
            "travel_day", "travel_month", "travel_weekday",
            "dep_hour", "dep_min", "arr_hour", "arr_min",
            "dur_min", "transit_count", "day_change"
        ]
        return X[keep]

# LOAD MODEL & DATA
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "modelLinearRegression.joblib"))

@st.cache_data
def load_airports():
    df = pd.read_csv(os.path.join(BASE_DIR, "airports_clean (1).csv"))
    df["label"] = df["nama_bandara"] + " (" + df["airport_code"] + ")"
    return df

model = load_model()
bandara = load_airports()

# HAVERSINE 
def deg2rad(deg):
    return deg * (math.pi / 180)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = deg2rad(lat2 - lat1)
    dLon = deg2rad(lon2 - lon1)
    a = (
        math.sin(dLat / 2) ** 2
        + math.cos(deg2rad(lat1))
        * math.cos(deg2rad(lat2))
        * math.sin(dLon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# HEADER 
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;">
  <div>
    <h1>✈️ Prediksi Harga Tiket Pesawat <h1>
    <div class="caption">Untuk kebutuhan maskapai dan simulasi tiketing berdasarkan harga</div>
  </div>
  <div class="caption">
    Model: Linear Regression v1.0<br>
    Last Update: Jan 2026
  </div>
</div>
""", unsafe_allow_html=True)

# FORM
st.markdown('<div class="section-title">✈️ Detail Penerbangan</div>', unsafe_allow_html=True)
st.caption("Variabel operasional yang memengaruhi harga tiket")

col1, col2 = st.columns(2)

with col1:
    maskapai = st.selectbox("Maskapai", [
        "Garuda Indonesia", "Lion Air", "Batik Air", "Transit Maskapai",
        "Citilink", "Sriwijaya Air", "AirAsia Indonesia", "Super Air Jet",
        "Transit Premium", "Garuda Indonesia Business",
        "Sriwijaya Air Premium", "Wings Air"
    ])

    tgl = st.date_input("Tanggal Perjalanan", value=date(2019, 3, 24))
    waktu_dep = st.time_input("Waktu Berangkat", value=time(22, 20))
    waktu_arr = st.time_input("Waktu Tiba", value=time(1, 10))

    dep_dt = datetime.combine(tgl, waktu_dep)
    arr_dt = datetime.combine(tgl, waktu_arr)
    if arr_dt <= dep_dt:
        arr_dt += timedelta(days=1)

    durasi_menit = int((arr_dt - dep_dt).total_seconds() / 60)
    durasi = f"{durasi_menit//60}h {durasi_menit%60}m"
    st.text_input("Durasi Penerbangan", value=durasi, disabled=True)

with col2:
    transit = st.selectbox("Jumlah Transit", ["non-stop", "1 stop", "2 stops"])
    info = st.selectbox("Informasi Tambahan", [
        "No info", "In-flight meal not included",
        "No check-in baggage included", "1 Long layover",
        "Change airports", "Business class",
        "1 Short layover", "Red-eye flight"
    ])

st.markdown('<div class="section-title">Rute Penerbangan</div>', unsafe_allow_html=True)
st.caption("Validasi jarak antar bandara keberangkatan dan tujuan")

col3, col4 = st.columns(2)
asal = col3.selectbox("Bandara Asal", bandara["label"].tolist(), index=None)
tujuan = col4.selectbox("Bandara Tujuan", bandara["label"].tolist(), index=None)

st.markdown("<br>", unsafe_allow_html=True)
predict = st.button("📈 Prediksi Harga", use_container_width=True)

# OUTPUT
if predict and asal != tujuan:
    b1 = bandara[bandara["label"] == asal].iloc[0]
    b2 = bandara[bandara["label"] == tujuan].iloc[0]

    jarak_km = haversine(b1["latitude"], b1["longitude"], b2["latitude"], b2["longitude"])

    input_df = pd.DataFrame([{
        "Maskapai": maskapai,
        "Tanggal_Perjalanan": tgl,
        "Waktu_Berangkat": waktu_dep,
        "Waktu_Tiba": waktu_arr,
        "Durasi_Penerbangan": durasi,
        "Jumlah_Transit": transit,
        "Informasi_Tambahan": info,
        "Jarak_Km": jarak_km
    }])

    pred = model.predict(input_df)[0]
    low, high = pred * 0.9, pred * 1.1

    st.markdown(f"""
    <div class="card">
        <div class="price">Rp {pred:,.0f}</div>
        <div class="caption">Estimasi Harga: Rp {low:,.0f} – Rp {high:,.0f}</div>
        <hr>
        <ul class="caption">
            <li>Jarak: {jarak_km:.0f} km</li>
            <li>Durasi: {durasi}</li>
            <li>Transit: {transit}</li>
            <li>Maskapai: {maskapai}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # MAP
    map_df = pd.DataFrame([
        {"name": "Origin", "lat": b1["latitude"], "lon": b1["longitude"], "color": [30,144,255]},
        {"name": "Destination", "lat": b2["latitude"], "lon": b2["longitude"], "color": [220,38,38]}
    ])

    route_df = pd.DataFrame([{
        "from_lon": b1["longitude"], "from_lat": b1["latitude"],
        "to_lon": b2["longitude"], "to_lat": b2["latitude"]
    }])

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v10",
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[lon, lat]",
                get_radius=10000,
                get_fill_color="color",
                pickable=True
            ),
            pdk.Layer(
                "ArcLayer",
                data=route_df,
                get_source_position="[from_lon, from_lat]",
                get_target_position="[to_lon, to_lat]",
                get_width=4,
                get_source_color=[0,0,0],
                get_target_color=[0,0,0]
            )
        ],
        initial_view_state=pdk.ViewState(
            latitude=(b1["latitude"] + b2["latitude"]) / 2,
            longitude=(b1["longitude"] + b2["longitude"]) / 2,
            zoom=4
        ),
        tooltip={"text": "{name}"}
    ))

    st.caption("Biru: Bandara Keberangkatan • Merah: Bandara Tujuan")

    st.markdown("""
    <hr>
    <div style="text-align:center; font-size:14px; color:#6B7280; line-height:1.6;">
        Copyright © 2026 by Pengelola MK Praktikum Unggulan (Praktikum DGX), Universitas Gunadarma
        <br>
        <a href="https://www.praktikum-hpc.gunadarma.ac.id/" target="_blank">
            https://www.praktikum-hpc.gunadarma.ac.id/
        </a>
        <br>
        <a href="https://www.hpc-hub.gunadarma.ac.id/" target="_blank">
            https://www.hpc-hub.gunadarma.ac.id/
        </a>
    </div>
    """, unsafe_allow_html=True)
