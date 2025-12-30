import streamlit as st

#  Judul
st.set_page_config(
    page_title="Prediksi Harga Tiket Pesawat",
    layout="centered"
)


# CSS (Ui)
st.markdown("""
<style>
/* ===== FORCE LIGHT MODE ===== */
html, body, [data-testid="stApp"] {
    background-color: #F5F7FA !important;
    color: #1F2937 !important;
}

/* Container */
.block-container {
    background-color: #F5F7FA !important;
}

/* ===== STREAMLIT INPUT FIX ===== */

/* Selectbox, TextInput, TextArea */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border-radius: 12px !important;
    border: 1px solid #D1D5DB !important;
}

/* Text inside input */
input, textarea {
    color: #111827 !important;
    background-color: #FFFFFF !important;
}

/* Placeholder */
input::placeholder {
    color: #9CA3AF !important;
}

/* Dropdown menu */
ul {
    background-color: #FFFFFF !important;
    color: #111827 !important;
}

/* ===== TITLES ===== */
h1 {
    color: #1A73E8;
    font-weight: 700;
}

/* Section title */
.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-top: 24px;
    margin-bottom: 12px;
    color: #111827;
}

/* ===== CARD ===== */
.card {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    margin-top: 24px;
    border: 1px solid #E5E7EB;
}

/* Price */
.price {
    font-size: 32px;
    font-weight: 700;
    color: #16A34A;
}

/* Caption */
.caption {
    color: #6B7280;
    font-size: 14px;
}

/* ===== BUTTON ===== */
.stButton>button {
    background-color: #1A73E8 !important;
    color: white !important;
    border-radius: 12px;
    padding: 12px 20px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    background-color: #1558C0 !important;
}
/* ===== FIX TEXT COLOR STREAMLIT INPUT ===== */

/* Selected value (selectbox) */
div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* Dropdown options */
div[data-baseweb="menu"] span {
    color: #111827 !important;
}

/* Placeholder selectbox */
div[data-baseweb="select"] [aria-selected="false"] {
    color: #6B7280 !important;
}

/* Input cursor */
input {
    caret-color: #111827 !important;
}

/* Disabled / faded text */
div[aria-disabled="true"] {
    color: #9CA3AF !important;
}
</style>
""", unsafe_allow_html=True)




# import
import os
import pandas as pd
import numpy as np
import joblib
import math
from sklearn.base import BaseEstimator, TransformerMixin

# Feature Engineering 
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


# Load Model &+Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model_linear_regression.joblib")
    return joblib.load(model_path)

@st.cache_data
def load_airports():
    csv_path = os.path.join(BASE_DIR, "airports_clean (1).csv")
    df = pd.read_csv(csv_path)
    df["label"] = df["nama_bandara"] + " (" + df["airport_code"] + ")"
    return df

model = load_model()
bandara = load_airports()

# Haversine ( itung Jarak berdasar La & Long)
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
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Markdown sesuai data
MASKAPAI_OPTIONS = [
    "Garuda Indonesia",
    "Lion Air",
    "Batik Air",
    "Transit Maskapai",
    "Citilink",
    "Sriwijaya Air",
    "AirAsia Indonesia",
    "Super Air Jet",
    "Transit Premium",
    "Garuda Indonesia Business",
    "Sriwijaya Air Premium",
    "Wings Air"
]

INFO_OPTIONS = [
    "No info",
    "In-flight meal not included",
    "No check-in baggage included",
    "1 Long layover",
    "Change airports",
    "Business class",
    "No Info",
    "1 Short layover",
    "Red-eye flight",
    "2 Long layover"
]

# Header
st.title(" Prediksi Harga Tiket Pesawat")
st.markdown("Cek estimasi harga tiket berdasarkan detail penerbangan Anda.")

# Form Input
with st.container():
    st.markdown('<div class="section-title"> Detail Penerbangan</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        maskapai = st.selectbox("Maskapai", MASKAPAI_OPTIONS)
        tgl = st.text_input("Tanggal Perjalanan (dd/mm/yyyy)", "24/03/2019")
        waktu_dep = st.text_input("Waktu Berangkat (HH:MM)", "22:20")
        durasi = st.text_input("Durasi Penerbangan", "2h 50m")

    with col2:
        transit = st.selectbox("Jumlah Transit", ["non-stop", "1 stop", "2 stops"])
        info = st.selectbox("Informasi Tambahan", INFO_OPTIONS)
        waktu_arr = st.text_input("Waktu Tiba", "01:10 22 Mar")

    st.markdown('<div class="section-title"> Rute Penerbangan</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        asal = st.selectbox(
            "Bandara Asal",
            options=bandara["label"].tolist(),
            index=None,
            placeholder="Ketik nama bandara atau kode (CGK, SUB, DPS...)"
        )

    with col4:
        tujuan = st.selectbox(
            "Bandara Tujuan",
            options=bandara["label"].tolist(),
            index=None,
            placeholder="Ketik nama bandara atau kode (CGK, SUB, DPS...)"
        )


# Button
st.markdown("<br>", unsafe_allow_html=True)
center = st.columns([1,2,1])
with center[1]:
    predict = st.button(" Prediksi Harga", use_container_width=True)

# PPrediksi (Model)
if predict:
    if asal == tujuan:
        st.error("Bandara asal dan tujuan tidak boleh sama.")
    else:
        b1 = bandara[bandara["label"] == asal].iloc[0]
        b2 = bandara[bandara["label"] == tujuan].iloc[0]

        jarak_km = haversine(
            b1["latitude"], b1["longitude"],
            b2["latitude"], b2["longitude"]
        )

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

        st.markdown(f"""
        <div class="card">
            <div class="price"> Rp {pred:,.0f}</div>
            <div class="caption"> Jarak penerbangan: {jarak_km:.2f} km</div>
        </div>
        """, unsafe_allow_html=True)




