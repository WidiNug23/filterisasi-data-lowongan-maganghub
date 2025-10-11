import requests
import pandas as pd
import streamlit as st
import time, re, joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# =====================
# KONFIGURASI DASAR
# =====================
BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100
MAX_PAGE = 30
MODEL_PATH = "model_maganghub.pkl"
VECTORIZER_PATH = "vectorizer_maganghub.pkl"

# =====================
# THEME DAN PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Filterisasi Lowongan Magang",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Sistem Analisis Lowongan MagangHub")

# Sidebar Theme
theme_option = st.sidebar.selectbox("Pilih Theme", ["Light", "Dark"])
if theme_option == "Dark":
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: #f0f0f0;
        }
        .stButton>button {
            background-color: #1e293b;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

# =====================
# FUNGSI AMBIL DATA API
# =====================
def ambil_data_api():
    all_data = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for page in range(1, MAX_PAGE + 1):
        url = f"{BASE_URL}?page={page}&limit={LIMIT}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            st.error(f"Gagal ambil data di halaman {page}: {e}")
            break

        json_data = response.json()
        data = json_data.get("data", [])
        if not data:
            break

        all_data.extend(data)
        status_text.text(f"✅ Halaman {page} — {len(data)} data diambil")
        progress_bar.progress(page / MAX_PAGE)
        time.sleep(0.1)

    status_text.text("✅ Pengambilan data selesai!")
    progress_bar.empty()
    return all_data

# =====================
# LOGIKA LABEL OTOMATIS
# =====================
def label_otomatis(nama, alamat, deskripsi):
    text = f"{nama} {alamat} {deskripsi}".lower()
    if re.search(r"\b(kementerian|dinas|badan|lembaga|sekretariat|pemerintah|provinsi|kabupaten|universitas negeri|politeknik negeri|bank rakyat indonesia)\b", text):
        return "Pemerintah / Negeri"
    elif re.search(r"\b(pt|cv)\b", text) and re.search(r"\b(alfa|astra|indofood|unilever|mustika|midi|bank|finance|group|holding|retail|industri|corporate|international|mining|energy|technology|chemical|indonesia)\b", text):
        return "Swasta Besar"
    elif "pt" in text or "cv" in text:
        return "Swasta Kecil"
    else:
        return "Lainnya"

# =====================
# LOAD DATA OTOMATIS
# =====================
@st.cache_data(show_spinner=False)
def load_data():
    all_data = ambil_data_api()
    if not all_data:
        return pd.DataFrame()

    # Load model jika ada
    model, vectorizer = None, None
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

    records = []
    texts = [f"{item.get('perusahaan', {}).get('nama_perusahaan','')} "
             f"{item.get('perusahaan', {}).get('alamat','')} "
             f"{item.get('perusahaan', {}).get('deskripsi_perusahaan','')}"
             for item in all_data]

    # Prediksi batch
    if model and vectorizer:
        X_text = vectorizer.transform(texts)
        predictions = model.predict(X_text)
    else:
        predictions = [label_otomatis(
            f"{item.get('perusahaan', {}).get('nama_perusahaan','')}",
            f"{item.get('perusahaan', {}).get('alamat','')}",
            f"{item.get('perusahaan', {}).get('deskripsi_perusahaan','')}")
            for item in all_data]

    for item, jenis in zip(all_data, predictions):
        perusahaan = item.get("perusahaan", {})
        jadwal = item.get("jadwal", {})
        created_at = item.get("created_at", item.get("tanggal_mulai"))
        try:
            created_at_dt = pd.to_datetime(created_at)
        except:
            created_at_dt = pd.NaT

        records.append({
            "Judul": item.get("posisi", ""),
            "Instansi": perusahaan.get("nama_perusahaan", ""),
            "Lokasi": f"{perusahaan.get('nama_kabupaten', '')}, {perusahaan.get('nama_provinsi', '')}",
            "Tanggal Mulai": jadwal.get("tanggal_mulai", ""),
            "Tanggal Selesai": jadwal.get("tanggal_selesai", ""),
            "Jumlah Kuota": item.get("jumlah_kuota", ""),
            "Jumlah Terdaftar": item.get("jumlah_terdaftar", ""),
            "Status": item.get("ref_status_posisi", {}).get("nama_status_posisi", ""),
            "Jenis Instansi": jenis,
            "Tanggal Ditambahkan": created_at_dt,
            "Banner": perusahaan.get("banner", ""),
        })

    df = pd.DataFrame(records)
    df.sort_values(by="Tanggal Ditambahkan", ascending=False, inplace=True)
    return df

df = load_data()

# =====================
# TAMPILKAN DATA
# =====================
if df.empty:
    st.warning("⚠️ Tidak ada data lowongan yang tersedia.")
else:
    st.subheader("📊 Data Lowongan Magang")
    
    # Sidebar filter
    jenis_filter = st.sidebar.selectbox("Filter Jenis Instansi", ["Semua"] + sorted(df["Jenis Instansi"].unique().tolist()))
    if jenis_filter != "Semua":
        df = df[df["Jenis Instansi"] == jenis_filter]

    search = st.sidebar.text_input("Cari Instansi atau Posisi")
    if search:
        df = df[df["Instansi"].str.contains(search, case=False, na=False) |
                df["Judul"].str.contains(search, case=False, na=False)]

    st.write(f"Menampilkan {len(df)} data dari total {len(df)} lowongan.")
    st.dataframe(df, use_container_width=True)

    st.subheader("🖼️ Contoh Banner Perusahaan")
    for _, row in df.head(3).iterrows():
        st.markdown(f"**{row['Instansi']}** — {row['Judul']} — {row['Tanggal Ditambahkan']}")
        if row["Banner"]:
            st.image(row["Banner"], width=400)
        else:
            st.write("_Tidak ada banner tersedia_")

    # Tombol download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", data=csv, file_name="lowongan_maganghub.csv", mime="text/csv")
