import requests
import pandas as pd
import streamlit as st
import joblib
import time

# === Konfigurasi dasar ===
st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("🎯 Sistem Filterisasi Lowongan MagangHub")

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100  # batas per halaman dari API

# === Load model dan vectorizer ===
@st.cache_resource
def load_model():
    model = joblib.load("model_maganghub.pkl")
    vectorizer = joblib.load("vectorizer_maganghub.pkl")
    return model, vectorizer

model, vectorizer = load_model()


# === Ambil semua data API tanpa batas halaman ===
def ambil_data_api():
    all_data = []
    page = 1
    status = st.empty()
    progress = st.progress(0)

    while True:
        url = f"{BASE_URL}?page={page}&limit={LIMIT}"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
        except Exception as e:
            st.error(f"⚠️ Gagal ambil data halaman {page}: {e}")
            break

        data = res.json().get("data", [])
        if not data:
            break  # berhenti jika halaman kosong

        all_data.extend(data)
        status.text(f"📄 Mengambil halaman {page} ({len(data)} data)... Total: {len(all_data)}")
        progress.progress(min(page * LIMIT / 2500, 1.0))  # estimasi total data ±2500
        page += 1
        time.sleep(0.05)

    progress.empty()
    status.text(f"✅ Selesai! Total data diperoleh: {len(all_data):,}")
    return all_data


# === Fungsi muat dan olah data ===
@st.cache_data(show_spinner=False)
def load_data():
    data = ambil_data_api()
    records = []
    for item in data:
        perusahaan = item.get("perusahaan", {}) or {}
        nama = perusahaan.get("nama_perusahaan", "")
        alamat = perusahaan.get("alamat", "")
        deskripsi = perusahaan.get("deskripsi_perusahaan", "")
        teks = f"{nama} {alamat} {deskripsi}"

        # Prediksi jenis instansi (Negeri / Swasta)
        jenis_pred = model.predict(vectorizer.transform([teks]))[0]
        if jenis_pred not in ["Negeri", "Swasta"]:
            jenis_pred = "Swasta"

        kuota = item.get("jumlah_kuota", 0)
        daftar = item.get("jumlah_terdaftar", 0)
        peluang = 100 if daftar == 0 else min(round((kuota / (daftar + 1)) * 100), 100)

        records.append({
            "Lowongan": item.get("posisi", ""),
            "Instansi": nama,
            "Jenis Instansi": jenis_pred,
            "Lokasi": f"{perusahaan.get('nama_kabupaten', '')}, {perusahaan.get('nama_provinsi', '')}",
            "Jumlah Kuota": kuota,
            "Jumlah Terdaftar": daftar,
            "Peluang Lolos (%)": peluang,
            "Tanggal Publikasi": pd.to_datetime(item.get("created_at", None), errors="coerce")
        })

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["Lowongan", "Instansi"], inplace=True)
    return df


# === Load data utama ===
with st.spinner("🔄 Memuat data dari MagangHub..."):
    df = load_data()

if df.empty:
    st.warning("⚠️ Tidak ada data yang ditemukan.")
    st.stop()

# === Inisialisasi session_state ===
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = df.copy()

# === Filter Input ===
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("🔍 Masukkan kata kunci (Instansi/Posisi Lowongan/Lokasi)", key="search")
with col2:
    jenis_filter = st.selectbox("🏢 Jenis Instansi", ["Semua", "Negeri", "Swasta"], key="jenis")
with col3:
    cari_btn = st.button("🔎 Cari")

# === Fungsi filter ===
def apply_filter():
    filtered = df.copy()

    # Filter jenis instansi (bisa tanpa keyword)
    if jenis_filter != "Semua":
        filtered = filtered[filtered["Jenis Instansi"] == jenis_filter]

    # Filter berdasarkan kata kunci jika diisi
    if search.strip():
        filtered = filtered[
            filtered["Instansi"].str.contains(search, case=False, na=False) |
            filtered["Lowongan"].str.contains(search, case=False, na=False) |
            filtered["Lokasi"].str.contains(search, case=False, na=False)
        ]
    return filtered


# === Jalankan filter jika tombol diklik atau Enter ditekan ===
show_count = False
if cari_btn or search != "":
    # Jalankan filter apapun kondisi inputnya
    st.session_state.filtered_df = apply_filter()

    # Tampilkan jumlah hanya jika user isi keyword
    show_count = bool(search.strip())

filtered_df = st.session_state.filtered_df

# === Tampilkan jumlah hasil hanya jika ada kata kunci ===
if show_count:
    st.markdown(f"📄 Menampilkan **{len(filtered_df):,}** hasil pencarian.")

# === Pewarnaan angka peluang ===
def peluang_label(val):
    if pd.isna(val):
        return ""
    if val >= 75:
        warna = "#00CC66"  # hijau
    elif val >= 50:
        warna = "#FFD700"  # kuning
    elif val >= 25:
        warna = "#FF9900"  # oranye
    else:
        warna = "#FF4B4B"  # merah
    return f"<span style='color:{warna}; font-weight:bold;'>{val}%</span>"


# === Format data tampil ===
df_tampil = filtered_df.copy()
df_tampil["Tanggal Publikasi"] = df_tampil["Tanggal Publikasi"].dt.strftime("%d %b %Y %H:%M")

# === Tabel tampil ===
st.dataframe(
    df_tampil.style.format({
        "Peluang Lolos (%)": lambda x: f"{x}%" if pd.notna(x) else "-"
    }).apply(
        lambda s: [
            "color: #00CC66; font-weight:bold;" if v >= 75 else
            "color: #FFD700; font-weight:bold;" if v >= 50 else
            "color: #FF9900; font-weight:bold;" if v >= 25 else
            "color: #FF4B4B; font-weight:bold;" if pd.notna(v) else ""
            for v in s
        ],
        subset=["Peluang Lolos (%)"]
    ),
    use_container_width=True,
    height=850
)

# === Sembunyikan toolbar & ubah ke dark mode ===
custom_css = """
    <style>
    [data-testid="stToolbar"] {visibility: hidden !important;}
    [data-testid="stDecoration"] {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    #MainMenu, header, footer {visibility: hidden !important;}

    [data-testid="stDecoration"] {visibility: hidden !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important;}
    .stAppDeployButton {display: none !important;}
    [data-testid="stStreamlitBadge"] {visibility: hidden !important;}
    footer:has([alt="Streamlit"]) {display: none !important;}
    div[data-testid="stBottomBlockContainer"] {visibility: hidden !important;}

    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }

    div[data-testid="stDataFrame"] table {
        background-color: #1E1E1E !important;
        color: #FAFAFA !important;
    }

    .stMarkdown, .stTextInput label, .stSelectbox label {
        color: #FFFFFF !important;
    }

    .stButton>button {
        background-color: #00CC66 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
    }

    .stButton>button:hover {
        background-color: #00994C !important;
    }

        /* Hilangkan banner 'Hosted with Streamlit' */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    [data-testid="stStreamlitBadge"] {
        display: none !important;
    }
    footer:has([alt="Streamlit"]) {
        display: none !important;
    }
    /* Pastikan kontainer bawah benar-benar kosong */
    div[data-testid="stBottomBlockContainer"] {
        display: none !important;
    }
    </style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# === Tombol download CSV ===
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Download Hasil CSV",
    data=csv,
    file_name=f"lowongan_maganghub_{int(time.time())}.csv",
    mime="text/csv"
)
