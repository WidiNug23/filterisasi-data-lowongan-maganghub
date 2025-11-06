import streamlit as st
import pandas as pd
import requests
import time
import json
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Konfigurasi dasar ===
st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("📊 Sistem Filterisasi Lowongan MagangHub (Real-Time)")

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100
MAKS_HALAMAN = 200
MAKS_WORKER = 15
REFRESH_INTERVAL = 300  # refresh tiap 5 menit

# === CSS tampilan bersih ===
st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stStreamlitBadge"],
.stAppDeployButton, div[data-testid="stBottomBlockContainer"],
div[class*="_profilePreview_"], div[data-testid="appCreatorAvatar"],
a[href*="share.streamlit.io/user/"], img[alt="App Creator Avatar"],
[data-testid="stHeader"] [href*="github.com"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# === Load model & vectorizer ===
@st.cache_resource
def load_model():
    model = joblib.load("model_maganghub.pkl")
    vectorizer = joblib.load("vectorizer_maganghub.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# === Ambil satu halaman data ===
def ambil_halaman(page, uniq):
    """Ambil satu halaman data dengan tambahan parameter unik agar tidak di-cache."""
    url = f"{BASE_URL}?page={page}&limit={LIMIT}&t={uniq}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json().get("data", [])
    except Exception:
        return []

# === Ambil semua data API (stop otomatis bila kosong 3x) ===
def ambil_data_api():
    all_data = []
    status = st.empty()
    progress = st.progress(0)
    stop_counter = 0
    berhasil = 0
    batch_size = 10
    uniq = int(time.time())  # buat token unik agar cache API tidak menempel

    for start_page in range(1, MAKS_HALAMAN + 1, batch_size):
        pages = list(range(start_page, min(start_page + batch_size, MAKS_HALAMAN + 1)))

        with ThreadPoolExecutor(max_workers=MAKS_WORKER) as executor:
            futures = {executor.submit(ambil_halaman, p, uniq): p for p in pages}
            for future in as_completed(futures):
                page = futures[future]
                try:
                    data = future.result()
                    if data:
                        all_data.extend(data)
                        berhasil += 1
                        stop_counter = 0
                    else:
                        stop_counter += 1
                except Exception:
                    stop_counter += 1

                progress.progress(min(page / MAKS_HALAMAN, 1.0))
                status.text(f"📄 Memuat halaman {page} — total {len(all_data):,} data...")

        if stop_counter >= 3:  # hentikan bila 3 batch kosong berturut
            break

    progress.empty()
    status.text(f"✅ Diperoleh total {len(all_data):,} lowongan (dari {berhasil} halaman berhasil).")
    return all_data

# === Konversi data ke DataFrame ===
def load_data():
    data = ambil_data_api()
    records = []

    for item in data:
        perusahaan = item.get("perusahaan", {}) or {}
        nama = perusahaan.get("nama_perusahaan", "")
        alamat = perusahaan.get("alamat", "")
        deskripsi = perusahaan.get("deskripsi_perusahaan", "")
        teks = f"{nama} {alamat} {deskripsi}"

        jenis_pred = model.predict(vectorizer.transform([teks]))[0]
        if jenis_pred not in ["Negeri", "Swasta"]:
            jenis_pred = "Swasta"

        kuota = item.get("jumlah_kuota", 0)
        daftar = item.get("jumlah_terdaftar", 0)
        peluang = 100 if daftar == 0 else min(round((kuota / (daftar + 1)) * 100), 100)

        prog_studi_raw = item.get("program_studi")
        if prog_studi_raw:
            try:
                prog_list = json.loads(prog_studi_raw)
                program_studi = ", ".join([p.get("title", "").strip() for p in prog_list if p.get("title")])
            except Exception:
                program_studi = ""
        else:
            program_studi = ""

        jenjang_raw = item.get("jenjang")
        jenjang = ""
        if jenjang_raw:
            try:
                jenjang_list = json.loads(jenjang_raw)
                if isinstance(jenjang_list, list):
                    if all(isinstance(j, dict) for j in jenjang_list):
                        jenjang = ", ".join([j.get("title", "").strip() for j in jenjang_list if j.get("title")])
                    elif all(isinstance(j, str) for j in jenjang_list):
                        jenjang = ", ".join([j.strip() for j in jenjang_list if j.strip()])
            except Exception:
                jenjang = str(jenjang_raw).strip()

        records.append({
            "Lowongan": item.get("posisi", ""),
            "Instansi": nama,
            "Jenis Instansi": jenis_pred,
            "Program Studi": program_studi,
            "Jenjang": jenjang,
            "Lokasi": f"{perusahaan.get('nama_kabupaten', '')}, {perusahaan.get('nama_provinsi', '')}",
            "Jumlah Kuota": kuota,
            "Jumlah Terdaftar": daftar,
            "Peluang Lolos (%)": peluang,
            "Tanggal Publikasi": pd.to_datetime(item.get("created_at", None), errors="coerce")
        })

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["Lowongan", "Instansi"], inplace=True)
    return df

# === Logika refresh otomatis ===
def perlu_refresh():
    last_time = st.session_state.get("last_update_time", 0)
    return (time.time() - last_time) > REFRESH_INTERVAL

# Paksa refresh setiap reload halaman (menghapus cache session lama)
if "session_flag" not in st.session_state:
    st.session_state.clear()
    st.session_state.session_flag = True

if "df" not in st.session_state or perlu_refresh():
    with st.spinner("🔄 Mengambil data terbaru dari MagangHub..."):
        st.session_state.df = load_data()
        st.session_state.last_update_time = time.time()

df = st.session_state.df

if df.empty:
    st.warning("⚠️ Tidak ada data ditemukan.")
    st.stop()

# === Filter input ===
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("🔍 Kata kunci (Instansi/Posisi/Lokasi/Prodi/Jenjang)")
with col2:
    jenis_filter = st.selectbox("🏢 Jenis Instansi", ["Semua", "Negeri", "Swasta"])
with col3:
    cari_btn = st.button("🔎 Cari")

def apply_filter():
    filtered = df.copy()
    if jenis_filter != "Semua":
        filtered = filtered[filtered["Jenis Instansi"] == jenis_filter]
    if search.strip():
        for kw in search.split():
            mask = (
                filtered["Instansi"].str.contains(kw, case=False, na=False) |
                filtered["Lowongan"].str.contains(kw, case=False, na=False) |
                filtered["Lokasi"].str.contains(kw, case=False, na=False) |
                filtered["Program Studi"].str.contains(kw, case=False, na=False) |
                filtered["Jenjang"].str.contains(kw, case=False, na=False)
            )
            filtered = filtered[mask]
    return filtered

if cari_btn or search.strip():
    filtered_df = apply_filter()
else:
    filtered_df = df.copy()

# === Tabel dengan pewarnaan ===
df_tampil = filtered_df.copy()
df_tampil["Tanggal Publikasi"] = df_tampil["Tanggal Publikasi"].dt.strftime("%d %b %Y %H:%M")

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

# === Tombol download ===
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Download CSV",
    data=csv,
    file_name=f"lowongan_maganghub_{int(time.time())}.csv",
    mime="text/csv"
)
