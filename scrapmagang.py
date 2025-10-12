import requests
import pandas as pd
import streamlit as st
import joblib
import time
import json

# === Konfigurasi dasar ===
st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("Sistem Filterisasi Lowongan MagangHub")

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100  # batas per halaman dari API

# === CSS untuk sembunyikan elemen default Streamlit ===
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

# === Load model dan vectorizer ===
@st.cache_resource
def load_model():
    model = joblib.load("model_maganghub.pkl")
    vectorizer = joblib.load("vectorizer_maganghub.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# === Fungsi ambil data API ===
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
            break

        all_data.extend(data)
        status.text(f"📄 Mengambil halaman {page} ({len(data)} data)... Total: {len(all_data)} lowongan")
        progress.progress(min(page * LIMIT / 3000, 1.0))
        page += 1
        time.sleep(0.05)

    progress.empty()
    final_text = f"Diperoleh {format(len(all_data), ',').replace(',', '.')} lowongan"
    status.text(final_text)
    st.session_state["last_status"] = final_text
    return all_data

# === Fungsi olah data API menjadi DataFrame ===
def load_data():
    data = ambil_data_api()
    records = []
    for item in data:
        perusahaan = item.get("perusahaan", {}) or {}
        nama = perusahaan.get("nama_perusahaan", "")
        alamat = perusahaan.get("alamat", "")
        deskripsi = perusahaan.get("deskripsi_perusahaan", "")
        teks = f"{nama} {alamat} {deskripsi}"

        # Prediksi jenis instansi
        jenis_pred = model.predict(vectorizer.transform([teks]))[0]
        if jenis_pred not in ["Negeri", "Swasta"]:
            jenis_pred = "Swasta"

        kuota = item.get("jumlah_kuota", 0)
        daftar = item.get("jumlah_terdaftar", 0)
        peluang = 100 if daftar == 0 else min(round((kuota / (daftar + 1)) * 100), 100)

        # === Ambil dan parse field program_studi ===
        raw_prodi = item.get("program_studi")
        if raw_prodi:
            try:
                parsed_prodi = json.loads(raw_prodi)
                list_prodi = [p.get("title", "").strip() for p in parsed_prodi if p.get("title")]
                nama_prodi = ", ".join(list_prodi) if list_prodi else "-"
            except Exception:
                nama_prodi = "-"
        else:
            nama_prodi = "-"

        records.append({
            "Lowongan": item.get("posisi", ""),
            "Program Studi": nama_prodi,
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
if "df" not in st.session_state:
    with st.spinner("Memuat data dari MagangHub..."):
        st.session_state.df = load_data()

df = st.session_state.df

if df.empty:
    st.warning("⚠️ Tidak ada data yang ditemukan.")
    st.stop()

# === Session state untuk filtered df ===
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = df.copy()

# === Filter input ===
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("🔍 Masukkan kata kunci (Instansi/Posisi/Program Studi/Lokasi)", key="search")
with col2:
    jenis_filter = st.selectbox("🏢 Jenis Instansi", ["Semua", "Negeri", "Swasta"], key="jenis")
with col3:
    cari_btn = st.button("🔎 Cari")

# === Fungsi filter ===
def apply_filter():
    filtered = st.session_state.df.copy()
    if jenis_filter != "Semua":
        filtered = filtered[filtered["Jenis Instansi"] == jenis_filter]

    if search.strip():
        filtered = filtered[
            filtered["Instansi"].str.contains(search, case=False, na=False) |
            filtered["Lowongan"].str.contains(search, case=False, na=False) |
            filtered["Program Studi"].str.contains(search, case=False, na=False) |
            filtered["Lokasi"].str.contains(search, case=False, na=False)
        ]
    return filtered

# === Jalankan filter ===
show_count = False
if cari_btn or search != "":
    st.session_state.filtered_df = apply_filter()
    show_count = bool(search.strip())

filtered_df = st.session_state.filtered_df

# === Tampilkan jumlah hasil ===
if show_count:
    st.markdown(f"📄 Menampilkan **{len(filtered_df):,}** hasil pencarian.")

# === Format tabel ===
df_tampil = filtered_df.copy()
df_tampil["Tanggal Publikasi"] = df_tampil["Tanggal Publikasi"].dt.strftime("%d %b %Y %H:%M")
df_tampil.rename(columns={"Jenis Instansi": "* Jenis Instansi"}, inplace=True)

st.dataframe(
    df_tampil.style.format({
        "Peluang Lolos (%)": lambda x: f"{x}%" if pd.notna(x) else "-"
    }),
    use_container_width=True,
    height=850
)

# === Tombol download CSV ===
csv = df_tampil.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Download Hasil CSV",
    data=csv,
    file_name=f"lowongan_maganghub_{int(time.time())}.csv",
    mime="text/csv"
)
