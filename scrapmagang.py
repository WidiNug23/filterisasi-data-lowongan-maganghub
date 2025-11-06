import requests
import pandas as pd
import streamlit as st
import joblib
import time
import json
import time
import requests

def ambil_data_api():
    semua_data = []
    for page in range(1, 51):
        url = f"https://maganghub.kemnaker.go.id/api/lowongan?page={page}"
        for attempt in range(3):  # coba 3 kali
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
                data = response.json().get("data", [])
                if not data:
                    print(f"Hentikan di halaman {page}, tidak ada data lagi.")
                    return semua_data
                semua_data.extend(data)
                print(f"✅ Halaman {page} berhasil diambil ({len(data)} data)")
                break  # keluar dari loop retry kalau berhasil
            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout halaman {page}, percobaan ke-{attempt+1}")
                time.sleep(3)
            except Exception as e:
                print(f"⚠️ Gagal ambil data halaman {page}: {e}")
                break
        time.sleep(1)  # beri jeda agar server tidak overload
    return semua_data


# === Konfigurasi dasar ===
st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("Sistem Filterisasi Lowongan MagangHub")

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100  # batas per halaman dari API

# === CSS global ===
st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], 
[data-testid="stStatusWidget"], [data-testid="stStreamlitBadge"], 
.stAppDeployButton, div[data-testid="stBottomBlockContainer"],
div[class*="_profilePreview_"], div[data-testid="appCreatorAvatar"],
a[href*="share.streamlit.io/user/"], img[alt="App Creator Avatar"],
[data-testid="stHeader"] [href*="github.com"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    position: fixed !important;
    z-index: -999 !important;
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
        progress.progress(min(page * LIMIT / 10000, 1.0))
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

        # === Parsing program studi ===
        prog_studi_raw = item.get("program_studi")
        if prog_studi_raw:
            try:
                prog_list = json.loads(prog_studi_raw)
                program_studi = ", ".join([p.get("title", "").strip() for p in prog_list if p.get("title")])
            except Exception:
                program_studi = ""
        else:
            program_studi = ""

        # === Parsing jenjang ===
        jenjang_raw = item.get("jenjang")
        jenjang = ""
        if jenjang_raw:
            try:
                jenjang_list = json.loads(jenjang_raw)
                if isinstance(jenjang_list, list):
                    # Bisa berupa list of dict atau list of string
                    if all(isinstance(j, dict) for j in jenjang_list):
                        jenjang = ", ".join([j.get("title", "").strip() for j in jenjang_list if j.get("title")])
                    elif all(isinstance(j, str) for j in jenjang_list):
                        jenjang = ", ".join([j.strip() for j in jenjang_list if j.strip()])
            except Exception:
                # fallback kalau bukan JSON valid
                jenjang = str(jenjang_raw).strip()

        # === Simpan ke records ===
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

# === Load data utama dengan animasi teks ===
if "df" not in st.session_state:
    messages = [
        "Memuat data dari MagangHub...",
        "Semakin banyak data, semakin lama loading-nya 😅",
        "Periksa juga koneksi internetmu cuyy 🌐",
        "Take your time xixi ☕",
    ]
    
    # Placeholder untuk teks yang berganti
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # Ganti teks setiap 3 detik selama data dimuat
    for i in range(len(messages) * 2):  # total durasi = jumlah_pesan * 3 detik * 2 loop
        msg = messages[i % len(messages)]
        status_text.markdown(f"**{msg}**")
        progress_bar.progress((i + 1) / (len(messages) * 2))
        time.sleep(3)
        
        # Simulasi memuat data (ganti dengan fungsi aslimu)
        if i == 2:  # contoh: setelah beberapa detik data selesai
            st.session_state.df = load_data()
            break
    
    status_text.empty()
    progress_bar.empty()

# === Setelah data berhasil dimuat ===
df = st.session_state.df

if df.empty:
    st.warning("⚠️ Tidak ada data yang ditemukan.")
    st.stop()
else:
    st.success("✅ Data berhasil dimuat!")

# === Session state untuk filtered df ===
if "filtered_df" not in st.session_state:
    st.session_state.filtered_df = df.copy()

# === Filter input ===
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    search = st.text_input("🔍 Masukkan kata kunci (Instansi/Posisi/Lokasi/Program Studi/Jenjang)", key="search")
with col2:
    jenis_filter = st.selectbox("🏢 Jenis Instansi", ["Semua", "Negeri", "Swasta"], key="jenis")
with col3:
    cari_btn = st.button("🔎 Cari")

# === Fungsi filter ===
def apply_filter():
    filtered = st.session_state.df.copy()

    # Filter jenis instansi (jika dipilih)
    if jenis_filter != "Semua":
        filtered = filtered[filtered["Jenis Instansi"] == jenis_filter]

    # Filter kombinasi keyword
    if search.strip():
        keywords = [k.strip() for k in search.split() if k.strip()]
        for kw in keywords:
            mask = (
                filtered["Instansi"].str.contains(kw, case=False, na=False) |
                filtered["Lowongan"].str.contains(kw, case=False, na=False) |
                filtered["Lokasi"].str.contains(kw, case=False, na=False) |
                filtered["Program Studi"].str.contains(kw, case=False, na=False) |
                filtered["Jenjang"].str.contains(kw, case=False, na=False)
            )
            filtered = filtered[mask]

    return filtered


# === Jalankan filter ===
show_count = False
if cari_btn or search != "":
    st.session_state.filtered_df = apply_filter()
    show_count = bool(search.strip())

filtered_df = st.session_state.filtered_df

# === Jumlah hasil ===
if show_count:
    st.markdown(f"📄 Menampilkan **{len(filtered_df):,}** hasil pencarian.")

# === Pewarnaan peluang ===
def peluang_label(val):
    if pd.isna(val):
        return ""
    if val >= 75:
        warna = "#00CC66"
    elif val >= 50:
        warna = "#FFD700"
    elif val >= 25:
        warna = "#FF9900"
    else:
        warna = "#FF4B4B"
    return f"<span style='color:{warna}; font-weight:bold;'>{val}%</span>"

df_tampil = filtered_df.copy()
df_tampil["Tanggal Publikasi"] = df_tampil["Tanggal Publikasi"].dt.strftime("%d %b %Y %H:%M")
df_tampil.rename(columns={"Jenis Instansi": "* Jenis Instansi"}, inplace=True)

st.markdown(
    """
    <p style='color:#AAAAAA; font-size:13px; font-style:italic; text-align:left; margin-top:-10px; margin-bottom:10px;'>
    * data jenis instansi masih dalam tahap training dan pengembangan
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
        """
    <p style='color:#AAAAAA; font-size:13px; font-style:italic; text-align:left; margin-top:-10px; margin-bottom:10px;'>
    [untuk mengurutkan, klik/tekan masing-masing judul kolom]
    </p>
    """,
    unsafe_allow_html=True
)

# === Tabel utama ===
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

# === Tombol download CSV ===
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "💾 Download Hasil CSV",
    data=csv,
    file_name=f"lowongan_maganghub_{int(time.time())}.csv",
    mime="text/csv"
)
