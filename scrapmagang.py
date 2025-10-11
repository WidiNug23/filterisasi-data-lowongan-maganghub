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

st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("Sistem Analisis Lowongan MagangHub")

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
        time.sleep(0.2)

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
# STATE DATA
# =====================
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

# =====================
# TOMBOL LATIH MODEL
# =====================
if st.button("🔁 Latih Ulang Model (Otomatis)"):
    st.info("⏳ Sedang melatih model berdasarkan data MagangHub...")
    data_latih = ambil_data_api()

    records = []
    for item in data_latih:
        perusahaan = item.get("perusahaan", {})
        nama = perusahaan.get("nama_perusahaan", "")
        alamat = perusahaan.get("alamat", "")
        deskripsi = perusahaan.get("deskripsi_perusahaan", "")
        label = label_otomatis(nama, alamat, deskripsi)
        records.append({"text": f"{nama} {alamat} {deskripsi}", "label": label})

    df_latih = pd.DataFrame(records)
    vectorizer = TfidfVectorizer(stop_words=None, max_features=1000)
    X = vectorizer.fit_transform(df_latih["text"])
    y = df_latih["label"]

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    st.success("✅ Model berhasil dilatih ulang dan disimpan!")

# =====================
# TOMBOL AMBIL DATA
# =====================
if st.button("📥 Ambil Data Lowongan"):
    st.info("⏳ Mengambil data lowongan magang dari MagangHub...")
    all_data = ambil_data_api()

    if not all_data:
        st.warning("⚠️ Tidak ada data yang diambil dari API.")
    else:
        model, vectorizer = None, None
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)

        records = []
        with st.spinner("📌 Memproses data..."):
            texts = []
            for item in all_data:
                perusahaan = item.get("perusahaan", {})
                nama = perusahaan.get("nama_perusahaan", "")
                alamat = perusahaan.get("alamat", "")
                deskripsi = perusahaan.get("deskripsi_perusahaan", "")
                texts.append(f"{nama} {alamat} {deskripsi}")

            if model and vectorizer:
                X_text = vectorizer.transform(texts)
                predictions = model.predict(X_text)
            else:
                predictions = [label_otomatis(f"{item.get('perusahaan', {}).get('nama_perusahaan','')}",
                                               f"{item.get('perusahaan', {}).get('alamat','')}",
                                               f"{item.get('perusahaan', {}).get('deskripsi_perusahaan','')}") 
                               for item in all_data]

            for item, jenis in zip(all_data, predictions):
                perusahaan = item.get("perusahaan", {})
                jadwal = item.get("jadwal", {})
                created_at = item.get("created_at", item.get("tanggal_mulai"))  # fallback jika tidak ada created_at
                try:
                    created_at_dt = pd.to_datetime(created_at)
                except:
                    created_at_dt = pd.NaT

                records.append({
                    "judul": item.get("posisi"),
                    "instansi_nama": perusahaan.get("nama_perusahaan", ""),
                    "lokasi": f"{perusahaan.get('nama_kabupaten', '')}, {perusahaan.get('nama_provinsi', '')}",
                    "tanggal_mulai": jadwal.get("tanggal_mulai"),
                    "tanggal_selesai": jadwal.get("tanggal_selesai"),
                    "jumlah_kuota": item.get("jumlah_kuota"),
                    "jumlah_terdaftar": item.get("jumlah_terdaftar"),
                    "status": item.get("ref_status_posisi", {}).get("nama_status_posisi"),
                    "jenis_instansi": jenis,
                    "created_at": created_at_dt,
                    "banner": perusahaan.get("banner"),
                })

        df = pd.DataFrame(records)
        # Urutkan dari terbaru ke terlama
        df.sort_values(by="created_at", ascending=False, inplace=True)
        st.session_state.dataframe = df
        st.success("✅ Data berhasil diambil dan disimpan ke memori!")

# =====================
# TAMPILKAN DATA
# =====================
if st.session_state.dataframe is not None:
    df = st.session_state.dataframe.copy()

    st.subheader("Data Lowongan Magang")

    # Filter Jenis Instansi
    jenis_filter = st.selectbox("🧩 Filter Jenis Instansi", ["Semua"] + sorted(df["jenis_instansi"].unique().tolist()))
    if jenis_filter != "Semua":
        df = df[df["jenis_instansi"] == jenis_filter]

    # Pencarian
    search = st.text_input("🔍 Cari nama instansi atau posisi:")
    if search:
        df = df[df["instansi_nama"].str.contains(search, case=False, na=False) |
                df["judul"].str.contains(search, case=False, na=False)]

    st.write(f"Menampilkan {len(df)} data dari total {len(st.session_state.dataframe)} lowongan.")
    st.dataframe(df, use_container_width=True)

    # Banner preview
    st.subheader("🖼️ Contoh Banner Perusahaan")
    for _, row in df.head(3).iterrows():
        st.markdown(f"**{row['instansi_nama']}** — {row['judul']} — {row['created_at']}")
        if row["banner"]:
            st.image(row["banner"], width=400)
        else:
            st.write("_Tidak ada banner tersedia_")

    # Tombol Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download CSV", data=csv, file_name="lowongan_maganghub.csv", mime="text/csv")

else:
    st.info("Klik 📥 Ambil Data Lowongan untuk mulai menampilkan data.")
