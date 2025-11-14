import streamlit as st
import pandas as pd
import requests
import time
import json
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# === Konfigurasi dasar ===
st.set_page_config(page_title="Filterisasi Lowongan Magang", layout="wide")
st.title("Sistem Filterisasi Lowongan MagangHub")

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 1400                   # Ubah ke nilai aman
MAKS_HALAMAN = 600
MAKS_WORKER = 30            # 100 terlalu besar, bikin server throttle
REFRESH_INTERVAL = 5000
ITEMS_PER_PAGE = 40 # agar 3 kolom pas

# === CSS Modern & Neon + sembunyikan navbar ===
st.markdown("""
<style>
/* Sembunyikan navbar Streamlit */
header, footer {visibility: hidden;}
/* Card */
.card {
    border-radius: 12px;
    padding: 16px;
    margin: 10px 5px;
    background-color: #121212;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    transition: 0.3s;
    border: 2px solid transparent;
    min-height: 220px;
}
.card:hover { 
    border: 2px solid #00FFFF; 
    box-shadow: 0 8px 24px rgba(0,255,255,0.6); 
}
.card-title { font-size: 18px; font-weight: 700; color: #00FFFF; margin-bottom: 4px; }
.card-subtitle { font-size: 16px; color: #D78FEE; margin-bottom: 2px; }
.card-subtitle2 { font-size: 14px; color: #B0B0B0; margin-bottom: 6px; }
.card-detail { font-size: 12px; color: #E0E0E0; margin-bottom: 3px; }
.peluang-high { color: #39FF14; font-weight:bold; }
.peluang-medium { color: #FFD700; font-weight:bold; }
.peluang-low { color: #FF4500; font-weight:bold; }
.peluang-verylow { color: #FF073A; font-weight:bold; }

/* Pagination responsive horizontal */
.page-nav { display: flex; justify-content: center; flex-wrap: wrap; margin: 12px 0; gap:4px; }
.page-btn {
    background-color: #121212; 
    color: #00FFFF; 
    border: 1px solid #00FFFF; 
    padding: 4px 10px; 
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}
.page-btn:hover { background-color: #00FFFF; color: #121212; }
.page-btn.active { background-color: #00FFFF; color: #121212; }

/* Tombol neon */
.neon-btn {
    background-color: #121212; 
    color: #00FFFF;
    border: 2px solid #00FFFF;
    padding: 6px 16px;
    font-weight: bold;
    border-radius: 10px;
    text-align: center;
    cursor: pointer;
    box-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 20px #00FFFF;
    transition: 0.3s;
}
.neon-btn:hover { 
    color: #121212; 
    background-color: #00FFFF; 
    box-shadow: 0 0 10px #00FFFF, 0 0 20px #00FFFF, 0 0 30px #00FFFF;
}
.info-note { font-size: 10px; color: #B0B0B0; margin-bottom: 10px; }

.card-subtitle3 { 
    font-size: 13px; 
    color: #87CEFA; 
    font-style: italic;
    margin-bottom: 4px; 
}

/* Responsive untuk mobile */
@media (max-width: 768px) {
    .page-nav { flex-wrap: wrap; gap:2px; }
    .page-btn { padding: 3px 6px; font-size: 12px; }
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

# === Fungsi ambil data API ===
def ambil_halaman(page, uniq, retries=10):
    url = f"{BASE_URL}?page={page}&limit={LIMIT}&t={uniq}"

    for attempt in range(1, retries + 1):
        try:
            res = requests.get(url, timeout=12)

            if res.status_code == 200:
                data = res.json().get("data", [])
                return data if isinstance(data, list) else []

            time.sleep(0.3 * attempt)

        except:
            time.sleep(0.5 * attempt)

    return []


def ambil_data_api():
    status = st.empty()
    progress = st.progress(0)

    uniq = int(time.time())

    all_data = []
    kosong_beruntun = 0
    BATAS_KOSONG = 5  # berhenti setelah 5 halaman kosong beruntun

    page = 1

    while True:
        data = ambil_halaman(page, uniq)

        if data:
            kosong_beruntun = 0
            all_data.extend(data)
        else:
            kosong_beruntun += 1

        # status
        status.text(f"Memuat {len(all_data):,} data... Halaman {page}")
        progress.progress(min(1.0, kosong_beruntun / BATAS_KOSONG))

        # Jika 5 halaman kosong berturut-turut → data habis
        if kosong_beruntun >= BATAS_KOSONG:
            break

        page += 1

    status.text(f"✅ Total {len(all_data):,} data berhasil diambil")
    progress.progress(1.0)

    return all_data



def load_data():
    data = ambil_data_api()
    records = []
    for item in data:
        perusahaan = item.get("perusahaan", {}) or {}
        nama = perusahaan.get("nama_perusahaan", "")
        teks = f"{nama} {perusahaan.get('alamat', '')} {perusahaan.get('deskripsi_perusahaan','')}"
        jenis_pred = model.predict(vectorizer.transform([teks]))[0]
        if jenis_pred not in ["Negeri", "Swasta"]: jenis_pred = "Swasta"
        kuota = item.get("jumlah_kuota",0)
        daftar = item.get("jumlah_terdaftar",0)
        peluang = 100 if daftar==0 else min(round((kuota/daftar)*100,2),100)

                # Ambil nama kementerian (jika ada)
        kementerian = ""
        if item.get("government_agency") and isinstance(item["government_agency"], dict):
            kementerian = item["government_agency"].get("government_agency_name", "")


        # Program studi
        program_studi=""
        try:
            prog_list = json.loads(item.get("program_studi","[]"))
            if isinstance(prog_list,list):
                program_studi = ", ".join([p.get("title","").strip() for p in prog_list if p.get("title")])
        except: pass

        # Jenjang
        jenjang=""
        try:
            jenjang_list=json.loads(item.get("jenjang","[]"))
            if isinstance(jenjang_list,list):
                if all(isinstance(j,dict) for j in jenjang_list):
                    jenjang = ", ".join([j.get("title","").strip() for j in jenjang_list if j.get("title")])
                elif all(isinstance(j,str) for j in jenjang_list):
                    jenjang = ", ".join(jenjang_list)
        except: jenjang=str(item.get("jenjang",""))

        records.append({
            "Lowongan": item.get("posisi",""),
            "Instansi": nama,
            "Kementerian": kementerian,  # ✅ tambahan
            "Jenis Instansi": jenis_pred,
            "Program Studi": program_studi,
            "Jenjang": jenjang,
            "Lokasi": f"{perusahaan.get('nama_kabupaten','')}, {perusahaan.get('nama_provinsi','')}",
            "Jumlah Kuota": kuota,
            "Jumlah Pendaftar": daftar,
            "Peluang Lolos (%)": peluang,
            "Tanggal Publikasi": pd.to_datetime(item.get("created_at",None),errors="coerce")
        })

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["Lowongan","Instansi"], inplace=True)
    return df

def perlu_refresh():
    last_time = st.session_state.get("last_update_time",0)
    return (time.time()-last_time) > REFRESH_INTERVAL

# Load data
if "df" not in st.session_state or perlu_refresh():
    with st.spinner("Mengambil data terbaru..."):
        st.session_state.df = load_data()
        st.session_state.last_update_time = time.time()

df = st.session_state.df
if df.empty: st.warning("Tidak ada data ditemukan."); st.stop()

# === Filter Otomatis ===
col1,col2,col3,col4=st.columns([3,2,2,1])
with col1: 
    search = st.text_input("Kata kunci", placeholder="Cari posisi, lokasi, nama perusahaan, atau prodi")
with col2: jenis_filter = st.selectbox("Jenis Instansi",["Semua","Negeri","Swasta"])
with col3: sort_option = st.selectbox("Urut Berdasarkan",["Tidak Urut","Peluang Lolos Terbesar","Peluang Lolos Terkecil","Jumlah Kuota Terbesar","Jumlah Kuota Terkecil","Jumlah Pendaftar Terbesar","Jumlah Pendaftar Terkecil"])
with col4: st.markdown('<button class="neon-btn">Cari</button>', unsafe_allow_html=True)
st.markdown('<div class="info-note">Catatan: Filter "Jenis Instansi" masih dalam tahap pengembangan.</div>', unsafe_allow_html=True)

def apply_filter(df_in):
    filtered = df_in.copy()
    if jenis_filter!="Semua": filtered=filtered[filtered["Jenis Instansi"]==jenis_filter]
    if search.strip():
        for kw in search.split():
            mask = (
                filtered["Instansi"].str.contains(kw,case=False,na=False)|
                filtered["Lowongan"].str.contains(kw,case=False,na=False)|
                filtered["Lokasi"].str.contains(kw,case=False,na=False)|
                filtered["Program Studi"].str.contains(kw,case=False,na=False)|
                filtered["Jenjang"].str.contains(kw,case=False,na=False)
            )
            filtered=filtered[mask]
    if sort_option!="Tidak Urut":
        ascending="Terkecil" in sort_option
        if "Peluang" in sort_option: filtered=filtered.sort_values(by="Peluang Lolos (%)",ascending=ascending)
        elif "Kuota" in sort_option: filtered=filtered.sort_values(by="Jumlah Kuota",ascending=ascending)
        elif "Pendaftar" in sort_option: filtered=filtered.sort_values(by="Jumlah Pendaftar",ascending=ascending)
    return filtered

filtered_df = apply_filter(df)

# === Pagination Setup ===
if "page_num" not in st.session_state: st.session_state.page_num=1
total_items=len(filtered_df)
total_pages=max(1,math.ceil(total_items/ITEMS_PER_PAGE))
start_idx=(st.session_state.page_num-1)*ITEMS_PER_PAGE
end_idx=start_idx+ITEMS_PER_PAGE
df_page=filtered_df.iloc[start_idx:end_idx]

def get_peluang_class(p):
    return "peluang-high" if p>=75 else "peluang-medium" if p>=50 else "peluang-low" if p>=25 else "peluang-verylow"

# === Tampilkan card 3 kolom ===
columns = st.columns(3)
for idx,row in df_page.iterrows():
    col = columns[idx%3]
    col.markdown(f"""
    <div class="card">
        <div class="card-title">{row['Lowongan']}</div>
        <div class="card-subtitle">{row['Instansi']}</div>
        <div class="card-subtitle3">{row["Kementerian"]}
        <div class="card-subtitle2">{row['Lokasi']}</div>
        <div class="card-detail"><b>Program Studi:</b> {row['Program Studi'] or '-'}</div>
        <div class="card-detail"><b>Jenjang:</b> {row['Jenjang'] or '-'}</div>
        <div class="card-detail"><b>Jumlah Kuota:</b> {row['Jumlah Kuota']}, Pendaftar: {row['Jumlah Pendaftar']}</div>
        <div class="card-detail"><b>Peluang Lolos:</b> <span class="{get_peluang_class(row['Peluang Lolos (%)'])}">{row['Peluang Lolos (%)']}%</span></div>
        <div class="card-detail"><b>Tanggal Publikasi:</b> {row['Tanggal Publikasi'].strftime("%d %b %Y %H:%M")}</div>
    </div>
    """, unsafe_allow_html=True)


# === Pagination ===
if "page_num" not in st.session_state:
    st.session_state.page_num = 1

total_pages = max(1, math.ceil(total_items / ITEMS_PER_PAGE))

# Fungsi render tombol pagination dengan "..."
def render_pagination(current_page, total_pages, delta=2):
    pages = []
    left = max(1, current_page - delta)
    right = min(total_pages, current_page + delta)
    for i in range(1, total_pages + 1):
        if i == 1 or i == total_pages or left <= i <= right:
            pages.append(i)
        elif pages[-1] != "...":
            pages.append("...")
    return pages

# Tampilkan pagination horizontal
def show_pagination_horizontal():
    pages = render_pagination(st.session_state.page_num, total_pages)
    
    cols = st.columns(len(pages) + 2)  # +2 untuk Prev dan Next

    # Prev button
    if cols[0].button("⏪ Prev") and st.session_state.page_num > 1:
        st.session_state.page_num -= 1

    # Page buttons horizontal
    for idx, p in enumerate(pages):
        if p == "...":
            cols[idx+1].markdown("...")
        else:
            label = f"**{p}**" if p == st.session_state.page_num else str(p)
            if cols[idx+1].button(label):
                st.session_state.page_num = p

    # Next button
    if cols[-1].button("Next ⏩") and st.session_state.page_num < total_pages:
        st.session_state.page_num += 1

# CSS tombol neon horizontal
st.markdown("""
<style>
div.stButton > button {
    background-color: #121212;
    color: #00FFFF;
    border: 1px solid #00FFFF;
    padding: 6px 12px;
    border-radius: 6px;
    font-weight: bold;
    cursor: pointer;
    min-width: 50px;
}
div.stButton > button:hover {
    background-color: #00FFFF;
    color: #121212;
}
</style>
""", unsafe_allow_html=True)

# Tampilkan pagination horizontal
show_pagination_horizontal()
st.write(f"Halaman aktif: {st.session_state.page_num} dari {total_pages}")
# Tombol download CSV
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download CSV",data=csv,file_name=f"lowongan_maganghub_{int(time.time())}.csv",mime="text/csv")
