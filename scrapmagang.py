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
LIMIT = 1000
MAKS_HALAMAN = 200
MAKS_WORKER = 100
REFRESH_INTERVAL = 300
ITEMS_PER_PAGE = 18  # agar 3 kolom pas

# === CSS Modern & Neon ===
st.markdown("""
<style>
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
def ambil_halaman(page, uniq, retries=3):
    url = f"{BASE_URL}?page={page}&limit={LIMIT}&t={uniq}"
    for _ in range(retries):
        try:
            res = requests.get(url, timeout=15)
            if res.status_code == 200: return res.json().get("data", [])
        except: time.sleep(1)
    return []

def ambil_data_api():
    all_data = []
    status = st.empty()
    progress = st.progress(0)
    uniq = int(time.time())
    total_estimasi = MAKS_HALAMAN * LIMIT

    for batch_start in range(1, MAKS_HALAMAN + 1, MAKS_WORKER):
        pages = list(range(batch_start, min(batch_start + MAKS_WORKER, MAKS_HALAMAN + 1)))
        with ThreadPoolExecutor(max_workers=MAKS_WORKER) as executor:
            futures = {executor.submit(ambil_halaman, p, uniq): p for p in pages}
            for future in as_completed(futures):
                page = futures[future]
                try:
                    data = future.result()
                    if data: all_data.extend(data)
                except: pass
                progress.progress(min(len(all_data)/total_estimasi, 1.0))
                status.text(f"Memuat {len(all_data):,} data. Sabar ya, Kak.")
    status.text(f"✅ Total {len(all_data):,} lowongan")
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
        <div class="card-subtitle2">{row['Lokasi']}</div>
        <div class="card-detail"><b>Program Studi:</b> {row['Program Studi'] or '-'}</div>
        <div class="card-detail"><b>Jenjang:</b> {row['Jenjang'] or '-'}</div>
        <div class="card-detail"><b>Jumlah Kuota:</b> {row['Jumlah Kuota']}, Pendaftar: {row['Jumlah Pendaftar']}</div>
        <div class="card-detail"><b>Peluang Lolos:</b> <span class="{get_peluang_class(row['Peluang Lolos (%)'])}">{row['Peluang Lolos (%)']}%</span></div>
        <div class="card-detail"><b>Tanggal Publikasi:</b> {row['Tanggal Publikasi'].strftime("%d %b %Y %H:%M")}</div>
    </div>
    """, unsafe_allow_html=True)

# Fungsi pagination "..."
def render_pagination(page_num, total_pages, delta=2):
    pages = []
    left = page_num - delta
    right = page_num + delta
    for i in range(1, total_pages + 1):
        if i == 1 or i == total_pages or (left <= i <= right):
            pages.append(i)
        elif pages[-1] != '...':
            pages.append('...')
    return pages

# Callback untuk tombol halaman
def go_to_page(p):
    st.session_state.page_num = p

# Callback Prev / Next
def prev_page():
    if st.session_state.page_num > 1:
        st.session_state.page_num -= 1

def next_page():
    if st.session_state.page_num < total_pages:
        st.session_state.page_num += 1

# Navigasi horizontal
pages_to_render = render_pagination(st.session_state.page_num, total_pages)
cols = st.columns(len(pages_to_render) + 2)  # +2 untuk Prev/Next

# Tombol Prev
cols[0].button("⏪", on_click=prev_page)

# Tombol halaman
for idx, p in enumerate(pages_to_render):
    if p == "...":
        cols[idx + 1].markdown("<b>...</b>", unsafe_allow_html=True)
    else:
        cols[idx + 1].button(str(p), on_click=go_to_page, args=(p,))

# Tombol Next
cols[-1].button("⏩", on_click=next_page)


# === Tombol download CSV ===
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Download CSV",data=csv,file_name=f"lowongan_maganghub_{int(time.time())}.csv",mime="text/csv")
