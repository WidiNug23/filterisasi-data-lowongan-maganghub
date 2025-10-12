import requests
import pandas as pd
import re
import time

BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100
MAX_PAGE = 200

def ambil_data_api():
    all_data = []
    for page in range(1, MAX_PAGE + 1):
        url = f"{BASE_URL}?page={page}&limit={LIMIT}"
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"⚠️ Gagal ambil data halaman {page}: {e}")
            break
        data = r.json().get("data", [])
        if not data:
            break
        all_data.extend(data)
        print(f"✅ Halaman {page}: {len(data)} data diambil")
        time.sleep(0.1)
    return all_data

def label_otomatis(nama, alamat, deskripsi):
    text = f"{nama} {alamat} {deskripsi}".lower()
    if re.search(r"\b(persero|kementerian|dinas|badan|lembaga|sekretariat|pemerintah|provinsi|kabupaten|universitas negeri|politeknik negeri|bumn|bank rakyat indonesia|bank negara indonesia|pt\s+kereta api|pertamina|pln|telkom)\b", text):
        return "Negeri"
    return "Swasta"

print("🚀 Mengambil data dari API...")
data = ambil_data_api()

records = []
for item in data:
    perusahaan = item.get("perusahaan", {})
    nama = perusahaan.get("nama_perusahaan", "")
    alamat = perusahaan.get("alamat", "")
    desk = perusahaan.get("deskripsi_perusahaan", "")
    kategori = label_otomatis(nama, alamat, desk)

    records.append({
        "nama": nama,
        "alamat": alamat,
        "deskripsi": desk,
        "kategori": kategori
    })

df = pd.DataFrame(records).drop_duplicates(subset=["nama", "alamat"])
df.to_csv("dataset_manual.csv", index=False, encoding="utf-8-sig")
print(f"✅ Dataset disimpan ({len(df)} baris)")
print(df["kategori"].value_counts())
