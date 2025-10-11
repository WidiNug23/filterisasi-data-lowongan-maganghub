import requests
import pandas as pd
import re
import time

# === Konfigurasi API ===
BASE_URL = "https://maganghub.kemnaker.go.id/be/v1/api/list/vacancies-aktif"
LIMIT = 100
MAX_PAGE = 20

def ambil_data_api():
    all_data = []
    for page in range(1, MAX_PAGE + 1):
        url = f"{BASE_URL}?page={page}&limit={LIMIT}"
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Gagal ambil data halaman {page}: {e}")
            break
        json_data = response.json()
        data = json_data.get("data", [])
        if not data:
            break
        all_data.extend(data)
        print(f"✅ Halaman {page}: {len(data)} data diambil")
        time.sleep(0.3)
    return all_data

# === Fungsi label otomatis diperbarui ===
def label_otomatis(nama, alamat, deskripsi):
    text = f"{nama} {alamat} {deskripsi}".lower()

    # Negeri / Pemerintah
    if re.search(r"\b(kementerian|dinas|badan|lembaga|sekretariat|pemerintah|provinsi|kabupaten|universitas negeri|politeknik negeri|bank rakyat indonesia|bank negara indonesia)\b", text):
        return "Negeri"

    # Swasta Besar
    elif re.search(r"\b(pt|cv)\b", text) and re.search(r"\b(alfa|astra|indofood|unilever|mustika|midi|bank|finance|group|holding|retail|industri|corporate|international|mining|energy|technology|chemical|aeon|indonesia)\b", text):
        return "Swasta Besar"

    # Swasta Kecil
    elif re.search(r"\b(pt|cv)\b", text):
        return "Swasta Kecil"

    # Default ke Swasta Kecil jika tidak cocok
    else:
        return "Swasta Kecil"

# === Ambil data dan buat dataset ===
print("🚀 Mengambil data dari API...")
data = ambil_data_api()

records = []
for item in data:
    perusahaan = item.get("perusahaan", {})
    nama = perusahaan.get("nama_perusahaan", "")
    alamat = perusahaan.get("alamat", "")
    deskripsi = perusahaan.get("deskripsi_perusahaan", "")
    kategori = label_otomatis(nama, alamat, deskripsi)

    records.append({
        "nama": nama,
        "alamat": alamat,
        "deskripsi": deskripsi,
        "kategori": kategori
    })

# === Simpan ke CSV ===
df = pd.DataFrame(records)
df.drop_duplicates(subset=["nama", "alamat"], inplace=True)
df.to_csv("dataset_manual.csv", index=False, encoding="utf-8-sig")

print(f"✅ Dataset awal disimpan sebagai dataset_manual.csv ({len(df)} baris)")
