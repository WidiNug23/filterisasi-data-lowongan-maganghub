import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Baca dataset manual ===
# File ini harus punya kolom: nama, alamat, deskripsi, kategori
df = pd.read_csv("dataset_manual.csv")

# Gabungkan teks jadi satu kolom untuk dilatih
df["text"] = df["nama"].fillna('') + " " + df["alamat"].fillna('') + " " + df["deskripsi"].fillna('')

# === 2. Pisahkan fitur & label ===
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["kategori"], test_size=0.2, random_state=42, stratify=df["kategori"])

# === 3. Vectorisasi teks ===
vectorizer = TfidfVectorizer(max_features=1500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 4. Latih model ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# === 5. Evaluasi ===
y_pred = model.predict(X_test_vec)
print("=== Laporan Evaluasi ===")
print(classification_report(y_test, y_pred, zero_division=0))  # zero_division=0 mencegah warning

# === Tambahkan confusion matrix ===
cm = confusion_matrix(y_test, y_pred, labels=df["kategori"].unique())
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=df["kategori"].unique(), yticklabels=df["kategori"].unique(), cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Sebenarnya")
plt.title("Confusion Matrix")
plt.show()

# === 6. Simpan model & vectorizer ===
joblib.dump(model, "model_maganghub.pkl")
joblib.dump(vectorizer, "vectorizer_maganghub.pkl")

print("✅ Model dan vectorizer berhasil disimpan!")
