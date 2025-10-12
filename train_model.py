import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib, seaborn as sns, matplotlib.pyplot as plt

df = pd.read_csv("dataset_manual.csv")
df["text"] = df["nama"].fillna('') + " " + df["alamat"].fillna('') + " " + df["deskripsi"].fillna('')

print(df["kategori"].value_counts())  # pastikan seimbang

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["kategori"], test_size=0.2, random_state=42, stratify=df["kategori"]
)

vectorizer = TfidfVectorizer(max_features=1500)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=["Negeri", "Swasta"])
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Negeri","Swasta"], yticklabels=["Negeri","Swasta"], cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Sebenarnya")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, "model_maganghub.pkl")
joblib.dump(vectorizer, "vectorizer_maganghub.pkl")
print("✅ Model dan vectorizer berhasil disimpan!")
