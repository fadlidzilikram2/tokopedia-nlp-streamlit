import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import preprocess_text

# Load data
df = pd.read_csv("ulasan_tokopedia.csv")

# Pastikan kolom ada dan bersih
df = df.dropna(subset=["review", "sentiment"])

# Preprocessing
df["clean_review"] = df["review"].astype(str).apply(preprocess_text)

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"], df["sentiment"], test_size=0.2, random_state=42
)

# Vectorize teks dengan TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Latih model Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Evaluasi performa
y_pred = nb.predict(X_test_tfidf)
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, average="weighted", zero_division=0))
print("F1-Score :", f1_score(y_test, y_pred, average="weighted", zero_division=0))

# Simpan model ke file .pkl
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(nb, "nb_model.pkl")

print("Model sudah tersimpan: tfidf.pkl dan nb_model.pkl")
