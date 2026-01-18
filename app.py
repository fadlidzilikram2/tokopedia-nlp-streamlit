import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessing import preprocess_text
from model_utils import load_models

st.set_page_config(page_title="Analisis Tokopedia", layout="wide")
st.title("📊 Analisis Topik & Sentimen Tokopedia")

tfidf, nb = load_models()

uploaded_file = st.file_uploader("Upload CSV (kolom: review)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    df["review"] = df["review"].fillna("")
    df["clean_review"] = df["review"].astype(str).apply(preprocess_text)

    X = tfidf.transform(df["clean_review"])
    df["sentiment_pred"] = nb.predict(X)

    st.subheader("Hasil Sentimen")
    st.dataframe(df[["review", "sentiment_pred"]].head())
