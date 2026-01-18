import joblib

def load_models():
    tfidf = joblib.load("tfidf.pkl")
    lda = joblib.load("lda_model.pkl")
    nb = joblib.load("nb_model.pkl")
    return tfidf, lda, nb
