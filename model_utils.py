import joblib

def load_models():
    tfidf = joblib.load("tfidf.pkl")
    nb = joblib.load("nb_model.pkl")
    return tfidf, nb
