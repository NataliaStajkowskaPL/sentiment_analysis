# src/vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)

def vectorize_text(text_data):
    vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer.fit_transform(text_data)
