# src/preprocessing.py

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    words = word_tokenize(text.lower()) 
    words = [word for word in words if word not in stopwords.words('english')]  # Usuwanie stop words
    return ' '.join(words)
