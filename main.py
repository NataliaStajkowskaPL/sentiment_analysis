from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from src.model import train_model, evaluate_model
from src.preprocessing import preprocess_text
from src.vectorizer import vectorize_text
import pandas as pd
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def load_data(data_path):
    texts = []
    labels = []
    for sentiment in ["pos", "neg"]:
        sentiment_path = os.path.join(data_path, sentiment)
        for filename in os.listdir(sentiment_path):
            file_path = os.path.join(sentiment_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                texts.append(file.read())
                labels.append(1 if sentiment == "pos" else 0)
    return pd.DataFrame({"text": texts, "label": labels})

def vectorize_text(text_data):
    vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer.fit_transform(text_data)

if __name__ == '__main__':
    data = load_data('data/aclImdb/train')  
    data['text'] = data['text'].apply(preprocess_text)  


    data['sentiment_score'] = data['label']  
    data['sentiment_label'] = data['sentiment_score'].apply(lambda x: 'positive' if x == 1 else 'negative')

    X = vectorize_text(data['text'])
    y = data['sentiment_score']

    print(data['sentiment_label'].value_counts())

    print(data[['text', 'sentiment_score', 'sentiment_label']].head())

    print(data[['text', 'label']].sample(5))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(y_train.value_counts())
    print(y_test.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
