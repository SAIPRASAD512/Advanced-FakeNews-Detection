import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import pipeline

# ----------------------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------------------
df = pd.read_csv("train.csv")  # Kaggle dataset
df = df.dropna()

df["content"] = df["author"].fillna('') + " " + df["title"].fillna('')

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

df["clean_content"] = df["content"].apply(clean_text)

# ----------------------------------------------------------------------
# 3. Feature Engineering (TF-IDF for ML models)
# ----------------------------------------------------------------------
X = df["clean_content"].values
y = df["label"].values

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------------------
# 4. ML Models (Baseline + Advanced)
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

print("\n=== ML Models Evaluation ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

# ----------------------------------------------------------------------
# 5. Deep Learning (LSTM)
# ----------------------------------------------------------------------
print("\n=== Training LSTM Model ===")

max_words = 10000
max_len = 300

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["clean_content"])

X_seq = tokenizer.texts_to_sequences(df["clean_content"])
X_pad = pad_sequences(X_seq, maxlen=max_len)

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(
    X_pad, y, test_size=0.2, stratify=y, random_state=42
)

lstm_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_dl, y_train_dl, validation_split=0.2, epochs=3, batch_size=64)

lstm_eval = lstm_model.evaluate(X_test_dl, y_test_dl)
print("\nLSTM Accuracy:", lstm_eval[1])

# ----------------------------------------------------------------------
# 6. Transformer (BERT)
# ----------------------------------------------------------------------
print("\n=== Using BERT for Fake News Detection ===")

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# HuggingFace pipeline for inference
bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

# Example Prediction
example_text = "Breaking: Stock market crashes due to inflation concerns."
print("\nBERT Prediction →", bert_pipeline(example_text)[0])

# ----------------------------------------------------------------------
# 7. Final Prediction Function
# ----------------------------------------------------------------------
def predict_news(text, model="bert"):
    if model == "bert":
        return bert_pipeline(text)[0]["label"]
    elif model == "lstm":
        seq = tokenizer.texts_to_sequences([clean_text(text)])
        pad = pad_sequences(seq, maxlen=max_len)
        pred = (lstm_model.predict(pad) > 0.5).astype(int)[0][0]
        return "Fake News" if pred == 1 else "Real News"
    else:
        vec = vectorizer.transform([clean_text(text)])
        pred = models["XGBoost"].predict(vec)[0]
        return "Fake News" if pred == 1 else "Real News"


if __name__ == "__main__":
    sample = "Government announces new healthcare scheme."
    print("\nSample Prediction:", predict_news(sample, model="xgboost"))
