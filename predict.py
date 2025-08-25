from src.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertForSequenceClassification, pipeline

# Assume models are already trained & loaded
vectorizer = None
ml_models = {}
lstm_model = None
bert_pipeline = None
tokenizer = None
max_len = 300

def predict_news(text, model="bert"):
    if model == "bert":
        return bert_pipeline(text)[0]["label"]

    elif model == "lstm":
        seq = tokenizer.texts_to_sequences([clean_text(text)])
        pad = pad_sequences(seq, maxlen=max_len)
        pred = (lstm_model.predict(pad) > 0.5).astype(int)[0][0]
        return "Fake News" if pred == 1 else "Real News"

    else:  # ML (XGBoost / LR / SVM)
        vec = vectorizer.transform([clean_text(text)])
        pred = ml_models["XGBoost"].predict(vec)[0]
        return "Fake News" if pred == 1 else "Real News"
