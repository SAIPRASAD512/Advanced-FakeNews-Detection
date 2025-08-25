from src.train_ml import train_ml_models
from src.train_lstm import train_lstm_model
from src.train_bert import load_bert_model

if __name__ == "__main__":
    print("🚀 Training ML Models...")
    train_ml_models()

    print("\n🚀 Training LSTM Model...")
    train_lstm_model()

    print("\n🚀 Loading BERT Model...")
    load_bert_model()

    print("\n✅ All models ready! Use src/predict.py to test new news articles.")
