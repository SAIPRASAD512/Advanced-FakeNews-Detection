
# 📰 Fake News Detection (ML + Deep Learning + BERT)

An advanced **Fake News Detection System** using:
- ✅ Logistic Regression, SVM, Random Forest, XGBoost
- ✅ LSTM (Deep Learning with Keras)
- ✅ BERT (Transformers - HuggingFace)

---

## 🚀 Features
- Text Preprocessing (stopwords removal, stemming, TF-IDF)
- Machine Learning Models (LR, SVM, RF, XGBoost)
- Deep Learning (LSTM with embeddings)
- Transformer (BERT for state-of-the-art NLP)
- Unified Prediction Function (`xgboost`, `lstm`, or `bert`)

---

## 📂 Project Structure

```
fake-news-detection/
│── data/                 # dataset (train.csv)
│── src/                  # all source code
│── notebooks/            # optional EDA/experiments
│── main.py               # entry point
│── requirements.txt      # dependencies
│── README.md             # documentation
```

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

---

## ▶️ Usage
### Train & Evaluate Models
```bash
python main.py
```

### Predict on New Text
```python
from src.predict import predict_news

text = "Breaking: Government announces new healthcare scheme."
print(predict_news(text, model="bert"))
```

---

## 📊 Results
| Model              | Accuracy |
|--------------------|----------|
| Logistic Regression | ~92%    |
| SVM                | ~93%    |
| Random Forest      | ~91%    |
| **XGBoost**        | ~94%    |
| **LSTM**           | ~95%    |
| **BERT**           | ~97%    |

---

## 📦 Requirements
- Python 3.8+
- scikit-learn
- xgboost
- tensorflow
- torch
- transformers
- nltk
- pandas, numpy
