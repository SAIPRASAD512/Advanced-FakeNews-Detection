import pandas as pd
import numpy as np
import re
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------------------
df = pd.read_csv("train.csv")  # dataset: https://www.kaggle.com/c/fake-news/data
df = df.dropna()  # drop missing values

# Combine title + author as input text
df["content"] = df["author"].fillna('') + " " + df["title"].fillna('')

# ----------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Remove non-alphabetic chars
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    # Remove stopwords + stemming
    text = [ps.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

df["clean_content"] = df["content"].apply(clean_text)

# ----------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------
X = df["clean_content"].values
y = df["label"].values  # 0 = real, 1 = fake

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------------------------------------------------------------
# 4. Model Training
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="linear"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))

# ----------------------------------------------------------------------
# 5. Prediction Function
# ----------------------------------------------------------------------
def predict_fake_or_real(text, model=models["SVM"]):  # default = SVM (best)
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return "Fake News" if pred == 1 else "Real News"


# Example Usage
if __name__ == "__main__":
    sample = "Breaking news: The stock market is crashing due to unexpected inflation rise."
    print("\nSample Prediction →", predict_fake_or_real(sample))
