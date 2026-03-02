import pandas as pd
import string
import joblib
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("stopwords")

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Prepare stopwords
stop_words = set(stopwords.words("english"))

# Clean text
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

X = df["message"]
y = df["label"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 Improved TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),     # Unigram + Bigram
    max_df=0.9,            # Remove very common words
    min_df=2,              # Remove rare words
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔥 Slight smoothing tuning
model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("🔥 Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ High-Accuracy Model Saved")