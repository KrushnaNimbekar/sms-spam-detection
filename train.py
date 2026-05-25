import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from nltk.stem.porter import PorterStemmer
import nltk

# download nltk
nltk.download('punkt')

# =========================
# STEMMER
# =========================
ps = PorterStemmer()

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/spam.csv", encoding="latin-1")

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# =========================
# LABEL ENCODING
# ham = 0
# spam = 1
# =========================
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove links
    text = re.sub(r'http\\S+', '', text)

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove numbers
    text = re.sub(r'\\d+', '', text)

    # tokenize
    words = text.split()

    # stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

# apply cleaning
df['message'] = df['message'].apply(clean_text)

# =========================
# FEATURES + LABELS
# =========================
X = df['message']
y = df['label']

# =========================
# TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\\nAccuracy:", round(accuracy * 100, 2), "%")

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, "model.pkl")


print("Model saved successfully")
