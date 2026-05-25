from flask import Flask, render_template, request
import joblib
import re
import string
from nltk.stem.porter import PorterStemmer

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# LOAD MODEL + VECTORIZER
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# STEMMER
# =========================
ps = PorterStemmer()

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove links
    text = re.sub(r'http\S+', '', text)

    # remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # split words
    words = text.split()

    # stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

# =========================
# HOME PAGE
# =========================
@app.route('/')
def home():
    return render_template("index.html")

# =========================
# PREDICTION ROUTE
# =========================
@app.route('/predict', methods=['POST'])
def predict():

    # get message from textarea
    message = request.form['message']

    # clean message
    cleaned_message = clean_text(message)

    # vectorize
    vector_input = vectorizer.transform([cleaned_message])

    # predict
    result = model.predict(vector_input)[0]

    # output
    if result == 1:
        prediction = "🚨 SPAM MESSAGE"
    else:
        prediction = "✅ NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        user_message=message
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)