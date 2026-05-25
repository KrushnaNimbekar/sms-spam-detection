import streamlit as st
import joblib
import re
import string
from nltk.stem.porter import PorterStemmer

# =========================
# CACHE MODEL LOADING (IMPORTANT FOR SPEED)
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# =========================
# STEMMER
# =========================
ps = PorterStemmer()

# =========================
# CLEAN TEXT FUNCTION
# =========================
def clean_text(text):

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'\d+', '', text)

    words = text.split()

    words = [ps.stem(word) for word in words]

    return " ".join(words)

# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("📩 SMS Spam Detection System")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# =========================
# INPUT
# =========================
message = st.text_area("Enter SMS Message", height=150)

# =========================
# PREDICT BUTTON
# =========================
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message first.")

    else:
        cleaned_message = clean_text(message)

        vector_input = vectorizer.transform([cleaned_message])

        result = model.predict(vector_input)[0]

        # =========================
        # OUTPUT
        # =========================
        if result == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")
