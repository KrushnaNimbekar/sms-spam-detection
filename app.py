import streamlit as st
import joblib
import re
import string
from nltk.stem.porter import PorterStemmer

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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
# UI DESIGN
# =========================
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS Spam Detection System")

st.write("Enter a message and check if it is SPAM or NOT")

# input box
message = st.text_area("Enter SMS Message")

# button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message")
    else:
        cleaned = clean_text(message)

        vector_input = vectorizer.transform([cleaned])

        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")