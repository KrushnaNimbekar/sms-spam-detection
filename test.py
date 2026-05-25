import joblib

# load model + vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# test messages
messages = [
    "Congratulations! You won a free iPhone",
    "Hey, what are you doing today?",
    "URGENT! Bank account blocked",
    "Let's meet tomorrow"
]

for msg in messages:
    vec = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]

    if prediction == 1:
        print("SPAM 🚨 ->", msg)
    else:
        print("HAM ✅ ->", msg)