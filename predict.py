import re

import joblib


model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,?!:;%-]", "", text)
    text = text.replace("_", "")
    return text


def predict_intent(text):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)

    return prediction, confidence


print("House MD intent tahmin sistemi")
print("Cikmak icin 'q' yazabilirsin.")
print("-" * 50)

while True:
    user_text = input("\nBir cumle gir: ")

    if user_text.lower() == "q":
        print("Program kapatildi.")
        break

    prediction, confidence = predict_intent(user_text)

    print("\nTahmin edilen intent:", prediction)

    if confidence is not None:
        print(f"Guven skoru: {confidence * 100:.2f}%")

        if confidence < 0.50:
            print("Uyari: Model bu tahminden cok emin degil.")
    else:
        print(f"Guven skoru: {confidence * 100:.2f}%")
        
