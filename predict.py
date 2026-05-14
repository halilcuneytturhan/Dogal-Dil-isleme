import joblib
import re



model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s.,?!:;%-]", "", text)

    return text



def predict_intent(text):
    cleaned_text = clean_text(text)

    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    confidence = max(probabilities)

    return prediction, confidence



print("House MD Konuşma Niyeti Tahmin Sistemi")
print("Çıkmak için 'q' yazabilirsin.")
print("-" * 50)

while True:
    user_text = input("\nBir cümle gir: ")

    if user_text.lower() == "q":
        print("Program kapatıldı.")
        break

    prediction, confidence = predict_intent(user_text)

    print("\nTahmin edilen intent:", prediction)
    print(f"Güven skoru: {confidence * 100:.2f}%")

    if confidence < 0.50:
        print("Uyarı: Model bu tahminden çok emin değil.")