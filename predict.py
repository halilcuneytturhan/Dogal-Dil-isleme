import joblib
import re


# =========================
# 1. Kaydedilen model ve TF-IDF dosyasını yükle
# =========================

model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# =========================
# 2. Metin temizleme fonksiyonu
# =========================

def clean_text(text):
    text = text.lower()
    text = text.strip()

    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text)

    # Gereksiz karakterleri temizle
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s.,?!:;%-]", "", text)

    return text


# =========================
# 3. Tahmin fonksiyonu
# =========================

def predict_intent(text):
    cleaned_text = clean_text(text)

    # Metni TF-IDF formatına dönüştür
    text_vector = vectorizer.transform([cleaned_text])

    # Model ile tahmin yap
    prediction = model.predict(text_vector)[0]

    # Olasılık değerlerini al
    probabilities = model.predict_proba(text_vector)[0]

    # En yüksek olasılık
    confidence = max(probabilities)

    return prediction, confidence


# =========================
# 4. Kullanıcıdan metin al
# =========================

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