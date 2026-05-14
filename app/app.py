import streamlit as st
import joblib
import re
from pathlib import Path


st.set_page_config(
    page_title="House MD Intent Sınıflandırma",
    page_icon="🩺",
    layout="centered"
)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "intent_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


model, vectorizer = load_model()


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

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)

    return prediction, confidence, cleaned_text


st.title("🩺 HouseMD-Intent: Medikal Diyalog Niyeti Sınıflandırma Sistemi")

st.markdown(
    """
    **HouseMD-Intent**, House MD veri seti kullanılarak geliştirilen bir doğal dil işleme
    uygulamasıdır. Bu sistem, kullanıcı tarafından girilen medikal içerikli bir diyaloğu
    analiz ederek cümlenin hangi konuşma niyetine ait olduğunu tahmin eder.

    Model, girilen metni önce **TF-IDF** yöntemiyle sayısal forma dönüştürür ve ardından
    eğitilmiş sınıflandırma modeli ile intent tahmini yapar.
    """
)

st.info(
    """
    Bu uygulama; açıklama, hipotez, soru, talimat, tanı, tedavi, değerlendirme ve
    test/prosedür sınıfları arasında tahmin yapmaktadır.
    """
)



st.divider()


st.subheader("Analiz edilecek cümle giriniz")

user_text = st.text_area(
    label="",
    placeholder="Veri setinden örnek: Prednizon bağışıklık sistemini baskılar. Sahip olmadığı hastalık için verdiğim ilaç bu değil mi?.",
    height=120
)

predict_button = st.button("Analiz")


if predict_button:
    if user_text.strip() == "":
        st.warning("Lütfen bir cümle giriniz.")
    else:
        prediction, confidence, cleaned_text = predict_intent(user_text)

        st.divider()

        st.subheader("Tahmin Sonucu")

        st.success(f"Tahmin edilen intent: **{prediction}**")

        if confidence is not None:
            st.info(f"Güven skoru: **{confidence * 100:.2f}%**")

            if confidence < 0.40:
                st.warning("Model bu tahminden çok emin değil.")
        else:
            st.info("Bu model güven skoru üretmemektedir.")

        st.divider()


st.sidebar.title("Intent Sınıfları")

st.sidebar.write(
    """
    - **Açıklama** 
    - **Hipotez** 
    - **Soru** 
    - **Talimat** 
    - **Tanı** 
    - **Tedavi** 
    - **Değerlendirme**  
    - **Test / Prosedür** 
    """
)

st.sidebar.divider()

st.sidebar.write(
    """
    **Model:** TF-IDF + Calibrated SVM  
    **Veri:** Kendi oluşturduğumuz HouseMD Dataset
    """
)