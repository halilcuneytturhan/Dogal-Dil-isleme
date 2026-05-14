import pandas as pd
import re
import os

input_path = r"C:\Users\cavus\Desktop\Doğal Dil İşleme\data\Last_HouseMD_DataSet.xlsx"
output_csv_path = "data/cleaned_intent_dataset.csv"
output_excel_path = "data/cleaned_intent_dataset.xlsx"

df = pd.read_excel(input_path)

print("Orijinal Dataset boyutu = ", df.shape)
print("Sütunlar = ", df.columns.to_list())

df = df[["text", "Intent"]].copy()

df.rename(columns={"Intent": "label"}, inplace=True)

df.dropna(subset=["text","label"], inplace=True)



df['text'] = df["text"].astype(str)
df['label'] = df["label"].astype(str)

def temizlik(text):
    text = text.lower()
    text = text.strip()

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZğüşöçıİĞÜŞÖÇ0-9\s.,?!:;%-]", "", text)

    return text

df["text"] = df["text"].apply(temizlik)

def temiz_label(label):
    label = label.lower()
    label = label.strip()
    
    replacements = {
    "açıklama": "açıklama",
    "aciklama": "açıklama",
    "açiklama": "açıklama",
    "durum bilgilendirmesi": "açıklama",
    "bilgilendirme": "açıklama",
    "gözlem": "açıklama",

    "hipotez": "hipotez",
    "olasılık": "hipotez",
    "varsayım": "hipotez",

    "soru": "soru",
    "sorgulama": "soru",

    "talimat": "talimat",
    "emir": "talimat",
    "yönlendirme": "talimat",

    "tanı": "tanı",
    "tani": "tanı",
    "teşhis": "tanı",
    "teshis": "tanı",

    "tedavi": "tedavi",

    "değerlendirme": "değerlendirme",
    "degerlendirme": "değerlendirme",

    "test": "test/prosedür",
    "prosedür": "test/prosedür",
    "prosedur": "test/prosedür",
    "test/prosedür": "test/prosedür",
    "test/prosedur": "test/prosedür",
    }
    
    return replacements.get(label, label)

df["label"] = df["label"].apply(temiz_label)

df = df[df["text"].str.len() > 2]
df = df[df["label"].str.len() > 1]

valid_labels = [
    "açıklama",
    "hipotez",
    "soru",
    "talimat",
    "tanı",
    "tedavi",
    "test/prosedür",
    "değerlendirme"
]

df = df[df["label"].isin(valid_labels)]

df.drop_duplicates(subset=["text", "label"], inplace=True)

print("\nTemizlenmiş dataset boyutu:", df.shape)
print("\nSınıf dağılımı:")
print(df["label"].value_counts())

os.makedirs("data", exist_ok=True)
df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
df.to_excel(output_excel_path, index=False)

print("\nTemizlenmiş dosyalar kaydedildi:")
print(output_csv_path)
print(output_excel_path)