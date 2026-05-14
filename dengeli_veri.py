import pandas as pd
import os

input_path = "data/cleaned_intent_dataset.csv"
output_csv_path = "data/balanced_intent_dataset.csv"
output_excel_path = "data/balanced_intent_dataset.xlsx"

df = pd.read_csv(input_path)

print("Orijinal temiz dataset boyutu:", df.shape)

print("\nOrijinal sınıf dağılımı:")
print(df["label"].value_counts())


min_class_count = df["label"].value_counts().min()

print("\nEn az veriye sahip sınıftaki veri sayısı:", min_class_count)

balanced_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min_class_count, random_state=42))
)


balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDengelenmiş dataset boyutu:", balanced_df.shape)

print("\nDengelenmiş sınıf dağılımı:")
print(balanced_df["label"].value_counts())

print("\nİlk 10 satır:")
print(balanced_df.head(10))

os.makedirs("data", exist_ok=True)

balanced_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
balanced_df.to_excel(output_excel_path, index=False)

print("\nDengelenmiş dosyalar kaydedildi:")
print(output_csv_path)
print(output_excel_path)