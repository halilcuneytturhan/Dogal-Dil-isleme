# Doğal Dil İşleme - Medikal Intent Sınıflandırma

Bu proje, medikal metinlerdeki cümlelerin hangi niyete/intent'e ait olduğunu sınıflandırmak için geliştirilmiş bir doğal dil işleme uygulamasıdır. Model, TF-IDF özellik çıkarımı ve LinearSVC tabanlı sınıflandırıcı kullanarak metinleri farklı intent sınıflarına ayırır.

## Sınıflar
Model aşağıdaki intent sınıflarını tahmin eder:

- açıklama
- hipotez
- tanı
- soru
- talimat
- tedavi
- değerlendirme
- test/prosedür

## Kullanılan Teknolojiler

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- TF-IDF
- LinearSVC


## Model Eğitimi
Modeli yeniden eğitmek için proje ana dizininde şu komutu çalıştırın:


Eğitim sırasında:

- Veri seti okunur.
- Sınıf dağılımı görüntülenir.
- Tüm temiz veri eğitim akışında kullanılır.
- Word ve character n-gram TF-IDF ile metinler sayısallaştırılır.
- Az örnekli sınıflara ek ağırlık veren LinearSVC tabanlı model eğitilir.
- Accuracy, balanced accuracy, macro F1, classification report ve confusion matrix üretilir.
- Model dosyaları `models/` klasörüne kaydedilir.
- Rapor görselleri `reports/` klasörüne kaydedilir.



## Streamlit Uygulamasını Çalıştırma

Arayüzü başlatmak için proje ana dizininde şu komutu çalıştırın:

```bash
streamlit run app/app.py
```

Uygulama açıldıktan sonra kullanıcı bir metin girer ve model bu metnin hangi intent sınıfına ait olduğunu tahmin eder.

## Örnek Tahminler

Model aşağıdaki gibi metinleri sınıflandırabilir:

```text
Metin: Bu lupus olabilir mi?
Tahmin: hipotez

Metin: Hemen MR çekin.
Tahmin: talimat

Metin: Tedaviye antibiyotikle başlayalım.
Tahmin: tedavi
```

## Performans

Son eğitimde model yaklaşık olarak şu değerleri üretmiştir:

```text
Accuracy: 0.6603
Balanced accuracy: 0.4401
Macro F1: 0.4272
Weighted F1: 0.6587
```

Model performansı veri setinin dengesi, sınıf sayısı ve metinlerin benzerliğine göre değişebilir. Daha yüksek başarı için veri seti genişletilebilir, sınıf dengesi iyileştirilebilir veya farklı modeller denenebilir.

## Notlar

Eğer daha önce farklı bir scikit-learn sürümüyle kaydedilmiş model dosyaları kullanılırsa `InconsistentVersionWarning` uyarısı görülebilir. Bu durumda en güvenli çözüm modeli mevcut scikit-learn sürümüyle yeniden eğitmektir.

## Amaç

Bu proje, medikal metinlerde intent sınıflandırması yaparak doğal dil işleme, metin sınıflandırma, model eğitimi ve model değerlendirme süreçlerini uygulamalı olarak göstermeyi amaçlar.
