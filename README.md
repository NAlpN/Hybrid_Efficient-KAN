# Hybrid Efficient-KAN: Koroid Tümörlerinin Sınıflandırılması için Çok Modlu Bir Derin Öğrenme Çerçevesi

Bu depo, nadir görülen koroid tümörlerinin (melanom, hemanjiom ve metastatik karsinom) sınıflandırılmasında karşılaşılan eksik modalite ve sınıf dengesizliği problemlerinin ele alınması amacıyla geliştirilen **Hybrid Efficient-KAN** modelinin resmi kodlarını ve ablasyon çalışmalarını içermektedir.

## 📌 Proje Özeti
Önerilen Hybrid Efficient-KAN modelinde, özellik çıkarımı aşamasında parametre verimliliği yüksek EfficientNet-B0 mimarisi, sınıflandırma aşamasında ise öğrenilebilir aktivasyon fonksiyonlarına sahip Kolmogorov–Arnold Ağları (KAN) kullanılmıştır. Geleneksel CNN mimarileri ile yüksek temsilli özellikler elde edilirken, KAN’ın esnek yapısı sayesinde eksik modalite problemine karşı daha dayanıklı, lokalize ve sağlam bir öğrenme süreci sağlanmıştır.

## 🔬 Veri Seti (CTI Dataset)
Model, tanısal doğruluğu en üst düzeye çıkarmak için 750 hastadan elde edilen üç farklı tıbbi görüntüleme modalitesini eşzamanlı olarak işler:
* **FA (Flöresein Anjiyografi):** Retina ve yüzeyel damar ağının incelenmesi.
* **ICGA (İndosiyanin Yeşili Anjiyografisi):** Koroidin daha derin vasküler yapılarının görüntülenmesi.
* **US (Oküler Ultrasonografi):** Tümörün morfolojik yapısı ve akustik yoğunluğu.

## 🚀 Performans ve Ablasyon Çalışmaları
Farklı derin öğrenme mimarileri (DenseNet-121, ResNet-50) ve veri çoğaltma stratejileri (Zero Padding, Gaussian Noise, Diffusion) ile gerçekleştirilen kapsamlı deneyler sonucunda, önerilen **Hybrid Efficient-KAN (Zero Padding)** modeli en yüksek performansı sergilemiştir:
* **Doğruluk (Accuracy):** %94.83
* **Özgüllük (Specificity):** %97.87
* **Kesinlik (Precision):** %92.86
* **Hassasiyet (Sensitivity):** %90.81
* **F1-Skoru:** %90.81

## ⚙️ Kurulum ve Çalıştırma

**1. Gereksinimlerin Yüklenmesi:**
Projeyi klonladıktan sonra gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
