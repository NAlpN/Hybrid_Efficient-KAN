# [cite_start]Hybrid Efficient-KAN: Koroid Tümörlerinin Sınıflandırılması için Çok Modlu Bir Derin Öğrenme Çerçevesi [cite: 1]

[cite_start]Bu depo, nadir görülen koroid tümörlerinin (melanom, hemanjiom ve metastatik karsinom) sınıflandırılmasında karşılaşılan eksik modalite ve sınıf dengesizliği problemlerinin ele alınması amacıyla geliştirilen **Hybrid Efficient-KAN** modelinin resmi uygulamasını ve ablasyon çalışmalarını içermektedir[cite: 4].

## 📌 Proje Özeti
[cite_start]Önerilen modelde, özellik çıkarımı aşamasında parametre verimliliği yüksek EfficientNet-B0 mimarisi kullanılırken sınıflandırma aşamasında öğrenilebilir aktivasyon fonksiyonlarına sahip Kolmogorov–Arnold Ağları (KAN) kullanılmıştır[cite: 5]. [cite_start]Elde edilen bulgular, çok modlu tıbbi görüntüleme verilerinin birlikte modellenmesinin tanısal doğruluğu anlamlı ölçüde artırdığını göstermektedir[cite: 8].

## 🔬 Veri Seti
[cite_start]Bu çalışmada 750 hastaya ait görüntüleri içeren CTI (Choroid Tri-Modal Imaging) veri seti kullanılmıştır[cite: 60, 61]. [cite_start]Model, çoklu modalite mimarisi sayesinde aşağıdaki üç farklı görüntüleme tipini eşzamanlı olarak işler[cite: 63]:
* [cite_start]**FA (Flöresein Anjiyografi):** Retina ve yüzeyel damar ağının incelenmesi[cite: 23].
* [cite_start]**ICGA (İndosiyanin Yeşili Anjiyografisi):** Koroidin daha derin vasküler yapılarının görüntülenmesi[cite: 23].
* [cite_start]**US (Oküler Ultrasonografi):** Tümörün morfolojik yapısı, iç yankılanma özellikleri ve akustik yoğunluğu[cite: 24].

## 🚀 Performans
[cite_start]Önerilen **EfficientNet-B0 (Zero Padding)** konfigürasyonu ile gerçekleştirilen testlerde aşağıdaki yüksek performans metrikleri elde edilmiştir[cite: 79]:
* [cite_start]**Doğruluk (Accuracy):** %94,83 [cite: 79]
* [cite_start]**Özgüllük (Specificity):** %97,87 [cite: 79]
* [cite_start]**Kesinlik (Precision):** %92,86 [cite: 79]
* [cite_start]**Hassasiyet (Sensitivity):** %90,81 [cite: 79]
* [cite_start]**F1-Skoru:** %90,81 [cite: 79]

## ⚙️ Kurulum ve Çalıştırma (Run)

1) Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
