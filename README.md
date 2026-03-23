# Hybrid Efficient-KAN: Koroid Tümörlerinin Sınıflandırılması için Çok Modlu Bir Derin Öğrenme Çerçevesi

Bu depo, nadir görülen koroid tümörlerinin (melanom, hemanjiom ve metastatik karsinom) sınıflandırılmasında karşılaşılan eksik modalite ve sınıf dengesizliği problemlerinin ele alınması amacıyla geliştirilen **Hybrid Efficient-KAN** modelinin resmi kodlarını ve ablasyon çalışmalarını içermektedir.

## Proje Özeti
Önerilen Hybrid Efficient-KAN modelinde, özellik çıkarımı aşamasında parametre verimliliği yüksek EfficientNet-B0 mimarisi, sınıflandırma aşamasında ise öğrenilebilir aktivasyon fonksiyonlarına sahip Kolmogorov–Arnold Ağları (KAN) kullanılmıştır. Geleneksel CNN mimarileri ile yüksek temsilli özellikler elde edilirken, KAN’ın esnek yapısı sayesinde eksik modalite problemine karşı daha dayanıklı ve sağlam bir öğrenme süreci sağlanmıştır.

## Veri Seti (CTI Dataset)
Model, tanısal doğruluğu en üst düzeye çıkarmak için 750 hastadan elde edilen üç farklı tıbbi görüntüleme modalitesini eşzamanlı olarak işler:
* **FA (Flöresein Anjiyografi):** Retina ve yüzeyel damar ağının incelenmesi.
* **ICGA (İndosiyanin Yeşili Anjiyografisi):** Koroidin daha derin vasküler yapılarının görüntülenmesi.
* **US (Oküler Ultrasonografi):** Tümörün morfolojik yapısı ve akustik yoğunluğu.

**Veri Setine ulaşmak için: https://drive.google.com/drive/folders/1YwDhqC_M9ACBnGjn_8IZouWHgJx1ue5Q**

## Performans
Farklı derin öğrenme mimarileri ve veri çoğaltma stratejileri ile gerçekleştirilen kapsamlı deneyler sonucunda, önerilen **Hybrid Efficient-KAN (Zero Padding)** modeli en yüksek performansı sergilemiştir:
* **Doğruluk (Accuracy):** %94.83
* **Özgüllük (Specificity):** %97.87
* **Kesinlik (Precision):** %92.86
* **Hassasiyet (Sensitivity):** %90.81
* **F1-Skoru:** %90.81

--------------------------------------------------------------------------------------------------------------

### Gereksinimlerin Yüklenmesi
Projeyi klonladıktan sonra öncelikle gerekli Python kütüphanelerini yükleyin:
```bash
pip install -r requirements.txt
```

## Ana Modelin Eğitimi
Tüm sistemin belkemiği olan ve en yüksek performansı (%94.83) veren ana mimariyi eğitmek ve test etmek için aşağıdaki betiği çalıştırın:
```bash
python hybrid_kan_model.py
```

## Ablasyon Modellerinin Çalıştırılması
Önerilen modelin başarısını literatürdeki diğer popüler mimarilerle ve farklı veri çoğaltma (Gaussian Noise vb.) stratejileriyle kıyaslamak için hazırlanan ablasyon betiklerini doğrudan çalıştırabilirsiniz:
```bash
python DenseNet-121.py
python DenseNet-121_GN.py
python ResNet-50.py
python ResNet-50_GN.py
python Gaussian.py
```

## Jupyter Notebook Analizleri
Modelin Diffusion veri artırma tekniği altındaki davranışını ve Test-Time Augmentation (TTA) kullanılmayan durumdaki detaylı istatistiksel sonuçlarını incelemek için analiz defterini başlatın:
```bash
jupyter notebook "Diffusion&No_TTA.ipynb"
```

## Çıktılar
Eğitim süreçleri tamamlandığında; sınıflandırma performansını gösteren karmaşıklık matrisleri, ROC eğrileri, t-SNE dağılım grafikleri ve detaylı CSV dökümleri betikler tarafından otomatik olarak **experiments/** dizinine kaydedilecektir.
