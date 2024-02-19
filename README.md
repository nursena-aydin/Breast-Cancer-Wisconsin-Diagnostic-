# Breast Cancer Predictor(Streamlit)

Bu proje, meme kanseri hücre kümelerinin iyi huylu (benign) veya kötü huylu (malignant) olup olmadığını tahmin etmek için bir makine öğrenimi modeli kullanır. Model, sitoloji laboratuvarınızdan yapılan ölçümlere dayanır ve Streamlit kullanılarak bir web uygulaması olarak sunulmuştur.

# Nasıl Kullanılır
*Veri Yüklemek: Yan taraftaki menüden bir veri seti yükleyin. Veri setinizin sütunlarının özelliklerini ve tanılarını içermesi beklenir.

*Özellikleri Güncellemek: Yan taraftaki kaydırmalı çubukları kullanarak özellik değerlerini güncelleyin. Bu, tahminlerinizi bireyselleştirmenize ve farklı senaryoları denemenize olanak tanır.

*Model Seçmek: Yan taraftaki menüden kullanılacak modeli seçin. Kullanılabilir modeller arasında K-En Yakın Komşular, Destek Vektör Makineleri ve Naif Bayes bulunmaktadır.

*Tahminleri Görüntülemek: Güncellenmiş özelliklere dayanarak yapılan tahminleri gözlemleyin. Tahmin edilen hücre kümesinin iyi huylu mu yoksa kötü huylu mu olduğunu ve bu tahminlerin olasılıklarını görüntüleyin.

*Model Değerlendirmesi: Proje, seçilen modelin doğruluğunu, hassasiyetini, geri çağırmasını ve F1 skorunu hesaplar. Ayrıca bir karışıklık matrisi de sunar.

## Kurulum
Proje Python tabanlıdır. Projeyi çalıştırmak için öncelikle gerekli paketleri yüklemeniz gerekmektedir:
`pip install pandas`
`pip install streamlit`
`pip install plotly`

## Uygulamayı başlatmak için aşağıdaki komutu çalıştırmanız yeterlidir:
`streamlit run app/main.py`
