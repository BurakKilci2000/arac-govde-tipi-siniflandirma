# 🚗 Araba Gövde Tipi Sınıflandırma (EfficientNet-B0)

Görsel girdiden aracın gövde tipini **8 sınıf** arasından tahmin eden derin öğrenme projesi.
ImageNet üzerinde ön-eğitilmiş **EfficientNet-B0** transfer öğrenme ile yeniden eğitilmiş,
**Streamlit** tabanlı bir web arayüzüyle canlı tahmin yapılabilir hâle getirilmiştir.

> Kocaeli Üniversitesi – Bilgisayar Mühendisliği – Yazılım Laboratuvarı II – Proje III

## Sınıflar

| No | Sınıf | No | Sınıf |
|---|---|---|---|
| 1 | SUV | 5 | Açık Tekerlekli (F1) |
| 2 | Van | 6 | Sedan |
| 3 | Station Wagon | 7 | Hatchback |
| 4 | Micro | 8 | Pick-up |

## Sonuçlar (Test Seti, 659 görsel)

| Metrik | Değer |
|---|---|
| Makro F1 | **0.8714** |
| Ağırlıklı F1 | **0.8726** |
| Accuracy | **%87.41** |
| Model boyutu | ~20 MB |
| Çıkarım süresi (RTX 2060) | ~62 ms |

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Açık Tekerlekli | 1.000 | 1.000 | 1.000 |
| Van | 0.954 | 0.991 | 0.972 |
| Micro | 0.944 | 0.944 | 0.944 |
| Pick-up | 0.957 | 0.833 | 0.891 |
| SUV | 0.844 | 0.871 | 0.857 |
| Sedan | 0.683 | 0.990 | 0.808 |
| Hatchback | 0.939 | 0.629 | 0.753 |
| Station Wagon | 0.864 | 0.655 | 0.745 |

En sık karışıklık **Hatchback → Sedan** (%31) ve **Station Wagon → Sedan** (%17) arasındadır;
bu sınıflar geometrik olarak birbirine çok yakındır. Grafikler ve karışıklık matrisi `ciktilar/` klasöründedir.

## Yöntem

- **Model:** EfficientNet-B0 (ImageNet ağırlıkları), son katman `Dropout(0.3) → Linear(1280 → 8)`
- **Kayıp:** Sınıf ağırlıklı Cross-Entropy + label smoothing (0.1)
- **Optimizasyon:** AdamW, ayrımcı öğrenme oranı (omurga 1e-4, baş 1e-3), Cosine Annealing
- **Veri çoğaltma:** RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter, Perspective, RandomErasing
- **Eğitim:** 30 epoch, batch 16, erken durdurma (sabır 7), en düşük doğrulama kaybındaki model kaydedilir

## Proje Yapısı

```
├── veri_seti_hazirla.py   # Ham görselleri egitim/dogrulama/test (%75/%15/%10) olarak böler
├── egitim.py              # Modeli eğitir, grafikleri ve metrikleri üretir
├── tahmin.py              # Tahminleyici sınıfı (tek görsel tahmini)
├── arayuz.py              # Streamlit web arayüzü
├── toplu_tahmin.py        # Bir klasördeki görselleri toplu tahmin eder → preds.txt
├── PredictionScript.txt   # Değerlendirme için Colab uyumlu Predict(file_path) fonksiyonu
├── gereksinimler.txt
├── rapor.pdf              # IEEE formatında proje raporu
└── ciktilar/              # Eğitilmiş model, grafikler ve sınıflandırma raporu
```

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r gereksinimler.txt
```

GPU (CUDA 11.8) için PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Kullanım

**Web arayüzü:**
```bash
streamlit run arayuz.py
```

**Tek görsel tahmini:**
```bash
python tahmin.py --model ciktilar/en_iyi_model.pth --gorsel araba.jpg
```

**Toplu tahmin (preds.txt üretir):**
```bash
python toplu_tahmin.py --gorsel_klasor test_gorselleri --model ciktilar/en_iyi_model.pth
```

## Modeli Sıfırdan Eğitmek

Veri seti boyutu nedeniyle repoya eklenmemiştir. Aşağıdaki kaynaklardan indirip
`ham_veri/<SINIF_ADI>/` klasörlerine yerleştirin:

| Sınıf | Kaynak |
|---|---|
| SUV, Van, Sedan, Hatchback, Pick-up | [Car Body Types Images Dataset (Kaggle)](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped) |
| Station Wagon | [Stanford Car Body Type Data (Kaggle)](https://www.kaggle.com/datasets/mayurmahurkar/stanford-car-body-type-data) |
| Açık Tekerlekli | [F1 Image Classification (Kaggle)](https://www.kaggle.com/datasets/loveymishra/f1-image-classification-updated) |
| Micro | Web aramasıyla manuel toplandı (Citroen Ami, Smart Fortwo, Renault Twizy, Microlino, Aixam) |

```bash
python veri_seti_hazirla.py --ham_klasor ham_veri --cikti_klasor veri
python egitim.py --veri_klasor veri --epoch 30 --batch_boyut 16 --isci_sayisi 0
```

## Teknolojiler

Python 3.11 · PyTorch · torchvision · scikit-learn · Streamlit · matplotlib · seaborn
