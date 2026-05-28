# 🌾 SPK Rekomendasi Tanaman
**Kelompok 5 — Sistem Pendukung Keputusan (DSS)**

Aplikasi berbasis Machine Learning untuk merekomendasikan tanaman optimal berdasarkan kondisi tanah dan iklim.

## Fitur
- 🎯 **Klasifikasi** — Random Forest, top-3 rekomendasi + persentase kecocokan
- 🗺️ **Zonasi Ekologi** — K-Means clustering, identifikasi zona lahan
- 📊 **Eksplorasi Data** — Distribusi, korelasi, boxplot interaktif
- 🔬 **What-If Analysis** — Simulasi perubahan parameter
- 📡 **Sensitivity Analysis** — Heatmap sensitivitas per fitur

## Cara Jalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/USERNAME/spk-tanaman.git
cd spk-tanaman

# 2. Install dependencies
pip install -r requirements.txt

# 3. Jalankan
streamlit run app.py
```

## Struktur File
```
spk-tanaman/
├── app.py                    # Aplikasi Streamlit utama
├── Crop_recommendation.csv   # Dataset (wajib ada)
├── requirements.txt          # Dependensi Python
├── README.md
└── .gitignore
```

## Dataset
Dataset dari [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset)
- 2.200 sampel · 22 jenis tanaman · 7 fitur input

## Deploy ke Streamlit Cloud
Lihat panduan di bawah atau file `DEPLOY.md`
