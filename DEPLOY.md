# 🚀 Panduan Deploy — Streamlit Community Cloud (GRATIS)

## Persiapan File

Pastikan folder project berisi:
```
spk-tanaman/
├── app.py
├── Crop_recommendation.csv   ← WAJIB ADA
├── requirements.txt
└── README.md
```

---

## Langkah 1 — Buat Akun GitHub
1. Buka https://github.com dan daftar (gratis)
2. Klik **New repository**
3. Nama repo: `spk-tanaman`
4. Centang **Public**
5. Klik **Create repository**

---

## Langkah 2 — Upload File ke GitHub

### Cara A — Via Website (Termudah)
1. Buka repo yang baru dibuat
2. Klik **Add file → Upload files**
3. Drag & drop semua file: `app.py`, `Crop_recommendation.csv`, `requirements.txt`, `README.md`
4. Klik **Commit changes**

### Cara B — Via Terminal (Git)
```bash
git init
git add .
git commit -m "Initial commit SPK Tanaman"
git branch -M main
git remote add origin https://github.com/USERNAME/spk-tanaman.git
git push -u origin main
```

---

## Langkah 3 — Deploy ke Streamlit Cloud

1. Buka **https://share.streamlit.io**
2. Login dengan akun **GitHub**
3. Klik **New app**
4. Isi form:
   - **Repository**: `USERNAME/spk-tanaman`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **Deploy!**
6. Tunggu ~2–3 menit
7. ✅ Aplikasi live di: `https://USERNAME-spk-tanaman-app-XXXX.streamlit.app`

---

## Selesai!

URL bisa langsung dibagikan ke siapa saja.
Setiap kali push ke GitHub, aplikasi otomatis update.

---

## Troubleshooting

| Error | Solusi |
|-------|--------|
| `FileNotFoundError: Crop_recommendation.csv` | Pastikan CSV ada di root folder repo |
| `ModuleNotFoundError` | Cek `requirements.txt` sudah ada dan benar |
| App loading lama | Normal untuk pertama kali, model sedang di-cache |
| Memory error | Kurangi `n_estimators` dari 200 ke 100 di `app.py` |

---

## Alternatif Platform Deploy

| Platform | Kelebihan | Link |
|----------|-----------|------|
| **Streamlit Cloud** ⭐ | Gratis, paling mudah untuk Streamlit | share.streamlit.io |
| **Hugging Face Spaces** | Gratis, support Streamlit & Gradio | huggingface.co/spaces |
| **Railway** | Gratis $5/bulan credit, support semua framework | railway.app |
| **Render** | Gratis (sleep setelah idle) | render.com |
