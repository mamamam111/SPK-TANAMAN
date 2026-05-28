"""
AgroSPK — Sistem Pendukung Keputusan Rekomendasi Tanaman
Kelompok 5 | Clean UI • Ramah Petani • Analisis Mendalam
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, silhouette_score
)
from sklearn.decomposition import PCA
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSPK — Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #052e16 0%, #14532d 40%, #166534 100%);
    border-right: none; box-shadow: 4px 0 20px rgba(0,0,0,.25);
}
section[data-testid="stSidebar"] * { color: #bbf7d0 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(187,247,208,.15) !important; }
section[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,.05); border-radius: 10px;
    padding: 9px 14px; display: flex; align-items: center; gap: 8px;
    transition: all .2s; border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,.1); border-color: rgba(187,247,208,.2);
}

/* Main button */
.stButton > button {
    background: linear-gradient(135deg, #15803d, #16a34a, #22c55e) !important;
    color: white !important; border: none !important; border-radius: 14px !important;
    font-weight: 700 !important; font-size: 1.05rem !important;
    padding: 14px 28px !important; width: 100% !important;
    box-shadow: 0 4px 15px rgba(22,163,74,.35) !important;
    transition: all .2s !important; letter-spacing: .3px !important;
}
.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(22,163,74,.5) !important;
    transform: translateY(-2px) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { margin-top: 0 !important; }

/* ── Custom Components ── */
.page-banner {
    background: linear-gradient(135deg, #052e16 0%, #14532d 50%, #15803d 100%);
    border-radius: 20px; padding: 32px 40px; margin-bottom: 24px;
    color: white; position: relative; overflow: hidden;
}
.page-banner::after {
    content: "🌾"; position: absolute; font-size: 110px;
    opacity: .07; right: 30px; bottom: -10px;
}
.page-banner h1 { font-size: 1.8rem; font-weight: 800; margin: 0 0 6px; }
.page-banner p  { font-size: .92rem; opacity: .8; margin: 0; max-width: 580px; }
.page-banner .badge {
    display: inline-block; background: rgba(255,255,255,.15);
    border-radius: 20px; padding: 3px 12px; font-size: .75rem; margin-top: 10px; margin-right: 6px;
}

.section-title {
    font-size: 1rem; font-weight: 700; color: #14532d;
    text-transform: uppercase; letter-spacing: 1px;
    padding-bottom: 8px; border-bottom: 2px solid #dcfce7; margin-bottom: 14px;
}

/* Input cards */
.input-section {
    background: white; border-radius: 18px; padding: 22px 24px;
    border: 1px solid #dcfce7; box-shadow: 0 2px 12px rgba(22,163,74,.08);
}
.input-label {
    font-size: .78rem; font-weight: 600; color: #166534;
    text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px;
}
.input-hint { font-size: .7rem; color: #6b7280; margin-top: -8px; }

/* Result cards */
.rec-card {
    background: linear-gradient(135deg, #052e16, #14532d);
    border-radius: 20px; padding: 28px 24px; text-align: center; color: white;
    box-shadow: 0 8px 32px rgba(5,46,22,.4); height: 100%;
}
.rec-rank1 { border: 2px solid #4ade80; }
.rec-rank2 { border: 1px solid rgba(255,255,255,.2); }
.rec-rank3 { border: 1px solid rgba(255,255,255,.1); }
.rec-emoji { font-size: 3rem; display: block; margin-bottom: 8px; line-height: 1; }
.rec-label { font-size: .65rem; opacity: .7; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.rec-name  { font-size: 1.15rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; }
.rec-pct   { font-size: 2rem; font-weight: 800; color: #4ade80; margin: 6px 0; }
.rec-desc  { font-size: .72rem; opacity: .75; line-height: 1.5; }
.rec-musim { display: inline-block; background: rgba(255,255,255,.12);
             border-radius: 20px; padding: 3px 10px; font-size: .68rem; margin-top: 8px; }

/* Zone card */
.zona-hero {
    border-radius: 18px; padding: 22px 20px; text-align: center;
    height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,.1);
}
.zona-icon  { font-size: 2.8rem; display: block; margin-bottom: 8px; }
.zona-nomor { font-size: .7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: .75; }
.zona-nama  { font-size: 1.1rem; font-weight: 800; margin: 4px 0; }
.zona-desc  { font-size: .72rem; opacity: .8; line-height: 1.6; }

/* Status table */
.status-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 9px 14px; border-bottom: 1px solid #f0fdf4; font-size: .82rem;
}
.status-row:last-child { border-bottom: none; }
.status-key { color: #374151; font-weight: 500; }
.status-val { font-weight: 700; color: #111827; }
.st-ok   { background: #dcfce7; color: #14532d; border-radius: 20px; padding: 2px 10px; font-size: .72rem; font-weight: 600; }
.st-warn { background: #fef9c3; color: #854d0e; border-radius: 20px; padding: 2px 10px; font-size: .72rem; font-weight: 600; }
.st-bad  { background: #fee2e2; color: #991b1b; border-radius: 20px; padding: 2px 10px; font-size: .72rem; font-weight: 600; }

/* Metric boxes */
.kpi { background: white; border-radius: 14px; padding: 18px 20px;
       border: 1px solid #dcfce7; text-align: center;
       box-shadow: 0 1px 8px rgba(22,163,74,.07); }
.kpi-val { font-size: 1.9rem; font-weight: 800; color: #15803d; }
.kpi-lbl { font-size: .72rem; color: #6b7280; margin-top: 2px; text-transform: uppercase; letter-spacing: .5px; }

/* Prob bars */
.pbar-wrap { margin-bottom: 12px; }
.pbar-head { display: flex; justify-content: space-between; font-size: .8rem;
             font-weight: 600; color: #1f2937; margin-bottom: 4px; }
.pbar-pct  { color: #16a34a; font-weight: 700; }
.pbar-bg   { background: #f0fdf4; border-radius: 6px; height: 9px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 6px; }

/* Model comparison table */
.model-tbl { width: 100%; border-collapse: collapse; font-size: .83rem; }
.model-tbl th { background: #f0fdf4; color: #14532d; padding: 11px 14px;
                text-align: left; font-size: .72rem; text-transform: uppercase;
                letter-spacing: .5px; border-bottom: 2px solid #dcfce7; }
.model-tbl td { padding: 10px 14px; border-bottom: 1px solid #f9fafb; color: #374151; }
.model-tbl tr:hover td { background: #fafff9; }
.model-tbl .best { color: #15803d; font-weight: 700; }
.model-badge { border-radius: 20px; padding: 2px 10px; font-size: .72rem; font-weight: 600;
               background: #dcfce7; color: #14532d; display: inline-block; }

/* Tips */
.tips { background: #f0fdf4; border-left: 4px solid #22c55e; border-radius: 0 10px 10px 0;
        padding: 12px 16px; font-size: .82rem; color: #14532d; margin: 8px 0 16px; }

/* White card */
.wcard { background: white; border-radius: 16px; padding: 22px;
         border: 1px solid #dcfce7; box-shadow: 0 2px 12px rgba(22,163,74,.06); }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────────────────────
FITUR   = ["N","P","K","temperature","humidity","ph","rainfall"]
F_NAMA  = {"N":"Nitrogen","P":"Fosfor","K":"Kalium",
           "temperature":"Suhu","humidity":"Kelembaban",
           "ph":"pH Tanah","rainfall":"Curah Hujan"}
F_SAT   = {"N":"ppm","P":"ppm","K":"ppm","temperature":"°C",
           "humidity":"%","ph":"","rainfall":"mm"}
F_LONG  = {"N":"Nitrogen (N)","P":"Fosfor (P)","K":"Kalium (K)",
           "temperature":"Suhu Udara (°C)","humidity":"Kelembaban (%)",
           "ph":"pH Tanah","rainfall":"Curah Hujan (mm)"}
F_HELP  = {
    "N":"Kandungan nitrogen dalam tanah. Makin tinggi makin subur untuk tanaman sayur/daun.",
    "P":"Kandungan fosfor, penting untuk akar kuat dan pembentukan buah.",
    "K":"Kandungan kalium, meningkatkan kualitas buah dan ketahanan penyakit.",
    "temperature":"Suhu rata-rata harian di lokasi lahan.",
    "humidity":"Kelembaban udara rata-rata di sekitar lahan.",
    "ph":"Tingkat keasaman tanah. pH 7 = netral, <7 = asam, >7 = basa.",
    "rainfall":"Total curah hujan tahunan yang diterima lahan.",
}
F_MIN   = {"N":0,"P":5,"K":5,"temperature":8.0,"humidity":14.0,"ph":3.5,"rainfall":20.0}
F_MAX   = {"N":140,"P":145,"K":205,"temperature":44.0,"humidity":100.0,"ph":10.0,"rainfall":300.0}
F_DEF   = {"N":50,"P":53,"K":48,"temperature":25.0,"humidity":71.0,"ph":6.5,"rainfall":100.0}
F_STEP  = {"N":1,"P":1,"K":1,"temperature":0.5,"humidity":0.5,"ph":0.1,"rainfall":1.0}

TANAMAN = {
    "rice":{"e":"🌾","s":"Basah","d":"Padi sawah, butuh banyak air dan irigasi yang baik."},
    "maize":{"e":"🌽","s":"Kering","d":"Jagung, cocok untuk lahan kering dengan drainase baik."},
    "chickpea":{"e":"🫘","s":"Dingin","d":"Kacang arab, toleran kering, bagus untuk rotasi tanaman."},
    "kidneybeans":{"e":"🫘","s":"Hangat","d":"Kacang merah, kelembaban sedang, pH netral."},
    "pigeonpeas":{"e":"🌿","s":"Kering","d":"Kacang gude, cocok lahan kering dan berlereng."},
    "mothbeans":{"e":"🌿","s":"Kering","d":"Kacang moth, sangat toleran kekeringan ekstrem."},
    "mungbean":{"e":"🫛","s":"Hangat","d":"Kacang hijau, siklus panen pendek ±65 hari."},
    "blackgram":{"e":"🫘","s":"Hangat","d":"Kacang urad, toleran panas, curah hujan sedang."},
    "lentil":{"e":"🫘","s":"Dingin","d":"Lentil, tumbuh baik di tanah lempung berdrainase baik."},
    "pomegranate":{"e":"🍎","s":"Panas","d":"Delima, tahan kering, nilai komersial tinggi."},
    "banana":{"e":"🍌","s":"Basah","d":"Pisang, butuh kelembaban dan tanah subur tinggi."},
    "mango":{"e":"🥭","s":"Panas","d":"Mangga, buah tropis dengan nilai jual tinggi."},
    "grapes":{"e":"🍇","s":"Panas","d":"Anggur, butuh pH agak asam dan drainase sangat baik."},
    "watermelon":{"e":"🍉","s":"Panas","d":"Semangka, tanah berpasir dan sinar matahari penuh."},
    "muskmelon":{"e":"🍈","s":"Panas","d":"Melon, butuh musim panas panjang dan drainase baik."},
    "apple":{"e":"🍏","s":"Dingin","d":"Apel, cocok untuk dataran tinggi dengan suhu sejuk."},
    "orange":{"e":"🍊","s":"Hangat","d":"Jeruk, butuh pH sedikit asam dan curah hujan teratur."},
    "papaya":{"e":"🍑","s":"Basah","d":"Pepaya, tumbuh cepat, cocok di tanah gembur."},
    "coconut":{"e":"🥥","s":"Basah","d":"Kelapa, cocok tanah pantai dengan kelembaban tinggi."},
    "cotton":{"e":"🌼","s":"Panas","d":"Kapas, cocok lahan kering panas dengan drainase baik."},
    "jute":{"e":"🌿","s":"Basah","d":"Jute (rami), butuh lahan basah dan curah hujan tinggi."},
    "coffee":{"e":"☕","s":"Basah","d":"Kopi, tumbuh baik di ketinggian sedang dengan pH asam."},
}

ZONA_WARNA  = ["#0ea5e9","#f59e0b","#a855f7","#22c55e","#ef4444","#06b6d4","#f97316"]
ZONA_GELAP  = ["#0c4a6e","#78350f","#581c87","#14532d","#7f1d1d","#164e63","#7c2d12"]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def wz(i): return ZONA_WARNA[i % len(ZONA_WARNA)]
def wd(i): return ZONA_GELAP[i % len(ZONA_GELAP)]

def nama_zona(c, i):
    N,P,K,t,h,ph,r = c
    if r > 155 and h > 72: return f"Zona {i} — Tropis Basah 🌧️"
    if r < 85  and h < 58: return f"Zona {i} — Semi-Kering 🌵"
    if K > 90  or P > 90:  return f"Zona {i} — Subur Mineral 💎"
    return f"Zona {i} — Komersial Sedang 🌿"

def cek_param(val, ideal):
    p = abs(val - ideal) / (ideal + 1e-9) * 100
    if p <= 20: return "✅ Sesuai",    "st-ok"
    if p <= 40: return "⚠️ Perhatikan","st-warn"
    return "❌ Perlu Koreksi",         "st-bad"

def bar(pct, warna="#16a34a", opacity="cc"):
    return (f'<div class="pbar-bg">'
            f'<div class="pbar-fill" style="width:{pct:.1f}%;background:{warna}{opacity}"></div>'
            f'</div>')


# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def muat_data():
    return pd.read_csv("Crop_recommendation.csv")


@st.cache_resource
def latih_semua(_df):
    X = _df[FITUR]; y = _df["label"]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    sc = StandardScaler()
    X_trs = sc.fit_transform(X_tr); X_tes = sc.transform(X_te)

    # Semua 6 model
    models = {
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42),
        "Naive Bayes":         GaussianNB(),
        "SVM":                 SVC(gamma="auto",probability=True,random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000,random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),
        "XGBoost":             None,  # handled separately
    }
    le = LabelEncoder(); y_tr_enc = le.fit_transform(y_tr); y_te_enc = le.transform(y_te)

    hasil = {}
    trained = {}
    for nama, m in models.items():
        if nama == "XGBoost":
            m = xgb.XGBClassifier(use_label_encoder=False,eval_metric="mlogloss",random_state=42,n_jobs=-1)
            m.fit(X_trs, y_tr_enc)
            yp = m.predict(X_tes)
            acc = accuracy_score(y_te_enc, yp)
            prec= precision_score(y_te_enc,yp,average="weighted",zero_division=0)
            rec = recall_score(y_te_enc,yp,average="weighted",zero_division=0)
            f1  = f1_score(y_te_enc,yp,average="weighted",zero_division=0)
            cvs = cross_val_score(m, sc.transform(X), le.transform(y), cv=5)
        else:
            m.fit(X_trs, y_tr)
            yp = m.predict(X_tes)
            acc = accuracy_score(y_te, yp)
            prec= precision_score(y_te,yp,average="weighted",zero_division=0)
            rec = recall_score(y_te,yp,average="weighted",zero_division=0)
            f1  = f1_score(y_te,yp,average="weighted",zero_division=0)
            cvs = cross_val_score(m, sc.transform(X), y, cv=5)

        hasil[nama]  = {"acc":acc,"prec":prec,"rec":rec,"f1":f1,
                        "cv_mean":cvs.mean(),"cv_std":cvs.std()}
        trained[nama]= m

    rf  = trained["Random Forest"]
    fi  = rf.feature_importances_
    cm  = confusion_matrix(y_te, trained["Random Forest"].predict(X_tes), labels=rf.classes_)
    return trained, sc, hasil, fi, cm, rf.classes_, le


@st.cache_resource
def latih_cluster(_df, k=4):
    X=_df[FITUR]; sc=StandardScaler(); Xs=sc.fit_transform(X)
    km=KMeans(n_clusters=k,random_state=42,n_init=15)
    lb=km.fit_predict(Xs); sil=silhouette_score(Xs,lb)
    pca=PCA(n_components=2,random_state=42); Xp=pca.fit_transform(Xs)
    raw=sc.inverse_transform(km.cluster_centers_)
    return km,sc,lb,sil,Xp,raw


df            = muat_data()
trained,sc,hasil_model,fi,cm_rf,kelas,le = latih_semua(df)
rf_model      = trained["Random Forest"]


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:16px 0 20px">
      <div style="font-size:2.4rem">🌾</div>
      <div style="font-size:1.1rem;font-weight:800;color:#bbf7d0">AgroSPK</div>
      <div style="font-size:.72rem;opacity:.6;margin-top:3px">Rekomendasi Tanaman Berbasis AI</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    menu = st.radio("", [
        "🌱  Dashboard Petani",
        "🗺️  Peta Zona Ekologi",
        "📊  Analisis & Model",
        "🔬  Simulasi Lanjutan",
    ], label_visibility="collapsed")

    st.divider()
    rf_h = hasil_model["Random Forest"]
    st.markdown(f"""
    <div style="font-size:.72rem;color:#86efac;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">
    Model Terbaik (RF)</div>
    <div style="display:flex;justify-content:space-between;font-size:.8rem;padding:3px 0">
        <span style="opacity:.7">Akurasi</span><b style="color:#4ade80">{rf_h["acc"]:.2%}</b></div>
    <div style="display:flex;justify-content:space-between;font-size:.8rem;padding:3px 0">
        <span style="opacity:.7">F1-Score</span><b style="color:#4ade80">{rf_h["f1"]:.2%}</b></div>
    <div style="display:flex;justify-content:space-between;font-size:.8rem;padding:3px 0">
        <span style="opacity:.7">CV 5-fold</span>
        <b style="color:#4ade80">{rf_h["cv_mean"]:.2%}±{rf_h["cv_std"]:.2%}</b></div>
    <div style="font-size:.72rem;margin-top:10px;opacity:.55">
        📁 {len(df):,} sampel · 22 tanaman · 7 fitur</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DASHBOARD PETANI
# ══════════════════════════════════════════════════════════════
if menu == "🌱  Dashboard Petani":

    st.markdown("""
    <div class="page-banner">
      <h1>Selamat Datang, Pak/Bu Tani! 👋</h1>
      <p>Masukkan kondisi tanah dan iklim lahan Anda, lalu klik <b>Analisis Lahan</b>
         untuk mendapatkan rekomendasi tanaman terbaik beserta zona ekologi lahan Anda.</p>
      <span class="badge">🎯 Rekomendasi Otomatis</span>
      <span class="badge">🗺️ Deteksi Zona Ekologi</span>
      <span class="badge">📋 Analisis Kondisi Lahan</span>
    </div>
    """, unsafe_allow_html=True)

    # ── INPUT SECTION ──
    with st.container():
        st.markdown("""
        <div style="background:white;border-radius:18px;padding:22px 24px 12px;
             border:1px solid #dcfce7;box-shadow:0 2px 12px rgba(22,163,74,.08);margin-bottom:20px">
        <div class="section-title">⚗️ Kondisi Tanah & Iklim Lahan Anda</div>
        </div>""", unsafe_allow_html=True)

        ca,cb,cc,cd,ce,cf,cg = st.columns(7)
        with ca:
            st.markdown('<div class="input-label">🧪 Nitrogen</div>', unsafe_allow_html=True)
            N = st.slider("N_s",0,140,50,1,label_visibility="collapsed",key="N",
                          help=F_HELP["N"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{N} ppm</b></div>', unsafe_allow_html=True)
        with cb:
            st.markdown('<div class="input-label">🌱 Fosfor</div>', unsafe_allow_html=True)
            P = st.slider("P_s",5,145,53,1,label_visibility="collapsed",key="P",
                          help=F_HELP["P"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{P} ppm</b></div>', unsafe_allow_html=True)
        with cc:
            st.markdown('<div class="input-label">💧 Kalium</div>', unsafe_allow_html=True)
            K = st.slider("K_s",5,205,48,1,label_visibility="collapsed",key="K",
                          help=F_HELP["K"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{K} ppm</b></div>', unsafe_allow_html=True)
        with cd:
            st.markdown('<div class="input-label">🌡️ Suhu</div>', unsafe_allow_html=True)
            T = st.slider("T_s",8.0,44.0,25.0,.5,label_visibility="collapsed",key="T",
                          help=F_HELP["temperature"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{T}°C</b></div>', unsafe_allow_html=True)
        with ce:
            st.markdown('<div class="input-label">💦 Kelembaban</div>', unsafe_allow_html=True)
            H = st.slider("H_s",14.0,100.0,71.0,.5,label_visibility="collapsed",key="H",
                          help=F_HELP["humidity"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{H}%</b></div>', unsafe_allow_html=True)
        with cf:
            st.markdown('<div class="input-label">⚗️ pH Tanah</div>', unsafe_allow_html=True)
            PH = st.slider("PH_s",3.5,10.0,6.5,.1,label_visibility="collapsed",key="PH",
                           help=F_HELP["ph"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{PH}</b></div>', unsafe_allow_html=True)
        with cg:
            st.markdown('<div class="input-label">🌧️ Curah Hujan</div>', unsafe_allow_html=True)
            R = st.slider("R_s",20.0,300.0,100.0,1.0,label_visibility="collapsed",key="R",
                          help=F_HELP["rainfall"])
            st.markdown(f'<div class="input-hint">Nilai: <b>{R} mm</b></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analisis = st.button("🔍 Analisis Lahan Saya — Dapatkan Rekomendasi", use_container_width=True)

    # ── HASIL ──
    if analisis or st.session_state.get("show_result"):
        st.session_state["show_result"] = True
        inp    = np.array([[N,P,K,T,H,PH,R]])
        inp_sc = sc.transform(inp)
        pred   = rf_model.predict(inp_sc)[0]
        proba  = rf_model.predict_proba(inp_sc)[0]
        top5   = sorted(zip(kelas, proba), key=lambda x:-x[1])[:5]
        conf   = proba[list(kelas).index(pred)]
        info   = TANAMAN.get(pred,{"e":"🌿","s":"-","d":"-"})

        # Zona
        km,km_sc,km_lb,sil,_,raw = latih_cluster(df,4)
        zona_id = km.predict(km_sc.transform(inp))[0]
        nz   = nama_zona(raw[zona_id], zona_id)
        warna_z = wz(zona_id)
        dark_z  = wd(zona_id)

        # Tanaman dominan zona
        df_z = df.copy(); df_z["zona"] = km.predict(km_sc.transform(df[FITUR]))
        top_z = df_z[df_z["zona"]==zona_id]["label"].value_counts().head(5)

        st.markdown("---")
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:14px;
             padding:14px 20px;margin-bottom:20px;font-size:.9rem;color:#14532d">
          ✅ Analisis selesai! Berikut hasil rekomendasi untuk kondisi lahan
          <b>N={N} · P={P} · K={K} · Suhu={T}°C · Kelembaban={H}% · pH={PH} · Hujan={R}mm</b>
        </div>""", unsafe_allow_html=True)

        # ── ROW 1: Top-3 Rekomendasi ──
        st.markdown('<div class="section-title">🏆 Rekomendasi Tanaman Terbaik</div>',
                    unsafe_allow_html=True)
        cr1,cr2,cr3,cr4 = st.columns([1,1,1,1])

        for i, (col, (nama, pct)) in enumerate(zip([cr1,cr2,cr3], top5[:3])):
            ti = TANAMAN.get(nama,{"e":"🌿","s":"-","d":"-"})
            rank_cls = ["rec-rank1","rec-rank2","rec-rank3"][i]
            rank_lbl = ["🥇 Rekomendasi Utama","🥈 Alternatif 1","🥉 Alternatif 2"][i]
            opacity  = ["ff","cc","99"][i]
            with col:
                st.markdown(f"""
                <div class="rec-card {rank_cls}" style="opacity:{1 if i==0 else .85 if i==1 else .7}">
                  <div class="rec-label">{rank_lbl}</div>
                  <span class="rec-emoji">{ti["e"]}</span>
                  <div class="rec-name">{nama.upper()}</div>
                  <div class="rec-pct">{pct:.0%}</div>
                  <div class="rec-desc">{ti["d"]}</div>
                  <span class="rec-musim">Musim {ti["s"]}</span>
                </div>""", unsafe_allow_html=True)

        # Top-5 bar di kolom ke-4
        with cr4:
            st.markdown("""<div class="wcard" style="height:100%;padding:16px 18px">
            <div style="font-size:.78rem;font-weight:700;color:#14532d;margin-bottom:10px;
                 text-transform:uppercase;letter-spacing:.5px">Semua Kandidat</div>""",
                        unsafe_allow_html=True)
            W5 = ["#15803d","#16a34a","#22c55e","#86efac","#dcfce7"]
            for i,(n,p) in enumerate(top5):
                ti = TANAMAN.get(n,{"e":"🌿"})
                st.markdown(
                    f'<div class="pbar-head"><span>{ti["e"]} {n.capitalize()}</span>'
                    f'<span class="pbar-pct">{p:.0%}</span></div>'
                    + bar(p*100, W5[i]),
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW 2: Zona + Status + Radar ──
        st.markdown('<div class="section-title">🗺️ Zona Ekologi & Kondisi Lahan</div>',
                    unsafe_allow_html=True)
        cz1,cz2,cz3 = st.columns([1,1.4,1.6])

        # Zona card
        with cz1:
            st.markdown(f"""
            <div class="zona-hero" style="background:linear-gradient(135deg,{dark_z},{warna_z});color:white">
              <span class="zona-icon">🗺️</span>
              <div class="zona-nomor">Zona Ekologi Lahan Anda</div>
              <div class="zona-nama">{nz}</div>
              <div style="margin:10px 0;height:1px;background:rgba(255,255,255,.2)"></div>
              <div class="zona-desc">Curah hujan rata-rata {raw[zona_id][6]:.0f}mm ·
                Kelembaban {raw[zona_id][4]:.0f}%</div>
              <div style="margin-top:12px;font-size:.72rem;opacity:.8">
                Tanaman umum di zona ini:<br>
                <b>{" · ".join([TANAMAN.get(c,{}).get("e","🌿")+" "+c.capitalize() for c in top_z.index])}</b>
              </div>
            </div>""", unsafe_allow_html=True)

        # Status parameter
        with cz2:
            ideal = df[df["label"]==pred][FITUR].mean()
            vals  = [N,P,K,T,H,PH,R]
            rows  = ""
            for f,v in zip(FITUR,vals):
                teks,cls = cek_param(v, ideal[f])
                rows += (f'<div class="status-row">'
                         f'<span class="status-key">{F_LONG[f]}</span>'
                         f'<span class="status-val">{v} {F_SAT[f]}</span>'
                         f'<span class="{cls}">{teks}</span></div>')

            st.markdown(f"""
            <div class="wcard" style="padding:0;overflow:hidden">
              <div style="background:#f0fdf4;padding:12px 16px;font-size:.78rem;font-weight:700;
                   color:#14532d;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #dcfce7">
                📋 Status Kondisi vs Profil Ideal {TANAMAN.get(pred,{}).get("e","")} {pred.capitalize()}
              </div>
              {rows}
            </div>""", unsafe_allow_html=True)

        # Radar chart
        with cz3:
            norm  = lambda v,f: (v-F_MIN[f])/(F_MAX[f]-F_MIN[f])*100
            lbl   = [F_NAMA[f] for f in FITUR]
            ideal_norm = [norm(ideal[f],f) for f in FITUR]
            input_norm = [norm(v,f) for v,f in zip(vals,FITUR)]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=ideal_norm+[ideal_norm[0]], theta=lbl+[lbl[0]],
                fill="toself", name=f"Profil Ideal ({pred.capitalize()})",
                line=dict(color="#16a34a",width=2),
                fillcolor="rgba(22,163,74,.15)"))
            fig_r.add_trace(go.Scatterpolar(
                r=input_norm+[input_norm[0]], theta=lbl+[lbl[0]],
                fill="toself", name="Lahan Anda",
                line=dict(color="#f59e0b",width=2.5),
                fillcolor="rgba(245,158,11,.12)"))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True,range=[0,100],
                                    tickfont=dict(size=8),gridcolor="#dcfce7"),
                    bgcolor="white"),
                showlegend=True,
                legend=dict(font=dict(size=10),orientation="h",y=-.05),
                height=290, margin=dict(t=10,b=30,l=10,r=10),
                paper_bgcolor="white")
            st.plotly_chart(fig_r, use_container_width=True)

        # ── Semua Zona ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">🌏 Semua Zona Ekologi</div>',
                    unsafe_allow_html=True)
        cols_z = st.columns(4)
        for i in range(4):
            nz_i  = nama_zona(raw[i],i)
            wz_i  = wz(i); wd_i = wd(i)
            aktif = (i == zona_id)
            bdr   = f"3px solid {wz_i}" if aktif else f"1px solid {wz_i}40"
            bg    = f"linear-gradient(135deg,{wd_i}22,{wz_i}18)" if aktif else "white"
            with cols_z[i]:
                label_aktif = f'<div style="font-size:.68rem;font-weight:700;color:{wz_i};margin-bottom:4px">📍 LAHAN ANDA</div>' if aktif else ""
                st.markdown(f"""
                <div style="border:{bdr};background:{bg};border-radius:14px;
                     padding:16px;text-align:center;height:100%">
                  {label_aktif}
                  <div style="font-size:1.4rem">{["🌧️","🌵","💎","🌿"][i]}</div>
                  <div style="font-size:.88rem;font-weight:700;color:#1f2937;margin-top:6px">{nz_i}</div>
                  <div style="font-size:.72rem;color:#6b7280;margin-top:6px;line-height:1.6">
                    Hujan {raw[i][6]:.0f}mm<br>Lembab {raw[i][4]:.0f}%<br>Suhu {raw[i][3]:.0f}°C
                  </div>
                </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PETA ZONA
# ══════════════════════════════════════════════════════════════
elif menu == "🗺️  Peta Zona Ekologi":
    st.markdown("""
    <div class="page-banner">
      <h1>🗺️ Peta Zona Ekologi Lahan</h1>
      <p>K-Means Clustering mengelompokkan 2.200 lahan ke dalam zona berdasarkan
         kesamaan profil tanah & iklim. Kenali zona lahan Anda untuk strategi tanam lebih tepat.</p>
    </div>""", unsafe_allow_html=True)

    k_val = st.sidebar.slider("Jumlah Zona (K)",2,8,4)
    km,km_sc,km_lb,sil,X_pca,raw = latih_cluster(df, k_val)
    df_z = df.copy(); df_z["Zona"]=km_lb
    df_z["NamaZona"] = [nama_zona(raw[i],i) for i in km_lb]
    df_z["PCA1"]=X_pca[:,0]; df_z["PCA2"]=X_pca[:,1]

    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="kpi"><div class="kpi-val">{k_val}</div><div class="kpi-lbl">Jumlah Zona</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi"><div class="kpi-val">{sil:.3f}</div><div class="kpi-lbl">Silhouette Score</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi"><div class="kpi-val">{"Baik ✅" if sil>.3 else "Cukup ⚠️"}</div><div class="kpi-lbl">Kualitas Cluster</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📍 Visualisasi","📋 Profil Zona","📈 Elbow Method","🔍 Identifikasi"])

    with tab1:
        st.markdown('<div class="tips">Setiap titik = 1 data lahan. Warna berbeda = zona berbeda.</div>', unsafe_allow_html=True)
        fig=px.scatter(df_z,x="PCA1",y="PCA2",color="NamaZona",
                       hover_data={"label":True,"PCA1":False,"PCA2":False},
                       color_discrete_sequence=px.colors.qualitative.Set2,height=500)
        fig.update_traces(marker=dict(size=5,opacity=.75))
        fig.update_layout(plot_bgcolor="#fafff9",paper_bgcolor="white",
                          legend=dict(title="Zona",font=dict(size=11)))
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        profil=df_z.groupby("Zona")[FITUR].mean().round(1)
        profil.index=[nama_zona(raw[i],i) for i in profil.index]
        profil.columns=[F_LONG[f] for f in FITUR]
        st.dataframe(profil,use_container_width=True)
        st.markdown("#### Persebaran Tanaman per Zona")
        cd=pd.crosstab(df_z["NamaZona"],df_z["label"])
        fig2=px.imshow(cd,text_auto=True,color_continuous_scale="YlGn",height=320)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        st.markdown('<div class="tips">Cari titik "siku" pada grafik Inertia — itulah K terbaik.</div>', unsafe_allow_html=True)
        iner,sils=[],[]
        Xs=km_sc.transform(df[FITUR])
        for k in range(2,11):
            kt=KMeans(n_clusters=k,random_state=42,n_init=10); lb=kt.fit_predict(Xs)
            iner.append(kt.inertia_); sils.append(silhouette_score(Xs,lb))
        fig3=make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Scatter(x=list(range(2,11)),y=iner,mode="lines+markers",name="Inertia",
                                  line=dict(color="#15803d",width=2.5),marker=dict(size=8)),secondary_y=False)
        fig3.add_trace(go.Scatter(x=list(range(2,11)),y=sils,mode="lines+markers",name="Silhouette",
                                  line=dict(color="#f59e0b",width=2.5,dash="dash"),marker=dict(size=8)),secondary_y=True)
        fig3.update_xaxes(title_text="Jumlah Zona (K)")
        fig3.update_yaxes(title_text="Inertia",secondary_y=False)
        fig3.update_yaxes(title_text="Silhouette Score",secondary_y=True)
        fig3.update_layout(height=380,plot_bgcolor="#fafff9",paper_bgcolor="white")
        st.plotly_chart(fig3,use_container_width=True)
        st.info(f"💡 K optimal = **{list(range(2,11))[np.argmax(sils)]}** (Silhouette tertinggi: {max(sils):.3f})")

    with tab4:
        st.markdown("#### Masukkan kondisi lahan untuk mengetahui zonanya")
        c1,c2=st.columns(2)
        with c1:
            N2=st.slider("N",0,140,50,key="z2_n"); P2=st.slider("P",5,145,53,key="z2_p")
            K2=st.slider("K",5,205,48,key="z2_k"); T2=st.slider("Suhu",8.0,44.0,25.0,key="z2_t")
        with c2:
            H2=st.slider("Kelembaban",14.0,100.0,71.0,key="z2_h")
            PH2=st.slider("pH",3.5,10.0,6.5,key="z2_ph"); R2=st.slider("Curah Hujan",20.0,300.0,100.0,key="z2_r")
        if st.button("🗺️ Identifikasi Zona",use_container_width=True,key="btn_z2"):
            inp2=np.array([[N2,P2,K2,T2,H2,PH2,R2]]); z=km.predict(km_sc.transform(inp2))[0]
            nz=nama_zona(raw[z],z); wz_=wz(z)
            st.success(f"**{nz}** · Hujan rata-rata {raw[z][6]:.0f}mm · Kelembaban {raw[z][4]:.0f}%")
            top_zc=df_z[df_z["Zona"]==z]["label"].value_counts().head(5)
            st.markdown("Tanaman umum: " + "  ·  ".join([f"{TANAMAN.get(c,{}).get('e','🌿')} {c.capitalize()}" for c in top_zc.index]))


# ══════════════════════════════════════════════════════════════
#  ANALISIS & MODEL
# ══════════════════════════════════════════════════════════════
elif menu == "📊  Analisis & Model":
    st.markdown("""
    <div class="page-banner">
      <h1>📊 Analisis Data & Perbandingan Model</h1>
      <p>Evaluasi 6 model klasifikasi yang diuji pada dataset ini, beserta eksplorasi data dan feature importance.</p>
    </div>""", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["🏆 Perbandingan 6 Model","🔥 Feature Importance","🔗 Korelasi","📦 Distribusi Data"])

    with tab1:
        # KPI row
        c1,c2,c3,c4 = st.columns(4)
        rf_h = hasil_model["Random Forest"]
        c1.markdown(f'<div class="kpi"><div class="kpi-val">{rf_h["acc"]:.2%}</div><div class="kpi-lbl">Akurasi RF</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi"><div class="kpi-val">{rf_h["f1"]:.2%}</div><div class="kpi-lbl">F1-Score RF</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi"><div class="kpi-val">{rf_h["cv_mean"]:.2%}</div><div class="kpi-lbl">CV Mean (5-fold)</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="kpi"><div class="kpi-val">6</div><div class="kpi-lbl">Model Diuji</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabel perbandingan
        best_acc = max(v["acc"] for v in hasil_model.values())
        rows_tbl = ""
        sorted_models = sorted(hasil_model.items(), key=lambda x:-x[1]["acc"])
        for rank,(nama,h) in enumerate(sorted_models,1):
            is_best = h["acc"]==best_acc
            badge = '<span class="model-badge">⭐ Terbaik</span>' if is_best else ""
            cls   = 'class="best"' if is_best else ''
            rows_tbl += f"""<tr>
              <td><b>#{rank}</b></td>
              <td {cls}>{nama} {badge}</td>
              <td {cls}>{h["acc"]:.2%}</td>
              <td>{h["prec"]:.2%}</td>
              <td>{h["rec"]:.2%}</td>
              <td>{h["f1"]:.2%}</td>
              <td>{h["cv_mean"]:.2%} ±{h["cv_std"]:.2%}</td>
            </tr>"""

        st.markdown(f"""
        <div class="wcard" style="padding:0;overflow:hidden">
          <table class="model-tbl">
            <thead><tr>
              <th>#</th><th>Model</th><th>Akurasi</th>
              <th>Precision</th><th>Recall</th><th>F1-Score</th><th>CV Score (5-fold)</th>
            </tr></thead>
            <tbody>{rows_tbl}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Bar chart perbandingan
        names  = [n for n,_ in sorted_models]
        accs   = [h["acc"]*100   for _,h in sorted_models]
        f1s    = [h["f1"]*100    for _,h in sorted_models]
        cvs    = [h["cv_mean"]*100 for _,h in sorted_models]
        x = np.arange(len(names)); w = 0.25
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(x=list(range(len(names))),y=accs,name="Akurasi",
                                 marker_color="#15803d",width=w,
                                 text=[f"{v:.1f}%" for v in accs],textposition="outside"))
        fig_cmp.add_trace(go.Bar(x=[i+w for i in range(len(names))],y=f1s,name="F1-Score",
                                 marker_color="#0ea5e9",width=w,
                                 text=[f"{v:.1f}%" for v in f1s],textposition="outside"))
        fig_cmp.add_trace(go.Bar(x=[i+2*w for i in range(len(names))],y=cvs,name="CV Score",
                                 marker_color="#f59e0b",width=w,
                                 text=[f"{v:.1f}%" for v in cvs],textposition="outside"))
        fig_cmp.update_layout(
            xaxis=dict(tickvals=[i+w for i in range(len(names))],ticktext=names,tickangle=-15),
            yaxis=dict(title="Score (%)",range=[75,103]),
            barmode="group",height=380,
            plot_bgcolor="#fafff9",paper_bgcolor="white",
            legend=dict(orientation="h",y=1.05))
        st.plotly_chart(fig_cmp,use_container_width=True)

        # Confusion matrix RF
        st.markdown("#### Confusion Matrix — Random Forest")
        fig_cm,ax=plt.subplots(figsize=(12,10))
        sns.heatmap(cm_rf,annot=True,fmt="d",cmap="Greens",
                    xticklabels=kelas,yticklabels=kelas,ax=ax,linewidths=.5)
        ax.set_xlabel("Prediksi",fontsize=11); ax.set_ylabel("Aktual",fontsize=11)
        ax.tick_params(axis="x",rotation=45,labelsize=8); ax.tick_params(axis="y",labelsize=8)
        plt.tight_layout(); st.pyplot(fig_cm)

    with tab2:
        st.markdown('<div class="tips">Semakin panjang batangnya, semakin besar pengaruh parameter tersebut dalam menentukan rekomendasi tanaman.</div>', unsafe_allow_html=True)
        fi_df=pd.DataFrame({"Fitur":[F_LONG[f] for f in FITUR],"Skor":fi}).sort_values("Skor",ascending=True)
        fig_fi=go.Figure(go.Bar(x=fi_df["Skor"]*100,y=fi_df["Fitur"],orientation="h",
                                marker_color="#15803d",
                                text=[f"{v:.1f}%" for v in fi_df["Skor"]*100],textposition="outside"))
        fig_fi.update_layout(xaxis_title="Tingkat Pengaruh (%)",height=360,
                             plot_bgcolor="#fafff9",paper_bgcolor="white",
                             margin=dict(t=10,b=10,r=90))
        st.plotly_chart(fig_fi,use_container_width=True)

    with tab3:
        st.markdown('<div class="tips">Merah = keduanya naik bersama. Biru = satu naik, satu turun. Angka mendekati ±1 = hubungan kuat.</div>', unsafe_allow_html=True)
        corr=df[FITUR].corr()
        corr.columns=[F_LONG[f] for f in FITUR]; corr.index=[F_LONG[f] for f in FITUR]
        fig_c=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,height=480)
        st.plotly_chart(fig_c,use_container_width=True)

    with tab4:
        feat_sel=st.selectbox("Pilih Parameter",[F_LONG[f] for f in FITUR])
        feat_key=[f for f in FITUR if F_LONG[f]==feat_sel][0]
        fig_bx=px.box(df,x="label",y=feat_key,color="label",
                      color_discrete_sequence=px.colors.qualitative.Set2,
                      labels={feat_key:feat_sel,"label":"Tanaman"},height=500)
        fig_bx.update_layout(showlegend=False,xaxis_tickangle=-45,
                             plot_bgcolor="#fafff9",paper_bgcolor="white")
        st.plotly_chart(fig_bx,use_container_width=True)
        stats=df.groupby("label")[feat_key].agg(["min","mean","max"]).round(1)
        stats.columns=["Minimum","Rata-rata","Maksimum"]
        st.dataframe(stats,use_container_width=True,height=280)


# ══════════════════════════════════════════════════════════════
#  SIMULASI LANJUTAN
# ══════════════════════════════════════════════════════════════
elif menu == "🔬  Simulasi Lanjutan":
    st.markdown("""
    <div class="page-banner">
      <h1>🔬 Simulasi & Analisis Sensitivitas</h1>
      <p>Simulasikan perubahan parameter dan analisis seberapa sensitif rekomendasi terhadap perubahan kondisi lahan.</p>
    </div>""", unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["🔄 What-If: Ubah Satu Parameter","📡 Sensitivitas Semua Parameter"])

    with tab1:
        st.markdown('<div class="tips">Atur kondisi dasar, pilih satu parameter yang ingin diubah-ubah, dan lihat bagaimana rekomendasi berubah.</div>', unsafe_allow_html=True)
        st.markdown("**Kondisi Dasar Lahan**")
        c1,c2,c3,c4=st.columns(4)
        bN=c1.number_input("Nitrogen (N)",0,140,50,key="wN"); bP=c1.number_input("Fosfor (P)",5,145,53,key="wP")
        bK=c2.number_input("Kalium (K)",5,205,48,key="wK"); bT=c2.number_input("Suhu (°C)",8.0,44.0,25.0,key="wT")
        bH=c3.number_input("Kelembaban (%)",14.0,100.0,71.0,key="wH"); bPH=c3.number_input("pH",3.5,10.0,6.5,key="wPH")
        bR=c4.number_input("Curah Hujan (mm)",20.0,300.0,100.0,key="wR")

        base=[bN,bP,bK,bT,bH,bPH,bR]
        bpred=rf_model.predict(sc.transform([base]))[0]
        bconf=rf_model.predict_proba(sc.transform([base]))[0][list(kelas).index(bpred)]
        st.info(f"Rekomendasi dasar: {TANAMAN.get(bpred,{}).get('e','🌿')} **{bpred.upper()}** — Kecocokan {bconf:.0%}")

        col1_,col2_=st.columns([1,1])
        with col1_:
            pu=st.selectbox("Parameter yang Disimulasikan",[F_LONG[f] for f in FITUR])
        with col2_:
            ns=st.slider("Jumlah Titik Simulasi",5,30,15)
        pk=[f for f in FITUR if F_LONG[f]==pu][0]; idx_p=FITUR.index(pk)
        vals=np.linspace(df[pk].min(),df[pk].max(),ns)
        rows=[]
        for v in vals:
            inp2=base.copy(); inp2[idx_p]=v; sc2=sc.transform([inp2])
            pd2=rf_model.predict(sc2)[0]; pp2=rf_model.predict_proba(sc2)[0]
            rows.append({"Nilai":round(v,2),"Tanaman":pd2,"Kecocokan":pp2.max()})
        dw=pd.DataFrame(rows)

        cl,cr=st.columns([2,1])
        with cl:
            fig=go.Figure()
            for crop in dw["Tanaman"].unique():
                sub=dw[dw["Tanaman"]==crop]; ei=TANAMAN.get(crop,{"e":"🌿"})
                fig.add_trace(go.Scatter(x=sub["Nilai"],y=sub["Kecocokan"]*100,
                    mode="lines+markers",name=f"{ei['e']} {crop}",line=dict(width=2.5),marker=dict(size=8)))
            fig.update_layout(xaxis_title=pu,yaxis_title="Kecocokan (%)",
                              height=380,plot_bgcolor="#fafff9",paper_bgcolor="white")
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            disp=dw.copy()
            disp["Kecocokan"]=disp["Kecocokan"].map(lambda x:f"{x:.0%}")
            disp["Tanaman"]=disp["Tanaman"].apply(lambda c:f"{TANAMAN.get(c,{}).get('e','🌿')} {c.capitalize()}")
            st.dataframe(disp,use_container_width=True,height=380)

        perubahan=[{"Dari":dw.loc[i-1,"Tanaman"],"Ke":dw.loc[i,"Tanaman"],pu:dw.loc[i,"Nilai"]}
                   for i in range(1,len(dw)) if dw.loc[i,"Tanaman"]!=dw.loc[i-1,"Tanaman"]]
        if perubahan:
            st.warning(f"⚡ **{len(perubahan)} titik perubahan** rekomendasi ditemukan:")
            st.dataframe(pd.DataFrame(perubahan),use_container_width=True)
        else:
            st.success("✅ Rekomendasi stabil di seluruh rentang nilai parameter ini.")

    with tab2:
        st.markdown('<div class="tips">Heatmap menunjukkan kecocokan untuk tanaman rekomendasi dasar ketika setiap parameter diubah ±10%/±20%/±30%. Warna merah = kecocokan turun drastis.</div>', unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        sN=c1.slider("N",0,140,50,key="sN"); sP=c1.slider("P",5,145,53,key="sP")
        sK=c2.slider("K",5,205,48,key="sK"); sT=c2.slider("Suhu",8.0,44.0,25.0,key="sT")
        sH=c3.slider("Kelembaban",14.0,100.0,71.0,key="sH"); sPH=c3.slider("pH",3.5,10.0,6.5,key="sPH")
        sR=c3.slider("Curah Hujan",20.0,300.0,100.0,key="sR")
        base_s=[sN,sP,sK,sT,sH,sPH,sR]
        bp=rf_model.predict(sc.transform([base_s]))[0]
        bpr=rf_model.predict_proba(sc.transform([base_s]))[0][list(kelas).index(bp)]
        st.info(f"Rekomendasi dasar: {TANAMAN.get(bp,{}).get('e','🌿')} **{bp.upper()}** — Kecocokan {bpr:.0%}")

        pcts=[-30,-20,-10,0,10,20,30]; sens={}
        for i,f in enumerate(FITUR):
            vals=[]
            for pct in pcts:
                inp3=base_s.copy(); inp3[i]*=(1+pct/100)
                p3=rf_model.predict_proba(sc.transform([inp3]))[0][list(kelas).index(bp)]
                vals.append(p3*100)
            sens[f]=vals
        sens_df=pd.DataFrame(sens,index=[f"{p:+d}%" for p in pcts])
        sens_df.columns=[F_LONG[f] for f in FITUR]
        fig_h=px.imshow(sens_df,text_auto=".0f",color_continuous_scale="RdYlGn",
                        height=340,zmin=0,zmax=100,labels=dict(color="Kecocokan (%)"))
        fig_h.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_h,use_container_width=True)
        skor={F_LONG[f]:round(np.std(sens[f]),2) for f in FITUR}
        rank=pd.DataFrame(list(skor.items()),columns=["Parameter","Skor Sensitivitas"]
                          ).sort_values("Skor Sensitivitas",ascending=False)
        rank["Level"]=rank["Skor Sensitivitas"].apply(
            lambda x:"🔴 Sangat Sensitif" if x>10 else("🟡 Cukup Sensitif" if x>5 else "🟢 Stabil"))
        st.dataframe(rank,use_container_width=True,hide_index=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:.78rem;padding:6px'>"
    "🌾 AgroSPK · Kelompok 5 · Sistem Pendukung Keputusan (DSS) · "
    "Dataset: Crop Recommendation — 2.200 sampel, 22 tanaman, 6 model"
    "</div>", unsafe_allow_html=True)
