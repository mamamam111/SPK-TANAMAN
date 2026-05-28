"""
SPK Rekomendasi Tanaman — Kelompok 5
Clean UI: ramah petani + analisis mendalam
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, silhouette_score
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSPK — Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.main { background: #F7FDF9; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 60%, #1B4332 100%);
}
section[data-testid="stSidebar"] * { color: #D8F3DC !important; }

.hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
    border-radius: 20px; padding: 36px 40px; color: white; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero::after { content:"🌾"; position:absolute; font-size:120px; opacity:.07; right:30px; top:-10px; }
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0 0 8px; }
.hero p  { font-size: 1rem; opacity: 0.85; margin: 0; max-width: 560px; }

.hasil-utama {
    background: linear-gradient(135deg, #1B4332, #2D6A4F);
    border-radius: 20px; padding: 28px; text-align: center; color: white; margin-bottom: 20px;
}
.hasil-emoji { font-size: 3.5rem; display: block; margin-bottom: 6px; }
.hasil-nama  { font-size: 2rem; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; }
.hasil-cocok { font-size: 1rem; opacity: 0.85; margin-top: 4px; }
.hasil-tag   {
    display:inline-block; background:rgba(255,255,255,.15);
    border-radius:20px; padding:4px 14px; font-size:.8rem; margin:8px 4px 0;
}

.stat-box {
    background:white; border-radius:12px; padding:16px; text-align:center;
    border:1px solid #E8F5EE; box-shadow:0 1px 6px rgba(45,106,79,.06);
}
.stat-angka { font-size:1.8rem; font-weight:800; color:#2D6A4F; }
.stat-label { font-size:.75rem; color:#6B7280; margin-top:2px; text-transform:uppercase; letter-spacing:.5px; }

.section-hd {
    font-size:1.25rem; font-weight:800; color:#1B4332;
    margin:24px 0 12px; padding-bottom:8px; border-bottom:2px solid #D8F3DC;
}
.section-sub { font-size:.83rem; color:#6B7280; margin-bottom:16px; }

.badge-ok   { background:#D8F3DC; color:#1B4332; border-radius:20px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.badge-warn { background:#FFF3CD; color:#856404; border-radius:20px; padding:3px 10px; font-size:.75rem; font-weight:600; }
.badge-bad  { background:#FFE0E0; color:#922B21; border-radius:20px; padding:3px 10px; font-size:.75rem; font-weight:600; }

.prob-track { background:#F0FFF4; border-radius:6px; height:10px; overflow:hidden; margin-bottom:10px; }
.prob-fill  { height:100%; border-radius:6px; }

.tabel-param { width:100%; border-collapse:collapse; font-size:.83rem; }
.tabel-param th { background:#F0FFF4; color:#1B4332; padding:10px 12px; text-align:left; font-size:.75rem; text-transform:uppercase; }
.tabel-param td { padding:9px 12px; border-bottom:1px solid #F0FFF4; color:#374151; }

.tips {
    background:#EBF8F0; border-left:4px solid #40916C;
    border-radius:0 10px 10px 0; padding:12px 16px;
    font-size:.83rem; color:#1B4332; margin:8px 0 16px;
}

.stButton button {
    background: linear-gradient(135deg, #2D6A4F, #40916C) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 12px 24px !important;
}
.stButton button:hover {
    box-shadow: 0 6px 20px rgba(45,106,79,.35) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────────────────────
FITUR    = ["N","P","K","temperature","humidity","ph","rainfall"]
F_NAMA   = {"N":"Nitrogen (N)","P":"Fosfor (P)","K":"Kalium (K)",
            "temperature":"Suhu Udara","humidity":"Kelembaban",
            "ph":"Keasaman (pH)","rainfall":"Curah Hujan"}
F_SATUAN = {"N":"ppm","P":"ppm","K":"ppm","temperature":"°C",
            "humidity":"%","ph":"","rainfall":"mm"}
F_MIN    = {"N":0,"P":5,"K":5,"temperature":8.0,"humidity":14.0,"ph":3.5,"rainfall":20.0}
F_MAX    = {"N":140,"P":145,"K":205,"temperature":44.0,"humidity":100.0,"ph":10.0,"rainfall":300.0}
F_DEF    = {"N":50,"P":53,"K":48,"temperature":25.0,"humidity":71.0,"ph":6.5,"rainfall":100.0}

TANAMAN = {
    "rice":{"emoji":"🌾","musim":"Basah","singkat":"Padi sawah, butuh banyak air"},
    "maize":{"emoji":"🌽","musim":"Kering","singkat":"Jagung, lahan kering"},
    "chickpea":{"emoji":"🫘","musim":"Dingin","singkat":"Kacang arab, toleran kering"},
    "kidneybeans":{"emoji":"🫘","musim":"Hangat","singkat":"Kacang merah"},
    "pigeonpeas":{"emoji":"🌿","musim":"Kering","singkat":"Kacang gude"},
    "mothbeans":{"emoji":"🌿","musim":"Kering","singkat":"Sangat toleran kering"},
    "mungbean":{"emoji":"🫛","musim":"Hangat","singkat":"Kacang hijau, siklus pendek"},
    "blackgram":{"emoji":"🫘","musim":"Hangat","singkat":"Kacang urad, toleran panas"},
    "lentil":{"emoji":"🫘","musim":"Dingin","singkat":"Lentil, tanah lempung"},
    "pomegranate":{"emoji":"🍎","musim":"Panas","singkat":"Delima, buah komersial"},
    "banana":{"emoji":"🍌","musim":"Basah","singkat":"Pisang, kelembaban tinggi"},
    "mango":{"emoji":"🥭","musim":"Panas","singkat":"Mangga, buah tropis"},
    "grapes":{"emoji":"🍇","musim":"Panas","singkat":"Anggur, drainase baik"},
    "watermelon":{"emoji":"🍉","musim":"Panas","singkat":"Semangka, tanah berpasir"},
    "muskmelon":{"emoji":"🍈","musim":"Panas","singkat":"Melon, sinar matahari penuh"},
    "apple":{"emoji":"🍏","musim":"Dingin","singkat":"Apel, dataran tinggi"},
    "orange":{"emoji":"🍊","musim":"Hangat","singkat":"Jeruk, curah hujan teratur"},
    "papaya":{"emoji":"🍑","musim":"Basah","singkat":"Pepaya, tumbuh cepat"},
    "coconut":{"emoji":"🥥","musim":"Basah","singkat":"Kelapa, tanah pantai"},
    "cotton":{"emoji":"🌼","musim":"Panas","singkat":"Kapas, lahan kering"},
    "jute":{"emoji":"🌿","musim":"Basah","singkat":"Jute/rami"},
    "coffee":{"emoji":"☕","musim":"Basah","singkat":"Kopi, ketinggian sedang"},
}

# ──────────────────────────────────────────────────────────────
# LOAD & TRAIN
# ──────────────────────────────────────────────────────────────
@st.cache_data
def muat_data():
    return pd.read_csv("Crop_recommendation.csv")

@st.cache_resource
def latih_klasifikasi(_df):
    X = _df[FITUR]; y = _df["label"]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    m = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
    m.fit(X_tr_s,y_tr)
    y_p = m.predict(X_te_s)
    acc = accuracy_score(y_te,y_p)
    cm  = confusion_matrix(y_te,y_p,labels=m.classes_)
    cvs = cross_val_score(m,sc.transform(X),y,cv=5)
    return m, sc, acc, cm, cvs, m.classes_

@st.cache_resource
def latih_cluster(_df, k=4):
    X=_df[FITUR]; sc=StandardScaler(); X_s=sc.fit_transform(X)
    km=KMeans(n_clusters=k,random_state=42,n_init=15); lb=km.fit_predict(X_s)
    sil=silhouette_score(X_s,lb); pca=PCA(n_components=2,random_state=42)
    return km, sc, lb, sil, pca.fit_transform(X_s), km.cluster_centers_

def nama_zona(center_raw, idx):
    N,P,K,temp,hum,ph,rain = center_raw
    if rain>160 and hum>75: return f"Zona {idx} — Tropis Basah 🌧️"
    elif rain<80 and hum<55: return f"Zona {idx} — Semi-Kering 🌵"
    elif K>100 or P>100:     return f"Zona {idx} — Subur Mineral 💎"
    else:                    return f"Zona {idx} — Komersial Sedang 🌿"

def warna_zona(i):
    return ["#0EA5E9","#F59E0B","#A855F7","#22C55E","#EF4444","#06B6D4","#F97316","#84CC16"][i%8]

def cek_status(nilai, ideal):
    pct = abs(nilai-ideal)/(ideal+1e-9)*100
    if pct<=20: return "✅ Sesuai","badge-ok"
    elif pct<=40: return "⚠️ Perlu Perhatian","badge-warn"
    else: return "❌ Perlu Perbaikan","badge-bad"

def bar_html(pct, warna="#2D6A4F"):
    return f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%;background:{warna}"></div></div>'

df = muat_data()
model,scaler,akurasi,cm,cv_scores,kelas = latih_klasifikasi(df)
importances = model.feature_importances_

# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌾 AgroSPK")
    st.markdown("<small style='opacity:.7'>Rekomendasi Tanaman Berbasis AI<br>Kelompok 5 — DSS</small>",
                unsafe_allow_html=True)
    st.markdown("---")
    menu = st.radio("", [
        "🏠  Beranda",
        "🌱  Cek Lahan Saya",
        "🗺️  Peta Zona Lahan",
        "📊  Analisis Data",
        "🔬  Analisis Lanjutan",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f"**Model:** Random Forest (200 pohon)")
    st.markdown(f"**Akurasi:** `{akurasi:.2%}`")
    st.markdown(f"**CV 5-fold:** `{cv_scores.mean():.2%} ± {cv_scores.std():.2%}`")
    st.markdown("---")
    st.markdown(f"📁 **{len(df):,}** sampel · **22** tanaman")


# ══════════════════════════════════════════════════════════════
# BERANDA
# ══════════════════════════════════════════════════════════════
if menu == "🏠  Beranda":
    st.markdown("""
    <div class="hero">
      <h1>Selamat Datang di AgroSPK 🌾</h1>
      <p>Masukkan kondisi tanah dan iklim lahan Anda — sistem ini merekomendasikan tanaman yang paling
         cocok berdasarkan data dari 2.200 lahan pertanian.</p>
    </div>""", unsafe_allow_html=True)

    for col, angka, label in zip(st.columns(4),
        ["2.200","22",f"{akurasi:.0%}","4"],
        ["Data Lahan Latih","Jenis Tanaman","Akurasi Model","Zona Ekologi"]):
        col.markdown(f'<div class="stat-box"><div class="stat-angka">{angka}</div>'
                     f'<div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown("""<div style="background:white;border-radius:16px;padding:24px;border:1px solid #E8F5EE">
          <h3 style="color:#1B4332;margin:0 0 12px">🎯 Cara Pakai</h3>
          <ol style="color:#374151;line-height:2.2;font-size:.9rem;padding-left:18px">
            <li>Buka menu <b>🌱 Cek Lahan Saya</b></li>
            <li>Geser slider sesuai kondisi tanah & iklim lahan</li>
            <li>Klik tombol <b>Rekomendasikan</b></li>
            <li>Lihat 3 tanaman terbaik dan persentase kecocoknya</li>
            <li>Periksa juga zona ekologi lahan Anda</li>
          </ol></div>""", unsafe_allow_html=True)
    with cr:
        st.markdown("""<div style="background:white;border-radius:16px;padding:24px;border:1px solid #E8F5EE">
          <h3 style="color:#1B4332;margin:0 0 12px">💡 Fitur Aplikasi</h3>
          <div style="color:#374151;font-size:.88rem;line-height:2.2">
            ✅ Rekomendasi 3 tanaman terbaik + persentase kecocokan<br>
            ✅ Identifikasi zona ekologi lahan secara otomatis<br>
            ✅ Perbandingan kondisi lahan vs profil ideal tanaman<br>
            ✅ Simulasi "bagaimana jika" satu parameter diubah<br>
            ✅ Analisis sensitivitas seluruh parameter sekaligus
          </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-hd">🌿 22 Tanaman yang Didukung</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i,(nama,info) in enumerate(TANAMAN.items()):
        with cols[i%6]:
            st.markdown(f"""<div style="text-align:center;padding:10px 6px;background:white;
                border-radius:10px;border:1px solid #E8F5EE;margin-bottom:8px">
              <div style="font-size:1.6rem">{info['emoji']}</div>
              <div style="font-size:.76rem;font-weight:600;color:#1B4332">{nama.capitalize()}</div>
              <div style="font-size:.68rem;color:#6B7280">{info['musim']}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CEK LAHAN
# ══════════════════════════════════════════════════════════════
elif menu == "🌱  Cek Lahan Saya":
    st.markdown('<div class="section-hd">🌱 Cek Lahan Saya</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Geser slider sesuai hasil uji tanah dan kondisi iklim lahan Anda.</div>',
                unsafe_allow_html=True)

    st.markdown("#### ⚗️ Kondisi Tanah")
    c1,c2,c3 = st.columns(3)
    with c1: N = st.slider("🧪 Nitrogen (N) — ppm",0,140,50,help="Kandungan nitrogen, makin tinggi makin subur.")
    with c2: P = st.slider("🌱 Fosfor (P) — ppm",5,145,53,help="Penting untuk akar dan buah.")
    with c3: K = st.slider("💧 Kalium (K) — ppm",5,205,48,help="Meningkatkan kualitas dan ketahanan.")

    st.markdown("#### 🌤️ Kondisi Iklim")
    c4,c5,c6,c7 = st.columns(4)
    with c4: suhu   = st.slider("🌡️ Suhu (°C)",8.0,44.0,25.0,step=0.5)
    with c5: lembab = st.slider("💦 Kelembaban (%)",14.0,100.0,71.0,step=0.5)
    with c6: ph     = st.slider("⚗️ pH Tanah",3.5,10.0,6.5,step=0.1,help="7=netral, <7=asam, >7=basa")
    with c7: hujan  = st.slider("🌧️ Curah Hujan (mm)",20.0,300.0,100.0,step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Rekomendasikan Tanaman untuk Lahan Ini", use_container_width=True):

        inp    = np.array([[N,P,K,suhu,lembab,ph,hujan]])
        inp_sc = scaler.transform(inp)
        pred   = model.predict(inp_sc)[0]
        proba  = model.predict_proba(inp_sc)[0]
        top5   = [(kelas[i],proba[i]) for i in np.argsort(proba)[::-1][:5]]
        info   = TANAMAN.get(pred,{"emoji":"🌿","musim":"-","singkat":"-"})
        conf   = proba[list(kelas).index(pred)]

        st.markdown("---")
        col_h, col_d = st.columns([1,1])

        with col_h:
            # Kartu hasil utama
            st.markdown(f"""
            <div class="hasil-utama">
              <span class="hasil-emoji">{info['emoji']}</span>
              <div class="hasil-nama">{pred.upper()}</div>
              <div class="hasil-cocok">Tingkat Kecocokan: <b>{conf:.0%}</b></div>
              <span class="hasil-tag">🗓️ Musim {info['musim']}</span>
              <span class="hasil-tag">ℹ️ {info['singkat']}</span>
            </div>""", unsafe_allow_html=True)

            # Top-5 bar
            st.markdown("##### 🏆 Peringkat Kecocokan")
            WARNA = ["#1B4332","#2D6A4F","#40916C","#74C69D","#B7E4C7"]
            for i,(nama,pct) in enumerate(top5):
                ti = TANAMAN.get(nama,{"emoji":"🌿"})
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:.83rem;margin-bottom:4px">'
                    f'<span><b>#{i+1}</b> {ti["emoji"]} {nama.capitalize()}</span>'
                    f'<span style="font-weight:700;color:{WARNA[i]}">{pct:.0%}</span></div>'
                    + bar_html(pct*100, WARNA[i]),
                    unsafe_allow_html=True)

        with col_d:
            # Tabel status parameter
            st.markdown("##### 📋 Status Kondisi Lahan vs Profil Ideal")
            ideal = df[df["label"]==pred][FITUR].mean()
            vals  = [N,P,K,suhu,lembab,ph,hujan]
            rows  = "".join([
                f"<tr><td>{F_NAMA[f]}</td><td><b>{v} {F_SATUAN[f]}</b></td>"
                f"<td style='color:#6B7280'>{ideal[f]:.1f}</td>"
                f"<td><span class='{cek_status(v,ideal[f])[1]}'>{cek_status(v,ideal[f])[0]}</span></td></tr>"
                for f,v in zip(FITUR,vals)
            ])
            st.markdown(f"""<table class="tabel-param">
              <thead><tr><th>Parameter</th><th>Nilai Anda</th><th>Nilai Ideal</th><th>Status</th></tr></thead>
              <tbody>{rows}</tbody></table>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Radar
            norm = lambda v,f: (v-F_MIN[f])/(F_MAX[f]-F_MIN[f])*100
            fig  = go.Figure()
            lbl  = [F_NAMA[f] for f in FITUR]
            fig.add_trace(go.Scatterpolar(r=[norm(ideal[f],f) for f in FITUR],theta=lbl,
                fill="toself",name=f"Ideal ({pred})",line_color="#2D6A4F",fillcolor="rgba(45,106,79,.15)"))
            fig.add_trace(go.Scatterpolar(r=[norm(v,f) for v,f in zip(vals,FITUR)],theta=lbl,
                fill="toself",name="Lahan Anda",line_color="#F77F00",fillcolor="rgba(247,127,0,.1)"))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),
                              showlegend=True,height=280,margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        # Zona ekologi
        st.markdown("---")
        st.markdown("##### 🗺️ Zona Ekologi Lahan Anda")
        km,km_sc,km_lb,sil,_,centers = latih_cluster(df,4)
        raw   = km_sc.inverse_transform(centers)
        zona  = km.predict(km_sc.transform(inp))[0]
        df_z  = df.copy(); df_z["zona"] = km.predict(km_sc.transform(df[FITUR]))

        cols_z = st.columns(4)
        for i in range(4):
            nz = nama_zona(raw[i],i); wz = warna_zona(i)
            aktif = (i==zona)
            border = f"3px solid {wz}" if aktif else f"1px solid {wz}40"
            bg = f"{wz}18" if aktif else "white"
            with cols_z[i]:
                st.markdown(f"""<div style="border:{border};background:{bg};border-radius:14px;
                    padding:14px;text-align:center">
                  <div style="font-size:.7rem;font-weight:700;color:{wz};min-height:16px">
                    {'📍 Zona Anda' if aktif else ''}</div>
                  <div style="font-size:.9rem;font-weight:700;color:#1B4332;margin-top:4px">{nz}</div>
                  <div style="font-size:.72rem;color:#6B7280;margin-top:6px">
                    Hujan {raw[i][6]:.0f}mm · Lembab {raw[i][4]:.0f}%</div>
                </div>""", unsafe_allow_html=True)

        top_c = df_z[df_z["zona"]==zona]["label"].value_counts().head(6)
        st.markdown("**Tanaman yang umum berhasil di zona ini:**  " +
                    "  ·  ".join([f"{TANAMAN.get(c,{}).get('emoji','🌿')} {c.capitalize()}" for c in top_c.index]))


# ══════════════════════════════════════════════════════════════
# PETA ZONA
# ══════════════════════════════════════════════════════════════
elif menu == "🗺️  Peta Zona Lahan":
    st.markdown('<div class="section-hd">🗺️ Peta Zona Ekologi Lahan</div>', unsafe_allow_html=True)

    n_zona = st.sidebar.slider("Jumlah Zona (K)",2,8,4)
    km,km_sc,km_lb,sil,X_pca,centers = latih_cluster(df,n_zona)
    raw   = km_sc.inverse_transform(centers)
    df_z  = df.copy(); df_z["Zona"]=km_lb
    df_z["PCA1"]=X_pca[:,0]; df_z["PCA2"]=X_pca[:,1]
    df_z["Nama Zona"] = df_z["Zona"].apply(lambda i: nama_zona(raw[i],i))

    c1,c2,c3 = st.columns(3)
    c1.metric("Jumlah Zona",n_zona)
    c2.metric("Silhouette Score",f"{sil:.3f}")
    c3.metric("Kualitas","Baik ✅" if sil>0.30 else "Cukup ⚠️")
    st.markdown("---")

    tab1,tab2,tab3,tab4 = st.tabs(["📍 Visualisasi PCA","📋 Profil Zona","📈 Elbow Method","🔍 Cari Zona Lahan"])

    with tab1:
        st.markdown('<div class="section-sub">Setiap titik = 1 data lahan, warna = zona yang ditetapkan</div>',
                    unsafe_allow_html=True)
        fig = px.scatter(df_z,x="PCA1",y="PCA2",color="Nama Zona",hover_data={"label":True,"PCA1":False,"PCA2":False},
                         color_discrete_sequence=px.colors.qualitative.Set2,height=480)
        fig.update_traces(marker=dict(size=5,opacity=0.75))
        fig.update_layout(plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE")
        st.plotly_chart(fig,use_container_width=True)

    with tab2:
        profil = df_z.groupby("Zona")[FITUR].mean().round(1)
        profil.index = [nama_zona(raw[i],i) for i in profil.index]
        profil.columns = [F_NAMA[f] for f in FITUR]
        st.dataframe(profil,use_container_width=True)
        st.markdown("#### Persebaran Tanaman per Zona")
        cd = pd.crosstab(df_z["Nama Zona"],df_z["label"])
        fig2 = px.imshow(cd,text_auto=True,color_continuous_scale="YlGn",height=320)
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2,use_container_width=True)

    with tab3:
        st.markdown('<div class="tips">Cari titik "siku" pada grafik Inertia — itulah K optimal.</div>',
                    unsafe_allow_html=True)
        iner,sils = [],[]
        X_s = km_sc.transform(df[FITUR])
        for k in range(2,11):
            km_t=KMeans(n_clusters=k,random_state=42,n_init=10); lb=km_t.fit_predict(X_s)
            iner.append(km_t.inertia_); sils.append(silhouette_score(X_s,lb))
        fig3 = make_subplots(specs=[[{"secondary_y":True}]])
        fig3.add_trace(go.Scatter(x=list(range(2,11)),y=iner,mode="lines+markers",name="Inertia",
                                  line=dict(color="#2D6A4F",width=2.5),marker=dict(size=8)),secondary_y=False)
        fig3.add_trace(go.Scatter(x=list(range(2,11)),y=sils,mode="lines+markers",name="Silhouette",
                                  line=dict(color="#F59E0B",width=2.5,dash="dash"),marker=dict(size=8)),secondary_y=True)
        fig3.update_xaxes(title_text="Jumlah Zona (K)")
        fig3.update_yaxes(title_text="Inertia",secondary_y=False)
        fig3.update_yaxes(title_text="Silhouette Score",secondary_y=True)
        fig3.update_layout(height=360,plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE")
        st.plotly_chart(fig3,use_container_width=True)
        st.info(f"💡 K optimal: **K = {list(range(2,11))[np.argmax(sils)]}** (Silhouette tertinggi: {max(sils):.3f})")

    with tab4:
        st.markdown("#### Masukkan Data Lahan untuk Mengetahui Zonanya")
        c1,c2 = st.columns(2)
        with c1:
            N2=st.slider("N",0,140,50,key="z_n"); P2=st.slider("P",5,145,53,key="z_p")
            K2=st.slider("K",5,205,48,key="z_k"); T2=st.slider("Suhu",8.0,44.0,25.0,key="z_t")
        with c2:
            H2=st.slider("Kelembaban",14.0,100.0,71.0,key="z_h")
            PH2=st.slider("pH",3.5,10.0,6.5,key="z_ph"); R2=st.slider("Curah Hujan",20.0,300.0,100.0,key="z_r")
        if st.button("🗺️ Cari Zona",use_container_width=True,key="btn_z"):
            inp2=np.array([[N2,P2,K2,T2,H2,PH2,R2]]); z=km.predict(km_sc.transform(inp2))[0]
            nz=nama_zona(raw[z],z); wz=warna_zona(z)
            st.markdown(f"""<div style="background:{wz}18;border:2px solid {wz};border-radius:14px;
                padding:20px;text-align:center;margin-top:12px">
              <div style="font-size:1.4rem;font-weight:800;color:{wz}">{nz}</div>
              <div style="color:#374151;font-size:.85rem;margin-top:8px">
                Hujan rata-rata {raw[z][6]:.0f}mm · Kelembaban {raw[z][4]:.0f}% · Suhu {raw[z][3]:.0f}°C
              </div></div>""", unsafe_allow_html=True)
            top_z = df_z[df_z["Zona"]==z]["label"].value_counts().head(5)
            st.markdown("**Tanaman umum di zona ini:**  " +
                        "  ·  ".join([f"{TANAMAN.get(c,{}).get('emoji','🌿')} {c.capitalize()}" for c in top_z.index]))


# ══════════════════════════════════════════════════════════════
# ANALISIS DATA
# ══════════════════════════════════════════════════════════════
elif menu == "📊  Analisis Data":
    st.markdown('<div class="section-hd">📊 Analisis Data & Performa Model</div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4 = st.tabs(["📈 Performa Model","🔥 Feature Importance","🔗 Korelasi","📦 Distribusi"])

    with tab1:
        c1,c2,c3=st.columns(3)
        c1.metric("Akurasi Test Set",f"{akurasi:.2%}")
        c2.metric("CV Mean (5-fold)",f"{cv_scores.mean():.2%}")
        c3.metric("CV Std Dev",f"±{cv_scores.std():.2%}")
        cl,cr = st.columns(2)
        with cl:
            st.markdown("#### Skor tiap Fold")
            fig=go.Figure(go.Bar(x=[f"Fold {i+1}" for i in range(len(cv_scores))],
                y=cv_scores*100,marker_color="#2D6A4F",
                text=[f"{s:.1%}" for s in cv_scores],textposition="outside"))
            fig.add_hline(y=cv_scores.mean()*100,line_dash="dash",line_color="#F77F00",
                          annotation_text=f"Rata-rata: {cv_scores.mean():.1%}")
            fig.update_layout(yaxis_title="Akurasi (%)",height=300,
                              plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE",margin=dict(t=30,b=10))
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            st.markdown("#### Confusion Matrix")
            fig2,ax=plt.subplots(figsize=(10,9))
            sns.heatmap(cm,annot=True,fmt="d",cmap="Greens",xticklabels=kelas,yticklabels=kelas,ax=ax,
                        linewidths=.5,linecolor="white")
            ax.set_xlabel("Prediksi",fontsize=11); ax.set_ylabel("Aktual",fontsize=11)
            ax.tick_params(axis="x",rotation=45,labelsize=8); ax.tick_params(axis="y",labelsize=8)
            plt.tight_layout(); st.pyplot(fig2)

    with tab2:
        st.markdown("#### Seberapa Besar Pengaruh Tiap Parameter?")
        st.markdown('<div class="tips">Parameter dengan nilai lebih tinggi lebih berpengaruh dalam menentukan tanaman yang cocok untuk lahan Anda.</div>',
                    unsafe_allow_html=True)
        fi=pd.DataFrame({"Fitur":[F_NAMA[f] for f in FITUR],"Skor":importances}).sort_values("Skor",ascending=True)
        fig=go.Figure(go.Bar(x=fi["Skor"]*100,y=fi["Fitur"],orientation="h",marker_color="#2D6A4F",
                             text=[f"{v:.1f}%" for v in fi["Skor"]*100],textposition="outside"))
        fig.update_layout(xaxis_title="Tingkat Pengaruh (%)",height=380,
                          plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE",margin=dict(t=10,b=10,r=80))
        st.plotly_chart(fig,use_container_width=True)

    with tab3:
        st.markdown("#### Hubungan Antar Parameter Lahan")
        st.markdown('<div class="section-sub">Merah = keduanya naik bersama. Biru = satu naik satu turun.</div>',
                    unsafe_allow_html=True)
        corr=df[FITUR].corr(); corr.columns=[F_NAMA[f] for f in FITUR]; corr.index=[F_NAMA[f] for f in FITUR]
        fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,height=460)
        st.plotly_chart(fig,use_container_width=True)

    with tab4:
        feat_sel=st.selectbox("Pilih Parameter",[F_NAMA[f] for f in FITUR])
        feat_key=[f for f in FITUR if F_NAMA[f]==feat_sel][0]
        fig=px.box(df,x="label",y=feat_key,color="label",
                   color_discrete_sequence=px.colors.qualitative.Set2,
                   labels={feat_key:feat_sel,"label":"Tanaman"},height=480)
        fig.update_layout(showlegend=False,xaxis_tickangle=-45,
                          plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE")
        st.plotly_chart(fig,use_container_width=True)
        stats=df.groupby("label")[feat_key].agg(["min","mean","max"]).round(1)
        stats.columns=["Minimum","Rata-rata","Maksimum"]
        st.dataframe(stats,use_container_width=True,height=280)


# ══════════════════════════════════════════════════════════════
# ANALISIS LANJUTAN
# ══════════════════════════════════════════════════════════════
elif menu == "🔬  Analisis Lanjutan":
    st.markdown('<div class="section-hd">🔬 Analisis Lanjutan</div>', unsafe_allow_html=True)

    tab1,tab2 = st.tabs(["🔄 Simulasi Perubahan Parameter","📡 Sensitivitas Fitur"])

    with tab1:
        st.markdown("#### Simulasi: Apa yang Terjadi Jika Satu Parameter Diubah?")
        st.markdown('<div class="tips">Atur kondisi dasar lahan, lalu pilih satu parameter yang ingin disimulasikan. Grafik akan menunjukkan bagaimana rekomendasi berubah.</div>',
                    unsafe_allow_html=True)
        st.markdown("**Kondisi Dasar**")
        c1,c2,c3,c4=st.columns(4)
        bN=c1.number_input("N",0,140,50,key="wN"); bP=c1.number_input("P",5,145,53,key="wP")
        bK=c2.number_input("K",5,205,48,key="wK"); bT=c2.number_input("Suhu",8.0,44.0,25.0,key="wT")
        bH=c3.number_input("Kelembaban",14.0,100.0,71.0,key="wH")
        bPH=c3.number_input("pH",3.5,10.0,6.5,key="wPH")
        bR=c4.number_input("Curah Hujan",20.0,300.0,100.0,key="wR")

        base=[bN,bP,bK,bT,bH,bPH,bR]
        bpred=model.predict(scaler.transform([base]))[0]
        bconf=model.predict_proba(scaler.transform([base]))[0][list(kelas).index(bpred)]
        st.info(f"**Rekomendasi Dasar:** {TANAMAN.get(bpred,{}).get('emoji','🌿')} **{bpred.upper()}** — Kecocokan {bconf:.0%}")

        st.markdown("---")
        pu=st.selectbox("Parameter yang Disimulasikan",[F_NAMA[f] for f in FITUR])
        pk=[f for f in FITUR if F_NAMA[f]==pu][0]; idx_p=FITUR.index(pk)
        ns=st.slider("Jumlah Titik Simulasi",5,30,15)
        vals=np.linspace(df[pk].min(),df[pk].max(),ns)
        rows=[]
        for v in vals:
            inp2=base.copy(); inp2[idx_p]=v; sc2=scaler.transform([inp2])
            pd2=model.predict(sc2)[0]; pp2=model.predict_proba(sc2)[0]
            rows.append({"Nilai":round(v,2),"Tanaman":pd2,"Kecocokan":pp2.max()})
        dw=pd.DataFrame(rows)

        cl,cr=st.columns([2,1])
        with cl:
            fig=go.Figure()
            for crop in dw["Tanaman"].unique():
                sub=dw[dw["Tanaman"]==crop]; ei=TANAMAN.get(crop,{"emoji":"🌿"})
                fig.add_trace(go.Scatter(x=sub["Nilai"],y=sub["Kecocokan"]*100,
                    mode="lines+markers",name=f"{ei['emoji']} {crop}",
                    line=dict(width=2.5),marker=dict(size=8)))
            fig.update_layout(xaxis_title=pu,yaxis_title="Kecocokan (%)",
                              height=380,plot_bgcolor="#FAFFFE",paper_bgcolor="#FAFFFE")
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            disp=dw.copy()
            disp["Kecocokan"]=disp["Kecocokan"].map(lambda x:f"{x:.0%}")
            disp["Tanaman"]=disp["Tanaman"].apply(lambda c:f"{TANAMAN.get(c,{}).get('emoji','🌿')} {c.capitalize()}")
            st.dataframe(disp,use_container_width=True,height=380)

        perubahan=[{"Dari":dw.loc[i-1,"Tanaman"],"Ke":dw.loc[i,"Tanaman"],pu:dw.loc[i,"Nilai"]}
                   for i in range(1,len(dw)) if dw.loc[i,"Tanaman"]!=dw.loc[i-1,"Tanaman"]]
        if perubahan:
            st.warning(f"**⚡ {len(perubahan)} titik perubahan rekomendasi** ditemukan:")
            st.dataframe(pd.DataFrame(perubahan),use_container_width=True)
        else:
            st.success("✅ Rekomendasi tetap stabil di seluruh rentang nilai parameter ini.")

    with tab2:
        st.markdown("#### Sensitivitas: Parameter Mana yang Paling Berpengaruh?")
        st.markdown('<div class="tips">Nilai merah = rekomendasi berubah drastis jika parameter ini diubah. Nilai hijau = rekomendasi stabil.</div>',
                    unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        sN=c1.slider("N",0,140,50,key="sN"); sP=c1.slider("P",5,145,53,key="sP")
        sK=c2.slider("K",5,205,48,key="sK"); sT=c2.slider("Suhu",8.0,44.0,25.0,key="sT")
        sH=c3.slider("Kelembaban",14.0,100.0,71.0,key="sH")
        sPH=c3.slider("pH",3.5,10.0,6.5,key="sPH"); sR=c3.slider("Curah Hujan",20.0,300.0,100.0,key="sR")

        base_s=[sN,sP,sK,sT,sH,sPH,sR]
        bpred_s=model.predict(scaler.transform([base_s]))[0]
        bprob_s=model.predict_proba(scaler.transform([base_s]))[0][list(kelas).index(bpred_s)]
        st.info(f"**Rekomendasi Dasar:** {TANAMAN.get(bpred_s,{}).get('emoji','🌿')} **{bpred_s.upper()}** — Kecocokan {bprob_s:.0%}")

        pcts=[-30,-20,-10,0,10,20,30]; sens={}
        for i,f in enumerate(FITUR):
            vals=[]
            for pct in pcts:
                inp3=base_s.copy(); inp3[i]*=(1+pct/100)
                p3=model.predict_proba(scaler.transform([inp3]))[0][list(kelas).index(bpred_s)]
                vals.append(p3*100)
            sens[f]=vals

        sens_df=pd.DataFrame(sens,index=[f"{p:+d}%" for p in pcts])
        sens_df.columns=[F_NAMA[f] for f in FITUR]
        fig=px.imshow(sens_df,text_auto=".0f",color_continuous_scale="RdYlGn",
                      height=340,zmin=0,zmax=100,labels=dict(color="Kecocokan (%)"))
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig,use_container_width=True)

        skor={F_NAMA[f]:round(np.std(sens[f]),2) for f in FITUR}
        rank=pd.DataFrame(list(skor.items()),columns=["Parameter","Skor Sensitivitas"]
                          ).sort_values("Skor Sensitivitas",ascending=False)
        rank["Interpretasi"]=rank["Skor Sensitivitas"].apply(
            lambda x:"🔴 Sangat Sensitif" if x>10 else("🟡 Cukup Sensitif" if x>5 else "🟢 Stabil"))
        st.dataframe(rank,use_container_width=True,hide_index=True)


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#9CA3AF;font-size:.8rem;padding:8px'>"
    "🌾 AgroSPK · Kelompok 5 · Sistem Pendukung Keputusan · "
    "Dataset: Crop Recommendation (2.200 sampel, 22 tanaman)"
    "</div>", unsafe_allow_html=True)
