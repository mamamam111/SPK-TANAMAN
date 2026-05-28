import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, silhouette_score
)
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title { font-size:2.1rem; font-weight:700; color:#2d6a4f; text-align:center; }
    .sub-title  { font-size:1rem; color:#52796f; text-align:center; margin-bottom:1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-radius:12px; padding:1rem 1.5rem; text-align:center;
        box-shadow:0 2px 8px rgba(0,0,0,.08);
    }
    .metric-value { font-size:2rem; font-weight:700; color:#1b4332; }
    .metric-label { font-size:.85rem; color:#52796f; }
    .result-box {
        background:linear-gradient(135deg,#2d6a4f,#40916c);
        color:white; border-radius:16px; padding:1.5rem; text-align:center;
        font-size:1.6rem; font-weight:700; margin:1rem 0;
        box-shadow:0 4px 15px rgba(45,106,79,.3);
    }
    section[data-testid="stSidebar"] { background:linear-gradient(180deg,#081c15,#1b4332); }
    section[data-testid="stSidebar"] * { color:#d8f3dc !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KONSTANTA
# ─────────────────────────────────────────────
FITUR_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

FITUR_LABEL = {
    "N": "Nitrogen (N)", "P": "Fosfor (P)", "K": "Kalium (K)",
    "temperature": "Suhu (°C)", "humidity": "Kelembaban (%)",
    "ph": "pH Tanah", "rainfall": "Curah Hujan (mm)"
}

CROP_INFO = {
    "rice":        {"emoji":"🌾","musim":"Basah",  "catatan":"Butuh irigasi baik, tanah sawah"},
    "maize":       {"emoji":"🌽","musim":"Kering", "catatan":"Cocok lahan kering, drainase baik"},
    "chickpea":    {"emoji":"🫘","musim":"Dingin", "catatan":"Toleran kering, baik untuk rotasi"},
    "kidneybeans": {"emoji":"🫘","musim":"Hangat", "catatan":"pH netral, kelembaban sedang"},
    "pigeonpeas":  {"emoji":"🌿","musim":"Kering", "catatan":"Toleran kering, nitrogen tinggi"},
    "mothbeans":   {"emoji":"🌿","musim":"Kering", "catatan":"Sangat toleran kekeringan"},
    "mungbean":    {"emoji":"🫛","musim":"Hangat", "catatan":"Siklus pendek, mudah tumbuh"},
    "blackgram":   {"emoji":"🫘","musim":"Hangat", "catatan":"Toleran panas, curah hujan sedang"},
    "lentil":      {"emoji":"🫘","musim":"Dingin", "catatan":"Tanah lempung, drainase baik"},
    "pomegranate": {"emoji":"🍎","musim":"Panas",  "catatan":"Tahan kering, buah komersil tinggi"},
    "banana":      {"emoji":"🍌","musim":"Basah",  "catatan":"Kelembaban tinggi, tanah subur"},
    "mango":       {"emoji":"🥭","musim":"Panas",  "catatan":"Buah tropis, musim kering ringan"},
    "grapes":      {"emoji":"🍇","musim":"Panas",  "catatan":"pH asam, drainase sangat baik"},
    "watermelon":  {"emoji":"🍉","musim":"Panas",  "catatan":"Tanah berpasir, sinar matahari penuh"},
    "muskmelon":   {"emoji":"🍈","musim":"Panas",  "catatan":"Musim panas panjang, drainase baik"},
    "apple":       {"emoji":"🍏","musim":"Dingin", "catatan":"Dataran tinggi, suhu sejuk"},
    "orange":      {"emoji":"🍊","musim":"Hangat", "catatan":"pH asam, curah hujan teratur"},
    "papaya":      {"emoji":"🍑","musim":"Basah",  "catatan":"Tumbuh cepat, tanah gembur"},
    "coconut":     {"emoji":"🥥","musim":"Basah",  "catatan":"Tanah pantai, kelembaban tinggi"},
    "cotton":      {"emoji":"🌼","musim":"Panas",  "catatan":"Lahan kering, drainase baik"},
    "jute":        {"emoji":"🌿","musim":"Basah",  "catatan":"Lahan basah, curah hujan tinggi"},
    "coffee":      {"emoji":"☕","musim":"Basah",  "catatan":"pH asam, ketinggian sedang"},
}

ZONE_INFO = {
    0: {"name":"Tropis Basah",    "icon":"🌧️", "color":"#0ea5e9",
        "desc":"Curah hujan & kelembaban tinggi. Cocok padi, kelapa, kopi, jute."},
    1: {"name":"Semi-Kering",     "icon":"🌵", "color":"#f59e0b",
        "desc":"Curah hujan rendah. Cocok kacang-kacangan toleran kering."},
    2: {"name":"Subur Mineral",   "icon":"💎", "color":"#a855f7",
        "desc":"Kandungan P & K sangat tinggi. Ideal buah premium: anggur, apel."},
    3: {"name":"Komersial Sedang","icon":"🌿", "color":"#22c55e",
        "desc":"Kondisi seimbang. Cocok jagung, kapas, semangka."},
}

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # Path relatif — file CSV harus ada di folder yang sama dengan app.py
    df = pd.read_csv("Crop_recommendation.csv")
    return df

@st.cache_resource
def train_classifier(_df):
    X = _df[FITUR_COLS]
    y = _df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    y_pred   = model.predict(X_test_sc)
    acc      = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    cm       = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

    return model, scaler, acc, report, cm, cv_scores, model.classes_, X_test, y_test, y_pred

@st.cache_resource
def train_clustering(_df, n_clusters=4):
    X = _df[FITUR_COLS]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    km     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    sil    = silhouette_score(X_sc, labels)

    pca    = PCA(n_components=2, random_state=42)
    X_pca  = pca.fit_transform(X_sc)

    return km, scaler, labels, sil, X_pca

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
df = load_data()
clf_model, clf_scaler, clf_acc, clf_report, clf_cm, cv_scores, classes, X_test, y_test, y_pred = train_classifier(df)
importances = clf_model.feature_importances_

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 SPK Tanaman")
    st.markdown("**Kelompok 5 — DSS**")
    st.markdown("---")
    menu = st.radio("Navigasi", [
        "🏠 Beranda",
        "🎯 Rekomendasi Tanaman",
        "🗺️ Zonasi Ekologi",
        "📊 Eksplorasi Data",
        "🔬 What-If Analysis",
        "📡 Sensitivity Analysis",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Info Dataset")
    st.markdown(f"- **{len(df):,}** sampel data")
    st.markdown(f"- **{df['label'].nunique()}** jenis tanaman")
    st.markdown(f"- **{len(FITUR_COLS)}** fitur input")
    st.markdown("---")
    st.markdown("### Akurasi Model")
    st.markdown(f"**Random Forest:** `{clf_acc:.2%}`")
    st.markdown(f"**CV (5-fold):** `{cv_scores.mean():.2%} ± {cv_scores.std():.2%}`")

# ─────────────────────────────────────────────
# BERANDA
# ─────────────────────────────────────────────
if menu == "🏠 Beranda":
    st.markdown('<div class="main-title">🌾 SPK Rekomendasi Tanaman</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Klasifikasi & Clustering berbasis Data Mining untuk Optimasi Pertanian</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1,"2.200","Total Sampel"),
        (c2,"22","Jenis Tanaman"),
        (c3,f"{clf_acc:.1%}","Akurasi Model"),
        (c4,"7","Fitur Input"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 🎯 Fitur 1 — Klasifikasi (Random Forest)")
        st.markdown("Prediksi tanaman optimal dari 7 parameter lahan dengan **top-3 rekomendasi + persentase kecocokan**.")
        st.markdown("#### 🔬 Fitur 3 — What-If Analysis")
        st.markdown("Simulasi perubahan satu parameter sekaligus dan lihat dampaknya terhadap rekomendasi secara real-time.")
    with col_r:
        st.markdown("#### 🗺️ Fitur 2 — Clustering (K-Means)")
        st.markdown("Lahan dikelompokkan ke **4 zona ekologi** berdasarkan profil tanah & iklim + visualisasi PCA.")
        st.markdown("#### 📡 Fitur 4 — Sensitivity Analysis")
        st.markdown("Analisis seberapa sensitif rekomendasi terhadap perubahan ±30% setiap fitur.")

    st.markdown("---")
    st.markdown("### 🌿 Tanaman yang Didukung")
    cols = st.columns(6)
    for i, (crop, info) in enumerate(CROP_INFO.items()):
        with cols[i % 6]:
            st.markdown(f"**{info['emoji']} {crop.capitalize()}**")
            st.caption(info["musim"])

# ─────────────────────────────────────────────
# REKOMENDASI TANAMAN
# ─────────────────────────────────────────────
elif menu == "🎯 Rekomendasi Tanaman":
    st.markdown("## 🎯 Rekomendasi Tanaman")
    st.markdown("Masukkan kondisi lahan untuk mendapatkan rekomendasi tanaman terbaik.")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 50)
        P = st.slider("Fosfor (P)", 5, 145, 53)
        K = st.slider("Kalium (K)", 5, 205, 48)
    with col2:
        temp = st.slider("Suhu (°C)", 8.0, 44.0, 25.0, step=0.5)
        hum  = st.slider("Kelembaban (%)", 14.0, 100.0, 71.0, step=0.5)
    with col3:
        ph       = st.slider("pH Tanah", 3.5, 10.0, 6.5, step=0.1)
        rainfall = st.slider("Curah Hujan (mm)", 20.0, 300.0, 100.0, step=1.0)

    if st.button("🌱 Dapatkan Rekomendasi", use_container_width=True, type="primary"):
        inp      = np.array([[N, P, K, temp, hum, ph, rainfall]])
        inp_sc   = clf_scaler.transform(inp)
        pred     = clf_model.predict(inp_sc)[0]
        proba    = clf_model.predict_proba(inp_sc)[0]
        top5_idx = np.argsort(proba)[::-1][:5]
        top5     = [(classes[i], proba[i]) for i in top5_idx]

        info = CROP_INFO.get(pred, {"emoji":"🌿","musim":"-","catatan":"-"})
        conf = proba[list(classes).index(pred)]

        st.markdown(f'<div class="result-box">{info["emoji"]} Rekomendasi Utama: <u>{pred.upper()}</u> — Kecocokan {conf:.1%}</div>', unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info(f"**Musim:** {info['musim']}")
            st.info(f"**Catatan:** {info['catatan']}")

        with col_b:
            st.markdown("#### Top-5 Kandidat")
            fig_bar = go.Figure(go.Bar(
                x=[f"{CROP_INFO.get(c,{}).get('emoji','🌿')} {c}" for c,_ in top5],
                y=[p*100 for _,p in top5],
                marker_color=["#2d6a4f" if c==pred else "#95d5b2" for c,_ in top5],
                text=[f"{p:.1%}" for _,p in top5], textposition="outside",
            ))
            fig_bar.update_layout(yaxis_title="Probabilitas (%)", height=280,
                                  margin=dict(t=10,b=10), plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c:
            st.markdown("#### Input vs Profil Ideal")
            crop_mean  = df[df["label"]==pred][FITUR_COLS].mean()
            input_vals = [N, P, K, temp, hum, ph, rainfall]
            fig_radar  = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=list(crop_mean), theta=list(FITUR_LABEL.values()),
                fill="toself", name=f"Ideal {pred}", line_color="#2d6a4f"))
            fig_radar.add_trace(go.Scatterpolar(r=input_vals, theta=list(FITUR_LABEL.values()),
                fill="toself", name="Input Anda", line_color="#f77f00", opacity=0.7))
            fig_radar.update_layout(height=280, margin=dict(t=20,b=20))
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Zona Ekologi ──
        st.markdown("---")
        st.markdown("### 🗺️ Zona Ekologi Lahan Anda")
        km_model, km_scaler, km_labels, sil_score, _ = train_clustering(df, 4)
        inp_sc2   = km_scaler.transform(inp)
        zona      = km_model.predict(inp_sc2)[0]
        zone_info = ZONE_INFO[zona]
        st.success(f"**{zone_info['icon']} Zona {zona} — {zone_info['name']}**: {zone_info['desc']}")

    # ── Performa Model ──
    st.markdown("---")
    st.markdown("### 📈 Performa Model")
    t1, t2, t3 = st.tabs(["Akurasi & CV", "Confusion Matrix", "Feature Importance"])

    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Test Set", f"{clf_acc:.2%}")
        c2.metric("CV Mean (5-fold)", f"{cv_scores.mean():.2%}")
        c3.metric("CV Std Dev", f"±{cv_scores.std():.2%}")
        fig_cv = go.Figure(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores*100, marker_color="#2d6a4f",
            text=[f"{s:.1%}" for s in cv_scores], textposition="outside"
        ))
        fig_cv.add_hline(y=cv_scores.mean()*100, line_dash="dash", line_color="#f77f00",
                         annotation_text=f"Mean: {cv_scores.mean():.1%}")
        fig_cv.update_layout(yaxis_title="Akurasi (%)", height=300,
                             margin=dict(t=10,b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cv, use_container_width=True)

    with t2:
        fig_cm, ax = plt.subplots(figsize=(14,12))
        sns.heatmap(clf_cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5)
        ax.set_xlabel("Prediksi", fontsize=12)
        ax.set_ylabel("Aktual", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with t3:
        feat_imp = pd.DataFrame({"Fitur":list(FITUR_LABEL.values()),"Importance":importances}
                                ).sort_values("Importance", ascending=True)
        fig_imp = go.Figure(go.Bar(
            x=feat_imp["Importance"]*100, y=feat_imp["Fitur"], orientation="h",
            marker_color="#2d6a4f",
            text=[f"{v:.1f}%" for v in feat_imp["Importance"]*100], textposition="outside"
        ))
        fig_imp.update_layout(xaxis_title="Feature Importance (%)", height=380,
                              margin=dict(t=10,b=10,r=80), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────
# ZONASI EKOLOGI
# ─────────────────────────────────────────────
elif menu == "🗺️ Zonasi Ekologi":
    st.markdown("## 🗺️ Zonasi Ekologi Lahan (K-Means)")
    st.markdown("Lahan dikelompokkan berdasarkan kesamaan profil tanah & iklim.")

    n_clust = st.sidebar.slider("Jumlah Zona (K)", 2, 10, 4)
    km_model, km_scaler, km_labels, sil_score, X_pca = train_clustering(df, n_clust)

    df_c = df.copy()
    df_c["Zona"]  = km_labels
    df_c["PCA1"]  = X_pca[:,0]
    df_c["PCA2"]  = X_pca[:,1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Zona", n_clust)
    c2.metric("Silhouette Score", f"{sil_score:.3f}")
    c3.metric("Kualitas Cluster", "Baik ✅" if sil_score > 0.3 else "Cukup ⚠️")

    st.markdown("---")
    t1, t2, t3, t4 = st.tabs(["Visualisasi PCA", "Profil Zona", "Elbow Method", "Identifikasi Lahan"])

    with t1:
        fig_pca = px.scatter(df_c, x="PCA1", y="PCA2", color=df_c["Zona"].astype(str),
                             hover_data={"label":True},
                             labels={"color":"Zona"},
                             color_discrete_sequence=px.colors.qualitative.Set2,
                             height=450)
        fig_pca.update_traces(marker=dict(size=5, opacity=0.7))
        fig_pca.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pca, use_container_width=True)

    with t2:
        cluster_profile = df_c.groupby("Zona")[FITUR_COLS].mean().round(2)
        cluster_profile.index = [f"Zona {k}" for k in cluster_profile.index]
        st.dataframe(cluster_profile.rename(columns=FITUR_LABEL), use_container_width=True)

        st.markdown("#### Distribusi Tanaman per Zona")
        crop_dist = pd.crosstab(df_c["Zona"], df_c["label"])
        crop_dist.index = [f"Zona {k}" for k in crop_dist.index]
        fig_h = px.imshow(crop_dist, text_auto=True, color_continuous_scale="Greens",
                          height=350)
        st.plotly_chart(fig_h, use_container_width=True)

    with t3:
        inertias, sils = [], []
        X_sc = km_scaler.transform(df[FITUR_COLS])
        for k in range(2,11):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl    = km_tmp.fit_predict(X_sc)
            inertias.append(km_tmp.inertia_)
            sils.append(silhouette_score(X_sc, lbl))

        fig_elbow = make_subplots(specs=[[{"secondary_y":True}]])
        fig_elbow.add_trace(go.Scatter(x=list(range(2,11)), y=inertias, mode="lines+markers",
            name="Inertia", line=dict(color="#2d6a4f",width=2)), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(x=list(range(2,11)), y=sils, mode="lines+markers",
            name="Silhouette", line=dict(color="#f77f00",width=2,dash="dash")), secondary_y=True)
        fig_elbow.update_xaxes(title_text="Jumlah Zona (K)")
        fig_elbow.update_yaxes(title_text="Inertia", secondary_y=False)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig_elbow.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_elbow, use_container_width=True)
        best_k = list(range(2,11))[np.argmax(sils)]
        st.info(f"💡 K optimal berdasarkan Silhouette tertinggi: **K = {best_k}**")

    with t4:
        st.markdown("#### Identifikasi Zona Lahan Baru")
        c1, c2 = st.columns(2)
        with c1:
            N2    = st.slider("N",   0, 140, 50, key="cn")
            P2    = st.slider("P",   5, 145, 53, key="cp")
            K2    = st.slider("K",   5, 205, 48, key="ck")
            t2_   = st.slider("Suhu",8.0,44.0,25.0, key="ct")
        with c2:
            h2    = st.slider("Kelembaban",14.0,100.0,71.0, key="ch")
            ph2   = st.slider("pH",3.5,10.0,6.5, key="cp2")
            r2    = st.slider("Curah Hujan",20.0,300.0,100.0, key="cr")

        if st.button("🗺️ Identifikasi Zona", key="zone_btn", use_container_width=True):
            inp2   = np.array([[N2,P2,K2,t2_,h2,ph2,r2]])
            inp_sc = km_scaler.transform(inp2)
            zona   = km_model.predict(inp_sc)[0]
            zi     = ZONE_INFO.get(zona, {"name":f"Zona {zona}","icon":"🌿","desc":"-"})
            st.success(f"**{zi['icon']} Zona {zona} — {zi['name']}**: {zi['desc']}")
            st.dataframe(cluster_profile.iloc[zona:zona+1].rename(columns=FITUR_LABEL),
                         use_container_width=True)
            crops_in = df_c[df_c["Zona"]==zona]["label"].value_counts().head(5)
            st.markdown("**Tanaman dominan di zona ini:**")
            for crop, cnt in crops_in.items():
                ei = CROP_INFO.get(crop, {"emoji":"🌿"})
                st.write(f"- {ei['emoji']} {crop.capitalize()} ({cnt} sampel)")

# ─────────────────────────────────────────────
# EKSPLORASI DATA
# ─────────────────────────────────────────────
elif menu == "📊 Eksplorasi Data":
    st.markdown("## 📊 Eksplorasi Dataset")
    st.markdown("### Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")
    t1, t2, t3 = st.tabs(["Distribusi Fitur", "Matriks Korelasi", "Boxplot per Tanaman"])

    with t1:
        feat_sel = st.selectbox("Pilih Fitur", list(FITUR_LABEL.values()))
        key      = [k for k,v in FITUR_LABEL.items() if v==feat_sel][0]
        fig_h    = px.histogram(df, x=key, color="label", nbins=40, barmode="overlay",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                labels={key:feat_sel}, height=400)
        fig_h.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, use_container_width=True)

    with t2:
        corr     = df[FITUR_COLS].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with t3:
        feat_box = st.selectbox("Pilih Fitur untuk Boxplot", list(FITUR_LABEL.values()), key="bx")
        key_box  = [k for k,v in FITUR_LABEL.items() if v==feat_box][0]
        fig_box  = px.box(df, x="label", y=key_box, color="label",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          labels={key_box:feat_box,"label":"Tanaman"}, height=450)
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-45,
                              plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("### Data Mentah")
    filter_crop = st.multiselect("Filter Tanaman", sorted(df["label"].unique()))
    df_show     = df[df["label"].isin(filter_crop)] if filter_crop else df
    st.dataframe(df_show, use_container_width=True, height=350)
    st.caption(f"Menampilkan {len(df_show):,} dari {len(df):,} baris")

# ─────────────────────────────────────────────
# WHAT-IF ANALYSIS
# ─────────────────────────────────────────────
elif menu == "🔬 What-If Analysis":
    st.markdown("## 🔬 What-If Analysis")
    st.markdown("Ubah satu parameter dan lihat bagaimana rekomendasi berubah.")

    c1,c2,c3,c4 = st.columns(4)
    bN  = c1.number_input("N Dasar",  0, 140, 50)
    bP  = c1.number_input("P Dasar",  5, 145, 53)
    bK  = c2.number_input("K Dasar",  5, 205, 48)
    bT  = c2.number_input("Suhu",   8.0,44.0,25.0)
    bH  = c3.number_input("Kelembaban",14.0,100.0,71.0)
    bPH = c3.number_input("pH",    3.5,10.0,6.5)
    bR  = c4.number_input("Curah Hujan",20.0,300.0,100.0)

    base = [bN,bP,bK,bT,bH,bPH,bR]
    base_sc   = clf_scaler.transform([base])
    base_pred = clf_model.predict(base_sc)[0]
    base_prob = clf_model.predict_proba(base_sc)[0]
    base_conf = base_prob[list(classes).index(base_pred)]
    ei = CROP_INFO.get(base_pred,{"emoji":"🌿"})
    st.info(f"**Rekomendasi Dasar:** {ei['emoji']} **{base_pred.upper()}** — Kepercayaan: {base_conf:.1%}")

    st.markdown("---")
    fitur_ubah = st.selectbox("Parameter yang Diubah", list(FITUR_LABEL.values()))
    fitur_key  = [k for k,v in FITUR_LABEL.items() if v==fitur_ubah][0]
    idx        = FITUR_COLS.index(fitur_key)
    n_steps    = st.slider("Jumlah Skenario", 5, 30, 15)
    vals       = np.linspace(df[fitur_key].min(), df[fitur_key].max(), n_steps)

    rows = []
    for v in vals:
        inp = base.copy(); inp[idx] = v
        inp_sc = clf_scaler.transform([inp])
        pred   = clf_model.predict(inp_sc)[0]
        proba  = clf_model.predict_proba(inp_sc)[0]
        rows.append({"Nilai":round(v,2),"Rekomendasi":pred,"Kepercayaan":proba.max()})
    df_what = pd.DataFrame(rows)

    cl, cr = st.columns(2)
    with cl:
        fig_w = go.Figure()
        for crop in df_what["Rekomendasi"].unique():
            sub = df_what[df_what["Rekomendasi"]==crop]
            fig_w.add_trace(go.Scatter(x=sub["Nilai"],y=sub["Kepercayaan"]*100,
                mode="lines+markers",name=crop))
        fig_w.update_layout(xaxis_title=fitur_ubah,yaxis_title="Kepercayaan (%)",
                            height=380,plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_w, use_container_width=True)
    with cr:
        st.dataframe(df_what.assign(Kepercayaan=df_what["Kepercayaan"].map(lambda x:f"{x:.1%}")),
                     use_container_width=True, height=380)

    transitions = [{"Dari":df_what.loc[i-1,"Rekomendasi"],"Ke":df_what.loc[i,"Rekomendasi"],
                    "Pada":df_what.loc[i,"Nilai"]}
                   for i in range(1,len(df_what)) if df_what.loc[i,"Rekomendasi"]!=df_what.loc[i-1,"Rekomendasi"]]
    if transitions:
        st.markdown("#### ⚡ Titik Perubahan Rekomendasi")
        st.dataframe(pd.DataFrame(transitions), use_container_width=True)
    else:
        st.success("Rekomendasi stabil di seluruh rentang nilai.")

# ─────────────────────────────────────────────
# SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
elif menu == "📡 Sensitivity Analysis":
    st.markdown("## 📡 Sensitivity Analysis")
    st.markdown("Seberapa sensitif rekomendasi terhadap perubahan ±30% setiap fitur?")

    c1,c2,c3 = st.columns(3)
    sN  = c1.slider("N",   0,140,50,key="sn")
    sP  = c1.slider("P",   5,145,53,key="sp")
    sK  = c2.slider("K",   5,205,48,key="sk")
    sT  = c2.slider("Suhu",8.0,44.0,25.0,key="st")
    sH  = c3.slider("Kelembaban",14.0,100.0,71.0,key="sh")
    sPH = c3.slider("pH", 3.5,10.0,6.5,key="sph")
    sR  = c3.slider("Curah Hujan",20.0,300.0,100.0,key="sr")

    base  = [sN,sP,sK,sT,sH,sPH,sR]
    bsc   = clf_scaler.transform([base])
    bpred = clf_model.predict(bsc)[0]
    bprob = clf_model.predict_proba(bsc)[0][list(classes).index(bpred)]
    ei    = CROP_INFO.get(bpred,{"emoji":"🌿"})
    st.info(f"**Rekomendasi Dasar:** {ei['emoji']} **{bpred.upper()}** | Kepercayaan: {bprob:.1%}")

    pct_changes = [-30,-20,-10,0,10,20,30]
    sens = {}
    for i,f in enumerate(FITUR_COLS):
        vals = []
        for pct in pct_changes:
            inp = base.copy(); inp[i] *= (1+pct/100)
            isc = clf_scaler.transform([inp])
            p   = clf_model.predict_proba(isc)[0]
            vals.append(p[list(classes).index(bpred)]*100)
        sens[f] = vals

    sens_df = pd.DataFrame(sens, index=[f"{p:+d}%" for p in pct_changes])
    sens_df.columns = list(FITUR_LABEL.values())

    fig_heat = px.imshow(sens_df, text_auto=".1f", color_continuous_scale="RdYlGn",
                         height=350, zmin=0, zmax=100,
                         labels=dict(color="Kepercayaan (%)"))
    st.plotly_chart(fig_heat, use_container_width=True)

    scores = {FITUR_LABEL[f]: np.std(sens[f]) for f in FITUR_COLS}
    sr_df  = pd.DataFrame(list(scores.items()), columns=["Fitur","Skor"]).sort_values("Skor",ascending=False)
    sr_df["Interpretasi"] = sr_df["Skor"].apply(
        lambda x: "🔴 Sangat Sensitif" if x>10 else ("🟡 Cukup Sensitif" if x>5 else "🟢 Stabil"))
    st.dataframe(sr_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#52796f;font-size:.8rem'>"
    "SPK Rekomendasi Tanaman · Kelompok 5 · Dataset: Crop Recommendation (2.200 sampel, 22 tanaman)"
    "</div>", unsafe_allow_html=True
)
