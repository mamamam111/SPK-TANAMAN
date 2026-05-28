import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# CONFIG HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SPK Rekomendasi Tanaman",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2d6a4f;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #52796f;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #d8f3dc, #b7e4c7);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1b4332;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #52796f;
    }
    .result-box {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(45,106,79,0.3);
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #0d6efd;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #081c15, #1b4332);
    }
    section[data-testid="stSidebar"] * {
        color: #d8f3dc !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA & MODEL (cache agar tidak reload terus)
# ─────────────────────────────────────────────
DATA_PATH = "Crop_recommendation.csv"

CROP_INFO = {
    "rice":        {"emoji": "🌾", "musim": "Basah",  "catatan": "Butuh irigasi baik, tanah sawah"},
    "maize":       {"emoji": "🌽", "musim": "Kering", "catatan": "Cocok lahan kering, drainase baik"},
    "chickpea":    {"emoji": "🫘", "musim": "Dingin", "catatan": "Toleran kering, baik untuk rotasi"},
    "kidneybeans": {"emoji": "🫘", "musim": "Hangat", "catatan": "pH netral, kelembaban sedang"},
    "pigeonpeas":  {"emoji": "🌿", "musim": "Kering", "catatan": "Toleran kering, nitrogen tinggi"},
    "mothbeans":   {"emoji": "🌿", "musim": "Kering", "catatan": "Sangat toleran kekeringan"},
    "mungbean":    {"emoji": "🫛", "musim": "Hangat", "catatan": "Siklus pendek, mudah tumbuh"},
    "blackgram":   {"emoji": "🫘", "musim": "Hangat", "catatan": "Toleran panas, curah hujan sedang"},
    "lentil":      {"emoji": "🫘", "musim": "Dingin", "catatan": "Tanah lempung, drainase baik"},
    "pomegranate": {"emoji": "🍎", "musim": "Panas",  "catatan": "Tahan kering, buah komersil tinggi"},
    "banana":      {"emoji": "🍌", "musim": "Basah",  "catatan": "Kelembaban tinggi, tanah subur"},
    "mango":       {"emoji": "🥭", "musim": "Panas",  "catatan": "Buah tropis, musim kering ringan"},
    "grapes":      {"emoji": "🍇", "musim": "Panas",  "catatan": "pH asam, drainase sangat baik"},
    "watermelon":  {"emoji": "🍉", "musim": "Panas",  "catatan": "Tanah berpasir, sinar matahari penuh"},
    "muskmelon":   {"emoji": "🍈", "musim": "Panas",  "catatan": "Musim panas panjang, drainase baik"},
    "apple":       {"emoji": "🍏", "musim": "Dingin", "catatan": "Dataran tinggi, suhu sejuk"},
    "orange":      {"emoji": "🍊", "musim": "Hangat", "catatan": "pH asam, curah hujan teratur"},
    "papaya":      {"emoji": "🍑", "musim": "Basah",  "catatan": "Tumbuh cepat, tanah gembur"},
    "coconut":     {"emoji": "🥥", "musim": "Basah",  "catatan": "Tanah pantai, kelembaban tinggi"},
    "cotton":      {"emoji": "🌼", "musim": "Panas",  "catatan": "Lahan kering, drainase baik"},
    "jute":        {"emoji": "🌿", "musim": "Basah",  "catatan": "Lahan basah, curah hujan tinggi"},
    "coffee":      {"emoji": "☕", "musim": "Basah",  "catatan": "pH asam, ketinggian sedang"},
}

FITUR_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
FITUR_LABEL = {
    "N": "Nitrogen (N)", "P": "Fosfor (P)", "K": "Kalium (K)",
    "temperature": "Suhu (°C)", "humidity": "Kelembaban (%)",
    "ph": "pH Tanah", "rainfall": "Curah Hujan (mm)"
}

@st.cache_data
def load_data():
    try:
        # Gunakan relative path (hanya panggil nama filenya saja)
        df = pd.read_csv("Crop_recommendation.csv", sep=',')
    except FileNotFoundError:
        st.error("File `Crop_recommendation.csv` tidak ditemukan. Pastikan file CSV ada di folder yang sama dengan script ini.")
        st.stop()
    return df

@st.cache_resource
def train_classifier(df):
    X = df[FITUR_COLS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")

    return model, scaler, acc, report, cm, cv_scores, model.classes_, X_test, y_test, y_pred

@st.cache_resource
def train_clustering(df, n_clusters=5):
    X = df[FITUR_COLS]
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, labels)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    return km, scaler, labels, sil, X_pca

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = load_data()
clf_model, clf_scaler, clf_acc, clf_report, clf_cm, cv_scores, classes, X_test, y_test, y_pred = train_classifier(df)
importances = clf_model.feature_importances_

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 SPK Tanaman")
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["Beranda", "Eksplorasi Data","Klasifikasi", "Clustering", "What-If Analysis", "Sensitivity Analysis"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### Info Dataset")
    st.markdown(f"- **{len(df):,}** sampel data")
    st.markdown(f"- **{df['label'].nunique()}** jenis tanaman")
    st.markdown(f"- **{len(FITUR_COLS)}** fitur input")
    st.markdown("---")
    st.markdown("### Akurasi Model")
    st.markdown(f"**Random Forest:** `{clf_acc:.1%}`")
    st.markdown(f"**CV (5-fold):** `{cv_scores.mean():.1%} ± {cv_scores.std():.2%}`")

# ─────────────────────────────────────────────
# ══════════ HALAMAN BERANDA ══════════
# ─────────────────────────────────────────────
if menu == "Beranda":
    st.markdown('<div class="main-title">🌾 Sistem Pendukung Keputusan Rekomendasi Tanaman</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Memanfaatkan Data Mining — Klasifikasi & Clustering untuk Optimasi Pertanian</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-value">2.200</div><div class="metric-label">Total Sampel</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">22</div><div class="metric-label">Jenis Tanaman</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{clf_acc:.1%}</div><div class="metric-label">Akurasi Klasifikasi</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">7</div><div class="metric-label">Fitur Input</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧩 Arsitektur SPK")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        #### Fitur 1 — Klasifikasi (Random Forest)
        Memprediksi tanaman paling cocok berdasarkan 7 parameter kondisi lahan:
        - Kandungan N, P, K dalam tanah
        - Suhu & kelembaban udara
        - pH tanah & curah hujan

        **Algoritma:** Random Forest (200 pohon, cross-validated)
        """)
        st.markdown("""
        #### Fitur 3 — What-If Analysis
        Simulasi "bagaimana jika" parameter diubah — petani dapat menjelajahi
        dampak perubahan kondisi lahan terhadap rekomendasi.
        """)
    with col_r:
        st.markdown("""
        #### Fitur 2 — Clustering (K-Means)
        Mengelompokkan lahan ke dalam klaster berdasarkan kesamaan
        karakteristik tanah & iklim, untuk strategi manajemen lahan
        yang lebih tepat sasaran.

        **Algoritma:** K-Means + Silhouette Score + PCA Visualization
        """)
        st.markdown("""
        #### Fitur 4 — Sensitivity Analysis
        Menganalisis seberapa sensitif rekomendasi terhadap perubahan
        masing-masing fitur — membantu prioritas intervensi lahan.
        """)

    st.markdown("---")
    st.markdown("### 🌿 Tanaman yang Didukung")
    cols = st.columns(6)
    for i, (crop, info) in enumerate(CROP_INFO.items()):
        with cols[i % 6]:
            st.markdown(f"**{info['emoji']} {crop.capitalize()}**")
            st.caption(info["musim"])

# ─────────────────────────────────────────────
# ══════════ HALAMAN KLASIFIKASI ══════════
# ─────────────────────────────────────────────
elif menu == "Klasifikasi":
    st.markdown("## Fitur 1: Rekomendasi Tanaman (Klasifikasi)")
    st.markdown("Masukkan kondisi lahan Anda untuk mendapatkan rekomendasi tanaman terbaik.")

    st.markdown("### ⚙️ Parameter Lahan")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.slider("Nitrogen (N)", 0, 140, 50, help="Kandungan Nitrogen dalam tanah (kg/ha)")
        P = st.slider("Fosfor (P)", 5, 145, 53, help="Kandungan Fosfor dalam tanah (kg/ha)")
        K = st.slider("Kalium (K)", 5, 205, 48, help="Kandungan Kalium dalam tanah (kg/ha)")
    with col2:
        temp = st.slider("Suhu (°C)", 8.0, 44.0, 25.0, step=0.5)
        hum  = st.slider("Kelembaban (%)", 14.0, 100.0, 71.0, step=0.5)
    with col3:
        ph       = st.slider("pH Tanah", 3.5, 10.0, 6.5, step=0.1)
        rainfall = st.slider("Curah Hujan (mm)", 20.0, 300.0, 100.0, step=1.0)

    if st.button(" Dapatkan Rekomendasi", use_container_width=True, type="primary"):
        input_arr = np.array([[N, P, K, temp, hum, ph, rainfall]])
        input_sc  = clf_scaler.transform(input_arr)
        pred      = clf_model.predict(input_sc)[0]
        proba     = clf_model.predict_proba(input_sc)[0]

        # Top-5 prediksi
        top5_idx   = np.argsort(proba)[::-1][:5]
        top5_crops = [(classes[i], proba[i]) for i in top5_idx]

        info = CROP_INFO.get(pred, {"emoji": "🌿", "musim": "-", "catatan": "-"})
        st.markdown(f'<div class="result-box">{info["emoji"]} Rekomendasi: <u>{pred.upper()}</u></div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Detail Tanaman")
            st.info(f"**Musim:** {info['musim']}")
            st.info(f"**Catatan:** {info['catatan']}")
            st.success(f"**Kepercayaan Model:** {proba[list(classes).index(pred)]:.1%}")
        with col_b:
            st.markdown("#### Top-5 Kandidat")
            fig_bar = go.Figure(go.Bar(
                x=[f"{CROP_INFO.get(c,{}).get('emoji','🌿')} {c}" for c, _ in top5_crops],
                y=[p * 100 for _, p in top5_crops],
                marker_color=["#2d6a4f" if c == pred else "#95d5b2" for c, _ in top5_crops],
                text=[f"{p:.1%}" for _, p in top5_crops],
                textposition="outside",
            ))
            fig_bar.update_layout(
                yaxis_title="Probabilitas (%)", height=300,
                margin=dict(t=10, b=10), plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Perbandingan input vs rata-rata tanaman terprediksi
        st.markdown("#### Perbandingan Input vs Profil Ideal")
        crop_mean = df[df["label"] == pred][FITUR_COLS].mean()
        input_vals = [N, P, K, temp, hum, ph, rainfall]
        labels_radar = list(FITUR_LABEL.values())

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=list(crop_mean), theta=labels_radar, fill="toself",
            name=f"Profil Ideal {pred}", line_color="#2d6a4f"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=input_vals, theta=labels_radar, fill="toself",
            name="Input Anda", line_color="#f77f00", opacity=0.7
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            height=380, margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Performa Model")
    tab1, tab2, tab3 = st.tabs(["Akurasi & CV", "Confusion Matrix", "Feature Importance"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Akurasi Test Set", f"{clf_acc:.2%}")
        c2.metric("CV Mean (5-fold)", f"{cv_scores.mean():.2%}")
        c3.metric("CV Std Dev", f"±{cv_scores.std():.2%}")

        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(cv_scores))],
            y=cv_scores * 100, marker_color="#2d6a4f",
            text=[f"{s:.1%}" for s in cv_scores], textposition="outside"
        ))
        fig_cv.add_hline(y=cv_scores.mean()*100, line_dash="dash", line_color="#f77f00",
                         annotation_text=f"Mean: {cv_scores.mean():.1%}")
        fig_cv.update_layout(yaxis_title="Akurasi (%)", height=300,
                             margin=dict(t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cv, use_container_width=True)

    with tab2:
        fig_cm, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(clf_cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    linewidths=0.5, linecolor="white")
        ax.set_xlabel("Prediksi", fontsize=12)
        ax.set_ylabel("Aktual", fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig_cm)

    with tab3:
        feat_imp = pd.DataFrame({
            "Fitur": list(FITUR_LABEL.values()),
            "Importance": importances
        }).sort_values("Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=feat_imp["Importance"] * 100, y=feat_imp["Fitur"],
            orientation="h", marker_color="#2d6a4f",
            text=[f"{v:.1f}%" for v in feat_imp["Importance"] * 100],
            textposition="outside"
        ))
        fig_imp.update_layout(
            xaxis_title="Feature Importance (%)", height=380,
            margin=dict(t=10, b=10, r=80), plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ─────────────────────────────────────────────
# ══════════ HALAMAN CLUSTERING ══════════
# ─────────────────────────────────────────────
elif menu == "Clustering":
    st.markdown("## Fitur 2: Pengelompokan Lahan (Clustering)")
    st.markdown("K-Means mengelompokkan lahan berdasarkan kemiripan profil tanah & iklim.")

    n_clust = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, 5)
    km_model, km_scaler, km_labels, sil_score, X_pca = train_clustering(df, n_clust)

    df_clust = df.copy()
    df_clust["Klaster"] = km_labels
    df_clust["PCA1"]    = X_pca[:, 0]
    df_clust["PCA2"]    = X_pca[:, 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Klaster", n_clust)
    c2.metric("Silhouette Score", f"{sil_score:.3f}")
    c3.metric("Interpretasi", "Baik ✅" if sil_score > 0.35 else "Cukup ⚠️")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["Visualisasi PCA", "Profil Klaster", "Elbow Method", "Identifikasi Lahan"])

    with tab1:
        st.markdown("#### Distribusi Klaster (Proyeksi PCA 2D)")
        colors = px.colors.qualitative.Set2
        fig_pca = go.Figure()
        for k in range(n_clust):
            mask = df_clust["Klaster"] == k
            fig_pca.add_trace(go.Scatter(
                x=df_clust[mask]["PCA1"], y=df_clust[mask]["PCA2"],
                mode="markers", name=f"Klaster {k+1}",
                marker=dict(color=colors[k % len(colors)], size=5, opacity=0.7),
                text=df_clust[mask]["label"],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}"
            ))
        fig_pca.update_layout(
            xaxis_title="Principal Component 1", yaxis_title="Principal Component 2",
            height=450, plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab2:
        st.markdown("#### Profil Rata-Rata Setiap Klaster")
        cluster_profile = df_clust.groupby("Klaster")[FITUR_COLS].mean().round(2)
        cluster_profile.index = [f"Klaster {k+1}" for k in cluster_profile.index]
        st.dataframe(cluster_profile.rename(columns=FITUR_LABEL), use_container_width=True)

        st.markdown("#### Distribusi Tanaman per Klaster")
        crop_dist = pd.crosstab(df_clust["Klaster"], df_clust["label"])
        crop_dist.index = [f"Klaster {k+1}" for k in crop_dist.index]
        fig_heat = px.imshow(
            crop_dist, text_auto=True, color_continuous_scale="Greens",
            labels=dict(x="Tanaman", y="Klaster", color="Jumlah"),
            height=350
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Radar Profil Klaster")
        # Normalisasi untuk radar
        cluster_norm = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min() + 1e-9)
        fig_radar = go.Figure()
        for k in range(n_clust):
            row = cluster_norm.iloc[k]
            fig_radar.add_trace(go.Scatterpolar(
                r=list(row) + [row.iloc[0]],
                theta=list(FITUR_LABEL.values()) + [list(FITUR_LABEL.values())[0]],
                mode="lines", name=f"Klaster {k+1}",
                line_color=colors[k % len(colors)]
            ))
        fig_radar.update_layout(height=420)
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.markdown("#### 📐 Elbow Method — Menentukan K Optimal")
        inertias, sil_scores = [], []
        k_range = range(2, 11)
        with st.spinner("Menghitung Elbow & Silhouette..."):
            X_sc = km_scaler.transform(df[FITUR_COLS])
            for k in k_range:
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                lbl = km_tmp.fit_predict(X_sc)
                inertias.append(km_tmp.inertia_)
                sil_scores.append(silhouette_score(X_sc, lbl))

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            name="Inertia (WCSS)", line=dict(color="#2d6a4f", width=2),
            marker=dict(size=8)
        ), secondary_y=False)
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=sil_scores, mode="lines+markers",
            name="Silhouette Score", line=dict(color="#f77f00", width=2, dash="dash"),
            marker=dict(size=8)
        ), secondary_y=True)
        fig_elbow.update_xaxes(title_text="Jumlah Klaster (K)")
        fig_elbow.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
        fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
        fig_elbow.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_elbow, use_container_width=True)
        best_k = list(k_range)[np.argmax(sil_scores)]
        st.info(f"💡 K optimal berdasarkan Silhouette Score tertinggi: **K = {best_k}**")

    with tab4:
        st.markdown("#### Identifikasi Klaster Lahan Baru")
        col1, col2 = st.columns(2)
        with col1:
            N2 = st.slider("N", 0, 140, 50, key="cn")
            P2 = st.slider("P", 5, 145, 53, key="cp")
            K2 = st.slider("K", 5, 205, 48, key="ck")
            temp2 = st.slider("Suhu", 8.0, 44.0, 25.0, key="ct")
        with col2:
            hum2 = st.slider("Kelembaban", 14.0, 100.0, 71.0, key="ch")
            ph2  = st.slider("pH", 3.5, 10.0, 6.5, key="cp2")
            rain2 = st.slider("Curah Hujan", 20.0, 300.0, 100.0, key="cr")

        if st.button("Identifikasi Klaster", key="cluster_btn", use_container_width=True):
            inp = np.array([[N2, P2, K2, temp2, hum2, ph2, rain2]])
            inp_sc = km_scaler.transform(inp)
            cluster_id = km_model.predict(inp_sc)[0]
            st.success(f" Lahan Anda masuk ke **Klaster {cluster_id + 1}**")
            st.dataframe(
                cluster_profile.iloc[cluster_id:cluster_id+1].rename(columns=FITUR_LABEL),
                use_container_width=True
            )
            crops_in_cluster = df_clust[df_clust["Klaster"] == cluster_id]["label"].value_counts().head(5)
            st.markdown("**Tanaman umum di klaster ini:**")
            for crop, cnt in crops_in_cluster.items():
                info = CROP_INFO.get(crop, {"emoji": "🌿"})
                st.write(f"- {info['emoji']} {crop.capitalize()} ({cnt} sampel)")

# ─────────────────────────────────────────────
# ══════════ HALAMAN WHAT-IF ANALYSIS ══════════
# ─────────────────────────────────────────────
elif menu == " What-If Analysis":
    st.markdown("## What-If Analysis")
    st.markdown("Simulasikan **perubahan parameter** dan lihat bagaimana rekomendasi berubah secara real-time.")

    st.markdown("### Kondisi Dasar")
    col1, col2, col3, col4 = st.columns(4)
    base_N    = col1.number_input("N Dasar", 0, 140, 50)
    base_P    = col1.number_input("P Dasar", 5, 145, 53)
    base_K    = col2.number_input("K Dasar", 5, 205, 48)
    base_temp = col2.number_input("Suhu Dasar (°C)", 8.0, 44.0, 25.0)
    base_hum  = col3.number_input("Kelembaban Dasar (%)", 14.0, 100.0, 71.0)
    base_ph   = col3.number_input("pH Dasar", 3.5, 10.0, 6.5)
    base_rain = col4.number_input("Curah Hujan Dasar (mm)", 20.0, 300.0, 100.0)

    base_input = [base_N, base_P, base_K, base_temp, base_hum, base_ph, base_rain]
    base_sc = clf_scaler.transform([base_input])
    base_pred = clf_model.predict(base_sc)[0]
    base_proba = clf_model.predict_proba(base_sc)[0]
    info_base = CROP_INFO.get(base_pred, {"emoji": "🌿"})
    st.info(f"**Rekomendasi Dasar:** {info_base['emoji']} **{base_pred.upper()}** (kepercayaan: {base_proba[list(classes).index(base_pred)]:.1%})")

    st.markdown("---")
    st.markdown("### Parameter yang Diubah")

    fitur_ubah = st.selectbox("Pilih Parameter", list(FITUR_LABEL.values()))
    fitur_key = [k for k, v in FITUR_LABEL.items() if v == fitur_ubah][0]
    idx = FITUR_COLS.index(fitur_key)
    base_val = base_input[idx]

    min_v, max_v = df[fitur_key].min(), df[fitur_key].max()
    n_steps = st.slider("Jumlah Skenario", 5, 30, 15)
    vals = np.linspace(min_v, max_v, n_steps)

    results = []
    for v in vals:
        inp = base_input.copy()
        inp[idx] = v
        inp_sc = clf_scaler.transform([inp])
        pred = clf_model.predict(inp_sc)[0]
        proba = clf_model.predict_proba(inp_sc)[0]
        top_conf = proba.max()
        results.append({"Nilai": round(v, 2), "Rekomendasi": pred, "Kepercayaan": top_conf})

    df_what = pd.DataFrame(results)

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Perubahan Rekomendasi")
        fig_what = go.Figure()
        unique_crops = df_what["Rekomendasi"].unique()
        color_map = {c: px.colors.qualitative.Set2[i % 8] for i, c in enumerate(unique_crops)}

        for crop in unique_crops:
            mask = df_what["Rekomendasi"] == crop
            sub = df_what[mask]
            fig_what.add_trace(go.Scatter(
                x=sub["Nilai"], y=sub["Kepercayaan"] * 100,
                mode="markers+lines", name=crop,
                line=dict(color=color_map[crop], width=2),
                marker=dict(size=8),
            ))
        fig_what.update_layout(
            xaxis_title=fitur_ubah, yaxis_title="Kepercayaan (%)",
            height=380, plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_what, use_container_width=True)

    with col_right:
        st.markdown("#### Tabel Skenario")
        df_display = df_what.copy()
        df_display["Emoji"] = df_display["Rekomendasi"].map(lambda c: CROP_INFO.get(c, {}).get("emoji", "🌿"))
        df_display["Kepercayaan"] = df_display["Kepercayaan"].map(lambda x: f"{x:.1%}")
        st.dataframe(df_display[["Nilai", "Emoji", "Rekomendasi", "Kepercayaan"]], use_container_width=True, height=380)

    # Ringkasan titik perubahan
    transitions = []
    for i in range(1, len(df_what)):
        if df_what.loc[i, "Rekomendasi"] != df_what.loc[i-1, "Rekomendasi"]:
            transitions.append({
                "Dari": df_what.loc[i-1, "Rekomendasi"],
                "Ke": df_what.loc[i, "Rekomendasi"],
                "Pada": f"{df_what.loc[i, 'Nilai']:.2f}"
            })
    if transitions:
        st.markdown("#### ⚡ Titik Perubahan Rekomendasi")
        st.dataframe(pd.DataFrame(transitions), use_container_width=True)
    else:
        st.success("Rekomendasi stabil — tidak berubah di seluruh rentang nilai.")

# ─────────────────────────────────────────────
# ══════════ HALAMAN SENSITIVITY ANALYSIS ══════════
# ─────────────────────────────────────────────
elif menu == "Sensitivity Analysis":
    st.markdown("## Sensitivity Analysis")
    st.markdown("Analisis seberapa sensitif rekomendasi terhadap perubahan **setiap fitur** dari kondisi dasar.")

    st.markdown("### Kondisi Dasar")
    c1, c2, c3 = st.columns(3)
    N_s    = c1.slider("N", 0, 140, 50, key="s_n")
    P_s    = c1.slider("P", 5, 145, 53, key="s_p")
    K_s    = c1.slider("K", 5, 205, 48, key="s_k")
    temp_s = c2.slider("Suhu", 8.0, 44.0, 25.0, key="s_t")
    hum_s  = c2.slider("Kelembaban", 14.0, 100.0, 71.0, key="s_h")
    ph_s   = c3.slider("pH", 3.5, 10.0, 6.5, key="s_ph")
    rain_s = c3.slider("Curah Hujan", 20.0, 300.0, 100.0, key="s_r")

    base = [N_s, P_s, K_s, temp_s, hum_s, ph_s, rain_s]
    base_sc = clf_scaler.transform([base])
    base_pred = clf_model.predict(base_sc)[0]
    base_proba = clf_model.predict_proba(base_sc)[0][list(classes).index(base_pred)]

    st.info(f"**Rekomendasi Dasar:** {CROP_INFO.get(base_pred,{}).get('emoji','🌿')} **{base_pred.upper()}** | Kepercayaan: {base_proba:.1%}")

    pct_changes = [-30, -20, -10, 0, 10, 20, 30]
    sens_results = {}

    with st.spinner("Menghitung sensitivity..."):
        for i, fitur in enumerate(FITUR_COLS):
            vals = []
            for pct in pct_changes:
                inp = base.copy()
                inp[i] = inp[i] * (1 + pct / 100)
                inp_sc = clf_scaler.transform([inp])
                p = clf_model.predict_proba(inp_sc)[0]
                # kepercayaan untuk tanaman dasar
                base_idx = list(classes).index(base_pred)
                vals.append(p[base_idx] * 100)
            sens_results[fitur] = vals

    # Heatmap sensitivitas
    st.markdown("#### Heatmap Sensitivitas")
    sens_df = pd.DataFrame(sens_results, index=[f"{p:+d}%" for p in pct_changes])
    sens_df.columns = list(FITUR_LABEL.values())

    fig_heat = px.imshow(
        sens_df, text_auto=".1f", color_continuous_scale="RdYlGn",
        labels=dict(x="Fitur", y="Perubahan (%)", color="Kepercayaan (%)"),
        height=350, zmin=0, zmax=100
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Grafik garis sensitivity per fitur
    st.markdown("#### Grafik Sensitivity per Fitur")
    fig_lines = go.Figure()
    for fitur, label in FITUR_LABEL.items():
        fig_lines.add_trace(go.Scatter(
            x=[f"{p:+d}%" for p in pct_changes],
            y=sens_results[fitur], mode="lines+markers",
            name=label
        ))
    fig_lines.add_hline(y=base_proba * 100, line_dash="dash",
                        annotation_text="Baseline", line_color="black")
    fig_lines.update_layout(
        xaxis_title="Perubahan dari Kondisi Dasar",
        yaxis_title=f"Kepercayaan untuk '{base_pred}' (%)",
        height=420, plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_lines, use_container_width=True)

    # Skor sensitivitas total per fitur (std dari perubahan kepercayaan)
    st.markdown("#### Ranking Sensitivitas Fitur")
    sens_scores = {FITUR_LABEL[f]: np.std(sens_results[f]) for f in FITUR_COLS}
    sens_rank = pd.DataFrame(list(sens_scores.items()), columns=["Fitur", "Skor Sensitivitas"]).sort_values("Skor Sensitivitas", ascending=False)
    sens_rank["Interpretasi"] = sens_rank["Skor Sensitivitas"].apply(
        lambda x: "🔴 Sangat Sensitif" if x > 10 else ("🟡 Cukup Sensitif" if x > 5 else "🟢 Stabil")
    )
    st.dataframe(sens_rank, use_container_width=True, hide_index=True)

    fig_rank = go.Figure(go.Bar(
        x=sens_rank["Skor Sensitivitas"],
        y=sens_rank["Fitur"],
        orientation="h",
        marker_color=["#d62728" if s > 10 else "#ff7f0e" if s > 5 else "#2ca02c"
                      for s in sens_rank["Skor Sensitivitas"]],
        text=[f"{s:.2f}" for s in sens_rank["Skor Sensitivitas"]],
        textposition="outside"
    ))
    fig_rank.update_layout(xaxis_title="Skor Sensitivitas (Std Dev Kepercayaan)",
                           height=350, margin=dict(t=10, r=80), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rank, use_container_width=True)

# ─────────────────────────────────────────────
# ══════════ HALAMAN EKSPLORASI DATA ══════════
# ─────────────────────────────────────────────
elif menu == "Eksplorasi Data":
    st.markdown("## Eksplorasi Dataset")

    st.markdown("### Statistik Deskriptif")
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distribusi Fitur", "Matriks Korelasi", "Boxplot per Tanaman"])

    with tab1:
        fitur_sel = st.selectbox("Pilih Fitur", list(FITUR_LABEL.values()))
        key = [k for k, v in FITUR_LABEL.items() if v == fitur_sel][0]
        fig_hist = px.histogram(df, x=key, color="label", nbins=40, barmode="overlay",
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                labels={key: fitur_sel}, height=400)
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        corr = df[FITUR_COLS].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             zmin=-1, zmax=1, height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        fitur_box = st.selectbox("Pilih Fitur untuk Boxplot", list(FITUR_LABEL.values()), key="box")
        key_box = [k for k, v in FITUR_LABEL.items() if v == fitur_box][0]
        fig_box = px.box(df, x="label", y=key_box, color="label",
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         labels={key_box: fitur_box, "label": "Tanaman"}, height=450)
        fig_box.update_layout(showlegend=False, xaxis_tickangle=-45, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    st.markdown("### Data Mentah")
    filter_crop = st.multiselect("Filter berdasarkan tanaman", sorted(df["label"].unique()), default=[])
    df_show = df[df["label"].isin(filter_crop)] if filter_crop else df
    st.dataframe(df_show, use_container_width=True, height=350)
    st.caption(f"Menampilkan {len(df_show):,} dari {len(df):,} baris")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#52796f; font-size:0.8rem;'>"
    "SPK Rekomendasi Tanaman — Klasifikasi & Clustering dengan Data Mining | "
    "Dataset: Crop Recommendation (2.200 sampel, 22 tanaman)"
    "</div>",
    unsafe_allow_html=True
)
