"""
app.py — Home page for the AI-Powered Skin Cancer Detection app.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from config import DATASETS
from styles import inject, sidebar_brand, DS_COLORS, DS_ICONS, MODEL_COLORS

st.set_page_config(
    page_title="Prognos AI — Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_brand()
st.sidebar.markdown("""
<div style="font-size:0.78rem; color:#64748B; line-height:1.7;">
    <b style="color:#94A3B8;">📌 Navigation</b><br>
    Use the pages below to explore<br>training results or run a live<br>prediction on your own image.
</div>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:white; border-radius:16px; padding:32px 36px 28px;
            border:1px solid #E2E8F0; box-shadow:0 1px 4px rgba(0,0,0,0.05);
            margin-bottom:28px;">
    <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.12em; color:#3B82F6; margin-bottom:8px;">
        IEEE EMBS Pune Chapter · Student Internship Program 2026
    </div>
    <h1 style="margin:0 0 6px; font-size:2.1rem; font-weight:800; color:#0F172A;
               letter-spacing:-0.02em;">
        🔬 Prognos AI — Skin Cancer Detection
    </h1>
    <p style="margin:0; font-size:1.05rem; color:#475569; font-weight:400; line-height:1.5;">
        Deep learning classification across four benchmark dermoscopy datasets,
        with Grad-CAM explainability for clinical transparency.
    </p>
    <div style="margin-top:20px; display:flex; gap:10px; flex-wrap:wrap;">
        <span style="background:#EFF6FF; color:#1D4ED8; padding:5px 14px; border-radius:20px;
                     font-size:0.8rem; font-weight:600;">EfficientNet-B0</span>
        <span style="background:#ECFDF5; color:#065F46; padding:5px 14px; border-radius:20px;
                     font-size:0.8rem; font-weight:600;">ResNet50</span>
        <span style="background:#F5F3FF; color:#5B21B6; padding:5px 14px; border-radius:20px;
                     font-size:0.8rem; font-weight:600;">DenseNet121</span>
        <span style="background:#FFF7ED; color:#9A3412; padding:5px 14px; border-radius:20px;
                     font-size:0.8rem; font-weight:600;">Grad-CAM XAI</span>
        <span style="background:#F0FDF4; color:#166534; padding:5px 14px; border-radius:20px;
                     font-size:0.8rem; font-weight:600;">PyTorch + timm</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Abstract ──────────────────────────────────────────────────────────────────
with st.expander("📄 Project Abstract", expanded=False):
    st.markdown("""
    This system applies deep learning to analyse dermoscopic images of skin lesions and flag
    potential concerns at an early stage. Three CNN architectures — **EfficientNet-B0**,
    **ResNet50**, and **DenseNet121** — were trained with transfer learning (ImageNet pre-training,
    timm library) across four benchmark datasets spanning binary and multi-class tasks.

    Grad-CAM visualisations highlight which image regions drove each prediction, providing
    clinical interpretability alongside quantitative metrics (Accuracy, F1, ROC-AUC).

    The goal is accessible, explainable preliminary screening — not a replacement for
    dermatologists, but a tool to support earlier detection in under-resourced settings.
    """)

# ── Dataset cards ─────────────────────────────────────────────────────────────
st.markdown("## 📊 Datasets")
cols = st.columns(4)

for col, (name, cfg) in zip(cols, DATASETS.items()):
    accent = DS_COLORS[name]
    icon   = DS_ICONS[name]
    best_m = max(cfg["metrics"], key=lambda m: cfg["metrics"][m]["Accuracy"])
    best_a = cfg["metrics"][best_m]["Accuracy"]
    task_label = "Binary" if cfg["task"] == "binary" else f"{cfg['num_classes']}-class"

    with col:
        st.markdown(f"""
        <div class="ds-card" style="--accent:{accent};">
            <div class="ds-icon">{icon}</div>
            <h4>{name}</h4>
            <p class="ds-desc">{cfg['description']}</p>
            <div class="ds-stat">
                <span class="chip">{task_label}</span>
                <span class="chip">{cfg['num_classes']} classes</span>
            </div>
            <div class="best-line">
                Best: <b>{best_m}</b><br>
                <span class="best-acc" style="color:{accent};">{best_a:.1f}%</span>
                <span style="font-size:0.78rem; color:#64748B;"> accuracy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

# ── Architecture overview ─────────────────────────────────────────────────────
st.markdown("## 🏗️ System Architecture")
arch_col1, arch_col2 = st.columns([1, 2])

with arch_col1:
    st.markdown("""
    <div class="card">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.08em; color:#94A3B8; margin-bottom:12px;">Pipeline</div>
        <ol style="margin:0; padding-left:1.2em; color:#334155; line-height:2.1; font-size:0.9rem;">
            <li>Image upload &amp; preprocessing</li>
            <li>Resize → 224×224, ImageNet normalisation</li>
            <li>CNN classification (3 architectures)</li>
            <li>Grad-CAM explainability overlay</li>
            <li>Prediction report &amp; confidence scores</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

with arch_col2:
    st.markdown("""
    <div class="card">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.08em; color:#94A3B8; margin-bottom:12px;">Frameworks &amp; Components</div>
    """, unsafe_allow_html=True)

    tech_df = pd.DataFrame({
        "Component":  ["Deep Learning", "Model Backbone", "Grad-CAM Target Layers",
                       "Preprocessing", "Web Interface", "Datasets"],
        "Technology": [
            "PyTorch 2.x",
            "EfficientNet-B0 · ResNet50 · DenseNet121 (timm)",
            "conv_head · layer4[-1] · features.norm5",
            "torchvision transforms, OpenCV",
            "Streamlit",
            "ISIC 2018, ISIC 2019, ISIC 2020, MILK10K",
        ]
    })
    st.dataframe(tech_df.set_index("Component"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Key results table ─────────────────────────────────────────────────────────
st.markdown("## 📈 Key Results at a Glance")

rows = []
for ds_name, cfg in DATASETS.items():
    for model, m in cfg["metrics"].items():
        rows.append({
            "Dataset":      ds_name,
            "Model":        model,
            "Accuracy (%)": m["Accuracy"],
            "Precision (%)": m["Precision"],
            "Recall (%)":   m["Recall"],
            "F1 (%)":       m["F1"],
            "ROC-AUC":      m["ROC-AUC"] if m["ROC-AUC"] else "—",
        })

df = pd.DataFrame(rows)
st.dataframe(
    df.set_index(["Dataset", "Model"]).style.format({
        "Accuracy (%)":  "{:.2f}",
        "Precision (%)": "{:.2f}",
        "Recall (%)":    "{:.2f}",
        "F1 (%)":        "{:.2f}",
    }).background_gradient(subset=["Accuracy (%)", "F1 (%)"], cmap="Blues"),
    use_container_width=True,
)

# ── Footer hint ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:32px; padding:16px 20px; background:white; border-radius:10px;
            border:1px solid #E2E8F0; display:flex; align-items:center; gap:12px;">
    <span style="font-size:1.3rem;">👈</span>
    <span style="font-size:0.88rem; color:#475569;">
        Use the <b style="color:#1E293B;">sidebar</b> to open the
        <b style="color:#1E293B;">Results Dashboard</b> or run a
        <b style="color:#1E293B;">Live Prediction</b> on your own image.
    </span>
</div>
""", unsafe_allow_html=True)
