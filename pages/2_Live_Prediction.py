"""
pages/2_Live_Prediction.py — Upload a dermoscopic image and get real-time
classification + Grad-CAM overlay.
"""

import os, sys
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, CKPT, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, ensure_checkpoint
from model_utils import load_model
from gradcam_utils import run_gradcam_pipeline
from styles import inject, sidebar_brand, MODEL_COLORS

st.set_page_config(page_title="Live Prediction · Prognos AI",
                   page_icon="🔬", layout="wide")
inject()

# ── Preprocessing ─────────────────────────────────────────────────────────────
try:
    from torchvision import transforms
    PREPROCESS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    HAS_TV = True
except ImportError:
    HAS_TV = False


@st.cache_resource(show_spinner="Loading model weights…")
def get_model(dataset: str, arch: str):
    ckpt_path, dl_err = ensure_checkpoint(dataset, arch)
    if dl_err:
        return None, f"Could not load checkpoint: {dl_err}"
    num_classes = DATASETS[dataset]["num_classes"]
    try:
        model = load_model(arch, num_classes, ckpt_path)
        return model, None
    except Exception as e:
        return None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_brand()

st.sidebar.markdown("""
<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
            letter-spacing:0.08em; color:#475569; margin-bottom:10px;">
    Configuration
</div>
""", unsafe_allow_html=True)

dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
arch         = st.sidebar.selectbox("Architecture", ["EfficientNet-B0", "ResNet50", "DenseNet121"])
show_gcam    = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
top_k        = st.sidebar.slider("Top-K predictions",
                                  1, min(5, DATASETS[dataset_name]["num_classes"]), 3)

st.sidebar.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="font-size:0.78rem; color:#64748B; line-height:1.7;">
    <b style="color:#94A3B8;">Selected:</b><br>
    <span style="color:{MODEL_COLORS.get(arch,'#CBD5E1')}; font-weight:600;">{arch}</span>
    <br>{dataset_name}
    <br>{DATASETS[dataset_name]['num_classes']} classes
</div>
""", unsafe_allow_html=True)

cfg     = DATASETS[dataset_name]
classes = cfg["classes"]

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
    <h1 style="margin:0 0 4px; font-size:1.75rem; font-weight:800; color:#0F172A;">
        🔬 Live Prediction
    </h1>
    <p style="margin:0; color:#64748B; font-size:0.92rem;">
        Upload a dermoscopic image to receive an AI classification with Grad-CAM explanation.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Upload section ────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:white; border-radius:12px; padding:20px 24px 4px;
            border:1px solid #E2E8F0; margin-bottom:20px;">
    <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.08em; color:#94A3B8; margin-bottom:10px;">
        Upload Image
    </div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop or click to browse — JPG / PNG",
    type=["jpg", "jpeg", "png"],
    help="Use a real dermoscopic image for best results.",
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div style="background:white; border-radius:12px; padding:40px;
                border:1px solid #E2E8F0; text-align:center; color:#94A3B8;">
        <div style="font-size:2rem; margin-bottom:8px;">🖼️</div>
        <div style="font-size:0.9rem; font-weight:500;">
            Upload a dermoscopic image above to begin.
        </div>
        <div style="font-size:0.8rem; margin-top:4px;">
            Use the sidebar to select a dataset and model architecture.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load image & model ────────────────────────────────────────────────────────
image = Image.open(uploaded).convert("RGB")

col_img, col_info = st.columns([1, 2])
with col_img:
    st.markdown("""
    <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.08em; color:#94A3B8; margin-bottom:6px;">
        Input Image
    </div>""", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

with col_info:
    st.markdown(f"""
    <div class="card">
        <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.08em; color:#94A3B8; margin-bottom:12px;">
            Inference Settings
        </div>
        <table style="width:100%; font-size:0.85rem; border-collapse:collapse;">
            <tr><td style="padding:5px 0; color:#64748B; width:40%;">Dataset</td>
                <td style="padding:5px 0; font-weight:600; color:#1E293B;">{dataset_name}</td></tr>
            <tr><td style="padding:5px 0; color:#64748B;">Architecture</td>
                <td style="padding:5px 0; font-weight:600;
                    color:{MODEL_COLORS.get(arch,'#1E293B')};">{arch}</td></tr>
            <tr><td style="padding:5px 0; color:#64748B;">Task</td>
                <td style="padding:5px 0; font-weight:600; color:#1E293B;">
                    {'Binary' if cfg['task']=='binary' else 'Multi-class'} ({cfg['num_classes']} classes)</td></tr>
            <tr><td style="padding:5px 0; color:#64748B;">Classes</td>
                <td style="padding:5px 0; color:#475569; font-size:0.8rem;">
                    {', '.join(classes)}</td></tr>
            <tr><td style="padding:5px 0; color:#64748B;">Input size</td>
                <td style="padding:5px 0; color:#475569;">224 × 224 · ImageNet norm</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner(f"Loading {arch} weights for {dataset_name}…"):
        model, err = get_model(dataset_name, arch)

    if err:
        st.error(f"❌ Could not load checkpoint.\n\n{err}")
        st.stop()

    st.markdown("""
    <div style="display:inline-flex; align-items:center; gap:8px; background:#F0FDF4;
                border:1px solid #86EFAC; border-radius:8px; padding:8px 14px;
                font-size:0.85rem; font-weight:600; color:#166534; margin-top:4px;">
        ✅ Model loaded successfully
    </div>
    """, unsafe_allow_html=True)

if not HAS_TV:
    st.error("torchvision is not installed. Run: `pip install torchvision`")
    st.stop()

# ── Inference ─────────────────────────────────────────────────────────────────
input_tensor = PREPROCESS(image).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    probs  = F.softmax(logits, dim=1)[0]

pred_idx   = probs.argmax().item()
pred_class = classes[pred_idx]
pred_conf  = probs[pred_idx].item() * 100

# Pick card colour based on prediction
is_malignant = any(w in pred_class.upper() for w in ["MALIGNANT", "MEL", "SCC", "BCC"])
is_benign    = any(w in pred_class.upper() for w in ["BENIGN", "BEN", "NV"])
pred_color   = "#DC2626" if is_malignant else ("#16A34A" if is_benign else "#3B82F6")

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("<hr style='margin:24px 0 20px; border-color:#E2E8F0;'>", unsafe_allow_html=True)
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    # Prediction card
    st.markdown(f"""
    <div class="pred-card" style="--pred-color:{pred_color};">
        <p class="pred-tag">Prediction</p>
        <p class="pred-name">{pred_class}</p>
        <p class="pred-pct">{pred_conf:.1f}%</p>
        <p class="pred-sub">Confidence Score</p>
        <div style="margin-top:14px; padding-top:12px; border-top:1px solid #F1F5F9;
                    font-size:0.78rem; color:#94A3B8;">
            {arch} · {dataset_name}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Top-K bar chart
    top_indices = probs.topk(top_k).indices.tolist()
    top_probs   = [probs[i].item() * 100 for i in top_indices]
    top_labels  = [classes[i] for i in top_indices]
    bar_colors  = [pred_color if i == pred_idx else "#CBD5E1" for i in top_indices]

    fig_bar = go.Figure(go.Bar(
        x=top_probs,
        y=top_labels,
        orientation="h",
        marker=dict(color=bar_colors, line_width=0),
        text=[f"<b>{p:.1f}%</b>" for p in top_probs],
        textposition="outside",
        textfont=dict(size=12),
    ))
    fig_bar.update_layout(
        title=dict(text=f"Top-{top_k} Predictions",
                   font=dict(size=13, color="#1E293B", family="Inter"), x=0),
        xaxis=dict(range=[0, 115], title="Confidence (%)",
                   tickfont=dict(size=10), showgrid=True,
                   gridcolor="#F1F5F9"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#334155")),
        height=max(180, top_k * 56),
        margin=dict(l=0, r=10, t=40, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter"),
    )
    fig_bar.update_yaxes(showgrid=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with res_col2:
    if show_gcam:
        with st.spinner("Computing Grad-CAM…"):
            try:
                input_grad = PREPROCESS(image).unsqueeze(0)
                overlay_pil, _, cam_arr = run_gradcam_pipeline(
                    model, arch, input_grad, image, class_idx=pred_idx
                )
                st.markdown("""
                <div style="font-size:0.78rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:0.08em; color:#94A3B8; margin-bottom:10px;">
                    Grad-CAM Explainability
                </div>""", unsafe_allow_html=True)

                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("<div style='font-size:0.8rem; font-weight:600; "
                                "color:#475569; margin-bottom:4px;'>Original</div>",
                                unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with gc2:
                    st.markdown(f"<div style='font-size:0.8rem; font-weight:600; "
                                f"color:{pred_color}; margin-bottom:4px;'>"
                                f"Grad-CAM · {pred_class}</div>",
                                unsafe_allow_html=True)
                    st.image(overlay_pil, use_container_width=True)

                st.markdown("""
                <div style="background:#F8FAFC; border-radius:8px; padding:10px 14px;
                            font-size:0.78rem; color:#64748B; margin-top:8px;
                            border:1px solid #E2E8F0; line-height:1.5;">
                    🌡️ <b>Heatmap guide:</b> Red/yellow regions had the highest influence on
                    the prediction. Blue/green regions were less important to the model.
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"Grad-CAM could not be computed: {e}")
    else:
        st.markdown("""
        <div style="background:white; border-radius:12px; padding:40px 24px;
                    border:1px solid #E2E8F0; text-align:center; color:#94A3B8; height:100%;">
            <div style="font-size:1.8rem; margin-bottom:8px;">🌡️</div>
            <div style="font-size:0.88rem; font-weight:500;">
                Enable <b>Show Grad-CAM heatmap</b> in the sidebar<br>to see spatial attention.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Full probability table ────────────────────────────────────────────────────
with st.expander("📊 Full Probability Distribution", expanded=False):
    prob_df = pd.DataFrame({
        "Class":          classes,
        "Confidence (%)": [f"{p*100:.2f}%" for p in probs.tolist()],
        "Probability":    probs.tolist(),
    }).sort_values("Probability", ascending=False).drop(columns="Probability")
    st.dataframe(prob_df.set_index("Class"), use_container_width=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠️ <b>Medical Disclaimer:</b> This tool is intended for research and educational purposes only.
    It is <b>not</b> a substitute for professional medical diagnosis. Always consult a qualified
    dermatologist for clinical decisions. Model outputs reflect statistical patterns learned
    from training data and may not generalise to all clinical contexts.
</div>
""", unsafe_allow_html=True)
