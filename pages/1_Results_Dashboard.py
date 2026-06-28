"""
pages/1_Results_Dashboard.py — Per-dataset metrics tables and training plots.
"""

import os, sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, PLOTS
from styles import inject, sidebar_brand, DS_COLORS, DS_ICONS, MODEL_COLORS

st.set_page_config(page_title="Results Dashboard · Prognos AI",
                   page_icon="📊", layout="wide")
inject()

# ── Sidebar ───────────────────────────────────────────────────────────────────
sidebar_brand()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:24px;">
    <h1 style="margin:0 0 4px; font-size:1.75rem; font-weight:800; color:#0F172A;">
        📊 Results Dashboard
    </h1>
    <p style="margin:0; color:#64748B; font-size:0.92rem;">
        Training metrics, performance charts, and visualisations for all four datasets.
    </p>
</div>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_plot(dataset_name: str, plot_key: str) -> Image.Image | None:
    base = PLOTS.get(dataset_name, "")
    if not base:
        return None
    rel = DATASETS[dataset_name]["plot_files"].get(plot_key, "")
    if not rel:
        return None
    path = os.path.join(base, rel)
    return Image.open(path) if os.path.isfile(path) else None


def metrics_chart(metrics_dict: dict, title: str) -> go.Figure:
    models  = list(metrics_dict.keys())
    metrics = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()
    for model in models:
        vals = [metrics_dict[model].get(m, 0) for m in metrics]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=vals,
            marker_color=MODEL_COLORS.get(model, "#94A3B8"),
            marker_line_width=0,
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="#334155"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1E293B", family="Inter"),
                   x=0, pad=dict(l=0)),
        barmode="group",
        yaxis=dict(range=[0, 115], title="Score (%)", tickfont=dict(size=11),
                   gridcolor="#F1F5F9", gridwidth=1),
        xaxis=dict(tickfont=dict(size=12, color="#334155")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11)),
        height=380,
        margin=dict(t=60, b=20, l=0, r=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter"),
    )
    fig.update_xaxes(showgrid=False)
    return fig


def metrics_table(metrics_dict: dict) -> pd.DataFrame:
    rows = []
    for model, m in metrics_dict.items():
        rows.append({
            "Model":        model,
            "Accuracy":     m["Accuracy"],
            "Precision":    m["Precision"],
            "Recall":       m["Recall"],
            "F1":           m["F1"],
            "ROC-AUC":      f"{m['ROC-AUC']:.4f}" if m["ROC-AUC"] else "—",
        })
    return pd.DataFrame(rows).set_index("Model")


def show_plots(dataset_name: str, plot_order: list):
    available = [(n, load_plot(dataset_name, n)) for n in plot_order]
    available = [(n, img) for n, img in available if img]
    if available:
        st.markdown("**Training Plots**")
        for i in range(0, len(available), 2):
            pair = available[i:i+2]
            c = st.columns(len(pair))
            for col, (name, img) in zip(c, pair):
                with col:
                    st.markdown(f"<div style='font-size:0.82rem; font-weight:600; "
                                f"color:#475569; margin-bottom:4px;'>{name}</div>",
                                unsafe_allow_html=True)
                    st.image(img, use_container_width=True)


def dataset_header(name: str, cfg: dict):
    accent = DS_COLORS[name]
    icon   = DS_ICONS[name]
    st.markdown(f"""
    <div style="background:white; border-radius:10px; padding:16px 20px;
                border:1px solid #E2E8F0; border-left:4px solid {accent};
                margin-bottom:16px; display:flex; align-items:center; gap:14px;">
        <span style="font-size:1.5rem;">{icon}</span>
        <div>
            <div style="font-size:1rem; font-weight:700; color:#1E293B;">{name}</div>
            <div style="font-size:0.8rem; color:#64748B; margin-top:2px;">{cfg['description']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = [f"{DS_ICONS[n]} {n}" for n in ["MILK10K", "ISIC 2020", "ISIC 2018", "ISIC 2019"]]
tabs = st.tabs(tab_labels)

# ════ MILK10K ═════════════════════════════════════════════════════════════════
with tabs[0]:
    cfg = DATASETS["MILK10K"]
    dataset_header("MILK10K", cfg)

    st.markdown(f"<div style='margin-bottom:10px; font-size:0.82rem; color:#64748B;'>"
                f"Classes: {'  ·  '.join(f'<code>{c}</code>' for c in cfg['classes'])}"
                f"</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("**Performance Metrics**")
        tbl = metrics_table(cfg["metrics"])
        st.dataframe(
            tbl.style.format("{:.2f}", subset=["Accuracy","Precision","Recall","F1"])
                     .background_gradient(subset=["Accuracy","F1"], cmap="Blues"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(metrics_chart(cfg["metrics"], "MILK10K — Model Comparison"),
                        use_container_width=True)

    show_plots("MILK10K", ["Class Distribution","Sample Images","Training Curves",
                            "Confusion Matrices","Model Comparison"])

# ════ ISIC 2020 ══════════════════════════════════════════════════════════════
with tabs[1]:
    cfg = DATASETS["ISIC 2020"]
    dataset_header("ISIC 2020", cfg)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("**Performance Metrics**")
        tbl = metrics_table(cfg["metrics"])
        st.dataframe(
            tbl.style.format("{:.2f}", subset=["Accuracy","Precision","Recall","F1"])
                     .background_gradient(subset=["Accuracy","F1"], cmap="Blues"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(metrics_chart(cfg["metrics"], "ISIC 2020 — Model Comparison"),
                        use_container_width=True)

    st.markdown("**Grad-CAM Case Studies (from report)**")
    gcol1, gcol2 = st.columns(2)
    with gcol1:
        st.markdown("""
        <div class="card">
            <div style="font-size:0.8rem; font-weight:700; color:#16A34A; margin-bottom:10px;">
                ✅ Correct Classification — Benign Lesion
            </div>
            <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
                <tr style="background:#F8FAFC;">
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Model</th>
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Pred</th>
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Conf.</th>
                </tr>
                <tr><td style="padding:6px 8px;">EfficientNet-B0</td>
                    <td style="padding:6px 8px; color:#16A34A; font-weight:600;">Benign</td>
                    <td style="padding:6px 8px;">97.35%</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:6px 8px;">ResNet50</td>
                    <td style="padding:6px 8px; color:#16A34A; font-weight:600;">Benign</td>
                    <td style="padding:6px 8px;">68.29%</td></tr>
                <tr><td style="padding:6px 8px;">DenseNet121</td>
                    <td style="padding:6px 8px; color:#16A34A; font-weight:600;">Benign</td>
                    <td style="padding:6px 8px;">96.20%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with gcol2:
        st.markdown("""
        <div class="card">
            <div style="font-size:0.8rem; font-weight:700; color:#DC2626; margin-bottom:10px;">
                ❌ Misclassification — BCC Lesion
            </div>
            <table style="width:100%; font-size:0.8rem; border-collapse:collapse;">
                <tr style="background:#F8FAFC;">
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Model</th>
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">True</th>
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Predicted</th>
                    <th style="padding:6px 8px; text-align:left; color:#64748B; font-weight:600;">Conf.</th>
                </tr>
                <tr><td style="padding:6px 8px;">EfficientNet-B0</td>
                    <td style="padding:6px 8px;">BCC</td>
                    <td style="padding:6px 8px; color:#DC2626;">AKIEC</td>
                    <td style="padding:6px 8px;">39.96%</td></tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:6px 8px;">ResNet50</td>
                    <td style="padding:6px 8px;">BCC</td>
                    <td style="padding:6px 8px; color:#DC2626;">AKIEC</td>
                    <td style="padding:6px 8px;">51.39%</td></tr>
                <tr><td style="padding:6px 8px;">DenseNet121</td>
                    <td style="padding:6px 8px;">BCC</td>
                    <td style="padding:6px 8px; color:#DC2626;">BKL</td>
                    <td style="padding:6px 8px;">32.20%</td></tr>
            </table>
            <div style="font-size:0.75rem; color:#94A3B8; margin-top:8px;">
                All models focused on lesion region; confusion due to visual similarity between BCC, AKIEC, and BKL.
            </div>
        </div>
        """, unsafe_allow_html=True)

    show_plots("ISIC 2020", ["Class Distribution","Sample Images","Training Curves",
                              "Confusion Matrices","Model Comparison"])


# ════ ISIC 2018 ══════════════════════════════════════════════════════════════
with tabs[2]:
    cfg = DATASETS["ISIC 2018"]
    dataset_header("ISIC 2018", cfg)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("**Performance Metrics**")
        tbl = metrics_table(cfg["metrics"])
        st.dataframe(
            tbl.style.format("{:.2f}", subset=["Accuracy","Precision","Recall","F1"])
                     .background_gradient(subset=["Accuracy","F1"], cmap="Blues"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(metrics_chart(cfg["metrics"], "ISIC 2018 — Model Comparison"),
                        use_container_width=True)

    show_plots("ISIC 2018", ["Class Distribution","Sample Images","Training Curves",
                              "ROC Curves","Model Comparison","Confusion Matrices"])

    # Grad-CAM plots from report
    gcam_files = cfg.get("gradcam_files", {})
    base = PLOTS.get("ISIC 2018", "")
    if base and gcam_files:
        gcam_imgs = [(m, os.path.join(base, p)) for m, p in gcam_files.items()
                     if os.path.isfile(os.path.join(base, p))]
        if gcam_imgs:
            st.markdown("**Grad-CAM Visualisations**")
            gcols = st.columns(len(gcam_imgs))
            for col, (model_name, fpath) in zip(gcols, gcam_imgs):
                with col:
                    st.markdown(f"<div style='font-size:0.82rem; font-weight:600; "
                                f"color:{MODEL_COLORS.get(model_name,'#475569')}; "
                                f"margin-bottom:4px;'>{model_name}</div>",
                                unsafe_allow_html=True)
                    st.image(Image.open(fpath), use_container_width=True)

# ════ ISIC 2019 ══════════════════════════════════════════════════════════════
with tabs[3]:
    cfg = DATASETS["ISIC 2019"]
    dataset_header("ISIC 2019", cfg)

    st.markdown(f"<div style='margin-bottom:10px; font-size:0.82rem; color:#64748B;'>"
                f"Classes: {'  ·  '.join(f'<code>{c}</code>' for c in cfg['classes'])}"
                f"</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.markdown("**Performance Metrics**")
        tbl = metrics_table(cfg["metrics"])
        st.dataframe(
            tbl.style.format("{:.2f}", subset=["Accuracy","Precision","Recall","F1"])
                     .background_gradient(subset=["Accuracy","F1"], cmap="Purples"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(metrics_chart(cfg["metrics"], "ISIC 2019 — Model Comparison"),
                        use_container_width=True)

    show_plots("ISIC 2019", ["Class Distribution","Sample Images","Training Curves",
                              "Model Comparison","Confusion Matrices"])


# ── Cross-dataset comparison ──────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<h2 style="margin:0 0 4px; font-size:1.25rem; font-weight:700; color:#0F172A;">
    🔀 Cross-Dataset Accuracy Comparison
</h2>
<p style="margin:0 0 16px; font-size:0.85rem; color:#64748B;">
    How each architecture performs across all four datasets.
</p>
""", unsafe_allow_html=True)

all_rows = []
for ds_name, cfg in DATASETS.items():
    for model, m in cfg["metrics"].items():
        all_rows.append({
            "Dataset": ds_name, "Model": model,
            "Accuracy": m["Accuracy"], "F1": m["F1"],
        })
df_all = pd.DataFrame(all_rows)

fig_cross = go.Figure()
for model in ["EfficientNet-B0", "ResNet50", "DenseNet121"]:
    sub = df_all[df_all["Model"] == model]
    fig_cross.add_trace(go.Scatter(
        x=sub["Dataset"], y=sub["Accuracy"],
        mode="lines+markers+text",
        name=model,
        line=dict(color=MODEL_COLORS[model], width=2.5),
        marker=dict(size=10, color=MODEL_COLORS[model],
                    line=dict(width=2, color="white")),
        text=[f"<b>{v:.1f}%</b>" for v in sub["Accuracy"]],
        textposition="top center",
        textfont=dict(size=11),
    ))

fig_cross.update_layout(
    yaxis=dict(title="Accuracy (%)", range=[40, 105],
               gridcolor="#F1F5F9", tickfont=dict(size=11)),
    xaxis=dict(tickfont=dict(size=12, color="#334155")),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=12)),
    height=380,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=20, b=20, l=0, r=0),
    font=dict(family="Inter"),
)
fig_cross.update_xaxes(showgrid=False)
st.plotly_chart(fig_cross, use_container_width=True)
