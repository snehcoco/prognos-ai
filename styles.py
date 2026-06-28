"""
styles.py — Shared CSS and colour constants for all pages.
Call inject() at the top of every page after set_page_config().
"""

import streamlit as st

# ── Consistent colour palette ─────────────────────────────────────────────────
MODEL_COLORS = {
    "EfficientNet-B0": "#3B82F6",   # blue
    "ResNet50":        "#10B981",   # emerald
    "DenseNet121":     "#8B5CF6",   # purple
}

DS_COLORS = {
    "ISIC 2018": "#3B82F6",   # blue
    "ISIC 2019": "#8B5CF6",   # purple
    "ISIC 2020": "#10B981",   # emerald
    "MILK10K":   "#F59E0B",   # amber
}

DS_ICONS = {
    "ISIC 2018": "🧬",
    "ISIC 2019": "🧫",
    "ISIC 2020": "🔭",
    "MILK10K":   "🩺",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, h1, h2, h3, h4, h5, label {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── App background ── */
.stApp { background-color: #F1F5F9; }
.block-container { padding-top: 2rem; }

/* ── Dark sidebar ── */
section[data-testid="stSidebar"] {
    background: #0F172A !important;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
}
section[data-testid="stSidebar"] label {
    color: #94A3B8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ── Sidebar selectbox / slider dark override ── */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSlider {
    background: #1E293B !important;
    border-radius: 8px;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #1E293B !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #E2E8F0;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 6px 16px;
    color: #64748B;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1E293B !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem;
}

/* ── White content cards ── */
.card {
    background: white;
    border-radius: 12px;
    padding: 22px 26px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    margin-bottom: 14px;
}

/* ── Dataset summary cards ── */
.ds-card {
    background: white;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.ds-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: var(--accent, #3B82F6);
    border-radius: 12px 12px 0 0;
}
.ds-card .ds-icon {
    font-size: 1.6rem;
    margin-bottom: 6px;
}
.ds-card h4 {
    margin: 0 0 6px 0;
    font-size: 1rem;
    font-weight: 700;
    color: #1E293B;
}
.ds-card .ds-desc {
    font-size: 0.79rem;
    color: #64748B;
    margin: 0 0 10px 0;
    line-height: 1.45;
}
.ds-card .ds-stat {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.ds-card .chip {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.73rem;
    font-weight: 600;
}
.ds-card .best-line {
    margin-top: 10px;
    font-size: 0.82rem;
    color: #1E293B;
    font-weight: 500;
}
.ds-card .best-acc {
    font-weight: 800;
    font-size: 1.1rem;
    color: var(--accent, #3B82F6);
}

/* ── Section title divider ── */
.stMarkdown h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #1E293B !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E2E8F0;
    margin-top: 1.75rem !important;
}

/* ── Prediction card ── */
.pred-card {
    background: white;
    border-radius: 14px;
    padding: 28px 24px;
    text-align: center;
    border: 2px solid var(--pred-color, #3B82F6);
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.pred-card .pred-tag {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94A3B8;
    margin: 0;
}
.pred-card .pred-name {
    font-size: 1.9rem;
    font-weight: 800;
    margin: 8px 0 4px;
    line-height: 1.1;
    color: var(--pred-color, #1E293B);
}
.pred-card .pred-pct {
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    color: var(--pred-color, #1E293B);
    margin: 0;
}
.pred-card .pred-sub {
    font-size: 0.78rem;
    color: #94A3B8;
    font-weight: 500;
    margin-top: 4px;
}

/* ── Disclaimer banner ── */
.disclaimer {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 4px solid #F97316;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #9A3412;
    line-height: 1.5;
    margin-top: 16px;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame th { background: #F8FAFC; font-weight: 600; }

/* ── Info / warning / success alerts ── */
.stAlert { border-radius: 10px !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px !important;
    border: 2px dashed #CBD5E1 !important;
    background: white !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #1E293B;
    background: white;
    border-radius: 10px !important;
}
</style>
"""


def inject():
    """Inject global CSS. Call once per page, right after set_page_config()."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand():
    """Render consistent sidebar branding."""
    st.sidebar.markdown("""
    <div style="padding: 4px 0 16px;">
        <div style="font-size:1.25rem; font-weight:800; color:#F1F5F9; letter-spacing:-0.01em;">
            🔬 Prognos AI
        </div>
        <div style="font-size:0.72rem; color:#64748B; margin-top:2px; font-weight:500;">
            IEEE EMBS Pune Chapter · SIP 2026
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:0 0 16px;'>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="font-size:0.72rem; font-weight:700; text-transform:uppercase;
                letter-spacing:0.08em; color:#475569; margin-bottom:8px;">Team</div>
    <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.9;">
        Subhasree Paul<br>Snehansha Das<br>Srijoni Sarkar
    </div>
    <div style="margin-top:10px; font-size:0.82rem; color:#94A3B8;">
        <span style="color:#475569; font-weight:600;">Mentor</span><br>Dr. Ajey Kumar
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
