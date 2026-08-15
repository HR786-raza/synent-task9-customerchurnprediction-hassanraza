"""
Customer Churn Prediction & Retention Analytics
TelePredict — Production Streamlit App
"""

import os, json, warnings, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="TelePredict · Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data",   "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.joblib")
META_PATH  = os.path.join(BASE_DIR, "models", "metadata.json")

# ─── Session state ────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}
.stApp { background: #0B0D17 !important; }
.block-container {
    padding: 1.8rem 2.2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0D1025 !important;
    border-right: 1px solid #1C1F3A !important;
    width: 230px !important;
    min-width: 230px !important;
    max-width: 230px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
    overflow-x: hidden !important;
}

/* ── ALL sidebar buttons: shared base ── */
section[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 10px 10px 0 !important;
    color: #6B7299 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 11px 18px !important;
    width: 100% !important;
    text-align: left !important;
    cursor: pointer !important;
    margin: 1px 0 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #1A1E38 !important;
    color: #C8CCEC !important;
    border-left-color: #3D4580 !important;
    transform: none !important;
}
section[data-testid="stSidebar"] .stButton button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* ── Active nav button wrapper ── */
.nav-active .stButton button {
    background: linear-gradient(90deg, #1C2B6A22, #1A256008) !important;
    border-left: 3px solid #4F8EF7 !important;
    color: #4F8EF7 !important;
    font-weight: 700 !important;
}
.nav-active .stButton button:hover {
    background: linear-gradient(90deg, #1C2B6A33, #1A256015) !important;
    color: #6FA8FF !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0D1025; }
::-webkit-scrollbar-thumb { background: #1C1F3A; border-radius: 3px; }

/* ── Page banner ── */
.page-banner {
    background: linear-gradient(135deg, #0E1535 0%, #131A3E 60%, #0E1230 100%);
    border: 1px solid #1C1F3A;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-banner::after {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, #4F8EF712 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.page-banner h1 {
    font-size: 1.85rem; font-weight: 800;
    color: #E4E7FF; margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.page-banner p {
    font-size: 0.92rem; color: #6B7299; margin: 0; line-height: 1.55;
}

/* ── KPI card ── */
.kpi-card {
    background: #131629;
    border: 1px solid #1C1F3A;
    border-radius: 14px;
    padding: 20px 18px 16px;
    text-align: center;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #2A2F5A; }
.kpi-icon { font-size: 1.3rem; margin-bottom: 8px; line-height: 1; }
.kpi-val  { font-size: 1.85rem; font-weight: 800; line-height: 1; margin: 2px 0 7px; letter-spacing: -1px; }
.kpi-lbl  { font-size: 0.65rem; font-weight: 700; color: #4A5080; text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Section heading ── */
.section-heading {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 16px;
}
.section-heading .bar {
    width: 4px; height: 20px; border-radius: 2px; flex-shrink: 0;
    background: linear-gradient(180deg, #4F8EF7, #9775FA);
}
.section-heading .title {
    font-size: 1.05rem; font-weight: 700; color: #E4E7FF; letter-spacing: -0.2px;
}

/* ── Insight card ── */
.insight-card {
    background: #131629;
    border: 1px solid #1C1F3A;
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
}
.insight-card .ic-icon  { font-size: 1.3rem; margin-bottom: 8px; }
.insight-card .ic-stat  { font-size: 1.5rem; font-weight: 800; margin: 4px 0 2px; letter-spacing: -0.5px; }
.insight-card .ic-title { font-size: 0.85rem; font-weight: 700; color: #C8CCEC; margin-bottom: 6px; }
.insight-card .ic-body  { font-size: 0.78rem; color: #6B7299; line-height: 1.6; }

/* ── Chart wrapper ── */
.chart-card {
    background: #131629;
    border: 1px solid #1C1F3A;
    border-radius: 14px;
    padding: 4px;
    overflow: hidden;
}

/* ── Form ── */
.form-label {
    font-size: 0.65rem; font-weight: 700; color: #4A5080;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin: 20px 0 10px;
    display: flex; align-items: center; gap: 10px;
}
.form-label::after { content: ''; flex: 1; height: 1px; background: #1C1F3A; }
div[data-testid="stSelectbox"] > label,
div[data-testid="stNumberInput"] > label,
div[data-testid="stSlider"]     > label {
    font-size: 0.8rem !important; font-weight: 500 !important; color: #8890B5 !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #181B2E !important; border: 1px solid #1C1F3A !important; border-radius: 8px !important;
}
div[data-testid="stNumberInput"] > div > div {
    background: #181B2E !important; border: 1px solid #1C1F3A !important; border-radius: 8px !important;
}

/* ── Predict button ── */
.stForm .stButton button {
    background: linear-gradient(135deg, #4F8EF7, #7B5CEA) !important;
    color: white !important; border: none !important; border-left: none !important;
    border-radius: 10px !important; font-size: 0.95rem !important;
    font-weight: 700 !important; padding: 13px !important;
    width: 100% !important; letter-spacing: 0.01em !important;
    box-shadow: 0 4px 20px #4F8EF728 !important;
    transform: none !important;
}
.stForm .stButton button:hover {
    opacity: 0.88 !important; border-left: none !important;
}

/* ── Result banner ── */
.result-churn {
    background: linear-gradient(135deg, #1F0A0A, #2E1010);
    border: 1px solid #F75C5C28; border-radius: 18px;
    padding: 32px 36px; text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #071A10, #0C2A1C);
    border: 1px solid #2DCE8928; border-radius: 18px;
    padding: 32px 36px; text-align: center;
}
.result-icon  { font-size: 3rem; margin-bottom: 10px; line-height: 1; }
.result-title { font-size: 1.55rem; font-weight: 800; margin: 0 0 10px; letter-spacing: -0.4px; }
.result-pct   { font-size: 3.2rem; font-weight: 900; letter-spacing: -2px; line-height: 1; margin: 8px 0 2px; }
.result-sub   { font-size: 0.78rem; color: #6B7299; }

/* ── Risk pill ── */
.pill-high   { display:inline-block; background:#F75C5C15; color:#F75C5C; border:1px solid #F75C5C35; border-radius:20px; padding:4px 14px; font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin:6px 0; }
.pill-medium { display:inline-block; background:#FFA94D15; color:#FFA94D; border:1px solid #FFA94D35; border-radius:20px; padding:4px 14px; font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin:6px 0; }
.pill-low    { display:inline-block; background:#2DCE8915; color:#2DCE89; border:1px solid #2DCE8935; border-radius:20px; padding:4px 14px; font-size:0.72rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; margin:6px 0; }

/* ── Prob bar ── */
.prob-track { background: #1C1F3A; border-radius: 99px; height: 9px; margin: 8px 0 4px; overflow: hidden; }
.prob-fill  { height: 100%; border-radius: 99px; }

/* ── Snapshot box ── */
.snapshot-box {
    background: #181B2E; border: 1px solid #1C1F3A;
    border-radius: 12px; padding: 14px 16px;
}
.snap-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid #1C1F3A; font-size: 0.83rem;
}
.snap-row:last-child { border-bottom: none; }
.snap-lbl { color: #6B7299; }
.snap-val { color: #E4E7FF; font-weight: 600; }

/* ── Rec card ── */
.rec-card {
    background: #181B2E; border: 1px solid #1C1F3A;
    border-radius: 12px; padding: 16px 18px; height: 100%;
}
.rec-icon  { font-size: 1.3rem; margin-bottom: 8px; }
.rec-title { font-size: 0.88rem; font-weight: 700; margin-bottom: 6px; }
.rec-body  { font-size: 0.78rem; color: #6B7299; line-height: 1.65; }

/* ── Model comparison table ── */
div[data-testid="stDataFrame"] table thead tr th {
    background: #181B2E !important; color: #6B7299 !important; font-size: 0.75rem !important;
}
div[data-testid="stDataFrame"] table tbody tr:hover td { background: #1A1E38 !important; }

/* ── CM mini card ── */
.cm-card {
    border-radius: 10px; padding: 12px 14px; margin-bottom: 0;
}

/* ── Metric bar ── */
.mbar-row  { display:flex; align-items:center; gap:10px; margin-bottom:12px; }
.mbar-lbl  { font-size:0.76rem; font-weight:600; color:#6B7299; width:80px; flex-shrink:0; }
.mbar-track{ flex:1; background:#1C1F3A; border-radius:99px; height:7px; overflow:hidden; }
.mbar-fill { height:100%; border-radius:99px; }
.mbar-val  { font-size:0.8rem; font-weight:700; color:#E4E7FF; width:42px; text-align:right; flex-shrink:0; }

/* ── Stat row ── */
.srow { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #1C1F3A; font-size:0.83rem; }
.srow:last-child { border-bottom: none; }
.srow-lbl { color:#6B7299; }
.srow-val { color:#E4E7FF; font-weight:600; }

/* ── About card ── */
.about-card { background:#131629; border:1px solid #1C1F3A; border-radius:14px; padding:22px 24px; margin-bottom:14px; }
.about-card h3 { font-size:0.95rem; font-weight:700; color:#E4E7FF; margin:0 0 10px; }
.about-card p  { font-size:0.85rem; color:#6B7299; line-height:1.75; margin:0; }

/* ── Expander ── */
div[data-testid="stExpander"] { background:#131629 !important; border:1px solid #1C1F3A !important; border-radius:12px !important; }
summary { color:#8890B5 !important; font-size:0.85rem !important; font-weight:600 !important; }

/* ── Misc ── */
hr { border-color: #1C1F3A !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)

@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df

def mk_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#131629")
    ax.set_facecolor("#131629")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1C1F3A")
    ax.tick_params(colors="#4A5080", labelsize=8)
    ax.xaxis.label.set_color("#4A5080")
    ax.yaxis.label.set_color("#4A5080")
    ax.title.set_color("#E4E7FF")
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax

def section(title):
    st.markdown(f"""
    <div class="section-heading">
        <div class="bar"></div>
        <div class="title">{title}</div>
    </div>""", unsafe_allow_html=True)

# ─── Load artifacts ───────────────────────────────────────────────────────────
try:
    model    = load_model()
    meta     = load_metadata()
    df_raw   = load_raw_data()
    ok = True
except Exception as e:
    ok = False
    err_msg = str(e)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  — pure st.button navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:26px 16px 20px; text-align:center; border-bottom:1px solid #1C1F3A;">
        <div style="font-size:2.6rem; line-height:1;">📡</div>
        <div style="font-size:1.1rem; font-weight:800; color:#E4E7FF; margin-top:9px; letter-spacing:-0.2px;">
            TelePredict
        </div>
        <div style="font-size:0.67rem; color:#4A5080; margin-top:3px;
                    text-transform:uppercase; letter-spacing:0.12em;">
            Churn Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav label
    st.markdown("""
    <div style="padding:14px 18px 6px; font-size:0.62rem; font-weight:700;
                color:#4A5080; text-transform:uppercase; letter-spacing:0.14em;">
        Main Menu
    </div>
    """, unsafe_allow_html=True)

    # Nav items
    nav_items = [
        ("Dashboard",         "🏠", "Dashboard"),
        ("Predict Customer",  "🔮", "Predict Customer"),
        ("Model Performance", "📊", "Model Performance"),
        ("About",             "ℹ️",  "About"),
    ]
    for key, icon, label in nav_items:
        is_active = st.session_state.page == key
        wrap_cls = "nav-active" if is_active else "nav-inactive"
        st.markdown(f'<div class="{wrap_cls}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Model info panel
    st.markdown("""
    <div style="margin:18px 14px 0; border-top:1px solid #1C1F3A; padding-top:16px;">
        <div style="font-size:0.62rem; font-weight:700; color:#4A5080;
                    text-transform:uppercase; letter-spacing:0.14em; margin-bottom:12px;">
            Model Info
        </div>
    </div>
    """, unsafe_allow_html=True)

    info_rows = [
        ("Algorithm",  "Gradient Boosting", "#E4E7FF"),
        ("ROC-AUC",    "84.4%",             "#2DCE89"),
        ("Accuracy",   "80.4%",             "#4F8EF7"),
        ("Dataset",    "7,043 rows",         "#E4E7FF"),
        ("Version",    "v1.0",              "#9775FA"),
    ]
    for lbl, val, col in info_rows:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    font-size:0.76rem; padding:5px 14px;">
            <span style="color:#4A5080;">{lbl}</span>
            <span style="color:{col}; font-weight:600;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    # Status badge
    st.markdown("""
    <div style="margin:16px 14px 20px; background:#0F1822;
                border:1px solid #163328; border-radius:10px;
                padding:9px 14px; display:flex; align-items:center; gap:9px;">
        <div style="width:7px; height:7px; border-radius:50%; background:#2DCE89;
                    box-shadow:0 0 7px #2DCE8980; flex-shrink:0;"></div>
        <span style="font-size:0.73rem; color:#6B7299;">Model Active &amp; Ready</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Active page ──────────────────────────────────────────────────────────────
page = st.session_state.page

if not ok:
    st.error(f"⚠️ Could not load model artifacts. Run `python src/train_model.py` first.\n\n`{err_msg}`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.markdown("""
    <div class="page-banner">
        <h1>Customer Churn Analytics</h1>
        <p>Monitor churn trends, explore key risk drivers, and understand your customer base at a glance.</p>
    </div>""", unsafe_allow_html=True)

    di = meta["dataset_info"]
    m  = meta["best_metrics"]
    churn_n = int(df_raw["Churn"].map({"Yes":1,"No":0}).sum())
    safe_n  = len(df_raw) - churn_n

    # ── KPI row ──────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "👥", f"{di['total_rows']:,}", "Total Customers",    "#4F8EF7"),
        (k2, "⚠️", f"{churn_n:,}",           "Churned",            "#F75C5C"),
        (k3, "✅", f"{safe_n:,}",             "Retained",           "#2DCE89"),
        (k4, "🎯", f"{m['ROC-AUC']:.1%}",    "ROC-AUC Score",     "#9775FA"),
        (k5, "📈", f"{di['churn_rate']:.1%}", "Churn Rate",        "#FFA94D"),
    ]
    for col, icon, val, lbl, color in kpis:
        col.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-val" style="color:{color};">{val}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # ── Row 1: Donut + Contract ───────────────────────────────────
    section("Churn Distribution & Contract Analysis")
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        counts = df_raw["Churn"].value_counts()
        fig, ax = mk_fig(6, 4)
        ax.pie(
            counts, labels=["Retained","Churned"],
            autopct="%1.1f%%", colors=["#2DCE89","#F75C5C"],
            startangle=90,
            wedgeprops=dict(edgecolor="#131629", linewidth=2.5, width=0.58),
            pctdistance=0.78,
        )
        for text in ax.texts:
            text.set_fontsize(9)
            text.set_color("#8890B5" if "%" not in text.get_text() else "white")
        ax.set_title("Overall Churn Distribution", pad=12)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        ct = df_raw.groupby("Contract")["Churn"].apply(
            lambda x: (x=="Yes").mean()
        ).sort_values()
        fig, ax = mk_fig(6, 4)
        bars = ax.barh(ct.index, ct.values,
                       color=["#2DCE89","#FFA94D","#F75C5C"],
                       edgecolor="none", height=0.45)
        ax.set_xlabel("Churn Rate")
        ax.set_title("Churn Rate by Contract Type", pad=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
        ax.set_xlim(0, ct.max()*1.28)
        for bar in bars:
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.1%}", va="center",
                    color="#C8CCEC", fontsize=8.5, fontweight="600")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Tenure + Monthly charges ──────────────────────────
    section("Tenure & Monthly Charges Analysis")
    col3, col4 = st.columns(2, gap="medium")

    with col3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = mk_fig(6, 4)
        for lbl, mask, col_h in [
            ("Retained", df_raw["Churn"]=="No",  "#2DCE89"),
            ("Churned",  df_raw["Churn"]=="Yes", "#F75C5C"),
        ]:
            ax.hist(df_raw.loc[mask, "tenure"], bins=24, alpha=0.72,
                    color=col_h, label=lbl, edgecolor="none")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Customers")
        ax.set_title("Churn by Tenure")
        ax.legend(facecolor="#181B2E", edgecolor="#1C1F3A",
                  labelcolor="#C8CCEC", fontsize=8)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig, ax = mk_fig(6, 4)
        for lbl, mask, col_h in [
            ("Retained", df_raw["Churn"]=="No",  "#2DCE89"),
            ("Churned",  df_raw["Churn"]=="Yes", "#F75C5C"),
        ]:
            ax.hist(df_raw.loc[mask, "MonthlyCharges"], bins=24, alpha=0.72,
                    color=col_h, label=lbl, edgecolor="none")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Customers")
        ax.set_title("Churn by Monthly Charges")
        ax.legend(facecolor="#181B2E", edgecolor="#1C1F3A",
                  labelcolor="#C8CCEC", fontsize=8)
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Internet + Payment ────────────────────────────────
    section("Service & Payment Patterns")
    col5, col6 = st.columns(2, gap="medium")

    with col5:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        grp = df_raw.groupby("InternetService")["Churn"].apply(
            lambda x: (x=="Yes").mean()
        ).sort_values()
        fig, ax = mk_fig(6, 4)
        bars = ax.bar(grp.index, grp.values,
                      color=["#2DCE89","#4F8EF7","#F75C5C"],
                      edgecolor="none", width=0.45)
        ax.set_ylabel("Churn Rate")
        ax.set_title("Churn Rate by Internet Service")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
        ax.set_ylim(0, grp.max()*1.22)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.1%}", ha="center",
                    color="#C8CCEC", fontsize=8.5, fontweight="600")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        pm = df_raw.groupby("PaymentMethod")["Churn"].apply(
            lambda x: (x=="Yes").mean()
        ).sort_values()
        rename = {
            "Electronic check":             "E-Check",
            "Mailed check":                 "Mailed Check",
            "Bank transfer (automatic)":    "Bank Transfer*",
            "Credit card (automatic)":      "Credit Card*",
        }
        pm.index = [rename.get(k,k) for k in pm.index]
        fig, ax = mk_fig(6, 4)
        bars = ax.barh(pm.index, pm.values,
                       color=["#2DCE89","#4F8EF7","#FFA94D","#F75C5C"],
                       edgecolor="none", height=0.42)
        ax.set_xlabel("Churn Rate")
        ax.set_title("Churn Rate by Payment Method")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0%}"))
        ax.set_xlim(0, pm.max()*1.28)
        for bar in bars:
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.1%}", va="center",
                    color="#C8CCEC", fontsize=8.5, fontweight="600")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Key insights ─────────────────────────────────────────────
    section("Key Business Insights")
    i1, i2, i3, i4 = st.columns(4, gap="medium")
    insights = [
        (i1, "📅", "43%",  "#F75C5C", "Month-to-Month Churn",
         "Monthly contract customers churn at 43% vs just 3% on 2-year plans. Contract upgrades are the #1 retention lever."),
        (i2, "💳", "45%",  "#FFA94D", "E-Check Churn Rate",
         "Electronic check users churn most. Auto-pay enrollment strongly correlates with lower churn."),
        (i3, "🌐", "42%",  "#9775FA", "Fiber Optic Churn",
         "Despite fast speeds, fiber customers churn at 42%. Price sensitivity is high — targeted offers can help."),
        (i4, "⏱️", "0–12", "#4F8EF7", "Highest Risk Tenure",
         "The first 12 months are critical. Customers past year one churn at far lower rates — prioritise early engagement."),
    ]
    for col, icon, stat, color, title, body in insights:
        col.markdown(f"""
        <div class="insight-card">
            <div class="ic-icon">{icon}</div>
            <div class="ic-stat" style="color:{color};">{stat}</div>
            <div class="ic-title">{title}</div>
            <div class="ic-body">{body}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Customer":
    st.markdown("""
    <div class="page-banner">
        <h1>Customer Churn Predictor</h1>
        <p>Enter customer details to get an instant churn probability score and retention recommendations.</p>
    </div>""", unsafe_allow_html=True)

    with st.form("predict_form"):

        st.markdown('<div class="form-label">Personal Information</div>', unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        gender     = p1.selectbox("Gender",         ["Male","Female"])
        senior     = p2.selectbox("Senior Citizen", ["No","Yes"])
        partner    = p3.selectbox("Partner",        ["Yes","No"])
        dependents = p4.selectbox("Dependents",     ["No","Yes"])

        st.markdown('<div class="form-label">Account & Billing</div>', unsafe_allow_html=True)
        a1, a2, a3 = st.columns(3)
        tenure        = a1.slider("Tenure (months)", 0, 72, 24)
        monthly_chg   = a2.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        total_default = float(round(max(tenure * monthly_chg, monthly_chg), 2))
        total_chg     = a3.number_input("Total Charges ($)", 0.0, 9000.0, total_default, step=1.0)

        a4, a5, a6 = st.columns(3)
        contract       = a4.selectbox("Contract",         ["Month-to-month","One year","Two year"])
        payment_method = a5.selectbox("Payment Method",   ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        paperless      = a6.selectbox("Paperless Billing",["Yes","No"])

        st.markdown('<div class="form-label">Phone & Internet</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        phone_svc    = s1.selectbox("Phone Service",    ["Yes","No"])
        multi_lines  = s2.selectbox("Multiple Lines",   ["No","Yes","No phone service"])
        internet_svc = s3.selectbox("Internet Service", ["DSL","Fiber optic","No"])

        st.markdown('<div class="form-label">Security & Support Add-ons</div>', unsafe_allow_html=True)
        s4, s5, s6 = st.columns(3)
        online_sec  = s4.selectbox("Online Security",   ["No","Yes","No internet service"])
        online_bkp  = s5.selectbox("Online Backup",     ["No","Yes","No internet service"])
        device_prot = s6.selectbox("Device Protection", ["No","Yes","No internet service"])

        st.markdown('<div class="form-label">Streaming & Tech Support</div>', unsafe_allow_html=True)
        s7, s8, s9 = st.columns(3)
        tech_sup   = s7.selectbox("Tech Support",     ["No","Yes","No internet service"])
        stream_tv  = s8.selectbox("Streaming TV",     ["No","Yes","No internet service"])
        stream_mov = s9.selectbox("Streaming Movies", ["No","Yes","No internet service"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        t = max(tenure, 1)
        svc_count = sum([phone_svc=="Yes", multi_lines=="Yes", online_sec=="Yes",
                         online_bkp=="Yes", device_prot=="Yes", tech_sup=="Yes",
                         stream_tv=="Yes", stream_mov=="Yes"])
        has_stream = int(stream_tv=="Yes" or stream_mov=="Yes")
        avg_spend  = total_chg / t
        tg = ("0-12 mo" if tenure<=12 else "13-24 mo" if tenure<=24
              else "25-48 mo" if tenure<=48 else "49-60 mo" if tenure<=60 else "61-72 mo")

        row = pd.DataFrame([{
            "gender":gender, "SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner, "Dependents":dependents,
            "tenure":tenure, "PhoneService":phone_svc,
            "MultipleLines":multi_lines, "InternetService":internet_svc,
            "OnlineSecurity":online_sec, "OnlineBackup":online_bkp,
            "DeviceProtection":device_prot, "TechSupport":tech_sup,
            "StreamingTV":stream_tv, "StreamingMovies":stream_mov,
            "Contract":contract, "PaperlessBilling":paperless,
            "PaymentMethod":payment_method,
            "MonthlyCharges":monthly_chg, "TotalCharges":total_chg,
            "ServiceCount":svc_count, "AvgMonthlySpend":avg_spend,
            "HasStreaming":has_stream, "TenureGroup":tg,
        }])

        try:
            prob = model.predict_proba(row)[0][1]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        pct  = prob * 100
        pred = int(prob >= 0.5)

        if prob >= 0.70:   risk_cls, risk_lbl, risk_icon = "pill-high",   "HIGH RISK",   "🔴"
        elif prob >= 0.40: risk_cls, risk_lbl, risk_icon = "pill-medium", "MEDIUM RISK", "🟡"
        else:              risk_cls, risk_lbl, risk_icon = "pill-low",    "LOW RISK",    "🟢"

        bar_color = "#F75C5C" if prob>=0.70 else "#FFA94D" if prob>=0.40 else "#2DCE89"

        section("Prediction Result")
        res_col, gauge_col = st.columns([3, 2], gap="large")

        with res_col:
            cls   = "result-churn" if pred==1 else "result-safe"
            color = "#F75C5C"      if pred==1 else "#2DCE89"
            icon  = "⚠️"           if pred==1 else "✅"
            title = "Customer Likely to Churn" if pred==1 else "Customer Unlikely to Churn"
            st.markdown(f"""
            <div class="{cls}">
                <div class="result-icon">{icon}</div>
                <div class="result-title" style="color:{color};">{title}</div>
                <span class="{risk_cls}">{risk_icon} {risk_lbl}</span>
                <div class="result-pct" style="color:{color};">{pct:.1f}%</div>
                <div class="result-sub">Churn Probability Score</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:14px;">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.76rem;color:#4A5080;margin-bottom:5px;">
                    <span>Probability Meter</span>
                    <span style="color:{bar_color};font-weight:700;">{pct:.1f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{min(pct,100):.1f}%;background:{bar_color};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            font-size:0.67rem;color:#4A5080;margin-top:3px;">
                    <span>0% — Safe</span><span>100% — Definite Churn</span>
                </div>
            </div>""", unsafe_allow_html=True)

        with gauge_col:
            # Gauge
            fig2, ax2 = plt.subplots(figsize=(4, 3), subplot_kw=dict(polar=True))
            fig2.patch.set_facecolor("#131629")
            ax2.set_facecolor("#131629")
            theta_bg = np.linspace(0, np.pi, 300)
            ax2.plot(theta_bg, np.ones(300), color="#1C1F3A", lw=18, solid_capstyle="round")
            fill_end = np.pi * min(prob, 1.0)
            if fill_end > 0:
                theta_fill = np.linspace(0, fill_end, 300)
                ax2.plot(theta_fill, np.ones(300), color=bar_color, lw=18, solid_capstyle="round")
            ax2.set_theta_zero_location("W")
            ax2.set_theta_direction(-1)
            ax2.set_ylim(0, 1.6)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.spines["polar"].set_visible(False)
            ax2.text(np.pi/2, 0.1,  f"{pct:.0f}%",     ha="center", va="center",
                     fontsize=26, fontweight="bold", color=bar_color, transform=ax2.transData)
            ax2.text(np.pi/2, -0.62, "Churn Risk",     ha="center", va="center",
                     fontsize=8, color="#4A5080", transform=ax2.transData)
            ax2.text(0.02,  0.10, "0%",   ha="center", color="#4A5080", fontsize=7, transform=ax2.transAxes)
            ax2.text(0.98, 0.10, "100%", ha="center", color="#4A5080", fontsize=7, transform=ax2.transAxes)
            fig2.subplots_adjust(top=0.95, bottom=0.0)
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            # Snapshot
            st.markdown(f"""
            <div class="snapshot-box">
                <div style="font-size:0.62rem;font-weight:700;color:#4A5080;
                            text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">
                    Customer Snapshot
                </div>
                <div class="snap-row"><span class="snap-lbl">Tenure</span><span class="snap-val">{tenure} months</span></div>
                <div class="snap-row"><span class="snap-lbl">Monthly</span><span class="snap-val">${monthly_chg:.0f}</span></div>
                <div class="snap-row"><span class="snap-lbl">Contract</span><span class="snap-val">{contract.split()[0]}</span></div>
                <div class="snap-row"><span class="snap-lbl">Services</span><span class="snap-val">{svc_count} active</span></div>
                <div class="snap-row"><span class="snap-lbl">Internet</span><span class="snap-val">{internet_svc}</span></div>
            </div>""", unsafe_allow_html=True)

        # Recommendations
        section("Retention Recommendations")
        if prob >= 0.70:
            recs = [
                ("🎁","#F75C5C","Loyalty Discount",   "Offer 15–20% off the next 3 months to immediately reduce the motivation to switch."),
                ("📝","#FFA94D","Contract Upgrade",    "Pitch a 1- or 2-year plan with waived setup fees. This single action most dramatically cuts future churn."),
                ("🛡️","#9775FA","Service Bundle",      "Combine tech support, online security, and device protection at a discounted price to increase stickiness."),
                ("📞","#4F8EF7","Proactive Outreach",  "Assign an account manager to call within 48 hours with a personalised retention offer."),
            ]
        elif prob >= 0.40:
            recs = [
                ("🌟","#FFA94D","Loyalty Reward",    "A surprise thank-you discount shows appreciation before dissatisfaction compounds."),
                ("📊","#4F8EF7","Plan Review",        "Offer a free plan audit to ensure the customer is on the right package for their usage."),
                ("💬","#9775FA","NPS Survey",         "Trigger a short satisfaction survey to surface and resolve hidden friction early."),
            ]
        else:
            recs = [
                ("✅","#2DCE89","Maintain Engagement", "Regular touchpoints via newsletters and usage reports reinforce value continuously."),
                ("📈","#4F8EF7","Upsell Opportunity",  "This loyal customer is an ideal candidate for premium tier or add-on upgrades."),
                ("🔔","#9775FA","Quarterly Check-in",  "Schedule a quarterly review to catch any emerging signals before risk grows."),
            ]
        rec_cols = st.columns(len(recs))
        for col, (icon2, color2, title2, body2) in zip(rec_cols, recs):
            col.markdown(f"""
            <div class="rec-card">
                <div class="rec-icon">{icon2}</div>
                <div class="rec-title" style="color:{color2};">{title2}</div>
                <div class="rec-body">{body2}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("""
    <div class="page-banner">
        <h1>Model Performance</h1>
        <p>Full evaluation report — model comparison, confusion matrix, ROC curve, and feature importance.</p>
    </div>""", unsafe_allow_html=True)

    m   = meta["best_metrics"]
    mc  = meta["model_comparison"]
    cm  = meta["confusion_matrix"]
    fi  = meta.get("feature_importance", {})
    bn  = meta["best_model_name"]

    # Champion KPIs
    section(f"Champion Model — {bn}")
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, lbl, val, color, icon in [
        (k1,"Accuracy",  m["Accuracy"],  "#4F8EF7","📋"),
        (k2,"Precision", m["Precision"], "#9775FA","🎯"),
        (k3,"Recall",    m["Recall"],    "#F75C5C","📣"),
        (k4,"F1 Score",  m["F1 Score"],  "#FFA94D","⚖️"),
        (k5,"ROC-AUC",   m["ROC-AUC"],   "#2DCE89","📈"),
    ]:
        col.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {color};">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-val" style="color:{color};">{val:.1%}</div>
            <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    # Model comparison table
    section("All Models Compared")
    comp_df = pd.DataFrame(mc).T.reset_index()
    comp_df.columns = ["Model","Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
    st.dataframe(
        comp_df.style
            .highlight_max(subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"], color="#1A2E5A")
            .format({c:"{:.4f}" for c in ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]}),
        use_container_width=True, hide_index=True,
    )
    st.markdown(f"<div style='font-size:0.8rem;color:#4A5080;margin-top:6px;'>"
                f"Best model highlighted. <strong style='color:#4F8EF7;'>{bn}</strong> selected by highest ROC-AUC.</div>",
                unsafe_allow_html=True)

    with st.expander("What do these metrics mean for the business?"):
        st.markdown("""
| Metric | What it tells you | Business impact |
|---|---|---|
| **Accuracy** | % of all predictions correct | General reliability |
| **Precision** | Of flagged churners, how many actually churn | Controls wasted retention spend |
| **Recall** | Of actual churners, how many we caught | Revenue saved — the most critical metric |
| **F1 Score** | Balance of Precision & Recall | Overall model quality |
| **ROC-AUC** | Discrimination power across all thresholds | Best single ranking metric |

> **Key:** Missing a churner (False Negative) costs a full customer lifetime value. A false alarm costs only one retention incentive. **Recall and ROC-AUC are prioritised.**
        """)

    # Confusion matrix + ROC
    section("Confusion Matrix & ROC Curve")
    cm_col, roc_col = st.columns(2, gap="large")

    with cm_col:
        tn,fp,fn,tp = cm["tn"],cm["fp"],cm["fn"],cm["tp"]
        fig, ax = mk_fig(5.5, 4.5)
        sns.heatmap(
            np.array([[tn,fp],[fn,tp]]),
            annot=True, fmt="d",
            cmap=sns.light_palette("#4F8EF7", as_cmap=True),
            xticklabels=["Predicted No","Predicted Yes"],
            yticklabels=["Actual No","Actual Yes"],
            ax=ax, linewidths=2, linecolor="#0B0D17",
            annot_kws={"size":16,"weight":"bold","color":"white"},
            cbar=False,
        )
        ax.set_title("Confusion Matrix", pad=12)
        ax.tick_params(colors="#8890B5", labelsize=8)
        fig.tight_layout(pad=1.0)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        c1,c2 = st.columns(2)
        c1.markdown(f"""<div class="cm-card" style="background:#0B1F14;border:1px solid #163328;">
            <div style="font-size:1.1rem;font-weight:800;color:#2DCE89;">{tn:,}</div>
            <div style="font-size:0.71rem;color:#4A5080;margin-top:2px;">True Negatives</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="cm-card" style="background:#111530;border:1px solid #1E2458;">
            <div style="font-size:1.1rem;font-weight:800;color:#4F8EF7;">{tp:,}</div>
            <div style="font-size:0.71rem;color:#4A5080;margin-top:2px;">True Positives</div></div>""", unsafe_allow_html=True)
        c1.markdown(f"""<div class="cm-card" style="background:#1A1408;border:1px solid #2E2210;">
            <div style="font-size:1.1rem;font-weight:800;color:#FFA94D;">{fp:,}</div>
            <div style="font-size:0.71rem;color:#4A5080;margin-top:2px;">False Positives</div></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="cm-card" style="background:#1F0A0A;border:1px solid #3A1010;">
            <div style="font-size:1.1rem;font-weight:800;color:#F75C5C;">{fn:,}</div>
            <div style="font-size:0.71rem;color:#4A5080;margin-top:2px;">False Negatives</div></div>""", unsafe_allow_html=True)

    with roc_col:
        roc = meta.get("roc_curve",{})
        if roc:
            fpr, tpr = roc["fpr"], roc["tpr"]
            fig, ax = mk_fig(5.5, 4.5)
            ax.fill_between(fpr, tpr, alpha=0.10, color="#4F8EF7")
            ax.plot(fpr, tpr, color="#4F8EF7", lw=2.5,
                    label=f"Gradient Boosting  AUC = {m['ROC-AUC']:.3f}")
            ax.plot([0,1],[0,1], color="#2A2D4A", lw=1.5, linestyle="--", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", pad=12)
            ax.legend(facecolor="#181B2E", edgecolor="#1C1F3A",
                      labelcolor="#C8CCEC", fontsize=8.5, loc="lower right")
            fig.tight_layout(pad=1.0)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:#181B2E;border:1px solid #1C1F3A;border-left:3px solid #9775FA;
                        border-radius:10px;padding:14px 16px;margin-top:10px;">
                <div style="font-size:1.3rem;font-weight:900;color:#9775FA;">AUC = {m['ROC-AUC']:.4f}</div>
                <div style="font-size:0.78rem;color:#6B7299;margin-top:4px;line-height:1.6;">
                    The model correctly ranks a random churner above a random non-churner
                    <strong style="color:#E4E7FF;">{m['ROC-AUC']:.1%}</strong> of the time.
                    Scores above 0.80 are considered strong for churn prediction.
                </div>
            </div>""", unsafe_allow_html=True)

    # Feature importance
    if fi:
        section("Top Feature Importances")
        fi_sorted = sorted(fi.items(), key=lambda x: x[1])
        n = len(fi_sorted)
        fig, ax = mk_fig(10, 5.5)
        colors_fi = ["#9775FA" if i < n-5 else "#4F8EF7" for i in range(n)]
        bars = ax.barh([x[0] for x in fi_sorted], [x[1] for x in fi_sorted],
                       color=colors_fi, edgecolor="none", height=0.6)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance — Top Predictors of Churn", pad=12)
        ax.set_xlim(0, max(x[1] for x in fi_sorted)*1.18)
        for bar in bars:
            ax.text(bar.get_width()+max(x[1] for x in fi_sorted)*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.4f}", va="center", color="#4A5080", fontsize=7.5)
        fig.tight_layout(pad=1.0)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        top3 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
        t1,t2,t3 = st.columns(3)
        for col, (feat, imp), rank, color in zip(
            [t1,t2,t3], top3,
            ["#1 Driver","#2 Driver","#3 Driver"],
            ["#4F8EF7","#9775FA","#FFA94D"],
        ):
            col.markdown(f"""
            <div style="background:#181B2E;border:1px solid #1C1F3A;border-top:3px solid {color};
                        border-radius:12px;padding:16px;text-align:center;margin-top:12px;">
                <div style="font-size:0.65rem;font-weight:700;color:#4A5080;text-transform:uppercase;
                            letter-spacing:0.1em;">{rank}</div>
                <div style="font-size:0.9rem;font-weight:700;color:#E4E7FF;margin:7px 0 4px;">{feat}</div>
                <div style="font-size:1.05rem;font-weight:800;color:{color};">{imp:.4f}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class="page-banner">
        <h1>About This Project</h1>
        <p>An end-to-end machine learning system built to solve a real telecom business problem.</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("""
        <div class="about-card">
            <h3>Business Problem</h3>
            <p>Customer churn — when a subscriber cancels their service — is one of the most costly challenges
            in the telecom industry. Acquiring a new customer costs <strong style="color:#E4E7FF;">5–7x more</strong>
            than retaining an existing one. This system predicts churn <em>before it happens</em>, enabling
            targeted retention campaigns that reduce revenue loss and improve customer lifetime value.</p>
        </div>
        <div class="about-card">
            <h3>Project Objective</h3>
            <p>Build a production-grade, end-to-end data science application demonstrating the complete ML
            lifecycle: data collection, cleaning, exploratory analysis, feature engineering, preprocessing,
            model training, evaluation, model selection, serialisation, and deployment as an interactive
            Streamlit web application suitable for real business use.</p>
        </div>
        <div class="about-card">
            <h3>How Predictions Work</h3>
            <p>Customer details pass through the same ColumnTransformer pipeline used during training
            (StandardScaler for numerical features + OneHotEncoder for categorical features). The processed
            input is fed into the <strong style="color:#E4E7FF;">Gradient Boosting</strong> classifier —
            selected after comparing five algorithms on ROC-AUC. The model outputs a 0–100% probability.
            Scores above 50% are classified as churn. Risk tiers (Low / Medium / High) trigger different
            retention action playbooks at the 40% and 70% thresholds.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        di = meta["dataset_info"]
        m  = meta["best_metrics"]

        st.markdown(f"""
        <div style="background:#131629;border:1px solid #1C1F3A;border-radius:14px;padding:22px;margin-bottom:14px;">
            <div style="font-size:0.62rem;font-weight:700;color:#4A5080;text-transform:uppercase;
                        letter-spacing:0.14em;margin-bottom:14px;">Dataset</div>
            <div class="srow"><span class="srow-lbl">Source</span><span class="srow-val">IBM Telco / Kaggle</span></div>
            <div class="srow"><span class="srow-lbl">Customers</span><span class="srow-val">{di['total_rows']:,}</span></div>
            <div class="srow"><span class="srow-lbl">Features</span><span class="srow-val">{di['features']} (incl. engineered)</span></div>
            <div class="srow"><span class="srow-lbl">Churn Rate</span><span class="srow-val">{di['churn_rate']:.1%}</span></div>
            <div class="srow"><span class="srow-lbl">Train / Test</span><span class="srow-val">80% / 20% stratified</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#131629;border:1px solid #1C1F3A;border-radius:14px;padding:22px;">
            <div style="font-size:0.62rem;font-weight:700;color:#4A5080;text-transform:uppercase;
                        letter-spacing:0.14em;margin-bottom:6px;">Champion Model</div>
            <div style="font-size:1rem;font-weight:800;color:#4F8EF7;margin-bottom:16px;">
                {meta['best_model_name']}
            </div>""", unsafe_allow_html=True)

        for metric, val in m.items():
            bw = int(val * 100)
            col_m = "#2DCE89" if val>=0.80 else "#4F8EF7" if val>=0.70 else "#FFA94D"
            st.markdown(f"""
            <div class="mbar-row">
                <div class="mbar-lbl">{metric}</div>
                <div class="mbar-track"><div class="mbar-fill" style="width:{bw}%;background:{col_m};"></div></div>
                <div class="mbar-val">{val:.1%}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
