import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR    = os.getenv("MODEL_DIR", "model_cache")
MODEL_FILE   = os.path.join(MODEL_DIR, "model.pkl")
PIPE_FILE    = os.path.join(MODEL_DIR, "pipeline.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "metrics.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]
OCEAN_OPTIONS = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]

# ── ML helpers ─────────────────────────────────────────────────────────────────
def build_pipeline() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), NUM_ATTRIBS),
        ("cat", Pipeline([
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_ATTRIBS),
    ])


def train_model(df: pd.DataFrame):
    df = df.copy()
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr, te in sss.split(df, df["income_cat"]):
        train_df = df.loc[tr].drop("income_cat", axis=1)
        test_df  = df.loc[te].drop("income_cat", axis=1)

    labels   = train_df["median_house_value"].copy()
    features = train_df.drop("median_house_value", axis=1)

    pipeline = build_pipeline()
    X_train  = pipeline.fit_transform(features)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, labels)

    cv_rmse = float(-cross_val_score(
        model, X_train, labels,
        scoring="neg_root_mean_squared_error", cv=5,
    ).mean())
    test_rmse = float(root_mean_squared_error(
        test_df["median_house_value"],
        model.predict(pipeline.transform(test_df.drop("median_house_value", axis=1))),
    ))

    metrics = dict(cv_rmse=cv_rmse, test_rmse=test_rmse,
                   train_size=len(train_df), test_size=len(test_df))

    joblib.dump(model,    MODEL_FILE,   compress=("lz4", 3))
    joblib.dump(pipeline, PIPE_FILE,    compress=("lz4", 3))
    joblib.dump(metrics,  METRICS_FILE, compress=("lz4", 3))
    return model, pipeline, metrics


@st.cache_resource(show_spinner=False)
def load_artefacts(mtime: str):
    model    = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPE_FILE)
    metrics  = joblib.load(METRICS_FILE) if os.path.exists(METRICS_FILE) else {}
    return model, pipeline, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Page config & global CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar polish ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #60a5fa !important; }
[data-testid="stSidebar"] hr { border-color: #1e293b !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #1e293b !important;
    border-color: #334155 !important;
    border-radius: 10px !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: #3b82f6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 0.55rem 1rem !important;
    width: 100%;
    transition: background 0.18s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb !important;
}
[data-testid="stSidebar"] .stButton > button.delete-btn {
    background: #7f1d1d !important;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e1b4b 55%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.25rem 2.5rem;
    margin-bottom: 1.75rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border: 1px solid #1e293b;
}
.hero-icon {
    font-size: 3rem;
    line-height: 1;
    flex-shrink: 0;
}
.hero h1 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #f8fafc;
    margin: 0 0 .3rem;
    line-height: 1.2;
}
.hero p {
    margin: 0;
    color: #64748b;
    font-size: 0.875rem;
}
.hero .pill {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border-radius: 20px;
    padding: .15rem .65rem;
    font-size: 0.75rem;
    margin-right: .35rem;
    margin-top: .4rem;
    border: 1px solid #334155;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.75rem;
}
.metric-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.metric-card.accent-blue::before  { background: #3b82f6; }
.metric-card.accent-violet::before { background: #8b5cf6; }
.metric-card.accent-teal::before  { background: #14b8a6; }
.metric-card.accent-amber::before { background: #f59e0b; }

.metric-card .m-label {
    font-size: .72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #94a3b8;
    margin-bottom: .4rem;
}
.metric-card .m-value {
    font-size: 1.55rem;
    font-weight: 600;
    color: #0f172a;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-card .m-sub {
    font-size: .75rem;
    color: #94a3b8;
    margin-top: .35rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #f8fafc;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    width: fit-content;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: .45rem 1.1rem !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.08) !important;
}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
    border-radius: 12px !important;
    border: 1.5px dashed #cbd5e1 !important;
    background: #f8fafc !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
}

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #86efac;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.pred-result .pred-label {
    font-size: .8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #16a34a;
    margin-bottom: .5rem;
}
.pred-result .pred-value {
    font-size: 3rem;
    font-weight: 700;
    color: #14532d;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.pred-result .pred-sub {
    font-size: .8rem;
    color: #4ade80;
    margin-top: .5rem;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: .6rem;
    margin-bottom: 1rem;
}
.section-header .sh-icon {
    width: 32px; height: 32px;
    background: #eff6ff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    border: 1px solid #bfdbfe;
    flex-shrink: 0;
}
.section-header h3 {
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0;
}
.section-header p {
    font-size: .8rem;
    color: #94a3b8;
    margin: 0;
}

/* ── Info callout ── */
.callout {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: .75rem 1rem;
    font-size: .825rem;
    color: #1e40af;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}

/* ── Form inputs ── */
.stNumberInput input, .stSelectbox select, .stSelectbox > div > div {
    border-radius: 8px !important;
    font-size: .875rem !important;
}

/* ── Primary action buttons (main area) ── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: .875rem !important;
    padding: .55rem 1.4rem !important;
    transition: all 0.18s !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    color: #475569 !important;
    border-radius: 8px !important;
    font-size: .85rem !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    background: #f1f5f9 !important;
    border-color: #cbd5e1 !important;
}

/* ── Badge ── */
.badge {
    display: inline-flex; align-items: center; gap: .3rem;
    border-radius: 20px; padding: .25rem .7rem;
    font-size: .75rem; font-weight: 600;
}
.badge-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
.badge-amber { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
.badge-red   { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }

/* ── No-model gate ── */
.gate-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    max-width: 480px;
    margin: 2rem auto;
}
.gate-card .gate-icon { font-size: 3rem; margin-bottom: 1rem; }
.gate-card h2 { font-size: 1.25rem; font-weight: 600; color: #0f172a; margin-bottom: .5rem; }
.gate-card p  { font-size: .875rem; color: #64748b; margin: 0; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Success / Error boxes ── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 10px !important;
}

/* ── Input group label ── */
.input-group-label {
    font-size: .7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #3b82f6;
    margin-bottom: .4rem;
    margin-top: 1.25rem;
}
            
.e1yxiy6j6{
    display: none;            
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: .75rem 0 1rem;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9; letter-spacing: -.01em;">
            🏡 Housing Predictor
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**Train model**")
    train_file = st.file_uploader(
        "housing.csv",
        type="csv",
        label_visibility="collapsed",
        help="Raw Kaggle California housing dataset.",
    )

    if train_file:
        if st.button("🚀 Start training", width="stretch"):
            with st.spinner("Training… ~30 s on full dataset"):
                t0  = time.time()
                df  = pd.read_csv(train_file)
                _m, _p, _met = train_model(df)
                elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.1f} s")
            st.cache_resource.clear()
    else:
        st.caption("Upload a CSV above to enable training.")

    st.divider()

    if os.path.exists(MODEL_FILE):
        size_mb = os.path.getsize(MODEL_FILE) / 1_048_576
        cls = "badge-green" if size_mb < 90 else "badge-amber"
        st.markdown(
            f'<span class="badge {cls}">● model.pkl — {size_mb:.1f} MiB</span>',
            unsafe_allow_html=True,
        )
        if st.button("🗑 Delete model cache", width="stretch"):
            for f in (MODEL_FILE, PIPE_FILE, METRICS_FILE):
                if os.path.exists(f): os.remove(f)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.markdown('<span class="badge badge-red">✕ No model cached</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Hero header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-icon">🏡</div>
  <div>
    <h1>California Housing Price Predictor</h1>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model gate
# ══════════════════════════════════════════════════════════════════════════════
model_ready = os.path.exists(MODEL_FILE) and os.path.exists(PIPE_FILE)

if not model_ready:
    st.markdown("""
    <div class="gate-card">
        <div class="gate-icon">🔧</div>
        <h2>No model found</h2>
        <p>Upload <code>housing.csv</code> in the sidebar and click <b>Start training</b> to get started.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

try:
    model, pipeline, metrics = load_artefacts(str(os.path.getmtime(MODEL_FILE)))
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Metric cards
# ══════════════════════════════════════════════════════════════════════════════
if metrics and isinstance(metrics, dict):
    cv   = metrics.get("cv_rmse", 0)
    te   = metrics.get("test_rmse", 0)
    tr_n = metrics.get("train_size", 0)
    te_n = metrics.get("test_size", 0)
    r2_approx = max(0, 1 - (te**2) / (250_000**2))  # rough indicator

    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card accent-blue">
        <div class="m-label">CV RMSE (5-fold)</div>
        <div class="m-value">${cv:,.0f}</div>
        <div class="m-sub">Cross-validated error</div>
      </div>
      <div class="metric-card accent-violet">
        <div class="m-label">Test RMSE</div>
        <div class="m-value">${te:,.0f}</div>
        <div class="m-sub">Held-out test set</div>
      </div>
      <div class="metric-card accent-teal">
        <div class="m-label">Training samples</div>
        <div class="m-value">{tr_n:,}</div>
        <div class="m-sub">80% of dataset</div>
      </div>
      <div class="metric-card accent-amber">
        <div class="m-label">Test samples</div>
        <div class="m-value">{te_n:,}</div>
        <div class="m-sub">20% stratified split</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_single, tab_bulk = st.tabs(["  🔢  Single prediction  ", "  📂  Bulk CSV inference  "])


# ── Tab 1: Single prediction ──────────────────────────────────────────────────
with tab_single:
    # Row 1: location
    st.markdown('<div class="input-group-label">📌 Location</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.4f",
                                    help="West-negative, e.g. -118.25 for Los Angeles")
    with c2:
        latitude  = st.number_input("Latitude",  value=37.88,  format="%.4f",
                                    help="North-positive, e.g. 34.05 for Los Angeles")
    with c3:
        ocean_proximity = st.selectbox("Ocean proximity", OCEAN_OPTIONS)

    # Row 2: property
    st.markdown('<div class="input-group-label">🏠 Property</div>', unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    with c4:
        housing_median_age = st.number_input("Median house age (yrs)", value=41, min_value=1, max_value=100)
    with c5:
        total_rooms    = st.number_input("Total rooms",    value=880, min_value=1)
    with c6:
        total_bedrooms = st.number_input("Total bedrooms", value=129, min_value=1)

    # Row 3: population
    st.markdown('<div class="input-group-label">👥 Demographics</div>', unsafe_allow_html=True)
    c7, c8, c9 = st.columns(3)
    with c7:
        population    = st.number_input("Population",             value=322,    min_value=1)
    with c8:
        households    = st.number_input("Households",             value=126,    min_value=1)
    with c9:
        median_income = st.number_input("Median income (scaled)", value=8.3252, format="%.4f", min_value=0.0,
                                        help="Scaled value, e.g. 3.0 ≈ $30,000 annual household income")

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        predict_clicked = st.button("✨ Predict price", width="stretch")

    if predict_clicked:
        row = pd.DataFrame([{
            "longitude": longitude, "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms, "total_bedrooms": total_bedrooms,
            "population": population, "households": households,
            "median_income": median_income, "ocean_proximity": ocean_proximity,
        }])
        with st.spinner("Running model…"):
            pred = model.predict(pipeline.transform(row))[0]

        st.markdown(f"""
        <div class="pred-result">
            <div class="pred-label">Estimated market value</div>
            <div class="pred-value">${pred:,.0f}</div>
            <div class="pred-sub">Random Forest · 100 estimators · stratified training</div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 2: Bulk CSV inference ─────────────────────────────────────────────────
with tab_bulk:
    st.markdown("""
    <div class="section-header">
      <div class="sh-icon">📂</div>
      <div>
        <p>Upload a CSV with property features and download predictions.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
    <b>Required columns:</b>
    longitude · latitude · housing_median_age · total_rooms · total_bedrooms ·
    population · households · median_income · ocean_proximity<br>
    <b>Optional:</b> include <code>median_house_value</code> to get residual analysis.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a CSV here or click to browse", type="csv",
                                key="infer_upload", label_visibility="collapsed")

    if uploaded:
        input_df = pd.read_csv(uploaded)

        missing = set(NUM_ATTRIBS + CAT_ATTRIBS) - set(input_df.columns)
        if missing:
            st.error(f"Missing columns: `{'`, `'.join(sorted(missing))}`")
            st.stop()

        with st.spinner(f"Running inference on {len(input_df):,} rows…"):
            features    = input_df[NUM_ATTRIBS + CAT_ATTRIBS]
            transformed = pipeline.transform(features)
            preds       = model.predict(transformed)

        out_df = input_df.copy()
        out_df["predicted_value"] = np.round(preds, 2)

        # Stats row
        has_truth = "median_house_value" in input_df.columns
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows processed",   f"{len(out_df):,}")
        s2.metric("Avg prediction",   f"${preds.mean():,.0f}")
        s3.metric("Min",              f"${preds.min():,.0f}")
        s4.metric("Max",              f"${preds.max():,.0f}")

        if has_truth:
            out_df["residual"] = out_df["median_house_value"] - out_df["predicted_value"]
            rmse = root_mean_squared_error(out_df["median_house_value"], out_df["predicted_value"])
            st.info(f"**Inference RMSE vs. ground truth:** ${rmse:,.0f}")

        st.dataframe(
            out_df.style.format({
                "predicted_value": "${:,.0f}",
                **({ "median_house_value": "${:,.0f}", "residual": "${:,.0f}" } if has_truth else {}),
            }),
            width='stretch',
            height=400,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                "⬇️ Download predictions CSV",
                data=out_df.to_csv(index=False).encode(),
                file_name="predictions.csv",
                mime="text/csv",
                width='stretch',
            )
    else:
        # Empty state
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; color: #94a3b8;">
            <div style="font-size: 2.5rem; margin-bottom: .75rem;">📄</div>
            <div style="font-size: .9rem; font-weight: 500; color: #64748b; margin-bottom: .35rem;">
                No file selected
            </div>
            <div style="font-size: .8rem;">
                Upload a CSV to run batch predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)