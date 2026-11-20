from __future__ import annotations

import re
import warnings
import base64
import json
import html
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Page setup and project paths
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Discover India | Smart Tourism",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = next((p for p in (BASE_DIR / "Dataset", BASE_DIR / "dataset") if p.exists()), BASE_DIR / "Dataset")
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
SEGMENTATION_DIR = OUTPUT_DIR / "segmentation"
RECOMMENDATION_DIR = OUTPUT_DIR / "recommendations"

HISTORY_PATH = OUTPUT_DIR / "overnight_tourism_decoded.csv"
ATTRACTIONS_PATH = DATA_DIR / "india_tourist_destinations.csv"
CATBOOST_PATH = MODEL_DIR / "best_model_catboost.cbm"
CATBOOST_META_PATH = MODEL_DIR / "catboost_metadata.pkl"
SEGMENTATION_MODEL_PATH = MODEL_DIR / "tourist_segmentation_model.pkl"
CLUSTER_PROFILES_PATH = SEGMENTATION_DIR / "cluster_profiles.csv"

# Features used by the final CatBoost experiment (the model was trained without district).
CATBOOST_FALLBACK_FEATURES = [
    "household_members",
    "average_member_age",
    "trip_duration_nights",
    "trip_type_label",
    "travel_mode_label",
    "accommodation_type_label",
    "leading_purpose_label",
    "trip_month_label",
    "main_destination_label",
    "destination_state_label",
    "state",
    "sector_label",
]

SEGMENT_NUMERIC = ["household_members", "average_member_age", "trip_duration_nights"]
SEGMENT_CATEGORICAL = [
    "trip_type_label",
    "travel_mode_label",
    "accommodation_type_label",
    "leading_purpose_label",
    "trip_month_label",
    "main_destination_label",
    "sector_label",
]

NUMERIC_SIMILARITY = ["household_members", "average_member_age", "trip_duration_nights"]
CATEGORICAL_SIMILARITY = [
    "leading_purpose_label",
    "travel_mode_label",
    "accommodation_type_label",
    "main_destination_label",
]

STATE_CODE_TO_NAME = {
    1: "Jammu and Kashmir", 2: "Himachal Pradesh", 3: "Punjab", 4: "Chandigarh",
    5: "Uttarakhand", 6: "Haryana", 7: "Delhi", 8: "Rajasthan", 9: "Uttar Pradesh",
    10: "Bihar", 11: "Sikkim", 12: "Arunachal Pradesh", 13: "Nagaland", 14: "Manipur",
    15: "Mizoram", 16: "Tripura", 17: "Meghalaya", 18: "Assam", 19: "West Bengal",
    20: "Jharkhand", 21: "Odisha", 22: "Chhattisgarh", 23: "Madhya Pradesh",
    24: "Gujarat", 25: "Dadra and Nagar Haveli and Daman and Diu", 27: "Maharashtra",
    28: "Andhra Pradesh", 29: "Karnataka", 30: "Goa", 31: "Lakshadweep", 32: "Kerala",
    33: "Tamil Nadu", 34: "Puducherry", 35: "Andaman and Nicobar Islands",
    36: "Telangana", 37: "Ladakh",
}
STATE_NAME_TO_CODE = {v: k for k, v in STATE_CODE_TO_NAME.items()}

REGIONS = {
    "North India": ["Jammu and Kashmir", "Himachal Pradesh", "Punjab", "Chandigarh", "Uttarakhand", "Haryana", "Delhi", "Rajasthan", "Uttar Pradesh", "Ladakh"],
    "South India": ["Andhra Pradesh", "Karnataka", "Goa", "Kerala", "Tamil Nadu", "Puducherry", "Telangana", "Lakshadweep"],
    "East India": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West India": ["Gujarat", "Maharashtra", "Dadra and Nagar Haveli and Daman and Diu"],
    "Central India": ["Madhya Pradesh", "Chhattisgarh"],
    "North East India": ["Sikkim", "Arunachal Pradesh", "Nagaland", "Manipur", "Mizoram", "Tripura", "Meghalaya", "Assam"],
    "Islands": ["Andaman and Nicobar Islands"],
}

PURPOSES = [
    "Holiday, leisure and recreation",
    "Pilgrimage and religious activities",
    "Health and medical",
    "Shopping",
]
TRIP_TYPES = ["Non-package", "Package"]
TRAVEL_MODES = [
    "Train", "Bus", "Air", "Own motorised transport", "Hired motorised transport",
    "On foot", "Ship/Boat", "Own non-motorised transport", "Hired non-motorised transport", "Others",
]
ACCOMMODATIONS = [
    "Hotel", "Guest house", "Dharamshala", "Rented house", "Homestay",
    "Other commercial accommodation", "Friends and relatives", "Other non-commercial accommodation",
]
DESTINATION_TYPES = [
    "Within the district",
    "Outside district but within state",
    "Outside state but within India",
    "International trip departure port in India",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
SECTORS = ["Rural", "Urban"]
CATEGORIES = ["Any", "Adventure", "Beach", "Heritage", "Nature", "Recreation", "Rural", "Spiritual", "Wildlife"]


# -----------------------------------------------------------------------------
# Editorial travel design inspired by the supplied reference
# -----------------------------------------------------------------------------
HERO_IMAGE = "https://media.assettype.com/outlooktraveller/2024-01/cca0398d-b197-46fa-83f7-4d9e207735d2/shutterstock_1033276267.jpeg?ar=40%3A21&auto=format%2Ccompress&enlarge=true&mode=crop&ogImage=true&overlay=false&overlay_position=bottom&overlay_width=100&w=1600"
TAJ_IMAGE = "https://media.cntraveler.com/photos/67c0b5a379fa3291633d3b26/master/w_1600%2Cc_limit/2179901360"
KERALA_IMAGE = "https://junglecamp.in/uploads/2026/02/north-vs-south-india-cultural-richness-and-tourism-highlights.webp"
FORT_IMAGE = "https://media.assettype.com/thequint%2F2015-12%2F1a5f1044-7d87-48fb-a163-45b2c4bfe661%2FSW_Balloon_Safari_2.JPG"
JOURNAL_IMAGE = "https://images.unsplash.com/photo-1455390582262-044cdead277a?auto=format&fit=crop&w=1200&q=85"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Bebas+Neue&family=Cormorant+Garamond:wght@500;600;700&display=swap');
    :root {
      --ink:#202b2a; --deep:#174f55; --teal:#1d6870; --coral:#df755c;
      --cream:#f5ecdf; --paper:#fff8ee; --sand:#e5d5c2; --muted:#777d76;
    }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:var(--cream); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(245,236,223,.94); }
    [data-testid="stToolbar"] { right:1rem; }
    [data-testid="stSidebar"] { display:none; }
    .block-container { max-width:1440px; padding:1.0rem 2rem 3rem; }
    h1,h2,h3 { color:var(--ink); }
    h1,h2,.serif,.section-title,.destination-name { font-family:'Cormorant Garamond',Georgia,serif !important; }
    p, label, .stMarkdown { color:var(--ink); }
    div[data-testid="stRadio"] > label { display:none; }
    div[data-testid="stRadio"] div[role="radiogroup"] {
      gap:1.2rem; flex-wrap:wrap; justify-content:flex-end; align-items:center;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
      border:0; border-radius:0; border-bottom:2px solid transparent;
      padding:.48rem .18rem; background:transparent; color:#244647 !important;
      font-size:.82rem; font-weight:700; letter-spacing:.025em; margin:0; opacity:1 !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
      background:rgba(223,117,92,.10); color:#bd604b !important; border-bottom:2px solid var(--coral);
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label p { color:inherit !important; }
    .brandline { display:flex; align-items:center; gap:.7rem; color:var(--deep); }
    .brandmark { width:42px; height:42px; border:1px solid var(--deep); border-radius:50%; display:grid; place-items:center; color:var(--deep); font-size:1.25rem; }
    .brandname { display:block; font-family:'Cormorant Garamond',Georgia,serif; font-size:1.16rem; letter-spacing:.08em; line-height:1; font-weight:700; }
    .brandtag { display:block; margin-top:.22rem; font-size:.55rem; letter-spacing:.18em; color:#78817b; text-transform:uppercase; }
    .hero-wrap {
      position:relative; overflow:hidden; min-height:465px; border-radius:2px;
      margin-top:.55rem; padding:3.2rem 3rem; color:#fff8ec;
      background-image:linear-gradient(90deg,rgba(15,50,54,.85) 0%,rgba(15,50,54,.43) 46%,rgba(15,50,54,.04) 100%),url('https://media.assettype.com/outlooktraveller/2024-01/cca0398d-b197-46fa-83f7-4d9e207735d2/shutterstock_1033276267.jpeg');
      background-position:center 43%; background-size:cover; display:flex; flex-direction:column; justify-content:center;
      box-shadow:0 18px 45px rgba(49,48,39,.12);
    }
    .hero-wrap:after { content:'INDIA  ·  TRAVEL COLLECTION'; position:absolute; right:2rem; bottom:1.5rem; color:#fff4df; font-size:.58rem; letter-spacing:.22em; writing-mode:vertical-rl; opacity:.88; }
    .hero-kicker { color:#f3b49a; text-transform:uppercase; letter-spacing:.24em; font-weight:700; font-size:.68rem; margin-bottom:1rem; }
    .hero-title { color:#fff8ec; font-family:'Bebas Neue',Impact,sans-serif; font-size:clamp(4.2rem,8vw,7.8rem); line-height:.82; letter-spacing:.015em; max-width:650px; margin:0; }
    .hero-title em { color:#f3a28a; font-family:'Cormorant Garamond',Georgia,serif; font-size:.58em; line-height:1; font-weight:600; }
    .hero-copy { color:#f6efe4; font-size:.94rem; line-height:1.75; max-width:440px; margin-top:1.25rem; }
    .hero-side-note { position:absolute; right:3rem; top:3.4rem; color:#fff7ed; text-transform:uppercase; letter-spacing:.18em; font-size:.61rem; line-height:2.2; text-align:left; }
    .section-kicker { color:var(--coral); text-transform:uppercase; letter-spacing:.22em; font-size:.65rem; font-weight:700; margin-bottom:.3rem; }
    .section-title { font-family:'Bebas Neue',Impact,sans-serif !important; font-size:2.55rem; line-height:.97; letter-spacing:.015em; color:var(--ink); margin:0 0 .45rem 0; }
    .section-copy { color:#687671; margin:0 0 1.05rem 0; font-size:.9rem; line-height:1.65; }
    .soft-card { background:var(--paper); border:1px solid #eadbca; border-radius:3px; padding:1.15rem 1.2rem; height:100%; box-shadow:0 9px 22px rgba(70,55,39,.04); }
    .destination-card { background:var(--paper); border:1px solid #e6d8c7; border-radius:3px; overflow:hidden; height:100%; box-shadow:0 9px 24px rgba(70,55,39,.08); }
    .destination-photo { width:100%; height:205px; object-fit:cover; display:block; background:#e8ddcf; }
    .destination-body { padding:1rem 1.05rem 1.1rem; }
    .photo-credit { color:#899089; font-size:.60rem; letter-spacing:.04em; margin-top:.55rem; line-height:1.4; }
    .photo-placeholder { height:205px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.45rem; color:#31595a; background:linear-gradient(135deg,#d9e3d9,#e8d4bd); text-align:center; padding:1rem; }
    .photo-placeholder .symbol { font-size:2rem; }
    .photo-placeholder .note { font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; }
    .card-actions { display:flex; gap:.5rem; align-items:center; flex-wrap:wrap; margin-top:.7rem; }
    .explorer-panel { background:#eee1d0; border:1px solid #dfcfbc; padding:1.2rem; }
    .detail-note { color:#66746e; font-size:.84rem; line-height:1.65; }
    .destination-icon { width:38px; height:38px; display:grid; place-items:center; border-radius:50%; background:#e3ece5; font-size:1.15rem; margin-bottom:.6rem; }
    .destination-name { color:var(--ink); font-size:1.55rem; font-weight:700; line-height:1.05; }
    .destination-meta { color:#79837c; font-size:.74rem; margin:.35rem 0 .7rem; text-transform:uppercase; letter-spacing:.08em; }
    .score-pill { display:inline-block; border-radius:0; background:#f5dfd3; color:#94523e; padding:.26rem .55rem; font-size:.65rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .metric-card { background:var(--paper); border:1px solid #e8d9c8; border-radius:3px; padding:1rem 1.15rem; height:100%; }
    .metric-label { color:#78837d; font-size:.66rem; text-transform:uppercase; letter-spacing:.13em; }
    .metric-value { color:var(--deep); font-family:'Cormorant Garamond',Georgia,serif; font-size:2.05rem; font-weight:700; margin-top:.18rem; }
    .metric-note { color:#8b938c; font-size:.72rem; }
    .journal-panel { display:grid; grid-template-columns:1.1fr 1fr; min-height:270px; background:#e9d4bd; }
    .journal-panel img { width:100%; height:100%; min-height:270px; object-fit:cover; display:block; }
    .journal-copy { padding:2.1rem 2.2rem; display:flex; flex-direction:column; justify-content:center; }
    .journal-title { font-family:'Bebas Neue',Impact,sans-serif; color:#242b29; font-size:2.8rem; line-height:.92; }
    .route-panel { background:var(--deep); color:#fff4e7; padding:2.2rem 2.3rem; position:relative; overflow:hidden; }
    .route-kicker { color:#f1a38a; font-size:.65rem; font-weight:700; letter-spacing:.22em; text-transform:uppercase; }
    .route-title { color:#fff4e7; font-family:'Bebas Neue',Impact,sans-serif; font-size:2.7rem; line-height:.95; margin:.35rem 0 .7rem; }
    .route-copy { color:#e2e4db; font-size:.84rem; max-width:430px; line-height:1.6; }
    .region-chip { display:inline-block; border:1px solid rgba(255,244,231,.45); color:#fff4e7; padding:.42rem .75rem; margin:.25rem .25rem .15rem 0; font-size:.67rem; letter-spacing:.1em; text-transform:uppercase; }
    .quote-strip { background:#e88b73; color:#243b3b; padding:1.1rem 1.5rem; display:flex; align-items:center; gap:1.2rem; }
    .quote-mark { font-family:'Cormorant Garamond',Georgia,serif; font-size:3.8rem; line-height:.8; color:#fff4e7; }
    .quote-text { font-family:'Cormorant Garamond',Georgia,serif; font-size:1.3rem; font-weight:700; line-height:1.1; }
    .footer { background:#202624; color:#d5d6cd; margin:2.6rem calc(-1 * max(2rem, (100vw - 1440px)/2)) -3rem; padding:2.2rem max(2rem, calc((100vw - 1300px)/2)) 1.4rem; font-size:.76rem; }
    .footer strong { color:#fff1df; font-family:'Cormorant Garamond',Georgia,serif; font-size:1.3rem; letter-spacing:.1em; }
    .stButton > button, .stFormSubmitButton > button { background:var(--coral); color:#fff8ee; border:1px solid var(--coral); border-radius:2px; padding:.68rem 1.25rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; font-size:.72rem; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { background:#c96350; color:white; border-color:#c96350; }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div { border-radius:2px; background:#fffaf3; border-color:#e2d4c4; }
    .stSlider { padding-top:.3rem; }
    div[data-testid="stMetric"] { background:var(--paper); border:1px solid #e8d9c8; border-radius:3px; padding:.8rem 1rem; }
    div[data-testid="stMetricLabel"] { color:#718080; }
    div[data-testid="stMetricValue"] { color:var(--deep); }
    .stAlert { border-radius:3px; }
    [data-testid="stDataFrame"] { background:var(--paper); }
    /* --- Header: high contrast, sticky, text-only navigation --- */
    [data-testid="stHeader"] { background:rgba(245,236,223,.98) !important; }
    [data-testid="stToolbar"], [data-testid="stAppDeployButton"], #MainMenu, footer { display:none !important; }
    .brandline, .brandline * { color:#174f55 !important; opacity:1 !important; }
    .brandmark { border-color:#174f55 !important; color:#174f55 !important; background:rgba(255,250,243,.7); }
    .brandname { color:#174f55 !important; font-weight:700 !important; text-shadow:0 1px 0 rgba(255,255,255,.35); }
    .brandtag { color:#526d68 !important; font-weight:700; }
    div[data-testid="stHorizontalBlock"]:has(.brandline) {
      position:sticky; top:0; z-index:999; align-items:center;
      background:rgba(255,248,238,.97); backdrop-filter:blur(14px);
      border-bottom:1px solid #dfd0bd; border-radius:0 0 3px 3px;
      padding:.62rem 1rem; margin:-1rem -2rem 1.2rem;
      width:calc(100% + 4rem); box-shadow:0 5px 20px rgba(62,48,34,.045);
    }
    div[data-testid="stRadio"] > label { display:none !important; }
    div[data-testid="stRadio"] [role="radiogroup"] {
      display:flex !important; gap:clamp(.55rem,1.5vw,1.5rem) !important;
      flex-wrap:wrap; justify-content:flex-end; align-items:center;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label {
      position:relative; display:inline-flex !important; align-items:center;
      padding:.52rem .12rem .58rem !important; margin:0 !important;
      border:0 !important; border-radius:0 !important; background:transparent !important;
      color:#244647 !important; opacity:1 !important; cursor:pointer !important;
      font-size:.82rem !important; font-weight:700 !important; letter-spacing:.025em;
      transition:color .22s ease, transform .22s ease;
    }
    /* Hide the radio dot: the label itself stays a real clickable navigation item. */
    div[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child,
    div[data-testid="stRadio"] [role="radiogroup"] > label [data-baseweb="radio"] {
      display:none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label p,
    div[data-testid="stRadio"] [role="radiogroup"] > label span {
      color:inherit !important; opacity:1 !important; font-weight:inherit !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:hover {
      color:#bd604b !important; transform:translateY(-2px);
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked),
    div[data-testid="stRadio"] [role="radiogroup"] > label:has([aria-checked="true"]) {
      color:#b95745 !important; background:transparent !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:after {
      content:""; position:absolute; left:0; right:0; bottom:0; height:2px;
      background:#df755c; transform:scaleX(0); transform-origin:left;
      transition:transform .24s ease;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > label:hover:after,
    div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked):after,
    div[data-testid="stRadio"] [role="radiogroup"] > label:has([aria-checked="true"]):after { transform:scaleX(1); }

    /* --- Motion: subtle editorial entrance, living hero, and interactive cards --- */
    html { scroll-behavior:smooth; }
    @keyframes yatraFadeUp { from {opacity:0; transform:translateY(20px)} to {opacity:1; transform:translateY(0)} }
    @keyframes yatraHeroReveal { from {opacity:0; transform:translateY(14px) scale(.992)} to {opacity:1; transform:translateY(0) scale(1)} }
    @keyframes yatraPhotoDrift { from {background-position:48% 48%} to {background-position:53% 43%} }
    @keyframes yatraFloat { 0%,100% {transform:translate3d(0,0,0) rotate(-5deg)} 50% {transform:translate3d(0,-15px,0) rotate(5deg)} }
    @keyframes yatraGlow { 0%,100% {opacity:.30; transform:scale(.92)} 50% {opacity:.72; transform:scale(1.08)} }
    @keyframes yatraUnderline { from {transform:scaleX(0)} to {transform:scaleX(1)} }
    .hero-wrap { animation:yatraHeroReveal .9s cubic-bezier(.2,.7,.2,1) both, yatraPhotoDrift 24s ease-in-out 1s infinite alternate; }
    .hero-wrap:before {
      content:""; position:absolute; width:132px; height:132px; right:18%; top:22%;
      border:1px solid rgba(255,239,213,.8); border-radius:50%;
      box-shadow:0 0 0 12px rgba(255,239,213,.07), 0 0 45px rgba(255,239,213,.18);
      pointer-events:none; animation:yatraFloat 7s ease-in-out infinite, yatraGlow 5s ease-in-out infinite;
    }
    .hero-kicker, .hero-title, .hero-copy { position:relative; z-index:1; animation:yatraFadeUp .8s ease both; }
    .hero-title { animation-delay:.12s; }
    .hero-copy { animation-delay:.24s; }
    .section-kicker { animation:yatraFadeUp .55s ease both; }
    .destination-card {
      transition:transform .32s cubic-bezier(.2,.7,.2,1), box-shadow .32s ease, border-color .32s ease;
      animation:yatraFadeUp .65s ease both;
    }
    .destination-card:hover {
      transform:translateY(-7px); border-color:#d99a82;
      box-shadow:0 18px 38px rgba(70,55,39,.15);
    }
    .destination-photo { transition:transform .55s cubic-bezier(.2,.7,.2,1), filter .4s ease; }
    .destination-card:hover .destination-photo { transform:scale(1.045); filter:saturate(1.08); }
    .soft-card, .metric-card { transition:transform .28s ease, box-shadow .28s ease; }
    .soft-card:hover, .metric-card:hover { transform:translateY(-4px); box-shadow:0 14px 28px rgba(70,55,39,.11); }
    .journal-panel, .route-panel, .quote-strip { animation:yatraFadeUp .8s ease both; }
    .region-chip { transition:background .22s ease, color .22s ease, transform .22s ease; }
    .region-chip:hover { background:#fff4e7; color:#174f55; transform:translateY(-2px); }
    a { color:#b95745 !important; text-decoration:none !important; transition:color .2s ease; }
    a:hover { color:#174f55 !important; text-decoration:underline !important; }
    .stButton > button { transition:transform .2s ease, box-shadow .2s ease, background .2s ease; }
    .stButton > button:hover { transform:translateY(-2px); box-shadow:0 8px 18px rgba(185,87,69,.18); }
    .stTextInput input:focus, .stNumberInput input:focus { box-shadow:0 0 0 2px rgba(223,117,92,.18) !important; border-color:#df755c !important; }
    @media (prefers-reduced-motion: reduce) {
      *, *:before, *:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; }
    }
    @media (max-width:800px) {
      .block-container { padding-left:1rem; padding-right:1rem; }
      .hero-wrap { padding:2rem 1.3rem; min-height:390px; }
      .hero-side-note { display:none; }
      .hero-title { font-size:4.4rem; }
      .journal-panel { grid-template-columns:1fr; }
      .journal-panel img { min-height:190px; max-height:220px; }
      .footer { margin-left:-1rem; margin-right:-1rem; padding-left:1.2rem; padding-right:1.2rem; }
      div[data-testid="stRadio"] [role="radiogroup"] { justify-content:flex-start !important; gap:.65rem !important; }
      div[data-testid="stHorizontalBlock"]:has(.brandline) { margin-left:-1rem; margin-right:-1rem; width:calc(100% + 2rem); padding:.65rem .8rem; position:relative; }
      .brandname { font-size:1rem !important; }
      div[data-testid="stRadio"] [role="radiogroup"] > label { font-size:.73rem !important; }
      .hero-wrap:before { width:80px; height:80px; right:9%; top:18%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utilities and cached data/model loading
# -----------------------------------------------------------------------------
def inr(value: float | int) -> str:
    try:
        number = float(value)
        return f"₹{number:,.0f}" if np.isfinite(number) else "Not available"
    except (TypeError, ValueError):
        return "Not available"


def clean_options(values: pd.Series | list[str], fallback: list[str]) -> list[str]:
    if isinstance(values, pd.Series):
        vals = values.dropna().astype(str).str.strip().unique().tolist()
    else:
        vals = [str(v).strip() for v in values if pd.notna(v)]
    vals = [v for v in vals if v and v.lower() not in {"nan", "none"}]
    return sorted(set(vals)) if vals else fallback


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY_PATH, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_attractions() -> pd.DataFrame:
    if not ATTRACTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ATTRACTIONS_PATH, low_memory=False)


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    if not CLUSTER_PROFILES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLUSTER_PROFILES_PATH, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_catboost_model():
    if not CATBOOST_PATH.exists():
        return None
    try:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(CATBOOST_PATH))
        return model
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def load_catboost_metadata():
    if not CATBOOST_META_PATH.exists():
        return {}
    try:
        obj = joblib.load(CATBOOST_META_PATH)
        return obj if isinstance(obj, dict) else {"metadata_object": obj}
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_segmentation_artifact():
    if not SEGMENTATION_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(SEGMENTATION_MODEL_PATH)
    except Exception as exc:
        return exc


@st.cache_resource(show_spinner=False)
def get_similarity_engine():
    """Build the historical-neighbour engine lazily, only when recommendations are requested."""
    history = load_history()
    required = NUMERIC_SIMILARITY + CATEGORICAL_SIMILARITY
    if history.empty or any(c not in history.columns for c in required):
        return None, None, "Required historical features are missing."

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import NearestNeighbors
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)])
        preprocessor = ColumnTransformer(
            [("num", numeric_pipe, NUMERIC_SIMILARITY), ("cat", categorical_pipe, CATEGORICAL_SIMILARITY)],
            remainder="drop",
        )
        features = history[required].copy()
        matrix = preprocessor.fit_transform(features)
        k = min(2000, len(history))
        neighbours = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        neighbours.fit(matrix)
        return preprocessor, neighbours, None
    except Exception as exc:
        return None, None, str(exc)


def option_from_history(df: pd.DataFrame, column: str, defaults: list[str]) -> list[str]:
    if column not in df.columns:
        return defaults
    return clean_options(df[column], defaults)


def residence_state_options(df: pd.DataFrame) -> list[str]:
    """Show readable state names while retaining the survey's original state code for prediction."""
    if "state" not in df.columns:
        return sorted(STATE_NAME_TO_CODE)
    raw = pd.to_numeric(df["state"], errors="coerce")
    if raw.notna().mean() > 0.8:
        codes = sorted({int(x) for x in raw.dropna().unique() if int(x) in STATE_CODE_TO_NAME})
        names = [STATE_CODE_TO_NAME[x] for x in codes]
        return names or sorted(STATE_NAME_TO_CODE)
    return clean_options(df["state"], sorted(STATE_NAME_TO_CODE))


def residence_state_model_value(display_name: str, df: pd.DataFrame):
    if display_name in STATE_NAME_TO_CODE:
        return STATE_NAME_TO_CODE[display_name]
    # Some copies of the dataset may store the state as a readable string or a numeric code.
    try:
        return int(float(display_name))
    except (TypeError, ValueError):
        return display_name


def parse_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [x.strip().casefold() for x in re.split(r"[;,|]", str(value)) if x.strip()]


def category_icon(category: str) -> str:
    icons = {
        "Adventure": "⛰️", "Beach": "🌊", "Heritage": "🏛️", "Nature": "🌿",
        "Recreation": "🎡", "Rural": "🌾", "Spiritual": "🪷", "Wildlife": "🐘",
    }
    return icons.get(str(category), "✦")


# Original-attraction image overrides for destinations visible in the supplied UI screenshots.
# If these do not load, the app searches Wikimedia Commons for the attraction, then for the state.
ATTRACTION_PHOTO_OVERRIDES = {
    "taj mahal": (
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/4e/Taj_Mahal%2C_Agra%2C_UP%2C_India.jpg/1280px-Taj_Mahal%2C_Agra%2C_UP%2C_India.jpg",
        "Yann Forget · Wikimedia Commons · CC BY-SA",
        "https://commons.wikimedia.org/wiki/File:Taj_Mahal,_Agra,_UP,_India.jpg",
    ),
    "alappuzha backwaters": (
        "https://s7ap1.scene7.com/is/image/incredibleindia/alappuzha-kerala-attr-about?qlt=82&ts=1751441626047",
        "Incredible India",
        "https://www.incredibleindia.gov.in/en/kerala/alappuzha/backwaters-of-alappuzha",
    ),
    "backwaters of alappuzha": (
        "https://s7ap1.scene7.com/is/image/incredibleindia/alappuzha-kerala-attr-about?qlt=82&ts=1751441626047",
        "Incredible India",
        "https://www.incredibleindia.gov.in/en/kerala/alappuzha/backwaters-of-alappuzha",
    ),
    "kerala backwaters": (
        "https://s7ap1.scene7.com/is/image/incredibleindia/alappuzha-kerala-attr-about?qlt=82&ts=1751441626047",
        "Incredible India",
        "https://www.incredibleindia.gov.in/en/kerala/alappuzha/backwaters-of-alappuzha",
    ),
    "loktak lake": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a3/Loktak_Lake%2C_Manipur.jpg",
        "Nandkishordudhe · Wikimedia Commons · CC BY-SA",
        "https://commons.wikimedia.org/wiki/File:Loktak_Lake,_Manipur.jpg",
    ),
}

# State-level fallback photographs. Other states are resolved through the Commons search below.
STATE_PHOTO_OVERRIDES = {
    "kerala": (
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/d/d3/Alappuzha_Backwaters.jpg/1280px-Alappuzha_Backwaters.jpg",
        "Anjali Mathew · Wikimedia Commons · CC BY-SA",
        "https://commons.wikimedia.org/wiki/File:Alappuzha_Backwaters.jpg",
    ),
    "manipur": (
        "https://upload.wikimedia.org/wikipedia/commons/a/a3/Loktak_Lake%2C_Manipur.jpg",
        "Nandkishordudhe · Wikimedia Commons · CC BY-SA",
        "https://commons.wikimedia.org/wiki/File:Loktak_Lake,_Manipur.jpg",
    ),
}


def _photo_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def download_photo_bytes(url: str) -> bytes | None:
    """Download and cache a remote image on the Streamlit server, not in the browser."""
    if not url or not str(url).startswith("https://"):
        return None
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "YatraSmartTourismAcademicProject/1.0",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            },
        )
        with urlopen(req, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            data = response.read(12 * 1024 * 1024 + 1)
        if not content_type.startswith("image/") or len(data) > 12 * 1024 * 1024:
            return None
        # Quick signature check prevents an HTML error page being rendered as an image.
        if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG") or data.startswith(b"RIFF") or data.startswith(b"GIF8") or data.startswith(b"\x00\x00\x01\x00")):
            return None
        return data
    except Exception:
        return None


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def commons_photo_candidates(query: str) -> list[dict[str, str]]:
    """Find real photographs on Wikimedia Commons for the attraction or its state."""
    if not query.strip():
        return []
    params = urlencode({
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": 5,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1200,
        "format": "json",
    })
    request_url = "https://commons.wikimedia.org/w/api.php?" + params
    try:
        req = Request(request_url, headers={"User-Agent": "YatraSmartTourismAcademicProject/1.0"})
        with urlopen(req, timeout=9) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        pages = payload.get("query", {}).get("pages", {})
        found: list[dict[str, str]] = []
        for page in pages.values():
            title = str(page.get("title", ""))
            title_lower = title.casefold()
            if any(word in title_lower for word in ("logo", "flag", "coat of arms", "map of", "location map", "diagram", ".svg", ".pdf")):
                continue
            imageinfo = (page.get("imageinfo") or [{}])[0]
            image_url = imageinfo.get("thumburl") or imageinfo.get("url")
            if not image_url:
                continue
            metadata = imageinfo.get("extmetadata") or {}
            artist = metadata.get("Artist", {}).get("value", "")
            license_name = metadata.get("LicenseShortName", {}).get("value", "")
            description_url = imageinfo.get("descriptionurl", "")
            credit_parts = [re.sub(r"<[^>]+>", " ", str(x)).strip() for x in (artist, license_name) if x]
            found.append({
                "image_url": str(image_url),
                "description_url": str(description_url),
                "title": title,
                "credit": " · ".join(x for x in credit_parts if x) or "Wikimedia Commons",
            })
        return found
    except Exception:
        return []


def resolve_place_photo(attraction_name: str, destination_name: str, state: str) -> tuple[bytes | None, str, str, bool]:
    """Prefer an attraction photo; use a state photo when a specific photo is unavailable."""
    attraction_key = _photo_key(attraction_name)
    destination_key = _photo_key(destination_name)
    state_key = _photo_key(state)

    # 1. Carefully selected original-place photos for common catalogue entries.
    override = ATTRACTION_PHOTO_OVERRIDES.get(attraction_key)
    if not override:
        for key, value in ATTRACTION_PHOTO_OVERRIDES.items():
            if key and (key in attraction_key or key in destination_key):
                override = value
                break
    if override:
        data = download_photo_bytes(override[0])
        if data:
            return data, override[1], override[2], False

    # 2. Search for the exact attraction / destination name in Wikimedia Commons.
    exact_queries = []
    if attraction_name:
        exact_queries.append(f'"{attraction_name}" {destination_name} {state} India')
    elif destination_name:
        exact_queries.append(f'"{destination_name}" {state} India tourism')
    for query in exact_queries:
        for candidate in commons_photo_candidates(query):
            data = download_photo_bytes(candidate["image_url"])
            if data:
                return data, candidate["credit"], candidate["description_url"], False

    # 3. Use a state-specific fallback image before doing a broader state photo search.
    state_override = STATE_PHOTO_OVERRIDES.get(state_key)
    if state_override:
        data = download_photo_bytes(state_override[0])
        if data:
            return data, state_override[1], state_override[2], True

    if state:
        for candidate in commons_photo_candidates(f'"{state}" India landscape tourism photograph'):
            data = download_photo_bytes(candidate["image_url"])
            if data:
                return data, candidate["credit"], candidate["description_url"], True

    # No broken <img> element: the card renderer will show a branded placeholder instead.
    return None, "Photo unavailable", "", True


def image_data_mime(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def render_destination_card(
    *,
    attraction_name: str,
    destination_name: str,
    state: str,
    category: str,
    description: str,
    score: int | None = None,
    planning_notes: str = "",
    data_status: str = "",
    catalogue_source: str = "",
    catalogue_url: str = "",
    key: str = "place",
    allow_save: bool = True,
) -> None:
    photo_bytes, photo_credit, photo_source_url, is_state_fallback = resolve_place_photo(
        attraction_name, destination_name, state
    )
    safe_name = html.escape(str(attraction_name or destination_name))
    safe_destination = html.escape(str(destination_name or "India"))
    safe_state = html.escape(str(state or "India"))
    safe_category = html.escape(str(category or "Destination").upper())
    safe_description = html.escape(str(description or "A place to add to your travel list."))

    if photo_bytes:
        encoded = base64.b64encode(photo_bytes).decode("ascii")
        mime_type = image_data_mime(photo_bytes)
        image_html = f"<img class='destination-photo' src='data:{mime_type};base64,{encoded}' alt='{safe_name}' loading='lazy'>"
        image_kind = f"STATE PHOTO · {safe_state}" if is_state_fallback else "ORIGINAL ATTRACTION PHOTO"
    else:
        image_html = (
            "<div class='photo-placeholder'><div class='symbol'>✧</div>"
            f"<div class='note'>{safe_state} · INDIA</div>"
            "<div style='font-family:Cormorant Garamond,serif;font-size:1.1rem'>A place waiting to be discovered</div></div>"
        )
        image_kind = "PHOTO NOT AVAILABLE"

    score_html = f"<span class='score-pill'>MATCH SCORE {int(score)}</span>" if score is not None else "<span class='score-pill'>DISCOVER THE REGION</span>"
    credit_html = f"<div class='photo-credit'>{html.escape(image_kind)} · {html.escape(photo_credit)}</div>"
    st.markdown(
        f"<div class='destination-card'>{image_html}<div class='destination-body'>"
        f"<div class='section-kicker'>{safe_category} · INDIA</div>"
        f"<div class='destination-name'>{safe_name}</div>"
        f"<div class='destination-meta'>{safe_destination} · {safe_state}</div>"
        f"<p style='font-size:.83rem;color:#647675;min-height:3.4rem;line-height:1.55'>{safe_description}</p>"
        f"{score_html}{credit_html}</div></div>",
        unsafe_allow_html=True,
    )

    detail_text = str(planning_notes or "").strip()
    if data_status and str(data_status).strip().casefold() not in {"nan", "none", ""}:
        detail_text = (detail_text + "\n\nCatalogue status: " + str(data_status)).strip()
    with st.expander("Place details & travel notes", expanded=False):
        st.write(description or "No additional description is available in the project catalogue.")
        if detail_text:
            st.markdown("**Planning notes**")
            st.write(detail_text)
        if catalogue_source:
            st.caption(f"Catalogue source: {catalogue_source}")
        links = []
        if catalogue_url and str(catalogue_url).startswith(("https://", "http://")):
            links.append(f"[Destination information ↗]({catalogue_url})")
        if photo_source_url and str(photo_source_url).startswith(("https://", "http://")):
            links.append(f"[Photo source / licence ↗]({photo_source_url})")
        if links:
            st.markdown(" · ".join(links))
        elif not detail_text and not catalogue_source and not catalogue_url:
            st.caption("The project catalogue does not contain further travel notes for this entry.")

    if allow_save:
        saved = st.session_state.setdefault("saved_places", {})
        place_id = "|".join([str(attraction_name), str(destination_name), str(state)])
        button_label = "✓ SAVED TO SHORTLIST" if place_id in saved else "＋ SAVE TO SHORTLIST"
        if st.button(button_label, key=f"save_{key}_{re.sub(r'[^a-zA-Z0-9]+', '_', place_id)[:55]}", use_container_width=True):
            saved = st.session_state.setdefault("saved_places", {})
            if place_id in saved:
                del saved[place_id]
            else:
                saved[place_id] = {
                    "Attraction": str(attraction_name),
                    "Destination": str(destination_name),
                    "State / UT": str(state),
                    "Category": str(category),
                    "Description": str(description),
                }
            st.rerun()


def render_saved_shortlist() -> None:
    saved = st.session_state.get("saved_places", {})
    if not saved:
        return
    st.write("")
    section_heading("YOUR PERSONAL COLLECTION", "Saved places", f"You have saved {len(saved)} place(s) to revisit.")
    saved_df = pd.DataFrame(list(saved.values()))
    st.dataframe(saved_df, use_container_width=True, hide_index=True)
    st.download_button(
        "DOWNLOAD MY SHORTLIST ↓",
        data=saved_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="yatra_saved_places.csv",
        mime="text/csv",
        use_container_width=True,
    )


def predict_catboost(user_values: dict[str, Any]) -> tuple[float | None, str | None]:
    model = load_catboost_model()
    if model is None:
        return None, f"Model file not found: {CATBOOST_PATH}"
    if isinstance(model, Exception):
        return None, f"Could not load CatBoost model: {model}"

    metadata = load_catboost_metadata()
    feature_names = getattr(model, "feature_names_", None)
    if feature_names is None or len(feature_names) == 0:
        feature_names = metadata.get("feature_columns") or metadata.get("features") or CATBOOST_FALLBACK_FEATURES
    if isinstance(feature_names, str):
        feature_names = CATBOOST_FALLBACK_FEATURES

    row = {name: user_values.get(name, "Not applicable" if name == "destination_state_label" else 0) for name in feature_names}
    frame = pd.DataFrame([row], columns=list(feature_names))
    numeric_names = {"household_members", "average_member_age", "trip_duration_nights"}
    for col in frame.columns:
        if col in numeric_names:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        else:
            frame[col] = frame[col].fillna("Not applicable").astype(str)
    try:
        prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
        return max(0.0, prediction), None
    except Exception as exc:
        return None, f"Prediction failed: {exc}"


def run_segmentation_prediction(user_values: dict[str, Any]) -> tuple[int | None, str | None]:
    artifact = load_segmentation_artifact()
    if artifact is None:
        return None, f"Segmentation model not found: {SEGMENTATION_MODEL_PATH}"
    if isinstance(artifact, Exception):
        return None, f"Could not load segmentation model: {artifact}"

    row = pd.DataFrame([{c: user_values.get(c) for c in SEGMENT_NUMERIC + SEGMENT_CATEGORICAL}])
    try:
        if isinstance(artifact, dict):
            model = artifact.get("model") or artifact.get("kmeans") or artifact.get("cluster_model")
            preprocessor = artifact.get("preprocessor") or artifact.get("transformer")
            if model is None:
                # Support artifacts that store a complete sklearn pipeline under another key.
                model = artifact.get("pipeline") or artifact.get("segmentation_pipeline")
                preprocessor = None
            if model is None:
                return None, "The saved segmentation artifact does not contain a recognizable model."
            transformed = preprocessor.transform(row) if preprocessor is not None else row
            cluster = int(np.asarray(model.predict(transformed)).reshape(-1)[0])
            return cluster, None
        if isinstance(artifact, (tuple, list)) and len(artifact) >= 2:
            preprocessor, model = artifact[0], artifact[1]
            return int(np.asarray(model.predict(preprocessor.transform(row))).reshape(-1)[0]), None
        # A standalone Pipeline or estimator.
        return int(np.asarray(artifact.predict(row)).reshape(-1)[0]), None
    except Exception as exc:
        return None, f"Could not generate segment: {exc}"


def recommendation_results(preferences: dict[str, Any], attractions: pd.DataFrame, history: pd.DataFrame):
    if attractions.empty:
        return pd.DataFrame(), pd.DataFrame(), "Destination catalogue is missing."

    # Filter and rank the curated attraction catalogue. The catalogue is not a live booking feed.
    required = {"destination_name", "attraction_name", "state_or_ut", "category", "suitable_purpose"}
    if not required.issubset(attractions.columns):
        return pd.DataFrame(), pd.DataFrame(), "The attraction catalogue is missing one or more expected columns."

    pool = attractions.copy()
    purpose = str(preferences["purpose"]).casefold()
    pool["_purpose_match"] = pool["suitable_purpose"].apply(lambda x: purpose in parse_list(x) or purpose in str(x).casefold())
    pool = pool[pool["_purpose_match"]].copy()
    if pool.empty:
        pool = attractions.copy()
        pool["_purpose_match"] = False

    state = preferences.get("state", "Any")
    region = preferences.get("region", "Any")
    location_pool = pool
    location_note = None
    if state != "Any":
        exact = location_pool[location_pool["state_or_ut"].astype(str).str.casefold() == state.casefold()]
        if not exact.empty:
            location_pool = exact
        else:
            location_note = f"No catalogue entries were found for {state}; showing broader matches instead."
    elif region != "Any":
        region_states = [x.casefold() for x in REGIONS.get(region, [])]
        regional = location_pool[location_pool["state_or_ut"].astype(str).str.casefold().isin(region_states)]
        if not regional.empty:
            location_pool = regional
        else:
            location_note = f"No catalogue entries were found for {region}; showing broader matches instead."

    preferred_category = preferences.get("category", "Any")
    pool = location_pool.copy()
    pool["_score"] = 0
    pool["_score"] += pool["_purpose_match"].astype(int) * 5
    if state != "Any":
        pool["_score"] += (pool["state_or_ut"].astype(str).str.casefold() == state.casefold()).astype(int) * 5
    elif region != "Any":
        region_states = [x.casefold() for x in REGIONS.get(region, [])]
        pool["_score"] += pool["state_or_ut"].astype(str).str.casefold().isin(region_states).astype(int) * 3
    if preferred_category != "Any":
        pool["_score"] += (pool["category"].astype(str).str.casefold() == preferred_category.casefold()).astype(int) * 5
    pool = pool.sort_values(["_score", "destination_name", "attraction_name"], ascending=[False, True, True])

    # Obtain similar survey trips and group their observed travel patterns.
    patterns = pd.DataFrame()
    if not history.empty:
        preprocessor, neighbours, err = get_similarity_engine()
        if preprocessor is not None and neighbours is not None:
            query = pd.DataFrame([{
                "household_members": preferences["travellers"],
                "average_member_age": preferences["age"],
                "trip_duration_nights": preferences["nights"],
                "leading_purpose_label": preferences["purpose"],
                "travel_mode_label": preferences["mode"],
                "accommodation_type_label": preferences["accommodation"],
                "main_destination_label": preferences["destination_type"],
            }])
            try:
                qmatrix = preprocessor.transform(query)
                distances, indices = neighbours.kneighbors(qmatrix)
                idx = indices[0]
                matched = history.iloc[idx].copy()
                matched["similarity"] = 1 - distances[0]
                if "mult" in matched.columns:
                    matched["survey_weight"] = pd.to_numeric(matched["mult"], errors="coerce").fillna(100) / 100
                elif "survey_weight" not in matched.columns:
                    matched["survey_weight"] = 1.0
                if "total_trip_expenditure" in matched.columns:
                    matched["total_trip_expenditure"] = pd.to_numeric(matched["total_trip_expenditure"], errors="coerce")
                    budget_matches = matched[matched["total_trip_expenditure"] <= preferences["budget"]]
                    if not budget_matches.empty:
                        matched = budget_matches
                group_cols = [c for c in ["travel_mode_label", "accommodation_type_label"] if c in matched.columns]
                if group_cols:
                    rows = []
                    for keys, group in matched.groupby(group_cols, dropna=False):
                        if not isinstance(keys, tuple):
                            keys = (keys,)
                        item = dict(zip(group_cols, keys))
                        item["matched_trips"] = len(group)
                        item["median_expenditure"] = group["total_trip_expenditure"].median() if "total_trip_expenditure" in group else np.nan
                        item["average_duration"] = group["trip_duration_nights"].mean() if "trip_duration_nights" in group else np.nan
                        item["average_similarity"] = group["similarity"].mean()
                        item["weighted_support"] = (group["similarity"] * group["survey_weight"]).sum()
                        rows.append(item)
                    patterns = pd.DataFrame(rows).sort_values(["average_similarity", "weighted_support"], ascending=False).head(5)
            except Exception:
                patterns = pd.DataFrame()

    return pool, patterns, location_note


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>"


def section_heading(kicker: str, title: str, copy: str = ""):
    st.markdown(
        f"<div class='section-kicker'>{kicker}</div><div class='section-title'>{title}</div><div class='section-copy'>{copy}</div>",
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class='footer'>
          <div style='display:flex;justify-content:space-between;gap:2rem;flex-wrap:wrap'>
            <div style='min-width:210px;max-width:340px'>
              <strong>YATRA / SMART TOURISM</strong>
              <p style='color:#c0c7bf;line-height:1.6;margin-top:.7rem'>Discover India through thoughtful travel planning and machine-learning insights.</p>
              <div style='color:#9ea9a2;font-size:.67rem;letter-spacing:.1em'>AN ACADEMIC PROJECT PROTOTYPE</div>
            </div>
            <div><div style='color:#f1a38a;font-size:.65rem;letter-spacing:.16em;margin-bottom:.55rem'>EXPLORE</div><div>Experiences<br>Destinations<br>Travel planner</div></div>
            <div><div style='color:#f1a38a;font-size:.65rem;letter-spacing:.16em;margin-bottom:.55rem'>INTELLIGENCE</div><div>Expenditure prediction<br>Traveller segments<br>Historical travel patterns</div></div>
            <div><div style='color:#f1a38a;font-size:.65rem;letter-spacing:.16em;margin-bottom:.55rem'>PROJECT</div><div>About the research<br>Data and methodology<br>Model limitations</div></div>
          </div>
          <div style='border-top:1px solid #46514b;margin-top:1.5rem;padding-top:.8rem;color:#9da8a0;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;font-size:.66rem'>
            <span>© 2026 SMART TOURISM MANAGEMENT · INDIA</span>
            <span>Historical survey insights · Not a live booking service</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------
def navigate_to(destination: str):
    st.session_state["navigation"] = destination


header_left, header_right = st.columns([1.15, 2.85], vertical_alignment="center")
with header_left:
    st.markdown(
        "<div class='brandline'><span class='brandmark'>✧</span><span><span class='brandname'>YATRA / SMART TOURISM</span><span class='brandtag'>India · Intelligent travel</span></span></div>",
        unsafe_allow_html=True,
    )
with header_right:
    page = st.radio(
        "Navigate",
        ["Discover", "Plan a Trip", "Budget Predictor", "Tourist Segments", "About"],
        horizontal=True,
        label_visibility="collapsed",
        key="navigation",
    )

history = load_history()
attractions = load_attractions()


# -----------------------------------------------------------------------------
# Page 1: Discover / home
# -----------------------------------------------------------------------------
if page == "Discover":
    st.markdown(
        f"""
        <div class="hero-wrap" style="background-image:linear-gradient(90deg,rgba(14,48,52,.86) 0%,rgba(15,57,60,.55) 42%,rgba(15,57,60,.04) 100%),url('{HERO_IMAGE}')">
          <div class="hero-kicker">A new perspective on every journey</div>
          <div class="hero-title">TRAVEL<br>BY <em>INSIGHT</em></div>
          <div class="hero-copy">Explore India with a little more curiosity. Discover meaningful places, understand travel patterns and plan with machine-learning insights shaped by real domestic tourism records.</div>
          <div class="hero-side-note">SCENIC INDIA<br>CURATED DISCOVERIES<br>TRAVEL JOURNALS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    section_heading("OUR CURATED JOURNEYS", "Signature experiences across India", "A few ways to begin exploring — from living heritage and sacred cities to quiet backwaters and wild landscapes.")

    if not history.empty:
        trip_count = len(history)
        state_count = history["state"].nunique() if "state" in history else 0
        purpose_count = history["leading_purpose_label"].nunique() if "leading_purpose_label" in history else 0
        median_spend = pd.to_numeric(history.get("total_trip_expenditure", pd.Series(dtype=float)), errors="coerce").median()
    else:
        trip_count, state_count, purpose_count, median_spend = 0, 0, 0, np.nan

    featured_journeys = [
        ("Taj Mahal", "Agra", "Uttar Pradesh", "Heritage", "A marble landmark on the banks of the Yamuna, known for its detailed architecture."),
        ("Alappuzha Backwaters", "Alappuzha", "Kerala", "Nature", "Houseboats, palm-lined canals and a slower rhythm of life along Kerala's backwaters."),
        ("Loktak Lake", "Bishnupur", "Manipur", "Nature", "A remarkable freshwater lake known for its floating phumdis and fishing communities."),
    ]
    journey_cols = st.columns(3, gap="medium")
    for idx, (name, destination, state, category, desc) in enumerate(featured_journeys):
        with journey_cols[idx]:
            render_destination_card(
                attraction_name=name, destination_name=destination, state=state,
                category=category, description=desc, key=f"featured_{idx}"
            )

    st.write("")
    section_heading("EXPLORE THE CATALOGUE", "Find places across India", "Search the curated attraction catalogue by place, state and experience. Open each card for planning notes and save places to your shortlist.")
    if attractions.empty:
        st.info("The attraction catalogue is not available yet. Add Dataset/india_tourist_destinations.csv to enable destination browsing.")
    else:
        state_options = clean_options(attractions.get("state_or_ut", pd.Series(dtype=str)).tolist(), [])
        category_options = clean_options(attractions.get("category", pd.Series(dtype=str)).tolist(), [])
        with st.form("discover_catalogue_filters"):
            filter_cols = st.columns([1.5, 1, 1, .7])
            with filter_cols[0]:
                discover_query = st.text_input("Search place or destination", placeholder="Try: Loktak, Varanasi, backwaters...")
            with filter_cols[1]:
                discover_state = st.selectbox("State / UT", ["All India"] + state_options, key="discover_state_filter")
            with filter_cols[2]:
                discover_category = st.selectbox("Experience", ["Any"] + category_options, key="discover_category_filter")
            with filter_cols[3]:
                discover_limit = st.selectbox("Show", [6, 9, 12], index=0, key="discover_limit")
            discover_submit = st.form_submit_button("FIND DESTINATIONS  ↗", use_container_width=True)
        if discover_submit:
            browse = attractions.copy()
            if discover_query.strip():
                searchable = [c for c in ["destination_name", "attraction_name", "city_or_area", "state_or_ut", "keywords", "description"] if c in browse.columns]
                mask = pd.Series(False, index=browse.index)
                for col_name in searchable:
                    mask |= browse[col_name].fillna("").astype(str).str.contains(discover_query.strip(), case=False, regex=False)
                browse = browse[mask]
            if discover_state != "All India" and "state_or_ut" in browse.columns:
                browse = browse[browse["state_or_ut"].astype(str).str.casefold() == discover_state.casefold()]
            if discover_category != "Any" and "category" in browse.columns:
                browse = browse[browse["category"].astype(str).str.casefold() == discover_category.casefold()]
            st.session_state["discover_catalogue_results"] = browse.head(int(discover_limit)).to_dict("records")
            st.session_state["discover_catalogue_total"] = len(browse)
        if "discover_catalogue_results" in st.session_state:
            browse_rows = st.session_state["discover_catalogue_results"]
            total_found = st.session_state.get("discover_catalogue_total", len(browse_rows))
            st.caption(f"Showing {len(browse_rows)} of {total_found:,} matching catalogue entries.")
            if not browse_rows:
                st.info("No matching destinations. Try another name, state or category.")
            for start_i in range(0, len(browse_rows), 3):
                browse_cols = st.columns(3, gap="medium")
                for col_i, row in enumerate(browse_rows[start_i:start_i + 3]):
                    with browse_cols[col_i]:
                        render_destination_card(
                            attraction_name=str(row.get("attraction_name", row.get("destination_name", "Attraction"))),
                            destination_name=str(row.get("destination_name", "India")),
                            state=str(row.get("state_or_ut", "India")),
                            category=str(row.get("category", "Destination")),
                            description=str(row.get("description", "A place to add to your travel list.")),
                            planning_notes=str(row.get("planning_notes", "")),
                            data_status=str(row.get("data_status", "")),
                            catalogue_source=str(row.get("source_name", "")),
                            catalogue_url=str(row.get("source_url", "")),
                            key=f"browse_{start_i}_{col_i}",
                        )

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Historical trips", f"{trip_count:,}", "Overnight survey records"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("States & UTs", f"{state_count}", "Represented in the data"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Travel purposes", f"{purpose_count}", "Purpose categories"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Median trip spend", inr(median_spend) if pd.notna(median_spend) else "—", "Observed expenditure · INR"), unsafe_allow_html=True)

    st.write("")
    st.markdown(
        f"""
        <div class='journal-panel'>
          <img src='{JOURNAL_IMAGE}' alt='Travel journal and notebook on a desk'>
          <div class='journal-copy'>
            <div class='section-kicker'>THE YATRA COLLECTION</div>
            <div class='journal-title'>THE<br>TRAVELLER'S<br>JOURNAL</div>
            <p style='color:#5e6d67;font-size:.87rem;line-height:1.65;margin:.7rem 0 1.1rem'>Every journey leaves a pattern. Explore the stories hidden in domestic tourism data, discover how travellers move and find inspiration for the next place on your map.</p>
            <div style='color:#bd624f;font-size:.68rem;font-weight:700;letter-spacing:.14em'>INSIGHTS · PLACES · POSSIBILITIES</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    route_left, route_right = st.columns([1.05, 1.95], gap="large", vertical_alignment="center")
    with route_left:
        st.markdown(
            "<div class='route-panel' style='height:100%'><div class='route-kicker'>EXPLORE OUR ROUTES</div><div class='route-title'>INDIA,<br>REGION BY REGION</div><div class='route-copy'>Choose a part of the country and start shaping a journey around your interests. Regional filters are available in the personalized trip planner.</div><div style='margin-top:1rem'><span class='region-chip'>North</span><span class='region-chip'>South</span><span class='region-chip'>East</span><span class='region-chip'>West</span><span class='region-chip'>Central</span><span class='region-chip'>North East</span><span class='region-chip'>Islands</span></div></div>",
            unsafe_allow_html=True,
        )
    with route_right:
        st.markdown(
            """
            <div style='background:#eadbc8;padding:1.25rem 1.4rem;min-height:240px;position:relative;overflow:hidden'>
              <div style='font-size:.63rem;color:#b96853;font-weight:700;letter-spacing:.19em'>A COUNTRY OF MANY JOURNEYS</div>
              <div style='font-family:Bebas Neue,Impact,sans-serif;font-size:2rem;color:#244e50;line-height:1;margin:.35rem 0 1rem'>FIND YOUR OWN WAY THROUGH INDIA</div>
              <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem'>
                <div style='border-top:1px solid #b7a58f;padding-top:.55rem'><div style='font-family:Cormorant Garamond,serif;font-size:1.45rem;color:#244e50;font-weight:700'>01</div><b style='font-size:.78rem'>Choose a purpose</b><div style='font-size:.72rem;color:#697770'>Leisure, pilgrimage, health or shopping</div></div>
                <div style='border-top:1px solid #b7a58f;padding-top:.55rem'><div style='font-family:Cormorant Garamond,serif;font-size:1.45rem;color:#244e50;font-weight:700'>02</div><b style='font-size:.78rem'>Set your preferences</b><div style='font-size:.72rem;color:#697770'>Budget, duration and travel style</div></div>
                <div style='border-top:1px solid #b7a58f;padding-top:.55rem'><div style='font-family:Cormorant Garamond,serif;font-size:1.45rem;color:#244e50;font-weight:700'>03</div><b style='font-size:.78rem'>Explore matches</b><div style='font-size:.72rem;color:#697770'>See attractions and similar trips</div></div>
              </div>
              <div style='position:absolute;right:1rem;bottom:.25rem;font-family:Cormorant Garamond,serif;font-size:5rem;color:rgba(29,87,88,.10);font-weight:700'>INDIA</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown(
        f"<div class='quote-strip'><div class='quote-mark'>“</div><div><div class='quote-text'>The best journeys start with curiosity — and become more meaningful when shaped around the way you love to travel.</div><div style='font-size:.64rem;letter-spacing:.16em;margin-top:.45rem;font-weight:700'>A SMARTER WAY TO DISCOVER INDIA</div></div></div>",
        unsafe_allow_html=True,
    )
    st.write("")
    section_heading("EXPLORE THE EXPERIENCE", "Three tools. One more thoughtful journey.", "Move from inspiration to practical planning with the three machine-learning components in this project.")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("<div class='soft-card'><div class='destination-icon'>₹</div><div class='destination-name'>Estimate expenditure</div><p style='color:#718080;font-size:.86rem;line-height:1.6'>Use the trained CatBoost regression model to estimate overnight trip expenditure from your trip details.</p></div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div class='soft-card'><div class='destination-icon'>◈</div><div class='destination-name'>Understand traveller groups</div><p style='color:#718080;font-size:.86rem;line-height:1.6'>Explore broad behavioural segments identified using MiniBatch K-Means clustering.</p></div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div class='soft-card'><div class='destination-icon'>✧</div><div class='destination-name'>Find places to explore</div><p style='color:#718080;font-size:.86rem;line-height:1.6'>Get purpose-led attraction ideas and historical travel patterns similar to your preferences.</p></div>", unsafe_allow_html=True)

    render_saved_shortlist()
    render_footer()

# -----------------------------------------------------------------------------
# Page 2: Personalized recommendations
# -----------------------------------------------------------------------------
elif page == "Plan a Trip":
    section_heading("PERSONALIZED TRAVEL PLANNER", "Find a journey that feels like yours", "Share a few preferences to discover relevant attractions and see what similar historical trips looked like.")
    if history.empty:
        st.error(f"Historical dataset not found. Expected: {HISTORY_PATH}")
    elif attractions.empty:
        st.error(f"Destination catalogue not found. Expected: {ATTRACTIONS_PATH}")
    else:
        states = option_from_history(history, "state", ["Delhi", "Rajasthan", "Uttar Pradesh", "Kerala", "Goa"])
        # Include states in the catalogue even if not observed in the historical sample.
        if "state_or_ut" in attractions.columns:
            states = sorted(set(states + clean_options(attractions["state_or_ut"], [])))
        with st.form("trip_preferences"):
            top1, top2, top3 = st.columns(3)
            with top1:
                purpose = st.selectbox("Main travel purpose", PURPOSES)
                travellers = st.number_input("Number of travellers", min_value=1, max_value=16, value=2, step=1)
                age = st.slider("Average traveller age", min_value=18, max_value=80, value=35)
            with top2:
                budget = st.number_input("Total trip budget (INR)", min_value=500, max_value=5000000, value=25000, step=1000)
                nights = st.number_input("Trip duration (nights)", min_value=1, max_value=180, value=4, step=1)
                mode = st.selectbox("Preferred transport", TRAVEL_MODES)
            with top3:
                accommodation = st.selectbox("Preferred accommodation", ACCOMMODATIONS)
                destination_type = st.selectbox("Destination distance", DESTINATION_TYPES, index=1)
                month = st.selectbox("Preferred travel month", MONTHS, index=0)
            loc1, loc2, loc3 = st.columns(3)
            with loc1:
                preferred_state = st.selectbox("Preferred state / UT", ["Any"] + states)
            with loc2:
                preferred_region = st.selectbox("Or explore a region", ["Any"] + list(REGIONS.keys()))
            with loc3:
                preferred_category = st.selectbox("Experience category", CATEGORIES)
            submitted = st.form_submit_button("Discover recommendations  ↗", use_container_width=True)

        if submitted:
            prefs = {
                "purpose": purpose,
                "travellers": int(travellers),
                "age": int(age),
                "budget": float(budget),
                "nights": int(nights),
                "mode": mode,
                "accommodation": accommodation,
                "destination_type": destination_type,
                "month": month,
                "state": preferred_state,
                "region": preferred_region if preferred_state == "Any" else "Any",
                "category": preferred_category,
            }
            with st.spinner("Matching your preferences with the survey patterns and destination catalogue..."):
                ranked, patterns, note = recommendation_results(prefs, attractions, history)
            st.session_state["last_trip_recommendations"] = ranked
            st.session_state["last_trip_patterns"] = patterns
            st.session_state["last_trip_note"] = note
            st.session_state["last_trip_preferences"] = prefs

        if "last_trip_recommendations" in st.session_state:
            ranked = st.session_state["last_trip_recommendations"]
            patterns = st.session_state["last_trip_patterns"]
            note = st.session_state["last_trip_note"]
            prefs = st.session_state["last_trip_preferences"]
            st.divider()
            section_heading("YOUR DESTINATION SHORTLIST", "Places worth exploring", f"Matched to {prefs['purpose'].lower()} · {prefs['state'] if prefs['state'] != 'Any' else prefs['region']}")
            if note:
                st.info(note)
            if ranked.empty:
                st.warning("No attractions matched these preferences in the current catalogue.")
            else:
                display = ranked.head(9).reset_index(drop=True)
                for start in range(0, len(display), 3):
                    cols = st.columns(3)
                    for col, (_, item) in zip(cols, display.iloc[start:start + 3].iterrows()):
                        with col:
                            name = str(item.get("attraction_name", "Attraction"))
                            destination = str(item.get("destination_name", "India"))
                            state = str(item.get("state_or_ut", ""))
                            category = str(item.get("category", "Destination"))
                            description = str(item.get("description", "A place to add to your travel list."))
                            if len(description) > 150:
                                description = description[:147].rstrip() + "..."
                            score = int(item.get("_score", 0))
                            render_destination_card(
                                attraction_name=name,
                                destination_name=destination,
                                state=state,
                                category=category,
                                description=description,
                                score=score,
                                planning_notes=str(item.get("planning_notes", "")),
                                data_status=str(item.get("data_status", "")),
                                catalogue_source=str(item.get("source_name", "")),
                                catalogue_url=str(item.get("source_url", "")),
                                key=f"recommendation_{start}_{len(str(name))}_{name[:12]}",
                            )
            st.write("")
            section_heading("INSIGHTS FROM SIMILAR TRIPS", "How travellers with similar preferences travelled", "Historical patterns are evidence from the survey, not live fares or availability.")
            if patterns.empty:
                st.info("Similar-trip pattern summaries could not be generated. The destination shortlist is still available.")
            else:
                p_display = patterns.reset_index(drop=True)
                for i, row in p_display.iterrows():
                    mode_label = row.get("travel_mode_label", "Not available")
                    stay_label = row.get("accommodation_type_label", "Not available")
                    expenditure = inr(row.get("median_expenditure", np.nan))
                    duration = f"{row.get('average_duration', np.nan):.1f} nights" if pd.notna(row.get("average_duration")) else "—"
                    similarity = f"{row.get('average_similarity', 0):.0%}"
                    st.markdown(
                        f"<div class='soft-card' style='margin-bottom:.55rem;padding:.85rem 1rem'><b>{i+1}. {mode_label} · {stay_label}</b><div style='color:#718080;font-size:.84rem;margin-top:.25rem'>Median observed expenditure: {expenditure} &nbsp;·&nbsp; Average duration: {duration} &nbsp;·&nbsp; Similarity: {similarity} &nbsp;·&nbsp; Matched trips: {int(row.get('matched_trips',0)):,}</div></div>",
                        unsafe_allow_html=True,
                    )
            st.caption("The budget filter applies to observed historical trip expenditure only. The attraction catalogue does not contain verified current prices.")
        render_saved_shortlist()
    render_footer()


# -----------------------------------------------------------------------------
# Page 3: Expenditure prediction
# -----------------------------------------------------------------------------
elif page == "Budget Predictor":
    section_heading("MACHINE-LEARNING ESTIMATE", "Plan your trip budget", "Estimate total overnight trip expenditure from the trained CatBoost model using your trip details.")
    if history.empty:
        st.error(f"Historical dataset not found. Expected: {HISTORY_PATH}")
    else:
        states = residence_state_options(history)
        if "destination_state_label" in history:
            destination_states = option_from_history(history, "destination_state_label", ["Not applicable"])
        else:
            destination_states = ["Not applicable"]
        with st.form("budget_prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                household_members = st.number_input("Number of travellers", min_value=1, max_value=16, value=2, step=1)
                average_age = st.slider("Average age", min_value=18, max_value=80, value=35)
                duration = st.number_input("Trip duration (nights)", min_value=1, max_value=180, value=4, step=1)
                trip_type = st.selectbox("Trip type", TRIP_TYPES)
                purpose = st.selectbox("Purpose", PURPOSES)
                trip_month = st.selectbox("Travel month", MONTHS)
            with c2:
                travel_mode = st.selectbox("Main transport", TRAVEL_MODES)
                accommodation = st.selectbox("Accommodation", ACCOMMODATIONS)
                destination_type = st.selectbox("Destination type", DESTINATION_TYPES, index=1)
                state = st.selectbox("Your state of residence", states)
                sector = st.selectbox("Residence sector", SECTORS)
                if destination_type in DESTINATION_TYPES[:2]:
                    destination_state = "Not applicable"
                    st.text_input("Destination state", value="Not applicable (not collected for this trip type)", disabled=True)
                else:
                    destination_state = st.selectbox("Destination state", [x for x in destination_states if x != "Not applicable"] or states)
            submitted_prediction = st.form_submit_button("Estimate trip expenditure  ↗", use_container_width=True)

        if submitted_prediction:
            values = {
                "household_members": int(household_members),
                "average_member_age": float(average_age),
                "trip_duration_nights": float(duration),
                "trip_type_label": trip_type,
                "travel_mode_label": travel_mode,
                "accommodation_type_label": accommodation,
                "leading_purpose_label": purpose,
                "trip_month_label": trip_month,
                "main_destination_label": destination_type,
                "destination_state_label": destination_state,
                "state": residence_state_model_value(state, history),
                "sector_label": sector,
            }
            with st.spinner("Calculating your estimate..."):
                estimate, error = predict_catboost(values)
            if error:
                st.error(error)
                st.caption("Check that outputs/models/best_model_catboost.cbm and catboost_metadata.pkl are present.")
            else:
                st.session_state["last_budget_estimate"] = estimate
                st.session_state["last_budget_input"] = values

        if "last_budget_estimate" in st.session_state:
            estimate = st.session_state["last_budget_estimate"]
            vals = st.session_state["last_budget_input"]
            st.write("")
            st.markdown(
                f"<div class='hero-wrap' style='min-height:220px;padding:2.2rem 2.5rem;background-image:linear-gradient(110deg,rgba(14,55,57,.96),rgba(23,81,80,.88))'><div class='hero-kicker'>Your estimated total trip expenditure</div><div class='hero-title' style='font-size:3rem'>{inr(estimate)}</div><div class='hero-copy'>For {vals['household_members']} traveller(s) · {vals['trip_duration_nights']:.0f} nights · {vals['leading_purpose_label']}</div></div>",
                unsafe_allow_html=True,
            )
            st.info("This is a statistical estimate from historical survey data, not a live quote. Actual costs can differ by destination, dates, prices and personal choices.")
            if "total_trip_expenditure" in history.columns:
                median_value = pd.to_numeric(history["total_trip_expenditure"], errors="coerce").median()
                st.write(f"For context, the dataset-wide median observed trip expenditure is **{inr(median_value)}**. This is not a recommended budget.")
    render_footer()


# -----------------------------------------------------------------------------
# Page 4: Tourist segmentation
# -----------------------------------------------------------------------------
elif page == "Tourist Segments":
    section_heading("UNSUPERVISED LEARNING", "Explore traveller segments", "The final segmentation model groups trips by travel behaviour, not by expenditure. These are broad patterns rather than sharply separated classes.")
    profiles = load_cluster_profiles()
    if profiles.empty:
        st.warning(f"Cluster profile file not found: {CLUSTER_PROFILES_PATH}")
    else:
        st.markdown("**Observed cluster profiles**")
        st.dataframe(profiles, use_container_width=True, hide_index=True)

    st.write("")
    section_heading("TRY THE SEGMENT MODEL", "Which group does this trip resemble?", "Enter a sample trip profile and the saved MiniBatch K-Means model will assign its nearest cluster.")
    if history.empty:
        st.error(f"Historical dataset not found: {HISTORY_PATH}")
    else:
        with st.form("segmentation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                seg_members = st.number_input("Household / traveller count", min_value=1, max_value=16, value=2, key="seg_members")
                seg_age = st.slider("Average age", min_value=18, max_value=80, value=35, key="seg_age")
                seg_nights = st.number_input("Trip duration (nights)", min_value=1, max_value=180, value=4, key="seg_nights")
                seg_trip_type = st.selectbox("Trip type", TRIP_TYPES, key="seg_trip_type")
            with c2:
                seg_mode = st.selectbox("Travel mode", TRAVEL_MODES, key="seg_mode")
                seg_stay = st.selectbox("Accommodation", ACCOMMODATIONS, key="seg_stay")
                seg_purpose = st.selectbox("Purpose", PURPOSES, key="seg_purpose")
                seg_month = st.selectbox("Travel month", MONTHS, key="seg_month")
            with c3:
                seg_destination = st.selectbox("Destination type", DESTINATION_TYPES, index=1, key="seg_destination")
                seg_sector = st.selectbox("Residence sector", SECTORS, key="seg_sector")
            seg_submit = st.form_submit_button("Identify traveller segment  ↗", use_container_width=True)
        if seg_submit:
            seg_values = {
                "household_members": int(seg_members),
                "average_member_age": float(seg_age),
                "trip_duration_nights": float(seg_nights),
                "trip_type_label": seg_trip_type,
                "travel_mode_label": seg_mode,
                "accommodation_type_label": seg_stay,
                "leading_purpose_label": seg_purpose,
                "trip_month_label": seg_month,
                "main_destination_label": seg_destination,
                "sector_label": seg_sector,
            }
            cluster, error = run_segmentation_prediction(seg_values)
            if error:
                st.error(error)
                st.caption("The saved artifact must contain the preprocessor and K-Means model used by finalize_segmentation.py.")
            else:
                st.success(f"This trip is assigned to **Cluster {cluster}**.")
                if not profiles.empty:
                    cluster_col = next((c for c in profiles.columns if c.lower() in {"cluster", "cluster_id", "segment"}), None)
                    if cluster_col:
                        match = profiles[pd.to_numeric(profiles[cluster_col], errors="coerce") == cluster]
                        if not match.empty:
                            st.markdown("**Profile for this cluster**")
                            st.dataframe(match, use_container_width=True, hide_index=True)
    # Display saved distribution chart when available.
    cluster_chart = SEGMENTATION_DIR / "cluster_distribution.png"
    if cluster_chart.exists():
        st.write("")
        st.image(str(cluster_chart), caption="Distribution of trips across the selected clusters", use_container_width=True)
    st.caption("The latest saved segmentation experiment selected K = 3. Its silhouette score was modest, so interpret cluster profiles as broad behavioural groups.")
    render_footer()


# -----------------------------------------------------------------------------
# Page 5: About and methodology
# -----------------------------------------------------------------------------
elif page == "About":
    section_heading("ABOUT THE PROJECT", "AI-driven smart tourism management", "A machine-learning capstone combining expenditure prediction, tourist segmentation and personalized travel planning.")
    st.markdown(
        """
        <div class="soft-card">
          <div class="section-kicker">DATA SOURCE</div>
          <div class="destination-name">Domestic Tourism Expenditure Survey</div>
          <p style="color:#647675">The expenditure and travel-pattern analysis uses overnight domestic tourism records prepared from the Government of India's DTES microdata supplied for this project.</p>
          <div class="section-kicker">MACHINE-LEARNING COMPONENTS</div>
          <p style="color:#647675"><b>Expenditure prediction:</b> CatBoost regression predicts total overnight trip expenditure from household, trip, transport, accommodation, purpose and destination attributes.</p>
          <p style="color:#647675"><b>Tourist segmentation:</b> MiniBatch K-Means groups trips using behavioural features. Expenditure is excluded from the final behavioural clustering feature set.</p>
          <p style="color:#647675"><b>Personalized planning:</b> Similar historical trips are retrieved using feature similarity, while destination attractions are ranked using purpose, location and category preferences.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    section_heading("IMPORTANT INTERPRETATION NOTES", "What this prototype can and cannot tell you", "Keep these limitations in mind when interpreting predictions and recommendations.")
    notes = [
        "Expenditure predictions are estimates from historical survey records. They are not guaranteed prices or quotations.",
        "Historical similarity is not a live travel search. The app does not verify hotel availability, transport schedules or current ticket prices.",
        "The attraction catalogue is curated for this project and is not exhaustive. Verify destination details before travelling.",
        "Destination cards try to display an attraction-specific photograph first, then a photograph of the relevant state. Image source and licence links are shown in the place details when available.",
        "The segmentation silhouette score is relatively low, which indicates broad, overlapping behavioural groups.",
        "Model evaluation should be reported using regression metrics such as R², MAE and RMSE, not classification accuracy.",
    ]
    for note in notes:
        st.markdown(f"- {note}")
    st.write("")
    section_heading("MODEL PERFORMANCE", "Final test-set comparison", "Values below are from the completed evaluation of the current DTES expenditure models.")
    final_comparison_path = OUTPUT_DIR / "final_evaluation" / "final_model_comparison.csv"
    comparison = safe_read_csv(final_comparison_path)
    if comparison is not None and not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    else:
        st.info("Final model comparison file is not available. The model metrics can be added after running final_model_evaluation.py.")
    render_footer()

