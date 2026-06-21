"""
app.py
------
Interface Streamlit — Detection de Faux Billets
Style : blanc, bleu marine, professionnel
"""

import os
import json
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
# CONFIG PAGE  — doit etre le 1er appel Streamlit
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FinPRO — Detection de Faux Billets",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

API_URL         = "https://finpro-api-0o4g.onrender.com"
FEATURE_NAMES   = ["diagonal","height_left","height_right","margin_low","margin_up","length"]
HISTORIQUE_FILE      = "historique_analyses.json"   # historique admin (toutes les analyses)
HISTORIQUE_USER_FILE = "historique_user.json"       # historique utilisateur (ses analyses)

# ─────────────────────────────────────────────
# COMPTES UTILISATEURS
# ─────────────────────────────────────────────

COMPTES = {
    "utilisateur": {"mot_de_passe": "user123",  "role": "user"},
    "Madara": {"mot_de_passe": "madara24",  "role": "user"},
    "admin":       {"mot_de_passe": "admin123",  "role": "admin"},
}

# Palette — bleu marine + vert/rouge sobres
NAVY    = "#1B3A6B"
BLUE    = "#2563EB"
BLUE_L  = "#EFF6FF"
BLUE_B  = "#BFDBFE"
GREEN   = "#16A34A"
GREEN_L = "#F0FDF4"
GREEN_B = "#BBF7D0"
RED     = "#DC2626"
RED_L   = "#FEF2F2"
RED_B   = "#FECACA"
ORANGE  = "#D97706"
ORANGE_L= "#FFFBEB"
GRAY    = "#475569"
GRAY_L  = "#F8FAFC"
BORDER  = "#E2E8F0"
TEXT    = "#0F172A"
TEXT2   = "#334155"

# ─────────────────────────────────────────────
# CSS — style financier professionnel
# ─────────────────────────────────────────────

st.markdown(f"""
<style>
/* ── Reset & typographie ── */
html, body, [class*="css"] {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* ── Fond general blanc ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
section.main > div {{
    background-color: #F8FAFC !important;
}}

/* ── Tous les textes en NOIR — sauf h1/h2 qui restent bleu marine ── */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span:not([data-testid]),
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] small,
.stMarkdown p, .stMarkdown li,
.stMarkdown span {{ color: #000000 !important; }}

/* Labels des sliders en noir */
[data-testid="stSlider"] label,
[data-testid="stSlider"] label p,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {{ color: #000000 !important; }}

/* Valeur affichee au-dessus du curseur en noir */
[data-testid="stSlider"] p {{ color: #000000 !important; }}

/* Labels des champs texte en noir */
[data-testid="stTextInput"] label,
[data-testid="stTextInput"] label p {{ color: #000000 !important; }}

/* Captions en noir attenué */
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span {{ color: #222222 !important; font-size: 13px !important; }}

/* ── Sidebar — bleu marine ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY} 0%, #112854 100%) !important;
    border-right: none !important;
}}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {{
    color: #CBD5E1 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.10) !important;
    margin: 14px 0 !important;
}}
[data-testid="stSidebar"] .stRadio > label {{
    font-size: 12px !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {{
    font-size: 14px !important;
    color: #CBD5E1 !important;
    padding: 8px 12px !important;
    border-radius: 7px !important;
    transition: background .15s !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {{
    background: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
}}

/* ── Titres ── */
h1 {{
    font-size: 26px !important; font-weight: 700 !important;
    color: {NAVY} !important; letter-spacing: -.02em !important;
    margin-bottom: 4px !important;
}}
h2 {{
    font-size: 19px !important; font-weight: 600 !important;
    color: {NAVY} !important; margin-top: 8px !important;
}}
h3 {{
    font-size: 15px !important; font-weight: 600 !important;
    color: {NAVY} !important;
}}

/* ── Metriques ── */
[data-testid="metric-container"] {{
    background: #FFFFFF !important;
    border: 1px solid {BORDER} !important;
    border-top: 3px solid {BLUE} !important;
    border-radius: 10px !important;
    padding: 18px 22px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}}
[data-testid="stMetricValue"] {{
    font-size: 30px !important; font-weight: 700 !important;
    color: {NAVY} !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: 11px !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: .07em;
    color: #000000 !important;
}}

/* ── Bouton principal ── */
.stButton > button[kind="primary"] {{
    background: {BLUE} !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 7px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.28) !important;
    transition: all .18s ease !important;
    letter-spacing: .01em !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: #1D4ED8 !important;
    color: #000000 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.38) !important;
    transform: translateY(-1px) !important;
}}

.stButton > button[kind="primary"]:active {{
    transform: translateY(0) !important;
}}

/* ── Bouton secondaire ── */
.stButton > button[kind="secondary"] {{
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid {BORDER} !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
    transition: all .15s ease !important;
}}

.stButton > button[kind="secondary"]:hover {{
    background: {GRAY_L} !important;
    color: #000000 !important;
    border-color: #CBD5E1 !important;
}}

/* Forcer le texte des boutons en noir */
.stButton button,
.stButton button p,
.stButton button span {{
    color: #000000 !important;
}}
/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: #FFFFFF !important;
    border: 2px dashed {BLUE_B} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color .2s !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {BLUE} !important; }}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    color: #1E293B !important; font-size: 14px !important;
}}
[data-testid="stFileUploader"] small {{ color: #334155 !important; }}
[data-testid="stFileUploader"] button {{
    color: {BLUE} !important;
    border-color: {BLUE} !important;
    background: {BLUE_L} !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: #FFFFFF !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}}
[data-testid="stExpander"] summary {{
    color: {NAVY} !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
}}
[data-testid="stExpander"] summary span {{ color: {NAVY} !important; }}
[data-testid="stExpander"] p,
[data-testid="stExpander"] span,
[data-testid="stExpander"] li,
[data-testid="stExpander"] div {{ color: {TEXT} !important; }}
[data-testid="stExpander"] pre,
[data-testid="stExpander"] code {{ color: {NAVY} !important; background: {BLUE_L} !important; }}

/* ── Alertes ── */
.stAlert {{ border-radius: 9px !important; font-weight: 500 !important; border-left-width: 5px !important; }}
.stAlert p, .stAlert span, .stAlert div {{ color: inherit !important; }}

/* ── Succes ── */
div[data-baseweb="notification"][kind="positive"],
div[class*="stSuccess"] {{ background: {GREEN_L} !important; border-color: {GREEN} !important; }}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #FFFFFF !important;
}}

/* ── Separateur ── */
hr {{ border: none !important; border-top: 1px solid {BORDER} !important; margin: 22px 0 !important; }}

/* ── Code inline ── */
code {{ color: {NAVY} !important; background: {BLUE_L} !important; border-radius: 4px !important; padding: 1px 6px !important; }}
pre  {{ color: {TEXT} !important; background: {GRAY_L} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important; }}

/* ── Spinner texte ── */
[data-testid="stSpinner"] p {{ color: #334155 !important; }}

/* ── Badges risque ── */
.badge-ok     {{ display:inline-block; padding:5px 16px; border-radius:999px; font-size:13px; font-weight:700; background:{GREEN_L};  color:{GREEN};  border:1px solid {GREEN_B};  }}
.badge-alerte {{ display:inline-block; padding:5px 16px; border-radius:999px; font-size:13px; font-weight:700; background:{ORANGE_L}; color:{ORANGE}; border:1px solid #FDE68A; }}
.badge-danger {{ display:inline-block; padding:5px 16px; border-radius:999px; font-size:13px; font-weight:700; background:{RED_L};    color:{RED};    border:1px solid {RED_B};   }}

/* ── Section card (wrapper blanc) ── */
.section-card {{
    background: #FFFFFF;
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {{
    background: #FFFFFF !important;
    color: {BLUE} !important;
    border: 1px solid {BLUE_B} !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all .15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: {BLUE_L} !important;
    border-color: {BLUE} !important;
}}

/* ── Bouton Se connecter (form_submit_button) — bleu marine visible ── */
[data-testid="stFormSubmitButton"] > button {{
    background: {NAVY} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    letter-spacing: .03em !important;
    box-shadow: 0 4px 14px rgba(27,58,107,0.35) !important;
    transition: all .2s ease !important;
    text-transform: uppercase !important;
}}
[data-testid="stFormSubmitButton"] > button:hover {{
    background: {BLUE} !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
    transform: translateY(-2px) !important;
}}
[data-testid="stFormSubmitButton"] > button:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(27,58,107,0.3) !important;
}}
[data-testid="stFormSubmitButton"] > button p,
[data-testid="stFormSubmitButton"] > button span {{
    color: #FFFFFF !important;
}}

/* ── Sliders — bleu foncé ── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background-color: {NAVY} !important;
    border-color: {NAVY} !important;
}}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] + div > div:first-child {{
    background-color: {NAVY} !important;
}}
/* Barre remplie du slider (track gauche) */
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div:first-child {{
    background-color: {NAVY} !important;
}}
/* Thumb (poignée) */
[data-testid="stSlider"] div[role="slider"] {{
    background-color: {NAVY} !important;
    border-color: {NAVY} !important;
    box-shadow: 0 0 0 3px rgba(27,58,107,0.18) !important;
}}
/* Partie active de la track */
[data-baseweb="slider"] [data-testid="stSlider"] div {{
    background: {NAVY} !important;
}}
/* Override global accent color pour les sliders */
[data-testid="stSlider"] .st-emotion-cache-1gv3huu,
[data-testid="stSlider"] [class*="st-"] {{
    color: {NAVY} !important;
}}
/* Track remplie — sélecteur universel Streamlit */
div[data-testid="stSlider"] > div > div > div > div:nth-child(2) {{
    background: {NAVY} !important;
}}
div[data-testid="stSlider"] > div > div > div > div > div[role="slider"] {{
    background-color: {NAVY} !important;
    border: 2px solid {NAVY} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HISTORIQUE
# ─────────────────────────────────────────────

def _fichier_historique(role):
    return HISTORIQUE_FILE if role == "admin" else HISTORIQUE_USER_FILE

def charger_historique(role=None):
    if role is None:
        role = st.session_state.get("role", "user")
    fichier = _fichier_historique(role)
    if not os.path.exists(fichier):
        return []
    try:
        with open(fichier, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def sauvegarder_historique(h, role=None):
    if role is None:
        role = st.session_state.get("role", "user")
    fichier = _fichier_historique(role)
    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(h, f, ensure_ascii=False, indent=2)

def ajouter_entree_historique(nom_fichier, nb_total, nb_auth, nb_faux):
    role     = st.session_state.get("role", "user")
    username = st.session_state.get("username", "inconnu")
    entree   = {
        "date"       : datetime.now().strftime("%d/%m/%Y"),
        "heure"      : datetime.now().strftime("%H:%M:%S"),
        "utilisateur": username,
        "fichier"    : nom_fichier,
        "nb_total"   : nb_total,
        "nb_auth"    : nb_auth,
        "nb_faux"    : nb_faux,
        "taux_faux"  : round((nb_faux / nb_total) * 100, 1) if nb_total > 0 else 0.0,
        "statut"     : "ALERTE" if nb_faux > 0 else "OK"
    }
    # Sauvegarder dans le fichier du role connecte
    h = charger_historique(role)
    h.insert(0, entree)
    sauvegarder_historique(h[:50], role)
    # Si utilisateur normal, copier aussi dans l'historique admin
    if role == "user":
        h_admin = charger_historique("admin")
        h_admin.insert(0, entree)
        sauvegarder_historique(h_admin[:50], "admin")


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────

def verifier_api():
    try:
        r = requests.get(f"{API_URL}/sante", timeout=3)
        return r.status_code == 200
    except:
        return False

def appeler_api_predire(mesures):
    r = requests.post(f"{API_URL}/predire", json=mesures, timeout=10)
    r.raise_for_status()
    return r.json()

def appeler_api_predire_lot(liste_mesures):
    r = requests.post(f"{API_URL}/predire-lot", json={"billets": liste_mesures}, timeout=30)
    r.raise_for_status()
    return r.json()

def nettoyer_lot(df):
    nb_nan = df[FEATURE_NAMES].isnull().sum().sum()
    for col in FEATURE_NAMES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())
    return df, nb_nan


# ─────────────────────────────────────────────
# MATPLOTLIB — style professionnel clair
# ─────────────────────────────────────────────

def style_matplotlib():
    plt.rcParams.update({
        "figure.facecolor" : "#FFFFFF",
        "axes.facecolor"   : "#F8FAFC",
        "axes.edgecolor"   : BORDER,
        "axes.labelcolor"  : GRAY,
        "axes.titlecolor"  : NAVY,
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.grid"        : True,
        "grid.color"       : BORDER,
        "grid.linewidth"   : 1,
        "grid.linestyle"   : "-",
        "grid.alpha"       : 1,
        "xtick.color"      : GRAY,
        "ytick.color"      : GRAY,
        "font.family"      : "DejaVu Sans",
        "font.size"        : 11,
        "axes.titlesize"   : 13,
        "axes.titleweight" : "bold",
        "axes.labelsize"   : 11,
        "legend.framealpha": 0.95,
        "legend.edgecolor" : BORDER,
        "legend.fontsize"  : 10,
    })


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def afficher_sidebar():
    with st.sidebar:

        # Logo
        st.markdown(f"""
        <div style="padding:28px 0 22px 0; text-align:center;">
            <div style="font-size:38px; margin-bottom:8px;">💵</div>
            <div style="font-size:20px; font-weight:800; color:#FFFFFF; letter-spacing:-.01em;">FinPRO</div>
            <div style="font-size:12px; color:#94A3B8; margin-top:3px; font-weight:400;">Detection de Faux Billets</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("<div style='font-size:11px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:.08em;padding:0 4px;margin-bottom:8px;'>Menu</div>", unsafe_allow_html=True)

        role_actuel  = st.session_state.get("role", "user")
        pages_user   = ["🔍  Analyser un billet", "📦  Analyser un lot", "📋  Mon historique"]
        pages_admin  = ["🔍  Analyser un billet", "📦  Analyser un lot", "📋  Mon historique", "🗂️  Historique global", "📊  Tableau de bord Admin", "ℹ️  A propos"]
        options_menu = pages_admin if role_actuel == "admin" else pages_user

        page = st.radio(
            "Menu",
            options_menu,
            label_visibility="collapsed"
        )

        st.divider()

        # Statut API — visible uniquement pour l'admin
        api_ok = verifier_api()
        if role_actuel == "admin":
            st.markdown("<div style='font-size:11px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;'>Statut API</div>", unsafe_allow_html=True)
            if api_ok:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;background:rgba(22,163,74,0.15);border:1px solid rgba(22,163,74,0.3);border-radius:8px;padding:10px 14px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#22C55E;box-shadow:0 0 6px rgba(34,197,94,0.6);"></div>
                    <span style="color:#FFFFFF !important;font-size:13px;font-weight:600;">En ligne</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;background:rgba(220,38,38,0.15);border:1px solid rgba(220,38,38,0.3);border-radius:8px;padding:10px 14px;">
                    <div style="width:8px;height:8px;border-radius:50%;background:#EF4444;"></div>
                    <span style="color:#FFFFFF !important;font-size:13px;font-weight:600;">Hors ligne</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div style='font-size:11px;color:#64748B;margin-top:6px;font-family:monospace;padding:0 2px;'>uvicorn api:app --reload</div>", unsafe_allow_html=True)

            st.divider()

       

        # Utilisateur connecté + déconnexion
        username = st.session_state.get("username", "")
        role     = st.session_state.get("role", "user")
        role_label = "Administrateur" if role == "admin" else "Utilisateur"
        role_icon  = "🔑" if role == "admin" else "👤"

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12);
                    border-radius:10px; padding:12px 14px; margin-bottom:12px;">
            <div style="font-size:13px; font-weight:700; color:#FFFFFF;">{role_icon} {username}</div>
            <div style="font-size:11px; color:#94A3B8; margin-top:2px;">{role_label}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪  Se déconnecter", use_container_width=True, type="secondary"):
            se_deconnecter()
            st.rerun()

    return page, api_ok


# ─────────────────────────────────────────────
# PAGE 1 — ANALYSER UN BILLET
# ─────────────────────────────────────────────

def page_analyser_billet(api_ok):

    # En-tête
    st.markdown(f"<h1>Analyser un billet</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Renseignez les six mesures physiques du billet pour obtenir le verdict du modèle.</p>", unsafe_allow_html=True)
    st.divider()

    with st.form("formulaire_billet"):
        st.markdown(f"<h2 style='margin-bottom:20px;'>Mesures physiques</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")
        with col1:
            diagonal     = st.slider("Diagonale (mm)",        171.0, 173.5, 172.0, 0.01)
            height_left  = st.slider("Hauteur gauche (mm)",   103.0, 105.0, 104.0, 0.01)
            height_right = st.slider("Hauteur droite (mm)",   103.0, 105.0, 104.0, 0.01)
        with col2:
            margin_low   = st.slider("Marge inferieure (mm)", 2.5,   7.0,   4.5,   0.01)
            margin_up    = st.slider("Marge superieure (mm)", 2.0,   4.5,   3.0,   0.01)
            length       = st.slider("Longueur (mm)",          109.0, 115.0, 112.5, 0.01)

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        soumettre = st.form_submit_button("Analyser ce billet", use_container_width=True, type="primary")

    if not soumettre:
        return

    if not api_ok:
        st.error("L'API est hors ligne. Lancez `uvicorn api:app --reload` dans un terminal.")
        return

    mesures = {
        "diagonal": diagonal, "height_left": height_left, "height_right": height_right,
        "margin_low": margin_low, "margin_up": margin_up, "length": length
    }

    with st.spinner("Analyse en cours..."):
        try:
            res      = appeler_api_predire(mesures)
            est_auth = res["prediction"] == 1

            st.divider()
            st.markdown("<h2>Résultat</h2>", unsafe_allow_html=True)

            # Verdict
            if est_auth:
                st.success(f"✅  BILLET AUTHENTIQUE  —  Confiance : {res['probabilite_auth']:.1f} %")
            else:
                st.error(f"🚨  FAUX BILLET DÉTECTÉ  —  Confiance : {res['probabilite_faux']:.1f} %")

            # Métriques
            c1, c2 = st.columns(2)
            c1.metric("Probabilité — Authentique", f"{res['probabilite_auth']:.2f} %")
            c2.metric("Probabilité — Faux billet", f"{res['probabilite_faux']:.2f} %")

            # Barre
            st.markdown(f"<p style='color:#334155;font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin:16px 0 6px 0;'>Niveau de confiance (Authentique)</p>", unsafe_allow_html=True)
            st.progress(int(res["probabilite_auth"]))

            # ── Visualisations graphiques ──────────────────────
            st.divider()
            st.markdown("<h2>Visualisation des probabilités</h2>", unsafe_allow_html=True)

            style_matplotlib()
            col_pie, col_bar = st.columns(2, gap="large")

            prob_auth = res["probabilite_auth"]
            prob_faux = res["probabilite_faux"]

            # ── Diagramme circulaire ──
            with col_pie:
                st.markdown(
                    f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>"
                    f"Répartition Authentique / Faux</p>",
                    unsafe_allow_html=True
                )
                fig_pie, ax_pie = plt.subplots(figsize=(4.2, 3.5))

                vals_pie  = [prob_auth, prob_faux]
                labels_pie = [
                    f"Authentique\n({prob_auth:.1f} %)",
                    f"Faux\n({prob_faux:.1f} %)"
                ]
                colors_pie = [GREEN, RED]
                explode_pie = (0.04, 0.04)

                wedges, texts, autotexts = ax_pie.pie(
                    vals_pie,
                    labels=labels_pie,
                    colors=colors_pie,
                    autopct="%1.1f%%",
                    startangle=90,
                    explode=explode_pie,
                    textprops={"fontsize": 10, "color": TEXT},
                    wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
                )
                for at in autotexts:
                    at.set_fontsize(11)
                    at.set_fontweight("bold")
                    at.set_color("white")

                ax_pie.set_title("Part de probabilité par classe", pad=12)
                plt.tight_layout()
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            # ── Diagramme en barres horizontales ──
            with col_bar:
                st.markdown(
                    f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>"
                    f"Comparaison des probabilités</p>",
                    unsafe_allow_html=True
                )
                fig_bar, ax_bar = plt.subplots(figsize=(4.2, 3.5))

                categories  = ["Authentique", "Faux billet"]
                valeurs     = [prob_auth, prob_faux]
                bar_colors  = [GREEN if prob_auth >= prob_faux else GREEN_B,
                               RED   if prob_faux >  prob_auth else RED_B]

                bars = ax_bar.barh(
                    categories, valeurs,
                    color=bar_colors,
                    height=0.45,
                    edgecolor="white",
                    linewidth=1.5
                )

                # Ligne de seuil à 50 %
                ax_bar.axvline(50, color=ORANGE, linewidth=1.5,
                               linestyle="--", alpha=0.8, label="Seuil 50 %")

                # Valeurs sur les barres
                for bar, val in zip(bars, valeurs):
                    ax_bar.text(
                        min(val + 1.5, 97), bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f} %",
                        va="center", ha="left",
                        fontsize=11, fontweight="bold", color=TEXT
                    )

                ax_bar.set_xlim(0, 105)
                ax_bar.set_xlabel("Probabilité (%)")
                ax_bar.set_title("Probabilité par classe (verdict modèle)", pad=12)
                ax_bar.legend(loc="lower right", fontsize=9)
                ax_bar.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close(fig_bar)

            # ── Tableau mesures ──
            st.divider()
            st.markdown("<h2>Mesures saisies</h2>", unsafe_allow_html=True)
            df_aff = pd.DataFrame([mesures]).T.rename(columns={0: "Valeur (mm)"})
            df_aff.index.name = "Feature"
            st.dataframe(df_aff, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur : {e}")


# ─────────────────────────────────────────────
# PAGE 2 — ANALYSER UN LOT
# ─────────────────────────────────────────────

def page_analyser_lot(api_ok):

    st.markdown("<h1>Analyser un lot de billets</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Importez un fichier CSV pour analyser plusieurs billets en une seule opération.</p>", unsafe_allow_html=True)
    st.divider()

    with st.expander("Format attendu du fichier CSV"):
        st.markdown(f"<p style='color:{TEXT};'>Le fichier doit contenir ces <strong>6 colonnes</strong> :</p>", unsafe_allow_html=True)
        st.code(
            "diagonal;height_left;height_right;margin_low;margin_up;length\n"
            "171.81;104.86;104.95;4.52;2.89;112.83\n"
            "172.19;104.63;104.18;5.27;3.37;110.97",
            language="text"
        )
        st.caption("Séparateur accepté : point-virgule (;) ou virgule (,)")

    fichier = st.file_uploader("Choisissez votre fichier CSV", type=["csv"])

    if fichier is None:
        st.markdown(f"<p style='color:#334155;font-size:14px;margin-top:12px;'>En attente d'un fichier CSV...</p>", unsafe_allow_html=True)
        return

    try:
        df_lot = pd.read_csv(fichier, sep=";")
        if len(df_lot.columns) == 1:
            fichier.seek(0)
            df_lot = pd.read_csv(fichier, sep=",")
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        return

    colonnes_manquantes = [c for c in FEATURE_NAMES if c not in df_lot.columns]
    if colonnes_manquantes:
        st.error(f"Colonnes manquantes : {colonnes_manquantes}")
        return

   
    st.success(f"✅  Fichier chargé — **{len(df_lot)} billets** détectés")

    with st.expander("Aperçu des données"):
        st.dataframe(df_lot[FEATURE_NAMES].head(5), use_container_width=True)

    st.divider()

    if not st.button("Analyser tous les billets", type="primary", use_container_width=True):
        return

    if not api_ok:
        st.error("L'API est hors ligne.")
        return

    with st.spinner(f"Analyse de {len(df_lot)} billets en cours..."):
        try:
            res      = appeler_api_predire_lot(df_lot[FEATURE_NAMES].to_dict(orient="records"))
            nb_auth  = res["nb_authentiques"]
            nb_faux  = res["nb_faux"]
            nb_total = res["total_analyse"]
            taux     = round((nb_faux / nb_total) * 100, 1) if nb_total > 0 else 0

            ajouter_entree_historique(fichier.name, nb_total, nb_auth, nb_faux)

            st.divider()

            # ── Indicateur de risque ──
            st.markdown("<h2>Niveau de risque</h2>", unsafe_allow_html=True)

            if taux == 0:
                badge = "<span class='badge-ok'>✅  LOT SAIN</span>"
                desc  = "Aucun faux billet détecté. Le lot peut être mis en circulation."
            elif taux <= 10:
                badge = "<span class='badge-alerte'>⚠️  LOT SUSPECT</span>"
                desc  = f"{taux} % des billets sont suspects. Une vérification manuelle est recommandée."
            else:
                badge = "<span class='badge-danger'>🚨  LOT DANGEREUX</span>"
                desc  = f"{taux} % des billets sont des contrefaçons. Ne pas mettre en circulation."

            st.markdown(badge, unsafe_allow_html=True)
            st.markdown(f"<p style='color:{TEXT};font-size:14px;margin-top:10px;'>{desc}</p>", unsafe_allow_html=True)
            st.caption("Taux de faux billets dans le lot")
            st.progress(min(int(taux), 100))

            st.divider()

            # ── Métriques ──
            st.markdown("<h2>Résumé du lot</h2>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total analysé",  str(nb_total))
            c2.metric("Authentiques",   str(nb_auth))
            c3.metric("Faux détectés",  str(nb_faux))
            c4.metric("Taux de faux",   f"{taux} %")

            if nb_faux > 0:
                indices = [i+1 for i, r in enumerate(res["resultats"]) if r["prediction"] == 0]
                st.error(f"🚨  ALERTE — {nb_faux} faux billet(s) — Billets n° : {indices}")
            else:
                st.success("✅  Tous les billets du lot sont authentiques.")

            st.divider()

            # ── Tableau résultats ──
            st.markdown("<h2>Résultats détaillés</h2>", unsafe_allow_html=True)
            rows = [{
                "Billet"        : f"Billet {i+1}",
                "Verdict"       : r["verdict"],
                "P_Authentique" : round(r["probabilite_auth"], 2),
                "P_Faux"        : round(r["probabilite_faux"], 2)
            } for i, r in enumerate(res["resultats"])]

            df_res = pd.DataFrame(rows)

            def colorier(row):
                if row["Verdict"] == "FAUX BILLET":
                    return [f"background-color:{RED_L};color:{RED};font-weight:600"] * len(row)
                return [f"background-color:{GREEN_L};color:{GREEN};font-weight:500"] * len(row)

            st.dataframe(df_res.style.apply(colorier, axis=1), use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️  Télécharger les résultats (CSV)",
                df_res.to_csv(index=False, sep=";"),
                "resultats_analyse.csv", "text/csv",
                use_container_width=True
            )

            st.divider()

            # ── Graphiques ──
            st.markdown("<h2>Visualisations</h2>", unsafe_allow_html=True)
            style_matplotlib()

            # Camembert + Histogramme
            col_g1, col_g2 = st.columns(2, gap="large")

            with col_g1:
                st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:12px;'>Répartition du lot</p>", unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(4.5, 4))
                vals  = [v for v in [nb_auth, nb_faux] if v > 0]
                etiq  = [e for e, v in zip([f"Authentiques\n({nb_auth})", f"Faux billets\n({nb_faux})"], [nb_auth, nb_faux]) if v > 0]
                cols  = [c for c, v in zip([GREEN, RED], [nb_auth, nb_faux]) if v > 0]
                wedges, texts, autotexts = ax1.pie(
                    vals, labels=etiq, colors=cols,
                    autopct="%1.1f%%", startangle=90,
                    textprops={"fontsize": 11, "color": TEXT},
                    wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
                    explode=[0.03] * len(vals)
                )
                for at in autotexts:
                    at.set_fontsize(12); at.set_fontweight("bold"); at.set_color("white")
                ax1.set_title("Distribution du lot", pad=14)
                plt.tight_layout()
                st.pyplot(fig1); plt.close(fig1)

            with col_g2:
                st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:12px;'>Niveaux de confiance</p>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(4.5, 4))
                probas = [r["probabilite_auth"] for r in res["resultats"]]
                preds  = [r["prediction"]       for r in res["resultats"]]
                p_a    = [p for p, c in zip(probas, preds) if c == 1]
                p_f    = [p for p, c in zip(probas, preds) if c == 0]
                if p_a:
                    ax2.hist(p_a, bins=15, color=GREEN, alpha=0.82, label=f"Authentiques ({len(p_a)})", edgecolor="white", linewidth=0.8)
                if p_f:
                    ax2.hist(p_f, bins=15, color=RED,   alpha=0.82, label=f"Faux billets ({len(p_f)})",  edgecolor="white", linewidth=0.8)
                ax2.set_xlabel("Probabilité d'authenticité (%)")
                ax2.set_ylabel("Nombre de billets")
                ax2.set_title("Confiance du modèle")
                ax2.legend()
                plt.tight_layout()
                st.pyplot(fig2); plt.close(fig2)

            st.caption("Si les barres sont concentrées à 95-100 %, le modèle est très certain de ses prédictions.")
            st.divider()

            # Barres groupées features
            st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:12px;'>Valeurs moyennes des 6 mesures par classe</p>", unsafe_allow_html=True)
            df_mes = df_lot[FEATURE_NAMES].copy().reset_index(drop=True)
            df_mes["prediction"] = [r["prediction"] for r in res["resultats"]]
            g_a = df_mes[df_mes["prediction"] == 1][FEATURE_NAMES]
            g_f = df_mes[df_mes["prediction"] == 0][FEATURE_NAMES]
            m_a = g_a.mean() if len(g_a) > 0 else pd.Series([0]*6, index=FEATURE_NAMES)
            m_f = g_f.mean() if len(g_f) > 0 else pd.Series([0]*6, index=FEATURE_NAMES)

            x = np.arange(len(FEATURE_NAMES)); w = 0.35
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            b_a = ax3.bar(x-w/2, m_a.values, w, label=f"Authentiques ({len(g_a)})", color=GREEN, alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3)
            b_f = ax3.bar(x+w/2, m_f.values, w, label=f"Faux billets ({len(g_f)})",  color=RED,   alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3)
            for bars, col in [(b_a, GREEN), (b_f, RED)]:
                for b in bars:
                    h = b.get_height()
                    if h > 0:
                        ax3.text(b.get_x()+b.get_width()/2, h+0.04, f"{h:.2f}",
                                 ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
            ax3.set_xticks(x); ax3.set_xticklabels(FEATURE_NAMES)
            ax3.set_ylabel("Valeur moyenne (mm)")
            ax3.set_title("Comparaison des mesures moyennes — Authentiques vs Faux billets")
            ax3.legend()
            plt.tight_layout()
            st.pyplot(fig3); plt.close(fig3)
            st.caption("Plus l'écart entre les barres est grand, plus cette mesure aide le modèle à distinguer les deux classes.")

            st.divider()

            # Scatter plot
            st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:4px;'>Nuage de points — length vs margin_low</p>", unsafe_allow_html=True)
            st.caption("Les deux mesures les plus discriminantes du dataset.")
            fig4, ax4 = plt.subplots(figsize=(10, 4.5))
            pts_a = df_mes[df_mes["prediction"] == 1]
            pts_f = df_mes[df_mes["prediction"] == 0]
            ax4.scatter(pts_a["length"], pts_a["margin_low"], c=GREEN, alpha=0.70, s=55, zorder=3,
                        label=f"Authentiques ({len(pts_a)})", edgecolors="white", linewidths=0.6)
            ax4.scatter(pts_f["length"], pts_f["margin_low"], c=RED,   alpha=0.85, s=75, zorder=4,
                        marker="X", label=f"Faux billets ({len(pts_f)})", edgecolors="white", linewidths=0.6)
            ax4.set_xlabel("length (mm)"); ax4.set_ylabel("margin_low (mm)")
            ax4.set_title("Séparation des classes : length vs margin_low")
            ax4.legend()
            plt.tight_layout()
            st.pyplot(fig4); plt.close(fig4)
            st.caption("Les faux billets (×) se situent généralement dans la zone length courte + margin_low élevée.")

        except Exception as e:
            st.error(f"Erreur : {e}")


# ─────────────────────────────────────────────
# PAGE 3 — HISTORIQUE
# ─────────────────────────────────────────────

def page_historique(role_page="user"):
    """
    role_page="user"  -> Mon historique   (analyses de l'utilisateur connecté)
    role_page="admin" -> Historique global (toutes les analyses, tous utilisateurs)
    """
    if role_page == "admin":
        st.markdown("<h1>Historique global</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Toutes les analyses effectuées par tous les utilisateurs.</p>", unsafe_allow_html=True)
    else:
        username = st.session_state.get("username", "")
        st.markdown("<h1>Mon historique</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Vos analyses de lots — compte : <strong>{username}</strong>.</p>", unsafe_allow_html=True)
    st.divider()

    h = charger_historique(role_page)

    if not h:
        st.info("Aucune analyse effectuée. Importez un fichier CSV dans 'Analyser un lot'.")
        return

    nb_s  = len(h)
    tot_b = sum(e["nb_total"] for e in h)
    tot_f = sum(e["nb_faux"]  for e in h)
    nb_al = sum(1 for e in h if e["statut"] == "ALERTE")

    st.markdown("<h2>Résumé global</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions",              str(nb_s))
    c2.metric("Billets analysés",      str(tot_b))
    c3.metric("Faux détectés",         str(tot_f))
    c4.metric("Sessions avec alerte",  str(nb_al))

    st.divider()

    if nb_s >= 2:
        st.markdown("<h2>Évolution du taux de faux</h2>", unsafe_allow_html=True)
        style_matplotlib()
        chron   = list(reversed(h))
        lbl_x   = [f"{e['date']}\n{e['heure']}" for e in chron]
        taux_y  = [e["taux_faux"] for e in chron]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(lbl_x, taux_y, marker="o", linewidth=2.5, color=BLUE,
                markerfacecolor=RED, markeredgecolor="white", markeredgewidth=1.5, markersize=9, zorder=3)
        ax.fill_between(lbl_x, taux_y, alpha=0.08, color=BLUE)
        ax.axhline(0, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_ylabel("Taux de faux (%)")
        ax.set_title("Taux de faux billets par session (ordre chronologique)")
        plt.xticks(rotation=30, ha="right", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.divider()

    st.markdown("<h2>Détail des sessions</h2>", unsafe_allow_html=True)
    rows = [{
        "Date": e["date"], "Heure": e["heure"],
        "Utilisateur": e.get("utilisateur", "—"),
        "Fichier": e["fichier"],
        "Total": e["nb_total"], "Authentiques": e["nb_auth"], "Faux": e["nb_faux"],
        "Taux faux": f"{e['taux_faux']} %", "Statut": e["statut"]
    } for e in h]
    df_h = pd.DataFrame(rows)

    def col_statut(row):
        if row["Statut"] == "ALERTE":
            return [f"background-color:{RED_L};color:{RED}"] * len(row)
        return [f"background-color:{GREEN_L};color:{GREEN}"] * len(row)

    st.dataframe(df_h.style.apply(col_statut, axis=1), use_container_width=True, hide_index=True)

    st.divider()

    ca, cb = st.columns(2)
    with ca:
        st.download_button("⬇️  Télécharger l'historique (CSV)",
                           df_h.to_csv(index=False, sep=";"),
                           "historique_analyses.csv", "text/csv",
                           use_container_width=True)
    with cb:
        if st.button("Effacer l'historique", use_container_width=True, type="secondary"):
            st.session_state["conf_supp"] = True

    if st.session_state.get("conf_supp", False):
        st.warning("Confirmer la suppression ? Cette action est irréversible.")
        co, cn = st.columns(2)
        with co:
            if st.button("Oui, effacer", type="primary", use_container_width=True):
                sauvegarder_historique([])
                st.session_state["conf_supp"] = False
                st.success("Historique effacé.")
                st.rerun()
        with cn:
            if st.button("Annuler", use_container_width=True):
                st.session_state["conf_supp"] = False
                st.rerun()


# ─────────────────────────────────────────────
# PAGE 4 — A PROPOS
# ─────────────────────────────────────────────

def page_a_propos():
    st.markdown("<h1>À propos</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Système de détection automatique de faux billets par Machine Learning.</p>", unsafe_allow_html=True)
    st.divider()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.info(f"**API FastAPI** — `api.py`\n\nReçoit les mesures, normalise et prédit avec la Régression Logistique.\n\n`http://localhost:8000`")
    with c2:
        st.success(f"**Interface Streamlit** — `app.py`\n\nAffiche les formulaires, envoie les données à l'API et présente les résultats.\n\n`http://localhost:8501`")

    st.divider()

    with st.expander("Modèle de Machine Learning"):
        a, b = st.columns(2)
        a.markdown(f"<p style='color:{TEXT};'>• <strong>Algorithme :</strong> Régression Logistique<br>• <strong>Dataset :</strong> 1 500 billets CHF<br>• <strong>Split :</strong> 80 % train / 20 % test</p>", unsafe_allow_html=True)
        b.markdown(f"<p style='color:{TEXT};'>• <strong>Accuracy :</strong> 99.33 %<br>• <strong>AUC-ROC :</strong> 0.9996<br>• <strong>Faux Positifs :</strong> 0 / 300</p>", unsafe_allow_html=True)

    with st.expander("Features analysées"):
        a, b = st.columns(2)
        a.markdown(f"<p style='color:{TEXT};'><code>diagonal</code> — Diagonale (mm)<br><code>height_left</code> — Hauteur gauche (mm)<br><code>height_right</code> — Hauteur droite (mm)</p>", unsafe_allow_html=True)
        b.markdown(f"<p style='color:{TEXT};'><code>margin_low</code> — Marge inférieure ⭐<br><code>margin_up</code> — Marge supérieure (mm)<br><code>length</code> — Longueur totale ⭐</p>", unsafe_allow_html=True)
        st.caption("⭐ Features les plus discriminantes")

    with st.expander("Graphiques disponibles dans Analyser un lot"):
        st.markdown(f"""
        <ol style='color:{TEXT};line-height:2;'>
            <li><strong>Camembert</strong> — proportion authentiques / faux</li>
            <li><strong>Histogramme</strong> — distribution des niveaux de confiance</li>
            <li><strong>Barres groupées</strong> — valeurs moyennes des 6 mesures par classe</li>
            <li><strong>Scatter plot</strong> — nuage de points length vs margin_low</li>
            <li><strong>Indicateur de risque</strong> — Sain / Suspect / Dangereux</li>
        </ol>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# AUTHENTIFICATION
# ─────────────────────────────────────────────

def verifier_identifiants(nom, mot_de_passe):
    """Vérifie les identifiants et retourne le rôle si OK, sinon None."""
    compte = COMPTES.get(nom)
    if compte and compte["mot_de_passe"] == mot_de_passe:
        return compte["role"]
    return None

def se_deconnecter():
    """Réinitialise la session."""
    st.session_state["connecte"]  = False
    st.session_state["role"]      = None
    st.session_state["username"]  = None


# ─────────────────────────────────────────────
# PAGE LOGIN
# ─────────────────────────────────────────────

def page_login():
    """Affiche l'écran de connexion centré."""

    # Centrage avec colonnes
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:

        # Logo + titre
        st.markdown(f"""
        <div style="text-align:center; padding: 48px 0 32px 0;">
            <div style="font-size:56px; margin-bottom:12px;">💵</div>
            <div style="font-size:26px; font-weight:800; color:{NAVY}; letter-spacing:-.02em;">FinPRO</div>
            <div style="font-size:14px; color:#334155; margin-top:4px;">Détection de Faux Billets</div>
        </div>
        """, unsafe_allow_html=True)

        # Carte de connexion
        
        with st.form("form_login"):
            nom = st.text_input(
                "Nom d'utilisateur",
                placeholder="utilisateur  ou  admin",
                label_visibility="visible"
            )
            mdp = st.text_input(
                "Mot de passe",
                type="password",
                placeholder="••••••••",
                label_visibility="visible"
            )
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            connexion = st.form_submit_button(
                "Se connecter",
                use_container_width=True,
                type="primary"
            )

        if connexion:
            if not nom or not mdp:
                st.error("Veuillez remplir tous les champs.")
                return

            role = verifier_identifiants(nom.strip(), mdp.strip())
            if role:
                st.session_state["connecte"] = True
                st.session_state["role"]     = role
                st.session_state["username"] = nom.strip()
                st.rerun()
            else:
                st.error("❌  Identifiants incorrects. Vérifiez votre nom d'utilisateur et mot de passe.")

        # Aide discrète
        st.markdown(f"""
        <div style="text-align:center; margin-top:20px;">
            <span style="font-size:12px; color:#334155;">
                Problème de connexion ? Contactez l'administrateur.
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 5 — TABLEAU DE BORD ADMIN
# ─────────────────────────────────────────────

def page_admin_dashboard():
    api_ok = verifier_api()

    st.markdown("<h1>🔑 Tableau de bord — Administrateur</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#334155;font-size:15px;margin-bottom:24px;'>Vue d'ensemble du système, statistiques globales et gestion.</p>", unsafe_allow_html=True)
    st.divider()

    # ── Section 1 : Statut du système ──────────────────────
    st.markdown("<h2>Statut du système</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Statut API
    with c1:
        if api_ok:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {BORDER};border-top:3px solid {GREEN};
                        border-radius:10px;padding:20px 22px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#334155;margin-bottom:8px;">API FastAPI</div>
                <div style="font-size:22px;font-weight:700;color:{GREEN};">✅ En ligne</div>
                <div style="font-size:12px;color:#334155;margin-top:4px;">localhost:8000</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#FFFFFF;border:1px solid {BORDER};border-top:3px solid {RED};
                        border-radius:10px;padding:20px 22px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
                <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#334155;margin-bottom:8px;">API FastAPI</div>
                <div style="font-size:22px;font-weight:700;color:{RED};">🔴 Hors ligne</div>
                <div style="font-size:12px;color:#334155;margin-top:4px;">uvicorn api:app --reload</div>
            </div>""", unsafe_allow_html=True)

    # Fichier historique admin
    with c2:
        taille_admin = os.path.getsize(HISTORIQUE_FILE) if os.path.exists(HISTORIQUE_FILE) else 0
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid {BORDER};border-top:3px solid {BLUE};
                    border-radius:10px;padding:20px 22px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
            <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#334155;margin-bottom:8px;">Historique global</div>
            <div style="font-size:22px;font-weight:700;color:{NAVY};">📁 {taille_admin} octets</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">{HISTORIQUE_FILE}</div>
        </div>""", unsafe_allow_html=True)

    # Fichier historique user
    with c3:
        taille_user = os.path.getsize(HISTORIQUE_USER_FILE) if os.path.exists(HISTORIQUE_USER_FILE) else 0
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid {BORDER};border-top:3px solid {ORANGE};
                    border-radius:10px;padding:20px 22px;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
            <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;color:#334155;margin-bottom:8px;">Historique utilisateur</div>
            <div style="font-size:22px;font-weight:700;color:{NAVY};">📁 {taille_user} octets</div>
            <div style="font-size:12px;color:#334155;margin-top:4px;">{HISTORIQUE_USER_FILE}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Section 2 : Statistiques globales ──────────────────
    st.markdown("<h2>Statistiques globales</h2>", unsafe_allow_html=True)

    h_admin = charger_historique("admin")

    if not h_admin:
        st.info("Aucune analyse enregistrée pour l'instant.")
    else:
        nb_sessions  = len(h_admin)
        tot_billets  = sum(e["nb_total"] for e in h_admin)
        tot_auth     = sum(e["nb_auth"]  for e in h_admin)
        tot_faux     = sum(e["nb_faux"]  for e in h_admin)
        taux_global  = round((tot_faux / tot_billets) * 100, 1) if tot_billets > 0 else 0
        nb_alertes   = sum(1 for e in h_admin if e["statut"] == "ALERTE")

        # KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Sessions totales",   str(nb_sessions))
        k2.metric("Billets analysés",   str(tot_billets))
        k3.metric("Authentiques",       str(tot_auth))
        k4.metric("Faux détectés",      str(tot_faux))
        k5.metric("Taux faux global",   f"{taux_global} %")

        st.divider()

        # Graphique évolution + camembert côte à côte
        style_matplotlib()
        col_g1, col_g2 = st.columns([2, 1], gap="large")

        with col_g1:
            st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>Évolution du taux de faux par session</p>", unsafe_allow_html=True)
            chron  = list(reversed(h_admin))
            lbl_x  = [f"{e['date']}\n{e['heure']}" for e in chron]
            taux_y = [e["taux_faux"] for e in chron]

            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(lbl_x, taux_y, marker="o", linewidth=2.5, color=BLUE,
                    markerfacecolor=RED, markeredgecolor="white",
                    markeredgewidth=1.5, markersize=9, zorder=3)
            ax.fill_between(lbl_x, taux_y, alpha=0.08, color=BLUE)
            ax.axhline(0, color=GREEN, linewidth=1.2, linestyle="--", alpha=0.6)
            ax.set_ylabel("Taux de faux (%)")
            ax.set_title("Taux de faux par session (ordre chronologique)")
            plt.xticks(rotation=30, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with col_g2:
            st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>Répartition globale</p>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(4, 3.2))
            vals  = [v for v in [tot_auth, tot_faux] if v > 0]
            etiq  = [e for e, v in zip([f"Auth.\n({tot_auth})", f"Faux\n({tot_faux})"], [tot_auth, tot_faux]) if v > 0]
            cols  = [c for c, v in zip([GREEN, RED], [tot_auth, tot_faux]) if v > 0]
            wedges, texts, autotexts = ax2.pie(
                vals, labels=etiq, colors=cols, autopct="%1.1f%%", startangle=90,
                textprops={"fontsize": 10, "color": TEXT},
                wedgeprops={"linewidth": 2.5, "edgecolor": "white"},
                explode=[0.03] * len(vals)
            )
            for at in autotexts:
                at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
            ax2.set_title("Tous billets confondus")
            plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

        st.divider()

        # Tableau par utilisateur
        st.markdown("<h2>Activité par utilisateur</h2>", unsafe_allow_html=True)
        utilisateurs = {}
        for e in h_admin:
            u = e.get("utilisateur", "inconnu")
            if u not in utilisateurs:
                utilisateurs[u] = {"sessions": 0, "billets": 0, "faux": 0, "alertes": 0}
            utilisateurs[u]["sessions"] += 1
            utilisateurs[u]["billets"]  += e["nb_total"]
            utilisateurs[u]["faux"]     += e["nb_faux"]
            utilisateurs[u]["alertes"]  += 1 if e["statut"] == "ALERTE" else 0

        rows_u = [{
            "Utilisateur"   : u,
            "Sessions"      : d["sessions"],
            "Billets"       : d["billets"],
            "Faux détectés" : d["faux"],
            "Taux faux"     : f"{round((d['faux']/d['billets'])*100,1) if d['billets']>0 else 0} %",
            "Alertes"       : d["alertes"]
        } for u, d in utilisateurs.items()]

        df_u = pd.DataFrame(rows_u)
        st.dataframe(df_u, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section 3 : Gestion du système ─────────────────────
    st.markdown("<h2>Gestion du système</h2>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>Réinitialiser les historiques</p>", unsafe_allow_html=True)

        choix_reset = st.selectbox(
            "Choisir le fichier à effacer",
            ["Historique utilisateur", "Historique global (tous)", "Les deux"],
            label_visibility="visible"
        )

        if st.button("🗑️  Effacer", type="secondary", use_container_width=True):
            st.session_state["confirm_reset"] = True

        if st.session_state.get("confirm_reset", False):
            st.warning(f"⚠️ Confirmer l'effacement : **{choix_reset}** ? Action irréversible.")
            r1, r2 = st.columns(2)
            with r1:
                if st.button("✅ Confirmer", type="primary", use_container_width=True):
                    if choix_reset in ["Historique utilisateur", "Les deux"]:
                        sauvegarder_historique([], "user")
                    if choix_reset in ["Historique global (tous)", "Les deux"]:
                        sauvegarder_historique([], "admin")
                    st.session_state["confirm_reset"] = False
                    st.success("Historique effacé avec succès.")
                    st.rerun()
            with r2:
                if st.button("Annuler", use_container_width=True):
                    st.session_state["confirm_reset"] = False
                    st.rerun()

    with col_b:
        st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-bottom:10px;'>Exporter les données</p>", unsafe_allow_html=True)

        h_export = charger_historique("admin")
        if h_export:
            rows_exp = [{
                "Date"        : e["date"],
                "Heure"       : e["heure"],
                "Utilisateur" : e.get("utilisateur", "—"),
                "Fichier"     : e["fichier"],
                "Total"       : e["nb_total"],
                "Authentiques": e["nb_auth"],
                "Faux"        : e["nb_faux"],
                "Taux faux"   : f"{e['taux_faux']} %",
                "Statut"      : e["statut"]
            } for e in h_export]
            df_exp = pd.DataFrame(rows_exp)
            st.download_button(
                "⬇️  Exporter historique global (CSV)",
                df_exp.to_csv(index=False, sep=";"),
                "export_admin_historique.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.caption("Aucune donnée à exporter.")

        # Infos modèle via API
        if api_ok:
            try:
                r = requests.get(f"{API_URL}/sante", timeout=3)
                info = r.json()
                st.markdown(f"<p style='font-weight:600;color:{NAVY};font-size:14px;margin-top:18px;margin-bottom:6px;'>Infos modèle (API)</p>", unsafe_allow_html=True)
                st.json(info)
            except:
                pass


def main():
    # ── Initialisation de la session ──
    if "connecte" not in st.session_state:
        st.session_state["connecte"] = False
    if "role"      not in st.session_state:
        st.session_state["role"]     = None
    if "username"  not in st.session_state:
        st.session_state["username"] = None

    # ── Si non connecté → écran de login ──
    if not st.session_state["connecte"]:
        page_login()
        return

    # ── Si connecté → interface normale ──
    page, api_ok = afficher_sidebar()

    if   page == "🔍  Analyser un billet":    page_analyser_billet(api_ok)
    elif page == "📦  Analyser un lot":        page_analyser_lot(api_ok)
    elif page == "📋  Mon historique":         page_historique(role_page="user")
    elif page == "🗂️  Historique global":      page_historique(role_page="admin")
    elif page == "📊  Tableau de bord Admin":  page_admin_dashboard()
    elif page == "ℹ️  A propos":               page_a_propos()


if __name__ == "__main__":
    main()
