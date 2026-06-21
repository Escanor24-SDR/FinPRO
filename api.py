"""
api.py
------
API FastAPI pour la detection de faux billets.
Lancement : uvicorn api:app --reload
"""

import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ──────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────

FEATURE_NAMES = [
    "diagonal",
    "height_left",
    "height_right",
    "margin_low",
    "margin_up",
    "length"
]

MODELS_DIR = os.path.join(os.getcwd(), "models")


# ──────────────────────────────────────────────
# CREATION DE L'APPLICATION FASTAPI
# ──────────────────────────────────────────────

app = FastAPI(
    title="API Detection de Faux Billets",
    description=(
        "API de detection de faux billets par Machine Learning. "
        "Envoyez les mesures physiques d'un billet et obtenez "
        "un verdict : AUTHENTIQUE ou FAUX BILLET."
    ),
    version="1.0.0"
)


# ──────────────────────────────────────────────
# CHARGEMENT DU MODELE ET DU SCALER
# ──────────────────────────────────────────────

def charger_modele_et_scaler():
    fichiers_pkl = [
        f for f in os.listdir(MODELS_DIR)
        if f.endswith(".pkl") and f != "scaler.pkl"
    ]
    chemin_modele = os.path.join(MODELS_DIR, fichiers_pkl[0])
    chemin_scaler = os.path.join(MODELS_DIR, "scaler.pkl")

    modele = joblib.load(chemin_modele)
    scaler = joblib.load(chemin_scaler)

    print(f"[OK] Modele charge : {fichiers_pkl[0]}")
    print(f"[OK] Scaler charge : scaler.pkl")

    return modele, scaler


modele, scaler = charger_modele_et_scaler()


# ──────────────────────────────────────────────
# SCHEMAS PYDANTIC
# ──────────────────────────────────────────────

class BilletEntree(BaseModel):
    """
    Schema des donnees d'entree pour un seul billet.
    Pydantic valide automatiquement que chaque valeur
    est bien un nombre decimal (float).
    Field() permet d'ajouter des contraintes et descriptions.
    """
    diagonal     : float = Field(..., description="Longueur de la diagonale en mm",       example=171.81)
    height_left  : float = Field(..., description="Hauteur cote gauche en mm",             example=104.86)
    height_right : float = Field(..., description="Hauteur cote droit en mm",              example=104.95)
    margin_low   : float = Field(..., description="Marge inferieure en mm",                example=4.52)
    margin_up    : float = Field(..., description="Marge superieure en mm",                example=2.89)
    length       : float = Field(..., description="Longueur totale en mm",                 example=112.83)

    class Config:
        json_schema_extra = {
            "example": {
                "diagonal"     : 171.81,
                "height_left"  : 104.86,
                "height_right" : 104.95,
                "margin_low"   : 4.52,
                "margin_up"    : 2.89,
                "length"       : 112.83
            }
        }


class BilletSortie(BaseModel):
    """
    Schema des donnees de sortie — ce que l'API renvoie
    apres avoir analyse un billet.
    """
    prediction      : int    # 0 = faux, 1 = authentique
    verdict         : str    # "AUTHENTIQUE" ou "FAUX BILLET"
    probabilite_auth: float  # probabilite d'etre authentique (0 a 100)
    probabilite_faux: float  # probabilite d'etre faux (0 a 100)
    mesures         : dict   # les mesures originales envoyees


class LotEntree(BaseModel):
    """
    Schema pour analyser plusieurs billets en meme temps.
    On envoie une liste de BilletEntree.
    """
    billets: List[BilletEntree]


class LotSortie(BaseModel):
    """
    Schema de sortie pour un lot de billets.
    """
    total_analyse   : int
    nb_authentiques : int
    nb_faux         : int
    resultats       : List[BilletSortie]


# ──────────────────────────────────────────────
# ROUTE 1 : ACCUEIL
# ──────────────────────────────────────────────

@app.get("/")
def accueil():
    """Route d'accueil — verifie que l'API fonctionne."""
    return {
        "message" : "API Detection de Faux Billets",
        "version" : "1.0.0",
        "status"  : "en ligne",
        "routes"  : ["/", "/sante", "/predire", "/predire-lot"],
        "doc"     : "http://localhost:8000/docs"
    }


# ──────────────────────────────────────────────
# ROUTE 2 : SANTE
# ──────────────────────────────────────────────

@app.get("/sante")
def sante():
    """Verifie que le modele est bien charge."""
    return {
        "status"        : "ok",
        "modele_charge" : modele is not None,
        "scaler_charge" : scaler is not None,
        "features"      : FEATURE_NAMES
    }


# ──────────────────────────────────────────────
# ROUTE 3 : PREDIRE — UN SEUL BILLET
# ──────────────────────────────────────────────

@app.post("/predire", response_model=BilletSortie)
def predire(billet: BilletEntree):
    """
    Analyse un seul billet et retourne le verdict.

    - Recoit les 6 mesures physiques du billet
    - Normalise les donnees avec le scaler
    - Predit avec le modele ML
    - Retourne : verdict + probabilites + mesures

    Exemple d'utilisation :
        POST http://localhost:8000/predire
        Body : {"diagonal": 171.81, "height_left": 104.86, ...}
    """

    try:
        # ── Etape 1 : Mise en forme ───────────────────────────
        # On convertit le schema Pydantic en dictionnaire
        # puis en DataFrame pandas avec les colonnes dans le bon ordre
        mesures = {
            "diagonal"     : billet.diagonal,
            "height_left"  : billet.height_left,
            "height_right" : billet.height_right,
            "margin_low"   : billet.margin_low,
            "margin_up"    : billet.margin_up,
            "length"       : billet.length
        }

        X = pd.DataFrame([mesures])[FEATURE_NAMES]

        # ── Etape 2 : Normalisation ───────────────────────────
        # On applique le meme scaler que celui utilise
        # pendant l'entrainement
        X_scaled = scaler.transform(X)

        # ── Etape 3 : Prediction ──────────────────────────────
        prediction   = int(modele.predict(X_scaled)[0])
        probabilites = modele.predict_proba(X_scaled)[0]

        proba_auth = round(float(probabilites[1]) * 100, 2)
        proba_faux = round(float(probabilites[0]) * 100, 2)

        # ── Etape 4 : Construction de la reponse ──────────────
        verdict = "AUTHENTIQUE" if prediction == 1 else "FAUX BILLET"

        return BilletSortie(
            prediction       = prediction,
            verdict          = verdict,
            probabilite_auth = proba_auth,
            probabilite_faux = proba_faux,
            mesures          = mesures
        )

    except Exception as e:
        # Si une erreur survient, on renvoie une erreur HTTP 500
        # avec un message explicatif
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prediction : {str(e)}"
        )


# ──────────────────────────────────────────────
# ROUTE 4 : PREDIRE — UN LOT DE BILLETS
# ──────────────────────────────────────────────

@app.post("/predire-lot", response_model=LotSortie)
def predire_lot(lot: LotEntree):
    """
    Analyse plusieurs billets en une seule requete.

    - Recoit une liste de billets
    - Les analyse tous en une seule operation
    - Retourne un resume + le detail de chaque billet

    Exemple d'utilisation :
        POST http://localhost:8000/predire-lot
        Body : {"billets": [{"diagonal": 171.81, ...}, {...}]}
    """

    try:
        # ── Mise en forme du lot ──────────────────────────────
        # On construit un DataFrame avec autant de lignes
        # qu'il y a de billets dans la liste
        liste_mesures = [
            {
                "diagonal"     : b.diagonal,
                "height_left"  : b.height_left,
                "height_right" : b.height_right,
                "margin_low"   : b.margin_low,
                "margin_up"    : b.margin_up,
                "length"       : b.length
            }
            for b in lot.billets
        ]

        X_lot        = pd.DataFrame(liste_mesures)[FEATURE_NAMES]
        X_lot_scaled = scaler.transform(X_lot)

        # ── Predictions sur tout le lot ───────────────────────
        predictions  = modele.predict(X_lot_scaled)
        probabilites = modele.predict_proba(X_lot_scaled)

        # ── Construction des resultats ────────────────────────
        resultats = []
        for i, (pred, proba, mesures) in enumerate(
            zip(predictions, probabilites, liste_mesures)
        ):
            resultats.append(BilletSortie(
                prediction       = int(pred),
                verdict          = "AUTHENTIQUE" if pred == 1 else "FAUX BILLET",
                probabilite_auth = round(float(proba[1]) * 100, 2),
                probabilite_faux = round(float(proba[0]) * 100, 2),
                mesures          = mesures
            ))

        nb_authentiques = int((predictions == 1).sum())
        nb_faux         = int((predictions == 0).sum())

        return LotSortie(
            total_analyse   = len(predictions),
            nb_authentiques = nb_authentiques,
            nb_faux         = nb_faux,
            resultats       = resultats
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse du lot : {str(e)}"
        )
