# FinPRO — Détection de Faux Billets 💵

Projet de fin d'études — Licence Informatique de Gestion  
UCAO/ISG Saint Michel, Dakar — 2025-2026  
Réalisé par : Isidore & Dame FALL  
Superviseur : M. Aidara

## Description
Application web de détection de faux billets par Machine Learning
(Régression Logistique — 99.33% de précision).

## Technologies
- **Frontend** : Streamlit
- **Backend** : FastAPI
- **Modèle ML** : Scikit-learn (Régression Logistique)

## Structure du projet
FinPRO/

├── app.py          → Interface Streamlit

├── api.py          → Backend FastAPI

├── requirements.txt

├── models/         → Modèle ML et Scaler

├── data/           → Datasets CSV et historiques

└── notebooks/      → Notebooks d'entraînement
## Lancement
```bash
pip install -r requirements.txt
uvicorn api:app --reload
streamlit run app.py
```
