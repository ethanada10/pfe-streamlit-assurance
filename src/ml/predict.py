from __future__ import annotations
import os
import joblib
import pandas as pd

DEFAULT_FEATURES = [
    "age_souscription",
    "duree",
    "capital_emprunte",
    "taux_interet_annuel",
    "taux_technique_annuel",
]

def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    obj = joblib.load(model_path)

    # Si c'est un pack (dict): {"model":..., "features":..., "metrics":...}
    if isinstance(obj, dict) and "model" in obj:
        if "features" not in obj or obj["features"] is None:
            obj["features"] = DEFAULT_FEATURES
        return obj

    # Sinon: modèle brut
    return {"model": obj, "features": DEFAULT_FEATURES, "metrics": None}

def predict_new_loan_df(
    *,
    model_pack,
    age_souscription: int,
    duree: int,
    capital_emprunte: float,
    taux_interet_annuel: float,
    taux_technique_annuel: float,
) -> float:
    model = model_pack["model"]
    features = model_pack.get("features") or DEFAULT_FEATURES

    X_new = pd.DataFrame([{
        "age_souscription": age_souscription,
        "duree": duree,
        "capital_emprunte": capital_emprunte,
        "taux_interet_annuel": taux_interet_annuel,
        "taux_technique_annuel": taux_technique_annuel,
    }])[features]

    return float(model.predict(X_new.to_numpy())[0])
