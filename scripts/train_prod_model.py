import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "age_souscription",
    "duree",
    "capital_emprunte",
    "taux_interet_annuel",
    "taux_technique_annuel",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dataset_40k_lignes.csv")
    parser.add_argument("--out", default="models/prod/premium_model_prod.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    X = df[FEATURES].to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "csv": args.csv,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "metrics": metrics}, args.out)

    print("✅ Modèle PROD sauvegardé:", args.out)
    print("📊 Metrics:", metrics)

if __name__ == "__main__":
    main()
