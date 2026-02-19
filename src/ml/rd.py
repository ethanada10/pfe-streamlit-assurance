from __future__ import annotations
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

FEATURES = [
    "age_souscription",
    "duree",
    "capital_emprunte",
    "taux_interet_annuel",
    "taux_technique_annuel",
]

MODELS = {
    "LinearRegression": lambda params: LinearRegression(),
    "RandomForest": lambda params: RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 500)),
        max_depth=None,
        min_samples_leaf=int(params.get("min_samples_leaf", 2)),
        random_state=int(params.get("random_state", 42)),
        n_jobs=-1,
    ),
    "HistGradientBoosting": lambda params: HistGradientBoostingRegressor(
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 6)),
        max_iter=int(params.get("max_iter", 400)),
        random_state=int(params.get("random_state", 42)),
    ),
}

def list_available_datasets(data_dir: str) -> list[str]:
    if not os.path.isdir(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f.startswith("dataset_")]
    return sorted(files, key=_dataset_size_key)

def _dataset_size_key(name: str) -> int:
    # dataset_20k_lignes.csv -> 20000
    m = re.search(r"dataset_(\d+)(k)?", name)
    if not m:
        return 10**12
    n = int(m.group(1))
    if m.group(2) == "k":
        n *= 1000
    return n

def _load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + ["target"] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {os.path.basename(csv_path)}: {missing}")
    return df

def fit_and_eval(
    *,
    csv_path: str,
    model_name: str,
    params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = _load_dataset(csv_path)
    X = df[FEATURES].to_numpy()
    y = df["target"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state)
    )

    model = MODELS[model_name](params)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    metrics = {
        "model": model_name,
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset": os.path.basename(csv_path),
    }

    # preview
    preview = pd.DataFrame(X_test[:200], columns=FEATURES)
    preview["y_true"] = y_test[:200]
    preview["y_pred"] = pred_test[:200]
    preview["abs_error"] = (preview["y_true"] - preview["y_pred"]).abs()

    # importances for RF only
    feat_imp = None
    if model_name == "RandomForest":
        feat_imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    return model, metrics, preview, feat_imp

def cv_mae(
    *,
    csv_path: str,
    model_name: str,
    params: dict,
    k_folds: int = 5,
    random_state: int = 42,
):
    df = _load_dataset(csv_path)
    X = df[FEATURES].to_numpy()
    y = df["target"].to_numpy()

    model = MODELS[model_name](params)
    kf = KFold(n_splits=int(k_folds), shuffle=True, random_state=int(random_state))
    scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
    return {
        "k_folds": int(k_folds),
        "mae_mean": float((-scores).mean()),
        "mae_std": float((-scores).std()),
    }

def learning_curve_multi_csv(
    *,
    data_dir: str,
    model_name: str,
    params: dict,
    test_size: float = 0.2,
    random_state: int = 42,
):
    files = list_available_datasets(data_dir)
    if not files:
        raise ValueError("Aucun dataset trouvé pour la learning curve.")

    rows = []
    for f in files:
        csv_path = os.path.join(data_dir, f)
        _, metrics, _, _ = fit_and_eval(
            csv_path=csv_path,
            model_name=model_name,
            params=params,
            test_size=test_size,
            random_state=random_state,
        )
        rows.append({
            "dataset": f,
            "size": _dataset_size_key(f),
            "test_mae": metrics["test_mae"],
            "test_r2": metrics["test_r2"],
        })

    df_res = pd.DataFrame(rows).sort_values("size")

    fig_mae = plt.figure(figsize=(10, 6))
    plt.plot(df_res["size"], df_res["test_mae"], marker="o")
    plt.xscale("log")
    plt.xlabel("Nombre d'observations (log)")
    plt.ylabel("MAE test")
    plt.title(f"Learning curve — {model_name} — MAE test")
    plt.grid(True, alpha=0.3)

    fig_r2 = plt.figure(figsize=(10, 6))
    plt.plot(df_res["size"], df_res["test_r2"], marker="o")
    plt.xscale("log")
    plt.xlabel("Nombre d'observations (log)")
    plt.ylabel("R² test")
    plt.title(f"Learning curve — {model_name} — R² test")
    plt.grid(True, alpha=0.3)

    return df_res, fig_mae, fig_r2

def save_model_pack(model, model_name: str, params: dict, metrics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pack = {
        "model": model,
        "features": FEATURES,
        "model_name": model_name,
        "params": params,
        "metrics": metrics,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(pack, out_path)
