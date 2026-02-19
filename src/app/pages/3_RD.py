import os
import sys
import time
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.ui import metric_card
from src.ml.rd import (
    list_available_datasets,
    fit_and_eval,
    cv_mae,
    learning_curve_multi_csv,
    save_model_pack,
)

st.header("Recherche et développement")
st.caption("Comparaison de modèles, métriques, validation croisée et learning curves. Aucun impact sur la production.")

DATA_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "data")
RD_DIR = os.path.join(PROJECT_ROOT, "models", "rd")

st.markdown(
    """
<div class="card">
  <div class="card-title">Règle de séparation</div>
  <div>Les modèles R&D sont enregistrés dans <code>models/rd/</code>. Le modèle de production reste dans <code>models/prod/</code>.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Dataset")
    data_dir = st.text_input("Dossier datasets", value=DATA_DIR_DEFAULT)
    datasets = list_available_datasets(data_dir)
    if datasets:
        dataset_name = st.selectbox("Choisir un dataset", datasets, index=min(2, len(datasets)-1))
    else:
        dataset_name = st.text_input("Nom dataset", value="dataset_40k_lignes.csv")

    csv_path = os.path.join(data_dir, dataset_name)

    st.divider()
    st.subheader("Modèle")
    model_name = st.selectbox("Type de modèle", ["LinearRegression", "RandomForest", "HistGradientBoosting"])

    st.divider()
    st.subheader("Paramètres")
    test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    st.markdown("**Paramètres Random Forest**")
    n_estimators = st.slider("Nombre d’arbres", 100, 2000, 500, 50)
    min_samples_leaf = st.slider("Taille minimale des feuilles", 1, 10, 2, 1)

    st.markdown("**Paramètres HistGradientBoosting**")
    learning_rate = st.slider("Learning rate", 0.005, 0.2, 0.05, 0.005)
    max_depth = st.slider("Profondeur max", 2, 12, 6, 1)
    max_iter = st.slider("Nombre d’itérations", 100, 2000, 400, 50)

    st.divider()
    st.subheader("Validation")
    use_cv = st.checkbox("Activer validation croisée (MAE)", value=False)
    k_folds = st.selectbox("Nombre de folds", [3, 4, 5], index=0, disabled=(not use_cv))
    do_learning = st.checkbox("Calculer learning curves", value=True)

    run = st.button("Lancer l’analyse")

params = {
    "n_estimators": int(n_estimators),
    "min_samples_leaf": int(min_samples_leaf),
    "learning_rate": float(learning_rate),
    "max_depth": int(max_depth),
    "max_iter": int(max_iter),
    "random_state": int(random_state),
}

if run:
    try:
        model, metrics, preview, feat_imp = fit_and_eval(
            csv_path=csv_path,
            model_name=model_name,
            params=params,
            test_size=float(test_size),
            random_state=int(random_state),
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("MAE test", f"{metrics['test_mae']:.6f}")
        with c2:
            metric_card("RMSE test", f"{metrics['test_rmse']:.6f}")
        with c3:
            metric_card("R² test", f"{metrics['test_r2']:.6f}")

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("Détails des métriques")
        st.write(metrics)

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("Aperçu des prédictions")
        st.dataframe(preview, use_container_width=True, height=320)

        if feat_imp is not None:
            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader("Importance des variables")
            st.dataframe(feat_imp, use_container_width=True, height=220)

        if use_cv:
            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader("Validation croisée")
            cvres = cv_mae(csv_path=csv_path, model_name=model_name, params=params, k_folds=int(k_folds), random_state=int(random_state))
            cc1, cc2 = st.columns(2)
            with cc1:
                metric_card("MAE moyenne", f"{cvres['mae_mean']:.6f}")
            with cc2:
                metric_card("Écart-type", f"{cvres['mae_std']:.6f}")

        if do_learning:
            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader("Learning curves")
            df_lc, fig_mae, fig_r2 = learning_curve_multi_csv(
                data_dir=data_dir,
                model_name=model_name,
                params=params,
                test_size=float(test_size),
                random_state=int(random_state),
            )
            st.dataframe(df_lc, use_container_width=True, height=260)
            st.pyplot(fig_mae, clear_figure=True)
            st.pyplot(fig_r2, clear_figure=True)

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("Sauvegarde du modèle R&D")
        os.makedirs(RD_DIR, exist_ok=True)
        out_name = f"{model_name}_{dataset_name.replace('.csv','')}_{int(time.time())}.joblib"
        out_path = os.path.join(RD_DIR, out_name)

        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown(f"<div class='card'><div class='card-title'>Chemin</div><div>{out_path}</div></div>", unsafe_allow_html=True)
        with colB:
            if st.button("Sauvegarder"):
                save_model_pack(model, model_name, params, metrics, out_path)
                st.success("Modèle sauvegardé dans R&D.")

    except Exception as e:
        st.error(f"Erreur: {e}")
