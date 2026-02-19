import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.layout import inject_css, sidebar_branding, header
from src.app.ui import metric_card
from src.actuarial.pricing import creer_tableau_amortissement
from src.ml.predict import load_model, predict_new_loan_df

def eur(x: float) -> str:
    return f"{x:,.2f} €".replace(",", " ").replace(".", ",")

inject_css(PROJECT_ROOT)
sidebar_branding(
    title="Moteur",
    subtitle="IA production",
    badges=["Modèle figé", "Random Forest"],
)

header("Moteur IA", "Prédiction de la prime mensuelle via un modèle de production figé.")

PROD_MODEL = os.path.join(PROJECT_ROOT, "models", "prod", "premium_model_prod.joblib")

st.markdown(
    f"""
<div class="card">
  <div class="card-title">Modèle de production</div>
  <div>Chemin: <code>{PROD_MODEL}</code></div>
  <div style="opacity:0.75; margin-top:6px;">Ce modèle n’est jamais modifié par la page R&D.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Paramètres du prêt")
    capital = st.number_input("Capital emprunté", min_value=1000.0, value=200000.0, step=1000.0)
    duree = st.number_input("Durée en années", min_value=1, max_value=40, value=20, step=1)
    taux_interet_annuel = st.number_input("Taux du crédit annuel", min_value=0.0, value=0.035, step=0.001, format="%.4f")

    st.divider()
    st.subheader("Paramètres assurantiels")
    age_souscription = st.number_input("Âge à la souscription", min_value=18, max_value=90, value=35, step=1)
    taux_technique_annuel = st.number_input("Taux technique annuel", min_value=0.0, value=0.02, step=0.001, format="%.4f")

    run = st.button("Prédire")

if run:
    try:
        model_pack = load_model(PROD_MODEL)
        prime_ml = float(
            predict_new_loan_df(
                model_pack=model_pack,
                age_souscription=int(age_souscription),
                duree=int(duree),
                capital_emprunte=float(capital),
                taux_interet_annuel=float(taux_interet_annuel),
                taux_technique_annuel=float(taux_technique_annuel),
            )
        )

        amort = creer_tableau_amortissement(float(capital), int(duree), float(taux_interet_annuel))
        amort["prime_mensuelle"] = prime_ml
        amort["mensualite_totale"] = amort["mensualites"] + amort["prime_mensuelle"]

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Prime mensuelle prédite", eur(prime_ml))
        with c2:
            metric_card("Mensualité du crédit", eur(float(amort.loc[1, "mensualites"])), "Échéance 1")
        with c3:
            metric_card("Mensualité totale", eur(float(amort.loc[1, "mensualite_totale"])), "Échéance 1")

        if model_pack.get("metrics"):
            st.markdown('<div class="section"></div>', unsafe_allow_html=True)
            st.subheader("Indicateurs du modèle de production")
            m = model_pack["metrics"]
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                metric_card("MAE test", f"{m.get('test_mae', 0):.6f}")
            with cc2:
                metric_card("RMSE test", f"{m.get('test_rmse', 0):.6f}")
            with cc3:
                metric_card("R² test", f"{m.get('test_r2', 0):.6f}")

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("Tableau d’amortissement")
        st.dataframe(amort, use_container_width=True, height=520)

        st.download_button(
            "Télécharger le tableau (CSV)",
            data=amort.to_csv(index=False).encode("utf-8"),
            file_name="amortissement_ia_prod.csv",
            mime="text/csv",
        )

    except FileNotFoundError:
        st.error("Modèle de production introuvable. Lance : python3 scripts/train_prod_model.py --csv data/dataset_40k_lignes.csv")
    except Exception as e:
        st.error(f"Erreur: {e}")
