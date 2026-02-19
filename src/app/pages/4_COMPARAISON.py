import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.layout import inject_css, sidebar_branding, header
from src.app.ui import metric_card

from src.actuarial.mortality_table import load_mortality_table_with_abattement
from src.actuarial.pricing import compute_mensual_premium_direct_with_schedule, creer_tableau_amortissement
from src.ml.predict import load_model, predict_new_loan_df

def eur(x: float) -> str:
    return f"{x:,.2f} €".replace(",", " ").replace(".", ",")

inject_css(PROJECT_ROOT)
sidebar_branding(
    title="Application",
    subtitle="Assurance emprunteur",
    badges=["Actuariat vs IA", "Modèle IA: production figée"],
)

header("Comparaison des moteurs", "Comparaison directe sur des inputs identiques : prime actuarielle vs prime ML.")

DEFAULT_TABLE = os.path.join(PROJECT_ROOT, "assets", "TH00-02.xlsx")
PROD_MODEL = os.path.join(PROJECT_ROOT, "models", "prod", "premium_model_prod.joblib")

with st.sidebar:
    st.subheader("Paramètres du prêt")
    capital = st.number_input("Capital emprunté", min_value=1000.0, value=200000.0, step=1000.0)
    duree = st.number_input("Durée en années", min_value=1, max_value=40, value=20, step=1)
    taux_interet_annuel = st.number_input("Taux du crédit annuel", min_value=0.0, value=0.035, step=0.001, format="%.4f")

    st.divider()
    st.subheader("Paramètres assurantiels")
    age_souscription = st.number_input("Âge à la souscription", min_value=18, max_value=90, value=35, step=1)
    taux_technique_annuel = st.number_input("Taux technique annuel", min_value=0.0, value=0.02, step=0.001, format="%.4f")

    st.divider()
    st.subheader("Actuariat")
    table_path = st.text_input("Table de mortalité", value=DEFAULT_TABLE)
    taux_abattement = st.slider("Abattement sur qx", min_value=0.0, max_value=0.9, value=0.35, step=0.05)

    run = st.button("Comparer")

if run:
    try:
        # --- Actuariat ---
        table = load_mortality_table_with_abattement(table_path=table_path, taux_abattement=float(taux_abattement))
        prime_actu, amort_actu = compute_mensual_premium_direct_with_schedule(
            table_mortalite=table,
            capital=float(capital),
            duree=int(duree),
            taux_emprunt_annuel=float(taux_interet_annuel),
            age=int(age_souscription),
            taux_technique_annuel=float(taux_technique_annuel),
        )

        # --- IA (PROD) ---
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
        amort_ml = creer_tableau_amortissement(float(capital), int(duree), float(taux_interet_annuel))
        amort_ml["prime_mensuelle"] = prime_ml
        amort_ml["mensualite_totale"] = amort_ml["mensualites"] + amort_ml["prime_mensuelle"]

        # --- Cards summary ---
        delta = prime_ml - prime_actu
        delta_pct = (delta / prime_actu * 100) if prime_actu != 0 else 0.0

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Prime mensuelle actuarielle", eur(prime_actu))
        with c2:
            metric_card("Prime mensuelle IA production", eur(prime_ml))
        with c3:
            metric_card("Écart IA - Actuariat", f"{eur(delta)}", f"{delta_pct:.2f}%")

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["Amortissement actuariat", "Amortissement IA production"])

        with t1:
            st.dataframe(amort_actu, use_container_width=True, height=520)
            st.download_button(
                "Télécharger CSV actuariat",
                data=amort_actu.to_csv(index=False).encode("utf-8"),
                file_name="amortissement_actuariat.csv",
                mime="text/csv",
            )
        with t2:
            st.dataframe(amort_ml, use_container_width=True, height=520)
            st.download_button(
                "Télécharger CSV IA",
                data=amort_ml.to_csv(index=False).encode("utf-8"),
                file_name="amortissement_ia_prod.csv",
                mime="text/csv",
            )

    except FileNotFoundError:
        st.error("Modèle IA production introuvable. Lance : python3 scripts/train_prod_model.py --csv data/dataset_40k_lignes.csv")
    except Exception as e:
        st.error(f"Erreur: {e}")
