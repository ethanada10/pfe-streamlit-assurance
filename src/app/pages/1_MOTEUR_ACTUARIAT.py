import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.layout import inject_css, sidebar_branding, header
from src.app.ui import metric_card
from src.actuarial.mortality_table import load_mortality_table_with_abattement
from src.actuarial.pricing import compute_mensual_premium_direct_with_schedule

def eur(x: float) -> str:
    return f"{x:,.2f} €".replace(",", " ").replace(".", ",")

inject_css(PROJECT_ROOT)
sidebar_branding(
    title="Moteur",
    subtitle="Actuariat",
    badges=["Table de mortalité", "Prime mensuelle sur CRD"],
)

header("Moteur Actuariat", "Calcul actuariel de la prime mensuelle sur CRD, mortalité et actualisation.")

DEFAULT_TABLE = os.path.join(PROJECT_ROOT, "assets", "TH00-02.xlsx")

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
    st.subheader("Table de mortalité")
    table_path = st.text_input("Chemin du fichier", value=DEFAULT_TABLE)
    taux_abattement = st.slider("Abattement sur qx", min_value=0.0, max_value=0.9, value=0.35, step=0.05)

    run = st.button("Lancer le calcul")

if run:
    try:
        table = load_mortality_table_with_abattement(table_path=table_path, taux_abattement=float(taux_abattement))

        prime_mensuelle, amort = compute_mensual_premium_direct_with_schedule(
            table_mortalite=table,
            capital=float(capital),
            duree=int(duree),
            taux_emprunt_annuel=float(taux_interet_annuel),
            age=int(age_souscription),
            taux_technique_annuel=float(taux_technique_annuel),
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Prime mensuelle", eur(prime_mensuelle))
        with c2:
            metric_card("Mensualité du crédit", eur(float(amort.loc[1, "mensualites"])), "Échéance 1")
        with c3:
            metric_card("Mensualité totale", eur(float(amort.loc[1, "mensualite_totale"])), "Échéance 1")

        st.markdown('<div class="section"></div>', unsafe_allow_html=True)
        st.subheader("Tableau d’amortissement")
        st.dataframe(amort, use_container_width=True, height=520)

        st.download_button(
            "Télécharger le tableau (CSV)",
            data=amort.to_csv(index=False).encode("utf-8"),
            file_name="amortissement_actuariat.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Erreur: {e}")
