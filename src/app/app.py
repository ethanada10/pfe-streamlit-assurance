import os
import sys
import streamlit as st

# Make "src" importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Global CSS
CSS_PATH = os.path.join(PROJECT_ROOT, "assets", "styles.css")
if os.path.exists(CSS_PATH):
    st.markdown(f"<style>{open(CSS_PATH, 'r', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Assurance emprunteur — Actuariat & Machine Learning",
    layout="wide",
)

st.title("Assurance emprunteur décès — Calcul actuariel et prédiction ML")

st.markdown(
    """
<div class="card">
  <div class="card-title">Objectif</div>
  <div>Comparer deux moteurs de tarification de la prime mensuelle d’assurance emprunteur décès :</div>
  <ul>
    <li><b>Moteur Actuariat</b> : calcul basé sur table de mortalité + actualisation + CRD.</li>
    <li><b>Moteur IA (Production)</b> : prédiction via un modèle ML figé (Random Forest), entraîné sur les datasets générés.</li>
  </ul>
</div>

<div class="card">
  <div class="card-title">Organisation de l’application</div>
  <ul>
    <li><b>Moteur Actuariat</b> : saisie des paramètres, prime mensuelle actuarielle, tableau d’amortissement.</li>
    <li><b>Moteur IA</b> : saisie identique, prime prédite par le modèle de production, tableau d’amortissement.</li>
    <li><b>R&D</b> : expérimentation (comparaison de modèles, métriques, validation croisée, learning curves). Aucun impact sur le modèle de production.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
<div class="card">
  <div class="card-title">Production</div>
  <div>Le modèle utilisé par le moteur IA est stocké dans <code>models/prod/</code> et ne change pas via la page R&D.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="card">
  <div class="card-title">Traçabilité</div>
  <div>Les paramètres d’entrée sont identiques entre les deux moteurs pour rendre la comparaison cohérente.</div>
</div>
""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
<div class="card">
  <div class="card-title">Export</div>
  <div>Les tableaux d’amortissement peuvent être exportés au format CSV directement depuis l’interface.</div>
</div>
""",
        unsafe_allow_html=True,
    )
