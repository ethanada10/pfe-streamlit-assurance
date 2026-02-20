# PFE — Assurance Emprunteur  
## Actuariat & Intelligence Artificielle — Application Streamlit

Ce projet est une application Streamlit multi-pages permettant de :

- calculer une **prime mensuelle actuarielle** d’assurance emprunteur décès indexée sur le **capital restant dû (CRD)** ;
- approximer cette prime via un **modèle de Machine Learning** ;
- comparer les deux approches dans une interface claire et structurée.

L’objectif est de relier un moteur actuariel rigoureux à une approximation IA exploitable en contexte opérationnel.

---

# 🎯 Objectif du projet

Le projet repose sur quatre blocs complémentaires :

### 🔹 1. Moteur Actuariat  
- Calcul de la prime mensuelle via le principe d’équivalence actuarielle  
- Conversion mortalité annuelle → mensuelle  
- Calcul de la valeur actuelle espérée des sinistres (EPV)  
- Génération du tableau d’amortissement  
- Export CSV  

### 🔹 2. Moteur IA (Production)  
- Modèle Random Forest figé  
- Prédiction instantanée de la prime mensuelle  
- Strictement isolé des expérimentations R&D  

### 🔹 3. R&D  
- Comparaison de modèles  
- Validation croisée (k = 3 / 4 / 5)  
- Learning curves  
- Analyse des métriques  
- Sauvegarde de modèles expérimentaux (sans impact production)  

### 🔹 4. Comparaison  
- Prime actuarielle vs prime IA  
- Écart absolu  
- Écart relatif  
- Comparaison des échéanciers  

---

# 🗂️ Structure du projet

pfe-streamlit-assurance/
├── src/
│   ├── actuarial/          # amortissement, mortalité, EPV, prime mensuelle
│   ├── ml/                 # features, modèle prod, R&D
│   └── app/
│       ├── app.py          # page d'accueil
│       └── pages/          # pages Streamlit
├── models/
│   ├── prod/               # modèle figé (joblib)
│   └── rd/                 # modèles R&D sauvegardés
├── assets/                 # images, logo
├── data/                   # datasets CSV
├── requirements.txt
└── README.md

---

# ✅ Prérequis

- Python 3.10 ou supérieur (3.11 / 3.12 recommandé)
- pip à jour

---

# 🚀 Installation (Mac / Linux)

1) Cloner le projet

```bash
git clone https://github.com/ethanada10/pfe-streamlit-assurance.git
cd pfe-streamlit-assurance
```

2) Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4) Lancer l’application

```bash
streamlit run src/app/app.py
```

---

# 🚀 Installation (Windows)

1) Cloner le projet

```powershell
git clone https://github.com/ethanada10/pfe-streamlit-assurance.git
cd pfe-streamlit-assurance
```

2) Créer un environnement virtuel

```powershell
python -m venv .venv
```

3) Activer l’environnement

PowerShell :
```powershell
.\.venv\Scripts\Activate.ps1
```

CMD :
```cmd
.\.venv\Scripts\activate.bat
```

4) Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5) Lancer l’application

```powershell
streamlit run src/app/app.py
```

---

# 📦 Dépendances principales

- streamlit  
- pandas  
- numpy  
- scikit-learn  
- joblib  
- matplotlib  

Les versions exactes sont précisées dans requirements.txt.

---

# 📁 Organisation des fichiers

Datasets CSV à placer dans :

data/
  dataset_20k.csv
  dataset_30k.csv
  dataset_70k.csv

La page R&D permet de sélectionner les datasets pour entraîner et analyser les modèles.

---

Modèle de production :

models/prod/
  premium_model_prod.joblib

⚠️ Attention : GitHub limite les fichiers à 100 MB.  
Si le modèle est trop volumineux :
- le régénérer localement,
- réduire sa taille,
- utiliser Git LFS,
- ou fournir un script de training.

---

# 🧪 Utilisation

Page Actuariat  
→ Saisir capital, durée, âge, taux crédit, taux technique, abattement, quotité  
→ Résultat : prime actuarielle + échéancier  

Page IA Production  
→ Même inputs  
→ Prédiction instantanée  

Page R&D  
→ Comparaison modèles  
→ MAE / RMSE / R²  
→ Validation croisée  
→ Learning curves  
→ Importance des variables  

Page Comparaison  
→ Actuariat vs IA  
→ Écart absolu  
→ Écart relatif  

---

# 🛠️ Problèmes fréquents

ModuleNotFoundError: No module named 'src'

Toujours lancer Streamlit depuis la racine :

```bash
cd pfe-streamlit-assurance
streamlit run src/app/app.py
```

Problèmes Python :

Mac/Linux :
```bash
which python
python --version
```

Windows :
```powershell
where python
python --version
```

Conflit numpy / sklearn :

```bash
pip uninstall -y numpy scikit-learn
pip install -r requirements.txt
```

---

# 👤 Auteurs

Adel Kechid  
Ethan Ada  
Valentin Beaufils  