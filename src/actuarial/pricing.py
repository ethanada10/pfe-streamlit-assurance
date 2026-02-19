from __future__ import annotations
import pandas as pd

def creer_tableau_amortissement(capital: float, duree: int, taux_emprunt_annuel: float) -> pd.DataFrame:
    mensualite = capital * ((taux_emprunt_annuel / 12) / (1 - (1 + taux_emprunt_annuel / 12) ** (-12 * duree)))
    nbr_mensualites = 12 * duree

    capital_restant_du = [capital]
    interets = [0.0]
    mensualites = [0.0]
    capital_rembourse = [0.0]

    for m in range(1, nbr_mensualites + 1):
        interet = capital_restant_du[m - 1] * (taux_emprunt_annuel / 12)
        capital_r = mensualite - interet
        capital_rest = capital_restant_du[m - 1] - capital_r

        interets.append(round(interet, 2))
        mensualites.append(round(mensualite, 2))
        capital_rembourse.append(round(capital_r, 2))
        capital_restant_du.append(round(capital_rest, 2))

    return pd.DataFrame({
        "Num_échéance": range(nbr_mensualites + 1),
        "capital_restant_du": capital_restant_du,
        "interets": interets,
        "mensualites": mensualites,
        "capital_rembourse": capital_rembourse,
    })

def compute_mensual_premium_direct_with_schedule(
    *,
    table_mortalite: pd.DataFrame,
    capital: float,
    duree: int,
    taux_emprunt_annuel: float,
    age: int,
    taux_technique_annuel: float,
) -> tuple[float, pd.DataFrame]:
    """
    Version Streamlit-friendly de ton notebook :
    - interpolation de l(x(t))
    - q(x(t)) = (l(x(t)) - l(x(t)+1 mois)) / l(x(t))
    - tp[x(0)] = l(x(t))/l(x(0))
    - facteur actu mensuel avec (t+0.5)
    - perte = CRD * tp * q * actu
    - prime mensuelle = prime unique / annuity factor
    """

    def create_data_frame(age: int, table_mortalite: pd.DataFrame, duree: int, taux_technique_annuel: float, CRD: pd.Series):
        df = pd.DataFrame()
        df["t"] = range(0, 12 * duree + 1)
        df["y(t)"] = age + df["t"] // 12
        df["x(t)"] = age + df["t"] / 12

        temp = table_mortalite[["Age", "lx"]].rename(columns={"lx": "l_temp"})
        df = df.merge(temp, left_on="y(t)", right_on="Age", how="left")
        df["l(y(t))"] = df["l_temp"]
        df.drop(columns=["Age", "l_temp"], inplace=True)

        age_floor = (df["t"] // 12).astype(int)
        frac = (df["t"] / 12 - age_floor)

        lx_by_month = df["l(y(t))"].values
        idx0 = (age_floor * 12).clip(0, len(df) - 1)
        idx1 = (age_floor * 12 + 12).clip(0, len(df) - 1)

        lx0 = lx_by_month[idx0]
        lx1 = lx_by_month[idx1]
        df["l(x(t))"] = lx0 + frac * (lx1 - lx0)

        qx_dict = table_mortalite.set_index("Age")["qx"].to_dict()
        df["q(y(t))"] = df["y(t)"].map(qx_dict)

        df["q(x(t))"] = (df["l(x(t))"] - df["l(x(t))"].shift(-1)) / df["l(x(t))"]
        df["tp[x(0)]"] = df["l(x(t))"] / df.loc[0, "l(x(t))"]

        taux_technique_mensuel = (1 + taux_technique_annuel) ** (1 / 12) - 1
        df["facteur_actu"] = 1 / (1 + taux_technique_mensuel) ** (df["t"] + 0.5)

        df["CRD"] = CRD.values
        df["Perte"] = df["CRD"] * df["tp[x(0)]"] * df["q(x(t))"] * df["facteur_actu"]

        return df.iloc[:-1]

    def compute_single_premium(df: pd.DataFrame) -> float:
        return float(df["Perte"].sum())

    def compute_annuity_factor(df: pd.DataFrame, taux_technique_annuel: float) -> float:
        df = df.copy()
        df["actu_factor_annuity"] = (1 / ((1 + taux_technique_annuel) ** (1 / 12))) ** df["t"]
        return float((df["tp[x(0)]"] * df["actu_factor_annuity"]).sum())

    amort = creer_tableau_amortissement(capital, duree, taux_emprunt_annuel)
    df_calc = create_data_frame(age, table_mortalite, duree, taux_technique_annuel, amort["capital_restant_du"])

    U = compute_single_premium(df_calc)
    a = compute_annuity_factor(df_calc, taux_technique_annuel)
    prime_mensuelle = float(U / a)

    amort = amort.copy()
    amort["prime_mensuelle"] = prime_mensuelle
    amort["mensualite_totale"] = amort["mensualites"] + amort["prime_mensuelle"]

    return prime_mensuelle, amort
