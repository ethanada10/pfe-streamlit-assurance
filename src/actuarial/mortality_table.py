from __future__ import annotations
import pandas as pd

def abattement(table_mortalite: pd.DataFrame, taux_abattement: float) -> pd.DataFrame:
    table = table_mortalite.copy()

    if "Age" not in table.columns:
        raise ValueError("La table doit contenir une colonne 'Age'.")
    if "qx" not in table.columns or "lx" not in table.columns:
        raise ValueError("La table doit contenir les colonnes 'qx' et 'lx'.")

    table["qx"] = table["qx"].astype(float) * (1 - float(taux_abattement))
    table.at[table.index[-1], "qx"] = 1.00

    table["lx"] = table["lx"].astype(float)
    for i in range(1, len(table)):
        table.loc[i, "lx"] = table.loc[i - 1, "lx"] * (1 - table.loc[i - 1, "qx"])

    return table

def load_mortality_table_with_abattement(*, table_path: str, taux_abattement: float = 0.35) -> pd.DataFrame:
    table = pd.read_excel(table_path)
    return abattement(table, taux_abattement)
