"""
Préparation commune des données pour la modélisation (Membre 3).
Réutilise temporal_split (Membre 1) et features.py (Membre 2).
"""
import pandas as pd
import numpy as np
from temporal_split import temporal_split
from features import fit_feature_stats, prepare_data

RANDOM_STATE = 42
TARGET_COL = "reservation_annulee"
DATE_COL = "date_reservation"
ID_COL = "reservation_id"

# Colonnes exclues du modèle (identifiants, dates brutes, hautes cardinalités
# non exploitées telles quelles) -- cohérent avec le choix de Membre 1
COLS_TO_DROP_BASE = [TARGET_COL, DATE_COL, "date_arrivee", ID_COL, "hotel_id", "agent_id"]

def load_and_prepare():
    train_df = pd.read_csv("reservations_train.csv", parse_dates=[DATE_COL, "date_arrivee"])
    test_df = pd.read_csv("reservations_test.csv", parse_dates=[DATE_COL, "date_arrivee"])

    # 1. Split temporel AVANT tout calcul de stats (anti-fuite)
    train_sub_raw, val_raw = temporal_split(train_df, DATE_COL, frac_train=0.8)

    # 2. Stats de feature engineering calculées UNIQUEMENT sur train_sub
    stats = fit_feature_stats(train_sub_raw)

    # 3. Application des mêmes stats partout
    train_sub = prepare_data(train_sub_raw, stats)
    val = prepare_data(val_raw, stats)
    test_ready = prepare_data(test_df, stats)

    y_train = train_sub[TARGET_COL]
    y_val = val[TARGET_COL]

    X_train = train_sub.drop(columns=COLS_TO_DROP_BASE)
    X_val = val.drop(columns=COLS_TO_DROP_BASE)
    X_test = test_ready.drop(columns=[c for c in COLS_TO_DROP_BASE if c != TARGET_COL])

    return X_train, y_train, X_val, y_val, X_test, test_ready[ID_COL]

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, test_ids = load_and_prepare()
    print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)
    print("Taux annulation train_sub:", y_train.mean().round(4))
    print("Taux annulation val:", y_val.mean().round(4))
    print(X_train.dtypes)
