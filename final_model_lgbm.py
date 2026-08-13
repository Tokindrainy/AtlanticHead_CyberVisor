"""
Modèle final retenu (Membre 3) — LightGBM
==========================================
Réentraîné sur l'intégralité du train (8000 lignes), avec les
hyperparamètres optimaux trouvés par Optuna sur la validation temporelle,
et le seuil de décision optimal.

Livrable pour Membre 4 : ce script + best_lgbm_hyperparams.json +
best_lgbm_threshold.json + preview des prédictions sur le vrai test.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
import json

from features import fit_feature_stats, prepare_data

RANDOM_STATE = 42
TARGET_COL = "reservation_annulee"
DATE_COL = "date_reservation"
ID_COL = "reservation_id"
COLS_TO_DROP_BASE = [TARGET_COL, DATE_COL, "date_arrivee", ID_COL, "hotel_id", "agent_id"]

BEST_PARAMS = {
    'n_estimators': 256, 'learning_rate': 0.014464020877451817, 'num_leaves': 36,
    'max_depth': 4, 'min_child_samples': 59, 'subsample': 0.7004229904782565,
    'colsample_bytree': 0.7724567389301411, 'reg_alpha': 0.0016742237510088004,
    'reg_lambda': 0.13601246953903182, 'scale_pos_weight': 3.6629470290120234,
}
BEST_THRESHOLD = 0.51  # trouvé sur la validation temporelle (derniers 20% du train)

def main():
    train_df = pd.read_csv("reservations_train.csv", parse_dates=[DATE_COL, "date_arrivee"])
    test_df = pd.read_csv("reservations_test.csv", parse_dates=[DATE_COL, "date_arrivee"])

    # Stats de feature engineering recalculées sur TOUT le train (pas juste
    # train_sub) puisqu'on n'a plus besoin de réserver une validation ici :
    # la validation temporelle a déjà servi à choisir hyperparamètres + seuil.
    stats = fit_feature_stats(train_df)
    train_ready = prepare_data(train_df, stats)
    test_ready = prepare_data(test_df, stats)

    y_train = train_ready[TARGET_COL]
    X_train = train_ready.drop(columns=COLS_TO_DROP_BASE)
    X_test = test_ready.drop(columns=[c for c in COLS_TO_DROP_BASE if c != TARGET_COL])

    cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        cats = X_train[c].cat.categories
        X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats)

    params = dict(BEST_PARAMS)
    params.update({"objective": "binary", "metric": "None", "verbosity": -1,
                    "boosting_type": "gbdt", "random_state": RANDOM_STATE})
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= BEST_THRESHOLD).astype(int)

    result = pd.DataFrame({
        "reservation_id": test_ready[ID_COL].values,
        "probabilite_annulation": proba_test,
        "reservation_annulee": pred_test,
    })

    result.to_csv("predictions_membre3_lgbm.csv", index=False)

    print("Modèle final : LightGBM")
    print("Hyperparamètres :", BEST_PARAMS)
    print("Seuil de décision retenu :", BEST_THRESHOLD)
    print()
    print("Prédictions générées sur reservations_test.csv :", result.shape)
    print("Taux d'annulation prédit sur le test :", round(result['reservation_annulee'].mean(), 4))
    print()
    print(result.head(10))

    model.booster_.save_model("lgbm_final_model.txt")

    with open("lgbm_final_config.json", "w") as f:
        json.dump({"hyperparams": BEST_PARAMS, "threshold": BEST_THRESHOLD,
                    "categorical_features": cat_cols}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
