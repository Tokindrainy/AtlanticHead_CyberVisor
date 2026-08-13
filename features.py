"""
features.py
============
Module de feature engineering — Atlantic Haven Hotels (hackathon ML).

Rôle : transformer les données brutes de réservation en variables exploitables
par les modèles, SANS fuite de cible ni fuite temporelle.

Principe clé (anti-leakage) :
- `fit_feature_stats(train_df)` calcule toutes les statistiques (moyennes,
  médianes, etc.) UNIQUEMENT sur le train.
- `prepare_data(df, stats)` applique ces statistiques à n'importe quel
  dataframe (train, validation, test) sans jamais recalculer sur ce dataframe.

Usage attendu par l'équipe :

    from features import fit_feature_stats, prepare_data

    stats = fit_feature_stats(train_df)          # à faire UNE SEULE FOIS sur le train
    train_ready = prepare_data(train_df, stats)
    val_ready   = prepare_data(val_df, stats)     # même stats, pas de recalcul
    test_ready  = prepare_data(test_df, stats)    # idem pour reservations_test.csv
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Calcul des statistiques d'entraînement (à faire UNE SEULE FOIS sur train)
# ---------------------------------------------------------------------------

def fit_feature_stats(train_df: pd.DataFrame) -> dict:
    """
    Calcule toutes les statistiques nécessaires aux features, à partir du
    train UNIQUEMENT. Ce dictionnaire doit être réutilisé tel quel pour
    transformer la validation et le test.
    """
    stats = {}

    # Médiane du prix par région + catégorie d'hôtel (imputation ciblée,
    # plus pertinente qu'une médiane globale)
    stats["prix_median_par_region_cat"] = (
        train_df.groupby(["region_hotel", "categorie_hotel"])["prix_moyen_nuit_eur"]
        .median()
    )
    stats["prix_median_global"] = train_df["prix_moyen_nuit_eur"].median()

    # Moyenne du prix par région + catégorie (pour l'écart au prix moyen)
    stats["prix_moyenne_par_region_cat"] = (
        train_df.groupby(["region_hotel", "categorie_hotel"])["prix_moyen_nuit_eur"]
        .mean()
    )
    stats["prix_moyenne_globale"] = train_df["prix_moyen_nuit_eur"].mean()

    # Médianes pour les autres colonnes à valeurs manquantes
    stats["enfants_median"] = train_df["enfants"].median()
    stats["demandes_speciales_median"] = train_df["demandes_speciales"].median()

    return stats


# ---------------------------------------------------------------------------
# 2. Application des transformations (train, validation OU test)
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Applique le feature engineering à un dataframe donné, en utilisant
    uniquement les statistiques précalculées sur le train (`stats`).
    Ne modifie jamais `df` en place — retourne une copie.
    """
    out = df.copy()

    # -----------------------------------------------------------------
    # A. Imputation des valeurs manquantes (ciblée, pas de fillna(0) aveugle)
    # -----------------------------------------------------------------
    out["enfants"] = out["enfants"].fillna(stats["enfants_median"])
    out["demandes_speciales"] = out["demandes_speciales"].fillna(
        stats["demandes_speciales_median"]
    )

    # Imputation du prix par région+catégorie, avec repli sur médiane globale
    # si la combinaison région/catégorie n'existe pas dans les stats du train
    def _impute_prix(row):
        if pd.notna(row["prix_moyen_nuit_eur"]):
            return row["prix_moyen_nuit_eur"]
        key = (row["region_hotel"], row["categorie_hotel"])
        return stats["prix_median_par_region_cat"].get(key, stats["prix_median_global"])

    out["prix_moyen_nuit_eur"] = out.apply(_impute_prix, axis=1)

    # -----------------------------------------------------------------
    # B. Historique client
    # -----------------------------------------------------------------
    out["client_sans_historique"] = (out["reservations_passees"] == 0).astype(int)
    out["taux_annulation_historique"] = np.where(
        out["reservations_passees"] > 0,
        out["annulations_passees"] / out["reservations_passees"],
        0.0,  # valeur neutre pour les nouveaux clients, flag séparé ci-dessus
    )

    # -----------------------------------------------------------------
    # C. Séjour / composition du groupe
    # -----------------------------------------------------------------
    out["taille_groupe"] = out["adultes"] + out["enfants"]
    out["ratio_nuits_chambres"] = out["nuits"] / out["chambres"].replace(0, np.nan)
    out["ratio_nuits_chambres"] = out["ratio_nuits_chambres"].fillna(out["nuits"])

    # -----------------------------------------------------------------
    # D. Prix — écart à la moyenne région/catégorie (calculée sur train)
    # -----------------------------------------------------------------
    def _ecart_prix(row):
        key = (row["region_hotel"], row["categorie_hotel"])
        moyenne = stats["prix_moyenne_par_region_cat"].get(key, stats["prix_moyenne_globale"])
        return row["prix_moyen_nuit_eur"] - moyenne

    out["ecart_prix_vs_moyenne_region"] = out.apply(_ecart_prix, axis=1)

    # -----------------------------------------------------------------
    # E. Commercial
    # -----------------------------------------------------------------
    out["is_reservation_directe"] = out["agent_id"].isna().astype(int)

    out["tarif_remboursable_bin"] = (
        out["tarif_remboursable"].astype(str).str.lower().isin(["oui", "true", "1", "yes"])
    ).astype(int)

    out["acompte_x_remboursable"] = (
        out["type_acompte"].astype(str) + "_" + out["tarif_remboursable_bin"].astype(str)
    )

    # -----------------------------------------------------------------
    # F. Comportement avant arrivée
    # -----------------------------------------------------------------
    out["a_modifie_reservation"] = (out["modifications_reservation"] > 0).astype(int)
    out["a_attendu_liste"] = (out["jours_liste_attente"] > 0).astype(int)

    return out


# ---------------------------------------------------------------------------
# 3. Test rapide en exécution directe (à adapter avec les vrais fichiers)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Exemple d'utilisation — à remplacer par les vrais chemins le jour J
    train_df = pd.read_csv("reservations_train.csv", parse_dates=["date_reservation", "date_arrivee"])
    test_df = pd.read_csv("reservations_test.csv", parse_dates=["date_reservation", "date_arrivee"])

    stats = fit_feature_stats(train_df)
    train_ready = prepare_data(train_df, stats)
    test_ready = prepare_data(test_df, stats)

    print("Train prêt :", train_ready.shape)
    print("Test prêt :", test_ready.shape)
    print(train_ready[[
        "taux_annulation_historique", "taille_groupe", "ecart_prix_vs_moyenne_region",
        "is_reservation_directe", "acompte_x_remboursable"
    ]].head())