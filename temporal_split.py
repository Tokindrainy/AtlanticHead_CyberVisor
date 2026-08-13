import pandas as pd

def temporal_split(df, date_col, frac_train=0.8):
    """Trie par date et coupe en train_sub / val (PAS de shuffle, PAS de K-Fold)."""
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    idx = int(len(df_sorted) * frac_train)
    return df_sorted.iloc[:idx].copy(), df_sorted.iloc[idx:].copy()
