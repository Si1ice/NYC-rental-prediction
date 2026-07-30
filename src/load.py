"""Загрузка данных, очистка выбросов, time-based split."""
import pandas as pd
from pathlib import Path


def load_raw_data(path: str = "dataset/raw/train.json") -> pd.DataFrame:
    df = pd.read_json(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def remove_outliers(df: pd.DataFrame, target: str = "price",
                    lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    lower = df[target].quantile(lower_q)
    upper = df[target].quantile(upper_q)
    before = len(df)
    df_clean = df[(df[target] >= lower) & (df[target] <= upper)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    print(f"Outliers removed: {before - len(df_clean):,} "
          f"({(before - len(df_clean)) / before * 100:.2f}%)")
    print(f"Bounds: ${lower:,.2f} - ${upper:,.2f}")
    return df_clean


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8,
                     date_col: str = "created"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    split_date = df[date_col].quantile(train_ratio)
    train_df = df[df[date_col] <= split_date].copy()
    test_df = df[df[date_col] > split_date].copy()
    print(f"Time-based split (cutoff: {split_date.date()}):")
    print(f"  Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%)")
    return train_df, test_df


def save_processed(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   output_dir: str = "dataset/processed") -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_path = f"{output_dir}/train_clean.json"
    test_path = f"{output_dir}/test_clean.json"
    train_df.to_json(train_path, orient="records", date_format="iso")
    test_df.to_json(test_path, orient="records", date_format="iso")
    print(f"Saved: {train_path} ({len(train_df):,} rows)")
    print(f"Saved: {test_path} ({len(test_df):,} rows)")