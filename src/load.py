import json
import pandas as pd
from pathlib import Path

# Константы для предобработки и разделения данных
TARGET_COLUMN = "price"
DATE_COLUMN = "created"
SPLIT_DATE = "2016-05-01"
OUTLIER_LOWER_QUANTILE = 0.01
OUTLIER_UPPER_QUANTILE = 0.99


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def clean_outliers(df: pd.DataFrame, target_col: str, lower_q: float, upper_q: float) -> pd.DataFrame:
    lower_limit = df[target_col].quantile(lower_q)
    upper_limit = df[target_col].quantile(upper_q)
    
    initial_shape = df.shape
    df_clean = df[(df[target_col] >= lower_limit) & (df[target_col] <= upper_limit)].copy()
    df_clean.reset_index(drop=True, inplace=True)
    
    print(f"Очистка выбросов: {initial_shape[0]} -> {df_clean.shape[0]} строк.")
    return df_clean


def split_data_by_time(df: pd.DataFrame, split_date: str, date_col: str) -> tuple:
    """
    Разделяет датасет на обучающую и тестовую выборки по временной метке.
    """
    split_datetime = pd.to_datetime(split_date)
    df[date_col] = pd.to_datetime(df[date_col])
    
    train_df = df[df[date_col] < split_datetime].copy()
    test_df = df[df[date_col] >= split_datetime].copy()
    
    print(f"Разделение данных: Train = {len(train_df)}, Test = {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def process_and_save_data(config: dict) -> tuple:
    """
    Основной пайплайн загрузки, очистки и сохранения данных.
    """
    paths = config['paths']
    
    Path("dataset/processed").mkdir(parents=True, exist_ok=True)
    
    print("Загрузка исходных данных...")
    df = pd.read_json(paths['raw_train'])
    
    print("Очистка выбросов...")
    df_clean = clean_outliers(
        df, 
        target_col=TARGET_COLUMN,
        lower_q=OUTLIER_LOWER_QUANTILE,
        upper_q=OUTLIER_UPPER_QUANTILE
    )
    
    print("Разделение на train и test...")
    train_df, test_df = split_data_by_time(
        df_clean,
        split_date=SPLIT_DATE,
        date_col=DATE_COLUMN
    )
    
    print("Сохранение обработанных данных...")
    train_df.to_json(paths['processed_train'], orient='records', lines=False)
    test_df.to_json(paths['processed_test'], orient='records', lines=False)
    print("Обработанные данные сохранены в dataset/processed/")
    
    return train_df, test_df


if __name__ == "__main__":
    cfg = load_config()
    train, test = process_and_save_data(cfg)