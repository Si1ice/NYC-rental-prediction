import json
import pandas as pd
from pathlib import Path

def load_config(config_path: str = "config.json") -> dict:
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_raw_data(config: dict) -> pd.DataFrame:
    """Загружает сырые данные из JSON файла."""
    paths = config['paths']
    print(f"Загрузка данных из: {paths['raw_train']}")
    df = pd.read_json(paths['raw_train'])
    print(f"Размер датасета: {df.shape}")
    return df