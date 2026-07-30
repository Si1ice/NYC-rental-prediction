"""Точка входа. Запускает пайплайн."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.load import load_raw_data, remove_outliers, time_based_split, save_processed


def main():
    print("NYC Rental Price Prediction - Pipeline")

    print("\n[1/3] Loading data")
    df = load_raw_data("dataset/raw/train.json")

    print("\n[2/3] Removing outliers")
    df = remove_outliers(df)

    print("\n[3/3] Time-based split")
    train_df, test_df = time_based_split(df)
    save_processed(train_df, test_df)

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()