"""
split_dataset.py

Splits the processed weather dataset into
training, development, and testing datasets.

The split is chronological for each city
to prevent time-series data leakage.
"""

import os
import pandas as pd

# ---------------------------------
# Paths
# ---------------------------------

INPUT_FILE = "data/processed/features.csv"
OUTPUT_DIR = "data/splits"

TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15

os.makedirs(OUTPUT_DIR, exist_ok=True)


def split_dataset():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. "
            "Run preprocess.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    df["time"] = pd.to_datetime(df["time"])

    train_sets = []
    dev_sets = []
    test_sets = []

    # ---------------------------------
    # Split each city independently
    # ---------------------------------

    for city in df["city"].unique():

        city_df = (
            df[df["city"] == city]
            .sort_values("time")
            .reset_index(drop=True)
        )

        n = len(city_df)

        train_end = int(n * TRAIN_RATIO)
        dev_end = int(n * (TRAIN_RATIO + DEV_RATIO))

        train_sets.append(city_df.iloc[:train_end])

        dev_sets.append(
            city_df.iloc[train_end:dev_end]
        )

        test_sets.append(
            city_df.iloc[dev_end:]
        )

    # ---------------------------------
    # Combine all cities
    # ---------------------------------

    train_df = pd.concat(
        train_sets,
        ignore_index=True
    )

    dev_df = pd.concat(
        dev_sets,
        ignore_index=True
    )

    test_df = pd.concat(
        test_sets,
        ignore_index=True
    )

    # ---------------------------------
    # Save datasets
    # ---------------------------------

    train_df.to_csv(
        os.path.join(OUTPUT_DIR, "train.csv"),
        index=False
    )

    dev_df.to_csv(
        os.path.join(OUTPUT_DIR, "dev.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(OUTPUT_DIR, "test.csv"),
        index=False
    )

    print("-------------------------------------")
    print("Dataset successfully split.")
    print("-------------------------------------")
    print(f"Training rows:   {len(train_df):,}")
    print(f"Development rows:{len(dev_df):,}")
    print(f"Testing rows:    {len(test_df):,}")
    print("-------------------------------------")
    print("Saved to:")
    print("data/splits/train.csv")
    print("data/splits/dev.csv")
    print("data/splits/test.csv")


if __name__ == "__main__":
    split_dataset()