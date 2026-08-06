"""
preprocess.py

Reads all raw weather CSV files, performs feature engineering,
and saves a cleaned dataset for model training.
"""

import os
import glob

import numpy as np
import pandas as pd

RAW_DATA_PATH = "data/raw"
OUTPUT_PATH = "data/processed"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def preprocess():

    csv_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.csv"))

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "No CSV files found inside data/raw."
        )

    dataframes = []

    for file in csv_files:

        print(f"Processing {file}")

        df = pd.read_csv(file)

        # -----------------------------
        # City name from filename
        # -----------------------------
        city = os.path.splitext(os.path.basename(file))[0].replace("_", " ")
        df["city"] = city

        # -----------------------------
        # Convert datetime
        # -----------------------------
        df["time"] = pd.to_datetime(df["time"])

        # -----------------------------
        # Remove missing values
        # -----------------------------
        df = df.dropna()

        # -----------------------------
        # Time features
        # -----------------------------
        df["year"] = df["time"].dt.year
        df["month"] = df["time"].dt.month
        df["day"] = df["time"].dt.day
        df["hour"] = df["time"].dt.hour
        df["day_of_week"] = df["time"].dt.dayofweek
        df["day_of_year"] = df["time"].dt.dayofyear

        # -----------------------------
        # Cyclical Encoding
        # -----------------------------
        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        df["day_sin"] = np.sin(
            2 * np.pi * df["day_of_year"] / 365
        )

        df["day_cos"] = np.cos(
            2 * np.pi * df["day_of_year"] / 365
        )

        dataframes.append(df)

    # -----------------------------
    # Combine every city
    # -----------------------------
    final_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    # -----------------------------
    # Sort chronologically
    # -----------------------------
    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # -----------------------------
    # Save processed dataset
    # -----------------------------
    output_file = os.path.join(
        OUTPUT_PATH,
        "features.csv"
    )

    final_df.to_csv(
        output_file,
        index=False
    )

    print("--------------------------------")
    print("Finished preprocessing.")
    print(f"Rows: {len(final_df):,}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    preprocess()