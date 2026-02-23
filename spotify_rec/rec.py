import pandas as pd
import numpy as np
import matplotlib as plt

def dataset_sanity_check(df):
    print("\n==============================")
    print("DATASET OVERVIEW")
    print("==============================")
    print(f"Shape: {df.shape}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    print("\n------------------------------")
    print("COLUMN NAMES")
    print("------------------------------")
    for col in df.columns:
        print(col)

    print("\n------------------------------")
    print("DATA TYPES")
    print("------------------------------")
    print(df.dtypes)

    print("\n------------------------------")
    print("MISSING VALUES (Top 10)")
    print("------------------------------")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(10))

    print("\n------------------------------")
    print("DUPLICATE ROWS")
    print("------------------------------")
    print("Exact duplicate rows:", df.duplicated().sum())

    if "track_id" in df.columns:
        print("Duplicate track_id count:", df["track_id"].duplicated().sum())

    print("\n------------------------------")
    print("NUMERIC SUMMARY")
    print("------------------------------")
    print(df.describe())

    print("\n------------------------------")
    print("NON-FINITE VALUES CHECK")
    print("------------------------------")
    numeric_df = df.select_dtypes(include=[np.number])
    non_finite = ~np.isfinite(numeric_df)
    print("Total non-finite values:", non_finite.sum().sum())

    print("\n------------------------------")
    print("GENRE DISTRIBUTION")
    print("------------------------------")
    if "playlist_genre" in df.columns:
        print(df["playlist_genre"].value_counts())

    print("\n==============================")
    print("SANITY CHECK COMPLETE")
    print("==============================\n")


def main():
    df = pd.read_csv("./data/spotify_songs.csv")
    dataset_sanity_check(df)


if __name__ == "__main__":
    main()