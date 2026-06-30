"""Build breakout labels and features from raw batter season data.

For each player-season (year N), the label is whether the player's WAR
jumps by >= 2.0 in year N+1, conditional on being a qualified batter
(PA >= 300) in year N+1. Features are computed using ONLY data through
year N -- no leakage from the labeling year.

Usage:
    python -m src.features.build_features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WAR_JUMP_THRESHOLD = 2.0
MIN_PA_NEXT_YEAR = 300
MIN_PA_CURRENT_YEAR = 300  # filter out tiny-sample current-year rows


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Consolidate duplicate columns from the merge
    df = df.rename(columns={"PA_x": "PA", "G_x": "G"})
    df = df.drop(columns=["PA_y", "G_y", "Name", "mlb_ID"], errors="ignore")

    return df


def compute_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate stats normalized by PA/AB, safe against div-by-zero."""
    df = df.copy()
    pa = df["PA"].replace(0, np.nan)
    ab = df["AB"].replace(0, np.nan)

    df["bb_rate"] = df["BB"] / pa
    df["k_rate"] = df["SO"] / pa
    df["hr_rate"] = df["HR"] / pa
    df["iso"] = df["SLG"] - df["BA"]  # isolated power
    df["babip"] = (df["H"] - df["HR"]) / (ab - df["SO"] - df["HR"] + df["SF"]).replace(0, np.nan)

    return df


def build_labeled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """For each player-season, attach next-year WAR and the breakout label."""
    df = df.sort_values(["player_ID", "year_ID"]).reset_index(drop=True)

    # Self-join: for each row, find this player's row in year_ID + 1
    next_year = df[["player_ID", "year_ID", "WAR", "PA"]].copy()
    next_year = next_year.rename(
        columns={"year_ID": "year_ID_next", "WAR": "WAR_next", "PA": "PA_next"}
    )
    next_year["year_ID"] = next_year["year_ID_next"] - 1  # join key: prior year

    merged = df.merge(
        next_year[["player_ID", "year_ID", "WAR_next", "PA_next", "year_ID_next"]],
        on=["player_ID", "year_ID"],
        how="left",
    )

    merged["has_next_year"] = merged["year_ID_next"].notna()
    merged["next_year_qualified"] = merged["PA_next"] >= MIN_PA_NEXT_YEAR
    merged["war_jump"] = merged["WAR_next"] - merged["WAR"]

    merged["breakout"] = (
        merged["has_next_year"]
        & merged["next_year_qualified"]
        & (merged["war_jump"] >= WAR_JUMP_THRESHOLD)
    ).astype(int)

    return merged


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year trend features (this year vs last year)."""
    df = df.sort_values(["player_ID", "year_ID"]).reset_index(drop=True)

    prev_year = df[["player_ID", "year_ID", "WAR", "OPS", "PA"]].copy()
    prev_year = prev_year.rename(
        columns={"WAR": "WAR_prev", "OPS": "OPS_prev", "PA": "PA_prev"}
    )
    prev_year["year_ID"] = prev_year["year_ID"] + 1  # join key: shift forward

    df = df.merge(
        prev_year[["player_ID", "year_ID", "WAR_prev", "OPS_prev", "PA_prev"]],
        on=["player_ID", "year_ID"],
        how="left",
    )

    df["has_prior_year"] = df["WAR_prev"].notna()
    df["war_trend"] = df["WAR"] - df["WAR_prev"]
    df["ops_trend"] = df["OPS"] - df["OPS_prev"]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DIR / "batters_2015_2024.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "breakout_features.parquet",
    )
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = load_raw(args.input)
    print(f"  Loaded {len(df):,} player-seasons")

    print("\nFiltering to meaningful sample sizes (PA >= {})...".format(MIN_PA_CURRENT_YEAR))
    df = df[df["PA"] >= MIN_PA_CURRENT_YEAR].copy()
    print(f"  Remaining: {len(df):,} player-seasons")

    print("\nComputing rate stats...")
    df = compute_rate_stats(df)

    print("\nBuilding breakout labels (WAR jump >= {} next year)...".format(WAR_JUMP_THRESHOLD))
    df = build_labeled_dataset(df)

    print("\nAdding year-over-year trend features...")
    df = add_trend_features(df)

    # Only keep rows where we can actually evaluate the label
    # (i.e. has_next_year is known -- rows in the final season can't be labeled)
    labeled = df[df["has_next_year"]].copy()

    print(f"\n=== Label summary (labeled rows only: {len(labeled):,}) ===")
    print(labeled["breakout"].value_counts())
    print(f"Breakout rate: {labeled['breakout'].mean():.3%}")

    print(f"\n=== Full dataset (including unlabeled final-season rows): {len(df):,} ===")
    df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output}")

    print("\n=== war_jump distribution (for labeled rows) ===")
    print(labeled["war_jump"].describe())


if __name__ == "__main__":
    main()
