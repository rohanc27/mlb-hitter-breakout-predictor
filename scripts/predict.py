"""CLI to score a player's breakout probability for next season.

Usage:
    python scripts/predict.py --name "Bobby Witt Jr." --year 2023
    python scripts/predict.py --list-players  # see available names
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logreg.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "breakout_features.parquet"

FEATURES = [
    "Age", "WAR", "PA", "BA", "OBP", "SLG", "OPS",
    "bb_rate", "k_rate", "hr_rate", "iso", "babip",
    "war_trend", "ops_trend",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, help="Player name (partial match OK)")
    parser.add_argument("--year", type=int, help="Season year to score from")
    parser.add_argument("--list-players", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(FEATURES_PATH)

    if args.list_players:
        names = sorted(df["name_common"].dropna().unique())
        print("\n".join(names))
        return

    if not args.name or not args.year:
        parser.error("--name and --year are required unless using --list-players")

    matches = df[
        df["name_common"].str.contains(args.name, case=False, na=False)
        & (df["year_ID"] == args.year)
    ]

    if len(matches) == 0:
        print(f"No match found for '{args.name}' in {args.year}.")
        print("Try --list-players to see available names.")
        return
    if len(matches) > 1:
        print(f"Multiple matches found:")
        print(matches[["name_common", "year_ID"]].to_string(index=False))
        return

    row = matches.iloc[0]
    if row[FEATURES].isna().any():
        print(f"Warning: {row['name_common']} ({args.year}) is missing required "
              f"features (likely no prior-year data). Cannot score reliably.")
        return

    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([row[FEATURES]])
    probability = float(model.predict_proba(X)[0, 1])

    print(f"\n{row['name_common']} — {args.year} season")
    print(f"  Age: {row['Age']:.0f}, WAR: {row['WAR']:.2f}, OPS: {row['OPS']:.3f}")
    print(f"  WAR trend (vs prior year): {row['war_trend']:+.2f}")
    print(f"\n  Predicted breakout probability for {args.year + 1}: {probability:.1%}")

    if row.get("has_next_year") and not pd.isna(row.get("war_jump")):
        actual = "YES" if row["breakout"] == 1 else "no"
        print(f"  (Actual outcome: breakout = {actual}, "
              f"WAR jump = {row['war_jump']:+.2f})")


if __name__ == "__main__":
    main()
