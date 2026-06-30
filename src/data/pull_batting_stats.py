"""Pull batter season stats (2015-2024) using Baseball-Reference data.

FanGraphs blocks pybaseball's scraper (403), so we use two
Baseball-Reference-backed pybaseball functions instead:

  - bwar_bat(): static text file with WAR, pulled once for full history,
    filtered down to batters in our season range. Source of our label.
  - batting_stats_bref(season): per-season counting/rate stats, looped
    over each season since it only accepts one year per call. Source
    of our features.

The two are merged on mlb_ID/mlbID + year, with multi-stint seasons
(player traded mid-year) aggregated into a single row per player-season.

Usage:
    python -m src.data.pull_batting_stats
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from pybaseball import batting_stats_bref, bwar_bat

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

START_SEASON = 2015
END_SEASON = 2024


def pull_war_data(start: int, end: int) -> pd.DataFrame:
    """Pull bwar_bat, filter to batters in our season range, aggregate stints."""
    print("Pulling bwar_bat() — full history WAR table (one request)...")
    war = bwar_bat()
    print(f"  Raw: {len(war):,} rows")

    war = war[war["pitcher"] == "N"].copy()
    war = war[(war["year_ID"] >= start) & (war["year_ID"] <= end)].copy()
    print(f"  After batter + season filter: {len(war):,} rows")

    # Aggregate multi-stint seasons (traded mid-year) into one row.
    # Sum the additive stats; keep the first non-null for identifiers.
    agg = (
        war.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg(
            name_common=("name_common", "first"),
            player_ID=("player_ID", "first"),
            G=("G", "sum"),
            PA=("PA", "sum"),
            WAR=("WAR", "sum"),
            WAA=("WAA", "sum"),
            runs_above_avg=("runs_above_avg", "sum"),
            runs_above_avg_off=("runs_above_avg_off", "sum"),
            runs_above_avg_def=("runs_above_avg_def", "sum"),
        )
    )
    print(f"  After stint aggregation: {len(agg):,} player-seasons")
    return agg


def pull_rate_stats(start: int, end: int) -> pd.DataFrame:
    """Loop batting_stats_bref over each season (one year per call)."""
    frames = []
    for year in range(start, end + 1):
        print(f"  Pulling batting_stats_bref({year})...")
        df = batting_stats_bref(year)
        df["year_ID"] = year
        frames.append(df)
        time.sleep(1.0)  # polite delay between requests

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows across {end - start + 1} seasons")
    return combined


def fix_encoding(df: pd.DataFrame, col: str) -> pd.Series:
    """Fix literal-escape-sequence names (e.g. 'Jos\\xc3\\xa9' -> 'José').

    pybaseball's HTML scraper sometimes returns the literal text
    "\\xc3\\xa9" (8 visible characters: backslash, x, c, 3, ...) instead
    of decoding it into the corresponding UTF-8 character. We reverse
    this by treating the text as a Python escape sequence, decoding it
    into raw bytes, then decoding those bytes as UTF-8.
    """
    def _fix(val):
        if not isinstance(val, str):
            return val
        if "\\x" not in val:
            return val  # no escape sequence present, nothing to fix
        try:
            return val.encode("utf-8").decode("unicode_escape").encode("latin1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return val  # fixing failed, keep original rather than crash
    return df[col].apply(_fix)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=START_SEASON)
    parser.add_argument("--end", type=int, default=END_SEASON)
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR / f"batters_{START_SEASON}_{END_SEASON}.parquet",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.out.exists() and not args.force:
        existing = pd.read_parquet(args.out)
        print(f"Already pulled: {len(existing):,} rows in {args.out.name}")
        return

    war_df = pull_war_data(args.start, args.end)
    print()
    rate_df = pull_rate_stats(args.start, args.end)

    # Fix encoding on the name column before merging/saving
    rate_df["Name"] = fix_encoding(rate_df, "Name")
    war_df["name_common"] = fix_encoding(war_df, "name_common")

    print()
    print("Merging WAR data with rate stats on mlb_ID/mlbID + year...")
    merged = rate_df.merge(
        war_df,
        left_on=["mlbID", "year_ID"],
        right_on=["mlb_ID", "year_ID"],
        how="inner",
    )
    print(f"  Merged: {len(merged):,} rows")
    print(f"  Rate stats unmatched: {len(rate_df) - len(merged):,} rows dropped")

    merged.to_parquet(args.out, index=False)
    print(f"\nSaved to {args.out}")

    print()
    print("=== Per-season row counts ===")
    print(merged["year_ID"].value_counts().sort_index())

    print()
    print("=== Columns ===")
    print(sorted(merged.columns.tolist()))

    print()
    print("=== Sample row ===")
    print(merged.iloc[0])


if __name__ == "__main__":
    main()