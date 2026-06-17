"""
collect_team_stats.py
---------------------
Fetch team-level advanced stats (pace, offensive/defensive rating, net rating)
for each season from 2015-16 through 2024-25 using nba_api.

Output: data/raw/team_stats.csv

Usage:
    python -m src.data.collect_team_stats
"""

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.static import teams as nba_teams
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = DATA_DIR / "team_stats.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASONS = [
    "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
]
API_DELAY = 0.7  # seconds between requests

TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in nba_teams.get_teams()}

# Columns we want from the Advanced measure type
KEEP_COLS = [
    "TEAM_ID", "TEAM_NAME",
    "GP", "W", "L", "W_PCT",
    "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE",
    "PIE",  # Player Impact Estimate (team-level)
    "AST_PCT", "AST_TO", "AST_RATIO",
    "OREB_PCT", "DREB_PCT", "REB_PCT",
    "EFG_PCT", "TS_PCT", "TM_TOV_PCT",
]


def fetch_team_advanced(season: str) -> pd.DataFrame | None:
    """Fetch advanced team stats for one season."""
    try:
        result = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense="Advanced",
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
        )
        df = result.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"   [WARN] Error fetching {season}: {e}")
        return None


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []

    for season in tqdm(SEASONS, desc="Fetching team stats", unit="season"):
        time.sleep(API_DELAY)
        df = fetch_team_advanced(season)
        if df is None or df.empty:
            tqdm.write(f"   [SKIP] {season}: No data")
            continue

        # Keep only the columns that exist (some may not in older seasons)
        available = [c for c in KEEP_COLS if c in df.columns]
        df = df[available].copy()
        df["TEAM_ABBREVIATION"] = df["TEAM_ID"].map(TEAM_ID_TO_ABBR)
        df["season"] = season
        all_frames.append(df)
        tqdm.write(f"   [OK] {season}: {len(df)} teams")

    if not all_frames:
        print("[ERROR] No data collected!")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Wrote {len(combined)} rows ({len(SEASONS)} seasons) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
