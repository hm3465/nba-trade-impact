"""
collect_game_logs.py
--------------------
For every trade in data/raw/trades.csv, fetch the player's full-season game
logs from nba_api, split them into pre-trade and post-trade windows, compute
derived stats (TS%, USG%), and write:

    data/raw/player_game_logs.csv      – every individual game row
    data/raw/player_trade_splits.csv   – aggregated pre/post means + deltas

Usage:
    python -m src.data.collect_game_logs
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelog
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRADES_PATH = DATA_DIR / "trades.csv"
TEAM_STATS_PATH = DATA_DIR / "team_stats.csv"
GAME_LOGS_PATH = DATA_DIR / "player_game_logs.csv"
SPLITS_PATH = DATA_DIR / "player_trade_splits.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_GAMES = 10          # Minimum games on each side to include
API_DELAY = 0.7         # seconds between nba_api calls (avoid 429s)

# ---------------------------------------------------------------------------
# Stats columns from PlayerGameLog
# ---------------------------------------------------------------------------
STAT_COLS = [
    "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]


def load_team_pace() -> dict[tuple[str, str], float]:
    """
    Load team PACE values from team_stats.csv.

    Returns a dict mapping (TEAM_ABBREVIATION, season) → PACE.
    Falls back to empty dict if file doesn't exist.
    """
    if not TEAM_STATS_PATH.exists():
        print("   [INFO] team_stats.csv not found; using default PACE estimate.")
        return {}

    try:
        df = pd.read_csv(TEAM_STATS_PATH)
        pace_map: dict[tuple[str, str], float] = {}
        for _, row in df.iterrows():
            abbr = row.get("TEAM_ABBREVIATION", "")
            season = row.get("season", "")
            pace = row.get("PACE", None)
            if abbr and season and pace is not None and not pd.isna(pace):
                pace_map[(str(abbr), str(season))] = float(pace)
        print(f"   [INFO] Loaded PACE data for {len(pace_map)} team-seasons.")
        return pace_map
    except Exception as e:
        print(f"   [WARN] Error loading team_stats.csv: {e}")
        return {}


def compute_ts_pct(pts: float, fga: float, fta: float) -> float:
    """True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))."""
    denom = 2 * (fga + 0.44 * fta)
    if denom == 0:
        return 0.0
    return pts / denom


def compute_usg_rate(fga: float, fta: float, tov: float,
                     minutes: float, team_pace: float | None = None) -> float:
    """
    Compute USG% using team PACE when available.

    Full formula:
        USG% = 100 * (FGA + 0.44*FTA + TOV) * (team_pace * minutes_per_game / 48)
               / (minutes * team_possessions_per_game / 5)

    Simplified with PACE (possessions per 48 min):
        team_possessions_in_player_minutes ≈ PACE * (minutes / 48)
        USG% ≈ (FGA + 0.44*FTA + TOV) / (team_possessions_in_player_minutes / 5) * 100

    When PACE is unavailable, falls back to per-minute proxy:
        USG% ≈ ((FGA + 0.44*FTA + TOV) / minutes) * (48 / 5) * 100
    """
    if minutes == 0:
        return 0.0

    possessions_used = fga + 0.44 * fta + tov

    if team_pace is not None and team_pace > 0:
        # Team possessions during this player's minutes
        team_poss_in_minutes = team_pace * (minutes / 48.0)
        # Player's share = possessions_used / (team_poss / 5)
        return (possessions_used / (team_poss_in_minutes / 5.0)) * 100.0
    else:
        # Fallback: per-minute proxy (assumes ~100 possessions per 48 min)
        return (possessions_used / minutes) * (48.0 / 5.0) * 100.0


def parse_team_from_matchup(matchup: str) -> str:
    """
    Extract the player's team abbreviation from the MATCHUP string.
    Examples:
        'LAL vs. PHX' → 'LAL'
        'LAL @ DEN'   → 'LAL'
    """
    return matchup.strip().split(" ")[0].upper()


def fetch_season_game_log(player_id: int, season: str) -> pd.DataFrame | None:
    """Fetch a single season's game log. Returns DataFrame or None on error."""
    try:
        gl = playergamelog.PlayerGameLog(
            player_id=str(player_id),
            season=season,
            season_type_all_star="Regular Season",
        )
        df = gl.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"   [WARN] API error for player {player_id}, season {season}: {e}")
        return None


def parse_minutes(min_str) -> float:
    """Convert MIN column to float minutes. Handles 'MM:SS' and plain float."""
    if pd.isna(min_str):
        return 0.0
    s = str(min_str).strip()
    if ":" in s:
        parts = s.split(":")
        return float(parts[0]) + float(parts[1]) / 60.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def aggregate_window(df: pd.DataFrame, team_pace: float | None = None) -> dict:
    """Compute mean stats for a window of games."""
    n = len(df)
    if n == 0:
        return {}

    agg = {"games": n}
    for col in STAT_COLS:
        if col in df.columns:
            agg[col.lower()] = df[col].mean()

    # Derived stats from totals (more accurate than averaging per-game TS%)
    total_pts = df["PTS"].sum()
    total_fga = df["FGA"].sum()
    total_fta = df["FTA"].sum()
    total_tov = df["TOV"].sum()
    total_min = df["MIN_FLOAT"].sum() if "MIN_FLOAT" in df.columns else df["MIN"].sum()

    agg["ts_pct"] = compute_ts_pct(total_pts, total_fga, total_fta)
    agg["usg_rate"] = compute_usg_rate(
        total_fga / n, total_fta / n, total_tov / n,
        total_min / n,
        team_pace=team_pace,
    )

    return agg


def main() -> None:
    # Load trades
    if not TRADES_PATH.exists():
        print(f"[ERROR] {TRADES_PATH} not found. Run collect_trades.py first.")
        sys.exit(1)

    trades_df = pd.read_csv(TRADES_PATH)
    print(f"Loaded {len(trades_df)} trades from {TRADES_PATH}")

    # Load team PACE data
    pace_map = load_team_pace()

    all_game_rows: list[dict] = []
    split_rows: list[dict] = []
    skipped = 0

    pbar = tqdm(trades_df.iterrows(), total=len(trades_df),
                desc="Processing trades", unit="trade")

    for idx, trade in pbar:
        player_name = trade["player_name"]
        player_id = int(trade["player_id"])
        season = trade["season"]
        trade_date = pd.to_datetime(trade["trade_date"])
        team_from = trade["team_from"]
        team_to = trade["team_to"]

        pbar.set_postfix_str(f"{player_name} ({team_from}->{team_to})")

        # Fetch game log
        time.sleep(API_DELAY)
        gl_df = fetch_season_game_log(player_id, season)
        if gl_df is None or gl_df.empty:
            tqdm.write(f"   [SKIP] {player_name}: No game log data")
            skipped += 1
            continue

        # Parse dates and minutes
        gl_df["GAME_DATE_PARSED"] = pd.to_datetime(gl_df["GAME_DATE"], format="mixed")
        gl_df["MIN_FLOAT"] = gl_df["MIN"].apply(parse_minutes)
        gl_df["TEAM_ABR"] = gl_df["MATCHUP"].apply(parse_team_from_matchup)

        # Split into pre-trade and post-trade
        pre_df = gl_df[gl_df["GAME_DATE_PARSED"] < trade_date].copy()
        post_df = gl_df[gl_df["GAME_DATE_PARSED"] >= trade_date].copy()

        # Further filter: pre should be team_from, post should be team_to
        pre_df = pre_df[pre_df["TEAM_ABR"] == team_from]
        post_df = post_df[post_df["TEAM_ABR"] == team_to]

        if len(pre_df) < MIN_GAMES or len(post_df) < MIN_GAMES:
            tqdm.write(f"   [SKIP] {player_name}: pre={len(pre_df)}, "
                       f"post={len(post_df)} (need {MIN_GAMES}+)")
            skipped += 1
            continue

        # Store individual game rows
        for _, g in pd.concat([pre_df, post_df]).iterrows():
            all_game_rows.append({
                "player_id":    player_id,
                "player_name":  player_name,
                "season":       season,
                "trade_date":   trade_date.strftime("%Y-%m-%d"),
                "game_date":    g["GAME_DATE"],
                "game_id":      g.get("Game_ID", ""),
                "team":         g["TEAM_ABR"],
                "matchup":      g["MATCHUP"],
                "wl":           g.get("WL", ""),
                "period":       "pre" if g["GAME_DATE_PARSED"] < trade_date else "post",
                **{col.lower(): g.get(col, 0) for col in STAT_COLS},
                "min_float":    g.get("MIN_FLOAT", 0),
            })

        # Look up team PACE for pre/post teams
        pre_pace = pace_map.get((team_from, season))
        post_pace = pace_map.get((team_to, season))

        # Aggregate
        pre_agg = aggregate_window(pre_df, team_pace=pre_pace)
        post_agg = aggregate_window(post_df, team_pace=post_pace)

        row: dict = {
            "player_id":    player_id,
            "player_name":  player_name,
            "season":       season,
            "trade_date":   trade_date.strftime("%Y-%m-%d"),
            "team_from":    team_from,
            "team_to":      team_to,
            "pre_games":    pre_agg["games"],
            "post_games":   post_agg["games"],
        }

        # Add pre/post/delta for each stat
        stat_keys = [c.lower() for c in STAT_COLS] + ["ts_pct", "usg_rate"]
        for key in stat_keys:
            pre_val = pre_agg.get(key, 0.0)
            post_val = post_agg.get(key, 0.0)
            row[f"pre_{key}"] = round(pre_val, 4)
            row[f"post_{key}"] = round(post_val, 4)
            row[f"delta_{key}"] = round(post_val - pre_val, 4)

        split_rows.append(row)
        delta_pts = post_agg.get("pts", 0) - pre_agg.get("pts", 0)
        tqdm.write(
            f"   [OK] {player_name}: PTS {pre_agg.get('pts', 0):.1f} -> "
            f"{post_agg.get('pts', 0):.1f} (d{delta_pts:+.1f})"
        )

    # ── Write game logs CSV ──────────────────────────────────────────
    if all_game_rows:
        gl_out = pd.DataFrame(all_game_rows)
        gl_out.to_csv(GAME_LOGS_PATH, index=False)
        print(f"\n[OK] Wrote {len(gl_out)} game log rows to {GAME_LOGS_PATH}")

    # ── Write splits CSV ─────────────────────────────────────────────
    if split_rows:
        splits_out = pd.DataFrame(split_rows)
        splits_out.to_csv(SPLITS_PATH, index=False)
        print(f"[OK] Wrote {len(splits_out)} trade splits to {SPLITS_PATH}")

    print(f"\nDone. {len(split_rows)} trades processed, {skipped} skipped.")


if __name__ == "__main__":
    main()
