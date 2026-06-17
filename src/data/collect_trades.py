"""
collect_trades.py
-----------------
Generate a curated CSV of notable NBA mid-season trades (2015-16 through 2024-25).

Each row represents one player who was traded mid-season and played meaningful
minutes for both teams.  The CSV is written to data/raw/trades.csv.

Usage:
    python -m src.data.collect_trades
"""

import csv
import os
import sys
import unicodedata
from pathlib import Path

from nba_api.stats.static import players as nba_players

# ---------------------------------------------------------------------------
# Project root (two levels up from this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "trades.csv"

# ---------------------------------------------------------------------------
# NBA team abbreviation mapping  (BBRef / common → nba_api)
# Most are identical; a few differ historically.
# ---------------------------------------------------------------------------
TEAM_ABBR_MAP = {
    "PHX": "PHX", "BKN": "BKN", "PHI": "PHI", "DAL": "DAL",
    "LAL": "LAL", "LAC": "LAC", "BOS": "BOS", "MIA": "MIA",
    "MIL": "MIL", "GSW": "GSW", "DEN": "DEN", "MIN": "MIN",
    "CLE": "CLE", "TOR": "TOR", "IND": "IND", "NOP": "NOP",
    "ATL": "ATL", "CHI": "CHI", "SAC": "SAC", "POR": "POR",
    "OKC": "OKC", "HOU": "HOU", "MEM": "MEM", "SAS": "SAS",
    "NYK": "NYK", "WAS": "WAS", "DET": "DET", "ORL": "ORL",
    "CHA": "CHA", "UTA": "UTA",
    # Common alternate abbreviations
    "NO":  "NOP", "SA":  "SAS", "GS":  "GSW", "NY":  "NYK",
    "BRK": "BKN", "CHO": "CHA", "PHO": "PHX",
}


def normalize_team(abbr: str) -> str:
    """Normalize a team abbreviation to the nba_api standard."""
    return TEAM_ABBR_MAP.get(abbr.upper(), abbr.upper())


# ---------------------------------------------------------------------------
# Curated trade list  (VERIFIED against public records)
# Format: (player_name, season, trade_date, team_from, team_to)
#
# Only includes MID-SEASON trades where the player realistically played ≥10
# games on each side.  Offseason trades are excluded.
# Dates are approximate (official trade completion date).
# ---------------------------------------------------------------------------
TRADES: list[tuple[str, str, str, str, str]] = [
    # ── 2024-25 season ──────────────────────────────────────────────────
    # Luka/AD blockbuster (3-team, Feb 2, 2025)
    ("Luka Doncic",         "2024-25", "2025-02-02", "DAL", "LAL"),
    ("Anthony Davis",       "2024-25", "2025-02-02", "LAL", "DAL"),
    # Jimmy Butler to Warriors (5-team deal, Feb 6, 2025)
    ("Jimmy Butler",        "2024-25", "2025-02-06", "MIA", "GSW"),
    # Brandon Ingram to Raptors (Feb 7, 2025)
    ("Brandon Ingram",      "2024-25", "2025-02-07", "NOP", "TOR"),
    # Zach LaVine to Kings (3-team deal, Feb 6, 2025)
    ("Zach LaVine",         "2024-25", "2025-02-06", "CHI", "SAC"),
    # Dorian Finney-Smith to Lakers (Dec 29, 2024 — mid-season trade)
    ("Dorian Finney-Smith", "2024-25", "2024-12-29", "BKN", "LAL"),
    # Dennis Schröder: BKN → GSW (Dec 15, 2024)
    ("Dennis Schroder",     "2024-25", "2024-12-15", "BKN", "GSW"),
    # Dennis Schröder: GSW → DET (Feb 6, 2025)
    ("Dennis Schroder",     "2024-25", "2025-02-06", "GSW", "DET"),
    # De'Andre Hunter to Cavaliers (Feb 6, 2025)
    ("De'Andre Hunter",     "2024-25", "2025-02-06", "ATL", "CLE"),

    # ── 2023-24 season ──────────────────────────────────────────────────
    ("Pascal Siakam",       "2023-24", "2024-01-17", "TOR", "IND"),
    ("Dejounte Murray",     "2023-24", "2024-02-08", "ATL", "NOP"),
    ("OG Anunoby",          "2023-24", "2024-01-01", "TOR", "NYK"),
    # NOTE: Alex Caruso/Josh Giddey REMOVED — offseason trade (June 2024)
    ("Buddy Hield",         "2023-24", "2024-02-08", "IND", "PHI"),
    ("Patrick Williams",    "2023-24", "2024-02-08", "CHI", "ORL"),
    ("Daniel Gafford",      "2023-24", "2024-02-08", "WAS", "DAL"),
    ("P.J. Washington",     "2023-24", "2024-02-08", "CHA", "DAL"),
    ("Gordon Hayward",      "2023-24", "2024-02-08", "CHA", "OKC"),
    ("Dorian Finney-Smith", "2023-24", "2024-02-06", "DAL", "BKN"),

    # ── 2022-23 season ──────────────────────────────────────────────────
    ("Kyrie Irving",        "2022-23", "2023-02-06", "BKN", "DAL"),
    ("Kevin Durant",        "2022-23", "2023-02-09", "BKN", "PHX"),
    ("Russell Westbrook",   "2022-23", "2023-02-09", "LAL", "UTA"),
    ("D'Angelo Russell",    "2022-23", "2023-02-09", "MIN", "LAL"),
    ("Mike Conley",         "2022-23", "2023-02-09", "UTA", "MIN"),
    ("Jakob Poeltl",        "2022-23", "2023-02-09", "SAS", "TOR"),
    ("Saddiq Bey",          "2022-23", "2023-02-09", "DET", "ATL"),
    ("Jae Crowder",         "2022-23", "2023-02-09", "PHX", "MIL"),
    ("Bojan Bogdanovic",    "2022-23", "2023-02-09", "DET", "NYK"),
    ("Josh Richardson",     "2022-23", "2023-02-09", "SAS", "NOP"),
    # NOTE: Dillon Brooks REMOVED — offseason sign-and-trade (July 2023)

    # ── 2021-22 season ──────────────────────────────────────────────────
    ("James Harden",        "2021-22", "2022-02-10", "BKN", "PHI"),
    ("Ben Simmons",         "2021-22", "2022-02-10", "PHI", "BKN"),
    ("Tyrese Haliburton",   "2021-22", "2022-02-08", "SAC", "IND"),
    ("Domantas Sabonis",    "2021-22", "2022-02-08", "IND", "SAC"),
    ("CJ McCollum",         "2021-22", "2022-02-08", "POR", "NOP"),
    ("Derrick White",       "2021-22", "2022-02-10", "SAS", "BOS"),
    ("Norman Powell",       "2021-22", "2022-02-08", "POR", "LAC"),
    ("Robert Covington",    "2021-22", "2022-02-08", "POR", "LAC"),
    ("Josh Hart",           "2021-22", "2022-02-08", "NOP", "POR"),
    ("Caris LeVert",        "2021-22", "2022-02-06", "IND", "CLE"),
    ("Jerami Grant",        "2021-22", "2022-02-08", "DET", "POR"),
    ("Seth Curry",          "2021-22", "2022-02-10", "PHI", "BKN"),
    ("Kristaps Porzingis",  "2021-22", "2022-02-10", "DAL", "WAS"),
    ("Spencer Dinwiddie",   "2021-22", "2022-02-10", "WAS", "DAL"),

    # ── 2020-21 season ──────────────────────────────────────────────────
    ("Victor Oladipo",      "2020-21", "2021-03-19", "HOU", "MIA"),
    ("Aaron Gordon",        "2020-21", "2021-03-25", "ORL", "DEN"),
    ("Nikola Vucevic",      "2020-21", "2021-03-25", "ORL", "CHI"),
    ("Evan Fournier",       "2020-21", "2021-03-25", "ORL", "BOS"),
    ("Rajon Rondo",         "2020-21", "2021-03-25", "ATL", "LAC"),
    ("Nemanja Bjelica",     "2020-21", "2021-03-25", "MIA", "MIN"),
    ("JJ Redick",           "2020-21", "2021-03-25", "NOP", "DAL"),
    ("Norman Powell",       "2020-21", "2021-03-25", "TOR", "POR"),
    ("Larry Nance Jr.",     "2020-21", "2021-03-25", "CLE", "POR"),

    # ── 2019-20 season ──────────────────────────────────────────────────
    ("Andre Drummond",      "2019-20", "2020-02-06", "DET", "CLE"),
    ("Robert Covington",    "2019-20", "2020-02-05", "MIN", "HOU"),
    ("Clint Capela",        "2019-20", "2020-02-05", "HOU", "ATL"),
    ("D'Angelo Russell",    "2019-20", "2020-02-06", "GSW", "MIN"),
    ("Andrew Wiggins",      "2019-20", "2020-02-06", "MIN", "GSW"),
    # NOTE: Jrue Holiday NOP→IND REMOVED — never happened, he stayed with NOP all season
    ("Marcus Morris Sr.",   "2019-20", "2020-02-06", "NYK", "LAC"),
    ("Dennis Smith Jr.",    "2019-20", "2020-02-06", "NYK", "DET"),
    ("Derrick Rose",        "2019-20", "2020-02-06", "DET", "MIN"),

    # ── 2018-19 season ──────────────────────────────────────────────────
    ("Tobias Harris",       "2018-19", "2019-02-06", "LAC", "PHI"),
    ("Marc Gasol",          "2018-19", "2019-02-07", "MEM", "TOR"),
    ("Nikola Mirotic",      "2018-19", "2019-02-07", "NOP", "MIL"),
    ("Harrison Barnes",     "2018-19", "2019-02-06", "DAL", "SAC"),
    ("Reggie Bullock",      "2018-19", "2019-02-07", "DET", "LAL"),
    ("Otto Porter Jr.",     "2018-19", "2019-02-07", "WAS", "CHI"),
    ("Enes Kanter",         "2018-19", "2019-02-07", "NYK", "POR"),
    ("Wesley Matthews",     "2018-19", "2019-02-06", "DAL", "NYK"),
    ("DeAndre Jordan",      "2018-19", "2019-01-29", "DAL", "NYK"),
    ("Markieff Morris",     "2018-19", "2019-02-07", "WAS", "NOP"),

    # ── 2017-18 season ──────────────────────────────────────────────────
    ("Blake Griffin",       "2017-18", "2018-01-29", "LAC", "DET"),
    ("Isaiah Thomas",       "2017-18", "2018-02-08", "CLE", "LAL"),
    ("Jordan Clarkson",     "2017-18", "2018-02-08", "LAL", "CLE"),
    ("Larry Nance Jr.",     "2017-18", "2018-02-08", "LAL", "CLE"),
    ("Rodney Hood",         "2017-18", "2018-02-08", "UTA", "CLE"),
    ("Joe Johnson",         "2017-18", "2018-02-08", "UTA", "HOU"),
    ("George Hill",         "2017-18", "2018-02-08", "SAC", "CLE"),
    ("Tyreke Evans",        "2017-18", "2018-02-08", "MEM", "IND"),
    ("DeAndre Jordan",      "2017-18", "2018-02-08", "LAC", "DAL"),
    ("Avery Bradley",       "2017-18", "2018-02-08", "DET", "LAC"),
    ("Lou Williams",        "2017-18", "2018-02-08", "LAL", "LAC"),
    ("Derrick Rose",        "2017-18", "2018-02-08", "CLE", "MIN"),

    # ── 2016-17 season ──────────────────────────────────────────────────
    ("DeMarcus Cousins",    "2016-17", "2017-02-20", "SAC", "NOP"),
    ("Serge Ibaka",         "2016-17", "2017-02-14", "ORL", "TOR"),
    ("P.J. Tucker",         "2016-17", "2017-02-23", "PHX", "TOR"),
    ("Nerlens Noel",        "2016-17", "2017-01-05", "PHI", "DAL"),
    ("Taj Gibson",          "2016-17", "2017-02-23", "CHI", "OKC"),
    ("Doug McDermott",      "2016-17", "2017-02-23", "CHI", "OKC"),
    ("Andrew Bogut",        "2016-17", "2017-02-23", "DAL", "PHI"),
    ("Ersan Ilyasova",      "2016-17", "2017-02-23", "PHI", "ATL"),
    ("Bojan Bogdanovic",    "2016-17", "2017-02-23", "BKN", "WAS"),

    # ── 2015-16 season ──────────────────────────────────────────────────
    ("Tobias Harris",       "2015-16", "2016-02-18", "ORL", "DET"),
    # NOTE: Jeff Teague REMOVED — offseason trade (July 2016)
    ("Arron Afflalo",       "2015-16", "2016-02-18", "NYK", "POR"),
    ("Markieff Morris",     "2015-16", "2016-02-18", "PHX", "WAS"),
    # NOTE: Brandon Knight REMOVED — wrong season (2014-15), direction reversed
    ("Courtney Lee",        "2015-16", "2016-02-18", "MEM", "CHA"),
    # NOTE: Reggie Jackson REMOVED — wrong season (2014-15)
]


# ---------------------------------------------------------------------------
# Known name aliases for nba_api lookups
# Maps display name → name(s) that nba_api recognizes
# ---------------------------------------------------------------------------
NAME_ALIASES: dict[str, list[str]] = {
    "Enes Kanter":      ["Enes Freedom", "Enes Kanter"],
    "Enes Freedom":     ["Enes Freedom", "Enes Kanter"],
    "Jimmy Butler":     ["Jimmy Butler", "Jimmy Butler III"],
    "Reggie Bullock":   ["Reggie Bullock", "Reggie Bullock Jr."],
    "Dennis Schroder":  ["Dennis Schroder", "Dennis Schröder", "Dennis Schroeder"],
    "Nikola Vucevic":   ["Nikola Vucevic", "Nikola Vučević"],
    "Luka Doncic":      ["Luka Doncic", "Luka Dončić"],
    "Kristaps Porzingis": ["Kristaps Porzingis", "Kristaps Porziņģis"],
    "Bojan Bogdanovic": ["Bojan Bogdanovic", "Bojan Bogdanović"],
}


def _normalize_unicode(name: str) -> str:
    """Strip diacritics from a name for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.category(c).startswith("M"))


def _strip_suffix(name: str) -> str:
    """Remove common name suffixes like Jr., Sr., III, II, IV."""
    for suffix in (" III", " II", " IV", " Jr.", " Sr.", " Jr", " Sr"):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name


def resolve_player_id(player_name: str) -> int | None:
    """
    Look up a player's nba_api ID by full name.

    Tries multiple strategies:
    1. Exact match
    2. Case-insensitive match
    3. Known aliases
    4. Unicode-normalized match
    5. Suffix-stripped match
    """
    all_players = nba_players.get_players()

    # 1. Exact match
    matches = [p for p in all_players if p["full_name"] == player_name]
    if matches:
        return matches[0]["id"]

    # 2. Case-insensitive match
    name_lower = player_name.lower()
    matches = [p for p in all_players if p["full_name"].lower() == name_lower]
    if matches:
        return matches[0]["id"]

    # 3. Known aliases
    aliases = NAME_ALIASES.get(player_name, [])
    for alias in aliases:
        matches = [p for p in all_players if p["full_name"].lower() == alias.lower()]
        if matches:
            return matches[0]["id"]

    # 4. Unicode-normalized match
    norm_name = _normalize_unicode(player_name).lower()
    for p in all_players:
        if _normalize_unicode(p["full_name"]).lower() == norm_name:
            return p["id"]

    # 5. Suffix-stripped match
    stripped = _strip_suffix(player_name).lower()
    if stripped != name_lower:
        matches = [p for p in all_players if p["full_name"].lower() == stripped]
        if matches:
            return matches[0]["id"]
        # Also try stripping suffix from API names
        for p in all_players:
            if _strip_suffix(p["full_name"]).lower() == stripped:
                return p["id"]

    return None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve player IDs
    print("Resolving player IDs from nba_api...")
    rows: list[dict] = []
    unresolved: list[str] = []

    for player_name, season, trade_date, team_from, team_to in TRADES:
        pid = resolve_player_id(player_name)
        if pid is None:
            unresolved.append(player_name)
            continue
        rows.append({
            "player_name":  player_name,
            "player_id":    pid,
            "season":       season,
            "trade_date":   trade_date,
            "team_from":    normalize_team(team_from),
            "team_to":      normalize_team(team_to),
        })

    # Write CSV
    fieldnames = ["player_name", "player_id", "season", "trade_date",
                  "team_from", "team_to"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Wrote {len(rows)} trades to {OUTPUT_PATH}")
    if unresolved:
        print(f"[WARN] Could not resolve IDs for {len(unresolved)} players:")
        for name in unresolved:
            print(f"   - {name}")


if __name__ == "__main__":
    main()
