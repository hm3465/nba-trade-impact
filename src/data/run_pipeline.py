"""
run_pipeline.py
---------------
Convenience script to run the full Phase 1 data collection pipeline.

Executes in order:
    1. collect_trades.py      – Generate curated trade registry
    2. collect_team_stats.py  – Fetch team advanced stats per season
    3. collect_game_logs.py   – Fetch player game logs + compute trade splits

Usage:
    python -m src.data.run_pipeline
"""

import time

from src.data import collect_trades, collect_team_stats, collect_game_logs


def main() -> None:
    print("=" * 70)
    print("  NBA Trade Impact — Phase 1 Data Collection Pipeline")
    print("=" * 70)

    steps = [
        ("1/3  Trade Registry", collect_trades.main),
        ("2/3  Team Stats",     collect_team_stats.main),
        ("3/3  Game Logs",      collect_game_logs.main),
    ]

    overall_start = time.time()

    for label, func in steps:
        print(f"\n{'─' * 70}")
        print(f"  Step {label}")
        print(f"{'─' * 70}\n")
        step_start = time.time()
        func()
        elapsed = time.time() - step_start
        print(f"\n  Step completed in {elapsed:.1f}s")

    total = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete!  Total time: {total:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
