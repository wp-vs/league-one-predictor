#!/usr/bin/env python3
"""
League One Monte Carlo Season Predictor

Fetches xG data from FBref, fits a Dixon-Coles model, and runs
Monte Carlo simulations of the remaining fixtures to predict
final league standings probabilities.

Usage:
    python run_simulation.py                        # Standings-based (default)
    python run_simulation.py --fbref                # Fetch xG data from FBref
    python run_simulation.py --manual               # Use manual match CSV
    python run_simulation.py --sims 50000           # Custom simulation count
    python run_simulation.py --team "Cardiff"       # Detailed view for a team
    python run_simulation.py --no-cache             # Force fresh data fetch
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

from league_one_predictor.data_fetcher import (
    try_fetch_fbref,
    get_played_matches,
    get_remaining_fixtures,
    load_manual_data,
    load_standings,
    load_remaining_fixtures,
)
from league_one_predictor.dixon_coles import fit_dixon_coles, predict_match
from league_one_predictor.standings_estimator import (
    estimate_from_standings,
    refine_strengths_iterative,
)
from league_one_predictor.simulator import (
    build_current_table,
    run_monte_carlo,
    apply_points_deductions,
)
from league_one_predictor.display import (
    format_current_table,
    format_simulation_results,
    format_team_detail,
    format_model_params,
)


# Known points deductions for 2025-26 season (update as needed)
POINTS_DEDUCTIONS = {
    # e.g. "Reading": 6,
}


def run_standings_mode(args):
    """
    Run simulation from league standings + remaining fixtures.

    This is the practical default mode when match-level xG data
    is unavailable. Estimates team strengths from goals scored/conceded.
    """
    print("\n--- Loading standings data ---")
    standings = load_standings(args.standings_file)
    remaining = load_remaining_fixtures(args.fixtures_file)

    print(f"Teams: {len(standings)}")
    print(f"Remaining fixtures: {len(remaining)}")

    # Build current table from standings DataFrame
    current_table = standings.set_index("team")
    if POINTS_DEDUCTIONS:
        current_table = apply_points_deductions(current_table, POINTS_DEDUCTIONS)
        print(f"(Applied points deductions: {POINTS_DEDUCTIONS})")

    print("\n--- Current League Table ---")
    print(format_current_table(current_table))

    # Estimate team strengths from standings
    print("\n--- Estimating Team Strengths ---")
    print("Using iterative strength estimation from goals scored/conceded")
    model = refine_strengths_iterative(standings)
    print(f"Home advantage: {model['home_advantage']:.3f} "
          f"(+{(np.exp(model['home_advantage'])-1)*100:.0f}% expected goals boost)")
    print(f"Dixon-Coles rho: {model['rho']:.3f}")

    return model, current_table, remaining


def run_match_mode(args):
    """
    Run simulation from match-level data (FBref xG or manual CSV).

    This is the more accurate mode when individual match xG data is available.
    """
    print("\n--- Loading match data ---")
    try:
        if args.manual or args.data_file:
            df = load_manual_data(args.data_file)
            print(f"Loaded {len(df)} fixtures from manual data")
        else:
            df = try_fetch_fbref(use_cache=not args.no_cache)
    except Exception as e:
        print(f"\nError fetching data: {e}")
        print("\nTip: Use standings mode instead:")
        print("  python run_simulation.py --standings")
        print("\nOr provide manual match data:")
        print("  python run_simulation.py --manual --data-file path/to/matches.csv")
        sys.exit(1)

    played = get_played_matches(df)
    remaining = get_remaining_fixtures(df)

    if len(played) == 0:
        print("No played matches found. Check your data.")
        sys.exit(1)

    print(f"\nPlayed matches: {len(played)}")
    print(f"Remaining fixtures: {len(remaining)}")

    xg_available = "home_xg" in played.columns and played["home_xg"].notna().sum() > 0
    xg_count = played["home_xg"].notna().sum() if xg_available else 0
    print(f"Matches with xG data: {xg_count}")

    # Build current table
    print("\n--- Current League Table ---")
    current_table = build_current_table(played)
    if POINTS_DEDUCTIONS:
        current_table = apply_points_deductions(current_table, POINTS_DEDUCTIONS)
        print(f"(Applied points deductions: {POINTS_DEDUCTIONS})")
    print(format_current_table(current_table))

    # Fit Dixon-Coles model
    print("\n--- Fitting Dixon-Coles Model ---")
    use_xg = not args.use_goals and xg_available
    print(f"Using {'xG' if use_xg else 'actual goals'} for model fitting")
    print(f"Time decay half-life: {args.half_life} days")

    t0 = time.time()
    model = fit_dixon_coles(played, use_xg=use_xg, half_life_days=args.half_life)
    print(f"Model fitted in {time.time()-t0:.1f}s")
    print(f"Home advantage: {model['home_advantage']:.3f} "
          f"(+{(np.exp(model['home_advantage'])-1)*100:.0f}% xG boost)")
    print(f"Dixon-Coles rho: {model['rho']:.3f}")

    return model, current_table, remaining


def main():
    parser = argparse.ArgumentParser(
        description="League One Monte Carlo Season Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py                          # Default: use standings data
  python run_simulation.py --fbref                  # Fetch xG from FBref
  python run_simulation.py --sims 50000             # More simulations
  python run_simulation.py --team "Cardiff"         # Detailed team view
  python run_simulation.py --team "Bolton" --sims 100000

Data modes:
  Default uses data/standings.csv + data/remaining_fixtures.csv
  --fbref fetches match-level xG data from FBref (may be blocked)
  --manual uses match-level data from data/matches.csv
        """
    )

    # Data source
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fbref", action="store_true",
                        help="Fetch match-level xG data from FBref")
    source.add_argument("--manual", action="store_true",
                        help="Use manual match CSV from data/matches.csv")

    # Data files
    parser.add_argument("--data-file", type=str, default=None,
                        help="Path to manual match data CSV")
    parser.add_argument("--standings-file", type=str, default=None,
                        help="Path to standings CSV (default: data/standings.csv)")
    parser.add_argument("--fixtures-file", type=str, default=None,
                        help="Path to remaining fixtures CSV")

    # Simulation params
    parser.add_argument("--sims", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Model params
    parser.add_argument("--half-life", type=int, default=120,
                        help="Time decay half-life in days for match-level mode")
    parser.add_argument("--use-goals", action="store_true",
                        help="Use actual goals instead of xG for model fitting")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force fresh data fetch from FBref")

    # Output
    parser.add_argument("--team", type=str, default=None,
                        help="Show detailed results for a specific team")

    args = parser.parse_args()

    print("=" * 60)
    print("  LEAGUE ONE MONTE CARLO SEASON PREDICTOR")
    print("  2025-26 Season Run-In Simulation")
    print("=" * 60)

    # Choose data mode
    if args.fbref or args.manual or args.data_file:
        model, current_table, remaining = run_match_mode(args)
    else:
        model, current_table, remaining = run_standings_mode(args)

    # Run Monte Carlo simulation
    if len(remaining) == 0:
        print("\nNo remaining fixtures — season is complete!")
        print("\nFinal table is the current table above.")
        return

    print(f"\n--- Running {args.sims:,} Monte Carlo Simulations ---")
    t0 = time.time()
    results = run_monte_carlo(
        model, current_table, remaining,
        n_simulations=args.sims,
        points_deductions=POINTS_DEDUCTIONS,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s ({args.sims/elapsed:.0f} sims/sec)")

    # Display results
    print("\n--- Projected Final Standings ---")
    print(format_simulation_results(results, current_table))

    # Show team detail if requested
    if args.team:
        matches = [t for t in results if args.team.lower() in t.lower()]
        if matches:
            for t in matches:
                print(format_team_detail(t, results, model))
        else:
            print(f"\nTeam '{args.team}' not found. Available teams:")
            for t in sorted(results.keys()):
                print(f"  {t}")

    # Show model parameters
    print(format_model_params(model))

    print("\n" + "=" * 60)
    print("  Notes:")
    print("  - Auto promotion: positions 1-2")
    print("  - Playoffs: positions 3-6")
    print("  - Relegation: bottom 4")
    print("  - Model: Dixon-Coles (1997) inspired strength estimation")
    print("  - Update data/standings.csv with latest table for best results")
    print("=" * 60)


if __name__ == "__main__":
    main()
