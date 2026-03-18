"""
Monte Carlo simulation engine for League One season run-in.

Simulates remaining fixtures N times using the Dixon-Coles model,
tracks league table outcomes, and produces probability distributions
for final positions, promotion, playoffs, and relegation.

Optimized: precomputes score distribution CDFs for all fixtures,
then uses vectorized sampling for fast simulation.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from .dixon_coles import predict_match


def build_current_table(played_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Build the current league table from played match results.

    Uses actual goals (not xG) for the table.
    """
    teams = sorted(set(played_matches["home_team"]) | set(played_matches["away_team"]))
    table = {t: {"played": 0, "won": 0, "drawn": 0, "lost": 0,
                  "gf": 0, "ga": 0, "gd": 0, "points": 0}
             for t in teams}

    for _, m in played_matches.iterrows():
        ht = m["home_team"]
        at = m["away_team"]
        hg = int(m["home_goals"])
        ag = int(m["away_goals"])

        table[ht]["played"] += 1
        table[at]["played"] += 1
        table[ht]["gf"] += hg
        table[ht]["ga"] += ag
        table[at]["gf"] += ag
        table[at]["ga"] += hg

        if hg > ag:
            table[ht]["won"] += 1
            table[ht]["points"] += 3
            table[at]["lost"] += 1
        elif hg < ag:
            table[at]["won"] += 1
            table[at]["points"] += 3
            table[ht]["lost"] += 1
        else:
            table[ht]["drawn"] += 1
            table[at]["drawn"] += 1
            table[ht]["points"] += 1
            table[at]["points"] += 1

    for t in teams:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]

    df = pd.DataFrame.from_dict(table, orient="index")
    df.index.name = "team"
    df = df.sort_values(["points", "gd", "gf"], ascending=[False, False, False])
    return df


def apply_points_deductions(table: pd.DataFrame, deductions: dict) -> pd.DataFrame:
    """Apply any known points deductions (e.g. administration penalties)."""
    table = table.copy()
    for team, pts in deductions.items():
        if team in table.index:
            table.loc[team, "points"] -= pts
    return table.sort_values(["points", "gd", "gf"], ascending=[False, False, False])


def _precompute_fixtures(model, remaining_fixtures, max_goals=7):
    """
    Precompute score distributions for all remaining fixtures.

    Returns:
        fixtures: list of (home_team, away_team)
        score_cdfs: numpy array of shape (n_fixtures, (max_goals+1)^2)
            cumulative distribution for fast sampling
        score_home: array mapping flat index -> home goals
        score_away: array mapping flat index -> away goals
    """
    n_scores = (max_goals + 1) ** 2
    fixtures = []
    score_probs = []

    for _, row in remaining_fixtures.iterrows():
        ht = row["home_team"]
        at = row["away_team"]

        if ht not in model["attack"] or at not in model["attack"]:
            continue

        fixtures.append((ht, at))
        pred = predict_match(model, ht, at, max_goals=max_goals)
        flat_probs = pred["score_matrix"].flatten()
        score_probs.append(flat_probs)

    score_probs = np.array(score_probs)  # (n_fixtures, n_scores)

    # Precompute CDF for each fixture
    score_cdfs = np.cumsum(score_probs, axis=1)

    # Precompute score lookup
    score_home = np.arange(n_scores) // (max_goals + 1)
    score_away = np.arange(n_scores) % (max_goals + 1)

    return fixtures, score_cdfs, score_home, score_away


def run_monte_carlo(model, current_table, remaining_fixtures,
                    n_simulations=10000, points_deductions=None, seed=42):
    """
    Run Monte Carlo simulation of the season run-in.

    Optimized: precomputes all match distributions, then samples
    all fixtures across all simulations in bulk using vectorized ops.
    """
    rng = np.random.default_rng(seed)
    teams = list(current_table.index)
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    # Precompute fixture distributions
    fixtures, score_cdfs, score_home_lookup, score_away_lookup = \
        _precompute_fixtures(model, remaining_fixtures)
    n_fixtures = len(fixtures)

    if n_fixtures == 0:
        return {}

    # Base standings as arrays
    base_points = np.array([int(current_table.loc[t, "points"]) for t in teams])
    base_gd = np.array([int(current_table.loc[t, "gd"]) for t in teams])
    base_gf = np.array([int(current_table.loc[t, "gf"]) for t in teams])

    # Apply deductions to base
    if points_deductions:
        for team, pts in points_deductions.items():
            if team in team_idx:
                base_points[team_idx[team]] -= pts

    # Fixture team indices
    home_idx = np.array([team_idx[h] for h, a in fixtures])
    away_idx = np.array([team_idx[a] for h, a in fixtures])

    # Sample all scorelines at once: (n_simulations, n_fixtures)
    uniform_samples = rng.random((n_simulations, n_fixtures))

    # Accumulators
    all_points = np.zeros((n_simulations, n_teams), dtype=np.int32)
    all_gd = np.zeros((n_simulations, n_teams), dtype=np.int32)
    all_gf = np.zeros((n_simulations, n_teams), dtype=np.int32)

    # For each fixture, find the score index from the CDF
    for f in range(n_fixtures):
        # Binary search for score index across all simulations
        score_indices = np.searchsorted(score_cdfs[f], uniform_samples[:, f])
        score_indices = np.clip(score_indices, 0, len(score_home_lookup) - 1)

        hg = score_home_lookup[score_indices]  # (n_simulations,)
        ag = score_away_lookup[score_indices]

        hi = home_idx[f]
        ai = away_idx[f]

        # Update goals
        all_gf[:, hi] += hg
        all_gf[:, ai] += ag
        all_gd[:, hi] += hg - ag
        all_gd[:, ai] += ag - hg

        # Update points
        home_wins = hg > ag
        away_wins = ag > hg
        draws = hg == ag

        all_points[:, hi] += home_wins * 3 + draws * 1
        all_points[:, ai] += away_wins * 3 + draws * 1

    # Add base standings
    all_points += base_points
    all_gd += base_gd
    all_gf += base_gf

    # Determine positions for each simulation
    # Sort by (points, gd, gf) descending - use negative for argsort ascending
    sort_keys = -(all_points * 10000 + all_gd * 100 + all_gf // 10).astype(np.int64)
    positions = np.zeros((n_simulations, n_teams), dtype=np.int32)
    for sim in range(n_simulations):
        order = np.argsort(sort_keys[sim])
        positions[sim, order] = np.arange(1, n_teams + 1)

    # Aggregate results
    results = {}
    for i, team in enumerate(teams):
        team_points = all_points[:, i]
        team_positions = positions[:, i]

        pos_probs = {}
        for p in range(1, n_teams + 1):
            pos_probs[p] = np.sum(team_positions == p) / n_simulations

        promotion_prob = sum(pos_probs.get(p, 0) for p in range(1, 3))
        playoff_prob = sum(pos_probs.get(p, 0) for p in range(3, 7))
        top_half_prob = sum(pos_probs.get(p, 0) for p in range(1, n_teams // 2 + 1))
        relegation_prob = sum(pos_probs.get(p, 0) for p in range(n_teams - 3, n_teams + 1))

        results[team] = {
            "avg_points": float(np.mean(team_points)),
            "avg_position": float(np.mean(team_positions)),
            "position_probs": pos_probs,
            "promotion_prob": promotion_prob,
            "playoff_prob": playoff_prob,
            "top_half_prob": top_half_prob,
            "relegation_prob": relegation_prob,
            "min_points": int(np.min(team_points)),
            "max_points": int(np.max(team_points)),
        }

    return results
