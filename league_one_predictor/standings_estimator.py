"""
Estimate Dixon-Coles team strengths directly from league standings.

This is the fallback approach when match-level xG data is unavailable.
Instead of fitting from individual match results, we estimate attack and
defence strengths from goals scored/conceded per game, adjusted for
strength of schedule.

Less precise than match-level fitting but practical and still useful
for Monte Carlo simulation.
"""

import numpy as np
from scipy.optimize import minimize


def estimate_from_standings(standings_df):
    """
    Estimate team strengths from a standings DataFrame.

    Expected columns: team, played, won, drawn, lost, gf, ga, gd, points

    Returns a model dict compatible with dixon_coles.predict_match().
    """
    teams = list(standings_df["team"])
    n = len(teams)

    # Average goals per game in the league
    total_gf = standings_df["gf"].sum()
    total_games = standings_df["played"].sum() / 2  # each game counted twice
    avg_goals_per_team = total_gf / standings_df["played"].sum()

    # Initial strength estimates from goals per game
    attack_init = np.log(standings_df["gf"].values / standings_df["played"].values / avg_goals_per_team)
    defence_init = np.log(standings_df["ga"].values / standings_df["played"].values / avg_goals_per_team)

    # Centre the attack ratings (identifiability constraint)
    attack_init -= attack_init.mean()

    # Estimate home advantage from league-wide home/away split
    # Typical League One home advantage: ~0.25-0.35 in log space
    home_advantage = 0.28

    # Rho (Dixon-Coles correction) - typical value
    rho = -0.05

    return {
        "attack": {teams[i]: float(attack_init[i]) for i in range(n)},
        "defence": {teams[i]: float(defence_init[i]) for i in range(n)},
        "home_advantage": home_advantage,
        "rho": rho,
        "teams": teams,
    }


def refine_strengths_iterative(standings_df, n_iterations=20):
    """
    Iteratively refine team strengths accounting for strength of schedule.

    Basic idea: a team's goals scored against strong defences should count
    more than goals against weak defences, and vice versa.
    """
    teams = list(standings_df["team"])
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    played = standings_df["played"].values.astype(float)
    gf = standings_df["gf"].values.astype(float)
    ga = standings_df["ga"].values.astype(float)

    avg_gpg = gf.sum() / played.sum()

    # Initialize
    attack = gf / played / avg_gpg
    defence = ga / played / avg_gpg

    # Iterative refinement (simplified Bradley-Terry style)
    for _ in range(n_iterations):
        # In a round-robin, each team plays every other team roughly equally
        # Adjust attack by average opposition defence, and vice versa
        avg_opp_def = (defence.sum() - defence) / (n - 1)
        avg_opp_att = (attack.sum() - attack) / (n - 1)

        attack_new = (gf / played) / (avg_opp_def * avg_gpg)
        defence_new = (ga / played) / (avg_opp_att * avg_gpg)

        # Normalize
        attack_new *= n / attack_new.sum()
        defence_new *= n / defence_new.sum()

        attack = 0.7 * attack_new + 0.3 * attack
        defence = 0.7 * defence_new + 0.3 * defence

    # Convert to log scale for Dixon-Coles compatibility
    log_attack = np.log(attack)
    log_defence = np.log(defence)

    # Centre attack
    log_attack -= log_attack.mean()

    home_advantage = 0.28
    rho = -0.05

    return {
        "attack": {teams[i]: float(log_attack[i]) for i in range(n)},
        "defence": {teams[i]: float(log_defence[i]) for i in range(n)},
        "home_advantage": home_advantage,
        "rho": rho,
        "teams": teams,
    }
