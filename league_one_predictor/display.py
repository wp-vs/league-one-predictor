"""
Display and formatting utilities for simulation results.
"""

import pandas as pd
from tabulate import tabulate


def format_current_table(table: pd.DataFrame) -> str:
    """Format the current league table for display."""
    display = table.copy()
    display = display.reset_index()
    display.columns = ["Team", "P", "W", "D", "L", "GF", "GA", "GD", "Pts"]
    display.insert(0, "Pos", range(1, len(display) + 1))
    return tabulate(display, headers="keys", tablefmt="simple", showindex=False)


def format_simulation_results(results: dict, current_table: pd.DataFrame) -> str:
    """
    Format Monte Carlo simulation results into a readable table.

    Shows: current points, projected points, promotion/playoff/relegation probs.
    """
    rows = []
    for team in results:
        r = results[team]
        curr_pts = int(current_table.loc[team, "points"]) if team in current_table.index else 0
        rows.append({
            "Team": team,
            "Curr Pts": curr_pts,
            "Proj Pts": f"{r['avg_points']:.1f}",
            "Avg Pos": f"{r['avg_position']:.1f}",
            "Auto Up%": f"{r['promotion_prob']*100:.1f}",
            "Playoff%": f"{r['playoff_prob']*100:.1f}",
            "Releg%": f"{r['relegation_prob']*100:.1f}",
            "Pts Range": f"{r['min_points']}-{r['max_points']}",
        })

    # Sort by average position
    rows.sort(key=lambda x: float(x["Avg Pos"]))

    return tabulate(rows, headers="keys", tablefmt="simple", showindex=False)


def format_team_detail(team: str, results: dict, model: dict) -> str:
    """Show detailed position probability distribution for a team."""
    r = results[team]
    lines = [
        f"\n{'='*50}",
        f"  {team}",
        f"{'='*50}",
        f"  Projected points: {r['avg_points']:.1f} (range: {r['min_points']}-{r['max_points']})",
        f"  Average position: {r['avg_position']:.1f}",
        f"  Attack strength:  {model['attack'][team]:.3f}",
        f"  Defence strength: {model['defence'][team]:.3f}",
        "",
        "  Outcome probabilities:",
        f"    Auto promotion (1st-2nd):  {r['promotion_prob']*100:.1f}%",
        f"    Playoffs (3rd-6th):        {r['playoff_prob']*100:.1f}%",
        f"    Relegation (bottom 4):     {r['relegation_prob']*100:.1f}%",
        "",
        "  Position distribution:",
    ]

    # Show top positions with >0.5% probability
    n_teams = len(results)
    for pos in range(1, n_teams + 1):
        prob = r["position_probs"].get(pos, 0)
        if prob >= 0.005:
            bar = "█" * int(prob * 50)
            lines.append(f"    {pos:>2}. {prob*100:5.1f}% {bar}")

    return "\n".join(lines)


def format_model_params(model: dict) -> str:
    """Display fitted model parameters."""
    lines = [
        "\nModel Parameters:",
        f"  Home advantage: {model['home_advantage']:.3f} "
        f"(+{(np.exp(model['home_advantage'])-1)*100:.0f}% expected goals)",
        f"  Dixon-Coles rho: {model['rho']:.3f}",
        "",
        "  Team strengths (attack / defence):",
        "  Higher attack = stronger offence, lower (more negative) defence = better defence",
    ]

    # Sort by attack strength
    teams_by_attack = sorted(model["teams"], key=lambda t: model["attack"][t], reverse=True)
    for t in teams_by_attack:
        lines.append(f"    {t:>25s}  att: {model['attack'][t]:+.3f}  def: {model['defence'][t]:+.3f}")

    return "\n".join(lines)


# Need numpy for format_model_params
import numpy as np
