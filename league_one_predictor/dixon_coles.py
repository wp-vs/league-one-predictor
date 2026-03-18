"""
Dixon-Coles model for football match prediction.

Based on Dixon & Coles (1997): "Modelling Association Football Scores and
Inefficiencies in the Football Betting Market"

The model estimates:
- Attack strength (alpha) for each team
- Defence strength (beta) for each team
- Home advantage parameter (gamma)
- Correlation parameter (rho) for low-scoring adjustment

Team strengths can be estimated from either actual goals or xG (preferred).
A time-decay function gives more weight to recent matches.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from collections import defaultdict


def tau(x, y, lam, mu, rho):
    """
    Dixon-Coles correction factor for low-scoring outcomes.

    Adjusts probabilities for 0-0, 1-0, 0-1, and 1-1 scorelines
    which are more correlated than independent Poisson would predict.
    """
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def dc_log_likelihood(params, matches, teams, xi=0.0018):
    """
    Negative log-likelihood for the Dixon-Coles model.

    params layout:
        [0..n-1]        = attack strengths (one per team)
        [n..2n-1]       = defence strengths (one per team)
        [2n]            = home advantage (gamma)
        [2n+1]          = correlation (rho)

    xi: time decay parameter (per day). Default ~0.5 per year.
    """
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack = params[:n]
    defence = params[n:2*n]
    gamma = params[2*n]
    rho = params[2*n + 1]

    log_lik = 0.0

    for _, match in matches.iterrows():
        home = team_idx[match["home_team"]]
        away = team_idx[match["away_team"]]

        # Expected goals for each team
        lam = np.exp(attack[home] + defence[away] + gamma)  # home expected goals
        mu = np.exp(attack[away] + defence[home])            # away expected goals

        # Clamp to avoid numerical issues
        lam = max(lam, 0.001)
        mu = max(mu, 0.001)

        # Use xG-weighted goals if available, otherwise actual goals
        home_g = match.get("home_xg") if "home_xg" in match.index and not np.isnan(match.get("home_xg", np.nan)) else match["home_goals"]
        away_g = match.get("away_xg") if "away_xg" in match.index and not np.isnan(match.get("away_xg", np.nan)) else match["away_goals"]

        # For the likelihood we need integer goals (actual results)
        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        # Time decay weight
        if "weight" in match.index:
            w = match["weight"]
        else:
            w = 1.0

        # Dixon-Coles adjusted probability
        p_home = poisson.pmf(hg, lam)
        p_away = poisson.pmf(ag, mu)
        tau_adj = tau(hg, ag, lam, mu, rho)

        prob = p_home * p_away * tau_adj
        if prob > 0:
            log_lik += w * np.log(prob)
        else:
            log_lik += w * (-30)  # penalty for impossible outcomes

    return -log_lik


def dc_log_likelihood_xg(params, matches, teams, xi=0.0018):
    """
    Alternative likelihood using xG as the observed outcome rather than goals.

    This treats xG as a continuous approximation - we use a Gaussian
    approximation around the Poisson mean to get a smooth likelihood.
    When xG is available, this gives more stable strength estimates
    than using noisy actual scorelines.
    """
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack = params[:n]
    defence = params[n:2*n]
    gamma = params[2*n]
    rho = params[2*n + 1]

    log_lik = 0.0

    for _, match in matches.iterrows():
        home = team_idx[match["home_team"]]
        away = team_idx[match["away_team"]]

        lam = np.exp(attack[home] + defence[away] + gamma)
        mu = np.exp(attack[away] + defence[home])

        lam = max(lam, 0.001)
        mu = max(mu, 0.001)

        # Use xG if available, else actual goals
        home_xg = match.get("home_xg", np.nan)
        away_xg = match.get("away_xg", np.nan)

        if np.isnan(home_xg):
            home_xg = float(match["home_goals"])
        if np.isnan(away_xg):
            away_xg = float(match["away_goals"])

        w = match.get("weight", 1.0)

        # Poisson log-likelihood with continuous xG (using Stirling approx)
        # log P(x|λ) ≈ x*log(λ) - λ - x*log(x) + x (for continuous x)
        ll_home = home_xg * np.log(lam) - lam
        ll_away = away_xg * np.log(mu) - mu

        log_lik += w * (ll_home + ll_away)

    return -log_lik


def fit_dixon_coles(matches, use_xg=True, half_life_days=120):
    """
    Fit the Dixon-Coles model to match data.

    Parameters:
        matches: DataFrame with columns date, home_team, away_team,
                 home_goals, away_goals, home_xg, away_xg
        use_xg: If True, use xG-based likelihood (smoother, more stable)
        half_life_days: Half-life for time decay weighting

    Returns:
        dict with keys: attack, defence, home_advantage, rho, teams
    """
    matches = matches.copy()

    # Compute time weights (exponential decay)
    if "date" in matches.columns:
        max_date = matches["date"].max()
        days_ago = (max_date - matches["date"]).dt.days
        xi = np.log(2) / half_life_days
        matches["weight"] = np.exp(-xi * days_ago)
    else:
        matches["weight"] = 1.0

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n = len(teams)

    # Initial params: zero attack/defence, small home advantage, zero rho
    x0 = np.zeros(2 * n + 2)
    x0[2*n] = 0.25  # initial home advantage

    # Choose likelihood function
    if use_xg and "home_xg" in matches.columns:
        lik_fn = dc_log_likelihood_xg
    else:
        lik_fn = dc_log_likelihood

    # Constraint: sum of attack strengths = 0 (identifiability)
    constraints = [{
        "type": "eq",
        "fun": lambda p, n=n: np.sum(p[:n])
    }]

    # Bounds: rho between -1 and 1, others unconstrained
    bounds = [(None, None)] * (2*n) + [(None, None)] + [(-0.99, 0.99)]

    result = minimize(
        lik_fn,
        x0,
        args=(matches, teams),
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 500, "ftol": 1e-8}
    )

    if not result.success:
        print(f"Warning: optimization did not converge: {result.message}")

    params = result.x
    team_idx = {t: i for i, t in enumerate(teams)}

    return {
        "attack": {t: params[i] for t, i in team_idx.items()},
        "defence": {t: params[n + i] for t, i in team_idx.items()},
        "home_advantage": params[2*n],
        "rho": params[2*n + 1],
        "teams": teams,
    }


def predict_match(model, home_team, away_team, max_goals=8):
    """
    Predict match outcome probabilities using the fitted model.

    Returns:
        dict with:
            home_win_prob, draw_prob, away_win_prob,
            expected_home_goals, expected_away_goals,
            score_matrix (max_goals+1 x max_goals+1 probability matrix)
    """
    attack = model["attack"]
    defence = model["defence"]
    gamma = model["home_advantage"]
    rho = model["rho"]

    lam = np.exp(attack[home_team] + defence[away_team] + gamma)
    mu = np.exp(attack[away_team] + defence[home_team])

    # Score probability matrix with Dixon-Coles adjustment
    score_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson.pmf(i, lam) * poisson.pmf(j, mu) * tau(i, j, lam, mu, rho)
            score_matrix[i, j] = p

    # Normalize (small correction due to truncation)
    score_matrix /= score_matrix.sum()

    home_win = np.sum(np.tril(score_matrix, -1))  # below diagonal
    draw = np.sum(np.diag(score_matrix))
    away_win = np.sum(np.triu(score_matrix, 1))  # above diagonal

    # Wait — convention: score_matrix[i,j] = P(home=i, away=j)
    # home wins when i > j, which is below diagonal
    home_win = sum(score_matrix[i, j] for i in range(max_goals+1)
                   for j in range(max_goals+1) if i > j)
    draw = sum(score_matrix[i, j] for i in range(max_goals+1)
               for j in range(max_goals+1) if i == j)
    away_win = sum(score_matrix[i, j] for i in range(max_goals+1)
                   for j in range(max_goals+1) if i < j)

    return {
        "home_win_prob": home_win,
        "draw_prob": draw,
        "away_win_prob": away_win,
        "expected_home_goals": lam,
        "expected_away_goals": mu,
        "score_matrix": score_matrix,
        "lambda": lam,
        "mu": mu,
    }


def sample_scoreline(prediction, rng=None):
    """
    Sample a random scoreline from the predicted score distribution.
    Used in Monte Carlo simulation.
    """
    if rng is None:
        rng = np.random.default_rng()

    matrix = prediction["score_matrix"]
    flat = matrix.flatten()
    idx = rng.choice(len(flat), p=flat)
    home_goals = idx // matrix.shape[1]
    away_goals = idx % matrix.shape[1]
    return home_goals, away_goals
