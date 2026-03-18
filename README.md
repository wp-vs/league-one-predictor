# League One Monte Carlo Season Predictor

Monte Carlo simulation of the EFL League One 2025-26 season run-in, using a Dixon-Coles inspired model to predict final standings probabilities.

## How it works

1. **Team strength estimation** — Attack and defence ratings are derived from goals scored/conceded per game, refined iteratively to account for strength of schedule (or fitted via Dixon-Coles MLE from match-level xG data when available)
2. **Match simulation** — Each remaining fixture is modelled as independent Poisson processes with a Dixon-Coles correction for low-scoring correlation, plus home advantage
3. **Monte Carlo aggregation** — 10,000+ simulations of all remaining fixtures produce probability distributions for final positions, promotion, playoffs, and relegation

## Quick start

```bash
pip install -r requirements.txt
python run_simulation.py
```

## Usage

```bash
# Default: use standings data (fast, practical)
python run_simulation.py

# More simulations for tighter confidence
python run_simulation.py --sims 100000

# Detailed view for a specific team
python run_simulation.py --team "Bolton"

# Fetch match-level xG data from FotMob API
python run_simulation.py --fotmob

# Use your own match data CSV
python run_simulation.py --manual --data-file path/to/matches.csv
```

## Updating the data

Edit `data/standings.csv` with the latest league table and `data/remaining_fixtures.csv` with upcoming fixtures. The standings CSV format is:

```
team,played,won,drawn,lost,gf,ga,gd,points
Lincoln City,37,24,8,5,65,28,37,80
...
```

## Data modes

| Mode | Command | Data source | Accuracy |
|------|---------|-------------|----------|
| **Standings** (default) | `python run_simulation.py` | `data/standings.csv` + `data/remaining_fixtures.csv` | Good |
| **FotMob xG** | `python run_simulation.py --fotmob` | FotMob API (match-level xG) | Best (when available) |
| **Manual matches** | `python run_simulation.py --manual` | `data/matches.csv` | Best (with xG) |

### xG data sources for League One

- **FotMob** (recommended) — Provides free match-level xG for League One via their unofficial API. The `--fotmob` flag fetches this automatically.
- **FBref** — As of January 2026, FBref no longer has any xG data (Opta/Stats Perform terminated access). Even before that, League One was not covered by FBref's advanced stats.
- **Understat** — Only covers top 5 European leagues, no EFL coverage.
- **FootyStats** — Has League One xG but match-level data requires a paid subscription (~£20).

## Model details

- **Dixon-Coles (1997)** — Adjusts independent Poisson for the empirical correlation in low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
- **Time decay** — Recent matches weighted more heavily (configurable half-life, default 120 days)
- **xG-based fitting** — When match-level xG is available, strengths are estimated from expected goals rather than actual goals, giving more stable signals
- **Home advantage** — Calibrated boost to home team's expected goals (~30%)

## Key references

- Dixon, M.J. & Coles, S.G. (1997). "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
- Maher, M.J. (1982). "Modelling association football scores"
