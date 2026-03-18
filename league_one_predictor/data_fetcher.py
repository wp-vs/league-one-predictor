"""
Fetch League One match data with xG from FotMob API.

FotMob provides free match-level xG data for League One.
FBref no longer has xG data (Opta terminated access Jan 2026)
and never covered League One with advanced stats anyway.

Also supports loading standings + remaining fixtures as a fallback
when match-level data is unavailable.
"""

import time
import json
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# FotMob League One ID
FOTMOB_LEAGUE_ID = 108

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_fotmob_matches(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch match results with xG from FotMob's API.

    FotMob provides free xG for League One matches via their
    unofficial API at fotmob.com/api/leagues and /api/matchDetails.

    Returns DataFrame with columns:
        date, home_team, away_team, home_goals, away_goals, home_xg, away_xg
    """
    cache_path = DATA_DIR / "fotmob_matches.csv"

    if use_cache and cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 6:
            print(f"Using cached FotMob data ({age_hours:.1f}h old)")
            return pd.read_csv(cache_path, parse_dates=["date"])

    print("Fetching League One data from FotMob API...")

    # Step 1: Get league overview with all matches
    league_url = f"https://www.fotmob.com/api/leagues?id={FOTMOB_LEAGUE_ID}"
    resp = requests.get(league_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    league_data = resp.json()

    # Extract matches from the league data
    matches = []
    all_matches = league_data.get("matches", {}).get("allMatches", [])
    if not all_matches:
        # Try alternative structure
        for round_data in league_data.get("matches", {}).get("data", []):
            all_matches.extend(round_data.get("matches", []))

    print(f"Found {len(all_matches)} matches in league data")

    # Step 2: For each played match, get xG from match details
    played_count = 0
    for match in all_matches:
        match_id = match.get("id")
        status = match.get("status", {})
        is_finished = status.get("finished", False)

        home_info = match.get("home", {})
        away_info = match.get("away", {})

        home_team = home_info.get("name", "")
        away_team = away_info.get("name", "")
        date = match.get("utcTime", "")[:10]  # YYYY-MM-DD

        home_goals = None
        away_goals = None
        home_xg = None
        away_xg = None

        if is_finished:
            home_goals = home_info.get("score")
            away_goals = away_info.get("score")
            played_count += 1

            # Get xG from match details (rate limited)
            if match_id:
                try:
                    time.sleep(0.5)  # Be respectful to FotMob
                    detail_url = f"https://www.fotmob.com/api/matchDetails?matchId={match_id}"
                    detail_resp = requests.get(detail_url, headers=HEADERS, timeout=15)
                    if detail_resp.status_code == 200:
                        detail = detail_resp.json()
                        content = detail.get("content", {})
                        stats = content.get("stats", {})
                        # Look for xG in stats
                        if stats:
                            for period in stats.get("Ede5A3ihS", stats):
                                if isinstance(period, dict):
                                    for stat_group in period.get("stats", []):
                                        for stat in stat_group.get("stats", []):
                                            if "expected_goals" in stat.get("key", "").lower() or \
                                               "xg" in stat.get("title", "").lower():
                                                vals = stat.get("stats", [])
                                                if len(vals) >= 2:
                                                    home_xg = float(vals[0])
                                                    away_xg = float(vals[1])
                except Exception:
                    pass  # xG not available for this match

        matches.append({
            "date": date,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "home_xg": home_xg,
            "away_xg": away_xg,
        })

    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Save cache
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache_path, index=False)
    xg_count = df["home_xg"].notna().sum()
    print(f"Fetched {len(df)} fixtures ({played_count} played, {xg_count} with xG)")

    return df


def get_played_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Return only completed matches."""
    played = df.dropna(subset=["home_goals", "away_goals"]).copy()
    played["home_goals"] = played["home_goals"].astype(int)
    played["away_goals"] = played["away_goals"].astype(int)
    return played


def get_remaining_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Return fixtures not yet played."""
    return df[df["home_goals"].isna()].copy()


def load_manual_data(path: str = None) -> pd.DataFrame:
    """
    Load manually provided match data from a CSV file.

    Expected columns: date, home_team, away_team, home_goals, away_goals, home_xg, away_xg
    For remaining fixtures, leave goals and xG columns empty.
    """
    if path is None:
        path = DATA_DIR / "matches.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_standings(path: str = None) -> pd.DataFrame:
    """
    Load league standings from CSV.

    Expected columns: team, played, won, drawn, lost, gf, ga, gd, points
    """
    if path is None:
        path = DATA_DIR / "standings.csv"
    return pd.read_csv(path)


def load_remaining_fixtures(path: str = None) -> pd.DataFrame:
    """
    Load remaining fixtures from CSV.

    Expected columns: date, home_team, away_team
    """
    if path is None:
        path = DATA_DIR / "remaining_fixtures.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def try_fetch_fotmob(use_cache: bool = True):
    """Try FotMob API to get match data with xG."""
    try:
        return fetch_fotmob_matches(use_cache=use_cache)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch from FotMob: {e}\n"
            "Use default standings mode instead:\n"
            "  python run_simulation.py"
        )
